"""Surface-crossing Monte Carlo for an isotropic flash, in plain NumPy.

This is a second, independent sampler next to :mod:`shell_mc`. The difference
is the estimator, not only the code. ``shell_mc`` integrates track length
inside a shell of finite width and divides by its volume, so its answer carries
a shell-width bias that has to be refined away. Here every crossing of an
infinitely thin sphere is found analytically, and the classical surface
estimators are accumulated:

    scalar fluence   Q(r) = (1 / 4 pi r**2) sum_c w_c / |mu_c|
    outward current  F(r) = (1 / 4 pi r**2) sum_c w_c sign(mu_c)

with ``mu_c`` the cosine between the photon direction and the outward normal at
the crossing. Both inward and outward crossings count, and a photon that
re-enters contributes again; that is the volume-averaged fluence the transport
equation defines, not a first-entry probability.

Absorption is carried as the weight ``exp(-a S)`` on the accumulated path ``S``,
so one set of trajectories serves several absorption coefficients and the
derivative identity ``d<f>/da = -Cov(f, S)`` can be checked on the same paths
rather than across independent runs. Flights are sampled with the scattering
coefficient alone, and directions turn with the Henyey-Greenstein kernel by
exact inversion.

Nothing here imports the angular solver, the cache, the eigenmodes or Numba.
The ``1/|mu|`` weight has a heavy tail at grazing crossings, so batch spreads
are the honest uncertainty statement and a narrow one is not a guarantee.
"""
import numpy as np

__all__ = ["crossing_estimate", "mean_cosine", "ORDER_LABELS", "OBSERVABLES"]

ORDER_LABELS = ("zero", "one", "two_or_more")
OBSERVABLES = ("weight", "weight_cosine", "weight_path", "weight_cosine_path")


def _henyey_greenstein_cosines(rng, g, size):
    """Exact inversion of the HG cumulative distribution."""
    uniform = rng.random(size)
    if abs(g) < 1e-12:
        return 2 * uniform - 1
    ratio = (1 - g * g) / (1 - g + 2 * g * uniform)
    return np.clip((1 + g * g - ratio * ratio) / (2 * g), -1.0, 1.0)


def _turn(rng, directions, g):
    """Rotate each direction by an HG polar angle and a uniform azimuth."""
    cosines = _henyey_greenstein_cosines(rng, g, len(directions))
    sines = np.sqrt(np.maximum(1 - cosines ** 2, 0.0))
    phis = 2 * np.pi * rng.random(len(directions))
    helper = np.zeros_like(directions)
    steep = np.abs(directions[:, 2]) < 0.9
    helper[steep, 2] = 1.0
    helper[~steep, 0] = 1.0
    first = np.cross(directions, helper)
    first /= np.linalg.norm(first, axis=1)[:, None]
    second = np.cross(directions, first)
    return (cosines[:, None] * directions
            + sines[:, None] * (np.cos(phis)[:, None] * first
                                + np.sin(phis)[:, None] * second))


def crossing_estimate(*, scattering_per_m=0.022, g=0.9, absorptions=(0.02, 0.07),
                      radii_m=(20.0, 80.0), photons_per_batch=20000, batches=8,
                      max_path_m=1000.0, seed=20260917, max_collisions=4000):
    """Per-batch crossing sums. Axes: batch, absorption, radius, order, observable.

    The observables are the four sums the fluence, the current and the
    path-covariance identity need: ``w``, ``w mu``, ``w S`` and ``w mu S``,
    each already divided by ``|mu|`` and by ``4 pi r**2 photons``. The scalar
    fluence of order ``o`` is therefore ``sum[..., 0]`` and its flux-weighted
    mean radial cosine is ``sum[..., 1] / sum[..., 0]``.
    """
    absorption = np.atleast_1d(np.asarray(absorptions, float))
    radii = np.atleast_1d(np.asarray(radii_m, float))
    if (not np.isfinite(absorption).all() or np.any(absorption < 0)
            or not np.isfinite(radii).all() or np.any(radii <= 0)):
        raise ValueError("Absorptions must be nonnegative and radii positive")
    if not np.isfinite(scattering_per_m) or scattering_per_m <= 0 or not -1 < g < 1:
        raise ValueError("Require positive scattering and -1 < g < 1")
    if max_path_m <= radii.max():
        raise ValueError("max_path_m must exceed every radius")
    for value in (photons_per_batch, batches, max_collisions):
        if not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError("Counts must be positive integers")
    out = np.zeros((int(batches), len(absorption), len(radii),
                    len(ORDER_LABELS), len(OBSERVABLES)))
    area = 4 * np.pi * radii ** 2
    for batch in range(int(batches)):
        rng = np.random.default_rng(int(seed) + batch)
        count = int(photons_per_batch)
        position = np.zeros((count, 3))
        # A fixed initial direction is equivalent to isotropic emission for
        # observables that depend only on the radius.
        direction = np.tile(np.array([0.0, 0.0, 1.0]), (count, 1))
        path = np.zeros(count)
        order = np.zeros(count, dtype=np.int64)
        for _ in range(int(max_collisions)):
            if not count:
                break
            flight = rng.exponential(1 / scattering_per_m, count)
            flight = np.minimum(flight, max_path_m - path)
            along = np.einsum("ij,ij->i", position, direction)
            square = np.einsum("ij,ij->i", position, position)
            for index, radius in enumerate(radii):
                discriminant = along ** 2 - square + radius ** 2
                valid = discriminant > 0
                root = np.sqrt(np.where(valid, discriminant, 0.0))
                for step in (-root, root):
                    distance = -along + step
                    hit = valid & (distance >= 0) & (distance < flight)
                    if not np.any(hit):
                        continue
                    where = np.flatnonzero(hit)
                    point = position[where] + distance[where, None] * direction[where]
                    cosine = np.einsum("ij,ij->i", point, direction[where]) / radius
                    magnitude = np.abs(cosine)
                    # A crossing exactly tangent to the sphere carries no
                    # fluence weight of its own; drop it rather than divide.
                    keep = magnitude > 1e-12
                    where, cosine, magnitude = where[keep], cosine[keep], magnitude[keep]
                    total_path = path[where] + distance[where]
                    for ia, value in enumerate(absorption):
                        weight = np.exp(-value * total_path) / magnitude
                        bucket = np.minimum(order[where], 2)
                        # The current estimator is w sign(mu); written against
                        # the fluence weight w/|mu| that is a factor mu, so the
                        # ratio of the two sums is the flux-weighted cosine.
                        for observable, quantity in enumerate((
                                np.ones_like(weight), cosine,
                                total_path, cosine * total_path)):
                            np.add.at(out[batch, ia, index, :, observable],
                                      bucket, weight * quantity)
            position = position + flight[:, None] * direction
            path = path + flight
            alive = path < max_path_m
            position, direction, path = position[alive], direction[alive], path[alive]
            order = order[alive] + 1
            count = len(path)
            if count:
                direction = _turn(rng, direction, g)
        out[batch] /= int(photons_per_batch)
        out[batch] /= area[None, :, None, None]
    return out


def mean_cosine(batch_data, order=2):
    """Flux-weighted mean radial cosine of one scattering order, with an error.

    The ratio of ensemble means is formed first, then the batch spread of the
    residual gives the standard error: the two sums are strongly correlated, so
    a naive error on each separately would be wrong.
    """
    weight = batch_data[..., order, 0]
    weighted_cosine = batch_data[..., order, 1]
    mean = weighted_cosine.mean(axis=0) / weight.mean(axis=0)
    residual = weighted_cosine - mean[None, ...] * weight
    error = residual.std(axis=0, ddof=1) / (np.sqrt(len(weight))
                                            * weight.mean(axis=0))
    return mean, error
