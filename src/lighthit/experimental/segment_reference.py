"""Independent single-scattering reference for directed flashes and segments.

Nothing here touches :mod:`lighthit.angular`, :mod:`lighthit.cache` or any
real-``k`` Fourier inversion. The first-scattered signal is built directly in
coordinate space from the three factors a photon experiences: an exponential
flight to the scattering point, one phase-function turn, and a second
exponential flight to the receiver. The receiver is an isotropic point of unit
effective area, so a signal here is photons per square metre per emitted
photon, the same normalisation :mod:`lighthit.cache` uses.

Two phase functions are provided. ``"hg"`` is the full Henyey-Greenstein
kernel; ``"truncated"`` keeps only degrees ``0..L`` of its Legendre series,

    p_L(x) = sum_{l<=L} (2l+1)/(4 pi) g**l P_l(x),

which is the kernel a finite-``L`` angular solve actually applies at first
order. Comparing a cache against ``"hg"`` therefore mixes angular truncation
with every other numerical effect; comparing it against ``"truncated"`` at the
same ``L`` isolates the rest. Both are available on purpose.

Two independent routes to the same finite-segment number are provided as well.
:func:`segment_first_order` integrates over emission points and cone azimuth,
with an inner flight quadrature. :func:`segment_first_order_mc` samples arrival
directions at the receiver instead and needs the unscattered cone field, not
the emission parametrisation. They share no quadrature and no variable.

The geometry is exact in every case: emission points follow the straight
segment, each with its own time, and the Cherenkov cone is integrated over its
own azimuth rather than replaced by an average direction.
"""
from dataclasses import dataclass
import numpy as np
from scipy.integrate import quad_vec
from scipy.special import eval_legendre, roots_legendre

from ..medium import Medium, C_VACUUM_M_PER_NS

__all__ = ["phase_function", "directed_first_order", "segment_first_order",
           "segment_first_order_mc", "segment_ballistic_field",
           "ballistic_by_ray_counting", "SegmentGeometry"]


def phase_function(cosine, medium: Medium, *, kind="hg", degree=None):
    """Scattering kernel per steradian, normalised to one over the sphere."""
    x = np.asarray(cosine, float)
    g = medium.g
    if kind == "hg":
        if abs(g) < 1e-15:
            return np.full(x.shape, 1 / (4 * np.pi))
        return (1 - g * g) / (4 * np.pi * (1 + g * g - 2 * g * x) ** 1.5)
    if kind != "truncated":
        raise ValueError("kind must be 'hg' or 'truncated'")
    if degree is None:
        raise ValueError("the truncated kernel needs an explicit degree")
    ell = np.arange(int(degree) + 1)
    weights = (2 * ell + 1) * g ** ell / (4 * np.pi)
    return np.tensordot(weights, eval_legendre(ell.reshape((-1,) + (1,) * x.ndim),
                                               x[None, ...]), axes=1)


def directed_first_order(omega, sources_m, directions, detector_m, medium: Medium,
                         *, kind="hg", degree=None, epsabs=1e-32, epsrel=1e-10,
                         limit=400, panels=16):
    """Once-scattered spectrum of directed point flashes, by one quadrature.

    A photon leaves ``sources_m`` along a unit vector, scatters once at
    distance ``d`` along that ray and continues to the receiver:

        K1(omega) = int_0^inf dd  mu_s e^{-mu_t d} p(cos theta)
                    e^{-mu_t l} / l**2  exp(i omega (d + l) / v),

    with ``l`` the distance from the scattering point to the receiver and
    ``cos theta`` the turn angle there. The ``1/l**2`` spreading factor is the
    only singular feature; it is integrable, and the quadrature is told where
    each ray's closest approach sits.

    Sources and directions broadcast against each other, so one call can carry
    a whole set of emission points and cone azimuths. The result has shape
    ``(rays, frequencies)``, or ``(frequencies,)`` for a single ray.
    """
    w = np.atleast_1d(np.asarray(omega, float))
    starts = np.asarray(sources_m, float)
    rays = np.asarray(directions, float)
    single = starts.ndim == 1 and rays.ndim == 1
    starts, rays = np.atleast_2d(starts), np.atleast_2d(rays)
    starts, rays = np.broadcast_arrays(starts, rays)
    det = np.asarray(detector_m, float)
    if det.shape != (3,) or starts.shape[1:] != (3,) or rays.shape[1:] != (3,):
        raise ValueError("sources, directions and detector must be 3-vectors")
    if not np.allclose(np.linalg.norm(rays, axis=1), 1, atol=1e-12):
        raise ValueError("directions must be unit vectors")
    relative = det[None, :] - starts
    along = np.einsum("ij,ij->i", relative, rays)
    impact = np.sqrt(np.maximum(np.einsum("ij,ij->i", relative, relative)
                                - along ** 2, 0.0))
    if np.any((impact < 1e-12) & (along > 0)):
        raise ValueError("A receiver sits on an emission ray; the 1/l**2 "
                         "integrand is not integrable there")
    speed = medium.speed_m_per_ns

    def integrand(d):
        gap = det[None, :] - (starts + d * rays)
        ell = np.sqrt(np.einsum("ij,ij->i", gap, gap))
        cos_turn = np.einsum("ij,ij->i", gap, rays) / ell
        kernel = phase_function(cos_turn, medium, kind=kind, degree=degree)
        amplitude = (medium.scattering_per_m * kernel
                     * np.exp(-medium.extinction_per_m * (d + ell)) / ell ** 2)
        return (amplitude[:, None]
                * np.exp(1j * w[None, :] * (d + ell)[:, None] / speed)).ravel()

    # Panel the range around where the rays' integrands peak, at their closest
    # approaches, and stop a few extinction lengths past the last one. One
    # panel per ray would mean thousands of adaptive calls on a batch this
    # size, so the peaks are summarised by quantiles and the adaptive rule is
    # left to find the rest; ``panels`` trades cost against that reliance.
    peaks = np.maximum(along, 0.0)
    far = float(peaks.max()) + 40 / medium.extinction_per_m
    if len(peaks) <= panels:
        marks = np.unique(peaks)
    else:
        marks = np.unique(np.quantile(peaks, np.linspace(0, 1, int(panels))))
    breaks = np.unique(np.concatenate([[0.0], marks, [far]]))
    total = np.zeros(len(rays) * len(w), complex)
    for lo, hi in zip(breaks, breaks[1:]):
        if hi <= lo:
            continue
        inside = [float(p) for p in np.unique(peaks[(peaks > lo) & (peaks < hi)])]
        value, _ = quad_vec(integrand, lo, hi, epsabs=epsabs, epsrel=epsrel,
                            limit=limit,
                            points=inside[:8] if 0 < len(inside) <= 8 else None)
        total += value
    total = total.reshape((len(rays), len(w)))
    return total[0] if single else total


@dataclass(frozen=True)
class SegmentGeometry:
    """The part of a cone segment this reference needs, in plain numbers."""
    start_m: tuple
    direction: tuple
    length_m: float
    beta: float
    phase_index: float
    photons_per_m: float
    start_time_ns: float = 0.0

    @classmethod
    def of(cls, segment):
        """Adopt a :class:`~lighthit.experimental.cone_segment.ConeSegment`."""
        if isinstance(segment, cls):
            return segment
        return cls(tuple(segment.start_m), tuple(segment.direction),
                   float(segment.length_m), float(segment.beta),
                   float(segment.phase_index), float(segment.photons_per_m),
                   float(segment.start_time_ns))

    @property
    def cone_cosine(self):
        return 1 / (self.beta * self.phase_index)

    @property
    def speed_m_per_ns(self):
        return self.beta * C_VACUUM_M_PER_NS

    def frame(self):
        """The segment axis and two unit vectors across it."""
        u = np.asarray(self.direction, float)
        helper = np.eye(3)[int(np.argmin(np.abs(u)))]
        e1 = np.cross(u, helper)
        e1 /= np.linalg.norm(e1)
        return u, e1, np.cross(u, e1)

    def cone_directions(self, azimuths):
        """Unit emission vectors at the cone angle, one per azimuth."""
        u, e1, e2 = self.frame()
        azimuths = np.atleast_1d(np.asarray(azimuths, float))
        mu = self.cone_cosine
        sine = np.sqrt(max(1 - mu * mu, 0.0))
        return (mu * u[None, :]
                + sine * (np.cos(azimuths)[:, None] * e1[None, :]
                          + np.sin(azimuths)[:, None] * e2[None, :]))

    def ballistic_root(self, detector_m):
        """Emission point whose cone ray reaches the receiver, and its azimuth.

        Returns ``(a, azimuth)``; ``a`` outside ``[0, length)`` means the cone
        never reaches that receiver, and the once-scattered integrand is then
        free of the near-ray peak.
        """
        u, e1, e2 = self.frame()
        relative = np.asarray(detector_m, float) - np.asarray(self.start_m, float)
        along = float(relative @ u)
        across = relative - along * u
        impact = float(np.linalg.norm(across))
        mu = self.cone_cosine
        sine = np.sqrt(max(1 - mu * mu, 0.0))
        azimuth = float(np.arctan2(across @ e2, across @ e1)) if impact > 0 else 0.0
        return along - impact * mu / sine, azimuth


def segment_ballistic_field(points_m, segment, medium: Medium):
    """Unscattered cone fluence at arbitrary points, with its arrival data.

    Returns ``(fluence, incoming_direction, arrival_time)``. The fluence is the
    same quantity :func:`cone_segment.ballistic_spectrum` evaluates at a
    receiver, written here as a field so that it can be scattered once more.
    Points the cone never reaches get zero fluence. The emission root, the
    flight distance, the incoming direction and the time all follow the exact
    straight segment.
    """
    geometry = SegmentGeometry.of(segment)
    x = np.atleast_2d(np.asarray(points_m, float))
    x0 = np.asarray(geometry.start_m, float)
    u = np.asarray(geometry.direction, float)
    mu = geometry.cone_cosine
    sine = np.sqrt(max(1 - mu * mu, 0.0))
    relative = x - x0
    along = relative @ u
    impact = np.linalg.norm(relative - along[:, None] * u, axis=1)
    safe = np.maximum(impact, 1e-300)
    root = along - safe * mu / sine
    distance = safe / sine
    lit = (root >= 0) & (root < geometry.length_m) & (impact > 0)
    fluence = np.where(lit, geometry.photons_per_m
                       * np.exp(-medium.extinction_per_m * distance)
                       / (2 * np.pi * safe * sine), 0.0)
    incoming = (x - (x0 + root[:, None] * u)) / np.maximum(distance, 1e-300)[:, None]
    time = (geometry.start_time_ns + root / geometry.speed_m_per_ns
            + distance / medium.speed_m_per_ns)
    return fluence, incoming, time


def _polar_sectors(root, length, half_width):
    """Polar-angle ranges in which the emission rectangle is visible.

    The emission domain is ``a`` in ``[0, length]`` and one full turn of the
    cone azimuth, the latter rescaled to a transverse length so that the two
    coordinates are comparable. Seen from a centre inside the rectangle every
    direction meets it, and the corners split the circle into four sectors.
    Seen from a centre outside it — a cone that just misses the receiver, where
    the integrand still has that peak sitting beyond the segment end — only one
    arc meets it, and the corners split that arc instead. Both cases are
    returned as a list of ranges, each with a smooth radial extent inside it.
    """
    corners = np.array([(0.0 - root, -half_width), (length - root, -half_width),
                        (length - root, half_width), (0.0 - root, half_width)])
    angles = np.arctan2(corners[:, 1], corners[:, 0])
    inside = (0.0 <= root <= length) and half_width > 0
    if inside:
        ordered = np.sort(angles % (2 * np.pi))
        return [(float(ordered[i]),
                 float(ordered[(i + 1) % 4] + (2 * np.pi if i == 3 else 0.0)))
                for i in range(4)]
    # Outside: unwrap the corner angles around the direction of the rectangle's
    # centre, then the visible arc is simply their span.
    middle = float(np.arctan2(0.0, length / 2 - root))
    shifted = np.sort(middle + np.angle(np.exp(1j * (angles - middle))))
    return [(float(lo), float(hi)) for lo, hi in zip(shifted, shifted[1:]) if hi > lo]


def _polar_extent(psi, root, length, half_width):
    """Entry and exit radii of the rectangle along each direction ``psi``.

    Slab clipping, so it works whether the centre is inside the rectangle
    (entry radius zero) or outside it.
    """
    cos, sin = np.cos(psi), np.sin(psi)
    near = np.zeros(psi.shape)
    far = np.full(psi.shape, np.inf)
    for component, low, high in ((cos, 0.0 - root, length - root),
                                 (sin, -half_width, half_width)):
        with np.errstate(divide="ignore", invalid="ignore"):
            first = np.where(np.abs(component) > 1e-15, low / component, -np.inf)
            second = np.where(np.abs(component) > 1e-15, high / component, np.inf)
        lower = np.minimum(first, second)
        upper = np.maximum(first, second)
        parallel = np.abs(component) <= 1e-15
        outside = parallel & ((low > 0) | (high < 0))
        near = np.maximum(near, np.where(parallel, 0.0, lower))
        far = np.minimum(far, np.where(parallel, np.inf, upper))
        far = np.where(outside, 0.0, far)
    return near, np.maximum(far, near)


def segment_first_order(omega, segment, detector_m, medium: Medium, *,
                        kind="hg", degree=None, longitudinal_order=32,
                        azimuth_order=64, radial_order=24, polar_order=24,
                        product_rule=False, **quadrature):
    """Once-scattered spectrum of a finite straight Cherenkov segment.

    The emission-point and cone-azimuth integrals are explicit; the flight
    integral inside each of them is :func:`directed_first_order`. Every
    emission point carries its own time ``t0 + a / (beta c)``, so the
    longitudinal phase is never averaged.

    When the cone reaches the receiver ballistically, the once-scattered
    integrand has an integrable ``1/rho`` peak at that emission point and
    azimuth: photons scattered by a vanishing angle just before the receiver
    arrive from rays passing arbitrarily close to it. A product rule in
    ``(a, phi)`` converges very slowly against such a peak, and it converges
    slowly again when the peak sits just *outside* the segment, which happens
    whenever the cone almost reaches the receiver. Both cases are handled by
    integrating in polar coordinates centred on that point, inside or outside
    the domain, with the azimuth rescaled into a transverse distance first; the
    polar Jacobian cancels the peak exactly and the rectangle is clipped
    sector by sector. ``radial_order`` and ``polar_order`` control it.

    ``product_rule=True`` selects the plain tensor rule in ``(a, phi)``
    instead, with ``longitudinal_order`` and ``azimuth_order``. It is there as
    an independent second discretisation, not as the default: for geometries
    far from any ballistic root the two agree to many digits, and where they
    do not the polar rule is the one that converges. Convergence must be
    checked by raising the orders, never assumed.
    """
    w = np.atleast_1d(np.asarray(omega, float))
    geometry = SegmentGeometry.of(segment)
    det = np.asarray(detector_m, float)
    x0 = np.asarray(geometry.start_m, float)
    u = np.asarray(geometry.direction, float)
    root, azimuth = geometry.ballistic_root(det)

    def contribution(a, phi, weight):
        """Integrate a batch of emission points and azimuths."""
        sources = x0[None, :] + a[:, None] * u[None, :]
        rays = geometry.cone_directions(phi)
        inner = directed_first_order(w, sources, rays, det, medium,
                                     kind=kind, degree=degree, **quadrature)
        phase = np.exp(1j * w[None, :] * (geometry.start_time_ns
                                          + a[:, None] / geometry.speed_m_per_ns))
        return (weight[:, None] * geometry.photons_per_m * phase * inner).sum(axis=0)

    if product_rule:
        nodes, weights = roots_legendre(int(longitudinal_order))
        half = geometry.length_m / 2
        a = half * (nodes + 1)
        phi = 2 * np.pi * np.arange(int(azimuth_order)) / int(azimuth_order)
        grid_a = np.repeat(a, len(phi))
        grid_phi = np.tile(phi, len(a))
        weight = np.repeat(half * weights, len(phi)) / len(phi)
        return contribution(grid_a, grid_phi, weight)

    # Measure the azimuth as the transverse distance it moves the ray at the
    # receiver, so that one polar radius means the same thing along both axes.
    lever = max(np.linalg.norm(det - (x0 + root * u)), 1e-12) \
        * np.sqrt(max(1 - geometry.cone_cosine ** 2, 0.0))
    half_width = np.pi * lever
    psi_nodes, psi_weights = roots_legendre(int(polar_order))
    rho_nodes, rho_weights = roots_legendre(int(radial_order))
    total = np.zeros(len(w), complex)
    for lo, hi in _polar_sectors(root, geometry.length_m, half_width):
        half = (hi - lo) / 2
        psi = lo + half * (psi_nodes + 1)
        near, far = _polar_extent(psi, root, geometry.length_m, half_width)
        span = far - near
        rho = near[:, None] + span[:, None] * (rho_nodes[None, :] + 1) / 2
        # The polar Jacobian rho is what removes the 1/rho peak; the 1/lever
        # turns the transverse coordinate back into an azimuth measure.
        weight = (half * psi_weights)[:, None] * (span[:, None] / 2) \
            * rho_weights[None, :] * rho / (2 * np.pi * lever)
        a = root + rho * np.cos(psi)[:, None]
        phi = azimuth + rho * np.sin(psi)[:, None] / lever
        keep = weight != 0
        if not np.any(keep):
            continue
        total += contribution(a[keep], phi[keep], weight[keep])
    return total


def segment_first_order_mc(omega, segment, detector_m, medium: Medium, *,
                           kind="hg", degree=None, samples=200000, seed=11,
                           batches=16):
    """Once-scattered segment spectrum from arrival directions, by sampling.

    Writing the last scattering point as ``x' = R - l n`` turns the volume
    element into ``l**2 dl dOmega_n``, which cancels the ``1/l**2`` spreading
    factor of the final flight. What is left,

        K1 = int dOmega_n int dl  mu_s p(s.n) F(x') e^{-mu_t l}
             exp(i omega (t(x') + l / v)),

    is sampled with ``n`` uniform on the sphere and ``l`` exponential with rate
    ``mu_t``. ``F`` and ``t`` come from :func:`segment_ballistic_field`, so
    this route shares no variable, quadrature or series with
    :func:`segment_first_order`. Returns ``(mean, standard_error)`` over
    batches; the ``1/impact`` ridge of the unscattered cone field makes the
    batch spread the honest error estimate, not a Gaussian guarantee.
    """
    w = np.atleast_1d(np.asarray(omega, float))
    geometry = SegmentGeometry.of(segment)
    det = np.asarray(detector_m, float)
    rng = np.random.default_rng(seed)
    speed = medium.speed_m_per_ns
    mu_t = medium.extinction_per_m
    estimates = np.zeros((int(batches), len(w)), complex)
    for batch in range(int(batches)):
        normals = rng.normal(size=(int(samples), 3))
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        ell = rng.exponential(1 / mu_t, int(samples))
        points = det[None, :] - ell[:, None] * normals
        fluence, incoming, time = segment_ballistic_field(points, geometry, medium)
        kernel = phase_function(np.einsum("ij,ij->i", incoming, normals),
                                medium, kind=kind, degree=degree)
        # Uniform sphere density 1/(4 pi) and exponential density mu_t e^{-mu_t l}
        # divide out; the exponential exactly cancels the flight attenuation.
        weight = 4 * np.pi * medium.scattering_per_m * kernel * fluence / mu_t
        estimates[batch] = (weight[:, None]
                            * np.exp(1j * w[None, :] * (time[:, None] + ell[:, None] / speed))
                            ).mean(axis=0)
    mean = estimates.mean(axis=0)
    error = (estimates.real.std(axis=0, ddof=1) + 1j * estimates.imag.std(axis=0, ddof=1))
    return mean, error / np.sqrt(int(batches))


def ballistic_by_ray_counting(segment, detector_m, medium: Medium, *,
                              acceptance_radius_m=0.05, samples=200000, seed=7,
                              azimuth_window=None):
    """Unscattered fluence at a small sphere, counted ray by ray.

    A pure geometry estimate with no factor in common with
    ``cone_segment.ballistic_spectrum``: emission points and cone azimuths are
    sampled, each ray is tested for passage through a sphere of radius ``rho``
    around the receiver, and the surviving weight ``exp(-mu_t d)`` is divided
    by the sphere's cross-section ``pi rho**2``. As ``rho`` shrinks the
    estimate converges to the point-receiver fluence, so it certifies the
    analytic Jacobian to the level of its own bias and statistics, not to
    machine precision.

    Only a narrow band of azimuths can possibly hit a small sphere, so the
    azimuth is drawn from a window around the one aimed at the receiver and
    the weight carries the window's width. The window is sampling only: no
    part of the answer assumes it, and the returned ``edge_fraction`` says how
    close the accepted rays came to its rim. A nonzero edge fraction means the
    window was too narrow and the number is a lower bound.
    """
    rng = np.random.default_rng(seed)
    geometry = SegmentGeometry.of(segment)
    det = np.asarray(detector_m, float)
    x0 = np.asarray(geometry.start_m, float)
    u, e1, e2 = geometry.frame()
    # Only emission points whose cone can pass near the sphere matter. Find
    # that stretch by scanning the smallest reachable impact parameter, which
    # is geometry, not the Jacobian under test, then sample inside it.
    scan = np.linspace(0.0, geometry.length_m, 4001)
    reach = det[None, :] - (x0[None, :] + scan[:, None] * u[None, :])
    span = np.linalg.norm(reach, axis=1)
    opening = np.arccos(np.clip((reach @ u) / span, -1, 1))
    closest = span * np.sin(np.abs(opening - np.arccos(geometry.cone_cosine)))
    near = np.flatnonzero(closest < 8 * acceptance_radius_m)
    if len(near):
        low = max(0.0, scan[near[0]] - 8 * acceptance_radius_m)
        high = min(geometry.length_m, scan[near[-1]] + 8 * acceptance_radius_m)
    else:
        low, high = 0.0, geometry.length_m
    a = rng.uniform(low, high, int(samples))
    gap = det[None, :] - (x0[None, :] + a[:, None] * u[None, :])
    distance = np.linalg.norm(gap, axis=1)
    aim = np.arctan2(gap @ e2, gap @ e1)
    if azimuth_window is None:
        # Generous: eight times the angular radius the sphere subtends, and
        # never less than a milliradian.
        azimuth_window = float(min(np.pi, max(1e-3, 8 * acceptance_radius_m
                                              / max(distance.min(), 1e-9))))
    offset = azimuth_window * (2 * rng.random(int(samples)) - 1)
    rays = geometry.cone_directions(aim + offset)
    along = np.einsum("ij,ij->i", gap, rays)
    impact2 = np.einsum("ij,ij->i", gap, gap) - along ** 2
    hit = (along > 0) & (impact2 < acceptance_radius_m ** 2)
    weight = np.where(hit, np.exp(-medium.extinction_per_m * np.abs(along)), 0.0)
    scale = (geometry.photons_per_m * (high - low)
             * (azimuth_window / np.pi) / (np.pi * acceptance_radius_m ** 2))
    if np.any(hit):
        edge = np.maximum(np.abs(offset[hit]) / azimuth_window,
                          np.where((a[hit] - low) < (high - a[hit]),
                                   1 - 2 * (a[hit] - low) / max(high - low, 1e-30),
                                   1 - 2 * (high - a[hit]) / max(high - low, 1e-30)))
        edge_fraction = float(np.mean(edge > 0.9))
    else:
        edge_fraction = 0.0
    return (scale * weight.mean(),
            scale * weight.std(ddof=1) / np.sqrt(int(samples)),
            edge_fraction)
