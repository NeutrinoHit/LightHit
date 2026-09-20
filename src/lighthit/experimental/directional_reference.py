r"""Independent references for the directional-OM kernel.

Two of them, at two different distances from the production path.

``khat_quadrature_response`` keeps the azimuthal-block solve of
:mod:`lighthit.angular` and replaces everything above it -- the
Clebsch-Gordan contraction, the parity rule, the collapse into the ``r_hat``
frame -- by an explicit numerical integral over the direction of ``k``, with
the source and the module axis rotated into each node's own frame by hand.
It is the check of the *factorisation*.

``dense_angular_reference`` goes further and drops the blocks as well. It
builds the streaming operator ``<Y_lm| s |Y_l'm'>`` once by quadrature over the
photon direction, assembles the full lab-frame matrix for every ``(k, k_hat)``
and inverts it. Nothing in it knows that the problem separates in ``m``, that
``a_lm = sqrt((l^2-m^2)/(4l^2-1))``, or that a parity rule exists. It is
affordable only at small degree, which is exactly where it is wanted.

``axisymmetric_reference`` covers the special case in which the source axis,
the module axis and the displacement all coincide: the azimuthal integral is
then trivial and the whole response is a one-dimensional quadrature.

None of these is a production path and none is optimised.
"""
import numpy as np
from scipy.special import roots_legendre

from ..angular import (_degree, coupling_coefficients, resolvent_rows)
from ..cache import MOMENT_ORDERS
from .event_moments import real_spherical_harmonics

__all__ = ["spherical_quadrature", "khat_quadrature_response",
           "dense_angular_reference", "axisymmetric_reference"]


def spherical_quadrature(polar_order=32, azimuth_count=None):
    """Gauss-Legendre in ``cos(theta)`` times the uniform rule in azimuth.

    Exact for every spherical harmonic of degree below ``2*polar_order`` and of
    azimuthal order below ``azimuth_count``; the uniform azimuthal rule is the
    spectrally exact one on a circle. A Lebedev set would be the same object
    with fewer nodes and a lower exactness degree, and is not worth a table
    here: the reference is allowed to be slow.
    """
    polar_order = _degree(polar_order, "polar_order")
    if polar_order < 1:
        raise ValueError("polar_order must be positive")
    azimuth = 2 * polar_order + 1 if azimuth_count is None else int(azimuth_count)
    if azimuth < 1:
        raise ValueError("azimuth_count must be positive")
    x, w = roots_legendre(polar_order)
    phi = 2 * np.pi * (np.arange(azimuth) + 0.5) / azimuth
    sine = np.sqrt(np.clip(1 - x * x, 0, None))
    nodes = np.stack(
        (np.outer(sine, np.cos(phi)).ravel(),
         np.outer(sine, np.sin(phi)).ravel(),
         np.repeat(x, azimuth)), axis=1)
    weights = np.repeat(w, azimuth) * (2 * np.pi / azimuth)
    return nodes, weights


def _frame_from_axis(axis):
    """An orthonormal frame whose third vector is ``axis``."""
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis, axis=-1, keepdims=True)
    helper = np.where(np.abs(axis[..., 2:3]) > 0.9,
                      np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]))
    first = np.cross(helper, axis)
    first = first / np.linalg.norm(first, axis=-1, keepdims=True)
    return first, np.cross(axis, first), axis


def khat_quadrature_response(medium, settings, omega_per_ns, source_moments,
                             displacement_m, look_direction, alpha, *,
                             acceptance_degree=None, polar_order=40,
                             backend="numpy"):
    r"""Response of one point source at the origin, by explicit ``k_hat`` quadrature.

    ``source_moments`` are the real lab-frame moments ``S_lm`` of the emission
    distribution, flat in the ``l*l + l + m`` order of
    :func:`~lighthit.experimental.event_moments.real_spherical_harmonics`.
    Returns the two scattered orders of :data:`~lighthit.cache.MOMENT_ORDERS`.
    """
    moments = np.asarray(source_moments, float)
    size = int(round(np.sqrt(len(moments))))
    if size * size != len(moments):
        raise ValueError("source_moments must have length (degree+1)**2")
    degree = size - 1
    L_A = degree if acceptance_degree is None else _degree(acceptance_degree,
                                                           "acceptance_degree")
    alpha = np.asarray(alpha, float)
    displacement = np.asarray(displacement_m, float)
    look = np.asarray(look_direction, float)
    look = look / np.linalg.norm(look)
    k, kw = settings.quadrature()
    first, multiple = resolvent_rows(k, omega_per_ns, medium,
                                     settings.scattering_degree, degree, L_A,
                                     backend=backend)
    nodes, weights = spherical_quadrature(polar_order)
    # The module axis and the source moments must be expressed in ONE frame per
    # node -- the sum over nu is invariant under the choice of transverse axes,
    # but only if both factors use the same choice.
    from ..directional import real_rotation_rows, _rotate_into_frame
    _, cosine, phi = _node_angles(nodes)
    look_local = _rotate_into_frame(
        np.broadcast_to(look, (len(nodes), 3)), cosine, phi)
    look_harmonics = real_spherical_harmonics(L_A, look_local)
    # rotate the source moments into each node frame: S'_lnu = sum_m E_lnu_m S_lm
    rows = real_rotation_rows(degree, L_A, degree, cosine, phi)
    rotated = np.zeros((len(nodes), degree + 1, 2 * L_A + 1))
    for l in range(degree + 1):
        block = moments[l * l:(l + 1) * (l + 1)]
        rotated[:, l] = rows[:, l, :, degree - l:degree + l + 1] @ block
    carrier = np.exp(1j * k[:, None] * (nodes @ displacement)[None, :])
    total = np.zeros(len(MOMENT_ORDERS), complex)
    weight = (kw * k * k) / (2 * np.pi) ** 3
    for axis, rows_by_block in enumerate((first, multiple)):
        value = np.zeros((len(k), len(nodes)), complex)
        for (m, lam), block in rows_by_block.items():
            for sign in ((0,) if m == 0 else (+1, -1)):
                nu = sign * m
                if abs(nu) > lam:
                    continue
                contribution = (block[:, :degree + 1]
                                @ rotated[:, :, nu + L_A].T)   # (k, nodes)
                value += (alpha[lam] * look_harmonics[:, lam * lam + lam + nu]
                          [None, :] * contribution)
        total[axis] = np.sum(weight[:, None] * weights[None, :] * carrier * value)
    return total


def _node_angles(nodes):
    radius = np.linalg.norm(nodes, axis=1)
    cosine = np.clip(nodes[:, 2] / radius, -1.0, 1.0)
    return radius, cosine, np.arctan2(nodes[:, 1], nodes[:, 0])


def _batched_solve(matrices, vectors):
    rhs = np.broadcast_to(np.asarray(vectors, complex),
                          matrices.shape[:-1])[..., None]
    return np.linalg.solve(matrices, rhs)[..., 0]


def _direction_operator(degree, quadrature_order=64):
    """``<Yreal_lm | s_a | Yreal_l'm'>`` for ``a = x, y, z``, by quadrature.

    Built without any recurrence coefficient, so a reference that uses it knows
    nothing about ``a_lm``.
    """
    nodes, weights = spherical_quadrature(quadrature_order)
    harmonics = real_spherical_harmonics(degree, nodes)
    return np.einsum("na,ni,nj->aij", nodes * weights[:, None], harmonics,
                     harmonics, optimize=True)


def dense_angular_reference(medium, settings, omega_per_ns, source_moments,
                            displacement_m, look_direction, alpha, *,
                            acceptance_degree=None, polar_order=24,
                            quadrature_order=64):
    """Full lab-frame dense solve, no azimuthal blocks anywhere.

    Only for small degrees: the cost is one dense solve per ``(k, k_hat)`` node
    of size ``(degree+1)**2``.
    """
    moments = np.asarray(source_moments, float)
    size = int(round(np.sqrt(len(moments))))
    if size * size != len(moments):
        raise ValueError("source_moments must have length (degree+1)**2")
    degree = size - 1
    L_A = degree if acceptance_degree is None else _degree(acceptance_degree,
                                                           "acceptance_degree")
    alpha = np.asarray(alpha, float)
    displacement = np.asarray(displacement_m, float)
    look = np.asarray(look_direction, float)
    look = look / np.linalg.norm(look)
    k, kw = settings.quadrature()
    nodes, node_weights = spherical_quadrature(polar_order)
    operator = _direction_operator(degree, quadrature_order)
    d0 = medium.extinction_per_m - 1j * omega_per_ns / medium.speed_m_per_ns
    ell = np.concatenate([np.full(2 * l + 1, l) for l in range(degree + 1)])
    gamma = np.where(ell <= settings.scattering_degree,
                     medium.scattering_per_m * medium.g ** ell, 0.0)
    look_harmonics = real_spherical_harmonics(L_A, look[None, :])[0]
    detector = np.zeros(len(moments))
    for lam in range(L_A + 1):
        for mu in range(-lam, lam + 1):
            detector[lam * lam + lam + mu] = alpha[lam] * look_harmonics[
                lam * lam + lam + mu]
    weight = (kw * k * k) / (2 * np.pi) ** 3
    total = np.zeros(len(MOMENT_ORDERS), complex)
    free_diag = np.diag(d0 * np.ones(len(moments))).astype(complex)
    full_diag = np.diag(d0 - gamma).astype(complex)
    identity = np.eye(len(moments))
    for index, node in enumerate(nodes):
        streaming = np.einsum("a,aij->ij", node, operator)
        phase = np.exp(1j * k * float(node @ displacement))
        stack = 1j * k[:, None, None] * streaming[None, :, :]
        free = free_diag[None, :, :] + stack
        full = full_diag[None, :, :] + stack
        g0 = _batched_solve(free, moments)
        first = _batched_solve(free, gamma[None, :] * g0)
        g = _batched_solve(full, moments)
        rest = g - g0 - first
        scale = weight * node_weights[index] * phase
        total[0] += np.sum(scale * (first @ detector))
        total[1] += np.sum(scale * (rest @ detector))
    return total


def axisymmetric_reference(medium, settings, omega_per_ns, moments_m0,
                           radius_m, alpha, *, acceptance_degree=None,
                           polar_order=400, backend="numpy"):
    r"""Source axis, module axis and displacement all along one line.

    ``moments_m0[l]`` is the real ``m = 0`` moment of the emission distribution
    about that common axis, and the module looks along the same axis. Every
    node frame then differs from the lab frame by a single polar angle, the
    azimuthal integral over ``k_hat`` is exactly ``2 pi``, and the response
    becomes a product of two one-dimensional quadratures.

    Note what does *not* simplify: the azimuthal blocks ``nu != 0`` still
    contribute, because the polar axis of each node frame is ``k_hat`` and not
    the symmetry axis. A reference that dropped them would be wrong, and the
    test that uses this function is what says so.
    """
    moments = np.asarray(moments_m0, float)
    degree = len(moments) - 1
    L_A = degree if acceptance_degree is None else _degree(acceptance_degree,
                                                           "acceptance_degree")
    alpha = np.asarray(alpha, float)
    k, kw = settings.quadrature()
    first, multiple = resolvent_rows(k, omega_per_ns, medium,
                                     settings.scattering_degree, degree, L_A,
                                     backend=backend)
    x, w = roots_legendre(int(polar_order))
    sine = np.sqrt(np.clip(1 - x * x, 0.0, None))
    # z of the lab frame, as seen in the node frame: (-sin theta, 0, cos theta)
    seen = np.stack((-sine, np.zeros_like(x), x), axis=1)
    harmonics = real_spherical_harmonics(max(degree, L_A), seen)
    ell = np.arange(degree + 1)
    scale = np.sqrt(4 * np.pi / (2 * ell + 1))
    total = np.zeros(len(MOMENT_ORDERS), complex)
    weight = (kw * k * k) / (2 * np.pi) ** 3
    phase = np.exp(1j * k[:, None] * radius_m * x[None, :])
    for axis, rows in enumerate((first, multiple)):
        value = np.zeros((len(k), len(x)), complex)
        for (m, lam), block in rows.items():
            for sign in ((0,) if m == 0 else (+1, -1)):
                nu = sign * m
                if abs(nu) > lam:
                    continue
                source_nodes = (harmonics[:, ell * ell + ell + nu]
                                * (moments * scale)[None, :])
                source_nodes[:, ell < abs(nu)] = 0.0
                value += (alpha[lam] * harmonics[:, lam * lam + lam + nu][None, :]
                          * (block[:, :degree + 1] @ source_nodes.T))
        total[axis] = 2 * np.pi * np.sum(
            weight[:, None] * w[None, :] * phase * value)
    return total
