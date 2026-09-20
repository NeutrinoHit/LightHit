"""Numba-fused unscattered (ballistic) response -- same closed form, no arrays.

``scripts/run_shower_moments.py::vectorised_ballistic`` already evaluates the
exact, closed-form order-0 term of @eq-ballistic-fluence -- no cache lookup,
no angular sum, one root per (element, receiver) pair. But it is still written
as a numpy reduction over all elements *per receiver*, which means every one
of ``along``, ``impact``, ``root``, ``distance`` and ``weight`` is a full
221 393-long array, allocated and read back from RAM, 576 times over. That is
the same memory-bound pattern diagnosed for the scattered orders -- just with
nothing to interpolate, so there is no reason for it to cost more than a
handful of flops per pair.

This module is that formula as one fused loop: the running sum for a receiver
lives in a scalar accumulator, nothing indexed by (element, receiver) is ever
materialised, and both loops (receivers, elements) are compiled so neither is
a Python-level iteration. Requires numba; there is no numpy fallback here on
purpose, because a numpy fallback would just be ``vectorised_ballistic``
again under a different name.

The physics is untouched -- same root ``a*``, same ``1/(2 pi b sin theta_C)``,
same ``exp(-mu_t d*)`` -- only where the intermediate numbers live changes.
"""
from __future__ import annotations

import numpy as np

try:
    import numba
    from numba import njit, prange
except ImportError as exc:  # pragma: no cover - exercised only without numba
    raise ImportError(
        "ballistic_fast needs numba (pip install 'lighthit[accelerate]'); "
        "the uncompiled formula lives in "
        "scripts/run_shower_moments.py:vectorised_ballistic"
    ) from exc

__all__ = ["vectorised_ballistic_fast", "vectorised_ballistic_bins_fast",
           "ballistic_directional_fast"]


@njit(parallel=True, cache=True, fastmath=False)
def _ballistic_kernel(receivers, start, direction, length, photons,
                       cone_cosine, sine, extinction_per_m):
    n_recv = receivers.shape[0]
    n_el = start.shape[0]
    total = np.zeros(n_recv, dtype=np.float64)
    two_pi = 2.0 * np.pi
    for r in prange(n_recv):
        rx = receivers[r, 0]
        ry = receivers[r, 1]
        rz = receivers[r, 2]
        acc = 0.0
        for i in range(n_el):
            dx = rx - start[i, 0]
            dy = ry - start[i, 1]
            dz = rz - start[i, 2]
            ux = direction[i, 0]
            uy = direction[i, 1]
            uz = direction[i, 2]
            along = dx * ux + dy * uy + dz * uz
            dot = dx * dx + dy * dy + dz * dz
            impact2 = dot - along * along
            if impact2 < 1e-300:
                impact2 = 1e-300
            impact = np.sqrt(impact2)
            s = sine[i]
            mu = cone_cosine[i]
            root = along - impact * mu / s
            len_i = length[i]
            if root < 0.0 or root >= len_i:
                continue
            distance = impact / s
            safe_len = len_i if len_i > 1e-300 else 1e-300
            acc += (photons[i] / safe_len
                    * np.exp(-extinction_per_m * distance)
                    / (two_pi * impact * s))
        total[r] = acc
    return total


@njit(parallel=True, cache=True, fastmath=False)
def _ballistic_bins_kernel(receivers, start, direction, length, photons,
                           cone_cosine, sine, start_ns, end_ns,
                           extinction_per_m, speed_m_per_ns, origins, edges):
    """Accumulate exact ballistic charge and arrival bins in one element pass."""
    n_recv = receivers.shape[0]
    n_el = start.shape[0]
    n_bins = len(edges) - 1
    total = np.zeros(n_recv, dtype=np.float64)
    bins = np.zeros((n_recv, n_bins), dtype=np.float64)
    two_pi = 2.0 * np.pi
    for r in prange(n_recv):
        rx = receivers[r, 0]
        ry = receivers[r, 1]
        rz = receivers[r, 2]
        acc = 0.0
        for i in range(n_el):
            dx = rx - start[i, 0]
            dy = ry - start[i, 1]
            dz = rz - start[i, 2]
            ux = direction[i, 0]
            uy = direction[i, 1]
            uz = direction[i, 2]
            along = dx * ux + dy * uy + dz * uz
            dot = dx * dx + dy * dy + dz * dz
            impact2 = dot - along * along
            if impact2 < 1e-300:
                impact2 = 1e-300
            impact = np.sqrt(impact2)
            s = sine[i]
            mu = cone_cosine[i]
            root = along - impact * mu / s
            len_i = length[i]
            if root < 0.0 or root >= len_i:
                continue
            distance = impact / s
            safe_len = len_i if len_i > 1e-300 else 1e-300
            weight = (photons[i] / safe_len
                      * np.exp(-extinction_per_m * distance)
                      / (two_pi * impact * s))
            acc += weight
            fraction = root / safe_len
            arrival = (start_ns[i] + fraction * (end_ns[i] - start_ns[i])
                       + distance / speed_m_per_ns - origins[r])
            # searchsorted(edges, arrival, side="right") - 1, written here so
            # no temporary arrival array or atomic histogram is needed.
            left = 0
            right = len(edges)
            while left < right:
                middle = (left + right) // 2
                if arrival < edges[middle]:
                    right = middle
                else:
                    left = middle + 1
            index = left - 1
            if 0 <= index < n_bins:
                bins[r, index] += weight
        total[r] = acc
    return total, bins


def vectorised_ballistic_fast(elements, receivers, medium, cone_cosine):
    """Drop-in, numba-fused replacement for ``vectorised_ballistic``.

    Same signature, same return (one real number per receiver, the ballistic
    charge of @eq-ballistic-fluence summed over every element). Verified
    against the numpy version to float64 precision on synthetic data in
    ``tests/test_ballistic_fast.py``.
    """
    receivers = np.ascontiguousarray(receivers, dtype=np.float64)
    if receivers.ndim != 2 or receivers.shape[1] != 3:
        raise ValueError("receivers must be (n, 3)")
    sine = np.sqrt(np.maximum(1.0 - np.asarray(cone_cosine, dtype=np.float64) ** 2, 0.0))
    start = np.ascontiguousarray(elements.start_m, dtype=np.float64)
    direction = np.ascontiguousarray(elements.direction, dtype=np.float64)
    length = np.ascontiguousarray(elements.length_m, dtype=np.float64)
    photons = np.ascontiguousarray(elements.photons, dtype=np.float64)
    cone_cosine = np.ascontiguousarray(cone_cosine, dtype=np.float64)
    if not (len(start) == len(direction) == len(length) == len(photons)
            == len(cone_cosine)):
        raise ValueError("elements arrays and cone_cosine must share one length")
    return _ballistic_kernel(receivers, start, direction, length, photons,
                              cone_cosine, sine, float(medium.extinction_per_m))


def vectorised_ballistic_bins_fast(elements, receivers, medium, cone_cosine,
                                   time_origin_ns, relative_edges_ns):
    """Exact ballistic charge and per-OM relative-time histogram.

    The same closed cone root and fluence as :func:`vectorised_ballistic_fast`
    are evaluated once.  Arrival time uses the element's linearly interpolated
    stored emission time plus the optical flight time.  Values outside the
    requested window remain in the returned integrated charge and are not
    silently folded into an edge bin.
    """
    receivers = np.ascontiguousarray(receivers, dtype=np.float64)
    if receivers.ndim != 2 or receivers.shape[1] != 3:
        raise ValueError("receivers must be (n, 3)")
    origins = np.ascontiguousarray(time_origin_ns, dtype=np.float64)
    edges = np.ascontiguousarray(relative_edges_ns, dtype=np.float64)
    if origins.shape != (len(receivers),) or not np.isfinite(origins).all():
        raise ValueError("time_origin_ns must contain one finite value per receiver")
    if (edges.ndim != 1 or len(edges) < 2 or not np.isfinite(edges).all()
            or np.any(np.diff(edges) <= 0)):
        raise ValueError("relative_edges_ns must be finite and strictly increasing")
    cone_cosine = np.ascontiguousarray(cone_cosine, dtype=np.float64)
    sine = np.sqrt(np.maximum(1.0 - cone_cosine ** 2, 0.0))
    arrays = [np.ascontiguousarray(value, dtype=np.float64) for value in (
        elements.start_m, elements.direction, elements.length_m, elements.photons,
        elements.start_ns, elements.end_ns)]
    start, direction, length, photons, start_ns, end_ns = arrays
    if not all(len(value) == len(start) for value in
               (direction, length, photons, cone_cosine, start_ns, end_ns)):
        raise ValueError("elements arrays and cone_cosine must share one length")
    return _ballistic_bins_kernel(
        receivers, start, direction, length, photons, cone_cosine, sine,
        start_ns, end_ns, float(medium.extinction_per_m),
        float(medium.speed_m_per_ns), origins, edges)


@njit(cache=True, nogil=True, inline="always")
def _acceptance(alpha, x):
    """``sum_l alpha_l (2l+1) P_l(x) / (4 pi)``, by the Legendre recurrence."""
    total = alpha[0] / (4.0 * np.pi)
    if len(alpha) == 1:
        return total
    previous = 1.0
    current = x
    total += alpha[1] * 3.0 * current / (4.0 * np.pi)
    for ell in range(2, len(alpha)):
        value = ((2 * ell - 1) * x * current - (ell - 1) * previous) / ell
        previous = current
        current = value
        total += alpha[ell] * (2 * ell + 1) * value / (4.0 * np.pi)
    return total


@njit(parallel=True, cache=True, fastmath=False)
def _ballistic_directional_kernel(receivers, looks, start, direction, length,
                                  photons, cone_cosine, sine, extinction_per_m,
                                  alpha, speed_m_per_ns, start_ns, end_ns,
                                  origins, edges, want_bins):
    """Same closed form as ``_ballistic_bins_kernel``, with the module response
    evaluated at the exact arrival direction of each element's cone photon."""
    n_recv = receivers.shape[0]
    n_el = start.shape[0]
    n_bins = len(edges) - 1
    total = np.zeros(n_recv, dtype=np.float64)
    bins = np.zeros((n_recv, n_bins if want_bins else 1), dtype=np.float64)
    two_pi = 2.0 * np.pi
    for r in prange(n_recv):
        rx = receivers[r, 0]
        ry = receivers[r, 1]
        rz = receivers[r, 2]
        nx = looks[r, 0]
        ny = looks[r, 1]
        nz = looks[r, 2]
        acc = 0.0
        for i in range(n_el):
            dx = rx - start[i, 0]
            dy = ry - start[i, 1]
            dz = rz - start[i, 2]
            ux = direction[i, 0]
            uy = direction[i, 1]
            uz = direction[i, 2]
            along = dx * ux + dy * uy + dz * uz
            dot = dx * dx + dy * dy + dz * dz
            impact2 = dot - along * along
            if impact2 < 1e-300:
                impact2 = 1e-300
            impact = np.sqrt(impact2)
            s = sine[i]
            mu = cone_cosine[i]
            root = along - impact * mu / s
            len_i = length[i]
            if root < 0.0 or root >= len_i:
                continue
            distance = impact / s
            safe_len = len_i if len_i > 1e-300 else 1e-300
            # exact incoming direction of this element's cone photon
            sx = (dx - root * ux) / distance
            sy = (dy - root * uy) / distance
            sz = (dz - root * uz) / distance
            cosine = sx * nx + sy * ny + sz * nz
            if cosine > 1.0:
                cosine = 1.0
            elif cosine < -1.0:
                cosine = -1.0
            weight = (photons[i] / safe_len
                      * np.exp(-extinction_per_m * distance)
                      / (two_pi * impact * s)
                      * _acceptance(alpha, cosine))
            acc += weight
            if want_bins:
                fraction = root / safe_len
                arrival = (start_ns[i] + fraction * (end_ns[i] - start_ns[i])
                           + distance / speed_m_per_ns - origins[r])
                left = 0
                right = len(edges)
                while left < right:
                    middle = (left + right) // 2
                    if arrival < edges[middle]:
                        right = middle
                    else:
                        left = middle + 1
                index = left - 1
                if 0 <= index < n_bins:
                    bins[r, index] += weight
        total[r] = acc
    return total, bins


def ballistic_directional_fast(elements, receivers, look_directions, medium,
                               cone_cosine, alpha, *, time_origin_ns=None,
                               relative_edges_ns=None):
    """Fused twin of :func:`lighthit.ballistic.ballistic_directional`."""
    receivers = np.ascontiguousarray(receivers, dtype=np.float64)
    looks = np.ascontiguousarray(look_directions, dtype=np.float64)
    if receivers.ndim != 2 or receivers.shape[1] != 3 or not len(receivers):
        raise ValueError("receivers must be a nonempty (n, 3) array")
    if looks.shape != receivers.shape:
        raise ValueError("look_directions must match receivers")
    if not np.allclose(np.linalg.norm(looks, axis=1), 1.0, rtol=0, atol=1e-9):
        raise ValueError("look_directions must be unit vectors")
    cone_cosine = np.ascontiguousarray(cone_cosine, dtype=np.float64)
    sine = np.sqrt(np.maximum(1.0 - cone_cosine ** 2, 0.0))
    start = np.ascontiguousarray(elements.start_m, dtype=np.float64)
    direction = np.ascontiguousarray(elements.direction, dtype=np.float64)
    length = np.ascontiguousarray(elements.length_m, dtype=np.float64)
    photons = np.ascontiguousarray(elements.photons, dtype=np.float64)
    if not (len(start) == len(direction) == len(length) == len(photons)
            == len(cone_cosine)):
        raise ValueError("elements arrays and cone_cosine must share one length")
    want_bins = time_origin_ns is not None and relative_edges_ns is not None
    if want_bins:
        origins = np.ascontiguousarray(time_origin_ns, dtype=np.float64)
        edges = np.ascontiguousarray(relative_edges_ns, dtype=np.float64)
        if origins.shape != (len(receivers),) or not np.isfinite(origins).all():
            raise ValueError("time_origin_ns must contain one finite value per receiver")
        if (edges.ndim != 1 or len(edges) < 2 or not np.isfinite(edges).all()
                or np.any(np.diff(edges) <= 0)):
            raise ValueError("relative_edges_ns must be finite and strictly increasing")
        start_ns = np.ascontiguousarray(elements.start_ns, dtype=np.float64)
        end_ns = np.ascontiguousarray(elements.end_ns, dtype=np.float64)
    else:
        origins = np.zeros(len(receivers))
        edges = np.array([0.0, 1.0])
        start_ns = np.zeros(len(start))
        end_ns = np.zeros(len(start))
    charge, bins = _ballistic_directional_kernel(
        receivers, looks, start, direction, length, photons, cone_cosine, sine,
        float(medium.extinction_per_m), np.ascontiguousarray(alpha, np.float64),
        float(medium.speed_m_per_ns), start_ns, end_ns, origins, edges,
        bool(want_bins))
    return charge, (bins if want_bins else None)
