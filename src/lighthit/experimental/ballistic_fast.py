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
        "ballistic_fast needs numba (pip install -e '.[accelerate]'); "
        "the uncompiled formula lives in "
        "scripts/run_shower_moments.py:vectorised_ballistic"
    ) from exc

__all__ = ["vectorised_ballistic_fast"]


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
