"""Optional compiled evaluation of the free Legendre moments and ratios.

Only degree sweeps are compiled. No scattering/spatial truncations, threshold
constants or Fourier conventions are changed. Each k node has its own Miller
start degree, unlike the worst-case start shared by the NumPy batch.

This module is imported only after the user explicitly selects ``numba``.
"""
import numpy as np
from numba import njit


@njit(cache=True, fastmath=False)
def _nodewise(kk, z, decay, eta, f0, norm, ratio_norm, N):
    """Scalar recurrence per independent k node; no parallel thread pool."""
    count = len(kk)
    moments = np.empty((count, N + 1), np.complex128)
    ratios = np.empty((count, N + 1), np.complex128)
    for i in range(count):
        if eta[i] * (N + 3) < 3:
            previous = f0[i]
            current = z[i] * previous + 2 / (1j * kk[i])
            moments[i, 0] = previous * norm[0]
            ratios[i, 0] = (current / previous) * ratio_norm[0]
            for ell in range(1, N + 1):
                following = ((2 * ell + 1) * z[i] * current - ell * previous) / (ell + 1)
                moments[i, ell] = current * norm[ell]
                ratios[i, ell] = (following / current) * ratio_norm[ell]
                previous, current = current, following
        else:
            # Same 32/28 safety rule as NumPy, applied per node rather than to
            # eta.min() over the whole batch. Do not replace these by fit values.
            top = N + 1 + max(32, int(np.ceil(28 / eta[i])))
            ratio = decay[i] * (1 - 0.5 / (top + 1))
            for ell in range(top, 0, -1):
                ratio = ell / ((2 * ell + 1) * z[i] - (ell + 1) * ratio)
                if ell <= N + 1:
                    ratios[i, ell - 1] = ratio
            product = 1.0 + 0.0j
            for ell in range(N + 1):
                moments[i, ell] = f0[i] * product * norm[ell]
                product *= ratios[i, ell]
                ratios[i, ell] *= ratio_norm[ell]
    return moments, ratios


def free_moments_and_ratios(k, d0, N):
    """Validated k, d0, N from angular.py; returns Fortran-contiguous arrays.

    b_l = sqrt((2*l+1)/2) F_l, ratio_l = b_(l+1)/b_l. In the Miller
    branch ratios are computed before b can underflow; no division of tiny
    computed moments is used there. The k=0 result is handled explicitly.
    """
    b = np.zeros((len(k), N + 1), dtype=complex, order="F")
    ratios = np.zeros_like(b, order="F")
    positive = k > 0
    b[~positive, 0] = np.sqrt(2) / d0
    if not np.any(positive):
        return b, ratios
    kk = k[positive]
    z = 1j * d0 / kk
    root = np.sqrt(z - 1) * np.sqrt(z + 1)
    decay = 1 / (z + root)
    decay = np.where(abs(decay) > 1, 1 / decay, decay)
    eta = -np.log(np.abs(decay))
    f0 = 2 * np.arctan(kk / d0) / kk
    ell = np.arange(N + 1)
    norm = np.sqrt((2 * ell + 1) / 2)
    ratio_norm = np.sqrt((2 * ell + 3) / (2 * ell + 1))
    b[positive], ratios[positive] = _nodewise(kk, z, decay, eta, f0, norm, ratio_norm, N)
    return b, ratios
