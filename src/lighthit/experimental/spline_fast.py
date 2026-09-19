"""Numba-fused replacement for ``ResponseCache.moments_at`` / ``_contract``.

``ResponseCache.moments_at`` (``cache.py:289-339``) rebuilds a
``scipy.interpolate.CubicSpline`` the first time a given ``(degree,
frequency_index)`` key is asked for, and even once that object is cached,
every call still evaluates it and materialises a fresh (frequency, points,
degree, order) array. That function is the shared hot path behind four
different routes: ``KernelChannels.multipoles`` (hence ``direct_response``
and ``evaluate_moments``'s ``KernelChannels.channels``) and
``cone_segment.py``'s ``_scatter``. It is the "spline" item named in the
axial-route profiling of chapter 9, paid again by every one of them.

``PreparedMultipoles`` builds the Horner coefficients of the same
log-radius cubic spline once, over the full degree and frequency range --
the identical construction already used and verified for the axial source in
``axial_blocked.py::BlockedAxialKernel`` and ``axial_fast.py`` -- and
evaluates them with a compiled loop instead of a scipy spline object per
call.

It is a duck-typed drop-in, not a new API: anything that only reads
``cache.degree``, ``cache.grid.omega_per_ns``, ``cache.medium``,
``cache.radius_range_m`` and calls ``cache.moments_at(radii, degrees=...,
frequency_index=...)`` -- ``KernelChannels``, ``direct_response``,
``cone_segment.py``'s ``_scatter`` and ``segment_spectrum``, and
``evaluate_moments`` through ``KernelChannels.channels`` -- accepts a
``PreparedMultipoles`` wherever it accepted the ``ResponseCache`` band it
wraps, with no code of its own changed.

Wraps one band. A :class:`~lighthit.cache.BandedResponseCache` is not
dispatched here; wrap each of its bands separately if that is needed.

Requires numba; there is no numpy fallback here on purpose, for the same
reason ``ballistic_fast`` has none -- a numpy fallback would just be
``ResponseCache.moments_at`` again.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline

try:
    import numba
    from numba import njit, prange
except ImportError as exc:  # pragma: no cover - exercised only without numba
    raise ImportError(
        "spline_fast needs numba (pip install -e '.[accelerate]'); "
        "the uncompiled path is ResponseCache.moments_at"
    ) from exc

__all__ = ["PreparedMultipoles"]


@njit(parallel=True, cache=True)
def _horner_eval(log_radii_grid, coefficients, query_log_radii):
    """Evaluate a per-(degree, order, frequency) cubic Horner polynomial.

    ``coefficients`` is (interval, degree, power, order, frequency); returns
    (frequency, points, degree, order), matching ``ResponseCache.moments_at``.
    """
    n_interval, n_degree, n_power, n_order, n_freq = coefficients.shape
    n_points = query_log_radii.shape[0]
    n_nodes = log_radii_grid.shape[0]
    out = np.zeros((n_freq, n_points, n_degree, n_order), dtype=np.complex128)
    for p in prange(n_points):
        x = query_log_radii[p]
        left = 0
        right = n_nodes - 1
        while left < right - 1:
            mid = (left + right) // 2
            if log_radii_grid[mid] <= x:
                left = mid
            else:
                right = mid
        interval = left
        if interval > n_interval - 1:
            interval = n_interval - 1
        step = x - log_radii_grid[interval]
        for d in range(n_degree):
            for o in range(n_order):
                for w in range(n_freq):
                    acc = coefficients[interval, d, 0, o, w]
                    for power in range(1, n_power):
                        acc = acc * step + coefficients[interval, d, power, o, w]
                    out[w, p, d, o] = acc
    return out


class PreparedMultipoles:
    """Horner-ready radial spline of one ``ResponseCache`` band."""

    def __init__(self, cache, degree, omega_per_ns, log_radii, coefficients,
                 absorption_per_m, speed_m_per_ns, radial_phase):
        self._cache = cache
        self.degree = degree
        self.omega_per_ns = omega_per_ns
        self.log_radii = log_radii
        self.coefficients = coefficients
        self.absorption_per_m = absorption_per_m
        self.speed_m_per_ns = speed_m_per_ns
        self.radial_phase = radial_phase

    @classmethod
    def of(cls, cache, degree=None):
        if hasattr(cache, "bands"):
            raise ValueError(
                "PreparedMultipoles wraps one ResponseCache band; a "
                "BandedResponseCache is not dispatched here, wrap each "
                "band separately")
        degree = cache.degree if degree is None else int(degree)
        if not 0 <= degree <= cache.degree:
            raise ValueError("degree must lie within the cache")
        radii = np.asarray(cache.grid.radii_m, float)
        omega = np.asarray(cache.grid.omega_per_ns, float)
        scale = np.exp(-cache.medium.absorption_per_m * radii) / (4 * np.pi * radii ** 2)
        if not np.isfinite(scale).all() or np.any(scale == 0):
            raise ValueError("radial scale underflowed; narrow the radius range")
        values = cache.moments[:, :, :degree + 1, :] / scale[None, :, None, None]
        phase = getattr(cache, "radial_phase", "none")
        if phase not in ("none", "flight"):
            raise ValueError("unknown radial_phase")
        if phase == "flight":
            values = values * np.exp(
                -1j * omega[:, None] * radii[None, :] / cache.medium.speed_m_per_ns
            )[:, :, None, None]
        spline = CubicSpline(np.log(radii), values, axis=1)
        # spline.c is (power, interval, frequency, degree, order)
        coefficients = np.ascontiguousarray(np.transpose(spline.c, (1, 3, 0, 4, 2)))
        return cls(cache, degree, omega, np.log(radii), coefficients,
                   float(cache.medium.absorption_per_m),
                   float(cache.medium.speed_m_per_ns), phase)

    # -- duck-typed passthroughs, so this wraps wherever the band did -----

    @property
    def grid(self):
        return self._cache.grid

    @property
    def medium(self):
        return self._cache.medium

    @property
    def radius_range_m(self):
        return self._cache.radius_range_m

    @property
    def moments(self):
        return self._cache.moments

    # -- the accelerated call ---------------------------------------------

    def moments_at(self, radii_m, *, degrees=None, frequency_index=None):
        """Drop-in for ``ResponseCache.moments_at``, same shape and math."""
        radii = np.atleast_1d(np.asarray(radii_m, float))
        if radii.ndim != 1 or not len(radii) or not np.isfinite(radii).all() or np.any(radii <= 0):
            raise ValueError("radii_m must be a nonempty finite positive 1-D array")
        low, high = self.radius_range_m
        if np.any(radii < low) or np.any(radii > high):
            raise ValueError(
                f"radii_m outside the cached range [{low:g}, {high:g}] m; "
                "extrapolation is not provided")
        top = self.degree + 1 if degrees is None else int(degrees)
        if not 1 <= top <= self.degree + 1:
            raise ValueError("degrees must be between 1 and spatial_degree+1")
        if frequency_index is None:
            sel = slice(None)
        else:
            index = int(frequency_index)
            if index >= len(self.omega_per_ns):
                raise ValueError("frequency_index outside the cache")
            sel = slice(index, index + 1)
        coeff = np.ascontiguousarray(self.coefficients[:, :top, :, :, sel])
        out = _horner_eval(self.log_radii, coeff, np.log(radii))
        scale = np.exp(-self.absorption_per_m * radii) / (4 * np.pi * radii ** 2)
        out = out * scale[None, :, None, None]
        if self.radial_phase == "flight":
            freq_vals = self.omega_per_ns[sel]
            out = out * np.exp(
                1j * freq_vals[:, None] * radii[None, :] / self.speed_m_per_ns
            )[:, :, None, None]
        return out
