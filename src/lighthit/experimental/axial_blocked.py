"""The compact-source apply, reordered, without a compiled backend.

``axial_fast`` needs numba. This module performs the same reordering in numpy,
and exists to answer one question: how much of that route's speed is the new
*order* of operations and how much is compilation?

The answer, measured on the stored event at full size -- 4935 cells, 277
channels, 41 frequencies, both scattered orders -- is that the reordering on
its own is worth **nothing**: 3.4 s per module against 3.0 s for the
straightforward version, agreeing to 4e-15. Every saving in the compiled route
comes from never materialising the intermediates at all, which numpy cannot
express: an array indexed per (receiver, cell, degree, order, frequency) has to
exist in numpy and does not have to exist in a fused loop.

So this module is a measurement and a fallback, not a fast path. It is kept
because the negative result is worth having written down, and because it
returns both scattered orders from one call, which the straightforward route
does not.

Three changes, all of them exact:

1. **The radial moment is never expanded over m.** The cache holds one
   ``M_l`` per degree; the straightforward version indexes it up to the
   ``(l, m)`` channel list, an 8.4-fold duplication at ``L_q=32, M=4`` that is
   then multiplied and thrown away. Here the azimuthal channels are contracted
   first, so the degree axis stays 33 long.

2. **Both scattered orders come out of one pass.** They share the distances,
   the directions, the harmonics and the angular contraction; only the radial
   factor differs, and the cache stores both on the same grid.

3. **The radial interpolation is a Horner evaluation on prepared coefficients**
   rather than a spline object called once per frequency, so the long array of
   interpolated moments is never built.

Nothing about the physics, the grids or the truncations changes: this returns
the same numbers as :func:`~lighthit.experimental.axial_source.axial_response`
to rounding.
"""
from dataclasses import dataclass
import numpy as np
from scipy.interpolate import CubicSpline

from .axial_source import channel_index
from .event_moments import real_spherical_harmonics

__all__ = ["BlockedAxialKernel"]


@dataclass(frozen=True)
class BlockedAxialKernel:
    """Radial spline coefficients of a cache, laid out for Horner evaluation."""
    degree: int
    omega_per_ns: np.ndarray
    log_radii: np.ndarray
    coefficients: np.ndarray      # (interval, degree, power, order, frequency)
    absorption_per_m: float
    speed_m_per_ns: float
    radial_phase: str

    @classmethod
    def of(cls, cache, degree=None):
        if hasattr(cache, "bands"):
            raise ValueError("pass a single ResponseCache; bands are not dispatched here")
        degree = cache.degree if degree is None else int(degree)
        if not 0 <= degree <= cache.degree:
            raise ValueError("degree must lie within the cache")
        radii = np.asarray(cache.grid.radii_m, float)
        omega = np.asarray(cache.grid.omega_per_ns, float)
        # The same detrending the cache itself interpolates on, so that what is
        # splined is smooth and the Horner step below is the cache's own answer.
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
        return cls(degree, omega, np.log(radii), coefficients,
                   float(cache.medium.absorption_per_m),
                   float(cache.medium.speed_m_per_ns), phase)

    def apply(self, source, receivers_m, *, cell_block=512, receiver_block=8):
        """Both scattered orders at once, shape ``(frequency, receiver, 2)``."""
        receivers = np.atleast_2d(np.asarray(receivers_m, float))
        if self.degree > source.degree:
            raise ValueError("the source carries fewer degrees than the kernel")
        kept, degrees = channel_index(self.degree, min(source.azimuthal_degree,
                                                       self.degree))
        if not np.array_equal(source.kept[:len(kept)], kept):
            raise ValueError("unexpected source harmonic ordering")
        channels = source.channels[:, :len(kept), :]
        omega = self.omega_per_ns
        if channels.shape[2] != len(omega):
            raise ValueError("source and kernel frequency grids differ in length")
        bounds = np.searchsorted(degrees, np.arange(self.degree + 2))
        weight = 4 * np.pi / (2 * np.arange(self.degree + 1) + 1)
        points = source.points_m()
        total = np.zeros((len(receivers), len(omega), 2), complex)

        for first in range(0, len(receivers), receiver_block):
            here = receivers[first:first + receiver_block]
            for begin in range(0, len(points), cell_block):
                block = points[begin:begin + cell_block]
                vectors = here[:, None, :] - block[None, :, :]
                radii = np.linalg.norm(vectors, axis=2)
                logs = np.log(radii)
                interval = np.clip(np.searchsorted(self.log_radii, logs, side="right") - 1,
                                   0, len(self.log_radii) - 2)
                step = logs - self.log_radii[interval]
                # (receivers, cells, degree + 1, 2, frequency), by Horner
                picked = self.coefficients[interval]
                radial = picked[..., 0, :, :]
                for power in range(1, picked.shape[-3]):
                    radial = radial * step[..., None, None, None] + picked[..., power, :, :]
                envelope = (np.exp(-self.absorption_per_m * radii)
                            / (4 * np.pi * radii ** 2))
                if self.radial_phase == "flight":
                    envelope = (envelope[..., None]
                                * np.exp(1j * omega * (radii / self.speed_m_per_ns)[..., None]))
                    radial = radial * envelope[..., None, None, :]
                else:
                    radial = radial * envelope[..., None, None, None]

                local = source.frame.rotate(vectors.reshape(-1, 3))
                harmonics = real_spherical_harmonics(
                    self.degree, local, source.azimuthal_degree
                ).reshape(len(here), len(block), -1)[:, :, :len(kept)]
                # Contract m inside each degree before the degree axis is ever
                # broadcast to channels: this is the 8.4-fold saving.
                folded = np.empty((len(here), len(block), self.degree + 1, len(omega)),
                                  complex)
                piece = channels[begin:begin + cell_block]
                for ell in range(self.degree + 1):
                    low, high = bounds[ell], bounds[ell + 1]
                    folded[:, :, ell, :] = weight[ell] * np.einsum(
                        "rbc,bcw->rbw", harmonics[:, :, low:high], piece[:, low:high, :],
                        optimize=True)
                total[first:first + len(here)] += np.einsum(
                    "rblow,rblw->rwo", radial, folded, optimize=True)

        carrier = np.exp(1j * omega * source.reference_ns)
        return np.transpose(total, (1, 0, 2)) * carrier[:, None, None]
