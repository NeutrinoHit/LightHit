"""Cheap receiver screening before a full event-spectrum evaluation.

The proxy is intentionally simple: photon yield is accumulated into a short
sequence of source-axis cells and propagated as absorption-only isotropic point
emission.  It is a heuristic, not a mathematical upper bound on directional
Cherenkov light.  Production screening therefore pairs it with an exact
omega=0 axial evaluation of rejected receivers before any spectrum is skipped.
"""
import numpy as np

from .axial_source import AxisFrame

__all__ = ["isotropic_axial_proxy", "screening_candidates"]


def isotropic_axial_proxy(elements, receivers_m, medium, *, bins=64, frame=None):
    """Absorption-only charge proxy per unit area for every receiver.

    The cost is O(elements + bins*receivers), independent of frequency count
    and angular degree.  Each axial bin sits at its photon-weighted 3-D
    centroid, so rotations and translations of source plus detector leave the
    estimate invariant.
    """
    if isinstance(bins, bool) or not isinstance(bins, (int, np.integer)) or bins < 1:
        raise ValueError("bins must be a positive integer")
    receivers = np.atleast_2d(np.asarray(receivers_m, float))
    if (receivers.ndim != 2 or receivers.shape[1] != 3 or not len(receivers)
            or not np.isfinite(receivers).all()):
        raise ValueError("receivers_m must be a finite nonempty (N,3) array")
    frame = frame or AxisFrame.of(elements)
    points = np.asarray(elements.midpoints_m, float)
    photons = np.asarray(elements.photons, float)
    if not len(points) or np.any(photons < 0) or not np.isfinite(photons).all():
        raise ValueError("elements must carry finite nonnegative photon weights")
    z = (points - frame.centre_m) @ frame.axis
    low, high = float(z.min()), float(z.max())
    if high == low:
        index = np.zeros(len(z), dtype=np.int64)
        bins = 1
    else:
        edges = np.linspace(low, high, bins + 1)
        index = np.clip(np.searchsorted(edges, z, side="right") - 1, 0, bins - 1)
    weight = np.bincount(index, weights=photons, minlength=bins)
    centres = np.zeros((bins, 3), float)
    safe = np.maximum(weight, np.finfo(float).tiny)
    for axis in range(3):
        centres[:, axis] = np.bincount(
            index, weights=photons * points[:, axis], minlength=bins) / safe
    keep = weight > 0
    distance = np.linalg.norm(
        receivers[:, None, :] - centres[None, keep, :], axis=2)
    distance = np.maximum(distance, np.finfo(float).tiny)
    return np.sum(
        weight[keep][None, :] * np.exp(-medium.absorption_per_m * distance)
        / (4 * np.pi * distance * distance), axis=1)


def screening_candidates(ballistic_per_m2, scattered_proxy_per_m2, *,
                         threshold_pe=0.01, effective_area_m2=0.053,
                         efficiency=0.20, safety_factor=10.0):
    """Initial full-spectrum mask from exact ballistic charge plus the proxy."""
    ballistic = np.asarray(ballistic_per_m2, float)
    proxy = np.asarray(scattered_proxy_per_m2, float)
    if ballistic.shape != proxy.shape or ballistic.ndim != 1:
        raise ValueError("ballistic and proxy must be equal-length vectors")
    values = np.array([threshold_pe, effective_area_m2, efficiency, safety_factor], float)
    if not np.isfinite(values).all() or threshold_pe < 0 or effective_area_m2 <= 0 \
            or not 0 < efficiency <= 1 or safety_factor < 1:
        raise ValueError("invalid screening threshold, exposure or safety factor")
    exposure = effective_area_m2 * efficiency
    estimate = exposure * (np.maximum(ballistic, 0.0)
                           + safety_factor * np.maximum(proxy, 0.0))
    return estimate >= threshold_pe, estimate
