"""Unscattered light with the module acceptance at the true arrival direction.

The closed form of the cone root is the one already used by
``lighthit.experimental.ballistic_fast``: for each emitting element and each
receiver there is at most one point ``a*`` along the element's chord whose
Cherenkov cone passes through the receiver, and the fluence there is

    photons / length * exp(-mu_t d*) / (2 pi rho sin theta_C).

What is added here is the only thing a directional module needs and the only
thing the previous code had to approximate: the photon that arrives from that
point travels along

    s_hat = (receiver - start - a* u) / d*,

so the acceptance argument is ``s_hat . n_look`` and not the cosine of the
line from the source centroid. The acceptance is rebuilt from its Legendre
coefficients, which is exact for a band-limited module response and is the
same ``alpha`` the scattered orders use -- one definition, not two.

This module is numpy only and always importable; the fused variant lives in
``lighthit.experimental.ballistic_fast``.
"""
import numpy as np
from scipy.special import eval_legendre

__all__ = ["acceptance_from_coefficients", "ballistic_directional"]


def acceptance_from_coefficients(alpha, cosines):
    """``A(x) = sum_l alpha_l (2l+1) P_l(x) / (4 pi)``."""
    alpha = np.asarray(alpha, float)
    ell = np.arange(len(alpha))
    basis = eval_legendre(ell[None, :], np.atleast_1d(cosines)[:, None])
    return basis @ (alpha * (2 * ell + 1)) / (4 * np.pi)


def ballistic_directional(elements, receivers, look_directions, medium,
                          cone_cosine, alpha, *, time_origin_ns=None,
                          relative_edges_ns=None, element_chunk=65536):
    """Exact ballistic charge, and optionally time bins, with directional OMs.

    Returns ``(charge, bins)``; ``bins`` is ``None`` unless both
    ``time_origin_ns`` and ``relative_edges_ns`` are given. Arrivals outside
    the requested window stay in the charge and are not folded into an edge
    bin, exactly as in the fused kernel.
    """
    receivers = np.ascontiguousarray(receivers, dtype=float)
    looks = np.ascontiguousarray(look_directions, dtype=float)
    if receivers.ndim != 2 or receivers.shape[1] != 3 or not len(receivers):
        raise ValueError("receivers must be a nonempty (n, 3) array")
    if looks.shape != receivers.shape:
        raise ValueError("look_directions must match receivers")
    if not np.allclose(np.linalg.norm(looks, axis=1), 1.0, rtol=0, atol=1e-9):
        raise ValueError("look_directions must be unit vectors")
    cone_cosine = np.ascontiguousarray(cone_cosine, dtype=float)
    sine = np.sqrt(np.maximum(1.0 - cone_cosine ** 2, 0.0))
    start = np.ascontiguousarray(elements.start_m, dtype=float)
    direction = np.ascontiguousarray(elements.direction, dtype=float)
    length = np.ascontiguousarray(elements.length_m, dtype=float)
    photons = np.ascontiguousarray(elements.photons, dtype=float)
    if not (len(start) == len(direction) == len(length) == len(photons)
            == len(cone_cosine)):
        raise ValueError("elements arrays and cone_cosine must share one length")
    want_bins = time_origin_ns is not None and relative_edges_ns is not None
    if want_bins:
        origins = np.ascontiguousarray(time_origin_ns, dtype=float)
        edges = np.ascontiguousarray(relative_edges_ns, dtype=float)
        if origins.shape != (len(receivers),) or not np.isfinite(origins).all():
            raise ValueError("time_origin_ns must contain one finite value per receiver")
        if (edges.ndim != 1 or len(edges) < 2 or not np.isfinite(edges).all()
                or np.any(np.diff(edges) <= 0)):
            raise ValueError("relative_edges_ns must be finite and strictly increasing")
        start_ns = np.ascontiguousarray(elements.start_ns, dtype=float)
        end_ns = np.ascontiguousarray(elements.end_ns, dtype=float)
        bins = np.zeros((len(receivers), len(edges) - 1))
    else:
        bins = None
    charge = np.zeros(len(receivers))
    safe_length = np.where(length > 1e-300, length, 1e-300)
    for index in range(len(receivers)):
        for low in range(0, len(start), element_chunk):
            piece = slice(low, low + element_chunk)
            offset = receivers[index][None, :] - start[piece]
            unit = direction[piece]
            along = np.einsum("ij,ij->i", offset, unit)
            impact2 = np.maximum(np.einsum("ij,ij->i", offset, offset)
                                 - along ** 2, 1e-300)
            impact = np.sqrt(impact2)
            s = sine[piece]
            root = along - impact * cone_cosine[piece] / s
            valid = (root >= 0.0) & (root < length[piece])
            if not np.any(valid):
                continue
            root = root[valid]
            impact = impact[valid]
            s = s[valid]
            distance = impact / s
            emitted = offset[valid] - root[:, None] * unit[valid]
            arrival_direction = emitted / distance[:, None]
            cosine = np.clip(arrival_direction @ looks[index], -1.0, 1.0)
            weight = (photons[piece][valid] / safe_length[piece][valid]
                      * np.exp(-medium.extinction_per_m * distance)
                      / (2 * np.pi * impact * s)
                      * acceptance_from_coefficients(alpha, cosine))
            charge[index] += float(weight.sum())
            if want_bins:
                fraction = root / safe_length[piece][valid]
                arrival = (start_ns[piece][valid]
                           + fraction * (end_ns[piece][valid] - start_ns[piece][valid])
                           + distance / medium.speed_m_per_ns - origins[index])
                slot = np.searchsorted(edges, arrival, side="right") - 1
                inside = (slot >= 0) & (slot < len(edges) - 1)
                np.add.at(bins[index], slot[inside], weight[inside])
    return charge, bins
