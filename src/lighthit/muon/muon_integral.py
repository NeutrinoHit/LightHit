"""Straight-piece order-one integrals of a curved primary track.

The primary steps are fitted by straight pieces within geometric tolerances and
each piece is integrated with the direct physical-time segment quadrature.
The tolerances are geometric; they are not a bound on the detected signal.
"""
from dataclasses import dataclass, replace
from time import perf_counter

import numpy as np

from ..sources import SpectralLightElements


@dataclass(frozen=True)
class TrackIntegralFit:
    """Straight pieces fitted to a primary track and their geometric errors."""
    elements: SpectralLightElements
    original_elements: int
    ranges: np.ndarray
    maximum_position_error_m: float
    maximum_time_error_ns: float
    maximum_direction_rms_rad: float


def fit_track_integrals(source, *, max_length_m=10., position_tolerance_m=.01,
                        time_tolerance_ns=.03, direction_tolerance_rad=.001,
                        beta_tolerance=1e-4, yield_tolerance=.001,
                        phase_indices=()):
    """Replace consecutive steps by straight, emission-weighted pieces.

    A recursive bisection preserves total C0 and C2 exactly per piece, fits
    a line about the emission centroid and mean time, and checks every
    original midpoint and every step endpoint. Discontinuous position/time
    records are never joined across a missing, nonemitting interval.
    Original low-beta steps survive when a piece would mix
    different Cherenkov cones.  Ballistic light must still use ``source``.
    """
    limits = (max_length_m, position_tolerance_m, time_tolerance_ns,
              direction_tolerance_rad, beta_tolerance, yield_tolerance)
    if not np.isfinite(limits).all() or min(limits) <= 0:
        raise ValueError("fit tolerances must be finite and positive")
    if len(np.unique(source.uid)) != 1:
        raise ValueError("fit_track_integrals needs one primary track UID")
    phases = np.asarray(phase_indices, float)
    if phases.ndim != 1 or np.any(~np.isfinite(phases)) or np.any(phases <= 0):
        raise ValueError("phase_indices must contain positive finite indices")
    if np.any(np.diff(source.start_ns) < 0):
        raise ValueError("primary steps must be in chronological order")
    mid = source.midpoints_m
    ends = source.start_m + source.length_m[:, None] * source.direction
    times = .5 * (source.start_ns + source.end_ns)
    # Reconstructing a chord endpoint incurs ordinary floating-point error.
    # Permit only that coordinate-scaled roundoff, not a fit-tolerance-sized
    # dark gap whose light would otherwise be spread along the fitted piece.
    eps = 64 * np.finfo(float).eps
    position_scale = np.maximum(1., np.maximum(
        np.max(np.abs(ends[:-1]), axis=1), np.max(np.abs(source.start_m[1:]), axis=1)))
    time_scale = np.maximum(1., np.maximum(np.abs(source.end_ns[:-1]),
                                           np.abs(source.start_ns[1:])))
    gaps = ((np.max(np.abs(ends[:-1] - source.start_m[1:]), axis=1) > eps * position_scale)
            | (np.abs(source.end_ns[:-1] - source.start_ns[1:]) > eps * time_scale))
    boundaries = np.r_[0, np.flatnonzero(gaps) + 1, len(source)]
    accepted = []

    def fit(lo, hi):
        """Accept steps lo..hi as one piece, or split and recurse."""
        w = source.coefficient0[lo:hi]
        total = w.sum()
        if hi - lo == 1 or total <= 0:
            for i in range(lo, hi):
                accepted.append((i, i+1, source.start_m[i], source.direction[i],
                                 source.length_m[i], source.start_ns[i],
                                 source.end_ns[i], source.beta[i], 0., 0., 0.))
            return
        w = w / total
        direction = np.sum(w[:, None] * source.direction[lo:hi], axis=0)
        norm = np.linalg.norm(direction)
        if norm < 1e-12:
            split = (lo+hi)//2
            fit(lo, split)
            fit(split, hi)
            return
        direction /= norm
        centre = np.sum(w[:, None] * mid[lo:hi], axis=0)
        low = float((source.start_m[lo] - centre) @ direction)
        end = ends[hi-1]
        high = float((end - centre) @ direction)
        length = high - low
        z = (mid[lo:hi] - centre) @ direction
        start = centre + low * direction
        position_error = float(np.max(np.linalg.norm(
            mid[lo:hi] - centre - np.clip(z, low, high)[:, None]*direction, axis=1)))
        slope = float((source.end_ns[hi-1]-source.start_ns[lo]) / length) if length > 0 else 0.
        mean_time = float(w @ times[lo:hi])
        t0, t1 = mean_time + low*slope, mean_time + high*slope
        time_error = float(np.max(np.abs(times[lo:hi]-mean_time-z*slope)))
        for nodes, node_times in ((source.start_m[lo:hi], source.start_ns[lo:hi]),
                                  (ends[lo:hi], source.end_ns[lo:hi])):
            projected = np.clip((nodes-centre) @ direction, low, high)
            position_error = max(position_error, float(np.max(np.linalg.norm(
                nodes-centre-projected[:, None]*direction, axis=1))))
            time_error = max(time_error, float(np.max(np.abs(
                node_times-mean_time-projected*slope))))
        angular = np.arccos(np.clip(source.direction[lo:hi] @ direction, -1., 1.))
        rms = float(np.sqrt(w @ angular**2))
        # Preserve the mean cone cosine: beta_eff^-1 = <beta^-1>.
        beta = float(1. / (w @ (1. / source.beta[lo:hi])))
        density = source.coefficient0[lo:hi] / source.length_m[lo:hi]
        beta_low, beta_high = np.min(source.beta[lo:hi]), np.max(source.beta[lo:hi])
        same_threshold = not np.any((beta_low * phases <= 1.) & (beta_high * phases > 1.))
        ok = (0 < length <= max_length_m and position_error <= position_tolerance_m
              and time_error <= time_tolerance_ns and rms <= direction_tolerance_rad
              and beta_high - beta_low <= beta_tolerance and same_threshold
              and np.ptp(density) <= yield_tolerance * (w @ density))
        if ok:
            accepted.append((lo, hi, start, direction, length, t0, t1, beta,
                             position_error, time_error, rms))
        else:
            split = (lo+hi)//2
            fit(lo, split)
            fit(split, hi)

    for lo, hi in zip(boundaries[:-1], boundaries[1:]):
        fit(int(lo), int(hi))
    ranges = np.asarray([(x[0], x[1]) for x in accepted], np.int64)
    elements = SpectralLightElements(
        start_m=np.asarray([x[2] for x in accepted]),
        direction=np.asarray([x[3] for x in accepted]),
        length_m=np.asarray([x[4] for x in accepted]),
        coefficient0=np.asarray([source.coefficient0[lo:hi].sum() for lo, hi in ranges]),
        coefficient2=np.asarray([source.coefficient2[lo:hi].sum() for lo, hi in ranges]),
        beta=np.asarray([x[7] for x in accepted]),
        start_ns=np.asarray([x[5] for x in accepted]),
        end_ns=np.asarray([x[6] for x in accepted]),
        row_index=source.row_index[ranges[:, 0]], uid=source.uid[ranges[:, 0]],
        reference_phase_index=source.reference_phase_index,
        cone_model=source.cone_model,
        provenance={**(source.provenance or {}), "experimental_track_integrals": True})
    return TrackIntegralFit(elements, len(source), ranges,
                            max(x[8] for x in accepted), max(x[9] for x in accepted),
                            max(x[10] for x in accepted))


def order1_integrals(kernel, elements, *, time_origin_ns, config=None):
    """Direct order-one integration without recomputing ballistic/selection.

    All supplied receivers are calculated. The spectral cone, optical tables,
    acceptance, bin edges and quadrature are the production prompt ones.
    ``config.element_method`` selects segment, grouped or ring controls.
    No RTE cache is touched and no active-OM threshold is applied here.
    """
    from .. import prompt
    config = config or prompt.PromptConfig(element_method="segment", screen=False)
    if config.element_method not in ("segment", "grouped", "ring"):
        raise ValueError("choose an explicit element method")
    config = replace(config, screen=False)
    tables = prompt._spectral_tables(kernel)
    alpha = np.ascontiguousarray(kernel.acceptance()["alpha"], float)
    coef, ipow, jpow, degree = prompt.acceptance_monomials(alpha)
    detector = kernel.detector
    positions = np.ascontiguousarray(detector.positions_m, float)
    origins = np.asarray(time_origin_ns, float)
    if origins.shape != (len(positions),) or not np.isfinite(origins).all():
        raise ValueError("provide one finite time origin per receiver")
    looks = np.ascontiguousarray(-detector.orientations, float)
    areas = np.asarray(detector.effective_area_m2, float)
    edges = np.asarray(kernel.config.relative_time_edges_ns, float)
    charge, bins = np.zeros(len(positions)), np.zeros((len(positions), len(edges)-1))
    began = perf_counter()
    diagnostics = []
    phases = (tables["phase_index"] if elements.cone_model == "spectral" else
              np.full_like(tables["phase_index"], elements.reference_phase_index))
    config = replace(config, batch_charge_tolerance_pe=config.batch_charge_tolerance_pe/len(phases))
    for lam, phase in enumerate(phases):
        if tables['mus'][lam]==0 or (tables['s0w'][lam]==0 and tables['s2w'][lam]==0):
            continue
        mask = elements.beta * phase > 1
        if not np.any(mask):
            continue
        source = elements if np.all(mask) else elements.subset(mask)
        source = replace(source, reference_phase_index=float(phase), cone_model="frozen")
        segments = prompt._element_segments(source)
        single = {key: value[lam:lam+1] for key, value in tables.items()}
        q, b, meta = prompt._order1(
            config.element_method, kernel, source, source, segments,
            positions, looks, areas, origins, np.arange(len(positions)), single,
            coef, ipow, jpow, degree, float(kernel.medium.g), edges, config,
            np.zeros(len(positions)))
        if meta.get("order1_segments_too_close", 0):
            raise ValueError("a fitted segment comes inside min_distance_m of an OM")
        charge += q
        bins += b
        diagnostics.append(meta)
    return charge, bins, {"elapsed_seconds": perf_counter()-began,
                          "elements": len(elements), "wavelengths": diagnostics}
