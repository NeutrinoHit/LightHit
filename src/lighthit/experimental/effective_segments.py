"""Method 7: fit a stored shower to a chosen number of straight segments.

@sec-cone-segment already has an exact closed form for one straight Cherenkov
segment -- ballistic in closed form, orders 1 and >=2 by a short Gauss
quadrature against the cached multipoles, both cheap because there is only
one track and a handful of receivers to serve. The question this module
answers is what happens when that machinery is asked to stand in for a real,
curved, 221 393-element shower: how many straight pieces does it take, and
what does each extra piece cost against what it buys.

The shower is cut along its own photon-weighted axis (:class:`AxisFrame`,
the same one the needle reduction of ``axial_source.py`` uses) into
``segments`` equal-length intervals of that coordinate. Growing the count
never substitutes a coarser assumption for a finer one: each piece is fit
independently, from only its own true elements --

* its own photon-weighted **direction** (not the shower's global axis, so a
  piece can follow real curvature),
* its own photon-weighted **cone cosine**, which fixes an effective beta
  through ``cone_cosine = 1/(beta * phase_index)`` at the contract's fixed
  phase index. This preserves that scalar mean, but not the angular dipole
  ``<cone_cosine * direction>``: averaging the cone cosine and direction
  separately does not preserve their joint moment,
* its own **start time**, fit by a weighted least-squares intercept against
  the position each element occupies along the piece's own direction, at the
  piece's own effective speed -- exact at the piece's photon-weighted mean
  position and time, not assumed linear from a global start.

So "one segment" (``segments=1``) is the crude whole-shower test. Increasing
``segments`` shrinks only the spread in axial position. It does **not** shrink
the transverse or directional distribution at a fixed shower depth. The
construction therefore converges for a genuinely one-dimensional curved
track, but need not converge for an electromagnetic shower: one physical
Cherenkov cone cannot in general reproduce even the angular dipole of a broad
mixture of particle directions. Convergence on the synthetic track fixture is
a regression check, not a theorem about stored showers.

Within the reduced source, :func:`~lighthit.experimental.cone_segment.segment_spectrum`
still evaluates the longitudinal quadrature, closed-form ballistic term and
addition-theorem angular projection without further approximation. The public
``full_response`` deliberately replaces its fitted ballistic column with the
exact per-element result.
"""
from dataclasses import dataclass
import numpy as np

from ..medium import C_VACUUM_M_PER_NS
from .axial_source import AxisFrame
from .cone_segment import ConeSegment, segment_spectrum

__all__ = ["EffectiveSegments", "fit_effective_segments", "effective_segment_response",
           "full_response"]


@dataclass(frozen=True)
class EffectiveSegments:
    """The straight-segment fit of a shower, and how the photons split."""
    segments: tuple
    frame: AxisFrame
    photons_per_segment: np.ndarray
    elements_per_segment: np.ndarray
    empty_bins: int

    def __len__(self):
        return len(self.segments)

    def summary(self):
        return {
            "segments": len(self.segments),
            "empty_bins_skipped": self.empty_bins,
            "photons_per_segment": self.photons_per_segment.tolist(),
            "elements_per_segment": self.elements_per_segment.tolist(),
            "length_m": [float(s.length_m) for s in self.segments],
            "beta": [float(s.beta) for s in self.segments],
        }


def fit_effective_segments(elements, segments, *, frame=None, minimum_length_m=1e-4):
    """Reduce ``elements`` to at most ``segments`` straight ConeSegments.

    Bins with no element are skipped rather than raising: a shower's photon
    density is uneven along its own axis, and asking for more segments than
    the data can fill some of is a normal, reportable outcome, not an error.

    ``segments`` controls the axial partition only. It does not resolve a
    broad mixture of directions inside one slab; use a harmonic source such
    as :class:`~lighthit.experimental.axial_source.AxialSource` when that
    angular content must be retained.
    """
    if not isinstance(segments, (int, np.integer)) or segments < 1:
        raise ValueError("segments must be a positive integer")
    frame = frame if frame is not None else AxisFrame.of(elements)
    contract = elements.contract
    phase_index = float(contract.phase_index)

    mid = elements.midpoints_m
    weights = np.asarray(elements.photons, float)
    direction = elements.direction
    cone_cosine = np.asarray(elements.cone_cosine, float)
    time = elements.midpoint_time_ns()

    z = (mid - frame.centre_m) @ frame.axis
    low, high = z.min(), z.max()
    if not np.isfinite([low, high]).all():
        raise ValueError("non-finite axial coordinate; check the element positions")
    span = high - low
    if span <= 0:
        edges = np.array([low, low + 1.0])
        segments = 1
    else:
        edges = low + span * np.arange(segments + 1) / segments

    built, photon_counts, element_counts = [], [], []
    empty = 0
    for i in range(segments):
        upper_inclusive = i == segments - 1
        mask = (z >= edges[i]) & (z < edges[i + 1] if not upper_inclusive else z <= edges[i + 1])
        n = int(mask.sum())
        if n == 0:
            empty += 1
            continue
        w = weights[mask]
        total = w.sum()
        share = w / total

        u_eff = (share[:, None] * direction[mask]).sum(axis=0)
        norm = np.linalg.norm(u_eff)
        if norm < 1e-12:
            u_eff = frame.axis.copy()
        else:
            u_eff = u_eff / norm

        cone_eff = float((share * cone_cosine[mask]).sum())
        # Every physical input cone obeys mu_C >= 1 / n_phase (beta <= 1),
        # but a weighted sum of values sitting exactly on that boundary can
        # round a few ulps below it.  The old generic ``1e-6`` lower clip did
        # not enforce the actual physical bound and made sufficiently fine K
        # scans fail spuriously with beta just above one.
        physical_min = 1.0 / phase_index
        tolerance = 16 * np.finfo(float).eps * max(1.0, abs(physical_min))
        if cone_eff < physical_min - tolerance:
            raise ValueError(
                f"segment {i}: fitted cone cosine {cone_eff:g} is below the "
                f"physical beta=1 boundary {physical_min:g}")
        cone_eff = min(max(cone_eff, physical_min), np.nextafter(1.0, 0.0))
        beta_eff = 1.0 / (phase_index * cone_eff)
        if not 0 < beta_eff <= 1.0:
            raise ValueError(
                f"segment {i}: fitted beta {beta_eff:g} is unphysical; the "
                "photon-weighted cone cosine in this bin does not correspond "
                "to a beta <= 1 at this phase index")
        speed_eff = beta_eff * C_VACUUM_M_PER_NS

        centroid = (share[:, None] * mid[mask]).sum(axis=0)
        s = (mid[mask] - centroid) @ u_eff
        s_min, s_max = s.min(), s.max()
        length = max(float(s_max - s_min), minimum_length_m)
        start_point = centroid + s_min * u_eff

        mean_s = float((share * s).sum())
        mean_t = float((share * time[mask]).sum())
        start_time = mean_t - (mean_s - s_min) / speed_eff

        photons_per_m = float(total) / length

        built.append(ConeSegment(
            start_m=tuple(start_point), direction=tuple(u_eff), length_m=length,
            beta=beta_eff, phase_index=phase_index, photons_per_m=photons_per_m,
            start_time_ns=start_time))
        photon_counts.append(float(total))
        element_counts.append(n)

    if not built:
        raise ValueError("every bin was empty; too many segments for this event")
    return EffectiveSegments(tuple(built), frame, np.array(photon_counts),
                              np.array(element_counts), empty)


def effective_segment_response(cache, effective, receivers_m, *, longitudinal_order=16,
                               detector_chunk=64):
    """Sum ``segment_spectrum`` over every fitted piece.

    Returns ``(frequency, receiver, 3)``: ballistic, finite-L order 1,
    order >=2 -- the same three columns :func:`segment_spectrum` returns for
    one segment, because the response is linear in the source and this is
    exactly the superposition @sec-segment-event already checks (rotating,
    translating, delaying and splitting a segment all recover the whole).
    """
    receivers = np.atleast_2d(np.asarray(receivers_m, float))
    total = None
    for segment in effective.segments:
        piece = segment_spectrum(cache, segment, receivers,
                                 longitudinal_order=longitudinal_order,
                                 detector_chunk=detector_chunk)
        total = piece if total is None else total + piece
    return total


def full_response(cache, elements, medium, effective, receivers_m, *,
                  longitudinal_order=16, detector_chunk=64, ballistic=None):
    """The response this method should actually be used for.

    Column 0 (ballistic) is *not* taken from the segment fit: it is the
    exact per-element closed form of @eq-ballistic-fluence, computed on the
    real elements directly (``ballistic_fast.vectorised_ballistic_fast`` if
    numba is available, else the numpy ``vectorised_ballistic``). Measured
    directly (see ``fit_effective_segments`` and the accompanying tests): a
    handful of straight pieces cannot reproduce the ballistic term, however
    many are used, because it is a near-singular function of exact cone
    alignment, and it is already fast in closed form -- there is no reason
    to approximate what costs almost nothing to get exactly. Orders 1 and
    >=2 are the only columns supplied by the fit. Whether they converge
    with this axial-only partition is source dependent; a broad shower can
    retain an irreducible angular mixture in every slab.

    ``ballistic`` may be passed precomputed (shape ``(receivers,)``) to
    avoid recomputing it across a scan over ``effective`` at fixed elements
    and receivers.
    """
    receivers = np.atleast_2d(np.asarray(receivers_m, float))
    scattered = effective_segment_response(cache, effective, receivers,
                                           longitudinal_order=longitudinal_order,
                                           detector_chunk=detector_chunk)
    if ballistic is None:
        try:
            from .ballistic_fast import vectorised_ballistic_fast as _ballistic
        except ImportError:
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
            from run_shower_moments import vectorised_ballistic as _ballistic
        ballistic = _ballistic(elements, receivers, medium, elements.cone_cosine)
    out = scattered.copy()
    out[:, :, 0] = ballistic[None, :]
    return out
