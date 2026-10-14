"""Choice of OMs for the detailed muon calculation.

A cheap preview estimates every OM; candidates above the threshold with a
margin are computed in detail, then sentinels and reopening rounds widen the
set where the detailed charges disagree with the preview. The preview is an
estimate, not a bound: OMs left out keep their preview charge and are marked
as such.
"""
from copy import copy
from dataclasses import dataclass, replace, asdict
from time import perf_counter
import numpy as np


@dataclass(frozen=True)
class AdaptiveScreenConfig:
    """Margins of the preview-driven OM selection and its reopening rounds."""
    scatter_margin: float = 4.
    feedback_safety: float = 1.5
    sentinel_fraction: float = 1/64
    max_rounds: int = 3
    maximum_margin: float = 128.
    audit_all_rejected: bool = False

    def __post_init__(self):
        """Reject non-finite or inconsistent margins."""
        if not np.isfinite([self.scatter_margin, self.feedback_safety,
                            self.sentinel_fraction, self.maximum_margin]).all():
            raise ValueError("screen parameters must be finite")
        if (self.scatter_margin < 1 or self.feedback_safety < 1
                or not 0 <= self.sentinel_fraction <= 1
                or self.maximum_margin < self.scatter_margin or self.max_rounds < 1):
            raise ValueError("invalid adaptive screen parameters")


def detector_groups(detector):
    """Cluster and string label of every OM (identifiers, else trivial labels)."""
    identifiers = detector.identifiers
    n = len(detector.positions_m)
    def labels(names, fallback):
        """First identifier column present among ``names``."""
        return np.asarray(next((identifiers[k] for k in names if k in identifiers), fallback))
    clusters = labels(('cluster_id', 'cluster'), np.zeros(n, int))
    strings = labels(('string_id', 'string', 'subcluster'), np.arange(n))
    return clusters, strings


def adaptive_candidates(preview, detector, threshold, evaluate, config=None, extra_charge=None):
    """Evaluate new candidates only. evaluate(indices) returns fine Q0/Q1/Qge2.

    Retained arrays are expected charges. Negative preview values are preserved
    in output but never used to cancel a positive selection score. Unsafe
    preview points are always evaluated. All selection decisions depend only
    on the independent preview and results of already evaluated receivers.

    ``extra_charge`` (per OM) is light of other sources of the same event
    already computed elsewhere; it enters every threshold comparison, so the
    selection is made on the event sum. None is identical to zero.
    """
    config = config or AdaptiveScreenConfig()
    q = np.asarray(preview.charge_components_pe, float).copy()
    n = len(detector.positions_m)
    if n == 0 or q.shape != (n, 3) or not np.isfinite(threshold) or threshold < 0:
        raise ValueError("invalid preview shape or threshold")
    clusters, strings = detector_groups(detector)
    raw = np.nan_to_num(q, nan=0., posinf=0., neginf=0.)
    scattered = np.maximum(raw[:, 1:], 0.).sum(1)
    q0 = np.maximum(raw[:, 0], 0.)
    known = np.zeros(n) if extra_charge is None else np.maximum(np.asarray(extra_charge, float), 0.)
    if known.shape != (n,) or not np.isfinite(known).all():
        raise ValueError("extra_charge must be finite with one value per OM")
    q0 = q0 + known
    forced = np.asarray(preview.unsafe_mask, bool).copy()
    if forced.shape != (n,):
        raise ValueError("unsafe mask must match detector")
    forced |= ~np.isfinite(q).all(1)
    evaluated = np.zeros(n, bool)
    margin = config.scatter_margin
    history = []
    sentinel = np.zeros(n, bool)
    # One probe at the strongest predicted OM of each non-negligible string.
    # The threshold applies to the SUM of the entire event, never one packet.
    if threshold > 0:
        for cluster in np.unique(clusters):
            for string in np.unique(strings[clusters == cluster]):
                indices = np.flatnonzero((clusters == cluster) & (strings == string))
                best = indices[np.argmax(q0[indices] + scattered[indices])]
                if q0[best] + scattered[best] >= config.sentinel_fraction * threshold:
                    sentinel[best] = True
    if not np.any((q0+margin*scattered >= threshold) | forced | sentinel):
        sentinel[int(np.argmax(q0+scattered))] = True
    fallback = None
    for iteration in range(config.max_rounds + 1):
        score = q0 + margin*scattered
        wanted = (score >= threshold) | forced | sentinel
        if threshold == 0 or iteration == config.max_rounds:
            if np.any(wanted & ~evaluated):
                wanted[:] = True
                fallback = 'all OMs: threshold zero or refinement limit'
        new = np.flatnonzero(wanted & ~evaluated)
        if not len(new):
            break
        began = perf_counter()
        fine = np.asarray(evaluate(new), float)
        if fine.shape != (len(new), 3) or not np.isfinite(fine).all():
            raise ValueError("fine evaluator must return finite three-order charges")
        q[new] = fine
        evaluated[new] = True
        total = fine.sum(1) + known[new]
        surprise = new[(score[new] < threshold) & (total >= threshold)]
        reopened = np.unique(clusters[surprise])
        for cluster in reopened:
            forced |= clusters == cluster
        # Calibrate the combined scattered predictor only where q0 alone does
        # not already retain the OM. A poor q1 estimate can be rescued by qge2.
        informative = (q0[new] < threshold) & (total >= threshold/8)
        true_scatter = np.maximum(fine[:, 1:].sum(1), 0.)
        blind = informative & (scattered[new] <= threshold*1e-8) & (true_scatter > threshold/8)
        if np.any(blind):
            forced[:] = True
            fallback = 'all OMs: observed blind scattered predictor'
        safe = informative & (scattered[new] > threshold*1e-8)
        required = (config.feedback_safety * float(np.max(
            true_scatter[safe]/scattered[new][safe], initial=0.)))
        old_margin = margin
        if required > config.maximum_margin:
            forced[:] = True
            fallback = 'all OMs: observed predictor error exceeds allowed margin'
        else:
            margin = max(margin, required)
        history.append(dict(round=iteration, indices=new.tolist(), receivers=len(new),
            fine_seconds=perf_counter()-began, active_in_batch=int(np.sum(total >= threshold)),
            sentinel_surprises=surprise.tolist(), reopened_clusters=reopened.tolist(),
            margin_before=old_margin, margin_after=margin))
    if config.audit_all_rejected and np.any(~evaluated):
        remaining = np.flatnonzero(~evaluated)
        began = perf_counter()
        fine = np.asarray(evaluate(remaining), float)
        if fine.shape != (len(remaining), 3) or not np.isfinite(fine).all():
            raise ValueError("invalid complete-audit result")
        q[remaining] = fine
        evaluated[remaining] = True
        history.append(dict(round='complete rejected audit', indices=remaining.tolist(),
            receivers=len(remaining), fine_seconds=perf_counter()-began,
            recovered_active=remaining[fine.sum(1)>=threshold].tolist()))
    return q, evaluated, dict(config=asdict(config), rounds=history, margin=margin,
        initial_score_candidates=int(np.sum(q0+config.scatter_margin*scattered >= threshold)),
        forced_by_preview=int(np.sum(preview.unsafe_mask)), sentinel_oms=np.flatnonzero(sentinel).tolist(),
        fine_receivers=int(evaluated.sum()), coarse_only_receivers=int((~evaluated).sum()),
        fallback=fallback, all_rejections_audited=bool(evaluated.all()),
        inactivity_certified=False,
        guarantee_scope='empirical preview with sentinels and observed-error reopening; no unseen-peak guarantee')


def __getattr__(name):
    # The earlier reference route lives in lighthit.experimental.smart_muon.
    """Resolve ``transport_smart_muon`` from its experimental module on demand."""
    if name == 'transport_smart_muon':
        from ..experimental.smart_muon import transport_smart_muon
        return transport_smart_muon
    raise AttributeError(name)
