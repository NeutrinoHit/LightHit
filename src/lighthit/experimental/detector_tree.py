"""Conservative cluster -> string -> OM queries for a finite radial domain.

This geometry helper does not run transport. Rejection beyond the requested
radius means outside that query domain, not zero physical light: a physical
far-tail bound remains a separate requirement. Points potentially inside the
cache's lower radius are returned for an explicit near-source treatment.

Optional charge pruning consumes ONE absolute-error budget per OM across all
source blocks. A callback must certify a uniform per-OM bound for the ENTIRE
source block and detector subtree. Empirical scores can prioritize traversal
but never certify rejection. Do not use a per-packet PE threshold: weak packets
may together activate an OM. Numerical-cache bounds do not bound cache error
against the physical RTE, and order-0/1 contributions must also be included in
any later full-event threshold decision.
"""
from dataclasses import dataclass
from math import hypot

import numpy as np


def _points(values, name):
    values = np.asarray(values, float)
    if values.ndim != 2 or values.shape[1:] != (3,) or not len(values) or not np.isfinite(values).all():
        raise ValueError(f"{name} must be a finite nonempty (N,3) array")
    return values


def _indices(values):
    values = np.asarray(values)
    if (values.ndim != 1 or not len(values) or not np.issubdtype(values.dtype, np.integer)
            or np.any(values < 0) or len(np.unique(values)) != len(values)):
        raise ValueError("indices must be unique nonnegative integers")
    result = values.astype(np.int64, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class AABB:
    lower_m: np.ndarray
    upper_m: np.ndarray

    def __post_init__(self):
        low, high = np.asarray(self.lower_m, float), np.asarray(self.upper_m, float)
        if (low.shape != (3,) or high.shape != (3,) or not np.isfinite(low).all()
                or not np.isfinite(high).all() or np.any(low > high)):
            raise ValueError("AABB endpoints must be finite ordered three-vectors")
        for name, value in (("lower_m", low), ("upper_m", high)):
            value = value.copy()
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    @classmethod
    def from_points(cls, points_m):
        points = _points(points_m, "points_m")
        return cls(points.min(0), points.max(0))


def aabb_distances(first, second):
    """Outward-rounded lower/upper distances for any pair of box points.

    Guard subtraction, sum-of-squares and square-root roundoff using coordinate
    scale. A box can contain holes in the actual point set: a small minimum is
    only a possible encounter, while a large minimum proves every pair is far.
    """
    gap = np.maximum(np.maximum(first.lower_m-second.upper_m,
                                second.lower_m-first.upper_m), 0.)
    far = np.maximum(np.abs(first.lower_m-second.upper_m),
                     np.abs(first.upper_m-second.lower_m))
    scale = max(1., float(np.max(np.abs(np.r_[first.lower_m, first.upper_m,
                                              second.lower_m, second.upper_m]))))
    guard = 64*np.finfo(float).eps*scale
    minimum = max(0., float(np.nextafter(hypot(*gap)-guard, -np.inf)))
    maximum = float(np.nextafter(hypot(*far)+guard, np.inf))
    return minimum, maximum


@dataclass(frozen=True)
class SourceBlock:
    box: AABB
    indices: np.ndarray
    label: object = None

    def __post_init__(self):
        if not isinstance(self.box, AABB):
            raise TypeError("source block box must be an AABB")
        object.__setattr__(self, "indices", _indices(self.indices))


def make_source_blocks(points_m, block_size=256, *, indices=None, end_points_m=None):
    """Bound contiguous groups of supplied source indices without renumbering.

    Input order is retained; callers can spatially order indices first. With
    ``end_points_m`` each index identifies a chord and its entire support is
    included, rather than only its start or midpoint. Splitting a long track
    into local blocks avoids one enormous box bridging the whole detector.
    """
    points = _points(points_m, "points_m")
    if isinstance(block_size, bool) or not isinstance(block_size, (int, np.integer)) or block_size < 1:
        raise ValueError("block_size must be a positive integer")
    index = _indices(np.arange(len(points)) if indices is None else indices)
    if np.any(index >= len(points)):
        raise ValueError("source indices are outside points_m")
    ends = None if end_points_m is None else _points(end_points_m, "end_points_m")
    if ends is not None and ends.shape != points.shape:
        raise ValueError("end_points_m must match points_m")
    blocks = []
    for first in range(0, len(index), int(block_size)):
        chosen = index[first:first+int(block_size)]
        support = points[chosen] if ends is None else np.concatenate((points[chosen], ends[chosen]))
        blocks.append(SourceBlock(AABB.from_points(support), chosen, len(blocks)))
    return tuple(blocks)


@dataclass(frozen=True)
class DetectorNode:
    node_id: int
    kind: str
    label: object
    box: AABB
    indices: np.ndarray
    children: tuple = ()


@dataclass(frozen=True)
class AbsoluteChargeBound:
    """Uniform bound on absolute charge from one whole block to EACH node OM.

    ``proof_status='certified'`` is an explicit promise by the callback, not
    an automatic proof. Its reference must identify the bound being used.
    Non-certified bounds cannot consume an error budget or prune a node.
    """
    per_om_pe: float
    proof_status: str = "empirical"
    proof_reference: str = ""
    scope: str = "numerical_cache_charge"

    def __post_init__(self):
        if np.isnan(self.per_om_pe) or self.per_om_pe < 0:
            raise ValueError("charge bound must be nonnegative and not NaN")
        if self.proof_status not in ("certified", "empirical", "unknown"):
            raise ValueError("proof_status must be certified, empirical or unknown")
        if self.proof_status == "certified" and not self.proof_reference.strip():
            raise ValueError("a certified bound requires a proof_reference")
        if not self.scope:
            raise ValueError("a charge bound must name its scope")


@dataclass
class TraversalStats:
    source_blocks: int = 0
    source_indices: int = 0
    node_pairs_visited: int = 0
    radius_rejected_subtrees: int = 0
    radius_excluded_point_om_pairs: int = 0
    certified_rejected_subtrees: int = 0
    bound_calls: int = 0
    uncertified_bounds_ignored: int = 0
    om_leaves_visited: int = 0
    candidate_block_om_pairs: int = 0
    candidate_point_om_pairs: int = 0


@dataclass(frozen=True)
class BlockCandidates:
    source: SourceBlock
    detector_indices: np.ndarray
    possible_near_indices: np.ndarray


@dataclass(frozen=True)
class RangeCandidates:
    detector_indices: np.ndarray
    possible_near_indices: np.ndarray
    blocks: tuple
    stats: TraversalStats
    omitted_charge_bound_pe: np.ndarray
    bound_scope: str


class DetectorTree:
    """Static hierarchy with original DetectorArray row indices at every node."""

    def __init__(self, detector):
        self.positions_m = _points(detector.positions_m, "detector positions").copy()
        self.positions_m.setflags(write=False)
        size = len(self.positions_m)
        identifiers = getattr(detector, "identifiers", {})

        def labels(*names):
            value = next((identifiers[name] for name in names if name in identifiers), None)
            if value is None:
                return np.full(size, None, object)
            value = np.asarray(value)
            if value.shape != (size,):
                raise ValueError("detector identifiers must have one scalar per OM")
            return value

        clusters = labels("cluster_id", "cluster")
        strings = labels("string_id", "string", "subcluster")

        def groups(indices, values):
            result = {}
            for index in indices:
                key = values[index]
                if isinstance(key, np.generic):
                    key = key.item()
                if isinstance(key, float) and np.isnan(key):
                    key = None
                try:
                    result.setdefault(key, []).append(int(index))
                except TypeError as exc:
                    raise ValueError("detector identifiers must be hashable scalars") from exc
            return [(key, np.asarray(value, np.int64)) for key, value in result.items()]

        nodes = []

        def node(kind, label, indices, child_builder=None):
            node_id = len(nodes)
            nodes.append(None)
            indices = _indices(indices)
            children = () if child_builder is None else tuple(child_builder(indices))
            value = DetectorNode(node_id, kind, label, AABB.from_points(self.positions_m[indices]), indices, children)
            nodes[node_id] = value
            return value

        def string_nodes(indices):
            return [node("string", label, selected,
                         lambda rows: [node("om", int(row), np.array([row])) for row in rows])
                    for label, selected in groups(indices, strings)]

        self.root = node("detector", None, np.arange(size),
                         lambda rows: [node("cluster", label, selected, string_nodes)
                                       for label, selected in groups(rows, clusters)])
        self.nodes = tuple(nodes)

    @classmethod
    def from_detector(cls, detector):
        return cls(detector)

    def range_candidates(self, blocks, *, maximum_distance_m=300., minimum_cache_distance_m=3.,
                         boundary_guard_m=1e-9, prioritizer=None, charge_bound=None,
                         per_om_error_budget_pe=0., bound_scope="numerical_cache_charge"):
        """Return possible pairs; geometric false positives are intentional.

        ``prioritizer(block,node,(minimum,maximum))`` returns a larger-is-first
        score. ``charge_bound`` takes the same arguments and returns an
        AbsoluteChargeBound or None. Its certified bound covers all source
        points/parts in that block, not one representative point. One budget
        ledger is shared by every block in THIS call: pass all event source
        parts together rather than creating a fresh budget per packet.

        Cache charge pruning is disabled when the boxes might approach below
        the lower cache radius. Such near pairs remain explicit candidates.
        ``omitted_charge_bound_pe`` excludes geometric radius exclusions and
        cannot be presented as a bound on uncomputed physical far tails.
        """
        low, high, guard = map(float, (minimum_cache_distance_m, maximum_distance_m, boundary_guard_m))
        if not np.isfinite([low, high, guard]).all() or low < 0 or high <= low or guard < 0:
            raise ValueError("require finite 0 <= minimum_cache_distance_m < maximum_distance_m and nonnegative guard")
        blocks = tuple(blocks)
        if not all(isinstance(block, SourceBlock) for block in blocks):
            raise TypeError("blocks must contain SourceBlock instances")
        budget = np.broadcast_to(np.asarray(per_om_error_budget_pe, float), (len(self.positions_m),)).copy()
        if not np.isfinite(budget).all() or np.any(budget < 0):
            raise ValueError("per-OM error budgets must be finite and nonnegative")
        spent = np.zeros(len(self.positions_m))
        stats = TraversalStats(source_blocks=len(blocks), source_indices=sum(len(block.indices) for block in blocks))
        overall = np.zeros(len(self.positions_m), bool)
        possible_near = np.zeros(len(self.positions_m), bool)
        results = []
        for block in blocks:
            candidates, near = [], []
            stack = [self.root]
            while stack:
                current = stack.pop()
                stats.node_pairs_visited += 1
                distance = aabb_distances(block.box, current.box)
                if distance[0] > high+guard:
                    stats.radius_rejected_subtrees += 1
                    stats.radius_excluded_point_om_pairs += len(block.indices)*len(current.indices)
                    continue
                if charge_bound is not None and distance[0] >= low+guard:
                    stats.bound_calls += 1
                    bound = charge_bound(block, current, distance)
                    if bound is not None:
                        if not isinstance(bound, AbsoluteChargeBound):
                            raise TypeError("charge_bound must return AbsoluteChargeBound or None")
                        if bound.proof_status != "certified":
                            stats.uncertified_bounds_ignored += 1
                        elif bound.scope != bound_scope:
                            raise ValueError("certified charge bound scope differs from requested scope")
                        elif np.isfinite(bound.per_om_pe):
                            # Round UP the accumulated error bound; subtraction
                            # of a remaining budget could otherwise round high.
                            updated = (spent[current.indices] if bound.per_om_pe == 0. else
                                       np.nextafter(spent[current.indices]+bound.per_om_pe, np.inf))
                            if np.all(updated <= budget[current.indices]):
                                spent[current.indices] = updated
                                stats.certified_rejected_subtrees += 1
                                continue
                if current.children:
                    children = list(current.children)
                    if prioritizer is not None:
                        def priority(child):
                            value = float(prioritizer(block, child, aabb_distances(block.box, child.box)))
                            return value if np.isfinite(value) else 0.
                        children.sort(key=priority, reverse=True)
                    stack.extend(reversed(children))
                else:
                    stats.om_leaves_visited += 1
                    index = int(current.indices[0])
                    candidates.append(index)
                    if distance[0] <= low+guard:
                        near.append(index)
            candidate_array = np.sort(np.asarray(candidates, np.int64))
            near_array = np.sort(np.asarray(near, np.int64))
            overall[candidate_array] = True
            possible_near[near_array] = True
            stats.candidate_block_om_pairs += len(candidate_array)
            stats.candidate_point_om_pairs += len(block.indices)*len(candidate_array)
            results.append(BlockCandidates(block, candidate_array, near_array))
        return RangeCandidates(np.flatnonzero(overall), np.flatnonzero(possible_near),
                               tuple(results), stats, spent, bound_scope)


__all__ = ["AABB", "aabb_distances", "SourceBlock", "make_source_blocks", "DetectorNode",
           "DetectorTree", "AbsoluteChargeBound", "BlockCandidates", "RangeCandidates", "TraversalStats"]
