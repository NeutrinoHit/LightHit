"""Conservative geometry and explicit error-budget accounting; no RTE runs."""
from itertools import product
from types import SimpleNamespace

import numpy as np
import pytest

from lighthit.experimental.detector_tree import (
    AABB, AbsoluteChargeBound, DetectorTree, SourceBlock, aabb_distances, make_source_blocks)


def detector(points, cluster=None, string=None):
    identifiers = {}
    if cluster is not None:
        identifiers["cluster_id"] = np.asarray(cluster)
    if string is not None:
        identifiers["string_id"] = np.asarray(string)
    return SimpleNamespace(positions_m=np.asarray(points, float), identifiers=identifiers)


def corners(box):
    return np.array(list(product(*zip(box.lower_m, box.upper_m))))


def test_box_distance_bounds_enclose_all_corner_pairs_and_interior_samples():
    rng = np.random.default_rng(3)
    for _ in range(100):
        lo_a, lo_b = rng.normal(size=(2, 3))*1000
        a = AABB(lo_a, lo_a+rng.uniform(0, 1000, 3))
        b = AABB(lo_b, lo_b+rng.uniform(0, 1000, 3))
        low, high = aabb_distances(a, b)
        first = np.vstack((corners(a), rng.uniform(a.lower_m, a.upper_m, (20, 3))))
        second = np.vstack((corners(b), rng.uniform(b.lower_m, b.upper_m, (20, 3))))
        distances = np.linalg.norm(first[:, None]-second[None, :], axis=2)
        assert low <= distances.min() and high >= distances.max()
        assert high == pytest.approx(distances.max(), rel=1e-12)
        np.testing.assert_allclose(aabb_distances(b, a), (low, high), rtol=0, atol=0)
    assert aabb_distances(AABB([0, 0, 0], [2, 2, 2]), AABB([1, 1, 1], [3, 3, 3]))[0] == 0


@pytest.mark.parametrize("block_size", [1, 4, 16])
def test_no_geometric_false_rejection_against_exhaustive_pairs(block_size):
    rng = np.random.default_rng(15)
    positions = rng.uniform(-700, 700, (120, 3))
    source = rng.uniform(-900, 900, (47, 3))
    tree = DetectorTree.from_detector(detector(positions, np.arange(120)//24, np.arange(120)//6 % 4))
    blocks = make_source_blocks(source, block_size=block_size)
    found = tree.range_candidates(blocks, maximum_distance_m=300.)
    truth = np.linalg.norm(source[:, None]-positions[None, :], axis=2) <= 300.
    assert set(np.flatnonzero(truth.any(0))) <= set(found.detector_indices)
    for item in found.blocks:
        assert set(np.flatnonzero(truth[item.source.indices].any(0))) <= set(item.detector_indices)
    if block_size == 1:
        np.testing.assert_array_equal(found.detector_indices, np.flatnonzero(truth.any(0)))
        assert found.stats.radius_rejected_subtrees > 0
        assert found.stats.candidate_point_om_pairs == int(truth.sum())


def test_cluster_boundary_and_high_z_string_boxes_do_not_hide_near_oms():
    positions = [[299., 0., 0.], [401., 0., 0.], [0., 0., -1000.],
                 [0., 0., 1000.], [0., 0., 50.]]
    tree = DetectorTree.from_detector(detector(positions, [0, 0, 1, 1, 1], [0, 0, 0, 0, 0]))
    result = tree.range_candidates(make_source_blocks([[0., 0., 0.]], 1))
    np.testing.assert_array_equal(result.detector_indices, [0, 4])
    assert result.stats.radius_rejected_subtrees > 0


def test_equality_nextafter_and_large_coordinate_roundoff_are_retained():
    origin = 1e12
    positions = [[origin+300., 0., 0.], [np.nextafter(origin+300., np.inf), 0., 0.],
                 [origin+301., 0., 0.]]
    tree = DetectorTree.from_detector(detector(positions))
    result = tree.range_candidates(make_source_blocks([[origin, 0., 0.]], 1))
    assert 0 in result.detector_indices and 1 in result.detector_indices
    assert 2 not in result.detector_indices


def test_long_source_blocks_reduce_pairs_without_losing_either_end():
    x = np.linspace(-5000., 5000., 1001)
    source = np.column_stack((x, np.zeros(len(x)), np.zeros(len(x))))
    positions = [[-4500., 0., 0.], [0., 0., 0.], [4500., 0., 0.], [0., 400., 0.]]
    tree = DetectorTree.from_detector(detector(positions, [0, 1, 2, 1], [0, 0, 0, 1]))
    one = tree.range_candidates(make_source_blocks(source, len(source)))
    many = tree.range_candidates(make_source_blocks(source, 20))
    np.testing.assert_array_equal(one.detector_indices, [0, 1, 2])
    np.testing.assert_array_equal(many.detector_indices, one.detector_indices)
    assert many.stats.candidate_point_om_pairs < one.stats.candidate_point_om_pairs/3


def test_chord_boxes_include_the_support_between_endpoints():
    tree = DetectorTree.from_detector(detector([[0., 0., 0.], [0., 301., 0.]]))
    blocks = make_source_blocks([[-1000., 0., 0.]], end_points_m=[[1000., 0., 0.]])
    result = tree.range_candidates(blocks)
    np.testing.assert_array_equal(result.detector_indices, [0])
    np.testing.assert_array_equal(result.possible_near_indices, [0])


def test_lower_cache_radius_is_flagged_and_never_geometrically_discarded():
    tree = DetectorTree.from_detector(detector([[.5, 0., 0.], [5., 0., 0.]]))
    blocks = make_source_blocks([[0., 0., 0.]])
    geometric = tree.range_candidates(blocks)
    np.testing.assert_array_equal(geometric.detector_indices, [0, 1])
    np.testing.assert_array_equal(geometric.possible_near_indices, [0])
    # Even a cache certificate must not be applied below its valid inner radius.
    bounded = tree.range_candidates(blocks, charge_bound=lambda *args:
                                    AbsoluteChargeBound(0., "certified", "zero numerical cache component"))
    np.testing.assert_array_equal(bounded.detector_indices, [0])


@pytest.mark.parametrize("identifiers", [{}, {"cluster_id": [1, 1, 2]},
                                         {"string_id": [5, 5, 7]},
                                         {"cluster": [1, 1, 2], "subcluster": [0, 1, 0]},
                                         {"cluster_id": [None, None, 2], "string_id": [np.nan, 0., 1.]}])
def test_missing_identifiers_and_aliases_preserve_original_indices(identifiers):
    # Physical row 1 is deliberately unrelated to the module's public identifier.
    item = detector([[900., 0., 0.], [10., 0., 0.], [600., 0., 0.]])
    item.identifiers = {**identifiers, "module_id": [99, 101, 45]}
    tree = DetectorTree.from_detector(item)
    result = tree.range_candidates(make_source_blocks([[0., 0., 0.]]))
    np.testing.assert_array_equal(result.detector_indices, [1])
    assert [node.kind for node in tree.nodes].count("om") == 3
    np.testing.assert_array_equal(tree.root.indices, [0, 1, 2])


def test_source_indices_and_static_detector_snapshot_are_preserved():
    source = np.array([[200., 0., 0.], [0., 0., 0.], [100., 0., 0.]])
    blocks = make_source_blocks(source, block_size=1, indices=[2, 0])
    np.testing.assert_array_equal(np.concatenate([block.indices for block in blocks]), [2, 0])
    item = detector([[10., 0., 0.]])
    tree = DetectorTree.from_detector(item)
    item.positions_m[:] = 1000.
    np.testing.assert_array_equal(tree.positions_m, [[10., 0., 0.]])
    with pytest.raises(ValueError):
        tree.positions_m[0, 0] = 3.


def test_empirical_scores_and_bounds_never_certify_rejection():
    tree = DetectorTree.from_detector(detector([[10., 0., 0.], [20., 0., 0.]], [0, 1], [0, 0]))
    blocks = make_source_blocks([[0., 0., 0.]])
    baseline = tree.range_candidates(blocks)
    reordered = tree.range_candidates(blocks, prioritizer=lambda block, node, distance: distance[1],
                                      charge_bound=lambda *args: AbsoluteChargeBound(0., "empirical"),
                                      per_om_error_budget_pe=1.)
    np.testing.assert_array_equal(reordered.detector_indices, baseline.detector_indices)
    np.testing.assert_array_equal(reordered.omitted_charge_bound_pe, [0., 0.])
    assert reordered.stats.certified_rejected_subtrees == 0
    assert reordered.stats.uncertified_bounds_ignored > 0


def test_certified_omission_budget_accumulates_across_all_source_parts():
    tree = DetectorTree.from_detector(detector([[0., 0., 0.]]))
    blocks = make_source_blocks([[10., 0., 0.], [20., 0., 0.]], 1)
    # Each entire block contributes at most .006 PE to this synthetic operator.
    # A fresh .01 budget per block would incorrectly permit omitting .012 PE.
    result = tree.range_candidates(blocks, per_om_error_budget_pe=.01,
                                   charge_bound=lambda *args:
                                   AbsoluteChargeBound(.006, "certified", "synthetic constant operator: q_block<=.006"))
    assert len(result.blocks[0].detector_indices) == 0
    np.testing.assert_array_equal(result.blocks[1].detector_indices, [0])
    assert .006 <= result.omitted_charge_bound_pe[0] <= .01
    np.testing.assert_array_equal(result.detector_indices, [0])


def test_per_om_budgets_descend_when_a_subtree_cannot_share_the_same_omission():
    tree = DetectorTree.from_detector(detector([[0., 0., 0.], [1., 0., 0.]], [0, 0], [0, 0]))
    result = tree.range_candidates(make_source_blocks([[20., 0., 0.]], 1),
                                   per_om_error_budget_pe=[.01, .001],
                                   charge_bound=lambda *args:
                                   AbsoluteChargeBound(.006, "certified", "synthetic uniform absolute charge"))
    np.testing.assert_array_equal(result.detector_indices, [1])
    assert result.omitted_charge_bound_pe[0] >= .006
    assert result.omitted_charge_bound_pe[1] == 0


def test_invalid_certificate_scope_or_missing_proof_is_rejected():
    with pytest.raises(ValueError, match="proof_reference"):
        AbsoluteChargeBound(.1, "certified")
    tree = DetectorTree.from_detector(detector([[10., 0., 0.]]))
    with pytest.raises(ValueError, match="scope"):
        tree.range_candidates(make_source_blocks([[0., 0., 0.]]), per_om_error_budget_pe=1.,
                               charge_bound=lambda *args: AbsoluteChargeBound(.1, "certified", "different operator", "physical_charge"))


def test_empty_block_query_is_well_defined():
    result = DetectorTree.from_detector(detector([[10., 0., 0.]])).range_candidates(())
    assert not len(result.detector_indices) and result.stats.node_pairs_visited == 0
    np.testing.assert_array_equal(result.omitted_charge_bound_pe, [0.])


def test_invalid_bounds_and_source_indices_are_rejected():
    with pytest.raises(ValueError):
        AABB([1., 0., 0.], [0., 0., 0.])
    with pytest.raises(ValueError):
        AABB([np.nan, 0., 0.], [1., 1., 1.])
    with pytest.raises(ValueError):
        make_source_blocks([[0., 0., 0.]], indices=[2])
    with pytest.raises(ValueError):
        SourceBlock(AABB([0., 0., 0.], [1., 1., 1.]), np.array([0, 0]))
