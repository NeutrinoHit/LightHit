"""Finite-segment primitive invariants, including tiny remote positive tails."""
import numpy as np
import pytest

pytest.importorskip("numba")
from lighthit.experimental.positive_fluence import PositiveFluenceAtlas, endpoint_histogram


def atlas(z_edges=None, profile=None):
    z_edges = np.asarray([-5., -2., 0., 1., 4., 9.] if z_edges is None else z_edges)
    profile = np.asarray([.2, 1., 3., .4, .01] if profile is None else profile)
    # Distinct beta, order and radial factors exercise all interpolation lanes.
    density = (np.array([1., 2.])[:, None, None, None]
               * np.array([1., 7.])[None, :, None, None]
               * np.array([1., 3., 2.])[None, None, :, None]
               * profile[None, None, None, :])
    return PositiveFluenceAtlas(np.array([0., 1., 3., 6.]), z_edges,
                               np.array([.8, 1.]), density, np.zeros_like(density), {})


def test_primitive_matches_independent_cell_overlap_integral():
    a = atlas()
    lo, hi = -3.3, 6.2
    value, outside = a.segments(a.rho_nodes[1], lo, hi, .9)
    overlaps = np.maximum(0., np.minimum(hi, a.z_edges_m[1:])
                          - np.maximum(lo, a.z_edges_m[:-1]))
    expected = ((a.density[0, :, 1] + a.density[1, :, 1])/2) @ overlaps
    np.testing.assert_allclose(value[0], expected, rtol=3e-15)
    assert not outside[0]


def test_segment_additivity_positivity_and_zero_length():
    a = atlas()
    rho = np.linspace(.01, a.rho_nodes[-1], 101)
    beta = np.linspace(.8, 1., 101)
    whole, flags = a.segments(rho, -4.7, 8.5, beta)
    first, _ = a.segments(rho, -4.7, -.3, beta)
    last, _ = a.segments(rho, -.3, 8.5, beta)
    np.testing.assert_allclose(first+last, whole, rtol=4e-15)
    assert np.all(whole > 0) and not np.any(flags)
    zero, _ = a.segments(rho, .7, .7, beta)
    np.testing.assert_array_equal(zero, 0.)


def test_right_tail_avoids_saturated_left_prefix_cancellation():
    a = atlas(np.array([-2., 0., 1., 2., 3.]), np.array([1e20, 1., 1e-12, 2e-12]))
    assert a.left[0, 0, 0, -1] - a.left[0, 0, 0, -2] == 0.
    value, outside = a.segments(a.rho_nodes[0], 2.1, 2.9, .8)
    np.testing.assert_allclose(value[0], [1.6e-12, 1.12e-11], rtol=2e-15)
    assert not outside[0]


def test_left_tail_avoids_saturated_right_prefix_cancellation():
    a = atlas(np.array([-3., -2., -1., 0., 2.]), np.array([2e-12, 1e-12, 1., 1e20]))
    value, outside = a.segments(a.rho_nodes[0], -2.9, -2.1, .8)
    np.testing.assert_allclose(value[0], [1.6e-12, 1.12e-11], rtol=2e-15)
    assert not outside[0]


def test_outside_queries_are_explicit_and_clipped_not_certified_zero():
    a = atlas()
    rho = [a.rho_nodes[0], a.rho_nodes[-1]+.1, -1., a.rho_nodes[0]]
    value, outside = a.segments(rho, [-100., -1., -1., -1.], [100., 1., 1., 1.], [.9, .9, .9, 1.1])
    assert outside.tolist() == [True, True, True, True]
    inside, _ = a.segments(rho[0], a.z_edges_m[0], a.z_edges_m[-1], .9)
    np.testing.assert_allclose(value[0], inside[0])
    np.testing.assert_array_equal(value[1:3], 0.)
    assert np.all(value[3] > 0)  # Clamped beta is also flagged, not silently trusted.


def test_histogram_retains_weight_and_weight_squared():
    xyz = np.array([[.3, .4, .1], [.6, .8, .2], [0., 1., -2.]])
    counts, squares = endpoint_histogram(xyz, np.array([2., 3., 5.]), 1., 1.,
                                        np.array([0., 2.]), np.array([-3., 0., 3.]))
    np.testing.assert_array_equal(counts, [[5., 5.]])
    np.testing.assert_array_equal(squares, [[25., 13.]])


def test_rejects_negative_density_and_preserves_roundtrip(tmp_path):
    a = atlas()
    path = tmp_path/'atlas.npz'
    a.save(path)
    b = PositiveFluenceAtlas.load(path)
    np.testing.assert_array_equal(a.segments(2., -3., 4., .91)[0],
                                  b.segments(2., -3., 4., .91)[0])
    a.density[0, 0, 0, 0] = -1.
    with pytest.raises(ValueError, match='positive atlas'):
        PositiveFluenceAtlas(a.rho_edges_m, a.z_edges_m, a.beta_nodes,
                             a.density, a.sigma, {})
