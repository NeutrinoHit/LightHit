"""Directional fluence moments against independent photon-ensemble averages."""
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip('numba')
from lighthit.muon.angular_preview import (
    endpoint_directions, angular_histogram, cubic_response, query_angular_groups)
from lighthit.muon.positive_fluence import (
    PositiveFluenceAtlas, universal_endpoints, endpoint_histogram)


@pytest.mark.parametrize('order', [1, 2])
@pytest.mark.parametrize('g', [0., .9])
def test_retaining_final_ray_preserves_paths_and_random_stream(order, g):
    xyz, orders, directions = endpoint_directions(1000, order, 19, g)
    old_xyz, old_orders = universal_endpoints(1000, order, 19, g)
    np.testing.assert_array_equal(xyz, old_xyz)
    np.testing.assert_array_equal(orders, old_orders)
    np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1., atol=3e-15)


@pytest.mark.parametrize('cosine', [.7, 1.])
def test_directional_histogram_preserves_scalar_counts_and_sampling_spread(cosine):
    xyz, orders, directions = endpoint_directions(2000, 2, 21)
    weights = .3**orders/.5**(orders-1)
    rho, z = np.linspace(0., 20., 17), np.linspace(-20., 20., 31)
    moments, squares = angular_histogram(xyz, directions, weights, cosine, 2., rho, z)
    old, old_squares = endpoint_histogram(xyz, weights, cosine, 2., rho, z)
    np.testing.assert_array_equal(moments[..., 0], old)
    np.testing.assert_array_equal(squares, old_squares)


def test_cubic_contraction_equals_weighted_photon_acceptance_for_tilted_om():
    rng = np.random.default_rng(7)
    rays = rng.normal(size=(300, 3)); rays /= np.linalg.norm(rays, axis=1)[:, None]
    weights = rng.uniform(.1, 2., len(rays))
    # Every endpoint lies on +x; the local meridian directions are (ray_x, ray_z).
    xyz = np.tile([1., 0., 0.], (len(rays), 1))
    moments, _ = angular_histogram(xyz, rays, weights, 1., 1.,
                                   np.array([0., 2.]), np.array([-1., 1.]))
    polynomial = np.array([.5, .3, .1, .1])
    for look in np.vstack((np.eye(3), rng.normal(size=(12, 3)))):
        look = look / np.linalg.norm(look)
        reflected = rays.copy(); reflected[:, 1] *= -1
        expected = .5*np.dot(weights, np.polynomial.polynomial.polyval(rays@look, polynomial)
                             + np.polynomial.polynomial.polyval(reflected@look, polynomial))
        actual = cubic_response(moments[0, 0], look[0], look[2], look[1]**2, polynomial)
        np.testing.assert_allclose(actual, expected, rtol=3e-15)


def directional_atlas():
    xyz = np.tile([1., 0., 0.], (2, 1))
    rays = np.array([[.6, 0., .8], [-.2, np.sqrt(.87), -.3]])
    row, _ = angular_histogram(xyz, rays, np.array([2., 1.]), 1., 1.,
                               np.array([0., 2.]), np.array([-2., 2.]))
    moments = np.broadcast_to(row[0, 0], (2, 2, 2, 2, 10)).copy()
    return PositiveFluenceAtlas(np.array([0., 1., 3.]), np.array([-2., 0., 2.]),
                                np.array([.8, 1.]), moments[..., 0].copy(),
                                np.zeros(moments.shape[:-1]), {}, moments)


def test_lookup_uses_actual_orientation_and_is_invariant_under_rotation():
    atlas = directional_atlas()
    groups = SimpleNamespace(points=np.zeros((1, 3)), directions=np.array([[0., 0., 1.]]),
                             coefficient0=np.array([2.]), beta=np.array([.9]))
    detector = SimpleNamespace(positions_m=np.array([[1., 0., 0.], [1., 0., 0.]]),
        orientations=np.array([[0., 0., -1.], [0., 0., 1.]]), effective_area_m2=np.array([.1, .2]))
    # A(c) = 1 + .5 c.  z=0 is a histogram edge: use the same right cell as the scalar query.
    alpha = 4*np.pi*np.array([1., .5/3])
    result = query_angular_groups(atlas, groups, detector, alpha)
    np.testing.assert_allclose(result, [[.73, .73], [.94, .94]], atol=2e-15)
    rotation = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
    groups.directions = groups.directions@rotation.T
    detector.positions_m = detector.positions_m@rotation.T
    detector.orientations = detector.orientations@rotation.T
    np.testing.assert_allclose(query_angular_groups(atlas, groups, detector, alpha), result)


def test_isotropic_acceptance_and_axis_limit_match_scalar_fluence():
    atlas = directional_atlas()
    groups = SimpleNamespace(points=np.zeros((1, 3)), directions=np.array([[0., 0., 1.]]),
                             coefficient0=np.array([2.]), beta=np.array([.9]))
    detector = SimpleNamespace(positions_m=np.array([[1., 0., .5], [0., 0., .5], [0., 0., .5]]),
        orientations=np.array([[0., 0., -1.], [1., 0., 0.], [0., 1., 0.]]), effective_area_m2=np.ones(3))
    scalar, _ = atlas.points([1., 0., 0.], .5, .9)
    np.testing.assert_allclose(query_angular_groups(atlas, groups, detector, [4*np.pi]), 2*scalar)
    alpha = 4*np.pi*np.array([1., .5/3])
    actual = query_angular_groups(atlas, groups, detector, alpha)
    np.testing.assert_array_equal(actual[1], actual[2])


def test_angular_atlas_roundtrip_and_zeroth_moment_contract(tmp_path):
    atlas = directional_atlas()
    path = tmp_path/'directional.npz'
    atlas.save(path)
    restored = PositiveFluenceAtlas.load(path)
    np.testing.assert_array_equal(restored.angular_moments, atlas.angular_moments)
    atlas.angular_moments[..., 0] *= 2
    with pytest.raises(ValueError, match='arrival-direction moments'):
        PositiveFluenceAtlas(atlas.rho_edges_m, atlas.z_edges_m, atlas.beta_nodes,
                             atlas.density, atlas.sigma, {}, atlas.angular_moments)
