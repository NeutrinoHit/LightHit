"""Geometry and collection area are independent detector inputs."""
from dataclasses import replace
import numpy as np
import pytest

import lighthit as lh
from lighthit.ballistic_interval import ballistic_interval, BallisticIntervalConfig
from test_ballistic_finite_receiver import setup, element
from test_model_file import model


def test_ray_entry_uses_the_surface_and_retains_misses():
    surface = lh.SphereSurface(2.)
    origin = np.array([[-5., 0., 0.], [-5., 1., 0.], [-5., 3., 0.], [0., 0., 0.]])
    distance = surface.entry_distance(origin, np.tile([1., 0., 0.], (4, 1)))
    np.testing.assert_allclose(distance[:2], [3., 5. - np.sqrt(3.)])
    assert np.isinf(distance[2]) and distance[3] == 0.


def test_surface_roundtrip_and_receiver_subset_keep_physical_radii(tmp_path):
    original = model()
    loaded = lh.load_model(original.save(tmp_path / 'model.npz'))
    np.testing.assert_array_equal(loaded.detector.receiver_surface.radii(3), [.216, .20, .25])
    sub = loaded.detector.subset([2, 0])
    np.testing.assert_array_equal(sub.receiver_surface.radii(2), [.25, .216])
    np.testing.assert_array_equal(sub.positions_m, loaded.detector.positions_m[[2, 0]])


def test_collection_area_does_not_change_surface_geometry():
    cot = 1. / np.sqrt(1.35 ** 2 - 1.)
    kernel = setup([2., 0., 2. * cot + .025])
    source = element([[0., 0., 0.]], .05)
    first = ballistic_interval(kernel, source, config=BallisticIntervalConfig(receiver_mode="surface")).charge_pe
    kernel.detector = replace(kernel.detector, effective_area_m2=.025)
    second = ballistic_interval(kernel, source, config=BallisticIntervalConfig(receiver_mode="surface")).charge_pe
    np.testing.assert_allclose(second, .5 * first, rtol=1e-12)


def test_surface_cannot_be_inferred_from_collection_area():
    kernel = setup([2., 0., 3.])
    kernel.detector = replace(kernel.detector, receiver_surface=None)
    source = element([[0., 0., 0.]], .05)
    with pytest.raises(ValueError, match='receiver_surface'):
        ballistic_interval(kernel, source, config=BallisticIntervalConfig(receiver_mode="surface"))
    ballistic_interval(kernel, source, config=BallisticIntervalConfig(receiver_mode='point'))


def test_duplicated_emission_has_no_source_size_or_distance_switch():
    cot = 1. / np.sqrt(1.35 ** 2 - 1.)
    kernel = setup([60., 0., 60. * cot + .025])
    single = element([[0., 0., 0.]], .05)
    count = 8001
    duplicate = element(np.zeros((count, 3)), np.full(count, .05), c0=1e8/count)
    q = ballistic_interval(kernel, single, config=BallisticIntervalConfig(receiver_mode="surface")).charge_pe
    np.testing.assert_allclose(ballistic_interval(kernel, duplicate, config=BallisticIntervalConfig(receiver_mode="surface")).charge_pe, q, rtol=2e-11)
