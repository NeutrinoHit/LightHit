import numpy as np
import pytest

import lighthit as lh


def detector():
    return lh.DetectorArray(
        [[-2., 0., -5.], [2., 0., 5.], [10., 4., 0.]],
        [[0., 0., 1.]] * 3, .05,
        lambda x: np.ones_like(np.asarray(x, float)),
        lambda w: np.ones_like(np.asarray(w, float)),
        identifiers={"cluster": [1, 1, 2]})


def test_detector_geometry_summary_reports_explicit_bounds_and_clusters():
    summary = lh.describe_geometry(detector())
    np.testing.assert_allclose(summary.centroid_m, [10 / 3, 4 / 3, 0])
    np.testing.assert_array_equal(summary.bounds_min_m, [-2, 0, -5])
    np.testing.assert_array_equal(summary.bounds_max_m, [10, 4, 5])
    np.testing.assert_array_equal(summary.size_m, [12, 4, 10])
    assert summary.array.modules == 3
    assert set(summary.clusters) == {1, 2}
    assert summary.clusters[1].modules == 2
    assert summary.as_dict()["clusters"]["2"]["centroid_m"] == [10., 4., 0.]


def test_point_source_radial_range_covers_all_modules_independently_of_brightness():
    array = detector()
    position = np.array([0., 0., 0.])
    low, high = lh.point_source_radial_range(
        array, position, minimum_range_m=(1., 5.))
    radii = np.linalg.norm(array.positions_m - position[None, :], axis=1)
    assert low == 1.
    assert high > radii.max()
    with pytest.raises(ValueError, match="closer"):
        lh.point_source_radial_range(array, [-2., 0., -5.])


def test_bright_laser_example_resolves_spatial_quadrature_at_far_om(tmp_path):
    from examples.laser_cache import laser_config

    config = laser_config(detector(), cache_directory=tmp_path,
                          cache_policy="require")
    assert config.k_panel_per_m <= 2 * np.pi / config.radial_range_m[1]
    assert config.radial_range_m[1] >= 300.
    wider = laser_config(detector(), cache_directory=tmp_path,
                         max_radius_m=500.)
    assert wider.radial_range_m == (config.radial_range_m[0], 500.)
    assert wider.k_panel_per_m < config.k_panel_per_m


def test_laser_example_reports_an_outdated_installed_package(tmp_path, monkeypatch):
    from examples.laser_cache import laser_config

    monkeypatch.delattr(lh, "point_source_radial_range")
    with pytest.raises(RuntimeError, match="require LightHit 0.2.0a8 or newer") as exc:
        laser_config(detector(), cache_directory=tmp_path)
    assert lh.__file__ in str(exc.value)
    assert "pip install -e" in str(exc.value)


def test_track_elements_are_placed_by_numeric_centroid_axis_and_time():
    elements = lh.CherenkovTrack.centered(
        [0, 0, 0], [0, 0, 1], 4., time_ns=3.).to_elements(step_m=.25)
    pose = lh.SourcePose([12., -4., 7.], [1., 2., 3.], time_ns=19.)
    placed = elements.placed(pose)
    np.testing.assert_allclose(placed.centroid_m, pose.position_m, atol=1e-12)
    np.testing.assert_allclose(placed.principal_axis, pose.direction, atol=1e-12)
    assert placed.start_ns.min() == pytest.approx(19.)
    assert placed.provenance["pose"]["target_centroid_m"] == [12., -4., 7.]
    assert np.linalg.det(np.asarray(placed.provenance["pose"]["rotation"])) == pytest.approx(1.)


def test_synthetic_shower_is_deterministic_and_explicitly_placed():
    pose = lh.SourcePose([30., 20., -100.], [.2, -.3, .9], time_ns=5.)
    first = lh.SyntheticShower.gaussian(pose, elements=24, seed=7)
    second = lh.SyntheticShower.gaussian(pose, elements=24, seed=7)
    np.testing.assert_array_equal(first.elements.start_m, second.elements.start_m)
    np.testing.assert_allclose(first.centroid_m, pose.position_m, atol=1e-12)
    np.testing.assert_allclose(first.principal_axis, pose.direction, atol=1e-12)
    assert first.elements.start_ns.min() == pytest.approx(5.)
    assert first.elements.coefficient0.sum() > 0


def test_g4_shower_placement_preserves_input_identity():
    raw = lh.CherenkovTrack.centered([0, 0, 0], [0, 0, 1], 2.).to_elements(step_m=.2)
    shower = lh.G4Shower(raw, "event.h5", 5)
    pose = lh.SourcePose([1, 2, 3], [0, 1, 0], 11.)
    placed = shower.placed(pose)
    assert placed.input_path == "event.h5"
    assert placed.event == 5
    np.testing.assert_allclose(placed.centroid_m, pose.position_m, atol=1e-12)
    np.testing.assert_allclose(placed.principal_axis, pose.direction, atol=1e-12)


def test_long_track_time_origin_minimizes_emission_plus_group_flight():
    track = lh.CherenkovTrack([0., 0., -210.], [0., 0., 1.], 420., beta=.99)
    receivers = np.array([[30., 0., 0.], [0., 0., -200.], [30., 0., 300.]])
    group_index = 1.37
    origins = track.earliest_arrival_ns(receivers, group_index)
    distance = np.linspace(0., track.length_m, 100_001)
    start = np.asarray(track.start_m, float)
    direction = np.asarray(track.direction, float)
    points = start[None, :] + distance[:, None] * direction
    for index, receiver in enumerate(receivers):
        samples = (track.time_ns + distance / (track.beta * .299792458)
                   + np.linalg.norm(points - receiver[None, :], axis=1)
                   * group_index / .299792458)
        assert origins[index] == pytest.approx(samples.min(), abs=.02)
    old_origin = np.linalg.norm(receivers[0] - start) * group_index / .299792458
    assert old_origin - origins[0] > 60.
