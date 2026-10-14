"""Joint source conservation, clocks, poses and exact spectral thresholds."""
from dataclasses import replace
import numpy as np
import pytest
from scipy.special import eval_legendre

from lighthit.sources import SpectralLightElements
from lighthit.experimental.axial_source import AxisFrame, VACUUM_M_PER_NS
from lighthit.experimental.axial_fast import _layout
from lighthit.experimental.event_moments import real_spherical_harmonics
from lighthit.experimental.joint_source import compile_joint_source, compile_photon_histogram


def source():
    direction = np.array([[0., 0., 1.], [.6, 0., .8], [0., -.8, .6], [0., 0., 1.]])
    beta = np.array([1., .76, .72, .5])
    c0 = np.array([2., 3., 4., 5.])
    return SpectralLightElements(np.array([[0., 0., 0.], [.02, .01, .1], [-.02, -.01, .2], [0., 0., .3]]),
                                direction, np.array([.03, .02, .01, .02]), c0, -c0/beta**2, beta,
                                np.array([0., .3, .7, 1.]), np.array([.1, .4, .8, 1.1]),
                                np.arange(4), np.arange(4), cone_model='spectral')


def frame():
    return AxisFrame(np.zeros(3), np.array([0., 0., 1.]), np.array([1., 0., 0.]), np.array([0., 1., 0.]))


def test_mass_and_exact_variable_beta_cone_moments():
    s = source(); wl = np.array([400., 500.]); phase = np.array([1.3, 1.4])
    result = compile_joint_source(s, wl, phase, frame=frame(), degree=5, azimuthal_degree=3, chunk=2)
    physical = np.where(s.beta[:, None]*phase > 1,
                        np.maximum(0., s.coefficient0[:, None]+s.coefficient2[:, None]/phase**2), 0.)/wl**2
    np.testing.assert_allclose(result.photons_per_nm.sum(axis=1), physical.sum(axis=0), rtol=2e-15)
    _, _, _, _, ell = _layout(5, 3)
    y = real_spherical_harmonics(5, s.direction, 3)
    expected = np.stack([np.sum(physical[:, band, None]*y
                        * eval_legendre(ell[None, :], 1/(s.beta[:, None]*phase[band])), axis=0)
                        for band in range(2)])
    np.testing.assert_allclose(result.angular.sum(axis=1), expected, rtol=2e-13, atol=1e-19)
    np.testing.assert_allclose(result.angular[:, :, 0]*np.sqrt(4*np.pi), result.photons_per_nm, rtol=2e-15)
    assert result.summary['emitting_rows'] == 3
    assert not result.summary['azimuth_symmetry_assumed']
    # The slower third element emits at 1.4 but not at 1.3; no mean-beta model.
    assert physical[2, 0] == 0 and physical[2, 1] > 0


def test_global_clock_shift_and_pose_invariance():
    s = source(); options = dict(degree=4, azimuthal_degree=4, chunk=3)
    a = compile_joint_source(s, [400., 500.], [1.3, 1.4], frame=frame(), **options)
    shift = 1e6
    b = compile_joint_source(replace(s, start_ns=s.start_ns+shift, end_ns=s.end_ns+shift),
                             [400., 500.], [1.3, 1.4], frame=frame(), **options)
    np.testing.assert_array_equal(a.keys, b.keys)
    np.testing.assert_allclose(a.angular, b.angular, rtol=1e-14, atol=1e-18)
    np.testing.assert_allclose(b.times_ns-shift, a.times_ns, atol=2e-10)
    rng = np.random.default_rng(1); rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(rotation) < 0: rotation[:, 0] *= -1
    translation = np.array([500., -100., 17.]); f = frame()
    moved_frame = AxisFrame(translation, rotation @ f.axis, rotation @ f.first, rotation @ f.second)
    moved = replace(s, start_m=s.start_m @ rotation.T+translation, direction=s.direction @ rotation.T)
    c = compile_joint_source(moved, [400., 500.], [1.3, 1.4], frame=moved_frame, **options)
    np.testing.assert_array_equal(a.keys, c.keys)
    np.testing.assert_allclose(a.angular, c.angular, rtol=2e-12, atol=1e-18)
    np.testing.assert_allclose(c.points_m, a.points_m @ rotation.T+translation, atol=1e-12)
    np.testing.assert_allclose(c.times_ns, a.times_ns, atol=1e-12)


def test_front_zero_residual_and_joint_correlation():
    s = source(); beta = np.ones(4); length = np.full(4, .01)
    position = np.array([[0., 0., z] for z in [0., 1., 2., 3.]])
    start = position[:, 2]/VACUUM_M_PER_NS
    s = replace(s, start_m=position, direction=np.tile([0., 0., 1.], (4, 1)), length_m=length,
                beta=beta, coefficient2=-s.coefficient0, start_ns=start,
                end_ns=start+length/VACUUM_M_PER_NS)
    result = compile_joint_source(s, [450.], [1.35], frame=frame(), z_step_m=.5, tau_step_ns=.05)
    assert np.all(result.keys[:, 2] == 0)
    assert np.max(result.residual_variance_ns2) < 1e-28
    np.testing.assert_allclose(result.times_ns, result.points_m[:, 2]/VACUUM_M_PER_NS, atol=2e-15)
    # Equal z but opposite mu/time remain distinct cells, not product marginals.
    paired = replace(s, start_m=np.zeros((4, 3)), direction=np.array([[0., 0., 1.], [0., 0., -1.]]*2),
                     start_ns=np.array([0., 2., 0., 2.]), end_ns=np.array([.03, 2.03, .03, 2.03]))
    joint = compile_joint_source(paired, [450.], [1.35], frame=frame(), chord_order=1, tau_step_ns=.1)
    assert len(joint.keys) == 2
    assert joint.keys[0, 1]*joint.keys[1, 1] < 0
    assert joint.keys[0, 2] != joint.keys[1, 2]


def test_no_tail_clipping_and_empty_dark_source():
    s = source(); s = replace(s, start_ns=np.array([0., .3, 6e5, 1.]), end_ns=np.array([.1, .4, 6e5+.1, 1.1]))
    result = compile_joint_source(s, [450.], [1.4], frame=frame(), degree=2)
    assert result.times_ns.max() > 5e5
    dark = compile_joint_source(s, [450.], [.99], frame=frame(), degree=2)
    assert dark.angular.shape == (1, 0, 9)
    assert dark.points_m.shape == (0, 3)
    np.testing.assert_array_equal(dark.spectrum([0., 1.]), np.zeros((1, 9, 2)))


@pytest.mark.parametrize('bad', [dict(mu_step=0), dict(tau_step_ns=np.inf), dict(degree=True), dict(chunk=0)])
def test_invalid_grids_fail(bad):
    with pytest.raises(ValueError):
        compile_joint_source(source(), [450.], [1.35], frame=frame(), **bad)


def test_positive_photon_histogram_conserves_mass_and_exposes_original_samples():
    s = source(); result = compile_photon_histogram(s, 480., 1.4, frame=frame(), ring_nodes=16, retain_samples=True)
    expected = np.where(s.beta*1.4 > 1, np.maximum(0., s.coefficient0+s.coefficient2/1.4**2), 0.).sum()/480**2
    assert np.all(result.photon_weights > 0)
    np.testing.assert_allclose(result.photon_weights.sum(), expected, rtol=3e-15)
    np.testing.assert_allclose(result.original_photons_per_nm.sum(), expected, rtol=3e-15)
    np.testing.assert_allclose(np.linalg.norm(result.direction, axis=1), 1, atol=3e-15)
    np.testing.assert_allclose(np.linalg.norm(result.original_direction, axis=1), 1, atol=3e-15)
    assert result.original_position.shape == (3*2*16, 3)
    # Photon directions cover nonzero azimuth, rather than an axisymmetric marginal.
    assert len(np.unique(result.geometry_keys[:, 2])) > 4
    exact = compile_joint_source(s, [480.], [1.4], frame=frame(), degree=5, azimuthal_degree=5)
    sampled = np.sum(real_spherical_harmonics(5, result.original_direction)
                     * result.original_photons_per_nm[:, None], axis=0)
    # Uniform ring quadrature exactly integrates retained finite harmonics
    # when the ring has more nodes than their azimuthal bandwidth.
    np.testing.assert_allclose(sampled, exact.angular.sum(axis=1)[0], rtol=3e-13, atol=1e-18)
    group = result.original_geometry_id
    assert group.shape == result.original_times_ns.shape
    for geometry in range(len(result.position)):
        selected = group == geometry; weight = result.original_photons_per_nm[selected]
        weight = weight/weight.sum(); point = result.original_position[selected]
        direction = result.original_direction[selected]
        centre = np.sum(weight[:, None]*point, axis=0)
        mean = np.sum(weight[:, None]*direction, axis=0)
        np.testing.assert_allclose(centre, result.position[geometry], atol=1e-14)
        np.testing.assert_allclose(mean, result.geometry_direction_mean[geometry], atol=1e-14)
        pcov = np.einsum('n,ni,nj->ij', weight, point-centre, point-centre)
        dcov = np.einsum('n,ni,nj->ij', weight, direction-mean, direction-mean)
        cross = np.einsum('n,ni,nj->ij', weight, point-centre, direction-mean)
        np.testing.assert_allclose(pcov, result.geometry_position_covariance_m2[geometry], atol=1e-16)
        np.testing.assert_allclose(dcov, result.geometry_direction_covariance[geometry], atol=1e-16)
        np.testing.assert_allclose(cross, result.geometry_position_direction_covariance_m[geometry], atol=1e-16)
        angles = np.arccos(np.clip(direction @ result.direction[geometry], -1, 1))
        assert np.sqrt(np.sum(weight*angles**2)) <= result.geometry_angular_rms_upper_rad[geometry]+3e-8
        assert angles.max() <= result.geometry_sample_angular_max_rad[geometry]+3e-8


def test_positive_geometry_is_shared_across_clocks_without_factorising_source():
    s = source().subset(np.array([True, False, False, False]))
    s = replace(s, start_m=np.tile(s.start_m, (2, 1)), direction=np.tile(s.direction, (2, 1)),
                length_m=np.tile(s.length_m, 2), coefficient0=np.tile(s.coefficient0, 2),
                coefficient2=np.tile(s.coefficient2, 2), beta=np.tile(s.beta, 2),
                start_ns=np.array([0., 2.]), end_ns=np.array([.1, 2.1]), row_index=np.arange(2), uid=np.arange(2))
    a = compile_photon_histogram(s, 480., 1.35, frame=frame(), ring_nodes=8, phi_step_rad=.8, tau_step_ns=.1)
    assert len(a.geometry_id) > len(a.position)
    assert a.summary['clock_reuse_ratio'] > 1
    shift = 1e5
    b = compile_photon_histogram(replace(s, start_ns=s.start_ns+shift, end_ns=s.end_ns+shift),
                                 480., 1.35, frame=frame(), ring_nodes=8, phi_step_rad=.8, tau_step_ns=.1)
    np.testing.assert_array_equal(a.clock_keys, b.clock_keys)
    np.testing.assert_allclose(a.photon_weights, b.photon_weights)
    np.testing.assert_allclose(a.emission_times_ns, b.emission_times_ns-shift, atol=2e-11)


def test_payload_budget_prevents_dense_source_allocation():
    with pytest.raises(ValueError, match='payload'):
        compile_joint_source(source(), [450.], [1.35], frame=frame(), max_payload_bytes=1)


@pytest.mark.parametrize('degree,m', [(5, 0), (11, 4), (11, 11)])
def test_once_charged_harmonics_ordered_accumulator_matches_original_scatter(degree, m):
    from lighthit.experimental.joint_source import _charged_harmonics, _accumulate_spectral_cones
    from lighthit._shower_basis import _cone_harmonics
    rng = np.random.default_rng(8)
    vectors = rng.normal(size=(257, 3)); vectors /= np.linalg.norm(vectors, axis=1)[:, None]
    starts, aa, bb, kept, _ = _layout(degree, m)
    cones = rng.uniform(.65, 1., (9, len(vectors)))
    weights = rng.uniform(0., 20., cones.shape); weights[rng.random(weights.shape) < .2] = 0.
    inverse = rng.integers(0, 19, len(vectors), dtype=np.int64)
    expected = np.zeros((9, 19, len(kept))); expected_mass = np.zeros((9, 19))
    for band in range(9):
        temporary = np.empty((len(vectors), len(kept)))
        _cone_harmonics(vectors, cones[band], starts, aa, bb, m, temporary)
        np.add.at(expected[band], inverse, temporary*weights[band, :, None])
        np.add.at(expected_mass[band], inverse, weights[band])
    actual = np.zeros_like(expected); actual_mass = np.zeros_like(expected_mass)
    harmonics = _charged_harmonics(vectors, starts, aa, bb, m)
    _accumulate_spectral_cones(harmonics, cones, weights, inverse, starts, actual, actual_mass)
    np.testing.assert_allclose(actual, expected, rtol=3e-14, atol=1e-12)
    np.testing.assert_array_equal(actual_mass, expected_mass)


def test_vectorised_positive_histogram_matches_streaming_source():
    from lighthit.experimental.joint_source import _compile_photon_histogram_streaming
    options = dict(frame=frame(), ring_nodes=24, chord_order=2, retain_samples=True, chunk=2)
    a = compile_photon_histogram(source(), 480., 1.4, **options)
    b = _compile_photon_histogram_streaming(source(), 480., 1.4, **options)
    for name in ['clock_keys', 'geometry_keys', 'geometry_id', 'original_position',
                 'original_direction', 'original_times_ns', 'original_photons_per_nm']:
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
    for name in ['position', 'direction', 'emission_times_ns', 'photon_weights']:
        np.testing.assert_allclose(getattr(a, name), getattr(b, name), rtol=2e-13, atol=2e-13)


@pytest.mark.parametrize('scale', [20, 2**40])
def test_scalar_key_sort_matches_sparse_tuple_sort_including_overflow_fallback(scale):
    from lighthit.experimental.joint_source import _unique_integer_rows
    rng = np.random.default_rng(4); keys = rng.integers(-scale, scale, (130, 6), dtype=np.int64)
    keys[100:] = keys[:30]
    expected, inverse = np.unique(keys, axis=0, return_inverse=True)
    actual, labels = _unique_integer_rows(keys)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(labels, inverse)


def test_retained_positive_covariances_are_stable_under_global_clock_shift():
    s = source(); options = dict(frame=frame(), ring_nodes=16, retain_samples=True, chunk=2)
    a = compile_photon_histogram(s, 480., 1.4, **options)
    shift = 6e5
    b = compile_photon_histogram(replace(s, start_ns=s.start_ns+shift, end_ns=s.end_ns+shift),
                                 480., 1.4, **options)
    np.testing.assert_array_equal(a.original_geometry_id, b.original_geometry_id)
    np.testing.assert_allclose(a.emission_times_ns, b.emission_times_ns-shift, atol=2e-10)
    np.testing.assert_allclose(a.geometry_position_covariance_m2, b.geometry_position_covariance_m2, atol=1e-16)
    np.testing.assert_allclose(a.geometry_direction_covariance, b.geometry_direction_covariance, atol=1e-16)
