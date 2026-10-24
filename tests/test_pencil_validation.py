"""Geometry, units, unbiased weights and equal-observable Green comparisons."""
from dataclasses import replace
import numpy as np
import pytest
from scipy.integrate import quad

from lighthit import synthetic_medium, PointGreenSolver, SolverSettings
from lighthit.angular import angular_components
from lighthit.validation import AxisymmetricCell
from lighthit.validation.beam_rte import ballistic_cells, cell_spectra, global_spectra


def test_cell_volume_and_refinement_keep_the_forward_distribution():
    c = AxisymmetricCell(9., 11., .9, 1.)
    assert c.volume_m3 == pytest.approx(2*np.pi/3*.1*(11**3-9**3))
    refined = c.refined()
    assert refined.mu_max == 1.
    assert refined.mu_min == pytest.approx(.95)
    assert refined.r_max_m-refined.r_min_m == 1.
    with pytest.raises(ValueError): AxisymmetricCell(0., 1., -.5, .5)
    with pytest.raises(ValueError): AxisymmetricCell(1., 2., .5, .5)


def test_global_collision_expansion_matches_the_zero_wave_number_resolvent():
    omega = np.array([0., .03, .1])
    for g in (-.4, 0., .9):
        medium = replace(synthetic_medium(), g=g)
        expected = global_spectra(medium, omega)
        for iw, w in enumerate(omega):
            values = angular_components(np.array([0.]), w, medium, 12, 0)
            got = np.array([v[0, 0] for v in values])/np.sqrt(2)
            np.testing.assert_allclose(got, expected[iw], rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(expected.sum(1),
            1/(medium.absorption_per_m-1j*omega/medium.speed_m_per_ns), rtol=1e-14)


def test_cell_average_reduces_to_a_regular_point_away_from_the_axis():
    medium = synthetic_medium()
    settings = SolverSettings(12, 48, 3., .1, 6)
    cell = AxisymmetricCell(9.999, 10.001, .5999, .6001)
    averaged = cell_spectra(medium, [cell], [0., .03], settings=settings,
                           radial_order=4, polar_order=4)
    point = PointGreenSolver(medium, settings).solve([0., .03], cell.centre_m)
    np.testing.assert_allclose(averaged.components[:, 0], point.components[:, 0],
                               rtol=2e-5, atol=1e-14)


def test_full_shell_beam_average_equals_an_isotropic_source():
    medium = synthetic_medium()
    settings = SolverSettings(8, 32, 2., .1, 6)
    cells = [AxisymmetricCell(9., 11., -1., 1.)]
    a = cell_spectra(medium, cells, settings=settings, radial_order=4, polar_order=24)
    b = cell_spectra(medium, cells, settings=settings, source='isotropic', radial_order=4)
    np.testing.assert_allclose(a.components, b.components, rtol=2e-5, atol=1e-12)


@pytest.fixture(scope='module')
def mc():
    pytest.importorskip('numba')
    from lighthit.validation import beam_mc
    return beam_mc


def test_signed_cones_and_two_shell_crossings_against_dense_ray_sampling(mc):
    cases = [
        (np.array([-10., 2., 0.]), np.array([1., 0., 0.]), AxisymmetricCell(3., 5., -.2, .2)),
        (np.array([2., 1., 8.]), np.array([.2, .3, -1.]), AxisymmetricCell(2., 7., -.8, -.1)),
        (np.array([3., -2., -8.]), np.array([-.1, .2, 1.]), AxisymmetricCell(2., 7., .1, .8)),
    ]
    axis = np.array([0., 0., 1.])
    for x, direction, cell in cases:
        direction /= np.linalg.norm(direction)
        bounds, count = mc._intervals(x, direction, axis, 25., np.array([
            cell.r_min_m, cell.r_max_m, cell.mu_min, cell.mu_max]))
        exact = np.sum(bounds[:count, 1]-bounds[:count, 0])
        n = 200000
        step = 25/n
        points = x+((np.arange(n)+.5)*step)[:, None]*direction
        radius = np.linalg.norm(points, axis=1)
        cosine = points[:, 2]/radius
        inside = ((radius >= cell.r_min_m) & (radius <= cell.r_max_m)
                  & (cosine >= cell.mu_min) & (cosine <= cell.mu_max))
        assert abs(exact-inside.sum()*step) < 10*step
    forward = np.array([9., 11., .9, 1.])
    bounds, count = mc._intervals(np.zeros(3), axis, axis, np.inf, forward)
    np.testing.assert_allclose(bounds[:count], [[9., 11.]])
    bounds, count = mc._intervals(np.zeros(3), axis, axis, np.inf,
                                  np.array([9., 11., .5, .9]))
    assert count == 0


def test_weighted_complex_segment_integral_against_quad(mc):
    for rate, frequency in [(0., 0.), (.07, 0.), (.07, .13), (0., .13)]:
        exact = mc._integral(rate, frequency, .22, 13., 2., 27., 7.)
        f = lambda a: np.exp(-rate*a+1j*frequency*(7+(13+a)/.22))
        numeric = quad(lambda x: f(x).real, 2., 27., epsabs=1e-12)[0]
        numeric += 1j*quad(lambda x: f(x).imag, 2., 27., epsabs=1e-12)[0]
        np.testing.assert_allclose(exact, numeric, rtol=1e-12, atol=1e-12)


def test_hg_inverse_poles_and_first_two_moments(mc):
    rng = np.random.default_rng(81)
    uniform = rng.random(40000)
    for g in (-.8, 0., 1e-10, .9):
        cosine = np.array([mc._hg_cosine(u, g) for u in uniform])
        p2 = (3*cosine*cosine-1)/2
        for value, expected in [(cosine, g), (p2, g*g)]:
            assert abs(value.mean()-expected) < 6*value.std(ddof=1)/np.sqrt(len(value))
        for axis in (np.array([0., 0., 1.]), np.array([0., 0., -1.]), np.array([1., 0., 0.])):
            out = mc._turn(axis, .7, .3)
            assert np.linalg.norm(out) == pytest.approx(1., abs=2e-15)
            assert np.dot(out, axis) == pytest.approx(.7, abs=2e-15)


def test_no_scattering_is_exact_including_axis_frequency_clock_and_bins(mc):
    medium = replace(synthetic_medium(), scattering_per_m=0.)
    cells = [AxisymmetricCell(9., 11., .9, 1.), AxisymmetricCell(9., 11., .5, .9)]
    omega = np.array([0., .05])
    r = mc.estimate_cells(medium, cells, omega_per_ns=omega, photons_per_batch=20,
                         batches=3, emission_time_ns=7., time_edges_ns=[-10., 7., 30., 1000.])
    exact = ballistic_cells(medium, cells, omega, emission_time_ns=7.)
    np.testing.assert_allclose(r.components[:, :, 0], exact, rtol=1e-14, atol=1e-17)
    np.testing.assert_array_equal(r.components[:, :, 1:], 0.)
    np.testing.assert_allclose(r.batch_global.mean(0)[:, 0],
        global_spectra(medium, omega)[:, 0]*np.exp(1j*omega*7), rtol=1e-14)
    np.testing.assert_array_equal(r.batch_bins[:, :, :2], 0.)
    np.testing.assert_allclose(r.batch_bins.mean(0).sum(1)[:, 0], exact[0].real, rtol=1e-14)
    np.testing.assert_allclose(r.effective_histories()[0, 0], 60., rtol=1e-14)


def test_partition_and_correlated_total_uncertainty(mc):
    cells = [AxisymmetricCell(8., 12., -1., 1.)]
    edges = [-1., -.6, 0., .6, 1.]
    cells += [AxisymmetricCell(8., 12., a, b) for a, b in zip(edges[:-1], edges[1:])]
    r = mc.estimate_cells(synthetic_medium(), cells, photons_per_batch=400, batches=5)
    volumes = np.array([c.volume_m3 for c in cells])
    np.testing.assert_allclose(r.batch_components[:, :, 0]*volumes[0],
        (r.batch_components[:, :, 1:]*volumes[None, None, 1:, None]).sum(2),
        rtol=2e-13, atol=1e-14)
    expected = r.batch_components.sum(-1).real.std(0, ddof=1)/np.sqrt(5)
    np.testing.assert_allclose(r.standard_error(total=True)[..., 0], expected)
    np.testing.assert_array_equal(r.standard_error()[..., 1], 0.)
    assert np.all(r.history_standard_error() >= 0)


@pytest.mark.parametrize('mode', ['implicit', 'analog'])
def test_global_normalization_and_frequency_phase_with_independent_batches(mc, mode):
    medium = synthetic_medium()
    omega = np.array([0., .03])
    r = mc.estimate_cells(medium, [AxisymmetricCell(9., 11., .4, .8)],
        omega_per_ns=omega, photons_per_batch=1200, batches=10, seed=812,
        absorption_mode=mode, roulette_weight=.001)
    difference = r.batch_global.mean(0)-global_spectra(medium, omega)
    error = r.standard_error(global_space=True)
    assert np.all(abs(difference.real) <= 6*error[..., 0]+1e-12)
    assert np.all(abs(difference.imag) <= 6*error[..., 1]+1e-12)


def test_analog_and_weighted_estimators_and_roulette_thresholds_agree(mc):
    medium = synthetic_medium()
    cells = [AxisymmetricCell(8., 12., -.5, .5)]
    result = [mc.estimate_cells(medium, cells, photons_per_batch=2000, batches=10,
              seed=seed, absorption_mode=mode, roulette_weight=threshold)
              for mode, threshold, seed in [('analog', .001, 192),
                  ('implicit', .01, 193), ('implicit', 1e-5, 194)]]
    for r in result[1:]:
        difference = r.components.real-result[0].components.real
        se = np.sqrt(r.standard_error()[..., 0]**2+result[0].standard_error()[..., 0]**2)
        assert np.all(abs(difference) <= 6*se+1e-12)


def test_isotropic_direction_proposal_keeps_global_collision_normalization(mc):
    medium = synthetic_medium()
    result = mc.estimate_cells(medium, [AxisymmetricCell(9., 11., -.8, -.4)],
        photons_per_batch=2000, batches=12, isotropic_fraction=.3, seed=195)
    exact = global_spectra(medium, [0.])
    error = result.standard_error(global_space=True)[..., 0]
    assert np.all(abs(result.batch_global.mean(0).real-exact.real) < 6*error+1e-12)
    assert result.components[0, 0, 0] == 0.


def test_local_generator_reproducibility_and_readable_backend(mc):
    cells = [AxisymmetricCell(8., 12., -.5, .5)]
    options = dict(photons_per_batch=20, batches=2, seed=196)
    state = np.random.get_state()
    a = mc.estimate_cells(synthetic_medium(), cells, **options)
    b = mc.estimate_cells(synthetic_medium(), cells, **options)
    c = mc.estimate_cells(synthetic_medium(), cells, backend='python', **options)
    np.testing.assert_array_equal(a.batch_components, b.batch_components)
    np.testing.assert_allclose(a.batch_components, c.batch_components, rtol=1e-12, atol=1e-15)
    after = np.random.get_state()
    np.testing.assert_array_equal(state[1], after[1])
    assert state[2:] == after[2:]


@pytest.mark.parametrize('option', [dict(batches=1), dict(photons_per_batch=True),
    dict(roulette_weight=0), dict(axis=[0, 0, 2]), dict(omega_per_ns=[.1, 0.]),
    dict(source='cone'), dict(absorption_mode='cutoff'), dict(time_edges_ns=[0.]),
    dict(seed=-1), dict(isotropic_fraction=1.), dict(isotropic_fraction=-.1)])
def test_invalid_monte_carlo_options_are_explicit(mc, option):
    with pytest.raises(ValueError):
        mc.estimate_cells(synthetic_medium(), [AxisymmetricCell(9., 11., .5, .9)], **option)


def test_cached_histories_reject_changed_inputs_and_interrupted_files(mc, tmp_path):
    from lighthit.validation.pencil_experiment import _cached_mc
    cells = [AxisymmetricCell(8., 12., .8, 1.)]
    options = (tmp_path, 'cache', synthetic_medium(), cells, [0.], 20, 2)
    first = _cached_mc(*options, 310, False)
    repeated = _cached_mc(*options, 310, True)
    np.testing.assert_array_equal(first.batch_components, repeated.batch_components)
    assert repeated.diagnostics['cache_reused'] is True
    changed = _cached_mc(*options, 311, True)
    assert not np.array_equal(first.batch_components, changed.batch_components)
    (tmp_path/'mc-cache.npz').write_bytes(b'interrupted')
    recovered = _cached_mc(*options, 310, True)
    np.testing.assert_array_equal(first.batch_components, recovered.batch_components)
    assert recovered.diagnostics['cache_reused'] is False


def test_cached_rte_rejects_changed_medium(mc, tmp_path):
    from lighthit.validation.pencil_experiment import _cached_rte
    medium = replace(synthetic_medium(), scattering_per_m=0.)
    cells = [AxisymmetricCell(8., 12., .8, 1.)]
    settings = SolverSettings(0, 0, 1., .1, 4)
    a = _cached_rte(tmp_path, 'cache', medium, cells, [0.], settings, 2, 2, False)
    b = _cached_rte(tmp_path, 'cache', replace(medium, absorption_per_m=.08),
                    cells, [0.], settings, 2, 2, True)
    assert a.components[0, 0, 0] != b.components[0, 0, 0]
