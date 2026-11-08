"""Continuous-history reference: exact limits and complete path accounting."""
import numpy as np
import pytest

pytest.importorskip("numba")

from lighthit.medium import Medium
from lighthit.experimental.halfspace import SurfacePort
from lighthit.experimental.halfspace_boundary import HalfSpaceBoundary
from lighthit.experimental.halfspace_mc import sample_halfspace


def test_absorption_only_reference_and_exact_surface_entry():
    medium = Medium(.2, 0., .9, 1.47)
    sample = sample_halfspace(medium, HalfSpaceBoundary(1.4), SurfacePort.pencil(), photons=100, seed=1)
    balance = sample.photon_balance()
    assert balance["specular"] == pytest.approx(1/36)
    assert balance["absorbed"] == pytest.approx(35/36)
    assert balance["escaped"] == 0
    assert balance["residual"] == pytest.approx(0., abs=1e-15)


def test_full_hg_reference_matches_exact_isotropic_first_collision():
    medium = Medium(.2, 1., 0., 1.4)
    sample = sample_halfspace(medium, HalfSpaceBoundary(), SurfacePort.pencil(), photons=150000, seed=2271)
    estimate = sample.spectrum([[0., 0.]], [0.])
    exact = medium.scattering_per_m/(2*medium.extinction_per_m)*(1-np.log(2))
    assert abs(estimate.mean[0, 0, 1].real-exact) < 5*estimate.stderr_real[0, 0, 1]
    assert estimate.mean[0, 0, 0] == 0
    offset = .4
    estimate = sample.spectrum([[0., 0.]], [0.], absorption_offset_per_m=offset)
    exact = medium.scattering_per_m/(2*(medium.extinction_per_m+offset))*(1-np.log(2))
    assert abs(estimate.mean[0, 0, 1].real-exact) < 5*estimate.stderr_real[0, 0, 1]


def test_recorded_cell_lengths_include_reflections_and_time_accounting():
    medium = Medium(.2, 1., .3, 1.4)
    # This cell contains every finite history in this bounded sample.
    cell = np.array([[[-1e8, -1e8, 0.], [1e8, 1e8, 1e8]]])
    sample = sample_halfspace(medium, HalfSpaceBoundary(1.4), SurfacePort.pencil(),
                             photons=5000, seed=391, sensitivity_cells_m=cell)
    np.testing.assert_allclose(sample.cell_path_m[:, 0], sample.path_m, atol=1e-12)
    balance = sample.photon_balance()
    assert abs(balance["residual"]) < 1e-15
    end = sample.path_m.max()/medium.speed_m_per_ns + 1
    bins = sample.time_bins(np.linspace(0., end, 20))
    assert bins.mean.sum() == pytest.approx(balance["escaped"], abs=2e-15)
    derivative = sample.spectrum([[0., 0.]], [0.], absorption_derivative=True)
    assert derivative.mean.sum().real < 0


def test_event_limit_is_an_error_instead_of_a_truncated_reference():
    medium = Medium(.001, 10., 0., 1.4)
    with pytest.raises(RuntimeError, match="no truncated estimate"):
        sample_halfspace(medium, HalfSpaceBoundary(), SurfacePort.pencil(),
                         photons=50, seed=9, max_events=1)


def test_cell_sensitivity_orders_partition_the_total():
    medium = Medium(.2, 1., .3, 1.4)
    cells = np.array([[[0., -.2, .1], [.4, .2, .4]], [[.6, -.2, .1], [1., .2, .4]]])
    sample = sample_halfspace(medium, HalfSpaceBoundary(1.4), SurfacePort.pencil(), photons=20000,
                              seed=17, source_sigma_m=.4, sensitivity_cells_m=cells)
    total = sample.cell_absorption_sensitivity(detector_sigma_m=.4, half_angle_rad=.7)
    orders = sample.cell_absorption_sensitivity(detector_sigma_m=.4, half_angle_rad=.7, by_order=True)
    assert orders.mean.shape == (2, 3) and not orders.mean[:, 0].any()
    np.testing.assert_allclose(orders.mean.sum(axis=1), total.mean, rtol=1e-12)


def test_cell_sensitivity_at_frequency_weights_each_history_by_its_flight_time():
    medium = Medium(.2, 1., .3, 1.4)
    cells = np.array([[[0., -.2, .1], [.4, .2, .4]], [[.6, -.2, .1], [1., .2, .4]]])
    sample = sample_halfspace(medium, HalfSpaceBoundary(1.4), SurfacePort.pencil(), photons=20000,
                              seed=19, source_sigma_m=.4, sensitivity_cells_m=cells)
    omega = .7
    stationary = sample.cell_absorption_sensitivity(detector_sigma_m=.4, half_angle_rad=.7)
    np.testing.assert_array_equal(
        sample.cell_absorption_sensitivity(detector_sigma_m=.4, half_angle_rad=.7, omega_per_ns=0.).mean,
        stationary.mean)
    actual = sample.cell_absorption_sensitivity(detector_sigma_m=.4, half_angle_rad=.7, omega_per_ns=omega,
                                                by_order=True)
    gaussian = np.exp(-((sample.x_m-1.)**2+sample.y_m**2)/(2*.4**2))/(2*np.pi*.4**2)
    phase = np.exp(1j*omega*sample.path_m/sample.speed_m_per_ns)
    volume = np.prod(cells[:, 1]-cells[:, 0], axis=1)
    value = -sample.cell_path_m*(sample._weights(.7)*gaussian*phase)[:, None]/volume
    np.testing.assert_allclose(actual.mean.sum(axis=1), value.mean(axis=0), rtol=1e-12)
    assert np.all(np.abs(actual.mean[:, 2].imag) > 0) and not actual.mean[:, 0].any()
