import warnings

import numpy as np
import pytest
from scipy.special import ndtr
from lighthit.readout import inverse_bins, readout_bins
from lighthit.transport import TransportResponse


def test_frequency_inversion_gaussian_delta():
    omega=np.linspace(0,5,2001); t0=12.;sigma=2.
    edges=np.arange(-4,30,1.)
    spectrum=np.exp(1j*omega*t0)[:,None]
    actual=inverse_bins(omega,spectrum,edges,sigma)[0]
    expected=ndtr((edges[1:]-t0)/sigma)-ndtr((edges[:-1]-t0)/sigma)
    np.testing.assert_allclose(actual,expected,atol=5e-14)


@pytest.mark.parametrize("omega,edges,sigma", [([0],[0,1],0),([.1,.2],[0,1],0),([0,.1],[1,0],0),([0,.1],[0,1],-1)])
def test_invalid_readout(omega,edges,sigma):
    with pytest.raises(ValueError):inverse_bins(omega,np.ones((len(omega),1)),edges,sigma)


def test_response_rebins_only_fourier_derived_orders_at_smaller_cutoff():
    omega = np.array([0.0, 0.5, 1.0])
    edges = np.array([-1.0, 0.0, 1.0])
    origin = np.array([3.0])
    spectrum = np.zeros((3, 1, 3), complex)
    spectrum[:, 0, 2] = np.exp(1j * omega * origin[0])
    components = np.zeros((1, 2, 3))
    components[0, :, 1] = [0.25, 0.75]
    components[0, :, 2] = inverse_bins(
        omega, spectrum[:, :, 2] * np.exp(-1j * omega[:, None] * origin), edges)[0]
    response = TransportResponse(
        None, omega, edges, spectrum, components, np.zeros((1, 3)), origin,
        np.ones(1, bool), "test", {"fourier_inverted_orders": [2]})

    got = response.components_at_frequency_cutoff(0.5)
    expected_tail = inverse_bins(
        omega[:2], spectrum[:2, :, 2] * np.exp(-1j * omega[:2, None] * origin), edges)[0]
    np.testing.assert_array_equal(got[0, :, 1], components[0, :, 1])
    np.testing.assert_allclose(got[0, :, 2], expected_tail)


@pytest.mark.parametrize("cutoff", [0.0, np.nan, 0.1])
def test_response_rebin_rejects_invalid_frequency_cutoff(cutoff):
    response = TransportResponse(
        None, np.array([0.0, 0.5]), np.array([0.0, 1.0]),
        np.zeros((2, 1, 3), complex), np.zeros((1, 1, 3)),
        np.zeros((1, 3)), np.zeros(1), np.ones(1, bool), "test", {})
    with pytest.raises(ValueError):
        response.components_at_frequency_cutoff(cutoff)


def _exponential_pulse(tau_ns, omega, edges):
    """Spectrum and exact bins of exp(-t/tau)/tau, t>0 (unit charge)."""
    spectrum = (1 / (1 - 1j * np.asarray(omega) * tau_ns))[:, None]
    cumulative = np.where(edges > 0, 1 - np.exp(-np.clip(edges, 0, None) / tau_ns), 0.)
    return spectrum, np.diff(cumulative)


def test_table_grid_repeats_the_waveform_inside_a_long_window():
    # 65 table frequencies up to 1.2 rad/ns repeat every 2 pi / 0.01875 = 335 ns;
    # the 805 ns window of the route configuration then holds the pulse ~2.4 times.
    omega, edges = np.linspace(0., 1.2, 65), np.arange(-60., 745., 5.)
    spectrum, _ = _exponential_pulse(40., omega, edges)
    with pytest.warns(RuntimeWarning, match="period of the frequency grid"):
        aliased = readout_bins(omega, spectrum, edges)[0]
    assert aliased.sum() > 2.


def test_readout_grid_removes_the_copies_and_keeps_the_charge():
    # The spline of the table-grid samples resolves a pulse much shorter than
    # the 335 ns table period (here 20 ns); longer tails are attenuated.
    omega, edges = np.linspace(0., 1.2, 65), np.arange(-60., 745., 5.)
    readout = np.linspace(0., 1.2, 321)
    spectrum, exact = _exponential_pulse(20., omega, edges)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        bins = readout_bins(omega, spectrum, edges, readout)[0]
    assert bins.sum() == pytest.approx(1., abs=1e-2)
    direct = inverse_bins(readout, _exponential_pulse(20., readout, edges)[0], edges)[0]
    assert np.abs(bins - direct).sum() < 1e-2
    assert np.abs(bins - exact).sum() < .05


def test_kernel_transport_reads_out_on_the_readout_grid(tmp_path):
    import lighthit as lh
    from dataclasses import replace
    medium = lh.SpectralMedium([400., 500.], [.04, .05], [.1, .12], [1.34, 1.36],
                               [1.37, 1.39], g=.9, provenance="test spectrum")
    detector = lh.DetectorArray(
        [[10., 0., 0.], [0., 20., 0.]], [[-1., 0., 0.], [0., -1., 0.]],
        .05, lambda x: np.clip((1 + np.asarray(x)) / 2, 0, 1),
        lambda wavelength: np.ones_like(np.asarray(wavelength, float)) * .2)
    omega, readout = np.linspace(0., .4, 33), np.linspace(0., .4, 161)
    edges = np.arange(-10., 1501., 10.)        # three periods of the table grid
    config = lh.KernelConfig(
        omega_per_ns=omega, relative_time_edges_ns=edges, wavelength_nodes=2,
        scattering_degree=4, source_degree=4, azimuthal_degree=2, cell_m=1.,
        k_max_per_m=2., k_panel_per_m=.2, k_order=4, radial_range_m=(2., 60.),
        radial_nodes=8, angular_backend="numpy", threshold_pe=0, cache_directory=tmp_path)
    source = lh.IsotropicFlash.monochromatic([0, 0, 0], 1e5, 450.)
    with pytest.warns(RuntimeWarning, match="period of the frequency grid"):
        aliased = lh.TransportKernel(medium, detector, config).transport(source)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        response = lh.TransportKernel(
            medium, detector, replace(config, readout_omega_per_ns=readout)).transport(source)
    relative = (response.spectrum_pe
                * np.exp(-1j * omega[:, None] * response.time_origin_ns[None, :])[:, :, None])
    for order in (1, 2):
        np.testing.assert_allclose(
            response.components_pe[:, :, order],
            readout_bins(omega, relative[:, :, order], edges, readout), rtol=1e-12, atol=1e-15)
        ratio = (aliased.components_pe[:, :, order].sum(1)
                 / response.components_pe[:, :, order].sum(1))
        assert np.all(ratio > 2.5)
