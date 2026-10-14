"""Public continuous-spectrum wrapper and prompt integration controls."""
from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import quad

pytest.importorskip('numba')
import lighthit as lh
from lighthit.ballistic_interval import ballistic_interval, BallisticIntervalConfig, prepare_optics


def setup():
    medium = lh.SpectralMedium([350., 450., 610.], [.07,.04,.09], [0.,0.,0.],
                              [1.38,1.35,1.32], [1.43,1.39,1.36], g=.9)
    def efficiency(w):
        return np.interp(w, [350.,410.,415.,610.], [.1,.2,.35,.05])
    efficiency.wavelength_knots_nm = np.array([350.,410.,415.,610.])
    detector = lh.DetectorArray([[10.,0.,25.]], [[0.,0.,-1.]], .05,
                               lambda x: np.ones_like(x), efficiency)
    kernel = lh.TransportKernel(medium, detector, lh.KernelConfig(
        omega_per_ns=np.array([0.]), wavelength_nodes=2, threshold_pe=0., acceptance_degree=0))
    source = lh.SpectralLightElements(np.array([[0.,0.,0.]]), np.array([[0.,0.,1.]]),
        np.array([50.]), np.array([1e8]), np.array([-1e8]), np.array([1.]),
        np.array([0.]), np.array([50./.299792458]), np.array([0]), np.array([0]), cone_model='spectral')
    return kernel, source


def test_real_callable_knots_and_independent_charge_quad():
    kernel, source = setup()
    result = ballistic_interval(kernel, source, time_origin_ns=[0.], relative_edges_ns=[0.,1000.])
    def value(w):
        band = kernel.medium.sample(w)
        n = band['phase_index']; sine = np.sqrt(1-1/n**2)
        photons = 1e8*(1-1/n**2)/w**2
        return (.05*kernel.detector.spectral_weight(w)*photons/50
                *np.exp(-band['absorption_per_m']*10/sine)/(2*np.pi*10*sine))
    points = [350.,410.,415.,450.,610.]
    expected = sum(quad(value,a,b,epsabs=1e-13,epsrel=1e-11)[0] for a,b in zip(points[:-1],points[1:]))
    np.testing.assert_allclose(result.charge_pe, [expected], rtol=2e-9)
    np.testing.assert_allclose(result.bins_pe.sum(1), result.charge_pe, rtol=2e-14)
    assert result.metadata['optical_panels']['supplied_response_knots']
    assert result.metadata['quadrature'][-1]['failed'] == 0


def test_prompt_opt_in_uses_continuous_charge_and_correct_bins():
    kernel, source = setup()
    origins = np.array([0.])
    config = lh.PromptConfig(ballistic_method='interval', screen=False)
    direct = ballistic_interval(kernel, source, time_origin_ns=origins,
                                relative_edges_ns=kernel.config.relative_time_edges_ns)
    response = kernel.transport_prompt(source, config, compute_bins_for_all=True, time_origin_ns=origins)
    np.testing.assert_allclose(response.charge_orders_pe[:,0], direct.charge_pe, rtol=1e-13)
    np.testing.assert_allclose(response.bins_orders_pe[:,:,0], direct.bins_pe, rtol=1e-13)
    np.testing.assert_array_equal(response.charge_orders_pe[:,1], 0.)
    assert response.metadata['ballistic_interval']['method'] == 'continuous-spectrum-intervals'


def test_invalid_requests_do_not_silently_change_model():
    kernel, source = setup()
    with pytest.raises(ValueError, match='spectral'):
        ballistic_interval(kernel, replace(source, cone_model='frozen'))
    with pytest.raises(ValueError, match='provided together'):
        ballistic_interval(kernel, source, time_origin_ns=[0.])
    with pytest.raises(ValueError, match='physical Cherenkov'):
        ballistic_interval(kernel, replace(source, coefficient2=np.array([-2e8])))
    with pytest.raises(ValueError):
        BallisticIntervalConfig(order=4, max_order=4)
    with pytest.raises(ValueError):
        lh.PromptConfig(ballistic_method='unknown')


def test_efficiency_panel_refinement_is_checked():
    kernel, _ = setup()
    detector = replace(kernel.detector, spectral_efficiency=lambda w: np.exp((np.asarray(w)-610)/25))
    other = lh.TransportKernel(kernel.medium, detector, kernel.config)
    panels = prepare_optics(other)
    assert panels['metadata']['efficiency_checks'][0]['failed'] > 0
    assert panels['metadata']['efficiency_checks'][-1]['failed'] == 0
    assert not panels['metadata']['supplied_response_knots']


def test_prompt_keeps_emission_between_endpoint_and_first_global_node():
    kernel, source = setup()
    detector = replace(kernel.detector, positions_m=np.array([[.11,0.,25.]]))
    kernel = lh.TransportKernel(kernel.medium, detector, kernel.config)
    source = replace(source, beta=np.array([.725]), coefficient2=-source.coefficient0/.725**2)
    assert np.all(source.beta[0]*kernel.medium.sample(kernel.wavelength.wavelength_nm)['phase_index'] < 1)
    response = kernel.transport_prompt(source, lh.PromptConfig(ballistic_method='interval', screen=False),
                                       compute_bins_for_all=True, time_origin_ns=[0.])
    assert response.charge_orders_pe[0,0] > 0
    np.testing.assert_array_equal(response.charge_orders_pe[:,1], 0.)


def test_primary_component_excludes_other_uids_and_respects_spectral_range():
    from lighthit.experimental.primary_prompt import transport_primary_prompt
    from lighthit.experimental.primary_polyline import primary_spectral_quadrature
    kernel, primary = setup()
    kernel = lh.TransportKernel(kernel.medium, kernel.detector,
                                 replace(kernel.config, wavelength_range_nm=(400.,500.)))
    arrays = {name: np.repeat(getattr(primary,name),2,axis=0) for name in
              ('start_m','direction','length_m','coefficient0','coefficient2','beta','start_ns','end_ns','row_index','uid')}
    arrays['uid'] = np.array([0,9]); arrays['row_index'] = np.arange(2)
    mixed = lh.SpectralLightElements(**arrays,cone_model='spectral')
    result = transport_primary_prompt(kernel,mixed,time_origin_ns=[0.])
    direct = ballistic_interval(kernel,primary,time_origin_ns=[0.],relative_edges_ns=kernel.config.relative_time_edges_ns)
    np.testing.assert_allclose(result.charge_orders_pe[:,0],direct.charge_pe,rtol=1e-13)
    np.testing.assert_array_equal(result.charge_orders_pe[:,1],0.)
    assert not result.metadata['secondaries_computed'] and not result.metadata['ge2_computed']
    rule = primary_spectral_quadrature(kernel.medium,kernel.detector,wavelength_range_nm=(400.,500.))
    np.testing.assert_allclose(rule.weight_nm.sum(),100.,rtol=1e-14)
    assert np.all((rule.wavelength_nm>400)&(rule.wavelength_nm<500))


def test_pair_list_passes_equal_full_receiver_element_loop_bitwise():
    # Several receivers and elements, some failing the geometry precheck.
    kernel, source = setup()
    rng = np.random.default_rng(3)
    n = 12
    direction = rng.normal(size=(n, 3)); direction /= np.linalg.norm(direction, axis=1)[:, None]
    beta = rng.uniform(.75, 1., n)
    c0 = rng.uniform(1e7, 1e8, n)
    length = rng.uniform(.5, 20., n)
    start_ns = rng.uniform(0., 50., n)
    many = lh.SpectralLightElements(rng.uniform(-20., 20., (n, 3)), direction, length, c0, -c0/beta**2, beta,
        start_ns, start_ns+length/(beta*.299792458), np.arange(n), np.arange(n), cone_model='spectral')
    positions = np.array([[10., 0., 25.], [-30., 5., 0.], [0., 40., -10.], [80., -20., 30.]])
    kernel.detector = replace(kernel.detector, positions_m=positions,
        orientations=np.tile([0., 0., -1.], (4, 1)), effective_area_m2=np.full(4, .05))
    origins = np.zeros(4); edges = np.linspace(-20., 400., 43)
    results = [ballistic_interval(kernel, many, config=BallisticIntervalConfig(pair_list=flag),
                                  time_origin_ns=origins, relative_edges_ns=edges) for flag in (False, True)]
    np.testing.assert_array_equal(results[0].charge_pe, results[1].charge_pe)
    np.testing.assert_array_equal(results[0].bins_pe, results[1].bins_pe)
    assert np.any(results[0].charge_pe > 0)
