"""Exact tabulated OM response avoids a polynomial-subdivision error budget."""
import numpy as np
import pytest
from dataclasses import replace
from scipy.integrate import quad

pytest.importorskip('numba')
import lighthit as lh
from lighthit.bgvd import OMResponse
from lighthit.ballistic_interval import prepare_optics, ballistic_interval, BallisticIntervalConfig


def make_kernel():
    response = OMResponse([1.], [350., 470., 610.], [.1, .4, .2],
                           [350., 450., 610.], [.01, .6, .9])
    angular, spectral = response.functions()
    medium = lh.SpectralMedium([350., 610.], [.01, .01], [.02, .02],
                                [1.35, 1.35], [1.37, 1.37])
    detector = lh.DetectorArray([[3., 0., 10.]], [[0., 0., 1.]], .1, angular, spectral)
    return medium, lh.TransportKernel(medium, detector, lh.KernelConfig(
        omega_per_ns=np.array([0.]), threshold_pe=0., acceptance_degree=0))


def test_exact_panels_evaluate_callable_and_preserve_default_cubic_contract():
    _, kernel = make_kernel()
    tables = prepare_optics(kernel, exact_response=True)
    cubic = prepare_optics(kernel)
    assert tables['efficiency_coefficients'].shape[1] == 6
    assert cubic['efficiency_coefficients'].shape[1] == 4
    assert len(tables['panel_edges']) < len(cubic['panel_edges'])
    q = np.array([.02, .23, .51, .97])
    for (lo, hi), c in zip(zip(tables['panel_edges'][:-1], tables['panel_edges'][1:]),
                            tables['efficiency_coefficients']):
        predicted = np.polynomial.polynomial.polyval(q, c[:4])*np.exp(c[4]+c[5]*q)
        np.testing.assert_allclose(predicted, kernel.detector.spectral_weight(lo+(hi-lo)*q), rtol=8e-15)


def test_exact_response_direct_charge_matches_independent_spectral_integral():
    _, kernel = make_kernel()
    length, coefficient, n, rho = 20., 1e8, 1.35, 3.
    source = lh.SpectralLightElements([[0., 0., 0.]], [[0., 0., 1.]], [length],
        [coefficient*length], [-coefficient*length], [1.], [0.], [length/.299792458],
        [0], [0], cone_model='spectral')
    actual = ballistic_interval(kernel, source, config=BallisticIntervalConfig(receiver_mode='point'))
    spectral = kernel.detector.spectral_weight
    integral = quad(lambda w: float(spectral(w))/w**2, 350., 610.,
                    points=[450., 470.], epsabs=1e-18, epsrel=1e-12)[0]
    sine = np.sqrt(1.-1./n**2)
    expected = coefficient*sine/(2*np.pi*rho)*np.exp(-.03*rho/sine)*.1*integral
    np.testing.assert_allclose(actual.charge_pe[0], expected, rtol=2e-12)


def test_inconsistent_advertised_response_is_rejected():
    _, kernel = make_kernel()
    original = kernel.detector.spectral_efficiency
    def incorrect(w):
        return 2*original(w)
    incorrect.linear_exponential_tables = original.linear_exponential_tables
    kernel = lh.TransportKernel(kernel.medium, replace(kernel.detector, spectral_efficiency=incorrect),
                                kernel.config)
    with pytest.raises(ValueError, match='tables differ'):
        prepare_optics(kernel, exact_response=True)
