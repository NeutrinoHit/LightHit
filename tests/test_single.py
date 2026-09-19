import numpy as np
import pytest
from dataclasses import replace
from scipy.integrate import quad
from lighthit import synthetic_medium
from lighthit.single import (single_scattering_rate, single_spectrum, single_bins,
                             single_quadrature_backend, hg_phase)


def test_hg_normalization():
    for g in [-0.7, 0, 0.7, 0.9]:
        value = 2*np.pi*quad(lambda mu: hg_phase(mu, g), -1, 1, epsabs=1e-11)[0]
        assert abs(value-1) < 1e-10


def test_isotropic_first_analytic_time():
    m = replace(synthetic_medium(), g=0.0)
    r = 20.0
    t = r/m.speed_m_per_ns + np.geomspace(.01, 500, 35)
    path = t*m.speed_m_per_ns
    exact = m.scattering_per_m*np.exp(-m.extinction_per_m*path)/(4*np.pi*r*t)*np.log((path+r)/(path-r))
    np.testing.assert_allclose(single_scattering_rate(t, r, None, m), exact, rtol=1e-12)


@pytest.mark.parametrize("mu", [-1.0, -0.3, 0.5, 0.9])
def test_directed_charge_against_independent_first_flight_integral(mu):
    m = synthetic_medium(); r = 20.0
    def flight(a):
        b = np.sqrt(r*r+a*a-2*r*mu*a)
        c = (r*mu-a)/b
        return m.scattering_per_m*hg_phase(c,m.g)*np.exp(-m.extinction_per_m*(a+b))/(b*b)
    reference = quad(flight,0,np.inf,epsabs=1e-14,epsrel=1e-10)[0]
    result, _ = single_spectrum(np.array([0.0]), r, mu, m)
    np.testing.assert_allclose(result[0], reference, rtol=1e-9, atol=1e-14)


def test_single_raw_bins_are_causal_and_integrate_charge():
    m=synthetic_medium(); r=20.; front=r/m.speed_m_per_ns
    edges=np.r_[0,front-1,front,front+1,front+20,front+200,front+2000]
    bins=single_bins(edges,r,.5,m)
    assert bins[0] == bins[1] == 0
    assert np.all(bins>=0)
    spectrum,_=single_spectrum([0.0],r,.5,m)
    np.testing.assert_allclose(bins.sum(),spectrum[0].real,rtol=1e-8)


def test_zero_scattering():
    m=replace(synthetic_medium(),scattering_per_m=0)
    assert single_scattering_rate(100,20,.5,m)==0
    np.testing.assert_array_equal(single_spectrum([0,.1],20,.5,m)[0],0)


def test_isotropic_spectrum_is_finite_at_the_integrable_front():
    m = synthetic_medium()
    spectrum, error = single_spectrum(np.linspace(0.0, 0.4, 9), 20.0, None, m)
    assert np.isfinite(spectrum).all()
    assert np.isfinite(error)
    assert spectrum[0].real > 0


@pytest.mark.parametrize("function,args", [
    (single_spectrum, ([0.0], 20.0, None, synthetic_medium())),
    (single_bins, ([0.0, 100.0], 20.0, None, synthetic_medium())),
])
def test_single_backend_name_is_validated(function, args):
    with pytest.raises(ValueError, match="single_backend"):
        function(*args, backend="typo")


def test_auto_backend_resolves_to_an_available_implementation():
    assert single_quadrature_backend() in ("numpy", "numba")
