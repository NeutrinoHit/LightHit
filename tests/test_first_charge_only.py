"""Stationary q1 preserves directional physics and ignores source clocks."""
from dataclasses import replace
import numpy as np
import pytest
from lighthit.model import SpectralMedium,DetectorArray
from lighthit import SpectralLightElements,KernelConfig
from lighthit.transport import TransportKernel
from lighthit.prompt import PromptConfig
from lighthit.experimental.first_charge_only import first_scattering_charge_only
from lighthit.experimental.muon_integral import order1_integrals


def case():
    medium=SpectralMedium(np.array([350.,610.]),np.array([.05,.05]),np.array([.015,.015]),np.array([1.34,1.34]),np.array([1.37,1.37]),g=.9)
    detector=DetectorArray(np.array([[30.,0.,0.],[10.,0.,25.],[50.,0.,-20.]]),
        np.tile([0.,0.,-1.],(3,1)),1.,lambda x:1.+.2*x,lambda w:np.ones_like(w))
    kernel=TransportKernel(medium,detector,KernelConfig(omega_per_ns=np.array([0.]),wavelength_nodes=1,
        relative_time_edges_ns=np.arange(-60.,745.,5.),cache_policy='require'))
    source=SpectralLightElements(np.array([[0.,0.,-40.]]),np.array([[0.,0.,1.]]),np.array([20.]),
        np.array([20.]),np.array([-20./.95**2]),np.array([.95]),np.array([0.]),np.array([20/.28]),
        np.array([0]),np.array([0]),cone_model='spectral')
    return kernel,source


def test_stationary_directional_charge_matches_time_transport():
    kernel,source=case();config=PromptConfig(element_method='segment',screen=False)
    q,meta=first_scattering_charge_only(kernel,source,config=config)
    expected,bins,_=order1_integrals(kernel,source,time_origin_ns=np.zeros(3),config=config)
    np.testing.assert_allclose(q,expected,rtol=3e-3,atol=1e-16)
    assert meta['computed_bins'] is False
    assert np.all(q>0)


def test_charge_independent_of_emission_clock_and_time_window():
    kernel,source=case();q,_=first_scattering_charge_only(kernel,source)
    changed=replace(source,start_ns=source.start_ns+4321.,end_ns=source.end_ns+5555.)
    kernel.config=replace(kernel.config,relative_time_edges_ns=np.array([-1.,0.,1.]))
    other,_=first_scattering_charge_only(kernel,changed)
    np.testing.assert_array_equal(q,other)


def test_true_beta_threshold_no_artificial_light():
    kernel,source=case();dark=replace(source,beta=np.array([.7]),coefficient2=-source.coefficient0/.7**2)
    q,meta=first_scattering_charge_only(kernel,dark)
    np.testing.assert_array_equal(q,np.zeros(3));assert not meta['wavelengths']


def test_zero_scattering_has_exactly_zero_first_order():
    kernel,source=case()
    kernel.medium=replace(kernel.medium,scattering_per_m=np.zeros(2))
    for order in [0,6]:
        q,meta=first_scattering_charge_only(kernel,source,gauss_order=order)
        np.testing.assert_array_equal(q,np.zeros(3));assert not meta['wavelengths']
    q,b,_=order1_integrals(kernel,source,time_origin_ns=np.zeros(3))
    np.testing.assert_array_equal(q,np.zeros(3));np.testing.assert_array_equal(b,0.)


def test_uniform_segmentation_preserves_stationary_response():
    kernel,source=case();q,_=first_scattering_charge_only(kernel,source)
    parts=SpectralLightElements(np.array([[0.,0.,-40.],[0.,0.,-30.]]),np.tile([0.,0.,1.],(2,1)),np.array([10.,10.]),
        np.array([10.,10.]),np.array([-10./.95**2,-10./.95**2]),np.array([.95,.95]),np.array([0.,10/.28]),
        np.array([10/.28,20/.28]),np.array([0,1]),np.array([0,0]),cone_model='spectral')
    qp,_=first_scattering_charge_only(kernel,parts)
    np.testing.assert_allclose(q,qp,rtol=3e-3,atol=1e-16)


@pytest.mark.parametrize('theta,radius,g',[(1e-5,10.,.99),(.6,100.,.9),(2.8,500.,.9)])
def test_stationary_gauss_against_independent_scattering_angle_integral(theta,radius,g):
    from scipy.integrate import quad
    from lighthit._prompt_numba import _gauss_legendre,_batch_sigma,_stationary_path_gauss,_batch_add_pencil
    mut=.06;sine=np.sin(theta);cosine=np.cos(theta);apoly=np.array([1.,.3,.1])
    sigma=_batch_sigma(radius,sine,np.tan(theta/2),(1-g)/(1+g))
    x,w=_gauss_legendre(6);s=np.empty(2048);weights=np.empty(2048)
    n=_stationary_path_gauss(radius,sigma,mut,25.,1.5,x,w,s,weights)
    a=np.zeros(2048);b=np.zeros(2048)
    _batch_add_pencil(n,s,radius,sine,cosine,1/(radius*sine),sine,1.,0.,-.4,.8,apoly,g,a,b)
    candidate=np.sum(weights[:n]*np.exp(-mut*(s[:n]-radius))*a[:n])
    y0=np.tan(theta/2);upper=2*np.arctan(y0+25/(mut*radius*sine))
    def integrand(chi):
        c=np.cos(chi);v=np.sin(chi);arg=-.4*c+.8*v
        hg=(1-g*g)/(4*np.pi*(1+g*g-2*g*c)**1.5)
        return np.exp(-mut*radius*sine*(np.tan(chi/2)-y0))*hg*(1+.3*arg+.1*arg*arg)/(radius*sine)
    reference=quad(integrand,theta,upper,epsabs=1e-12,epsrel=1e-10)[0]
    # Ten-ppm path criterion, much tighter than the existing source/cache error.
    np.testing.assert_allclose(candidate,reference,rtol=1e-5,atol=1e-14)


def test_stationary_gauss_is_clock_independent_and_converges():
    kernel,source=case()
    q,_=first_scattering_charge_only(kernel,source,gauss_order=6,panel_width=1.5)
    refined,_=first_scattering_charge_only(kernel,source,gauss_order=10,panel_width=.5)
    np.testing.assert_allclose(q,refined,rtol=5e-7,atol=1e-16)
    changed=replace(source,start_ns=source.start_ns+4321.,end_ns=source.end_ns+5555.)
    shifted,_=first_scattering_charge_only(kernel,changed,gauss_order=6,panel_width=1.5)
    np.testing.assert_array_equal(q,shifted)
