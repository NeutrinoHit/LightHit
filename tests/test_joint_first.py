"""Independent physical-time checks for a reusable pencil Q1 response."""
import numpy as np
import pytest
from scipy.integrate import quad
from lighthit.experimental.joint_first import PencilFirstCDF, contract_joint_pencils, direct_photon_histogram


def response(theta=.6, radius=40., step=.015, g=.9, mus=.015):
    return PencilFirstCDF.build([radius*np.sin(theta), 0., radius*np.cos(theta)],
        [0., 0., 1.], [.8, 0., -.6], [1., .3, .1], scattering_per_m=mus,
        absorption_per_m=.05, g=g, speed_m_per_ns=.22, path_step=step)


def independent_integral(theta, radius, x, g=.9):
    upper=2*np.arctan(np.tan(theta/2)+x/(radius*np.sin(theta)))
    def density(chi):
        extra=radius*np.sin(theta)*(np.tan(chi/2)-np.tan(theta/2))
        arg=-.6*np.cos(chi)+.8*np.sin(chi)
        hg=(1-g*g)/(4*np.pi*(1+g*g-2*g*np.cos(chi))**1.5)
        return .015*hg*(1+.3*arg+.1*arg*arg)*np.exp(-.065*(radius+extra))/(radius*np.sin(theta))
    return quad(density, theta, upper, epsabs=1e-18, epsrel=1e-11)[0]


@pytest.mark.parametrize("theta,radius,g",[(1e-4,10.,.99),(.6,40.,.9),(2.8,150.,.9)])
def test_cdf_against_independent_scatter_angle_quadrature(theta,radius,g):
    result=response(theta,radius,g=g)
    x=np.array([.0001,.05,1.,10.,100.,25/.065])
    expected=np.array([independent_integral(theta,radius,z,g) for z in x])
    np.testing.assert_allclose(result.cdf_path(x),expected,rtol=2e-6,atol=expected[-1]*1e-9)
    np.testing.assert_allclose(result.charge_per_m2,expected[-1],rtol=2e-6)


def test_charge_clock_invariance_and_causal_window():
    result=response(); edges=np.arange(-5.,806.,5.)
    q,b=result.contract([0.,25.],[3.,4.],edges)
    late,empty=result.contract([1e6,1e6+25.],[3.,4.],edges)
    assert q==late
    np.testing.assert_array_equal(empty,0.)
    np.testing.assert_array_equal(b[edges[1:]<=40/.22],0.)
    shifted_q,shifted_b=result.contract([1e6,1e6+25.],[3.,4.],edges,time_origin_ns=1e6)
    assert shifted_q==q
    np.testing.assert_allclose(shifted_b,b,rtol=2e-11,atol=1e-20)


def test_temporal_histogram_is_linear_and_converges():
    coarse,fine=response(step=.015),response(step=.0075)
    times=np.linspace(-40.,40.,31);weights=np.exp(-times**2/100.)
    edges=np.arange(-5.,806.,5.)+.37
    q,b=coarse.contract(times,weights,edges)
    fq,fb=fine.contract(times,weights,edges)
    np.testing.assert_allclose(q,fq,rtol=3e-7)
    assert abs(b-fb).sum()/abs(fb).sum()<2e-6
    individual=sum((coarse.contract([t],[w],edges)[1] for t,w in zip(times,weights)))
    np.testing.assert_allclose(b,individual,rtol=1e-14,atol=1e-22)


def test_zero_scattering_and_input_guards():
    result=response(mus=0.)
    q,b=result.contract([0.],[1.],[-5.,500.,1000.])
    assert q==0.;np.testing.assert_array_equal(b,0.)
    with pytest.raises(ValueError,match="collinear"):
        PencilFirstCDF.build([0.,0.,40.],[0.,0.,1.],[0.,0.,-1.],[1.],
            scattering_per_m=.015,absorption_per_m=.05,g=.9,speed_m_per_ns=.22)
    with pytest.raises(ValueError):result.contract([0.],[1.],[0.,0.])


def test_relative_geometry_rotation_invariance():
    base=response()
    angle=.7;rot=np.array([[np.cos(angle),-np.sin(angle),0.],[np.sin(angle),np.cos(angle),0.],[0.,0.,1.]])
    other=PencilFirstCDF.build(rot@np.array([40*np.sin(.6),0.,40*np.cos(.6)]),
        rot@np.array([0.,0.,1.]),rot@np.array([.8,0.,-.6]),[1.,.3,.1],
        scattering_per_m=.015,absorption_per_m=.05,g=.9,speed_m_per_ns=.22)
    np.testing.assert_allclose(base.cdf_path([.1,1.,20.]),other.cdf_path([.1,1.,20.]),rtol=1e-13)


def test_joint_group_contract_preserves_all_clocks_and_weights():
    responses=[response(theta=.4),response(theta=1.1)]
    ids=np.array([1,0,1,0]);t=np.array([0.,3.,12.,1e6]);w=np.array([1.,2.,3.,4.])
    edges=np.arange(-5.,806.,5.)
    q,b=contract_joint_pencils(responses,ids,t,w,edges)
    expected=[responses[key].contract([clock],[weight],edges) for key,clock,weight in zip(ids,t,w)]
    np.testing.assert_allclose(q,sum(value[0] for value in expected),rtol=1e-15)
    np.testing.assert_allclose(b,sum((value[1] for value in expected)),rtol=1e-15,atol=1e-22)


def test_exact_backward_limit_against_distance_integral():
    result=PencilFirstCDF.build([0.,0.,-40.],[0.,0.,1.],[.8,0.,-.6],[1.,.3,.1],
        scattering_per_m=.015,absorption_per_m=.05,g=.9,speed_m_per_ns=.22)
    hg=(1-.9**2)/(4*np.pi*(1+.9**2+2*.9)**1.5)
    acc=1+.3*.6+.1*.6**2
    expected=quad(lambda a:.015*hg*acc*np.exp(-.065*(40+2*a))/(40+a)**2,
                  0.,25/(2*.065),epsabs=1e-20,epsrel=1e-11)[0]
    np.testing.assert_allclose(result.charge_per_m2,expected,rtol=2e-7)


def test_positive_histogram_direct_interface_has_no_double_spectral_yield():
    from types import SimpleNamespace
    from lighthit.model import SpectralMedium,DetectorArray
    from lighthit.transport import TransportKernel
    from lighthit import KernelConfig
    from lighthit.prompt import acceptance_polynomial,acceptance_monomials,C_M_PER_NS
    medium=SpectralMedium(np.array([350.,610.]),np.full(2,.05),np.full(2,.015),
        np.full(2,1.34),np.full(2,1.37),g=.9)
    look=np.array([.8,0.,-.6]);delta=np.array([40*np.sin(.6),0.,40*np.cos(.6)])
    detector=DetectorArray(delta[None,:],-look[None,:],1.,lambda x:1+.2*x,lambda w:np.ones_like(w))
    kernel=TransportKernel(medium,detector,KernelConfig(omega_per_ns=np.array([0.]),cache_policy='require'))
    hist=SimpleNamespace(wavelength_nm=np.array([480.]),geometry_id=np.array([0,0]),
        position=np.zeros((1,3)),direction=np.array([[0.,0.,1.]]),
        emission_times_ns=np.array([0.,25.]),photon_weights=np.array([[3.,4.]]))
    edges=np.arange(-5.,806.,5.)
    q,b=direct_photon_histogram(kernel,hist,np.array([0]),edges,np.zeros(1),wavelength_weight_nm=2.)
    apoly=acceptance_polynomial(*acceptance_monomials(kernel.acceptance()['alpha']))
    sample=medium.sample([480.])
    cdf=PencilFirstCDF.build(delta,[0.,0.,1.],look,apoly,
        scattering_per_m=sample['scattering_per_m'][0],absorption_per_m=sample['absorption_per_m'][0],
        g=.9,speed_m_per_ns=C_M_PER_NS/sample['group_index'][0])
    cq,cb=cdf.contract(hist.emission_times_ns,hist.photon_weights[0]*2.,edges)
    np.testing.assert_allclose(q[0],cq,rtol=1e-6)
    assert abs(b[0]-cb).sum()/abs(cb).sum()<1e-6
