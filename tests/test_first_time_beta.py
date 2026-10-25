"""Beta interpolation preserves FT yield, source clocks and exact fallbacks."""
from types import SimpleNamespace
from dataclasses import replace
import numpy as np
import lighthit as lh
from lighthit.muon.first_time_beta import FirstTimeBetaAtlas,supported_time_part,C_M_PER_NS


class UniformResponse:
    radii_m=np.array([3.,10.]);x_edges=np.array([-8.5,8.5])
    def __init__(self,beta):self.beta=beta
    def query(self,rho,orientation,lo,hi,edges,*,coefficient0_per_m):
        # Exact one-band FT yield times a normalized positive time measure.
        charge=coefficient0_per_m*(hi-lo)*(1-.5/self.beta**2)
        cdf=np.clip(np.asarray(edges)/10,0.,1.)
        return charge,charge*np.diff(cdf)


def atlas():
    data=FirstTimeBetaAtlas.__new__(FirstTimeBetaAtlas)
    data.beta=np.array([.999,1.]);data.stores=[UniformResponse(b) for b in data.beta]
    data.radii_m=data.stores[0].radii_m
    return data


def test_beta_parameter_preserves_physical_spectral_yield_and_time_mass():
    beta=.9996;edges=np.array([-1.,0.,3.,7.,10.,15.])
    q,b=atlas().query(4.,np.array([0.,0.,1.]),-2.,3.,edges,beta,coefficient0_per_m=2.)
    expected=10*(1-.5/beta**2)
    np.testing.assert_allclose(q,expected,rtol=2e-15)
    np.testing.assert_allclose(b,expected*np.array([0.,.3,.4,.3,0.]),rtol=2e-15)


def test_beta_time_uses_actual_source_origin_and_keeps_residual_clock_guard():
    beta=.9995;length=4.;clock=1/(beta*C_M_PER_NS)
    source=lh.SpectralLightElements(np.zeros((1,3)),np.array([[0.,0.,1.]]),np.array([length]),
        np.array([length]),np.array([-length/beta**2]),np.array([beta]),np.array([2.]),
        np.array([2.+length*clock]),np.array([0]),np.array([0]),cone_model='spectral')
    edges=np.arange(-10.,31.,2.);position=np.array([3.2,0.,2.]);normal=np.array([0.,0.,1.])
    q,b,left,count=supported_time_part(SimpleNamespace(),source,position,normal,1.,edges,atlas())
    expected_q=length*(1-.5/beta**2);shift=2.+2*clock-1.
    np.testing.assert_allclose(q,expected_q,rtol=2e-15)
    np.testing.assert_allclose(b,expected_q*np.diff(np.clip((edges-shift)/10,0,1)),rtol=2e-14,atol=1e-15)
    assert left==[] and count==1
    changed=replace(source,end_ns=source.end_ns+.1)
    q,b,left,count=supported_time_part(SimpleNamespace(),changed,position,normal,1.,edges,atlas())
    assert q==0 and not b.any() and count==0 and left==[(0,0.,length)]


def test_radial_coordinate_aligns_glass_front_and_preserves_stored_nodes():
    from lighthit.muon.track_first_primitive import TrackFirstAtlas,orientation_nodes
    radii=np.array([3.,5.]);origin=1.;front=3.;width=2.
    time=np.unique(np.r_[-5.,0.,front-origin/radii,front+width-origin/radii,10.])
    _,powers,_=orientation_nodes();ql=np.zeros((2,2,10));qr=np.zeros_like(ql)
    cl=np.zeros((2,2,len(time),10));cr=np.zeros_like(cl)
    for j,r in enumerate(radii):
        ql[j,1,0]=1.;qr[j,0,0]=1.
        cdf=np.clip((time+origin/r-front)/width,0.,1.)
        cl[j,1,:,0]=cdf;cr[j,0,:,0]=cdf
    data=TrackFirstAtlas(radii,np.array([-1.,1.]),time,powers,ql,qr,cl,cr,{})
    look=np.array([0.,0.,1.]);edges=np.linspace(0.,25.,101);rho=4.
    _,actual=data.query(rho,look,-rho,rho,edges,radial_time_origin_ns=origin)
    expected=np.diff(np.clip(((edges+origin)/rho-front)/width,0.,1.))
    np.testing.assert_allclose(actual,expected,rtol=1e-13,atol=2e-16)
    for r in radii:
        before=data.query(r,look,-r,r,edges)[1]
        after=data.query(r,look,-r,r,edges,radial_time_origin_ns=origin)[1]
        np.testing.assert_array_equal(before,after)
