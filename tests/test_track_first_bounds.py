"""Independent convolution, photon normalization and positive omission controls."""
from types import SimpleNamespace
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import k0,i0e,k0e
from lighthit.experimental.track_first_bounds import (
    yukawa_convolution_upper,yukawa_convolution_elementary_upper,
    infinite_track_first_scattering_bounds,combined_first_scattering_segment_bounds,
    hg_angle_integral_over_sine_upper)
from lighthit.sources import SpectralLightElements


@pytest.mark.parametrize('g',[0.,.5,.9,.99])
def test_analytic_hg_angular_bound_against_independent_quadrature(g):
    norm=(1-g*g)/(4*np.pi)
    values=[]
    for theta in np.r_[np.geomspace(1e-6,.1,6),np.linspace(.2,3.,8),np.pi-1e-7]:
        exact=quad(lambda chi:norm/(1+g*g-2*g*np.cos(chi))**1.5,
            theta,np.pi,epsabs=1e-12,epsrel=2e-10)[0]/np.sin(theta)
        upper=float(hg_angle_integral_over_sine_upper(theta,g))
        assert upper>=exact*(1-1e-9)
        values.append(exact)
    # Monotonicity allows minimum cone separation to bound every source ray.
    assert np.all(np.diff(values)<=1e-8*np.max(values))
    assert np.isinf(hg_angle_integral_over_sine_upper(0.,g))
    np.testing.assert_allclose(hg_angle_integral_over_sine_upper(np.pi,g),
        norm/(1+g)**3,rtol=5e-14)


def test_new_finite_ray_bound_exceeds_independent_first_flight_integral():
    from lighthit.experimental.track_first_primitive import first_scattering_segment_bounds
    from dataclasses import replace
    kernel,source=case()
    source=replace(source,start_m=np.zeros((1,3)),length_m=np.array([.2]),
        coefficient0=np.array([.2]),coefficient2=np.array([-.2]))
    phase=1.34;cone=1/phase;sincone=np.sqrt(1-cone**2)
    px,pw=np.polynomial.legendre.leggauss(24);phi=np.pi*(px+1);pw=pw/2
    ax,aw=np.polynomial.legendre.leggauss(8)
    directions=np.column_stack((sincone*np.cos(phi),sincone*np.sin(phi),np.full(len(phi),cone)))
    for receiver in [[40.,0.,20.],[180.,0.,-50.],[20.,0.,100.]]:
        bound,_=first_scattering_segment_bounds(kernel,source,receiver)
        value=0.
        for a,w in zip((ax+1)*.1,aw*.1):
            R=np.array(receiver)-np.array([0.,0.,a])
            for direction,phiw in zip(directions,pw):
                def integrand(ell):
                    last=R-ell*direction;distance=np.linalg.norm(last);cosine=last@direction/distance
                    hg=(1-.9**2)/(4*np.pi*(1+.9**2-2*.9*cosine)**1.5)
                    return .015*np.exp(-.065*(ell+distance))*hg/distance**2
                value+=w*phiw*quad(integrand,0,np.inf,epsabs=1e-20,epsrel=1e-8)[0]
        value*=(1-cone**2)/480**2
        assert 0<value<=bound[0]


@pytest.mark.parametrize('rho,a,b',[(.1,.12,.06),(3.,.12,.06),(30.,.12,.06),(300.,.12,.06),(100.,.065,.065)])
def test_bounds_exceed_independent_bessel_integral_and_elliptic_coordinate_identity(rho,a,b):
    direct=4*quad(lambda t:k0(rho*np.hypot(a*np.sin(t),b*np.cos(t))),0,np.pi/2,epsabs=1e-30,epsrel=1e-11)[0]
    c=rho*(a+b)/2;d=rho*(a-b)/2
    closed=2*np.pi*i0e(d)*k0e(c)*np.exp(-rho*b)
    np.testing.assert_allclose(direct,closed,rtol=2e-11)
    assert direct<=yukawa_convolution_upper(rho,a,b,nodes=32)
    assert direct<=yukawa_convolution_elementary_upper(rho,a,b)
    # The equality a=b reduces to 2*pi*K0(b*rho).
    if a==b:np.testing.assert_allclose(direct,2*np.pi*k0(b*rho),rtol=1e-12)


def case():
    def sample(w):
        return {key:np.full_like(w,value) for key,value in dict(absorption_per_m=.05,
            scattering_per_m=.015,phase_index=1.34,group_index=1.37).items()}
    kernel=SimpleNamespace(medium=SimpleNamespace(g=.9,sample=sample),
        wavelength=SimpleNamespace(wavelength_nm=np.array([480.]),weight_nm=np.array([1.])),
        detector=SimpleNamespace(spectral_weight=lambda w:1.,effective_area_m2=np.array([1.])),
        acceptance=lambda:{'alpha':np.array([4*np.pi])})
    source=SpectralLightElements(np.array([[0.,0.,-70.]]),np.array([[0.,0.,1.]]),
        np.array([60.]),np.array([60.]),np.array([-60.]),np.array([1.]),
        np.array([0.]),np.array([60/.299792458]),np.array([0]),np.array([0]),cone_model='spectral')
    return kernel,source


def test_cone_crossing_is_finite_and_exceeds_independent_direct_segment_integral():
    from lighthit.experimental.track_first_primitive import first_scattering_segment_bounds
    kernel,source=case();receiver=np.array([30.,0.,0.])
    old,_=first_scattering_segment_bounds(kernel,source,receiver)
    assert np.isinf(old[0])
    result,meta=combined_first_scattering_segment_bounds(kernel,source,receiver)
    assert np.isfinite(result[0]) and result[0]>0 and meta['line_tighter_pieces']==1
    # Independent quadrature of emitting position, cone azimuth and first flight.
    # Gauss azimuth excludes the measure-zero ray exactly through the point OM.
    x,w=np.polynomial.legendre.leggauss(24);positions=-70+(x+1)*30;sw=w*30
    az,aw=np.polynomial.legendre.leggauss(96);phi=np.pi*(az+1);aw=aw/2
    cone=1/1.34;directions=np.column_stack((np.sqrt(1-cone**2)*np.cos(phi),np.sqrt(1-cone**2)*np.sin(phi),np.full(len(phi),cone)))
    y,v=np.polynomial.legendre.leggauss(128);t=(y+1)/2;ell=30*t/(1-t);lw=v/2*30/(1-t)**2
    response=0.
    for z,weight in zip(positions,sw):
        R=np.array([30.,0.,-z]);last=R-directions[:,None,:]*ell[None,:,None]
        distance=np.linalg.norm(last,axis=2)
        cosine=np.sum(last*directions[:,None,:],axis=2)/distance
        phase=(1-.9**2)/(4*np.pi*(1+.9**2-2*.9*cosine)**1.5)
        values=.015*np.exp(-.065*(ell[None,:]+distance))*phase/distance**2
        response+=weight*(aw@(values@lw))
    response*=(1-1/1.34**2)/480**2
    assert 0<response<result[0]
    assert meta['continuum_spectral_error_included'] is False


def test_uniform_track_extension_normalization_and_zero_area():
    from dataclasses import replace
    kernel,source=case();receiver=np.array([300.,0.,0.])
    base,_=infinite_track_first_scattering_bounds(kernel,source,receiver)
    double=replace(source,length_m=source.length_m*2,coefficient0=source.coefficient0*2,coefficient2=source.coefficient2*2)
    same,_=infinite_track_first_scattering_bounds(kernel,double,receiver)
    np.testing.assert_allclose(base,same,rtol=0,atol=0)  # Same infinite yield/m.
    scaled,_=infinite_track_first_scattering_bounds(kernel,source,receiver,effective_area_m2=2.)
    np.testing.assert_allclose(scaled,base*2,rtol=1e-14)
    zero,_=infinite_track_first_scattering_bounds(kernel,source,[0.,0.,0.],effective_area_m2=0.)
    np.testing.assert_array_equal(zero,[0.])


def test_axis_and_bad_parameters_are_explicit():
    assert np.isinf(yukawa_convolution_elementary_upper(0.,.1,.05))
    for fn in (yukawa_convolution_elementary_upper,yukawa_convolution_upper):
        for args in [(-1.,.1,.05),(1.,.03,.05),(1.,.1,0.)]:
            with pytest.raises(ValueError):fn(*args)


def test_zero_scattering_transparent_medium_is_exactly_dark():
    kernel,source=case();sample=kernel.medium.sample
    def transparent(w):
        values=sample(w);values['scattering_per_m'][:]=0.;values['absorption_per_m'][:]=0.
        return values
    kernel.medium.sample=transparent
    line,_=infinite_track_first_scattering_bounds(kernel,source,[30.,0.,0.])
    combined,_=combined_first_scattering_segment_bounds(kernel,source,[30.,0.,0.])
    np.testing.assert_array_equal(line,[0.])
    np.testing.assert_array_equal(combined,[0.])
