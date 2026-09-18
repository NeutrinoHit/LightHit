"""Regression tests for reviewed public-cache defects (3ba25b7)."""
from dataclasses import replace
import numpy as np
import pytest
import lighthit.cache as module
from lighthit import Medium, SolverSettings
from lighthit.cache import ResponseCache, CacheGrid, BandedResponseCache

M=Medium(.04,.05,.7,1.35)
S=SolverSettings(12,32,2.0,.1,6)

@pytest.fixture(scope='module')
def cache():
    return ResponseCache.build(M,S,CacheGrid.geometric(10,20,6,[0,.03,.08]))

def test_streamed_build_agrees_and_cleans_scratch(cache,tmp_path):
    streamed=ResponseCache.build(M,S,cache.grid,radius_chunk=1,scratch_dir=tmp_path)
    np.testing.assert_allclose(streamed.moments,cache.moments,rtol=1e-14,atol=1e-20)
    assert streamed.timings_s['temporary_disk_bytes']>0
    assert streamed.timings_s['bessel_block_bytes']==len(S.quadrature()[0])*33*16
    assert list(tmp_path.iterdir())==[]

def test_streamed_exception_removes_files(cache,tmp_path):
    def stop(*args):raise RuntimeError('stop')
    with pytest.raises(RuntimeError,match='stop'):
        ResponseCache.build(M,S,cache.grid,radius_chunk=1,scratch_dir=tmp_path,progress=stop)
    assert list(tmp_path.iterdir())==[]

def test_directional_exact_first_is_rejected(cache):
    with pytest.raises(ValueError,match='only for isotropic'):
        cache.acceptance_spectrum([15.],[-1.],[2*np.pi,2*np.pi/3],exact_first_order=True)

def test_exact_first_rejects_inconsistent_callable(cache):
    with pytest.raises(ValueError,match='constant'):
        cache.acceptance_spectrum([15.],[0.],[4*np.pi],exact_first_order=True,
                                  acceptance=lambda x:(1+x)/2)

def test_constant_half_acceptance(cache):
    one=cache.acceptance_charge([15.],[0.],[4*np.pi],exact_first_order=True)
    half=cache.acceptance_charge([15.],[0.],[2*np.pi],exact_first_order=True)
    np.testing.assert_allclose(half,.5*one,rtol=1e-14)

@pytest.mark.parametrize('r,c',[([np.nan],[0.]),([15.],[np.nan]),([15.],[1.])])
def test_directed_invalid_or_singular_inputs(cache,r,c):
    with pytest.raises(ValueError):cache.directed_spectrum(r,c)

def test_does_not_silently_drop_acceptance(cache):
    alpha=np.zeros(cache.degree+3);alpha[0]=4*np.pi;alpha[-1]=.1
    with pytest.raises(ValueError,match='too few'):
        cache.acceptance_charge([15.],[.1],alpha)

def test_band_medium_and_frequency_must_match(cache):
    rr=cache.grid.radii_m*2
    other=ResponseCache(CacheGrid(rr,cache.grid.omega_per_ns),replace(M,g=.6),S,cache.moments)
    with pytest.raises(ValueError,match='same medium'):BandedResponseCache([cache,other])
    other=ResponseCache(CacheGrid(rr,np.array([0,.02,.08])),M,S,cache.moments)
    with pytest.raises(ValueError,match='frequency'):BandedResponseCache([cache,other])

def test_banded_nan_rejected(cache):
    with pytest.raises(ValueError):BandedResponseCache([cache]).directed_charge([np.nan],[0.])

def test_zero_frequency_only_in_charge(cache,monkeypatch):
    seen=[];original=module.single_spectrum
    def observe(omega,*args,**kwargs):
        seen.append(np.array(omega));return original(omega,*args,**kwargs)
    monkeypatch.setattr(module,'single_spectrum',observe)
    charge=cache.directed_charge([15.],[.3])
    assert len(seen)==1 and np.array_equal(seen[0],[0.])
    np.testing.assert_allclose(charge,cache.directed_spectrum([15.],[.3])[0].real)

def test_spline_reused_and_can_be_invalidated(cache,monkeypatch):
    cache.clear_interpolation_cache();seen=[];original=module.CubicSpline
    def record(*args,**kwargs):seen.append(1);return original(*args,**kwargs)
    monkeypatch.setattr(module,'CubicSpline',record)
    cache.moments_at([12.],degrees=2);cache.moments_at([13.],degrees=2)
    assert len(seen)==1
    cache.clear_interpolation_cache();cache.moments_at([12.],degrees=2)
    assert len(seen)==2

def test_phase_demodulation_and_roundtrip(tmp_path):
    settings=replace(S,spatial_degree=1)
    grid=CacheGrid.geometric(10,30,8,[0.,.7])
    r=grid.radii_m;w=grid.omega_per_ns
    scale=np.exp(-M.absorption_per_m*r)/(4*np.pi*r*r)
    f=scale[None,:]*(1+np.log(r)[None,:]**3)*np.exp(1j*w[:,None]*r[None,:]/M.speed_m_per_ns)
    values=np.broadcast_to(f[:,:,None,None],(2,len(r),2,2)).copy()
    c=ResponseCache(grid,M,settings,values,radial_phase='flight')
    rq=np.array([11.5,17.,26.])
    target=np.exp(-M.absorption_per_m*rq)/(4*np.pi*rq*rq)*(1+np.log(rq)**3)
    target=target[None,:]*np.exp(1j*w[:,None]*rq[None,:]/M.speed_m_per_ns)
    np.testing.assert_allclose(c.moments_at(rq)[:,:,0,0],target,rtol=1e-13,atol=1e-15)
    c.save(tmp_path/'c.npz');loaded=ResponseCache.load(tmp_path/'c.npz')
    assert loaded.radial_phase=='flight'
    np.testing.assert_allclose(loaded.moments_at(rq),c.moments_at(rq))

@pytest.mark.parametrize('photons',[-1,float('nan'),float('inf')])
def test_invalid_photon_count(cache,photons):
    with pytest.raises(ValueError):cache.directed_charge([15.],[0.],photons=photons)
