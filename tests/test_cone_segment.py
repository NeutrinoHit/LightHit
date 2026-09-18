from dataclasses import replace
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.spatial.transform import Rotation
from scipy.special import eval_legendre
from lighthit import Medium,SolverSettings
from lighthit.cache import CacheGrid,ResponseCache
from lighthit.experimental.cone_segment import ConeSegment,segment_spectrum,ballistic_spectrum

M=Medium(.04,.05,.7,1.35)
SEG=ConeSegment((0.,0.,0.),(0.,0.,1.),8.,.99,1.34,1.0)
POS=np.array([[10.,0.,14.],[-11.,4.,12.]])
@pytest.fixture(scope='module')
def cache():
    return ResponseCache.build(M,SolverSettings(32,80,4,.05,8),CacheGrid.geometric(9,25,20,[0.,.02,.1]),radial_phase='flight')

def test_cone_addition_identity():
    mu=.75;nu=.3
    phi=2*np.pi*np.arange(1024)/1024
    c=mu*nu+np.sqrt(1-mu*mu)*np.sqrt(1-nu*nu)*np.cos(phi)
    for l in range(24):
        assert np.mean(eval_legendre(l,c))==pytest.approx(eval_legendre(l,mu)*eval_legendre(l,nu),abs=2e-14)

def test_ballistic_jacobian():
    r=POS[:1];z=r[0,2];b=r[0,0];mu=SEG.cone_cosine
    root=z-b*mu/np.sqrt(1-mu*mu)
    sigma=1e-4
    def f(a):
        dist=np.sqrt(b*b+(z-a)**2); c=(z-a)/dist
        delta=np.exp(-.5*((c-mu)/sigma)**2)/(np.sqrt(2*np.pi)*sigma)
        return np.exp(-M.extinction_per_m*dist)/(2*np.pi*dist**2)*delta
    numeric=quad(f,root-.1,root+.1,epsabs=1e-13,points=[root],limit=100)[0]
    expected=ballistic_spectrum([0],SEG,r,M)[0,0].real
    assert expected==pytest.approx(numeric,rel=2e-6)

def test_additivity(cache):
    whole=segment_spectrum(cache,SEG,POS,longitudinal_order=32)
    a=replace(SEG,length_m=4.)
    b=replace(SEG,start_m=(0.,0.,4.),length_m=4.,start_time_ns=4/SEG.speed_m_per_ns)
    split=segment_spectrum(cache,a,POS,longitudinal_order=24)+segment_spectrum(cache,b,POS,longitudinal_order=24)
    scale=np.max(np.abs(whole),axis=(0,2))
    assert np.max(np.abs(whole-split)/scale[None,:,None])<2e-5

def test_rotations_translations_and_time(cache):
    base=segment_spectrum(cache,SEG,POS)
    Q=Rotation.from_rotvec([.2,-.4,.6]).as_matrix();shift=np.array([11.,-8.,3.])
    rotated=replace(SEG,start_m=tuple(shift),direction=tuple(Q@np.array(SEG.direction)))
    same=segment_spectrum(cache,rotated,POS@Q.T+shift)
    np.testing.assert_allclose(base,same,rtol=2e-12,atol=1e-18)
    delayed=segment_spectrum(cache,replace(SEG,start_time_ns=17.),POS)
    np.testing.assert_allclose(delayed,base*np.exp(1j*cache.grid.omega_per_ns*17)[:,None,None],rtol=1e-12,atol=1e-18)

def test_scaling(cache):
    np.testing.assert_allclose(segment_spectrum(cache,replace(SEG,photons_per_m=2),POS),2*segment_spectrum(cache,SEG,POS))

@pytest.mark.parametrize('kwargs',[{'beta':0.5},{'length_m':-1},{'direction':(0,0,2)}])
def test_bad_segment(kwargs):
    with pytest.raises(ValueError):replace(SEG,**kwargs)
