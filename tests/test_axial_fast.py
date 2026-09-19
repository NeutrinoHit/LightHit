"""Integration tests to run against the user's full 1bb11dc checkout."""
import numpy as np
import pytest
pytest.importorskip('numba')
from scipy.spatial.transform import Rotation
from lighthit import Medium,SolverSettings
from lighthit.cache import ResponseCache,CacheGrid
from lighthit.experimental.g4_source import LightElements,SourceContract
from lighthit.experimental.axial_source import AxialSource,AxisFrame,axial_response
from lighthit.experimental.event_moments import KernelChannels
from lighthit.experimental.axial_fast import (PreparedAxialKernel,
    compile_axial_source_fast,rigidly_moved_source)


def event():
    rng=np.random.default_rng(883);n=35
    x=rng.normal(size=(n,3))*[.1,.1,.8]
    u=rng.normal(size=(n,3));u[:,2]+=2;u/=np.linalg.norm(u,axis=1)[:,None]
    h=rng.uniform(.001,.03,n);t=rng.uniform(0,3,n)
    return LightElements(x,u,h,rng.uniform(.1,3,n),rng.uniform(.75,.8,n),t,t+h/.27,
        np.arange(n),np.arange(n),SourceContract(),{'synthetic':True})

@pytest.fixture(scope='module')
def cache():
    med=Medium(.04,.05,.7,1.35,450,'synthetic-fast-tests')
    return ResponseCache.build(med,SolverSettings(12,16,3,.06,8),
        CacheGrid.geometric(8,70,18,[0,.12,.45]))

@pytest.mark.parametrize('deposit',['linear','nearest'])
@pytest.mark.parametrize('order',[1,2,4])
def test_compiler_matches_original(deposit,order):
    e=event();w=np.array([0,.12,.45]);f=AxisFrame.of(e)
    kw=dict(azimuthal_degree=3,cell_m=.3,frame=f,deposit=deposit,element_order=order)
    old=AxialSource.of(e,8,w,**kw)
    fast=compile_axial_source_fast(e,8,w,chunk=13,**kw)
    assert np.array_equal(old.points_m(),fast.points_m())
    np.testing.assert_allclose(old.channels,fast.channels,rtol=1e-10,atol=1e-11*np.max(np.abs(old.channels)))

@pytest.mark.parametrize('phase',['none','flight'])
@pytest.mark.parametrize('m',[0,2,4])
def test_query_matches_original(cache,phase,m):
    cache.radial_phase=phase;cache.clear_interpolation_cache()
    w=cache.grid.omega_per_ns
    source=AxialSource.of(event(),16,w,azimuthal_degree=m,cell_m=.3)
    receivers=np.array([[18,2,6],[-12,18,3],[5,-25,4.]])
    fast=PreparedAxialKernel.from_cache(cache).apply(source,receivers,source_omega_per_ns=w)
    old=np.stack([axial_response(KernelChannels.of(cache,o,16),source,receivers) for o in range(2)],-1)
    scale=np.max(np.abs(old),axis=(0,2))
    assert np.max(np.abs(fast-old)/scale[None,:,None])<1e-9


def test_rigid_pose_reuses_the_source(cache):
    w=cache.grid.omega_per_ns
    source=compile_axial_source_fast(event(),16,w,azimuthal_degree=4,cell_m=.3)
    matrix=Rotation.from_rotvec([.21,-.37,.5]).as_matrix();shift=np.array([5.,2.,-1.])
    moved=rigidly_moved_source(source,matrix,shift,7.)
    assert moved.channels is source.channels
    rec=np.array([[18.,3,6],[-12,18,3]])
    kernel=PreparedAxialKernel.from_cache(cache)
    a=kernel.apply(source,rec,source_omega_per_ns=w)
    b=kernel.apply(moved,rec@matrix.T+shift,source_omega_per_ns=w)
    np.testing.assert_allclose(b,a*np.exp(1j*w*7)[:,None,None],rtol=1e-9,atol=1e-12*np.abs(a).max())


def test_frequency_mismatch_is_refused(cache):
    w=cache.grid.omega_per_ns
    source=compile_axial_source_fast(event(),16,w,azimuthal_degree=2,cell_m=.3)
    with pytest.raises(ValueError,match='frequency'):
        PreparedAxialKernel.from_cache(cache).apply(source,[[18,0,0]],source_omega_per_ns=w+.001)
