"""Spectral Q1 charges and certified coarse-time refinement on glass paths."""
import numpy as np
import pytest
from lighthit.receiver import SphereSurface
from lighthit.receiver_first_fold import acceptance_moment_weights
from lighthit.receiver_geometry import ReceiverGeometry,ReceiverGeometryConfig
from lighthit.receiver_first_spectral import SpectralFirstEntry,SpectralFirstEntryConfig


@pytest.fixture(scope='module')
def prepared():
    geometry=ReceiverGeometry.build(radii_unit=np.array([14.,14.7,15.435]),g=.5,max_excess_unit=500.,
        config=ReceiverGeometryConfig(hit_nodes=9,miss_nodes=33,path_grid_nodes=129,path_nodes=128))
    t=dict(mut=np.array([.1,.13]),mus=np.array([.02,.03]),speed=np.array([.22,.215]),
        s0w=np.array([.4,.6]),s2w=np.array([.2,.3]))
    coarse=SpectralFirstEntry.build(geometry=geometry,radius_m=.216,tables=t,config=SpectralFirstEntryConfig(17))
    fine=SpectralFirstEntry.build(geometry=geometry,radius_m=.216,tables=t,config=SpectralFirstEntryConfig(129))
    return geometry,t,coarse,fine


def test_spectral_time_bound_controls_actual_bin_l1_and_zero_budget(prepared,tmp_path):
    geometry,t,coarse,fine=prepared
    r=np.array([14.2,14.9,15.2])*.216;angle=np.array([.01,.2,2.]);clock=np.array([.1,-.2,.3]);spread=np.array([0.,.8,.4])
    angular=acceptance_moment_weights([.3,.4,np.sqrt(.75)],[1.,.2,.1,.02])
    moments=np.stack([np.array([[w],[-w]])*angular for w in [1.,.4,.2]])
    edges=np.arange(10.,70.,.5);single={k:v[:1] for k,v in t.items()};args=(r,angle,clock,spread,moments,edges)
    q0,b0,_=fine.sum_groups(*args,tables=single,geometry=geometry,relative_tolerance=1e6)
    q,b,meta=coarse.sum_groups(*args,tables=single,geometry=geometry,relative_tolerance=.02)
    assert q==pytest.approx(q0,rel=1e-14)
    assert abs(b-b0).sum()<=meta['bin_l1_bound']+1e-13
    assert meta['bin_l1_bound']<=meta['bin_l1_budget']+1e-14
    q,b,meta=coarse.sum_groups(*args,tables=single,geometry=geometry,relative_tolerance=0.)
    np.testing.assert_allclose(b,b0,rtol=1e-12,atol=1e-14)
    assert b.min()>-1e-13
    coarse.save(tmp_path);loaded=SpectralFirstEntry.load(tmp_path)
    assert isinstance(loaded.hit_cdf,np.memmap)
    assert loaded.band(single)==0
    other=dict(single,mut=single['mut']*1.01)
    assert loaded.band(other) is None
    np.testing.assert_array_equal(loaded.hit_error,coarse.hit_error)
    coarse.prepare_quadratic(geometry,tmp_path)
    loaded=SpectralFirstEntry.load(tmp_path)
    q,b,meta=loaded.sum_groups(*args,tables=single,geometry=geometry,relative_tolerance=.02)
    assert meta['quadratic_time']
    assert abs(b-b0).sum()<=meta['bin_l1_bound']+1e-13
    assert meta['bin_l1_bound']<=meta['bin_l1_budget']+1e-14
    _,exact,_=loaded.sum_groups(*args,tables=single,geometry=geometry,relative_tolerance=0.)
    np.testing.assert_allclose(exact,b0,rtol=1e-12,atol=1e-14)


def test_shared_absolute_budget_can_keep_weak_band_coarse_without_dropping_charge(prepared):
    geometry,t,coarse,fine=prepared
    angular=acceptance_moment_weights([0.,0.,1.],[1.,.2,.1])
    moments=np.array([[[1e-5],[-1e-5]]])*angular
    args=(np.array([3.1]),np.array([.2]),np.array([0.]),np.array([.5]),moments,np.arange(10.,70.,1.))
    single={k:v[1:] for k,v in t.items()}
    q,b,meta=coarse.sum_groups(*args,tables=single,geometry=geometry,relative_tolerance=0.,absolute_tolerance=1e-6)
    assert meta['fine_groups']==0 and q>0
    q0,b0,_=fine.sum_groups(*args,tables=single,geometry=geometry,relative_tolerance=1e6)
    assert q==pytest.approx(q0,rel=1e-14)
    assert abs(b-b0).sum()<=meta['bin_l1_bound']+1e-13


@pytest.mark.parametrize('tolerance',[0.,1e6])
def test_spectral_clock_window_preserves_stationary_charge_and_partial_overlap(prepared,tolerance):
    geometry,t,coarse,_=prepared;single={k:v[:1] for k,v in t.items()}
    r=np.array([3.1]);angle=np.array([.2]);spread=np.array([20.]);edges=np.arange(10.,71.,1.)
    moments=np.zeros((1,2,10));moments[0,0,0]=1.
    options=dict(tables=single,geometry=geometry,relative_tolerance=tolerance)
    q0,_,_=coarse.sum_groups(r,angle,np.zeros(1),spread,moments,**options)
    front=(r[0]-coarse.radius_m)/single['speed'][0]
    for clock in [-1e6,1e6,edges[-1]-front+spread[0]/2]:
        q,b,_=coarse.sum_groups(r,angle,np.array([clock]),spread,moments,edges,**options)
        assert q==q0
        np.testing.assert_array_equal(b,0.)
    clock=edges[-1]-front+spread[0]/2-1.
    q,b,_=coarse.sum_groups(r,angle,np.array([clock]),spread,moments,edges,**options)
    assert q==q0 and b[-1]>0


def test_parallel_spectral_blocks_equal_full_preparation(tmp_path):
    geometry=ReceiverGeometry.build(radii_unit=np.array([2.,2.1,2.2,2.3]),g=.5,max_excess_unit=20.,
        config=ReceiverGeometryConfig(hit_nodes=3,miss_nodes=3,path_grid_nodes=17,path_nodes=16,path_order=4,scatter_order=8))
    geometry.save(tmp_path/'geometry');tables=dict(mut=np.array([.1]),mus=np.array([.02]),speed=np.array([.22]),s0w=np.array([1.]),s2w=np.array([.5]))
    config=SpectralFirstEntryConfig(5)
    direct=SpectralFirstEntry.build(geometry=geometry,radius_m=.216,tables=tables,config=config)
    parallel=SpectralFirstEntry.build_parallel(geometry_directory=tmp_path/'geometry',radius_m=.216,tables=tables,
        output=tmp_path/'spectral',config=config,jobs=2)
    for name in ('hit_charge','hit_cdf','hit_error','miss_charge','miss_cdf','miss_error'):
        np.testing.assert_array_equal(getattr(parallel,name),getattr(direct,name))


def test_signed_moment_cancellation_is_checked_before_fast_deposition(prepared,tmp_path):
    geometry,t,coarse,fine=prepared
    coarse.prepare_quadratic(geometry,tmp_path)
    single={k:v[:1] for k,v in t.items()};r=np.array([3.1]);angle=np.array([.2])
    clock=np.zeros(1);spread=np.array([.5]);edges=np.arange(10.,70.,.5)
    m=np.zeros((1,2,10));m[0,0,0]=1.
    q0,_,_=fine.sum_groups(r,angle,clock,spread,m,tables=single,geometry=geometry)
    z=np.zeros_like(m);z[0,0,2]=1.
    qz,_,_=fine.sum_groups(r,angle,clock,spread,z,tables=single,geometry=geometry)
    m[0,0,2]=-q0/qz
    _,reference,_=fine.sum_groups(r,angle,clock,spread,m,edges,tables=single,geometry=geometry,relative_tolerance=0.)
    _,actual,meta=coarse.sum_groups(r,angle,clock,spread,m,edges,tables=single,geometry=geometry,
        relative_tolerance=0.,absolute_tolerance=1e-11)
    assert abs(actual-reference).sum()<=meta['bin_l1_bound']+1e-13
    assert meta['guard_refinements']>0 or meta['exact_groups']>0
