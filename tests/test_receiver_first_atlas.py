"""Spectral first-entry interpolation, grazing branches and source clocks."""
from dataclasses import replace
import numpy as np
import pytest
from scipy.integrate import quad_vec
from lighthit.receiver import SphereSurface
from lighthit.receiver_first import first_entry_profile
from lighthit.receiver_first_atlas import FirstEntryAtlas,FirstEntryAtlasConfig,_path_extent
from lighthit.receiver_first_fold import fold_first_entry,acceptance_moment_weights


@pytest.fixture(scope='module')
def atlas():
    tables=dict(mut=np.array([.1,.13]),mus=np.array([.02,.025]),speed=np.array([.22,.215]),
        s0w=np.array([.4,.6]),s2w=np.array([.4,.6])/1.35**2)
    return FirstEntryAtlas.build(radii_m=np.array([3.,3.15,3.3075]),surface=SphereSurface(.216),g=.5,
        tables=tables,config=FirstEntryAtlasConfig(hit_nodes=17,miss_nodes=65,
            time_nodes=257,path_nodes=128))


def test_interpolated_charge_and_bins_match_direct_glass_profiles(atlas):
    r=np.sqrt(3*3.15);R=atlas.radius_m;cap=np.arcsin(R/r)
    edges=(r-R)/max(atlas.tables['speed'])+np.arange(-1.,60.1,.5)
    angular=acceptance_moment_weights([.3,.4,np.sqrt(.75)],[1.,.3,.1])
    for theta in [0.,.5*cap,.99*cap,1.01*cap,.7,2.5]:
        q,cdf=atlas.moments(r,theta,edges)
        extent=np.sqrt((r-R)*(r+R)) if theta<=cap else _path_extent(r,R,atlas.g,atlas.tables,1e-8)
        profile=first_entry_profile(SphereSurface(R),distance_m=r,direction_cosine=np.cos(theta),
            g=atlas.g,max_path_m=extent,path_nodes=256)
        t=atlas.tables
        ref=fold_first_entry(profile,extinction_per_m=t['mut'],scattering_per_m=t['mus'],
            speed_m_per_ns=t['speed'],spectral_weights=np.stack((t['s0w'],t['s2w']),axis=-1),time_edges_ns=edges)
        actual=(q[0]-q[1])@angular;expected=(ref.charge[0]-ref.charge[1])@angular
        assert abs(actual/expected-1)<.03
        bins=np.diff((cdf[:,0]-cdf[:,1])@angular)
        target=(ref.bins[:,0]-ref.bins[:,1])@angular
        assert np.sum(abs(bins-target))/np.sum(abs(target))<.03
        assert bins.min()>-1e-14


def test_uniform_source_clock_uses_the_time_antiderivative(atlas):
    r=3.08;theta=.2;edges=np.arange(12.,35.,1.);width=.8
    q,smoothed=atlas.moments(r,theta,edges,clock_width_ns=width)
    exact=quad_vec(lambda shift:atlas.moments(r,theta,edges-shift)[1],
                   -width/2,width/2,epsabs=1e-11,epsrel=1e-8)[0]/width
    np.testing.assert_allclose(smoothed,exact,rtol=3e-7,atol=2e-11)
    np.testing.assert_array_equal(q,atlas.moments(r,theta)[0])


def test_mmap_round_trip_and_optical_contract(atlas,tmp_path):
    atlas.save(tmp_path);loaded=FirstEntryAtlas.load(tmp_path)
    assert isinstance(loaded.hit_cdf,np.memmap)
    assert loaded.metadata['convergence_verified'] is False
    query=(3.1,.3,np.array([10.,15.,20.,200.]))
    for x,y in zip(atlas.moments(*query),loaded.moments(*query)):
        np.testing.assert_array_equal(x,y)
    loaded.validate(tables=atlas.tables,surface=SphereSurface(.216),g=.5)
    with pytest.raises(ValueError,match='another'):
        loaded.validate(tables=atlas.tables,surface=SphereSurface(.217),g=.5)
    changed=dict(atlas.tables,mut=atlas.tables['mut']*1.01)
    with pytest.raises(ValueError,match='another'):
        loaded.validate(tables=changed,surface=SphereSurface(.216),g=.5)


def test_cdf_is_zero_before_first_glass_arrival(atlas):
    q,cdf=atlas.moments(3.1,.2,[-10.,0.,5.])
    np.testing.assert_array_equal(cdf,0.)
    assert q[0,0]>0
    with pytest.raises(ValueError,match='outside'):atlas.moments(.2,.2)
    with pytest.raises(ValueError,match='clock'):atlas.moments(3.1,.2,clock_width_ns=-1.)


def test_compiled_group_sum_matches_folded_moments_and_clocks(atlas):
    radii=np.array([3.01,3.2,3.1]);angles=np.array([.02,.5,2.1])
    clocks=np.array([.1,.7,-.3]);spreads=np.array([0.,.8,.4]);edges=np.arange(10.,70.,.5)
    coefficients=np.array([[1.,-1.],[.5,-.4],[.2,-.15]])
    looks=np.array([[0.,0.,1.],[.3,.4,np.sqrt(.75)],[.8,.6,0.]])
    moments=np.stack([coefficients[j,:,None]*acceptance_moment_weights(look,[1.,.2,.1])
                      for j,look in enumerate(looks)])
    expected_charge=0.;expected_bins=np.zeros(len(edges)-1)
    for r,theta,clock,spread,m in zip(radii,angles,clocks,spreads,moments):
        q,c=atlas.moments(r,theta,edges-clock,clock_width_ns=spread)
        expected_charge+=np.sum(q*m)
        expected_bins+=np.diff(np.einsum('tfm,fm->t',c,m))
    charge,bins=atlas.sum_groups(radii,angles,clocks,spreads,moments,edges)
    assert charge==pytest.approx(expected_charge,rel=1e-13)
    np.testing.assert_allclose(bins,expected_bins,rtol=2e-8,atol=1e-13)


@pytest.mark.parametrize('quadratic',[False,True])
def test_time_contiguous_rows_preserve_charge_bins_and_cancellation_guard(atlas,tmp_path,quadratic):
    atlas.save(tmp_path);data=FirstEntryAtlas.load(tmp_path)
    node=data.radii_m[1]
    r=np.array([3.01,np.nextafter(node,0.),node,np.nextafter(node,np.inf),3.2,3.3])
    angles=np.array([0.,.01,.07,.2,2.,np.pi-1e-12]);clocks=np.array([0.,.1,.7,-.3,-1e6,1e6])
    spread=np.array([0.,.8,.4,20.,.1,.5]);edges=np.arange(10.,70.,.5)
    rng=np.random.default_rng(918);moments=rng.normal(size=(len(r),2,10))
    before={};after={};args=(r,angles,clocks,spread,moments,edges)
    q0,b0=data.sum_groups(*args,quadratic_time=quadratic,contiguous_time=False,report=before)
    q,b=data.sum_groups(*args,quadratic_time=quadratic,report=after)
    assert q==q0
    np.testing.assert_array_equal(b,b0)
    assert after['contiguous_time'] and not before['contiguous_time']
    assert after['guard_refinements']==before['guard_refinements']


def test_insufficient_memory_retains_the_canonical_time_interpolator(atlas,tmp_path,monkeypatch):
    import lighthit._memory as memory
    atlas.save(tmp_path);data=FirstEntryAtlas.load(tmp_path)
    moments=np.zeros((1,2,10));moments[0,0,0]=1.
    args=(np.array([3.1]),np.array([.2]),np.zeros(1),np.array([.5]),moments,np.arange(10.,70.,1.))
    q0,b0=data.sum_groups(*args,contiguous_time=False)
    monkeypatch.setattr(memory,'available_bytes',lambda:0)
    report={};q,b=data.sum_groups(*args,report=report)
    assert q==q0 and not report['contiguous_time']
    np.testing.assert_array_equal(b,b0)


def test_concurrent_time_rows_match_the_canonical_interpolator(atlas,tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    atlas.save(tmp_path);data=FirstEntryAtlas.load(tmp_path)
    moments=np.zeros((64,2,10));moments[:,0,0]=1.;edges=np.arange(10.,70.,.5)
    cases=[(np.full(64,r),np.full(64,.2),np.zeros(64),np.full(64,.5),moments,edges) for r in [3.01,3.3]]
    expected=[data.sum_groups(*args,contiguous_time=False) for args in cases]
    barrier=Barrier(2)
    def query(args):
        barrier.wait()
        return data.sum_groups(*args)
    with ThreadPoolExecutor(max_workers=2) as pool:actual=list(pool.map(query,cases))
    for before,after in zip(expected,actual):
        assert before[0]==after[0]
        np.testing.assert_array_equal(before[1],after[1])


@pytest.mark.parametrize('quadratic',[False,True])
def test_geometry_and_spectrum_reuse_preserves_the_time_response(atlas,tmp_path,quadratic):
    from lighthit._receiver_first_lookup import charge_and_time_bounds
    atlas.save(tmp_path);data=FirstEntryAtlas.load(tmp_path)
    r=np.array([3.01,3.2,3.1]);angles=np.array([.02,.5,2.1]);clocks=np.array([.1,.7,-.3])
    spreads=np.array([0.,.8,20.]);edges=np.arange(10.,70.,.5)
    moments=np.random.default_rng(1334).normal(size=(3,2,10));t=data.tables
    _,_,prepared=charge_and_time_bounds(r,angles,moments,np.log(data.radii_m),data.radius_m,data.g,
        t['mut'],t['mus'],t['s0w'],t['s2w'],float(max(t['speed'])),data.hit_charge,data.miss_charge,
        np.zeros_like(data.hit_charge),np.zeros_like(data.miss_charge))
    args=(r,angles,clocks,spreads,moments,edges);before={};after={}
    q0,b0=data.sum_groups(*args,quadratic_time=quadratic,report=before)
    q,b=data.sum_groups(*args,quadratic_time=quadratic,report=after,_prepared=prepared)
    assert q==q0 and before['guard_refinements']==after['guard_refinements']
    np.testing.assert_array_equal(b,b0)


@pytest.mark.parametrize('quadratic',[False,True])
def test_clock_window_keeps_all_time_charge_and_edge_overlaps(atlas,quadratic):
    r=np.array([3.1]);angle=np.array([.2]);spread=np.array([20.]);edges=np.arange(10.,71.,1.)
    moments=np.zeros((1,2,10));moments[0,0,0]=1.
    q0,_=atlas.sum_groups(r,angle,np.zeros(1),spread,moments)
    front=(r[0]-atlas.radius_m)/max(atlas.tables['speed'])
    for clock in [-1e6,1e6,edges[-1]-front+spread[0]/2]:
        q,b=atlas.sum_groups(r,angle,np.array([clock]),spread,moments,edges,quadratic_time=quadratic)
        assert q==q0
        np.testing.assert_array_equal(b,0.)
    clock=edges[-1]-front+spread[0]/2-1.
    q,b=atlas.sum_groups(r,angle,np.array([clock]),spread,moments,edges,quadratic_time=quadratic)
    assert q==q0 and b[-1]>0


def _pencils(angles,variance=0.):
    angles=np.asarray(angles);n=len(angles)
    directions=np.stack((np.sin(angles),np.zeros(n),np.cos(angles)),axis=1)
    major=np.stack((np.cos(angles),np.zeros(n),-np.sin(angles)),axis=1)
    return dict(position=np.zeros((n,3)),direction=directions,major=major,
        w0=np.full(n,1./n),w2=np.full(n,-1./n),time=np.zeros(n),
        var_major=np.full(n,variance),var_minor=np.full(n,variance),pos_cov=np.zeros((n,6)))


@pytest.mark.parametrize('radial_only',[False,True])
@pytest.mark.parametrize('select',[False,True])
def test_exact_shared_emission_geometry_preserves_groups_bitwise(atlas,radial_only,select):
    from lighthit.prompt import PromptConfig
    from lighthit._receiver_first_groups import share_pencil_geometry
    angles=np.r_[.01,.03,.1,.2,.8,1.5,2.5,3.]*1.
    pencils=_pencils(angles,variance=.0003)
    pencils['position'][2,0]=-0.;pencils['time'][3]=1e-12
    receiver=np.array([0.,0.,3.08]);look=np.array([.3,.4,np.sqrt(.75)])
    settings=PromptConfig();options=dict(radial_only=radial_only)
    if select:options['selected']=np.arange(len(angles))%2==0
    before,_=atlas.pencil_groups(pencils,receiver,look,[.3082,.54192,.19831,-.04912],settings,**options)
    share_pencil_geometry(pencils)
    after,metadata=atlas.pencil_groups(pencils,receiver,look,[.3082,.54192,.19831,-.04912],settings,**options)
    assert metadata['cached_emission_geometries']==1
    for a,b in zip(before,after):np.testing.assert_array_equal(a,b)


def test_grazing_hit_and_miss_cannot_merge_into_one_source_group(atlas):
    from lighthit.prompt import PromptConfig
    r=3.08;cap=np.arcsin(atlas.radius_m/r);angles=[.999*cap,1.001*cap]
    pencils=_pencils(angles);receiver=np.array([0.,0.,r]);look=np.array([.3,.4,np.sqrt(.75)])
    edges=np.arange(10.,70.,.5);poly=[1.,.2,.1]
    q,b,meta=atlas.evaluate_pencils(pencils,receiver,look,poly,PromptConfig(),time_edges_ns=edges)
    assert meta['groups']==2
    expected_q=0.;expected_b=np.zeros(len(edges)-1)
    for angle in angles:
        one=_pencils([angle])
        part=atlas.evaluate_pencils(one,receiver,look,poly,PromptConfig(),time_edges_ns=edges)
        expected_q+=part[0]/2;expected_b+=part[1]/2
    assert q==pytest.approx(expected_q,rel=1e-12)
    np.testing.assert_allclose(b,expected_b,rtol=1e-9,atol=1e-14)


def test_source_width_uses_cubature_without_changing_emitted_photon_mass(atlas):
    from lighthit._receiver_first_groups import prepare
    from lighthit._prompt_numba import _prepare_pencil_shape
    from lighthit.receiver_first_atlas import _normal_rule
    from lighthit.prompt import PromptConfig
    p=_pencils([0.],variance=.001);cfg=PromptConfig();x,w=_normal_rule(3)
    shape=_prepare_pencil_shape(p['direction'],p['major'],p['pos_cov'])
    out=prepare(np.array([0.,0.,3.08]),np.array([0.,0.,1.]),0.,p['position'],p['direction'],
        p['w0'],p['w2'],p['time'],p['major'],p['var_major'],p['var_minor'],shape,
        np.array([1.,0.,0.,0.]),atlas.radius_m,cfg.merge_log_r,cfg.merge_tau_ns,
        cfg.merge_theta_min,cfg.merge_theta_ratio,atlas.config.hit_nodes,atlas.config.miss_nodes,
        max(atlas.tables['speed']),x,w,0.)
    assert out[5]==9
    assert out[4][:,0,0].sum()==pytest.approx(1.,rel=1e-14)
    assert out[4][:,1,0].sum()==pytest.approx(-1.,rel=1e-14)


def test_coarse_hierarchy_uses_parent_or_children_once(atlas):
    from lighthit._receiver_first_groups import select_screen
    from lighthit._prompt_numba import _prepare_pencil_shape
    p=_pencils([0.,-.01,.01],variance=.001)
    shape=_prepare_pencil_shape(p['direction'],p['major'],p['pos_cov'])
    parent=np.array([-2,0,0]);next_node=np.array([3,2,3])
    for receiver,expected in [(np.array([0.,0.,3.]),[False,True,True]),
                              (np.array([3.,0.,0.]),[True,False,False])]:
        selected=select_screen(receiver,p['position'],p['direction'],p['var_major'],shape,parent,next_node,.216,6.)
        np.testing.assert_array_equal(selected,expected)
        assert np.array([1.,.3,.7])[selected].sum()==pytest.approx(1.)
        from lighthit.prompt import PromptConfig
        config=PromptConfig()
        p['hierarchy_parent']=parent;p['next_node']=next_node
        arguments=(p,receiver,np.array([0.,0.,1.]),np.array([.3,.5,.2,-.05]),config)
        explicit,_=atlas.pencil_groups(*arguments,selected=selected)
        integrated,_=atlas.pencil_groups(*arguments)
        for before,after in zip(explicit,integrated):np.testing.assert_array_equal(before,after)


def test_parallel_radial_blocks_preserve_the_common_physical_time_coordinate(tmp_path):
    tables=dict(mut=np.array([.1]),mus=np.array([.02]),speed=np.array([.22]),
                s0w=np.array([1.]),s2w=np.array([.5]))
    options=dict(radii_m=np.array([3.,3.05,3.1,3.15]),surface=SphereSurface(.216),g=.5,tables=tables,
        config=FirstEntryAtlasConfig(hit_nodes=3,miss_nodes=3,time_nodes=17,path_nodes=16,path_order=4,scatter_order=8))
    direct=FirstEntryAtlas.build(**options)
    parallel=FirstEntryAtlas.build_parallel(**options,output=tmp_path/'blocks',jobs=2)
    for name in ('hit_charge','hit_cdf','hit_integral','miss_charge','miss_cdf','miss_integral'):
        np.testing.assert_array_equal(getattr(parallel,name),getattr(direct,name))
    assert parallel.metadata['parallel_workers']==2
    assert parallel.metadata['maximum_scaled_tail_bound']<1e-7


def test_near_glass_grazing_sides_use_impact_resolution_not_cosine_ulps():
    from lighthit.receiver_first_atlas import _geometry
    R=.216;r=R*1.001
    for branch,x in [('hit',1.),('miss',0.)]:
        theta,c=_geometry(r,R,branch,x)
        impact=(r/R)*np.sqrt((1-c)*(1+c))
        assert (c>0 and impact<=1)==(branch=='hit')
        assert np.isfinite(theta)


def test_far_glass_grazing_uses_cosine_resolution_without_a_long_impact_loop(monkeypatch):
    from lighthit.receiver_first_atlas import _geometry
    original=np.nextafter;calls=0
    def counted(*args):
        nonlocal calls
        calls+=1
        assert calls<64,'grazing classification must not walk unresolved impact ulps'
        return original(*args)
    monkeypatch.setattr(np,'nextafter',counted)
    for r in [1.001,14.,1790.,5000.,12000.,1e6]:
        for branch,x in [('hit',1.),('miss',0.)]:
            _,c=_geometry(r,1.,branch,x)
            hit=c>0 and r*np.sqrt((1-c)*(1+c))<=1
            assert hit==(branch=='hit')


def test_adaptive_time_error_bounds_actual_bin_changes(atlas,tmp_path):
    from lighthit.receiver_first_adaptive import AdaptiveFirstEntry
    a=AdaptiveFirstEntry.of(atlas,stride=4)
    radii=np.array([3.02,3.1,3.2]);theta=np.array([.01,.3,2.5])
    clocks=np.array([.1,-.3,.6]);spreads=np.array([0.,.8,.1]);edges=np.arange(10.,70.,.5)
    angular=acceptance_moment_weights([.3,.4,np.sqrt(.75)],[1.,.2,.1])
    moments=np.stack([np.array([[w],[-w]])*angular for w in [1.,.4,.2]])
    args=(radii,theta,clocks,spreads,moments,edges)
    _,reference=atlas.sum_groups(*args)
    q,b,metadata=a.sum_groups(*args,relative_tolerance=.02)
    assert abs(b-reference).sum()<=metadata['bin_l1_bound']+1e-11
    assert metadata['bin_l1_bound']<=metadata['bin_l1_budget']+1e-14
    _,exact,metadata=a.sum_groups(*args,relative_tolerance=0.)
    np.testing.assert_array_equal(exact,reference)
    a.save(tmp_path/'time-level');loaded=AdaptiveFirstEntry.load(tmp_path/'time-level',fine=atlas)
    np.testing.assert_array_equal(loaded.hit_error,a.hit_error)
