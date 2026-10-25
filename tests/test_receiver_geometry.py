"""Wavelength-independent sphere geometry and analytic one-band folding."""
import numpy as np
import pytest
from scipy.integrate import quad_vec
from scipy.interpolate import PchipInterpolator
from lighthit.receiver import SphereSurface
from lighthit.receiver_geometry import ReceiverGeometry,ReceiverGeometryConfig
from lighthit.receiver_first import first_entry_profile
from lighthit.receiver_first_fold import fold_first_entry,acceptance_moment_weights
from lighthit._receiver_geometry_fold import fold_coefficients


def test_exponential_polynomial_fold_and_positive_clock_ramps():
    paths=np.array([1.,1.2,2.,5.]);cdf=np.stack([paths,paths**2,paths**3],axis=-1)
    curve=PchipInterpolator(paths,cdf,axis=0);edges=np.arange(2.,26.,.3)
    for mu in [.0001,.2,10.]:
        for width in [0.,1e-7,.8,8.]:
            charge,value=fold_coefficients(paths,curve.c,mu,.02,.22,edges,width)
            density=lambda s:np.exp(-mu*s)*curve(s,1)*.02
            expected=quad_vec(density,paths[0],paths[-1],points=paths[1:-1],epsabs=1e-14,epsrel=1e-12)[0]
            np.testing.assert_allclose(charge,expected,rtol=2e-12,atol=1e-14)
            for i in [0,15,40,70]:
                lo=.22*(edges[i]-width/2);hi=.22*(edges[i]+width/2)
                weight=(lambda s:float(s<=hi)) if width==0 else (lambda s:np.clip((hi-s)/(.22*width),0.,1.))
                points=[s for s in [*paths[1:-1],lo,hi] if paths[0]<s<paths[-1]]
                ref=quad_vec(lambda s:density(s)*weight(s),paths[0],paths[-1],points=points,epsabs=1e-14)[0]
                np.testing.assert_allclose(value[i],ref,rtol=2e-10,atol=1e-14)


@pytest.fixture(scope='module')
def geometry():
    return ReceiverGeometry.build(radii_unit=np.array([14.,14.7,15.435]),g=.5,max_excess_unit=300.,
        config=ReceiverGeometryConfig(hit_nodes=17,miss_nodes=65,path_grid_nodes=513,path_nodes=256))


def test_geometry_fold_matches_direct_physical_paths_and_radius_scaling(geometry,tmp_path):
    R=.216;r=14.3*R;cap=np.arcsin(R/r);edges=np.arange(12.,60.,.5)
    angular=acceptance_moment_weights([.3,.4,np.sqrt(.75)],[1.,.2,.1])
    for theta in [0.,.7*cap,1.02*cap,.6,2.5]:
        extent=np.nextafter(np.sqrt((r-R)*(r+R)),np.inf) if theta<=cap else R*(14.3+300.)
        direct=first_entry_profile(SphereSurface(R),distance_m=r,direction_cosine=np.cos(theta),g=.5,
            max_path_m=extent,path_nodes=256)
        ref=fold_first_entry(direct,extinction_per_m=[.1],scattering_per_m=[.02],speed_m_per_ns=[.22],
            spectral_weights=[[1.,0.]],time_edges_ns=edges)
        q,f,tail=geometry.fold(r,theta,SphereSurface(R),extinction_per_m=.1,
            scattering_per_m=.02,speed_m_per_ns=.22,time_edges_ns=edges)
        target=ref.charge[0]@angular
        assert abs(q@angular/target-1)<.03
        assert np.sum(abs(np.diff(f@angular)-np.diff(ref.cumulative[:,0]@angular)))/target<.03
        assert (tail==0)==(theta<=cap)
    geometry.save(tmp_path);loaded=ReceiverGeometry.load(tmp_path)
    for a,b in zip(loaded.profile(r,.4,SphereSurface(R)),geometry.profile(r,.4,SphereSurface(R))):
        np.testing.assert_array_equal(a,b)
    # A geometrically similar receiver rescales paths by R and geometry by 1/R.
    p,a=geometry.profile(r,.4,SphereSurface(R));p2,a2=geometry.profile(2*r,.4,SphereSurface(2*R))
    np.testing.assert_array_equal(p2,2*p);np.testing.assert_array_equal(a2,a/2)


def test_parallel_geometry_uses_one_path_coordinate_for_all_blocks(tmp_path):
    cfg=ReceiverGeometryConfig(hit_nodes=3,miss_nodes=3,path_grid_nodes=17,path_nodes=16,path_order=4,scatter_order=8)
    options=dict(radii_unit=np.array([2.,2.1,2.2,2.3]),g=.5,max_excess_unit=20.,config=cfg)
    direct=ReceiverGeometry.build(**options)
    parallel=ReceiverGeometry.build_parallel(**options,output=tmp_path,jobs=2)
    for name in ('hit_cdf','hit_total','hit_entry','miss_cdf','miss_total'):
        np.testing.assert_array_equal(getattr(parallel,name),getattr(direct,name))


def test_compiled_scalar_pchip_matches_nonuniform_flat_and_turning_panels():
    from lighthit._receiver_geometry_lookup import _pchip
    path=np.array([1.,1.01,1.7,2.,5.,8.])
    for y in [np.array([0.,.3,.3,.35,.8,1.]),np.array([0.,.3,-.1,.35,.8,.6])]:
        actual=_pchip(path,y)[:,:,0]
        np.testing.assert_allclose(actual,PchipInterpolator(path,y).c,rtol=1e-12,atol=1e-12)


def test_compiled_partial_spectrum_matches_readable_moment_folds(geometry):
    R=.216;r=np.array([14.1,14.3,15.])*R
    angle=np.array([.01,.2,2.]);clock=np.array([.1,-.2,.3]);spread=np.array([0.,.8,.4])
    angular=acceptance_moment_weights([.3,.4,np.sqrt(.75)],[1.,.2,.1])
    moments=np.stack([np.array([[w],[-w]])*angular for w in [1.,.4,.2]])
    t=dict(mut=np.array([.1]),mus=np.array([.02]),speed=np.array([.22]),s0w=np.array([1.]),s2w=np.array([.5]))
    edges=np.arange(10.,70.,.5);q0=0.;f0=np.zeros(len(edges))
    for ri,ai,ci,wi,mi in zip(r,angle,clock,spread,moments):
        q,f,_=geometry.fold(ri,ai,SphereSurface(R),extinction_per_m=.1,scattering_per_m=.02,
            speed_m_per_ns=.22,time_edges_ns=edges-ci,clock_width_ns=wi)
        coef=mi[0]+.5*mi[1];q0+=q@coef;f0+=f@coef
    q,b,bound=geometry.sum_groups(r,angle,clock,spread,moments,edges,surface=SphereSurface(R),tables=t)
    assert abs(q/q0-1)<.001
    assert np.sum(abs(b-np.diff(f0)))/q0<.003
    assert bound>=0
    assert b.min()>-1e-13


def test_orientation_linear_panels_match_refined_moment_folds(geometry):
    from lighthit._receiver_geometry_lookup import sum_linear_pencils
    r=14.3*.216;theta=.2;edges=np.arange(10.,70.,.5)
    looks=np.array([[0.,0.,1.],[.3,.4,np.sqrt(.75)],[.8,.6,0.]])
    poly=np.array([1.,.2,.1,.03]);R=.216
    q,b=sum_linear_pencils(np.array([r]),np.array([theta]),np.array([.1]),np.array([1.]),
        np.array([[0.,0.,1.]]),np.array([[1.,0.,0.]]),looks,poly,edges,R,geometry.g,
        np.log(geometry.radii_unit*R),geometry.path_u_max,geometry.max_excess_unit,
        geometry.hit_cdf,geometry.hit_total,geometry.miss_cdf,geometry.miss_total,.1,.02,.22)
    refq,reff,_=geometry.fold(r,theta,SphereSurface(R),extinction_per_m=.1,scattering_per_m=.02,
        speed_m_per_ns=.22,time_edges_ns=edges-.1)
    for j,look in enumerate(looks):
        weight=acceptance_moment_weights(look,poly);expected=refq@weight
        assert abs(q[j]/expected-1)<.003
        assert abs(b[j]-np.diff(reff@weight)).sum()/expected<.005
        assert b[j].min()>-1e-14


def test_segment_cone_quadrature_preserves_mass_at_glass_grazing():
    from lighthit._receiver_segment_numba import segment_pencils
    axis=np.array([0.,0.,1.]);receiver=np.array([3.2,0.,0.]);cone=1/1.35
    root=3.2*cone/np.sqrt(1-cone*cone)
    ax,aw=np.polynomial.legendre.leggauss(3);gx,gw=np.polynomial.legendre.leggauss(4)
    rays=segment_pencils(receiver,np.array([0.,0.,-root-.5]),axis,1.,1.,cone,0.,3.4,0.,.216,.22,
        2.5,.3,1e-4,.5,1.,16,20000,ax,aw,gx,gw)
    assert np.sum(rays[3])==pytest.approx(1.,rel=3e-8)
    np.testing.assert_allclose(np.linalg.norm(rays[4],axis=1),1.,atol=1e-14)
    hit=rays[1]<np.arcsin(.216/rays[0])
    assert hit.any() and (~hit).any()
