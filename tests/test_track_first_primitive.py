"""Exact conservative query identities and explicit finite-window failures."""
import numpy as np
import pytest
from lighthit.experimental.track_first_primitive import TrackFirstPrimitive


def row():
    # Piecewise-uniform emitting source with deliberately different time laws.
    # This provides an independent analytic finite-rectangle integral.
    return TrackFirstPrimitive(np.array([-3.,-1.,2.,5.]),np.array([-2.,0.,3.,8.]),
        np.array([2.,7.,4.]),np.array([[.1,.4,.5],[.3,3.,1.],[.2,.8,2.]]),.25,{})


def rectangular_integral(r,lo,hi,edges):
    fraction=np.maximum(0,np.minimum(hi,r.u_edges_m[1:])-np.maximum(lo,r.u_edges_m[:-1]))/np.diff(r.u_edges_m)
    temporal=np.maximum(0,np.minimum(np.asarray(edges)[1:,None],r.tau_edges_ns[None,1:])-
                           np.maximum(np.asarray(edges)[:-1,None],r.tau_edges_ns[None,:-1]))/np.diff(r.tau_edges_ns)
    return fraction@r.charge_cells,temporal@(fraction@r.bins_cells)


def test_offgrid_rectangle_matches_independent_overlap_integral():
    r=row();edges=np.array([-1.7,.8,2.1,6.4])
    expected=rectangular_integral(r,-2.4,3.7,edges)
    actual=r.query(-2.4,3.7,edges)
    np.testing.assert_allclose(actual[0],expected[0],rtol=1e-14)
    np.testing.assert_allclose(actual[1],expected[1],rtol=1e-14)


def test_segment_translation_and_clock_use_correct_retarded_sign():
    r=row();tau=np.array([-1.,.5,4.,7.])
    for z,t0 in [(3.,12.),(2.,-9.)]:
        length=z+2.5
        absolute=tau+t0+z/r.speed_m_per_ns
        q,b=r.segment(z,length,t0,absolute,coefficient0_per_m=4.)
        qr,br=rectangular_integral(r,-2.5,z,tau)
        np.testing.assert_allclose(q,4*qr)
        np.testing.assert_allclose(b,4*br,atol=1e-14)


def test_arbitrary_partition_is_additive_and_empty_interval_zero():
    r=row();t=np.array([-2.,1.,5.,8.])
    q,b=r.query(-3.,5.,t)
    q0,b0=r.query(-3.,.13,t);q1,b1=r.query(.13,5.,t)
    np.testing.assert_allclose(q,q0+q1)
    np.testing.assert_allclose(b,b0+b1)
    qz,bz=r.query(.23,.23,t)
    assert qz==0 and np.all(bz==0)
    assert q>b.sum()  # Late/out-of-window photons remain in all-time charge.


def test_unsupported_tail_and_time_are_not_silently_clipped():
    r=row()
    for lo,hi in [(-4.,3.),(-2.,6.),(3.,2.),(np.nan,3.)]:
        with pytest.raises(ValueError,match='source interval'):
            r.query(lo,hi,[-1.,2.])
    for edges in [[-3.,2.],[-1.,9.],[2.,1.],[0.,np.inf]]:
        with pytest.raises(ValueError,match='time edges'):
            r.query(-2.,3.,edges)


def test_nonphysical_cells_are_rejected():
    r=row()
    for bins in [-r.bins_cells,r.bins_cells*100]:
        with pytest.raises(ValueError):
            TrackFirstPrimitive(r.u_edges_m,r.tau_edges_ns,r.charge_cells,bins,.25,{})


def toy_atlas():
    from lighthit.experimental.track_first_primitive import TrackFirstAtlas,orientation_nodes
    _,powers,_=orientation_nodes()
    radius=np.array([2.,4.]);x=np.array([-2.,-1.,0.,2.]);t=np.array([-1.,0.,1.,3.])
    amplitude=np.array([1.,.1,.2,0.,.1,0.,0.,0.,0.,0.])
    spatial=x-x[0];cdf=np.array([0.,.1,.5,.8])
    q=np.exp(-.1*radius[:,None,None])*spatial[None,:,None]*amplitude
    right=q[:,-1:, :]-q
    c=q[:,:,None,:]*cdf[None,None,:,None]
    cr=right[:,:,None,:]*cdf[None,None,:,None]
    return TrackFirstAtlas(radius,x,t,powers,q,right,c,cr,{'beta':1.,'speed_m_per_ns':.3})


def test_orientation_basis_is_complete_well_conditioned_cubic():
    from lighthit.experimental.track_first_primitive import orientation_nodes
    nodes,powers,matrix=orientation_nodes()
    assert np.linalg.cond(matrix)<32
    np.testing.assert_allclose(np.linalg.norm(nodes,axis=1),1.)
    coefficients=np.arange(10)*.123
    recovered=np.linalg.solve(matrix,matrix@coefficients)
    looks=np.array([[.23,.61,np.sqrt(1-.23**2-.61**2)],[-.8,.6,0.]])
    design=looks[:,0,None]**powers[:,0]*looks[:,2,None]**powers[:,1]
    np.testing.assert_allclose(design@recovered,design@coefficients,atol=1e-14)


def test_atlas_joint_rho_orientation_time_interpolation_and_additivity(tmp_path):
    from lighthit.experimental.track_first_primitive import TrackFirstAtlas
    atlas=toy_atlas();rho=3.;look=np.array([.3,.4,np.sqrt(.75)])
    time=np.array([-.8,.2,1.8])*rho
    q,b=atlas.query(rho,look,-1.7*rho,1.1*rho,time)
    angular=1+.1*look[2]+.2*look[0]+.1*look[0]*look[2]
    expected_q=np.exp(-.3)*2.8*angular
    expected_b=expected_q*np.diff(np.interp(time/rho,atlas.time_over_rho_edges,[0.,.1,.5,.8]))
    np.testing.assert_allclose(q,expected_q,rtol=1e-14)
    np.testing.assert_allclose(b,expected_b,rtol=1e-14)
    q0,b0=atlas.query(rho,look,-1.7*rho,-.13*rho,time)
    q1,b1=atlas.query(rho,look,-.13*rho,1.1*rho,time)
    np.testing.assert_allclose(q,q0+q1,rtol=1e-14)
    np.testing.assert_allclose(b,b0+b1,rtol=1e-14)
    reflected=look.copy();reflected[1]*=-1
    qr,br=atlas.query(rho,reflected,-1.7*rho,1.1*rho,time)
    np.testing.assert_array_equal(b,br)
    assert qr==q
    path=tmp_path/'atlas.npz';atlas.save(path)
    loaded=TrackFirstAtlas.load(path)
    ql,bl=loaded.query(rho,look,-1.7*rho,1.1*rho,time)
    assert ql==q
    np.testing.assert_array_equal(b,bl)


def test_atlas_rejects_unsupported_geometry_and_negative_angular_reconstruction():
    from dataclasses import replace
    atlas=toy_atlas()
    for rho in [1.,5.]:
        with pytest.raises(ValueError,match='radius outside'):
            atlas.query(rho,[0.,0.,1.],-1.,1.,[-.5,.5])
    broken=replace(atlas,charge_left=-atlas.charge_left,charge_right=-atlas.charge_right)
    with pytest.raises(ValueError,match='negative physical'):
        broken.query(3.,[0.,0.,1.],-1.,1.,[-.5,.5])


def bound_case(start_z=-400.,length=100.):
    from types import SimpleNamespace
    from lighthit.sources import SpectralLightElements
    def sample(w):
        return {key:np.full_like(w,value) for key,value in dict(absorption_per_m=.05,
            scattering_per_m=.015,phase_index=1.34,group_index=1.37).items()}
    kernel=SimpleNamespace(medium=SimpleNamespace(g=.9,sample=sample),
        wavelength=SimpleNamespace(wavelength_nm=np.array([480.]),weight_nm=np.array([1.])),
        detector=SimpleNamespace(spectral_weight=lambda w:1.,effective_area_m2=np.array([1.])),
        acceptance=lambda:{'alpha':np.array([4*np.pi])})
    source=SpectralLightElements(np.array([[0.,0.,start_z]]),np.array([[0.,0.,1.]]),
        np.array([length]),np.array([length]),np.array([-length]),np.array([1.]),
        np.array([0.]),np.array([length/.299792458]),np.array([0]),np.array([0]),cone_model='spectral')
    return kernel,source


def test_far_segment_charge_bound_exceeds_independent_positive_ray_integral():
    from lighthit.experimental.track_first_primitive import first_scattering_segment_bounds
    kernel,source=bound_case();receiver=np.array([30.,0.,0.])
    bound,meta=first_scattering_segment_bounds(kernel,source,receiver)
    # Independent Gauss quadrature of source position, cone azimuth and complete
    # first flight. No cached prompt or atlas code enters this reference.
    x,w=np.polynomial.legendre.leggauss(16);positions=-400+(x+1)*50;sw=w*50
    phi=(np.arange(96)+.5)*2*np.pi/96;cone=1/1.34
    directions=np.column_stack((np.sqrt(1-cone**2)*np.cos(phi),np.sqrt(1-cone**2)*np.sin(phi),np.full(len(phi),cone)))
    y,v=np.polynomial.legendre.leggauss(128);t=(y+1)/2
    ell=30*t/(1-t);lw=v/2*30/(1-t)**2
    result=0.
    for z,weight in zip(positions,sw):
        R=np.array([30.,0.,-z]);last=R-directions[:,None,:]*ell[None,:,None]
        distance=np.linalg.norm(last,axis=2)
        cosine=np.sum(last*directions[:,None,:],axis=2)/distance
        phase=(1-.9**2)/(4*np.pi*(1+.9**2-2*.9*cosine)**1.5)
        integral=.015*np.exp(-.065*(ell[None,:]+distance))*phase/distance**2
        result+=weight*np.mean(integral@lw)
    result*=(1-1/1.34**2)/480**2
    assert 0 < result < bound[0]
    assert meta['continuum_spectral_error_included'] is False


def test_cone_intersection_is_unsupported_by_far_bound_instead_of_pruned():
    from lighthit.experimental.track_first_primitive import first_scattering_segment_bounds
    kernel,source=bound_case(-40.,20.)
    bound,meta=first_scattering_segment_bounds(kernel,source,np.array([30.,0.,0.]))
    assert np.isinf(bound[0])
    assert meta['unsupported_pieces']==1
