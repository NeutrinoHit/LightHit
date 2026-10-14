"""Universal optics-only Q1 atlas against independent physical-time CDFs."""
import numpy as np
import pytest
from lighthit.experimental.joint_first import PencilFirstCDF
from lighthit.experimental.joint_first_atlas import UniversalFirstCDFAtlas,evaluate_photon_histogram_atlas

POLY=np.array([.3082,.54192,.19831,-.04912])


@pytest.fixture(scope='module')
def atlas():return UniversalFirstCDFAtlas.build()


@pytest.mark.parametrize('look',[[.8,0.,-.6],[.6,.8,0.],[0.,0.,1.],[0.,0.,-1.]])
def test_withheld_geometry_charge_and_physical_time(atlas,look):
    angles=np.array([1.1e-4,.0021,.031,.117,.213,.61,1.2,2.03,2.83,np.pi-.0007])
    radius=np.array([7.,11.,23.,50.,87.,137.,230.,57.,183.,97.])
    delta=np.c_[radius*np.sin(angles),np.zeros(len(radius)),radius*np.cos(angles)]
    directions=np.tile([0.,0.,1.],(len(radius),1))
    prepared=atlas.prepare(delta,directions,np.asarray(look),POLY,
        scattering_per_m=.015,absorption_per_m=.05,g=.9,speed_m_per_ns=.22)
    for i in range(len(radius)):
        ref=PencilFirstCDF.build(delta[i],directions[i],look,POLY,
            scattering_per_m=.015,absorption_per_m=.05,g=.9,speed_m_per_ns=.22,path_step=.0075)
        edges=np.r_[np.linspace(-5.,30.,71),np.geomspace(31.,2000.,150)]+radius[i]/.22
        q,b=prepared.contract(np.array([i]),[0.],[1.],edges)
        qr,br=ref.contract([0.],[1.],edges)
        assert abs(q-qr)/qr<.003
        assert abs(b-br).sum()/abs(br).sum()<.005
        assert np.maximum(-b,0.).sum()/qr<1e-6


def test_clocks_charge_causality_and_band_scaling(atlas):
    delta=np.array([[30.,0.,40.]]);directions=np.array([[0.,0.,1.]])
    response=atlas.prepare(delta,directions,np.array([.8,0.,-.6]),POLY,
        scattering_per_m=.015,absorption_per_m=.05,g=.9,speed_m_per_ns=.22)
    edges=np.arange(-5.,806.,5.);ids=np.array([0,0]);times=np.array([0.,25.]);weights=np.array([3.,4.])
    q,b=response.contract(ids,times,weights,edges)
    late,empty=response.contract(ids,times+1e6,weights,edges)
    assert late==q;np.testing.assert_array_equal(empty,0.)
    shifted,sbins=response.contract(ids,times+1e6,weights,edges,origin_ns=1e6)
    assert shifted==q;np.testing.assert_allclose(sbins,b,rtol=2e-10,atol=q*1e-13)
    np.testing.assert_array_equal(b[edges[1:]<=50/.22],0.)
    altered=atlas.prepare(delta,directions,np.array([.8,0.,-.6]),POLY,
        scattering_per_m=.03,absorption_per_m=.035,g=.9,speed_m_per_ns=.22)
    sq,sb=altered.contract(ids,times,weights,edges)
    np.testing.assert_allclose(sq,2*q,rtol=1e-13);np.testing.assert_allclose(sb,2*b,rtol=1e-13)


def test_out_of_domain_fallback_and_optical_contract(atlas):
    delta=np.array([[3.,0.,4.]])*1000
    response=atlas.prepare(delta,np.array([[0.,0.,1.]]),np.array([.8,0.,-.6]),POLY,
        scattering_per_m=.015,absorption_per_m=.05,g=.9,speed_m_per_ns=.22)
    assert list(response.fallback)==[0]
    with pytest.raises(ValueError,match='outside atlas'):
        atlas.prepare(delta,np.array([[0.,0.,1.]]),np.array([.8,0.,-.6]),POLY,
            scattering_per_m=.015,absorption_per_m=.05,g=.9,speed_m_per_ns=.22,direct_fallback=False)
    with pytest.raises(ValueError,match='contract mismatch'):
        atlas.prepare(delta,np.array([[0.,0.,1.]]),np.array([.8,0.,-.6]),POLY,
            scattering_per_m=.015,absorption_per_m=.05,g=.8,speed_m_per_ns=.22)


def test_persisted_atlas_has_no_event_inputs(atlas,tmp_path):
    path=tmp_path/'atlas.npz';atlas.save(path);loaded=UniversalFirstCDFAtlas.load(path)
    assert loaded.g==atlas.g;assert loaded.degree==3
    np.testing.assert_array_equal(loaded.modes,atlas.modes)
    with np.load(path) as z:
        assert set(z.files)=={'metadata','log_kappa','log_tangent','u_grid','powers','modes'}
        assert 'source_or_geometry_stored' in str(z['metadata'])


def test_stationary_wrapper_preserves_charge_without_cdf_or_clock(atlas):
    from types import SimpleNamespace
    from lighthit.model import SpectralMedium,DetectorArray
    from lighthit.transport import TransportKernel
    from lighthit import KernelConfig
    medium=SpectralMedium(np.array([350.,610.]),np.full(2,.05),np.full(2,.015),
        np.full(2,1.34),np.full(2,1.37),g=.9)
    look=np.array([.8,0.,-.6]);delta=np.array([40*np.sin(.6),0.,40*np.cos(.6)])
    detector=DetectorArray(delta[None,:],-look[None,:],1.,lambda x:1+.2*x,lambda w:np.ones_like(w))
    kernel=TransportKernel(medium,detector,KernelConfig(omega_per_ns=np.array([0.]),cache_policy='require'))
    hist=SimpleNamespace(wavelength_nm=np.array([480.]),geometry_id=np.array([0,0]),
        position=np.zeros((1,3)),direction=np.array([[0.,0.,1.]]),
        emission_times_ns=np.array([0.,25.]),photon_weights=np.array([[3.,4.]]))
    edges=np.arange(-5.,806.,5.);om=np.array([0]);origin=np.array([0.])
    q,b,meta=evaluate_photon_histogram_atlas(atlas,kernel,hist,om,edges,origin,wavelength_weight_nm=2.,source_guard_factor=0.)
    stationary,empty,sm=evaluate_photon_histogram_atlas(atlas,kernel,hist,om,edges,origin,
        wavelength_weight_nm=2.,compute_bins=False,source_guard_factor=0.)
    np.testing.assert_allclose(q,stationary,rtol=1e-13)
    assert empty.shape==(1,0);assert sm['prepare_cpu_s']==0.;assert not sm['bins_computed']
    hist.emission_times_ns+=1e6
    delayed,_,_=evaluate_photon_histogram_atlas(atlas,kernel,hist,om,edges,origin,
        wavelength_weight_nm=2.,compute_bins=False,source_guard_factor=0.)
    np.testing.assert_array_equal(delayed,stationary)


def test_source_guard_restores_members_without_double_counting(atlas):
    from types import SimpleNamespace
    from lighthit.model import SpectralMedium,DetectorArray
    from lighthit.transport import TransportKernel
    from lighthit import KernelConfig
    from lighthit.prompt import acceptance_polynomial,acceptance_monomials,C_M_PER_NS
    medium=SpectralMedium(np.array([350.,610.]),np.full(2,.05),np.full(2,.015),
        np.full(2,1.34),np.full(2,1.37),g=.9)
    look=np.array([.8,0.,-.6]);delta=40*np.array([np.sin(.04),0.,np.cos(.04)])
    detector=DetectorArray(delta[None,:],-look[None,:],1.,lambda x:1+.2*x,lambda w:np.ones_like(w))
    kernel=TransportKernel(medium,detector,KernelConfig(omega_per_ns=np.array([0.]),cache_policy='require'))
    directions=np.array([[np.sin(.05),0.,np.cos(.05)],[-np.sin(.05),0.,np.cos(.05)]])
    weights=np.array([3.,4.]);mean=np.average(directions,axis=0,weights=weights)
    covariance=np.einsum('i,ij,ik->jk',weights/weights.sum(),directions-mean,directions-mean)
    hist=SimpleNamespace(wavelength_nm=np.array([480.]),geometry_id=np.array([0,0]),position=np.zeros((1,3)),
        direction=(mean/np.linalg.norm(mean))[None,:],emission_times_ns=np.array([0.,25.]),
        photon_weights=weights[None,:],geometry_position_covariance_m2=np.zeros((1,3,3)),
        geometry_direction_covariance=covariance[None,:,:],original_geometry_id=np.array([0,0]),
        original_position=np.zeros((2,3)),original_direction=directions,original_times_ns=np.array([0.,25.]),
        original_photons_per_nm=weights)
    edges=np.arange(-5.,806.,5.);om=np.array([0]);origin=np.array([0.])
    q,b,meta=evaluate_photon_histogram_atlas(atlas,kernel,hist,om,edges,origin,wavelength_weight_nm=2.,raw_chunk=1)
    poly=acceptance_polynomial(*acceptance_monomials(kernel.acceptance()['alpha']))
    qr=0.;br=np.zeros(len(edges)-1)
    for i in range(2):
        ref=PencilFirstCDF.build(delta,directions[i],look,poly,scattering_per_m=.015,
            absorption_per_m=.05,g=.9,speed_m_per_ns=C_M_PER_NS/1.37,path_step=.0075)
        qq,bb=ref.contract([hist.original_times_ns[i]],[weights[i]*2.],edges);qr+=qq;br+=bb
    assert abs(q[0]-qr)/qr<.003;assert abs(b[0]-br).sum()/abs(br).sum()<.005
    assert meta['source_guard_fallback_geometries']==1;assert meta['source_guard_fallback_members']==2
    assert meta['source_guard_charge_fraction']==1.
    omitted,zero,pruned_meta=evaluate_photon_histogram_atlas(atlas,kernel,hist,om,edges,origin,
        wavelength_weight_nm=2.,bins_tolerance_pe=10.)
    np.testing.assert_array_equal(omitted,q);np.testing.assert_array_equal(zero,0.)
    assert pruned_meta['skipped_budget_members']==2;assert pruned_meta['time_cdf_raw_members']==0
    assert abs(b).sum()<=pruned_meta['omitted_bins_l1_bound_pe']<=10.
    hist.emission_times_ns+=1e6;hist.original_times_ns+=1e6
    late,empty,lm=evaluate_photon_histogram_atlas(atlas,kernel,hist,np.array([0]),edges,origin,wavelength_weight_nm=2.)
    np.testing.assert_array_equal(late,q);np.testing.assert_array_equal(empty,0.)
    assert lm['skipped_clock_members']==2;assert lm['prepare_cpu_s']==0.


def _pruning_case():
    from types import SimpleNamespace
    from lighthit.model import SpectralMedium,DetectorArray
    from lighthit.transport import TransportKernel
    from lighthit import KernelConfig
    medium=SpectralMedium(np.array([350.,610.]),np.full(2,.05),np.full(2,.015),
        np.full(2,1.34),np.full(2,1.37),g=.9)
    look=np.array([.8,0.,-.6])
    detector=DetectorArray(np.array([[30.,0.,40.]]),-look[None,:],1.,lambda x:1+.2*x,lambda w:np.ones_like(w))
    kernel=TransportKernel(medium,detector,KernelConfig(omega_per_ns=np.array([0.]),cache_policy='require'))
    hist=SimpleNamespace(wavelength_nm=np.array([480.]),geometry_id=np.array([0,1,2]),
        position=np.array([[0.,0.,0.],[-350.,0.,0.],[0.,0.,-10.]]),direction=np.tile([0.,0.,1.],(3,1)),
        emission_times_ns=np.array([0.,-1500.,1e6]),photon_weights=np.array([[3.,.1,4.]]))
    return kernel,hist,np.array([0]),np.arange(-5.,806.,5.),np.zeros(1)


def test_bin_budget_and_clock_gate_preserve_all_stationary_charge(atlas):
    k,h,ids,e,o=_pruning_case()
    q,b,full=evaluate_photon_histogram_atlas(atlas,k,h,ids,e,o,source_guard_factor=0.)
    pq,pb,meta=evaluate_photon_histogram_atlas(atlas,k,h,ids,e,o,source_guard_factor=0.,bins_tolerance_pe=1e-9)
    np.testing.assert_array_equal(pq,q)
    assert abs(b-pb).sum()<=meta['omitted_bins_l1_bound_pe']*(1+1e-6)+1e-30
    assert 0<meta['omitted_bins_l1_bound_pe']<=1e-9
    assert meta['skipped_budget_geometries']==1
    assert meta['skipped_clock_geometries']==1
    assert meta['time_cdf_geometries']==1
    assert full['skipped_clock_geometries']==1
    assert q[0]>0.


def test_tiny_packet_skips_every_cdf_but_not_charge(atlas):
    k,h,ids,e,o=_pruning_case();h.photon_weights*=1e-40
    q,b,meta=evaluate_photon_histogram_atlas(atlas,k,h,ids,e,o,source_guard_factor=0.,bins_tolerance_pe=1e-10)
    assert q[0]>0.;np.testing.assert_array_equal(b,0.)
    assert meta['prepare_cpu_s']==0.;assert meta['time_cdf_geometries']==0
    assert meta['profile_skipped_oms']==[0]
    assert meta['omitted_bins_l1_bound_pe']<1e-10


def test_pruning_requires_positive_source_and_valid_budget(atlas):
    k,h,ids,e,o=_pruning_case();h.photon_weights[0,0]=-3.
    with pytest.raises(ValueError,match='positive photon'):
        evaluate_photon_histogram_atlas(atlas,k,h,ids,e,o,source_guard_factor=0.,bins_tolerance_pe=1e-9)
    with pytest.raises(ValueError,match='bins_tolerance_pe'):
        evaluate_photon_histogram_atlas(atlas,k,h,ids,e,o,source_guard_factor=0.,bins_tolerance_pe=-1.)


def test_fast_stationary_outside_atlas_matches_physical_cdf(atlas):
    delta=np.array([[300.,0.,1900.]])
    direction=np.array([[0.,0.,1.]]);look=np.array([.8,0.,-.6])
    q,supported=atlas.charge_coefficients(delta,direction,look,POLY,
        scattering_per_m=.015,absorption_per_m=.05,g=.9)
    assert not supported[0]
    ref=PencilFirstCDF.build(delta[0],direction[0],look,POLY,
        scattering_per_m=.015,absorption_per_m=.05,g=.9,speed_m_per_ns=.22,path_step=.003)
    np.testing.assert_allclose(q[0],ref.charge_per_m2,rtol=2e-7,atol=0.)
