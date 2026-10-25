"""Finite Q1 keeps fused spectra, partial spectra, source mass and screening."""
from dataclasses import replace
import numpy as np
import pytest
import lighthit as lh
from lighthit.receiver import SphereSurface
from lighthit.receiver_first_atlas import FirstEntryAtlas,FirstEntryAtlasConfig
from lighthit.receiver_first_adaptive import AdaptiveFirstEntry
from lighthit.receiver_geometry import ReceiverGeometry,ReceiverGeometryConfig
from lighthit.prompt import PromptConfig,_spectral_tables
from test_prompt import config,spectral_medium,detector


@pytest.fixture(scope='module')
def kernel(tmp_path_factory):
    root=tmp_path_factory.mktemp('finite-prompt')
    d=replace(detector(count=2),positions_m=np.array([[0.,0.,3.2],[3.2,0.,0.]]),receiver_surface=SphereSurface(.216))
    d.spectral_efficiency.wavelength_knots_nm=np.empty(0)
    k=lh.TransportKernel(spectral_medium(),d,config(root/'cache',scattered_receiver='first-entry',
        first_entry_tables=root,relative_time_edges_ns=np.arange(-10.,80.,2.)))
    radii=np.array([3.,3.15,3.3075,3.472875]);tables=_spectral_tables(k)
    fine=FirstEntryAtlas.build(radii_m=radii,surface=SphereSurface(.216),g=k.medium.g,tables=tables,
        config=FirstEntryAtlasConfig(hit_nodes=9,miss_nodes=33,time_nodes=65,path_nodes=64))
    fine.save(root/'q1-reference');AdaptiveFirstEntry.of(fine).save(root/'q1-adaptive')
    geometry=ReceiverGeometry.build(radii_unit=np.geomspace(3.,6.,16)/.216,g=k.medium.g,max_excess_unit=700.,
        config=ReceiverGeometryConfig(hit_nodes=9,miss_nodes=33,path_grid_nodes=129,path_nodes=64))
    geometry.save(root/'geometry')
    from lighthit.receiver_first_spectral import SpectralFirstEntry,SpectralFirstEntryConfig
    SpectralFirstEntry.build(geometry=geometry,radius_m=.216,tables=tables,
        config=SpectralFirstEntryConfig(33)).save(root/'q1-bands')
    return k


def _source(beta=1.,scale=1.):
    return lh.SpectralLightElements(np.array([[-.01,0.,0.]]),np.array([[0.,0.,1.]]),
        np.array([.02]),np.array([scale]),np.array([-scale/beta**2]),np.array([beta]),
        np.array([0.]),np.array([.1]),np.array([0]),np.array([0]),cone_model='spectral')


@pytest.mark.parametrize('single_band',[False,True])
@pytest.mark.parametrize('threads',[1,2])
def test_batched_screen_matches_individual_coarse_queries(kernel,single_band,threads,monkeypatch):
    from lighthit import prompt
    monkeypatch.setattr(prompt,'_thread_count',lambda:threads)
    from lighthit.receiver_prompt import FirstEntryPrompt
    from lighthit.prompt import _element_segments
    from lighthit._prompt_screen import compress_screen
    from lighthit._prompt_numba import _prepare_pencil_shape
    tables=_spectral_tables(kernel)
    source=replace(_source(),cone_model='frozen',reference_phase_index=float(np.mean(tables['phase_index'])))
    config=PromptConfig(pixel_face=4,ring_nodes=16,screen_pixel_face=2,screen_ring_nodes=16)
    pencils=compress_screen(source,_element_segments(source),config)
    pencils['_first_entry_shape']=_prepare_pencil_shape(pencils['direction'],pencils['major'],pencils['pos_cov'])
    if single_band:tables={key:value[:1] for key,value in tables.items()}
    backend=FirstEntryPrompt(kernel);positions=kernel.detector.positions_m;looks=-kernel.detector.orientations
    origins=np.array([.1,-.3]);polynomial=np.array([.3,.5,.2,-.05]);targets=np.array([1,0])
    actual,counts,_=backend.screen(pencils,positions,looks,origins,targets,polynomial,config,tables)
    for j,om in enumerate(targets):
        expected,_,metadata=backend.evaluate(pencils,positions[om],looks[om],origins[om],
            polynomial,config,tables,np.empty(0),coarse=True)
        assert actual[j]==pytest.approx(expected,rel=2e-14)
        assert counts[j]==metadata['groups']


@pytest.mark.parametrize('beta',[1.,.745])
def test_prompt_preserves_emission_linearity_for_common_and_partial_spectra(kernel,beta,monkeypatch):
    from lighthit import _prompt_numba
    def centre(*args,**kwargs):raise AssertionError('centre Q1 must not run')
    monkeypatch.setattr(_prompt_numba,'grouped_pencils_order1',centre)
    cfg=PromptConfig(element_method='grouped',fuse_bands=True,screen=False,pixel_face=4,ring_nodes=16,
        fused_pixel_face=4,fused_ring_nodes=16)
    first=kernel.transport_prompt(_source(beta),config=cfg)
    second=kernel.transport_prompt(_source(beta,2.),config=cfg)
    assert np.all(first.charge_orders_pe[:,1]>0)
    np.testing.assert_allclose(second.charge_orders_pe[:,1],2*first.charge_orders_pe[:,1],rtol=2e-12)
    np.testing.assert_allclose(second.bins_orders_pe[:,:,1],2*first.bins_orders_pe[:,:,1],rtol=2e-10,atol=1e-16)
    assert np.all(first.bins_orders_pe[:,:,1]>=-1e-14)
    if beta<1:
        details=[m for unit in first.metadata['order1_by_wavelength']
                 for m in unit['order1_surface_numerics'].values()]
        assert all(m.get('prepared_band') for m in details)
        # A previous event's bright common spectrum cannot subsidize this one.
        for j in range(len(first.detector)):
            budget=sum(unit['order1_surface_numerics'][str(j)]['bin_l1_budget_pe']
                       for unit in first.metadata['order1_by_wavelength'])
            assert budget==pytest.approx(.003*first.charge_orders_pe[j,1],rel=1e-12)


def test_shared_bin_budget_covers_all_spectral_units(kernel):
    a,b=_source(),_source(.745,1e-5)
    keys=('start_m','direction','length_m','coefficient0','coefficient2','beta','start_ns','end_ns','row_index','uid')
    source=lh.SpectralLightElements(**{key:np.concatenate((getattr(a,key),getattr(b,key))) for key in keys},cone_model='spectral')
    cfg=PromptConfig(element_method='grouped',fuse_bands=True,screen=False,pixel_face=4,ring_nodes=16,
        fused_pixel_face=4,fused_ring_nodes=16)
    response=kernel.transport_prompt(source,config=cfg)
    for j in range(len(response.detector)):
        bound=sum(unit['order1_surface_numerics'][str(j)]['bin_l1_bound_pe']
                  for unit in response.metadata['order1_by_wavelength'])
        assert bound<=.003*response.charge_orders_pe[j,1]+1e-15


def test_continuous_primary_rows_keep_glass_and_exact_cubic_orientation(kernel):
    from lighthit.muon.track_first_primitive import build_track_first_rows,orientation_nodes
    nodes,powers,matrix=orientation_nodes();look=np.array([.23,.61,np.sqrt(1-.23**2-.61**2)])
    kwargs=dict(rho_m=3.2,beta=1.,speed_m_per_ns=.299792458,u_edges_m=[3.2,3.6],tau_edges_ns=np.arange(0.,81.,1.))
    rows=build_track_first_rows(kernel,orientations=nodes,**kwargs)
    direct=build_track_first_rows(kernel,orientations=look[None,:],**kwargs)[0]
    monomial=look[0]**powers[:,0]*look[2]**powers[:,1]
    inverse=np.linalg.inv(matrix)
    q=monomial@inverse@np.array([r.charge_cells[0] for r in rows])
    b=monomial@inverse@np.stack([r.bins_cells[0] for r in rows])
    np.testing.assert_allclose(q,direct.charge_cells[0],rtol=1e-11)
    np.testing.assert_allclose(b,direct.bins_cells[0],rtol=2e-10,atol=1e-20)
    assert direct.metadata['receiver_contract']['radius_m']==.216


def test_parallel_primary_cells_preserve_rows(kernel):
    from lighthit.muon.track_first_primitive import build_track_first_rows,orientation_nodes
    looks,_,_=orientation_nodes()
    options=dict(rho_m=3.2,orientations=looks,beta=1.,speed_m_per_ns=.299792458,
                 u_edges_m=[-.4,.1,.5],tau_edges_ns=np.arange(0.,81.,2.))
    direct=build_track_first_rows(kernel,**options)
    completed=[]
    parallel=build_track_first_rows(kernel,workers=2,progress=lambda n,total:completed.append((n,total)),**options)
    assert completed==[(1,2),(2,2)]
    for a,b in zip(direct,parallel):
        np.testing.assert_array_equal(a.charge_cells,b.charge_cells)
        np.testing.assert_array_equal(a.bins_cells,b.bins_cells)
        assert b.metadata['build_workers']==2


@pytest.mark.parametrize('beta',[1.,.745])
def test_surface_cone_preview_matches_dense_pencil_average(kernel,beta):
    from lighthit.muon._surface_preview import cone_first_moments
    from lighthit.receiver_prompt import FirstEntryPrompt
    from lighthit.receiver_first_fold import acceptance_moment_weights
    from lighthit._receiver_band_first import charge_only
    from lighthit.muon.angular_preview import cubic_response
    data=FirstEntryPrompt(kernel).spectral;t=_spectral_tables(kernel)
    rho=2.1;z=np.sqrt(3.2**2-rho*rho);gx,gw=np.polynomial.legendre.leggauss(4)
    args=(rho,np.array([z]),beta,.216,kernel.medium.g,np.log(data.radii_m),data.hit_charge,data.miss_charge,
          t['mut'],t['mus'],t['speed'],t['s0w'],t['s2w'],t['phase_index'],gx,gw)
    raw=cone_first_moments(*args,.5)[0]
    poly=np.array([.3,.5,.2,-.05]);r=3.2;h=np.array([rho/r,0.,z/r]);n=8192
    phi=2*np.pi*(np.arange(n)+.5)/n
    for look in [np.array([0.,0.,1.]),np.array([.8,0.,.6]),np.array([-.3,.4,np.sqrt(.75)])]:
        reference=0.
        for band,phase in enumerate(t['phase_index']):
            if beta*phase<=1:continue
            c=1/(beta*phase);s=np.sqrt(1-c*c)
            direction=np.column_stack((s*np.cos(phi),s*np.sin(phi),np.full(n,c)))
            cosine=direction@h;x=h-cosine[:,None]*direction;x/=np.linalg.norm(x,axis=1)[:,None]
            y=np.cross(direction,x);angles=np.arccos(np.clip(cosine,-1.,1.))
            local=np.column_stack((x@look,y@look,direction@look))
            weights=np.stack([acceptance_moment_weights(v,poly) for v in local])/n
            moments=np.stack((weights,-weights/beta**2),axis=1)
            reference+=charge_only(np.full(n,r),angles,moments,.216,kernel.medium.g,np.log(data.radii_m),
                t['mut'][band],t['mus'][band],t['speed'][band],t['s0w'][band],t['s2w'][band],
                data.hit_charge[band],data.miss_charge[band]).sum()
        actual=cubic_response(raw,look[0],look[2],look[1]**2,poly)
        assert actual==pytest.approx(reference,rel=.003)


def test_first_entry_charge_only_does_not_use_emission_clocks(kernel):
    from lighthit.muon.first_charge_only import first_scattering_charge_only
    source=_source();changed=replace(source,start_ns=source.start_ns+1e6,end_ns=source.end_ns+3e6)
    a,metadata=first_scattering_charge_only(kernel,source)
    b,_=first_scattering_charge_only(kernel,changed)
    np.testing.assert_array_equal(a,b)
    assert metadata['computed_bins'] is False and np.all(a>0)


def test_spherical_primary_tail_bound_exceeds_actual_first_entry_charge(kernel):
    from lighthit.muon.track_first_bounds import combined_first_scattering_segment_bounds
    from lighthit.muon.first_charge_only import first_scattering_charge_only
    for beta in [1.,.745]:
        source=_source(beta);actual,_=first_scattering_charge_only(kernel,source)
        for om,position in enumerate(kernel.detector.positions_m):
            bound,metadata=combined_first_scattering_segment_bounds(kernel,source,position,
                effective_area_m2=kernel.detector.effective_area_m2[om])
            assert bound.sum()>=actual[om]
            assert metadata['ray']['receiver_surface_factor']==4.
            assert metadata['infinite_line']['receiver_radius_m']==pytest.approx(.216)


def test_spherical_packet_bound_exceeds_actual_first_entry_charge(kernel):
    from lighthit.muon.packet_skip import packet_first_bounds,bound_constants
    from lighthit.muon.first_charge_only import first_scattering_charge_only
    for beta in [1.,.745]:
        source=_source(beta)
        bound=packet_first_bounds(source,kernel.detector.positions_m,kernel.detector.effective_area_m2,
            _spectral_tables(kernel),bound_constants(kernel),receiver_radius_m=.216,azimuth_nodes=16).sum(0)
        actual,_=first_scattering_charge_only(kernel,source)
        assert np.all(bound>=actual)


def test_secondary_packets_keep_independent_time_error_budgets(kernel):
    from lighthit.muon.secondary_first import evaluate_secondary_first
    from lighthit.muon.muon_packets import SourcePacket
    a,b=_source(),_source(.745,1e-5)
    keys=('start_m','direction','length_m','coefficient0','coefficient2','beta','start_ns','end_ns','row_index','uid')
    source=lh.SpectralLightElements(**{key:np.concatenate((getattr(a,key),getattr(b,key))) for key in keys},cone_model='spectral')
    packets=[SourcePacket('shower',np.array([0]),1),SourcePacket('general',np.array([1]),2)]
    cfg=PromptConfig(element_method='grouped',screen=False,pixel_face=4,ring_nodes=16)
    q,bins,meta=evaluate_secondary_first(kernel,source,packets,np.arange(2),np.zeros(2),config=cfg,
        fuse_bands=True,fused_pixel_face=4,fused_ring_nodes=16,skip_epsilon=.001)
    partial=[row for row in meta['packets'] if row['packet']==1][0]
    for entry in partial['wavelengths']:
        for j,numerics in entry['prompt_metadata']['order1_surface_numerics'].items():
            own_charge=meta['wavelengths'][0]['charge_pe_per_om'][int(j)]
            assert numerics['bin_l1_budget_pe']==pytest.approx(.003*own_charge,rel=1e-12)
    assert np.all(q>0) and np.isfinite(bins).all()
