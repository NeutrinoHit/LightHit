"""Routing/selection controls for the universal primary assembler."""
from types import SimpleNamespace
from dataclasses import replace
import numpy as np
import pytest

pytest.importorskip('numba')
import lighthit as lh
import lighthit.experimental.universal_primary as transport
from lighthit.experimental.track_ge2_primitive import track_ge2_optics_metadata


def setup():
    m=lh.SpectralMedium([350.,610.],[.05,.05],[0.,0.],[1.35,1.35],[1.4,1.4])
    def efficiency(w):return np.ones_like(w)*.2
    efficiency.wavelength_knots_nm=np.array([350.,610.])
    d=lh.DetectorArray([[10.,0.,25.]],[[0.,0.,-1.]],.05,lambda x:np.ones_like(x),efficiency)
    k=lh.TransportKernel(m,d,lh.KernelConfig(omega_per_ns=np.linspace(0,1.2,65),threshold_pe=.01))
    s=lh.SpectralLightElements(np.zeros((1,3)),np.array([[0.,0.,1.]]),np.array([50.]),
        np.array([1e8]),np.array([-1e8]),np.ones(1),np.zeros(1),np.array([50/.299792458]),
        np.zeros(1,int),np.zeros(1,int),cone_model='spectral')
    return k,s


class First:
    radii_m=np.array([3.,300.]);x_edges=np.array([-8.5,8.5])
    metadata={'speed_m_per_ns':.299792458};resident_bytes=0;time_row_loads=0
    def __init__(self,value):self.value=value;self.time_calls=0
    def validate_kernel(self,k):return True
    def query_charge_many(self,rho,*a,**kw):return np.full(len(rho),self.value)
    def query(self,rho,look,lo,hi,t,**kw):
        self.time_calls+=1
        return self.value,np.full(len(t)-1,self.value/(len(t)-1))


class Ge2:
    impact_nodes_m=np.array([3.01,300.]);omega_per_ns=np.linspace(0,1.2,65)
    resident_bytes=0;full_row_loads=0
    def __init__(self,k,value):
        self.metadata=track_ge2_optics_metadata(k);self.value=value;self.time_calls=0
    def query_supported(self,*a,**kw):return True
    def segment_charge_many(self,rho,*a,**kw):return np.full(len(rho),self.value)
    def segment(self,*a,**kw):
        self.time_calls+=1
        return self.value*np.exp(-self.omega_per_ns)


@pytest.mark.parametrize('level,expected_active',[(.2,True),(0.,False)])
def test_charge_then_selected_time_profiles(monkeypatch,level,expected_active):
    k,s=setup();first=First(level);ge2=Ge2(k,level)
    def zero(k,s,**kw):return SimpleNamespace(charge_pe=np.zeros(1),bins_pe=np.zeros((1,len(k.config.relative_time_edges_ns)-1)))
    monkeypatch.setattr(transport,'ballistic_interval',zero)
    r=transport.transport_universal_primary(k,s,first,ge2)
    np.testing.assert_allclose(r.charge_orders_pe,[[0.,level,level]])
    assert r.active.tolist()==[expected_active]
    assert first.time_calls==int(expected_active) and ge2.time_calls==int(expected_active)
    assert r.metadata['q1_fallback_oms']==0 and r.metadata['ge2_fallback_pairs']==0
    assert r.metadata['secondaries_computed'] is False
    if not expected_active:np.testing.assert_array_equal(r.bins_orders_pe,0.)


def test_optical_mismatch_and_missing_primary_rejected():
    k,s=setup();first=First(.2);ge2=Ge2(k,.2)
    ge2.metadata['optics_fingerprint']='different'
    with pytest.raises(ValueError,match='optical contract'):
        transport.transport_universal_primary(k,s,first,ge2)
    with pytest.raises(ValueError,match='primary UID'):
        transport.transport_universal_primary(k,replace(s,uid=np.array([9])),first,Ge2(k,.2))


def test_batched_geometry_preserves_independent_track_receiver_frames(monkeypatch):
    k,s=setup();d=k.detector
    positions=np.array([[10.,4.,25.],[-20.,3.,60.]])
    normals=np.array([[0.,0.,-1.],[.6,.8,0.]])
    k.detector=replace(d,positions_m=positions,orientations=normals,effective_area_m2=np.repeat(d.effective_area_m2,2))
    directions=np.array([[0.,0.,1.],[.1,.02,np.sqrt(1-.1**2-.02**2)]])
    starts=np.array([[0.,0.,0.],[2.,1.,50.]])
    s=lh.SpectralLightElements(starts,directions,np.array([50.,48.]),np.full(2,1e8),np.full(2,-1e8),
        np.ones(2),np.array([0.,200.]),np.array([50/.299792458,200+48/.299792458]),
        np.arange(2),np.zeros(2,int),cone_model='spectral')
    def zero(k,s,**kw):
        n=len(k.detector.positions_m)
        return SimpleNamespace(charge_pe=np.zeros(n),bins_pe=np.zeros((n,len(k.config.relative_time_edges_ns)-1)))
    monkeypatch.setattr(transport,'ballistic_interval',zero)
    first=First(0.);captured=[]
    def query(rho,look,lo,hi,**kw):
        captured.append((rho.copy(),look.copy(),lo.copy(),hi.copy()))
        return np.zeros(len(rho))
    first.query_charge_many=query
    transport.transport_universal_primary(k,s,first,Ge2(k,0.))
    expected=[]
    for pos,normal in zip(positions,normals):
        for start,axis,length in zip(starts,directions,s.length_m):
            delta=pos-start;z=delta@axis;radial=delta-z*axis;rho=np.linalg.norm(radial)
            ex=radial/rho;ey=np.cross(axis,ex)
            expected.append((rho,np.array([normal@ex,normal@ey,normal@axis]),z-length,z))
    assert len(captured)==1
    for column,values in zip(captured[0],zip(*expected)):
        np.testing.assert_allclose(column,np.array(values),rtol=1e-14,atol=1e-14)


def test_weak_beta_tail_has_one_charge_pass_and_no_time_pass(monkeypatch):
    import lighthit.experimental.ge2_beta_response as beta_response
    k,s=setup();s=replace(s,beta=np.array([.99]),coefficient2=-s.coefficient0/.99**2)
    calls=[]
    def zero(k,s,**kw):return SimpleNamespace(charge_pe=np.zeros(1),bins_pe=np.zeros((1,len(k.config.relative_time_edges_ns)-1)))
    monkeypatch.setattr(transport,'ballistic_interval',zero)
    monkeypatch.setattr(transport,'combined_first_scattering_segment_bounds',lambda k,s,*a,**kw:(np.ones(len(s)),{}))
    monkeypatch.setattr(transport,'first_scattering_charge_only',lambda *a,**kw:(np.array([.001]),{}))
    def batch(k,s,*a,**kw):
        calls.append(len(s))
        return SimpleNamespace(charge_pe=np.array([.002]),metadata={'test':True})
    monkeypatch.setattr(beta_response,'transport_ge2_tail_batch',batch)
    def forbidden(*a,**kw):pytest.fail('weak OM must not trigger time or per-pair charge transport')
    monkeypatch.setattr(transport,'order1_integrals',forbidden)
    monkeypatch.setattr(transport.DirectGe2,'__call__',forbidden)
    result=transport.transport_universal_primary(k,s,First(0.),Ge2(k,0.))
    assert calls==[1]
    np.testing.assert_allclose(result.charge_orders_pe,[[0.,.001,.002]])
    assert result.metadata['ge2_direct_charge_pairs']==0
    assert result.metadata['q1_fallback_time_oms']==0


def test_charge_atlas_does_not_depend_on_clock(monkeypatch):
    k,s=setup();s=replace(s,end_ns=s.end_ns+10.)
    def zero(k,s,**kw):return SimpleNamespace(charge_pe=np.zeros(1),bins_pe=np.zeros((1,len(k.config.relative_time_edges_ns)-1)))
    monkeypatch.setattr(transport,'ballistic_interval',zero)
    def forbidden(*a,**kw):pytest.fail('stationary charge must not fall back because of clock')
    monkeypatch.setattr(transport,'first_scattering_charge_only',forbidden)
    monkeypatch.setattr(transport.DirectGe2,'__call__',forbidden)
    result=transport.transport_universal_primary(k,s,First(.001),Ge2(k,.002))
    np.testing.assert_allclose(result.charge_orders_pe,[[0.,.001,.002]])


def test_reduced_charge_preserves_beta_and_keeps_full_spectrum_for_time(monkeypatch):
    from lighthit.model import WavelengthQuadrature
    k,s=setup()
    s=lh.SpectralLightElements(np.array([[0.,0.,0.],[0.,0.,50.]]),np.tile([0.,0.,1.],(2,1)),
        np.array([50.,50.]),np.array([1e8,1e8]),-np.array([1e8/.99**2,1e8/.9**2]),
        np.array([.99,.9]),np.array([0.,50/.299792458]),np.array([50/.299792458,100/.299792458]),
        np.arange(2),np.zeros(2,int),cone_model='spectral')
    cube=SimpleNamespace(metadata={'beta_min':.95},quadrature=WavelengthQuadrature([480.],[260.]),validate_optics=lambda k:True)
    calls=[];time_calls=[]
    def zero(k,s,**kw):return SimpleNamespace(charge_pe=np.zeros(1),bins_pe=np.zeros((1,len(k.config.relative_time_edges_ns)-1)))
    monkeypatch.setattr(transport,'ballistic_interval',zero)
    monkeypatch.setattr(transport,'combined_first_scattering_segment_bounds',lambda k,s,*a,**kw:(np.ones(len(s)),{}))
    def stationary(local,source,**kw):
        calls.append((len(local.wavelength.weight_nm),source.beta.copy()))
        return np.array([.01]),{}
    def timed(local,source,**kw):
        time_calls.append((len(local.wavelength.weight_nm),source.beta.copy()))
        return np.array([.02]),np.zeros((1,len(local.config.relative_time_edges_ns)-1)),{}
    monkeypatch.setattr(transport,'first_scattering_charge_only',stationary)
    monkeypatch.setattr(transport,'order1_integrals',timed)
    monkeypatch.setattr(transport.DirectGe2,'__call__',lambda *a,**kw:np.zeros(65,complex) if a[-1] else np.zeros(1,complex))
    result=transport.transport_universal_primary(k,s,First(0.),Ge2(k,0.),batch_beta_charge=False,first_cubature=cube)
    assert result.active.tolist()==[True]
    assert [c[0] for c in calls]==[1,2]
    np.testing.assert_array_equal(calls[0][1],[.99])
    np.testing.assert_array_equal(calls[1][1],[.9])
    assert len(time_calls)==1 and time_calls[0][0]==2
    np.testing.assert_array_equal(time_calls[0][1],[.99,.9])


@pytest.mark.parametrize('beta_charge',[.02,-2e-6])
def test_beta_charge_modes_replace_charge_fallback_but_keep_actual_beta_time(monkeypatch,beta_charge):
    k,s=setup();s=replace(s,beta=np.array([.995]),coefficient2=-s.coefficient0/.995**2)
    calls=[]
    beta_atlas=SimpleNamespace(metadata={'beta_min':.99},resident_bytes=100,
        validate_kernel=lambda k:True,supports=lambda *a:True,
        segment_charge_many=lambda rho,*a,**kw:np.full(len(rho),beta_charge))
    def zero(k,s,**kw):return SimpleNamespace(charge_pe=np.zeros(1),bins_pe=np.zeros((1,len(k.config.relative_time_edges_ns)-1)))
    monkeypatch.setattr(transport,'ballistic_interval',zero)
    monkeypatch.setattr(transport,'combined_first_scattering_segment_bounds',lambda k,s,*a,**kw:(np.ones(len(s)),{}))
    monkeypatch.setattr(transport,'first_scattering_charge_only',lambda *a,**kw:(np.array([.03]),{}))
    monkeypatch.setattr(transport,'order1_integrals',lambda k,s,**kw:(np.array([.01]),np.zeros((1,len(k.config.relative_time_edges_ns)-1)),{}))
    def direct(self,rho,look,beta,clock,lo,hi,density,closest,full):
        assert full, 'charge must already be provided by the beta modes'
        calls.append(beta)
        return np.zeros(65,complex)
    monkeypatch.setattr(transport.DirectGe2,'__call__',direct)
    result=transport.transport_universal_primary(k,s,First(0.),Ge2(k,0.),ge2_beta_atlas=beta_atlas)
    np.testing.assert_allclose(result.charge_orders_pe,[[0.,.03,beta_charge]])
    assert result.metadata['ge2_beta_charge_pairs']==1
    assert result.metadata['ge2_direct_charge_pairs']==0
    assert result.metadata['beta_charge_batch'] is None
    assert result.metadata['ge2_beta_negative_reconstructions']==int(beta_charge<0)
    assert calls==[.995]


def test_relativistic_source_automatically_skips_unneeded_beta_modes(monkeypatch):
    k,s=setup()
    def forbidden(*a,**kw):pytest.fail('beta-one source should use the existing charge atlas')
    beta_atlas=SimpleNamespace(metadata={'beta_min':.99},resident_bytes=100,
        validate_kernel=lambda k:True,supports=forbidden,segment_charge_many=forbidden)
    def zero(k,s,**kw):return SimpleNamespace(charge_pe=np.zeros(1),bins_pe=np.zeros((1,len(k.config.relative_time_edges_ns)-1)))
    monkeypatch.setattr(transport,'ballistic_interval',zero)
    result=transport.transport_universal_primary(k,s,First(.001),Ge2(k,.002),ge2_beta_atlas=beta_atlas)
    np.testing.assert_allclose(result.charge_orders_pe,[[0.,.001,.002]])
    assert result.metadata['ge2_beta_charge_requested'] is True
    assert result.metadata['ge2_beta_charge_active'] is False


@pytest.mark.parametrize('bad',[False,True])
def test_q1_beta_charge_partition_and_selected_exact_refinement(monkeypatch,bad):
    k,s=setup();s=replace(s,length_m=np.array([200.]),end_ns=np.array([200/.299792458]),
        beta=np.array([.9995]),coefficient2=-s.coefficient0/.9995**2)
    total=float(s.coefficient0.sum());direct_mass=[]
    def cache(rho,look,lo,hi,beta,**kw):
        return np.full(len(rho),-.001) if bad else .03*np.asarray(kw['coefficient0_per_m'])*(hi-lo)/total
    beta_atlas=SimpleNamespace(metadata={'beta_min':.999},resident_bytes=100,
        validate_kernel=lambda k:True,supports=lambda *a:True,query_charge_many=cache)
    def zero(k,s,**kw):return SimpleNamespace(charge_pe=np.zeros(1),bins_pe=np.zeros((1,len(k.config.relative_time_edges_ns)-1)))
    monkeypatch.setattr(transport,'ballistic_interval',zero)
    monkeypatch.setattr(transport,'combined_first_scattering_segment_bounds',lambda k,s,*a,**kw:(np.ones(len(s)),{}))
    def stationary(k,part,**kw):
        direct_mass.append(float(part.coefficient0.sum()))
        return np.array([.02*part.coefficient0.sum()/total]),{}
    def timed(k,part,**kw):
        np.testing.assert_allclose(part.coefficient0.sum(),total)
        return np.array([.02]),np.zeros((1,len(k.config.relative_time_edges_ns)-1)),{}
    monkeypatch.setattr(transport,'first_scattering_charge_only',stationary)
    monkeypatch.setattr(transport,'order1_integrals',timed)
    monkeypatch.setattr(transport.DirectGe2,'__call__',lambda *a,**kw:np.zeros(65,complex) if a[-1] else np.zeros(1,complex))
    result=transport.transport_universal_primary(k,s,First(0.),Ge2(k,0.),batch_beta_charge=False,first_beta_atlas=beta_atlas)
    np.testing.assert_allclose(result.charge_orders_pe,[[0.,.02,0.]])
    assert result.metadata['q1_beta_failed_pairs']==int(bad)
    assert result.metadata['q1_beta_charge_pairs']==int(not bad)
    np.testing.assert_allclose(sum(direct_mass),total if bad else .45*total)


def test_relativistic_source_skips_q1_shape_mode(monkeypatch):
    k,s=setup()
    def forbidden(*a,**kw):pytest.fail('no Q1 beta correction needed for beta-one source')
    mode=SimpleNamespace(metadata={'beta_min':.999},resident_bytes=100,
        validate_kernel=lambda k:True,supports=forbidden,query_charge_many=forbidden)
    def zero(k,s,**kw):return SimpleNamespace(charge_pe=np.zeros(1),bins_pe=np.zeros((1,len(k.config.relative_time_edges_ns)-1)))
    monkeypatch.setattr(transport,'ballistic_interval',zero)
    r=transport.transport_universal_primary(k,s,First(.001),Ge2(k,.002),first_beta_atlas=mode)
    assert r.metadata['q1_beta_charge_requested'] is True
    assert r.metadata['q1_beta_charge_active'] is False


def _far_short_piece(om_position):
    """A 5 cm piece at the origin along z and one OM at ``om_position``."""
    k,s=setup()
    k.detector=replace(k.detector,positions_m=np.array([om_position],float))
    s=replace(s,length_m=np.array([.05]),coefficient0=np.array([1e5]),coefficient2=np.array([-1e5]),
        end_ns=np.array([.05/.299792458]))
    return k,s


def _no_first_order(monkeypatch):
    def zero(k,s,**kw):return SimpleNamespace(charge_pe=np.zeros(1),bins_pe=np.zeros((1,len(k.config.relative_time_edges_ns)-1)))
    monkeypatch.setattr(transport,'ballistic_interval',zero)
    monkeypatch.setattr(transport,'combined_first_scattering_segment_bounds',lambda k,s,*a,**kw:(np.ones(len(s)),{}))
    monkeypatch.setattr(transport,'first_scattering_charge_only',lambda *a,**kw:(np.zeros(1),{}))


def test_line_passing_an_om_far_from_the_piece_is_not_near_field(monkeypatch):
    # The line of the piece passes 0.5 m from the OM, the piece itself is 100 m
    # away: the order >=2 integral goes to the direct fallback, not to an error.
    k,s=_far_short_piece([.5,0.,100.])
    _no_first_order(monkeypatch)
    calls=[]
    def direct(self,rho,look,beta,clock,lo,hi,density,closest,full):
        calls.append((rho,lo,hi));return np.zeros(65,complex)
    monkeypatch.setattr(transport.DirectGe2,'__call__',direct)
    result=transport.transport_universal_primary(k,s,First(0.),Ge2(k,0.))
    assert result.metadata['ge2_fallback_pairs']==1
    (rho,lo,hi),=calls
    assert rho==pytest.approx(.5) and lo==pytest.approx(99.95) and hi==pytest.approx(100.)


def test_part_of_a_piece_inside_the_inner_radius_is_left_to_the_near_field(monkeypatch):
    # A 10 m piece passes 1 m from the OM: only its part closer than 3 m
    # (|x| < sqrt(8)) leaves the tables; the rest goes to the direct fallback.
    k,s=_far_short_piece([1.,0.,5.])
    s=replace(s,length_m=np.array([10.]),end_ns=np.array([10/.299792458]))
    _no_first_order(monkeypatch)
    calls=[]
    def direct(self,rho,look,beta,clock,lo,hi,density,closest,full):
        calls.append((lo,hi));return np.zeros(65,complex)
    monkeypatch.setattr(transport.DirectGe2,'__call__',direct)
    result=transport.transport_universal_primary(k,s,First(0.),Ge2(k,0.))
    inner=np.sqrt(8.)
    np.testing.assert_allclose(sorted(calls),[(-5.,-inner),(inner,5.)])
    assert result.metadata['ge2_near_field_oms']==[0]


def test_cone_evaluator_accepts_small_impact_only_outside_the_radial_minimum():
    from lighthit.experimental.track_ge2_primitive import cone_point_evaluator
    coefficients=np.zeros((1,2,3,3,4,3),np.complex64)
    bundle=dict(degree=2,acceptance_degree=1,omega=np.array([0.,.3,.6]),
        pair_lambda=np.array([0,1,1]),pair_mu=np.array([0,0,1]),
        log_r=np.log([3.,20.,300.]),coefficients=coefficients,absorption=np.array([.03]),
        speed=np.array([.22]),phase='flight')
    evaluate=cone_point_evaluator(bundle,impact_m=.5,look=np.array([0.,0.,1.]),alpha=np.array([4*np.pi,1.]),
        wavelength_nm=np.array([480.]),wavelength_weight_nm=np.ones(1),phase_index=np.array([1.35]),
        efficiency=np.ones(1),effective_area_m2=.05,backend='fused-looks')
    assert np.isfinite(evaluate(np.array([-100.,-50.]))).all()
    with pytest.raises(ValueError,match='radial minimum'):
        evaluate(np.array([-100.,-1.]))
