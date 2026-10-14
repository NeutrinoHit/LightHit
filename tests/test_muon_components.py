from dataclasses import replace
import numpy as np
import pytest
import lighthit as lh

pytest.importorskip("numba")
from lighthit.experimental.muon_packets import partition_muon
from lighthit.experimental.muon_components import MuonFineStrategy, transport_muon_components
from lighthit.shower_basis import _window_report_from_photons
from test_prompt import config, detector, spectral_medium, shower


def source_and_ancestry():
    e = shower().elements
    count = 16
    primary = lh.SpectralLightElements(
        start_m=np.column_stack((np.full(count, 12.), np.full(count, 3.), np.arange(count)*.5)),
        direction=np.tile([0.,0.,1.], (count,1)), length_m=np.full(count,.5),
        coefficient0=np.full(count,1e7), coefficient2=np.full(count,-1e7), beta=np.ones(count),
        start_ns=np.arange(count)*.5/.299792458, end_ns=(np.arange(count)+1)*.5/.299792458,
        row_index=np.arange(count), uid=np.zeros(count,int), cone_model="spectral")
    e = replace(e, row_index=np.arange(len(e))+count, uid=np.ones(len(e),int), cone_model="spectral")
    names=("start_m","direction","length_m","coefficient0","coefficient2","beta","start_ns","end_ns","row_index","uid")
    source=lh.SpectralLightElements(**{k:np.concatenate([getattr(primary,k),getattr(e,k)]) for k in names},cone_model="spectral")
    ancestry=dict(uid=np.array([0,1]),pdgid=np.array([13,11]),energy_GeV=np.array([100.,1.]),
                   start_m=np.array([[12.,3.,0.],[12.,3.,0.]]),start_ns=np.zeros(2),
                   creator=np.array(["primary","muIoni"]),element_family=source.uid.copy())
    return source, ancestry


def test_family_packets_cover_source_and_preserve_fields_and_delays():
    source, ancestry=source_and_ancestry()
    time=source.start_ns.copy(); end=source.end_ns.copy()
    time[-2:]+=2000.;end[-2:]+=2000.
    source=replace(source,start_ns=time,end_ns=end)
    packets=partition_muon(source,ancestry)
    assert {p.kind for p in packets} == {"primary","shower","late"}
    assert np.array_equal(np.sort(np.concatenate([p.indices for p in packets])),np.arange(len(source)))
    for packet in packets:
        part=packet.source(source)
        np.testing.assert_array_equal(part.start_ns,source.start_ns[np.sort(packet.indices)])
    for name in ("coefficient0","coefficient2"):
        assert sum(getattr(p.source(source),name).sum() for p in packets)==pytest.approx(getattr(source,name).sum(),rel=1e-14)


def test_point_packet_uses_emission_axis():
    from lighthit.experimental.axial_source import AxisFrame
    source,_=source_and_ancestry()
    single=source.subset(np.arange(len(source))==0).field(0)
    frame=AxisFrame.of(single)
    np.testing.assert_allclose(frame.axis,single.direction[0],rtol=0,atol=1e-14)


def test_window_vectorization_preserves_counts_and_finite_zero_absorption(tmp_path):
    k=lh.TransportKernel(spectral_medium(),detector(4),config(tmp_path))
    points=np.array([[0.,0.,0.],[300.,0.,0.]])
    photons=np.array([2.,5.]);positions=k.detector.positions_m
    got=_window_report_from_photons(k,points,photons,positions,1.,200.,0.,1.)
    assert np.isfinite(got[3]).all()
    for i,p in enumerate(positions):
        r=np.linalg.norm(points-p,axis=1);mask=r>200.
        assert got[0][i]==sum((r>=1.)&~mask)
        expected=10*k.detector.effective_area_m2[i]*np.max(abs(k.detector.angular_acceptance(np.linspace(-1,1,257))))*sum(photons[mask]/(4*np.pi*r[mask]**2))
        assert got[3][i]==pytest.approx(expected,rel=1e-14)


def test_component_strategy_preserves_ballistic_and_global_selection(tmp_path):
    source, ancestry=source_and_ancestry()
    kernel=lh.TransportKernel(spectral_medium(),detector(3),config(tmp_path,omega_per_ns=np.linspace(0.,1.2,9),
                               scattering_degree=4,source_degree=4,radial_nodes=16))
    pc=lh.PromptConfig(element_method="grouped",screen=False)
    basis=lh.ShowerBasis(track_uid=0,spectral_fusion=True)
    ref=kernel.transport_hybrid(source,frequency_nodes=9,ge2_basis=basis,prompt_config=pc)
    got=transport_muon_components(kernel,source,ancestry,frequency_nodes=9,basis=basis,
                                   prompt_config=pc,prompt_radius_m=1e6)
    np.testing.assert_array_equal(got.charge_components_pe[:,0],ref.charge_components_pe[:,0])
    np.testing.assert_array_equal(got.components_pe[:,:,0],ref.components_pe[:,:,0])
    np.testing.assert_array_equal(got.spectrum_ge2_pe,ref.spectrum_ge2_pe)
    assert got.metadata["muon_components"]["primary_integral_pieces"] < 16
    assert got.metadata["muon_components"]["first_order_calls"]
    split=transport_muon_components(kernel,source,ancestry,frequency_nodes=9,basis=basis,
                                   prompt_config=pc,prompt_radius_m=1e6,partition_ge2=True)
    np.testing.assert_array_equal(split.charge_components_pe[:,0],ref.charge_components_pe[:,0])
    assert split.metadata["source_partition_components"]>=2
    assert np.isfinite(split.components_pe).all()


def test_source_partition_rejects_double_counting(tmp_path):
    from lighthit.shower_basis import transport_hybrid_basis
    source,_=source_and_ancestry()
    k=lh.TransportKernel(spectral_medium(),detector(2),config(tmp_path,omega_per_ns=np.linspace(0.,1.2,9)))
    def bad(elements,basis):
        return [("one",elements,2,.5,"stencil"),("two",elements,2,.5,"stencil")]
    with pytest.raises(ValueError,match="cover each element once"):
        transport_hybrid_basis(k,source,lh.ShowerBasis(),frequency_nodes=9,source_partition=bad)


@pytest.mark.parametrize("method", ["segment", "ring"])
def test_component_route_rejects_silent_strategy_bypass(tmp_path, method):
    source, ancestry = source_and_ancestry()
    kernel = lh.TransportKernel(spectral_medium(), detector(2), config(tmp_path))
    with pytest.raises(ValueError, match="element_method"):
        transport_muon_components(kernel, source, ancestry,
                                  prompt_config=lh.PromptConfig(element_method=method))


def test_component_route_rejects_unbounded_radius(tmp_path):
    source, ancestry = source_and_ancestry()
    kernel = lh.TransportKernel(spectral_medium(), detector(2), config(tmp_path))
    with pytest.raises(ValueError, match="finite and positive"):
        transport_muon_components(kernel, source, ancestry, prompt_radius_m=np.inf)


def test_spectral_point_blocks_preserve_spectrum_budget_and_selection(tmp_path, monkeypatch):
    import lighthit.shower_basis as implementation
    source, _ = source_and_ancestry()
    kernel = lh.TransportKernel(spectral_medium(), detector(3), config(tmp_path,
        omega_per_ns=np.linspace(0., 1.2, 9), scattering_degree=4, source_degree=4, radial_nodes=16))
    basis = lh.ShowerBasis(track_uid=0, spectral_fusion=True, apply_tolerance_pe=1e-6)
    settings = lh.PromptConfig(element_method="grouped", screen=False)
    reference = kernel.transport_hybrid(source, frequency_nodes=9, ge2_basis=basis, prompt_config=settings)
    monkeypatch.setattr(implementation, "_SPECTRAL_APPLY_POINTS", 3)
    blocked = kernel.transport_hybrid(source, frequency_nodes=9, ge2_basis=basis, prompt_config=settings)
    np.testing.assert_array_equal(blocked.active, reference.active)
    np.testing.assert_allclose(blocked.charge_components_pe, reference.charge_components_pe, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(blocked.components_pe, reference.components_pe, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(blocked.spectrum_ge2_pe, reference.spectrum_ge2_pe, rtol=1e-12, atol=1e-13)
