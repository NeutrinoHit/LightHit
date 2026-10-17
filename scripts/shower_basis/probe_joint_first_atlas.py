#!/usr/bin/env python3
"""Reusable Q1 optical atlas on a bounded real secondary packet, all bands."""
import paths  # inputs and outputs: scripts/shower_basis/paths.py
import argparse,hashlib,json
from pathlib import Path
from time import process_time,perf_counter
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--atlas',type=Path,default=paths.work('joint-first-20261011/atlas-test/production-grid-atlas.npz'))
    p.add_argument('--energy',type=int,choices=[300,1000,3000],default=1000)
    p.add_argument('--elements',type=int,default=128)
    p.add_argument('--om',type=int,default=197)
    p.add_argument('--ring-nodes',type=int,default=32)
    p.add_argument('--source-guard-factor',type=float,default=8.)
    p.add_argument('--nearest-packet-om',action='store_true')
    args=p.parse_args()
    if args.output.exists():p.error('fresh output required')
    import numba,ntsim,lighthit as lh
    from ntsim.LightTransport import bgvd_model_from_ntsim
    from lighthit.experimental.joint_source import compile_photon_histogram
    from lighthit.experimental.joint_first_atlas import UniversalFirstCDFAtlas,evaluate_photon_histogram_atlas,photon_source_guard
    from lighthit.experimental.axial_source import AxisFrame
    from lighthit.prompt import _spectral_tables,acceptance_polynomial,acceptance_monomials
    from lighthit._prompt_numba import pencils_order1,interval_table
    if numba.get_num_threads()!=1:p.error('NUMBA_NUM_THREADS=1 required')
    path=paths.muon_source(args.energy)
    with np.load(path) as z:
        source=lh.SpectralLightElements(**{k:z[k] for k in z.files if k!='metadata'},cone_model='spectral')
    with np.load(paths.muon_ancestry(args.energy)) as z:
        family=z['element_family']
    model=bgvd_model_from_ntsim(ntsim.bgvd.Telescope(n_clusters=(1,2,3,4,5)),ntsim.bgvd.Water())
    kernel=model.kernel(lh.KernelConfig(omega_per_ns=np.array([0.]),cache_policy='require'))
    tables=_spectral_tables(kernel);reference_band=int(np.argmin(abs(tables['wavelength_nm']-480.)))
    phase=tables['phase_index'][reference_band];receiver=kernel.detector.positions_m[args.om]
    look=-kernel.detector.orientations[args.om];area=float(kernel.detector.effective_area_m2[args.om])
    mass=np.maximum(source.coefficient0+source.coefficient2/phase**2,0.)
    valid=(source.uid!=0)&(source.beta*phase>1.)
    score=mass*np.exp(-np.linalg.norm(source.midpoints_m-receiver,axis=1)/50.)
    roots=np.unique(family[valid]);total=np.array([score[valid&(family==root)].sum() for root in roots])
    root=roots[total.argmax()];rows=np.flatnonzero(valid&(family==root))
    rows=rows[np.argsort(mass[rows])[-args.elements:]];mask=np.zeros(len(source),bool);mask[rows]=True
    packet=source.subset(mask)
    selection_om=args.om
    if args.nearest_packet_om:
        centre=np.average(packet.midpoints_m,axis=0,weights=mass[mask])
        args.om=int(np.argmin(np.linalg.norm(kernel.detector.positions_m-centre,axis=1)))
        receiver=kernel.detector.positions_m[args.om];look=-kernel.detector.orientations[args.om]
        area=float(kernel.detector.effective_area_m2[args.om])
    frame=AxisFrame.of(packet.field(0,phase_index=float(tables['phase_index'].max())))
    apoly=acceptance_polynomial(*acceptance_monomials(kernel.acceptance()['alpha']))
    cpu=process_time()
    if args.atlas.exists():atlas=UniversalFirstCDFAtlas.load(args.atlas);build_or_load='load'
    else:
        atlas=UniversalFirstCDFAtlas.build(g=float(kernel.medium.g),degree=len(apoly)-1)
        atlas.save(args.atlas);build_or_load='build_and_save'
    atlas_io_cpu=process_time()-cpu
    edges=np.arange(-5.,806.,5.)
    origin=float(np.min(packet.start_ns+np.linalg.norm(packet.start_m-receiver,axis=1)/tables['speed'].max()))
    originals=np.zeros(len(edges)-1);grouped=originals.copy();cached=originals.copy()
    guarded_reference=originals.copy();guarded_charge=0.
    original_charge=grouped_charge=cached_charge=0.;records=[]
    def direct(hist,band,raw=False):
        if raw:points,directions,times,weights=(hist.original_position,hist.original_direction,hist.original_times_ns,hist.original_photons_per_nm)
        else:
            points=hist.position[hist.geometry_id];directions=hist.direction[hist.geometry_id]
            times=hist.emission_times_ns;weights=hist.photon_weights[0]
        mus=float(tables['mus'][band]);mut=float(tables['mut'][band]);speed=float(tables['speed'][band])
        kappa=np.array([-mut*speed]);ee=np.exp(np.outer(kappa,edges));it=interval_table(edges,kappa)
        spectral=float(tables['weight_nm'][band]*tables['efficiency'][band]);g=float(kernel.medium.g)
        return pencils_order1(np.ascontiguousarray(receiver[None,:]),np.ascontiguousarray(look[None,:]),np.array([area]),np.array([origin]),
            np.ascontiguousarray(points),np.ascontiguousarray(directions),np.ascontiguousarray(weights),np.zeros(len(weights)),
            np.ascontiguousarray(times),apoly,g,(1-abs(g))/(1+abs(g)),.025,.025,25.,np.array([mus]),np.array([mut]),np.array([speed]),
            np.array([spectral]),np.zeros(1),edges,ee,it,4,4096)
    # Warm all compiled routines using a geometry outside the real packet.
    warm=atlas.prepare(np.array([[30.,0.,40.]]),np.array([[0.,0.,1.]]),look,apoly,
        scattering_per_m=.015,absorption_per_m=.05,g=float(kernel.medium.g),speed_m_per_ns=.22)
    warm.contract(np.array([0]),[0.],[1.],edges)
    args.output.mkdir(parents=True)
    wall,cpu=perf_counter(),process_time()
    for band,(wave,phase) in enumerate(zip(tables['wavelength_nm'],tables['phase_index'])):
        began=process_time()
        hist=compile_photon_histogram(packet,float(wave),float(phase),frame=frame,z_step_m=.5,
            mu_step=.03,phi_step_rad=.06,tau_step_ns=.1,transverse_step_m=.1,
            ring_nodes=args.ring_nodes,chord_order=2,retain_samples=True)
        source_cpu=process_time()-began
        if band==0:
            # One original point warms the physical reference's compiled code.
            from dataclasses import replace
            one=replace(hist,original_position=hist.original_position[:1],original_direction=hist.original_direction[:1],
                original_times_ns=hist.original_times_ns[:1],original_photons_per_nm=hist.original_photons_per_nm[:1])
            direct(one,band,True)
        began=process_time();qr,br=direct(hist,band,True);raw_cpu=process_time()-began
        began=process_time();qg,bg=direct(hist,band);hist_cpu=process_time()-began
        unsafe=photon_source_guard(hist,receiver,factor=args.source_guard_factor)
        if np.any(unsafe):
            from types import SimpleNamespace
            clocks=~unsafe[hist.geometry_id];members=unsafe[hist.original_geometry_id]
            mixed=SimpleNamespace(original_position=np.concatenate((hist.position[hist.geometry_id[clocks]],hist.original_position[members])),
                original_direction=np.concatenate((hist.direction[hist.geometry_id[clocks]],hist.original_direction[members])),
                original_times_ns=np.concatenate((hist.emission_times_ns[clocks],hist.original_times_ns[members])),
                original_photons_per_nm=np.concatenate((hist.photon_weights[0,clocks],hist.original_photons_per_nm[members])))
            began=process_time();qmix,bmix=direct(mixed,band,True);guarded_direct_cpu=process_time()-began
        else:qmix,bmix=qg,bg;guarded_direct_cpu=hist_cpu
        spectral=float(tables['weight_nm'][band]*tables['efficiency'][band])
        qvalues,bvalues,guard_meta=evaluate_photon_histogram_atlas(atlas,kernel,hist,np.array([args.om]),edges,
            np.array([origin]),wavelength_weight_nm=float(tables['weight_nm'][band]),
            source_guard_factor=args.source_guard_factor)
        q,b=float(qvalues[0]),bvalues[0]
        prepare_cpu=guard_meta['prepare_cpu_s'];apply_cpu=guard_meta['apply_cpu_s']
        original_charge+=float(qr[0]);grouped_charge+=float(qg[0]);cached_charge+=q
        guarded_charge+=float(qmix[0]);guarded_reference+=bmix[0]
        originals+=br[0];grouped+=bg[0];cached+=b
        item=dict(wavelength_nm=float(wave),geometry_cells=len(hist.position),clock_cells=len(hist.geometry_id),
            raw_photons=len(hist.original_times_ns),fallback_geometries=guard_meta['fallback_geometries'],
            source_guard_geometries=guard_meta['source_guard_fallback_geometries'],
            source_guard_members=guard_meta['source_guard_fallback_members'],
            source_guard_charge_fraction=guard_meta['source_guard_charge_fraction'],
            source_guard_cpu_s=guard_meta['source_guard_cpu_s'],
            source_compile_cpu_s=source_cpu,raw_cpu_s=raw_cpu,hist_direct_cpu_s=hist_cpu,
            guarded_direct_cpu_s=guarded_direct_cpu,
            atlas_prepare_cpu_s=prepare_cpu,atlas_apply_cpu_s=apply_cpu,
            operator_charge_relative_error=float(abs(q-qmix[0])/max(abs(qmix[0]),1e-300)),
            operator_bins_l1=float(abs(b-bmix[0]).sum()/max(abs(bmix[0]).sum(),1e-300)))
        records.append(item);print(json.dumps(item),flush=True)
        hist=None
    def errors(q,b,qr,br):return dict(charge_pe=float(q),charge_relative=float(abs(q-qr)/max(abs(qr),1e-300)),
        bins_l1=float(abs(b-br).sum()/max(abs(br).sum(),1e-300)),negative_bin_mass_pe=float(np.maximum(-b,0.).sum()))
    report=dict(energy_GeV=args.energy,source_family=int(root),source_elements=len(packet),om=args.om,
        packet_selection_om=selection_om,nearest_packet_om=bool(args.nearest_packet_om),
        geometry_clusters=5,threads=1,wavelength_nodes=len(records),waveforms_use_physical_time=True,
        source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),model_fingerprint=model.fingerprint,
        atlas_path=str(args.atlas),atlas_sha256=hashlib.sha256(args.atlas.read_bytes()).hexdigest(),
        atlas_megabytes=atlas.modes.nbytes/1024**2,atlas_io=build_or_load,atlas_io_cpu_s=atlas_io_cpu,
        source_guard_factor=args.source_guard_factor,
        wall_s=perf_counter()-wall,cpu_s=process_time()-cpu,time_origin_ns=origin,records=records,
        source_binning=errors(grouped_charge,grouped,original_charge,originals),
        guarded_source=errors(guarded_charge,guarded_reference,original_charge,originals),
        operator=errors(cached_charge,cached,guarded_charge,guarded_reference),
        total=errors(cached_charge,cached,original_charge,originals),
        timing_scope='small actual compact secondary packet, all nine reference wavelength nodes, one real OM in five-cluster model; not full event')
    report['raw_cpu_s']=sum(x['raw_cpu_s'] for x in records)
    report['hist_direct_cpu_s']=sum(x['hist_direct_cpu_s']+x['source_compile_cpu_s'] for x in records)
    report['atlas_cpu_s']=sum(x['atlas_prepare_cpu_s']+x['atlas_apply_cpu_s']+x['source_compile_cpu_s'] for x in records)
    (args.output/'report.json').write_text(json.dumps(report,indent=2))
    np.savez_compressed(args.output/'responses.npz',raw_charge=original_charge,raw_bins=originals,
        grouped_charge=grouped_charge,grouped_bins=grouped,atlas_charge=cached_charge,atlas_bins=cached,
        guarded_charge=guarded_charge,guarded_bins=guarded_reference,
        edges=edges,origin=origin)
    print(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2),flush=True)


if __name__=='__main__':main()
