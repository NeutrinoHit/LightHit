#!/usr/bin/env python3
"""Apply frozen, event-independent primary atlases to five-cluster G4 sources."""
import paths  # inputs and outputs: scripts/shower_basis/paths.py
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import resource
import sys
from time import perf_counter,process_time
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--energy',type=int,default=1000)
    p.add_argument('--source',type=Path)
    p.add_argument('--fallback-pack',type=Path,default=paths.GE2_PACK.path())
    p.add_argument('--q1-atlas',type=Path,default=paths.Q1_ATLAS.path())
    p.add_argument('--ge2-atlas',type=Path,default=paths.GE2_ATLAS.path())
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--controls',type=int,default=0,help='Direct controls: top N predicted OMs plus near-threshold and cluster sentinels')
    p.add_argument('--control-indices',type=int,nargs='+')
    p.add_argument('--receiver-indices',type=int,nargs='+')
    p.add_argument('--controls-only',action='store_true',help='Extend direct controls of an existing immutable result')
    p.add_argument('--all-active-controls',action='store_true')
    p.add_argument('--no-batch-beta-charge',action='store_true',help='Use per-segment >=2 charge fallback for comparison')
    p.add_argument('--beta-cell-m',type=float,default=.25,help='Spatial cell of the actual-beta stationary tail compiler')
    p.add_argument('--q1-response-cubature',type=Path,help='Explicit experimental reduced spectral rule for Q1 charge only')
    p.add_argument('--ge2-beta-charge',type=Path,help='Explicit stationary >=2 Taylor-response payload')
    p.add_argument('--q1-beta-charge',type=Path,help='Explicit near-beta-one Q1 stationary shape correction')
    p.add_argument('--direct-ge2-quadrature',choices=['fixed12','adaptive','filon'],default='fixed12',
        help='Experimental short-interval rule for the actual-beta/time RTE fallback')
    p.add_argument('--q1-charge-gauss-order',type=int,choices=[0,4,6,8],default=0,
        help='Explicit stationary-only Gauss path rule; zero retains the original grid')
    args=p.parse_args()
    if (args.output/'response.npz').exists() and not args.controls_only:p.error('choose a fresh output path')
    import lighthit as lh,ntsim,numba
    from ntsim.LightTransport import bgvd_model_from_ntsim
    from lighthit.experimental.track_first_primitive import TrackFirstAtlasStore
    from lighthit.experimental.track_ge2_primitive import DiskTrackGe2Atlas
    from lighthit.experimental.universal_primary import transport_universal_primary,subset_kernel,DirectGe2
    if numba.get_num_threads()!=1:p.error('NUMBA_NUM_THREADS=1 required')
    source_path=args.source or paths.muon_source(args.energy)
    with np.load(source_path) as z:
        source=lh.SpectralLightElements(**{k:z[k] for k in z.files if k!='metadata'},cone_model='spectral')
        source_meta=json.loads(str(z['metadata']))
    source=source.subset(source.uid==0)
    model=bgvd_model_from_ntsim(ntsim.bgvd.Telescope(n_clusters=(1,2,3,4,5)),ntsim.bgvd.Water())
    if source_meta['model_fingerprint']!=model.fingerprint:raise ValueError('source/model differs')
    global_ids=np.arange(len(model.detector.positions_m))
    if args.receiver_indices:
        global_ids=np.asarray(args.receiver_indices);d=model.detector
        model=replace(model,detector=replace(d,positions_m=d.positions_m[global_ids],orientations=d.orientations[global_ids],
            effective_area_m2=d.effective_area_m2[global_ids],identifiers={k:v[global_ids] for k,v in d.identifiers.items()}))
    kernel=model.kernel(lh.KernelConfig(omega_per_ns=np.linspace(0,1.2,65),
        readout_omega_per_ns=np.linspace(0,1.2,321),threshold_pe=.01,cache_policy='require',
        cache_directory=str(paths.RTE_CACHE.path())))
    manifests=[args.q1_atlas/'manifest.json',args.ge2_atlas/'manifest.json']
    fingerprints=[hashlib.sha256(path.read_bytes()).hexdigest() for path in manifests]
    start=perf_counter();first=TrackFirstAtlasStore(args.q1_atlas);ge2=DiskTrackGe2Atlas(args.ge2_atlas)
    load_seconds=perf_counter()-start
    if args.controls_only:
        from types import SimpleNamespace
        report=json.loads((args.output/'report.json').read_text())
        if report['source_sha256']!=hashlib.sha256(source_path.read_bytes()).hexdigest() or report['cache_manifest_sha256']!=fingerprints:
            raise ValueError('control input differs from saved source/cache')
        with np.load(args.output/'response.npz') as z:
            if not np.array_equal(z['positions_m'],model.detector.positions_m):raise ValueError('control geometry differs')
            result=SimpleNamespace(**{key:z[key].copy() for key in ('charge_orders_pe','bins_orders_pe','time_origin_ns','relative_time_edges_ns','active')},metadata=json.loads(str(z['metadata'])))
    else:
        # Warm on one source record and synthetic cache queries, before timing.
        from lighthit.ballistic_interval import ballistic_interval
        warm_start=perf_counter()
        ballistic_interval(subset_kernel(kernel,[0]),source.subset(np.arange(len(source))==0),
            time_origin_ns=[0.],relative_edges_ns=kernel.config.relative_time_edges_ns)
        from lighthit.experimental.first_charge_only import first_scattering_charge_only
        first_scattering_charge_only(subset_kernel(kernel,[0]),source.subset(np.arange(len(source))==0),
            gauss_order=args.q1_charge_gauss_order,panel_width=1.5)
        from lighthit.experimental.filon_ge2 import warm_direct_rte
        warm_direct_rte()
        rho=float(first.radii_m[len(first.radii_m)//2]);orientation=np.array([[0.,0.,1.]])
        first.query_charge_many([rho],orientation,[0.],[rho],coefficient0_per_m=[1.])
        rho=float(ge2.impact_nodes_m[len(ge2.impact_nodes_m)//2])
        ge2.segment_charge_many([rho],orientation,[rho],[0.],coefficient0_per_m=[1.])
        warm_seconds=perf_counter()-warm_start
        args.output.mkdir(parents=True,exist_ok=True)
        def progress(stage,details):print(stage,json.dumps(details),flush=True)
        cubature=None
        if args.q1_response_cubature:
            from lighthit.experimental.spectral_response_basis import ResponseCubature
            cubature=ResponseCubature.load(args.q1_response_cubature)
        beta_atlas=None
        if args.ge2_beta_charge:
            from lighthit.experimental.ge2_charge_beta_atlas import Ge2ChargeBetaAtlas
            beta_atlas=Ge2ChargeBetaAtlas(args.ge2_beta_charge,ge2)
        first_beta=None
        if args.q1_beta_charge:
            from lighthit.experimental.first_charge_beta_atlas import FirstChargeBetaAtlas
            first_beta=FirstChargeBetaAtlas(args.q1_beta_charge,first)
        result=transport_universal_primary(kernel,source,first,ge2,progress=progress,
            batch_beta_charge=not args.no_batch_beta_charge,beta_cell_m=args.beta_cell_m,first_cubature=cubature,
            ge2_beta_atlas=beta_atlas,first_beta_atlas=first_beta,direct_ge2_quadrature=args.direct_ge2_quadrature,
            q1_charge_gauss_order=args.q1_charge_gauss_order,fallback_pack=args.fallback_pack)
        if fingerprints != [hashlib.sha256(path.read_bytes()).hexdigest() for path in manifests]:
            raise RuntimeError('cache manifest changed during event evaluation')
        np.savez_compressed(args.output/'response.npz',charge_orders_pe=result.charge_orders_pe,
            bins_orders_pe=result.bins_orders_pe,time_origin_ns=result.time_origin_ns,
            relative_time_edges_ns=result.relative_time_edges_ns,active=result.active,
            positions_m=model.detector.positions_m,receiver_indices=global_ids,
            metadata=json.dumps(result.metadata))
        report=dict(source=str(source_path),source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
            source_metadata=source_meta,cache_manifest_sha256=fingerprints,cache_build_included=False,
            cache_load_seconds=load_seconds,jit_warm_seconds=warm_seconds,metadata=result.metadata,
            cache_total_resident_mib=result.metadata['atlas_resident_bytes']/2**20,
            process_peak_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(2**20 if sys.platform=='darwin' else 2**10),
            controls=[])
        (args.output/'report.json').write_text(json.dumps(report,indent=2))
        print('COMPLETE',json.dumps({k:result.metadata[k] for k in ['elapsed_seconds','cpu_seconds','active','timings',
            'q1_fallback_oms','ge2_fallback_pairs','q1_time_fallback_pairs','ge2_time_fallback_pairs']}),flush=True)
    if args.controls or args.control_indices or args.all_active_controls:
        from lighthit.experimental.primary_polyline import fit_primary_polyline,primary_spectral_quadrature
        from lighthit.experimental.muon_integral import order1_integrals
        from lighthit.prompt import PromptConfig
        from lighthit.readout import inverse_bins
        from scipy.interpolate import CubicSpline
        from copy import copy
        score=result.charge_orders_pe.sum(1)
        ids=list(np.argsort(score)[-args.controls:]) if args.controls else []
        ids.extend(np.flatnonzero((score>=.008)&(score<=.012)).tolist())
        if args.controls:
            for cluster in np.unique(model.detector.identifiers['cluster_id']):
                group=np.flatnonzero(model.detector.identifiers['cluster_id']==cluster)
                ids.append(int(group[np.argmax(score[group])]))
        if args.control_indices:ids.extend(int(np.flatnonzero(global_ids==i)[0]) for i in args.control_indices)
        if args.all_active_controls:ids.extend(np.flatnonzero(result.active).tolist())
        done={row['om'] for row in report['controls']}
        ids=sorted(i for i in set(ids) if int(global_ids[i]) not in done)
        pieces=fit_primary_polyline(source,phase_indices=kernel.medium.phase_index).elements
        q1kernel=copy(kernel);q1kernel.wavelength=primary_spectral_quadrature(kernel.medium,kernel.detector,order=2)
        direct=DirectGe2(kernel,args.fallback_pack)
        for om in ids:
            began,cpu=perf_counter(),process_time()
            q1,b1,_=order1_integrals(subset_kernel(q1kernel,[om]),pieces,
                time_origin_ns=result.time_origin_ns[[om]],config=PromptConfig(element_method='segment',screen=False))
            spectrum=np.zeros(len(ge2.omega_per_ns),complex)
            for i in range(len(pieces)):
                axis=pieces.direction[i];delta=model.detector.positions_m[om]-pieces.start_m[i]
                z=float(delta@axis);radial=delta-z*axis;rho=float(np.linalg.norm(radial))
                if rho>=300:continue
                reach=np.sqrt(300**2-rho**2);a=min(z,reach);b=max(z-pieces.length_m[i],-reach)
                if a<=b:continue
                ex=radial/rho;ey=np.cross(axis,ex);normal=-model.detector.orientations[om]
                look=np.array([normal@ex,normal@ey,normal@axis]);clock=(pieces.end_ns[i]-pieces.start_ns[i])/pieces.length_m[i]
                closest=pieces.start_ns[i]+z*clock-result.time_origin_ns[om]
                spectrum+=direct(rho,look,float(pieces.beta[i]),clock,b,a,pieces.coefficient0[i]/pieces.length_m[i],closest,True)
            b2=inverse_bins(np.linspace(0,1.2,321),CubicSpline(ge2.omega_per_ns,spectrum)(np.linspace(0,1.2,321))[:,None],result.relative_time_edges_ns)[0]
            refq=np.array([result.charge_orders_pe[om,0],q1[0],spectrum[0].real])
            refb=np.column_stack((result.bins_orders_pe[om,:,0],b1[0],b2))
            time_compared=bool(result.metadata['time_computed_mask'][om])
            row=dict(om=int(global_ids[om]),reference_charge_pe=refq.tolist(),candidate_charge_pe=result.charge_orders_pe[om].tolist(),
                charge_relative=(abs(result.charge_orders_pe[om]-refq)/np.maximum(abs(refq),1e-30)).tolist(),
                bins_relative_l1=((abs(result.bins_orders_pe[om]-refb).sum(0)/np.maximum(abs(refb).sum(0),1e-30)).tolist()
                                 if time_compared else [0.,None,None]),
                time_compared=time_compared,
                total_charge_absolute_error_pe=float(abs(result.charge_orders_pe[om].sum()-refq.sum())),
                total_charge_relative=float(abs(result.charge_orders_pe[om].sum()-refq.sum())/max(abs(refq.sum()),1e-30)),
                total_bins_relative_l1=(float(abs(result.bins_orders_pe[om].sum(1)-refb.sum(1)).sum()/max(abs(refb.sum(1)).sum(),1e-30))
                                       if time_compared else None),
                direct_active=bool(refq.sum()>=.01),candidate_active=bool(result.active[om]),
                direct_wall_seconds=perf_counter()-began,direct_cpu_seconds=process_time()-cpu)
            report['controls'].append(row)
            np.savez_compressed(args.output/f'control-{int(global_ids[om])}.npz',reference_charge=refq,reference_bins=refb)
            (args.output/'report.json').write_text(json.dumps(report,indent=2))
            print('CONTROL',json.dumps(row),flush=True)


if __name__=='__main__':main()
