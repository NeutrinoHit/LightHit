#!/usr/bin/env python3
"""Measure a completely independent q0/q1/ge2 charge preview on saved muons."""
import paths  # inputs and outputs: scripts/shower_basis/paths.py
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import signal
from time import perf_counter

import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--energy",type=int,default=1000)
    p.add_argument("--cpu-limit",type=float,default=120.)
    p.add_argument("--output",type=Path)
    p.add_argument("--detector-tree",action="store_true")
    p.add_argument("--degree",type=int,default=8)
    p.add_argument("--azimuthal-degree",type=int,default=0)
    p.add_argument("--cell-m",type=float,default=2.)
    p.add_argument("--refine-negative",type=Path,
                   help="Refine only negative-scattered OMs from this preserved preview NPZ")
    args=p.parse_args()
    import lighthit as lh
    import numba
    import ntsim
    from ntsim.LightTransport import bgvd_model_from_ntsim
    from lighthit.experimental.charge_preview import charge_preview,ChargePreviewConfig,ChargePreview
    from lighthit.shower_basis import Ge2KernelStore
    if numba.get_num_threads()!=1:
        p.error("requires NUMBA_NUM_THREADS=1")
    sourcefile=paths.muon_source(args.energy)
    with np.load(sourcefile) as z:
        source=lh.SpectralLightElements(**{k:z[k] for k in z.files if k!='metadata'},cone_model='spectral')
        source_metadata=json.loads(str(z['metadata']))
    model=bgvd_model_from_ntsim(ntsim.bgvd.Telescope(n_clusters=(1,2,3,4,5)),ntsim.bgvd.Water())
    if model.fingerprint!=source_metadata['model_fingerprint']:
        p.error('saved model provenance differs')
    full_positions=model.detector.positions_m.copy()
    selected_ids=np.arange(len(full_positions))
    previous_charge=None;previous_flags=None
    if args.refine_negative:
        previous_report=json.loads(args.refine_negative.with_name('report.json').read_text())
        if previous_report['source_sha256']!=hashlib.sha256(sourcefile.read_bytes()).hexdigest():
            p.error('previous preview belongs to a different source')
        names=('near_source','near_compiled_point','outside_all_compiled_points',
               'far_tail_estimate_above_tolerance','negative_scattered_charge','nonfinite')
        with np.load(args.refine_negative) as z:
            previous_charge=z['charge'].copy()
            previous_flags={name:z[name].copy() for name in names}
            if not np.array_equal(z['positions'],full_positions):
                p.error('previous preview detector geometry differs')
        selected_ids=np.flatnonzero(previous_flags['negative_scattered_charge'])
        if not len(selected_ids):
            p.error('previous preview has no negative-scattered OMs to refine')
        d=model.detector
        model=replace(model,detector=replace(d,positions_m=d.positions_m[selected_ids],
              orientations=d.orientations[selected_ids],effective_area_m2=d.effective_area_m2[selected_ids],
              identifiers={k:np.asarray(v)[selected_ids] for k,v in d.identifiers.items()}))
    config=lh.KernelConfig(omega_per_ns=np.linspace(0.,1.2,65),cache_policy='require',
             cache_directory=str(paths.RTE_CACHE.path()))
    kernel=model.kernel(config)
    store=Ge2KernelStore.load(str(paths.GE2_PACK.path()))
    first=str(paths.work('muon-segments-20261009/first-cache/pack65'))
    options=ChargePreviewConfig(track_uid=0,use_detector_tree=args.detector_tree,
              degree=args.degree,azimuthal_degree=args.azimuthal_degree,cell_m=args.cell_m)
    output=args.output or paths.work(f'smart-screen-20261009/preview/{args.energy}')
    output.mkdir(parents=True,exist_ok=True)
    if (output/'preview.npz').exists():
        p.error('preserve existing result and choose another output directory')
    detector=model.detector
    warm_model=replace(model,detector=replace(detector,positions_m=detector.positions_m[:1],
        orientations=detector.orientations[:1],effective_area_m2=detector.effective_area_m2[:1],
        identifiers={k:np.asarray(v)[:1] for k,v in detector.identifiers.items()}))
    warm_source=source.subset(np.arange(len(source))==int(np.flatnonzero(source.uid==0)[0]))
    began=perf_counter()
    charge_preview(warm_model.kernel(config),warm_source,ge2_store=store,
                   first_pack_directory=first,config=options)
    warm_seconds=perf_counter()-began
    print('WARM seconds',warm_seconds,flush=True)
    def checkpoint(info,partial):
        np.savez_compressed(output/'checkpoint.npz',charge_incomplete=partial,
                            selected_module_indices=selected_ids,complete=False)
        (output/'checkpoint.json').write_text(json.dumps(info,indent=2))
        print('CHECKPOINT',json.dumps(info),flush=True)
    def timeout(signum,frame):
        raise TimeoutError('preview CPU limit exceeded')
    signal.signal(signal.SIGPROF,timeout);signal.setitimer(signal.ITIMER_PROF,args.cpu_limit)
    try:
        result=charge_preview(kernel,source,ge2_store=store,first_pack_directory=first,config=options,
                             progress=checkpoint if args.refine_negative else None)
    except TimeoutError:
        (output/'timeout.json').write_text(json.dumps(dict(cpu_limit=args.cpu_limit,warm_seconds=warm_seconds)))
        raise
    finally:
        signal.setitimer(signal.ITIMER_PROF,0)
    if args.refine_negative:
        np.savez_compressed(output/'partial_preview.npz',charge=result.charge_components_pe,
               selected_module_indices=selected_ids,unsafe=result.unsafe_mask,**result.flags)
        merged=previous_charge.copy();merged[selected_ids]=result.charge_components_pe
        flags={name:values.copy() for name,values in previous_flags.items()}
        for name,values in result.flags.items():
            flags[name][selected_ids]=values
        result=ChargePreview(merged,flags,result.metadata)
    # Fine values are loaded AFTER the independent preview, solely for scoring.
    referencefile=paths.muon_response(args.energy)
    with np.load(referencefile) as z:
        reference=z['charge_components_pe'].copy();active=z['active'].copy()
    q=result.charge_orders_pe;threshold=.01;screens={}
    for margin in [2.,4.]:
        prompt=q[:,0]+margin*q[:,1]>=threshold
        total=q[:,0]+margin*(q[:,1]+q[:,2])>=threshold
        positive_total=q[:,0]+margin*(np.maximum(q[:,1],0)+np.maximum(q[:,2],0))>=threshold
        combined=prompt|(q.sum(1)>=threshold)
        for label,mask in [('prompt',prompt),('total',total),('total_nonnegative_scattered',positive_total),
                           ('prompt_with_total_guard',combined)]:
            conservative=mask|result.unsafe
            screens[f'{label}/margin{margin:g}']=dict(
                candidates=int(mask.sum()),missed_reference_active=np.flatnonzero(active&~mask).tolist(),
                added_candidates_vs_reference=np.flatnonzero(mask&~active).tolist(),
                with_unsafe_candidates=int(conservative.sum()),
                with_unsafe_missed_reference_active=np.flatnonzero(active&~conservative).tolist())
    report=dict(energy_GeV=args.energy,source_sha256=hashlib.sha256(sourcefile.read_bytes()).hexdigest(),
        reference_sha256=hashlib.sha256(referencefile.read_bytes()).hexdigest(),
        source_provenance=source_metadata,warm_seconds=warm_seconds,preview=result.metadata,
        refinement_of=str(args.refine_negative) if args.refine_negative else None,
        selected_module_indices=selected_ids.tolist(),
        merged_flag_counts={name:int(value.sum()) for name,value in result.flags.items()},
        reference_active=int(active.sum()),threshold_pe=threshold,screens=screens,
        maximum_original_ballistic_difference_pe=float(np.max(np.abs(q[:,0]-reference[:,0]))),
        negative_q1_count=int((q[:,1]<0).sum()),negative_ge2_count=int((q[:,2]<0).sum()),
        minimum_q1_pe=float(q[:,1].min()),minimum_ge2_pe=float(q[:,2].min()),
        independent_selection=True,fine_reference_loaded_after_preview=True,
        comparison_scope='held-out stored response used only after preview for scoring; no bins computed')
    np.savez_compressed(output/'preview.npz',charge=q,reference_charge=reference,reference_active=active,
                         unsafe=result.unsafe,positions=full_positions,selected_module_indices=selected_ids,
                         **result.flags)
    (output/'report.json').write_text(json.dumps(report,indent=2))
    compact={k:v for k,v in report.items() if k not in ['source_provenance','preview']}
    compact['elapsed_seconds']=result.metadata['elapsed_seconds'];compact['cpu_seconds']=result.metadata['cpu_seconds']
    compact['timings']=result.metadata['timings'];compact['flag_counts']=result.metadata['flag_counts']
    print(json.dumps(compact,indent=2),flush=True)


if __name__=='__main__':
    main()
