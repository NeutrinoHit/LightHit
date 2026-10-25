#!/usr/bin/env python3
"""Full paired shower check on exact saved photon-MC steps.

Each timed call includes source compilation, all collision orders, selection
and readout. JIT/table setup is reported separately; no compiled event source
is reused. The existing MC runs and their distance-bin criterion are retained.
"""
from pathlib import Path
from time import perf_counter
import argparse,hashlib,json
import numpy as np


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--steps',type=Path,required=True)
    p.add_argument('--centre-data',type=Path,required=True);p.add_argument('--surface-data',type=Path,required=True)
    p.add_argument('--mc-record',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--threads',type=int,default=1)
    p.add_argument('--repeats',type=int,default=2,help='one call may be used with an already warmed installation')
    p.add_argument('--kind',choices=['shower','muon'],default='shower')
    p.add_argument('--parts',type=Path,help='saved ancestry for a single-primary muon event')
    p.add_argument('--only-receiver',choices=['centre','first-entry'])
    p.add_argument('--profile-order1',action='store_true',help='record spectral-unit and grouping timings')
    args=p.parse_args(argv)
    if args.repeats<1:p.error('--repeats must be positive')
    import lighthit as lh
    import numba
    from lighthit.shower import ShowerTransport
    from lighthit.shower.api import FREQUENCIES_PER_NS,READOUT_PER_NS
    from harness.cases_mcmatrix import _nearest
    numba.set_num_threads(args.threads)
    unit_profiles=[]
    if args.profile_order1:
        from lighthit import receiver_prompt
        evaluate=receiver_prompt.grouped_order1
        def profiled(*values,**options):
            answer=evaluate(*values,**options);meta=answer[2]
            fine=list(meta['order1_surface_numerics'].values())
            unit_profiles.append(dict(source_elements=len(values[1]),bands=len(values[8]['mut']),
                timings=meta['timings'],groups=meta['order1_groups_total'],
                fine_group_seconds=sum(v['group_seconds'] for v in fine),
                fine_lookup_seconds=sum(v['lookup_seconds'] for v in fine),
                fine_groups=sum(v['groups'] for v in fine),
                source_pencils=max((v['source_pencils'] for v in fine),default=None)))
            return answer
        receiver_prompt.grouped_order1=profiled
    source_sha=hashlib.sha256(args.steps.read_bytes()).hexdigest();mc=json.loads(args.mc_record.read_text())
    if source_sha not in mc['inputs'].values():raise ValueError('source differs from the saved MC event')
    if not mc['metrics'].get('same_steps_in_all_runs',False):raise ValueError('MC runs must share the same charged event')
    with np.load(args.steps,allow_pickle=False) as data:
        source=lh.SpectralLightElements(**{k:data[k] for k in data.files if k!='metadata'},cone_model='spectral')
        source_metadata=json.loads(str(data['metadata'])) if 'metadata' in data.files else {}
    ancestry=None;primary_uid=0
    if args.kind=='muon':
        if args.parts is None:p.error('--parts is required for a muon')
        parts=source_metadata.get('parts',[])
        if len(parts)!=1 or parts[0]['kind']!='muon':p.error('a single-primary muon event is required')
        primary_uid=int(parts[0]['primary_uid'])
        with np.load(args.parts,allow_pickle=False) as data:
            if not np.array_equal(data['mask_0'],np.arange(len(source))):raise ValueError('ancestry does not cover the entire event')
            ancestry={k[len('ancestry_0_'):]:data[k] for k in data.files if k.startswith('ancestry_0_')}
    model=lh.load_model(args.model)
    config=lh.KernelConfig(omega_per_ns=FREQUENCIES_PER_NS,readout_omega_per_ns=READOUT_PER_NS)
    args.output.mkdir(parents=True,exist_ok=True);timings=[];answers={}
    for label,data in [('centre',args.centre_data),('first-entry',args.surface_data)]:
        if args.only_receiver is not None and label!=args.only_receiver:continue
        setup=perf_counter()
        if args.kind=='shower':engine=ShowerTransport(model.kernel(config),data,threads=args.threads)
        else:
            from lighthit.muon import MuonTransport
            engine=MuonTransport(model.kernel(config),data,threads=args.threads)
        engine.tables()
        setup=perf_counter()-setup
        for repeat in range(args.repeats):
            if hasattr(engine,'_compiled'):engine._compiled.clear()
            unit_profiles.clear()
            start=perf_counter()
            response=engine(source) if ancestry is None else engine(source,ancestry,primary_uid=primary_uid)
            elapsed=perf_counter()-start
            stage_timings=response.metadata.get('timings')
            if stage_timings is None and 'batches' in response.metadata:
                stage_timings=dict(ballistic_seconds=response.metadata['ballistic_seconds'],
                    primary_seconds=sum(b['primary']['elapsed_seconds'] for b in response.metadata['batches']),
                    secondary_seconds=sum(b['secondary']['wall_s'] for b in response.metadata['batches']))
            row=dict(receiver=label,repeat=repeat,seconds=elapsed,table_setup_seconds=setup,
                source_compilation_included=True,active=int(response.active.sum()),charge_by_order=response.charge_components_pe.sum(0).tolist(),
                timings=stage_timings,prompt_timings=response.metadata.get('prompt_timings'),
                receiver_reference_by_order=response.metadata.get('receiver_reference_by_order'))
            if args.profile_order1:row['order1_units']=list(unit_profiles)
            timings.append(row);print(json.dumps(row),flush=True)
            if repeat==args.repeats-1:response.save(args.output/(label+'.npz'));answers[label]=response
    distance=_nearest(source,model.detector.positions_m)
    result=dict(source_sha256=source_sha,model_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(),
        threads=args.threads,kind=args.kind,same_charged_event_in_mc=True,
        parts_sha256=None if args.parts is None else hashlib.sha256(args.parts.read_bytes()).hexdigest(),
        timings=timings,criterion='max(3%, 2 MC standard errors); distance groups use existing MC significance',
        totals={},by_distance=[])
    for label,response in answers.items():
        result['totals'][label]={}
        for j,order in enumerate(['0','1','>=2','total']):
            value=float(response.charge_components_pe[:,j].sum() if j<3 else response.charge_components_pe.sum())
            old=mc['metrics']['totals'][order];mean=old['mc_pe'];sigma=old['mc_sigma_pe'];ratio=value/mean if mean else None
            result['totals'][label][order]=dict(prediction_pe=value,mc_pe=mean,mc_sigma_pe=sigma,ratio=ratio,
                passed=None if ratio is None else abs(ratio-1)<=max(.03,2*ratio*sigma/mean))
    for row in mc['metrics']['by_distance']:
        low,high=row['distance_m'];mask=(distance>=low)&(distance<(np.inf if high is None else high))
        if int(mask.sum())!=row['oms']:raise ValueError('distance grouping differs from saved MC')
        group=dict(distance_m=row['distance_m'],oms=row['oms'],tested=row['tested'],orders={})
        for label,response in answers.items():
            group['orders'][label]={}
            for j,order in enumerate(['0','1','>=2','total']):
                value=float(response.charge_components_pe[mask,j].sum() if j<3 else response.charge_components_pe[mask].sum())
                old=row[order];mean=old['mc_pe'];sigma=old['mc_sigma_pe'];ratio=value/mean if mean else None
                group['orders'][label][order]=dict(prediction_pe=value,mc_pe=mean,mc_sigma_pe=sigma,ratio=ratio,
                    significant=old['mc_share']>.001,passed=None if ratio is None else abs(ratio-1)<=max(.03,2*ratio*sigma/mean))
        result['by_distance'].append(group)
    if len(answers)==2:
        x,y=answers['centre'],answers['first-entry']
        result['paired']=dict(same_origins=bool(np.array_equal(x.time_origin_ns,y.time_origin_ns)),
            same_q0_charge=bool(np.array_equal(x.charge_components_pe[:,0],y.charge_components_pe[:,0])),
            old_active=int(x.active.sum()),new_active=int(y.active.sum()))
    else:result['paired']=None
    (args.output/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(totals=result['totals'],paired=result['paired'])),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
