#!/usr/bin/env python3
"""Paired Q1 timing on identical saved steps, optics, OMs and source clocks.

This measures the prompt stage only. A full-event >=2 timing and the photon
MC accuracy criterion remain separate checks before changing data manifests.
"""
from pathlib import Path
from dataclasses import replace
from time import perf_counter
import argparse,hashlib,json
import numpy as np


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--steps',type=Path,required=True)
    p.add_argument('--first-entry-tables',type=Path,required=True);p.add_argument('--reference',type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--receivers',nargs='+',type=int)
    p.add_argument('--threads',type=int,default=1);p.add_argument('--threshold',type=float,default=.01)
    args=p.parse_args(argv)
    import numba
    import lighthit as lh
    from lighthit.shower import ShowerConfig
    from lighthit.receiver_vertex import ReceiverVertexConfig
    numba.set_num_threads(args.threads)
    model=lh.load_model(args.model)
    with np.load(args.steps,allow_pickle=False) as data:
        fields={k:data[k] for k in data.files if k!='metadata'}
    source=lh.SpectralLightElements(**fields,cone_model='spectral')
    ids=np.arange(len(model.detector)) if args.receivers is None else np.asarray(args.receivers,int)
    detector=model.detector.subset(ids);origins=extra=None
    if args.reference is not None:
        with np.load(args.reference,allow_pickle=False) as data:
            origins=data['time_origin_ns'][ids];extra=data['charge_components_pe'][ids,2]
    args.output.mkdir(parents=True,exist_ok=True)
    config=lh.KernelConfig(omega_per_ns=np.linspace(0.,1.2,65),readout_omega_per_ns=np.linspace(0.,1.2,321),
        relative_time_edges_ns=np.arange(-60.,745.,5.),threshold_pe=args.threshold,cache_policy='require')
    record=dict(model_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(),
        steps_sha256=hashlib.sha256(args.steps.read_bytes()).hexdigest(),receivers=ids.tolist(),threads=args.threads,
        threshold_pe=args.threshold,scope='prompt only; same saved >=2 charge used only for OM selection',runs=[])
    for reference in ['centre','first-entry']:
        kernel=lh.TransportKernel(model.medium,detector,replace(config,scattered_receiver=reference,
            first_entry_tables=args.first_entry_tables,receiver_vertex=ReceiverVertexConfig(angular_pairs=96,radial_nodes=400)))
        for repeat in range(2):
            start=perf_counter()
            response=kernel.transport_prompt(source,config=ShowerConfig().prompt,
                time_origin_ns=origins,selection_extra_charge=extra)
            wall=perf_counter()-start
            row=dict(receiver_reference=reference,repeat=repeat,wall_seconds=wall,
                timings=response.metadata.get('timings'),charge_by_order=response.charge_orders_pe.sum(0).tolist(),
                order1_level=np.asarray(response.metadata['order1_level']).tolist())
            record['runs'].append(row);print(json.dumps(row),flush=True)
            if repeat==1:response.save(args.output/(reference+'.npz'))
    (args.output/'timings.json').write_text(json.dumps(record,indent=2)+'\n')
    return 0


if __name__=='__main__':raise SystemExit(main())
