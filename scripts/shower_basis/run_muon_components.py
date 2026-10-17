#!/usr/bin/env python3
"""Run the experimental muon-integral/quasi-shower mixture on saved G4 sources."""
import paths  # inputs and outputs: scripts/shower_basis/paths.py
import argparse
from dataclasses import replace
import json
from pathlib import Path
import resource
import sys
from time import perf_counter, process_time

import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--energy",type=int,default=1000)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--om-indices",type=int,nargs="+")
    p.add_argument("--active-only",action="store_true")
    p.add_argument("--partition-ge2",action="store_true")
    p.add_argument("--track-centroid",action="store_true")
    p.add_argument("--shower-axisymmetric",action="store_true",help="test local EM shower m=0; soft/nuclear/late retain M4")
    p.add_argument("--prompt-radius",type=float,default=400.)
    p.add_argument("--soft-cell",type=float,default=5.)
    p.add_argument("--shower-energy",type=float,default=.1)
    p.add_argument("--track-length",type=float,default=1.)
    p.add_argument("--frequencies",type=int,default=33)
    p.add_argument("--time-step-ns",type=float,default=.02)
    p.add_argument("--time-deposit",choices=['linear','nearest','cubic'],default='linear')
    p.add_argument("--repeat",type=int,default=1)
    p.add_argument("--reference-prompt-only",action="store_true",help="test >=2 partition using saved prompt; not a full-event timing")
    p.add_argument("--reuse-prompt",type=Path,help="use this native saved prompt for a >=2-only control; not a full-event timing")
    args=p.parse_args()
    import lighthit as lh
    import ntsim,numba
    from ntsim.LightTransport import bgvd_model_from_ntsim
    from lighthit.experimental.muon_components import transport_muon_components
    base=paths.work('muon-five-cluster-20261008')
    with np.load(paths.muon_source(args.energy)) as z:
        source=lh.SpectralLightElements(**{k:z[k] for k in z.files if k!='metadata'},cone_model='spectral')
    with np.load(paths.muon_ancestry(args.energy)) as z:
        ancestry={k:z[k] for k in z.files if k!='metadata'}
    ref_path=(paths.ntsim_hybrid_response(args.energy)
              if args.energy in (300,1000) else paths.muon_response(3000))
    with np.load(ref_path) as z:
        ref={k:z[k] for k in z.files if k!='metadata'}
        refmeta=json.loads(str(z['metadata']))
    ids=np.asarray(args.om_indices if args.om_indices else
                   np.flatnonzero(ref['active']) if args.active_only else np.arange(len(ref['active'])))
    model=bgvd_model_from_ntsim(ntsim.bgvd.Telescope(n_clusters=(1,2,3,4,5)),ntsim.bgvd.Water())
    d=model.detector
    model=replace(model,detector=replace(d,positions_m=d.positions_m[ids],orientations=d.orientations[ids],
                   effective_area_m2=d.effective_area_m2[ids],identifiers={k:np.asarray(v)[ids] for k,v in d.identifiers.items()}))
    kernel=model.kernel(lh.KernelConfig(omega_per_ns=np.linspace(0.,1.2,65),readout_omega_per_ns=np.linspace(0.,1.2,321),
                          cache_directory=paths.RTE_CACHE.path(),cache_policy='require'))
    kernel.ge2_kernel_store(paths.GE2_PACK.path())
    kwargs=dict(frequency_nodes=args.frequencies,shower_energy_GeV=args.shower_energy,soft_cell_m=args.soft_cell,
                 track_max_length_m=args.track_length,prompt_radius_m=args.prompt_radius,partition_ge2=args.partition_ge2,
                 track_stencil_center='centroid' if args.track_centroid else 'axis')
    kwargs['shower_azimuthal_degree'] = 0 if args.shower_axisymmetric else None
    kwargs['basis'] = lh.ShowerBasis(track_uid=0,spectral_fusion=True,apply_tolerance_pe=1e-6,
        time_step_ns=args.time_step_ns,time_deposit=args.time_deposit)
    if args.reference_prompt_only or args.reuse_prompt:
        # The numerical RTE is linear and independent of prompt; this switch
        # isolates a source-representation control without repeating order1.
        from types import SimpleNamespace
        import lighthit.prompt as prompt_module
        prompt_ref, prompt_meta = ref, refmeta
        if args.reuse_prompt:
            with np.load(args.reuse_prompt) as archive:
                prompt_ref = {k: archive[k] for k in archive.files if k != 'metadata'}
                prompt_meta = json.loads(str(archive['metadata']))
        def saved_prompt(kernel,source,config=None,**kw):
            np.testing.assert_allclose(kw['time_origin_ns'],prompt_ref['time_origin_ns'][ids],rtol=0,atol=1e-10)
            return SimpleNamespace(relative_time_edges_ns=prompt_ref['relative_time_edges_ns'],time_origin_ns=prompt_ref['time_origin_ns'][ids],
                 charge_orders_pe=prompt_ref['charge_components_pe'][ids,:2],bins_orders_pe=prompt_ref['components_pe'][ids,:,:2],
                 charge_pe=prompt_ref['charge_components_pe'][ids,:2].sum(1),computed=np.asarray(prompt_meta['prompt_order1_level'])[ids]==2,
                 metadata=dict(prompt_algorithm='reused',order1_level=np.asarray(prompt_meta['prompt_order1_level'])[ids].tolist()))
        prompt_module.transport_prompt=saved_prompt
    args.output.mkdir(parents=True,exist_ok=True)
    print('receivers',len(ids),'threads',numba.get_num_threads(),'settings',kwargs,flush=True)
    for trial in range(args.repeat):
        start,cpu=perf_counter(),process_time()
        response=transport_muon_components(kernel,source,ancestry,**kwargs)
        wall,cpu=perf_counter()-start,process_time()-cpu
        response.save(args.output/f'response-{trial}.npz')
        union=response.active|ref['active'][ids]
        comparison={}
        for label,cols in [('0',[0]),('1',[1]),('ge2',[2]),('total',[0,1,2])]:
            a=ref['charge_components_pe'][ids][:,cols].sum(1);b=response.charge_components_pe[:,cols].sum(1)
            ab=ref['components_pe'][ids][:,:,cols].sum(2);bb=response.components_pe[:,:,cols].sum(2)
            comparison[label]=dict(charge_max_abs=float(np.max(abs(a-b))),
                 active_charge_max_rel=float(np.max(abs(a[union]-b[union])/np.maximum(abs(a[union]),1e-30),initial=0.)),
                 active_bins_l1_max_rel=float(np.max(abs(ab[union]-bb[union]).sum(1)/np.maximum(abs(ab[union]).sum(1),1e-30),initial=0.)),
                 active_bins_l1_global=float(abs(ab[union]-bb[union]).sum()/max(abs(ab[union]).sum(),1e-30)))
        report=dict(wall_seconds=wall,cpu_seconds=cpu,peak_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(2**20 if sys.platform=='darwin' else 2**10),
                    threads=numba.get_num_threads(),om_indices=ids.tolist(),settings=kwargs,reference=str(ref_path),
                    prompt_reused=bool(args.reference_prompt_only or args.reuse_prompt),
                    prompt_reused_from=str(args.reuse_prompt or ref_path) if args.reference_prompt_only or args.reuse_prompt else None,
                    active=int(response.active.sum()),
                    lost=int(np.sum(ref['active'][ids]&~response.active)),gained=int(np.sum(~ref['active'][ids]&response.active)),
                    comparison=comparison,metadata=response.metadata)
        (args.output/f'response-{trial}.json').write_text(json.dumps(report,indent=2,default=str))
        print(json.dumps({k:v for k,v in report.items() if k not in ('metadata','om_indices')},indent=2,default=str),flush=True)


if __name__=='__main__':main()
