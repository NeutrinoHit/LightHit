#!/usr/bin/env python3
"""Independent preview + adaptive fine muon transport, five clusters/one thread.

A saved fine reference is opened only AFTER transport for validation. Coarse
charges on unselected OMs are explicitly labelled, not counted as fine results.
"""
import paths  # inputs and outputs: scripts/shower_basis/paths.py
import argparse,json,resource,sys
from pathlib import Path
from time import perf_counter,process_time
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--energy',type=int,default=1000)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--margin',type=float,default=4.)
    args=p.parse_args()
    import lighthit as lh,ntsim,numba
    from ntsim.LightTransport import bgvd_model_from_ntsim
    from lighthit.experimental.adaptive_screen import transport_smart_muon,AdaptiveScreenConfig
    if numba.get_num_threads()!=1:p.error('use NUMBA_NUM_THREADS=1')
    if (args.output/'response.npz').exists():p.error('preserve old output; choose another directory')
    base=paths.work('muon-five-cluster-20261008')
    with np.load(paths.muon_source(args.energy)) as z:
        source=lh.SpectralLightElements(**{k:z[k] for k in z.files if k!='metadata'},cone_model='spectral')
        source_meta=json.loads(str(z['metadata']))
    with np.load(paths.muon_ancestry(args.energy)) as z:
        ancestry={k:z[k] for k in z.files if k!='metadata'}
    model=bgvd_model_from_ntsim(ntsim.bgvd.Telescope(n_clusters=(1,2,3,4,5)),ntsim.bgvd.Water())
    if source_meta['model_fingerprint']!=model.fingerprint:raise ValueError('source/model mismatch')
    kernel=model.kernel(lh.KernelConfig(omega_per_ns=np.linspace(0,1.2,65),
        readout_omega_per_ns=np.linspace(0,1.2,321),cache_policy='require',
        cache_directory=str(paths.RTE_CACHE.path())))
    kernel.ge2_kernel_store(str(paths.GE2_PACK.path()))
    args.output.mkdir(parents=True,exist_ok=True)
    print('independent preview + fine transport',args.energy,'GeV',len(model.detector.positions_m),'OMs',flush=True)
    start,cpu=perf_counter(),process_time()
    response=transport_smart_muon(kernel,source,ancestry,
        first_pack_directory=str(paths.work('muon-segments-20261009/first-cache/pack65')),
        screen_config=AdaptiveScreenConfig(scatter_margin=args.margin))
    elapsed,cpu=perf_counter()-start,process_time()-cpu
    response.save(args.output/'response.npz')
    # Reference is not visible to the selector or evaluator above.
    reference=(paths.ntsim_hybrid_response(1000, 'all', 'centroid') if args.energy==1000 else
          paths.ntsim_hybrid_response(args.energy, 'all'))
    with np.load(reference) as z:
        q=z['charge_components_pe'];b=z['components_pe'];active=z['active']
        np.testing.assert_allclose(z['time_origin_ns'],response.time_origin_ns,rtol=0,atol=1e-10)
    union=active|response.active; comparison={}
    for name,indices in [('0',[0]),('1',[1]),('ge2',[2]),('total',[0,1,2])]:
        qr=q[:,indices].sum(1);qc=response.charge_components_pe[:,indices].sum(1)
        br=b[:,:,indices].sum(2);bc=response.components_pe[:,:,indices].sum(2)
        comparison[name]=dict(active_charge_max_abs=float(np.max(abs(qr[union]-qc[union]),initial=0)),
            active_charge_max_rel=float(np.max(abs(qr[union]-qc[union])/np.maximum(abs(qr[union]),1e-30),initial=0)),
            active_bins_l1_max_rel=float(np.max(abs(br[union]-bc[union]).sum(1)/np.maximum(abs(br[union]).sum(1),1e-30),initial=0)),
            active_bins_l1_global=float(abs(br[union]-bc[union]).sum()/max(abs(br[union]).sum(),1e-30)))
    report=dict(energy_GeV=args.energy,wall_seconds=elapsed,cpu_seconds=cpu,
        peak_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(2**20 if sys.platform=='darwin' else 2**10),
        threads=numba.get_num_threads(),modules=len(model.detector.positions_m),
        active=int(response.active.sum()),lost=np.flatnonzero(active&~response.active).tolist(),
        gained=np.flatnonzero(~active&response.active).tolist(),comparison=comparison,
        reference=str(reference),metadata=response.metadata)
    (args.output/'report.json').write_text(json.dumps(report,indent=2,default=str))
    print(json.dumps({k:v for k,v in report.items() if k!='metadata'},indent=2),flush=True)
    print('screen',json.dumps({k:v for k,v in response.metadata['smart_screen'].items()
        if k not in ('preview','rounds')},indent=2),flush=True)


if __name__=='__main__':main()
