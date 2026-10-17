#!/usr/bin/env python3
"""Bounded Q1 path-quadrature checks on saved actual residual source groups."""
import paths  # inputs and outputs: scripts/shower_basis/paths.py
import argparse,json
from pathlib import Path
from dataclasses import replace
from time import process_time
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--groups',type=Path,default=paths.work('q1-stationary-20261010'))
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():p.error('fresh output required')
    import numba,lighthit as lh,ntsim
    from ntsim.LightTransport import bgvd_model_from_ntsim
    from lighthit.experimental.first_charge_only import first_scattering_charge_only
    from lighthit.experimental.primary_polyline import primary_spectral_quadrature
    from lighthit.experimental.spectral_response_basis import ResponseCubature
    from lighthit.prompt import PromptConfig
    if numba.get_num_threads()!=1:p.error('one core required')
    rows=json.loads((args.groups/'groups.json').read_text())
    cube=ResponseCubature.load(str(paths.work('joint-response-20261010/basis-rank16.npz')))
    model=bgvd_model_from_ntsim(ntsim.bgvd.Telescope(n_clusters=(1,2,3,4,5)),ntsim.bgvd.Water())
    kernel=model.kernel(lh.KernelConfig(omega_per_ns=np.array([0.]),cache_policy='require'))
    indices=set()
    for nodes in [17,82]:
        ids=[i for i,r in enumerate(rows) if r['nodes']==nodes]
        indices.update(sorted(ids,key=lambda i:rows[i]['cpu_s'],reverse=True)[:2])
        indices.update(sorted(ids,key=lambda i:rows[i]['charge_pe'],reverse=True)[:2])
        indices.update(np.random.default_rng(nodes).choice(ids,min(3,len(ids)),replace=False).tolist())
    saved=np.load(args.groups/'groups.npz');report=[]
    args.output.mkdir(parents=True)
    for index in sorted(indices):
        row=rows[index];lo,hi=saved['offsets'][index:index+2]
        source=lh.SpectralLightElements(**{k:saved[k][lo:hi] for k in saved.files if k!='offsets'},cone_model='spectral')
        from copy import copy
        local=copy(kernel);d=kernel.detector
        local.detector=replace(d,positions_m=np.array([row['position']]),orientations=np.array([row['normal']]),
            effective_area_m2=d.effective_area_m2[:1],identifiers={k:v[:1] for k,v in d.identifiers.items()})
        local.wavelength=cube.quadrature if row['nodes']==17 else primary_spectral_quadrature(kernel.medium,kernel.detector,order=2)
        # Warm the changed compiled kernel before timing, on one source record.
        first_scattering_charge_only(local,source.subset(np.arange(len(source))==0),gauss_order=6)
        item=dict(index=index,source=row,runs=[])
        for order,width in [(0,1.),(4,1.5),(6,1.5),(8,1.),(10,.5)]:
            cpu=process_time();q,meta=first_scattering_charge_only(local,source,gauss_order=order,panel_width=width)
            item['runs'].append(dict(order=order,panel_width=width,cpu_s=process_time()-cpu,charge_pe=float(q[0])))
        ref=item['runs'][-1]['charge_pe']
        for run in item['runs']:
            run['relative_error']=float(abs(run['charge_pe']-ref)/max(abs(ref),1e-30))
            run['absolute_error_pe']=float(abs(run['charge_pe']-ref))
        report.append(item);print('CHECK',json.dumps(item),flush=True)
        (args.output/'report.json').write_text(json.dumps(dict(threads=1,primary_only=True,groups=report,
            reference='order10 width0.5 in asinh optical path; same source/azimuth/spectrum quadratures'),indent=2))


if __name__=='__main__':main()
