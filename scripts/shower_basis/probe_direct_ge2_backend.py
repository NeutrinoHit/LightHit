#!/usr/bin/env python3
"""One-core, same-quadrature comparison of two direct RTE contractions.

Saved G4 primary, actual beta and emission clock. Both methods use the same
finite RTE pack; this is an implementation check, not independent physics.
"""
import paths  # inputs and outputs: scripts/shower_basis/paths.py
import argparse,json
from pathlib import Path
from time import process_time,perf_counter
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--energy',type=int,default=300)
    p.add_argument('--oms',type=int,nargs='+',default=[197,195,1208])
    p.add_argument('--pairs-per-om',type=int,default=8)
    args=p.parse_args()
    if args.output.exists():p.error('fresh output required')
    import lighthit as lh,ntsim,numba
    from ntsim.LightTransport import bgvd_model_from_ntsim
    from lighthit.experimental.universal_primary import DirectGe2
    from lighthit.experimental.primary_polyline import fit_primary_polyline
    from lighthit.readout import inverse_bins
    from scipy.interpolate import CubicSpline
    if numba.get_num_threads()!=1:p.error('one core required')
    with np.load(paths.muon_source(args.energy)) as z:
        source=lh.SpectralLightElements(**{k:z[k] for k in z.files if k!='metadata'},cone_model='spectral')
    model=bgvd_model_from_ntsim(ntsim.bgvd.Telescope(n_clusters=(1,2,3,4,5)),ntsim.bgvd.Water())
    kernel=model.kernel(lh.KernelConfig(omega_per_ns=np.linspace(0,1.2,65),cache_policy='require',
        cache_directory=str(paths.RTE_CACHE.path())))
    pieces=fit_primary_polyline(source.subset(source.uid==0),phase_indices=kernel.medium.phase_index).elements
    pack=str(paths.GE2_PACK.path())
    methods={name:DirectGe2(kernel,pack,backend=backend,quadrature=quadrature)
        for name,backend,quadrature in [('generic','generic','fixed12'),('fused-looks','fused-looks','fixed12'),
            ('adaptive','fused-looks','adaptive'),('refined','fused-looks','fixed24')]}
    with np.load(str(paths.work(f'primary-universal-20261009/results/seed42-{args.energy}/response.npz'))) as z:
        origins=z['time_origin_ns']
    rows=[];arguments=[]
    for om in args.oms:
        pairs=[]
        for i in range(len(pieces)):
            axis=pieces.direction[i];d=kernel.detector.positions_m[om]-pieces.start_m[i]
            z=float(d@axis);rad=d-z*axis;rho=float(np.linalg.norm(rad))
            if not 3.01<=rho<300:continue
            reach=np.sqrt(300**2-rho**2);a=min(z,reach);b=max(z-pieces.length_m[i],-reach)
            if a<=b:continue
            ex=rad/rho;ey=np.cross(axis,ex);normal=-kernel.detector.orientations[om]
            look=np.array([normal@ex,normal@ey,normal@axis]);clock=(pieces.end_ns[i]-pieces.start_ns[i])/pieces.length_m[i]
            pairs.append((i,(rho,look,float(pieces.beta[i]),clock,b,a,pieces.coefficient0[i]/pieces.length_m[i],
                float(pieces.start_ns[i]+z*clock-origins[om]),True)))
        # Include early/middle/stopping sources, including below-threshold bands.
        count=min(len(pairs),args.pairs_per_om) if args.pairs_per_om>0 else len(pairs)
        for j in np.unique(np.linspace(0,len(pairs)-1,count).astype(int)):
            i,aa=pairs[j];arguments.append(aa)
            rows.append(dict(om=om,piece=i,beta=aa[2],rho_m=aa[0],span_m=aa[5]-aa[4]))
    if not arguments:raise ValueError('no supported source/receiver pairs')
    for method in methods.values():method(*arguments[0])
    outputs={};times={}
    for name,method in methods.items():
        cpu,wall=process_time(),perf_counter()
        outputs[name]=np.stack([method(*aa) for aa in arguments])
        times[name]=dict(cpu_seconds=process_time()-cpu,wall_seconds=perf_counter()-wall)
    a=outputs['generic'];b=outputs['fused-looks'];delta=abs(a-b)
    for j,row in enumerate(rows):
        row['max_relative_spectral_error']=float(delta[j].max()/max(abs(a[j]).max(),1e-30))
        row['charge_absolute_error_pe']=float(abs(a[j,0].real-b[j,0].real))
    outomega=np.linspace(0,1.2,321);edges=kernel.config.relative_time_edges_ns
    bins={name:inverse_bins(outomega,CubicSpline(np.linspace(0,1.2,65),value.T,axis=0)(outomega),edges)
        for name,value in outputs.items()}
    ba=bins['generic'];bb=bins['fused-looks']
    aggregate=[]
    for om in args.oms:
        mask=np.array([r['om']==om for r in rows]);item=dict(om=om,pairs=int(mask.sum()),variants={})
        q=outputs['refined'][mask].sum(0)[0].real;br=bins['refined'][mask].sum(0)
        for name,value in outputs.items():
            value=value[mask].sum(0);bi=bins[name][mask].sum(0)
            item['variants'][name]=dict(charge_pe=float(value[0].real),
                charge_relative_error=float(abs(value[0].real-q)/max(abs(q),1e-30)),
                bins_relative_l1=float(abs(bi-br).sum()/max(abs(br).sum(),1e-30)))
        aggregate.append(item)
    report=dict(primary_only=True,energy_GeV=args.energy,threads=1,measurements=times,pairs=rows,
        max_relative_spectral_error=max(r['max_relative_spectral_error'] for r in rows),
        max_relative_bins_l1=float(np.max(abs(ba-bb).sum(1)/np.maximum(abs(ba).sum(1),1e-30))),
        aggregate=aggregate,
        cpu_speedup=times['generic']['cpu_seconds']/times['fused-looks']['cpu_seconds'],
        adaptive_cpu_speedup=times['generic']['cpu_seconds']/times['adaptive']['cpu_seconds'])
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output/'spectra.npz',**outputs)
    (args.output/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
