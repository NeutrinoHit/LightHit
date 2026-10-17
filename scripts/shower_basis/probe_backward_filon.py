#!/usr/bin/env python3
"""Same-RTE Filon check for actual backward fallback source/OM intervals."""
import paths  # inputs and outputs: scripts/shower_basis/paths.py
import argparse,json,hashlib
from pathlib import Path
from time import process_time,perf_counter
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=paths.holdout_primary())
    p.add_argument('--result',type=Path,default=paths.work('ge2-latency-20261010/profile43'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--orders',type=int,nargs='+',default=[6,8,10])
    args=p.parse_args()
    if args.output.exists():p.error('fresh output required')
    import lighthit as lh,ntsim,numba
    from ntsim.LightTransport import bgvd_model_from_ntsim
    from lighthit.experimental.primary_polyline import fit_primary_polyline
    from lighthit.experimental.track_ge2_primitive import DiskTrackGe2Atlas
    from lighthit.experimental.universal_primary import DirectGe2
    from lighthit.experimental.filon_ge2 import backward_ge2_filon_many
    from lighthit.readout import inverse_bins
    from scipy.interpolate import CubicSpline
    if numba.get_num_threads()!=1:p.error('one core required')
    saved=json.loads((args.result/'report.json').read_text())
    if saved['source_sha256']!=hashlib.sha256(args.source.read_bytes()).hexdigest():raise ValueError('different source')
    with np.load(args.source) as z:
        source=lh.SpectralLightElements(**{k:z[k] for k in z.files if k!='metadata'},cone_model='spectral')
    source=source.subset(source.uid==0)
    model=bgvd_model_from_ntsim(ntsim.bgvd.Telescope(n_clusters=(1,2,3,4,5)),ntsim.bgvd.Water())
    kernel=model.kernel(lh.KernelConfig(omega_per_ns=np.linspace(0,1.2,65),cache_policy='require',
        cache_directory=str(paths.RTE_CACHE.path())))
    pieces=fit_primary_polyline(source,phase_indices=kernel.medium.phase_index).elements
    atlas=DiskTrackGe2Atlas(str(paths.GE2_ATLAS.path()));atlas.validate_kernel(kernel)
    with np.load(args.result/'response.npz') as z:
        origins=z['time_origin_ns'];np.testing.assert_array_equal(z['positions_m'],kernel.detector.positions_m)
    computed=np.asarray(saved['metadata']['time_computed_mask'])
    alpha=atlas.metadata['clock_ns_per_m'];queries=[];labels=[];excluded=0
    for om in np.flatnonzero(computed):
        for i in range(len(pieces)):
            axis=pieces.direction[i];delta=kernel.detector.positions_m[om]-pieces.start_m[i]
            z=float(delta@axis);rad=delta-z*axis;rho=float(np.linalg.norm(rad))
            if not 3.01<=rho<300:continue
            reach=np.sqrt(300**2-rho*rho);a=min(z,reach);b=max(z-pieces.length_m[i],-reach)
            if a<=b:continue
            clock=float((pieces.end_ns[i]-pieces.start_ns[i])/pieces.length_m[i]);beta=float(pieces.beta[i])
            charge_cached=(atlas.impact_nodes_m[0]<=rho<=atlas.impact_nodes_m[-1] and abs(beta-1)<=1e-5
                and atlas.query_supported(rho,a,b,charge_only=True))
            direct=(not charge_cached or abs(clock-alpha)*(a-b)/2>1e-3 or
                not atlas.query_supported(rho,a,b,charge_only=False))
            if not direct:continue
            if a>0:excluded+=1;continue
            ex=rad/rho;ey=np.cross(axis,ex);look=-kernel.detector.orientations[om]
            local=np.array([look@ex,look@ey,look@axis]);density=pieces.coefficient0[i]/pieces.length_m[i]
            closest=float(pieces.start_ns[i]+z*clock-origins[om])
            queries.append((rho,local,beta,clock,b,a,density,closest));labels.append(dict(om=int(om),piece=i))
    if not queries:raise ValueError('no backward fallback intervals')
    direct=DirectGe2(kernel,str(paths.GE2_PACK.path()),quadrature='adaptive')
    aa=lambda q:(*q,True)
    direct(*aa(queries[0]))
    backward_ge2_filon_many(kernel,direct.bundle,queries[:1],order=6)
    cpu,wall=process_time(),perf_counter()
    reference=np.stack([direct(*aa(q)) for q in queries]);refcpu=process_time()-cpu;refwall=perf_counter()-wall
    outomega=np.linspace(0,1.2,321);edges=kernel.config.relative_time_edges_ns
    def bins(spectra):return inverse_bins(outomega,CubicSpline(direct.bundle['omega'],spectra.T,axis=0)(outomega),edges)
    bref=bins(reference);args.output.mkdir(parents=True)
    report=dict(primary_only=True,threads=1,source_sha256=saved['source_sha256'],
        queries=[dict(**label,arguments=[q[0],q[1].tolist(),*q[2:]]) for label,q in zip(labels,queries)],
        forward_fallbacks_excluded=excluded,reference_cpu_seconds=refcpu,reference_wall_seconds=refwall,
        same_process_peak_rss_not_comparable=True,runs=[])
    np.savez_compressed(args.output/'reference.npz',spectrum=reference,bins=bref)
    for order in args.orders:
        cpu,wall=process_time(),perf_counter();result,meta=backward_ge2_filon_many(kernel,direct.bundle,queries,order=order)
        elapsed=process_time()-cpu;seconds=perf_counter()-wall;bi=bins(result)
        rows=[]
        for j,label in enumerate(labels):
            rows.append(dict(**label,reference_charge_pe=float(reference[j,0].real),
                charge_absolute_error_pe=float(abs(result[j,0].real-reference[j,0].real)),
                bins_absolute_l1_pe=float(abs(bi[j]-bref[j]).sum()),
                spectral_relative_error=float(abs(result[j]-reference[j]).max()/max(abs(reference[j]).max(),1e-30))))
        row=dict(order=order,cpu_seconds=elapsed,wall_seconds=seconds,cpu_speedup=refcpu/elapsed,
            metadata=meta,per_query=rows,max_bin_absolute_l1_pe=max(r['bins_absolute_l1_pe'] for r in rows))
        report['runs'].append(row)
        np.savez_compressed(args.output/f'order{order}.npz',spectrum=result,bins=bi)
        (args.output/'report.json').write_text(json.dumps(report,indent=2))
        print('FILON',json.dumps({k:v for k,v in row.items() if k!='per_query'}),flush=True)


if __name__=='__main__':main()
