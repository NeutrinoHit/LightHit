#!/usr/bin/env python3
"""Paired backward-MC beta correction at the complete reference spectrum.

The beta-one integral is computed directly for this validation experiment.
Its preparation cost is reported separately: a production hybrid would need
a reusable base response, and cannot count this event-specific integral as free.
Both estimators use identical proposals, conservative union trees and random
streams. Batch sums retain cross-wavelength and actual/base covariance.
"""
import paths  # inputs and outputs: scripts/shower_basis/paths.py
import argparse,json,math
from pathlib import Path
from copy import copy
from dataclasses import replace
from time import perf_counter,process_time
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--samples',type=int,nargs='+',default=[4096,16384])
    args=p.parse_args()
    if args.output.exists():p.error('choose unused output')
    import numba,lighthit as lh,ntsim
    from ntsim.LightTransport import bgvd_model_from_ntsim
    from lighthit.experimental.primary_polyline import fit_primary_polyline,primary_spectral_quadrature
    from lighthit.experimental.universal_primary import subset_kernel
    from lighthit.experimental import prompt_backward as backward
    from lighthit.experimental.muon_integral import order1_integrals
    from lighthit.prompt import _element_segments,_spectral_tables,acceptance_polynomial,acceptance_monomials,PromptConfig
    from lighthit.model import WavelengthQuadrature
    if numba.get_num_threads()!=1:p.error('one thread required')
    batches=32;ids=np.array([197,195,1208])
    with np.load(paths.muon_source(300)) as z:
        source=lh.SpectralLightElements(**{k:z[k] for k in z.files if k!='metadata'},cone_model='spectral')
    source=source.subset(source.uid==0)
    model=bgvd_model_from_ntsim(ntsim.bgvd.Telescope(n_clusters=(1,2,3,4,5)),ntsim.bgvd.Water())
    kernel=model.kernel(lh.KernelConfig(omega_per_ns=np.array([0.]),cache_policy='require'))
    pieces=fit_primary_polyline(source,phase_indices=kernel.medium.phase_index).elements
    local=subset_kernel(kernel,ids);local.wavelength=primary_spectral_quadrature(kernel.medium,kernel.detector,order=2)
    tables=_spectral_tables(local);looks=np.ascontiguousarray(-local.detector.orientations)
    frames=np.ascontiguousarray(np.stack([backward._frame(v) for v in looks]))
    areas=np.ascontiguousarray(local.detector.effective_area_m2)
    apoly=acceptance_polynomial(*acceptance_monomials(local.acceptance()['alpha']))
    if np.polynomial.polynomial.polyval(np.linspace(-1,1,20001),apoly).min()<0:raise ValueError('negative proposal acceptance')
    integral=np.polynomial.polynomial.polyint(apoly)
    cdf=float(np.polynomial.polynomial.polyval(1.,integral)-np.polynomial.polynomial.polyval(-1.,integral))
    base=paths.work('primary-universal-20261009/results/seed42-300')
    with np.load(base/'response.npz') as z:origins=np.ascontiguousarray(z['time_origin_ns'][ids])
    qref=[];bref=[]
    for om in ids:
        with np.load(base/f'control-{om}.npz') as z:qref.append(z['reference_charge'][1]);bref.append(z['reference_bins'][:,1])
    qref=np.array(qref);bref=np.array(bref);edges=local.config.relative_time_edges_ns
    prepared=[];base_q=np.zeros(3);base_bins=np.zeros_like(bref)
    setup_cpu,setup_wall=process_time(),perf_counter()
    for lam,phase in enumerate(tables['phase_index']):
        active=pieces.beta*phase>1
        band=pieces if active.all() else pieces.subset(active)
        actual=replace(band,reference_phase_index=float(phase),cone_model='frozen')
        baseline=replace(actual,beta=np.ones(len(actual)),coefficient2=-actual.coefficient0)
        sa,sb=_element_segments(actual),_element_segments(baseline)
        tree=backward.build_tree(sa);cones=tree.node_cone.copy();theta=math.acos(1/float(phase))
        cones[:,0]=np.minimum(cones[:,0],theta);cones[:,1]=np.maximum(cones[:,1],theta)
        tree=replace(tree,node_cone=cones)
        lobes=np.zeros((3,256,4));lobe_cdf=np.zeros((3,256));lobe_count=np.zeros(3,np.int64)
        for om in range(3):
            directions,weights=backward.direct_lobes(sb,tree,local.detector.positions_m[om],looks[om],apoly,float(tables['mut'][lam]),256)
            lobe_count[om]=len(weights);lobes[om,:len(weights)]=directions;lobe_cdf[om,:len(weights)]=np.cumsum(weights)
        common=(np.ascontiguousarray(local.detector.positions_m),looks,frames,areas,origins)
        shared=(tree.order,tree.node_bound,tree.node_cone,tree.node_child,tree.node_range,float(local.medium.g),
            tables['mus'][lam:lam+1],tables['mut'][lam:lam+1],tables['speed'][lam:lam+1],
            tables['s0w'][lam:lam+1],tables['s2w'][lam:lam+1],float(tables['mut'][lam]),
            apoly,cdf,lobes,lobe_cdf,lobe_count,.3,.5,.05,edges)
        def arguments(s):return common+(s.start,s.direction,s.length,s.q0,s.q2,s.cone,s.t0,s.dtda)+shared
        aa,bb=arguments(sa),arguments(sb);prepared.append((aa,bb))
        if lam==0:
            backward._run(*aa,1,2,1);backward._run(*bb,1,2,1)
        single=copy(local);single.wavelength=WavelengthQuadrature(tables['wavelength_nm'][lam:lam+1],tables['weight_nm'][lam:lam+1])
        q,b,_=order1_integrals(single,baseline,time_origin_ns=origins,config=PromptConfig(element_method='segment',screen=False))
        base_q+=q;base_bins+=b
    setup_cpu,setup_wall=process_time()-setup_cpu,perf_counter()-setup_wall
    args.output.mkdir(parents=True)
    report=dict(primary_only=True,energy_GeV=300,modules=ids.tolist(),source_pieces=len(pieces),reference_nodes=len(prepared),
        threads=1,reference_charge=qref.tolist(),base_charge=base_q.tolist(),base_preparation_cpu_s=setup_cpu,
        base_preparation_wall_s=setup_wall,base_integral_event_specific=True,base_cost_excluded_from_mc_timings=True,
        proposal_shared=True,tree_covers_both_cones=True,runs=[])
    for seed in [42,43]:
        for samples in args.samples:
            per_batch=int(np.ceil(samples/batches));count=per_batch*batches
            qa=np.zeros((3,batches));qb=np.zeros_like(qa)
            ba=np.zeros((3,batches,len(edges)-1));bbins=np.zeros_like(ba)
            cpu,wall=process_time(),perf_counter()
            for aa,bb in prepared:
                bins,stats=backward._run(*aa,per_batch,batches,seed);qa+=stats[:,:,0]/per_batch;ba+=bins/per_batch
                bins,stats=backward._run(*bb,per_batch,batches,seed);qb+=stats[:,:,0]/per_batch;bbins+=bins/per_batch
            elapsed_cpu,elapsed_wall=process_time()-cpu,perf_counter()-wall
            residual=qa-qb;hybrid=base_q+residual.mean(1);plain=qa.mean(1)
            plain_se=qa.std(1,ddof=1)/np.sqrt(batches);hybrid_se=residual.std(1,ddof=1)/np.sqrt(batches)
            hybrid_bins=base_bins+(ba-bbins).mean(1)
            item=dict(seed=seed,samples_per_wavelength_per_om=count,cpu_s=elapsed_cpu,wall_s=elapsed_wall,
                hybrid_charge=hybrid.tolist(),plain_charge=plain.tolist(),
                hybrid_relative_charge_error=(abs(hybrid-qref)/qref).tolist(),plain_relative_charge_error=(abs(plain-qref)/qref).tolist(),
                hybrid_relative_charge_se=(hybrid_se/qref).tolist(),plain_relative_charge_se=(plain_se/qref).tolist(),
                variance_reduction=(plain_se**2/np.maximum(hybrid_se**2,1e-300)).tolist(),
                hybrid_bins_l1=(abs(hybrid_bins-bref).sum(1)/np.maximum(abs(bref).sum(1),1e-300)).tolist(),
                plain_bins_l1=(abs(ba.mean(1)-bref).sum(1)/np.maximum(abs(bref).sum(1),1e-300)).tolist())
            report['runs'].append(item);(args.output/'report.json').write_text(json.dumps(report,indent=2))
            np.savez_compressed(args.output/f'seed{seed}-n{count}.npz',qa=qa,qb=qb,ba=ba,bb=bbins,base_charge=base_q,
                base_bins=base_bins,reference_charge=qref,reference_bins=bref)
            print('PAIRED',json.dumps(item),flush=True)


if __name__=='__main__':main()
