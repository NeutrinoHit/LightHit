#!/usr/bin/env python3
"""Cached beta-one geometry plus paired full-spectrum backward-MC residual.

The artificial source is a geometry-only polyline with beta=1, clipped to the
validated charge cache. Curvature, real beta, dark bands and omitted source
intervals enter the residual. Cache interpolation bias remains; it is checked
against an independently computed base integral rather than assumed zero.
"""
import paths  # inputs and outputs: scripts/shower_basis/paths.py
import argparse,json,math
from copy import copy
from dataclasses import replace
from pathlib import Path
from time import perf_counter,process_time
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--samples',type=int,nargs='+',default=[128,512,2048])
    p.add_argument('--oms',type=int,nargs='+',default=[197,195,1208])
    args=p.parse_args()
    if args.output.exists():p.error('fresh output required')
    import numba,lighthit as lh,ntsim
    from ntsim.LightTransport import bgvd_model_from_ntsim
    from lighthit.experimental.primary_polyline import fit_primary_polyline,PrimaryPolylineConfig,primary_spectral_quadrature
    from lighthit.experimental.track_first_primitive import TrackFirstAtlasStore
    from lighthit.experimental.first_charge_beta_atlas import FirstChargeBetaAtlas
    from lighthit.experimental.first_charge_only import first_scattering_charge_only
    from lighthit.experimental.universal_primary import subset_kernel,split_source
    from lighthit.experimental import prompt_backward as backward
    from lighthit.prompt import _element_segments,_spectral_tables,acceptance_polynomial,acceptance_monomials
    if numba.get_num_threads()!=1:p.error('one thread required')
    with np.load(paths.muon_source(300)) as z:
        source=lh.SpectralLightElements(**{k:z[k] for k in z.files if k!='metadata'},cone_model='spectral')
    source=source.subset(source.uid==0)
    model=bgvd_model_from_ntsim(ntsim.bgvd.Telescope(n_clusters=(1,2,3,4,5)),ntsim.bgvd.Water())
    kernel=model.kernel(lh.KernelConfig(omega_per_ns=np.array([0.]),cache_policy='require'))
    actual=fit_primary_polyline(source,phase_indices=kernel.medium.phase_index).elements
    artificial=replace(source,beta=np.ones(len(source)),coefficient2=-source.coefficient0)
    artificial=fit_primary_polyline(artificial,config=PrimaryPolylineConfig(time_tolerance_ns=1e10),phase_indices=kernel.medium.phase_index).elements
    first=TrackFirstAtlasStore(str(paths.Q1_ATLAS.path()))
    atlas=FirstChargeBetaAtlas(str(paths.Q1_BETA.path()),first)
    local=copy(kernel);local.wavelength=primary_spectral_quadrature(kernel.medium,kernel.detector,order=2)
    tables=_spectral_tables(local);full_source_actual=actual
    apoly=acceptance_polynomial(*acceptance_monomials(local.acceptance()['alpha']))
    integral=np.polynomial.polynomial.polyint(apoly);cdf=float(np.polynomial.polynomial.polyval(1.,integral)-np.polynomial.polynomial.polyval(-1.,integral))
    if np.polynomial.polynomial.polyval(np.linspace(-1,1,20001),apoly).min()<0:raise ValueError('negative proposal acceptance')
    mu_star=float(tables['mut'].min());reports=[];args.output.mkdir(parents=True)
    for om in args.oms:
        pos=kernel.detector.positions_m[om];normal=kernel.detector.orientations[om];look=-normal
        parts=[];geometry=[]
        for i in range(len(artificial)):
            axis=artificial.direction[i];d=pos-artificial.start_m[i];z=float(d@axis);rad=d-z*axis;rho=float(np.linalg.norm(rad))
            if rho==0:continue
            lo=max(z-artificial.length_m[i],first.x_edges[0]*rho);hi=min(z,first.x_edges[-1]*rho)
            if not atlas.supports(rho,1.,lo,hi):continue
            ex=rad/rho;ey=np.cross(axis,ex);orient=np.array([normal@ex,normal@ey,normal@axis])
            geometry.append((rho,orient,lo,hi,artificial.coefficient0[i]/artificial.length_m[i]))
            parts.append((i,z-hi,z-lo))
        if not parts:raise ValueError('no supported control source for OM '+str(om))
        base_source=split_source(artificial,parts)
        cpu=process_time();cache=atlas.query_charge_many(np.array([g[0] for g in geometry]),np.array([g[1] for g in geometry]),
            np.array([g[2] for g in geometry]),np.array([g[3] for g in geometry]),np.ones(len(geometry)),
            coefficient0_per_m=np.array([g[4] for g in geometry])).sum();cache_cpu=process_time()-cpu
        if not np.isfinite(cache):raise ValueError('invalid cached control source')
        single=subset_kernel(local,[om]);cpu=process_time();qbase=first_scattering_charge_only(single,base_source)[0][0];base_check_cpu=process_time()-cpu
        # One proposal for all wavelengths and both sources. Its acceptance
        # component gives full support even if the actual cone moves.
        phase=float(kernel.medium.sample([480.])['phase_index'][0])
        ref=replace(base_source,reference_phase_index=phase,cone_model='frozen');sr=_element_segments(ref);tr=backward.build_tree(sr)
        dirs,weights=backward.direct_lobes(sr,tr,pos,look,apoly,mu_star,256)
        lobes=np.zeros((1,256,4));lcdf=np.zeros((1,256));lobes[0,:len(weights)]=dirs;lcdf[0,:len(weights)]=np.cumsum(weights)
        lcount=np.array([len(weights)],np.int64);origins=np.zeros(1);edges=single.config.relative_time_edges_ns
        common=(np.ascontiguousarray(pos[None,:]),np.ascontiguousarray(look[None,:]),np.ascontiguousarray(backward._frame(look)[None,:,:]),
            np.ascontiguousarray(single.detector.effective_area_m2),origins)
        prepared=[];cpu=process_time()
        # Geometry/direction tree topology does not depend on wavelength.
        # Use conservative spectral cone ranges, including dark-band zero
        # weights, and avoid rebuilding all spatial nodes82times per OM.
        reference_actual=replace(full_source_actual,beta=np.ones(len(full_source_actual)),
            coefficient2=-full_source_actual.coefficient0,reference_phase_index=float(phase),cone_model='frozen')
        geometry_actual=_element_segments(reference_actual)
        tree_actual=backward.build_tree(geometry_actual)
        lowtheta=np.arccos(np.clip(1/(full_source_actual.beta*tables['phase_index'].min()),-1,1))
        hightheta=np.arccos(np.clip(1/(full_source_actual.beta*tables['phase_index'].max()),-1,1))
        cone_bounds=tree_actual.node_cone.copy()
        for node,(lo,hi) in enumerate(tree_actual.node_range):
            members=tree_actual.order[lo:hi]
            cone_bounds[node]=[lowtheta[members].min(),hightheta[members].max()]
        tree_actual=replace(tree_actual,node_cone=cone_bounds)
        tree_base=backward.build_tree(sr)
        cone_bounds=tree_base.node_cone.copy()
        cone_bounds[:,0]=math.acos(1/float(tables['phase_index'].min()))
        cone_bounds[:,1]=math.acos(1/float(tables['phase_index'].max()))
        tree_base=replace(tree_base,node_cone=cone_bounds)
        for lam,phase in enumerate(tables['phase_index']):
            emitting=full_source_actual.beta*phase>1
            band=replace(full_source_actual,beta=np.where(emitting,full_source_actual.beta,1.),
                coefficient0=np.where(emitting,full_source_actual.coefficient0,0.),
                coefficient2=np.where(emitting,full_source_actual.coefficient2,0.))
            sa=_element_segments(replace(band,reference_phase_index=float(phase),cone_model='frozen'))
            sb=_element_segments(replace(base_source,reference_phase_index=float(phase),cone_model='frozen'))
            shared=(float(single.medium.g),tables['mus'][lam:lam+1],tables['mut'][lam:lam+1],tables['speed'][lam:lam+1],
                tables['s0w'][lam:lam+1],tables['s2w'][lam:lam+1],mu_star,apoly,cdf,lobes,lcdf,lcount,.3,.5,.05,edges)
            def arguments(s,tree):
                return common+(s.start,s.direction,s.length,s.q0,s.q2,s.cone,s.t0,s.dtda)+(
                    tree.order,tree.node_bound,tree.node_cone,tree.node_child,tree.node_range)+shared
            aa,bb=arguments(sa,tree_actual),arguments(sb,tree_base);prepared.append((aa,bb))
            if lam==0:backward._run(*aa,1,2,1);backward._run(*bb,1,2,1)
        prepare_cpu=process_time()-cpu
        with np.load(str(paths.work(f'primary-universal-20261009/results/seed42-300/control-{om}.npz'))) as z:qref=float(z['reference_charge'][1])
        item=dict(om=om,actual_pieces=len(actual),artificial_pieces=len(artificial),control_pieces=len(base_source),cached_base_pe=float(cache),
            direct_base_pe=float(qbase),base_relative_error=float(abs(cache-qbase)/max(abs(qbase),1e-30)),cache_cpu_s=cache_cpu,
            proposal_prepare_cpu_s=prepare_cpu,base_validation_cpu_s=base_check_cpu,reference_pe=qref,runs=[])
        for seed in [42,43]:
            for n in args.samples:
                batches=32;count=int(np.ceil(n/batches));delta=np.zeros(batches);cpu=process_time()
                for aa,bb in prepared:
                    _,s=backward._run(*aa,count,batches,seed);a=s[0,:,0]/count
                    _,s=backward._run(*bb,count,batches,seed);b=s[0,:,0]/count
                    delta+=a-b
                elapsed=process_time()-cpu;hybrid=cache+delta.mean();unbiased_base_check=qbase+delta.mean()
                row=dict(seed=seed,samples_per_om=count*batches,mc_cpu_s=elapsed,estimate_pe=float(hybrid),
                    total_cpu_s=cache_cpu+prepare_cpu+elapsed,
                    error_relative=float(abs(hybrid-qref)/max(abs(qref),1e-30)),
                    residual_only_error_relative=float(abs(unbiased_base_check-qref)/max(abs(qref),1e-30)),
                    estimated_se_relative=float(delta.std(ddof=1)/np.sqrt(batches)/max(abs(qref),1e-30)))
                item['runs'].append(row);print('MC',json.dumps(dict(om=om,**row)),flush=True)
        reports.append(item);(args.output/'report.json').write_text(json.dumps(dict(primary_only=True,energy_GeV=300,threads=1,
            reference_nodes=len(tables['wavelength_nm']),base_integral_event_specific=False,base_validation_excluded=True,modules=reports),indent=2))


if __name__=='__main__':main()
