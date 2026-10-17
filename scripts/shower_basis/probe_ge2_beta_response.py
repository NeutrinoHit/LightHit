#!/usr/bin/env python3
"""Bounded exact-beta batching check on the saved stopping primary.

No new G4 generation. Direct quadrature and the compact-source compiler use
same existing RTE cache and wavelength nodes; comparisons isolate source work.
"""
import paths  # inputs and outputs: scripts/shower_basis/paths.py
import argparse,json
from pathlib import Path
from dataclasses import replace
from time import process_time,perf_counter
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--energy',type=int,default=300)
    p.add_argument('--oms',type=int,nargs='+',default=[197,198,1208])
    p.add_argument('--cell-m',type=float,default=.25)
    p.add_argument('--azimuthal-degree',type=int,default=4)
    p.add_argument('--all-charge',action='store_true')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():p.error('choose unused output')
    import numba,ntsim,lighthit as lh
    from ntsim.LightTransport import bgvd_model_from_ntsim
    from lighthit.shower_basis import Ge2KernelStore
    from lighthit.experimental.primary_polyline import fit_primary_polyline
    from lighthit.experimental.ge2_beta_response import transport_ge2_tail_batch,primary_tail_mask
    from lighthit.experimental.track_ge2_primitive import cone_point_evaluator
    from lighthit.readout import inverse_bins
    from scipy.interpolate import CubicSpline
    from probe_positive_map_screen import read_source
    if numba.get_num_threads()!=1:p.error('one core required')
    source,_,_=read_source(args.energy);source=source.subset(source.uid==0)
    model=bgvd_model_from_ntsim(ntsim.bgvd.Telescope(n_clusters=(1,2,3,4,5)),ntsim.bgvd.Water())
    pieces=fit_primary_polyline(source,phase_indices=model.medium.phase_index).elements
    mask=primary_tail_mask(pieces);tail=pieces.subset(mask)
    ids=np.arange(len(model.detector.positions_m)) if args.all_charge else np.asarray(args.oms)
    d=model.detector;model=replace(model,detector=replace(d,positions_m=d.positions_m[ids],orientations=d.orientations[ids],
       effective_area_m2=d.effective_area_m2[ids],identifiers={k:v[ids] for k,v in d.identifiers.items()}))
    config=lh.KernelConfig(omega_per_ns=np.linspace(0,1.2,65),readout_omega_per_ns=np.linspace(0,1.2,321),
       cache_directory=paths.RTE_CACHE.path(),cache_policy='require',threshold_pe=0.)
    kernel=model.kernel(config);store=Ge2KernelStore.load(str(paths.GE2_PACK.path()))
    with np.load(paths.muon_response(args.energy)) as z:origins=z['time_origin_ns'][ids]
    charge=transport_ge2_tail_batch(kernel,tail,store,full=False,cell_m=args.cell_m,
        azimuthal_degree=args.azimuthal_degree,time_origin_ns=origins)
    result=dict(energy_GeV=args.energy,original_primary_steps=len(source),fitted_pieces=len(pieces),tail_pieces=len(tail),
        receivers=ids.tolist(),tail_arc_length_m=float(tail.length_m.sum()),charge_metadata=charge.metadata,
        charge_pe=charge.charge_pe.tolist())
    args.output.mkdir(parents=True)
    if not args.all_charge:
        full=transport_ge2_tail_batch(kernel,tail,store,full=True,cell_m=args.cell_m,
            azimuthal_degree=args.azimuthal_degree,time_origin_ns=origins)
        bundle=next(b for b in store._bundles.values() if len(b['omega'])==65);w=kernel.wavelength;sample=kernel.medium.sample(w.wavelength_nm)
        gx,gw=np.polynomial.legendre.leggauss(12)
        reference=np.zeros((65,len(ids)),complex);counts=[];cpu=process_time();wall=perf_counter()
        for j,pos in enumerate(kernel.detector.positions_m):
            count=0
            for i in range(len(tail)):
                axis=tail.direction[i];delta=pos-tail.start_m[i];z=float(delta@axis);rad=delta-z*axis;rho=float(np.linalg.norm(rad))
                if rho>=300:continue
                reach=np.sqrt(300**2-rho*rho);a=min(z,reach);b=max(z-tail.length_m[i],-reach)
                if a<=b:continue
                if rho<3:raise ValueError('unvalidated nearfield')
                e1=rad/rho;e2=np.cross(axis,e1);look=-kernel.detector.orientations[j]
                local_look=np.array([look@e1,look@e2,look@axis])
                evaluator=cone_point_evaluator(bundle,impact_m=rho,look=local_look,alpha=kernel.acceptance()['alpha'],
                    wavelength_nm=w.wavelength_nm,wavelength_weight_nm=w.weight_nm,phase_index=sample['phase_index'],
                    efficiency=kernel.detector.spectral_weight(w.wavelength_nm),beta=float(tail.beta[i]),
                    effective_area_m2=float(kernel.detector.effective_area_m2[j]),backend='fused-looks')
                edges=np.linspace(b,a,max(1,int(np.ceil((a-b)/2)))+1);half=np.diff(edges)/2
                u=(edges[:-1,None]+half[:,None]*(gx+1)).ravel();clock=(tail.end_ns[i]-tail.start_ns[i])/tail.length_m[i]
                closest=tail.start_ns[i]+z*clock-origins[j];density=tail.coefficient0[i]/tail.length_m[i]
                values=evaluator(u)*np.exp(-1j*u[:,None]*clock*bundle['omega'])
                reference[:,j]+=np.einsum('p,pf->f',(half[:,None]*gw).ravel(),values)*density*np.exp(1j*closest*bundle['omega']);count+=1
            counts.append(count)
        direct_cpu=process_time()-cpu;direct_wall=perf_counter()-wall
        rb=inverse_bins(config.readout_omega_per_ns,CubicSpline(bundle['omega'],reference,axis=0)(config.readout_omega_per_ns),config.relative_time_edges_ns)
        result.update(full_metadata=full.metadata,direct_cpu_seconds=direct_cpu,direct_wall_seconds=direct_wall,direct_pairs=counts,
            reference_charge_pe=reference[0].real.tolist(),charge_relative_error=(abs(charge.charge_pe-reference[0].real)/np.maximum(abs(reference[0].real),1e-30)).tolist(),
            bins_relative_l1=(abs(full.bins_pe-rb).sum(1)/np.maximum(abs(rb).sum(1),1e-30)).tolist())
        np.savez_compressed(args.output/'response.npz',omega=bundle['omega'],reference=reference,batch=full.spectrum_pe,
            bins_reference=rb,bins_batch=full.bins_pe,origin_ns=origins)
    (args.output/'report.json').write_text(json.dumps(result,indent=2,default=str))
    print(json.dumps({k:v for k,v in result.items() if k not in ['full_metadata','charge_metadata','receivers','charge_pe']},indent=2))
    print('charge_cpu',charge.metadata['cpu_seconds'],flush=True)
    if not args.all_charge:print('full_cpu',full.metadata['cpu_seconds'],flush=True)

if __name__=='__main__':main()
