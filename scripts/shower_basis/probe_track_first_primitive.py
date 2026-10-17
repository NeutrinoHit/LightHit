#!/usr/bin/env python3
"""Bounded orientation-specific q1 track-prefix validation on real event geometry.

This checks one reusable row, not a full event accelerator or rho/orientation
atlas. Both cache and held-out finite segments use 82-node optical quadrature.
"""
import paths  # inputs and outputs: scripts/shower_basis/paths.py
import argparse
from dataclasses import replace
from pathlib import Path
from time import perf_counter,process_time
import json
import numpy as np


def whole_primary_check(kernel,elements,detector,om,energy,atlas_path,tail_budget=0.,audit_omitted=False):
    """Complete primary: cached local interiors plus explicit direct tails."""
    from copy import copy
    from lighthit.experimental.track_first_primitive import TrackFirstAtlas,first_scattering_segment_bounds
    from lighthit.experimental.primary_polyline import primary_spectral_quadrature
    from lighthit.experimental.muon_integral import order1_integrals
    from lighthit.sources import SpectralLightElements
    from lighthit.prompt import PromptConfig
    from lighthit._shower_basis import earliest_element_arrival
    loading=perf_counter()
    atlas=TrackFirstAtlas.load(atlas_path)
    load_wall=perf_counter()-loading
    local=copy(kernel)
    local.wavelength=primary_spectral_quadrature(kernel.medium,kernel.detector,order=2,
        wavelength_range_nm=kernel.config.wavelength_range_nm)
    config=PromptConfig(element_method='segment',screen=False)
    edges=np.asarray(kernel.config.relative_time_edges_ns)
    with np.load(paths.muon_response(energy)) as stored:
        origin=float(stored['time_origin_ns'][om])
    absolute=edges+origin
    position=detector.positions_m[0];orientation=detector.orientations[0]
    warm_start=perf_counter()
    order1_integrals(local,elements.subset(np.arange(len(elements))==0),time_origin_ns=np.array([origin]),config=config)
    warmup_wall=perf_counter()-warm_start
    fallback=[];cached=[];qcache=0.;bcache=np.zeros(len(edges)-1)
    vcache=atlas.metadata['speed_m_per_ns'];betacache=atlas.metadata['beta']
    began,cpu=perf_counter(),process_time()
    transport_start,transport_cpu_start=began,cpu
    for i in range(len(elements)):
        axis=elements.direction[i];delta=position-elements.start_m[i]
        z=float(delta@axis);radial=delta-z*axis;rho=float(np.linalg.norm(radial))
        length=float(elements.length_m[i]);speed=length/(elements.end_ns[i]-elements.start_ns[i])
        if not atlas.radii_m[0]<=rho<=atlas.radii_m[-1]:
            fallback.append((i,0.,length,'rho'));continue
        ex=radial/rho;frame=np.column_stack((ex,np.cross(axis,ex),axis));look=orientation@frame
        ulo=max(z-length,float(atlas.x_edges[0]*rho));uhi=min(z,float(atlas.x_edges[-1]*rho))
        if ulo>=uhi:
            fallback.append((i,0.,length,'outside_source_range'));continue
        # Actual piece clock anchors tau. Its slope differs slightly from the
        # atlas slope; report that approximation explicitly below.
        tau=absolute-elements.start_ns[i]-z/speed
        density=elements.coefficient0[i]/length
        phases=kernel.medium.phase_index
        if np.any(elements.beta[i]*phases<=1):
            fallback.append((i,0.,length,'beta_threshold'));continue
        angle_error=np.max(abs(np.arccos(1/(elements.beta[i]*phases))-np.arccos(1/(betacache*phases))))
        clock_error=max(abs(ulo),abs(uhi))*abs(1/speed-1/vcache)
        if angle_error>1e-6 or clock_error>1e-3:
            fallback.append((i,0.,length,'beta_or_clock'));continue
        try:
            q,b=atlas.query(rho,look,ulo,uhi,tau,coefficient0_per_m=density)
        except ValueError as error:
            fallback.append((i,0.,length,'query:'+str(error)));continue
        low_s,high_s=z-uhi,z-ulo
        if low_s>0: fallback.append((i,0.,low_s,'left_tail'))
        if high_s<length: fallback.append((i,high_s,length,'right_tail'))
        qcache+=q;bcache+=b
        phases=kernel.medium.phase_index
        angle_error=np.max(abs(np.arccos(1/(elements.beta[i]*phases))-np.arccos(1/(betacache*phases))))
        # At a canonical source u the clock difference is u*(1/vcache-1/vactual).
        clock_error=max(abs(ulo),abs(uhi))*abs(1/speed-1/vcache)
        cached.append(dict(piece=i,rho_m=rho,length_m=high_s-low_s,charge_pe=q,
            beta=float(elements.beta[i]),speed_m_per_ns=float(speed),
            max_clock_difference_ns=float(clock_error),max_cone_angle_difference_rad=float(angle_error)))
    cache_wall,cache_cpu=perf_counter()-began,process_time()-cpu
    tail_bounds=None;bound_cpu=bound_wall=0.;tail_omitted=False;tail_audit=None
    if fallback:
        idx=np.array([a[0] for a in fallback]);a=np.array([a[1] for a in fallback]);b=np.array([a[2] for a in fallback])
        lengths=b-a;fraction=lengths/elements.length_m[idx]
        fallback_source=SpectralLightElements(elements.start_m[idx]+a[:,None]*elements.direction[idx],
            elements.direction[idx],lengths,elements.coefficient0[idx]*fraction,elements.coefficient2[idx]*fraction,
            elements.beta[idx],elements.start_ns[idx]+a/elements.length_m[idx]*(elements.end_ns[idx]-elements.start_ns[idx]),
            elements.start_ns[idx]+b/elements.length_m[idx]*(elements.end_ns[idx]-elements.start_ns[idx]),
            elements.row_index[idx],elements.uid[idx],reference_phase_index=elements.reference_phase_index,cone_model='spectral')
        began,cpu=perf_counter(),process_time()
        bounds,bound_metadata=first_scattering_segment_bounds(local,fallback_source,position)
        bound_wall,bound_cpu=perf_counter()-began,process_time()-cpu
        tail_bounds=dict(total_pe=float(bounds.sum()) if np.isfinite(bounds.sum()) else None,
            per_part_pe=[float(v) if np.isfinite(v) else None for v in bounds],metadata=bound_metadata)
        if tail_budget>0 and bounds.sum()<=tail_budget:
            tail_omitted=True
            qf=np.array([0.]);bf=np.zeros((1,len(edges)-1));fallback_wall=fallback_cpu=0.
            if audit_omitted:
                began,cpu=perf_counter(),process_time()
                qa,ba,_=order1_integrals(local,fallback_source,time_origin_ns=np.array([origin]),config=config)
                tail_audit=dict(charge_pe=float(qa[0]),window_charge_pe=float(ba.sum()),
                    cpu_s=process_time()-cpu,wall_s=perf_counter()-began,excluded_from_timing=True)
        else:
            began,cpu=perf_counter(),process_time()
            qf,bf,meta=order1_integrals(local,fallback_source,time_origin_ns=np.array([origin]),config=config)
            fallback_wall,fallback_cpu=perf_counter()-began,process_time()-cpu
    else:
        qf=np.array([0.]);bf=np.zeros((1,len(edges)-1));fallback_wall=fallback_cpu=0.
    total_warm_cpu=process_time()-transport_cpu_start-(tail_audit['cpu_s'] if tail_audit else 0.)
    total_warm_wall=perf_counter()-transport_start-(tail_audit['wall_s'] if tail_audit else 0.)
    # Same original fitted source and 82-node quadrature, no caching or dropping.
    began,cpu=perf_counter(),process_time()
    qr,br,meta=order1_integrals(local,elements,time_origin_ns=np.array([origin]),config=config)
    reference_wall,reference_cpu=perf_counter()-began,process_time()-cpu
    q=qcache+qf[0];bins=bcache+bf[0]
    return dict(energy_GeV=energy,om=om,primary_only=True,scattering_order=1,source_pieces=len(elements),
        atlas_path=str(atlas_path),atlas_build_cpu_s=atlas.metadata['build_cpu_s'],
        atlas_load_wall_s=load_wall,direct_warmup_wall_s=warmup_wall,
        query_cpu_s=cache_cpu,query_wall_s=cache_wall,fallback_cpu_s=fallback_cpu,fallback_wall_s=fallback_wall,
        total_warm_cpu_s=total_warm_cpu,total_warm_wall_s=total_warm_wall,
        reference_cpu_s=reference_cpu,reference_wall_s=reference_wall,
        warm_speedup_cpu=reference_cpu/total_warm_cpu,
        bound_cpu_s=bound_cpu,bound_wall_s=bound_wall,tail_budget_pe=tail_budget,
        tail_bounds=tail_bounds,tail_omitted=tail_omitted,tail_audit=tail_audit,
        charge_pe=q,reference_charge_pe=float(qr[0]),cached_charge_pe=qcache,fallback_charge_pe=float(qf[0]),
        charge_relative_error=float(abs(q-qr[0])/qr[0]),bins_l1_relative=float(abs(bins-br[0]).sum()/br[0].sum()),
        cached_pieces=len(cached),fallback_parts=len(fallback),
        fallback_computed_parts=0 if tail_omitted else len(fallback),
        omitted_tail_parts=len(fallback) if tail_omitted else 0,cached=cached,
        fallback=[dict(piece=i,start_m=a,end_m=b,reason=reason) for i,a,b,reason in fallback],
        clock_approximation_max_ns=max((p['max_clock_difference_ns'] for p in cached),default=0.),
        cone_approximation_max_rad=max((p['max_cone_angle_difference_rad'] for p in cached),default=0.),
        cache_clock_guard_ns=1e-3,cache_cone_angle_guard_rad=1e-6,
        no_source_tail_dropped=not tail_omitted,cache_build_excluded_from_warm_timing=True,
        cache_load_excluded_from_warm_timing=True,reference_is_polyline_not_original_G4_steps=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--energy',type=int,choices=[300,1000,3000],default=1000)
    p.add_argument('--om',type=int,default=None)
    p.add_argument('--cells',type=int,default=32)
    p.add_argument('--time-step',type=float,default=.5)
    p.add_argument('--radial-study',action='store_true')
    p.add_argument('--orientation-atlas',action='store_true')
    p.add_argument('--full-primary-atlas',type=Path)
    p.add_argument('--tail-budget-pe',type=float,default=0.)
    p.add_argument('--audit-omitted',action='store_true')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if not np.isfinite(args.tail_budget_pe) or args.tail_budget_pe<0:
        p.error('tail budget must be finite and nonnegative')
    import ntsim,numba,lighthit as lh
    from ntsim.LightTransport import bgvd_model_from_ntsim
    from lighthit.experimental.primary_polyline import fit_primary_polyline
    from lighthit.experimental.track_first_primitive import build_track_first_primitive
    from probe_positive_map_screen import read_source
    if numba.get_num_threads()!=1: p.error('NUMBA_NUM_THREADS=1 required')
    if (args.output/'report.json').exists(): p.error('choose a new output folder')
    source,_,_=read_source(args.energy)
    primary=source.subset(source.uid==0)
    model=bgvd_model_from_ntsim(ntsim.bgvd.Telescope(n_clusters=(1,2,3,4,5)),ntsim.bgvd.Water())
    fit=fit_primary_polyline(primary,phase_indices=model.medium.phase_index)
    elements=fit.elements
    om=args.om if args.om is not None else {300:197,1000:198,3000:198}[args.energy]
    receiver=model.detector.positions_m[om]
    delta=receiver-elements.start_m
    z=np.sum(delta*elements.direction,axis=1)
    closest=delta-np.clip(z,0,elements.length_m)[:,None]*elements.direction
    piece=int(np.argmin(np.linalg.norm(closest,axis=1)))
    axis=elements.direction[piece]
    radial=delta[piece]-z[piece]*axis
    rho=float(np.linalg.norm(radial)); ex=radial/rho
    frame=np.column_stack((ex,np.cross(axis,ex),axis))
    orientation=model.detector.orientations[om]@frame
    beta=float(elements.beta[piece]);speed=float(elements.length_m[piece]/(elements.end_ns[piece]-elements.start_ns[piece]))
    d=model.detector
    model=replace(model,detector=replace(d,positions_m=d.positions_m[[om]],orientations=d.orientations[[om]],
        effective_area_m2=d.effective_area_m2[[om]],identifiers={k:v[[om]] for k,v in d.identifiers.items()}))
    kernel=model.kernel(lh.KernelConfig(omega_per_ns=np.array([0.]),cache_policy='require'))
    if args.full_primary_atlas:
        report=whole_primary_check(kernel,elements,model.detector,om,args.energy,args.full_primary_atlas,args.tail_budget_pe,args.audit_omitted)
        args.output.mkdir(parents=True,exist_ok=True)
        (args.output/'report.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report,indent=2))
        return
    # Sinh spacing resolves the direct cone-root region; distant source cells
    # get larger. Endpoints never extend beyond the explicit local 300m sphere.
    extent=float(np.sqrt(300.**2-rho**2))
    root=rho/np.sqrt((beta*1.34)**2-1.)
    scale=max(.4,rho*.1)
    grid=np.linspace(np.arcsinh((-extent-root)/scale),np.arcsinh((extent-root)/scale),args.cells+1)
    u=root+scale*np.sinh(grid);u[0]=-extent;u[-1]=extent
    front=rho*np.sqrt((kernel.medium.group_index.min()/.299792458)**2-1/speed**2)
    split=np.ceil(front+40)
    t=np.unique(np.r_[-10.,np.arange(max(-9.,front-2.),split,args.time_step),np.arange(split,901.,2.),900.])
    # Explicit warmup distinguishes build cost from first-use JIT compilation.
    start=perf_counter()
    warm=build_track_first_primitive(kernel,rho_m=rho,orientation=orientation,beta=beta,speed_m_per_ns=speed,
        u_edges_m=[root-.01,root+.01],tau_edges_ns=t)
    warmup=perf_counter()-start
    row=build_track_first_primitive(kernel,rho_m=rho,orientation=orientation,beta=beta,speed_m_per_ns=speed,
        u_edges_m=u,tau_edges_ns=t)
    intervals=[(max(-extent,root-30.7),min(extent,root+42.3)),
               (max(-extent,root-7.3),min(extent,root+4.7)),
               (max(-extent,root+15.1),min(extent,root+90.3))]
    actual_lo=float(z[piece]-elements.length_m[piece]);actual_hi=float(z[piece])
    if -extent <= actual_lo < actual_hi <= extent:
        intervals.append((actual_lo,actual_hi))
    tests=[]
    for i,(lo,hi) in enumerate(intervals):
        edges=np.arange(-5.,806.,5.)+.37*i
        control=build_track_first_primitive(kernel,rho_m=rho,orientation=orientation,beta=beta,speed_m_per_ns=speed,
            u_edges_m=[lo,hi],tau_edges_ns=edges)
        q,b=row.query(lo,hi,edges)
        qr,br=control.charge_cells[0],control.bins_cells[0]
        tests.append(dict(u_m=[lo,hi],time_shift_ns=.37*i,charge=q,reference_charge=qr,
            charge_relative_error=float(abs(q-qr)/qr),bins_l1_relative=float(abs(b-br).sum()/br.sum()),
            reference_seconds=control.metadata['build_wall_s'],reference_cpu_s=control.metadata['build_cpu_s'],
            window_fraction=float(br.sum()/qr)))
    began,cpu=perf_counter(),process_time()
    for j in range(2000):
        lo,hi=intervals[j%len(intervals)]
        row.query(lo,hi,np.arange(-5.,806.,5.)+.37*(j%3))
    timing=dict(queries=2000,wall_s=perf_counter()-began,cpu_s=process_time()-cpu)
    report=dict(energy_GeV=args.energy,om=om,piece=piece,rho_m=rho,orientation=orientation.tolist(),
        beta=beta,speed_m_per_ns=speed,actual_piece_u_m=[float(z[piece]-elements.length_m[piece]),float(z[piece])],
        actual_coefficient0_per_m=float(elements.coefficient0[piece]/elements.length_m[piece]),
        warmup_s=warmup,cache=row.metadata,cells=len(u)-1,time_edges=len(t),tests=tests,
        timing=timing,bytes_arrays=sum(v.nbytes for v in vars(row).values() if isinstance(v,np.ndarray)),
        note='Orientation/rho-specific row; 82 spectral nodes; all-time charge retained separately; no source clipping.')
    if args.radial_study:
        radial_rows=[]
        xgrid=np.unique(np.r_[-8.5,(u/rho)[(u/rho>-8.5)&(u/rho<8.5)],8.5])
        for radius in [25.,30.,35.]:
            radial_rows.append(build_track_first_primitive(kernel,rho_m=radius,orientation=orientation,
                beta=beta,speed_m_per_ns=speed,u_edges_m=xgrid*radius,tau_edges_ns=t*radius/rho))
        lower,upper=radial_rows[:2]
        r0,r1=lower.metadata['rho_m'],upper.metadata['rho_m']
        w=(rho-r0)/(r1-r0)
        total0,total1=lower.charge_cells.sum(),upper.charge_cells.sum()
        total=np.exp((1-w)*np.log(total0)+w*np.log(total1))
        radial_tests=[]
        for i,((lo,hi),ref) in enumerate(zip(intervals,tests)):
            edges=np.arange(-5.,806.,5.)+.37*i
            q0,b0=lower.query(lo*r0/rho,hi*r0/rho,edges*r0/rho)
            q1,b1=upper.query(lo*r1/rho,hi*r1/rho,edges*r1/rho)
            # Independent direct finite-segment reference, preserving actual bins.
            control=build_track_first_primitive(kernel,rho_m=rho,orientation=orientation,beta=beta,
                speed_m_per_ns=speed,u_edges_m=[lo,hi],tau_edges_ns=edges)
            qr,br=control.charge_cells[0],control.bins_cells[0]
            out={}
            for mode,qv,bv in [('linear',(1-w)*q0+w*q1,(1-w)*b0+w*b1),
                    ('normalized_log_amplitude',total*((1-w)*q0/total0+w*q1/total1),
                     total*((1-w)*b0/total0+w*b1/total1))]:
                out[mode]=dict(charge_relative_error=float(abs(qv-qr)/qr),
                              bins_l1_relative=float(abs(bv-br).sum()/br.sum()))
            radial_tests.append(out)
        report['radial_study']=dict(nodes_m=[25.,30.,35.],held_out_rho_m=rho,
            shared_orientation=True,shared_beta=True,shared_emission_speed=True,
            build_cpu_s=sum(r.metadata['build_cpu_s'] for r in radial_rows),tests=radial_tests,
            source_variable='u/rho',time_variable='tau/rho',normalization_source_range=[-8.5,8.5],
            amplitude_note='global row amplitude; same factor for every source interval/time bin; source additivity preserved')
    atlas=None
    if args.orientation_atlas:
        from lighthit.experimental.track_first_primitive import build_track_first_atlas
        xgrid=np.unique(np.r_[-8.5,(u/rho)[(u/rho>-8.5)&(u/rho<8.5)],8.5])
        atlas=build_track_first_atlas(kernel,radii_m=[25.,30.,35.],x_edges=xgrid,
            time_over_rho_edges=t/rho,beta=beta,speed_m_per_ns=speed,
            progress=lambda item: print(json.dumps({'radius_completed':item}),flush=True))
        extra_look=np.array([.23,.61,np.sqrt(1-.23**2-.61**2)])
        atlas_tests=[]
        for look in [orientation,extra_look]:
            for i,(lo,hi) in enumerate(intervals):
                edges=np.arange(-5.,806.,5.)+.37*i
                control=build_track_first_primitive(kernel,rho_m=rho,orientation=look,beta=beta,
                    speed_m_per_ns=speed,u_edges_m=[lo,hi],tau_edges_ns=edges)
                qr,br=control.charge_cells[0],control.bins_cells[0]
                entry=dict(orientation=look.tolist(),u_m=[lo,hi],reference_charge=qr,
                    reference_cpu_s=control.metadata['build_cpu_s'])
                try:
                    qa,ba=atlas.query(rho,look,lo,hi,edges)
                    entry.update(charge_relative_error=float(abs(qa-qr)/qr),
                        bins_l1_relative=float(abs(ba-br).sum()/br.sum()),
                        minimum_bin=float(ba.min()))
                except ValueError as error:
                    entry['reconstruction_error']=str(error)
                atlas_tests.append(entry)
        began,cpu=perf_counter(),process_time()
        for j in range(2000):
            lo,hi=intervals[j%len(intervals)]
            atlas.query(rho,orientation,lo,hi,np.arange(-5.,806.,5.)+.37*(j%3))
        report['orientation_atlas']=dict(metadata=atlas.metadata,tests=atlas_tests,
            query_timing=dict(queries=2000,wall_s=perf_counter()-began,cpu_s=process_time()-cpu),
            bytes_arrays=sum(v.nbytes for v in vars(atlas).values() if isinstance(v,np.ndarray)),
            held_out_radius_m=rho,held_out_orientations=2)
    args.output.mkdir(parents=True,exist_ok=True)
    if atlas is not None: atlas.save(args.output/'atlas.npz')
    np.savez_compressed(args.output/'row.npz',u_edges_m=u,tau_edges_ns=t,charge_cells=row.charge_cells,bins_cells=row.bins_cells,
        speed_m_per_ns=speed,metadata=json.dumps(row.metadata))
    (args.output/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
