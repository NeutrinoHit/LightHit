#!/usr/bin/env python3
"""Bounded full-directional >=2 finite-segment primitive experiment.

The row is specialized to one real OM/chord frame, not a universal interpolated
atlas. Spectral/angle/frequency accuracy is inherited from existing RTE cache.
"""
import paths  # inputs and outputs: scripts/shower_basis/paths.py
import argparse,json
from pathlib import Path
from time import perf_counter,process_time
import numpy as np
from numpy.polynomial.legendre import leggauss


def probe_mini_atlas(args,factory,omega,clock,real_look,a,b,coefficient,sample,optics_metadata):
    from lighthit.experimental.track_ge2_primitive import TrackGe2MiniAtlas,SegmentPrimitive
    from lighthit.readout import inverse_bins
    from scipy.interpolate import CubicSpline
    impacts=np.asarray(args.impact_nodes,float)
    edges=np.linspace(-300.,300.,int(np.ceil(600/args.panel_m))+1)
    front=float(np.sqrt((sample['group_index'][len(sample['group_index'])//2]/.299792458)**2-clock**2))
    factory(30.,np.array([[0.,0.,1.]]))(np.array([0.,1.]))
    wall=perf_counter();cpu=process_time()
    if args.reuse_atlas:
        zz=np.load(args.reuse_atlas/'atlas.npz')
        old_report=json.loads((args.reuse_atlas/'report.json').read_text())
        if old_report['energy_GeV']!=args.energy or old_report['om_index']!=args.om_index:raise ValueError('reuse geometry/spectrum mismatch')
        if not np.array_equal(zz['edges_m'],edges) or not np.array_equal(zz['omega_per_ns'],omega) or float(zz['clock_ns_per_m'])!=clock:raise ValueError('reuse grid/clock mismatch')
        old_nodes=zz['impact_nodes_m'];new_nodes=np.setdiff1d(impacts,old_nodes)
        if len(new_nodes)==1:raise ValueError('this probe refines at least two new impact rows')
        if len(new_nodes):
            atlas=TrackGe2MiniAtlas.build(factory,new_nodes,edges,omega,order=args.order,clock_ns_per_m=clock,front_ns_per_m=front)
            rows=dict(zip(new_nodes,atlas.rows))
        else:
            from lighthit.experimental.track_ge2_primitive import look_polynomials
            atlas=TrackGe2MiniAtlas(impacts,[],zz['basis_looks'],np.linalg.inv(look_polynomials(zz['basis_looks'])),omega,front)
            rows={}
        for i,rho in enumerate(old_nodes):
            rows[rho]=SegmentPrimitive(edges,zz['integrated_coefficients'][i],zz['prefix'][i],np.tile(omega,10),clock,{})
        atlas.impact_nodes_m=impacts;atlas.rows=[rows[rho] for rho in impacts]
    else:
        atlas=TrackGe2MiniAtlas.build(factory,impacts,edges,omega,order=args.order,
             clock_ns_per_m=clock,front_ns_per_m=front)
    build_cpu=process_time()-cpu;build_wall=perf_counter()-wall
    output_omega=np.linspace(0,omega[-1],321);time_edges=np.arange(-60.,745.,5.)
    def bins(q):
        return inverse_bins(output_omega,CubicSpline(omega,q)(output_omega)[:,None],time_edges)[0]
    orientations=[real_look,np.array([.3,-.4,np.sqrt(.75)]),np.array([-.7,np.sqrt(.26),.5])]
    results=[]
    # Fixed-node arbitrary orientation verifies angular interpolation separately.
    validation_impacts=(args.validation_impacts if args.validation_impacts else np.r_[impacts[len(impacts)//2],(impacts[:-1]+impacts[1:])/2])
    for rho in validation_impacts:
        for ilook,look in enumerate(orientations):
            fine_edges=np.linspace(-300.,300.,301)
            directrow=SegmentPrimitive.build(factory(rho,look),fine_edges,omega,order=12,clock_ns_per_m=clock)
            for shape,aa,bb in [('real_chord',a,b),('whole_local_track',300.,-300.),('short_front',rho-5.,rho+5.)]:
                aa,bb=max(aa,bb),min(aa,bb)
                shift=-rho*front
                ref=directrow.segment(aa,bb,closest_time_ns=shift,coefficient0_per_m=coefficient)
                bref=bins(ref)
                item=dict(rho_m=rho,orientation=ilook,shape=shape,charge_pe=float(ref[0].real),methods={})
                for method in ('raw','front','scaled'):
                    q=atlas.segment(rho,look,aa,bb,alignment=method,closest_time_ns=shift,coefficient0_per_m=coefficient)
                    item['methods'][method]=dict(charge_rel=float(abs(q[0]-ref[0])/max(abs(ref[0]),1e-30)),
                        spectrum_l1=float(abs(q-ref).sum()/max(abs(ref).sum(),1e-30)),
                        bins_l1=float(abs(bins(q)-bref).sum()/max(abs(bref).sum(),1e-30)))
                results.append(item)
    rng=np.random.default_rng(92);aa=rng.uniform(-250,250,1000);bb=aa-rng.uniform(1,100,1000)
    wall=perf_counter();cpu=process_time();atlas.segment(27.5,real_look,aa,bb,alignment='scaled')
    query_cpu=process_time()-cpu;query_wall=perf_counter()-wall
    report=dict(kind='reusable3impact10orientationmoment>=2mini-atlas',energy_GeV=args.energy,om_index=args.om_index,
       impact_nodes_m=impacts.tolist(),basis_looks=atlas.basis_looks.tolist(),
       orientation_matrix_condition=float(np.linalg.cond(atlas.inverse_orientation_matrix)),
       offline_cpu_seconds=build_cpu,offline_wall_seconds=build_wall,reused_atlas=str(args.reuse_atlas) if args.reuse_atlas else None,
       memory_bytes=int(sum(row.prefix.nbytes+row.integrated_coefficients.nbytes for row in atlas.rows)),
       query_count=1000,query_cpu_seconds=query_cpu,query_wall_seconds=query_wall,
       front_ns_per_m=front,clock_ns_per_m=clock,
       limitations=['3impactnodes25..35m only','onebeta/clock andfixedoptical9lambda spectrum',
          'R300mtruncation lacks physicaltailbound','inheritsL32,65frequency RTEcache error',
          'finite-source quadrature andrhointerpolation, not full-event validation'],results=results)
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'report.json').write_text(json.dumps(report,indent=2))
    atlas.metadata=optics_metadata
    atlas.save(args.output/'atlas.npz')
    print(json.dumps(report,indent=2))


def probe_primary_atlas(args,bundle,model,pieces,lam,w,sample):
    from lighthit.experimental.track_ge2_primitive import TrackGe2MiniAtlas,cone_point_evaluator
    from lighthit.readout import inverse_bins
    from scipy.interpolate import CubicSpline
    atlas=TrackGe2MiniAtlas.load(args.primary_atlas)
    if atlas.metadata['model_fingerprint']!=model.fingerprint:raise ValueError('atlas/model mismatch')
    omega=bundle['omega'];position=model.detector.positions_m[args.om_index]
    global_look=-model.detector.orientations[args.om_index]
    with np.load(paths.muon_response(args.energy)) as old:
        origin=float(old['time_origin_ns'][args.om_index])
    common=dict(alpha=np.asarray(atlas.metadata['alpha']),wavelength_nm=lam,wavelength_weight_nm=w.weight_nm,
        phase_index=sample['phase_index'],efficiency=model.detector.spectral_weight(lam),
        effective_area_m2=float(model.detector.effective_area_m2[args.om_index]))
    if common['effective_area_m2']!=atlas.metadata['effective_area_m2']:raise ValueError('area differs')
    reference=np.zeros(len(omega),complex);candidate=reference.copy()
    direct_seconds=0.;query_seconds=0.;used=[];skipped=[];fallback=[];maxclock=0.;maxbeta=0.;warmup_seconds=None;fallback_seconds=0.;fallback_reasons={}
    nq,wq=leggauss(12)
    for i in range(len(pieces)):
        axis=pieces.direction[i];rel=position-pieces.start_m[i];z=float(rel@axis)
        transverse=rel-z*axis;rho=float(np.linalg.norm(transverse))
        if rho>=300:skipped.append(i);continue
        reach=np.sqrt(300**2-rho**2);a=min(z,reach);b=max(z-pieces.length_m[i],-reach)
        if a<=b:skipped.append(i);continue
        if rho<3:raise ValueError('source approaches belowcache')
        e1=transverse/rho;e2=np.cross(axis,e1);look=np.array([global_look@e1,global_look@e2,global_look@axis])
        clock=float((pieces.end_ns[i]-pieces.start_ns[i])/pieces.length_m[i])
        tclosest=float(pieces.start_ns[i]+z*clock-origin)
        density=float(pieces.coefficient0[i]/pieces.length_m[i])
        evaluator=cone_point_evaluator(bundle,impact_m=rho,look=look,beta=float(pieces.beta[i]),**common)
        if warmup_seconds is None:
            cpu=process_time();evaluator(np.array([0.,1.]));warmup_seconds=process_time()-cpu
        panels=np.linspace(b,a,int(np.ceil((a-b)/2))+1);half=np.diff(panels)/2
        u=(panels[:-1,None]+half[:,None]*(nq+1)).ravel()
        cpu=process_time()
        values=evaluator(u)*np.exp(-1j*u[:,None]*clock*omega)
        direct=np.einsum('p,pf->f',(half[:,None]*wq).ravel(),values)*density*np.exp(1j*tclosest*omega)
        piece_direct_seconds=process_time()-cpu;direct_seconds+=piece_direct_seconds;reference+=direct
        reason=None
        if not atlas.impact_nodes_m[0]<=rho<=atlas.impact_nodes_m[-1]:reason='impact outside mini-atlas'
        elif abs(pieces.beta[i]-atlas.metadata['beta'])>1e-5:reason='beta differs from prepared row'
        elif abs(atlas.rows[0].clock_ns_per_m-clock)*(a-b)/2>.001:reason='clock mismatch exceeds .001ns'
        if reason is not None:
            candidate+=direct;fallback.append(i);fallback_reasons[str(i)]=reason;fallback_seconds+=piece_direct_seconds;continue
        # Anchor the fixedclock primitive at this segment/window midpoint.
        correction=(atlas.rows[0].clock_ns_per_m-clock)*(a+b)/2
        maxclock=max(maxclock,abs(atlas.rows[0].clock_ns_per_m-clock)*(a-b)/2)
        maxbeta=max(maxbeta,abs(pieces.beta[i]-atlas.metadata['beta']))
        cpu=process_time();q=atlas.segment(rho,look,a,b,alignment='scaled',closest_time_ns=tclosest+correction,coefficient0_per_m=density)
        query_seconds+=process_time()-cpu;candidate+=q
        used.append(dict(piece=i,impact_m=rho,length_in_window_m=float(a-b)))
    output_omega=np.linspace(0,omega[-1],321);time_edges=np.arange(-60.,745.,5.)
    def bins(q):return inverse_bins(output_omega,CubicSpline(omega,q)(output_omega)[:,None],time_edges)[0]
    cb,rb=bins(candidate),bins(reference)
    report=dict(kind='oneOMcompletefittedprimary>=2',energy_GeV=args.energy,om_index=args.om_index,
        total_pieces=len(pieces),used=used,radially_skipped=skipped,direct_fallback=fallback,fallback_reasons=fallback_reasons,
        reference_charge_pe=float(reference[0].real),candidate_charge_pe=float(candidate[0].real),
        charge_relative_error=float(abs(candidate[0]-reference[0])/abs(reference[0])),
        bins_relative_l1=float(abs(cb-rb).sum()/abs(rb).sum()),
        direct_cpu_seconds=direct_seconds,query_cpu_seconds=query_seconds,warmup_cpu_seconds=warmup_seconds,fallback_cpu_seconds=fallback_seconds,candidate_apply_cpu_seconds=query_seconds+fallback_seconds,
        max_clock_mismatch_ns=maxclock,max_beta_mismatch=maxbeta,
        limitations=['referenceintegratesfittedpolyline notoriginalG4steps','same9lambda/L32RTEcache accuracy unvalidated',
                     'r300m truncation notcertified','mini-atlas fixedbeta/clock approximated;clockbound reported',
                     'oneOM check, not full5cluster event speed'])
    args.output.mkdir(parents=True,exist_ok=True);(args.output/'report.json').write_text(json.dumps(report,indent=2))
    np.savez_compressed(args.output/'response.npz',omega=omega,reference=reference,candidate=candidate,bins_reference=rb,bins_candidate=cb)
    print(json.dumps(report,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--energy',type=int,choices=[300,1000,3000],default=1000)
    p.add_argument('--om-index',type=int,default=198)
    p.add_argument('--panel-m',type=float,default=4.)
    p.add_argument('--order',type=int,default=8)
    p.add_argument('--mini-atlas',action='store_true')
    p.add_argument('--impact-nodes',type=float,nargs='+',default=[25.,30.,35.])
    p.add_argument('--reuse-atlas',type=Path)
    p.add_argument('--validation-impacts',type=float,nargs='+')
    p.add_argument('--primary-atlas',type=Path)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if (args.output/'report.json').exists():p.error('output exists')
    import numba,ntsim,lighthit as lh
    from ntsim.LightTransport import bgvd_model_from_ntsim
    from lighthit.shower_basis import Ge2KernelStore
    from lighthit.experimental.primary_polyline import fit_primary_polyline
    from lighthit.experimental.track_ge2_primitive import SegmentPrimitive,cone_point_evaluator
    from lighthit.readout import inverse_bins
    from probe_positive_map_screen import read_source
    if numba.get_num_threads()!=1:p.error('requires one Numba thread')
    source,provenance,_=read_source(args.energy);source=source.subset(source.uid==0)
    model=bgvd_model_from_ntsim(ntsim.bgvd.Telescope(n_clusters=(1,2,3,4,5)),ntsim.bgvd.Water())
    if provenance['model_fingerprint'] != model.fingerprint:raise ValueError('source/model fingerprint mismatch')
    store=Ge2KernelStore.load(str(paths.GE2_PACK.path()))
    key,bundle=next((key,b) for key,b in store._bundles.items() if len(b['omega'])==65)
    # Read wavelength provenance instead of assuming cache order.
    lam=[]
    for cachekey in key[0]:
        with np.load(paths.RTE_CACHE.path()/f'directional-{cachekey}.npz') as z:
            lam.append(json.loads(str(z['metadata']))['medium']['wavelength_nm'])
    lam=np.asarray(lam); w=model.kernel(lh.KernelConfig(omega_per_ns=np.array([0.]),cache_policy='require')).wavelength
    if not np.array_equal(w.wavelength_nm,lam):raise ValueError('wavelength order mismatch')
    sample=model.medium.sample(lam)
    if not np.allclose(sample['absorption_per_m'],bundle['absorption'],rtol=1e-12,atol=0):raise ValueError('cache/model absorption mismatch')
    if not np.allclose(.299792458/sample['group_index'],bundle['speed'],rtol=1e-12,atol=0):raise ValueError('cache/model group velocity mismatch')
    fit=fit_primary_polyline(source,phase_indices=sample['phase_index'])
    pieces=fit.elements;pos=model.detector.positions_m[args.om_index]
    rel=pos-pieces.start_m; along=np.sum(rel*pieces.direction,axis=1)
    closest=np.clip(along,0,pieces.length_m)
    gap=rel-closest[:,None]*pieces.direction
    if args.primary_atlas:
        return probe_primary_atlas(args,bundle,model,pieces,lam,w,sample)
    # The nearest fitted segment is a real nontrivial local geometry.
    index=int(np.argmin(np.linalg.norm(gap,axis=1)))
    axis=pieces.direction[index]; z=float(along[index])
    transverse=rel[index]-z*axis; rho=float(np.linalg.norm(transverse))
    e1=transverse/rho;e2=np.cross(axis,e1)
    look=-model.detector.orientations[args.om_index]
    local_look=np.array([look@e1,look@e2,look@axis])
    alpha=model.kernel(lh.KernelConfig(omega_per_ns=np.array([0.]),cache_policy='require')).acceptance()['alpha']
    evaluator=cone_point_evaluator(bundle,impact_m=rho,look=local_look,alpha=alpha,
        wavelength_nm=lam,wavelength_weight_nm=w.weight_nm,phase_index=sample['phase_index'],
        efficiency=model.detector.spectral_weight(lam),beta=float(pieces.beta[index]),
        effective_area_m2=float(model.detector.effective_area_m2[args.om_index]))
    clock=float((pieces.end_ns[index]-pieces.start_ns[index])/pieces.length_m[index])
    reach=float(np.sqrt(np.exp(bundle['log_r'][-1])**2-rho*rho))
    if args.mini_atlas:
        kwargs=dict(alpha=alpha,wavelength_nm=lam,wavelength_weight_nm=w.weight_nm,
            phase_index=sample['phase_index'],efficiency=model.detector.spectral_weight(lam),
            beta=float(pieces.beta[index]),effective_area_m2=float(model.detector.effective_area_m2[args.om_index]))
        factory=lambda impact,looks:cone_point_evaluator(bundle,impact_m=impact,look=looks,**kwargs)
        return probe_mini_atlas(args,factory,bundle['omega'],clock,local_look,z,z-pieces.length_m[index],
            float(pieces.coefficient0[index]/pieces.length_m[index]),sample,
            dict(model_fingerprint=model.fingerprint,cache_keys=list(key[0]),
                 source_degree=int(bundle['degree']),acceptance_degree=int(bundle['acceptance_degree']),
                 radius_window_m=np.exp(bundle['log_r'][[0,-1]]).tolist(),
                 convention='receiver-minus-emitter u; forward exp(+i omega t); C0 density per metre',
                 **{k:np.asarray(v).tolist() for k,v in kwargs.items()}))
    # Compile independently of offline timing.
    begin=perf_counter();evaluator(np.array([0.,1.]));warmup=perf_counter()-begin
    results={};primitives={}
    for factor in [1.,.5]:
        step=args.panel_m*factor
        edges=np.linspace(-reach,reach,int(np.ceil(2*reach/step))+1)
        wall=perf_counter();cpu=process_time()
        row=SegmentPrimitive.build(evaluator,edges,bundle['omega'],order=args.order,clock_ns_per_m=clock)
        cpu=process_time()-cpu;wall=perf_counter()-wall
        label='coarse' if factor==1 else 'fine';primitives[label]=row
        results[label]=dict(panel_m=float(np.max(np.diff(edges))),panels=len(edges)-1,
            offline_cpu_seconds=cpu,offline_wall_seconds=wall,nbytes=row.prefix.nbytes+row.integrated_coefficients.nbytes)
    a=z;b=z-pieces.length_m[index]
    midpoint_time=pieces.start_ns[index]+z*clock
    origin=midpoint_time+rho*np.sqrt(float(np.min(sample['group_index']))**2-1)/.299792458
    coefficient=pieces.coefficient0[index]/pieces.length_m[index]
    qs={name:row.segment(a,b,closest_time_ns=midpoint_time-origin,coefficient0_per_m=coefficient)
        for name,row in primitives.items()}
    # Independent composite Gauss direct source integral, no cumulative table.
    clipped_a=min(reach,a);clipped_b=max(-reach,b)
    ed=np.linspace(clipped_b,clipped_a,int(np.ceil((clipped_a-clipped_b)/(args.panel_m*.5)))+1)
    nodes,weights=leggauss(args.order+4);half=np.diff(ed)/2
    u=(ed[:-1,None]+half[:,None]*(nodes+1)).ravel()
    wall=perf_counter();cpu=process_time()
    values=evaluator(u)*np.exp(-1j*u[:,None]*clock*bundle['omega'])
    direct=np.einsum('p,pf->f',(half[:,None]*weights).ravel(),values)
    direct*=coefficient*np.exp(1j*(midpoint_time-origin)*bundle['omega'])
    direct_cpu=process_time()-cpu;direct_wall=perf_counter()-wall
    edges_time=np.arange(-60.,745.,5.)
    from scipy.interpolate import CubicSpline
    output_omega=np.linspace(0.,bundle['omega'][-1],321)
    def to_bins(q):
        relative=CubicSpline(bundle['omega'],q.real)(output_omega)+1j*CubicSpline(bundle['omega'],q.imag)(output_omega)
        return inverse_bins(output_omega,relative[:,None],edges_time)[0]
    bins={name:to_bins(q) for name,q in qs.items()}
    bins['direct']=to_bins(direct)
    for name,q in qs.items():
        results[name].update(charge_pe=float(q[0].real),
            charge_relative_error=float(abs(q[0]-direct[0])/abs(direct[0])),
            spectrum_relative_l1=float(abs(q-direct).sum()/abs(direct).sum()),
            bins_relative_l1=float(abs(bins[name]-bins['direct']).sum()/abs(bins['direct']).sum()))
    row=primitives['fine'];mid=(a+b)/2
    partition=row.segment(a,mid)+row.segment(mid,b)-row.segment(a,b)
    # Arbitrary endpoints. No expensive physics evaluator is invoked here.
    rng=np.random.default_rng(81);n=10000
    aa=rng.uniform(-reach,reach,n);bb=rng.uniform(-reach,reach,n)
    wall=perf_counter();cpu=process_time();qq=row.segment(np.maximum(aa,bb),np.minimum(aa,bb));query_cpu=process_time()-cpu;query_wall=perf_counter()-wall
    report=dict(energy_GeV=args.energy,om_index=args.om_index,polyline_pieces=len(pieces),piece_index=index,
        piece_length_m=float(pieces.length_m[index]),impact_m=rho,local_look=local_look.tolist(),
        beta=float(pieces.beta[index]),clock_ns_per_m=clock,frequencies=65,wavelengths_nm=lam.tolist(),
        radius_window_m=np.exp(bundle['log_r'][[0,-1]]).tolist(),warmup_wall_seconds=warmup,
        row_kind='one exact impact and receiver orientation; not a universal atlas',
        limitations=['no bound on omitted r>300m contribution','existing 9 spectral nodes and L32 cache accuracy inherited',
                     'single closest fitted primary chord, not whole-event timing','rows built offline are geometry-specific'],
        rows=results,direct=dict(charge_pe=float(direct[0].real),cpu_seconds=direct_cpu,wall_seconds=direct_wall,points=len(u)),
        partition_absolute_max=float(abs(partition).max()),queries=dict(count=n,cpu_seconds=query_cpu,wall_seconds=query_wall,
           cpu_seconds_per_segment=query_cpu/n,amortized_ratio_direct_to_query=direct_cpu/(query_cpu/n)),
        time_edges_ns=edges_time.tolist(),readout_frequency_nodes=321,readout_interpolation='cubic after time-origin demodulation')
    args.output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output/'primitive.npz',edges_m=row.edges_m,integrated_coefficients=row.integrated_coefficients,
        prefix=row.prefix,omega_per_ns=row.omega_per_ns,clock_ns_per_m=row.clock_ns_per_m)
    (args.output/'report.json').write_text(json.dumps(report,indent=2))
    np.savez_compressed(args.output/'responses.npz',omega=bundle['omega'],direct=direct,coarse=qs['coarse'],fine=qs['fine'],bins_direct=bins['direct'],bins_fine=bins['fine'])
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
