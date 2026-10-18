#!/usr/bin/env python3
"""Build event-independent beta=1 >=2 finite-track optical response rows.

Inputs are installed BaikalWater/OpticalModule tables and existing directional
RTE splines. No detector geometry, saved event, NTSim or Geant4 is imported.
The only geometry in the build is a generic straight source and test receivers.
"""
import argparse,json,hashlib
from pathlib import Path
from time import perf_counter,process_time
import numpy as np
from numpy.polynomial.legendre import leggauss


def load_optics(pack,cache):
    import lighthit as lh
    from bgvd_model.BaikalWater import BaikalWater
    from bgvd_model import OpticalModule
    from lighthit.shower_basis import Ge2KernelStore
    from lighthit.experimental.track_ge2_primitive import track_ge2_optics_metadata
    water=BaikalWater();parameters=np.asarray(OpticalModule.angular_parameters,float)
    medium=lh.SpectralMedium(np.asarray(water.wavelength),np.asarray(water.absorption_inv_length),
        np.asarray(water.scattering_inv_length),np.asarray(water.phase_refraction_index),
        np.asarray(water.group_refraction_index),g=.9,provenance='installed optical provider only')
    def angular(x):return np.maximum(sum(parameters[k]*(-np.asarray(x))**k for k in range(len(parameters))),0.)
    def efficiency(w):return OpticalModule.efficiency(w)*OpticalModule.transmission_gel_glass(w)
    # A dummy one-point detector satisfies the optical interface; its position
    # and orientation are not used in or included in the cache signature.
    detector=lh.DetectorArray(np.zeros((1,3)),np.array([[0.,0.,1.]]),np.pi*OpticalModule.radius**2,angular,efficiency)
    kernel=lh.TransportKernel(medium,detector,lh.KernelConfig(omega_per_ns=np.linspace(0,1.2,65),cache_policy='require'))
    metadata=track_ge2_optics_metadata(kernel)
    store=Ge2KernelStore.load(pack)
    key,bundle=next((key,b) for key,b in store._bundles.items() if len(b['omega'])==65)
    recorded=[]
    for name in key[0]:
        with np.load(cache/f'directional-{name}.npz',allow_pickle=False) as z:
            recorded.append(json.loads(str(z['metadata']))['medium'])
    wavelengths=np.array([r['wavelength_nm'] for r in recorded])
    if not np.array_equal(wavelengths,np.asarray(metadata['wavelength_nm'])):raise ValueError('cached wavelength mismatch')
    for field in ('absorption_per_m','scattering_per_m','group_index'):
        if not np.allclose([r[field] for r in recorded],metadata[field],rtol=1e-12,atol=0):raise ValueError('cached optical coefficients mismatch: '+field)
    if not np.allclose([r['g'] for r in recorded],metadata['g'],rtol=0,atol=1e-14):raise ValueError('HG mismatch')
    if kernel.acceptance()['degree']>3:raise ValueError('ten-component atlas requires acceptance degree<=3')
    metadata.update(cache_keys=list(key[0]),source_degree=int(bundle['degree']),acceptance_degree=int(bundle['acceptance_degree']),
        radius_window_m=np.exp(bundle['log_r'][[0,-1]]).tolist(),
        normalization='positive orientation-average local-infinite-track charge; fallback absolute channel norm',
        physical_limitations=[f'{len(key[0])} Gauss-Legendre wavelength nodes of the RTE cache (the kernel quadrature)',
            'same finite angular/frequency RTE approximation; no new independent transport solver',
            'local radial cutoff300m is not a bound on omitted physical light',
            'fixed beta=1 and clock1/c; nonrelativistic sources need direct fallback'],
        source_dependency='none: water and OM response only',
        interpolation_phase='individual cumulative endpoint causal front, with scaled u/rho',
        runtime_source_guard='beta=1 reference; near-beta1 clock mismatch needs explicit bound/fallback')
    return kernel,bundle,metadata


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pack',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--nodes',type=float,nargs='+',default=[3.01,5,8,12,18,25,35,50,70,100,150,220,299])
    p.add_argument('--passes',type=int,default=3)
    p.add_argument('--max-nodes',type=int,default=65)
    p.add_argument('--order',type=int,default=10)
    p.add_argument('--tolerance',type=float,default=.01)
    p.add_argument('--bright-charge-pe',type=float,default=1e-5)
    p.add_argument('--build-only',action='store_true')
    p.add_argument('--backward-absolute-pe',type=float,default=1e-6)
    args=p.parse_args()
    import numba
    from lighthit.experimental.track_ge2_primitive import (DiskTrackGe2Atlas,SegmentPrimitive,
        cone_point_evaluator,orientation_basis_looks,build_track_ge2_orientation_row,track_ge2_axial_edges)
    from lighthit.readout import inverse_bins
    from scipy.interpolate import CubicSpline
    if numba.get_num_threads()!=1:p.error('requires NUMBA_NUM_THREADS=1')
    kernel,bundle,metadata=load_optics(args.pack,args.cache)
    omega=bundle['omega'];clock=metadata['clock_ns_per_m'];high=metadata['radius_window_m'][1]
    front=float(np.sqrt((metadata['group_index'][len(metadata['group_index'])//2]/.299792458)**2-clock**2))
    output_omega=np.linspace(0,omega[-1],321);time_edges=np.arange(-60.,745.,5.)
    looks=np.array([[0.,0.,1.],[.6,0.,.8],[-.5,np.sqrt(.5),.5]])
    c0=2*np.pi/137.035999084*1e9
    kwargs={k:metadata[k] for k in ['alpha','wavelength_nm','wavelength_weight_nm','phase_index','efficiency','effective_area_m2','beta']}
    def factory(rho,look):return cone_point_evaluator(bundle,impact_m=float(rho),look=look,backend='fused-looks',**kwargs)
    def to_bins(q):return inverse_bins(output_omega,CubicSpline(omega,q)(output_omega)[:,None],time_edges)[0]
    args.output.mkdir(parents=True,exist_ok=True);(args.output/'rows').mkdir(exist_ok=True);(args.output/'references').mkdir(exist_ok=True)
    manifest_path=args.output/'manifest.json'
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text())
        if manifest['metadata']['optics_fingerprint']!=metadata['optics_fingerprint']:raise ValueError('resume optics mismatch')
        rows={float(item['impact_m']):item for item in manifest['rows']}
        manifest['status']='building'
        manifest['metadata']=metadata
    else:
        rows={};manifest=dict(schema='lighthit/track-ge2-disk-atlas/1',status='building',metadata=metadata,
            omega_per_ns=omega.tolist(),front_ns_per_m=front,rows=[],validation=[])
    began=perf_counter();cpu_began=process_time()
    def save_manifest():
        manifest['rows']=[rows[rho] for rho in sorted(rows)]
        temporary=manifest_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(manifest,indent=2));temporary.replace(manifest_path)
    def build(rho):
        rho=float(rho)
        if rho in rows:return
        edges=track_ge2_axial_edges(rho,high,max_panel_m=4.,asinh_step=.15)
        cpu=process_time();wall=perf_counter()
        row=build_track_ge2_orientation_row(factory(rho,orientation_basis_looks()),edges,omega,order=args.order,clock_ns_per_m=clock)
        zero=row.prefix[-1].reshape(10,len(omega))[:,0].real
        norm=float(zero[0]+(zero[3]+zero[5])/3)
        if norm<=0:norm=float(np.linalg.norm(zero))
        if not np.isfinite(norm) or norm<=0:raise ValueError('invalid row normalization')
        row.integrated_coefficients/=norm;row.prefix/=norm
        stem=f'rows/rho-{rho:.9f}'
        np.savez_compressed(args.output/(stem+'.npz'),edges_m=row.edges_m,integrated_coefficients=row.integrated_coefficients,prefix=row.prefix)
        np.savez_compressed(args.output/(stem+'-charge.npz'),edges_m=row.edges_m,
            integrated_coefficients=row.integrated_coefficients.reshape(len(edges)-1,args.order+1,10,len(omega))[...,0].real.copy(),
            prefix=row.prefix.reshape(len(edges),10,len(omega))[...,0].real.copy())
        item=dict(impact_m=rho,file=stem+'.npz',charge_file=stem+'-charge.npz',normalization=norm,
            cpu_seconds=process_time()-cpu,wall_seconds=perf_counter()-wall,panels=len(edges)-1,
            full_bytes=row.integrated_coefficients.nbytes+row.prefix.nbytes,
            charge_bytes=(row.integrated_coefficients.nbytes+row.prefix.nbytes)//(2*len(omega)),order=args.order,
            sha256=hashlib.sha256((args.output/(stem+'.npz')).read_bytes()).hexdigest(),
            charge_sha256=hashlib.sha256((args.output/(stem+'-charge.npz')).read_bytes()).hexdigest())
        rows[rho]=item;save_manifest();print(json.dumps(dict(event='row',**item)),flush=True)
    factory(10.,looks[:1])(np.array([0.,1.]))  # warmup separate from row cost
    for rho in sorted(set(args.nodes)):build(rho)
    def reference(rho):
        path=args.output/f'references/rho-{rho:.9f}.npz'
        if path.exists():
            with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
        cpu=process_time()
        edges=track_ge2_axial_edges(rho,high,max_panel_m=2.,asinh_step=.075)
        nodes,_=leggauss(14);half=np.diff(edges)/2;u=(edges[:-1,None]+half[:,None]*(nodes+1)).ravel()
        values=factory(rho,looks)(u)
        reach=float(np.sqrt(high*high-rho*rho))
        cuts=np.array([[reach,-reach],[rho,-rho],[1.5*rho,.5*rho],[1.1*rho,.9*rho],[-.5*rho,-1.5*rho]])
        spectra=[]
        for i in range(len(looks)):
            row=SegmentPrimitive.build(lambda x:values[i],edges,omega,order=14,clock_ns_per_m=clock)
            spectra.append(row.segment(cuts[:,0],cuts[:,1],closest_time_ns=-rho*front,coefficient0_per_m=c0))
        data=dict(spectra=np.asarray(spectra),looks=looks,cuts=cuts,rho=np.array(rho),cpu_seconds=np.array(process_time()-cpu))
        np.savez_compressed(path,**data)
        return data
    def validate(rho,atlas):
        ref=reference(rho);details=[];passed=True;front_ok=True;charge_ok=True;back_ok=True
        for i,look in enumerate(looks):
            for j,(a,b) in enumerate(ref['cuts']):
                q=atlas.segment(rho,look,a,b,closest_time_ns=-rho*front,coefficient0_per_m=c0)
                r=ref['spectra'][i,j];qb,rb=to_bins(q),to_bins(r)
                errorq=float(abs(q[0]-r[0]));errorb=float(abs(qb-rb).sum())
                relq=errorq/max(abs(r[0]),args.bright_charge_pe);relb=errorb/max(abs(rb).sum(),args.bright_charge_pe)
                ok=max(relq,relb)<=args.tolerance;passed &= ok
                charge_ok &= relq<=args.tolerance
                if j==4:back_ok &= errorb<=max(args.backward_absolute_pe,args.tolerance*max(abs(rb).sum(),args.bright_charge_pe))
                else:front_ok &= max(relq,relb)<=args.tolerance
                whole_bins=to_bins(ref['spectra'][i,0])
                whole_relative=errorb/max(abs(whole_bins).sum(),args.bright_charge_pe)
                details.append(dict(orientation=i,cut=j,charge_pe=float(r[0].real),charge_error_pe=errorq,
                    bins_l1_error_pe=errorb,charge_scaled_error=relq,bins_scaled_l1=relb,
                    bins_l1_over_whole_track=whole_relative,passed=bool(ok)))
        result=dict(impact_m=float(rho),passed=bool(passed),front_passed=bool(front_ok),
            charge_passed=bool(charge_ok),backward_passed=bool(back_ok),max_charge_scaled=max(x['charge_scaled_error'] for x in details),
            max_bins_scaled=max(x['bins_scaled_l1'] for x in details),details=details)
        print(json.dumps({k:v for k,v in result.items() if k!='details'}),flush=True)
        return result
    if not args.build_only:
        for iteration in range(args.passes):
            nodes=np.array(sorted(rows));atlas=DiskTrackGe2Atlas(args.output,require_complete=False,require_validated=False)
            checks=[];failed=[]
            for lo,hi in zip(nodes[:-1],nodes[1:]):
                rho=float(np.sqrt(lo*hi));result=validate(rho,atlas)
                result.update(low_m=float(lo),high_m=float(hi));checks.append(result)
                if not result['front_passed'] or not result['charge_passed']:failed.append(rho)
            manifest['validation']=checks;manifest['validation_settings']=dict(tolerance=args.tolerance,bright_charge_pe=args.bright_charge_pe,
                meaning='relative charge/L1bins on bright cuts; absolute tolerance=tolerance*bright_charge_pe on weak cuts',
                reference='direct source quadrature same RTE cache; generic cuts,3look directions',iteration=iteration,
                backward_absolute_tolerance_pe=args.backward_absolute_pe,
                unsupported_region='fully backward segment u_start<=0 when interval.backward_passed isfalse',
                strict_results_retained=True,production_approved=False)
            save_manifest()
            if not failed or iteration+1>=args.passes:break
            allowance=max(0,args.max_nodes-len(rows))
            if allowance==0:break
            for rho in failed[:allowance]:build(rho)
    for item in rows.values():
        for file_key,hash_key in [('file','sha256'),('charge_file','charge_sha256')]:
            item[hash_key]=hashlib.sha256((args.output/item[file_key]).read_bytes()).hexdigest()
        with np.load(args.output/item['charge_file']) as charge:
            item['charge_bytes']=charge['integrated_coefficients'].real.nbytes+charge['prefix'].real.nbytes
    manifest['status']='complete';save_manifest()
    report=dict(rows=len(rows),build_cpu_seconds=sum(r['cpu_seconds'] for r in rows.values()),
        process_cpu_seconds=process_time()-cpu_began,wall_seconds=perf_counter()-began,
        full_disk_bytes=sum((args.output/r['file']).stat().st_size for r in rows.values()),
        charge_preload_bytes=sum(r['charge_bytes'] for r in rows.values()),
        largest3rows_ram_bytes=sum(sorted([r['full_bytes'] for r in rows.values()],reverse=True)[:3]),
        passed_intervals=sum(x['passed'] for x in manifest.get('validation',[])),
        front_passed_intervals=sum(x.get('front_passed',x['passed']) for x in manifest.get('validation',[])),
        charge_passed_intervals=sum(x.get('charge_passed',x['passed']) for x in manifest.get('validation',[])),
        backward_passed_intervals=sum(x.get('backward_passed',x['passed']) for x in manifest.get('validation',[])),
        total_intervals=len(manifest.get('validation',[])),optics_fingerprint=metadata['optics_fingerprint'])
    (args.output/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
