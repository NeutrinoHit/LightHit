#!/usr/bin/env python3
"""Two signed omega-zero beta modes on an existing universal radial atlas.

The existing directional RTE and optical provider are the only build inputs.
Synthetic held-out beta/interval checks precede event evaluation.
"""
import argparse,json,hashlib
from pathlib import Path
from time import perf_counter,process_time
import numpy as np
from numpy.polynomial.legendre import leggauss


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--pack',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--beta-min',type=float,default=.99)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    import numba
    from build_track_ge2_atlas import load_optics
    from lighthit.experimental.ge2_beta_response import cone_beta_taylor
    from lighthit.experimental.track_ge2_primitive import (DiskTrackGe2Atlas,cone_point_evaluator,
        orientation_basis_looks,build_track_ge2_orientation_row)
    from lighthit.experimental.ge2_charge_beta_atlas import Ge2ChargeBetaAtlas
    if numba.get_num_threads()!=1:p.error('one Numba thread required')
    kernel,bundle,optics=load_optics(args.pack,args.cache)
    if not np.isfinite(args.beta_min) or not 0<args.beta_min<=1 or args.beta_min*np.min(optics['phase_index'])<=1:
        p.error('beta-min must stay above all spectral Cherenkov thresholds and at most1')
    bundle=dict(bundle,omega=bundle['omega'][:1],coefficients=bundle['coefficients'][...,:1])
    base=DiskTrackGe2Atlas(args.base);base.validate_kernel(kernel)
    digest=hashlib.sha256((args.base/'manifest.json').read_bytes()).hexdigest()
    harmonics=cone_beta_taylor(optics['wavelength_nm'],optics['phase_index'],bundle['degree'])
    looks=orientation_basis_looks()
    common={k:optics[k] for k in ('alpha','wavelength_nm','wavelength_weight_nm','phase_index','efficiency','effective_area_m2')}
    begin,cpu=perf_counter(),process_time();coefficients=[[],[]];prefix=[[],[]];timings=[]
    (args.output/'rows').mkdir(exist_ok=True)
    for i,(rho,base_row) in enumerate(zip(base.impact_nodes_m,base._charge)):
        path=args.output/f'rows/{i:03d}.npz'
        if path.exists():
            with np.load(path) as data:
                if str(data['base_manifest_sha256'])!=digest:raise ValueError('resume base differs')
                for mode in range(2):coefficients[mode].append(data['coefficients'][mode].copy());prefix[mode].append(data['prefix'][mode].copy())
                timings.append(float(data['cpu_seconds']))
            continue
        row_cpu,row_wall=process_time(),perf_counter();cc=[];pp=[]
        for mode in [1,2]:
            evaluator=cone_point_evaluator(bundle,impact_m=float(rho),look=looks,
                source_harmonics=harmonics[mode],backend='fused-looks',**common)
            row=build_track_ge2_orientation_row(evaluator,base_row.edges_m,np.array([0.]),
                order=base_row.integrated_coefficients.shape[1]-1)
            norm=base.manifest['rows'][i]['normalization']
            cc.append(row.integrated_coefficients.real/norm);pp.append(row.prefix.real/norm)
        elapsed=process_time()-row_cpu;timings.append(elapsed)
        np.savez_compressed(path,coefficients=np.array(cc),prefix=np.array(pp),base_manifest_sha256=digest,cpu_seconds=elapsed)
        for mode in range(2):coefficients[mode].append(cc[mode]);prefix[mode].append(pp[mode])
        print('ROW',json.dumps(dict(index=i,rho_m=float(rho),cpu_s=elapsed,wall_s=perf_counter()-row_wall)),flush=True)
    metadata=dict(optics_fingerprint=optics['optics_fingerprint'],base_manifest_sha256=digest,beta_min=args.beta_min,
        parameter='delta=1/beta-1',taylor_order=2,charge_only=True,scattering_order='>=2',event_input=False,
        source_degree=int(bundle['degree']),wavelength_nodes=len(optics['wavelength_nm']),
        threshold_crossings_supported=False,source_clock_used=False,error_certified=False,
        interpolation='signed modes normalized by frozen base amplitude; same scaled quadratic radial interpolation')
    payload=dict(schema='lighthit/ge2-charge-beta/1',metadata=json.dumps(metadata),nodes_m=base.impact_nodes_m,
        edge_offsets=np.r_[0,np.cumsum([len(r.edges_m) for r in base._charge])].astype(np.int64),
        panel_offsets=np.r_[0,np.cumsum([len(r.edges_m)-1 for r in base._charge])].astype(np.int64),
        edges_m=np.concatenate([r.edges_m for r in base._charge]),
        coefficients=np.stack([np.concatenate(c) for c in coefficients]),prefix=np.stack([np.concatenate(p) for p in prefix]),
        lognorm=np.log([r['normalization'] for r in base.manifest['rows']]))
    path=args.output/'beta-charge.npz';np.savez_compressed(path,**payload)
    model=Ge2ChargeBetaAtlas(path,base,require_validated=False)
    # Independent quadrature at radii not used to build correction rows.
    test_looks=np.array([[0.,0.,1.],[.6,0.,.8],[-.5,np.sqrt(.5),.5]])
    gx,gw=leggauss(14);scale=2*np.pi/137.035999084*1e9
    details=[];validation_betas=sorted(set([1.,.9999,.999,.995,.99,.98,.95,args.beta_min]),reverse=True)
    for rho in [3.5,14.,40.,80.,200.,290.]:
        reach=np.sqrt(300**2-rho*rho)
        cuts=np.array([[reach,-reach],[rho,-rho],[1.5*rho,.5*rho],[1.1*rho,.9*rho],[-.5*rho,-1.5*rho]])
        cuts=np.clip(cuts,-reach,reach)
        for beta in validation_betas:
            exact=cone_point_evaluator(bundle,impact_m=rho,look=test_looks,beta=beta,backend='fused-looks',**common)
            delta=1/beta-1
            response=cone_point_evaluator(bundle,impact_m=rho,look=test_looks,source_harmonics=harmonics[0]+delta*harmonics[1]+delta**2*harmonics[2],backend='fused-looks',**common)
            for cut,(a,b) in enumerate(cuts):
                if a<=b:continue
                edges=np.linspace(b,a,max(1,int(np.ceil((a-b)/2)))+1);half=np.diff(edges)/2
                u=(edges[:-1,None]+half[:,None]*(gx+1)).ravel();weights=(half[:,None]*gw).ravel()*scale
                qref=np.einsum('p,lpf->l',weights,exact(u)).real
                qt=np.einsum('p,lpf->l',weights,response(u)).real
                qa=(model.segment_charge_many(np.full(3,rho),test_looks,np.full(3,a),np.full(3,b),np.full(3,beta),coefficient0_per_m=scale)
                    if beta>=args.beta_min else np.full(3,np.nan))
                for look in range(3):
                    floor=max(abs(qref[look]),1e-5)
                    details.append(dict(rho_m=rho,beta=beta,cut=cut,look=look,reference_pe=float(qref[look]),
                        taylor_scaled_error=float(abs(qt[look]-qref[look])/floor),
                        atlas_scaled_error=None if beta<args.beta_min else float(abs(qa[look]-qref[look])/floor)))
            print('VALIDATION',json.dumps(dict(rho_m=rho,beta=beta)),flush=True)
    summary=[]
    for beta in validation_betas:
        rows=[r for r in details if r['beta']==beta]
        summary.append(dict(beta=beta,max_taylor_scaled_error=max(r['taylor_scaled_error'] for r in rows),
            max_atlas_scaled_error=None if beta<args.beta_min else max(r['atlas_scaled_error'] for r in rows)))
    passed=max(r['max_taylor_scaled_error'] for r in summary if r['beta']>=args.beta_min)<=.002
    passed &= max(r['max_atlas_scaled_error'] for r in summary if r['beta']>=args.beta_min)<=.01
    metadata['synthetic_validation_passed']=bool(passed)
    payload['metadata']=json.dumps(metadata);np.savez_compressed(path,**payload)
    report=dict(metadata=metadata,summary=summary,details=details,build_cpu_seconds=sum(timings),
        elapsed_seconds=perf_counter()-begin,process_cpu_seconds=process_time()-cpu,
        disk_bytes=path.stat().st_size,resident_bytes=model.resident_bytes)
    (args.output/'report.json').write_text(json.dumps(report,indent=2));print('SUMMARY',json.dumps({k:v for k,v in report.items() if k!='details'}),flush=True)


if __name__=='__main__':main()
