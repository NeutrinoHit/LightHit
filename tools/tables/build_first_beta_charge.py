#!/usr/bin/env python3
"""One optics-only Q1 beta-shape mode; no event input or time CDF construction."""
import argparse,json,hashlib
from pathlib import Path
from copy import copy
from time import process_time,perf_counter
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--cubature',type=Path,required=True)
    p.add_argument('--anchor',type=float,default=.999)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    import numba
    from build_track_first_atlas import optical_kernel
    from lighthit import SpectralLightElements
    from lighthit.experimental.track_first_primitive import TrackFirstAtlasStore,orientation_nodes
    from lighthit.experimental.first_charge_only import first_scattering_charge_only
    from lighthit.experimental.first_charge_beta_atlas import FirstChargeBetaAtlas
    from lighthit.experimental.spectral_response_basis import ResponseCubature
    from lighthit.experimental.primary_polyline import primary_spectral_quadrature
    from lighthit.prompt import _spectral_tables
    if numba.get_num_threads()!=1:p.error('one thread required')
    kernel=optical_kernel();base=TrackFirstAtlasStore(args.base);base.validate_kernel(kernel)
    if not .99<=args.anchor<1:p.error('anchor must lie in[0.99,1)')
    cubature=ResponseCubature.load(args.cubature);cubature.validate_optics(kernel)
    digest=hashlib.sha256((args.base/'manifest.json').read_bytes()).hexdigest()
    rule_digest=hashlib.sha256(args.cubature.read_bytes()).hexdigest()
    full=copy(kernel);full.wavelength=primary_spectral_quadrature(kernel.medium,kernel.detector,order=2)
    tables=_spectral_tables(full);yield_slope=float(tables['s2w'].sum()/(tables['s0w'].sum()-tables['s2w'].sum()))
    sa=1/args.anchor**2-1;ra=1-sa*yield_slope
    looks,powers,matrix=orientation_nodes();inverse=np.linalg.inv(matrix)
    from build_response_cubature import scenario_kernel,source_for
    local=scenario_kernel(kernel,10.,looks);local.wavelength=cubature.quadrature
    first_scattering_charge_only(local,source_for(kernel,10.,1.,'root'))
    left=[];right=[];timings=[];(args.output/'rows').mkdir(exist_ok=True)
    begin=perf_counter()
    for i,rho in enumerate(base.radii_m):
        path=args.output/f'rows/{i:03d}.npz'
        if path.exists():
            with np.load(path) as data:
                if str(data['base_hash'])!=digest or str(data['rule_hash'])!=rule_digest or float(data['anchor'])!=args.anchor:
                    raise ValueError('resume input differs')
                left.append(data['left'].copy());right.append(data['right'].copy());timings.append(float(data['cpu_s']))
            continue
        local=scenario_kernel(kernel,float(rho),looks);local.wavelength=cubature.quadrature
        q0=[];qa=[];cpu=process_time();wall=perf_counter()
        for a,b in zip(base.x_edges[:-1]*rho,base.x_edges[1:]*rho):
            length=b-a
            for beta,output in [(1.,q0),(args.anchor,qa)]:
                source=SpectralLightElements(np.array([[0.,0.,-b]]),np.array([[0.,0.,1.]]),np.array([length]),
                    np.array([length]),np.array([-length/beta**2]),np.array([beta]),np.zeros(1),np.array([length/.299792458]),
                    np.array([0]),np.array([0]),cone_model='spectral')
                output.append(first_scattering_charge_only(local,source)[0])
        difference=(np.asarray(qa)/ra-np.asarray(q0))/sa
        coefficients=difference@inverse.T
        l=np.vstack((np.zeros((1,10)),np.cumsum(coefficients,axis=0)))
        r=np.vstack((np.cumsum(coefficients[::-1],axis=0)[::-1],np.zeros((1,10))))
        elapsed=process_time()-cpu;timings.append(elapsed);left.append(l);right.append(r)
        np.savez_compressed(path,left=l,right=r,cpu_s=elapsed,anchor=args.anchor,base_hash=digest,rule_hash=rule_digest)
        print('ROW',json.dumps(dict(rho_m=float(rho),cpu_s=elapsed,wall_s=perf_counter()-wall)),flush=True)
    metadata=dict(beta_min=args.anchor,anchor=args.anchor,parameter='s=beta^-2-1',yield_slope=yield_slope,
        base_manifest_sha256=digest,cubature_sha256=rule_digest,optics_fingerprint=base.metadata['optical_fingerprint'],
        spectral_nodes=len(cubature.quadrature.weight_nm),yield_nodes=len(full.wavelength.weight_nm),event_input=False,
        charge_only=True,time_response_approximated=False,error_certified=False,minimum_span_over_rho=.05,
        radial_interpolation='three-node quadratic log of positive segment charge',
        longitudinal_interpolation='componentwise PCHIP of angular charge prefixes')
    payload=dict(schema='lighthit/first-charge-beta/1',metadata=json.dumps(metadata),left=np.array(left),right=np.array(right))
    path=args.output/'first-beta-charge.npz';np.savez_compressed(path,**payload)
    model=FirstChargeBetaAtlas(path,base,require_validated=False)
    test_looks=np.array([[.8,0.,.6],[-.3,.4,np.sqrt(.75)],[0.,0.,-1.]])
    tests=[]
    for rho in [4.,10.,30.,80.,180.,280.]:
        for beta in [1.,.99999,.9999,.9995,args.anchor]:
            phase=float(kernel.medium.sample([480.])['phase_index'][0]);root=1/np.sqrt((beta*phase)**2-1)
            cuts=[(-1.,4.),(root-.4,root+.5),(root-1.,root+.03),(-6.,-2.),(root-.003,root+.003),
                (root-.025,root+.025),(root+.037-.025,root+.037+.025)]
            for cut,(a,b) in enumerate(cuts):
                length=(b-a)*rho
                source=SpectralLightElements(np.array([[0.,0.,-b*rho]]),np.array([[0.,0.,1.]]),np.array([length]),
                    np.array([length]),np.array([-length/beta**2]),np.array([beta]),np.zeros(1),np.array([length/.299792458]),
                    np.array([0]),np.array([0]),cone_model='spectral')
                ref=scenario_kernel(kernel,rho,test_looks);ref.wavelength=full.wavelength
                qr=first_scattering_charge_only(ref,source)[0]
                q=model.query_charge_many(np.full(3,rho),test_looks,np.full(3,a*rho),np.full(3,b*rho),np.full(3,beta),enforce_span=False)
                error=abs(q-qr)/np.maximum(abs(qr),1e-300)
                scale=2*np.pi/137.035999084*1e9
                scaled=abs(q-qr)*scale/np.maximum(abs(qr)*scale,1e-5)
                for look in range(3):tests.append(dict(rho_m=rho,beta=beta,cut=cut,look=look,reference=float(qr[look]),prediction=float(q[look]),relative_error=float(error[look]),scaled_error=float(scaled[look])))
        print('CHECK',rho,flush=True)
    # Narrow-root cuts also diagnose the existing beta-one endpoint grid.
    regular=[t for t in tests if t['cut']!=4]
    passed=all(np.isfinite(t['scaled_error']) for t in regular) and max(t['scaled_error'] for t in regular)<=.01
    metadata['synthetic_validation_passed']=bool(passed);payload['metadata']=json.dumps(metadata);np.savez_compressed(path,**payload)
    report=dict(metadata=metadata,build_cpu_s=sum(timings),elapsed_s=perf_counter()-begin,disk_bytes=path.stat().st_size,
        tests=tests,max_regular_error=max(t['relative_error'] for t in regular),max_regular_scaled_error=max(t['scaled_error'] for t in regular),
        max_narrow_error=max(t['relative_error'] for t in tests if t['cut']==4))
    (args.output/'report.json').write_text(json.dumps(report,indent=2));print('SUMMARY',json.dumps({k:v for k,v in report.items() if k!='tests'}),flush=True)


if __name__=='__main__':main()
