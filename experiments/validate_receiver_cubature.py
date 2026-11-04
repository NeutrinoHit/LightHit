#!/usr/bin/env python3
"""Check an existing positive wavelength rule on first-entry segment responses.

Weights and nodes are unchanged. A derived rule is written only after the
specified independent glass-response checks satisfy the existing 1% target.
The original training provenance is retained; no event data enter fitting.
"""
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--first-entry-tables',type=Path,required=True)
    p.add_argument('--rule',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--radii',nargs='+',type=float,default=[3.8,30.,150.])
    p.add_argument('--betas',nargs='+',type=float,default=[1.,.999])
    p.add_argument('--cuts',nargs='+',choices=['wide','root','edge','back'],default=['root','edge'])
    p.add_argument('--waveform',action='store_true',help='also validate the time-bin L1 error on the same physical segments')
    p.add_argument('--resume',action='store_true',help='reuse a complete rule with matching validation inputs')
    args=p.parse_args(argv)
    if args.output.exists() and not args.resume:p.error('output already exists; choose a new path')
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/tables'))
    from build_track_first_atlas import optical_kernel
    from build_response_cubature import scenario_kernel,source_for
    from lighthit.experimental.spectral_response_basis import ResponseCubature
    from lighthit.muon.primary_polyline import primary_spectral_quadrature
    from lighthit.muon.track_first_primitive import track_first_optical_contract
    from lighthit.muon.first_charge_only import first_scattering_charge_only
    from lighthit.muon.muon_integral import order1_integrals
    model=ResponseCubature.load(args.rule);kernel=optical_kernel(args.model,args.first_entry_tables)
    if args.output.exists():
        existing=ResponseCubature.load(args.output);existing.validate_optics(kernel)
        report=json.loads(args.output.with_suffix('.json').read_text())
        checks=report['checks']
        domain={(c['radius_m'],c['beta'],c['cut']) for c in checks}
        expected={(r,b,c) for r in args.radii for b in args.betas for c in args.cuts}
        if (not report['passed'] or domain!=expected
                or report['source_rule_sha256']!=hashlib.sha256(args.rule.read_bytes()).hexdigest()
                or report['model_sha256']!=hashlib.sha256(args.model.read_bytes()).hexdigest()
                or (args.waveform and not existing.metadata.get('first_entry_waveform_validation',{}).get('passed'))):
            raise ValueError('completed cubature has different validation inputs')
        print('REUSED',args.output,flush=True);return 0
    native=primary_spectral_quadrature(kernel.medium,kernel.detector,order=2)
    if not np.allclose(native.wavelength_nm[model.selected_indices],model.quadrature.wavelength_nm,rtol=1e-12,atol=0):
        raise ValueError('original rule nodes differ from the current native quadrature')
    looks=np.array([[.8,0.,.6],[-.3,.4,np.sqrt(.75)],[0.,0.,-1.]])
    checks=[]
    for radius in args.radii:
        for beta in args.betas:
            for cut in args.cuts:
                source=source_for(kernel,radius,beta,cut)
                local=scenario_kernel(kernel,radius,looks)
                if args.waveform:
                    from time import perf_counter
                    from lighthit.prompt import PromptConfig
                    def timed(quadrature):
                        local.wavelength=quadrature;start=perf_counter()
                        q,b,_=order1_integrals(local,source,time_origin_ns=np.zeros(len(looks)),
                            config=PromptConfig(element_method='segment',screen=False))
                        return q,b,dict(wall_seconds=perf_counter()-start)
                    reference,reference_bins,rm=timed(native)
                    prediction,prediction_bins,pm=timed(model.quadrature)
                else:
                    local.wavelength=native;reference,rm=first_scattering_charge_only(local,source)
                    local.wavelength=model.quadrature;prediction,pm=first_scattering_charge_only(local,source)
                error=abs(prediction-reference)/np.maximum(abs(reference),1e-300)
                row=dict(radius_m=radius,beta=beta,cut=cut,reference=reference.tolist(),prediction=prediction.tolist(),
                    relative_error=error.tolist(),native_seconds=rm['wall_seconds'],reduced_seconds=pm['wall_seconds'])
                if args.waveform:
                    row['bins_relative_l1']=(abs(prediction_bins-reference_bins).sum(1)/np.maximum(abs(reference_bins).sum(1),1e-300)).tolist()
                checks.append(row);print(json.dumps(row),flush=True)
    maximum=max(max(row['relative_error']) for row in checks)
    waveform_max=max((max(row['bins_relative_l1']) for row in checks if 'bins_relative_l1' in row),default=None)
    passed=maximum<=.01 and (waveform_max is None or waveform_max<=.01)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    report=dict(criterion_relative=.01,passed=passed,max_relative_error=maximum,checks=checks,
        max_bins_relative_l1=waveform_max,
        source_rule_sha256=hashlib.sha256(args.rule.read_bytes()).hexdigest(),model_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest())
    args.output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    if not passed:return 2
    optical_contract,identity=track_first_optical_contract(kernel)
    metadata=dict(model.metadata,original_optics_fingerprint=model.metadata['optics_fingerprint'],optics_fingerprint=identity,optical_contract=optical_contract,
        source_rule_sha256=report['source_rule_sha256'],receiver_transfer='unchanged positive rule checked on first-entry responses',
        first_entry_validation=dict(max_relative_error=maximum,radii_m=args.radii,betas=args.betas,cuts=args.cuts,criterion_relative=.01),
        error_certified=False)
    if args.waveform:metadata['first_entry_waveform_validation']=dict(passed=passed,max_relative_l1=waveform_max,
        criterion_relative=.01,radii_m=args.radii,betas=args.betas,cuts=args.cuts)
    temporary=args.output.with_suffix('.tmp.npz')
    np.savez_compressed(temporary,wavelength_nm=model.quadrature.wavelength_nm,weight_nm=model.quadrature.weight_nm,
        selected_indices=model.selected_indices,modes=model.modes,singular_values=model.singular_values,metadata=json.dumps(metadata))
    temporary.replace(args.output)
    return 0


if __name__=='__main__':raise SystemExit(main())
