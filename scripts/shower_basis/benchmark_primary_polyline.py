#!/usr/bin/env python3
"""Primary q1 compact-polyline check on three preselected difficult OMs.

Run with NUMBA_NUM_THREADS=OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=1.
Compares at matched spectral quadrature; nine-node saved original-step controls
are kept separate from improved-spectrum current-one-metre controls.
"""
import paths  # inputs and outputs: scripts/shower_basis/paths.py
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from time import perf_counter, process_time
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--energy', type=int, choices=[300,1000,3000], required=True)
    parser.add_argument('--wavelength-nodes', type=int, nargs='+', default=[9,129])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--composite-orders', type=int, nargs='+', help='check candidate only at union of actual medium/QE/transmission knots')
    args = parser.parse_args()
    import lighthit as lh
    import ntsim, numba
    from ntsim.LightTransport import bgvd_model_from_ntsim
    from lighthit.experimental.muon_integral import order1_integrals, fit_track_integrals
    from lighthit.experimental.primary_polyline import fit_primary_polyline
    if numba.get_num_threads() != 1:
        parser.error('set NUMBA_NUM_THREADS=1')
    source_path = paths.muon_source(args.energy)
    reference_path = paths.work(f'muon-segments-20261009/integral/exact{args.energy}/reference.npz')
    meta = json.loads(reference_path.with_suffix('.json').read_text())
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    if meta['source_sha256'] != source_hash or meta['method'] != 'segment' or meta.get('refined_quadrature',False):
        raise ValueError('saved original control does not match source/segment quadrature')
    with np.load(source_path) as z:
        source = lh.SpectralLightElements(**{k:z[k] for k in z.files if k!='metadata'},cone_model='spectral')
        smeta = json.loads(str(z['metadata']))
    source = source.subset(source.uid==0)
    with np.load(reference_path) as z:
        saved_q, saved_b, origins, edges = z['charge'],z['bins'],z['origins'],z['edges']
    model = bgvd_model_from_ntsim(ntsim.bgvd.Telescope(n_clusters=(1,2,3,4,5)),ntsim.bgvd.Water())
    if model.fingerprint != meta.get('model_fingerprint',smeta['model_fingerprint']):
        raise ValueError('saved control and detector/medium model differ')
    ids = np.asarray(meta['receiver_indices']); detector=model.detector
    model = replace(model,detector=replace(detector,positions_m=detector.positions_m[ids],
        orientations=detector.orientations[ids],effective_area_m2=detector.effective_area_m2[ids],
        identifiers={k:np.asarray(v)[ids] for k,v in detector.identifiers.items()}))
    start=perf_counter(); cpu=process_time()
    fit = fit_primary_polyline(source,phase_indices=model.medium.phase_index)
    fit_timing=dict(wall_seconds=perf_counter()-start,cpu_seconds=process_time()-cpu)
    current=fit_track_integrals(source,max_length_m=1.,phase_indices=model.medium.phase_index)
    args.output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output/'fit.npz',ranges=fit.ranges,**{
        k:getattr(fit.elements,k) for k in ('start_m','direction','length_m','coefficient0','coefficient2','beta','start_ns','end_ns','row_index','uid')})
    report=dict(energy_GeV=args.energy,source_sha256=source_hash,receiver_indices=ids.tolist(),
        fit=fit.diagnostics,fit_timing=fit_timing,current_one_metre_pieces=len(current.elements),threads=1,
        physical_spectral_convergence_certified=False,cases=[])
    previous_response = None
    for nodes in (args.composite_orders or args.wavelength_nodes):
        kernel=model.kernel(lh.KernelConfig(relative_time_edges_ns=edges,threshold_pe=0.,wavelength_nodes=nodes,
            cache_directory=str(paths.RTE_CACHE.path()),cache_policy='require'))
        if args.composite_orders:
            from bgvd_model import OpticalModule
            from lighthit.experimental.primary_polyline import primary_spectral_quadrature
            kernel.wavelength=primary_spectral_quadrature(model.medium,model.detector,order=nodes,
                efficiency_knots_nm=np.r_[OpticalModule.eff_wavelengths, OpticalModule.transmission_wavelengths])
        config=lh.PromptConfig(element_method='segment',screen=False)
        order1_integrals(kernel,source.subset(np.arange(len(source))==0),time_origin_ns=origins,config=config)
        case=dict(wavelength_nodes=len(kernel.wavelength.wavelength_nm), composite_order=nodes if args.composite_orders else None)
        responses={}
        for name,elements in ([('polyline',fit.elements)] if args.composite_orders else [('polyline',fit.elements),('current',current.elements)]):
            start=perf_counter();cpu=process_time()
            q,b,diagnostics=order1_integrals(kernel,elements,time_origin_ns=origins,config=config)
            case[name]=dict(charge_pe=q.tolist(),wall_seconds=perf_counter()-start,cpu_seconds=process_time()-cpu)
            responses[name]=(q,b)
            np.savez_compressed(args.output/f'{name}-{nodes}.npz',charge=q,bins=b,origins=origins,edges=edges)
        if args.composite_orders:
            q,b=responses['polyline']
            if previous_response is not None:
                rq,rb=previous_response
                case['versus_previous_order']=dict(relative_charge=((q-rq)/np.maximum(q,1e-30)).tolist(),
                    bins_l1=(abs(b-rb).sum(1)/np.maximum(abs(b).sum(1),1e-30)).tolist())
            previous_response=(q,b)
            report['cases'].append(case)
            (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps(dict(energy=args.energy,pieces=len(fit.elements),**case)),flush=True)
            continue
        q,b=responses['polyline'];rq,rb=responses['current']
        case['versus_current']=dict(relative_charge=((q-rq)/np.maximum(rq,1e-30)).tolist(),
            bins_l1=(abs(b-rb).sum(1)/np.maximum(abs(rb).sum(1),1e-30)).tolist(),
            speedup=case['current']['cpu_seconds']/case['polyline']['cpu_seconds'])
        if nodes==9:
            case['versus_original_saved_nine_nodes']=dict(relative_charge=((q-saved_q)/np.maximum(saved_q,1e-30)).tolist(),
                bins_l1=(abs(b-saved_b).sum(1)/np.maximum(abs(saved_b).sum(1),1e-30)).tolist())
        report['cases'].append(case)
        (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(dict(energy=args.energy,pieces=len(fit.elements),**case)),flush=True)


if __name__=='__main__':
    main()
