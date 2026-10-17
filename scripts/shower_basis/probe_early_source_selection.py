#!/usr/bin/env python3
"""Exact early spatial selection on an entire saved secondary G4 source.

Compare late/early compilation on the same requested source points. Five
clusters, all secondary records, same 65-frequency RTE operator; no prompt,
new OM threshold, mean shower, or radial-tail approximation is introduced.
"""
import paths  # inputs and outputs: scripts/shower_basis/paths.py
import argparse,gc,hashlib,json,resource,sys
from pathlib import Path
from dataclasses import replace
from time import perf_counter,process_time
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--energy',type=int,choices=[300,1000,3000],default=1000)
    p.add_argument('--band',type=int,default=4)
    p.add_argument('--time-step-ns',type=float,default=.5)
    p.add_argument('--oms',type=int,nargs='+')
    p.add_argument('--selection',choices=['all','controls'],default='all')
    p.add_argument('--pod-sample',action='store_true',help='save and test charge-preserving local angular/time ranks on a bounded sample')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('fresh output required')
    import lighthit as lh,ntsim,numba
    from ntsim.LightTransport import bgvd_model_from_ntsim
    from lighthit._shower_basis import compile_axial_slices,apply_multi,earliest_element_arrival
    from lighthit.experimental.axial_source import AxisFrame
    from lighthit.shower_basis import Ge2KernelStore
    from lighthit.readout import inverse_bins
    from scipy.interpolate import CubicSpline
    if numba.get_num_threads()!=1:p.error('one Numba thread required')
    path=paths.muon_source(a.energy)
    with np.load(path) as z:
        source=lh.SpectralLightElements(**{k:z[k] for k in z.files if k!='metadata'},cone_model='spectral')
        source_meta=json.loads(str(z['metadata']))
    source=source.subset(source.uid!=0)
    model=bgvd_model_from_ntsim(ntsim.bgvd.Telescope(n_clusters=(1,2,3,4,5)),ntsim.bgvd.Water())
    if source_meta['model_fingerprint']!=model.fingerprint:p.error('source/model mismatch')
    kernel=model.kernel(lh.KernelConfig(omega_per_ns=np.linspace(0,1.2,65),cache_policy='require'))
    wavelengths=kernel.wavelength.wavelength_nm
    if not 0<=a.band<len(wavelengths):p.error('invalid cached wavelength band')
    wavelength=float(wavelengths[a.band]);phase=float(kernel.medium.sample([wavelength])['phase_index'][0])
    active=source.beta*phase>1;source=source.subset(active)
    f0,f2=source.field(0,phase_index=phase),source.field(2,phase_index=phase)
    field=replace(f0,photons=np.ascontiguousarray((f0.photons-f2.photons/phase**2)/wavelength**2))
    frame=AxisFrame.of(field)
    # Use the prior active set only to choose reporting OMs; geometry pruning
    # defaults to *all* 1440 modules and makes no activity decision.
    refpath=(paths.ntsim_hybrid_response(a.energy)
        if a.energy in (300,1000) else paths.muon_response(3000))
    with np.load(refpath) as z:old_active=np.flatnonzero(z['active'])
    ids=np.array(list(dict.fromkeys((a.oms or old_active.tolist())+[196,197,198,342,666,1062,1241])))
    positions=np.ascontiguousarray(model.detector.positions_m[ids]);looks=np.ascontiguousarray(-model.detector.orientations[ids])
    areas=np.asarray(model.detector.effective_area_m2)[ids]
    alpha=np.ascontiguousarray(kernel.acceptance()['alpha'],float)
    store=Ge2KernelStore.load(str(paths.GE2_PACK.path()))
    bundle=store.bundle(kernel,np.arange(65));lam=a.band
    band=dict(bundle,coefficients=bundle['coefficients'][lam:lam+1],
        absorption=bundle['absorption'][lam:lam+1],speed=bundle['speed'][lam:lam+1])
    scale=(kernel.wavelength.weight_nm[lam]*float(model.detector.spectral_weight(wavelength))*areas)[None,:]
    omega=kernel.config.omega_per_ns;outomega=np.linspace(0,1.2,321);edges=kernel.config.relative_time_edges_ns
    origins=earliest_element_arrival(positions,source.start_m,source.direction,source.length_m,
        source.start_ns,float(kernel.medium.group_index.min()))
    options=dict(degree=32,azimuthal_degree=4,cell_m=.5,axial_order=3,
        transverse='stencil',stencil_m=.2,stencil_threshold_m=.03,frame=frame,
        time_step_ns=a.time_step_ns,time_deposit='cubic',channel_dtype=np.complex64,stencil_center='centroid')
    tiny=replace(field,**{name:getattr(field,name)[:1] for name in
        ['start_m','direction','length_m','photons','cone_cosine','start_ns','end_ns','row_index','uid']})
    warm=compile_axial_slices(tiny,omega_per_ns=omega,**options)
    apply_multi(band,warm,warm,positions[:1],looks[:1],alpha,np.ones(1),np.zeros(1),scale[:,:1],radial_window_m=(3.,300.))
    zero=compile_axial_slices(field,omega_per_ns=[0.],**options)
    spatial=zero.points_m();selected=np.zeros(len(spatial),bool)
    selection_positions=model.detector.positions_m if a.selection=='all' else positions
    for receiver in selection_positions:
        selected |= np.sum((spatial-receiver)**2,axis=1)<=(300.+1e-8)**2
    del zero;gc.collect()
    if not np.any(selected):p.error('no source points in the receiver radial windows')
    a.output.mkdir(parents=True)
    report=dict(energy_GeV=a.energy,source=str(path),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        model_fingerprint=model.fingerprint,threads=1,clusters=5,secondary_elements=len(source),
        wavelength_nm=wavelength,frequency_nodes=65,receiver_indices=ids.tolist(),
        geometry_selection_receivers=len(selection_positions),spatial_points=len(spatial),
        selected_points=int(selected.sum()),scope='entire secondary source at one wavelength; >=2 only',cases={})
    saved={};sample_ids=np.unique(np.linspace(0,int(selected.sum())-1,min(128,int(selected.sum()))).astype(int))
    reference_sample=None;reference_hash=None
    for label,prune in [('late',False),('early',True)]:
        cpu,wall=process_time(),perf_counter()
        compiled=compile_axial_slices(field,omega_per_ns=omega,point_selection=selected,prune_selected=prune,**options)
        compcpu,compwall=process_time()-cpu,perf_counter()-wall
        digest=hashlib.sha256(compiled.channels).hexdigest()
        if label=='late' and a.pod_sample:
            # Cover the spatial domain and large stationary source coefficients.
            important=np.argsort(np.linalg.norm(compiled.channels[:,:,0],axis=1))[-min(64,len(compiled.z_m)):]
            sample_ids=np.unique(np.r_[sample_ids,important])
        sample=compiled.channels[sample_ids].copy()
        cpu,wall=process_time(),perf_counter()
        spectrum=apply_multi(band,compiled,compiled,positions,looks,alpha,np.ones(1),np.zeros(1),scale,
            radial_window_m=(3.,300.))
        appcpu,appwall=process_time()-cpu,perf_counter()-wall
        bins=inverse_bins(outomega,CubicSpline(omega,spectrum*np.exp(-1j*omega[:,None]*origins[None,:]),axis=0)(outomega),edges)
        saved[label]=dict(spectrum=spectrum,bins=bins)
        if reference_sample is None:reference_sample=sample;reference_hash=digest
        row=dict(compile_cpu_s=compcpu,compile_wall_s=compwall,apply_cpu_s=appcpu,apply_wall_s=appwall,
            source_channels_sha256=digest,source_channels_bytes=compiled.channels.nbytes,
            temporary_deposit_bytes=compiled.summary['time_rows']*5*277*8,
            sample_max_absolute_difference=float(np.max(abs(sample-reference_sample))),
            summary=compiled.summary)
        report['cases'][label]=row
        print(a.energy,label,'compile',compcpu,'apply',appcpu,'rows',compiled.summary['time_rows'],
            'angular elements',compiled.summary['angular_elements_per_node'],flush=True)
        if label=='early' and a.pod_sample:
            from lighthit.experimental.local_source_rank import charge_preserving_ranks
            local=replace(compiled,z_m=compiled.z_m[sample_ids],transverse_m=compiled.transverse_m[sample_ids],channels=sample)
            pod_reference=apply_multi(band,local,local,positions,looks,alpha,np.ones(1),np.zeros(1),scale,
                radial_window_m=(3.,300.))
            def tobins(s):
                return inverse_bins(outomega,CubicSpline(omega,s*np.exp(-1j*omega[:,None]*origins[None,:]),axis=0)(outomega),edges)
            rb=tobins(pod_reference);rows=[]
            for rank,approx,diagnostics in charge_preserving_ranks(sample,[1,2,3,4,6]):
                trial=replace(local,channels=np.ascontiguousarray(approx,dtype=np.complex64))
                result=apply_multi(band,trial,trial,positions,looks,alpha,np.ones(1),np.zeros(1),scale,
                    radial_window_m=(3.,300.))
                bb=tobins(result)
                rows.append(dict(rank=rank,bins_relative_l1=(abs(bb-rb).sum(1)/np.maximum(abs(rb).sum(1),1e-30)).tolist(),
                    charge_max_absolute_pe=float(np.max(abs(result[0]-pod_reference[0]))),**diagnostics))
            report['pod_sample']=dict(points=len(sample_ids),scope='sampled source points only; same RTE operator; source compression costs included separately',ranks=rows)
            np.savez_compressed(a.output/'source-sample.npz',channels=sample,points_m=local.points_m(),omega=omega,
                source_indices=sample_ids,detector_indices=ids)
        del compiled;gc.collect()
    actual,ref=saved['early'],saved['late']
    report['comparison']=dict(source_channels_bitwise_equal=report['cases']['early']['source_channels_sha256']==reference_hash,
        charge_max_absolute_pe=float(np.max(abs(actual['spectrum'][0]-ref['spectrum'][0]))),
        spectrum_max_relative=float(np.max(abs(actual['spectrum']-ref['spectrum']))/max(abs(ref['spectrum']).max(),1e-30)),
        bins_max_relative_l1=float(np.max(abs(actual['bins']-ref['bins']).sum(1)/np.maximum(abs(ref['bins']).sum(1),1e-30))))
    report['peak_process_mib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(2**20 if sys.platform=='darwin' else 2**10)
    report['cpu_compile_speedup']=report['cases']['late']['compile_cpu_s']/report['cases']['early']['compile_cpu_s']
    np.savez_compressed(a.output/'responses.npz',**{f'{k}_{n}':v for k,items in saved.items() for n,v in items.items()},time_origin_ns=origins)
    (a.output/'report.json').write_text(json.dumps(report,indent=2,default=str))
    print(json.dumps(report['comparison']),flush=True)


if __name__=='__main__':main()
