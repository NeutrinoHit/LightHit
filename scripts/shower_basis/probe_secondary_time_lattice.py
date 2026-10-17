#!/usr/bin/env python3
"""Compare temporal source lattices for real muon descendant families.

The RTE operator, source records, spectrum, geometry and angular/spatial basis
are fixed. Only the linear emission-time deposition step changes. This probes
an implementation cost, not a mean-shower substitution or absolute accuracy.
"""
import paths  # inputs and outputs: scripts/shower_basis/paths.py
import argparse, json, gc
from itertools import product
from pathlib import Path
from dataclasses import replace
from time import process_time, perf_counter
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--energy',type=int,choices=[300,1000,3000],default=1000)
    p.add_argument('--families',type=int,default=3)
    p.add_argument('--steps-ns',type=float,nargs='+',default=[.02,.1,.2])
    p.add_argument('--all-bands',action='store_true',help='use all nine cached spectral quadrature nodes')
    p.add_argument('--cubic',action='store_true',help='reference stays linear .02 ns; other steps use cubic time')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('fresh output required')
    import lighthit as lh, ntsim, numba
    from ntsim.LightTransport import bgvd_model_from_ntsim
    from lighthit.experimental.axial_source import AxisFrame
    from lighthit.experimental.muon_packets import partition_muon
    from lighthit._shower_basis import compile_axial_slices,apply_multi,earliest_element_arrival
    from lighthit.shower_basis import Ge2KernelStore
    from lighthit.readout import inverse_bins
    from scipy.interpolate import CubicSpline
    if numba.get_num_threads()!=1:p.error('one Numba thread required')
    if a.steps_ns[0]!=.02 or min(a.steps_ns)<=0:p.error('first step must be .02 ns, all positive')
    with np.load(paths.muon_source(a.energy)) as z:
        source=lh.SpectralLightElements(**{k:z[k] for k in z.files if k!='metadata'},cone_model='spectral')
        source_metadata=json.loads(str(z['metadata']))
    with np.load(paths.muon_ancestry(a.energy)) as z:
        ancestry={k:z[k] for k in z.files if k!='metadata'}
    packets=partition_muon(source,ancestry)
    model=bgvd_model_from_ntsim(ntsim.bgvd.Telescope(n_clusters=(1,2,3,4,5)),ntsim.bgvd.Water())
    if source_metadata['model_fingerprint']!=model.fingerprint:
        p.error('saved G4 source and current optical/detector model differ')
    kernel=model.kernel(lh.KernelConfig(omega_per_ns=np.linspace(0,1.2,65),cache_policy='require'))
    store=Ge2KernelStore.load(str(paths.GE2_PACK.path()))
    bundle=store.bundle(kernel,np.arange(65))
    lam=len(kernel.wavelength.wavelength_nm)//2
    wavelength=float(kernel.wavelength.wavelength_nm[lam]);phase=float(kernel.medium.sample([wavelength])['phase_index'][0])
    def field_of(part):
        part=part.subset(part.beta*phase>1)
        f0,f2=part.field(0,phase_index=phase),part.field(2,phase_index=phase)
        return replace(f0,photons=np.ascontiguousarray((f0.photons-f2.photons/phase**2)/wavelength**2))
    candidates=[packet for packet in packets if packet.kind=='shower']
    chosen=sorted(candidates,key=lambda x:source.coefficient0[x.indices].sum(),reverse=True)[:a.families]
    # Keep non-shower physics represented in the bounded test as well.
    for kind in ['soft','general','late']:
        others=[x for x in packets if x.kind==kind]
        if others:chosen.append(max(others,key=lambda x:source.coefficient0[x.indices].sum()))
    # Include modules near each tested family, including families far from
    # the previously selected bright OMs. The seven fixed controls span five clusters.
    nearest=[int(i) for packet in chosen for i in np.argsort(np.linalg.norm(
        model.detector.positions_m-packet.source(source).centroid_m,axis=1))[:2]]
    ids=np.array(list(dict.fromkeys([196,197,198,342,666,1062,1241]+nearest)))
    positions=np.ascontiguousarray(model.detector.positions_m[ids]);looks=np.ascontiguousarray(-model.detector.orientations[ids])
    areas=np.asarray(model.detector.effective_area_m2)[ids]
    alpha=np.ascontiguousarray(kernel.acceptance()['alpha'],float)
    band=dict(bundle,coefficients=bundle['coefficients'][lam:lam+1],
        absorption=bundle['absorption'][lam:lam+1],speed=bundle['speed'][lam:lam+1])
    scale=(kernel.wavelength.weight_nm[lam]*float(model.detector.spectral_weight(wavelength))*areas)[None,:]
    omega=kernel.config.omega_per_ns;outomega=np.linspace(0,1.2,321);edges=kernel.config.relative_time_edges_ns
    origins=earliest_element_arrival(positions,source.start_m,source.direction,source.length_m,
        source.start_ns,float(kernel.medium.group_index.min()))
    def compile_field(field,frame,step):
        return compile_axial_slices(field,32,omega,azimuthal_degree=4,cell_m=.5,axial_order=3,
            transverse='stencil',stencil_m=.2,stencil_threshold_m=.03,frame=frame,
            time_step_ns=step,time_deposit='cubic' if a.cubic and step!=.02 else 'linear',
            channel_dtype=np.complex64,stencil_center='centroid')
    tiny=field_of(source.subset(np.arange(len(source))==0));warm=compile_field(tiny,AxisFrame.of(tiny),.02)
    apply_multi(band,warm,warm,positions[:1],looks[:1],alpha,np.ones(1),np.zeros(1),scale[:,:1],radial_window_m=(3.,300.))
    a.output.mkdir(parents=True)
    report=dict(energy_GeV=a.energy,threads=1,clusters=5,receiver_indices=ids.tolist(),
        model_fingerprint=model.fingerprint,
        selected_secondary_C0_fraction=float(sum(source.coefficient0[x.indices].sum() for x in chosen)/
            source.coefficient0[source.uid!=0].sum()),
        wavelength_nm=wavelength if not a.all_bands else kernel.wavelength.wavelength_nm.tolist(),
        frequency_nodes=65,scope='selected real descendant packets; cached spectral quadrature; >=2 only',
        candidate_time_deposit='cubic' if a.cubic else 'linear',
        reference='same RTE/spatial/angular operator with 0.02 ns source lattice',packets=[],aggregate={})
    summed={str(dt):np.zeros((65,len(ids)),complex) for dt in a.steps_ns}
    def bins_of(s):
        relative=s*np.exp(-1j*omega[:,None]*origins[None,:])
        return inverse_bins(outomega,CubicSpline(omega,relative,axis=0)(outomega),edges)
    def metric(s,ref):
        b,br=bins_of(s),bins_of(ref)
        return dict(charge_pe=s[0].real.tolist(),charge_absolute_error_pe=abs(s[0]-ref[0]).tolist(),
            bins_absolute_l1_pe=abs(b-br).sum(1).tolist(),
            bins_relative_l1=(abs(b-br).sum(1)/np.maximum(abs(br).sum(1),1e-30)).tolist())
    bands=range(len(kernel.wavelength.wavelength_nm)) if a.all_bands else [lam]
    for lam,packet in product(bands,chosen):
        wavelength=float(kernel.wavelength.wavelength_nm[lam]);phase=float(kernel.medium.sample([wavelength])['phase_index'][0])
        band=dict(bundle,coefficients=bundle['coefficients'][lam:lam+1],
            absorption=bundle['absorption'][lam:lam+1],speed=bundle['speed'][lam:lam+1])
        scale=(kernel.wavelength.weight_nm[lam]*float(model.detector.spectral_weight(wavelength))*areas)[None,:]
        field=field_of(packet.source(source));frame=AxisFrame.of(field)
        row=dict(kind=packet.kind,family_uid=packet.family_uid,root_energy_GeV=packet.root_energy_GeV,
            elements=len(field),wavelength_nm=wavelength,cases={})
        ref=None
        for dt in a.steps_ns:
            cpu,wall=process_time(),perf_counter();compiled=compile_field(field,frame,dt)
            compcpu,compwall=process_time()-cpu,perf_counter()-wall
            cpu,wall=process_time(),perf_counter()
            spectrum=apply_multi(band,compiled,compiled,positions,looks,alpha,np.ones(1),np.zeros(1),scale,
                radial_window_m=(3.,300.))
            appcpu,appwall=process_time()-cpu,perf_counter()-wall
            if ref is None:ref=spectrum.copy()
            summed[str(dt)]+=spectrum
            row['cases'][str(dt)]=dict(compile_cpu_s=compcpu,compile_wall_s=compwall,
                apply_cpu_s=appcpu,apply_wall_s=appwall,channels_bytes=compiled.channels.nbytes,
                summary=compiled.summary,**metric(spectrum,ref))
            print(a.energy,packet.kind,packet.family_uid,dt,'compile',compcpu,'apply',appcpu,
                'max L1',max(row['cases'][str(dt)]['bins_relative_l1']),flush=True)
            del compiled;gc.collect()
        report['packets'].append(row)
        (a.output/'report.json').write_text(json.dumps(report,indent=2,default=str))
    for dt,spectrum in summed.items():report['aggregate'][dt]=metric(spectrum,summed[str(a.steps_ns[0])])
    np.savez_compressed(a.output/'spectra.npz',**{f'step{dt}':s for dt,s in summed.items()},time_origin_ns=origins)
    (a.output/'report.json').write_text(json.dumps(report,indent=2,default=str))


if __name__=='__main__':main()
