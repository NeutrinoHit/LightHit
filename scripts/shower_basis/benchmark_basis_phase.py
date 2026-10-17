#!/usr/bin/env python3
"""Compare a frozen pre-optimisation basis contraction against the current one.

The baseline must be a local snapshot of lighthit/_shower_basis.py. Saved G4
secondaries, one real wavelength, original 65-frequency source coefficients.
"""
import paths  # inputs and outputs: scripts/shower_basis/paths.py
import argparse
import hashlib
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path
from time import process_time

import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--backend',choices=['single','spectral'],default='single')
    a=p.parse_args()
    if a.output.exists():p.error('fresh output directory required')
    if a.repeats<1:p.error('positive repeats required')
    import lighthit as lh
    import ntsim,numba
    from ntsim.LightTransport import bgvd_model_from_ntsim
    from lighthit._shower_basis import compile_axial_slices,apply_multi,apply_multi_spectral
    from lighthit.experimental.axial_source import AxisFrame
    from lighthit.shower_basis import Ge2KernelStore
    if numba.get_num_threads()!=1:p.error('one thread required')
    name='lighthit._shower_basis_phase_baseline'
    spec=importlib.util.spec_from_file_location(name,a.baseline)
    baseline=importlib.util.module_from_spec(spec);sys.modules[name]=baseline
    spec.loader.exec_module(baseline)
    path=paths.muon_source(1000)
    with np.load(path) as z:
        source=lh.SpectralLightElements(**{k:z[k] for k in z.files if k!='metadata'},cone_model='spectral')
    model=bgvd_model_from_ntsim(ntsim.bgvd.Telescope(n_clusters=(1,2,3,4,5)),ntsim.bgvd.Water())
    kernel=model.kernel(lh.KernelConfig(omega_per_ns=np.linspace(0,1.2,65),cache_policy='require'))
    sample=kernel.medium.sample(kernel.wavelength.wavelength_nm);phase=sample['phase_index']
    source=source.subset((source.uid!=0)&(source.beta*phase.max()>1))
    geometry=source.field(0,phase_index=float(phase.max()));frame=AxisFrame.of(geometry)
    lam=4;wavelength=kernel.wavelength.wavelength_nm[lam]
    source=source.subset(source.beta*phase[lam]>1)
    first,second=source.field(0,phase_index=phase[lam]),source.field(2,phase_index=phase[lam])
    field=replace(first,photons=np.ascontiguousarray((first.photons-second.photons/phase[lam]**2)/wavelength**2))
    ids=np.array([197,232,342,666,1062,1241])
    positions=np.ascontiguousarray(model.detector.positions_m[ids]);looks=np.ascontiguousarray(-model.detector.orientations[ids])
    options=dict(degree=32,azimuthal_degree=4,cell_m=.5,axial_order=3,transverse='stencil',
        stencil_m=.2,stencil_threshold_m=.03,frame=frame,time_step_ns=.1,time_deposit='cubic',
        stencil_center='centroid',geometry_reference=geometry,channel_dtype=np.complex64,
        axial_clock_ns_per_m=1/.299792458)
    zero=compile_axial_slices(field,omega_per_ns=[0.],**options);selected=np.zeros(len(zero.z_m),bool)
    for receiver in positions:selected|=np.sum((zero.points_m()-receiver)**2,axis=1)<=(300.+1e-8)**2
    compiled=compile_axial_slices(field,omega_per_ns=kernel.config.omega_per_ns,point_selection=selected,**options)
    bundle=Ge2KernelStore.load(str(paths.GE2_PACK.path())).bundle(kernel,np.arange(65))
    band=dict(bundle,coefficients=bundle['coefficients'][lam:lam+1],absorption=bundle['absorption'][lam:lam+1],speed=bundle['speed'][lam:lam+1])
    alpha=np.ascontiguousarray(kernel.acceptance()['alpha'],float)
    scale=(kernel.wavelength.weight_nm[lam]*float(model.detector.spectral_weight(wavelength))*np.asarray(model.detector.effective_area_m2)[ids])[None,:]
    def evaluate(method):
        if a.backend=='spectral':
            return method(band,[compiled],positions,looks,alpha,scale,
                          channel_dtype=np.complex64,radial_window_m=(3.,300.))
        return method(band,compiled,compiled,positions,looks,alpha,np.ones(1),np.zeros(1),scale,radial_window_m=(3.,300.))
    methods=({'old':baseline.apply_multi_spectral,'new':apply_multi_spectral} if a.backend=='spectral'
             else {'old':baseline.apply_multi,'new':apply_multi})
    timings={key:[] for key in methods};spectra={}
    for method in methods.values():evaluate(method)
    for repeat in range(a.repeats):
        for key in (['old','new'] if repeat%2==0 else ['new','old']):
            cpu=process_time();spectra[key]=evaluate(methods[key]);timings[key].append(process_time()-cpu)
    error=float(abs(spectra['old']-spectra['new']).max()/abs(spectra['old']).max())
    medians={key:float(np.median(values)) for key,values in timings.items()}
    report=dict(threads=1,backend=a.backend,energy_GeV=1000,wavelength_nm=float(wavelength),receiver_indices=ids.tolist(),
        source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        baseline_sha256=hashlib.sha256(a.baseline.read_bytes()).hexdigest(),
        current_sha256=hashlib.sha256((Path(lh.__file__).parent/'_shower_basis.py').read_bytes()).hexdigest(),
        source_channels_sha256=hashlib.sha256(compiled.channels).hexdigest(),
        points=len(compiled.z_m),cpu_repeats_s=timings,median_cpu_s=medians,
        speedup=medians['old']/medians['new'],normalized_spectrum_error=error,
        scope='one-band real secondary-basis apply, unchanged compiled source, not event timing')
    a.output.mkdir(parents=True);(a.output/'report.json').write_text(json.dumps(report,indent=2))
    np.savez_compressed(a.output/'spectra.npz',**spectra)
    print(json.dumps(report,indent=2),flush=True)
    if error>1e-10:raise AssertionError('common-phase contraction changed output')


if __name__=='__main__':main()
