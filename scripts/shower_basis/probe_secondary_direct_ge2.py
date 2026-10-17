#!/usr/bin/env python3
"""Original G4-chord >=2 control without spatial/time source interpolation.

Actual 3D positions, two chord Gauss nodes, beta-dependent cones, exact emission
phases. The same finite-L/M RTE cache and radial window remain approximations.
"""
import paths  # inputs and outputs: scripts/shower_basis/paths.py
import argparse,json,hashlib
from pathlib import Path
from dataclasses import replace
from time import process_time,perf_counter
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--om',type=int,default=232)
    p.add_argument('--chunk',type=int,default=512)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('fresh output directory required')
    import lighthit as lh,ntsim,numba
    from ntsim.LightTransport import bgvd_model_from_ntsim
    from lighthit._shower_basis import _cone_harmonics,apply_multi
    from lighthit.experimental.axial_fast import _layout
    from lighthit.experimental.axial_source import AxisFrame,AxialSource
    from lighthit.shower_basis import Ge2KernelStore
    from lighthit.readout import inverse_bins
    from scipy.interpolate import CubicSpline
    if numba.get_num_threads()!=1:p.error('one Numba thread required')
    r=json.loads((a.reference/'report.json').read_text())
    om=r['receiver_indices'].index(a.om)
    with np.load(a.reference/'responses.npz') as z:
        expected=z['early_spectrum'][:,om].copy();expected_bins=z['early_bins'][om].copy();origin=z['time_origin_ns'][om]
    path=Path(r['source'])
    if hashlib.sha256(path.read_bytes()).hexdigest()!=r['source_sha256']:p.error('source changed')
    with np.load(path) as z:source=lh.SpectralLightElements(**{k:z[k] for k in z.files if k!='metadata'},cone_model='spectral')
    model=bgvd_model_from_ntsim(ntsim.bgvd.Telescope(n_clusters=(1,2,3,4,5)),ntsim.bgvd.Water())
    if model.fingerprint!=r['model_fingerprint']:p.error('model changed')
    kernel=model.kernel(lh.KernelConfig(omega_per_ns=np.linspace(0,1.2,65),cache_policy='require'))
    wl=r['wavelength_nm'];lam=int(np.argmin(abs(kernel.wavelength.wavelength_nm-wl)))
    phase=float(kernel.medium.sample([wl])['phase_index'][0])
    source=source.subset((source.uid!=0)&(source.beta*phase>1))
    f0,f2=source.field(0,phase_index=phase),source.field(2,phase_index=phase)
    field=replace(f0,photons=(f0.photons-f2.photons/phase**2)/wl**2);frame=AxisFrame.of(field)
    position=np.ascontiguousarray(model.detector.positions_m[a.om:a.om+1]);look=np.ascontiguousarray(-model.detector.orientations[a.om:a.om+1])
    alpha=np.ascontiguousarray(kernel.acceptance()['alpha'],float)
    bundle=Ge2KernelStore.load(str(paths.GE2_PACK.path())).bundle(kernel,np.arange(65))
    band=dict(bundle,coefficients=bundle['coefficients'][lam:lam+1],absorption=bundle['absorption'][lam:lam+1],speed=bundle['speed'][lam:lam+1])
    scale=np.array([[kernel.wavelength.weight_nm[lam]*float(model.detector.spectral_weight(wl))*model.detector.effective_area_m2[a.om]]])
    omega=kernel.config.omega_per_ns;outomega=np.linspace(0,1.2,321);edges=kernel.config.relative_time_edges_ns
    starts,aa,bb,kept,degree=_layout(32,4);reference=float(field.start_ns.min());total=np.zeros(65,complex)
    gx,gw=np.polynomial.legendre.leggauss(2);points_count=0
    cpu,wall=process_time(),perf_counter();a.output.mkdir(parents=True)
    for fraction,weight in zip((gx+1)/2,gw/2):
        points=field.start_m+fraction*field.length_m[:,None]*field.direction
        distances=np.linalg.norm(points-position[0],axis=1)
        selected=np.flatnonzero((distances>=3.)&(distances<=300.))
        for start in range(0,len(selected),a.chunk):
            ids=selected[start:start+a.chunk];local=frame.rotate(points[ids]-frame.centre_m)
            harmonics=np.empty((len(ids),len(kept)))
            _cone_harmonics(np.ascontiguousarray(frame.rotate(field.direction[ids])),
                np.ascontiguousarray(field.cone_cosine[ids]),starts,aa,bb,4,harmonics)
            times=field.start_ns[ids]+fraction*(field.end_ns[ids]-field.start_ns[ids])
            channels=np.ascontiguousarray(harmonics[:,:,None]*(field.photons[ids]*weight)[:,None,None]*
                np.exp(1j*(times-reference)[:,None,None]*omega[None,None,:]))
            raw=AxialSource(frame,local[:,2],local[:,:2],channels,kept,degree,32,4,reference,{})
            total+=apply_multi(band,raw,raw,position,look,alpha,np.ones(1),np.zeros(1),scale,radial_window_m=(3.,300.))[:,0]
            points_count+=len(ids)
        print('node',fraction,'points',len(selected),'cpu',process_time()-cpu,flush=True)
    bins=inverse_bins(outomega,CubicSpline(omega,total*np.exp(-1j*omega*origin))(outomega)[:,None],edges)[0]
    report=dict(om=a.om,reference=str(a.reference),wavelength_nm=wl,source_sha256=r['source_sha256'],
        original_source_elements=len(field),quadrature_points=points_count,threads=1,
        cpu_seconds=process_time()-cpu,wall_seconds=perf_counter()-wall,
        direct_charge_pe=float(total[0].real),basis_charge_pe=float(expected[0].real),
        basis_charge_relative_error=float(abs(expected[0]-total[0])/max(abs(total[0]),1e-30)),
        basis_bins_relative_l1=float(abs(bins-expected_bins).sum()/max(abs(bins).sum(),1e-30)),
        limitations='same finite angular RTE/M4/window and 65-frequency readout; two chord nodes, no spatial or emission-phase interpolation')
    np.savez_compressed(a.output/'response.npz',spectrum=total,bins=bins,origin_ns=origin)
    (a.output/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
