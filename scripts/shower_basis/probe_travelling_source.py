#!/usr/bin/env python3
"""Coarser axial source cells with an explicit travelling-phase carrier.

Uses an existing entire-secondary fine-grid control, preserving its reporting
OMs, cached wavelength, source, medium, 65 frequencies and 5 ns readout. This is
an accuracy/speed prototype; no production defaults or source clocks change.
"""
import paths  # inputs and outputs: scripts/shower_basis/paths.py
import argparse,json,gc,hashlib
from pathlib import Path
from dataclasses import replace
from time import process_time,perf_counter
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--cells-m',type=float,nargs='+',default=[1.,2.])
    p.add_argument('--local-radius-m',type=float,nargs='+',help='correct coarse response with fine-minus-coarse nearby source, using fixed full-source geometry')
    p.add_argument('--travelling-only',action='store_true')
    p.add_argument('--time-step-ns',type=float,default=.5)
    p.add_argument('--oms',type=int,nargs='+')
    p.add_argument('--channel-dtype',choices=['complex64','complex128'],default='complex64')
    p.add_argument('--stencil-center',choices=['axis','centroid'],default='centroid')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('fresh output directory required')
    import lighthit as lh,ntsim,numba
    from ntsim.LightTransport import bgvd_model_from_ntsim
    from lighthit._shower_basis import compile_axial_slices,apply_multi
    from lighthit.experimental.axial_source import AxisFrame
    from lighthit.shower_basis import Ge2KernelStore
    from lighthit.readout import inverse_bins
    from scipy.interpolate import CubicSpline
    if numba.get_num_threads()!=1:p.error('one Numba thread required')
    r=json.loads((a.reference/'report.json').read_text());ids=np.asarray(r['receiver_indices'])
    with np.load(a.reference/'responses.npz') as z:
        expected=z['early_spectrum'].copy();expected_bins=z['early_bins'].copy();origins=z['time_origin_ns'].copy()
    if a.oms:
        columns=[r['receiver_indices'].index(i) for i in a.oms]
        ids=np.asarray(a.oms);expected=expected[:,columns];expected_bins=expected_bins[columns];origins=origins[columns]
    source_path=Path(r['source'])
    if hashlib.sha256(source_path.read_bytes()).hexdigest()!=r['source_sha256']:p.error('source changed')
    with np.load(source_path) as z:
        source=lh.SpectralLightElements(**{k:z[k] for k in z.files if k!='metadata'},cone_model='spectral')
    model=bgvd_model_from_ntsim(ntsim.bgvd.Telescope(n_clusters=(1,2,3,4,5)),ntsim.bgvd.Water())
    if model.fingerprint!=r['model_fingerprint']:p.error('model changed')
    kernel=model.kernel(lh.KernelConfig(omega_per_ns=np.linspace(0,1.2,65),cache_policy='require'))
    wavelength=r['wavelength_nm'];lam=int(np.argmin(abs(kernel.wavelength.wavelength_nm-wavelength)))
    if kernel.wavelength.wavelength_nm[lam]!=wavelength:p.error('cached spectral rule changed')
    phase=float(kernel.medium.sample([wavelength])['phase_index'][0])
    source=source.subset((source.uid!=0)&(source.beta*phase>1))
    f0,f2=source.field(0,phase_index=phase),source.field(2,phase_index=phase)
    field=replace(f0,photons=np.ascontiguousarray((f0.photons-f2.photons/phase**2)/wavelength**2));frame=AxisFrame.of(field)
    positions=np.ascontiguousarray(model.detector.positions_m[ids]);looks=np.ascontiguousarray(-model.detector.orientations[ids])
    areas=np.asarray(model.detector.effective_area_m2)[ids];alpha=np.ascontiguousarray(kernel.acceptance()['alpha'],float)
    bundle=Ge2KernelStore.load(str(paths.GE2_PACK.path())).bundle(kernel,np.arange(65))
    band=dict(bundle,coefficients=bundle['coefficients'][lam:lam+1],absorption=bundle['absorption'][lam:lam+1],speed=bundle['speed'][lam:lam+1])
    scale=(kernel.wavelength.weight_nm[lam]*float(model.detector.spectral_weight(wavelength))*areas)[None,:]
    omega=kernel.config.omega_per_ns;outomega=np.linspace(0,1.2,321);edges=kernel.config.relative_time_edges_ns
    controls=(model.detector.positions_m if r['geometry_selection_receivers']==1440 and not a.oms else positions)
    a.output.mkdir(parents=True)
    report=dict(reference=str(a.reference),energy_GeV=r['energy_GeV'],threads=1,receiver_indices=ids.tolist(),
        scope=r['scope'],source_sha256=r['source_sha256'],model_fingerprint=r['model_fingerprint'],cases=[])
    saved={}
    def metric(spectrum):
        bins=inverse_bins(outomega,CubicSpline(omega,spectrum*np.exp(-1j*omega[:,None]*origins[None,:]),axis=0)(outomega),edges)
        return dict(charge_relative_error=(abs(spectrum[0]-expected[0])/np.maximum(abs(expected[0]),1e-30)).tolist(),
            charge_absolute_error_pe=abs(spectrum[0]-expected[0]).tolist(),
            bins_relative_l1=(abs(bins-expected_bins).sum(1)/np.maximum(abs(expected_bins).sum(1),1e-30)).tolist(),
            bins_absolute_l1_pe=abs(bins-expected_bins).sum(1).tolist())
    def supported_points(points,part,cell):
        # Conservative cubic axial support for all chord quadrature nodes.
        grid=np.unique(points[:,0]);mark=np.zeros(len(grid)+1,np.int64)
        z0=frame.rotate(part.start_m-frame.centre_m)[:,2]
        z1=frame.rotate(part.start_m+part.length_m[:,None]*part.direction-frame.centre_m)[:,2]
        lo=np.searchsorted(grid,np.minimum(z0,z1)-2*cell-1e-8)
        hi=np.searchsorted(grid,np.maximum(z0,z1)+2*cell+1e-8,side='right')
        np.add.at(mark,lo,1);np.add.at(mark,hi,-1)
        return (np.cumsum(mark[:-1])>0)[np.searchsorted(grid,points[:,0])]
    for cell in a.cells_m:
        for slope in ([1/.299792458] if a.local_radius_m or a.travelling_only else [0.,1/.299792458]):
            options=dict(degree=32,azimuthal_degree=4,cell_m=cell,axial_order=3,
                transverse='stencil',stencil_m=.2,stencil_threshold_m=.03,frame=frame,
                time_step_ns=a.time_step_ns,time_deposit='cubic',channel_dtype=np.dtype(a.channel_dtype),
                stencil_center=a.stencil_center,axial_clock_ns_per_m=slope)
            zero=compile_axial_slices(field,omega_per_ns=[0.],**options)
            points=zero.points_m();selected=np.zeros(len(points),bool)
            for receiver in controls:selected|=np.sum((points-receiver)**2,axis=1)<=(300.+1e-8)**2
            del zero;gc.collect()
            cpu,wall=process_time(),perf_counter()
            compiled=compile_axial_slices(field,omega_per_ns=omega,point_selection=selected,**options)
            compcpu,compwall=process_time()-cpu,perf_counter()-wall
            cpu,wall=process_time(),perf_counter()
            spectrum=apply_multi(band,compiled,compiled,positions,looks,alpha,np.ones(1),np.zeros(1),scale,radial_window_m=(3.,300.))
            appcpu,appwall=process_time()-cpu,perf_counter()-wall
            bins=inverse_bins(outomega,CubicSpline(omega,spectrum*np.exp(-1j*omega[:,None]*origins[None,:]),axis=0)(outomega),edges)
            row=dict(cell_m=cell,clock_slope=slope,compile_cpu_s=compcpu,compile_wall_s=compwall,
                time_step_ns=a.time_step_ns,
                channel_dtype=a.channel_dtype,
                stencil_center=a.stencil_center,
                apply_cpu_s=appcpu,apply_wall_s=appwall,summary=compiled.summary,
                charge_relative_error=(abs(spectrum[0]-expected[0])/np.maximum(abs(expected[0]),1e-30)).tolist(),
                charge_absolute_error_pe=abs(spectrum[0]-expected[0]).tolist(),
                bins_relative_l1=(abs(bins-expected_bins).sum(1)/np.maximum(abs(expected_bins).sum(1),1e-30)).tolist(),
                bins_absolute_l1_pe=abs(bins-expected_bins).sum(1).tolist())
            print(r['energy_GeV'],cell,slope,'compile',compcpu,'apply',appcpu,'max bins L1',max(row['bins_relative_l1']),flush=True)
            report['cases'].append(row);saved[f'cell{cell}-clock{slope}']=spectrum
            del compiled;gc.collect()
            (a.output/'report.json').write_text(json.dumps(report,indent=2,default=str))
            if a.local_radius_m:
                from scipy.spatial import cKDTree
                nearest=cKDTree(model.detector.positions_m).query(field.midpoints_m,workers=1)[0]
                geometry=field
                for radius in a.local_radius_m:
                    mask=nearest<=radius+.5*field.length_m
                    part=replace(field,**{name:getattr(field,name)[mask] for name in
                        ['start_m','direction','length_m','photons','cone_cosine','start_ns','end_ns','row_index','uid']})
                    terms=[];clocks=[];summaries=[]
                    for h in [.5,cell]:
                        settings=dict(options,cell_m=h,geometry_reference=geometry)
                        z=compile_axial_slices(part,omega_per_ns=[0.],**settings)
                        zp=z.points_m();keep=supported_points(np.stack([z.z_m,z.z_m,z.z_m],axis=1),part,h)
                        reachable=np.zeros(len(zp),bool)
                        for receiver in controls:reachable|=np.sum((zp-receiver)**2,axis=1)<=(300.+1e-8)**2
                        keep&=reachable;del z;gc.collect()
                        cpu=process_time()
                        compiled=compile_axial_slices(part,omega_per_ns=omega,point_selection=keep,**settings)
                        compile_cpu=process_time()-cpu;cpu=process_time()
                        terms.append(apply_multi(band,compiled,compiled,positions,looks,alpha,np.ones(1),np.zeros(1),scale,radial_window_m=(3.,300.)))
                        clocks.append(dict(compile_cpu_s=compile_cpu,apply_cpu_s=process_time()-cpu))
                        summaries.append(compiled.summary);del compiled;gc.collect()
                    corrected=spectrum+terms[0]-terms[1]
                    rr=dict(cell_m=cell,clock_slope=slope,local_radius_m=radius,near_elements=len(part),
                        correction_timings=clocks,correction_summaries=summaries,**metric(corrected))
                    print('local correction',radius,'elements',len(part),'max bins L1',max(rr['bins_relative_l1']),flush=True)
                    report['cases'].append(rr);saved[f'cell{cell}-local{radius}']=corrected
                    (a.output/'report.json').write_text(json.dumps(report,indent=2,default=str))
    np.savez_compressed(a.output/'spectra.npz',**saved)


if __name__=='__main__':main()
