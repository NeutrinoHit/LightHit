#!/usr/bin/env python3
"""Original-step versus axial-source >=2, full cached spectral quadrature.

Five-cluster geometry, saved long G4 muon, actual 3D emission points and clocks.
Both sides retain L32/M4, cache radial window and frequency/readout approximations.
This measures source representation, not absolute RTE accuracy or a full event.
"""
import paths  # inputs and outputs: scripts/shower_basis/paths.py
import argparse,gc,hashlib,json
from dataclasses import replace
from pathlib import Path
from time import process_time,perf_counter
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--energy',type=int,choices=[300,1000,3000],default=1000)
    p.add_argument('--oms',type=int,nargs='+',default=[197,232,342,666,1062,1241])
    p.add_argument('--chunk',type=int,default=128)
    p.add_argument('--band-group',type=int,help='default: 9 for real points, 3 for complex fields')
    p.add_argument('--chord-order',type=int,choices=[2,4,8],default=2)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--direct-only',action='store_true')
    p.add_argument('--reuse-direct',type=Path)
    p.add_argument('--direct-backend',choices=['ordinary','point'],default='point')
    p.add_argument('--carrier-cells-m',type=float,nargs='+',default=[.5,1.])
    p.add_argument('--skip-standard',action='store_true',help='refinement against a saved direct control')
    a=p.parse_args()
    if a.band_group is None:a.band_group=9 if a.direct_backend=='point' else 3
    if a.output.exists():p.error('fresh output directory required')
    if a.chunk<1 or not 1<=a.band_group<=9:p.error('invalid chunk or band group')
    if (not np.isfinite(a.carrier_cells_m).all() or min(a.carrier_cells_m)<=0
            or len(set(a.carrier_cells_m))!=len(a.carrier_cells_m)):
        p.error('carrier cell sizes must be distinct positive finite values')
    import lighthit as lh,ntsim,numba
    from ntsim.LightTransport import bgvd_model_from_ntsim
    from lighthit._shower_basis import compile_axial_slices,_cone_harmonics,apply_multi,apply_multi_spectral,earliest_element_arrival
    from lighthit.experimental.axial_source import AxisFrame,AxialSource
    from lighthit.experimental.axial_fast import _layout
    from lighthit.shower_basis import Ge2KernelStore
    from lighthit.readout import inverse_bins
    from lighthit.experimental.point_ge2 import apply_point_spectral
    from scipy.interpolate import CubicSpline
    if numba.get_num_threads()!=1:p.error('one Numba thread required')
    path=paths.muon_source(a.energy)
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    with np.load(path) as z:
        source=lh.SpectralLightElements(**{k:z[k] for k in z.files if k!='metadata'},cone_model='spectral')
        provenance=json.loads(str(z['metadata']))
    model=bgvd_model_from_ntsim(ntsim.bgvd.Telescope(n_clusters=(1,2,3,4,5)),ntsim.bgvd.Water())
    if provenance['model_fingerprint']!=model.fingerprint:p.error('model/source mismatch')
    kernel=model.kernel(lh.KernelConfig(omega_per_ns=np.linspace(0,1.2,65),cache_policy='require'))
    wl=kernel.wavelength.wavelength_nm;weights=kernel.wavelength.weight_nm
    sample=kernel.medium.sample(wl);phase=sample['phase_index'];omega=kernel.config.omega_per_ns
    source=source.subset((source.uid!=0)&(source.beta*phase.max()>1))
    geometry=source.field(0,phase_index=float(phase.max()));frame=AxisFrame.of(geometry)
    ids=np.asarray(a.oms,np.int64)
    positions=np.ascontiguousarray(model.detector.positions_m[ids]);looks=np.ascontiguousarray(-model.detector.orientations[ids])
    areas=np.asarray(model.detector.effective_area_m2)[ids];alpha=np.ascontiguousarray(kernel.acceptance()['alpha'],float)
    scales=weights[:,None]*np.asarray(model.detector.spectral_weight(wl))[:,None]*areas[None,:]
    origins=earliest_element_arrival(positions,source.start_m,source.direction,source.length_m,source.start_ns,float(sample['group_index'].min()))
    store=Ge2KernelStore.load(str(paths.GE2_PACK.path()));bundle=store.bundle(kernel,np.arange(65))
    pack=paths.GE2_PACK.path() / 'pack.json'
    manifest=json.loads(pack.read_text());operator_hash=hashlib.sha256(pack.read_bytes())
    for entry in manifest['bundles']:
        if len(entry['key'][2])!=len(omega):continue
        with (pack.parent/entry['file']).resolve().open('rb') as stream:
            while block:=stream.read(16*1024*1024):operator_hash.update(block)
    def band_bundle(lo,hi):
        return dict(bundle,coefficients=bundle['coefficients'][lo:hi],absorption=bundle['absorption'][lo:hi],speed=bundle['speed'][lo:hi])
    starts,aa,bb,kept,channel_degree=_layout(32,4);reference=float(source.start_ns.min())
    def field_for(lam):
        valid=source.beta*phase[lam]>1
        if not np.any(valid):return None
        s=source if np.all(valid) else source.subset(valid)
        f0,f2=s.field(0,phase_index=float(phase[lam])),s.field(2,phase_index=float(phase[lam]))
        return replace(f0,photons=np.ascontiguousarray((f0.photons-f2.photons/phase[lam]**2)/wl[lam]**2))
    output_omega=np.linspace(0,1.2,321);edges=kernel.config.relative_time_edges_ns
    def bins_of(s):
        return inverse_bins(output_omega,CubicSpline(omega,s*np.exp(-1j*omega[:,None]*origins[None,:]),axis=0)(output_omega),edges)
    report=dict(energy_GeV=a.energy,source=str(path),source_sha256=digest,model_fingerprint=model.fingerprint,
        receiver_indices=ids.tolist(),clusters=5,threads=1,wavelength_nm=wl.tolist(),frequency_nodes=65,
        source_elements=len(source),shared_frame='C0 at maximum phase index',source_m=4,source_l=32,
        radial_window_m=[3.,300.],scope='entire secondary source; selected OMs; >=2 only',complete=False,direct_groups=[],variants={})
    report['direct_backend']=a.direct_backend
    report['band_group']=a.band_group
    report.update(kernel_pack_sha256=operator_hash.hexdigest(),source_omega_per_ns=omega.tolist(),
        readout_omega_per_ns=output_omega.tolist(),relative_time_edges_ns=edges.tolist(),chord_gauss_order=a.chord_order,
        waveform_scope='agreement at shared sampled/interpolated operator; not proof of late-time/Nyquist accuracy')
    a.output.mkdir(parents=True)
    def save_report():(a.output/'report.json').write_text(json.dumps(report,indent=2,default=str))
    # Warm both contractions with a real point, keeping initialization outside timings.
    vec=np.ascontiguousarray(frame.rotate(source.direction[:1]));harm=np.empty((1,len(kept)))
    _cone_harmonics(vec,np.array([1/(source.beta[0]*phase.max())]),starts,aa,bb,4,harm)
    loc=frame.rotate(source.start_m[:1]-frame.centre_m)
    tiny=AxialSource(frame,loc[:,2],loc[:,:2],np.ascontiguousarray(harm[:,:,None]*np.ones((1,1,65),complex)),kept,channel_degree,32,4,reference,{})
    apply_multi(band_bundle(0,1),tiny,tiny,positions[:1],looks[:1],alpha,np.ones(1),np.zeros(1),scales[:1,:1],radial_window_m=(3.,300.))
    apply_multi_spectral(band_bundle(0,1),[tiny],positions[:1],looks[:1],alpha,scales[:1,:1],radial_window_m=(3.,300.),channel_dtype=np.complex128)
    apply_point_spectral(band_bundle(0,1),frame,source.start_m[:1],harm[None,:,:],source.start_ns[:1],positions[:1],looks[:1],alpha,scales[:1,:1])
    direct=np.zeros((65,len(ids)),complex)
    if a.reuse_direct:
        old=json.loads((a.reuse_direct/'report.json').read_text())
        for key in ['source_sha256','model_fingerprint','receiver_indices','wavelength_nm','shared_frame',
                    'kernel_pack_sha256','source_l','source_m','source_omega_per_ns','readout_omega_per_ns',
                    'relative_time_edges_ns','chord_gauss_order','radial_window_m']:
            if old.get(key)!=report[key]:p.error(f'direct reference differs: {key}')
        if not old.get('direct_complete'):p.error('direct reference is incomplete')
        with np.load(a.reuse_direct/'direct.npz') as z:
            np.testing.assert_array_equal(z['time_origin_ns'],origins);direct=z['spectrum'].copy()
        report['direct_reused_from']=str(a.reuse_direct)
    else:
        gx,gw=np.polynomial.legendre.leggauss(a.chord_order)
        for lo in range(0,len(wl),a.band_group):
            hi=min(len(wl),lo+a.band_group);cpu,wall=process_time(),perf_counter();npoints=0
            for fraction,weight in zip((gx+1)/2,gw/2):
                points=source.start_m+fraction*source.length_m[:,None]*source.direction
                reachable=np.zeros(len(points),bool)
                for receiver in positions:
                    d2=np.sum((points-receiver)**2,axis=1);reachable|=(d2>=3**2)&(d2<=300**2)
                chosen=np.flatnonzero(reachable)
                for start in range(0,len(chosen),a.chunk):
                    take=chosen[start:start+a.chunk];local=frame.rotate(points[take]-frame.centre_m)
                    vectors=np.ascontiguousarray(frame.rotate(source.direction[take]));tau=source.start_ns[take]+fraction*(source.end_ns[take]-source.start_ns[take])-reference
                    sources=[];point_angles=[]
                    temporal=np.exp(1j*tau[:,None]*omega) if a.direct_backend=='ordinary' else None
                    for lam in range(lo,hi):
                        active=source.beta[take]*phase[lam]>1
                        cones=np.where(active,1/(source.beta[take]*phase[lam]),1.)
                        angular=np.empty((len(take),len(kept)))
                        _cone_harmonics(vectors,np.ascontiguousarray(cones),starts,aa,bb,4,angular)
                        photons=np.where(active,source.coefficient0[take]+source.coefficient2[take]/phase[lam]**2,0.)*weight/wl[lam]**2
                        if a.direct_backend=='point':
                            point_angles.append(angular*photons[:,None])
                        else:
                            channels=np.ascontiguousarray(angular[:,:,None]*photons[:,None,None]*temporal[:,None,:])
                            sources.append(AxialSource(frame,local[:,2],local[:,:2],channels,kept,channel_degree,32,4,reference,{}))
                    if a.direct_backend=='point':
                        direct+=apply_point_spectral(band_bundle(lo,hi),frame,points[take],np.stack(point_angles),tau+reference,
                            positions,looks,alpha,scales[lo:hi])
                    else:
                        direct+=apply_multi_spectral(band_bundle(lo,hi),sources,positions,looks,alpha,scales[lo:hi],radial_window_m=(3.,300.),channel_dtype=np.complex128)
                    npoints+=len(take)
                print('direct band group',lo,hi,'node',float(fraction),'points',len(chosen),'CPU',process_time()-cpu,flush=True)
            report['direct_groups'].append(dict(bands=[lo,hi],quadrature_points=npoints,cpu_s=process_time()-cpu,wall_s=perf_counter()-wall));save_report()
    report['direct_complete']=True
    np.savez_compressed(a.output/'direct.npz',spectrum=direct,bins=bins_of(direct),time_origin_ns=origins);save_report()
    if a.direct_only:report['complete']=True;save_report();return
    direct_bins=bins_of(direct)
    variants=[] if a.skip_standard else [('standard05',.5,0.)]
    variants += [(f'carrier{cell:g}'.replace('.',''),cell,1/.299792458)
                 for cell in a.carrier_cells_m]
    for name,cell,slope in variants:
        total=np.zeros_like(direct);measurements=[]
        for lam in range(len(wl)):
            f=field_for(lam)
            if f is None:continue
            options=dict(degree=32,azimuthal_degree=4,cell_m=cell,axial_order=3,transverse='stencil',
                stencil_m=.2,stencil_threshold_m=.03,frame=frame,time_step_ns=.1,time_deposit='cubic',
                channel_dtype=np.complex64,stencil_center='centroid',geometry_reference=geometry,
                axial_clock_ns_per_m=slope)
            select_cpu,select_wall=process_time(),perf_counter()
            zero=compile_axial_slices(f,omega_per_ns=[0.],**options);points=zero.points_m();selection=np.zeros(len(points),bool)
            for receiver in positions:selection|=np.sum((points-receiver)**2,axis=1)<=(300.+1e-8)**2
            sc,sw=process_time()-select_cpu,perf_counter()-select_wall
            del zero;gc.collect();cpu,wall=process_time(),perf_counter()
            compiled=compile_axial_slices(f,omega_per_ns=omega,point_selection=selection,**options)
            cc,cw=process_time()-cpu,perf_counter()-wall;cpu,wall=process_time(),perf_counter()
            total+=apply_multi(band_bundle(lam,lam+1),compiled,compiled,positions,looks,alpha,np.ones(1),np.zeros(1),scales[lam:lam+1],radial_window_m=(3.,300.))
            ac,aw=process_time()-cpu,perf_counter()-wall
            measurements.append(dict(band=lam,selection_cpu_s=sc,selection_wall_s=sw,
                compile_cpu_s=cc,compile_wall_s=cw,apply_cpu_s=ac,apply_wall_s=aw,summary=compiled.summary))
            del compiled;gc.collect();print(name,'band',lam,'CPU',cc+ac,flush=True)
        bins=bins_of(total);charge=direct[0].real
        row=dict(cell_m=cell,slope=slope,measurements=measurements,charge_pe=total[0].real.tolist(),
            charge_relative_error=(abs(total[0]-direct[0])/np.maximum(abs(direct[0]),1e-30)).tolist(),
            charge_absolute_error_pe=abs(total[0]-direct[0]).tolist(),
            bins_relative_l1=(abs(bins-direct_bins).sum(1)/np.maximum(abs(direct_bins).sum(1),1e-30)).tolist(),
            bins_absolute_l1_pe=abs(bins-direct_bins).sum(1).tolist())
        report['variants'][name]=row
        np.savez_compressed(a.output/f'{name}.npz',spectrum=total,bins=bins,time_origin_ns=origins)
        save_report();print(name,'max waveform difference',max(row['bins_relative_l1']),flush=True)
    report['direct_charge_pe']=direct[0].real.tolist();report['direct_bins_l1_pe']=abs(direct_bins).sum(1).tolist()
    report['complete']=True;save_report()


if __name__=='__main__':main()
