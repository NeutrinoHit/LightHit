#!/usr/bin/env python3
"""Independent original-G4 primary control for a frozen universal-atlas result.

No cache is built, no source/OM selection is changed. Default picks the brightest
predicted OM solely as a validation target. q1 uses ORIGINAL steps and 82-node
composite optical quadrature; >=2 uses the established spectral source compiler.
The latter retains its m=0, finite-radius and RTE-kernel approximations.
"""
import paths  # inputs and outputs: scripts/shower_basis/paths.py
import argparse,json,hashlib,resource
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
from copy import copy
from time import process_time,perf_counter
import numpy as np


def comparison(q,b,qr,br):
    return dict(charge_pe=float(q),reference_charge_pe=float(qr),
                charge_relative=float(abs(q-qr)/max(abs(qr),1e-30)),
                charge_absolute_pe=float(abs(q-qr)),
                bins_relative_l1=float(abs(b-br).sum()/max(abs(br).sum(),1e-30)),
                bins_absolute_l1_pe=float(abs(b-br).sum()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=paths.holdout_primary())
    p.add_argument('--comparison',type=Path,default=paths.work('primary-universal-20261009/results/holdout43'))
    p.add_argument('--output',type=Path,default=paths.work('primary-universal-20261009/original-controls/holdout43'))
    p.add_argument('--om-index',type=int)
    p.add_argument('--cell-m',type=float,default=.25)
    p.add_argument('--component',choices=['all','q1','ge2'],default='all')
    p.add_argument('--cpu-limit',type=int,default=120)
    a=p.parse_args()
    if (a.output/'report.json').exists():p.error('choose fresh output directory')
    resource.setrlimit(resource.RLIMIT_CPU,(a.cpu_limit,a.cpu_limit+2))
    import lighthit as lh,ntsim,numba
    from ntsim.LightTransport import bgvd_model_from_ntsim
    from lighthit.experimental.universal_primary import subset_kernel
    from lighthit.experimental.primary_polyline import fit_primary_polyline,primary_spectral_quadrature
    from lighthit.experimental.muon_integral import order1_integrals
    from lighthit.prompt import PromptConfig
    if numba.get_num_threads()!=1:p.error('single Numba thread required')
    with np.load(a.source,allow_pickle=False) as z:
        source=lh.SpectralLightElements(**{k:z[k] for k in z.files if k!='metadata'},cone_model='spectral')
        meta=json.loads(str(z['metadata']))
    source=source.subset(source.uid==0)
    with np.load(a.comparison/'response.npz',allow_pickle=False) as z:
        allq=z['charge_orders_pe'].copy();allb=z['bins_orders_pe'].copy();origins=z['time_origin_ns'].copy()
        edges=z['relative_time_edges_ns'].copy();indices=z['receiver_indices'].copy()
    local_index=int(np.argmax(allq.sum(axis=1))) if a.om_index is None else int(np.flatnonzero(indices==a.om_index)[0])
    om=int(indices[local_index]);origin=origins[[local_index]]
    control_path=a.comparison/f'control-{om}.npz'
    with np.load(control_path,allow_pickle=False) as z:
        fitted_reference_q=z['reference_charge'].copy();fitted_reference_b=z['reference_bins'].copy()
    model=bgvd_model_from_ntsim(ntsim.bgvd.Telescope(n_clusters=(1,2,3,4,5)),ntsim.bgvd.Water())
    if model.fingerprint!=meta['model_fingerprint']:raise ValueError('source/model mismatch')
    omega=np.linspace(0,1.2,65);outomega=np.linspace(0,1.2,321)
    kernel=model.kernel(lh.KernelConfig(omega_per_ns=omega,readout_omega_per_ns=outomega,
        relative_time_edges_ns=edges,threshold_pe=0.,cache_policy='require',
        cache_directory=str(paths.RTE_CACHE.path())))
    kernel=subset_kernel(kernel,[om])
    fit=fit_primary_polyline(source,phase_indices=kernel.medium.phase_index)
    report=dict(source_file=str(a.source),source_sha256=hashlib.sha256(a.source.read_bytes()).hexdigest(),
        comparison_file=str(a.comparison/'response.npz'),comparison_sha256=hashlib.sha256((a.comparison/'response.npz').read_bytes()).hexdigest(),
        om_index=om,selection='brightest prediction used only as independent validation target',
        original_steps=len(source),fitted_pieces=len(fit.elements),time_origin_ns=float(origin[0]),
        cell_m=a.cell_m,components={},cache_build=False)
    a.output.mkdir(parents=True,exist_ok=True)
    def save():
        (a.output/'report.json').write_text(json.dumps(report,indent=2,default=str))
        print(json.dumps(report,indent=2,default=str),flush=True)
    if a.component in ('all','q1'):
        local=copy(kernel);local.wavelength=primary_spectral_quadrature(kernel.medium,kernel.detector,order=2)
        cfg=PromptConfig(element_method='segment',screen=False)
        print('Q1 original BEGIN',om,len(source),len(local.wavelength.wavelength_nm),flush=True)
        order1_integrals(local,source.subset(np.arange(len(source))==0),time_origin_ns=origin,config=cfg)
        wall,cpu=perf_counter(),process_time()
        qr,br,diag=order1_integrals(local,source,time_origin_ns=origin,config=cfg)
        wall,cpu=perf_counter()-wall,process_time()-cpu
        qf,bf,_=order1_integrals(local,fit.elements,time_origin_ns=origin,config=cfg)
        np.testing.assert_allclose(qf[0],fitted_reference_q[1],rtol=1e-12,atol=1e-14)
        np.testing.assert_allclose(bf[0],fitted_reference_b[:,1],rtol=1e-12,atol=1e-14)
        report['components']['q1']=dict(cpu_seconds=cpu,wall_seconds=wall,spectral_nodes=len(local.wavelength.wavelength_nm),
            fitted_vs_original=comparison(qf[0],bf[0],qr[0],br[0]),
            atlas_vs_original=comparison(allq[local_index,1],allb[local_index,:,1],qr[0],br[0]),
            atlas_vs_fit=comparison(allq[local_index,1],allb[local_index,:,1],qf[0],bf[0]),metadata=diag)
        np.savez_compressed(a.output/'q1.npz',original_charge=qr,original_bins=br,fitted_charge=qf,fitted_bins=bf,
                            candidate_charge=allq[local_index,1],candidate_bins=allb[local_index,:,1],origin_ns=origin,edges=edges)
        save()
    if a.component in ('all','ge2'):
        from lighthit.shower_basis import transport_hybrid_basis
        from lighthit.readout import inverse_bins
        from scipy.interpolate import CubicSpline
        kernel.ge2_kernel_store(paths.GE2_PACK.path())
        def zero_prompt(*unused,**kwargs):
            baseline=np.asarray(kwargs['time_origin_ns'])
            return SimpleNamespace(relative_time_edges_ns=edges,time_origin_ns=baseline,
                charge_orders_pe=np.zeros((1,2)),charge_pe=np.zeros(1),computed=np.ones(1,bool),
                bins_orders_pe=np.zeros((1,len(edges)-1,2)),
                metadata={'prompt_algorithm':'disabled for isolated original-source >=2 control','order1_level':[0]})
        kernel.transport_prompt=zero_prompt
        basis=lh.ShowerBasis(track_uid=0,spectral_fusion=True,cell_m=a.cell_m,track_cell_m=a.cell_m,
            track_transverse='stencil',radial_high_m=300.,radial_tail_tolerance_pe=1e-4,apply_tolerance_pe=0.)
        print('GE2 original BEGIN',om,len(source),flush=True)
        wall,cpu=perf_counter(),process_time()
        result=transport_hybrid_basis(kernel,source,basis,frequency_nodes=65,track_stencil_center='centroid')
        wall,cpu=perf_counter()-wall,process_time()-cpu
        spectrum=result.spectrum_ge2_pe[:,0]*np.exp(-1j*omega*origin[0])
        br=inverse_bins(outomega,CubicSpline(omega,spectrum)(outomega)[:,None],edges)[0];qr=float(spectrum[0].real)
        report['components']['ge2']=dict(cpu_seconds=cpu,wall_seconds=wall,spectral_nodes=9,source_azimuthal_degree=0,
            fitted_vs_original=comparison(fitted_reference_q[2],fitted_reference_b[:,2],qr,br),
            atlas_vs_original=comparison(allq[local_index,2],allb[local_index,:,2],qr,br),
            atlas_vs_fit=comparison(allq[local_index,2],allb[local_index,:,2],fitted_reference_q[2],fitted_reference_b[:,2]),
            metadata=result.metadata,limitations=['same L32 RTE kernel, 9 wavelengths and 300m radius',
                'existing m0 spectral source compiler is an approximation to original G4 steps',
                'no separate refinement of compiler cell size in this bounded control'])
        np.savez_compressed(a.output/'ge2.npz',original_charge=qr,original_bins=br,original_spectrum=spectrum,
                            omega=omega,origin_ns=origin,edges=edges,candidate_charge=allq[local_index,2],
                            candidate_bins=allb[local_index,:,2],fitted_charge=fitted_reference_q[2],fitted_bins=fitted_reference_b[:,2])
        save()


    if {'q1','ge2'} <= set(report['components']):
        with np.load(a.output/'q1.npz') as one,np.load(a.output/'ge2.npz') as multi:
            original_q=allq[local_index,0]+float(one['original_charge'][0])+float(multi['original_charge'])
            original_b=allb[local_index,:,0]+one['original_bins'][0]+multi['original_bins']
        report['total_with_shared_original_q0']=comparison(allq[local_index].sum(),allb[local_index].sum(1),original_q,original_b)
        report['total_q0_scope']='same original-step interval ballistic response reused, not an independent q0 rerun'
        save()

if __name__=='__main__':main()
