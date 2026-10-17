#!/usr/bin/env python3
"""Isolate >=2 original-G4 primary against fitted-track primitive experiments.

The existing spectral source compiler is the independent source representation.
Prompt transport is explicitly replaced by zeros: no order0/1 work is done.
"""
import paths  # inputs and outputs: scripts/shower_basis/paths.py
import argparse,json
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
from time import perf_counter,process_time
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--energy',type=int,choices=[300,1000,3000],default=1000)
    p.add_argument('--om-index',type=int,default=198)
    p.add_argument('--cell-m',type=float,default=.5)
    p.add_argument('--full-azimuth',action='store_true')
    p.add_argument('--comparison',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():p.error('output must be unused')
    import numba,ntsim,lighthit as lh
    from ntsim.LightTransport import bgvd_model_from_ntsim
    from lighthit.shower_basis import transport_hybrid_basis
    from lighthit.readout import inverse_bins
    from scipy.interpolate import CubicSpline
    from probe_positive_map_screen import read_source
    if numba.get_num_threads()!=1:p.error('requires one Numba thread')
    source,provenance,_=read_source(args.energy);source=source.subset(source.uid==0)
    model=bgvd_model_from_ntsim(ntsim.bgvd.Telescope(n_clusters=(1,2,3,4,5)),ntsim.bgvd.Water())
    if provenance['model_fingerprint']!=model.fingerprint:raise ValueError('source/model mismatch')
    d=model.detector;ids=np.array([args.om_index]);model=replace(model,detector=replace(d,
        positions_m=d.positions_m[ids],orientations=d.orientations[ids],effective_area_m2=d.effective_area_m2[ids],
        identifiers={k:v[ids] for k,v in d.identifiers.items()}))
    omega=np.linspace(0,1.2,65);output_omega=np.linspace(0,1.2,321);edges=np.arange(-60,745,5.)
    config=lh.KernelConfig(omega_per_ns=omega,readout_omega_per_ns=output_omega,
        relative_time_edges_ns=edges,threshold_pe=0.,
        cache_directory=paths.RTE_CACHE.path(),cache_policy='require')
    kernel=model.kernel(config);kernel.ge2_kernel_store(paths.GE2_PACK.path())
    def zero_prompt(*unused,**kwargs):
        origins=np.asarray(kwargs['time_origin_ns'])
        return SimpleNamespace(relative_time_edges_ns=edges,time_origin_ns=origins,
            charge_orders_pe=np.zeros((1,2)),charge_pe=np.zeros(1),computed=np.ones(1,bool),
            bins_orders_pe=np.zeros((1,len(edges)-1,2)),
            metadata={'prompt_algorithm':'disabled for isolated >=2 control','order1_level':[0]})
    kernel.transport_prompt=zero_prompt
    basis=lh.ShowerBasis(track_uid=None if args.full_azimuth else 0,spectral_fusion=True,
        cell_m=args.cell_m,track_cell_m=args.cell_m,track_transverse='stencil',radial_high_m=300.,
        radial_tail_tolerance_pe=1e-4,apply_tolerance_pe=0.)
    wall=perf_counter();cpu=process_time()
    result=transport_hybrid_basis(kernel,source,basis,frequency_nodes=65,
        track_stencil_center='axis' if args.full_azimuth else 'centroid')
    cpu=process_time()-cpu;wall=perf_counter()-wall
    with np.load(paths.muon_response(args.energy)) as z:
        origin=float(z['time_origin_ns'][args.om_index])
    q=result.spectrum_ge2_pe[:,0]*np.exp(-1j*omega*origin)
    qb=inverse_bins(output_omega,CubicSpline(omega,q)(output_omega)[:,None],edges)[0]
    with np.load(args.comparison/'response.npz') as z:
        direct=z['reference'];candidate=z['candidate'];db=z['bins_reference'];cb=z['bins_candidate']
    def compare(x,xb):
        return dict(charge_relative_error=float(abs(x[0]-q[0])/abs(q[0])),
                    bins_relative_l1=float(abs(xb-qb).sum()/abs(qb).sum()),
                    spectrum_relative_l1=float(abs(x-q).sum()/abs(q).sum()))
    report=dict(energy_GeV=args.energy,om_index=args.om_index,source_steps=len(source),cell_m=args.cell_m,
        source_azimuthal_degree=4 if args.full_azimuth else 0,
        cpu_seconds=cpu,wall_seconds=wall,charge_original_ge2_pe=float(q[0].real),
        fitted_polyline_direct_vs_original=compare(direct,db),cached_vs_original=compare(candidate,cb),
        metadata=result.metadata,
        limitations=['original source is represented by existing cubic spectral compiler',
                     'm0 source approximation retained unless --full-azimuth',
                     'same 9 wavelength and L32 RTE kernel, no independent transport physics validation'])
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output/'response.npz',omega=omega,spectrum=q,bins=qb,origin_ns=origin)
    (args.output/'report.json').write_text(json.dumps(report,indent=2,default=str))
    print(json.dumps({k:v for k,v in report.items() if k!='metadata'},indent=2))

if __name__=='__main__':main()
