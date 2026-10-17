#!/usr/bin/env python3
"""Small real-G4 joint-photon Q1 experiment, physical time and one core.

Measures source binning separately from CDF transport error. The source is a
bounded compact muon-secondary packet. This is not a full-event timing. Geometry
responses are built from relative geometry and optics, never saved answers.
"""
import paths  # inputs and outputs: scripts/shower_basis/paths.py
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from time import process_time
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--energy',type=int,choices=[300,1000,3000],default=1000)
    p.add_argument('--elements',type=int,default=128)
    p.add_argument('--om',type=int,default=197)
    p.add_argument('--ring-nodes',type=int,default=32)
    p.add_argument('--z-step',type=float,default=.1)
    p.add_argument('--mu-step',type=float,default=.03)
    p.add_argument('--phi-step',type=float,default=.06)
    p.add_argument('--transverse-step',type=float,default=.03)
    args=p.parse_args()
    if args.output.exists():p.error('fresh output directory required')
    import numba,ntsim,lighthit as lh
    from ntsim.LightTransport import bgvd_model_from_ntsim
    from lighthit.experimental.joint_source import compile_photon_histogram
    from lighthit.experimental.joint_first import PencilFirstCDF,contract_joint_pencils
    from lighthit.prompt import _spectral_tables,acceptance_polynomial,acceptance_monomials
    from lighthit._prompt_numba import pencils_order1,interval_table
    if numba.get_num_threads()!=1:p.error('NUMBA_NUM_THREADS=1 required')
    path=paths.muon_source(args.energy)
    with np.load(path) as z:
        source=lh.SpectralLightElements(**{k:z[k] for k in z.files if k!='metadata'},cone_model='spectral')
    with np.load(paths.muon_ancestry(args.energy)) as z:
        family=z['element_family']
    model=bgvd_model_from_ntsim(ntsim.bgvd.Telescope(n_clusters=(1,2,3,4,5)),ntsim.bgvd.Water())
    kernel=model.kernel(lh.KernelConfig(omega_per_ns=np.array([0.]),cache_policy='require'))
    tables=_spectral_tables(kernel);lam=int(np.argmin(abs(tables['wavelength_nm']-480.)))
    phase=float(tables['phase_index'][lam]);wave=float(tables['wavelength_nm'][lam])
    receiver=kernel.detector.positions_m[args.om];look=-kernel.detector.orientations[args.om]
    area=float(kernel.detector.effective_area_m2[args.om])
    mass=np.maximum(source.coefficient0+source.coefficient2/phase**2,0.)
    valid=(source.uid!=0)&(source.beta*phase>1.)
    distance=np.linalg.norm(source.midpoints_m-receiver,axis=1)
    score=mass*np.exp(-distance/50.)
    roots=np.unique(family[valid]);totals=np.array([score[valid&(family==root)].sum() for root in roots])
    root=roots[totals.argmax()];rows=np.flatnonzero(valid&(family==root))
    rows=rows[np.argsort(mass[rows])[-args.elements:]]
    mask=np.zeros(len(source),bool);mask[rows]=True
    packet=source.subset(mask)
    cpu=process_time()
    hist=compile_photon_histogram(packet,wave,phase,z_step_m=args.z_step,
        mu_step=args.mu_step,phi_step_rad=args.phi_step,tau_step_ns=.1,
        transverse_step_m=args.transverse_step,ring_nodes=args.ring_nodes,
        chord_order=2,retain_samples=True)
    source_cpu=process_time()-cpu
    apoly=acceptance_polynomial(*acceptance_monomials(kernel.acceptance()['alpha']))
    mus=float(tables['mus'][lam]);mut=float(tables['mut'][lam]);speed=float(tables['speed'][lam])
    g=float(kernel.medium.g)
    origin=float(np.min(hist.original_times_ns+np.linalg.norm(hist.original_position-receiver,axis=1)/speed))
    edges=np.arange(-5.,806.,5.)
    kappa=np.array([-mut*speed]);eedge=np.exp(np.outer(kappa,edges));itab=interval_table(edges,kappa)
    def direct(points,directions,times,weights,step=.025):
        return pencils_order1(np.ascontiguousarray(receiver[None,:]),np.ascontiguousarray(look[None,:]),
            np.array([area]),np.array([origin]),np.ascontiguousarray(points),np.ascontiguousarray(directions),
            np.ascontiguousarray(weights),np.zeros(len(times)),np.ascontiguousarray(times),apoly,
            g,(1-abs(g))/(1+abs(g)),step,step,25.,np.array([mus]),np.array([mut]),np.array([speed]),
            np.ones(1),np.zeros(1),edges,eedge,itab,4,4096)
    # Warm independent pencil kernel and the temporal CDF evaluator.
    direct(hist.original_position[:1],hist.original_direction[:1],hist.original_times_ns[:1],hist.original_photons_per_nm[:1])
    sample=PencilFirstCDF.build(receiver-hist.position[0],hist.direction[0],look,apoly,
        scattering_per_m=mus,absorption_per_m=mut-mus,g=g,speed_m_per_ns=speed)
    sample.contract([origin],[1.],edges,time_origin_ns=origin, effective_area_m2=area)
    cpu=process_time();qr,br=direct(hist.original_position,hist.original_direction,hist.original_times_ns,hist.original_photons_per_nm)
    raw_cpu=process_time()-cpu
    cpu=process_time();qf,bf=direct(hist.original_position,hist.original_direction,hist.original_times_ns,hist.original_photons_per_nm,step=.0125)
    refined_cpu=process_time()-cpu
    pos=hist.position[hist.geometry_id];dirs=hist.direction[hist.geometry_id];weights=hist.photon_weights[0]
    cpu=process_time();qg,bg=direct(pos,dirs,hist.emission_times_ns,weights)
    grouped_reference_cpu=process_time()-cpu
    cpu=process_time();responses=[PencilFirstCDF.build(receiver-position,direction,look,apoly,
        scattering_per_m=mus,absorption_per_m=mut-mus,g=g,speed_m_per_ns=speed)
        for position,direction in zip(hist.position,hist.direction)]
    prepare_cpu=process_time()-cpu
    cpu=process_time();q,b=contract_joint_pencils(responses,hist.geometry_id,hist.emission_times_ns,weights,
        edges,time_origin_ns=origin,effective_area_m2=area)
    apply_cpu=process_time()-cpu
    def errors(q,b,refq,refb):
        return dict(charge_relative=float(abs(q-refq)/max(abs(refq),1e-300)),
            bins_l1=float(abs(b-refb).sum()/max(abs(refb).sum(),1e-300)),
            charge=float(q),negative_bin_mass=float(np.maximum(-b,0.).sum()))
    report=dict(energy_GeV=args.energy,om=args.om,family=int(root),source_elements=len(packet),
        original_photon_points=len(hist.original_times_ns),geometry_bins=len(responses),clock_bins=len(weights),
        wavelength_nm=wave,source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        model_fingerprint=model.fingerprint,threads=1,time_origin_ns=origin,
        clocks_span_ns=float(np.ptp(hist.original_times_ns)),source_compile_cpu_s=source_cpu,
        raw_reference_cpu_s=raw_cpu,refined_reference_cpu_s=refined_cpu,
        grouped_reference_cpu_s=grouped_reference_cpu,cdf_prepare_cpu_s=prepare_cpu,cdf_apply_cpu_s=apply_cpu,
        cdf_total_cpu_s=prepare_cpu+apply_cpu,
        raw_refinement=errors(qr[0],br[0],qf[0],bf[0]),
        source_binning=errors(qg[0],bg[0],qf[0],bf[0]),
        cdf_operator=errors(q,b,qg[0],bg[0]),
        total=errors(q,b,qf[0],bf[0]),summary=hist.summary,
        timing_scope='small real secondary packet, one wavelength, one real OM; cache build charged',
        limitations='finite ring/chord quadrature and centroid bin source; no claim of finite-bin integrated response')
    report['charge_unit']='accepted photons/nm before QE and wavelength quadrature; ratios unaffected'
    args.output.mkdir(parents=True)
    (args.output/'report.json').write_text(json.dumps(report,indent=2))
    np.savez_compressed(args.output/'responses.npz',reference_charge=qf,reference_bins=bf,
        original_charge=qr,original_bins=br,grouped_charge=qg,grouped_bins=bg,
        cdf_charge=q,cdf_bins=b,edges=edges,origin=origin)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
