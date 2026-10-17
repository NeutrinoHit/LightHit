#!/usr/bin/env python3
"""Check learned optical cubature on saved five-cluster primary controls."""
import paths  # inputs and outputs: scripts/shower_basis/paths.py
import argparse,json,hashlib
from pathlib import Path
from dataclasses import replace
from copy import copy
from time import perf_counter,process_time
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--basis',type=Path,required=True)
    p.add_argument('--energies',type=int,nargs='+',default=[300,1000,3000])
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():p.error('choose fresh output')
    import lighthit as lh,ntsim,numba
    from ntsim.LightTransport import bgvd_model_from_ntsim
    from lighthit.experimental.primary_polyline import fit_primary_polyline,primary_spectral_quadrature
    from lighthit.experimental.universal_primary import subset_kernel
    from lighthit.experimental.spectral_response_basis import ResponseCubature
    from lighthit.experimental.first_charge_only import first_scattering_charge_only
    from lighthit.experimental.muon_integral import order1_integrals
    from lighthit.prompt import PromptConfig
    if numba.get_num_threads()!=1:p.error('one thread required')
    model=bgvd_model_from_ntsim(ntsim.bgvd.Telescope(n_clusters=(1,2,3,4,5)),ntsim.bgvd.Water())
    kernel=model.kernel(lh.KernelConfig(omega_per_ns=np.array([0.]),threshold_pe=.01,cache_policy='require'))
    basis=ResponseCubature.load(args.basis);basis.validate_optics(kernel)
    prompt=PromptConfig(element_method='segment',screen=False)
    args.output.mkdir(parents=True);reports=[]
    for energy in args.energies:
        base=paths.work(f'primary-universal-20261009/results/seed42-{energy}')
        saved=json.loads((base/'report.json').read_text());ids=np.array([row['om'] for row in saved['controls']])
        source_path=paths.muon_source(energy)
        if hashlib.sha256(source_path.read_bytes()).hexdigest()!=saved['source_sha256']:
            raise ValueError('saved control source differs')
        if saved['source_metadata']['model_fingerprint']!=model.fingerprint:
            raise ValueError('saved control optical/detector model differs')
        with np.load(source_path) as z:
            source=lh.SpectralLightElements(**{k:z[k] for k in z.files if k!='metadata'},cone_model='spectral')
        source=source.subset(source.uid==0)
        pieces=fit_primary_polyline(source,phase_indices=kernel.medium.phase_index).elements
        good=pieces.beta>=basis.metadata['beta_min']
        fast=pieces.subset(good);tail=pieces.subset(~good) if not good.all() else None
        local=subset_kernel(kernel,ids);reference=copy(local)
        reference.wavelength=primary_spectral_quadrature(kernel.medium,kernel.detector,order=2)
        local.wavelength=basis.quadrature
        with np.load(base/'response.npz') as z:origins=z['time_origin_ns'][ids]
        qref=[];bref=[];ref_all=[];b_all=[];active=[]
        for row in saved['controls']:
            with np.load(base/f"control-{row['om']}.npz") as z:
                qref.append(z['reference_charge'][1]);bref.append(z['reference_bins'][:,1]);ref_all.append(z['reference_charge']);b_all.append(z['reference_bins'])
            active.append(row['direct_active'])
        qref=np.array(qref);bref=np.array(bref);active=np.array(active)
        # Warm on one element before the measurement.
        first_scattering_charge_only(subset_kernel(local,[0]),fast.subset(np.arange(len(fast))==0))
        cpu=process_time();wall=perf_counter();q=first_scattering_charge_only(local,fast)[0]
        if tail is not None:q+=first_scattering_charge_only(reference,tail)[0]
        candidate_cpu=process_time()-cpu;candidate_wall=perf_counter()-wall
        waveform=np.zeros_like(bref);selected=np.flatnonzero(active)
        cpu=process_time();wall=perf_counter()
        if len(selected):
            _,b,_=order1_integrals(subset_kernel(local,selected),fast,time_origin_ns=origins[selected],config=prompt)
            waveform[selected]=b
            if tail is not None:
                _,b,_=order1_integrals(subset_kernel(reference,selected),tail,time_origin_ns=origins[selected],config=prompt)
                waveform[selected]+=b
        rows=[]
        for index,row in enumerate(saved['controls']):
            others=np.array(ref_all[index])[[0,2]].sum()
            total_ref=others+qref[index];total_candidate=others+q[index]
            item=dict(om=int(ids[index]),reference_q1_pe=float(qref[index]),candidate_q1_pe=float(q[index]),
                q1_relative_error=float(abs(q[index]-qref[index])/max(abs(qref[index]),1e-30)),
                total_charge_relative_error=float(abs(q[index]-qref[index])/max(abs(total_ref),1e-30)),
                total_charge_absolute_error_pe=float(abs(q[index]-qref[index])),
                direct_active=bool(active[index]),candidate_active=bool(total_candidate>=.01),
                q1_bins_l1=float(abs(waveform[index]-bref[index]).sum()/max(abs(bref[index]).sum(),1e-30)) if active[index] else None)
            if active[index]:
                full_ref=np.array(b_all[index]).sum(1)
                item['total_bins_l1']=float(abs(waveform[index]-bref[index]).sum()/max(abs(full_ref).sum(),1e-30))
            rows.append(item)
        report=dict(energy_GeV=energy,primary_only=True,basis_file=str(args.basis),spectral_nodes=len(basis.quadrature.weight_nm),
            reference_nodes=len(reference.wavelength.weight_nm),controls=len(ids),fast_source_pieces=len(fast),direct_tail_pieces=0 if tail is None else len(tail),
            charge_cpu_s=candidate_cpu,charge_wall_s=candidate_wall,time_cpu_s=process_time()-cpu,time_wall_s=perf_counter()-wall,
            max_active_q1_charge_relative=max(r['q1_relative_error'] for r in rows if r['direct_active']),
            max_active_total_charge_relative=max(r['total_charge_relative_error'] for r in rows if r['direct_active']),
            max_active_q1_bins_l1=max(r['q1_bins_l1'] for r in rows if r['direct_active']),
            max_active_total_bins_l1=max(r['total_bins_l1'] for r in rows if r['direct_active']),
            activity_disagreements=[r['om'] for r in rows if r['direct_active']!=r['candidate_active']],rows=rows)
        reports.append(report)
        np.savez_compressed(args.output/f'response-{energy}.npz',ids=ids,charge=q,bins=waveform,reference_charge=qref,reference_bins=bref,origins=origins)
        (args.output/'report.json').write_text(json.dumps(reports,indent=2))
        print('EVENT',json.dumps({k:v for k,v in report.items() if k!='rows'}),flush=True)


if __name__=='__main__':main()
