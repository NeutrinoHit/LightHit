#!/usr/bin/env python3
"""Optics-only lambda/beta response POD and positive cubature, with holdouts.

G4 events are never training inputs. Charge is the first target because it is
the measured bottleneck; waveform validation remains independent.
"""
import argparse,json
from pathlib import Path
from dataclasses import replace
from copy import copy
from time import perf_counter,process_time
import numpy as np

C=.299792458
TRAIN_LOOKS=np.array([[-.68,np.sqrt(1-.68**2),0.],[.23,.61,np.sqrt(1-.23**2-.61**2)],[0.,0.,1.]])
TEST_LOOKS=np.array([[.8,0.,.6],[-.3,.4,np.sqrt(.75)],[0.,0.,-1.]])


def scenario_kernel(kernel,rho,looks):
    local=copy(kernel);d=kernel.detector;n=len(looks)
    local.detector=replace(d,positions_m=np.tile([rho,0.,0.],(n,1)),orientations=looks,
        effective_area_m2=np.repeat(d.effective_area_m2[:1],n),
        identifiers={key:np.repeat(val[:1],n) for key,val in d.identifiers.items()})
    return local


def source_for(kernel,rho,beta,cut):
    from lighthit import SpectralLightElements
    phase=float(kernel.medium.sample([480.])['phase_index'][0])
    root=1/np.sqrt((beta*phase)**2-1)
    intervals={'wide':(-1.,4.),'root':(root-.4,root+.5),
        'edge':(root-1.,root+.03),'back':(-6.,-2.)}
    a,b=np.array(intervals[cut])*rho;length=b-a
    return SpectralLightElements(np.array([[0.,0.,-b]]),np.array([[0.,0.,1.]]),
        np.array([length]),np.array([length]),np.array([-length/beta**2]),np.array([beta]),
        np.array([-b/(beta*C)]),np.array([-a/(beta*C)]),np.array([0]),np.array([0]),cone_model='spectral')


def snapshots(kernel,reference,states,looks,directory):
    from lighthit.experimental.first_charge_only import first_scattering_charge_only
    from lighthit.experimental.track_first_primitive import track_first_optical_contract
    from lighthit.model import WavelengthQuadrature
    values=[];directory.mkdir(parents=True,exist_ok=True)
    _,fingerprint=track_first_optical_contract(kernel)
    signature=dict(optics_fingerprint=fingerprint,wavelength_nm=reference.wavelength_nm.tolist(),
        weight_nm=reference.weight_nm.tolist(),looks=looks.tolist())
    contract=directory/'contract.json'
    if contract.exists():
        if json.loads(contract.read_text())!=signature:raise ValueError('snapshot optical/quadrature/orientation contract differs')
    else:
        if any(directory.glob('*.npz')):raise ValueError('old snapshots have no input contract; choose a fresh output')
        contract.write_text(json.dumps(signature,indent=2))
    for index,state in enumerate(states):
        path=directory/f'{index:03d}.npz'
        if path.exists():
            with np.load(path) as data:
                if json.loads(str(data['state']))!=state:raise ValueError('snapshot state differs')
                values.append(data['response'].copy())
            continue
        rho,beta,cut=state['rho_m'],state['beta'],state['cut']
        local=scenario_kernel(kernel,rho,looks);source=source_for(kernel,rho,beta,cut)
        response=np.zeros((len(looks),len(reference.wavelength_nm)))
        wall,cpu=perf_counter(),process_time()
        for j,wavelength in enumerate(reference.wavelength_nm):
            local.wavelength=WavelengthQuadrature(np.array([wavelength]),np.array([1.]))
            response[:,j]=first_scattering_charge_only(local,source)[0]
        np.savez_compressed(path,response=response,state=json.dumps(state))
        values.append(response)
        print('SNAPSHOT',json.dumps(dict(index=index,total=len(states),**state,cpu_s=process_time()-cpu,wall_s=perf_counter()-wall)),flush=True)
    return np.array(values)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    import numba
    from build_track_first_atlas import optical_kernel
    from lighthit.experimental.primary_polyline import primary_spectral_quadrature
    from lighthit.experimental.track_first_primitive import track_first_optical_contract
    from lighthit.experimental.spectral_response_basis import fit_response_cubature,evaluate_cubature
    from lighthit.experimental.first_charge_only import first_scattering_charge_only
    from lighthit.experimental.muon_integral import order1_integrals
    from lighthit.prompt import PromptConfig
    if numba.get_num_threads()!=1:parser.error('one Numba thread required')
    kernel=optical_kernel();reference=primary_spectral_quadrature(kernel.medium,kernel.detector,order=2)
    _,fingerprint=track_first_optical_contract(kernel)
    kernel.wavelength=reference
    first_scattering_charge_only(scenario_kernel(kernel,10.,TRAIN_LOOKS[:1]),source_for(kernel,10.,1.,'root'))
    train=[dict(rho_m=r,beta=b,cut=c) for r in [5.,40.,150.] for b in [1.,.999,.97,.95] for c in ['wide','root','edge','back']]
    held=[dict(rho_m=r,beta=b,cut=c) for r in [10.,80.,250.] for b in [.9999,.995,.96] for c in ['wide','root','edge','back']]
    raw=snapshots(kernel,reference,train,TRAIN_LOOKS,args.output/'train')
    check=snapshots(kernel,reference,held,TEST_LOOKS,args.output/'holdout')
    flat=raw.reshape(-1,len(reference.weight_nm));totals=flat@reference.weight_nm
    if np.any(totals<=0):raise ValueError('nonpositive training response')
    features=flat/totals[:,None]
    models=[];summaries=[];expected=check@reference.weight_nm
    for rank in [4,6,8,12,16,24]:
        model=fit_response_cubature(reference,features,rank,metadata=dict(optics_fingerprint=fingerprint,beta_min=.95,
            training_source='synthetic unit uniform segments only; no G4 inputs',target='stationary first-scattering charge',
            training_radii_m=[5.,40.,150.],validated_radii_m=[10.,80.,250.]))
        model.validate_optics(kernel);models.append(model)
        predicted=evaluate_cubature(model,check)
        error=abs(predicted-expected)/np.maximum(abs(expected),1e-300)
        summary=dict(rank=rank,nodes=len(model.selected_indices),max_relative_charge_error=float(error.max()),
            p95_relative_charge_error=float(np.quantile(error,.95)),max_absolute_charge_error=float(abs(predicted-expected).max()),
            worst_state=held[np.unravel_index(error.argmax(),error.shape)[0]],metadata=model.metadata)
        summaries.append(summary)
        np.savez_compressed(args.output/f'basis-rank{rank}.npz',wavelength_nm=model.quadrature.wavelength_nm,
            weight_nm=model.quadrature.weight_nm,selected_indices=model.selected_indices,modes=model.modes,
            singular_values=model.singular_values,metadata=json.dumps(model.metadata),
            reference_wavelength_nm=reference.wavelength_nm,reference_weight_nm=reference.weight_nm)
        print('BASIS',json.dumps(summary),flush=True)
    passing=[i for i,s in enumerate(summaries) if s['max_relative_charge_error']<=.002]
    selected=passing[0] if passing else len(models)-1
    candidate=models[selected]
    wave=[]
    for rho,beta,cut in [(10.,.9999,'wide'),(80.,.995,'edge'),(250.,.96,'root'),(30.,.95,'back')]:
        local=scenario_kernel(kernel,rho,TEST_LOOKS);source=source_for(kernel,rho,beta,cut)
        local.config=replace(local.config,relative_time_edges_ns=np.arange(-5.,2006.,5.)+.37)
        local.wavelength=reference;cpu=process_time();wall=perf_counter()
        qr,br,_=order1_integrals(local,source,time_origin_ns=np.zeros(3),config=PromptConfig(element_method='segment',screen=False))
        rcpu,rwall=process_time()-cpu,perf_counter()-wall
        local.wavelength=candidate.quadrature;cpu=process_time();wall=perf_counter()
        q,b,_=order1_integrals(local,source,time_origin_ns=np.zeros(3),config=PromptConfig(element_method='segment',screen=False))
        item=dict(rho_m=rho,beta=beta,cut=cut,charge_relative=(abs(q-qr)/np.maximum(abs(qr),1e-300)).tolist(),
            bins_relative_l1=(abs(b-br).sum(1)/np.maximum(abs(br).sum(1),1e-300)).tolist(),
            reference_cpu_s=rcpu,candidate_cpu_s=process_time()-cpu,reference_wall_s=rwall,candidate_wall_s=perf_counter()-wall)
        wave.append(item);print('WAVEFORM',json.dumps(item),flush=True)
        np.savez_compressed(args.output/f'wave-{rho}-{beta}-{cut}.npz',reference_charge=qr,candidate_charge=q,reference_bins=br,candidate_bins=b)
    report=dict(optics_fingerprint=fingerprint,reference_nodes=len(reference.weight_nm),training_states=train,
        heldout_states=held,training_looks=TRAIN_LOOKS.tolist(),heldout_looks=TEST_LOOKS.tolist(),models=summaries,
        selected_rank=summaries[selected]['rank'],charge_holdout_pass=bool(passing),waveform_checks=wave,
        beta_below_095_policy='direct full quadrature; not covered by learned rule',
        event_input_in_training=False,certified_error=False)
    (args.output/'report.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
