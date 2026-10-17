#!/usr/bin/env python3
"""Bounded one-core charge-only q1 validation on actual stopping-muon tails."""
import paths  # inputs and outputs: scripts/shower_basis/paths.py
import argparse,json
from pathlib import Path
from dataclasses import replace
from time import perf_counter,process_time
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('choose unused output directory')
    import lighthit as lh,ntsim,numba
    from ntsim.LightTransport import bgvd_model_from_ntsim
    from lighthit.experimental.primary_polyline import primary_spectral_quadrature
    from lighthit.experimental.universal_primary import subset_kernel
    from lighthit.experimental.first_charge_only import first_scattering_charge_only
    from lighthit.experimental.muon_integral import order1_integrals
    from lighthit.prompt import PromptConfig
    from lighthit._shower_basis import earliest_element_arrival
    if numba.get_num_threads()!=1:p.error('one Numba thread required')
    path=paths.muon_source(300)
    with np.load(path) as z:
        source=lh.SpectralLightElements(**{k:z[k] for k in z.files if k!='metadata'},cone_model='spectral')
    source=source.subset(source.uid==0)
    model=bgvd_model_from_ntsim(ntsim.bgvd.Telescope(n_clusters=(1,2,3,4,5)),ntsim.bgvd.Water())
    kernel=model.kernel(lh.KernelConfig(omega_per_ns=np.array([0.]),cache_policy='require'))
    kernel.wavelength=primary_spectral_quadrature(kernel.medium,kernel.detector,order=2)
    tail=source.subset(source.beta<.999)
    ids=np.argsort(np.linalg.norm(model.detector.positions_m-tail.centroid_m,axis=1))[:3]
    local=subset_kernel(kernel,ids)
    warm=tail.subset(np.arange(len(tail))==0)
    first_scattering_charge_only(subset_kernel(local,[0]),warm)
    order1_integrals(subset_kernel(local,[0]),warm,time_origin_ns=np.array([0.]),
                     config=PromptConfig(element_method='segment',screen=False))
    report=dict(source=str(path),energy_GeV=300,receiver_indices=ids.tolist(),
                threads=1,spectral_nodes=len(local.wavelength.wavelength_nm),cases=[])
    a.output.mkdir(parents=True)
    for cutoff in [.999,.99,.95]:
        selected=source.subset(source.beta<cutoff)
        origins=earliest_element_arrival(np.ascontiguousarray(local.detector.positions_m),selected.start_m,
            selected.direction,selected.length_m,selected.start_ns,float(local.medium.group_index.min()))
        wall,cpu=perf_counter(),process_time()
        expected,bins,_=order1_integrals(local,selected,time_origin_ns=origins,
            config=PromptConfig(element_method='segment',screen=False))
        refwall,refcpu=perf_counter()-wall,process_time()-cpu
        actual,meta=first_scattering_charge_only(local,selected)
        # Clock and readout changes must be irrelevant, even with real beta.
        changed=replace(selected,start_ns=selected.start_ns+1234.,end_ns=selected.end_ns+2345.)
        other,_=first_scattering_charge_only(local,changed)
        np.testing.assert_array_equal(actual,other)
        row=dict(beta_cut=cutoff,elements=len(selected),length_m=float(selected.length_m.sum()),
            reference_charge_pe=expected.tolist(),charge_only_pe=actual.tolist(),
            relative_error=(abs(actual-expected)/np.maximum(abs(expected),1e-30)).tolist(),
            reference_cpu_s=refcpu,reference_wall_s=refwall,charge_cpu_s=meta['cpu_seconds'],
            charge_wall_s=meta['wall_seconds'],cpu_speedup=refcpu/meta['cpu_seconds'],
            clock_invariant=True,computed_bins=meta['computed_bins'])
        report['cases'].append(row)
        np.savez_compressed(a.output/f'beta{cutoff}.npz',reference_charge=expected,reference_bins=bins,
                            charge_only=actual,time_origin_ns=origins)
        (a.output/'report.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(row),flush=True)

if __name__=='__main__':main()
