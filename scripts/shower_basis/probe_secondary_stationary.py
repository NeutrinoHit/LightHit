#!/usr/bin/env python3
"""One-core stationary grouping test on saved real G4 secondary sources."""
import paths  # inputs and outputs: scripts/shower_basis/paths.py
import argparse, json
from pathlib import Path
from dataclasses import replace
from time import process_time
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--energy', type=int, choices=[300,1000,3000], default=1000)
    p.add_argument('--oms', type=int, nargs='+', default=[196,197,198,342,666,1062,1241])
    p.add_argument('--wavelength', type=float, default=480.)
    p.add_argument('--output', type=Path, required=True)
    a=p.parse_args()
    if a.output.exists():p.error('fresh output directory required')
    import lighthit as lh, ntsim, numba
    from ntsim.LightTransport import bgvd_model_from_ntsim
    from lighthit.experimental.universal_primary import subset_kernel
    from lighthit.experimental.stationary_pencils import stationary_pencils_charge
    from lighthit.experimental.first_charge_only import first_scattering_charge_only
    if numba.get_num_threads()!=1:p.error('one Numba thread required')
    path=paths.muon_source(a.energy)
    with np.load(path) as z:
        source=lh.SpectralLightElements(**{k:z[k] for k in z.files if k!='metadata'},cone_model='spectral')
    source=source.subset(source.uid!=0) # Includes nuclear/late sources, not only e±.
    model=bgvd_model_from_ntsim(ntsim.bgvd.Telescope(n_clusters=(1,2,3,4,5)),ntsim.bgvd.Water())
    kernel=subset_kernel(model.kernel(lh.KernelConfig(omega_per_ns=np.array([0.]),cache_policy='require')),a.oms)
    kernel.wavelength=lh.WavelengthQuadrature(np.array([a.wavelength]),np.array([1.]))
    settings=lh.PromptConfig(element_method='grouped',screen=False,batch_charge_tolerance_pe=0.)
    warm=source.subset(np.arange(len(source))<1)
    stationary_pencils_charge(subset_kernel(kernel,[0]),warm,config=settings)
    first_scattering_charge_only(subset_kernel(kernel,[0]),warm,gauss_order=6)
    a.output.mkdir(parents=True)
    report=dict(energy_GeV=a.energy,source=str(path),secondary_elements=len(source),
        receiver_indices=a.oms,clusters=5,wavelength_nm=a.wavelength,weight_nm=1.,threads=1,
        scope='all secondary records, single wavelength, all-time q1 only',cases={})
    outputs={}
    ref, meta=first_scattering_charge_only(kernel,source,gauss_order=6)
    outputs['direct']=ref; report['reference']=meta
    print('direct',ref.tolist(),meta['cpu_seconds'],flush=True)
    for name,config,merge_time in [('ordinary',settings,True),('stationary',settings,False),
        ('stationary-fine',replace(settings,merge_log_r=.005,merge_theta_ratio=.02),False)]:
        q,meta=stationary_pencils_charge(kernel,source,config=config,merge_time=merge_time)
        outputs[name]=q
        report['cases'][name]=dict(charge_pe_per_nm=q.tolist(),
            relative_charge_error=(abs(q-ref)/np.maximum(abs(ref),1e-30)).tolist(),
            absolute_error_pe_per_nm=abs(q-ref).tolist(),**meta)
        print(name,q.tolist(),meta['cpu_seconds'],flush=True)
        (a.output/'report.json').write_text(json.dumps(report,indent=2))
    changed=replace(source,start_ns=source.start_ns+1234.,end_ns=source.end_ns+2345.)
    q,_=stationary_pencils_charge(kernel,changed,config=settings)
    np.testing.assert_allclose(q,outputs['stationary'],rtol=1e-11,atol=1e-20)
    report['clock_max_relative_roundoff']=float(np.max(abs(q-outputs['stationary'])/
        np.maximum(abs(q),1e-30)))
    report['clock_invariant']=True
    np.savez_compressed(a.output/'charges.npz',**outputs)
    (a.output/'report.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
