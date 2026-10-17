#!/usr/bin/env python3
"""Bounded angular-degree probe on the saved exact-point G4 benchmark sample.

No full-event accuracy claim: sample points are stratified by emission time,
not selected or reweighted to approximate an event integral.
"""
import argparse
import hashlib
import json
from pathlib import Path
from time import process_time

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sample', type=Path, required=True)
    p.add_argument('--degrees', type=int, nargs='+', default=[8,12,16,24,32])
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():p.error('fresh output directory required')
    if any(l < 4 or l > 32 for l in a.degrees):p.error('degree must lie in [4,32]')
    if a.repeats<1:p.error('positive repeats required')
    import lighthit as lh
    import ntsim
    import numba
    from ntsim.LightTransport import bgvd_model_from_ntsim
    from lighthit.experimental.point_ge2 import apply_point_spectral
    from lighthit.experimental.axial_source import AxisFrame
    from lighthit.shower_basis import Ge2KernelStore
    if numba.get_num_threads()!=1:p.error('one thread required')
    report=json.loads((a.sample/'report.json').read_text())
    if hashlib.sha256(Path(report['source']).read_bytes()).hexdigest()!=report['source_sha256']:
        p.error('source changed')
    with np.load(report['source']) as z:
        source=lh.SpectralLightElements(**{k:z[k] for k in z.files if k!='metadata'},cone_model='spectral')
    model=bgvd_model_from_ntsim(ntsim.bgvd.Telescope(n_clusters=(1,2,3,4,5)),ntsim.bgvd.Water())
    if model.fingerprint!=report['model_fingerprint']:p.error('model changed')
    kernel=model.kernel(lh.KernelConfig(omega_per_ns=np.array(report['omega_per_ns']),cache_policy='require'))
    phase=kernel.medium.sample(kernel.wavelength.wavelength_nm)['phase_index']
    source=source.subset((source.uid!=0)&(source.beta*phase.max()>1))
    frame=AxisFrame.of(source.field(0,phase_index=float(phase.max())))
    bundle=Ge2KernelStore.load(report['kernel_pack']).bundle(kernel,np.arange(65))
    bands=np.asarray(report['wavelength_indices'])
    if not np.array_equal(bands,np.arange(bands[0],bands[-1]+1)):
        p.error('sample requires consecutive wavelength bands')
    band_slice=slice(int(bands[0]),int(bands[-1])+1)
    bundle=dict(bundle,coefficients=bundle['coefficients'][band_slice],
                absorption=bundle['absorption'][band_slice],speed=bundle['speed'][band_slice])
    with np.load(a.sample/'sample.npz') as z:
        points=z['points_m'];times=z['times_ns'];angular=z['angular'];scale=z['output_scale']
    with np.load(a.sample/'spectra.npz') as z:
        expected=z['point'];positions=z['receivers_m'];looks=z['looks']
    alpha=np.ascontiguousarray(kernel.acceptance()['alpha'],float)
    rows={}
    for degree in a.degrees:
        channels=sum(2*min(l,4)+1 for l in range(degree+1))
        small=dict(bundle,degree=degree,coefficients=bundle['coefficients'][:,:,:degree+1])
        values=np.ascontiguousarray(angular[:,:,:channels])
        apply_point_spectral(small,frame,points,values,times,positions,looks,alpha,scale)
        timings=[]
        for _ in range(a.repeats):
            cpu=process_time()
            actual=apply_point_spectral(small,frame,points,values,times,positions,looks,alpha,scale)
            timings.append(process_time()-cpu)
        rows[str(degree)]=dict(cpu_s=float(np.median(timings)),cpu_repeats_s=timings,
            charge_relative_error=(abs(actual[0]-expected[0])/np.maximum(abs(expected[0]),1e-30)).tolist(),
            charge_absolute_error_pe=abs(actual[0]-expected[0]).tolist(),
            normalized_spectrum_error=(abs(actual-expected).max(0)/np.maximum(abs(expected).max(0),1e-30)).tolist())
    a.output.mkdir(parents=True)
    result=dict(sample=str(a.sample),source_sha256=report['source_sha256'],threads=1,
                sample_points=len(points),receiver_indices=report['receiver_indices'],
                scope='bounded sample, degree32 operator reference, not full-event error or speedup',results=rows)
    (a.output/'report.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
