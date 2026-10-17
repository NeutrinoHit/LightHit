"""Compare two CLI runs on identical observations, bins, and source normalization."""
import argparse
import json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser()
p.add_argument('candidate',type=Path)
p.add_argument('reference',type=Path)
a=p.parse_args()
with np.load(a.candidate/'spectrum.npz',allow_pickle=False) as c, np.load(a.reference/'spectrum.npz',allow_pickle=False) as r:
    for key in ('displacement_m','direction'):
        if not np.array_equal(c[key],r[key]):raise ValueError(f'Different {key}')
    cm=json.loads(str(c['metadata']));rm=json.loads(str(r['metadata']))
    for key in ('medium','photons','emission_time_ns','source'):
        if cm[key]!=rm[key]:raise ValueError(f'Different {key}')
    q=c['components'][np.flatnonzero(c['omega_per_ns']==0)[0]].sum(axis=-1).real
    qr=r['components'][np.flatnonzero(r['omega_per_ns']==0)[0]].sum(axis=-1).real
    denominator=np.abs(qr)
    if np.any(denominator==0):raise ValueError('Relative metric undefined for zero reference charge')
    out={'charge_relative':(np.abs(q-qr)/denominator).tolist()}
if (a.candidate/'profiles.npz').exists() and (a.reference/'profiles.npz').exists():
    with np.load(a.candidate/'profiles.npz',allow_pickle=False) as c, np.load(a.reference/'profiles.npz',allow_pickle=False) as r:
        if not np.allclose(c['edges_ns'],r['edges_ns'],rtol=0,atol=1e-10):raise ValueError('Different time bins')
        if c['sigma_ns']!=r['sigma_ns']:raise ValueError('Different readout sigma')
        for key in ('raw_components','readout_components'):
            out[key+'_L1_over_ref_charge']=(np.abs(c[key].sum(axis=-1)-r[key].sum(axis=-1)).sum(axis=1)/denominator).tolist()
print(json.dumps(out,indent=2))
