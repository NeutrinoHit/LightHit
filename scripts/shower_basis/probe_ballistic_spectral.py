#!/usr/bin/env python3
"""Measure exact spectral-fused ballistic charge on saved five-cluster muons."""
import paths  # inputs and outputs: scripts/shower_basis/paths.py
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from time import perf_counter, process_time

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--energy', type=int, choices=(300, 1000, 3000), default=1000)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import numba
    import ntsim
    import lighthit as lh
    from ntsim.LightTransport import bgvd_model_from_ntsim
    from lighthit.experimental.ballistic_spectral import spectral_ballistic_charge
    if numba.get_num_threads() != 1:
        parser.error('requires NUMBA_NUM_THREADS=1')
    if (args.output/'report.json').exists():
        parser.error('choose a new output directory to preserve results')
    args.output.mkdir(parents=True, exist_ok=True)
    sourcepath = paths.muon_source(args.energy)
    with np.load(sourcepath) as z:
        source = lh.SpectralLightElements(**{k: z[k] for k in z.files if k != 'metadata'},
                                          cone_model='spectral')
        provenance = json.loads(str(z['metadata']))
    model = bgvd_model_from_ntsim(ntsim.bgvd.Telescope(n_clusters=(1, 2, 3, 4, 5)), ntsim.bgvd.Water())
    if model.fingerprint != provenance['model_fingerprint']:
        parser.error('source/model fingerprints differ')
    config = lh.KernelConfig(omega_per_ns=np.linspace(0., 1.2, 65), cache_policy='require',
                             cache_directory=str(paths.RTE_CACHE.path()))
    kernel = model.kernel(config)
    d = model.detector
    warm_d = replace(d, positions_m=d.positions_m[:1], orientations=d.orientations[:1],
                     effective_area_m2=d.effective_area_m2[:1],
                     identifiers={k: np.asarray(v)[:1] for k, v in d.identifiers.items()})
    warm_source = source.subset(np.arange(len(source)) == int(np.argmax(source.beta)))
    start = perf_counter()
    spectral_ballistic_charge(replace(model, detector=warm_d).kernel(config), warm_source)
    warm_seconds = perf_counter()-start
    start, cpu = perf_counter(), process_time()
    charge = spectral_ballistic_charge(kernel, source)
    elapsed_seconds, cpu_seconds = perf_counter()-start, process_time()-cpu
    # Stored original q0 enters only the verification after calculation.
    referencepath = paths.work(f'positive-map-20261009/screen/point-{args.energy}-4m/preview.npz')
    reportpath = referencepath.with_name('report.json')
    old = json.loads(reportpath.read_text())
    if old['source_sha256'] != hashlib.sha256(sourcepath.read_bytes()).hexdigest():
        raise ValueError('reference source differs')
    with np.load(referencepath) as z:
        reference = z['charge'][:, 0].copy()
    difference = abs(charge-reference)
    report = dict(energy_GeV=args.energy, source_elements=len(source), receivers=len(charge),
                  elapsed_seconds=elapsed_seconds, cpu_seconds=cpu_seconds, warm_seconds=warm_seconds,
                  old_ballistic_seconds=old['original_ballistic_seconds'],
                  speedup=old['original_ballistic_seconds']/elapsed_seconds,
                  maximum_absolute_difference_pe=float(difference.max()),
                  charge_L1_relative=float(difference.sum()/max(reference.sum(), 1e-300)),
                  maximum_relative_difference_above_1e_10_pe=float(np.max(
                      difference[reference > 1e-10]/reference[reference > 1e-10])),
                  source_sha256=hashlib.sha256(sourcepath.read_bytes()).hexdigest(),
                  reference_loaded_after_calculation=True, numba_threads=numba.get_num_threads())
    np.testing.assert_allclose(charge, reference, rtol=5e-10, atol=1e-11)
    np.savez_compressed(args.output/'charge.npz', charge=charge, reference_charge=reference)
    (args.output/'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
