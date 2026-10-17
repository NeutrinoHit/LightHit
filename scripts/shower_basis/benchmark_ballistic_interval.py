#!/usr/bin/env python3
"""One-core continuous ballistic charge/bins on saved primary muons.

The old nine-node result is diagnostic only. Accuracy is checked against
independent interval_atoms (direct callable QE) and a tighter new run.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
from time import perf_counter

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--energy', type=int, default=1000)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    import numba
    import lighthit as lh
    import ntsim
    from ntsim.LightTransport import bgvd_model_from_ntsim
    from lighthit.ballistic_interval import ballistic_interval, BallisticIntervalConfig
    from lighthit.experimental.ballistic_spectral import spectral_ballistic_charge
    from lighthit._shower_basis import earliest_element_arrival
    from probe_muon_wiggle import interval_atoms
    from probe_positive_map_screen import read_source
    if numba.get_num_threads() != 1:
        p.error('NUMBA_NUM_THREADS=1 required')
    if (args.output/'report.json').exists():
        p.error('choose an unused output directory')
    source, meta, _ = read_source(args.energy)
    source = source.subset(source.uid == 0)
    model = bgvd_model_from_ntsim(ntsim.bgvd.Telescope(n_clusters=(1,2,3,4,5)), ntsim.bgvd.Water())
    if meta['model_fingerprint'] != model.fingerprint:
        raise ValueError('source/model mismatch')
    kernel = model.kernel(lh.KernelConfig(omega_per_ns=np.array([0.]), cache_policy='require'))
    detector = kernel.detector
    origins = earliest_element_arrival(np.ascontiguousarray(detector.positions_m), source.start_m,
        source.direction, source.length_m, source.start_ns, float(model.medium.group_index.min()))
    edges = kernel.config.relative_time_edges_ns
    ids = np.array([197])
    d = replace(detector, positions_m=detector.positions_m[ids], orientations=detector.orientations[ids],
                effective_area_m2=detector.effective_area_m2[ids],
                identifiers={k:v[ids] for k,v in detector.identifiers.items()})
    warm_kernel = replace(model, detector=d).kernel(kernel.config)
    begin = perf_counter()
    ballistic_interval(warm_kernel, source.subset(np.arange(len(source)) == 0),
                       time_origin_ns=origins[ids], relative_edges_ns=edges)
    warm_seconds = perf_counter()-begin
    # Full detector charge AND time bins, original primary steps.
    result = ballistic_interval(kernel, source, time_origin_ns=origins, relative_edges_ns=edges)
    q9 = spectral_ballistic_charge(kernel, source)
    selected = np.flatnonzero(result.charge_pe >= .005)
    d = replace(detector, positions_m=detector.positions_m[selected], orientations=detector.orientations[selected],
                effective_area_m2=detector.effective_area_m2[selected],
                identifiers={k:v[selected] for k,v in detector.identifiers.items()})
    small = replace(model, detector=d).kernel(kernel.config)
    began = perf_counter()
    exact = interval_atoms(small, source, 32)
    reference_seconds = perf_counter()-began
    qref = np.bincount(exact[0], weights=exact[2], minlength=len(selected))
    stricter = ballistic_interval(small, source, time_origin_ns=origins[selected], relative_edges_ns=edges,
        config=BallisticIntervalConfig(relative_tolerance=2e-7, absolute_tolerance_pe=1e-11,
                                      efficiency_relative_tolerance=1e-9, efficiency_absolute_tolerance=1e-13))
    q, bins = result.charge_pe[selected], result.bins_pe[selected]
    report = dict(energy_GeV=args.energy, primary_steps=len(source), receivers=len(detector),
                  warm_seconds=warm_seconds, result=result.metadata, selected=selected.tolist(),
                  independent_charge_max_relative=float(np.max(abs(q/qref-1))),
                  stricter_charge_max_relative=float(np.max(abs(q/stricter.charge_pe-1))),
                  stricter_bins_l1_max_relative=float(np.max(abs(bins-stricter.bins_pe).sum(1)/stricter.charge_pe)),
                  reference_seconds=reference_seconds, reference_receivers=len(selected),
                  old9_charge_l1_relative=float(abs(q9[selected]-qref).sum()/qref.sum()),
                  old9_pe=q9[selected].tolist(), corrected_pe=q.tolist(), independent_pe=qref.tolist())
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output/'response.npz', charge=result.charge_pe, bins=result.bins_pe, origins=origins, edges=edges)
    (args.output/'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
