#!/usr/bin/env python3
"""Positive-map preview plus existing fine transport; explicit research mode.

Unknown tails and empty cells prevent a certified exclusion. Saved fine data
are opened only after transport. Unselected OMs keep labelled coarse charges.
"""
import paths  # inputs and outputs: scripts/shower_basis/paths.py
import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
from time import perf_counter, process_time
from types import SimpleNamespace

import numpy as np


def acceptance_maximum(kernel):
    alpha = np.asarray(kernel.acceptance()['alpha'])
    polynomial = np.polynomial.Legendre(alpha*(2*np.arange(len(alpha))+1)/(4*np.pi))
    roots = polynomial.deriv().roots()
    candidates = [-1., 1.]+[float(r.real) for r in roots if abs(r.imag) < 1e-12 and -1 <= r.real <= 1]
    return float(np.max(polynomial(candidates)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--energy', type=int, choices=(300, 1000, 3000), default=1000)
    p.add_argument('--atlas', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--ballistic-method', choices=('original', 'spectral-fused'), default='original')
    args = p.parse_args()
    if (args.output/'response.npz').exists():
        p.error('preserve previous output')
    import numba
    import lighthit as lh
    import ntsim
    from ntsim.LightTransport import bgvd_model_from_ntsim
    from lighthit.experimental.positive_fluence import PositiveFluenceAtlas
    from lighthit.experimental.adaptive_screen import transport_smart_muon, AdaptiveScreenConfig
    from probe_positive_map_screen import (read_source, group_elements, preview_ballistic, query_groups)
    if numba.get_num_threads() != 1:
        p.error('NUMBA_NUM_THREADS=1 required')
    source, source_meta, source_path = read_source(args.energy)
    model = bgvd_model_from_ntsim(ntsim.bgvd.Telescope(n_clusters=(1,2,3,4,5)), ntsim.bgvd.Water())
    atlas = PositiveFluenceAtlas.load(args.atlas)
    if model.fingerprint != source_meta['model_fingerprint'] or model.fingerprint != atlas.metadata['model_fingerprint']:
        raise ValueError('source/atlas/model fingerprints differ')
    with np.load(paths.muon_ancestry(args.energy)) as archive:
        ancestry = {key: archive[key].copy() for key in archive.files if key != 'metadata'}
    kernel = model.kernel(lh.KernelConfig(omega_per_ns=np.linspace(0,1.2,65),
        readout_omega_per_ns=np.linspace(0,1.2,321), cache_policy='require',
        cache_directory=str(paths.RTE_CACHE.path())))
    kernel.ge2_kernel_store(str(paths.GE2_PACK.path()))
    args.output.mkdir(parents=True, exist_ok=True)
    began, cpu = perf_counter(), process_time()
    groups = group_elements(source)
    q0 = preview_ballistic(kernel, source, args.ballistic_method)
    result = query_groups(atlas, groups, kernel.detector.positions_m)
    factor = kernel.detector.effective_area_m2*acceptance_maximum(kernel)
    scattered = result['charge']*factor[:, None]
    sigma = result['sigma_sum']*factor[:, None]
    charge = np.column_stack((q0, scattered))
    # Force fine calculation near the source and on measurably noisy threshold
    # cases. This cannot certify unobserved empty-cell or out-of-grid tails.
    threshold = kernel.config.threshold_pe
    noisy = ((sigma.sum(1) > .25*scattered.sum(1))
             & (q0+scattered.sum(1)+4*sigma.sum(1) >= threshold/8))
    forced = (result['nearest_source_m'] < 3) | noisy
    preview_seconds = perf_counter()-began
    metadata = dict(kind='positive-map-scalar-acceptance-max', elapsed_seconds=preview_seconds,
        ballistic_method=args.ballistic_method,
        groups=len(groups.points), samples_per_order=atlas.metadata['count_per_order'],
        atlas_sha256=hashlib.sha256(args.atlas.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
        inactivity_certified=False, forced_count=int(forced.sum()),
        uncertainty='sampled tails, interpolation and grouping remain empirical')
    preview = SimpleNamespace(charge_components_pe=charge, charge_pe=charge.sum(1),
                              unsafe_mask=forced, metadata=metadata)
    np.savez_compressed(args.output/'independent-preview.npz', charge=charge, sigma=sigma, forced=forced)
    print('PREVIEW', json.dumps(metadata), flush=True)
    response = transport_smart_muon(kernel, source, ancestry, preview=preview,
        first_pack_directory=None, screen_config=AdaptiveScreenConfig(scatter_margin=2.))
    elapsed, cpu_seconds = perf_counter()-began, process_time()-cpu
    response.save(args.output/'response.npz')
    reference = (paths.ntsim_hybrid_response(1000, 'all', 'centroid')
                 if args.energy == 1000 else paths.ntsim_hybrid_response(args.energy, 'all'))
    with np.load(reference) as z:
        q, bins, active = z['charge_components_pe'], z['components_pe'], z['active']
        np.testing.assert_allclose(z['time_origin_ns'], response.time_origin_ns, rtol=0, atol=1e-10)
    union = active | response.active
    comparison = {}
    for label, cols in [('0',[0]), ('1',[1]), ('ge2',[2]), ('total',[0,1,2])]:
        a, b = q[:, cols].sum(1), response.charge_components_pe[:, cols].sum(1)
        ah, bh = bins[:, :, cols].sum(2), response.components_pe[:, :, cols].sum(2)
        comparison[label] = dict(active_max_abs_pe=float(np.max(abs(a[union]-b[union]), initial=0)),
            active_max_relative=float(np.max(abs(a[union]-b[union])/np.maximum(abs(a[union]),1e-30),initial=0)),
            active_bins_l1_global=float(abs(ah[union]-bh[union]).sum()/max(abs(ah[union]).sum(),1e-30)))
    report = dict(energy_GeV=args.energy, modules=len(q), active=int(response.active.sum()),
        lost=np.flatnonzero(active & ~response.active).tolist(), gained=np.flatnonzero(~active & response.active).tolist(),
        elapsed_seconds=elapsed, cpu_seconds=cpu_seconds, preview_seconds=preview_seconds,
        peak_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(2**20 if sys.platform=='darwin' else 2**10),
        threads=numba.get_num_threads(), comparison=comparison, reference=str(reference),
        screen=response.metadata['smart_screen'], fine_reference_loaded_after_transport=True)
    (args.output/'report.json').write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({key: val for key, val in report.items() if key != 'screen'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
