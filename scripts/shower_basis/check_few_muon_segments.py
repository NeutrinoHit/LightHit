#!/usr/bin/env python3
"""Few connected primary chords vs saved uncompressed physical-time q1.

No secondary light or production configuration is changed. Uses the same
source hashes, receivers, wavelengths, bins and numerical reference as the
earlier original-step segment control.
"""
import paths  # inputs and outputs: scripts/shower_basis/paths.py
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from time import perf_counter, process_time

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--energy', type=int, default=1000)
    p.add_argument('--pieces', type=int, nargs='+', default=[1, 2, 4, 8, 16, 32])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--legacy-fit', action='store_true', help='also time current one-metre primary fit')
    p.add_argument('--wavelength-nodes', type=int, default=9)
    args = p.parse_args()
    import lighthit as lh
    import ntsim, numba
    from ntsim.LightTransport import bgvd_model_from_ntsim
    from lighthit.experimental.muon_integral import order1_integrals
    from probe_muon_wiggle import fit_pieces
    source_path = paths.muon_source(args.energy)
    ref_path = paths.work(f'muon-segments-20261009/integral/exact{args.energy}/reference.npz')
    meta = json.loads(ref_path.with_suffix('.json').read_text())
    if meta['source_sha256'] != hashlib.sha256(source_path.read_bytes()).hexdigest():
        raise ValueError('reference belongs to a different source')
    if meta['method'] != 'segment' or meta.get('refined_quadrature', False):
        raise ValueError('expected original segment reference and default quadrature')
    if numba.get_num_threads() != 1:
        p.error('NUMBA_NUM_THREADS=1 required')
    with np.load(source_path) as z:
        source = lh.SpectralLightElements(**{k: z[k] for k in z.files if k != 'metadata'}, cone_model='spectral')
        source_meta = json.loads(str(z['metadata']))
    source = source.subset(source.uid == 0)
    with np.load(ref_path) as z:
        rq, rb, origins, edges = z['charge'], z['bins'], z['origins'], z['edges']
    model = bgvd_model_from_ntsim(ntsim.bgvd.Telescope(n_clusters=(1,2,3,4,5)), ntsim.bgvd.Water())
    if model.fingerprint != meta.get('model_fingerprint', source_meta['model_fingerprint']):
        raise ValueError('reference model differs')
    ids = np.array(meta['receiver_indices'])
    d = model.detector
    model = replace(model, detector=replace(d, positions_m=d.positions_m[ids], orientations=d.orientations[ids],
        effective_area_m2=d.effective_area_m2[ids], identifiers={k: np.asarray(v)[ids] for k, v in d.identifiers.items()}))
    kernel = model.kernel(lh.KernelConfig(relative_time_edges_ns=edges, threshold_pe=0., wavelength_nodes=args.wavelength_nodes,
        cache_directory=str(paths.RTE_CACHE.path()), cache_policy='require'))
    config = lh.PromptConfig(element_method='segment', screen=False)
    # Single real step only; timing excludes JIT and cache setup.
    order1_integrals(kernel, source.subset(np.arange(len(source)) == 0), time_origin_ns=origins, config=config)
    args.output.mkdir(parents=True, exist_ok=True)
    report = dict(energy_GeV=args.energy, wavelength_nodes=args.wavelength_nodes,
                  comparison_scope='source change at fixed nine-node spectrum' if args.wavelength_nodes == 9 else
                  'candidate uses a different spectrum; comparison to old reference mixes source and quadrature changes',
                  reference=str(ref_path), reference_sha256=hashlib.sha256(ref_path.read_bytes()).hexdigest(),
                  receiver_indices=ids.tolist(), scope='primary-only order1, connected endpoint chords', cases=[])
    for n in args.pieces + ([0] if args.legacy_fit else []):
        if n:
            candidate, geom = fit_pieces(source, n)
        else:
            from lighthit.experimental.muon_integral import fit_track_integrals
            fitted = fit_track_integrals(source, max_length_m=1.,
                phase_indices=model.medium.sample(kernel.wavelength.wavelength_nm)['phase_index'])
            candidate, geom = fitted.elements, []
        began, cpu = perf_counter(), process_time()
        q, bins, info = order1_integrals(kernel, candidate, time_origin_ns=origins, config=config)
        elapsed, cpu = perf_counter()-began, process_time()-cpu
        row = dict(pieces=len(candidate), current_one_metre_fit=not bool(n), wall_seconds=elapsed, cpu_seconds=cpu, geometry=geom,
                   charge_pe=q.tolist(), relative_charge=((q-rq)/rq).tolist(),
                   bin_l1_relative=(abs(bins-rb).sum(1)/np.maximum(abs(rb).sum(1),1e-30)).tolist(),
                   reference_charge_pe=rq.tolist())
        report['cases'].append(row)
        np.savez_compressed(args.output/f'chords-{n}.npz', charge=q, bins=bins, origins=origins, edges=edges)
        (args.output/'report.json').write_text(json.dumps(report, indent=2))
        print(json.dumps({k:v for k,v in row.items() if k != 'geometry'}), flush=True)


if __name__ == '__main__':
    main()
