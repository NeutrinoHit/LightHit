#!/usr/bin/env python3
"""Build an experimental spectral scalar cone atlas; never writes RTE caches."""
import argparse
import json
from pathlib import Path
from time import perf_counter, process_time

import numpy as np

from lighthit.experimental.positive_fluence import (
    PositiveFluenceAtlas, universal_endpoints, endpoint_histogram)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--count', type=int, default=3000000)
    parser.add_argument('--seed', type=int, default=71)
    parser.add_argument('--rho-bins', type=int, default=64)
    parser.add_argument('--z-bins', type=int, default=256)
    parser.add_argument('--extent-m', type=float, default=500.)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('choose a new output path')
    if args.count < 100 or min(args.rho_bins, args.z_bins) < 8 or args.extent_m < 10:
        parser.error('invalid sampling/grid size')
    import numba
    import lighthit as lh
    import ntsim
    from ntsim.LightTransport import bgvd_model_from_ntsim
    from lighthit.prompt import _spectral_tables
    if numba.get_num_threads() != 1:
        parser.error('NUMBA_NUM_THREADS=1 required')
    model = bgvd_model_from_ntsim(ntsim.bgvd.Telescope(n_clusters=(1,2,3,4,5)), ntsim.bgvd.Water())
    kernel = model.kernel(lh.KernelConfig())
    tables = _spectral_tables(kernel)
    # The production HG phase is wavelength-independent g=.9 for this model.
    # Verify against model data rather than infer it from the atlas filename.
    g = float(model.medium.g)
    rho = np.r_[0., np.geomspace(.1, args.extent_m, args.rho_bins)]
    z = np.sinh(np.linspace(-np.arcsinh(args.extent_m), np.arcsinh(args.extent_m), args.z_bins+1))
    beta = np.array([.735, .745, .76, .8, .9, .97, 1.])
    volume = np.pi*np.diff(rho**2)[:, None]*np.diff(z)[None, :]
    density = np.zeros((len(beta), 2, len(rho)-1, len(z)-1))
    sigma = np.zeros_like(density)
    rows = []
    # Compile before timing sampling and binning.
    sample, _ = universal_endpoints(4, 1, 0, g)
    endpoint_histogram(sample, np.ones(4), .75, 1., rho, z)
    began, cpu = perf_counter(), process_time()
    for kind, order in enumerate((1, 2)):
        xyz, n = universal_endpoints(args.count, order, args.seed+kind, g)
        proposal = np.ones(args.count) if order == 1 else .5**(n-1)
        for lam in range(len(tables['wavelength_nm'])):
            mut = tables['mut'][lam]
            p = tables['mus'][lam]/mut
            weights = p**n/(mut*proposal)
            for ib, b in enumerate(beta):
                cosine = 1/(b*tables['phase_index'][lam])
                if cosine >= 1:
                    continue
                spectral = tables['s0w'][lam]*(1-cosine*cosine)
                hist, squares = endpoint_histogram(xyz, weights, cosine, 1/mut, rho, z)
                density[ib, kind] += spectral*hist/(args.count*volume)
                # Sum per-wavelength SDs: does not assume reweighted samples independent.
                sigma[ib, kind] += spectral*np.sqrt(squares)/(args.count*volume)
                exact = p/mut if order == 1 else p*p/tables['mua'][lam]
                rows.append(dict(order=order, wavelength_nm=float(tables['wavelength_nm'][lam]),
                                 beta=float(b), mass_mc=float(weights.mean()), mass_exact=float(exact),
                                 in_map_fraction=float(hist.sum()/weights.sum())))
            print('ORDER', order, 'LAMBDA', tables['wavelength_nm'][lam],
                  'elapsed', perf_counter()-began, flush=True)
    metadata = dict(kind='positive-spectral-cone-histogram-v1', count_per_order=args.count,
                    seed=args.seed, g=g, proposal_p=.5, model_fingerprint=model.fingerprint,
                    tables={key: value.tolist() for key, value in tables.items()},
                    units='m^-2 per source coefficient0; multiply receiver area and acceptance',
                    coefficient_contract='coefficient2=-coefficient0/beta^2',
                    error_scope='sampling SD envelope only; no interpolation, beta or grouping error bound',
                    zero_cells_are_not_certified=True, inactivity_certified=False,
                    wall_seconds=perf_counter()-began, cpu_seconds=process_time()-cpu,
                    checks=rows)
    atlas = PositiveFluenceAtlas(rho, z, beta, density, sigma, metadata)
    atlas.save(args.output)
    args.output.with_suffix('.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps({key: metadata[key] for key in ['wall_seconds','cpu_seconds','count_per_order']}, indent=2))


if __name__ == '__main__':
    main()
