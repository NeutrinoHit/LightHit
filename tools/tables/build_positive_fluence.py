#!/usr/bin/env python3
"""Build spectral cone fluence and arrival-direction moments; preserve RTE caches."""
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
    parser.add_argument('--template', type=Path, help='reuse axes and optical tables of this atlas')
    parser.add_argument('--scalar-only', action='store_true', help='omit arrival-direction moments')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('choose a new output path')
    if args.count < 100 or min(args.rho_bins, args.z_bins) < 8 or args.extent_m < 10:
        parser.error('invalid sampling/grid size')
    import numba
    from lighthit.muon.angular_preview import endpoint_directions, angular_histogram
    if numba.get_num_threads() != 1:
        parser.error('NUMBA_NUM_THREADS=1 required')
    if args.template is not None:
        template = PositiveFluenceAtlas.load(args.template)
        tables = {key: np.asarray(value) for key, value in template.metadata['tables'].items()}
        g = float(template.metadata['g'])
        fingerprint = template.metadata['model_fingerprint']
        rho, z, beta = template.rho_edges_m, template.z_edges_m, template.beta_nodes
    else:
        import lighthit as lh
        import ntsim
        from ntsim.LightTransport import bgvd_model_from_ntsim
        from lighthit.prompt import _spectral_tables
        model = bgvd_model_from_ntsim(ntsim.bgvd.Telescope(n_clusters=(1,2,3,4,5)), ntsim.bgvd.Water())
        tables = _spectral_tables(model.kernel(lh.KernelConfig()))
        g, fingerprint = float(model.medium.g), model.fingerprint
        rho = np.r_[0., np.geomspace(.1, args.extent_m, args.rho_bins)]
        z = np.sinh(np.linspace(-np.arcsinh(args.extent_m), np.arcsinh(args.extent_m), args.z_bins+1))
        beta = np.array([.735, .745, .76, .8, .9, .97, 1.])
    volume = np.pi*np.diff(rho**2)[:, None]*np.diff(z)[None, :]
    density = np.zeros((len(beta), 2, len(rho)-1, len(z)-1))
    sigma = np.zeros_like(density)
    angular = None if args.scalar_only else np.zeros(density.shape+(10,))
    rows = []
    # Compile before timing sampling and binning.
    sample, _ = universal_endpoints(4, 1, 0, g)
    endpoint_histogram(sample, np.ones(4), .75, 1., rho, z)
    began, cpu = perf_counter(), process_time()
    for kind, order in enumerate((1, 2)):
        if angular is None:
            xyz, n = universal_endpoints(args.count, order, args.seed+kind, g)
            directions = None
        else:
            xyz, n, directions = endpoint_directions(args.count, order, args.seed+kind, g)
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
                if angular is None:
                    hist, squares = endpoint_histogram(xyz, weights, cosine, 1/mut, rho, z)
                else:
                    moments, squares = angular_histogram(xyz, directions, weights, cosine, 1/mut, rho, z)
                    hist = moments[..., 0]
                    angular[ib, kind] += spectral*moments/(args.count*volume[..., None])
                density[ib, kind] += spectral*hist/(args.count*volume)
                # Sum per-wavelength SDs: does not assume reweighted samples independent.
                sigma[ib, kind] += spectral*np.sqrt(squares)/(args.count*volume)
                exact = p/mut if order == 1 else p*p/tables['mua'][lam]
                rows.append(dict(order=order, wavelength_nm=float(tables['wavelength_nm'][lam]),
                                 beta=float(b), mass_mc=float(weights.mean()), mass_exact=float(exact),
                                 in_map_fraction=float(hist.sum()/weights.sum())))
            print('ORDER', order, 'LAMBDA', tables['wavelength_nm'][lam],
                  'elapsed', perf_counter()-began, flush=True)
    metadata = dict(kind='positive-spectral-cone-histogram-v2' if angular is not None else 'positive-spectral-cone-histogram-v1', count_per_order=args.count,
                    seed=args.seed, g=g, proposal_p=.5, model_fingerprint=fingerprint,
                    tables={key: value.tolist() for key, value in tables.items()},
                    units='m^-2 per source coefficient0; multiply receiver area and acceptance',
                    coefficient_contract='coefficient2=-coefficient0/beta^2',
                    error_scope='sampling SD envelope only; no interpolation, beta or grouping error bound',
                    zero_cells_are_not_certified=True, inactivity_certified=False,
                    wall_seconds=perf_counter()-began, cpu_seconds=process_time()-cpu,
                    checks=rows)
    metadata['angular_moments'] = None if angular is None else '1 x z xx xz zz xxx xxz xzz zzz; meridian frame; mirror-symmetric'
    atlas = PositiveFluenceAtlas(rho, z, beta, density, sigma, metadata, angular_moments=angular)
    atlas.save(args.output)
    args.output.with_suffix('.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps({key: metadata[key] for key in ['wall_seconds','cpu_seconds','count_per_order']}, indent=2))


if __name__ == '__main__':
    main()
