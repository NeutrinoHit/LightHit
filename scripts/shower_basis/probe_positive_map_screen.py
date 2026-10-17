#!/usr/bin/env python3
"""Independent positive-fluence screening of saved five-cluster muon events.

The proxy uses all source groups and the maximum detector acceptance. It is an
experimental estimator, not a certified upper bound. Fine responses are read
only after selection, for reporting; the production transport is not modified.
"""
from __future__ import annotations

import paths  # inputs and outputs: scripts/shower_basis/paths.py
import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from time import perf_counter, process_time

import numpy as np


@dataclass
class Groups:
    points: np.ndarray
    directions: np.ndarray
    coefficient0: np.ndarray
    beta: np.ndarray
    radius: np.ndarray
    primary: np.ndarray
    counts: np.ndarray


def group_elements(source, cell_m=4., direction_step=.5,
                   beta_edges=(.76, .8, .9, .97, 1.)):
    """Positive groups preserving total two-field spectral coefficients.

    Effective beta matches the summed c0 and c2 exactly. Geometry and angular
    averaging remain approximations and must be tested with smaller cells/bins.
    """
    if cell_m <= 0 or direction_step <= 0:
        raise ValueError('cell and direction sizes must be positive')
    c0, c2 = np.asarray(source.coefficient0), np.asarray(source.coefficient2)
    if not np.allclose(c2, -c0/source.beta**2, rtol=1e-12, atol=0):
        raise ValueError('atlas assumes Cherenkov coefficient2=-coefficient0/beta²')
    midpoint = source.midpoints_m
    keys = np.column_stack((np.floor(midpoint/cell_m).astype(np.int64),
                            np.floor(source.direction/direction_step).astype(np.int64),
                            np.searchsorted(beta_edges, source.beta), source.uid == 0))
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    count = int(inverse.max())+1
    def reduce(values):
        return np.bincount(inverse, weights=values, minlength=count)
    weights = reduce(c0)
    points = np.column_stack([reduce(c0*midpoint[:, i])/weights for i in range(3)])
    directions = np.column_stack([reduce(c0*source.direction[:, i])/weights for i in range(3)])
    norm = np.linalg.norm(directions, axis=1)
    if np.any(norm < 1e-8):
        raise ValueError('direction bins permitted cancelling directions')
    directions /= norm[:, None]
    radii = np.zeros(count)
    np.maximum.at(radii, inverse,
                  np.linalg.norm(midpoint-points[inverse], axis=1)+source.length_m/2)
    return Groups(np.ascontiguousarray(points), np.ascontiguousarray(directions),
                  weights, np.sqrt(-weights/reduce(c2)), radii,
                  reduce(c0*(source.uid == 0)) > 0,
                  np.bincount(inverse, minlength=count))


def original_ballistic(kernel, source):
    from lighthit import prompt
    tables = prompt._spectral_tables(kernel)
    elements = prompt._source_elements(kernel, source)
    fields = prompt._ballistic_fields(source, elements, tables)
    backend = prompt._ballistic_backend()
    positions = np.ascontiguousarray(kernel.detector.positions_m, float)
    looks = np.ascontiguousarray(-kernel.detector.orientations, float)
    alpha = np.ascontiguousarray(kernel.acceptance()['alpha'], float)
    area = kernel.detector.effective_area_m2
    charge = np.zeros(len(positions))
    for lam, wavelength in enumerate(tables['wavelength_nm']):
        item = fields(lam)
        if item is not None:
            field, cone = item
            charge += (tables['weight_nm'][lam]*tables['efficiency'][lam]*area
                       *backend(field, positions, looks,
                                kernel.medium.band(float(wavelength)), cone, alpha)[0])
    return charge


def preview_ballistic(kernel, source, method='original'):
    if method == 'original':
        return original_ballistic(kernel, source)
    if method == 'spectral-fused':
        from lighthit.experimental.ballistic_spectral import spectral_ballistic_charge
        return spectral_ballistic_charge(kernel, source)
    raise ValueError('unknown ballistic method')


def screen_points(atlas, points, directions, coefficient0, beta, receivers):
    """Fused point queries; sigma sum is a correlated-error diagnostic only."""
    return _screen_points(points, directions, coefficient0, beta, receivers,
                          atlas.rho_nodes, atlas.z_edges_m, atlas.beta_nodes,
                          atlas.density, atlas.sigma)


from numba import njit
from lighthit.experimental.positive_fluence import _bracket


@njit(cache=True)
def _screen_points(points, directions, coefficient0, beta, receivers,
                   rho_nodes, z_edges, beta_nodes, density, sigma):
    charge = np.zeros((len(receivers), 2))
    sigma_sum = np.zeros_like(charge)
    outside_pairs = np.zeros(len(receivers), np.int64)
    empty_pairs = np.zeros(len(receivers), np.int64)
    beta_clipped_charge = np.zeros_like(charge)
    nearest_source_m = np.full(len(receivers), np.inf)
    for j in range(len(receivers)):
        for i in range(len(points)):
            dx = receivers[j, 0]-points[i, 0]
            dy = receivers[j, 1]-points[i, 1]
            dz = receivers[j, 2]-points[i, 2]
            r2 = dx*dx+dy*dy+dz*dz
            z = dx*directions[i, 0]+dy*directions[i, 1]+dz*directions[i, 2]
            rho = np.sqrt(max(0., r2-z*z))
            nearest_source_m[j] = min(nearest_source_m[j], np.sqrt(r2))
            if rho > rho_nodes[-1] or z < z_edges[0] or z >= z_edges[-1]:
                outside_pairs[j] += 1
                continue
            ir, u = _bracket(rho_nodes, rho)
            ib, v = _bracket(beta_nodes, beta[i])
            iz = min(np.searchsorted(z_edges, z, side='right')-1, len(z_edges)-2)
            empty = True
            beta_out = beta[i] < beta_nodes[0] or beta[i] > beta_nodes[-1]
            for db in range(2):
                for dr in range(2):
                    w = coefficient0[i]*(v if db else 1-v)*(u if dr else 1-u)
                    for k in range(2):
                        q = w*density[ib+db, k, ir+dr, iz]
                        charge[j, k] += q
                        sigma_sum[j, k] += w*sigma[ib+db, k, ir+dr, iz]
                        empty = empty and q == 0
                        if beta_out:
                            beta_clipped_charge[j, k] += q
            if empty:
                empty_pairs[j] += 1
    return charge, sigma_sum, outside_pairs, empty_pairs, beta_clipped_charge, nearest_source_m


def query_groups(atlas, groups, receivers):
    arrays = screen_points(atlas, groups.points, groups.directions, groups.coefficient0,
                           groups.beta, np.ascontiguousarray(receivers))
    names = ('charge', 'sigma_sum', 'outside_pairs', 'empty_pairs',
             'beta_clipped_charge', 'nearest_source_m')
    return dict(zip(names, arrays))


def screen_primary_segments(atlas, source, receivers):
    from lighthit.experimental.muon_integral import fit_track_integrals
    from lighthit.experimental.positive_fluence import segment_queries, prefixes
    fitted = fit_track_integrals(source, max_length_m=10., position_tolerance_m=.01,
                                 direction_tolerance_rad=.001)
    elements = fitted.elements
    sigma_left, sigma_right = prefixes(atlas.sigma, atlas.z_edges_m)
    values = np.zeros((len(receivers), 2))
    errors = np.zeros_like(values)
    outside_pairs = np.zeros(len(receivers), np.int64)
    starts, directions = elements.start_m, elements.direction
    length = elements.length_m
    beta = np.sqrt(-elements.coefficient0/elements.coefficient2)
    per_m = elements.coefficient0/length
    for j, receiver in enumerate(receivers):
        delta = receiver-starts
        z = np.sum(delta*directions, axis=1)
        rho = np.sqrt(np.maximum(0., np.sum(delta**2, axis=1)-z*z))
        response, outside = segment_queries(rho, z-length, z, beta,
                    atlas.rho_nodes, atlas.z_edges_m, atlas.beta_nodes, atlas.left, atlas.right)
        error, _ = segment_queries(rho, z-length, z, beta,
                    atlas.rho_nodes, atlas.z_edges_m, atlas.beta_nodes, sigma_left, sigma_right)
        values[j] = np.sum(response*per_m[:, None], axis=0)
        errors[j] = np.sum(error*per_m[:, None], axis=0)
        outside_pairs[j] = outside.sum()
    details = dict(original_primary_elements=len(source), primary_segments=len(elements),
                   fit_position_error_m=fitted.maximum_position_error_m,
                   fit_direction_rms_rad=fitted.maximum_direction_rms_rad,
                   fit_time_error_ns=fitted.maximum_time_error_ns)
    return values, errors, outside_pairs, details


def read_source(energy):
    import lighthit as lh
    path = paths.muon_source(energy)
    with np.load(path) as z:
        source = lh.SpectralLightElements(
            **{k: z[k] for k in z.files if k != 'metadata'}, cone_model='spectral')
        provenance = json.loads(str(z['metadata']))
    return source, provenance, path


def self_check():
    from types import SimpleNamespace
    source = SimpleNamespace(
        coefficient0=np.array([2., 3., 4.]), beta=np.array([.99, .98, .8]),
        coefficient2=-np.array([2., 3., 4.])/np.array([.99, .98, .8])**2,
        midpoints_m=np.array([[.1, .1, .1], [.2, .1, .1], [2., 2., 2.]]),
        direction=np.tile([0., 1., 0.], (3, 1)), length_m=np.array([.1, .2, .1]),
        uid=np.array([0, 0, 1]))
    grouped = group_elements(source)
    assert np.isclose(grouped.coefficient0.sum(), source.coefficient0.sum())
    assert np.isclose(np.sum(-grouped.coefficient0/grouped.beta**2), source.coefficient2.sum())
    assert np.allclose(np.linalg.norm(grouped.directions, axis=1), 1.)
    assert grouped.counts.sum() == 3
    from lighthit.experimental.positive_fluence import PositiveFluenceAtlas
    rho_edges = np.array([0., 1., 2., 4.])
    z_edges = np.array([-5., 0., 5.])
    beta_nodes = np.array([.76, .9, 1.])
    density = np.arange(3*2*3*2).reshape(3, 2, 3, 2).astype(float)+1.
    atlas = PositiveFluenceAtlas(rho_edges, z_edges, beta_nodes, density,
                                density*.1, {})
    xyz = np.array([[1., 0., 2.], [2., 0., -2.], [100., 0., 0.]])
    fused = screen_points(atlas, np.zeros((1, 3)), np.array([[0., 0., 1.]]),
                          np.array([2.]), np.array([.95]), xyz)
    direct, outside = atlas.points(np.abs(xyz[:, 0]), xyz[:, 2], .95)
    assert np.allclose(fused[0], 2*direct)
    assert np.array_equal(fused[2] > 0, outside)
    assert np.allclose(fused[1], .1*fused[0])
    print('grouping spectral conservation and fused/direct atlas query checks passed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--energy', type=int, choices=(300, 1000, 3000), default=1000)
    parser.add_argument('--atlas', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--cell-m', type=float, default=4.)
    parser.add_argument('--direction-step', type=float, default=.5)
    parser.add_argument('--self-check', action='store_true')
    parser.add_argument('--primary-segments', action='store_true')
    parser.add_argument('--ballistic-method', choices=('original', 'spectral-fused'), default='original')
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    if args.atlas is None:
        parser.error('--atlas is required')
    import numba
    import ntsim
    import lighthit as lh
    from ntsim.LightTransport import bgvd_model_from_ntsim
    from lighthit.experimental.positive_fluence import PositiveFluenceAtlas
    if numba.get_num_threads() != 1:
        parser.error('requires NUMBA_NUM_THREADS=1')
    source, provenance, sourcepath = read_source(args.energy)
    model = bgvd_model_from_ntsim(ntsim.bgvd.Telescope(n_clusters=(1, 2, 3, 4, 5)), ntsim.bgvd.Water())
    if model.fingerprint != provenance['model_fingerprint']:
        parser.error('saved source/model fingerprints differ')
    output = args.output or paths.work(f'positive-map-20261009/screen/{args.energy}')
    output.mkdir(parents=True, exist_ok=True)
    if (output/'report.json').exists():
        parser.error('preserve previous result: choose another output directory')
    atlas = PositiveFluenceAtlas.load(args.atlas)
    if atlas.metadata.get('model_fingerprint') != model.fingerprint:
        parser.error('atlas/model fingerprints differ')
    kernel = model.kernel(lh.KernelConfig(
        omega_per_ns=np.linspace(0., 1.2, 65), cache_policy='require',
        cache_directory=str(paths.RTE_CACHE.path())))
    detector = model.detector
    # The maximum is a receiver angular factor, not a bound on map/group errors.
    alpha = np.asarray(kernel.acceptance()['alpha'])
    polynomial = np.polynomial.Legendre(alpha*(2*np.arange(len(alpha))+1)/(4*np.pi))
    stationary = polynomial.deriv().roots()
    real_stationary = stationary[np.abs(np.imag(stationary)) < 1e-12].real
    candidates = np.r_[-1., 1., real_stationary[np.abs(real_stationary) <= 1.]]
    peak_acceptance = float(np.max(polynomial(candidates)))
    # Warm compilation on a single original source element and receiver. The
    # full source is deliberately not transported during warm-up.
    from dataclasses import replace
    d = detector
    warm_d = replace(d, positions_m=d.positions_m[:1], orientations=d.orientations[:1],
                     effective_area_m2=d.effective_area_m2[:1],
                     identifiers={k: np.asarray(v)[:1] for k, v in d.identifiers.items()})
    warm_k = replace(model, detector=warm_d).kernel(kernel.config)
    warm_s = source.subset(np.arange(len(source)) == int(np.argmax(source.beta)))
    warm_start = perf_counter()
    preview_ballistic(warm_k, warm_s, args.ballistic_method)
    screen_points(atlas, np.asarray([[0., 0., 0.]]), np.asarray([[0., 1., 0.]]),
                  np.asarray([1.]), np.asarray([1.]), np.asarray([[10., 10., 0.]]))
    if args.primary_segments:
        screen_primary_segments(atlas, warm_s, detector.positions_m[:1])
    warm_seconds = perf_counter()-warm_start
    start, cpu = perf_counter(), process_time()
    stage = perf_counter()
    point_source = source.subset(source.uid != 0) if args.primary_segments else source
    groups = group_elements(point_source, args.cell_m, args.direction_step)
    grouping_seconds = perf_counter()-stage
    stage = perf_counter()
    q0 = preview_ballistic(kernel, source, args.ballistic_method)
    ballistic_seconds = perf_counter()-stage
    stage = perf_counter()
    result = query_groups(atlas, groups, detector.positions_m)
    map_seconds = perf_counter()-stage
    segment_details = {}
    segment_seconds = 0.
    if args.primary_segments:
        stage = perf_counter()
        qs, es, os, segment_details = screen_primary_segments(
            atlas, source.subset(source.uid == 0), detector.positions_m)
        segment_seconds = perf_counter()-stage
        result['charge'] += qs
        result['sigma_sum'] += es
        result['outside_pairs'] += os
    factor = detector.effective_area_m2*peak_acceptance
    q_scattered = result['charge']*factor[:, None]
    result['sigma_sum_pe'] = result.pop('sigma_sum')*factor[:, None]
    result['beta_clipped_charge_pe'] = result.pop('beta_clipped_charge')*factor[:, None]
    proxy = np.column_stack((q0, q_scattered))
    wall, cpu_seconds = perf_counter()-start, process_time()-cpu
    threshold = .01
    masks = {str(m): q0+m*q_scattered.sum(axis=1) >= threshold for m in (1., 2., 4.)}
    # No fine result is available to grouping, query, uncertainty or selection.
    referencepath = paths.muon_response(args.energy)
    with np.load(referencepath) as z:
        reference, active = z['charge_components_pe'].copy(), z['active'].copy()
    screens = {m: dict(candidates=int(mask.sum()),
                      missed_reference_active=np.flatnonzero(active & ~mask).tolist(),
                      extra_vs_reference=int(np.sum(mask & ~active))) for m, mask in masks.items()}
    report = dict(energy_GeV=args.energy, source_elements=len(source), source_groups=len(groups.points),
                  ballistic_method=args.ballistic_method,
                  cell_m=args.cell_m, direction_step=args.direction_step,
                  primary_segments=args.primary_segments, segment_seconds=segment_seconds,
                  segment_details=segment_details,
                  beta_range=[float(groups.beta.min()), float(groups.beta.max())],
                  original_ballistic_seconds=ballistic_seconds, grouping_seconds=grouping_seconds,
                  map_seconds=map_seconds, elapsed_seconds=wall, cpu_seconds=cpu_seconds,
                  warm_seconds=warm_seconds, peak_acceptance=peak_acceptance,
                  source_sha256=hashlib.sha256(sourcepath.read_bytes()).hexdigest(),
                  atlas_sha256=hashlib.sha256(args.atlas.read_bytes()).hexdigest(),
                  reference_sha256=hashlib.sha256(referencepath.read_bytes()).hexdigest(),
                  reference_active=int(active.sum()), threshold_pe=threshold, screens=screens,
                  maximum_ballistic_reference_difference_pe=float(np.max(np.abs(q0-reference[:, 0]))),
                  independent_selection=True, fine_reference_loaded_after_preview=True,
                  inactivity_certified=False, note='Maximum acceptance; finite map and grouped sources remain approximations.')
    for key, value in result.items():
        if key != 'charge':
            a = np.asarray(value)
            report[key+'_summary'] = dict(shape=list(a.shape), minimum=float(np.min(a)),
                                         maximum=float(np.max(a)), nonzero=int(np.count_nonzero(a)))
    np.savez_compressed(output/'preview.npz', charge=proxy, reference_charge=reference,
                        reference_active=active, positions=detector.positions_m,
                        **{k: v for k, v in result.items() if k != 'charge'})
    (output/'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
