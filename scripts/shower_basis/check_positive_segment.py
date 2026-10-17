#!/usr/bin/env python3
"""Independent finite-track MC versus cell-integrated positive primitives.

Fixed 480 nm, beta=1, scalar fluence only; no OM acceptance or full-event claim.
Training and validation use independent seeds. Reported sampling floor includes
both samples (Poisson approximation); radial interpolation bias is exposed by
grid refinement. This is deliberately a volume integral, never bin-centre truth.
"""
import paths  # inputs and outputs: scripts/shower_basis/paths.py
import argparse
import json
from pathlib import Path
from time import process_time

import numpy as np
from numpy.polynomial.legendre import leggauss

from lighthit.experimental.positive_fluence import (
    PositiveFluenceAtlas, endpoint_histogram, universal_endpoints)


def radial_integration(nodes, edges):
    """Exact integrals 2*pi*r of the atlas's linear radial basis."""
    result = np.zeros((len(edges)-1, len(nodes)))
    if edges[-1] > nodes[-1]:
        raise ValueError('validation volume must lie inside node domain')
    for row, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        splits = np.unique(np.r_[lo, nodes[(nodes > lo) & (nodes < hi)], hi])
        for a, b in zip(splits[:-1], splits[1:]):
            if b <= nodes[0]:
                result[row, 0] += np.pi*(b*b-a*a)
            else:
                j = np.searchsorted(nodes, (a+b)/2)-1
                area = np.pi*(b*b-a*a)
                upper = 2*np.pi*((b**3-a**3)/3-nodes[j]*(b*b-a*a)/2)/(nodes[j+1]-nodes[j])
                result[row, j] += area-upper
                result[row, j+1] += upper
    return result


def longitudinal_integration(source_edges, receiver_edges, length):
    """Exact convolution of source histogram cells, uniform track and bins."""
    a, b = -length/2, length/2
    e0, e1 = source_edges[:-1], source_edges[1:]

    def cumulative(z):
        h = lambda x: .5*np.maximum(x, 0.)**2
        return h(z-a-e0)-h(z-b-e0)-h(z-a-e1)+h(z-b-e1)

    return np.maximum(0., np.diff(np.array([cumulative(z) for z in receiver_edges]), axis=0))/length


def primitive_volume_quadrature(atlas, rho_bin, z_bin, length):
    """Independent split Gauss integral using actual public segment queries."""
    r0, r1 = rho_bin
    z0, z1 = z_bin
    rs = np.unique(np.r_[r0, atlas.rho_nodes[(atlas.rho_nodes > r0) & (atlas.rho_nodes < r1)], r1])
    breaks = np.r_[atlas.z_edges_m-length/2, atlas.z_edges_m+length/2]
    zs = np.unique(np.r_[z0, breaks[(breaks > z0) & (breaks < z1)], z1])
    x, w = leggauss(3)
    r = ((rs[:-1, None]+rs[1:, None])/2+np.diff(rs)[:, None]/2*x).ravel()
    rw = (np.diff(rs)[:, None]/2*w).ravel()
    z = ((zs[:-1, None]+zs[1:, None])/2+np.diff(zs)[:, None]/2*x).ravel()
    zw = (np.diff(zs)[:, None]/2*w).ravel()
    values, flags = atlas.segments(r[:, None], z[None, :]-length/2, z[None, :]+length/2, 1.)
    if np.any(flags):
        raise ValueError('quadrature unexpectedly left atlas')
    values = values[:, 0].reshape(len(r), len(z))
    return float((2*np.pi*r*rw) @ values @ zw/length)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--count', type=int, default=1_000_000)
    parser.add_argument('--output', type=Path, default=paths.work('positive-map-20261009/validation'))
    args = parser.parse_args()
    if args.count < 1000:
        parser.error('count must be at least 1000')
    args.output.mkdir(parents=True, exist_ok=True)
    mua, mus, phase = .0545, .0151, 1.339487509
    mut = mua+mus
    p = mus/mut
    cone = 1/phase
    length = 60.
    # Enough range for real backward light; tail cells remain statistically sparse.
    receiver_r = np.array([0., 2., 5., 10., 20., 40., 80., 160.])
    receiver_z = np.array([-120., -60., -30., 0., 30., 60., 120., 240.])
    universal_endpoints(2, 1, 1)
    universal_endpoints(2, 2, 1)
    start = process_time()
    report = dict(count_per_order_per_sample=args.count, independent_samples_per_order=2,
                  mua=mua, mus=mus, phase_index=phase, beta=1., g=.9,
                  segment_length_m=length, scalar_only=True, units='volume integral of F in metres',
                  statistical_floor='Poisson variance from both independent weighted histograms; approximate',
                  results=[])
    saved = dict(receiver_r_edges_m=receiver_r, receiver_z_edges_m=receiver_z)
    for order in (1, 2):
        seed = 90261200+10*order
        mass = p/mut if order == 1 else p*p/mua
        xyz, _ = universal_endpoints(args.count, order, seed, .9, p)
        ref, _ = universal_endpoints(args.count, order, seed+1, .9, p)
        # Independent source position sampled in the physical track frame.
        c, s = cone, np.sqrt(1-cone*cone)
        ref = ref @ np.array([[c, 0., -s], [0., 1., 0.], [s, 0., c]])/mut
        ref[:, 2] += np.random.default_rng(seed+2).uniform(-length/2, length/2, args.count)
        unit = np.ones(args.count)
        ref_counts, _ = endpoint_histogram(ref, unit, 1., 1., receiver_r, receiver_z)
        ref_mass = ref_counts*mass/args.count
        ref_variance = ref_counts*(mass/args.count)**2
        saved[f'order{order}_independent_mass'] = ref_mass
        for nr, nz in ((48, 144), (96, 288), (192, 576)):
            rho_edges = 180*np.linspace(0., 1., nr+1)**2
            z_edges = np.linspace(-360., 360., nz+1)
            counts, _ = endpoint_histogram(xyz, unit, cone, 1/mut, rho_edges, z_edges)
            volume = np.pi*np.diff(rho_edges**2)[:, None]*np.diff(z_edges)[None, :]
            field = counts*mass/(args.count*volume)
            density = np.tile(field[None, None, :, :], (2, 2, 1, 1))
            atlas = PositiveFluenceAtlas(rho_edges, z_edges, np.array([.99, 1.]), density,
                                        np.zeros_like(density), dict(monochromatic=True))
            radial = radial_integration(atlas.rho_nodes, receiver_r)
            longitudinal = longitudinal_integration(z_edges, receiver_z, length)
            predicted = radial @ field @ longitudinal.T
            # Linear projection from training cell counts propagates their variance.
            predicted_variance = (radial**2) @ (counts*(mass/(args.count*volume))**2) @ (longitudinal**2).T
            sigma = np.sqrt(predicted_variance+ref_variance)
            mask = ref_counts >= 25
            errors = predicted-ref_mass
            zscore = np.divide(errors, sigma, out=np.zeros_like(errors), where=sigma > 0)
            quadrature_error = []
            for i, j in ((0, 2), (3, 3), (5, 5)):
                quad = primitive_volume_quadrature(atlas, receiver_r[i:i+2], receiver_z[j:j+2], length)
                quadrature_error.append(abs(quad-predicted[i, j])/max(abs(quad), 1e-30))
            item = dict(order='1' if order == 1 else '>=2', grid=[nr, nz], seed_training=seed,
                        seed_validation=seed+1, mass_exact_full_space_m=mass,
                        mass_reference_volume_m=float(ref_mass.sum()), mass_predicted_volume_m=float(predicted.sum()),
                        l1_over_full_mass=float(np.abs(errors).sum()/mass),
                        expected_sampling_l1_over_full_mass=float(np.sqrt(2/np.pi)*sigma.sum()/mass),
                        populated_cells=int(mask.sum()), fraction_populated_over_3sigma=float(np.mean(abs(zscore[mask]) > 3)),
                        max_abs_zscore_populated=float(np.max(abs(zscore[mask]))),
                        primitive_vs_exact_cell_integral_max_relative=float(max(quadrature_error)),
                        caution='Finite grid and statistical tails remain; no local accuracy certificate.')
            report['results'].append(item)
            saved[f'order{order}_grid{nr}_predicted_mass'] = predicted
            saved[f'order{order}_grid{nr}_combined_sigma'] = sigma
    report['cpu_seconds_excluding_initial_endpoint_jit'] = process_time()-start
    np.savez_compressed(args.output/'segment-validation.npz', **saved)
    (args.output/'segment-validation.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
