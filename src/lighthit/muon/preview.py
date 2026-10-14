"""Cheap scattered-light preview used to choose OMs for the full muon solve.

The source is reduced to positive groups (cell, direction, beta class) and each
group is looked up in a precomputed positive fluence atlas. The result is an
estimate with a sampling spread, not a bound: OMs are selected from it with a
margin, and uncertain ones are forced into the detailed calculation.
"""
from dataclasses import dataclass

import numpy as np
from numba import njit

from .positive_fluence import PositiveFluenceAtlas, _bracket


@dataclass
class Groups:
    """Source groups: C0-weighted position and direction, total C0, beta."""
    points: np.ndarray
    directions: np.ndarray
    coefficient0: np.ndarray
    beta: np.ndarray
    radius: np.ndarray
    primary: np.ndarray
    counts: np.ndarray


@dataclass
class Preview:
    """Per-OM preview charge (Q0 column left zero) and forced-OM mask."""
    charge_components_pe: np.ndarray
    charge_pe: np.ndarray
    unsafe_mask: np.ndarray
    sigma_pe: np.ndarray
    metadata: dict


def group_elements(source, *, primary_uid=0, cell_m=4., direction_step=.5,
                   beta_edges=(.76, .8, .9, .97, 1.)):
    """Merge elements into groups keeping the summed C0 and C2 exactly.

    Elements are keyed by spatial cell, direction bin, beta class and whether
    they belong to the primary. Group beta follows from the summed C0 and C2,
    so the Cherenkov yield of each group is that of its members.
    """
    if cell_m <= 0 or direction_step <= 0:
        raise ValueError('cell and direction sizes must be positive')
    c0, c2 = np.asarray(source.coefficient0), np.asarray(source.coefficient2)
    if not np.allclose(c2, -c0/source.beta**2, rtol=1e-12, atol=0):
        raise ValueError('atlas assumes Cherenkov coefficient2=-coefficient0/beta²')
    primary = np.asarray(source.uid) == primary_uid
    midpoint = source.midpoints_m
    keys = np.column_stack((np.floor(midpoint/cell_m).astype(np.int64),
                            np.floor(source.direction/direction_step).astype(np.int64),
                            np.searchsorted(beta_edges, source.beta), primary))
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    count = int(inverse.max())+1

    def reduce(values):
        """Sum of ``values`` per group."""
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
                  reduce(c0*primary) > 0, np.bincount(inverse, minlength=count))


@njit(cache=True)
def _screen_points(points, directions, coefficient0, beta, receivers,
                   rho_nodes, z_edges, beta_nodes, density, sigma):
    """Bilinear (rho, beta) atlas lookup summed over groups for every OM."""
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
    """Atlas estimate per OM: Q1 and Q>=2 columns, spread and diagnostics."""
    arrays = _screen_points(groups.points, groups.directions, groups.coefficient0,
                            groups.beta, np.ascontiguousarray(receivers),
                            atlas.rho_nodes, atlas.z_edges_m, atlas.beta_nodes,
                            atlas.density, atlas.sigma)
    names = ('charge', 'sigma_sum', 'outside_pairs', 'empty_pairs',
             'beta_clipped_charge', 'nearest_source_m')
    return dict(zip(names, arrays))


def acceptance_maximum(kernel):
    """Largest value of the OM angular acceptance over all directions."""
    alpha = np.asarray(kernel.acceptance()['alpha'])
    polynomial = np.polynomial.Legendre(alpha*(2*np.arange(len(alpha))+1)/(4*np.pi))
    roots = polynomial.deriv().roots()
    candidates = [-1., 1.]+[float(r.real) for r in roots
                            if abs(r.imag) < 1e-12 and -1 <= r.real <= 1]
    return float(np.max(polynomial(candidates)))


def check_atlas_optics(atlas, kernel, rtol=1e-9):
    """Refuse an atlas built for other water optics or OM spectral response."""
    from ..prompt import _spectral_tables
    stored = atlas.metadata.get('tables')
    if not stored:
        raise ValueError('preview atlas carries no optical tables')
    tables = _spectral_tables(kernel)
    for key in ('wavelength_nm', 'mus', 'mut', 'phase_index', 's0w', 's2w'):
        if not np.allclose(np.asarray(stored[key], float), tables[key], rtol=rtol, atol=0):
            raise ValueError(f'preview atlas optics differ from the kernel ({key})')


def independent_preview(kernel, source, atlas, *, primary_uid=0):
    """Preview for ``transport_joint_muon``; no event response is loaded.

    The Q0 column stays zero: the transport replaces it by the exact direct
    light. An OM is forced into the detailed set when a source group is within
    3 m, or when the atlas spread is a large part of an estimate that can reach
    1/8 of the activity threshold.
    """
    if not isinstance(atlas, PositiveFluenceAtlas):
        atlas = PositiveFluenceAtlas.load(atlas)
    groups = group_elements(source, primary_uid=primary_uid)
    estimate = query_groups(atlas, groups, kernel.detector.positions_m)
    factors = np.asarray(kernel.detector.effective_area_m2)*acceptance_maximum(kernel)
    scattered = estimate['charge']*factors[:, None]
    sigma = estimate['sigma_sum']*factors[:, None]
    direct = np.zeros(len(kernel.detector.positions_m))
    coarse = np.column_stack((direct, scattered))
    threshold = float(kernel.config.threshold_pe)
    forced = (estimate['nearest_source_m'] < 3) | (
        (sigma.sum(1) > .25*scattered.sum(1))
        & (direct+scattered.sum(1)+4*sigma.sum(1) >= threshold/8))
    return Preview(coarse, coarse.sum(1), forced, sigma,
                   dict(independent=True, inactivity_certified=False,
                        fine_event_data_loaded=False, groups=len(groups.points)))
