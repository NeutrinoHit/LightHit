"""Positive stationary cone fluence maps used by the OM preview.

Maps include the spectral yield per C0, the wavelength quadrature and QE, but
not the OM area and acceptance. Cells are Monte Carlo estimates: a zero cell or
a small spread does not certify darkness.
"""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from numba import njit


@njit(cache=True)
def universal_endpoints(count, order, seed, g=.9, proposal_p=.5):
    """Exp(1) flights, fixed n=1 or geometric n>=2; initial direction +z."""
    np.random.seed(seed)
    xyz = np.zeros((count, 3))
    orders = np.empty(count, np.int32)
    for i in range(count):
        n = 1 if order == 1 else 1 + np.random.geometric(1-proposal_p)
        orders[i] = n
        sx, sy, sz = 0., 0., 1.
        for j in range(n+1):
            length = -np.log1p(-np.random.random())
            xyz[i, 0] += length*sx
            xyz[i, 1] += length*sy
            xyz[i, 2] += length*sz
            if j == n:
                break
            u = np.random.random()
            c = 2*u-1 if abs(g) < 1e-12 else (1+g*g-((1-g*g)/(1-g+2*g*u))**2)/(2*g)
            sn = np.sqrt(max(0., 1-c*c))
            phi = 2*np.pi*np.random.random()
            if abs(sz) < .9:
                ex, ey, ez = -sy, sx, 0.
            else:
                ex, ey, ez = 0., -sz, sy
            norm = np.sqrt(ex*ex+ey*ey+ez*ez)
            ex, ey, ez = ex/norm, ey/norm, ez/norm
            bx, by, bz = sy*ez-sz*ey, sz*ex-sx*ez, sx*ey-sy*ex
            cp, sp = np.cos(phi), np.sin(phi)
            sx, sy, sz = (c*sx+sn*(cp*ex+sp*bx), c*sy+sn*(cp*ey+sp*by),
                          c*sz+sn*(cp*ez+sp*bz))
    return xyz, orders


@njit(cache=True)
def endpoint_histogram(xyz, weights, cone_cos, scale, rho_edges, z_edges):
    """Positive volume counts; rotation is equivalent to uniform cone average."""
    out = np.zeros((len(rho_edges)-1, len(z_edges)-1))
    squares = np.zeros_like(out)
    s = np.sqrt(max(0., 1-cone_cos*cone_cos))
    for i in range(len(xyz)):
        x = (cone_cos*xyz[i, 0]+s*xyz[i, 2])*scale
        y = xyz[i, 1]*scale
        z = (-s*xyz[i, 0]+cone_cos*xyz[i, 2])*scale
        rho = np.hypot(x, y)
        ir = np.searchsorted(rho_edges, rho)-1
        iz = np.searchsorted(z_edges, z)-1
        if 0 <= ir < len(out) and 0 <= iz < out.shape[1]:
            w = weights[i]
            out[ir, iz] += w
            squares[ir, iz] += w*w
    return out, squares


def prefixes(density, z_edges):
    """Cumulative z integrals of the density from the left and from the right."""
    mass = density*np.diff(z_edges)
    zero = np.zeros(density.shape[:-1]+(1,))
    left = np.concatenate((zero, np.cumsum(mass, axis=-1)), axis=-1)
    right = np.concatenate((np.cumsum(mass[..., ::-1], axis=-1)[..., ::-1], zero), axis=-1)
    return left, right


@njit(cache=True)
def _bracket(axis, x):
    """Index and linear weight of ``x`` on an increasing axis, clamped to its ends."""
    if x <= axis[0]:
        return 0, 0.
    if x >= axis[-1]:
        return len(axis)-2, 1.
    i = np.searchsorted(axis, x)-1
    return i, (x-axis[i])/(axis[i+1]-axis[i])


@njit(cache=True)
def _prefix(z, edges, values):
    """Prefix value at ``z`` by linear interpolation between edges."""
    if z <= edges[0]:
        return values[0]
    if z >= edges[-1]:
        return values[-1]
    j = np.searchsorted(edges, z)-1
    t = (z-edges[j])/(edges[j+1]-edges[j])
    return (1-t)*values[j]+t*values[j+1]


@njit(cache=True)
def segment_queries(rho, zlo, zhi, beta, rho_nodes, z_edges, beta_nodes, left, right):
    """Integral of interpolated density over [zlo,zhi], positive, shape (N,2).

    A physical segment [a,b] seen at longitudinal z uses [z-b,z-a].
    Out-of-domain intervals are clipped; callers MUST handle the returned flag.
    """
    result = np.zeros((len(rho), 2))
    outside = np.zeros(len(rho), np.bool_)
    for q in range(len(rho)):
        outside[q] = (rho[q] < 0 or rho[q] > rho_nodes[-1]
                      or zlo[q] < z_edges[0] or zhi[q] > z_edges[-1]
                      or beta[q] < beta_nodes[0] or beta[q] > beta_nodes[-1])
        if rho[q] < 0 or rho[q] > rho_nodes[-1] or zhi[q] <= zlo[q]:
            continue
        ir, u = _bracket(rho_nodes, rho[q])
        ib, v = _bracket(beta_nodes, beta[q])
        for db in range(2):
            for dr in range(2):
                w = (v if db else 1-v)*(u if dr else 1-u)
                for k in range(2):
                    lp = left[ib+db, k, ir+dr]
                    rp = right[ib+db, k, ir+dr]
                    upper_left = _prefix(zhi[q], z_edges, lp)
                    upper_right = _prefix(zlo[q], z_edges, rp)
                    if upper_right < upper_left:
                        value = upper_right-_prefix(zhi[q], z_edges, rp)
                    else:
                        value = upper_left-_prefix(zlo[q], z_edges, lp)
                    result[q, k] += w*max(0., value)
    return result, outside


@njit(cache=True)
def point_queries(rho, z, beta, rho_nodes, z_edges, beta_nodes, density):
    """Linear rho/beta interpolation, histogram cell value in z."""
    result = np.zeros((len(rho), 2))
    outside = np.zeros(len(rho), np.bool_)
    for q in range(len(rho)):
        outside[q] = (rho[q] < 0 or rho[q] > rho_nodes[-1]
                      or z[q] < z_edges[0] or z[q] >= z_edges[-1]
                      or beta[q] < beta_nodes[0] or beta[q] > beta_nodes[-1])
        if rho[q] < 0 or rho[q] > rho_nodes[-1] or z[q] < z_edges[0] or z[q] >= z_edges[-1]:
            continue
        ir, u = _bracket(rho_nodes, rho[q])
        ib, v = _bracket(beta_nodes, beta[q])
        iz = min(np.searchsorted(z_edges, z[q], side='right')-1, len(z_edges)-2)
        for db in range(2):
            for dr in range(2):
                w = (v if db else 1-v)*(u if dr else 1-u)
                for k in range(2):
                    result[q, k] += w*density[ib+db, k, ir+dr, iz]
    return result, outside


@dataclass
class PositiveFluenceAtlas:
    """Positive cone fluence per C0 in (beta, order, rho, z) cells with sampling spread."""
    rho_edges_m: np.ndarray
    z_edges_m: np.ndarray
    beta_nodes: np.ndarray
    density: np.ndarray
    sigma: np.ndarray
    metadata: dict

    def __post_init__(self):
        """Check axes and arrays; precompute rho nodes and z prefixes."""
        if (len(self.rho_edges_m) < 3 or len(self.z_edges_m) < 2
                or not all(np.isfinite(x).all() for x in (self.rho_edges_m, self.z_edges_m, self.beta_nodes))
                or np.any(np.diff(self.rho_edges_m) <= 0) or self.rho_edges_m[0] != 0
                or np.any(np.diff(self.z_edges_m) <= 0)
                or len(self.beta_nodes) < 2 or np.any(np.diff(self.beta_nodes) <= 0)):
            raise ValueError('invalid atlas axes')
        expected = (len(self.beta_nodes), 2, len(self.rho_edges_m)-1, len(self.z_edges_m)-1)
        if (self.density.shape != expected or self.sigma.shape != expected
                or not np.isfinite(self.density).all() or not np.isfinite(self.sigma).all()
                or np.any(self.density < 0) or np.any(self.sigma < 0)):
            raise ValueError('invalid positive atlas values')
        self.rho_nodes = np.sqrt((self.rho_edges_m[:-1]**2+self.rho_edges_m[1:]**2)/2)
        self.left, self.right = prefixes(self.density, self.z_edges_m)

    @classmethod
    def load(cls, path):
        """Read an atlas written by :meth:`save`."""
        with np.load(path, allow_pickle=False) as data:
            return cls(*(data[k].copy() for k in ('rho_edges_m', 'z_edges_m', 'beta_nodes', 'density', 'sigma')),
                       json.loads(str(data['metadata'])))

    def save(self, path):
        """Write the atlas; an existing file is never overwritten."""
        path = Path(path)
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, rho_edges_m=self.rho_edges_m, z_edges_m=self.z_edges_m,
                            beta_nodes=self.beta_nodes, density=self.density, sigma=self.sigma,
                            metadata=json.dumps(self.metadata))

    def segments(self, rho, zlo, zhi, beta):
        """Fluence of straight pieces from z prefixes (orders 1 and >=2)."""
        values = [np.atleast_1d(np.asarray(x, float)) for x in (rho, zlo, zhi, beta)]
        values = [np.ascontiguousarray(x.ravel()) for x in np.broadcast_arrays(*values)]
        if not all(np.isfinite(x).all() for x in values) or np.any(values[2] < values[1]):
            raise ValueError('finite coordinates and zhi>=zlo required')
        return segment_queries(*values, self.rho_nodes, self.z_edges_m, self.beta_nodes, self.left, self.right)

    def points(self, rho, z, beta):
        """Fluence of point sources (orders 1 and >=2)."""
        values = [np.atleast_1d(np.asarray(x, float)) for x in (rho, z, beta)]
        values = [np.ascontiguousarray(x.ravel()) for x in np.broadcast_arrays(*values)]
        if not all(np.isfinite(x).all() for x in values):
            raise ValueError('finite coordinates required')
        return point_queries(*values, self.rho_nodes, self.z_edges_m, self.beta_nodes, self.density)
