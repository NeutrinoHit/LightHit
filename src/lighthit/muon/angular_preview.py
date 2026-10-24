"""Arrival-direction moments for the spectral cone-fluence preview.

An axisymmetric source and a reciprocal isotropic medium are mirror-symmetric
about the plane through the source axis and receiver. For cubic OM acceptance,
ten moments in the local radial/axial coordinates give the angular response
of the mirror-symmetrized photon ensemble exactly. The scalar fluence remains
the selection score at maximum acceptance; these moments supply output charges.
"""
import numpy as np
from numba import njit
from .positive_fluence import _bracket

# 1, x, z, x^2, xz, z^2, x^3, x^2 z, x z^2, z^3.
MOMENT_COUNT = 10


@njit(cache=True)
def endpoint_directions(count, order, seed, g=.9, proposal_p=.5):
    """Same path/RNG convention as universal_endpoints, retaining final rays."""
    np.random.seed(seed)
    xyz = np.zeros((count, 3))
    directions = np.empty((count, 3))
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
        directions[i, 0], directions[i, 1], directions[i, 2] = sx, sy, sz
    return xyz, orders, directions


@njit(cache=True)
def angular_histogram(xyz, directions, weights, cone_cos, scale, rho_edges, z_edges):
    """Rotate paths and final rays together, then bin in the meridian frame."""
    out = np.zeros((len(rho_edges)-1, len(z_edges)-1, MOMENT_COUNT))
    squares = np.zeros(out.shape[:2])
    sine = np.sqrt(max(0., 1-cone_cos*cone_cos))
    for i in range(len(xyz)):
        x = (cone_cos*xyz[i, 0]+sine*xyz[i, 2])*scale
        y = xyz[i, 1]*scale
        z = (-sine*xyz[i, 0]+cone_cos*xyz[i, 2])*scale
        rho = np.hypot(x, y)
        ir = np.searchsorted(rho_edges, rho)-1
        iz = np.searchsorted(z_edges, z)-1
        if 0 <= ir < out.shape[0] and 0 <= iz < out.shape[1]:
            dx = cone_cos*directions[i, 0]+sine*directions[i, 2]
            dy = directions[i, 1]
            dz = -sine*directions[i, 0]+cone_cos*directions[i, 2]
            dr = (x*dx+y*dy)/rho if rho > 0. else dx
            w = weights[i]
            row = out[ir, iz]
            row[0] += w
            row[1] += w*dr
            row[2] += w*dz
            row[3] += w*dr*dr
            row[4] += w*dr*dz
            row[5] += w*dz*dz
            row[6] += w*dr*dr*dr
            row[7] += w*dr*dr*dz
            row[8] += w*dr*dz*dz
            row[9] += w*dz*dz*dz
            squares[ir, iz] += w*w
    return out, squares


@njit(cache=True, inline='always')
def cubic_response(row, radial, axial, azimuthal_squared, polynomial):
    """Contract the ten stored moments with A(n dot photon direction)."""
    m0, x, z, xx, xz, zz, xxx, xxz, xzz, zzz = row
    first = radial*x+axial*z
    second = (radial*radial*xx+2*radial*axial*xz+axial*axial*zz
              +azimuthal_squared*(m0-xx-zz))
    third = (radial**3*xxx+3*radial*radial*axial*xxz
             +3*radial*axial*axial*xzz+axial**3*zzz
             +3*azimuthal_squared*(radial*(x-xxx-xzz)+axial*(z-xxz-zzz)))
    return polynomial[0]*m0+polynomial[1]*first+polynomial[2]*second+polynomial[3]*third


@njit(cache=True)
def angular_points(points, directions, coefficient0, beta, receivers, looks, areas,
                   rho_nodes, z_edges, beta_nodes, moments, polynomial):
    """Charge of grouped sources with the receiver's actual orientation."""
    result = np.zeros((len(receivers), 2))
    for j in range(len(receivers)):
        for i in range(len(points)):
            dx = receivers[j, 0]-points[i, 0]
            dy = receivers[j, 1]-points[i, 1]
            dz = receivers[j, 2]-points[i, 2]
            ux, uy, uz = directions[i]
            z = dx*ux+dy*uy+dz*uz
            px, py, pz = dx-z*ux, dy-z*uy, dz-z*uz
            rho = np.sqrt(px*px+py*py+pz*pz)
            if rho > rho_nodes[-1] or z < z_edges[0] or z >= z_edges[-1]:
                continue
            nz = looks[j, 0]*ux+looks[j, 1]*uy+looks[j, 2]*uz
            nr = (looks[j, 0]*px+looks[j, 1]*py+looks[j, 2]*pz)/rho if rho > 0. else 0.
            ny2 = max(1.-nr*nr-nz*nz, 0.)
            ir, u = _bracket(rho_nodes, rho)
            ib, v = _bracket(beta_nodes, beta[i])
            iz = min(np.searchsorted(z_edges, z, side='right')-1, len(z_edges)-2)
            for db in range(2):
                for dr in range(2):
                    weight = (v if db else 1-v)*(u if dr else 1-u)*coefficient0[i]*areas[j]
                    for order in range(2):
                        row = moments[ib+db, order, ir+dr, iz]
                        if rho == 0.:
                            # Exact axial symmetry removes the arbitrary meridian.
                            axial_row = row.copy()
                            axial_row[1], axial_row[4], axial_row[6], axial_row[8] = 0., 0., 0., 0.
                            axial_row[3] = .5*(row[0]-row[5])
                            axial_row[7] = .5*(row[2]-row[9])
                            value = cubic_response(axial_row, nr, nz, ny2, polynomial)
                        else:
                            value = cubic_response(row, nr, nz, ny2, polynomial)
                        if value < -1e-12*max(row[0], 1e-300):
                            raise ValueError('negative angular fluence response')
                        result[j, order] += weight*max(value, 0.)
    return result


def query_angular_groups(atlas, groups, detector, alpha):
    """Apply the detector acceptance polynomial after local-frame projection."""
    alpha = np.asarray(alpha, float)
    coefficient = np.polynomial.Legendre(alpha*(2*np.arange(len(alpha))+1)/(4*np.pi)).convert(
        kind=np.polynomial.Polynomial).coef
    if len(coefficient) > 4:
        raise ValueError('angular preview moments support acceptance degree <= 3')
    polynomial = np.zeros(4)
    polynomial[:len(coefficient)] = coefficient
    return angular_points(groups.points, groups.directions, groups.coefficient0, groups.beta,
                          np.ascontiguousarray(detector.positions_m),
                          np.ascontiguousarray(-detector.orientations),
                          np.ascontiguousarray(detector.effective_area_m2),
                          atlas.rho_nodes, atlas.z_edges_m, atlas.beta_nodes,
                          atlas.angular_moments, polynomial)
