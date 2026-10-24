"""Geometry-only first-entry profiles for building finite-OM response tables.

These profiles terminate the photon at its first glass intersection. They
are table-building primitives; the event transport routes do not use them
until their corresponding tables and interpolation contracts are supplied.
"""
from dataclasses import dataclass

import numpy as np

from .receiver import SphereSurface


MOMENT_POWERS = ((0, 0), (1, 0), (0, 1), (2, 0), (1, 1),
                (0, 2), (3, 0), (2, 1), (1, 2), (0, 3))


@dataclass(frozen=True)
class FirstEntryProfile:
    """Cumulative moments of the arrival direction x^i*z^j, in m^-1.

    The source pencil is the local z axis, and x points towards the receiver.
    To obtain a response, fold increments with mu_s*exp(-mu_t*S), spectral
    yield and collection area. The clock is t_emit + n_group*S/c, with no
    additional shift of the optical path. Detector transit time is separate.
    """
    path_m: np.ndarray
    cumulative_m_inv: np.ndarray
    receiver_radius_m: float
    distance_m: float
    direction_cosine: float
    g: float


def first_entry_profile(surface, *, distance_m, direction_cosine, g,
                        max_path_m, receiver=0, path_nodes=256,
                        path_order=8, scatter_order=16):
    """Build one geometry row using the supplied detector surface.

    ``max_path_m`` is an explicit table boundary, not a physical cutoff.
    The caller owns the spectral tail bound. The grid splits at the direct
    entry and tangent paths, including the internal logarithmic singularity.
    Angles whose initial pencil hits the OM stop their first flight there.
    """
    if not isinstance(surface, SphereSurface):
        raise ValueError('first-entry profiles require SphereSurface geometry')
    radius = float(surface.radius_m if surface.radius_m.ndim == 0
                   else surface.radius_m[receiver])
    if not np.isfinite(distance_m) or distance_m <= radius:
        raise ValueError('the source must be outside the receiver surface')
    if not np.isfinite(direction_cosine) or abs(direction_cosine) > 1:
        raise ValueError('direction_cosine must be finite and in [-1, 1]')
    if not np.isfinite(g) or abs(g) >= 1:
        raise ValueError('g must be finite and in (-1, 1)')
    if not np.isfinite(max_path_m) or max_path_m <= distance_m-radius:
        raise ValueError('max_path_m must exceed the glass path front')
    for name, value in (('path_nodes', path_nodes), ('path_order', path_order),
                        ('scatter_order', scatter_order)):
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer)) or value < 2):
            raise ValueError(f'{name} must be an integer of at least 2')
    r = distance_m/radius
    theta = np.arccos(direction_cosine)
    rho = r*np.sqrt((1-direction_cosine)*(1+direction_cosine))
    front = r-1
    tangent = np.sqrt((r-1)*(r+1))
    maximum = max_path_m/radius
    hit = direction_cosine > 0 and rho <= 1
    x = np.linspace(0., 1., path_nodes+1)
    cuts = [front]
    if hit:
        entry = r*direction_cosine-np.sqrt(max(0., 1-rho*rho))
        cuts.append(entry)
    cuts.extend((tangent, maximum))
    cuts = np.unique(np.clip(cuts, front, maximum))
    paths = []
    # The HG width only grades the numerical mesh; it changes no path weight.
    ag = abs(g)
    sigma = 1+r*(1-np.cos(theta)+(1-ag)/(1+ag)*np.sin(theta))
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        if lo >= tangent and hit:
            row = np.array([lo, hi])
        elif lo >= tangent:
            row = lo+sigma*np.sinh(np.arcsinh((hi-lo)/sigma)*x)
        else:
            row = lo+(hi-lo)*np.sin(np.pi*x/2)**2
        paths.append(row)
    paths = np.unique(np.concatenate(paths))
    from ._receiver_first_numba import cumulative
    values = cumulative(r, direction_cosine, paths, g,
                        *np.polynomial.legendre.leggauss(path_order),
                        *np.polynomial.legendre.leggauss(scatter_order))
    return FirstEntryProfile(paths*radius, values/radius, radius,
                             float(distance_m), float(direction_cosine), float(g))
