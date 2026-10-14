"""Certified omission of weak packet-receiver pairs in secondary Q1.

``packet_first_bounds`` returns, for one source packet, an all-time Q1 charge
upper bound per (band, receiver). Each element and band uses the smaller of
two independent valid bounds from ``track_first_bounds``: the finite-ray
cone-separation bound and the infinite-uniform-line bound. Taking the minimum
per band, rather than of band sums, is valid because each candidate bounds
its own band. The bounds apply to the supplied positive spectral quadrature,
not to its continuum integration error.

``select_skips`` chooses omitted packet-receiver pairs per band. A packet may be
omitted at receiver j only while the sum of omitted bounds stays below
``epsilon`` times the Q1 actually computed at j by the seed packets. Q1 of the
grouped solver is a nonnegative all-time mass, so the omitted charge at every
receiver is certified below ``epsilon`` times its computed secondary Q1.
"""
import math

import numpy as np
from numba import njit, prange


@njit(cache=True)
def _hg_angle_upper(theta, g):
    # Scalar form of track_first_bounds.hg_angle_integral_over_sine_upper.
    """Bound of the HG angular integral over sin(theta); scalar form for numba."""
    if not theta > 0.0:
        return np.inf
    backward = math.pi - theta
    if backward <= 1e-5:
        ratio = 1.0 + backward * backward / 6.0 + 7.0 * backward ** 4 / 360.0
    else:
        ratio = backward / math.sin(theta)
    v = math.sin(0.5 * theta)
    d0 = math.sqrt((1.0 - g) ** 2 + 4.0 * g * v * v)
    d1 = 1.0 + g
    norm = (1.0 - g * g) / (4.0 * math.pi)
    return norm * ratio * (1.0 + v) / (d1 * d0 * (d0 + v * d1))


@njit(cache=True)
def _yukawa_upper(rho, a, b):
    # Scalar form of track_first_bounds.yukawa_convolution_elementary_upper.
    """Bound of the 2D Yukawa convolution; scalar form for numba."""
    if not rho > 0.0:
        return np.inf
    c = 0.5 * rho * (a + b)
    d = 0.5 * rho * (a - b)
    angular = min(1.0, math.sqrt(math.pi / (8.0 * max(d, 1e-300))))
    return 2.0 * math.pi * math.exp(-rho * b) * math.sqrt(math.pi / (2.0 * c)) * angular


@njit(cache=True)
def _ray_bound(psi, rmin, mus, mut, g):
    # Per-ray all-time Q1 bound for angle psi to the receiver and distance
    # >= rmin; both candidates decrease with psi and with rmin.
    """All-time Q1 bound per photon for a ray at angle psi from the OM direction
    and distance >= rmin. Both candidates decrease with psi and rmin.
    """
    if not (psi > 1e-12 and rmin > 0.0):
        return np.inf
    distance = rmin * math.sin(min(psi, 0.5 * math.pi))
    kappa = 2.0 * math.sin(0.5 * psi) ** 2
    attenuation = math.exp(-mut * rmin)
    phase_max = (1.0 - g * g) / (4.0 * math.pi * (1.0 + g * g - 2.0 * g * math.cos(psi)) ** 1.5)
    first = mus * phase_max * attenuation / (mut * kappa * distance * distance)
    second = mus * attenuation / rmin * _hg_angle_upper(psi, g)
    return min(first, second)


@njit(cache=True)
def _azimuth_ray_bound(cone, theta_low, theta_high, rmin, mus, mut, g,
                       phi_cos, phi_weight):
    """Upper Riemann sum of the cone-azimuth average of the per-ray bound.

    Directions from the segment to the receiver share one plane through the
    track axis, at polar angles [theta_low, theta_high]. For photon azimuth
    phi the smallest angle psi_min(phi) to any of them increases with |phi| on
    [0, pi], and _ray_bound decreases with psi, so left-endpoint values on any
    partition of [0, pi] bound the mean over the uniform Cherenkov azimuth.
    """
    cc = math.cos(cone)
    sc = math.sin(cone)
    total = 0.0
    for k in range(phi_cos.shape[0]):
        cp = phi_cos[k]
        amplitude = math.sqrt(cc * cc + sc * sc * cp * cp)
        centre = math.atan2(sc * cp, cc)
        if theta_low <= centre <= theta_high:
            best = amplitude
        else:
            best = max(cc * math.cos(theta_low) + sc * cp * math.sin(theta_low),
                       cc * math.cos(theta_high) + sc * cp * math.sin(theta_high))
        psi = math.acos(min(1.0, max(-1.0, best)))
        value = _ray_bound(psi, rmin, mus, mut, g)
        if not value < np.inf:
            return np.inf
        total += phi_weight[k] * value
    return total


@njit(cache=True, parallel=True)
def _packet_bounds(start, direction, length, beta, c0, c2, receivers, areas,
                   phase, mus, mut, s0w, s2w, g, amax, pmax, phi_cos, phi_weight, out):
    """Accumulate the per-element, per-band minimum of ray and line bounds into ``out``."""
    n_band = phase.shape[0]
    for j in prange(receivers.shape[0]):
        scale = areas[j] * amax
        for e in range(start.shape[0]):
            dx = receivers[j, 0] - start[e, 0]
            dy = receivers[j, 1] - start[e, 1]
            dz = receivers[j, 2] - start[e, 2]
            ux = direction[e, 0]
            uy = direction[e, 1]
            uz = direction[e, 2]
            high = dx * ux + dy * uy + dz * uz
            low = high - length[e]
            px = dx - high * ux
            py = dy - high * uy
            pz = dz - high * uz
            rho = math.sqrt(px * px + py * py + pz * pz)
            if low > 0.0:
                nearest = low
            elif high < 0.0:
                nearest = -high
            else:
                nearest = 0.0
            rmin = math.hypot(rho, nearest)
            theta_low = math.atan2(rho, high)
            theta_high = math.atan2(rho, low)
            for b in range(n_band):
                bp = beta[e] * phase[b]
                if bp <= 1.0 or mus[b] <= 0.0:
                    continue
                photons = c0[e] * s0w[b] + c2[e] * s2w[b]
                if photons <= 0.0:
                    continue
                cone = math.acos(min(1.0, 1.0 / bp))
                separation = max(0.0, max(theta_low - cone, cone - theta_high))
                ray = photons * _ray_bound(separation, rmin, mus[b], mut[b], g)
                if phi_cos.shape[0] > 0 and ray < np.inf:
                    ray = min(ray, photons * _azimuth_ray_bound(
                        cone, theta_low, theta_high, rmin, mus[b], mut[b], g,
                        phi_cos, phi_weight))
                elif phi_cos.shape[0] > 0:
                    ray = photons * _azimuth_ray_bound(
                        cone, theta_low, theta_high, rmin, mus[b], mut[b], g,
                        phi_cos, phi_weight)
                line = np.inf
                sine = math.sqrt(max(0.0, 1.0 - 1.0 / (bp * bp)))
                if rho > 0.0 and sine > 0.0:
                    line = (photons / length[e] * mus[b] * pmax / (2.0 * sine)
                            * _yukawa_upper(rho, mut[b] / sine, mut[b]))
                out[b, j] += scale * min(ray, line)


def bound_constants(kernel):
    """Acceptance and HG maxima used by the bounds; HG 0<=g<1 required."""
    g = float(kernel.medium.g)
    if not 0 <= g < 1:
        raise ValueError('packet bounds require HG 0<=g<1')
    alpha = np.asarray(kernel.acceptance()['alpha'], float)
    amax = float(np.sum(abs(alpha) * (2 * np.arange(len(alpha)) + 1)) / (4 * np.pi))
    pmax = (1 + g) / (4 * np.pi * (1 - g) ** 2)
    return g, amax, pmax


def azimuth_partition(nodes):
    """Left endpoints (as cosines) and weights/pi of a partition of [0, pi].

    Quadratic spacing resolves the steep side near the cone. ``nodes=0``
    disables the azimuth refinement (worst-azimuth bound only).
    """
    if nodes == 0:
        return np.empty(0), np.empty(0)
    edges = np.pi * (np.arange(nodes + 1) / nodes) ** 2
    return np.cos(edges[:-1]), np.diff(edges) / np.pi


def packet_first_bounds(elements, receivers_m, areas, tables, constants, *,
                        azimuth_nodes=0):
    """All-time Q1 upper bounds, shape (bands, receivers), for one packet.

    ``tables`` are prompt._spectral_tables(kernel); every band's own Cherenkov
    threshold is applied to the supplied elements. Infinity marks pairs with
    no supported bound (cone crossing on the line through the receiver).
    """
    g, amax, pmax = constants
    if np.any(np.asarray(elements.length_m) <= 0):
        raise ValueError('positive segment lengths required')
    receivers = np.ascontiguousarray(receivers_m, float)
    out = np.zeros((len(tables['wavelength_nm']), len(receivers)))
    if len(receivers) and len(elements):
        _packet_bounds(np.ascontiguousarray(elements.start_m, float),
                       np.ascontiguousarray(elements.direction, float),
                       np.ascontiguousarray(elements.length_m, float),
                       np.ascontiguousarray(elements.beta, float),
                       np.ascontiguousarray(elements.coefficient0, float),
                       np.ascontiguousarray(elements.coefficient2, float),
                       receivers, np.ascontiguousarray(areas, float),
                       np.ascontiguousarray(tables['phase_index'], float),
                       np.ascontiguousarray(tables['mus'], float),
                       np.ascontiguousarray(tables['mut'], float),
                       np.ascontiguousarray(tables['s0w'], float),
                       np.ascontiguousarray(tables['s2w'], float),
                       g, amax, pmax, *azimuth_partition(int(azimuth_nodes)), out)
    # Absorb accumulated rounding of the sums; bounds stay upper bounds.
    return np.nextafter(out * (1 + 1e-12), np.inf)


def seed_packets(bounds, seed_fraction):
    """Per receiver, packets with the largest bounds covering ``seed_fraction``.

    ``bounds`` has shape (packets, receivers) with zeros where a packet does not
    reach a receiver. Returns a boolean (packets, receivers) seed mask. Every
    pair with an infinite bound and every receiver's largest bound is a seed.
    """
    bounds = np.asarray(bounds, float)
    seed = ~np.isfinite(bounds)
    finite = np.where(np.isfinite(bounds), bounds, 0.)
    for j in range(bounds.shape[1]):
        column = finite[:, j]
        total = column.sum()
        if total <= 0:
            continue
        order = np.argsort(-column, kind='stable')
        cumulative = np.cumsum(column[order])
        count = int(np.searchsorted(cumulative, seed_fraction * total) + 1)
        seed[order[:count], j] |= column[order[:count]] > 0
    return seed


def select_skips(bounds, cost, seed, computed_seed_charge, epsilon):
    """Return a boolean (packets, receivers) mask of certified omissions.

    Non-seed pairs are omitted greedily by largest cost/bound while the sum of
    omitted bounds at a receiver stays <= epsilon * computed seed charge.
    ``cost`` is a per-packet work estimate (pencils or elements).
    """
    bounds = np.asarray(bounds, float)
    # A zero bound certifies zero charge (no emitting element, no scattering).
    skip = (bounds == 0) & ~seed
    for j in range(bounds.shape[1]):
        budget = epsilon * float(computed_seed_charge[j])
        if not budget > 0:
            continue
        candidates = np.flatnonzero(~seed[:, j] & np.isfinite(bounds[:, j]) & (bounds[:, j] > 0))
        if not len(candidates):
            continue
        ratio = cost[candidates] / bounds[candidates, j]
        used = 0.
        for p in candidates[np.argsort(-ratio, kind='stable')]:
            if used + bounds[p, j] <= budget:
                used += bounds[p, j]
                skip[p, j] = True
    return skip
