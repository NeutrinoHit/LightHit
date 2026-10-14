"""Physical-time reusable one-scatter CDF for joint photon histograms.

A response belongs to a relative emitter/direction/receiver geometry and one
wavelength's optics, never to an event's emission clock or histogram weights.
It is a pencil response, not a claim that a finite source bin is a pencil.
Finite z/mu/azimuth bins can use quadrature sums of these responses. Only the
temporal histogram is contracted here; no azimuthal symmetry is assumed.
"""
from dataclasses import dataclass

import numpy as np
from numba import njit
from scipy.interpolate import CubicSpline


@njit(cache=True, nogil=True)
def _evaluate_path(x, sigma, knots, coefficients, total):
    out = np.empty(x.size)
    for i in range(x.size):
        if x[i] <= 0.:
            out[i] = 0.
            continue
        u = np.arcsinh(x[i] / sigma)
        if u >= knots[-1]:
            out[i] = total
            continue
        lo, hi = 0, knots.size - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if knots[mid] <= u:
                lo = mid
            else:
                hi = mid
        du = u - knots[lo]
        value = coefficients[0, lo]
        for k in range(1, coefficients.shape[0]):
            value = value * du + coefficients[k, lo]
        out[i] = value
    return out


@njit(cache=True, nogil=True)
def _temporal_contract(edges, clocks, weights, origin, radius, speed,
                       sigma, knots, coefficients, total):
    bins = np.zeros(edges.size - 1)
    for j in range(clocks.size):
        if weights[j] == 0.:
            continue
        x = speed * (edges + origin - clocks[j]) - radius
        cdf = _evaluate_path(x, sigma, knots, coefficients, total)
        for b in range(bins.size):
            bins[b] += weights[j] * (cdf[b + 1] - cdf[b])
    return total * weights.sum(), bins


@dataclass(frozen=True)
class PencilFirstCDF:
    """Per emitted photon, per m², without wavelength efficiency.

    Acceptance is a polynomial of photon propagation direction dot ``look``;
    use minus detector orientation for LightHit's convention. ``speed`` is
    group speed in m/ns. All-time charge is independent of source clocks.
    The finite path truncation exp(-tail_exponent) is explicit. Queries before
    the causal front return exactly zero; times may extend over many periods
    of any Fourier grid because this calculation has no Fourier grid.
    """
    radius_m: float
    speed_m_per_ns: float
    sigma_m: float
    knots: np.ndarray
    coefficients: np.ndarray
    charge_per_m2: float
    geometry: tuple
    path_step: float
    tail_exponent: float

    @classmethod
    def build(cls, displacement_m, direction, look, acceptance_polynomial, *,
              scattering_per_m, absorption_per_m, g, speed_m_per_ns,
              path_step=.015, tail_exponent=25., max_grid_nodes=16384):
        delta, direction, look = [np.asarray(v, float) for v in
                                   (displacement_m, direction, look)]
        if any(v.shape != (3,) or not np.all(np.isfinite(v)) for v in
               (delta, direction, look)):
            raise ValueError("finite 3D displacement/direction/look required")
        if not np.isclose(np.linalg.norm(direction), 1.) or not np.isclose(np.linalg.norm(look), 1.):
            raise ValueError("direction and look must be unit vectors")
        radius = float(np.linalg.norm(delta))
        if radius <= 0.:
            raise ValueError("point detector cannot coincide with source")
        cosine = float(np.clip(delta @ direction / radius, -1., 1.))
        sine = float(np.linalg.norm(np.cross(delta / radius, direction)))
        if sine < 1e-8 and cosine > 0.:
            raise ValueError("forward collinear pencil requires finite angular-bin integration")
        mus, mua, speed = float(scattering_per_m), float(absorption_per_m), float(speed_m_per_ns)
        apoly = np.asarray(acceptance_polynomial, float)
        if (apoly.ndim != 1 or not len(apoly) or not np.all(np.isfinite(apoly)) or
                not np.all(np.isfinite([mus, mua, g, speed, path_step, tail_exponent])) or
                mus < 0. or mua < 0. or mus + mua <= 0. or abs(g) >= 1. or
                min(speed, path_step, tail_exponent) <= 0.):
            raise ValueError("invalid optics or path rule")
        mut = mus + mua
        y_g = (1. - abs(g)) / (1. + abs(g))
        # sin²(theta)/(1+cos(theta))=1-cos(theta) removes the antipodal
        # cancellation. Exactly backward pencils have a regular finite response.
        sigma = max(.5 * radius * sine * y_g, .25 * radius * (1. - cosine))
        max_u = np.arcsinh(tail_exponent / (mut * sigma))
        count = max(8, int(np.ceil(max_u / path_step)))
        if isinstance(max_grid_nodes, bool) or int(max_grid_nodes) != max_grid_nodes or max_grid_nodes < 9:
            raise ValueError("max_grid_nodes must be an integer >=9")
        if count + 1 > max_grid_nodes:
            raise ValueError("path CDF exceeds max_grid_nodes; refine optical domain or increase explicit bound")
        u = np.linspace(0., max_u, count + 1)
        excess = sigma * np.sinh(u)
        path = radius + excess
        # Stable a avoids cancellation in S²-r² near the causal front.
        denominator = path - radius * cosine
        a = excess * (2. * radius + excess) / (2. * denominator)
        b = path - a
        outgoing = (delta[None, :] - a[:, None] * direction) / b[:, None]
        scatter_cos = np.clip(outgoing @ direction, -1., 1.)
        acceptance = np.polynomial.polynomial.polyval(outgoing @ look, apoly)
        hg = (1. - g*g) / (4. * np.pi * (1. + g*g - 2.*g*scatter_cos)**1.5)
        density_u = (mus * hg * acceptance * np.exp(-mut * path) /
                     (b * denominator) * sigma * np.cosh(u))
        # Integrate a cubic approximation in smooth, dimensionless path u.
        # Its antiderivative has a quartic local polynomial and exact knot
        # continuity. No renormalisation or positivity clipping is applied.
        antiderivative = CubicSpline(u, density_u).antiderivative()
        total = float(antiderivative(max_u))
        geometry = tuple(delta) + tuple(direction) + tuple(look) + tuple(apoly) + (mus, mua, g, speed)
        return cls(radius, speed, sigma, np.ascontiguousarray(u),
                   np.ascontiguousarray(antiderivative.c), total, geometry,
                   float(path_step), float(tail_exponent))

    def cdf_path(self, excess_path_m):
        x = np.asarray(excess_path_m, float)
        if not np.all(np.isfinite(x)):
            raise ValueError("finite excess path required")
        return _evaluate_path(x.ravel(), self.sigma_m, self.knots,
                              self.coefficients, self.charge_per_m2).reshape(x.shape)

    def contract(self, emission_times_ns, photon_weights, time_edges_ns, *,
                 time_origin_ns=0., effective_area_m2=1.):
        """Return all-time charge and physical-time bins for one shared geometry.

        Weights include photon number, wavelength integration and QE if wanted.
        Signed basis weights are accepted. No output-window charge substitution.
        """
        times, weights, edges = [np.ascontiguousarray(v, dtype=float) for v in
                                 (emission_times_ns, photon_weights, time_edges_ns)]
        if (times.ndim != 1 or weights.shape != times.shape or edges.ndim != 1 or
                len(edges) < 2 or np.any(np.diff(edges) <= 0.) or
                not all(np.all(np.isfinite(v)) for v in (times, weights, edges)) or
                not np.isfinite(time_origin_ns) or not np.isfinite(effective_area_m2) or
                effective_area_m2 < 0.):
            raise ValueError("invalid clocks, weights, edges or detector area")
        q, bins = _temporal_contract(edges, times, weights, float(time_origin_ns),
            self.radius_m, self.speed_m_per_ns, self.sigma_m, self.knots,
            self.coefficients, self.charge_per_m2)
        return q * effective_area_m2, bins * effective_area_m2


def contract_joint_pencils(responses, geometry_id, emission_times_ns,
                           photon_weights, time_edges_ns, *,
                           time_origin_ns=0., effective_area_m2=1.):
    """Contract positive photon-bin geometry responses with their clock masses.

    ``responses`` is a reusable list of PencilFirstCDF objects. Each row of the
    joint source identifies a geometry response and keeps its own emission
    clock/weight. The response list has no clocks or event weights in it.
    Caller supplies the wavelength-dependent list and weights for each band.
    """
    ids = np.asarray(geometry_id)
    times, weights = np.asarray(emission_times_ns, float), np.asarray(photon_weights, float)
    if ids.ndim != 1 or ids.dtype.kind not in "iu" or times.shape != ids.shape or weights.shape != ids.shape:
        raise ValueError("one integer geometry ID, clock and weight per source row required")
    if ids.size and (ids.min() < 0 or ids.max() >= len(responses)):
        raise ValueError("geometry ID outside response list")
    total, bins = 0., np.zeros(len(time_edges_ns) - 1)
    # Stable index sort makes temporal contraction contiguous while preserving
    # all clocks, in contrast to replacing their distribution by its mean.
    order = np.argsort(ids, kind="stable")
    lo = 0
    while lo < len(order):
        key = ids[order[lo]]
        hi = lo + 1
        while hi < len(order) and ids[order[hi]] == key:
            hi += 1
        rows = order[lo:hi]
        q, b = responses[key].contract(times[rows], weights[rows], time_edges_ns,
            time_origin_ns=time_origin_ns, effective_area_m2=effective_area_m2)
        total += q; bins += b
        lo = hi
    return total, bins


def direct_photon_histogram(kernel, histogram, receiver_indices, time_edges_ns,
                           time_origin_ns, *, wavelength_weight_nm=1.,
                           path_step=.025, tail_exponent=25., max_path_nodes=4096):
    """Physical-time HG evaluation of one positive photon-histogram band.

    A practical baseline that avoids per-event CDF preparation when geometry
    reuse is low. Weights already contain photons/nm, including Cherenkov beta
    threshold and lambda^-2; only wavelength weight, QE and OM area are applied
    here. The source binning is an explicit approximation; this evaluator does
    not add isotropy or average separate source clocks.
    """
    from .._prompt_numba import pencils_order1, interval_table
    from ..prompt import acceptance_monomials, acceptance_polynomial, C_M_PER_NS
    wave = np.asarray(histogram.wavelength_nm, float)
    if wave.shape != (1,):
        raise ValueError("direct_photon_histogram requires one wavelength band")
    ids = np.asarray(receiver_indices)
    edges, origins = np.ascontiguousarray(time_edges_ns, float), np.ascontiguousarray(time_origin_ns, float)
    if (ids.ndim != 1 or ids.dtype.kind not in "iu" or origins.shape != ids.shape or
            np.any(ids < 0) or np.any(ids >= len(kernel.detector.positions_m)) or
            edges.ndim != 1 or len(edges) < 2 or np.any(np.diff(edges) <= 0.) or
            not np.all(np.isfinite(edges)) or not np.all(np.isfinite(origins)) or
            not np.isfinite(wavelength_weight_nm) or wavelength_weight_nm < 0. or
            min(path_step, tail_exponent) <= 0. or max_path_nodes < 3):
        raise ValueError("invalid receiver indices, clocks, edges or path controls")
    sample = kernel.medium.sample(wave)
    mus = np.ascontiguousarray(sample['scattering_per_m'], float)
    mut = mus + np.asarray(sample['absorption_per_m'], float)
    speed = C_M_PER_NS / np.asarray(sample['group_index'], float)
    kappa = -mut * speed
    eedge = np.exp(np.outer(kappa, edges))
    itab = interval_table(edges, kappa)
    apoly = acceptance_polynomial(*acceptance_monomials(kernel.acceptance()['alpha']))
    geometry = np.asarray(histogram.geometry_id, np.int64)
    points = np.ascontiguousarray(histogram.position[geometry], float)
    directions = np.ascontiguousarray(histogram.direction[geometry], float)
    times = np.ascontiguousarray(histogram.emission_times_ns, float)
    weights = np.ascontiguousarray(histogram.photon_weights[0], float)
    efficiency = float(kernel.detector.spectral_weight(float(wave[0])))
    g = float(kernel.medium.g)
    return pencils_order1(np.ascontiguousarray(kernel.detector.positions_m[ids], float),
        np.ascontiguousarray(-kernel.detector.orientations[ids], float),
        np.ascontiguousarray(kernel.detector.effective_area_m2[ids], float), origins,
        points, directions, weights, np.zeros(len(weights)), times, apoly,
        g, (1-abs(g))/(1+abs(g)), float(path_step), float(path_step), float(tail_exponent),
        mus, np.ascontiguousarray(mut), np.ascontiguousarray(speed),
        np.array([wavelength_weight_nm*efficiency]), np.zeros(1), edges, eedge,
        itab, 4, int(max_path_nodes))
