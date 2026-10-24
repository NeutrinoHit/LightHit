"""Independent photon walks measured by finite-volume track length.

There are no RTE kernels, multipoles, first-entry stopping rules, fixed path
limits or collision-order cutoffs here. In implicit mode flights use mu_s;
absorption is integrated continuously as exp(-mu_a*S). Russian roulette at
small weight is unbiased and its threshold controls variance, not physics.
Analog mode instead samples one physical absorption distance for each ray.

Both modes measure the SAME volume-averaged fluence. Per-history scores are
accumulated before squaring, including all revisits. Independent batch means
retain correlations between orders, cells, frequencies and histogram bins.
"""
from dataclasses import dataclass
import numpy as np

try:
    from numba import njit
except ImportError:  # The readable implementation remains usable without JIT.
    def njit(*args, **kwargs):
        return lambda function: function

from .cells import cell_arrays


@njit(cache=True)
def _roots(cuts, count, a, b, c, left, right):
    """Split at roots of a*t²+2*b*t+c, using a dimensionless local t."""
    middle, half = (left+right)/2, (right-left)/2
    aa, bb = a*half*half, half*(a*middle+b)
    cc = (a*middle+2*b)*middle+c
    scale = max(abs(aa), abs(bb), abs(cc))
    if scale == 0.:
        return count
    if abs(aa) <= 8*np.finfo(np.float64).eps*scale:
        if abs(bb) > 8*np.finfo(np.float64).eps*scale:
            u = -cc/(2*bb)
            if -1 < u < 1:
                cuts[count] = middle+half*u
                count += 1
        return count
    disc = bb*bb-aa*cc
    if disc <= 0.:  # A tangent has zero path length.
        return count
    q = -bb-np.copysign(np.sqrt(disc), bb)
    u = q/aa
    if -1 < u < 1:
        cuts[count] = middle+half*u
        count += 1
    if q != 0.:
        u = cc/q
        if -1 < u < 1:
            cuts[count] = middle+half*u
            count += 1
    return count


@njit(cache=True)
def _intervals(position, direction, axis, flight, cell):
    """Exact radial/conical intersections, classified between sorted roots."""
    dot = np.dot(position, direction)
    perpendicular = position-dot*direction
    impact2 = np.dot(perpendicular, perpendicular)
    disc = cell[1]*cell[1]-impact2
    bounds = np.empty((9, 2))
    if disc <= 0.:
        return bounds, 0
    root = np.sqrt(disc)
    left, right = max(0., -dot-root), min(flight, -dot+root)
    if right <= left:
        return bounds, 0
    cuts = np.empty(10)
    cuts[0], cuts[1] = left, right
    square = np.dot(position, position)
    count = _roots(cuts, 2, 1., dot, square-cell[0]*cell[0], left, right)
    z, dz = np.dot(position, axis), np.dot(direction, axis)
    for mu in (cell[2], cell[3]):
        # Squaring also adds the opposite cone. Extra cuts are harmless:
        # the actual signed cosine is checked at every interval midpoint.
        count = _roots(cuts, count, dz*dz-mu*mu, z*dz-mu*mu*dot,
                       z*z-mu*mu*square, left, right)
    ordered = np.sort(cuts[:count])
    used = 0
    for j in range(count-1):
        a, b = ordered[j], ordered[j+1]
        if b <= a:
            continue
        point = position+(a+b)/2*direction
        radius = np.sqrt(np.dot(point, point))
        mu = max(-1., min(1., np.dot(point, axis)/radius))
        if cell[0] <= radius <= cell[1] and cell[2] <= mu <= cell[3]:
            bounds[used, 0], bounds[used, 1] = a, b
            used += 1
    return bounds, used


@njit(cache=True)
def _hg_cosine(uniform, g):
    """HG inverse CDF, algebraically rearranged to remain stable at g=0."""
    u = 2*uniform-1
    value = (u*(1+g*g)+.5*g*(3+u*u+g*g*(u*u-1)))/(1+g*u)**2
    return max(-1., min(1., value))


@njit(cache=True)
def _turn(direction, cosine, phi):
    if abs(direction[2]) < .9:
        first = np.array([-direction[1], direction[0], 0.])
    else:
        first = np.array([0., -direction[2], direction[1]])
    first /= np.sqrt(np.dot(first, first))
    second = np.cross(direction, first)
    sine = np.sqrt(max(0., (1-cosine)*(1+cosine)))
    result = cosine*direction+sine*(np.cos(phi)*first+np.sin(phi)*second)
    return result/np.sqrt(np.dot(result, result))


@njit(cache=True)
def _integral(rate, frequency, speed, path, left, right, emission_time):
    """Weighted line integral, with the +i*omega*t spectrum convention."""
    d = complex(rate, -frequency/speed)
    phase = np.exp(1j*frequency*(emission_time+path/speed))
    if d == 0.:
        return complex(right-left)*phase
    if np.isinf(right):
        return np.exp(-d*left)/d*phase
    return np.exp(-d*left)*(-np.expm1(-d*(right-left)))/d*phase


@njit(cache=True)
def _batch(rng, photons, absorption, scattering, g, speed, axis, isotropic,
           cells, omega, time_edges, emission_time, implicit, roulette_weight,
           isotropic_fraction):
    ncell, nw, nt = len(cells), len(omega), max(0, len(time_edges)-1)
    # Last receiver is the whole of space (normalization units: metres).
    sums = np.zeros((nw, ncell+1, 3), np.complex128)
    second = np.zeros((nw, ncell+1, 4, 2))
    histogram = np.zeros((ncell, nt, 3))
    score = np.zeros((nw, ncell+1, 3), np.complex128)
    collisions, roulettes = 0, 0
    for photon in range(photons):
        score[:] = 0.
        position = np.zeros(3)
        direction = axis.copy()
        if isotropic:
            direction = _turn(axis, 2*rng.random()-1, 2*np.pi*rng.random())
        death = np.inf if implicit else -np.log1p(-rng.random())/absorption
        path, weight, order = 0., 1., 0
        rate = absorption if implicit else 0.
        while True:
            flight = (-np.log1p(-rng.random())/scattering if scattering > 0. else np.inf)
            flight = min(flight, death-path)
            column = min(order, 2)
            for iw in range(nw):
                score[iw, ncell, column] += weight*_integral(
                    rate, omega[iw], speed, path, 0., flight, emission_time)
            for ic in range(ncell):
                intervals, count = _intervals(position, direction, axis, flight, cells[ic])
                for j in range(count):
                    left, right = intervals[j, 0], intervals[j, 1]
                    for iw in range(nw):
                        score[iw, ic, column] += weight*_integral(
                            rate, omega[iw], speed, path, left, right, emission_time)
                    start_bin = max(0, np.searchsorted(time_edges,
                        emission_time+(path+left)/speed, side='right')-1) if nt else 0
                    stop_bin = min(nt, np.searchsorted(time_edges,
                        emission_time+(path+right)/speed, side='left')) if nt else 0
                    for it in range(start_bin, stop_bin):
                        a = max(left, speed*(time_edges[it]-emission_time)-path)
                        b = min(right, speed*(time_edges[it+1]-emission_time)-path)
                        if b > a:
                            histogram[ic, it, column] += weight*_integral(
                                rate, 0., speed, path, a, b, emission_time).real
            if np.isinf(flight) or flight >= death-path:
                break
            position += flight*direction
            path += flight
            if implicit:
                weight *= np.exp(-absorption*flight)
                if weight < roulette_weight:
                    roulettes += 1
                    if rng.random() >= weight/roulette_weight:
                        break
                    weight = roulette_weight
            use_isotropic = isotropic_fraction > 0. and rng.random() < isotropic_fraction
            cosine = 2*rng.random()-1 if use_isotropic else _hg_cosine(rng.random(), g)
            if isotropic_fraction > 0.:
                gap = ((1-g)**2+2*g*(1-cosine) if g >= 0. else
                       (1+g)**2-2*g*(1+cosine))
                phase = (1-g)*(1+g)/gap**1.5  # HG divided by 1/(4*pi).
                weight *= phase/((1-isotropic_fraction)*phase+isotropic_fraction)
            direction = _turn(direction, cosine, 2*np.pi*rng.random())
            order += 1
            collisions += 1
        sums += score
        for iw in range(nw):
            for ic in range(ncell+1):
                total = 0j
                for order in range(3):
                    value = score[iw, ic, order]
                    second[iw, ic, order, 0] += value.real**2
                    second[iw, ic, order, 1] += value.imag**2
                    total += value
                second[iw, ic, 3, 0] += total.real**2
                second[iw, ic, 3, 1] += total.imag**2
    return sums, second, histogram, collisions, roulettes


def _count(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


@dataclass
class BeamMCResult:
    cells: tuple
    omega_per_ns: np.ndarray
    batch_components: np.ndarray  # batch, frequency, cell, order
    batch_global: np.ndarray      # batch, frequency, order; units m
    second_moments: np.ndarray    # batch, frequency, cell incl. global, 4 orders, real/imag
    time_edges_ns: np.ndarray
    batch_bins: np.ndarray        # batch, cell, bin, order
    photons_per_batch: int
    seed: int
    diagnostics: dict

    @property
    def components(self):
        return self.batch_components.mean(axis=0)

    def standard_error(self, *, total=False, global_space=False):
        """Independent-batch SE; sum orders BEFORE computing a total error."""
        values = self.batch_global if global_space else self.batch_components
        if total:
            values = values.sum(axis=-1)
        return np.stack((values.real.std(axis=0, ddof=1),
                         values.imag.std(axis=0, ddof=1)), axis=-1)/np.sqrt(len(values))

    def history_standard_error(self, *, global_space=False):
        """Per-history SE of each order and their correlated total."""
        values = self.batch_global.mean(0) if global_space else self.components
        mean = np.concatenate((values, values.sum(axis=-1, keepdims=True)), axis=-1)
        moments = (self.second_moments[:, :, -1] if global_space else
                   self.second_moments[:, :, :-1]).mean(axis=0)
        squared_mean = np.stack((mean.real**2, mean.imag**2), axis=-1)
        n = self.photons_per_batch*len(self.batch_components)
        return np.sqrt(np.maximum(0., moments-squared_mean)/(n-1))

    def effective_histories(self):
        """Positive zero-frequency score ESS, including each history's revisits."""
        indices = np.flatnonzero(self.omega_per_ns == 0.)
        if len(indices) != 1:
            raise ValueError("effective histories require omega=0")
        i = indices[0]
        mean = np.concatenate((self.components[i].real,
                               self.components[i].real.sum(axis=-1, keepdims=True)), axis=-1)
        second = self.second_moments[:, i, :-1, :, 0].mean(axis=0)
        n = self.photons_per_batch*len(self.batch_components)
        return np.divide(n*mean**2, second, out=np.zeros_like(mean), where=second > 0)


def estimate_cells(medium, cells, *, omega_per_ns=(0.,), photons_per_batch=10000,
                   batches=16, seed=78123, axis=(0., 0., 1.), source="beam",
                   absorption_mode="implicit", roulette_weight=1e-5,
                   isotropic_fraction=0.,
                   time_edges_ns=(), emission_time_ns=0., backend="numba", progress=None):
    """Independent volume fluence per emitted photon [m^-2].

    ``source='isotropic'`` is a useful additional normalization control.
    Implicit and analog absorption are different estimators of the same field.
    ``isotropic_fraction`` optionally proposes a mixture of physical HG and
    isotropic directions, dividing each score by the proposal probability.
    This helps resolve backward tails without changing their expectation.
    No arbitrary path/collision limit is imposed. ``roulette_weight`` is an
    unbiased stochastic stopping threshold used only in implicit mode.
    """
    from time import perf_counter
    cells, bounds = cell_arrays(cells)
    photons = _count(photons_per_batch, "photons_per_batch", 2)
    batches = _count(batches, "batches", 2)
    seed = _count(seed, "seed", 0)
    omega = np.asarray(omega_per_ns, float)
    edges = np.asarray(time_edges_ns, float)
    axis = np.asarray(axis, float)
    if (omega.ndim != 1 or not len(omega) or not np.isfinite(omega).all()
            or np.any(omega < 0) or np.any(np.diff(omega) <= 0)):
        raise ValueError("frequencies must be finite, nonnegative and increasing")
    if edges.ndim != 1 or not np.isfinite(edges).all() or (len(edges) and (
            len(edges) < 2 or np.any(np.diff(edges) <= 0))):
        raise ValueError("provide no time edges or at least two increasing finite edges")
    if axis.shape != (3,) or not np.isfinite(axis).all() or not np.isclose(np.linalg.norm(axis), 1., atol=1e-12, rtol=0):
        raise ValueError("axis must be a unit 3-vector")
    if source not in ("beam", "isotropic") or absorption_mode not in ("implicit", "analog"):
        raise ValueError("source must be beam/isotropic and absorption_mode implicit/analog")
    if not np.isfinite(roulette_weight) or not 0 < roulette_weight <= 1 or not np.isfinite(emission_time_ns):
        raise ValueError("require 0 < roulette_weight <= 1 and finite emission time")
    if not np.isfinite(isotropic_fraction) or not 0 <= isotropic_fraction < 1:
        raise ValueError("isotropic_fraction must be in [0,1)")
    if backend not in ("python", "numba"):
        raise ValueError("backend must be python or numba")
    if backend == "numba" and not hasattr(_batch, "py_func"):
        raise ImportError("install lighthit[accelerate] or use backend='python'")
    function = _batch if backend == "numba" else getattr(_batch, "py_func", _batch)
    volumes = np.r_[[c.volume_m3 for c in cells], 1.]
    samples, moments, histograms = [], [], []
    began = perf_counter()
    collisions = roulettes = 0
    for index in range(batches):
        rng = np.random.default_rng(np.random.SeedSequence([seed, index]))
        values, second, histogram, nc, nr = function(
            rng, photons, medium.absorption_per_m, medium.scattering_per_m, medium.g,
            medium.speed_m_per_ns, axis, source == "isotropic", bounds, omega,
            edges, float(emission_time_ns), absorption_mode == "implicit", float(roulette_weight),
            float(isotropic_fraction))
        samples.append(values/photons/volumes[None, :, None])
        moments.append(second/photons/volumes[None, :, None, None]**2)
        histograms.append(histogram/photons/volumes[:-1, None, None])
        collisions += nc
        roulettes += nr
        if progress is not None:
            progress(index+1, batches)
    samples = np.asarray(samples)
    return BeamMCResult(cells, omega.copy(), samples[:, :, :-1], samples[:, :, -1],
                        np.asarray(moments), edges.copy(), np.asarray(histograms), photons, seed,
                        dict(seconds=perf_counter()-began, photons=photons*batches,
                             collisions=collisions, roulettes=roulettes,
                             absorption_mode=absorption_mode, source=source,
                             roulette_weight=roulette_weight, backend=backend,
                             isotropic_fraction=isotropic_fraction,
                             finite_path_cutoff=False, finite_collision_cutoff=False))
