"""Independent analog half-space Monte Carlo with full HG scattering.

Continuous free flights and HG angles are sampled directly. A smooth
dielectric plane uses analog Fresnel reflection; no angular/depth mesh or
modal solver is used. Source-entry transmission is a packet weight. Every
entered history ends in absorption or one exterior escape. The event limit
raises an error instead of silently truncating light.
"""
from dataclasses import dataclass
import math

import numpy as np
from numba import njit


@njit(cache=True)
def _interface(mu, ratio):
    if ratio == 1.:
        return 0., mu
    sin2 = ratio * ratio * max(0., (1 - mu) * (1 + mu))
    if sin2 >= 1.:
        return 1., 0.
    transmitted = math.sqrt(1 - sin2)
    rs = (ratio * mu - transmitted) / (ratio * mu + transmitted)
    rp = (mu - ratio * transmitted) / (mu + ratio * transmitted)
    return (rs * rs + rp * rp) / 2, transmitted


@njit(cache=True)
def _segment_cells(x, y, z, ux, uy, uz, length, cells, track):
    position = (x, y, z)
    direction = (ux, uy, uz)
    for cell in range(len(cells)):
        low, high = 0., length
        for axis in range(3):
            if direction[axis] == 0:
                if position[axis] < cells[cell, 0, axis] or position[axis] > cells[cell, 1, axis]:
                    high = -1.
                    break
            else:
                a = (cells[cell, 0, axis] - position[axis]) / direction[axis]
                b = (cells[cell, 1, axis] - position[axis]) / direction[axis]
                low = max(low, min(a, b))
                high = min(high, max(a, b))
        if high > low:
            track[cell] += high - low


@njit(cache=True)
def _histories(mu_a, mu_s, g, n, directions, weights, photons, seed, max_events, sigma, cells):
    np.random.seed(seed)
    x_out = np.zeros(photons)
    y_out = np.zeros(photons)
    length_out = np.zeros(photons)
    cosine_out = np.zeros(photons)
    order_out = np.full(photons, -1, np.int64)
    entered = np.zeros(photons)
    cell_path = np.zeros((photons, len(cells)))
    cumulative = np.cumsum(weights)
    failed = 0
    extinction = mu_a + mu_s
    for p in range(photons):
        ray = min(np.searchsorted(cumulative, np.random.random()), len(weights) - 1)
        external_mu = directions[ray, 2]
        entry_r, uz = _interface(external_mu, 1 / n)
        entered[p] = 1 - entry_r
        if entered[p] == 0:
            continue
        ux, uy = directions[ray, 0] / n, directions[ray, 1] / n
        x, y, z = 0., 0., 0.
        if sigma > 0:
            x, y = sigma * np.random.normal(), sigma * np.random.normal()
        path, collisions = 0., 0
        track = np.zeros(len(cells))
        completed = False
        for event in range(max_events):
            flight = -math.log(max(np.random.random(), 1e-300)) / extinction
            boundary_distance = -z / uz if uz < 0 else np.inf
            if boundary_distance <= flight:
                _segment_cells(x, y, z, ux, uy, uz, boundary_distance, cells, track)
                x += ux * boundary_distance
                y += uy * boundary_distance
                path += boundary_distance
                z = 0.
                reflectance, transmitted_mu = _interface(-uz, n)
                if np.random.random() < reflectance:
                    uz = -uz
                    continue
                x_out[p], y_out[p] = x, y
                length_out[p], cosine_out[p] = path, transmitted_mu
                order_out[p] = min(collisions, 2)
                cell_path[p] = track
                completed = True
                break
            _segment_cells(x, y, z, ux, uy, uz, flight, cells, track)
            x += ux * flight
            y += uy * flight
            z += uz * flight
            path += flight
            if np.random.random() < mu_a / extinction:
                completed = True
                break
            collisions += 1
            uniform = np.random.random()
            if abs(g) < 1e-12:
                cosine = 2 * uniform - 1
            else:
                ratio = (1 - g*g) / (1 - g + 2*g*uniform)
                cosine = max(-1., min(1., (1 + g*g - ratio*ratio) / (2*g)))
            sine = math.sqrt(max(0., 1 - cosine*cosine))
            phi = 2 * math.pi * np.random.random()
            if abs(uz) < .9:
                ax, ay, az = uy, -ux, 0.
            else:
                ax, ay, az = 0., uz, -uy
            norm = math.sqrt(ax*ax + ay*ay + az*az)
            ax, ay, az = ax/norm, ay/norm, az/norm
            bx, by, bz = uy*az-uz*ay, uz*ax-ux*az, ux*ay-uy*ax
            c, s = math.cos(phi), math.sin(phi)
            ux, uy, uz = (cosine*ux+sine*(c*ax+s*bx),
                          cosine*uy+sine*(c*ay+s*by),
                          cosine*uz+sine*(c*az+s*bz))
        if not completed:
            failed += 1
    return x_out, y_out, length_out, cosine_out, order_out, entered, cell_path, failed


@dataclass(frozen=True)
class MonteCarloEstimate:
    mean: np.ndarray
    stderr_real: np.ndarray
    stderr_imag: np.ndarray


@dataclass
class HalfSpaceSamples:
    x_m: np.ndarray
    y_m: np.ndarray
    path_m: np.ndarray
    cosine_external: np.ndarray
    order: np.ndarray
    entered_weight: np.ndarray
    cell_path_m: np.ndarray
    speed_m_per_ns: float
    cells_m: np.ndarray

    def _estimate(self, value):
        value = np.asarray(value)
        count = len(self.order)
        mean = value.mean(axis=-1)
        real = np.sqrt(np.maximum(0., (np.mean(value.real**2, axis=-1) - mean.real**2) / (count - 1)))
        imag = np.sqrt(np.maximum(0., (np.mean(value.imag**2, axis=-1) - mean.imag**2) / (count - 1)))
        return MonteCarloEstimate(mean, real, imag)

    def _weights(self, half_angle_rad):
        angle = float(half_angle_rad)
        if not math.isfinite(angle) or not 0 < angle <= math.pi / 2:
            raise ValueError("aperture half-angle must lie in (0, pi/2]")
        return self.entered_weight * ((self.order >= 0) & (self.cosine_external >= math.cos(angle)))

    def spectrum(self, q_per_m, omega_per_ns, *, half_angle_rad=np.pi / 2,
                 absorption_derivative=False, absorption_offset_per_m=0.):
        """Escape spectrum, optionally reweighted to a larger absorption.

        Detected path probability changes by exp(-delta_mu_a * path) when
        only absorption changes. The offset is relative to the medium used
        to trace these histories. It does not change the stored base photon
        balance or generate trajectories for a smaller absorption.
        """
        q = np.atleast_2d(np.asarray(q_per_m, float))
        omega = np.atleast_1d(np.asarray(omega_per_ns, float))
        if (q.ndim != 2 or q.shape[1] != 2 or not np.isfinite(q).all()
                or omega.ndim != 1 or not np.isfinite(omega).all()):
            raise ValueError("q must contain finite two-vectors and omega a finite vector")
        offset = float(absorption_offset_per_m)
        if not math.isfinite(offset) or offset < 0:
            raise ValueError("absorption_offset_per_m must be finite and nonnegative")
        weight = self._weights(half_angle_rad)*np.exp(-offset*self.path_m)
        if absorption_derivative:
            weight = -weight * self.path_m
        shape = (len(omega), len(q), 3)
        mean, real, imag = np.zeros(shape, complex), np.zeros(shape), np.zeros(shape)
        for f, frequency in enumerate(omega):
            for k, wavevector in enumerate(q):
                phase = (frequency * self.path_m / self.speed_m_per_ns
                         - wavevector[0] * self.x_m - wavevector[1] * self.y_m)
                value = weight * np.exp(1j * phase)
                for order in range(3):
                    estimate = self._estimate(value * (self.order == order))
                    mean[f, k, order] = estimate.mean
                    real[f, k, order], imag[f, k, order] = estimate.stderr_real, estimate.stderr_imag
        return MonteCarloEstimate(mean, real, imag)

    def radial_density(self, radii_m, *, detector_sigma_m, half_angle_rad=np.pi / 2):
        radius = np.atleast_1d(np.asarray(radii_m, float))
        sigma = float(detector_sigma_m)
        if radius.ndim != 1 or not np.isfinite(radius).all() or np.any(radius < 0) or not math.isfinite(sigma) or sigma <= 0:
            raise ValueError("radii must be finite and nonnegative and detector sigma positive")
        weight = self._weights(half_angle_rad)
        mean = np.zeros((len(radius), 3))
        real = np.zeros_like(mean)
        for i, r in enumerate(radius):
            gaussian = np.exp(-((self.x_m - r)**2 + self.y_m**2) / (2*sigma*sigma)) / (2*np.pi*sigma*sigma)
            for order in range(3):
                estimate = self._estimate(weight * gaussian * (self.order == order))
                mean[i, order], real[i, order] = estimate.mean, estimate.stderr_real
        return MonteCarloEstimate(mean, real, np.zeros_like(real))

    def time_bins(self, edges_ns, *, half_angle_rad=np.pi / 2):
        edges = np.asarray(edges_ns, float)
        if edges.ndim != 1 or len(edges) < 2 or not np.isfinite(edges).all() or np.any(np.diff(edges) <= 0):
            raise ValueError("time edges must be finite and strictly increasing")
        weight = self._weights(half_angle_rad)
        time = self.path_m / self.speed_m_per_ns
        mean, real = np.zeros((len(edges)-1, 3)), np.zeros((len(edges)-1, 3))
        for i in range(len(edges)-1):
            mask = (time >= edges[i]) & (time < edges[i+1])
            for order in range(3):
                estimate = self._estimate(weight * mask * (self.order == order))
                mean[i, order], real[i, order] = estimate.mean, estimate.stderr_real
        return MonteCarloEstimate(mean, real, np.zeros_like(real))

    def photon_balance(self):
        entered = float(self.entered_weight.mean())
        absorbed = float(np.mean(self.entered_weight * (self.order < 0)))
        escaped = float(np.mean(self.entered_weight * (self.order >= 0)))
        return {"specular": 1-entered, "entered": entered, "absorbed": absorbed,
                "escaped": escaped, "residual": absorbed+escaped-entered}

    def cell_absorption_sensitivity(self, *, detector_xy_m=(1., 0.), detector_sigma_m,
                                    half_angle_rad=np.pi / 2, by_order=False):
        """Independent path-length derivative averaged over each supplied cell.

        With ``by_order`` the estimate has shape (cell, order) for detected
        volume orders (0, 1, >=2).
        """
        xy = np.asarray(detector_xy_m, float)
        sigma = float(detector_sigma_m)
        if xy.shape != (2,) or not np.isfinite(xy).all() or not math.isfinite(sigma) or sigma <= 0:
            raise ValueError("detector position must be finite and its sigma positive")
        gaussian = np.exp(-((self.x_m-xy[0])**2+(self.y_m-xy[1])**2)/(2*sigma*sigma))/(2*np.pi*sigma*sigma)
        weight = self._weights(half_angle_rad) * gaussian
        volume = np.prod(self.cells_m[:, 1]-self.cells_m[:, 0], axis=1)
        value = (-self.cell_path_m * weight[:, None] / volume).T
        if not by_order:
            return self._estimate(value)
        masks = np.stack([self.order == order for order in range(3)])
        return self._estimate(value[:, None, :] * masks[None, :, :])


def sample_halfspace(medium, boundary, source, *, photons=200000, seed=20261103,
                     max_events=10000, source_sigma_m=0., sensitivity_cells_m=None):
    """Trace continuous full-HG histories for a prescribed external ray port."""
    for value, name, minimum in ((photons, "photons", 2), (max_events, "max_events", 1)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if not isinstance(seed, (int, np.integer)) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an unsigned 32-bit integer")
    sigma = float(source_sigma_m)
    if not math.isfinite(sigma) or sigma < 0:
        raise ValueError("source_sigma_m must be finite and nonnegative")
    cells = np.empty((0, 2, 3)) if sensitivity_cells_m is None else np.asarray(sensitivity_cells_m, float)
    if (cells.ndim != 3 or cells.shape[1:] != (2, 3) or not np.isfinite(cells).all()
            or np.any(cells[:, 0] >= cells[:, 1]) or np.any(cells[:, 0, 2] < 0)):
        raise ValueError("sensitivity cells must have finite lower/upper corners in z>=0")
    result = _histories(medium.absorption_per_m, medium.scattering_per_m, medium.g,
                        boundary.n, source.directions_external, source.flux_weights,
                        int(photons), int(seed), int(max_events), sigma, cells)
    if result[-1]:
        raise RuntimeError(f"{result[-1]} histories exceeded max_events; no truncated estimate returned")
    return HalfSpaceSamples(*result[:-1], medium.speed_m_per_ns, cells)
