"""Point-detector Green spectra: exact 0, exact HG 1, finite-HG all >=2.

The spectral solve is streamed in omega and shared by every observation point
in a call. No private optical data, disk cache, or event model is required.
"""
from dataclasses import dataclass, asdict
from time import perf_counter
import warnings
import numpy as np
from scipy.special import roots_legendre, spherical_jn, eval_legendre
from .angular import angular_components, _degree, _check_backend
from .single import single_spectrum, single_quadrature_backend


@dataclass(frozen=True)
class SolverSettings:
    scattering_degree: int = 32
    spatial_degree: int = 160
    k_max_per_m: float = 6.0
    k_panel_per_m: float = 0.04
    k_order: int = 8
    taper: bool = True

    def __post_init__(self):
        _degree(self.scattering_degree, "scattering_degree")
        _degree(self.spatial_degree, "spatial_degree")
        _degree(self.k_order, "k_order")
        if self.k_order < 2:
            raise ValueError("k_order must be >= 2")
        if (not np.isfinite(self.k_max_per_m) or self.k_max_per_m <= 0
                or not np.isfinite(self.k_panel_per_m) or self.k_panel_per_m <= 0):
            raise ValueError("Positive finite k cutoff and panel width required")
        if not isinstance(self.taper, bool):
            raise ValueError("taper must be boolean")

    @classmethod
    def quick(cls):
        return cls(24, 96, 3.0, 0.05, 8)

    @classmethod
    def refined(cls):
        return cls(48, 208, 8.0, 0.025, 10)

    def quadrature(self):
        edges = np.linspace(0, self.k_max_per_m,
                            int(np.ceil(self.k_max_per_m / self.k_panel_per_m)) + 1)
        x, w = roots_legendre(self.k_order)
        half = np.diff(edges) / 2
        k = (edges[:-1, None] + half[:, None] * (1 + x)).ravel()
        weight = (half[:, None] * w).ravel()
        if self.taper:
            a = np.clip(2 * k / self.k_max_per_m - 1, 0, 1)
            weight *= (1 - a)**5 * (1 + 5*a + 15*a*a + 35*a**3 + 70*a**4)
        return k, weight


@dataclass
class GreenResult:
    omega_per_ns: np.ndarray
    displacement_m: np.ndarray
    direction: np.ndarray | None
    medium: object
    settings: SolverSettings
    components: np.ndarray  # (frequency, observation, [0,1,>=2]); units m^-2
    timings_s: dict
    first_quadrature_error: list
    photons: float = 1.0
    emission_time_ns: float = 0.0
    angular_backend: str = "numpy"
    single_backend: str = "numpy"

    @property
    def spectrum(self):
        return self.components.sum(axis=-1)

    @property
    def radii_m(self):
        return np.linalg.norm(self.displacement_m, axis=1)

    @property
    def cosines(self):
        if self.direction is None:
            return None
        return self.displacement_m @ self.direction / self.radii_m

    @property
    def front_time_ns(self):
        return self.emission_time_ns + self.radii_m / self.medium.speed_m_per_ns

    @property
    def charge_per_m2(self):
        indices = np.flatnonzero(self.omega_per_ns == 0)
        if len(indices) != 1:
            raise ValueError("Charge requires an explicitly computed omega=0")
        return self.spectrum[indices[0]].real

    def readout(self, edges_ns, sigma_ns=0.0):
        from .readout import time_bins
        return time_bins(self, edges_ns, sigma_ns)

    def save(self, path):
        """Save numerical output only where explicitly requested by the caller."""
        import json
        from pathlib import Path
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = dict(medium=asdict(self.medium), settings=asdict(self.settings),
                        angular_backend=self.angular_backend,
                        single_backend=self.single_backend,
                        timings_s=self.timings_s, photons=self.photons,
                        emission_time_ns=self.emission_time_ns,
                        component_names=["ballistic", "one_exact_HG", "two_or_more_HG_L"],
                        units="photons per m^2 effective area",
                        source="isotropic" if self.direction is None else "directed")
        np.savez_compressed(path, omega_per_ns=self.omega_per_ns,
                            displacement_m=self.displacement_m,
                            direction=np.array([]) if self.direction is None else self.direction,
                            components=self.components, metadata=json.dumps(metadata))


class PointGreenSolver:
    """A scalar, isotropic, 100%-efficient point detector, per unit area.

    Displacements are detector_position - emission_position, in metres.
    For a directed source the exactly forward ray is singular and rejected.
    direction=None selects an isotropic flash (one photon over the full sphere).
    Each solve recomputes transport; all observations in that solve share it.
    """
    def __init__(self, medium, settings=None, *, angular_backend="numpy",
                 single_backend="auto"):
        self.medium = medium
        self.settings = settings or SolverSettings()
        self.angular_backend = _check_backend(angular_backend)
        self.single_backend = single_quadrature_backend(single_backend)

    def solve(self, omega_per_ns, displacement_m, *, direction=(0.0, 0.0, 1.0),
              photons=1.0, emission_time_ns=0.0, progress=None):
        start = perf_counter()
        omega = np.asarray(omega_per_ns, float)
        rvec = np.atleast_2d(np.asarray(displacement_m, float))
        if (omega.ndim != 1 or not len(omega) or not np.isfinite(omega).all()
                or np.any(omega < 0) or np.any(np.diff(omega) <= 0)):
            raise ValueError("omega must be finite, nonnegative and strictly increasing")
        if rvec.ndim != 2 or rvec.shape[1] != 3 or not np.isfinite(rvec).all():
            raise ValueError("displacement_m must have shape (3,) or (D,3)")
        radius = np.linalg.norm(rvec, axis=1)
        if not len(radius) or np.any(radius <= 0):
            raise ValueError("Point source and point detector must be distinct")
        if not np.isfinite(photons) or photons < 0 or not np.isfinite(emission_time_ns):
            raise ValueError("Require photons >= 0 and finite emission time")
        cosine = None
        if direction is not None:
            direction = np.asarray(direction, float)
            if direction.shape != (3,) or not np.isfinite(direction).all() or not np.isclose(np.linalg.norm(direction), 1, rtol=0, atol=1e-12):
                raise ValueError("direction must be a unit 3-vector; use None for isotropic emission")
            cosine = np.clip(rvec @ direction / radius, -1, 1)
            if np.any(cosine >= 1 - 1e-12):
                raise ValueError("Exactly forward directed point-source/point-detector response is a distribution. Use a finite aperture or a nonzero angle.")
            if np.any(cosine > 0.98):
                warnings.warn("Near the forward ray, refine angular and spatial resolution independently", RuntimeWarning)
        settings = self.settings
        k, kw = settings.quadrature()
        J = settings.spatial_degree if direction is not None else 0
        ell = np.arange(J + 1)
        # Shape D,K,J+1. No grid over x,y,z,t and photon direction is allocated.
        spatial = np.empty((len(radius), len(k), J + 1), complex)
        norm = np.sqrt((2 * ell + 1) / 2) * (1j)**ell
        tk = perf_counter()
        for d, rad in enumerate(radius):
            angular = eval_legendre(ell, cosine[d]) if cosine is not None else np.ones(1)
            spatial[d] = (spherical_jn(ell[None, :], k[:, None] * rad)
                          * (angular * norm)[None, :]
                          * (kw * k * k)[:, None] / (2 * np.pi**2))
        bessel_time = perf_counter() - tk
        values = np.zeros((len(omega), len(radius), 3), complex)
        angular_time = 0.0
        spatial_time = 0.0
        if self.medium.scattering_per_m > 0 and photons > 0:
            for iw, frequency in enumerate(omega):
                before = perf_counter()
                _, _, multiple = angular_components(k, frequency, self.medium,
                                                     settings.scattering_degree, J,
                                                     backend=self.angular_backend)
                angular_time += perf_counter() - before
                before = perf_counter()
                values[iw, :, 2] = np.einsum("dkj,kj->d", spatial, multiple[:, :J + 1], optimize=False)
                spatial_time += perf_counter() - before
                if progress is not None:
                    progress(iw + 1, len(omega))
        first_start = perf_counter()
        errors = []
        for d, rad in enumerate(radius):
            if photons == 0:
                errors.append(0.0)
                continue
            first, error = single_spectrum(
                omega, rad, None if cosine is None else cosine[d], self.medium,
                backend=self.single_backend)
            values[:, d, 1] = first
            errors.append(error)
            if direction is None:
                q0 = np.exp(-self.medium.extinction_per_m * rad) / (4 * np.pi * rad * rad)
                values[:, d, 0] = q0 * np.exp(1j * omega * rad / self.medium.speed_m_per_ns)
            # Directed delta flash: ballistic is zero off its forward ray.
        first_time = perf_counter() - first_start
        values *= photons * np.exp(1j * omega * emission_time_ns)[:, None, None]
        timings = dict(spatial_weights=bessel_time, angular_solve=angular_time,
                       inverse_space=spatial_time, exact_low_orders=first_time,
                       total=perf_counter() - start, k_nodes=len(k),
                       observations=len(radius), frequencies=len(omega))
        return GreenResult(omega.copy(), rvec.copy(), None if direction is None else direction.copy(),
                           self.medium, settings, values, timings, errors,
                           float(photons), float(emission_time_ns), self.angular_backend,
                           self.single_backend)
