"""Volume averages of the existing point Green solver, not receiver tables.

The angular average of every spatial Legendre mode is analytic; only its
radial Bessel average uses Gauss quadrature. Q0 is an exact ray integral.
Q1 retains the solver's full-HG coordinate-space integral. Only >=2 uses the
truncated RTE resolvent and finite inverse spatial transform.
"""
from dataclasses import dataclass, asdict
from time import perf_counter
import numpy as np
from scipy.special import eval_legendre, roots_legendre, spherical_jn

from ..angular import angular_components
from ..green import SolverSettings
from ..single import single_spectrum
from .cells import cell_arrays


def global_spectra(medium, omega_per_ns):
    """Integral over all space: [Q0,Q1,Q>=2], units m per emitted photon."""
    omega = np.asarray(omega_per_ns, float)
    d = medium.extinction_per_m-1j*omega/medium.speed_m_per_ns
    a = medium.absorption_per_m-1j*omega/medium.speed_m_per_ns
    return np.stack((1/d, medium.scattering_per_m/d**2,
                     medium.scattering_per_m**2/(a*d**2)), axis=-1)


def ballistic_cells(medium, cells, omega_per_ns=(0.,), *, source="beam", emission_time_ns=0.):
    """Exact straight-ray volume fluence, including the forward distribution."""
    cells, _ = cell_arrays(cells)
    omega = np.asarray(omega_per_ns, float)
    d = medium.extinction_per_m-1j*omega/medium.speed_m_per_ns
    out = np.zeros((len(omega), len(cells)), complex)
    for i, c in enumerate(cells):
        fraction = float(c.mu_max == 1.) if source == "beam" else (c.mu_max-c.mu_min)/2
        out[:, i] = (fraction*np.exp(-d*c.r_min_m)*(-np.expm1(-d*(c.r_max_m-c.r_min_m)))
                     /d/c.volume_m3*np.exp(1j*omega*emission_time_ns))
    return out


def _angular_average(degree, low, high):
    out = np.ones(degree+1)
    if degree:
        j = np.arange(1, degree+1)
        def primitive(x):
            return (eval_legendre(j+1, x)-eval_legendre(j-1, x))/(2*j+1)
        out[1:] = (primitive(high)-primitive(low))/(high-low)
    return out


@dataclass
class BeamRTEResult:
    cells: tuple
    omega_per_ns: np.ndarray
    components: np.ndarray
    settings: SolverSettings
    diagnostics: dict


def cell_spectra(medium, cells, omega_per_ns=(0.,), *, settings=None,
                 source="beam", radial_order=8, polar_order=20,
                 angular_backend="numpy", single_backend="auto", progress=None):
    """RTE and exact low orders averaged over the same finite cells as MC.

    ``scattering_degree``, ``spatial_degree``, k cutoff/panels and the radial
    and polar quadratures are independent numerical resolutions. This helper
    constructs no event cache and changes no prepared receiver atlas.
    """
    cells, _ = cell_arrays(cells)
    omega = np.asarray(omega_per_ns, float)
    if (omega.ndim != 1 or not len(omega) or not np.isfinite(omega).all()
            or np.any(omega < 0) or np.any(np.diff(omega) <= 0)):
        raise ValueError("frequencies must be finite, nonnegative and increasing")
    if source not in ("beam", "isotropic"):
        raise ValueError("source must be beam or isotropic")
    for value in (radial_order, polar_order):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 2:
            raise ValueError("averaging orders must be integers >= 2")
    settings = settings or SolverSettings()
    start = perf_counter()
    x, w = roots_legendre(radial_order)
    tx, tw = roots_legendre(polar_order)
    values = np.zeros((len(omega), len(cells), 3), complex)
    values[:, :, 0] = ballistic_cells(medium, cells, omega, source=source)
    first_error = np.zeros(len(cells))
    if medium.scattering_per_m == 0.:
        return BeamRTEResult(cells, omega, values, settings,
                             dict(seconds=perf_counter()-start, radial_order=radial_order,
                                  polar_order=polar_order, k_nodes=0))
    k, kw = settings.quadrature()
    degree = settings.spatial_degree if source == "beam" else 0
    ell = np.arange(degree+1)
    weights = np.empty((len(cells), len(k), degree+1), complex)
    radial_cache = {}
    for i, c in enumerate(cells):
        key = (c.r_min_m, c.r_max_m)
        radii = (c.r_min_m+c.r_max_m)/2+(c.r_max_m-c.r_min_m)/2*x
        rw = (c.r_max_m-c.r_min_m)/2*w
        normalization = 3/((c.r_max_m-c.r_min_m)*(
            c.r_max_m**2+c.r_max_m*c.r_min_m+c.r_min_m**2))
        if key not in radial_cache:
            radial = np.zeros((len(k), degree+1))
            for r, weight in zip(radii, rw):
                radial += weight*r*r*spherical_jn(ell[None, :], k[:, None]*r)
            radial_cache[key] = radial*normalization
        angular = _angular_average(degree, c.mu_min, c.mu_max)
        weights[i] = (radial_cache[key]*angular[None, :]
                      * (1j)**ell*np.sqrt((2*ell+1)/2)[None, :]
                      * (kw*k*k/(2*np.pi**2))[:, None])
        low, high = np.arccos(c.mu_max), np.arccos(c.mu_min)
        theta = (low+high)/2+(high-low)/2*tx
        pw = (high-low)/2*tw
        if source == "isotropic":
            theta, pw = np.array([0.]), np.array([c.mu_max-c.mu_min])
        for r, radial_weight in zip(radii, rw):
            for angle, polar_weight in zip(theta, pw):
                first, error = single_spectrum(omega, float(r),
                    None if source == "isotropic" else float(np.cos(angle)),
                    medium, backend=single_backend)
                measure = (2*np.pi*radial_weight*r*r*polar_weight/c.volume_m3
                           * (1. if source == "isotropic" else np.sin(angle)))
                values[:, i, 1] += measure*first
                first_error[i] += abs(measure)*error
    prepared_seconds = perf_counter()-start
    for iw, frequency in enumerate(omega):
        _, _, multiple = angular_components(k, frequency, medium,
            settings.scattering_degree, degree, backend=angular_backend)
        values[iw, :, 2] = np.einsum("ckj,kj->c", weights, multiple[:, :degree+1], optimize=False)
        if progress is not None:
            progress(iw+1, len(omega))
    return BeamRTEResult(cells, omega.copy(), values, settings,
                        dict(seconds=perf_counter()-start, preparation_seconds=prepared_seconds,
                             radial_order=radial_order, polar_order=polar_order,
                             exact_first_error=first_error.tolist(), k_nodes=len(k),
                             source=source, settings=asdict(settings),
                             angular_backend=angular_backend))


def sphere_ballistic_probability(medium, centre_m, radius_m, *, direction=(0., 0., 1.)):
    """One collimated photon: survival to first glass entry, with unit response.

    Unlike cell fluence this is a dimensionless first-entry probability.
    It is used only to illustrate Q0 and never substitutes for scattered RTE.
    """
    from ..receiver import SphereSurface
    centre = np.asarray(centre_m, float)
    if centre.shape != (3,) or not np.isfinite(centre).all():
        raise ValueError("centre must be a finite 3-vector")
    if np.linalg.norm(centre) <= radius_m:
        raise ValueError("source must be outside the sphere")
    entry = SphereSurface(radius_m).entry_distance(-centre, direction)
    return float(np.exp(-medium.extinction_per_m*entry))
