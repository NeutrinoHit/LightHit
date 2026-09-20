"""Coordinate-space, full-HG first scattering of one monochromatic photon.

Both directed and isotropic point flashes are supported. The detector measures
scalar fluence per unit effective area. No finite sphere or PDE is included.
"""
from functools import lru_cache
import numpy as np
from scipy.integrate import quad, quad_vec
from scipy.special import roots_legendre, ndtr


def hg_phase(cosine, g):
    cosine = np.clip(np.asarray(cosine, dtype=float), -1, 1)
    return (1 - g * g) / (4 * np.pi * (1 + g * g - 2 * g * cosine) ** 1.5)


@lru_cache(maxsize=4)
def _unit_rule(order=48):
    x, w = roots_legendre(order)
    return (x + 1) / 2, w / 2


def _single_scattering_path_rate_numpy(path_m, radius_m, cosine,
                                       speed_m_per_ns, scattering_per_m,
                                       extinction_per_m, g, nodes, weights):
    """Scalar reference kernel used by adaptive quadrature without Numba."""
    if path_m <= radius_m:
        return 0.0
    attenuation = np.exp(-extinction_per_m * path_m)
    if not np.isnan(cosine):
        a = ((path_m - radius_m) * (path_m + radius_m)
             / (2 * (path_m - radius_m * cosine)))
        b = path_m - a
        scattering_cosine = (radius_m * cosine - a) / b
        return float(scattering_per_m * speed_m_per_ns
                     * hg_phase(scattering_cosine, g) * attenuation
                     / (b * (path_m - radius_m * cosine)))
    y_max = 0.5 * np.log1p(2 * radius_m / (path_m - radius_m))
    if g == 0:
        average = 1 / (4 * np.pi)
    else:
        delta = (path_m - radius_m) * (path_m + radius_m) / path_m**2
        angle = 1 - 2 * delta * np.cosh(y_max * nodes)**2
        average = float(hg_phase(angle, g) @ weights)
    return float(2 * speed_m_per_ns * scattering_per_m * attenuation
                 * y_max * average / (radius_m * path_m))


def _check_single_backend(backend):
    if backend not in ("auto", "numpy", "numba"):
        raise ValueError("single_backend must be 'auto', 'numpy' or 'numba'")
    return backend


@lru_cache(maxsize=3)
def _path_rate_backend(backend="auto"):
    """Return (scalar kernel, resolved backend), importing Numba only on demand."""
    backend = _check_single_backend(backend)
    if backend == "numpy":
        return _single_scattering_path_rate_numpy, "numpy"
    try:
        from .single_fast import single_scattering_path_rate
    except ModuleNotFoundError as exc:
        if exc.name != "numba":
            raise
        if backend == "numba":
            raise ImportError(
                "The numba single-scattering backend requires the optional "
                "accelerate extra: python -m pip install 'lighthit[accelerate]'"
            ) from exc
        return _single_scattering_path_rate_numpy, "numpy"
    return single_scattering_path_rate, "numba"


def single_quadrature_backend(backend="auto"):
    """Resolve the exact first-order scalar backend without running a solve."""
    return _path_rate_backend(backend)[1]


def _quadrature_geometry(radius_m, cosine):
    radius = float(radius_m)
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("Finite radius_m > 0 required")
    if cosine is None:
        return radius, np.nan
    cosine = float(cosine)
    if not np.isfinite(cosine) or not -1 <= cosine < 1:
        raise ValueError("Directed point receiver requires -1 <= cosine < 1")
    return radius, cosine


def single_scattering_rate(time_ns, radius_m, cosine, medium):
    """First-order density [m^-2 ns^-1], with exact HG (no Legendre cutoff).

    cosine = r_hat . s0 for a directed photon. cosine=None averages uniformly
    over all initial photon directions. The isotropic front has an integrable
    logarithmic singularity; its point value is returned as +inf.
    """
    t = np.asarray(time_ns, dtype=float)
    if not np.isfinite(t).all() or not np.isfinite(radius_m) or radius_m <= 0:
        raise ValueError("Finite times and radius_m > 0 required")
    if cosine is not None and not (-1 <= cosine < 1):
        raise ValueError("Directed point receiver requires -1 <= cosine < 1")
    out = np.zeros(t.shape, dtype=float)
    if medium.scattering_per_m == 0:
        return out
    v = medium.speed_m_per_ns
    r = float(radius_m)
    S = v * t
    mask = S > r
    path = S[mask]
    if not path.size:
        if cosine is None:
            out[S == r] = np.inf
        return out
    attenuation = np.exp(-medium.extinction_per_m * path)
    if cosine is not None:
        a = (path - r) * (path + r) / (2 * (path - r * cosine))
        b = path - a
        scattering_cosine = (r * cosine - a) / b
        out[mask] = (
            medium.scattering_per_m * v * hg_phase(scattering_cosine, medium.g)
            * attenuation / (b * (path - r * cosine))
        )
    else:
        # Ellipsoidal one-scatter integral: a=(S/2)*(1+tanh y).
        # This variable removes 1/(a*b) from the quadrature measure.
        Y = 0.5 * np.log1p(2 * r / (path - r))
        if medium.g == 0:
            average = np.full_like(path, 1 / (4 * np.pi))
        else:
            node, weight = _unit_rule()
            delta = (path - r) * (path + r) / path**2
            angle = 1 - 2 * delta[:, None] * np.cosh(Y[:, None] * node)**2
            average = hg_phase(angle, medium.g) @ weight
        out[mask] = 2 * v * medium.scattering_per_m * attenuation * Y * average / (r * path)
        out[S == r] = np.inf
    return out


def single_spectrum(omega, radius_m, cosine, medium, *, backend="auto"):
    """Frequency spectrum [m^-2]; 1-D adaptive quadrature in excess path.

    Omitted excess path has exp(-36) extinction. The numerical integration
    estimate returned separately does not include that tiny finite-cutoff tail.
    """
    omega = np.asarray(omega, float)
    if omega.ndim != 1 or not len(omega) or not np.isfinite(omega).all():
        raise ValueError("omega must be a finite nonempty 1-D array")
    path_rate, resolved_backend = _path_rate_backend(backend)
    if medium.scattering_per_m == 0:
        return np.zeros_like(omega, complex), 0.0
    v = medium.speed_m_per_ns
    r, cosine_value = _quadrature_geometry(radius_m, cosine)
    cutoff = 36 / medium.extinction_per_m
    scale = np.exp(-medium.extinction_per_m * r) / (4 * np.pi * r * r)
    nodes, weights = _unit_rule()

    if resolved_backend == "numba":
        from .single_fast import single_spectrum_integrand

        def integrand(excess):
            return single_spectrum_integrand(
                r + excess, r, cosine_value, v, medium.scattering_per_m,
                medium.extinction_per_m, medium.g, nodes, weights, omega)
    else:
        def integrand(excess):
            path = r + excess
            # Some quadrature implementations evaluate the endpoint.  The public
            # isotropic density is +inf there, but a single endpoint value has zero
            # measure and must not turn an integrable logarithm into a NaN result.
            if path <= r:
                return np.zeros_like(omega, dtype=complex)
            t = path / v
            value = path_rate(path, r, cosine_value, v,
                              medium.scattering_per_m, medium.extinction_per_m,
                              medium.g, nodes, weights) / v
            return value * np.exp(1j * omega * t)

    result, error = quad_vec(integrand, 0, cutoff, epsabs=max(scale * 1e-11, 1e-300),
                             epsrel=2e-10, limit=10000)
    return result, float(error)


def single_bins(edges, radius_m, cosine, medium, sigma_ns=0.0, *, backend="auto"):
    """Integrate the first order in physical time, without a frequency cutoff.

    edges are relative to emission time. A Gaussian readout, if requested,
    is integrated over each bin through differences of normal CDFs.
    """
    edges = np.asarray(edges, float)
    if edges.ndim != 1 or len(edges) < 2 or not np.isfinite(edges).all() or np.any(np.diff(edges) <= 0):
        raise ValueError("edges must be finite and strictly increasing")
    if not np.isfinite(sigma_ns) or sigma_ns < 0:
        raise ValueError("sigma_ns must be nonnegative")
    path_rate, _ = _path_rate_backend(backend)
    if medium.scattering_per_m == 0:
        return np.zeros(len(edges) - 1)
    radius_m, cosine_value = _quadrature_geometry(radius_m, cosine)
    speed = medium.speed_m_per_ns
    front = radius_m / speed
    scale = np.exp(-medium.extinction_per_m * radius_m) / (4 * np.pi * radius_m**2)
    eps = max(scale * 1e-10, 1e-300)
    nodes, weights = _unit_rule()

    def rate(t):
        return path_rate(speed * t, radius_m, cosine_value, speed,
                         medium.scattering_per_m, medium.extinction_per_m,
                         medium.g, nodes, weights)

    if sigma_ns == 0:
        result = np.zeros(len(edges) - 1)
        for j, (lo, hi) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
            if hi > front:
                result[j] = quad(rate, max(lo, front), hi, epsabs=eps,
                                 epsrel=1e-9, limit=200)[0]
        return result
    upper = min(front + 36 / (medium.extinction_per_m * speed),
                edges[-1] + 10 * sigma_ns)
    if upper <= front:
        return np.zeros(len(edges) - 1)

    def integrand(t):
        weight = ndtr((edges[1:] - t) / sigma_ns) - ndtr((edges[:-1] - t) / sigma_ns)
        return rate(t) * weight

    result, _ = quad_vec(integrand, front, upper, epsabs=eps, epsrel=1e-9,
                         limit=10000)
    return result
