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


def single_spectrum(omega, radius_m, cosine, medium):
    """Frequency spectrum [m^-2]; 1-D adaptive quadrature in excess path.

    Omitted excess path has exp(-36) extinction. The numerical integration
    estimate returned separately does not include that tiny finite-cutoff tail.
    """
    omega = np.asarray(omega, float)
    if omega.ndim != 1 or not len(omega) or not np.isfinite(omega).all():
        raise ValueError("omega must be a finite nonempty 1-D array")
    if medium.scattering_per_m == 0:
        return np.zeros_like(omega, complex), 0.0
    v = medium.speed_m_per_ns
    r = float(radius_m)
    cutoff = 36 / medium.extinction_per_m
    scale = np.exp(-medium.extinction_per_m * r) / (4 * np.pi * r * r)

    def integrand(excess):
        t = (r + excess) / v
        value = float(single_scattering_rate(t, r, cosine, medium)) / v
        return value * np.exp(1j * omega * t)

    result, error = quad_vec(integrand, 0, cutoff, epsabs=max(scale * 1e-11, 1e-300),
                             epsrel=2e-10, limit=10000)
    return result, float(error)


def single_bins(edges, radius_m, cosine, medium, sigma_ns=0.0):
    """Integrate the first order in physical time, without a frequency cutoff.

    edges are relative to emission time. A Gaussian readout, if requested,
    is integrated over each bin through differences of normal CDFs.
    """
    edges = np.asarray(edges, float)
    if edges.ndim != 1 or len(edges) < 2 or not np.isfinite(edges).all() or np.any(np.diff(edges) <= 0):
        raise ValueError("edges must be finite and strictly increasing")
    if not np.isfinite(sigma_ns) or sigma_ns < 0:
        raise ValueError("sigma_ns must be nonnegative")
    if medium.scattering_per_m == 0:
        return np.zeros(len(edges) - 1)
    front = radius_m / medium.speed_m_per_ns
    scale = np.exp(-medium.extinction_per_m * radius_m) / (4 * np.pi * radius_m**2)
    eps = max(scale * 1e-10, 1e-300)
    if sigma_ns == 0:
        result = np.zeros(len(edges) - 1)
        for j, (lo, hi) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
            if hi > front:
                result[j] = quad(lambda t: float(single_scattering_rate(t, radius_m, cosine, medium)),
                                 max(lo, front), hi, epsabs=eps, epsrel=1e-9, limit=200)[0]
        return result
    upper = min(front + 36 / (medium.extinction_per_m * medium.speed_m_per_ns),
                edges[-1] + 10 * sigma_ns)
    if upper <= front:
        return np.zeros(len(edges) - 1)

    def integrand(t):
        weight = ndtr((edges[1:] - t) / sigma_ns) - ndtr((edges[:-1] - t) / sigma_ns)
        return float(single_scattering_rate(t, radius_m, cosine, medium)) * weight

    result, _ = quad_vec(integrand, front, upper, epsabs=eps, epsrel=1e-9,
                         limit=10000)
    return result
