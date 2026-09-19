"""Optional Numba kernel for the exact full-HG single-scattering term.

The adaptive integrations remain in :mod:`scipy.integrate`; this module only
compiles the scalar rate evaluated at each quadrature node.  In particular it
does not change the 48-point Gauss--Legendre rule used for the isotropic-source
ellipsoidal average, any integration tolerance, or the full HG phase function.

Importing this module requires the optional ``accelerate`` dependency.  The
public functions in :mod:`lighthit.single` select it lazily and fall back to
their NumPy/SciPy implementation when Numba is unavailable.
"""
import numpy as np
from numba import njit


@njit(cache=True, fastmath=False, inline="always")
def _hg_phase_scalar(cosine, g):
    """Scalar Henyey--Greenstein density with the public function's clipping."""
    if cosine < -1.0:
        cosine = -1.0
    elif cosine > 1.0:
        cosine = 1.0
    return ((1.0 - g * g)
            / (4.0 * np.pi * (1.0 + g * g - 2.0 * g * cosine) ** 1.5))


@njit(cache=True, fastmath=False, inline="always")
def single_scattering_path_rate(path_m, radius_m, cosine, speed_m_per_ns,
                                scattering_per_m, extinction_per_m, g,
                                nodes, weights):
    """Exact first-order rate at one total path length.

    ``cosine=nan`` denotes an isotropic source.  The light-front value is set
    to zero for quadrature purposes: the isotropic density is infinite at that
    one point but its logarithmic singularity is integrable, so changing the
    endpoint value does not change either integral.
    """
    if path_m <= radius_m:
        return 0.0

    attenuation = np.exp(-extinction_per_m * path_m)
    if not np.isnan(cosine):
        a = ((path_m - radius_m) * (path_m + radius_m)
             / (2.0 * (path_m - radius_m * cosine)))
        b = path_m - a
        scattering_cosine = (radius_m * cosine - a) / b
        return (scattering_per_m * speed_m_per_ns
                * _hg_phase_scalar(scattering_cosine, g) * attenuation
                / (b * (path_m - radius_m * cosine)))

    y_max = 0.5 * np.log1p(2.0 * radius_m / (path_m - radius_m))
    if g == 0.0:
        average = 1.0 / (4.0 * np.pi)
    else:
        delta = ((path_m - radius_m) * (path_m + radius_m)
                 / (path_m * path_m))
        average = 0.0
        for index in range(len(nodes)):
            value = np.cosh(y_max * nodes[index])
            angle = 1.0 - 2.0 * delta * value * value
            average += _hg_phase_scalar(angle, g) * weights[index]
    return (2.0 * speed_m_per_ns * scattering_per_m * attenuation
            * y_max * average / (radius_m * path_m))


@njit(cache=True, fastmath=False)
def single_spectrum_integrand(path_m, radius_m, cosine, speed_m_per_ns,
                              scattering_per_m, extinction_per_m, g,
                              nodes, weights, omega_per_ns):
    """Fused scalar rate and complex phase for one adaptive path node."""
    out = np.empty(len(omega_per_ns), dtype=np.complex128)
    if path_m <= radius_m:
        out[:] = 0.0
        return out
    value = single_scattering_path_rate(
        path_m, radius_m, cosine, speed_m_per_ns, scattering_per_m,
        extinction_per_m, g, nodes, weights) / speed_m_per_ns
    time_ns = path_m / speed_m_per_ns
    for index in range(len(omega_per_ns)):
        phase = omega_per_ns[index] * time_ns
        out[index] = value * (np.cos(phase) + 1j * np.sin(phase))
    return out


__all__ = ["single_scattering_path_rate", "single_spectrum_integrand"]
