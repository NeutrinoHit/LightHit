"""Time-bin integrals. Detector timing is applied AFTER the transport solve."""
from dataclasses import dataclass
from time import perf_counter
import numpy as np
from scipy.special import ndtr
from .single import single_bins


def frequency_weights(omega):
    omega = np.asarray(omega, float)
    if (omega.ndim != 1 or len(omega) < 2 or not np.isfinite(omega).all()
            or omega[0] != 0 or np.any(np.diff(omega) <= 0)):
        raise ValueError("Time inversion requires at least two increasing frequencies starting at zero")
    step = np.diff(omega)
    return np.r_[step[0] / 2, (step[:-1] + step[1:]) / 2, step[-1] / 2]


def inverse_bins(omega, spectrum, edges_ns, sigma_ns=0.0):
    """Half-axis Fourier integral with exact integration of exp(-i*omega*t) per bin.

    spectrum: (frequency, observation). Result: (observation, bin).
    A finite frequency interval and quadrature may create signed ringing.
    Values are neither clipped nor renormalized.
    """
    omega = np.asarray(omega, float)
    edges = np.asarray(edges_ns, float)
    spectrum = np.asarray(spectrum, complex)
    if edges.ndim != 1 or len(edges) < 2 or not np.isfinite(edges).all() or np.any(np.diff(edges) <= 0):
        raise ValueError("Time edges must be finite and strictly increasing")
    if not np.isfinite(sigma_ns) or sigma_ns < 0:
        raise ValueError("sigma_ns must be finite and nonnegative")
    if spectrum.ndim != 2 or spectrum.shape[0] != len(omega) or not np.isfinite(spectrum).all():
        raise ValueError("spectrum must have shape (frequency, observation)")
    w = frequency_weights(omega)
    width = np.diff(edges)
    center = (edges[:-1] + edges[1:]) / 2
    kernel = (width[:, None] * np.sinc(width[:, None] * omega / (2 * np.pi))
              * np.exp(-1j * center[:, None] * omega)
              * (w * np.exp(-0.5 * (omega * sigma_ns)**2) / np.pi))
    return (kernel @ spectrum).real.T


@dataclass
class TimeProfile:
    edges_ns: np.ndarray
    components: np.ndarray  # observation, bin, [0,1,>=2]
    sigma_ns: float
    elapsed_s: float
    diagnostics: dict

    @property
    def values_per_m2(self):
        return self.components.sum(axis=-1)

    @property
    def rate_per_m2_ns(self):
        return self.values_per_m2 / np.diff(self.edges_ns)

    @property
    def centers_ns(self):
        return (self.edges_ns[:-1] + self.edges_ns[1:]) / 2


def time_bins(result, edges_ns, sigma_ns=0.0):
    start = perf_counter()
    edges = np.asarray(edges_ns, float)
    multi = inverse_bins(result.omega_per_ns, result.components[:, :, 2], edges, sigma_ns)
    parts = np.zeros((*multi.shape, 3))
    parts[:, :, 2] = multi
    for d, radius in enumerate(result.radii_m):
        cosine = None if result.cosines is None else float(result.cosines[d])
        if result.photons:
            parts[d, :, 1] = result.photons * single_bins(
                edges - result.emission_time_ns, radius, cosine, result.medium, sigma_ns,
                backend=result.single_backend)
        if result.direction is None:
            q0 = (result.photons * np.exp(-result.medium.extinction_per_m * radius)
                  / (4 * np.pi * radius * radius))
            front = result.front_time_ns[d]
            if sigma_ns:
                parts[d, :, 0] = q0 * (ndtr((edges[1:] - front) / sigma_ns)
                                       - ndtr((edges[:-1] - front) / sigma_ns))
            else:
                # Half-open [left,right) bins; final endpoint is also right-open.
                index = np.searchsorted(edges, front, side="right") - 1
                if 0 <= index < len(edges) - 1:
                    parts[d, index, 0] = q0
    full = parts.sum(axis=-1)
    charge = result.charge_per_m2
    denom = np.where(np.abs(charge) > 0, np.abs(charge), 1)
    before = edges[1:][None, :] <= result.front_time_ns[:, None]
    diagnostics = {
        "negative_mass_fraction": (np.maximum(-full, 0).sum(axis=1) / denom).tolist(),
        "window_charge_fraction": (full.sum(axis=1) / denom).tolist(),
        "prefront_absolute_mass_fraction": ((np.abs(full) * before).sum(axis=1) / denom).tolist(),
        "prefront_note": ("Gaussian readout produces a physical prefront tail" if sigma_ns else
                          "Exact unsmeared signal is causal; nonzero prefront mass is numerical"),
        "low_orders": "0 and 1 integrated in physical time; only >=2 Fourier-inverted",
    }
    return TimeProfile(edges.copy(), parts, float(sigma_ns), perf_counter() - start, diagnostics)
