"""Independent stationary P_N eigenmode check, isotropic point flash/source.

This is a DIFFERENT discretization from angular.py: the entire angular system
is truncated at degree N. Its positive eigenvalue sum eliminates the oscillatory
radial integral, but needs its own N-convergence study. No free-tail closure is
claimed here. It returns charge and outward current, not a time profile or a
full angular distribution. The construction is related to classical transport
spectral/rotated-reference-frame methods; see docs/research/absorption.md.
"""
from dataclasses import dataclass
from time import perf_counter
import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.special import roots_legendre
from scipy.integrate import quad_vec
from ..medium import Medium
from ..angular import _degree


@dataclass(frozen=True)
class StationaryModes:
    medium: Medium
    degree: int
    lengths_m: np.ndarray
    weights: np.ndarray
    build_seconds: float

    @classmethod
    def build(cls, medium: Medium, degree: int = 128):
        """Diagonalize A^-1/2 C A^-1/2; A_l=mu_t-mu_s*g**l."""
        degree = _degree(degree, "degree")
        if degree < 1:
            raise ValueError("degree must be at least 1")
        t0 = perf_counter()
        ell = np.arange(degree + 1)
        attenuation = medium.extinction_per_m - medium.scattering_per_m * medium.g ** ell
        j = np.arange(1, degree + 1)
        off = (j / np.sqrt(4 * j*j - 1)
               / np.sqrt(attenuation[:-1] * attenuation[1:]))
        length, vectors = eigh_tridiagonal(np.zeros(degree + 1), off)
        # The spectrum comes in +/- pairs. A zero eigenvalue contributes
        # only a distribution at r=0; that point is outside this API.
        keep = length > 100 * np.finfo(float).eps * np.max(np.abs(length))
        return cls(medium, degree, length[keep], vectors[0, keep]**2,
                   perf_counter() - t0)

    def scalar_current(self, radii_m, *, leading_only: bool = False):
        """Return (scalar charge Q, outward radial current F), each [m^-2].

        Q = sum w_n exp(-r/lambda_n)/(2*pi*mu_a*r*lambda_n**2).
        F follows photon balance outside the source. F/Q is mean radial cosine.
        leading_only retains the longest attenuation length, not all scattering.
        """
        r = np.atleast_1d(np.asarray(radii_m, float))
        if r.ndim != 1 or not len(r) or not np.isfinite(r).all() or np.any(r <= 0):
            raise ValueError("radii_m must be finite, positive and one-dimensional")
        length = self.lengths_m[-1:] if leading_only else self.lengths_m
        weight = self.weights[-1:] if leading_only else self.weights
        terms = (weight / length**2)[None, :] * np.exp(-r[:, None] / length)
        terms /= 2 * np.pi * self.medium.absorption_per_m * r[:, None]
        scalar = terms.sum(axis=1)
        current = (terms * (self.medium.absorption_per_m * length)[None, :]
                   * (1 + length[None, :] / r[:, None])).sum(axis=1)
        return scalar, current

    @property
    def leading_attenuation_per_m(self):
        return float(1 / self.lengths_m[-1])

    @property
    def leading_mean_cosine(self):
        return self.medium.absorption_per_m / self.leading_attenuation_per_m

    @property
    def isolated_leading_mode(self):
        """A necessary, not sufficient, indicator of an isolated continuum-limit pole."""
        return self.leading_attenuation_per_m < self.medium.extinction_per_m * (1 - 1e-10)


def single_scalar_current(radius_m: float, medium: Medium, angular_order: int = 96):
    """Positive coordinate-space one-scatter integral for isotropic emission.

    Independent of cache.py, angular.py and any real-k Fourier inversion.
    Use prolate coordinates S=a+b, a-b=S*tanh(y), 0<=y<=atanh(r/S).
    The symmetric final-direction cosine is (S/r)*(1-delta*cosh(y)**2),
    delta=1-r**2/S**2. The half-domain already includes a<->b symmetry.
    Return Q1 and F1, in m^-2. The omitted excess path has exp(-40) extinction.
    """
    r = float(radius_m)
    if not np.isfinite(r) or r <= 0:
        raise ValueError("radius_m must be finite and positive")
    angular_order = _degree(angular_order, "angular_order")
    if angular_order < 4:
        raise ValueError("angular_order must be at least 4")
    if medium.scattering_per_m == 0:
        return np.zeros(2)
    node, weight = roots_legendre(angular_order)
    node, weight = (node+1)/2, weight/2
    mu_t, mu_s, g = medium.extinction_per_m, medium.scattering_per_m, medium.g
    def integrand(excess):
        S = r + excess
        Y = 0.5 * np.log1p(2*r/excess)
        delta = excess * (2*r+excess) / S**2
        z = delta * np.cosh(Y*node)**2
        cos_scatter = np.clip(1-2*z, -1, 1)
        phase = (1-g*g) / (4*np.pi*(1+g*g-2*g*cos_scatter)**1.5)
        mu_arrival = S/r*(1-z)
        common = 2*mu_s*np.exp(-mu_t*excess)*Y/(r*S)
        return common * np.array([weight @ phase, weight @ (phase*mu_arrival)])
    value, _ = quad_vec(integrand, 0, 40/mu_t, epsabs=1e-13,
                        epsrel=2e-10, limit=1000)
    return value * np.exp(-mu_t*r)


def collision_components(radii_m, modes: StationaryModes):
    """Split total stationary Q,F into 0,1,>=2 via coordinate first order.

    Subtraction is used ONLY as a diagnostic. If >=2 is a tiny residual,
    increase N and check against a non-subtractive method before trusting it.
    Shape of each returned array is (radius, 3).
    """
    r = np.atleast_1d(np.asarray(radii_m, float))
    total_q, total_f = modes.scalar_current(r)
    q0 = np.exp(-modes.medium.extinction_per_m*r)/(4*np.pi*r*r)
    single = np.array([single_scalar_current(x, modes.medium) for x in r])
    q = np.column_stack([q0, single[:, 0], total_q-q0-single[:, 0]])
    f = np.column_stack([q0, single[:, 1], total_f-q0-single[:, 1]])
    return q, f
