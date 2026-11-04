"""Stationary P_N eigenmodes, collimated plane source in an infinite medium.

Geometry. A horizontally uniform collimated beam of unit flux (1 W m^-2, or one
photon s^-1 m^-2) crosses the plane z = 0 along +z (z is depth, mu the cosine
of the angle to +z). The medium (mu_a, mu_s, Henyey-Greenstein g) fills all z:
there is no boundary at z = 0, so light scattered to z < 0 travels on in the
same medium. The field depends on z and mu only. Everything is per unit
injected flux; multiply by the incident spectral flux for absolute values.

Order 0 is exact: the unscattered beam has flux exp(-mu_t z) for z >= 0,
mu_t = mu_a + mu_s, radiance exp(-mu_t z) delta(Omega - z_hat).

Scattered field. With phi_l(z) = int L P_l(mu) dOmega,
L = sum_l (2l+1)/(4 pi) phi_l P_l and the HG moments g^l, the transport equation
mu dL/dz + mu_t L = mu_s int p L dOmega' + mu_s exp(-mu_t z) H(z) p(mu)
(first-collision source of the beam) projects onto P_k as

    (k+1) phi'_{k+1} + k phi'_{k-1} + (2k+1) A_k phi_k = (2k+1) q_k(z),
    A_k = mu_t - mu_s g^k,   q_k = mu_s g^k exp(-mu_t z) H(z),

k = 0..N with phi_{N+1} = 0 (P_N closure). In matrix form T phi' + M phi = D q,
T symmetric tridiagonal with T_{j-1,j} = j, M = diag((2k+1) A_k), D = diag(2k+1).
With phi = M^{-1/2} U c and S = M^{-1/2} T M^{-1/2} = U diag(lambda) U^T,

    S_{j-1,j} = j / sqrt((4 j^2 - 1) A_{j-1} A_j),

the same tridiagonal matrix as ``stationary_modes`` (its point-source
eigen-lengths are these lambda). Each mode obeys lambda_j c_j' + c_j = s_j(z),
s_j = sigma_j exp(-mu_t z) H(z), sigma_j = mu_s sum_k U_kj sqrt((2k+1)/A_k) g^k:
the projection of the collimated source onto the modes. The bounded
infinite-medium solution is explicit:

    lambda > 0, z >= 0:  c = sigma (exp(-mu_t z) - exp(-z/lambda)) / (1 - mu_t lambda)
                           = sigma (z/lambda) exp(-a z) E(d z),
                         a = min(mu_t, 1/lambda), d = |mu_t - 1/lambda|,
                         E(x) = (1 - exp(-x))/x;   c = 0 for z < 0;
    lambda = -l < 0:     c = sigma exp(-mu_t z) / (1 + mu_t l)   (z >= 0),
                         c = sigma exp(z/l) / (1 + mu_t l)       (z < 0);
    lambda = 0 (only when N+1 is odd): c = s(z), a local term.

All terms of one mode are products of non-negative factors (expm1 form), so no
difference of nearly equal exponentials is formed; at depth one mode carries
the field and its relative precision is that of its own exponential (1e-15
relative and far below the surface value). The only sums of mixed sign are the
sums over modes; ``cancellation`` reports sum|term| / |sum| so that this is
measured, not assumed. A Fourier inversion in z would instead be limited by
its absolute error relative to the surface value.

The slowest mode lambda_max gives K_inf = 1/lambda_max, identical to
``StationaryModes.leading_attenuation_per_m`` at the same degree; its angular
shape is the asymptotic radiance distribution. Integrating the k = 0 equation
over all z gives mu_a int E_0 dz = 1 (beam included): every injected photon is
absorbed somewhere. Use odd N (an even number of moments, no lambda = 0 local
term) and check convergence in N.

Method, primary sources: the spherical-harmonics (P_N) method in plane
geometry (J. C. Mark, The Spherical Harmonics Method I and II, National Research
Council of Canada, Chalk River reports CRT-340 and CRT-338, 1944-45; B. Davison,
Neutron Transport Theory, Clarendon Press, Oxford 1957); eigenvalues of the P_N
tridiagonal system as the discrete spectrum (R. D. M. Garcia, C. E. Siewert,
J. Quant. Spectrosc. Radiat. Transfer 42, 385-394, 1989); particular solution
of the P_N equations for an inhomogeneous (beam) source (C. E. Siewert,
J. R. Thomas, JQSRT 43, 433-436, 1990); discrete eigenvalue and asymptotic
solution (K. M. Case, P. F. Zweifel, Linear Transport Theory, Addison-Wesley
1967); asymptotic underwater radiance (R. W. Preisendorfer, J. Mar. Res. 18,
1-9, 1959).
"""
from dataclasses import dataclass
from time import perf_counter
import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.special import roots_legendre
from ..medium import Medium
from ..angular import _degree


def _legendre(degree, mu):
    """P_0..P_degree at mu, shape (degree+1, len(mu)), upward recurrence."""
    mu = np.asarray(mu, float)
    p = np.empty((degree + 1, mu.size))
    p[0] = 1.0
    if degree:
        p[1] = mu
    for n in range(1, degree):
        p[n + 1] = ((2 * n + 1) * mu * p[n] - n * p[n - 1]) / (n + 1)
    return p


def _depths(depth_m):
    z = np.atleast_1d(np.asarray(depth_m, float))
    if z.ndim != 1 or not len(z) or not np.isfinite(z).all():
        raise ValueError("depth_m must be finite and one-dimensional")
    return z


def _relative_expm1(x):
    """(1 - exp(-x)) / x for x >= 0, equal to 1 at x = 0."""
    out = np.ones_like(x)
    np.divide(-np.expm1(-x), x, out=out, where=x > 0)
    return out


@dataclass(frozen=True)
class Irradiance:
    """Per unit injected flux. ``*_scattered`` excludes the unscattered beam."""
    depth_m: np.ndarray
    unscattered: np.ndarray
    downward_scattered: np.ndarray
    upward_scattered: np.ndarray
    scalar_scattered: np.ndarray

    @property
    def downward(self):
        return self.downward_scattered + self.unscattered

    @property
    def upward(self):
        return self.upward_scattered

    @property
    def scalar(self):
        return self.scalar_scattered + self.unscattered

    @property
    def net(self):
        return self.downward - self.upward


@dataclass(frozen=True)
class PlaneParallelModes:
    medium: Medium
    degree: int
    lengths_m: np.ndarray       # signed eigen-lengths lambda_j (m); 0 for a local term
    vectors: np.ndarray         # phi_l per unit c_j: (degree+1, modes)
    source: np.ndarray          # sigma_j, per unit injected flux
    downward_weights: np.ndarray
    upward_weights: np.ndarray
    build_seconds: float

    @classmethod
    def build(cls, medium: Medium, degree: int = 255):
        """Diagonalize S = M^-1/2 T M^-1/2 and project the beam source."""
        degree = _degree(degree, "degree")
        if degree < 1:
            raise ValueError("degree must be at least 1")
        t0 = perf_counter()
        ell = np.arange(degree + 1)
        attenuation = medium.extinction_per_m - medium.scattering_per_m * medium.g ** ell
        j = ell[1:]
        off = j / np.sqrt(4.0 * j * j - 1) / np.sqrt(attenuation[:-1] * attenuation[1:])
        length, u = eigh_tridiagonal(np.zeros(degree + 1), off)
        small = np.abs(length) <= 100 * np.finfo(float).eps * np.max(np.abs(length))
        length = np.where(small, 0.0, length)
        vectors = u / np.sqrt((2 * ell + 1) * attenuation)[:, None]
        source = medium.scattering_per_m * (
            (np.sqrt((2 * ell + 1) / attenuation) * medium.g ** ell) @ u)
        # half-range flux weights (2l+1)/2 int_0^1 mu P_l dmu, exact Gauss rule
        node, weight = roots_legendre(degree // 2 + 2)
        node, weight = (node + 1) / 2, weight / 2
        half = _legendre(degree, node) @ (weight * node)
        down = (2 * ell + 1) / 2 * half
        up = down * np.where(ell % 2 == 0, 1.0, -1.0)
        return cls(medium, degree, length, vectors, source, down @ vectors, up @ vectors,
                   perf_counter() - t0)

    # ----- modes -------------------------------------------------------
    @property
    def leading_attenuation_per_m(self):
        """K_inf = 1/lambda_max; same matrix as StationaryModes at this degree."""
        return float(1 / np.max(self.lengths_m))

    @property
    def leading_index(self):
        return int(np.argmax(self.lengths_m))

    def amplitudes(self, depth_m):
        """c_j(z), shape (depth, modes); every entry is >= 0 times sigma_j."""
        z = _depths(depth_m)
        lam = self.lengths_m
        mut = self.medium.extinction_per_m
        zp = np.maximum(z, 0.0)[:, None]
        out = np.zeros((len(z), len(lam)))
        pos, neg, loc = lam > 0, lam < 0, lam == 0
        if pos.any():
            inv = 1 / lam[pos]
            a = np.minimum(mut, inv)
            d = np.abs(mut - inv)
            out[:, pos] = zp * inv * np.exp(-a * zp) * _relative_expm1(d * zp)
        if neg.any():
            ell = -lam[neg]
            val = np.outer(np.exp(-mut * zp[:, 0]), 1 / (1 + mut * ell))
            above = z < 0
            if above.any():
                val[above] = np.exp(z[above, None] / ell) / (1 + mut * ell)
            out[:, neg] = val
        if loc.any():
            out[:, loc] = (np.where(z >= 0, np.exp(-mut * zp[:, 0]), 0.0))[:, None]
        return out * self.source[None, :]

    def amplitude_derivatives(self, depth_m, amplitudes=None):
        """dc_j/dz, shape (depth, modes); ``amplitudes`` may pass c_j(z)."""
        z = _depths(depth_m)
        c = self.amplitudes(z) if amplitudes is None else amplitudes
        lam = self.lengths_m
        mut = self.medium.extinction_per_m
        out = -mut * c
        pos, neg = lam > 0, lam < 0
        beam = np.where(z >= 0, np.exp(-mut * np.maximum(z, 0.0)), 0.0)
        out[:, pos] = (-c[:, pos] + np.outer(beam, self.source[pos])) / lam[pos]
        above = z < 0
        if above.any() and neg.any():
            sub = out[above]
            sub[:, neg] = c[above][:, neg] / (-lam[neg])
            out[above] = sub
        return out

    # ----- observables -------------------------------------------------
    def unscattered(self, depth_m):
        z = _depths(depth_m)
        return np.where(z >= 0, np.exp(-self.medium.extinction_per_m * np.maximum(z, 0)), 0.0)

    def moments(self, depth_m):
        """phi_l(z) = int L_scattered P_l dOmega, shape (depth, degree+1)."""
        return self.amplitudes(depth_m) @ self.vectors.T

    def irradiance(self, depth_m):
        z = _depths(depth_m)
        c = self.amplitudes(z)
        return Irradiance(z, self.unscattered(z), c @ self.downward_weights,
                          c @ self.upward_weights, c @ self.vectors[0])

    def cancellation(self, depth_m):
        """sum|terms|/|sum| of the mode sums for E_d, E_u, E_0 (scattered)."""
        c = self.amplitudes(depth_m)
        out = {}
        for name, w in (("downward", self.downward_weights), ("upward", self.upward_weights),
                        ("scalar", self.vectors[0])):
            t = c * w[None, :]
            with np.errstate(divide="ignore", invalid="ignore"):
                out[name] = np.abs(t).sum(1) / np.abs(t.sum(1))
        return out

    def diffuse_attenuation(self, depth_m):
        """K_d(z) = -d ln E_d / dz of the total downward irradiance (beam included)."""
        z = _depths(depth_m)
        c = self.amplitudes(z)
        dc = self.amplitude_derivatives(z, c)
        un = self.unscattered(z)
        ed = c @ self.downward_weights + un
        ded = dc @ self.downward_weights - self.medium.extinction_per_m * un
        return -ded / ed

    def angular_basis(self, mu):
        """r_j(mu) = sum_l (2l+1)/(4 pi) V_lj P_l(mu), shape (len(mu), modes)."""
        mu = np.atleast_1d(np.asarray(mu, float))
        if np.any(np.abs(mu) > 1):
            raise ValueError("mu must lie in [-1, 1]")
        ell = np.arange(self.degree + 1)
        return (_legendre(self.degree, mu).T * ((2 * ell + 1) / (4 * np.pi))) @ self.vectors

    def radiance(self, depth_m, mu, *, basis=None):
        """Scattered radiance per unit flux (sr^-1), shape (depth, len(mu)).

        The unscattered beam is a delta along mu = 1 (``unscattered``).
        ``basis`` may pass a precomputed ``angular_basis(mu)``.
        """
        r = self.angular_basis(mu) if basis is None else basis
        return self.amplitudes(depth_m) @ r.T

    def asymptotic_radiance(self, mu):
        """Shape of the slowest mode, normalised to unit scalar irradiance."""
        k = self.leading_index
        return self.angular_basis(mu)[:, k] / self.vectors[0, k]

    @property
    def asymptotic_mean_cosine(self):
        """E_net/E_0 of the slowest mode; equals mu_a lambda_max (k=0 equation)."""
        k = self.leading_index
        return float(self.vectors[1, k] / self.vectors[0, k])

    def integrated_scalar(self):
        """Analytic int E_0 dz over all z (m * per unit flux): beam and scattered.

        Every mode integrates to sigma_j/mu_t, so mu_a times this is 1 when the
        mode algebra is consistent. Also returns the part at z < 0.
        """
        mut = self.medium.extinction_per_m
        total = 1 / mut + (self.source / mut) @ self.vectors[0]
        neg = self.lengths_m < 0
        ell = -self.lengths_m[neg]
        above = (self.source[neg] * ell / (1 + mut * ell)) @ self.vectors[0, neg]
        return float(total), float(above)


def henyey_greenstein(mu, g):
    return (1 - g * g) / (4 * np.pi * (1 + g * g - 2 * g * np.asarray(mu, float)) ** 1.5)


def single_scattered_radiance(depth_m, mu, medium: Medium):
    """Exact once-scattered radiance of the collimated beam (sr^-1, per unit flux).

    Independent of the modes: mu_s p(mu) times the attenuated path integral,
    mu > 0, z >= 0: exp(-mu_t z) (1 - exp(-mu_t z (1-mu)/mu)) / (mu_t (1-mu));
    mu < 0: exp(-mu_t z) / (mu_t (1+|mu|)) for z >= 0 and
            exp(-mu_t |z|/|mu|) / (mu_t (1+|mu|)) for z < 0.
    """
    z = _depths(depth_m)[:, None]
    m = np.atleast_1d(np.asarray(mu, float))[None, :]
    mut = medium.extinction_per_m
    p = henyey_greenstein(m, medium.g) * medium.scattering_per_m
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        x = np.where(m > 0, mut * np.maximum(z, 0) * (1 - m) / np.where(m > 0, m, 1), 0.0)
        down = np.exp(-mut * np.maximum(z, 0)) * np.maximum(z, 0) * mut / np.where(m > 0, m, 1) \
            * _relative_expm1(x) / mut
        up = np.where(z >= 0, np.exp(-mut * np.maximum(z, 0)),
                      np.exp(-mut * np.abs(np.minimum(z, 0)) / np.abs(np.where(m < 0, m, -1)))) \
            / (mut * (1 + np.abs(m)))
    value = np.where(m > 0, np.where(z >= 0, down, 0.0), np.where(m < 0, up, 0.0))
    return value * p
