"""Discrete-ordinate half-space reference with analytic collimated sources.

Spatial transform: integral exp(-i q.rho) I d rho. Temporal inverse:
exp(-i omega t). No depth truncation or depth mesh is used. Only decaying
homogeneous modes are retained; the driven solution is integrated exactly
in depth. The scattering phase is the explicitly configured finite HG
Legendre expansion. Angular and transverse inversion convergence remain
independent accuracy requirements.
"""
from collections import OrderedDict
from dataclasses import dataclass
import math

import numpy as np
from scipy.linalg import eig, lapack, lu_factor, lu_solve
from scipy.special import eval_legendre, j0, jv, roots_legendre

from .halfspace_boundary import HalfSpaceBoundary


def _integer(value, name, minimum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _depths(value):
    z = np.atleast_1d(np.asarray(value, float))
    if z.ndim != 1 or not np.isfinite(z).all() or np.any(z < 0):
        raise ValueError("depths must be a finite nonnegative vector in metres")
    return z


def _phase(cosine, g, degree):
    x = np.clip(cosine, -1., 1.)
    result = np.zeros_like(x, dtype=float)
    for ell in range(degree + 1):
        result += (2 * ell + 1) * g**ell * eval_legendre(ell, x)
    return result / (4 * np.pi)


def _complex_eig(matrix):
    """Right eigenpairs of a complex matrix by LAPACK zgeev, minimal workspace.

    The minimal workspace keeps the unblocked Hessenberg reduction. With the
    optimal (blocked) workspace some platform LAPACK builds return wrong
    eigenvectors above about 128 unknowns; callers still check the residual.
    """
    n = len(matrix)
    lam, _, vectors, info = lapack.zgeev(matrix, compute_vl=0, lwork=max(1, 2*n))
    if info != 0:
        raise ArithmeticError(f"complex eigensolver failed (info={info})")
    return lam, vectors


def _normalized_legendre(x, degree, order):
    """Spherical-harmonic normalized associated Legendre functions.

    Returns P[m, l, i] = sqrt((2l+1)/(4 pi) (l-m)!/(l+m)!) P_l^m(x_i) for
    0 <= m <= order and m <= l <= degree (zero for l < m), without the
    Condon-Shortley sign; only products of equal order m are used.
    """
    x = np.asarray(x, float)
    sine = np.sqrt(np.maximum(0., (1 - x) * (1 + x)))
    result = np.zeros((order + 1, degree + 1, len(x)))
    diagonal = np.full(len(x), 1 / math.sqrt(4 * np.pi))
    for m in range(order + 1):
        if m:
            diagonal = math.sqrt((2 * m + 1) / (2 * m)) * sine * diagonal
        if m > degree:
            break
        result[m, m] = diagonal
        if m + 1 <= degree:
            result[m, m + 1] = math.sqrt(2 * m + 3) * x * diagonal
        for ell in range(m + 2, degree + 1):
            a = math.sqrt((4 * ell * ell - 1) / (ell * ell - m * m))
            b = math.sqrt((4 * (ell - 1)**2 - 1) / ((ell - 1)**2 - m * m))
            result[m, ell] = a * (x * result[m, ell - 1] - result[m, ell - 2] / b)
    return result


def _driven_exponential(lam, rate, z):
    """(exp(-rate*z)-exp(-lam*z))/(lam-rate), including equal rates."""
    a, b = lam[:, None, None], rate[None, :, None]
    difference = a - b
    positive = difference.real >= 0
    base = np.where(positive, b, a)
    argument = np.where(positive, -difference, difference) * z
    numerator = np.where(positive, -1., 1.) * np.exp(-base * z) * np.expm1(argument)
    limit = np.array(z * np.exp(-base * z), dtype=complex, copy=True)
    return np.divide(numerator, difference, out=limit, where=difference != 0)


@dataclass(frozen=True)
class SurfacePort:
    """An exterior angular port, represented by normalized incident-flux rays.

    Directions point into z>0 in the exterior medium. A receiver observes
    their exact reversed directions. ``collection_factor`` converts its
    weighted mean radiance to collected flux per surface area. For a pencil
    port this factor is one by convention: the receiver is differential in
    direction. Spatial footprints and source-detector offsets are separate.
    """
    directions_external: np.ndarray
    flux_weights: np.ndarray
    collection_factor: float = 1.
    axisymmetric: bool = False

    def __post_init__(self):
        direction = np.array(self.directions_external, dtype=float, copy=True)
        weight = np.array(self.flux_weights, dtype=float, copy=True)
        factor = float(self.collection_factor)
        if (direction.ndim != 2 or direction.shape[1] != 3 or not len(direction)
                or weight.shape != (len(direction),) or not np.isfinite(direction).all()
                or not np.isfinite(weight).all() or np.any(weight < 0)
                or not np.isclose(weight.sum(), 1., rtol=0, atol=1e-12)
                or not np.allclose(np.linalg.norm(direction, axis=1), 1., rtol=0, atol=1e-12)
                or np.any(direction[:, 2] <= 0) or not math.isfinite(factor) or factor <= 0):
            raise ValueError("a port needs unit inward directions, nonnegative unit-sum flux weights and positive collection_factor")
        direction.setflags(write=False)
        weight.setflags(write=False)
        object.__setattr__(self, "directions_external", direction)
        object.__setattr__(self, "flux_weights", weight)
        object.__setattr__(self, "collection_factor", factor)

    @classmethod
    def pencil(cls, direction=(0., 0., 1.)):
        direction = np.asarray(direction, float)
        return cls(direction[None, :], np.ones(1), axisymmetric=bool(np.all(direction[:2] == 0)))

    @classmethod
    def aperture(cls, half_angle_rad=np.pi / 2, *, polar_nodes=6, azimuth_nodes=12):
        angle = float(half_angle_rad)
        if not math.isfinite(angle) or not 0 < angle <= np.pi / 2:
            raise ValueError("aperture half-angle must lie in (0, pi/2]")
        polar_nodes = _integer(polar_nodes, "polar_nodes", 1)
        azimuth_nodes = _integer(azimuth_nodes, "azimuth_nodes", 2)
        x, w = roots_legendre(polar_nodes)
        low = max(0., math.cos(angle))
        mu = low + (x + 1) * (1 - low) / 2
        phi = 2 * np.pi * (np.arange(azimuth_nodes) + .5) / azimuth_nodes
        sine = np.sqrt((1 - mu) * (1 + mu))[:, None]
        direction = np.stack(np.broadcast_arrays(sine * np.cos(phi), sine * np.sin(phi),
                                                 mu[:, None]), axis=-1).reshape(-1, 3)
        weight = np.repeat(w * (1 - low) / 2 * mu * (2 * np.pi / azimuth_nodes), azimuth_nodes)
        factor = np.pi * math.sin(angle)**2
        return cls(direction, weight / weight.sum(), factor, axisymmetric=True)

    def internal_rays(self, boundary):
        c = boundary.coefficients(self.directions_external[:, 2], incidence="outside")
        direction = self.directions_external.copy()
        direction[:, :2] /= boundary.n
        direction[:, 2] = c.transmitted_cosine
        keep = (c.transmission > 0) & (direction[:, 2] > 0)
        return direction[keep], self.flux_weights[keep], c.transmission[keep]


@dataclass(frozen=True)
class HalfSpaceSettings:
    """Angular discretization.

    By default the volume field uses discrete ordinates on a uniform azimuth
    grid. ``azimuth_harmonics=M`` instead expands the volume field in the
    azimuthal cosine moments m=0..M about the transverse wavevector. Then
    ``azimuth_nodes`` is only the uniform readout grid used to interpolate
    the beam/diffuse cross terms of ``banana()``.
    """
    polar_nodes: int = 6
    azimuth_nodes: int = 12
    phase_degree: int = 8
    mode_cache_size: int = 8
    plane_parallel: bool = False
    azimuth_harmonics: int | None = None

    def __post_init__(self):
        _integer(self.polar_nodes, "polar_nodes", 2)
        _integer(self.azimuth_nodes, "azimuth_nodes", 1 if self.plane_parallel else 4)
        _integer(self.phase_degree, "phase_degree", 0)
        _integer(self.mode_cache_size, "mode_cache_size", 0)
        if self.azimuth_harmonics is not None:
            harmonics = _integer(self.azimuth_harmonics, "azimuth_harmonics", 0)
            if self.plane_parallel:
                raise ValueError("plane_parallel and azimuth_harmonics are exclusive")
            if self.azimuth_nodes % 2 or self.azimuth_nodes < 2 * (harmonics + 1):
                raise ValueError("azimuth_nodes must be even and at least 2*(azimuth_harmonics+1)")
            if self.phase_degree >= 2 * self.polar_nodes:
                raise ValueError("phase_degree must be below 2*polar_nodes")
            return
        if self.plane_parallel and self.azimuth_nodes != 1:
            raise ValueError("plane_parallel requires azimuth_nodes=1")
        if not self.plane_parallel and self.azimuth_nodes % 2:
            raise ValueError("azimuth_nodes must be even for exact direction reversal")
        limit = 2*self.polar_nodes if self.plane_parallel else min(2*self.polar_nodes, self.azimuth_nodes)
        if self.phase_degree >= limit:
            condition = "2*polar_nodes" if self.plane_parallel else "both 2*polar_nodes and azimuth_nodes"
            raise ValueError(f"phase_degree must be below {condition}")


class HalfSpaceSolver:
    """Monochromatic reciprocal RTE reference, independent of production routes."""
    def __init__(self, medium, boundary=None, settings=None):
        self.medium = medium
        self.boundary = HalfSpaceBoundary() if boundary is None else boundary
        self.settings = HalfSpaceSettings() if settings is None else settings
        x, w = roots_legendre(self.settings.polar_nodes)
        cuts = [0., 1.]
        if self.boundary.n > 1:
            cuts.insert(1, math.sqrt(1 - 1 / self.boundary.n**2))
        positive, weight = [], []
        for low, high in zip(cuts[:-1], cuts[1:]):
            positive.extend(low + (x + 1) * (high - low) / 2)
            weight.extend(w * (high - low) / 2)
        mu = np.asarray(positive)
        self.harmonics = self.settings.azimuth_harmonics
        if self.harmonics is not None:
            self._init_harmonics(mu, np.asarray(weight))
            return
        nphi = self.settings.azimuth_nodes
        phi = 2 * np.pi * (np.arange(nphi) + .5) / nphi
        sine = np.sqrt((1 - mu) * (1 + mu))[:, None]
        down = np.stack(np.broadcast_arrays(sine * np.cos(phi), sine * np.sin(phi),
                                            mu[:, None]), axis=-1).reshape(-1, 3)
        up = down.copy()
        up[:, 2] *= -1
        self.direction = np.concatenate([down, up])
        self.half = len(down)
        self.weight = np.tile(np.repeat(np.asarray(weight) * (2 * np.pi / nphi), nphi), 2)
        self.sqrt_weight = np.sqrt(self.weight)
        self.mu = self.direction[:, 2]
        self.reflection = self.boundary.coefficients(down[:, 2]).reflection
        self.transmission = self.boundary.coefficients(down[:, 2]).transmission
        index = np.arange(self.half).reshape(len(mu), nphi)
        phi_reverse = np.roll(index, nphi // 2, axis=1).ravel()
        self.reverse = np.concatenate([phi_reverse + self.half, phi_reverse])
        phase = self.angular_phase(self.direction, self.direction)
        self.phase_normalization_error = float(np.max(np.abs(phase @ self.weight - 1)))
        if self.phase_normalization_error > 2e-12:
            raise ValueError("angular quadrature does not normalize the configured phase function")
        self._check_phase_minimum()
        self.scatter = medium.scattering_per_m * (self.sqrt_weight[:, None] * phase
                                                  * self.sqrt_weight[None, :])
        self.reverse_sign = np.ones(len(self.mu))
        self._cache = OrderedDict()

    def _check_phase_minimum(self):
        medium = self.medium
        ell = np.arange(self.settings.phase_degree+1)
        polynomial = np.polynomial.legendre.Legendre((2*ell+1)*medium.g**ell/(4*np.pi))
        roots = polynomial.deriv().roots()
        roots = roots.real[(np.abs(roots.imag) < 1e-10) & (roots.real >= -1) & (roots.real <= 1)]
        self.phase_minimum = float(np.min(polynomial(np.r_[-1., roots, 1.])))
        if medium.scattering_per_m > 0 and self.phase_minimum < -1e-13:
            raise ValueError("the finite HG phase is negative; increase phase_degree and angular quadrature")

    def _init_harmonics(self, mu, weight):
        """Azimuthal cosine-moment basis about the transverse wavevector.

        Unknown (u, m) is sqrt(w_u) times the coefficient of the orthonormal
        function c_m(phi) = cos(m phi)/sqrt(pi) (c_0 = 1/sqrt(2 pi)), with
        polar node u (downward nodes first). Scattering is diagonal in m and
        is given by the Legendre addition theorem; streaming i q.Omega
        couples m to m+-1. On a uniform grid of 2(M+1) or more azimuths, with
        phase_degree <= azimuth_nodes/2, the discrete-ordinate equations for
        a field even about the wavevector are the same equations.
        """
        medium, harmonics = self.medium, self.harmonics
        degree = self.settings.phase_degree
        polar = np.r_[mu, -mu]
        self.polar_cosine = polar
        self.polar_weight = np.r_[weight, weight]
        self.polar_half = len(mu)
        count = harmonics + 1
        self.harmonic = np.tile(np.arange(count), len(polar))
        self.mu = np.repeat(polar, count)
        self.half = len(mu) * count
        self.weight = np.repeat(self.polar_weight, count)
        self.sqrt_weight = np.sqrt(self.weight)
        self.direction = None
        self.reflection = np.repeat(self.boundary.coefficients(mu).reflection, count)
        self.transmission = np.repeat(self.boundary.coefficients(mu).transmission, count)
        index = np.arange(self.half)
        self.reverse = np.concatenate([index + self.half, index])
        self.reverse_sign = np.where(self.harmonic % 2, -1., 1.)
        self.integral_weight = np.where(self.harmonic == 0, np.sqrt(2*np.pi*self.weight), 0.)
        self._legendre_scale = medium.g**np.arange(degree+1)
        self._node_legendre = _normalized_legendre(polar, degree, harmonics)
        # Kernel K[m, u, v] = 2 pi sum_l g^l P_l^m(mu_u) P_l^m(mu_v).
        kernel = 2*np.pi*np.einsum('mlu,l,mlv->muv', self._node_legendre,
                                   self._legendre_scale, self._node_legendre)
        self.phase_normalization_error = float(np.max(np.abs(kernel[0] @ self.polar_weight - 1)))
        if self.phase_normalization_error > 2e-12:
            raise ValueError("angular quadrature does not normalize the configured phase function")
        self._check_phase_minimum()
        root = np.sqrt(self.polar_weight)
        blocks = medium.scattering_per_m * root[None, :, None] * kernel * root[None, None, :]
        size = len(self.mu)
        scatter = np.zeros((len(polar), count, len(polar), count))
        for m in range(count):
            scatter[:, m, :, m] = blocks[m]
        self.scatter = scatter.reshape(size, size)
        # <c_m|cos(phi)|c_m+1>: 1/sqrt(2) for m=0, 1/2 otherwise.
        coupling = np.where(np.arange(harmonics) == 0, 1/math.sqrt(2), .5)
        stream = np.zeros((len(polar), count, count))
        sine = np.sqrt((1 - polar) * (1 + polar))
        stream[:, np.arange(harmonics), np.arange(1, count)] = sine[:, None] * coupling
        stream[:, np.arange(1, count), np.arange(harmonics)] = sine[:, None] * coupling
        self._stream = np.zeros((size, size))
        for u in range(len(polar)):
            block = slice(u*count, (u+1)*count)
            self._stream[block, block] = stream[u]
        self._cache = OrderedDict()

    @staticmethod
    def _axis(q):
        """Canonical line through q: angle in (-pi/2, pi/2] and signed |q|."""
        if not np.any(q):
            return 0., 0.
        angle = math.atan2(q[1], q[0])
        if angle <= -np.pi/2:
            angle += np.pi
        elif angle > np.pi/2:
            angle -= np.pi
        return angle, float(q[0]*math.cos(angle) + q[1]*math.sin(angle))

    def _check_even(self, port, angle):
        """A cosine-moment field needs a port even about the wavevector line."""
        rays, weights = port.directions_external, port.flux_weights
        c, s = math.cos(2*angle), math.sin(2*angle)
        mirror = np.column_stack([c*rays[:, 0] + s*rays[:, 1], s*rays[:, 0] - c*rays[:, 1], rays[:, 2]])
        distance = np.linalg.norm(mirror[:, None, :] - rays[None, :, :], axis=2)
        partner = np.argmin(distance, axis=1)
        if (np.max(distance[np.arange(len(rays)), partner]) > 1e-12
                or np.max(np.abs(weights[partner] - weights)) > 1e-14):
            raise ValueError("azimuth_harmonics requires a port mirror-symmetric about the wavevector")

    def basis_phase(self, directions, axis=0.):
        """Matrix whose product with U is the collision integral in directions.

        Row d is p(Omega_d . Omega_k) sqrt(W_k) for discrete ordinates and
        K_m(mu_d, mu_u) c_m(phi_d - axis) sqrt(w_u) for cosine moments.
        """
        direction = np.asarray(directions, float)
        if self.harmonics is None:
            return self.angular_phase(direction, self.direction) * self.sqrt_weight
        legendre = _normalized_legendre(direction[:, 2], self.settings.phase_degree, self.harmonics)
        kernel = 2*np.pi*np.einsum('mld,l,mlu->dum', legendre, self._legendre_scale, self._node_legendre)
        order = np.arange(self.harmonics + 1)
        azimuth = np.arctan2(direction[:, 1], direction[:, 0]) - axis
        moment = np.cos(order * azimuth[:, None]) / math.sqrt(np.pi)
        moment[:, 0] = 1 / math.sqrt(2*np.pi)
        result = kernel * moment[:, None, :] * np.sqrt(self.polar_weight)[None, :, None]
        return result.reshape(len(direction), -1)

    def phase(self, cosine):
        return _phase(cosine, self.medium.g, self.settings.phase_degree)

    def angular_phase(self, outgoing, incoming):
        """Angular scattering matrix, with analytic m=0 averaging when requested.

        At q=0, axial sources and receivers see only the azimuthal mean.
        The addition theorem gives its exact finite-HG kernel as the product
        of Legendre polynomials of the two normal cosines. No isotropic
        source replacement or reduced-scattering substitution is involved.
        """
        if not self.settings.plane_parallel:
            return self.phase(outgoing @ incoming.T)
        degree = self.settings.phase_degree
        ell = np.arange(degree+1)
        left = np.polynomial.legendre.legvander(outgoing[:, 2], degree)
        right = np.polynomial.legendre.legvander(incoming[:, 2], degree)
        return (left*((2*ell+1)*self.medium.g**ell/(4*np.pi))) @ right.T

    def _order_modes(self, lam, vectors):
        if not np.isfinite(lam).all() or not np.isfinite(vectors).all():
            raise ArithmeticError("half-space eigenmodes are not finite")
        positive = lam.real > 0
        if np.count_nonzero(positive) != self.half:
            raise ArithmeticError("the half-space stable subspace has the wrong dimension")
        order = np.r_[np.flatnonzero(positive), np.flatnonzero(~positive)]
        return lam[order], vectors[:, order]

    def _modes(self, q, omega):
        key = (*q, omega)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        base = self.medium.extinction_per_m - 1j * omega / self.medium.speed_m_per_ns
        if self.harmonics is not None:
            # A' = D^-1 A D is real up to the scalar -i omega/c for D=diag(i^m).
            # The pencil (A', M) has the eigenvalues of (A, M); eigenvectors
            # map back by D. M^-1 A' is solved as a standard problem: real at
            # omega=0, complex otherwise.
            _, along = self._axis(q)
            diagonal = np.full(len(self.mu), base)
            stream = along*(np.tril(self._stream) - np.triu(self._stream))
            if omega == 0:
                operator = self.medium.extinction_per_m*np.eye(len(self.mu)) - self.scatter + stream
                lam, vectors = eig(operator / self.mu[:, None], check_finite=False)
                lam, vectors = self._order_modes(lam, vectors)
                image = operator @ vectors.real + 1j*(operator @ vectors.imag)
            else:
                operator = base*np.eye(len(self.mu)) - self.scatter + stream
                lam, vectors = _complex_eig(operator / self.mu[:, None])
                lam, vectors = self._order_modes(lam, vectors)
                image = operator @ vectors
            residual = np.linalg.norm(image - self.mu[:, None] * vectors * lam) / max(1., np.linalg.norm(operator))
            vectors = np.array([1., 1j, -1., -1j])[self.harmonic % 4][:, None] * vectors
        else:
            diagonal = base + 1j * self.direction[:, :2] @ q
            operator = np.diag(diagonal) - self.scatter
            lam, vectors = eig(operator, np.diag(self.mu), check_finite=False)
            lam, vectors = self._order_modes(lam, vectors)
            residual = np.linalg.norm(operator @ vectors - self.mu[:, None] * vectors * lam) / max(1., np.linalg.norm(operator))
        if not np.isfinite(residual) or residual > 1e-10:
            raise ArithmeticError("half-space eigenmode residual exceeds tolerance")
        stable, unstable = vectors[:, :self.half], vectors[:, self.half:]
        match = stable[:self.half] - self.reflection[:, None] * stable[self.half:]
        data = (diagonal, lam, stable, unstable, lu_factor(vectors), lu_factor(match), residual)
        if self.settings.mode_cache_size:
            self._cache[key] = data
            while len(self._cache) > self.settings.mode_cache_size:
                self._cache.popitem(last=False)
        return data

    def solve(self, source=None, *, q_per_m=(0., 0.), omega_per_ns=0.):
        q = np.asarray(q_per_m, float)
        omega = float(omega_per_ns)
        if q.shape != (2,) or not np.isfinite(q).all() or not math.isfinite(omega):
            raise ValueError("q_per_m must be a finite two-vector and omega_per_ns finite")
        source = SurfacePort.pencil() if source is None else source
        if self.settings.plane_parallel and (np.any(q != 0) or not source.axisymmetric):
            raise ValueError("plane_parallel supports q=0 and axisymmetric source ports only")
        if self.harmonics is not None:
            self._check_even(source, self._axis(q)[0])
        return HalfSpaceSolution(self, source, q, omega)

    def response(self, source, detector, *, q_per_m=(0., 0.), omega_per_ns=0.):
        return self.solve(source, q_per_m=q_per_m, omega_per_ns=omega_per_ns).response(detector)

    def time_bins(self, source, detector, omega_per_ns, edges_ns, *, sigma_ns=0.):
        """Laterally integrated escaped flux per bin, in orders 0/1/>=2.

        Finite frequency range and spacing require independent convergence
        checks. Values are neither clipped nor renormalized. The pulse is
        emitted at exterior surface entry, at t=0.
        """
        from ..readout import inverse_bins
        omega = np.asarray(omega_per_ns, float)
        if omega.ndim != 1 or not np.isfinite(omega).all():
            raise ValueError("omega_per_ns must be a finite vector")
        spectrum = np.array([self.response(source, detector, omega_per_ns=o) for o in omega])
        return inverse_bins(omega, spectrum, edges_ns, sigma_ns).T

    def depth_sensitivity(self, source, detector, depths_m, *, q_per_m=(0., 0.), omega_per_ns=0.):
        """Absorption derivative per depth for a laterally uniform perturbation.

        Returns d(response)/d(mu_a(z)), including the two analytic
        beam/diffuse cross terms. This is the depth projection of the banana,
        not a spatially localized three-dimensional sensitivity map.
        """
        z = _depths(depths_m)
        q = np.asarray(q_per_m, float)
        forward = self.solve(source, q_per_m=q, omega_per_ns=omega_per_ns)
        reverse = self.solve(detector, q_per_m=-q, omega_per_ns=omega_per_ns)
        a, b = forward.field(z), reverse.field(z)
        pairing = np.sum(a * b[:, self.reverse] * self.weight * self.reverse_sign, axis=1)
        if len(forward.rays):
            cross = reverse.upward_field(-forward.rays, z)
            pairing += np.sum(forward.beam(z) * cross, axis=1)
        if len(reverse.rays):
            cross = forward.upward_field(-reverse.rays, z)
            pairing += np.sum(reverse.beam(z) * cross, axis=1)
        return -detector.collection_factor / self.boundary.n**2 * pairing

    def radial_response(self, source, detector, radii_m, *, q_max_per_m, q_nodes=48,
                        omega_per_ns=0., source_sigma_m=0., detector_sigma_m=0.):
        """Hankel inversion for axial ports and circular Gaussian footprints.

        Result: (radius, order) escaped photons/m^2 per exterior incident
        photon. The detector footprint is a normalized spatial averaging
        weight; multiply by its physical collection area separately.
        Check q cutoff, quadrature and angular resolution independently.
        """
        if self.settings.plane_parallel:
            raise ValueError("plane_parallel does not provide spatial inversion")
        radius = np.atleast_1d(np.asarray(radii_m, float))
        if not source.axisymmetric or not detector.axisymmetric:
            raise ValueError("Hankel inversion requires axisymmetric source and detector ports")
        qmax = float(q_max_per_m)
        if radius.ndim != 1 or not np.isfinite(radius).all() or np.any(radius < 0):
            raise ValueError("radii_m must be finite and nonnegative")
        if not math.isfinite(qmax) or qmax <= 0:
            raise ValueError("q_max_per_m must be positive")
        sigmas = np.asarray([source_sigma_m, detector_sigma_m], float)
        if not np.isfinite(sigmas).all() or np.any(sigmas < 0):
            raise ValueError("footprint sigmas must be finite and nonnegative")
        x, w = roots_legendre(_integer(q_nodes, "q_nodes", 4))
        q, weight = (x + 1) * qmax / 2, w * qmax / 2
        spectrum = np.array([self.response(source, detector, q_per_m=(k, 0.),
                                          omega_per_ns=omega_per_ns) for k in q])
        factor = weight * q / (2 * np.pi) * np.exp(-q*q * np.sum(sigmas**2) / 2)
        return (j0(radius[:, None] * q) * factor) @ spectrum

    def banana(self, source, detector, points_m, *, source_xy_m=(0., 0.),
               detector_xy_m=(1., 0.), source_sigma_m, detector_sigma_m,
               q_max_per_m, q_nodes=48, omega_per_ns=0., by_order=False):
        """Local absorption derivative dD/dmu_a(r), in m^-4 per incident photon.

        Both spatial ports have normalized circular Gaussian footprints and
        retain their prescribed angular shapes. The detector result is flux
        density averaged over its footprint. Integrating K(r)*delta_mu_a(r)
        over volume gives its first-order variation. For axial angular
        ports, azimuthal cosine moments are inverted with Bessel functions.
        With ``by_order`` the result has columns for detected volume orders
        (0, 1, >=2): order 1 pairs an uncollided leg with a once-scattered
        leg. Check transverse cutoff, quadrature and angular resolution.
        """
        if self.settings.plane_parallel:
            raise ValueError("plane_parallel does not provide spatial inversion")
        point = np.asarray(points_m, float)
        xy_s, xy_d = np.asarray(source_xy_m, float), np.asarray(detector_xy_m, float)
        sigmas = np.asarray([source_sigma_m, detector_sigma_m], float)
        qmax = float(q_max_per_m)
        if not source.axisymmetric or not detector.axisymmetric:
            raise ValueError("the Fourier-Bessel banana requires axisymmetric ports")
        if (point.ndim != 2 or point.shape[1] != 3 or not len(point)
                or not np.isfinite(point).all() or np.any(point[:, 2] < 0)
                or xy_s.shape != (2,) or xy_d.shape != (2,)
                or not np.isfinite(xy_s).all() or not np.isfinite(xy_d).all()
                or not np.isfinite(sigmas).all() or np.any(sigmas <= 0)
                or not math.isfinite(qmax) or qmax <= 0):
            raise ValueError("finite interior points, two surface positions, positive Gaussian sigmas and q cutoff are required")
        z, inverse = np.unique(point[:, 2], return_inverse=True)
        x, w = roots_legendre(_integer(q_nodes, "q_nodes", 4))
        waves, weights = (x+1)*qmax/2, w*qmax/2
        rays_s, _, _ = source.internal_rays(self.boundary)
        rays_d, _, _ = detector.internal_rays(self.boundary)
        harmonic = self.harmonics is not None
        if harmonic:
            volume_degree = np.arange(self.harmonics + 1)
            forward = np.zeros((len(point), len(self.polar_cosine), len(volume_degree)), complex)
        else:
            forward = np.zeros((len(point), len(self.mu)), complex)
        adjoint = np.zeros_like(forward)
        parts = 2 if by_order else 1
        forward_up = np.zeros((parts, len(point), len(rays_d)), complex)
        adjoint_up = np.zeros((parts, len(point), len(rays_s)), complex)
        nphi = self.settings.azimuth_nodes
        phi = 2*np.pi*(np.arange(nphi)+.5)/nphi
        degree = np.arange(nphi//2)
        basis = np.cos(degree[:, None]*phi)
        basis *= np.where(degree == 0, 1., 2.)[:, None]/nphi

        def cross_grid(directions):
            mu, pick = np.unique(-directions[:, 2], return_inverse=True)
            sine = np.sqrt((1-mu)*(1+mu))[:, None]
            grid = np.stack(np.broadcast_arrays(sine*np.cos(phi), sine*np.sin(phi),
                                                 -mu[:, None]), axis=-1).reshape(-1, 3)
            return grid, pick, np.arctan2(directions[:, 1], directions[:, 0])

        cross_s, cross_d = cross_grid(-rays_s), cross_grid(-rays_d)

        def invert(solution, center, cross, q):
            displacement = point[:, :2]-center
            radius = np.linalg.norm(displacement, axis=1)
            azimuth = np.arctan2(displacement[:, 1], displacement[:, 0])
            radial = jv(degree[None, :], q*radius[:, None])*(1j**degree)
            if harmonic:
                # Jacobi-Anger: moment m inverts with i^m J_m(q rho) about azimuth.
                coefficient = solution.field(z).reshape(len(z), -1, len(volume_degree))[inverse]
                field = coefficient * (jv(volume_degree[None, :], q*radius[:, None])
                                       * (1j**volume_degree))[:, None, :]
            else:
                regular = solution.field(z).reshape(len(z), -1, nphi)
                coefficient = np.einsum('zup,mp->zum', regular, basis)[inverse]
                angular = np.cos((phi[None, :, None]-azimuth[:, None, None])*degree)
                field = np.einsum('pum,pvm,pm->puv', coefficient, angular, radial).reshape(len(point), -1)
            grid, pick, target_phi = cross
            angular_up = np.cos((target_phi[None, :, None]-azimuth[:, None, None])*degree)
            upward = []
            for values in solution._upward_parts(grid, z)[2-parts:]:
                regular_up = values.reshape(len(z), -1, nphi)
                coefficient_up = np.einsum('zup,mp->zum', regular_up, basis)[inverse][:, pick]
                upward.append(np.sum(coefficient_up*angular_up*radial[:, None, :], axis=2))
            return field, np.array(upward)

        for q, weight in zip(waves, weights):
            a = self.solve(source, q_per_m=(q, 0.), omega_per_ns=omega_per_ns)
            b = self.solve(detector, q_per_m=(q, 0.), omega_per_ns=omega_per_ns)
            af, au = invert(a, xy_s, cross_d, q)
            bf, bu = invert(b, xy_d, cross_s, q)
            factor_s = weight*q/(2*np.pi)*np.exp(-sigmas[0]**2*q*q/2)
            factor_d = weight*q/(2*np.pi)*np.exp(-sigmas[1]**2*q*q/2)
            forward += factor_s*af
            forward_up += factor_s*au
            adjoint += factor_d*bf
            adjoint_up += factor_d*bu

        base = self.medium.extinction_per_m - 1j*omega_per_ns/self.medium.speed_m_per_ns

        def ballistic(port, xy, sigma):
            # Uncollided interior beam; no angular modes are needed.
            rays, flux, transmission = port.internal_rays(self.boundary)
            amplitude = flux * transmission / rays[:, 2]
            displacement = (point[:, None, :2] - xy
                            - point[:, None, 2:] * rays[None, :, :2] / rays[None, :, 2:])
            gaussian = np.exp(-np.sum(displacement**2, axis=2)/(2*sigma*sigma)) / (2*np.pi*sigma*sigma)
            return np.exp(-point[:, 2, None] * (base / rays[:, 2])) * amplitude * gaussian

        if harmonic:
            ds, dd = point[:, :2]-xy_s, point[:, :2]-xy_d
            difference = np.arctan2(ds[:, 1], ds[:, 0]) - np.arctan2(dd[:, 1], dd[:, 0])
            # Integral over azimuth of c_m(phi-a_s) c_m(phi+pi-a_d).
            angle = np.cos(volume_degree*difference[:, None]) * np.where(volume_degree % 2, -1., 1.)
            reverse = np.roll(np.arange(len(self.polar_cosine)), -self.polar_half)
            diffuse = np.einsum('pum,pum,u,pm->p', forward, adjoint[:, reverse], self.polar_weight, angle)
        else:
            diffuse = np.sum(forward * adjoint[:, self.reverse] * self.weight, axis=1)
        beam_s = ballistic(source, xy_s, sigmas[0])
        beam_d = ballistic(detector, xy_d, sigmas[1])
        scale = -detector.collection_factor / self.boundary.n**2
        if not by_order:
            pairing = diffuse + np.sum(beam_s * adjoint_up[0], axis=1)
            pairing += np.sum(beam_d * forward_up[0], axis=1)
            return scale * pairing
        # One volume collision: an uncollided leg times a once-scattered leg.
        first = np.sum(beam_s * adjoint_up[0], axis=1) + np.sum(beam_d * forward_up[0], axis=1)
        total = diffuse + np.sum(beam_s * adjoint_up[1], axis=1) + np.sum(beam_d * forward_up[1], axis=1)
        return scale * np.column_stack([np.zeros_like(first), first, total - first])


class HalfSpaceSolution:
    """One source at one transverse wavevector and temporal frequency."""
    def __init__(self, solver, source, q, omega):
        self.solver, self.source, self.q, self.omega = solver, source, q, omega
        self.rays, source_weights, self.entry_transmission = source.internal_rays(solver.boundary)
        self.entered_fraction = float(source_weights @ self.entry_transmission)
        self.specular_fraction = 1 - self.entered_fraction
        self.amplitude = source_weights * self.entry_transmission / self.rays[:, 2]
        self.base = solver.medium.extinction_per_m - 1j * omega / solver.medium.speed_m_per_ns
        self.rate = (self.base + 1j * self.rays[:, :2] @ q) / self.rays[:, 2]
        (self.diagonal, self.lam, self.stable, self.unstable, vector_lu,
         boundary_lu, self.eigen_residual) = solver._modes(q, omega)
        self.axis = solver._axis(q)[0]
        if solver.harmonics is None:
            self.forcing = (solver.medium.scattering_per_m * solver.sqrt_weight[:, None]
                            * solver.angular_phase(solver.direction, self.rays) * self.amplitude)
        else:
            self.forcing = (solver.medium.scattering_per_m
                            * solver.basis_phase(self.rays, self.axis).T * self.amplitude)
        g = lu_solve(vector_lu, self.forcing / solver.mu[:, None])
        self.gs, self.gu = g[:solver.half], g[solver.half:]
        self.part_u = self.gu / (self.lam[solver.half:, None] - self.rate)
        driven_boundary = self.unstable @ self.part_u.sum(axis=1)
        mismatch = (driven_boundary[:solver.half]
                    - solver.reflection * driven_boundary[solver.half:])
        self.c = lu_solve(boundary_lu, -mismatch)
        self.surface_weighted = self.stable @ self.c + driven_boundary

    def beam(self, depths_m):
        z = _depths(depths_m)
        return np.exp(-z[:, None] * self.rate) * self.amplitude

    def _coordinates(self, z):
        h = self.solver.half
        stable = np.exp(-self.lam[:h, None] * z) * self.c[:, None]
        stable += np.sum(self.gs[:, :, None] * _driven_exponential(self.lam[:h], self.rate, z), axis=1)
        unstable = self.part_u @ np.exp(-self.rate[:, None] * z)
        return stable, unstable

    def field(self, depths_m):
        """Scattered physical angular radiance, shape (depth, direction).

        The uncollided angular delta beams are returned separately by beam().
        In plane_parallel mode this is the azimuthal mean at each cosine.
        With azimuth_harmonics these are the cosine-moment coefficients:
        radiance(mu_u, phi) = sum_m field[:, (u, m)] c_m(phi - axis).
        """
        z = _depths(depths_m)
        stable, unstable = self._coordinates(z)
        return ((self.stable @ stable + self.unstable @ unstable)
                / self.solver.sqrt_weight[:, None]).T

    def first_field(self, depths_m):
        """One volume collision, including subsequent specular reflection."""
        if self.solver.harmonics is not None and np.any(self.q != 0):
            raise ValueError("first_field with azimuth_harmonics requires q=0")
        z = _depths(depths_m)
        s, h = self.solver, self.solver.half
        up = np.zeros((h, len(z)), complex)
        for k, rate in enumerate(self.rate):
            up += (self.forcing[h:, k] / (self.diagonal[h:] - s.mu[h:] * rate))[:, None] * np.exp(-rate * z)
        down = np.sum((self.forcing[:h] / s.mu[:h, None])[:, :, None]
                      * _driven_exponential(self.diagonal[:h] / s.mu[:h], self.rate, z), axis=1)
        up0 = np.sum(self.forcing[h:] / (self.diagonal[h:, None] - s.mu[h:, None] * self.rate), axis=1)
        down += (s.reflection * up0)[:, None] * np.exp(-(self.diagonal[:h] / s.mu[:h])[:, None] * z)
        return (np.concatenate([down, up]) / s.sqrt_weight[:, None]).T

    def upward_field(self, directions, depths_m, *, first_only=False):
        """Scattered radiance in arbitrary upward unit directions.

        Evaluated from the collision integral, without angular interpolation.
        """
        first, total = self._upward_parts(directions, depths_m, first_only=first_only)
        return first if first_only else total

    def _upward_parts(self, directions, depths_m, *, first_only=False):
        """(once-scattered, all scattered) upward radiance, shape (depth, direction)."""
        direction = np.asarray(directions, float)
        if (direction.ndim != 2 or direction.shape[1] != 3
                or not np.isfinite(direction).all() or np.any(direction[:, 2] >= 0)
                or not np.allclose(np.linalg.norm(direction, axis=1), 1., rtol=0, atol=1e-12)):
            raise ValueError("directions must be upward unit vectors")
        z = _depths(depths_m)
        s, h = self.solver, self.solver.half
        mu = -direction[:, 2]
        attenuation = self.base + 1j * direction[:, :2] @ self.q
        b = attenuation / mu
        beam_phase = s.angular_phase(direction, self.rays)
        exponent = np.exp(-self.rate[:, None] * z)
        first = s.medium.scattering_per_m * (
            (beam_phase * self.amplitude / (attenuation[:, None] + mu[:, None] * self.rate)) @ exponent).T
        if first_only:
            return first, None
        # Laplace transforms of the modal coordinates along each upward ray,
        # contracted with the collision kernel before the depth sum.
        stable_coord, _ = self._coordinates(z)
        phase = s.basis_phase(direction, self.axis)
        ps = (phase @ self.stable) / (b[:, None] + self.lam[None, :h])
        pu = phase @ self.unstable
        rates = (ps @ self.gs + pu @ self.part_u) / (b[:, None] + self.rate[None, :])
        laplace = ps @ stable_coord + rates @ exponent
        return first, first + (s.medium.scattering_per_m / mu[:, None] * laplace).T

    def response(self, detector):
        """Exterior collected flux spectrum in volume orders (0,1,>=2).

        Direct exterior specular reflection is separate in specular_fraction.
        Divide by detector.collection_factor for the reciprocal port response.
        """
        if self.solver.settings.plane_parallel and not detector.axisymmetric:
            raise ValueError("plane_parallel requires an axisymmetric detector port")
        rays, weight, transmission = detector.internal_rays(self.solver.boundary)
        output = np.zeros(3, complex)
        if not len(rays):
            return output
        factor = detector.collection_factor * weight * transmission / self.solver.boundary.n**2
        first, total = (value[0] for value in self._upward_parts(-rays, [0.]))
        output[1] = factor @ first
        output[2] = factor @ (total - first)
        return output

    def photon_balance(self):
        """All-time, laterally integrated absorption/escape/specular balance."""
        if np.any(self.q != 0) or self.omega != 0:
            raise ValueError("photon balance requires q=0 and omega=0")
        s, h = self.solver, self.solver.half
        integral_s = self.c / self.lam[:h] + np.sum(self.gs / self.rate / self.lam[:h, None], axis=1)
        integral_u = np.sum(self.part_u / self.rate, axis=1)
        integral = self.stable @ integral_s + self.unstable @ integral_u
        if s.harmonics is None:
            fluence = np.sum(self.amplitude / self.rate) + s.sqrt_weight @ integral
            outgoing = self.surface_weighted[h:] / s.sqrt_weight[h:]
            escaped = np.sum(s.weight[h:] * (-s.mu[h:]) * s.transmission * outgoing)
        else:
            fluence = np.sum(self.amplitude / self.rate) + s.integral_weight @ integral
            escaped = np.sum(s.integral_weight[h:] * (-s.mu[h:]) * s.transmission
                             * self.surface_weighted[h:])
        absorbed = s.medium.absorption_per_m * fluence
        return {"specular": self.specular_fraction, "entered": self.entered_fraction,
                "absorbed": float(absorbed.real), "escaped": float(escaped.real),
                "residual": float((absorbed + escaped).real - self.entered_fraction)}
