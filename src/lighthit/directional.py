r"""Exact directional-OM transport: the ``(l, lambda, mu)`` kernel.

The cache of :mod:`lighthit.cache` keeps one multipole per degree, which is
everything a problem with a *single* preferred direction needs: either a
directed source read by an isotropic detector, or an isotropic source read by a
directed detector. Both are the same array, because the angular system is
symmetric and its ``m = 0`` block serves the source index and the detector
index alike.

An event is directed and a real optical module is directed, so both directions
are present at once and that array is no longer enough. What replaces it is
derived in ``docs/research/directional-om-plan.md`` and summarised here.

Work in the frame whose polar axis is the source-to-detector direction
``r_hat``. The response is then

    R(r) = sum_l sum_(lambda<=L_A) sum_(|mu|<=min(l,lambda))
           alpha_lambda t_(l lambda |mu|)(r, omega)
           Y_(lambda mu)(n_hat) S_(l mu)(omega),                       (*)

with every harmonic evaluated in that frame, ``alpha_lambda`` the Legendre
coefficients of the module acceptance, ``S_(l mu)`` the source channels and

    t_(l lambda mu)(r) = (1 / 2 pi^2) sum_J i^J c(lambda, l, J, mu)
                         integral k^2 j_J(k r) g_(lambda l J)(k) dk,
    g_(lambda l J)(k) = sum_nu c(lambda, l, J, nu) G^(nu)_(lambda l)(k),
    c(lambda, l, J, mu) = (-1)^mu <lambda mu; l -mu | J 0>.

Three facts make this affordable and all three are derived, not assumed:

* the detector index is bounded by the *acceptance bandwidth* ``L_A`` -- three
  for a cubic module response -- and not by the source degree;
* the sum over the azimuthal block index ``nu`` factorises out of the geometry,
  so the cache is indexed by a triple and not by a quadruple;
* ``g`` vanishes unless ``l + lambda + J`` is even, which halves what is left.

At ``lambda = 0`` the whole construction collapses to the existing ``M_l``, and
that is a test (``tests/test_directional.py``), not a remark.
"""
from dataclasses import dataclass, asdict, field
from fractions import Fraction
from math import exp, factorial, lgamma, log, sqrt
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
import json

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.special import spherical_jn, eval_legendre, roots_legendre

from .angular import _degree, _check_backend, resolvent_rows
from .cache import CacheGrid, MOMENT_ORDERS, acceptance_coefficients
from .green import SolverSettings
from .medium import Medium

__all__ = ["clebsch_gordan", "acceptance_bandwidth", "CouplingTable",
           "wigner_small_d", "real_rotation_rows", "DirectionalCache",
           "directional_response"]


# ----------------------------------------------------------------- 3j / CG

def clebsch_gordan(j1, m1, j2, m2, J, M):
    """``<j1 m1 j2 m2 | J M>`` for integer angular momenta, by the Racah sum.

    Evaluated in exact integer arithmetic and converted to a float only at the
    end, through logarithms. The naive form multiplies six factorials before
    taking a square root, which overflows a double at ``j ~ 60`` --- well
    inside the source degrees this package already accepts --- and it does so
    by raising, not by returning a wrong number, which is how it was found.
    A coefficient forbidden by a selection rule is still exactly ``0.0`` here,
    because the exact sum is exactly zero before any float appears.
    """
    j1, m1, j2, m2, J, M = (int(x) for x in (j1, m1, j2, m2, J, M))
    if M != m1 + m2 or not abs(j1 - j2) <= J <= j1 + j2:
        return 0.0
    if abs(m1) > j1 or abs(m2) > j2 or abs(M) > J:
        return 0.0
    squared = Fraction(
        (2 * J + 1) * factorial(J + j1 - j2) * factorial(J - j1 + j2)
        * factorial(j1 + j2 - J) * factorial(J + M) * factorial(J - M)
        * factorial(j1 - m1) * factorial(j1 + m1) * factorial(j2 - m2)
        * factorial(j2 + m2),
        factorial(j1 + j2 + J + 1))
    total = Fraction(0)
    for step in range(j1 + j2 + J + 2):
        terms = (j1 + j2 - J - step, j1 - m1 - step, j2 + m2 - step,
                 J - j2 + m1 + step, J - j1 - m2 + step)
        if min(terms) < 0:
            continue
        denominator = factorial(step)
        for term in terms:
            denominator *= factorial(term)
        total += Fraction((-1) ** step, denominator)
    if total == 0:
        return 0.0
    sign = 1.0 if total > 0 else -1.0
    total = abs(total)
    logarithm = (0.5 * (log(squared.numerator) - log(squared.denominator))
                 + log(total.numerator) - log(total.denominator))
    return sign * exp(logarithm)


# ------------------------------------------------------- acceptance bandwidth

def acceptance_bandwidth(acceptance, *, max_degree=64, tolerance=1e-10,
                         quadrature_order=2048):
    """Measure the Legendre bandwidth of a module acceptance.

    Returns ``(bandwidth, alpha, residual)``. ``bandwidth`` is the largest
    degree whose coefficient exceeds ``tolerance`` times the peak; ``residual``
    is the largest coefficient *above* that degree, again relative to the peak,
    and is what says whether a nominally cubic response really is cubic.

    ``bandwidth == max_degree`` means the coefficients never fell below the
    tolerance inside the probed range: the response is **not** band-limited
    there. A module polynomial clipped at zero inside ``[-1, 1]`` does exactly
    that, and this is where it shows up rather than silently in the answer.
    """
    alpha = acceptance_coefficients(acceptance, max_degree,
                                    quadrature_order=quadrature_order)
    peak = float(np.max(np.abs(alpha)))
    if peak <= 0:
        raise ValueError("acceptance is identically zero")
    significant = np.flatnonzero(np.abs(alpha) > tolerance * peak)
    bandwidth = int(significant[-1]) if len(significant) else 0
    above = np.abs(alpha[bandwidth + 1:])
    residual = float(np.max(above) / peak) if len(above) else 0.0
    return bandwidth, alpha, residual


# ------------------------------------------------------------ coupling table

@dataclass(frozen=True)
class CouplingTable:
    """Everything that depends on ``(l, lambda, J, mu)`` and on nothing else.

    Built once per (source degree, acceptance degree) pair and shared by every
    wavelength, frequency, module and event.
    """
    source_degree: int
    acceptance_degree: int
    pair_lambda: np.ndarray          # (P,)   lambda of each stored (lambda,|mu|)
    pair_mu: np.ndarray              # (P,)   |mu| of each stored pair
    triple_lambda: np.ndarray        # (T,)
    triple_l: np.ndarray             # (T,)
    triple_J: np.ndarray             # (T,)
    nu_weight: np.ndarray            # (T, acceptance_degree+1) folded +-nu sum
    pair_weight: np.ndarray          # (T, P) complex, i^J c(lambda,l,J,mu)

    @property
    def pairs(self):
        return len(self.pair_lambda)

    @property
    def triples(self):
        return len(self.triple_lambda)

    @classmethod
    def build(cls, source_degree, acceptance_degree):
        L_s = _degree(source_degree, "source_degree")
        L_A = _degree(acceptance_degree, "acceptance_degree")
        pair_lambda, pair_mu = [], []
        for lam in range(L_A + 1):
            for mu in range(lam + 1):
                pair_lambda.append(lam)
                pair_mu.append(mu)
        pair_index = {(lam, mu): i for i, (lam, mu)
                      in enumerate(zip(pair_lambda, pair_mu))}
        triple_lambda, triple_l, triple_J = [], [], []
        for lam in range(L_A + 1):
            for l in range(L_s + 1):
                for J in range(abs(l - lam), l + lam + 1):
                    if (l + lam + J) % 2:
                        continue          # parity rule, derived in the note
                    triple_lambda.append(lam)
                    triple_l.append(l)
                    triple_J.append(J)
        n_triple = len(triple_lambda)
        nu_weight = np.zeros((n_triple, L_A + 1))
        pair_weight = np.zeros((n_triple, len(pair_lambda)), dtype=complex)
        for t, (lam, l, J) in enumerate(zip(triple_lambda, triple_l, triple_J)):
            top = min(lam, l)
            for nu in range(top + 1):
                coefficient = (-1.0) ** nu * clebsch_gordan(lam, nu, l, -nu, J, 0)
                nu_weight[t, nu] = coefficient * (1.0 if nu == 0 else 2.0)
                pair_weight[t, pair_index[(lam, nu)]] = 1j ** J * coefficient
        return cls(L_s, L_A, np.asarray(pair_lambda, np.int64),
                   np.asarray(pair_mu, np.int64),
                   np.asarray(triple_lambda, np.int64),
                   np.asarray(triple_l, np.int64), np.asarray(triple_J, np.int64),
                   nu_weight, pair_weight)


# --------------------------------------------------------------- Wigner d

def _start_value(j, mu, nu, half_cos, half_sin):
    """``d^j_(mu nu)`` where ``j = max(|mu|, |nu|)``, in closed form."""
    if j == 0:
        return 1.0
    sign = 1.0
    if abs(nu) == j and abs(mu) != j:
        mu, nu = nu, mu
        sign *= (-1.0) ** (mu - nu)
    if mu == -j:
        mu, nu = j, -nu
        sign *= (-1.0) ** (j + nu)
    # now mu == j
    # log form: the binomial alone overflows a double well before the degrees
    # this package accepts.
    binomial = exp(0.5 * (lgamma(2 * j + 1) - lgamma(j + nu + 1)
                          - lgamma(j - nu + 1)))
    return sign * binomial * half_cos ** (j + nu) * (-half_sin) ** (j - nu)


def wigner_small_d(degree, max_out, max_in, cos_theta):
    r"""``d^l_(mu nu)(theta)`` for ``|nu| <= max_out``, ``|mu| <= max_in``.

    Shape ``(points, degree+1, 2*max_out+1, 2*max_in+1)`` with the magnetic
    indices offset by ``max_out`` and ``max_in``. Entries with
    ``l < max(|mu|, |nu|)`` are zero. The ``l`` recurrence at fixed magnetic
    indices is the stable direction, exactly as for associated Legendre
    functions; the starting value is the closed form at ``l = max(|mu|,|nu|)``.
    """
    degree = _degree(degree, "degree")
    max_out = _degree(max_out, "max_out")
    max_in = _degree(max_in, "max_in")
    x = np.atleast_1d(np.asarray(cos_theta, float))
    if x.ndim != 1 or not np.isfinite(x).all() or np.any(np.abs(x) > 1 + 1e-12):
        raise ValueError("cos_theta must be a finite 1-D array in [-1, 1]")
    x = np.clip(x, -1.0, 1.0)
    half_cos = np.sqrt((1 + x) / 2)
    half_sin = np.sqrt((1 - x) / 2)
    out = np.zeros((len(x), degree + 1, 2 * max_out + 1, 2 * max_in + 1))
    for nu in range(-max_out, max_out + 1):
        for mu in range(-max_in, max_in + 1):
            start = max(abs(mu), abs(nu))
            if start > degree:
                continue
            previous = np.zeros(len(x))
            current = np.array([_start_value(start, mu, nu, c, s)
                                for c, s in zip(half_cos, half_sin)])
            out[:, start, nu + max_out, mu + max_in] = current
            for l in range(start, degree):
                if l == 0:
                    value = x * current
                else:
                    lower = sqrt(max((l * l - mu * mu) * (l * l - nu * nu), 0.0))
                    upper = sqrt(((l + 1) ** 2 - mu * mu)
                                 * ((l + 1) ** 2 - nu * nu))
                    value = (((2 * l + 1) * (l * (l + 1) * x - mu * nu) * current
                              - (l + 1) * lower * previous) / (l * upper))
                previous, current = current, value
                out[:, l + 1, nu + max_out, mu + max_in] = current
    return out


def _real_basis_matrix(degree, order):
    """Rows of the real harmonics in terms of the complex ones, one ``l``."""
    size = 2 * order + 1
    matrix = np.zeros((size, size), dtype=complex)
    matrix[order, order] = 1.0
    for m in range(1, order + 1):
        sign = (-1.0) ** m
        matrix[order + m, order + m] = sign / sqrt(2)
        matrix[order + m, order - m] = 1 / sqrt(2)
        matrix[order - m, order + m] = -1j * sign / sqrt(2)
        matrix[order - m, order - m] = 1j / sqrt(2)
    return matrix


def real_rotation_rows(degree, max_out, max_in, cos_theta, phi):
    r"""Real rotation rows ``E^l_(nu mu)`` for the frame whose axis is ``(theta, phi)``.

    Defined by ``Yreal_(l nu)(R^T v) = sum_mu E^l_(nu mu) Yreal_(l mu)(v)`` with
    ``R = Rz(phi) Ry(theta)``, so the same matrix also carries source moments
    from the source frame into the ``r_hat`` frame. Shape
    ``(points, degree+1, 2*max_out+1, 2*max_in+1)``.
    """
    d = wigner_small_d(degree, max_out, max_in, cos_theta)
    phi = np.atleast_1d(np.asarray(phi, float))
    if phi.shape != (d.shape[0],) or not np.isfinite(phi).all():
        raise ValueError("phi must match cos_theta in shape and be finite")
    mu = np.arange(-max_in, max_in + 1)
    carrier = np.exp(-1j * mu[None, :] * phi[:, None])         # (points, 2M+1)
    # D^l_(mu nu)(phi, theta, 0) = exp(-i mu phi) d^l_(mu nu); d is stored as [nu, mu].
    complex_rows = d * carrier[:, None, None, :]
    out_basis = _real_basis_matrix(degree, max_out)
    in_basis = _real_basis_matrix(degree, max_in)
    rows = np.einsum("ab,plbc,dc->plad", out_basis, complex_rows,
                     np.conjugate(in_basis), optimize=True)
    if np.max(np.abs(rows.imag)) > 1e-9 * max(np.max(np.abs(rows.real)), 1.0):
        raise FloatingPointError("real rotation rows acquired an imaginary part")
    return np.ascontiguousarray(rows.real)


# ------------------------------------------------------------------- cache

@dataclass
class DirectionalCache:
    """``t_(l lambda |mu|)(r, omega)`` on a distance grid, with the same lookup
    rule as :class:`~lighthit.cache.ResponseCache`.

    ``coefficients`` has shape ``(frequency, radius, source_degree+1, pairs, 2)``
    where the last axis holds the orders of
    :data:`~lighthit.cache.MOMENT_ORDERS` and ``pairs`` enumerates
    ``(lambda, |mu|)`` with ``0 <= |mu| <= lambda <= acceptance_degree``.
    """
    grid: CacheGrid
    medium: Medium
    settings: SolverSettings
    table: CouplingTable
    coefficients: np.ndarray
    timings_s: dict = field(default_factory=dict)
    angular_backend: str = "numpy"
    radial_phase: str = "none"
    _spline_cache: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        expected = (len(self.grid.omega_per_ns), len(self.grid.radii_m),
                    self.table.source_degree + 1, self.table.pairs,
                    len(MOMENT_ORDERS))
        self.coefficients = np.asarray(self.coefficients, dtype=complex)
        if self.coefficients.shape != expected:
            raise ValueError(f"coefficients must have shape {expected}")
        if len(self.grid.radii_m) < 2:
            raise ValueError("Interpolation requires at least two radii")
        if not np.isfinite(self.coefficients).all():
            raise ValueError("coefficients must be finite")
        if self.radial_phase not in ("none", "flight"):
            raise ValueError("radial_phase must be 'none' or 'flight'")

    @property
    def radius_range_m(self):
        return float(self.grid.radii_m[0]), float(self.grid.radii_m[-1])

    @property
    def degree(self):
        return self.table.source_degree

    @property
    def acceptance_degree(self):
        return self.table.acceptance_degree

    def clear_interpolation_cache(self):
        self._spline_cache.clear()

    # -- construction ----------------------------------------------------

    @classmethod
    def build(cls, medium, settings, grid, acceptance_degree, *,
              angular_backend="numpy", radius_chunk=None, progress=None,
              scratch_dir=None, radial_phase="none", report=None):
        """Contract the ``m``-block resolvent rows against a Bessel table.

        The loop structure mirrors :meth:`lighthit.cache.ResponseCache.build`:
        one Bessel table per radius block, one angular solve per frequency,
        one contraction. What changes is that the angular solve now returns the
        rows ``lambda <= acceptance_degree`` of every block ``|m| <= lambda``,
        and that the contraction runs over the ``(lambda, l, J)`` triples.
        """
        _check_backend(angular_backend)
        start = perf_counter()
        table = CouplingTable.build(settings.spatial_degree, acceptance_degree)
        radii = grid.radii_m
        omega = grid.omega_per_ns
        k, kw = settings.quadrature()
        L_s = table.source_degree
        L_A = table.acceptance_degree
        top_J = int(table.triple_J.max()) if table.triples else 0
        weight = (kw * k * k) / (2 * np.pi ** 2)
        if radius_chunk is None:
            per_radius = max(len(k) * (top_J + 1) * 16, 1)
            radius_chunk = max(1, int(256 * 1024 * 1024 // per_radius))
        radius_chunk = max(1, int(radius_chunk))

        coefficients = np.zeros((len(omega), len(radii), L_s + 1,
                                 table.pairs, len(MOMENT_ORDERS)), complex)
        bessel_time = angular_time = contract_time = 0.0
        scratch_bytes = 0
        max_block_bytes = 0
        solve_report = {}
        degrees = np.arange(top_J + 1)
        groups = [np.flatnonzero(table.triple_J == J) for J in range(top_J + 1)]
        rows_by_degree = [np.flatnonzero(table.triple_l == l) for l in range(L_s + 1)]
        # The Bessel table does not depend on the frequency. Building it inside
        # the frequency loop costs 133 s of a 147 s build at the production
        # settings; building it once costs 2.9 s. Same numbers, same order of
        # the final sum -- only where the table lives changes.
        with TemporaryDirectory(prefix="lighthit-bessel-", dir=scratch_dir) as tmp:
            paths = []
            memory_table = None
            if medium.scattering_per_m > 0:
                before = perf_counter()
                for low in range(0, len(radii), radius_chunk):
                    block = radii[low:low + radius_chunk]
                    bessel = spherical_jn(degrees[None, None, :],
                                          k[None, :, None] * block[:, None, None]
                                          ) * weight[None, :, None]
                    max_block_bytes = max(max_block_bytes, bessel.nbytes)
                    if len(radii) <= radius_chunk:
                        memory_table = bessel
                    else:
                        path = Path(tmp) / f"{low}.npy"
                        np.save(path, bessel, allow_pickle=False)
                        paths.append((low, path))
                        scratch_bytes += path.stat().st_size
                    del bessel
                bessel_time = perf_counter() - before
                for iw, frequency in enumerate(omega):
                    before = perf_counter()
                    first, multiple = resolvent_rows(
                        k, frequency, medium, settings.scattering_degree, L_s,
                        L_A, backend=angular_backend, report=solve_report)
                    angular_time += perf_counter() - before
                    before = perf_counter()
                    g = np.zeros((len(k), table.triples, len(MOMENT_ORDERS)),
                                 complex)
                    for axis, rows in enumerate((first, multiple)):
                        for (m, lam), block in rows.items():
                            index = np.flatnonzero(table.triple_lambda == lam)
                            if not len(index):
                                continue
                            g[:, index, axis] += (
                                table.nu_weight[index, m][None, :]
                                * block[:, table.triple_l[index]])
                    blocks = ([(0, memory_table)] if memory_table is not None
                              else [(low, None) for low, _ in paths])
                    for position, (low, cached) in enumerate(blocks):
                        if cached is None:
                            cached = np.load(paths[position][1], mmap_mode="r",
                                             allow_pickle=False)
                        moments = np.zeros((cached.shape[0], table.triples,
                                            len(MOMENT_ORDERS)), complex)
                        for J, index in enumerate(groups):
                            if not len(index):
                                continue
                            moments[:, index] = np.einsum(
                                "rk,kto->rto", cached[:, :, J], g[:, index],
                                optimize=True)
                        for l, rows_l in enumerate(rows_by_degree):
                            if not len(rows_l):
                                continue
                            coefficients[iw, low:low + cached.shape[0], l] = (
                                np.einsum("rto,tp->rpo", moments[:, rows_l],
                                          table.pair_weight[rows_l],
                                          optimize=True))
                        del cached
                    contract_time += perf_counter() - before
                    if progress is not None:
                        progress(iw + 1, len(omega))
        timings = dict(spatial_weights=bessel_time, angular_solve=angular_time,
                       contraction=contract_time, total=perf_counter() - start,
                       k_nodes=len(k), radii=len(radii), source_degree=L_s,
                       acceptance_degree=L_A, triples=table.triples,
                       pairs=table.pairs, frequencies=len(omega),
                       bessel_block_bytes=max_block_bytes,
                       temporary_disk_bytes=scratch_bytes,
                       output_bytes=coefficients.nbytes, **solve_report)
        if report is not None:
            report.update(timings)
        return cls(grid, medium, settings, table, coefficients, timings,
                   angular_backend, radial_phase=radial_phase)

    # -- lookup ----------------------------------------------------------

    def _scale(self, radii):
        radii = np.asarray(radii, float)
        return np.exp(-self.medium.absorption_per_m * radii) / (4 * np.pi * radii ** 2)

    def coefficients_at(self, radii_m):
        """Interpolated ``t``, shape (frequency, point, degree+1, pairs, 2)."""
        radii = np.atleast_1d(np.asarray(radii_m, float))
        if radii.ndim != 1 or not len(radii) or not np.isfinite(radii).all() or np.any(radii <= 0):
            raise ValueError("radii_m must be a nonempty finite positive 1-D array")
        low, high = self.radius_range_m
        if np.any(radii < low) or np.any(radii > high):
            raise ValueError(
                f"radii_m outside the cached range [{low:g}, {high:g}] m; "
                "extrapolation is not provided")
        key = ("spline", self.radial_phase)
        grid_r = self.grid.radii_m
        freq = self.grid.omega_per_ns
        if key not in self._spline_cache:
            scale = self._scale(grid_r)
            if np.any(scale == 0):
                raise FloatingPointError("Reference amplitude underflow: reduce radius range")
            detrended = self.coefficients / scale[None, :, None, None, None]
            if self.radial_phase == "flight":
                detrended = detrended * np.exp(
                    -1j * freq[:, None] * grid_r[None, :] / self.medium.speed_m_per_ns
                )[:, :, None, None, None]
            self._spline_cache[key] = CubicSpline(np.log(grid_r), detrended, axis=1)
        out = (self._spline_cache[key](np.log(radii))
               * self._scale(radii)[None, :, None, None, None])
        if self.radial_phase == "flight":
            out = out * np.exp(1j * freq[:, None] * radii[None, :]
                               / self.medium.speed_m_per_ns)[:, :, None, None, None]
        return out

    # -- persistence -----------------------------------------------------

    def _metadata(self):
        return dict(medium=asdict(self.medium), settings=asdict(self.settings),
                    acceptance_degree=self.table.acceptance_degree,
                    source_degree=self.table.source_degree,
                    timings_s=self.timings_s, angular_backend=self.angular_backend,
                    moment_orders=list(MOMENT_ORDERS), radial_phase=self.radial_phase,
                    detector_angular_model="exact_m_blocks",
                    units="photons per m^2 effective area")

    def save(self, path):
        from pathlib import Path
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, metadata=json.dumps(self._metadata()),
                            radii_m=self.grid.radii_m,
                            omega_per_ns=self.grid.omega_per_ns,
                            coefficients=self.coefficients)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"]))
            if tuple(metadata.get("moment_orders", MOMENT_ORDERS)) != MOMENT_ORDERS:
                raise ValueError("Unsupported cached scattering-order convention")
            grid = CacheGrid(data["radii_m"], data["omega_per_ns"])
            table = CouplingTable.build(metadata["source_degree"],
                                        metadata["acceptance_degree"])
            return cls(grid, Medium(**metadata["medium"]),
                       SolverSettings(**metadata["settings"]), table,
                       data["coefficients"], metadata.get("timings_s", {}),
                       metadata.get("angular_backend", "numpy"),
                       radial_phase=metadata.get("radial_phase", "none"))


# ------------------------------------------------------------------ apply

def _frame_angles(local_vectors):
    """``cos(theta)`` and ``phi`` of vectors already expressed in the source frame."""
    radius = np.linalg.norm(local_vectors, axis=-1)
    cosine = np.clip(local_vectors[..., 2] / radius, -1.0, 1.0)
    phi = np.arctan2(local_vectors[..., 1], local_vectors[..., 0])
    return radius, cosine, phi


def directional_response(cache, source, receivers_m, look_directions, alpha, *,
                         source_omega_per_ns, frequency_indices=None,
                         pair_chunk=64):
    r"""Equation (*) evaluated for every (cell, receiver) pair, in numpy.

    ``look_directions`` are unit vectors pointing the way the module looks, so a
    photon arriving head-on travels along ``+look``. Returns
    ``(frequency, receiver, 2)``.

    This is the readable definition of the kernel and the reference the fused
    kernel is tested against. Work is chunked over (receiver, cell) pairs, so
    nothing indexed by (receiver, cell, source channel, detector channel,
    frequency) is ever materialised; ``pair_chunk`` changes only the arithmetic
    order.
    """
    omega = np.asarray(source_omega_per_ns, float)
    receivers = np.atleast_2d(np.asarray(receivers_m, float))
    looks = np.atleast_2d(np.asarray(look_directions, float))
    alpha = np.asarray(alpha, float)
    if receivers.ndim != 2 or receivers.shape[1] != 3 or not len(receivers):
        raise ValueError("receivers must be a nonempty (D,3) array")
    if looks.shape != receivers.shape:
        raise ValueError("look_directions must match receivers")
    if not np.allclose(np.linalg.norm(looks, axis=1), 1.0, rtol=0, atol=1e-9):
        raise ValueError("look_directions must be unit vectors")
    if not isinstance(pair_chunk, (int, np.integer)) or pair_chunk < 1:
        raise ValueError("pair_chunk must be a positive integer")
    axis = np.asarray(cache.grid.omega_per_ns, float)
    if frequency_indices is None:
        indices = np.arange(len(axis))
    else:
        indices = np.asarray(frequency_indices)
        if indices.ndim != 1 or not len(indices) or indices.dtype.kind not in "iu":
            raise ValueError("frequency_indices must be a nonempty integer vector")
    if len(indices) != len(omega) or not np.array_equal(axis[indices], omega):
        raise ValueError("source and kernel frequency grids differ")
    table = cache.table
    L_A = table.acceptance_degree
    if len(alpha) < L_A + 1:
        raise ValueError("alpha must cover the acceptance degree of the cache")
    if source.degree < cache.degree:
        raise ValueError("the source carries fewer degrees than the cache")
    degree = cache.degree
    max_m = min(int(source.azimuthal_degree), degree)
    points = np.asarray(source.points_m(), float)
    if np.asarray(source.channels).shape[2] != len(omega):
        raise ValueError("source and kernel frequency grids differ")
    dense = dense_source_channels(source, degree, max_m)
    frame = source.frame
    centre = np.asarray(frame.centre_m, float)
    local_points = frame.rotate(points - centre)
    local_receivers = frame.rotate(receivers - centre)
    local_looks = frame.rotate(looks)
    alpha_pair = alpha[table.pair_lambda]
    cells = len(local_points)
    total = np.zeros((len(omega), len(receivers), len(MOMENT_ORDERS)), complex)
    for index in range(len(receivers)):
        for low in range(0, cells, pair_chunk):
            chunk = slice(low, min(low + pair_chunk, cells))
            vectors = local_receivers[index][None, :] - local_points[chunk]
            radius, cosine, phi = _frame_angles(vectors)
            rows = real_rotation_rows(degree, L_A, max_m, cosine, phi)
            rotated = np.einsum("plnm,plmw->plnw", rows, dense[chunk],
                                optimize=True)
            axis_local = _rotate_into_frame(
                np.broadcast_to(local_looks[index], vectors.shape), cosine, phi)
            harmonics = _real_harmonics(L_A, axis_local)
            coefficients = cache.coefficients_at(radius)[indices]
            weight = np.zeros((len(omega), len(radius), degree + 1,
                               2 * L_A + 1, len(MOMENT_ORDERS)), complex)
            for p, (lam, mu) in enumerate(zip(table.pair_lambda, table.pair_mu)):
                for sign in ((0,) if mu == 0 else (+1, -1)):
                    column = sign * mu + L_A
                    factor = alpha_pair[p] * harmonics[:, lam * lam + lam + sign * mu]
                    weight[:, :, :, column, :] += (coefficients[:, :, :, p, :]
                                                   * factor[None, :, None, None])
            total[:, index] += np.einsum("wplno,plnw->wo", weight, rotated,
                                         optimize=True)
    carrier = np.exp(1j * omega * source.reference_ns)
    return total * carrier[:, None, None]


def dense_source_channels(source, degree, max_m):
    """Source channels as a dense ``(cells, degree+1, 2*max_m+1, frequency)`` block."""
    channels = np.asarray(source.channels)
    dense = np.zeros((channels.shape[0], degree + 1, 2 * max_m + 1,
                      channels.shape[2]), dtype=complex)
    cursor = 0
    for l in range(source.degree + 1):
        width = 2 * min(l, source.azimuthal_degree) + 1
        if l <= degree:
            keep = min(l, max_m)
            middle = cursor + min(l, source.azimuthal_degree)
            dense[:, l, max_m - keep:max_m + keep + 1] = channels[
                :, middle - keep:middle + keep + 1]
        cursor += width
    return dense


def _rotate_into_frame(vectors, cosine, phi):
    """Components of ``vectors`` in the frame whose axis has angles (theta, phi)."""
    sin = np.sqrt(np.clip(1 - cosine ** 2, 0.0, None))
    cos_phi, sin_phi = np.cos(phi), np.sin(phi)
    e1 = np.stack((cosine * cos_phi, cosine * sin_phi, -sin), axis=-1)
    e2 = np.stack((-sin_phi, cos_phi, np.zeros_like(sin)), axis=-1)
    e3 = np.stack((sin * cos_phi, sin * sin_phi, cosine), axis=-1)
    return np.stack(((vectors * e1).sum(axis=-1), (vectors * e2).sum(axis=-1),
                     (vectors * e3).sum(axis=-1)), axis=-1)


def _real_harmonics(degree, vectors):
    from .experimental.event_moments import real_spherical_harmonics
    return real_spherical_harmonics(degree, vectors)


def rotation_rows_explicit(degree, max_out, max_in, cos_theta, phi):
    r"""The same ``E^l_(nu mu)`` as :func:`real_rotation_rows`, in real arithmetic.

    The complex route above is the definition; this is the arithmetic the fused
    kernel performs, written once in numpy so that it can be tested here and
    transcribed there. With ``d_ab = d^l_(ab)(theta)`` and ``p, n > 0``:

        E_(0,0)    = d_00
        E_(0,+n)   = sqrt2 (-1)^n d_(n0) cos(n phi)
        E_(0,-n)   = sqrt2 (-1)^n d_(n0) sin(n phi)
        E_(+p,0)   = sqrt2 (-1)^p d_(0p),            E_(-p,0) = 0
        C_pn = (-1)^p d_(np) + d_(n,-p),  S_pn = (-1)^p d_(np) - d_(n,-p)
        E_(+p,+n)  =  (-1)^n C_pn cos(n phi)
        E_(+p,-n)  =  (-1)^n C_pn sin(n phi)
        E_(-p,+n)  = -(-1)^n S_pn sin(n phi)
        E_(-p,-n)  =  (-1)^n S_pn cos(n phi)

    Only ``d^l_(mu nu)`` with ``0 <= mu <= max_in`` and ``|nu| <= max_out`` is
    needed, which is why the fused kernel carries a small Wigner table and not
    a rotation matrix.
    """
    degree = _degree(degree, "degree")
    max_out = _degree(max_out, "max_out")
    max_in = _degree(max_in, "max_in")
    d = wigner_small_d(degree, max_out, max_in, cos_theta)  # [point, l, nu, mu]
    phi = np.atleast_1d(np.asarray(phi, float))
    points = d.shape[0]
    out = np.zeros((points, degree + 1, 2 * max_out + 1, 2 * max_in + 1))
    root2 = np.sqrt(2.0)
    cosines = [np.cos(n * phi) for n in range(max_in + 1)]
    sines = [np.sin(n * phi) for n in range(max_in + 1)]

    def dval(l, mu, nu):
        return d[:, l, nu + max_out, mu + max_in]

    for l in range(degree + 1):
        out[:, l, max_out, max_in] = dval(l, 0, 0)
        for n in range(1, min(l, max_in) + 1):
            sign = (-1.0) ** n
            base = root2 * sign * dval(l, n, 0)
            out[:, l, max_out, max_in + n] = base * cosines[n]
            out[:, l, max_out, max_in - n] = base * sines[n]
        for p in range(1, min(l, max_out) + 1):
            out[:, l, max_out + p, max_in] = root2 * (-1.0) ** p * dval(l, 0, p)
            for n in range(1, min(l, max_in) + 1):
                cross = (-1.0) ** p * dval(l, n, p)
                straight = dval(l, n, -p)
                cc = (-1.0) ** n * (cross + straight)
                ss = (-1.0) ** n * (cross - straight)
                out[:, l, max_out + p, max_in + n] = cc * cosines[n]
                out[:, l, max_out + p, max_in - n] = cc * sines[n]
                out[:, l, max_out - p, max_in + n] = -ss * sines[n]
                out[:, l, max_out - p, max_in - n] = ss * cosines[n]
    return out
