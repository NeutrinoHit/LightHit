"""A reusable cache of multipole moments of the scattered response.

The angular solve of :mod:`lighthit.angular` depends on the medium, the
settings and the frequency, never on where the detector sits. A single
:class:`~lighthit.green.PointGreenSolver` call already shares it across the
observation points of that call, but nothing survives the call. This module
stores the radial contraction so that an arbitrary number of later
geometries costs a contraction instead of a solve.

What is stored is one step earlier than a response. The spatial inversion of
@sec-spatial is a sum over multipoles,

    K(r, nu, omega) = sum_l M_l(r, omega) P_l(nu),
    M_l(r, omega) = i^l sqrt((2l+1)/2) integral k^2 j_l(kr) h_l(k, omega) dk
                    / (2 pi^2),

and it is the moments ``M_l`` that are cached, not the sum. Keeping the
multipole index open is what makes one cache serve two different detectors,
since the weight applied to ``M_l`` is the only thing that distinguishes
them (@sec-om-acceptance):

* a directed flash read by an isotropic detector weights ``M_l`` by
  ``P_l(nu)``, with ``nu`` the cosine between the emission direction and the
  source-to-detector line;
* an isotropic flash read by a detector of acceptance ``A`` weights ``M_l``
  by ``alpha_l P_l(cos) / (4 pi)``, with ``alpha_l`` the Legendre
  coefficients of ``A`` and ``cos`` the cosine at which the source-to-detector
  line meets the head-on direction of the photocathode.

Both weights are exact, and neither is interpolated: the cache interpolates
in distance only, so the angular dependence carries no grid error at all.

For a directed point source, the output degree may need to follow ``k_max*r``.
For an isotropic source and a band-limited acceptance only its degrees are needed;
the scattering degree remains independent. :class:`BandedResponseCache` holds
one :class:`ResponseCache` per distance band, each with its own converged
settings, and dispatches a query to the band containing its radius.

Order 0 is analytic and is never cached. Order 1 is kept in both available
forms: as multipole moments of the finite-``L`` first order, which is what a
directional detector needs, and, for an isotropic detector, through the exact
full-Henyey-Greenstein quadrature of :mod:`lighthit.single` that the solver
itself reports. :func:`validation_report` measures the difference between
them rather than assuming it is negligible.
"""
from dataclasses import dataclass, asdict, field
from time import perf_counter
from tempfile import TemporaryDirectory
from pathlib import Path
import json
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.special import spherical_jn, eval_legendre, roots_legendre
from .angular import angular_components, _check_backend, _degree
from .green import SolverSettings, PointGreenSolver
from .medium import Medium
from .single import single_spectrum

def _validate_photons(photons):
    if not np.isfinite(photons) or photons < 0:
        raise ValueError("photons must be finite and nonnegative")


# Cached orders, in the order of the last moment axis.
MOMENT_ORDERS = ("one_finite_L", "two_or_more")


def _validate_axis(values, name, *, low=None, high=None):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError(f"{name} must be a nonempty finite 1-D array")
    if len(values) > 1 and np.any(np.diff(values) <= 0):
        raise ValueError(f"{name} must be strictly increasing")
    if low is not None and values[0] < low:
        raise ValueError(f"{name} must not go below {low}")
    if high is not None and values[-1] > high:
        raise ValueError(f"{name} must not exceed {high}")
    return values


def acceptance_coefficients(acceptance, degree, *, quadrature_order=512):
    """Legendre coefficients of a detector acceptance.

    ``acceptance`` maps the cosine between the detector axis and the photon
    arrival direction to a dimensionless efficiency. The returned
    ``alpha_l = 2 pi integral A(x) P_l(x) dx`` are normalized so that a
    perfectly isotropic, fully efficient detector has ``alpha_0 = 4 pi`` and
    no higher coefficient, reproducing the isotropic detector of
    :class:`~lighthit.green.PointGreenSolver`.
    """
    degree = _degree(degree, "degree")
    x, w = roots_legendre(int(quadrature_order))
    values = np.asarray(acceptance(x), dtype=float)
    if values.shape != x.shape or not np.isfinite(values).all():
        raise ValueError("acceptance must map an array of cosines to finite values")
    if np.any(values < 0):
        raise ValueError("acceptance must be nonnegative")
    ell = np.arange(degree + 1)
    return 2 * np.pi * (eval_legendre(ell[:, None], x[None, :]) * (w * values)).sum(axis=1)


def isotropic_acceptance(x):
    """A perfectly isotropic detector of unit efficiency."""
    return np.ones_like(np.asarray(x, dtype=float))


def cosine_acceptance(x):
    """A one-sided cosine response: ``max(x, 0)``, as a bare flat sensor."""
    return np.clip(np.asarray(x, dtype=float), 0.0, None)


def hemispherical_acceptance(x):
    """A smooth forward-weighted response, ``(1+x)/2``, nonzero over the sphere.

    This is a stand-in shape for a hemispherical photomultiplier behind a
    transparent sphere, chosen because it is smooth and has an elementary
    Legendre expansion. It is not a measured curve for any real module.
    """
    return (1.0 + np.asarray(x, dtype=float)) / 2.0


@dataclass(frozen=True)
class CacheGrid:
    """Distances and frequencies at which the numerical moments are stored.

    ``radii_m`` should be geometric rather than uniform: the response varies
    on a relative scale. ``omega_per_ns`` must contain zero for a cached
    charge to be defined.
    """
    radii_m: np.ndarray
    omega_per_ns: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, "radii_m",
                           _validate_axis(self.radii_m, "radii_m",
                                          low=np.nextafter(0, 1)))
        object.__setattr__(self, "omega_per_ns",
                           _validate_axis(self.omega_per_ns, "omega_per_ns", low=0.0))

    @classmethod
    def geometric(cls, r_min_m, r_max_m, n_radii, omega_per_ns):
        if not 0 < r_min_m < r_max_m:
            raise ValueError("Require 0 < r_min_m < r_max_m")
        _degree(n_radii, "n_radii")
        if n_radii < 4:
            raise ValueError("Cubic interpolation needs at least four radii")
        return cls(np.geomspace(r_min_m, r_max_m, n_radii),
                   np.atleast_1d(np.asarray(omega_per_ns, float)))


@dataclass
class ResponseCache:
    """Multipole moments on a distance grid, with interpolated lookup.

    ``moments`` has shape (frequency, radius, degree+1, 2); the last axis
    holds the orders named in :data:`MOMENT_ORDERS`. The units are those of
    :class:`~lighthit.green.GreenResult`, photons per square metre, once the
    angular weights are applied.
    """
    grid: CacheGrid
    medium: Medium
    settings: SolverSettings
    moments: np.ndarray
    timings_s: dict = field(default_factory=dict)
    angular_backend: str = "numpy"
    # "flight" interpolates exp(-i*omega*r/v)*moments. No readout is applied.
    radial_phase: str = "none"
    _spline_cache: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        expected = (len(self.grid.omega_per_ns), len(self.grid.radii_m),
                    self.settings.spatial_degree + 1, len(MOMENT_ORDERS))
        self.moments = np.asarray(self.moments, dtype=complex)
        if self.moments.shape != expected:
            raise ValueError(f"moments must have shape {expected}")
        if len(self.grid.radii_m) < 2:
            raise ValueError("Interpolation requires at least two radii")
        if not np.isfinite(self.moments).all():
            raise ValueError("moments must be finite")
        if self.radial_phase not in ("none", "flight"):
            raise ValueError("radial_phase must be 'none' or 'flight'")
        # The data are treated as immutable while a lazy interpolant exists.
        # Replacing/mutating them requires clear_interpolation_cache().

    def clear_interpolation_cache(self):
        """Invalidate lazy splines after an intentional change of moment data."""
        self._spline_cache.clear()

    @property
    def radius_range_m(self):
        return float(self.grid.radii_m[0]), float(self.grid.radii_m[-1])

    @property
    def degree(self):
        return self.settings.spatial_degree

    # -- construction ----------------------------------------------------

    @classmethod
    def build(cls, medium, settings, grid, *, angular_backend="numpy",
              radius_chunk=None, progress=None, scratch_dir=None, radial_phase="none"):
        """Contract the angular solution against a Bessel table, per radius.

        ``radius_chunk`` bounds the resident Bessel payload, not total RSS.
        With multiple blocks they are spooled to temporary .npy files under
        ``scratch_dir`` and mapped one at a time. The temporary files are
        removed even on failure. Output moments and one angular solve have
        additional memory costs. The default block target is 256 MiB.
        ``radial_phase='flight'`` changes only off-grid radial interpolation.
        """
        _check_backend(angular_backend)
        start = perf_counter()
        radii = grid.radii_m
        omega = grid.omega_per_ns
        k, kw = settings.quadrature()
        J = settings.spatial_degree
        ell = np.arange(J + 1)
        norm = np.sqrt((2 * ell + 1) / 2) * (1j) ** ell
        weight = (kw * k * k) / (2 * np.pi ** 2)
        if radius_chunk is None:
            per_radius = max(len(k) * (J + 1) * 16, 1)
            radius_chunk = max(1, int(256 * 1024 * 1024 // per_radius))
        radius_chunk = max(1, int(radius_chunk))

        moments = np.zeros((len(omega), len(radii), J + 1, len(MOMENT_ORDERS)),
                           complex)
        bessel_time = angular_time = contract_time = 0.0
        scratch_bytes = 0
        max_block_bytes = 0
        # Keep a single small block in RAM. If more blocks are needed, do
        # NOT keep them all in a list of ndarrays: that defeats radius_chunk.
        with TemporaryDirectory(prefix="lighthit-bessel-", dir=scratch_dir) as tmp:
            paths = []
            memory_table = None
            if medium.scattering_per_m > 0:
                before = perf_counter()
                for lo in range(0, len(radii), radius_chunk):
                    block = radii[lo:lo + radius_chunk]
                    table = np.empty((len(block), len(k), J + 1), complex)
                    for i, rad in enumerate(block):
                        table[i] = (spherical_jn(ell[None, :], k[:, None] * rad)
                                    * norm[None, :] * weight[:, None])
                    max_block_bytes = max(max_block_bytes, table.nbytes)
                    if len(radii) <= radius_chunk:
                        memory_table = table
                    else:
                        path = Path(tmp) / f"{lo}.npy"
                        np.save(path, table, allow_pickle=False)
                        paths.append((lo, path))
                        scratch_bytes += path.stat().st_size
                    del table
                bessel_time = perf_counter() - before
                for iw, frequency in enumerate(omega):
                    before = perf_counter()
                    _, first, tail = angular_components(
                        k, frequency, medium, settings.scattering_degree, J,
                        backend=angular_backend)
                    angular_time += perf_counter() - before
                    before = perf_counter()
                    if memory_table is not None:
                        for axis, block in enumerate((first[:, :J + 1], tail[:, :J + 1])):
                            moments[iw, :, :, axis] = np.einsum(
                                "rkj,kj->rj", memory_table, block, optimize=False)
                    else:
                        for lo, path in paths:
                            table = np.load(path, mmap_mode="r", allow_pickle=False)
                            for axis, block in enumerate((first[:, :J + 1], tail[:, :J + 1])):
                                moments[iw, lo:lo + len(table), :, axis] = np.einsum(
                                    "rkj,kj->rj", table, block, optimize=False)
                            del table  # release this mapping before the next block
                    contract_time += perf_counter() - before
                    if progress is not None:
                        progress(iw + 1, len(omega))
        timings = dict(spatial_weights=bessel_time, angular_solve=angular_time,
                       contraction=contract_time, total=perf_counter() - start,
                       k_nodes=len(k), radii=len(radii), degree=J,
                       frequencies=len(omega), bessel_block_bytes=max_block_bytes,
                       temporary_disk_bytes=scratch_bytes, output_bytes=moments.nbytes)
        return cls(grid, medium, settings, moments, timings, angular_backend,
                   radial_phase=radial_phase)

    # -- lookup ----------------------------------------------------------

    def _scale(self, radii):
        """A smooth positive reference that removes most of the dynamic range."""
        radii = np.asarray(radii, float)
        return np.exp(-self.medium.absorption_per_m * radii) / (4 * np.pi * radii ** 2)

    def moments_at(self, radii_m, *, degrees=None, frequency_index=None):
        """Interpolated moments, shape (frequency, point, degrees, 2).

        ``degrees`` keeps only the leading multipoles, which is what a smooth
        detector acceptance needs and avoids interpolating coefficients that
        are about to be multiplied by zero.
        """
        radii = np.atleast_1d(np.asarray(radii_m, float))
        if radii.ndim != 1 or not len(radii) or not np.isfinite(radii).all() or np.any(radii <= 0):
            raise ValueError("radii_m must be a nonempty finite positive 1-D array")
        if radii.ndim != 1 or not len(radii) or not np.isfinite(radii).all() or np.any(radii <= 0):
            raise ValueError("radii must be a nonempty finite positive 1-D array")
        low, high = self.radius_range_m
        if np.any(radii < low) or np.any(radii > high):
            raise ValueError(
                f"radii_m outside the cached range [{low:g}, {high:g}] m; "
                "extrapolation is not provided")
        top = self.degree + 1 if degrees is None else _degree(degrees, "degrees")
        if not 1 <= top <= self.degree + 1:
            raise ValueError("degrees must be between 1 and spatial_degree+1")
        if frequency_index is None:
            sel = slice(None)
        else:
            index = _degree(frequency_index, "frequency_index")
            if index >= len(self.grid.omega_per_ns):
                raise ValueError("frequency_index outside the cache")
            sel = slice(index, index + 1)
        freq = self.grid.omega_per_ns[sel]
        key = (top, frequency_index, self.radial_phase)
        if key not in self._spline_cache:
            grid_r = self.grid.radii_m
            block = self.moments[sel, :, :top]
            scale = self._scale(grid_r)
            if np.any(scale == 0):
                raise FloatingPointError("Reference amplitude underflow: reduce radius range")
            detrended = block / scale[None, :, None, None]
            if self.radial_phase == "flight":
                detrended = detrended * np.exp(
                    -1j * freq[:, None] * grid_r[None, :] / self.medium.speed_m_per_ns
                )[:, :, None, None]
            self._spline_cache[key] = CubicSpline(np.log(grid_r), detrended, axis=1)
        out = self._spline_cache[key](np.log(radii)) * self._scale(radii)[None, :, None, None]
        if self.radial_phase == "flight":
            out *= np.exp(1j * freq[:, None] * radii[None, :] /
                          self.medium.speed_m_per_ns)[:, :, None, None]
        return out

    def _contract(self, radii, weights, *, degrees=None):
        """Sum the interpolated moments against per-degree weights."""
        moments = self.moments_at(radii, degrees=degrees)
        return np.einsum("wrjo,rj->wro", moments, weights, optimize=True)

    def directed_spectrum(self, radii_m, cosines):
        """Directed flash, isotropic detector: weights are ``P_l(nu)``.

        Returns (frequency, point, 3). Order 1 is replaced by the exact
        full-Henyey-Greenstein quadrature, matching what the solver reports;
        order 0 vanishes off the forward ray.
        """
        radii = np.atleast_1d(np.asarray(radii_m, float))
        cos = np.atleast_1d(np.asarray(cosines, float))
        if cos.shape != radii.shape:
            raise ValueError("radii_m and cosines must have the same shape")
        if cos.ndim != 1 or not np.isfinite(cos).all() or np.any(np.abs(cos) > 1):
            raise ValueError("cosines must be a finite 1-D array in [-1, 1]")
        if np.any(cos >= 1 - 1e-12):
            raise ValueError("A directed point flash on the forward ray is singular")
        ell = np.arange(self.degree + 1)
        weights = eval_legendre(ell[None, :], cos[:, None])
        contracted = self._contract(radii, weights)
        out = np.zeros((len(self.grid.omega_per_ns), len(radii), 3), complex)
        out[:, :, 2] = contracted[:, :, 1]
        for i, rad in enumerate(radii):
            first, _ = single_spectrum(self.grid.omega_per_ns, float(rad),
                                       float(cos[i]), self.medium)
            out[:, i, 1] = first
        return out

    def acceptance_spectrum(self, radii_m, arrival_cosines, alpha, *,
                            exact_first_order=False, acceptance=None,
                            coefficient_tolerance=1e-12):
        """Isotropic flash read by a detector of acceptance coefficients ``alpha``.

        ``arrival_cosines`` is the argument of the acceptance itself: the
        cosine between the head-on direction of the photocathode and the
        source-to-detector line, so ``+1`` means the module faces the flash.
        :meth:`BandedResponseCache.charge_for_modules` derives it from module
        positions and an optical axis. Order 0 arrives exactly along the
        source-to-detector line and simply samples the acceptance there.
        """
        radii = np.atleast_1d(np.asarray(radii_m, float))
        cos = np.atleast_1d(np.asarray(arrival_cosines, float))
        if cos.shape != radii.shape:
            raise ValueError("radii_m and arrival_cosines must have the same shape")
        if np.any(np.abs(cos) > 1):
            raise ValueError("arrival_cosines must lie in [-1, 1]")
        alpha = np.asarray(alpha, dtype=float)
        if alpha.ndim != 1 or not len(alpha) or not np.isfinite(alpha).all():
            raise ValueError("alpha must be a nonempty finite 1-D array")
        if not np.isfinite(coefficient_tolerance) or coefficient_tolerance < 0:
            raise ValueError("coefficient_tolerance must be finite and nonnegative")
        peak = np.max(np.abs(alpha))
        if np.any(np.abs(alpha[self.degree + 1:]) > coefficient_tolerance * peak):
            raise ValueError("The cache has too few output degrees for this acceptance")
        if exact_first_order:
            if np.any(np.abs(alpha[1:]) > coefficient_tolerance * peak):
                raise ValueError("exact_first_order is implemented only for isotropic acceptance")
            if acceptance is not None:
                probe = np.asarray(acceptance(np.linspace(-1.0, 1.0, 65)), float)
                if not np.allclose(probe, alpha[0] / (4 * np.pi), rtol=1e-11, atol=1e-14):
                    raise ValueError("exact_first_order requires a constant, consistent acceptance")
        used = np.zeros(self.degree + 1)
        used[:min(len(alpha), self.degree + 1)] = alpha[:self.degree + 1]
        # The sum over degrees stops where the acceptance itself stops: a
        # smooth acceptance has few nonzero alpha_l, and evaluating Legendre
        # polynomials up to the spatial degree for every point would dominate
        # the query. Terms beyond the cut are zero, or below the tolerance.
        significant = np.flatnonzero(np.abs(used) > coefficient_tolerance
                                     * np.max(np.abs(used)))
        top = int(significant[-1]) if len(significant) else 0
        ell = np.arange(top + 1)
        weights = (eval_legendre(ell[None, :], cos[:, None]) * used[None, :top + 1]
                   / (4 * np.pi))
        contracted = self._contract(radii, weights, degrees=top + 1)
        omega = self.grid.omega_per_ns
        out = np.zeros((len(omega), len(radii), 3), complex)
        out[:, :, 1] = contracted[:, :, 0]
        out[:, :, 2] = contracted[:, :, 1]
        # Order 0 travels straight from the flash, so it arrives along the
        # source-to-detector direction and simply samples the acceptance
        # there. The callable is preferred when available: rebuilding a
        # non-smooth acceptance from truncated coefficients rings near a
        # kink, and the ballistic term is a point evaluation, not an integral.
        efficiency = (np.asarray(acceptance(cos), dtype=float) if acceptance is not None
                      else _acceptance_from_coefficients(used, cos))
        for i, rad in enumerate(radii):
            q0 = np.exp(-self.medium.extinction_per_m * rad) / (4 * np.pi * rad * rad)
            out[:, i, 0] = (efficiency[i] * q0
                            * np.exp(1j * omega * rad / self.medium.speed_m_per_ns))
            if exact_first_order:
                first, _ = single_spectrum(omega, float(rad), None, self.medium)
                out[:, i, 1] = (alpha[0] / (4 * np.pi)) * first
        return out

    def _zero_view(self):
        index = np.flatnonzero(self.grid.omega_per_ns == 0)
        if len(index) != 1:
            raise ValueError("Charge requires an explicitly cached omega=0")
        if len(self.grid.omega_per_ns) == 1:
            return self
        key = ("zero_view", self.radial_phase)
        if key not in self._spline_cache:
            i = int(index[0])
            self._spline_cache[key] = ResponseCache(
                CacheGrid(self.grid.radii_m, self.grid.omega_per_ns[i:i+1]),
                self.medium, self.settings, self.moments[i:i+1],
                angular_backend=self.angular_backend, radial_phase=self.radial_phase)
        return self._spline_cache[key]

    def directed_charge(self, radii_m, cosines, *, photons=1.0):
        _validate_photons(photons)
        view = self._zero_view()
        return photons * view.directed_spectrum(radii_m, cosines)[0].real

    def acceptance_charge(self, radii_m, arrival_cosines, alpha, *, photons=1.0,
                          exact_first_order=False, acceptance=None):
        _validate_photons(photons)
        view = self._zero_view()
        return photons * view.acceptance_spectrum(
            radii_m, arrival_cosines, alpha, exact_first_order=exact_first_order,
            acceptance=acceptance)[0].real

    def _charge(self, spectrum):
        index = np.flatnonzero(self.grid.omega_per_ns == 0)
        if len(index) != 1:
            raise ValueError("Charge requires an explicitly cached omega=0")
        return spectrum[index[0]].real

    # -- persistence -----------------------------------------------------

    def _arrays(self, prefix=""):
        return {f"{prefix}radii_m": self.grid.radii_m,
                f"{prefix}omega_per_ns": self.grid.omega_per_ns,
                f"{prefix}moments": self.moments}

    def _metadata(self):
        return dict(medium=asdict(self.medium), settings=asdict(self.settings),
                    timings_s=self.timings_s, angular_backend=self.angular_backend,
                    moment_orders=list(MOMENT_ORDERS), radial_phase=self.radial_phase,
                    units="photons per m^2 effective area")

    def save(self, path):
        from pathlib import Path
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, metadata=json.dumps(self._metadata()),
                            **self._arrays())

    @classmethod
    def _from_parts(cls, data, metadata, prefix=""):
        if tuple(metadata.get("moment_orders", MOMENT_ORDERS)) != MOMENT_ORDERS:
            raise ValueError("Unsupported cached scattering-order convention")
        grid = CacheGrid(data[f"{prefix}radii_m"], data[f"{prefix}omega_per_ns"])
        return cls(grid, Medium(**metadata["medium"]),
                   SolverSettings(**metadata["settings"]),
                   data[f"{prefix}moments"], metadata.get("timings_s", {}),
                   metadata.get("angular_backend", "numpy"),
                   radial_phase=metadata.get("radial_phase", "none"))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            return cls._from_parts(data, json.loads(str(data["metadata"])))


def _acceptance_from_coefficients(alpha, cosines, tolerance=1e-12):
    """Rebuild A(x) from its Legendre coefficients, for the ballistic term.

    Only the coefficients that carry weight are summed; the rest would
    contribute nothing while costing a Legendre evaluation per point.
    """
    alpha = np.asarray(alpha, dtype=float)
    peak = np.max(np.abs(alpha))
    significant = np.flatnonzero(np.abs(alpha) > tolerance * peak) if peak > 0 else []
    top = int(significant[-1]) if len(significant) else 0
    ell = np.arange(top + 1)
    basis = eval_legendre(ell[None, :], np.atleast_1d(cosines)[:, None])
    return basis @ (alpha[:top + 1] * (2 * ell + 1)) / (4 * np.pi)


@dataclass
class BandedResponseCache:
    """Several :class:`ResponseCache` bands covering one continuous range.

    Each band carries its own converged settings, because the spatial degree
    has to follow ``k_max * r``. Bands are given in increasing distance and
    must join without a gap; a query is served by the band containing its
    radius, the shared boundary going to the lower band.
    """
    bands: list

    def __post_init__(self):
        if not self.bands:
            raise ValueError("At least one band is required")
        first = self.bands[0]
        for band in self.bands[1:]:
            if band.medium != first.medium:
                raise ValueError("All bands must use the same medium")
            if not np.array_equal(band.grid.omega_per_ns, first.grid.omega_per_ns):
                raise ValueError("All bands must use exactly the same frequency axis")
        for lower, upper in zip(self.bands, self.bands[1:]):
            if not np.isclose(lower.radius_range_m[1], upper.radius_range_m[0],
                              rtol=1e-9, atol=0):
                raise ValueError("Bands must join without a gap or an overlap")

    @property
    def medium(self):
        return self.bands[0].medium

    @property
    def radius_range_m(self):
        return self.bands[0].radius_range_m[0], self.bands[-1].radius_range_m[1]

    @property
    def build_time_s(self):
        return sum(float(band.timings_s.get("total", 0.0)) for band in self.bands)

    def _dispatch(self, radii, call):
        radii = np.atleast_1d(np.asarray(radii, float))
        if radii.ndim != 1 or not len(radii) or not np.isfinite(radii).all() or np.any(radii <= 0):
            raise ValueError("radii must be a nonempty finite positive 1-D array")
        low, high = self.radius_range_m
        if np.any(radii < low) or np.any(radii > high):
            raise ValueError(f"radii outside the cached range [{low:g}, {high:g}] m")
        edges = np.array([band.radius_range_m[1] for band in self.bands[:-1]])
        which = np.searchsorted(edges, radii, side="left")
        n_omega = len(self.bands[0].grid.omega_per_ns)
        out = np.zeros((n_omega, len(radii), 3), complex)
        for index in np.unique(which):
            mask = which == index
            out[:, mask] = call(self.bands[index], mask)
        return out

    def directed_spectrum(self, radii_m, cosines):
        radii = np.atleast_1d(np.asarray(radii_m, float))
        cos = np.atleast_1d(np.asarray(cosines, float))
        return self._dispatch(
            radii, lambda band, mask: band.directed_spectrum(radii[mask], cos[mask]))

    def acceptance_spectrum(self, radii_m, arrival_cosines, alpha, *,
                            exact_first_order=False, acceptance=None,
                            coefficient_tolerance=1e-12):
        radii = np.atleast_1d(np.asarray(radii_m, float))
        cos = np.atleast_1d(np.asarray(arrival_cosines, float))
        return self._dispatch(
            radii, lambda band, mask: band.acceptance_spectrum(
                radii[mask], cos[mask], alpha, acceptance=acceptance,
                exact_first_order=exact_first_order,
                coefficient_tolerance=coefficient_tolerance))

    def _charge(self, spectrum):
        omega = self.bands[0].grid.omega_per_ns
        index = np.flatnonzero(omega == 0)
        if len(index) != 1:
            raise ValueError("Charge requires an explicitly cached omega=0")
        return spectrum[index[0]].real

    def _zero_view(self):
        return BandedResponseCache([band._zero_view() for band in self.bands])

    def directed_charge(self, radii_m, cosines, *, photons=1.0):
        _validate_photons(photons)
        return photons * self._zero_view().directed_spectrum(radii_m, cosines)[0].real

    def acceptance_charge(self, radii_m, arrival_cosines, alpha, *, photons=1.0,
                          exact_first_order=False, acceptance=None):
        _validate_photons(photons)
        return photons * self._zero_view().acceptance_spectrum(
            radii_m, arrival_cosines, alpha, exact_first_order=exact_first_order,
            acceptance=acceptance)[0].real

    def charge_for_modules(self, displacement_m, axis, alpha, *, photons=1.0,
                           acceptance=None):
        """Charge at modules sharing one optical axis, from an isotropic flash.

        ``displacement_m`` holds module positions relative to the flash and
        ``axis`` is the direction the photocathode faces, so a Baikal module
        looking at the lake floor takes ``axis=(0, 0, -1)``.
        """
        rvec = np.atleast_2d(np.asarray(displacement_m, float))
        if rvec.ndim != 2 or rvec.shape[1] != 3 or not np.isfinite(rvec).all():
            raise ValueError("displacement_m must have shape (3,) or (D,3)")
        radii = np.linalg.norm(rvec, axis=1)
        if np.any(radii <= 0):
            raise ValueError("Flash and module must be distinct")
        axis = np.asarray(axis, float)
        if axis.shape != (3,) or not np.isclose(np.linalg.norm(axis), 1,
                                                rtol=0, atol=1e-12):
            raise ValueError("axis must be a unit 3-vector")
        # ``axis`` points out of the photocathode, the way the module faces.
        # A ballistic photon travels away from the flash along r_hat, so it
        # arrives head-on when r_hat = -axis: the acceptance argument is
        # -(r_hat . axis), which is +1 for a module facing the flash.
        cos = np.clip(-(rvec @ axis) / radii, -1, 1)
        return self.acceptance_charge(radii, cos, alpha, photons=photons,
                                      acceptance=acceptance)

    def save(self, path):
        from pathlib import Path
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {}
        metadata = []
        for index, band in enumerate(self.bands):
            arrays.update(band._arrays(prefix=f"band{index}_"))
            metadata.append(band._metadata())
        np.savez_compressed(path, metadata=json.dumps(metadata), **arrays)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"]))
            return cls([ResponseCache._from_parts(data, meta, prefix=f"band{index}_")
                        for index, meta in enumerate(metadata)])


def validation_report(cache, radii_m, cosines, *, direction=(0.0, 0.0, 1.0)):
    """Compare interpolated directed charges with exact solves off the grid.

    The exact solve uses each band's own settings, so the residual isolates
    the interpolation error and excludes any discretisation the cache and the
    reference share. Errors are quoted relative to the *total* charge.
    Negative or sign-changing components indicate a numerical issue; they
    are not a physical zero-crossing of a nonnegative transport probability.
    """
    radii = np.atleast_1d(np.asarray(radii_m, float))
    cos = np.atleast_1d(np.asarray(cosines, float))
    bands = cache.bands if isinstance(cache, BandedResponseCache) else [cache]
    interpolated = cache.directed_charge(radii, cos)
    exact = np.zeros_like(interpolated)
    for band in bands:
        low, high = band.radius_range_m
        mask = (radii >= low) & (radii <= high)
        if band is not bands[0]:
            mask &= radii > low
        if not np.any(mask):
            continue
        solver = PointGreenSolver(band.medium, band.settings,
                                  angular_backend=band.angular_backend)
        sub, sub_cos = radii[mask], cos[mask]
        sin = np.sqrt(np.clip(1 - sub_cos ** 2, 0, None))
        axis = np.asarray(direction, float)
        if axis.shape != (3,) or not np.isfinite(axis).all() or not np.isclose(np.linalg.norm(axis), 1):
            raise ValueError("direction must be a unit 3-vector")
        e = np.eye(3)[np.argmin(np.abs(axis))]
        transverse = np.cross(axis, e)
        transverse /= np.linalg.norm(transverse)
        displacement = sub[:, None] * (sin[:, None] * transverse + sub_cos[:, None] * axis)
        exact[mask] = solver.solve([0.0], displacement, direction=direction
                                   ).components[0].real
    total_exact = exact.sum(axis=1)
    scale = np.abs(total_exact)
    total = np.abs(interpolated.sum(axis=1) - total_exact) / scale
    residual = np.abs(interpolated[:, 2] - exact[:, 2]) / scale
    return dict(n_points=int(len(radii)),
                max_relative_error_total=float(np.nanmax(total)),
                median_relative_error_total=float(np.nanmedian(total)),
                max_multiple_error_over_total=float(np.nanmax(residual)),
                median_multiple_error_over_total=float(np.nanmedian(residual)),
                radius_range_m=[float(radii.min()), float(radii.max())])


def first_order_consistency(cache, radii_m, *, alpha=None):
    """Compare the complete numerical first-order route with coordinate HG.

    The residual includes angular truncation, radial integration and interpolation;
    it must not be attributed to L alone.

    Both are isotropic-flash, isotropic-detector charges. The multipole route
    is the one a directional detector must use; the quadrature route is what
    the solver reports. Their difference is a property of ``L``.
    """
    radii = np.atleast_1d(np.asarray(radii_m, float))
    if alpha is None:
        alpha = np.array([4 * np.pi])
    cos = np.zeros_like(radii)
    multipole = cache.acceptance_charge(radii, cos, alpha)
    exact = cache.acceptance_charge(radii, cos, alpha, exact_first_order=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        relative = np.abs(multipole[:, 1] - exact[:, 1]) / np.abs(exact[:, 1])
        of_total = (np.abs(multipole[:, 1] - exact[:, 1])
                    / np.abs(exact.sum(axis=1)))
    return dict(n_points=int(len(radii)),
                max_relative_first_order=float(np.nanmax(relative)),
                median_relative_first_order=float(np.nanmedian(relative)),
                max_first_order_error_over_total=float(np.nanmax(of_total)),
                radius_range_m=[float(radii.min()), float(radii.max())])
