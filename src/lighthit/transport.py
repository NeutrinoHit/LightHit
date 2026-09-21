"""High-level multi-wavelength transport API."""
from dataclasses import asdict, dataclass, field, replace
from hashlib import sha256
from importlib.util import find_spec
from pathlib import Path
from time import perf_counter
from typing import Callable
import json

import numpy as np

from .ballistic import ballistic_directional
from .cache import CacheGrid, ResponseCache, acceptance_coefficients
from .directional import DirectionalCache, acceptance_bandwidth, directional_response
from .green import SolverSettings
from .model import DetectorArray, SpectralMedium, WavelengthQuadrature
from .readout import inverse_bins
from .single import single_bins
from .sources import CherenkovTrack, G4Shower, IsotropicFlash, SpectralLightElements


@dataclass(frozen=True)
class KernelConfig:
    """Numerical and operational settings shared by all source engines."""
    # Keep the established 0.00375 ns^-1 spacing (and therefore its long
    # 2*pi/delta-omega image period), while resolving the physical-time bins
    # with a substantially wider band.  Increasing only omega_max with the old
    # 41 nodes would move periodic Fourier images into the readout window.
    omega_per_ns: np.ndarray = field(default_factory=lambda: np.linspace(0.0, 1.2, 321))
    relative_time_edges_ns: np.ndarray = field(
        default_factory=lambda: np.arange(-60.0, 740.0 + 10.0, 20.0))
    wavelength_nodes: int = 9
    wavelength_range_nm: tuple[float, float] | None = None
    scattering_degree: int = 24
    source_degree: int = 32
    azimuthal_degree: int = 4
    cell_m: float = 0.12
    k_max_per_m: float = 8.0
    k_panel_per_m: float = 0.04
    k_order: int = 10
    radial_range_m: tuple[float, float] = (3.0, 300.0)
    radial_nodes: int = 220
    angular_backend: str = "auto"
    acceptance_degree: int | None = None
    acceptance_tolerance: float = 1e-10
    receiver_block: int = 8
    threshold_pe: float = 0.01
    cache_directory: str | Path | None = None
    allow_experimental: bool = False

    def __post_init__(self):
        omega = np.asarray(self.omega_per_ns, float)
        edges = np.asarray(self.relative_time_edges_ns, float)
        if (omega.ndim != 1 or not len(omega) or omega[0] != 0
                or not np.isfinite(omega).all() or np.any(np.diff(omega) <= 0)):
            raise ValueError("omega_per_ns must increase from zero")
        if (edges.ndim != 1 or len(edges) < 2 or not np.isfinite(edges).all()
                or np.any(np.diff(edges) <= 0)):
            raise ValueError("relative_time_edges_ns must strictly increase")
        if not 0 <= self.threshold_pe or not np.isfinite(self.threshold_pe):
            raise ValueError("threshold_pe must be finite and nonnegative")
        if self.angular_backend not in ("auto", "numpy", "numba"):
            raise ValueError("angular_backend must be 'auto', 'numpy' or 'numba'")
        object.__setattr__(self, "omega_per_ns", omega)
        object.__setattr__(self, "relative_time_edges_ns", edges)


@dataclass
class TransportResponse:
    """Detector response already folded over wavelength and OM efficiency."""
    detector: DetectorArray
    omega_per_ns: np.ndarray
    relative_time_edges_ns: np.ndarray
    spectrum_pe: np.ndarray
    components_pe: np.ndarray
    charge_components_pe: np.ndarray
    time_origin_ns: np.ndarray
    active: np.ndarray
    method: str
    metadata: dict

    @property
    def charge_pe(self):
        return self.charge_components_pe.sum(axis=1)

    @property
    def bins_pe(self):
        return self.components_pe.sum(axis=2)

    def components_at_frequency_cutoff(self, omega_max_per_ns):
        """Re-bin Fourier-derived orders using a prefix of the stored spectrum.

        Exact physical-time orders are copied unchanged.  This makes cutoff
        convergence checks cheap: build the widest frequency grid once, then
        compare nested cutoffs without rebuilding any transport cache.
        """
        cutoff = float(omega_max_per_ns)
        if not np.isfinite(cutoff) or cutoff <= 0:
            raise ValueError("omega_max_per_ns must be finite and positive")
        mask = self.omega_per_ns <= cutoff + 1e-12
        if np.count_nonzero(mask) < 2:
            raise ValueError("frequency cutoff must retain at least two nodes")
        omega = self.omega_per_ns[mask]
        phase = np.exp(-1j * omega[:, None] * self.time_origin_ns[None, :])
        relative = self.spectrum_pe[mask] * phase[:, :, None]
        components = self.components_pe.copy()
        orders = self.metadata.get("fourier_inverted_orders", (1, 2))
        for order in orders:
            if order not in (1, 2):
                raise ValueError("fourier_inverted_orders may contain only 1 and 2")
            components[:, :, order] = inverse_bins(
                omega, relative[:, :, order], self.relative_time_edges_ns)
        return components

    def save(self, path):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, omega_per_ns=self.omega_per_ns,
            relative_time_edges_ns=self.relative_time_edges_ns,
            spectrum_pe=self.spectrum_pe, components_pe=self.components_pe,
            charge_components_pe=self.charge_components_pe,
            time_origin_ns=self.time_origin_ns, active=self.active,
            metadata=json.dumps({"method": self.method, **self.metadata}, default=str))
        return path


@dataclass(frozen=True)
class _Method:
    function: Callable
    experimental: bool
    description: str


class TransportKernel:
    """One medium and detector, many sources and selectable engines.

    No global state is used.  Build/load the wavelength-dependent transport
    tables once, then call :meth:`transport` for each event.
    """
    def __init__(self, medium: SpectralMedium, detector: DetectorArray,
                 config: KernelConfig | None = None):
        self.medium = medium
        self.detector = detector
        self.config = config or KernelConfig()
        low, high = (self.config.wavelength_range_nm
                     or self.medium.wavelength_range_nm)
        self.wavelength = WavelengthQuadrature.gauss_legendre(
            low, high, self.config.wavelength_nodes)
        self._caches = {}
        # A method names how the SOURCE is handled. How the module is handled
        # is a property of the module, resolved from its measured acceptance
        # bandwidth and reported in metadata["detector_angular_model"]. The two
        # used to be conflated; they are not any more.
        self._methods = {
            "isotropic": _Method(self._transport_isotropic, False,
                                  "isotropic point flash; exact OM acceptance contraction"),
            "track": _Method(self._transport_elements, False,
                              "Cherenkov track split into axial source elements"),
            "axial": _Method(self._transport_elements, False,
                              "two-field axial G4/shower engine"),
            "directional": _Method(self._transport_directional, False,
                                   "axial source, exact directional OM, forced"),
            "axial_centroid": _Method(self._transport_axial, True,
                                       "superseded: OM acceptance at the source-centroid "
                                       "direction; kept for regression only"),
            "track_centroid": _Method(self._transport_axial, True,
                                       "superseded: track source with the centroid OM model"),
            "axial_full": _Method(self._transport_axial_full, True,
                                   "axial engine retaining all |m| <= source_degree"),
            "generic": _Method(self._transport_axial_full, True,
                               "slower full-m axial control for arbitrary element sources"),
        }
        self._acceptance = None
        self._directional_caches = {}

    def register_method(self, name, function, *, experimental=True, description="user method"):
        if not name or name in self._methods:
            raise ValueError("method name must be nonempty and new")
        self._methods[name] = _Method(function, bool(experimental), str(description))

    def available_methods(self):
        return {name: {"experimental": item.experimental,
                       "description": item.description} for name, item in self._methods.items()}

    def _solver_settings(self):
        c = self.config
        return SolverSettings(c.scattering_degree, c.source_degree,
                              c.k_max_per_m, c.k_panel_per_m, c.k_order, True)

    def resolved_angular_backend(self):
        """Concrete angular backend selected by the configuration.

        ``auto`` uses Numba when the optional dependency is installed and the
        NumPy reference implementation otherwise. An explicit ``numba`` choice
        still fails clearly when the dependency is absent.
        """
        if self.config.angular_backend != "auto":
            return self.config.angular_backend
        try:
            available = find_spec("numba") is not None
        except (ImportError, ValueError):
            available = False
        return "numba" if available else "numpy"

    def acceptance(self):
        """Measured Legendre coefficients and bandwidth of the module response.

        Cached on the kernel: the shape is common to every module of a BGVD-like
        array, only the orientation differs, so this is computed once and the
        request rotates nothing but an axis.
        """
        if self._acceptance is None:
            bandwidth, alpha, residual = acceptance_bandwidth(
                self.detector.angular_acceptance,
                max_degree=max(2 * self.config.scattering_degree, 64),
                tolerance=self.config.acceptance_tolerance)
            probe = max(2 * self.config.scattering_degree, 64)
            if bandwidth >= probe and self.config.acceptance_degree is None:
                raise ValueError(
                    "the module acceptance is not band-limited: its Legendre "
                    f"coefficients are still above the tolerance at degree {probe}. "
                    "A response clipped at zero inside [-1, 1] does this. Set "
                    "KernelConfig.acceptance_degree explicitly to accept a "
                    "truncation, and read the reported residual.")
            if self.config.acceptance_degree is not None:
                requested = int(self.config.acceptance_degree)
                if requested < bandwidth:
                    residual = float(np.max(np.abs(alpha[requested + 1:]))
                                     / np.max(np.abs(alpha)))
                bandwidth = requested
            if bandwidth > self.config.scattering_degree:
                raise ValueError(
                    f"module acceptance needs degree {bandwidth}, above "
                    f"scattering_degree={self.config.scattering_degree}")
            self._acceptance = {"degree": int(bandwidth),
                                "alpha": alpha[:bandwidth + 1].copy(),
                                "residual_above_degree": float(residual),
                                "full_alpha": alpha}
        return self._acceptance

    def _cache_key(self, wavelength_nm):
        band = self.medium.band(wavelength_nm)
        payload = {"medium": asdict(band), "settings": asdict(self._solver_settings()),
                   "omega": self.config.omega_per_ns.tolist(),
                   "range": list(self.config.radial_range_m),
                   "nodes": self.config.radial_nodes, "phase": "flight"}
        return sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:20]

    def _cache_for(self, wavelength_nm):
        wavelength_nm = float(wavelength_nm)
        key = self._cache_key(wavelength_nm)
        if key in self._caches:
            return self._caches[key]
        path = None
        if self.config.cache_directory is not None:
            directory = Path(self.config.cache_directory).expanduser().resolve()
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"transport-{key}.npz"
        if path is not None and path.exists():
            cache = ResponseCache.load(path)
        else:
            low, high = self.config.radial_range_m
            cache = ResponseCache.build(
                self.medium.band(wavelength_nm), self._solver_settings(),
                CacheGrid.geometric(low, high, self.config.radial_nodes,
                                    self.config.omega_per_ns),
                angular_backend=self.resolved_angular_backend(), radial_phase="flight")
            if path is not None:
                cache.save(path)
        self._caches[key] = cache
        return cache

    def _directional_cache_key(self, wavelength_nm):
        band = self.medium.band(wavelength_nm)
        payload = {"medium": asdict(band), "settings": asdict(self._solver_settings()),
                   "omega": self.config.omega_per_ns.tolist(),
                   "range": list(self.config.radial_range_m),
                   "nodes": self.config.radial_nodes, "phase": "flight",
                   "acceptance_degree": self.acceptance()["degree"],
                   "kernel": "directional_m_blocks"}
        return sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:20]

    def _directional_cache_for(self, wavelength_nm):
        wavelength_nm = float(wavelength_nm)
        key = self._directional_cache_key(wavelength_nm)
        if key in self._directional_caches:
            return self._directional_caches[key]
        path = None
        if self.config.cache_directory is not None:
            directory = Path(self.config.cache_directory).expanduser().resolve()
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"directional-{key}.npz"
        if path is not None and path.exists():
            cache = DirectionalCache.load(path)
        else:
            low, high = self.config.radial_range_m
            cache = DirectionalCache.build(
                self.medium.band(wavelength_nm), self._solver_settings(),
                CacheGrid.geometric(low, high, self.config.radial_nodes,
                                    self.config.omega_per_ns),
                self.acceptance()["degree"],
                angular_backend=self.resolved_angular_backend(),
                radial_phase="flight")
            if path is not None:
                cache.save(path)
        self._directional_caches[key] = cache
        return cache

    def cache_kinds(self, method="auto"):
        """Which radial tables ``method`` will actually ask for.

        ``"multipole"`` is the one-per-degree table of :mod:`lighthit.cache`;
        ``"directional"`` is the ``(l, lambda, |mu|)`` table of
        :mod:`lighthit.directional`. Building the wrong one is not wrong, only
        expensive, which is why :meth:`build` asks this first.
        """
        if method == "all":
            return ("multipole", "directional")
        if method in ("isotropic", "axial_centroid", "track_centroid"):
            return ("multipole",)
        if method in ("auto", "axial", "track", "directional", "axial_full",
                      "generic"):
            if method == "directional" or self.acceptance()["degree"] >= 1:
                return ("directional",)
            return ("multipole",)
        raise ValueError(f"unknown method {method!r}; available: "
                         f"{sorted(self._methods) + ['auto', 'all']}")

    def source_wavelengths(self, source):
        """The wavelengths a transport of ``source`` will actually ask for.

        A monochromatic flash asks for its own line and nothing else; every
        other source uses the configured quadrature. Prebuilding the quadrature
        for a monochromatic laser built two tables that were never read and
        then left the real one to be built on the first transport.
        """
        if isinstance(source, IsotropicFlash):
            return np.atleast_1d(np.asarray(self._flash_nodes(source)[0], float))
        return self.wavelength.wavelength_nm

    def build(self, wavelengths_nm=None, *, method="auto", source=None):
        """Build the tables ``method`` needs, at the wavelengths it will read.

        ``method="auto"`` builds what an element source would use on this
        detector: the directional table when the module has a measured
        acceptance bandwidth, the multipole table when it does not. Pass
        ``source=`` to resolve both the table and the wavelengths exactly
        instead of guessing, ``method="isotropic"`` before a flash, or
        ``method="all"`` for both tables. An explicit ``wavelengths_nm``
        overrides what ``source`` would have chosen.

        Building the multipole table and then transporting with the directional
        kernel was doing the expensive half of the work twice; so was building
        the quadrature for a source that only ever asks for one line.
        """
        if source is not None:
            method = self._auto_method(source)
        if wavelengths_nm is not None:
            wavelengths = np.atleast_1d(np.asarray(wavelengths_nm, float))
        elif source is not None:
            wavelengths = self.source_wavelengths(source)
        else:
            wavelengths = self.wavelength.wavelength_nm
        kinds = self.cache_kinds(method)
        for wavelength in wavelengths:
            if "multipole" in kinds:
                self._cache_for(float(wavelength))
            if "directional" in kinds:
                self._directional_cache_for(float(wavelength))
        return self

    def _auto_method(self, source):
        if isinstance(source, IsotropicFlash):
            return "isotropic"
        if isinstance(source, CherenkovTrack):
            return "track"
        if isinstance(source, (G4Shower, SpectralLightElements)):
            return "axial"
        raise TypeError("No automatic method for this source type")

    def transport(self, source, *, method="auto", allow_experimental=None):
        if method == "auto":
            method = self._auto_method(source)
        if method not in self._methods:
            raise ValueError(f"unknown method {method!r}; available: {sorted(self._methods)}")
        selected = self._methods[method]
        allowed = self.config.allow_experimental if allow_experimental is None else allow_experimental
        if selected.experimental and not allowed:
            raise ValueError(f"method {method!r} is experimental; pass allow_experimental=True")
        return selected.function(source, method=method)

    def _response(self, spectrum, components, charge, origins, active, method, metadata):
        return TransportResponse(
            self.detector, self.config.omega_per_ns.copy(),
            self.config.relative_time_edges_ns.copy(), spectrum, components,
            charge, origins, active, method, metadata)

    def _flash_nodes(self, source):
        if source.wavelength_nm is not None:
            return np.array([source.wavelength_nm]), np.array([source.photons])
        nodes, weights = self.wavelength.wavelength_nm, self.wavelength.weight_nm
        shape = np.asarray(source.spectral_shape(nodes), float)
        if shape.shape != nodes.shape or not np.isfinite(shape).all() or np.any(shape < 0):
            raise ValueError("flash spectral_shape must return finite nonnegative values")
        norm = float(weights @ shape)
        if norm <= 0:
            raise ValueError("flash spectral_shape has zero integral")
        return nodes, source.photons * weights * shape / norm

    def _transport_isotropic(self, source, *, method):
        if not isinstance(source, IsotropicFlash):
            raise TypeError("isotropic method requires IsotropicFlash")
        began = perf_counter()
        omega = self.config.omega_per_ns
        positions = self.detector.positions_m
        displacement = positions - source.position_m[None, :]
        radii = np.linalg.norm(displacement, axis=1)
        cosines = self.detector.head_on_cosine(source.position_m)
        wavelengths, photon_weight = self._flash_nodes(source)
        spectrum = np.zeros((len(omega), len(positions), 3), complex)
        ballistic_rows = []
        first_order_rows = []
        acceptance_probe = np.asarray(
            self.detector.angular_acceptance(np.linspace(-1.0, 1.0, 65)), float)
        constant_acceptance = (
            float(acceptance_probe[0])
            if np.allclose(acceptance_probe, acceptance_probe[0],
                           rtol=1e-11, atol=1e-14)
            else None)
        low, high = self.config.radial_range_m
        radial_inside = (radii >= low) & (radii <= high)
        upper = np.zeros(len(positions))
        for wavelength, photons in zip(wavelengths, photon_weight, strict=True):
            cache = self._cache_for(float(wavelength))
            alpha = acceptance_coefficients(self.detector.angular_acceptance, cache.degree)
            efficiency = float(self.detector.spectral_weight(float(wavelength)))
            scale = photons * self.detector.effective_area_m2 * efficiency
            if np.any(radial_inside):
                value = cache.acceptance_spectrum(
                    radii[radial_inside], cosines[radial_inside], alpha,
                    exact_first_order=constant_acceptance is not None,
                    acceptance=self.detector.angular_acceptance)
                spectrum[:, radial_inside] += value * scale[radial_inside][None, :, None]
            acceptance = np.asarray(self.detector.angular_acceptance(cosines), float)
            q0 = (scale * acceptance
                  * np.exp(-cache.medium.extinction_per_m * radii)
                  / (4 * np.pi * radii * radii))
            outside = ~radial_inside
            if np.any(outside):
                arrival = source.time_ns + radii[outside] / cache.medium.speed_m_per_ns
                spectrum[:, outside, 0] += (q0[outside][None, :]
                                             * np.exp(1j * omega[:, None] * arrival[None, :]))
            ballistic_rows.append((q0, source.time_ns + radii / cache.medium.speed_m_per_ns))
            if constant_acceptance is not None:
                first_order_rows.append((cache.medium, scale * constant_acceptance))
            upper += (photons * self.detector.effective_area_m2 * efficiency * acceptance
                      * np.exp(-cache.medium.absorption_per_m * radii)
                      / (4 * np.pi * radii * radii))
        unsafe = (~radial_inside) & (10 * upper >= self.config.threshold_pe)
        if np.any(unsafe):
            raise ValueError(
                f"{int(unsafe.sum())} OMs outside radial_range_m have an absorption-only "
                "estimate above threshold; extend a validated cache range")
        fastest = min(self.medium.band(float(w)).group_index for w in wavelengths)
        origins = source.time_ns + radii / (0.299792458 / fastest)
        relative = spectrum * np.exp(-1j * omega[:, None] * origins[None, :])[:, :, None]
        edges = self.config.relative_time_edges_ns
        components = np.zeros((len(positions), len(edges) - 1, 3))
        if len(omega) > 1:
            fourier_orders = (2,) if constant_acceptance is not None else (1, 2)
            for order in fourier_orders:
                components[:, :, order] = inverse_bins(
                    omega, relative[:, :, order], edges)
        else:
            fourier_orders = ()
        if constant_acceptance is not None:
            absolute_edges = edges[None, :] + origins[:, None] - source.time_ns
            for band, scale in first_order_rows:
                for detector_index in np.flatnonzero(radial_inside):
                    components[detector_index, :, 1] += scale[detector_index] * single_bins(
                        absolute_edges[detector_index], radii[detector_index], None, band,
                        backend=self.resolved_angular_backend())
        for weight, arrival in ballistic_rows:
            index = np.searchsorted(edges, arrival - origins, side="right") - 1
            bin_inside = (index >= 0) & (index < len(edges) - 1)
            np.add.at(components[:, :, 0],
                      (np.flatnonzero(bin_inside), index[bin_inside]), weight[bin_inside])
        charge = spectrum[0].real
        active = radial_inside & (charge.sum(axis=1) >= self.config.threshold_pe)
        if self.config.threshold_pe == 0:
            active = radial_inside
        return self._response(
            spectrum, components, charge, origins, active, method,
            {"elapsed_seconds": perf_counter() - began,
             "wavelength_nm": wavelengths.tolist(),
             "detector_angular_model": "exact multipole contraction",
             "first_order": (
                 "exact full-HG spectrum and physical-time bins"
                 if constant_acceptance is not None else
                 "finite-L directional-acceptance spectrum and Fourier bins"),
             "fourier_inverted_orders": list(fourier_orders),
             "radial_range_m": list(self.config.radial_range_m),
             "omitted_outside_range": int((~radial_inside).sum())})

    def _source_elements(self, source):
        if isinstance(source, G4Shower):
            return source.elements
        if isinstance(source, SpectralLightElements):
            return source
        if isinstance(source, CherenkovTrack):
            return source.to_elements(step_m=max(self.config.cell_m, 0.25))
        raise TypeError("axial/track methods require G4Shower, SpectralLightElements or CherenkovTrack")

    def _transport_elements(self, source, *, method, azimuthal_degree=None):
        """Route an element source to the module model its detector needs.

        A directional module gets the exact ``m``-block kernel; an isotropic one
        keeps the older engine, for which the centroid cosine is not an
        approximation at all --- a constant acceptance is a constant. Nothing
        that does not need the directional kernel pays for it.
        """
        if self.acceptance()["degree"] >= 1:
            return self._transport_directional(source, method=method,
                                                azimuthal_degree=azimuthal_degree)
        return self._transport_axial(source, method=method,
                                     azimuthal_degree=azimuthal_degree)

    def _transport_axial_full(self, source, *, method):
        return self._transport_elements(source, method=method,
                                        azimuthal_degree=self.config.source_degree)

    def _transport_axial(self, source, *, method, azimuthal_degree=None):
        from .experimental.axial_source import AxisFrame, AxialSource
        try:
            from .experimental.axial_fast import PreparedAxialKernel, compile_axial_source_fast
            compiler = compile_axial_source_fast
            backend = "numba axial"
        except ImportError:  # pragma: no cover - no-numba installation
            PreparedAxialKernel = None
            compiler = AxialSource.of
            backend = "numpy axial"
        from .experimental.ballistic_fast import (vectorised_ballistic_bins_fast,
                                                   vectorised_ballistic_fast)

        began = perf_counter()
        elements = self._source_elements(source)
        sample = self.medium.sample(self.wavelength.wavelength_nm)
        original_elements = len(elements)
        original_coefficient = float(elements.coefficient0.sum())
        valid = elements.beta * float(np.min(sample["phase_index"])) > 1
        if not np.any(valid):
            raise ValueError("source is below Cherenkov threshold over the wavelength range")
        if not np.all(valid):
            elements = elements.subset(valid)
        dropped_fraction = 1 - float(elements.coefficient0.sum()) / original_coefficient
        field0, field2 = elements.field(0), elements.field(2)
        frame = AxisFrame.of(field0)
        degree = self.config.source_degree
        max_m = self.config.azimuthal_degree if azimuthal_degree is None else azimuthal_degree
        kwargs = dict(azimuthal_degree=max_m, cell_m=self.config.cell_m,
                      element_order=2, frame=frame)
        source0 = compiler(field0, degree, self.config.omega_per_ns, **kwargs)
        source2 = compiler(field2, degree, self.config.omega_per_ns, **kwargs)
        compile_seconds = perf_counter() - began
        omega = self.config.omega_per_ns
        positions = self.detector.positions_m
        angular = self.detector.angular_weight(elements.centroid_m)
        scale_area = self.detector.effective_area_m2 * angular
        low, high = self.config.radial_range_m
        centre_distance = np.linalg.norm(positions - elements.centroid_m, axis=1)
        source_extent = float(np.max(np.linalg.norm(
            source0.points_m() - elements.centroid_m[None, :], axis=1)))
        # PreparedAxialKernel presently requires every retained source cell for
        # one OM to lie inside one cache. The bounding sphere is conservative.
        inside = ((centre_distance - source_extent >= low)
                  & (centre_distance + source_extent <= high))
        source0_factor = float(np.sum(
            self.wavelength.weight_nm / self.wavelength.wavelength_nm ** 2))
        source2_factor = float(np.sum(
            self.wavelength.weight_nm
            / (self.wavelength.wavelength_nm ** 2 * sample["phase_index"] ** 2)))
        emitted = float(elements.coefficient0.sum() * source0_factor
                        + elements.coefficient2.sum() * source2_factor)
        lower_distance = np.maximum(centre_distance - source_extent, low)
        upper = (10 * max(emitted, 0.0) * scale_area
                 * float(np.max(self.detector.spectral_weight(self.wavelength.wavelength_nm)))
                 * np.exp(-float(np.min(sample["absorption_per_m"])) * lower_distance)
                 / (4 * np.pi * lower_distance ** 2))
        unsafe = (~inside) & (upper >= self.config.threshold_pe)
        if np.any(unsafe):
            raise ValueError(
                f"{int(unsafe.sum())} OMs outside the validated axial radial range "
                "have a conservative estimate above threshold")
        work = np.flatnonzero(inside)
        # A common time origin; wavelength-dependent group delays remain inside
        # each spectral component before the wavelength sum.
        first = int(np.argmin(elements.start_ns))
        fastest_group = float(np.min(sample["group_index"]))
        origins = (float(elements.start_ns[first])
                   + np.linalg.norm(positions - elements.start_m[first], axis=1)
                   / (0.299792458 / fastest_group))
        charge = np.zeros((len(positions), 3))
        zero0 = replace(source0, channels=np.ascontiguousarray(source0.channels[:, :, :1]))
        zero2 = replace(source2, channels=np.ascontiguousarray(source2.channels[:, :, :1]))
        prepared = []
        # Exact omega=0 prepass in the current two-field/axial model.
        for wavelength, weight, phase in zip(
                self.wavelength.wavelength_nm, self.wavelength.weight_nm,
                sample["phase_index"], strict=True):
            cache = self._cache_for(float(wavelength))
            if PreparedAxialKernel is None:
                raise ImportError("multi-wavelength axial production requires lighthit[accelerate]")
            full = PreparedAxialKernel.from_cache(cache, degree=degree)
            zero = PreparedAxialKernel.from_cache(cache, degree=degree, frequency_indices=[0])
            prepared.append((cache, full))
            r0 = zero.apply(zero0, positions[work], source_omega_per_ns=omega[:1],
                            receiver_block=self.config.receiver_block)[0].real
            r2 = zero.apply(zero2, positions[work], source_omega_per_ns=omega[:1],
                            receiver_block=self.config.receiver_block)[0].real
            s0 = 1.0 / wavelength ** 2
            s2 = 1.0 / (wavelength ** 2 * phase ** 2)
            ballistic = (s0 * vectorised_ballistic_fast(
                field0, positions[work], cache.medium, field0.cone_cosine)
                - s2 * vectorised_ballistic_fast(
                    field2, positions[work], cache.medium, field2.cone_cosine))
            detector_scale = (weight * scale_area[work]
                              * float(self.detector.spectral_weight(float(wavelength))))
            charge[work, 0] += detector_scale * ballistic
            charge[work, 1:] += detector_scale[:, None] * (s0 * r0 - s2 * r2)
        active = inside & (charge.sum(axis=1) >= self.config.threshold_pe)
        if self.config.threshold_pe == 0:
            active = inside
        spectrum = np.zeros((len(omega), len(positions), 3), complex)
        components = np.zeros((len(positions), len(self.config.relative_time_edges_ns) - 1, 3))
        spectrum[0] = charge
        apply_start = perf_counter()
        for (wavelength, weight, phase, group_index, (cache, full)) in zip(
                self.wavelength.wavelength_nm, self.wavelength.weight_nm,
                sample["phase_index"], sample["group_index"], prepared, strict=True):
            s0 = 1.0 / wavelength ** 2
            s2 = 1.0 / (wavelength ** 2 * phase ** 2)
            detector_scale = (weight * scale_area[active]
                              * float(self.detector.spectral_weight(float(wavelength))))
            r0 = full.apply(source0, positions[active], source_omega_per_ns=omega,
                            receiver_block=self.config.receiver_block)
            r2 = full.apply(source2, positions[active], source_omega_per_ns=omega,
                            receiver_block=self.config.receiver_block)
            spectrum[:, active, 1:] += detector_scale[None, :, None] * (s0 * r0 - s2 * r2)
            _, b0 = vectorised_ballistic_bins_fast(
                field0, positions[active], cache.medium, field0.cone_cosine,
                origins[active], self.config.relative_time_edges_ns)
            _, b2 = vectorised_ballistic_bins_fast(
                field2, positions[active], cache.medium, field2.cone_cosine,
                origins[active], self.config.relative_time_edges_ns)
            components[active, :, 0] += detector_scale[:, None] * (s0 * b0 - s2 * b2)
        # Preserve exact prepass charges at omega=0 after nonzero-frequency fill.
        spectrum[0] = charge
        relative = (spectrum[:, active, 1:]
                    * np.exp(-1j * omega[:, None] * origins[None, active])[:, :, None])
        if len(omega) > 1:
            for order in range(2):
                components[active, :, order + 1] = inverse_bins(
                    omega, relative[:, :, order], self.config.relative_time_edges_ns)
        apply_seconds = perf_counter() - apply_start
        return self._response(
            spectrum, components, charge, origins, active, method,
            {"elapsed_seconds": perf_counter() - began,
             "compile_seconds": compile_seconds, "apply_seconds": apply_seconds,
             "backend": backend, "wavelength_nodes": len(self.wavelength.wavelength_nm),
             "spectral_source": "S0=lambda^-2; S2=lambda^-2*n_phase^-2",
             "source_fields": 2, "reference_phase_index": elements.reference_phase_index,
             "source_elements": len(elements),
             "source_elements_dropped_at_spectral_threshold": original_elements - len(elements),
             "source_coefficient_fraction_dropped": dropped_fraction,
             "active_modules": int(active.sum()), "threshold_pe": self.config.threshold_pe,
             "fourier_inverted_orders": [1, 2] if len(omega) > 1 else [],
             "radial_range_m": list(self.config.radial_range_m),
             "omitted_outside_range": int((~inside).sum()),
             "detector_angular_model": (
                 "source-centroid direction for all axial components; exact arbitrary "
                 "directional-OM axial transport is not yet implemented"),
             "ballistic_spectrum": "charge and exact bins stored; nonzero-frequency column omitted"})


    def _transport_directional(self, source, *, method, azimuthal_degree=None):
        """Axial source, exact directional module response, no centroid cosine.

        The only difference from :meth:`_transport_axial` in the physics is
        where the module acceptance enters: not as one scalar per OM taken at
        the source centroid, but inside the angular contraction for the
        scattered orders (equation (*) of :mod:`lighthit.directional`) and at
        the exact arrival direction of every element for the ballistic term.
        Everything else -- the two-field spectral model, the cone, the radial
        cache rule, the time origin, the readout -- is unchanged.
        """
        from .experimental.axial_source import AxisFrame, AxialSource
        try:
            from .experimental.axial_fast import compile_axial_source_fast
            compiler = compile_axial_source_fast
            compile_backend = "numba axial compile"
        except ImportError:  # pragma: no cover - no-numba installation
            compiler = AxialSource.of
            compile_backend = "numpy axial compile"
        try:
            from ._directional_numba import PreparedDirectionalKernel
            from .experimental.ballistic_fast import ballistic_directional_fast
            ballistic = ballistic_directional_fast
            apply_backend = "numba directional"
        except ImportError:  # pragma: no cover - no-numba installation
            PreparedDirectionalKernel = None
            ballistic = ballistic_directional
            apply_backend = "numpy directional"

        began = perf_counter()
        response = self.acceptance()
        alpha = np.ascontiguousarray(response["alpha"], float)
        elements = self._source_elements(source)
        sample = self.medium.sample(self.wavelength.wavelength_nm)
        original_elements = len(elements)
        original_coefficient = float(elements.coefficient0.sum())
        valid = elements.beta * float(np.min(sample["phase_index"])) > 1
        if not np.any(valid):
            raise ValueError("source is below Cherenkov threshold over the wavelength range")
        if not np.all(valid):
            elements = elements.subset(valid)
        dropped_fraction = 1 - float(elements.coefficient0.sum()) / original_coefficient
        field0, field2 = elements.field(0), elements.field(2)
        frame = AxisFrame.of(field0)
        degree = self.config.source_degree
        max_m = (self.config.azimuthal_degree if azimuthal_degree is None
                 else int(azimuthal_degree))
        kwargs = dict(azimuthal_degree=max_m, cell_m=self.config.cell_m,
                      element_order=2, frame=frame)
        source0 = compiler(field0, degree, self.config.omega_per_ns, **kwargs)
        source2 = compiler(field2, degree, self.config.omega_per_ns, **kwargs)
        compile_seconds = perf_counter() - began

        omega = self.config.omega_per_ns
        positions = self.detector.positions_m
        # The module looks the way a head-on photon travels: DetectorArray
        # orientations point towards the source, so the acceptance argument is
        # s_hat . (-orientation). This sign is applied once, here.
        looks = np.ascontiguousarray(-self.detector.orientations, float)
        scale_area = np.asarray(self.detector.effective_area_m2, float)
        low, high = self.config.radial_range_m
        centre_distance = np.linalg.norm(positions - elements.centroid_m, axis=1)
        source_extent = float(np.max(np.linalg.norm(
            source0.points_m() - elements.centroid_m[None, :], axis=1)))
        inside = ((centre_distance - source_extent >= low)
                  & (centre_distance + source_extent <= high))
        source0_factor = float(np.sum(
            self.wavelength.weight_nm / self.wavelength.wavelength_nm ** 2))
        source2_factor = float(np.sum(
            self.wavelength.weight_nm
            / (self.wavelength.wavelength_nm ** 2 * sample["phase_index"] ** 2)))
        emitted = float(elements.coefficient0.sum() * source0_factor
                        + elements.coefficient2.sum() * source2_factor)
        peak_acceptance = float(np.max(np.abs(np.asarray(
            self.detector.angular_acceptance(np.linspace(-1, 1, 257)), float))))
        lower_distance = np.maximum(centre_distance - source_extent, low)
        upper = (10 * max(emitted, 0.0) * scale_area * peak_acceptance
                 * float(np.max(self.detector.spectral_weight(self.wavelength.wavelength_nm)))
                 * np.exp(-float(np.min(sample["absorption_per_m"])) * lower_distance)
                 / (4 * np.pi * lower_distance ** 2))
        unsafe = (~inside) & (upper >= self.config.threshold_pe)
        if np.any(unsafe):
            raise ValueError(
                f"{int(unsafe.sum())} OMs outside the validated directional radial "
                "range have a conservative estimate above threshold")
        work = np.flatnonzero(inside)
        first = int(np.argmin(elements.start_ns))
        fastest_group = float(np.min(sample["group_index"]))
        origins = (float(elements.start_ns[first])
                   + np.linalg.norm(positions - elements.start_m[first], axis=1)
                   / (0.299792458 / fastest_group))
        charge = np.zeros((len(positions), 3))
        zero0 = replace(source0, channels=np.ascontiguousarray(source0.channels[:, :, :1]))
        zero2 = replace(source2, channels=np.ascontiguousarray(source2.channels[:, :, :1]))
        cache_seconds = 0.0
        prepass_seconds = 0.0
        caches = []
        for wavelength, weight, phase in zip(
                self.wavelength.wavelength_nm, self.wavelength.weight_nm,
                sample["phase_index"], strict=True):
            before = perf_counter()
            cache = self._directional_cache_for(float(wavelength))
            cache_seconds += perf_counter() - before
            caches.append(cache)
            before = perf_counter()
            r0, r2 = (self._directional_apply(
                PreparedDirectionalKernel, cache, item, positions[work],
                looks[work], alpha, omega[:1]) for item in (zero0, zero2))
            s0 = 1.0 / wavelength ** 2
            s2 = 1.0 / (wavelength ** 2 * phase ** 2)
            ballistic_charge = (
                s0 * ballistic(field0, positions[work], looks[work], cache.medium,
                               field0.cone_cosine, alpha)[0]
                - s2 * ballistic(field2, positions[work], looks[work], cache.medium,
                                 field2.cone_cosine, alpha)[0])
            detector_scale = (weight * scale_area[work]
                              * float(self.detector.spectral_weight(float(wavelength))))
            charge[work, 0] += detector_scale * ballistic_charge
            charge[work, 1:] += detector_scale[:, None] * (
                s0 * r0[0].real - s2 * r2[0].real)
            prepass_seconds += perf_counter() - before
        active = inside & (charge.sum(axis=1) >= self.config.threshold_pe)
        if self.config.threshold_pe == 0:
            active = inside
        spectrum = np.zeros((len(omega), len(positions), 3), complex)
        components = np.zeros((len(positions), len(self.config.relative_time_edges_ns) - 1, 3))
        spectrum[0] = charge
        apply_start = perf_counter()
        # Every module can fall below threshold -- a detector looking away from
        # the event does exactly that -- and an empty selection is a valid
        # answer, not an error.
        for (wavelength, weight, phase, cache) in ([] if not np.any(active) else zip(
                self.wavelength.wavelength_nm, self.wavelength.weight_nm,
                sample["phase_index"], caches, strict=True)):
            s0 = 1.0 / wavelength ** 2
            s2 = 1.0 / (wavelength ** 2 * phase ** 2)
            detector_scale = (weight * scale_area[active]
                              * float(self.detector.spectral_weight(float(wavelength))))
            r0, r2 = (self._directional_apply(
                PreparedDirectionalKernel, cache, item, positions[active],
                looks[active], alpha, omega) for item in (source0, source2))
            spectrum[:, active, 1:] += detector_scale[None, :, None] * (s0 * r0 - s2 * r2)
            _, b0 = ballistic(field0, positions[active], looks[active], cache.medium,
                              field0.cone_cosine, alpha,
                              time_origin_ns=origins[active],
                              relative_edges_ns=self.config.relative_time_edges_ns)
            _, b2 = ballistic(field2, positions[active], looks[active], cache.medium,
                              field2.cone_cosine, alpha,
                              time_origin_ns=origins[active],
                              relative_edges_ns=self.config.relative_time_edges_ns)
            components[active, :, 0] += detector_scale[:, None] * (s0 * b0 - s2 * b2)
        spectrum[0] = charge
        relative = (spectrum[:, active, 1:]
                    * np.exp(-1j * omega[:, None] * origins[None, active])[:, :, None])
        if len(omega) > 1 and np.any(active):
            for order in range(2):
                components[active, :, order + 1] = inverse_bins(
                    omega, relative[:, :, order], self.config.relative_time_edges_ns)
        apply_seconds = perf_counter() - apply_start
        return self._response(
            spectrum, components, charge, origins, active, method,
            {"elapsed_seconds": perf_counter() - began,
             "compile_seconds": compile_seconds, "apply_seconds": apply_seconds,
             "cache_seconds": cache_seconds, "prepass_seconds": prepass_seconds,
             "backend": f"{compile_backend} + {apply_backend}",
             "wavelength_nodes": len(self.wavelength.wavelength_nm),
             "spectral_source": "S0=lambda^-2; S2=lambda^-2*n_phase^-2",
             "source_fields": 2, "reference_phase_index": elements.reference_phase_index,
             "source_elements": len(elements),
             "source_elements_dropped_at_spectral_threshold": original_elements - len(elements),
             "source_coefficient_fraction_dropped": dropped_fraction,
             "active_modules": int(active.sum()), "threshold_pe": self.config.threshold_pe,
             "fourier_inverted_orders": [1, 2] if len(omega) > 1 else [],
             "radial_range_m": list(self.config.radial_range_m),
             "omitted_outside_range": int((~inside).sum()),
             "detector_angular_model": "exact_m_blocks",
             "detector_acceptance_degree": response["degree"],
             "detector_acceptance_residual_above_degree": response["residual_above_degree"],
             "azimuthal_degree": max_m,
             "ballistic_angular_model": "exact per-element arrival direction",
             "ballistic_spectrum": "charge and exact bins stored; nonzero-frequency column omitted"})

    def _directional_apply(self, prepared_class, cache, compiled, receivers,
                           looks, alpha, omega):
        """One (source, wavelength) apply, fused when numba is available."""
        indices = np.arange(len(omega))
        if prepared_class is None:
            return directional_response(cache, compiled, receivers, looks, alpha,
                                        source_omega_per_ns=omega,
                                        frequency_indices=indices)
        prepared = prepared_class.from_cache(cache, frequency_indices=indices)
        return prepared.apply(compiled, receivers, looks, alpha,
                              source_omega_per_ns=omega,
                              receiver_block=self.config.receiver_block)


__all__ = ["KernelConfig", "TransportResponse", "TransportKernel"]
