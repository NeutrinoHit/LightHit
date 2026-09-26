"""Prompt signal: scattering orders 0 and exactly 1, without the RTE cache.

``TransportKernel.transport_prompt(source)`` computes, for every module,

* order 0 -- unscattered light, with the **same** closed-form ballistic
  kernel, arguments and conventions as :meth:`TransportKernel.transport`;
* order 1 -- light scattered **exactly once**, with the full Henyey--Greenstein
  phase function, extinction ``mu_a + mu_s`` on both free flights, one factor
  of ``mu_s``, the wavelength-dependent group velocity, and the module
  acceptance evaluated at the true arrival direction after the scattering.

Orders ``>= 2`` are **not computed**; the returned
:class:`PromptTransportResponse` says so explicitly (``computed_orders`` is
``(0, 1)``) and refuses to select them, rather than reporting a zero.

Nothing here builds, loads or applies the multipole, directional or
spectral-folded RTE caches, and nothing calls :meth:`TransportKernel.transport`.

Order-1 formulation
-------------------
For one directed photon ("pencil") leaving ``x`` in direction ``s0`` at time
``t_e`` and a module at distance ``r`` with ``theta = angle(s0, R - x)``, put
``y = tan(chi/2)`` where ``chi`` is the scattering angle. Then exactly

    S(y)   = r cos(theta) + r sin(theta) y            (total path)
    s1     = cos(chi) s0 + sin(chi) e_perp            (arrival direction)
    N1     = mu_s / (r sin theta) int dchi exp(-mu_t S) p(cos chi) A(s1 . n)
    t      = t_e + S / v_g(lambda)

The ``1/rho^2`` of the second flight is absorbed by the change of variables;
the only singular factor, ``1/sin(theta)``, is integrable over the source
directions. The derivation and the numerical scheme are in
``docs/research/prompt-transport.md``. The slow independent reference is
:mod:`lighthit.prompt_reference`.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from importlib.util import find_spec
from math import comb
from pathlib import Path
from time import perf_counter
import json

import numpy as np
from numpy.polynomial import legendre as _legendre

from .ballistic import ballistic_directional
from .model import DetectorArray
from .sources import (CherenkovTrack, G4Shower, SpectralLightElements,
                      SyntheticShower)

C_M_PER_NS = 0.299792458

__all__ = ["PromptConfig", "PromptTransportResponse", "PromptComponent",
           "transport_prompt", "OrderNotComputedError"]


class OrderNotComputedError(ValueError):
    """Raised when a prompt response is asked for an order it never computed."""


@dataclass(frozen=True)
class PromptConfig:
    """Numerical controls of the direct order-1 calculation.

    Scattering-angle quadrature (all sources):

    ``path_step``, ``path_step_tail``
        Node spacing in ``u = asinh((S - r) / sigma)`` of the shared path grid
        of a batch, inside the time window and beyond it (beyond, only the
        integrated charge uses the nodes).  ``sigma`` is half the HG forward
        scale ``r sin(theta_min) max(y_g, tan(theta_min/2)/2)`` of the batch,
        ``y_g = (1-|g|)/(1+|g|)``.
    ``tail_exponent``
        Paths with ``mu_t,min (S - r) > tail_exponent`` are dropped; the
        omitted fraction of one pencil is below ``exp(-tail_exponent)``.
    ``direct_bins``
        Quadratic panels covering at most this many bins are integrated bin
        by bin in local variables; longer panels use the prefix-sum events.

    Straight segments (``CherenkovTrack``; element sources in ``segment``
    mode): tensor rule.  Along the segment, ``a_gauss``-point Gauss--Legendre
    panels graded towards the Cherenkov root by ``grade_ratio`` down to
    ``grade_min`` of the root scale, each spanning at most
    ``front_width_ns`` of front time and ``panel_max_relative`` of the
    module distance.  In azimuth, ``phi_gauss``-point panels of width
    ``phi_w_panel`` in ``w`` with ``phi = (theta_min/k) sinh(w)`` about the
    ring's closest approach to the module (``phi_uniform`` equal nodes when
    the ring stays more than 0.5 rad away).

    Element sources: ``element_method`` is ``"auto"`` (``segment`` for at
    most ``segment_max_elements`` elements, ``grouped`` above), ``"segment"``,
    ``"grouped"`` or ``"ring"``. ``grouped`` compresses the rings into
    pencils of ``cell_m`` cells and ``6 * pixel_face**2`` direction pixels
    sampled with ``ring_nodes`` azimuths, then merges pencils per module into
    groups of relative width ``merge_theta_ratio`` in angle above
    ``merge_theta_min``, ``merge_tau_ns`` in arrival time and
    ``merge_log_r`` in log distance. ``smoothing_factor`` sets where the
    Gaussian covariance mean of ``1/theta`` replaces the point value.
    ``ring`` evaluates every element as point cones of length at most
    ``ring_a_step_m`` (no grouping) with the same azimuth rule as segments.

    Grouped compression combines separate position and direction covariances
    without their cross covariance. Its geometry uses ``coefficient0`` weights
    for both spectral fields; the second retains its own total weight. These
    approximations are expected to be benign for validated ultra-relativistic
    electron and muon sources. For materially varying beta or difficult
    near-cone geometry, use ``element_method="ring"`` or ``"segment"`` as a
    convergence control.

    ``screen`` (grouped algorithm, nonzero threshold): a coarse pass
    (``screen_cell_m``, ``6 * screen_pixel_face**2`` pixels,
    ``screen_ring_nodes``) estimates every module; the full-resolution pass
    runs where ``q0 + screen_margin * q1_coarse >= threshold``.  The ratio of
    coarse to fine order 1 on the fine modules is reported; if it ever falls
    below ``1/screen_margin`` the fine set is widened with the observed ratio.
    Coarse-only modules keep the coarse order-1 charge and no bins.

    ``max_distance_m`` defaults to the kernel's ``radial_range_m[1]``;
    modules farther from every part of the source are omitted with the same
    conservative bound as the full path (an error is raised when that bound
    exceeds the threshold). ``min_distance_m`` rejects modules closer than
    this to a straight segment (the point-module model has no meaning there).
    """
    path_step: float = 0.1
    path_step_tail: float = 0.25
    tail_exponent: float = 25.0
    direct_bins: int = 4
    max_path_nodes: int = 2048
    a_gauss: int = 3
    front_width_ns: float = 2.5
    grade_ratio: float = 0.3
    grade_min: float = 1e-4
    panel_max_relative: float = 0.5
    phi_w_panel: float = 1.0
    phi_gauss: int = 4
    phi_uniform: int = 16
    max_a_breaks: int = 20_000
    min_distance_m: float = 0.1
    element_method: str = "auto"
    segment_max_elements: int = 256
    cell_m: float = 0.5
    pixel_face: int = 32
    ring_nodes: int = 96
    smoothing_factor: float = 6.0
    merge_theta_min: float = 2e-3
    merge_theta_ratio: float = 0.08
    merge_tau_ns: float = 0.5
    merge_log_r: float = 0.02
    ring_a_step_m: float = 0.05
    screen: bool = True
    screen_margin: float = 3.0
    screen_cell_m: float = 2.0
    screen_pixel_face: int = 8
    screen_ring_nodes: int = 24
    max_distance_m: float | None = None

    def __post_init__(self):
        positive = ("path_step", "path_step_tail", "tail_exponent", "grade_ratio",
                    "grade_min", "front_width_ns", "panel_max_relative",
                    "phi_w_panel", "cell_m", "screen_cell_m", "smoothing_factor",
                    "merge_theta_min", "merge_theta_ratio", "merge_tau_ns",
                    "merge_log_r", "ring_a_step_m")
        for name in positive:
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0 < self.grade_ratio < 1:
            raise ValueError("grade_ratio must be in (0, 1)")
        for name in ("direct_bins", "a_gauss", "phi_gauss",
                     "phi_uniform", "segment_max_elements",
                     "pixel_face", "ring_nodes"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name, minimum in (("max_a_breaks", 3), ("max_path_nodes", 3),
                              ("screen_pixel_face", 1), ("screen_ring_nodes", 1)):
            value = getattr(self, name)
            if (isinstance(value, (bool, np.bool_))
                    or not isinstance(value, (int, np.integer)) or value < minimum):
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.element_method not in ("auto", "segment", "grouped", "ring"):
            raise ValueError("element_method must be auto, segment, grouped or ring")
        if not np.isfinite(self.screen_margin) or self.screen_margin < 1:
            raise ValueError("screen_margin must be finite and >= 1")
        if not np.isfinite(self.min_distance_m) or self.min_distance_m < 0:
            raise ValueError("min_distance_m must be finite and nonnegative")
        if self.max_distance_m is not None and (
                not np.isfinite(self.max_distance_m) or self.max_distance_m <= 0):
            raise ValueError("max_distance_m must be finite and positive")


@dataclass
class PromptComponent:
    """One selected prompt signal on the ordinary OM and time axes."""
    charge_pe: np.ndarray
    bins_pe: np.ndarray
    time_origin_ns: np.ndarray
    relative_time_edges_ns: np.ndarray
    active: np.ndarray
    orders: tuple

    @property
    def rate_pe_per_ns(self):
        return self.bins_pe / np.diff(self.relative_time_edges_ns)[None, :]


@dataclass
class PromptTransportResponse:
    """Orders 0 and exactly 1; order >= 2 is *not computed*.

    ``charge_orders_pe[:, k]`` is the total integrated charge of order ``k``
    (all arrival times); ``bins_orders_pe[:, :, k]`` the charge inside each
    bin of ``relative_time_edges_ns`` measured from ``time_origin_ns``. Their
    difference is the charge outside the window, reported per order by
    :meth:`outside_window_pe`. Bins are filled for ``active`` modules only,
    exactly as in :class:`~lighthit.transport.TransportResponse`.
    """
    detector: DetectorArray
    relative_time_edges_ns: np.ndarray
    time_origin_ns: np.ndarray
    charge_orders_pe: np.ndarray
    bins_orders_pe: np.ndarray
    active: np.ndarray
    computed: np.ndarray
    method: str
    metadata: dict
    computed_orders: tuple = (0, 1)

    ORDER_GE2 = "not computed"

    @property
    def charge_pe(self):
        """Total order 0 + 1 charge (all times)."""
        return self.charge_orders_pe.sum(axis=1)

    @property
    def bins_pe(self):
        return self.bins_orders_pe.sum(axis=2)

    @property
    def window_charge_orders_pe(self):
        return self.bins_orders_pe.sum(axis=1)

    def outside_window_pe(self):
        """Charge of each order not contained in the time window (active OMs)."""
        out = self.charge_orders_pe - self.window_charge_orders_pe
        out[~self.active] = 0.0
        return out

    def select(self, component="0+1"):
        """Select ``0``, ``1`` or ``0+1``; ``>=2`` and ``all`` are refused."""
        choices = {0: (0,), 1: (1,), "0": (0,), "1": (1,), "0+1": (0, 1),
                   "prompt": (0, 1)}
        if component in (2, "2", ">=2", "all"):
            raise OrderNotComputedError(
                "order >= 2 was not computed by transport_prompt; use "
                "kernel.transport(source) for the full response")
        if component not in choices:
            raise ValueError("component must be 0, 1 or 0+1")
        selected = choices[component]
        return PromptComponent(
            self.charge_orders_pe[:, selected].sum(axis=1),
            self.bins_orders_pe[:, :, selected].sum(axis=2),
            self.time_origin_ns, self.relative_time_edges_ns, self.active,
            selected)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, relative_time_edges_ns=self.relative_time_edges_ns,
            time_origin_ns=self.time_origin_ns,
            charge_orders_pe=self.charge_orders_pe,
            bins_orders_pe=self.bins_orders_pe, active=self.active,
            computed=self.computed,
            computed_orders=np.asarray(self.computed_orders, int),
            metadata=json.dumps({"schema": "lighthit/prompt-response/1",
                                 "method": self.method,
                                 "computed_orders": list(self.computed_orders),
                                 "order_ge2": self.ORDER_GE2,
                                 **self.metadata}, default=_json_default))
        return path

    @classmethod
    def load(cls, path, detector):
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"]))
            if metadata.get("schema") != "lighthit/prompt-response/1":
                raise ValueError("not a LightHit prompt response")
            orders = tuple(int(v) for v in archive["computed_orders"])
            if orders != (0, 1) or metadata.get("order_ge2") != cls.ORDER_GE2:
                raise ValueError("prompt response must record computed_orders (0, 1)")
            if len(detector) != archive["charge_orders_pe"].shape[0]:
                raise ValueError("detector does not match the saved response")
            method = metadata.pop("method")
            for key in ("schema", "computed_orders", "order_ge2"):
                metadata.pop(key)
            return cls(detector, archive["relative_time_edges_ns"],
                       archive["time_origin_ns"], archive["charge_orders_pe"],
                       archive["bins_orders_pe"], archive["active"],
                       archive["computed"], method, metadata, orders)

    def viewer_components(self):
        """``(components, charge_components)`` padded to three columns for the
        viewer, together with ``computed_orders`` telling it that the third
        column is a placeholder, not a physical zero."""
        count, bins = self.bins_orders_pe.shape[:2]
        components = np.zeros((count, bins, 3))
        components[:, :, :2] = self.bins_orders_pe
        charge = np.zeros((count, 3))
        charge[:, :2] = self.charge_orders_pe
        return components, charge, list(self.computed_orders)


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


# ------------------------------------------------------------------ tables

def acceptance_monomials(alpha):
    """Monomial tables for ``A(x) = sum_l alpha_l (2l+1) P_l(x) / (4 pi)``.

    Returns ``(coef, ipow, jpow, degree)`` such that for
    ``x = cos(chi) c1 + sin(chi) c3``

        A = sum_m coef[m] cos(chi)^ipow[m] sin(chi)^jpow[m] c1^ipow[m] c3^jpow[m].
    """
    alpha = np.asarray(alpha, float)
    series = alpha * (2 * np.arange(len(alpha)) + 1) / (4 * np.pi)
    power = _legendre.leg2poly(series)
    degree = len(power) - 1
    coef, ipow, jpow = [], [], []
    for k in range(degree + 1):
        for i in range(k + 1):
            coef.append(power[k] * comb(k, i))
            ipow.append(i)
            jpow.append(k - i)
    return (np.asarray(coef, float), np.asarray(ipow, np.int64),
            np.asarray(jpow, np.int64), int(degree))


def acceptance_polynomial(coef, ipow, jpow, degree):
    """Monomial coefficients ``a_k`` of ``A(x) = sum_k a_k x^k``."""
    out = np.zeros(degree + 1)
    # the (i, j) = (k, 0) terms carry a_k * C(k, k) = a_k
    for c, i, j in zip(coef, ipow, jpow):
        if j == 0:
            out[i] = c
    return out


def _spectral_tables(kernel):
    """Per-wavelength medium and spectral weights, exactly the full path's.

    ``s0w/s2w`` multiply the two Cherenkov fields: quadrature weight times
    detector spectral efficiency times ``lambda^-2`` and
    ``lambda^-2 n_phase^-2``; the effective area is applied per module.
    """
    wavelengths = np.asarray(kernel.wavelength.wavelength_nm, float)
    weights = np.asarray(kernel.wavelength.weight_nm, float)
    sample = kernel.medium.sample(wavelengths)
    efficiency = np.array([float(kernel.detector.spectral_weight(float(w)))
                           for w in wavelengths])
    mus = np.asarray(sample["scattering_per_m"], float)
    mua = np.asarray(sample["absorption_per_m"], float)
    speed = C_M_PER_NS / np.asarray(sample["group_index"], float)
    phase = np.asarray(sample["phase_index"], float)
    s0w = weights * efficiency / wavelengths ** 2
    s2w = weights * efficiency / (wavelengths ** 2 * phase ** 2)
    return {"wavelength_nm": wavelengths, "weight_nm": weights,
            "efficiency": efficiency, "mus": mus, "mua": mua, "mut": mua + mus,
            "speed": speed, "phase_index": phase,
            "group_index": np.asarray(sample["group_index"], float),
            "s0w": s0w, "s2w": s2w}


def _thread_count():
    import numba
    return int(numba.get_num_threads())


# ----------------------------------------------------------------- sources

@dataclass(frozen=True)
class _Segments:
    """Straight Cherenkov segments with two spectral fields per metre."""
    start: np.ndarray
    direction: np.ndarray
    length: np.ndarray
    q0: np.ndarray
    q2: np.ndarray
    cone: np.ndarray
    t0: np.ndarray
    dtda: np.ndarray


def _track_segments(track):
    direction = np.asarray(track.direction, float)
    direction = direction / np.linalg.norm(direction)
    common = 2 * np.pi * 7.2973525693e-3 * 1e9  # photons / m / (nm^-1 lambda^-2)
    beta = float(track.beta)
    return _Segments(
        np.asarray(track.start_m, float)[None, :], direction[None, :],
        np.array([float(track.length_m)]), np.array([common]),
        np.array([-common / beta ** 2]),
        np.array([1.0 / (beta * track.reference_phase_index)]),
        np.array([float(track.time_ns)]),
        np.array([1.0 / (beta * C_M_PER_NS)]))


def _element_segments(elements):
    length = np.asarray(elements.length_m, float)
    return _Segments(
        np.ascontiguousarray(elements.start_m, float),
        np.ascontiguousarray(elements.direction, float), length,
        np.asarray(elements.coefficient0, float) / length,
        np.asarray(elements.coefficient2, float) / length,
        1.0 / (np.asarray(elements.beta, float) * elements.reference_phase_index),
        np.asarray(elements.start_ns, float),
        (np.asarray(elements.end_ns, float) - np.asarray(elements.start_ns, float))
        / length)


def _source_elements(kernel, source):
    """The element set the full path would transport, and why."""
    if isinstance(source, (G4Shower, SyntheticShower)):
        return source.elements
    if isinstance(source, SpectralLightElements):
        return source
    if isinstance(source, CherenkovTrack):
        # the full path's ballistic uses exactly this split
        return source.to_elements(step_m=max(kernel.config.cell_m, 0.25))
    raise TypeError("transport_prompt supports CherenkovTrack, G4Shower, "
                    "SyntheticShower and SpectralLightElements")


# ------------------------------------------------------------ main driver

def transport_prompt(kernel, source, config: PromptConfig | None = None):
    """Orders 0 and exactly 1 for ``source`` on ``kernel``'s detector.

    See the module docstring for the physics.  Uses the kernel's medium,
    detector, wavelength quadrature, time edges and threshold; never touches
    its RTE caches.
    """
    if find_spec("numba") is None:
        raise ImportError("transport_prompt requires the optional Numba accelerator; "
                          "install lighthit[accelerate]")
    config = config or PromptConfig()
    began = perf_counter()
    timings = {}
    jit_state = _jit_state()
    threads = _thread_count()

    stage = perf_counter()
    tables = _spectral_tables(kernel)
    response = kernel.acceptance()
    alpha = np.ascontiguousarray(response["alpha"], float)
    coef, ipow, jpow, degree = acceptance_monomials(alpha)
    detector = kernel.detector
    positions = np.ascontiguousarray(detector.positions_m, float)
    looks = np.ascontiguousarray(-detector.orientations, float)
    areas = np.asarray(detector.effective_area_m2, float)
    edges = np.asarray(kernel.config.relative_time_edges_ns, float)
    threshold = float(kernel.config.threshold_pe)
    g = float(kernel.medium.g)
    timings["tables_seconds"] = perf_counter() - stage

    stage = perf_counter()
    elements = _source_elements(kernel, source)
    spectral_cone = elements.cone_model == "spectral"
    original_elements = len(elements)
    original_coefficient = float(elements.coefficient0.sum())
    threshold_phase = (np.max(tables["phase_index"]) if spectral_cone
                       else np.min(tables["phase_index"]))
    valid = elements.beta * float(threshold_phase) > 1
    if not np.any(valid):
        raise ValueError("source is below Cherenkov threshold over the wavelength range")
    if not np.all(valid):
        elements = elements.subset(valid)
    dropped = 1 - float(elements.coefficient0.sum()) / original_coefficient
    fastest_group = float(np.min(tables["group_index"]))
    if isinstance(source, CherenkovTrack):
        if source.beta * float(threshold_phase) <= 1:
            raise ValueError("source is below Cherenkov threshold over the wavelength range")
        origins = source.earliest_arrival_ns(positions, fastest_group)
        segments = _track_segments(source)
        mode = "segment"
    else:
        first = int(np.argmin(elements.start_ns))
        origins = (float(elements.start_ns[first])
                   + np.linalg.norm(positions - elements.start_m[first], axis=1)
                   / (C_M_PER_NS / fastest_group))
        segments = _element_segments(elements)
        mode = config.element_method
        if mode == "auto":
            mode = "segment" if len(elements) <= config.segment_max_elements else "grouped"
    timings["source_seconds"] = perf_counter() - stage

    # ---------------------------------------------------- module selection
    stage = perf_counter()
    max_distance = (float(kernel.config.radial_range_m[1])
                    if config.max_distance_m is None else float(config.max_distance_m))
    distance = _min_distance_to_segments(positions, segments)
    inside = distance <= max_distance
    if spectral_cone:
        emitted = float(np.sum(tables["weight_nm"][None, :] * np.maximum(
            elements.photon_density_per_nm(tables["wavelength_nm"],
                                           tables["phase_index"]), 0.0)))
    else:
        emitted = float(np.sum(tables["weight_nm"] * (
            elements.coefficient0.sum() / tables["wavelength_nm"] ** 2
            + elements.coefficient2.sum()
            / (tables["wavelength_nm"] ** 2 * tables["phase_index"] ** 2))))
    peak_acceptance = float(np.max(np.abs(np.asarray(
        detector.angular_acceptance(np.linspace(-1, 1, 257)), float))))
    far = np.maximum(distance, 1e-9)
    upper = (10 * max(emitted, 0.0) * areas * peak_acceptance
             * float(np.max(tables["efficiency"]))
             * np.exp(-float(np.min(tables["mua"])) * far) / (4 * np.pi * far ** 2))
    unsafe = (~inside) & (upper >= threshold)
    if np.any(unsafe):
        raise ValueError(
            f"{int(unsafe.sum())} OMs beyond max_distance_m={max_distance:g} have a "
            "conservative estimate above threshold; increase max_distance_m")
    too_close = inside & (distance < config.min_distance_m)
    if np.any(too_close):
        raise ValueError(
            f"{int(too_close.sum())} OMs are closer than min_distance_m="
            f"{config.min_distance_m:g} m to the source")
    work = np.flatnonzero(inside)
    timings["selection_seconds"] = perf_counter() - stage

    # ------------------------------------------------------------ order 0
    stage = perf_counter()
    charge = np.zeros((len(positions), 2))
    ballistic = _ballistic_backend()
    fields = _ballistic_fields(source, elements, tables)
    for lam in range(len(tables["wavelength_nm"])):
        band = kernel.medium.band(float(tables["wavelength_nm"][lam]))
        detector_scale = (tables["weight_nm"][lam] * areas[work]
                          * tables["efficiency"][lam])
        field_and_cone = fields(lam)
        if field_and_cone is None:
            continue
        field_lam, cone = field_and_cone
        if len(work):
            value = ballistic(field_lam, positions[work], looks[work], band,
                              cone, alpha)[0]
            charge[work, 0] += detector_scale * value
    timings["order0_charge_seconds"] = perf_counter() - stage

    # ------------------------------------------------------------ order 1
    stage = perf_counter()
    if spectral_cone:
        order1_charge = np.zeros(len(positions))
        order1_bins = np.zeros((len(positions), len(edges) - 1))
        order1_meta = {"order1_spectral_cone": True,
                       "order1_by_wavelength": []}
        level = np.zeros(len(positions), dtype=np.int8)
        for lam, phase in enumerate(tables["phase_index"]):
            mask = elements.beta * phase > 1
            if not np.any(mask):
                continue
            subset = elements if np.all(mask) else elements.subset(mask)
            subset = replace(subset, reference_phase_index=float(phase),
                             cone_model="frozen")
            lam_segments = (_track_segments(replace(
                source, reference_phase_index=float(phase)))
                if isinstance(source, CherenkovTrack) else
                _element_segments(subset))
            single = {key: value[lam:lam + 1] for key, value in tables.items()}
            lam_charge, lam_bins, lam_meta = _order1(
                mode, kernel, source, subset, lam_segments,
                positions, looks, areas, origins, work, single, coef, ipow,
                jpow, degree, g, edges, replace(config, screen=False),
                charge[:, 0])
            order1_charge += lam_charge
            order1_bins += lam_bins
            lam_level = lam_meta.pop("order1_level", None)
            if lam_level is None:
                lam_level = np.zeros(len(positions), dtype=np.int8)
                lam_level[work] = 2
            level = np.maximum(level, lam_level)
            for key, value in lam_meta.pop("timings").items():
                timings[key] = timings.get(key, 0.0) + value
            order1_meta["order1_by_wavelength"].append({
                "wavelength_nm": float(tables["wavelength_nm"][lam]),
                "source_elements": len(subset), **lam_meta})
        order1_meta["order1_level"] = level
    else:
        order1_charge, order1_bins, order1_meta = _order1(
            mode, kernel, source, elements, segments, positions, looks, areas,
            origins, work, tables, coef, ipow, jpow, degree, g, edges, config,
            charge[:, 0])
        timings.update(order1_meta.pop("timings"))
    charge[:, 1] = order1_charge
    level = order1_meta.pop("order1_level", None)
    if level is None:
        level = np.zeros(len(positions), dtype=np.int8)
        level[work] = 2
    order1_meta["order1_level"] = np.asarray(level).tolist()
    order1_meta["order1_level_legend"] = {"0": "not computed (beyond max distance)",
                                          "1": "coarse screen only (below threshold)",
                                          "2": "full resolution"}
    timings["order1_seconds"] = perf_counter() - stage

    # ----------------------------------------------------- active and bins
    computed = inside.copy()
    active = computed & (charge.sum(axis=1) >= threshold)
    if threshold == 0:
        active = computed.copy()
    stage = perf_counter()
    bins = np.zeros((len(positions), len(edges) - 1, 2))
    active_index = np.flatnonzero(active)
    for lam in range(len(tables["wavelength_nm"])):
        band = kernel.medium.band(float(tables["wavelength_nm"][lam]))
        detector_scale = (tables["weight_nm"][lam] * areas[active_index]
                          * tables["efficiency"][lam])
        field_and_cone = fields(lam)
        if field_and_cone is None:
            continue
        field_lam, cone = field_and_cone
        if len(active_index):
            _, value = ballistic(field_lam, positions[active_index],
                                 looks[active_index], band, cone, alpha,
                                 time_origin_ns=origins[active_index],
                                 relative_edges_ns=edges)
            bins[active_index, :, 0] += detector_scale[:, None] * value
    timings["order0_bins_seconds"] = perf_counter() - stage
    bins[active, :, 1] = order1_bins[active]
    timings["elapsed_seconds"] = perf_counter() - began

    window = bins.sum(axis=1)
    metadata = {
        "computed_orders": [0, 1],
        "order_ge2": "not computed",
        "prompt_algorithm": mode,
        "source_type": type(source).__name__,
        "source_elements": len(elements),
        "source_elements_original": original_elements,
        "source_elements_dropped_at_spectral_threshold": original_elements - len(elements),
        "source_coefficient_fraction_dropped": dropped,
        "reference_phase_index": float(elements.reference_phase_index),
        "cherenkov_cone": ("wavelength-dependent medium phase index"
                            if spectral_cone else
                            "frozen at reference_phase_index (as the full path)"),
        "medium": {"provenance": kernel.medium.provenance, "g": g},
        "wavelength_nm": tables["wavelength_nm"].tolist(),
        "wavelength_weight_nm": tables["weight_nm"].tolist(),
        "scattering_per_m": tables["mus"].tolist(),
        "absorption_per_m": tables["mua"].tolist(),
        "group_index": tables["group_index"].tolist(),
        "phase_index": tables["phase_index"].tolist(),
        "detector": {"modules": len(detector), "provenance": detector.provenance,
                     "acceptance_degree": int(response["degree"]),
                     "acceptance_alpha": alpha.tolist(),
                     "acceptance_residual_above_degree":
                         float(response["residual_above_degree"])},
        "prompt_config": asdict(config),
        "threshold_pe": threshold,
        "max_distance_m": max_distance,
        "computed_modules": int(computed.sum()),
        "active_modules": int(active.sum()),
        "omitted_beyond_max_distance": int((~inside).sum()),
        "omitted_conservative_bound_max_pe": float(upper[~inside].max()) if np.any(~inside) else 0.0,
        "tail_bound_relative": float(np.exp(-config.tail_exponent)),
        "total_charge_pe": {"order0": float(charge[:, 0].sum()),
                            "order1": float(charge[:, 1].sum())},
        "window_charge_pe": {"order0": float(window[:, 0].sum()),
                             "order1": float(window[:, 1].sum())},
        "time_origin": ("earliest straight-track emission and group flight"
                        if isinstance(source, CherenkovTrack) else
                        "earliest element start and fastest group flight"),
        "ballistic": "identical to TransportKernel.transport: "
                     + ballistic.__module__ + "." + ballistic.__name__,
        "backend": "numba",
        "threads": threads,
        "jit_warm_before_call": jit_state,
        "timings": timings,
        **order1_meta,
    }
    return PromptTransportResponse(
        detector, edges.copy(), np.asarray(origins, float), charge, bins,
        active, computed, "prompt", metadata)


def _jit_state():
    """Whether the compiled prompt kernels were already compiled in-process."""
    try:
        from . import _prompt_numba as numba_kernels
    except ImportError:  # pragma: no cover
        return False
    names = ("segment_order1", "grouped_pencils_order1", "rings_order1",
             "compress_elements")
    return {name: bool(getattr(getattr(numba_kernels, name), "signatures", ())) for name in names}


def _ballistic_backend():
    if find_spec("numba") is not None:
        try:
            from .experimental.ballistic_fast import ballistic_directional_fast
            return ballistic_directional_fast
        except ImportError:  # pragma: no cover
            pass
    return ballistic_directional  # pragma: no cover


def _ballistic_fields(source, elements, tables):
    """Per-wavelength ballistic field, exactly as the full path builds it."""
    from dataclasses import replace
    if elements.cone_model == "spectral":
        def spectral(lam):
            phase = float(tables["phase_index"][lam])
            mask = elements.beta * phase > 1
            if not np.any(mask):
                return None
            subset = elements if np.all(mask) else elements.subset(mask)
            field0 = subset.field(0, phase_index=phase)
            field2 = subset.field(2, phase_index=phase)
            s0 = 1.0 / tables["wavelength_nm"][lam] ** 2
            s2 = s0 / phase ** 2
            return (replace(field0, photons=np.ascontiguousarray(
                s0 * field0.photons - s2 * field2.photons)),
                field0.cone_cosine)
        return spectral
    field0 = elements.field(0)
    field2 = elements.field(2)
    if isinstance(source, CherenkovTrack):
        beta = float(source.beta)
        factor = (1.0 / tables["wavelength_nm"] ** 2
                  * (1.0 - 1.0 / (beta ** 2 * tables["phase_index"] ** 2)))

        def track(lam):
            return (replace(field0, photons=np.ascontiguousarray(
                factor[lam] * field0.photons)), field0.cone_cosine)
        return track

    def shower(lam):
        s0 = 1.0 / tables["wavelength_nm"][lam] ** 2
        s2 = 1.0 / (tables["wavelength_nm"][lam] ** 2 * tables["phase_index"][lam] ** 2)
        return (replace(field0, photons=np.ascontiguousarray(
            s0 * field0.photons - s2 * field2.photons)), field0.cone_cosine)
    return shower


def _min_distance_to_segments(positions, segments):
    from . import _prompt_numba as kernels
    return kernels.min_distance_to_segments(
        np.ascontiguousarray(positions, float),
        np.ascontiguousarray(segments.start, float),
        np.ascontiguousarray(segments.direction, float),
        np.ascontiguousarray(segments.length, float))


# --------------------------------------------------------------- order 1

def _order1(mode, kernel, source, elements, segments, positions, looks, areas,
            origins, work, tables, coef, ipow, jpow, degree, g, edges, config,
            order0_charge):
    from . import _prompt_numba as kernels
    y_g = (1.0 - abs(g)) / (1.0 + abs(g)) if g != 0 else 1.0
    kappa = -tables["mut"] * tables["speed"]
    eedge = np.exp(np.outer(kappa, edges))
    itab = kernels.interval_table(edges, kappa)
    apoly = acceptance_polynomial(coef, ipow, jpow, degree)
    common = dict(g=g, y_g=y_g, h=config.path_step, h_tail=config.path_step_tail,
                  tail_exp=config.tail_exponent, mus=tables["mus"],
                  mut=tables["mut"], speed=tables["speed"], s0w=tables["s0w"],
                  s2w=tables["s2w"], edges=edges, eedge=eedge, itab=itab,
                  direct_bins=int(config.direct_bins))
    n = len(positions)
    receivers = np.ascontiguousarray(positions, float)
    looks = np.ascontiguousarray(looks, float)
    areas = np.ascontiguousarray(areas, float)
    origins = np.ascontiguousarray(origins, float)
    work = np.ascontiguousarray(work, np.int64)
    timings = {}
    meta = {}
    if mode == "segment":
        stage = perf_counter()
        charge, bins, pencils, batches, close = kernels.segment_order1(
            receivers, looks, areas, origins, work,
            segments.start, segments.direction, segments.length,
            segments.q0, segments.q2, segments.cone, segments.t0, segments.dtda,
            apoly, common["g"], common["y_g"], common["h"], common["h_tail"],
            common["tail_exp"], common["mus"], common["mut"], common["speed"],
            common["s0w"], common["s2w"], edges, eedge, itab,
            common["direct_bins"], int(config.a_gauss), float(config.front_width_ns),
            float(config.grade_ratio), float(config.grade_min),
            float(config.panel_max_relative), float(config.phi_w_panel),
            int(config.phi_gauss), int(config.phi_uniform),
            int(config.max_a_breaks), int(config.max_path_nodes),
            float(config.min_distance_m), _blocks(len(work)))
        timings["order1_eval_seconds"] = perf_counter() - stage
        meta.update({
            "order1_quadrature": "tensor rule per segment (a panels graded to the "
                                 "Cherenkov root and limited in front spread; sinh "
                                 "azimuth about the closest approach); one shared "
                                 "path grid per emission point; quadratic-in-time "
                                 "panels over the actual bin edges",
            "order1_segments": int(len(segments.length)),
            "order1_pencils_per_module_max": int(pencils.max()) if len(pencils) else 0,
            "order1_pencils_total": int(pencils.sum()),
            "order1_batches_total": int(batches.sum()),
            "order1_segments_too_close": int(close.sum()),
        })
        return charge, bins, {"timings": timings, **meta}
    if mode == "ring":
        stage = perf_counter()
        charge, bins, pencils = kernels.rings_order1(
            receivers, looks, areas, origins, work,
            np.ascontiguousarray(elements.start_m, float),
            np.ascontiguousarray(elements.direction, float),
            np.asarray(elements.length_m, float),
            np.asarray(elements.coefficient0, float),
            np.asarray(elements.coefficient2, float),
            np.asarray(segments.cone, float),
            np.asarray(elements.start_ns, float), np.asarray(elements.end_ns, float),
            float(config.ring_a_step_m), apoly,
            common["g"], common["y_g"], common["h"], common["h_tail"],
            common["tail_exp"], common["mus"], common["mut"], common["speed"],
            common["s0w"], common["s2w"], edges, eedge, itab,
            common["direct_bins"], int(config.max_path_nodes),
            float(config.phi_w_panel), int(config.phi_gauss), int(config.phi_uniform),
            _blocks(len(work)))
        timings["order1_eval_seconds"] = perf_counter() - stage
        meta.update({"order1_quadrature": "point cones (no grouping), sinh azimuth "
                                          "about the closest approach",
                     "order1_pencils_total": int(pencils.sum())})
        return charge, bins, {"timings": timings, **meta}
    # grouped pencils, optionally screened by a coarse pass
    def grouped(pencils, targets, merge):
        return kernels.grouped_pencils_order1(
            receivers, looks, areas, origins, np.ascontiguousarray(targets, np.int64),
            pencils["position"], pencils["direction"], pencils["w0"], pencils["w2"],
            pencils["time"], pencils["major"], pencils["var_major"],
            pencils["var_minor"], pencils["pos_cov"], kernels._GAUSS_TAU,
            kernels._GAUSS_W,
            coef, ipow, jpow, degree,
            common["g"], common["y_g"], common["h"], common["h_tail"],
            common["tail_exp"], common["mus"], common["mut"], common["speed"],
            common["s0w"], common["s2w"], edges, eedge, itab,
            common["direct_bins"], int(config.max_path_nodes),
            float(config.smoothing_factor), float(merge[0]), float(merge[1]),
            float(merge[2]), float(merge[3]), float(np.max(tables["speed"])),
            _blocks(len(targets)))

    fine_merge = (config.merge_theta_min, config.merge_theta_ratio,
                  config.merge_tau_ns, config.merge_log_r)
    threshold = float(kernel.config.threshold_pe)
    screen = config.screen and threshold > 0 and len(work) > 0
    charge = np.zeros(n)
    bins = np.zeros((n, len(edges) - 1))
    level = np.zeros(n, dtype=np.int8)          # 0 not computed, 1 coarse, 2 fine
    fine_work = work
    if screen:
        stage = perf_counter()
        coarse_config = replace(config, cell_m=config.screen_cell_m,
                                pixel_face=config.screen_pixel_face,
                                ring_nodes=config.screen_ring_nodes)
        coarse = compress(elements, segments, coarse_config)
        timings["order1_screen_compress_seconds"] = perf_counter() - stage
        stage = perf_counter()
        coarse_charge, _, _, _ = grouped(
            coarse, work, (config.merge_theta_min, 0.3, 2.0, 0.1))
        timings["order1_screen_eval_seconds"] = perf_counter() - stage
        charge[work] = coarse_charge[work]
        level[work] = 1
        margin = float(config.screen_margin)
        fine_work = work[order0_charge[work] + margin * coarse_charge[work] >= threshold]
    stage = perf_counter()
    pencils = compress(elements, segments, config)
    timings["order1_compress_seconds"] = perf_counter() - stage
    stage = perf_counter()
    fine_charge, fine_bins, groups, batches = grouped(pencils, fine_work, fine_merge)
    timings["order1_eval_seconds"] = perf_counter() - stage
    charge[fine_work] = fine_charge[fine_work]
    bins[fine_work] = fine_bins[fine_work]
    level[fine_work] = 2
    screen_meta = {"enabled": bool(screen)}
    if screen:
        ratio = coarse_charge[fine_work] / np.maximum(fine_charge[fine_work], 1e-300)
        worst = float(ratio.min()) if len(ratio) else 1.0
        extra = np.array([], dtype=np.int64)
        if worst * margin < 1.0:
            # the coarse screen underestimated some module by more than the
            # margin: widen the fine set with the observed ratio and redo
            wider = 1.5 / max(worst, 1e-3)
            candidates = work[(level[work] == 1)
                              & (order0_charge[work] + wider * coarse_charge[work]
                                 >= threshold)]
            if len(candidates):
                extra_charge, extra_bins, _, _ = grouped(pencils, candidates, fine_merge)
                charge[candidates] = extra_charge[candidates]
                bins[candidates] = extra_bins[candidates]
                level[candidates] = 2
                extra = candidates
        coarse_only = work[level[work] == 1]
        screen_meta.update({
            "margin": margin,
            "coarse_settings": {"cell_m": config.screen_cell_m,
                                "pixel_face": config.screen_pixel_face,
                                "ring_nodes": config.screen_ring_nodes,
                                "merge": [config.merge_theta_min, 0.3, 2.0, 0.1]},
            "fine_modules": int((level == 2).sum()),
            "coarse_only_modules": int(len(coarse_only)),
            "widened_modules": int(len(extra)),
            "coarse_over_fine_ratio_min": worst,
            "coarse_over_fine_ratio_max": float(ratio.max()) if len(ratio) else 1.0,
            "coarse_only_max_estimate_pe": float(
                (order0_charge[coarse_only] + charge[coarse_only]).max())
                if len(coarse_only) else 0.0,
            "note": "coarse-only modules carry the coarse order-1 charge and no "
                    "bins; they are below threshold by the screened estimate",
        })
    used = groups[fine_work]
    meta.update({
        "order1_quadrature": "rings -> (cell, pixel) pencils with moments and "
                             "shape -> per-module (tau, r, theta) groups -> one "
                             "shared path grid per (tau, r) batch -> quadratic-"
                             "in-time panels",
        "order1_pencils": int(len(pencils["w0"])),
        "order1_pencil_memory_mib": float(sum(v.nbytes for v in pencils.values()) / 2 ** 20),
        "order1_groups_per_module_mean": float(used.mean()) if len(used) else 0.0,
        "order1_groups_per_module_max": int(used.max()) if len(used) else 0,
        "order1_batches_total": int(batches.sum()),
        "order1_ring_nodes_deposited": int(pencils["ring_nodes_deposited"]),
        "order1_screen": screen_meta,
        "order1_level": level,
    })
    return charge, bins, {"timings": timings, **meta}


def _blocks(count):
    try:
        import numba
        threads = numba.get_num_threads()
    except ImportError:  # pragma: no cover
        threads = 1
    return int(max(1, min(count, 4 * threads)))


def compress(elements, segments, config):
    """Stage A: rings of all elements -> (cell, pixel) pencils with moments."""
    from . import _prompt_numba as kernels
    start = np.ascontiguousarray(elements.start_m, float)
    ends = start + np.asarray(elements.length_m, float)[:, None] * np.asarray(
        elements.direction, float)
    low = np.minimum(start.min(axis=0), ends.min(axis=0)) - config.cell_m
    high = np.maximum(start.max(axis=0), ends.max(axis=0)) + config.cell_m
    dims = np.ceil((high - low) / config.cell_m).astype(np.int64) + 1
    pixels = 6 * int(config.pixel_face) ** 2
    if float(np.prod(dims.astype(float))) * pixels >= 2.0 ** 62:
        raise ValueError("cell grid too large for 64-bit pencil keys; increase cell_m")
    origin = low
    arrays = (start, np.ascontiguousarray(elements.direction, float),
              np.asarray(elements.length_m, float),
              np.asarray(elements.coefficient0, float),
              np.asarray(elements.coefficient2, float),
              np.asarray(segments.cone, float), np.asarray(elements.start_ns, float),
              np.asarray(elements.end_ns, float))
    count = len(start)
    chunks = max(1, min(_thread_count() * 2, count // 2000 + 1))
    bounds = np.linspace(0, count, chunks + 1).astype(int)

    def run(index):
        return kernels.compress_elements(
            *arrays, float(config.cell_m), int(config.pixel_face),
            int(config.ring_nodes), 0.5 * float(config.cell_m), origin, dims,
            int(bounds[index]), int(bounds[index + 1]))

    if chunks == 1:
        parts = [run(0)]
    else:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=_thread_count()) as pool:
            parts = list(pool.map(run, range(chunks)))
    keys = np.concatenate([part[0] for part in parts])
    rows = np.concatenate([part[1] for part in parts])
    order = np.argsort(keys, kind="stable")
    (position, direction, w0, w2, time, major, var_major, var_minor,
     pos_cov) = kernels.finish_pencils(keys, rows, order)
    n_a = np.maximum(1, np.ceil(np.asarray(elements.length_m) / (0.5 * config.cell_m)))
    return {"position": position, "direction": direction, "w0": w0, "w2": w2,
            "time": time, "major": major, "var_major": var_major,
            "var_minor": var_minor, "pos_cov": pos_cov,
            "ring_nodes_deposited": np.array(int(n_a.sum() * config.ring_nodes))}
