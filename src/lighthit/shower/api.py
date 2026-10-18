"""Shower events to expected photoelectrons, all scattering orders.

This is the route used by default for electromagnetic and hadronic showers
(any compact G4 source without a dominant muon):

* direct light (order 0): exact on the original steps over the continuous
  spectrum (:func:`lighthit.ballistic_interval.ballistic_interval`);
* first scattering (order 1): the grouped physical-time prompt solver with
  the spectral bands fused into one pencil set (``PromptConfig.fuse_bands``);
* multiple scattering (order >=2): the axial shower basis compiled in three
  node bands and applied with band-folded RTE kernels
  (:mod:`lighthit.band_fold`); the zero-frequency charge used to select
  active OMs is predicted from a few collapsed points and computed exactly
  only for OMs whose threshold decision depends on it.

Accuracy and speed of each choice, against original-step and finer-basis
controls: docs/accuracy.md. The previous
per-band route is ``TransportKernel.transport_hybrid(source, ge2_basis=ShowerBasis())``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from time import perf_counter, process_time

import numpy as np

from ..prompt import PromptConfig
from ..shower_basis import ShowerBasis
from .data import ShowerData

#: RTE frequency grid of the prepared tables and the readout grid.
FREQUENCIES_PER_NS = np.linspace(0., 1.2, 65)
READOUT_PER_NS = np.linspace(0., 1.2, 321)


def _default_prompt():
    """Continuous-spectrum direct light; fused bands at 96 pixels per face.

    96/192 was more accurate than the muon route's 64/192 against original
    steps on all three controls (worst bin L1 3.3/2.2/3.1% and bias <= 0.6%)
    at about 1.4 times its order-1 cost.
    """
    return PromptConfig(ballistic_method="interval", fuse_bands=True,
                        fused_pixel_face=96, fused_ring_nodes=192)


@dataclass(frozen=True)
class ShowerConfig:
    """Numerical choices of the shower route.

    ``basis``: the order >=2 axial basis (slices, stencil, time lattice).
    ``prompt``: orders 0 and 1 (continuous-spectrum direct light, fused bands).
    ``band_rank`` node bands carry the order >=2 source (see
    :class:`lighthit.band_fold.BandFold`); ``fold_subdivide`` splits each
    radial interval of the folded kernel. The basis points collapsed to one per
    ``predictor_cell_m`` cube predict the zero-frequency >=2 charge of every OM; an OM gets the exact
    basis charge when ``q0 + q1 + predictor_safety * |prediction|`` reaches the
    threshold. If any exact charge exceeds ``predictor_safety`` times its
    prediction the exact charge is computed for every OM instead.

    Steps emitting more than ``prompt_delay_ns`` behind the light front
    (hadronic showers: neutron captures and decays, up to milliseconds) are
    not put on the prompt clock: the 65 RTE frequencies resolve only a few
    hundred ns, so such light would alias into the readout window. They are
    grouped in ``delayed_cell_ns`` cells, each demodulated by its own delay and
    read out on correspondingly shifted bin edges (an identity), and steps
    whose light cannot reach the window at all add only to the charge (see
    :func:`lighthit.shower.ge2.delay_groups`).
    """
    basis: ShowerBasis = field(default_factory=ShowerBasis)
    prompt: PromptConfig = field(default_factory=_default_prompt)
    frequency_nodes: int = 65
    band_rank: int = 3
    fold_subdivide: int = 1
    predictor_cell_m: float = 2.0
    predictor_safety: float = 2.0
    prompt_delay_ns: float = 20.0
    delayed_cell_ns: float = 20.0

    def __post_init__(self):
        """Check the route-specific settings."""
        if self.basis.track_uid is not None or self.basis.apply_tolerance_pe:
            raise ValueError("the shower route uses neither a track UID nor pair tolerance")
        if not 1 <= int(self.band_rank) <= 9 or int(self.fold_subdivide) < 1:
            raise ValueError("band_rank must lie in 1..9 and fold_subdivide be positive")
        if (not self.predictor_safety >= 1 or not self.predictor_cell_m > 0
                or not self.prompt_delay_ns >= 0 or not self.delayed_cell_ns > 0):
            raise ValueError("predictor_safety >= 1 and positive predictor/delay cells required")


def shower_kernel(kernel, data):
    """Kernel on the prepared cache with the route's frequency grids.

    The given kernel is reused when it already matches; otherwise a sibling
    keeps its medium, detector, bins and threshold (cache policy 'require').
    """
    from ..transport import TransportKernel
    config = kernel.config
    same = (config.cache_directory is not None
            and Path(config.cache_directory).expanduser().resolve() == Path(data.rte_cache).resolve()
            and np.array_equal(np.asarray(config.omega_per_ns), FREQUENCIES_PER_NS)
            and config.readout_omega_per_ns is not None
            and np.array_equal(np.asarray(config.readout_omega_per_ns), READOUT_PER_NS))
    if same:
        return kernel
    return TransportKernel(kernel.medium, kernel.detector,
                           replace(config, omega_per_ns=FREQUENCIES_PER_NS,
                                   readout_omega_per_ns=READOUT_PER_NS,
                                   cache_directory=str(data.rte_cache), cache_policy="require"))


class ShowerTransport:
    """Transport for shower events on one detector.

    ``data`` is a :class:`ShowerData`, a manifest path, or None to search
    next to the kernel's cache. The >=2 pack is memory-mapped and folded onto
    the node bands once (or a prepared fold is loaded); numba compiles its
    kernels once per installation. ``threads`` limits numba threads per event.
    """

    def __init__(self, kernel, data=None, *, config=None, threads=None):
        """Find the tables and build the route kernel; tables open on first use."""
        if data is None:
            data = ShowerData.find(kernel.config.cache_directory)
        elif not isinstance(data, ShowerData):
            data = ShowerData.load(data)
        self.data = data
        self.kernel = shower_kernel(kernel, data)
        self.config = config or ShowerConfig()
        self.threads = threads
        self._tables = None
        self._compiled = {}

    def tables(self):
        """Open the >=2 pack and the band fold once."""
        if self._tables is None:
            from ..band_fold import BandFold, fold_bundle, load_folded
            from ..shower_basis import _frequency_indices
            store = self.kernel.ge2_kernel_store(str(self.data.ge2_pack))
            fold = BandFold.of(self.kernel, rank=int(self.config.band_rank))
            folded = None
            path = self.data.ge2_band_fold
            if path is not None and Path(path).exists():
                folded = load_folded(path)
                if folded["band_fold"]["bands"] != list(fold.bands):
                    raise ValueError(f"{path} was folded on other node bands")
            began = perf_counter()
            if folded is None:
                indices = _frequency_indices(self.kernel, self.config.frequency_nodes)
                folded = fold_bundle(store.bundle(self.kernel, indices), fold,
                                     subdivide=int(self.config.fold_subdivide))
            self._tables = dict(store=store, fold=fold, folded=folded,
                                fold_seconds=perf_counter() - began)
        return self._tables

    def __call__(self, source, *, progress=None, time_origin_ns=None,
                 selection_extra_charge=None, receivers=None):
        """Response for one event (``HybridTransportResponse``).

        For events with several sources (:mod:`lighthit.event`):
        ``time_origin_ns`` is a common per-OM clock, ``selection_extra_charge``
        the light of the other sources added to every threshold decision, and
        ``receivers`` restricts the calculation to those OMs (all computed in
        full, threshold zero); the response then covers only ``receivers``.
        """
        options = dict(time_origin_ns=time_origin_ns,
                       selection_extra_charge=selection_extra_charge, receivers=receivers)
        if self.threads is not None:
            import numba
            previous = numba.get_num_threads()
            numba.set_num_threads(max(1, min(int(self.threads), numba.config.NUMBA_NUM_THREADS)))
            try:
                return self._event(source, **options)
            finally:
                numba.set_num_threads(previous)
        return self._event(source, **options)

    def _event(self, source, *, time_origin_ns=None, selection_extra_charge=None, receivers=None):
        """One event with the current thread setting."""
        tables = self.tables()
        kernel = self.kernel
        if receivers is not None:
            from ..muon.universal_primary import subset_kernel
            kernel = subset_kernel(kernel, receivers)
            kernel.config = replace(kernel.config, threshold_pe=0.)
        response = transport_shower(kernel, source, config=self.config,
                                    store=tables["store"], fold=tables["fold"],
                                    folded=tables["folded"], time_origin_ns=time_origin_ns,
                                    selection_extra_charge=selection_extra_charge,
                                    compiled=self._compiled)
        response.metadata["shower_data"] = str(self.data.manifest)
        response.metadata["route"] = "lighthit.shower"
        response.metadata["fold_prepare_seconds"] = tables["fold_seconds"]
        return response


def transport_shower(kernel, source, *, config=None, store=None, fold=None, folded=None,
                     time_origin_ns=None, selection_extra_charge=None, compiled=None):
    """All orders for one shower source; see the module docstring.

    ``store`` defaults to ``kernel.ge2_kernel_store()``; ``fold`` and
    ``folded`` are built from the kernel and the store when omitted.
    ``time_origin_ns`` replaces this source's own per-OM origin by a common
    event clock. ``selection_extra_charge`` (light of other sources of the
    event) is added to every threshold decision; ``active`` then refers to
    the event sum and every active OM has bins of this source.
    ``compiled`` is an optional dict that keeps the compiled node-band source
    of the last ``source`` object (it does not depend on the OMs), so repeated
    calls for other OM subsets of the same source skip the compilation.
    """
    from ..axial_source import AxisFrame
    from ..band_fold import BandFold, fold_bundle, zero_frequency
    from .._shower_basis import earliest_element_arrival
    from ..readout import inverse_bins
    from ..shower_basis import _frequency_indices, _window_report_from_photons
    from ..transport import HybridTransportResponse
    from .ge2 import apply_folded, collapse, delay_groups, node_sources

    config = config or ShowerConfig()
    basis = config.basis
    began, cpu = perf_counter(), process_time()
    timings = {}
    stage = perf_counter()
    store = kernel.ge2_kernel_store() if store is None else store
    indices = _frequency_indices(kernel, config.frequency_nodes)
    omega = kernel.config.omega_per_ns[indices]
    if omega[0] != 0.0:
        raise ValueError("the shower route needs omega = 0 in its frequency set")
    output_omega = (kernel.config.readout_omega_per_ns
                    if kernel.config.readout_omega_per_ns is not None else omega)
    elements = kernel._source_elements(source)
    if elements.cone_model != "spectral":
        raise ValueError("the shower route requires spectral cones (cone_model='spectral')")
    wavelengths = np.asarray(kernel.wavelength.wavelength_nm, float)
    weights = np.asarray(kernel.wavelength.weight_nm, float)
    sample = kernel.medium.sample(wavelengths)
    max_phase = float(np.max(sample["phase_index"]))
    valid = elements.beta * max_phase > 1
    if not np.any(valid):
        raise ValueError("source is below Cherenkov threshold over the wavelength range")
    original_coefficient = float(elements.coefficient0.sum())
    if not np.all(valid):
        elements = elements.subset(valid)
    efficiency = np.asarray(kernel.detector.spectral_weight(wavelengths), float)
    positions = np.asarray(kernel.detector.positions_m, float)
    looks = np.ascontiguousarray(-kernel.detector.orientations, float)
    area = np.asarray(kernel.detector.effective_area_m2, float)
    alpha = np.ascontiguousarray(kernel.acceptance()["alpha"], float)
    low, cache_high = kernel.config.radial_range_m
    high = cache_high if basis.radial_high_m is None else basis.radial_high_m
    threshold = float(kernel.config.threshold_pe)
    extra = (np.zeros(len(positions)) if selection_extra_charge is None
             else np.maximum(np.asarray(selection_extra_charge, float), 0.0))
    if extra.shape != (len(positions),) or not np.isfinite(extra).all():
        raise ValueError("selection_extra_charge must hold one finite value per OM")
    if fold is None:
        fold = BandFold.of(kernel, rank=int(config.band_rank))
    if folded is None:
        folded = fold_bundle(store.bundle(kernel, indices), fold, subdivide=int(config.fold_subdivide))
    if not np.allclose(folded["omega"], omega, rtol=0, atol=1e-12):
        raise ValueError("folded >=2 kernels and route frequencies differ")
    zero = zero_frequency(folded)
    timings["setup_seconds"] = perf_counter() - stage

    # ---- order >=2 source on the node bands (all frequencies, all points)
    stage = perf_counter()
    edges = np.asarray(kernel.config.relative_time_edges_ns, float)
    prompt_mask, late_groups, beyond = delay_groups(
        elements, prompt_delay_ns=float(config.prompt_delay_ns), cell_ns=float(config.delayed_cell_ns),
        window_end_ns=float(edges[-1]), group_index_max=float(np.max(sample["group_index"])))
    key = (fold.bands, basis, tuple(np.round(omega, 12)), int(kernel.config.source_degree),
           int(kernel.config.azimuthal_degree), float(config.prompt_delay_ns),
           float(config.delayed_cell_ns), float(edges[-1]))
    if compiled is not None and compiled.get("source") is source and compiled.get("key") == key:
        sources, late_zero, late_sources = compiled["sources"]
    else:
        def compiled_part(mask, frequencies):
            part = elements if np.all(mask) else elements.subset(mask)
            reference = part.field(0, phase_index=max_phase)
            return node_sources(kernel, part, basis, fold, frequencies, AxisFrame.of(reference),
                                reference)
        sources = compiled_part(prompt_mask, omega)
        late_zero = compiled_part(~prompt_mask, omega[:1]) if np.any(~prompt_mask) else None
        late_sources = [(shift, compiled_part(mask, omega)) for mask, shift in late_groups]
        if compiled is not None:
            compiled.clear()
            compiled.update(source=source, key=key, sources=(sources, late_zero, late_sources))
    timings["compile_seconds"] = perf_counter() - stage

    # ---- radial windows: every node-band point set (prompt and delayed);
    # the far-tail estimate uses the least absorbing band and scales the
    # node bands' detector weight to the whole quadrature (screening only).
    stage = perf_counter()
    used = np.zeros(len(positions), np.int64)
    far = np.zeros(len(positions), np.int64)
    tail = np.zeros(len(positions))
    near = np.zeros(len(positions), bool)
    absorption_min = float(np.min(sample["absorption_per_m"]))
    weight_ratio = float(np.sum(fold.scale) / np.sum(fold.scale[list(fold.bands)]))
    for group in [sources] + ([late_zero] if late_zero is not None else []):
        for band, item in zip(fold.bands, group):
            photons = np.sqrt(4 * np.pi) * np.abs(item.channels[:, 0, 0]) / efficiency[band]
            part_used, part_far, part_near, part_tail = _window_report_from_photons(
                kernel, item.points_m(), photons, positions, low, high, absorption_min,
                float(efficiency[band]))
            # Points closer than the radial minimum are left out of the
            # tables' sums; lighthit.near_field adds them back below.
            near |= np.asarray(part_near) > 0
            used = np.maximum(used, part_used)
            far = np.maximum(far, part_far)
            tail += weight_ratio * part_tail
    if np.any(tail > basis.radial_tail_tolerance_pe):
        raise ValueError(f"{int(np.count_nonzero(tail > basis.radial_tail_tolerance_pe))} OMs "
                         f"have estimated light beyond {high:g} m above "
                         f"{basis.radial_tail_tolerance_pe:g} pe; largest {float(tail.max()):g} pe")
    reach = np.flatnonzero(used > 0)

    # ---- predicted zero-frequency >=2 charge of every reachable OM
    predicted = np.zeros(len(positions))
    collapsed = collapse(sources, float(config.predictor_cell_m))
    late_charge = np.zeros(len(positions))
    if len(reach):
        predicted[reach] = apply_folded(zero, collapsed, positions[reach], looks[reach],
                                        alpha, area[reach], (low, high))[0].real
        if late_zero is not None:
            # Delayed light is small and its sources compact: exact everywhere.
            late_charge[reach] = apply_folded(zero, late_zero, positions[reach], looks[reach],
                                              alpha, area[reach], (low, high))[0].real
    timings["predict_seconds"] = perf_counter() - stage

    # ---- prompt orders 0 and 1 with the predicted >=2 in the selection
    fastest = float(np.min(sample["group_index"]))
    time_width = float(np.ptp(kernel.config.relative_time_edges_ns))
    source_time_span = float(np.max(elements.end_ns) - np.min(elements.start_ns))
    source_extent = float(np.linalg.norm(np.ptp(elements.midpoints_m, axis=0)))
    local_origin = source_extent > high or source_time_span > 0.5 * time_width
    if time_origin_ns is not None:
        origins = np.asarray(time_origin_ns, float)
        if origins.shape != (len(positions),) or not np.isfinite(origins).all():
            raise ValueError("time_origin_ns must hold one finite origin per OM")
        local_origin = True
    elif local_origin:
        origins = earliest_element_arrival(
            np.ascontiguousarray(positions), np.ascontiguousarray(elements.start_m),
            np.ascontiguousarray(elements.direction), np.ascontiguousarray(elements.length_m),
            np.ascontiguousarray(elements.start_ns), fastest)
    else:
        first = int(np.argmin(elements.start_ns))
        origins = (float(elements.start_ns[first])
                   + np.linalg.norm(positions - elements.start_m[first], axis=1)
                   / (0.299792458 / fastest))
    safety = float(config.predictor_safety)
    stage = perf_counter()
    prompt = kernel.transport_prompt(
        source, config.prompt,
        selection_extra_charge=safety * np.abs(predicted) + np.abs(late_charge) + extra,
        time_origin_ns=origins if local_origin else None)
    timings["prompt_seconds"] = perf_counter() - stage
    if not np.array_equal(prompt.relative_time_edges_ns, kernel.config.relative_time_edges_ns):
        raise ValueError("prompt and RTE time-bin edges differ")
    if not np.allclose(prompt.time_origin_ns, origins, rtol=0, atol=1e-10):
        raise ValueError("prompt and RTE time origins differ")

    # ---- exact zero-frequency >=2 where the threshold decision needs it
    stage = perf_counter()
    charge = np.zeros((len(positions), 3))
    charge[:, :2] = prompt.charge_orders_pe
    prompt_charge = charge[:, :2].sum(axis=1)
    computable = (used > 0) | (prompt.charge_pe > 0)
    candidate = (used > 0) & (prompt_charge + safety * np.abs(predicted) + late_charge + extra
                              >= threshold)
    if threshold == 0:
        candidate = used > 0
    zero_sources = [replace(item, channels=np.ascontiguousarray(item.channels[:, :, :1]))
                    for item in sources]

    def exact(mask):
        ids = np.flatnonzero(mask)
        values = np.zeros(len(positions))
        if len(ids):
            values[ids] = apply_folded(zero, zero_sources, positions[ids], looks[ids], alpha,
                                       area[ids], (low, high))[0].real
        return values

    level = np.where(used > 0, 1, 0).astype(np.int8)
    exact_charge = exact(candidate)
    ratio = np.abs(exact_charge[candidate]) / np.maximum(np.abs(predicted[candidate]), 1e-300)
    fallback = bool(np.any(np.abs(exact_charge[candidate]) > safety * np.abs(predicted[candidate])))
    if fallback:
        # The prediction failed its own margin: no screened decision is kept.
        candidate = used > 0
        exact_charge = exact(candidate)
    charge[:, 2] = predicted + late_charge
    charge[candidate, 2] = exact_charge[candidate] + late_charge[candidate]
    level[candidate] = 2
    active = computable & (charge.sum(axis=1) + extra >= threshold)
    if threshold == 0:
        active = computable
    if np.any(active & (level == 1)):
        raise RuntimeError("an OM was activated by a predicted >=2 charge")
    if not np.all(prompt.active[active]) and not fallback:
        raise RuntimeError("prompt bins are missing for an active OM")
    timings["zero_exact_seconds"] = perf_counter() - stage

    # ---- full frequencies for the active OMs, then the readout
    stage = perf_counter()
    spectrum = np.zeros((len(omega), len(positions)), complex)
    work = np.flatnonzero(active & (used > 0))
    if len(work):
        spectrum[:, work] = apply_folded(folded, sources, positions[work], looks[work], alpha,
                                         area[work], (low, high))
    timings["prompt_full_apply_seconds"] = perf_counter() - stage
    late_spectra = []
    for shift, group in late_sources:
        values = np.zeros((len(omega), len(positions)), complex)
        if len(work):
            values[:, work] = apply_folded(folded, group, positions[work], looks[work], alpha,
                                           area[work], (low, high))
        late_spectra.append((shift, values))
    timings["full_apply_seconds"] = perf_counter() - stage
    stage = perf_counter()
    if fallback and not np.all(prompt.computed[active]):
        raise ValueError("prompt did not compute every active OM")
    # The stored spectrum and its readout hold the prompt steps; delayed light
    # enters the bins through its own clocks below and the charge directly.
    spectrum[0] = charge[:, 2] - late_charge
    components = np.zeros((len(positions), len(kernel.config.relative_time_edges_ns) - 1, 3))
    if np.any(active):
        if not np.all(prompt.active[active]):
            # Bins of these OMs were not requested from the prompt (fallback).
            redo = kernel.transport_prompt(source, replace(config.prompt, screen=False),
                                           compute_bins_for_all=True,
                                           time_origin_ns=origins if local_origin else None)
            components[active, :, :2] = redo.bins_orders_pe[active]
        else:
            components[active, :, :2] = prompt.bins_orders_pe[active]
        relative = spectrum[:, active] * np.exp(-1j * omega[:, None] * origins[None, active])
        if len(omega) != len(output_omega):
            from scipy.interpolate import CubicSpline
            relative = (CubicSpline(omega, relative.real, axis=0)(output_omega)
                        + 1j * CubicSpline(omega, relative.imag, axis=0)(output_omega))
        components[active, :, 2] = inverse_bins(output_omega, relative,
                                                kernel.config.relative_time_edges_ns)
        for shift, values in late_spectra:
            # Light of a delayed group, demodulated by its own delay, read on
            # the window edges seen from that delay: an exact change of clock.
            relative = values[:, active] * np.exp(-1j * omega[:, None]
                                                  * (origins[None, active] + shift))
            if len(omega) != len(output_omega):
                from scipy.interpolate import CubicSpline
                relative = (CubicSpline(omega, relative.real, axis=0)(output_omega)
                            + 1j * CubicSpline(omega, relative.imag, axis=0)(output_omega))
            components[active, :, 2] += inverse_bins(output_omega, relative, edges - shift)
    timings["readout_seconds"] = perf_counter() - stage
    elapsed = perf_counter() - began
    timings.update(elapsed_seconds=elapsed, cpu_seconds=process_time() - cpu)
    metadata = {
        "route": "lighthit.shower", "timings": timings,
        "prompt_timings": prompt.metadata.get("timings"),
        "elapsed_seconds": elapsed, "prompt_elapsed_seconds": timings["prompt_seconds"],
        "rte_elapsed_seconds": elapsed - timings["prompt_seconds"],
        "ge2_source": "axial-slice shower basis compiled in the band-fold node bands (axis stencils)",
        "ge2_basis": asdict(basis), "band_fold": fold.metadata(),
        "band_fold_subdivide": int(folded.get("subdivide", 1)),
        "band_fold_source_residual": fold.source_residual(
            elements.beta, elements.coefficient0 * np.clip(1 - 1 / (elements.beta * max_phase) ** 2, 0, None)),
        "ge2_points": [len(item.z_m) for item in sources],
        "delayed_light": {"prompt_delay_ns": float(config.prompt_delay_ns),
                          "groups_in_window": len(late_sources),
                          "steps_in_window_groups": int(sum(int(m.sum()) for m, _ in late_groups)),
                          "steps_beyond_window": int(beyond.sum()),
                          "delayed_charge_pe": float(late_charge.sum()),
                          "stored_spectrum": "prompt steps only (delayed light in charge and bins)"},
        "ge2_charge_level": level.tolist(),
        "ge2_charge_level_legend": {"0": "outside the cache window", "1": "predicted (inactive)",
                                    "2": "exact basis charge"},
        "ge2_predictor": {"points": int(len(collapsed[0].z_m)), "cell_m": float(config.predictor_cell_m),
                          "safety": safety,
                          "candidates": int(candidate.sum()), "reachable": int(len(reach)),
                          "fallback_to_exact": fallback,
                          "exact_over_prediction_max": float(ratio.max(initial=0.0)),
                          "exact_over_prediction_min": float(ratio.min(initial=np.inf))
                          if len(ratio) else None},
        "radial_window_m": [float(low), float(high)],
        "radial_windowed_modules": int(np.count_nonzero(far)),
        "radial_tail_estimate_max_pe": float(tail.max(initial=0.0)),
        "time_origin": ("supplied event clock" if time_origin_ns is not None else
                        "earliest local segment and fastest group flight"
                        if local_origin else "earliest element start and fastest group flight"),
        "selection_extra_charge_pe": float(extra.sum()),
        "source_elements": len(elements),
        "source_coefficient_fraction_dropped": 1 - float(elements.coefficient0.sum()) / original_coefficient,
        "active_modules": int(active.sum()), "threshold_pe": threshold,
        "cone_model": "spectral", "spectrum_computed_orders": [2],
        "frequency_nodes_computed": len(omega), "frequency_nodes_readout": len(output_omega),
        "frequency_interpolation": "cubic after OM time-origin demodulation",
        "prompt_bins": "orders 0 and 1 integrated in physical time",
        "prompt_config": asdict(config.prompt),
        "prompt_algorithm": prompt.metadata.get("prompt_algorithm"),
        "prompt_order1_level": prompt.metadata.get("order1_level"),
        "prompt_ballistic": prompt.metadata.get("ballistic"),
        "inactive_order1_charge_may_be_coarse": bool(np.any(
            np.asarray(prompt.metadata.get("order1_level", [])) == 1)),
    }
    response = HybridTransportResponse(
        kernel.detector, omega.copy(), kernel.config.relative_time_edges_ns.copy(),
        spectrum, components, charge, origins, active, "shower", metadata)
    if np.any(near):
        from ..near_field import add_near_field
        add_near_field(response, kernel, elements, store.bundle(kernel, indices), label="shower")
    return response
