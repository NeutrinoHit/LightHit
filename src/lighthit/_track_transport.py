"""Fast exact path for a straight Cherenkov track and directional OMs.

A straight uniform track is axially symmetric, so its source has only m=0.
The directional detector indices remain untouched in the directional cache.
For each OM only source cells inside the validated radial cache are evaluated;
ballistic light is analytic and is never radially truncated.
"""
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter

import numpy as np

from .readout import inverse_bins


def _radial_windows(source, compiled, positions, low, high):
    """Per-OM contiguous cell window inside ``high`` for a straight track."""
    positions = np.asarray(positions, float)
    centre = np.asarray(compiled.frame.centre_m, float)
    local_points = np.asarray(compiled.frame.rotate(compiled.points_m() - centre), float)
    local_receivers = np.asarray(compiled.frame.rotate(positions - centre), float)
    if np.max(np.abs(local_points[:, :2])) > 1e-9:
        raise ValueError("track fast path requires source cells on one axis")
    z_cells = local_points[:, 2]
    if len(z_cells) > 1 and np.any(np.diff(z_cells) <= 0):
        raise ValueError("track source cells must be strictly ordered along the axis")

    b2 = np.sum(local_receivers[:, :2] ** 2, axis=1)
    rz = local_receivers[:, 2]
    safe_high = np.nextafter(float(high), 0.0)
    reach = np.sqrt(np.maximum(safe_high * safe_high - b2, 0.0))
    begin = np.searchsorted(z_cells, rz - reach, side="left").astype(np.int64)
    end = np.searchsorted(z_cells, rz + reach, side="right").astype(np.int64)
    outside_cylinder = b2 > safe_high * safe_high
    begin[outside_cylinder] = 0
    end[outside_cylinder] = 0

    # The lower radial edge cannot be handled by omission: it is the region
    # where the cached Green function is not validated and the response is large.
    start = np.asarray(source.start_m, float)
    direction = np.asarray(source.direction, float)
    direction = direction / np.linalg.norm(direction)
    rel = positions - start
    along = rel @ direction
    closest = np.clip(along, 0.0, float(source.length_m))
    nearest = rel - closest[:, None] * direction[None, :]
    minimum_distance = np.linalg.norm(nearest, axis=1)
    too_close = minimum_distance < low
    if np.any(too_close):
        raise ValueError(
            f"{int(too_close.sum())} OMs approach the Cherenkov track closer than "
            f"the validated radial minimum {low:g} m")

    # Nearest omitted compiled cell.  This is the correct distance for the
    # omission bound because the transport engine acts on the compiled source,
    # not on an ideal continuum after the source has been deposited.
    omitted_distance = np.full(len(positions), np.inf)
    for index in range(len(positions)):
        candidates = []
        if begin[index] == end[index]:
            nearest = int(np.searchsorted(z_cells, rz[index], side="left"))
            if nearest > 0:
                candidates.append(nearest - 1)
            if nearest < len(z_cells):
                candidates.append(nearest)
        else:
            if begin[index] > 0:
                candidates.append(begin[index] - 1)
            if end[index] < len(z_cells):
                candidates.append(end[index])
        for cell in candidates:
            dz = z_cells[cell] - rz[index]
            distance = np.sqrt(b2[index] + dz * dz)
            if distance < omitted_distance[index]:
                omitted_distance[index] = distance

    return begin, end, omitted_distance


def _track_frame(source, AxisFrame):
    """Exact source frame for a straight track; no PCA is needed."""
    axis = np.asarray(source.direction, float)
    axis = axis / np.linalg.norm(axis)
    centre = np.asarray(source.start_m, float) + 0.5 * float(source.length_m) * axis
    helper = np.array([0.0, 0.0, 1.0])
    if abs(float(axis @ helper)) > 0.9:
        helper = np.array([1.0, 0.0, 0.0])
    first = np.cross(axis, helper)
    first /= np.linalg.norm(first)
    return AxisFrame(centre, axis, first, np.cross(axis, first))


def transport_track_directional(kernel, source, *, method):
    """Directional-OM transport specialized to a straight Cherenkov track."""
    from ._directional_numba import PreparedDirectionalKernel
    from .experimental.axial_fast import compile_axial_source_fast
    from .experimental.axial_source import AxisFrame
    from .experimental.ballistic_fast import ballistic_directional_fast

    began = perf_counter()
    response = kernel.acceptance()
    alpha = np.ascontiguousarray(response["alpha"], float)
    elements = kernel._source_elements(source)
    wavelengths = kernel.wavelength.wavelength_nm
    weights = kernel.wavelength.weight_nm
    sample = kernel.medium.sample(wavelengths)

    if source.beta * float(np.min(sample["phase_index"])) <= 1:
        raise ValueError("source is below Cherenkov threshold over the wavelength range")

    field = elements.field(0)
    frame = _track_frame(source, AxisFrame)
    degree = kernel.config.source_degree
    omega = kernel.config.omega_per_ns

    # A straight track is exactly axially symmetric: only m=0 exists.  Keep the
    # configured longitudinal spacing but do not create transverse CIC cells.
    compiled = compile_axial_source_fast(
        field, degree, omega, azimuthal_degree=0,
        cell_m=kernel.config.cell_m, axial_only=True,
        element_order=2, frame=frame)
    compile_seconds = perf_counter() - began

    positions = np.asarray(kernel.detector.positions_m, float)
    looks = np.ascontiguousarray(-kernel.detector.orientations, float)
    scale_area = np.asarray(kernel.detector.effective_area_m2, float)
    low, high = kernel.config.radial_range_m
    cell_begin, cell_end, omitted_distance = _radial_windows(
        source, compiled, positions, low, high)
    has_window = cell_end > cell_begin
    work = np.flatnonzero(has_window)
    folded = None
    folded_load_seconds = 0.0
    if kernel.config.spectral_folded_cache is not False:
        from ._spectral_fold import optional_folded_cache
        started = perf_counter()
        folded = optional_folded_cache(kernel)
        folded_load_seconds = perf_counter() - started

    # Fraction of the actually compiled m=0 source omitted for each OM.
    # At omega=0 the l=0 channel is positive and proportional to photon weight.
    cell_weight = np.maximum(np.asarray(compiled.channels[:, 0, 0].real), 0.0)
    prefix = np.concatenate(([0.0], np.cumsum(cell_weight)))
    retained = prefix[cell_end] - prefix[cell_begin]
    omitted_fraction = np.clip(1.0 - retained / prefix[-1], 0.0, 1.0)

    # The two Cherenkov spectral fields are exactly proportional for one track:
    # field2 = field0 / beta^2.  Therefore one compiled source is sufficient.
    spectral_factor = (
        1.0 / wavelengths ** 2
        * (1.0 - 1.0 / (source.beta ** 2 * sample["phase_index"] ** 2)))
    emitted = float(elements.coefficient0.sum() * np.sum(weights * spectral_factor))
    peak_acceptance = float(np.max(np.abs(np.asarray(
        kernel.detector.angular_acceptance(np.linspace(-1, 1, 257)), float))))
    spectral_peak = float(np.max(kernel.detector.spectral_weight(wavelengths)))
    tail_distance = omitted_distance
    tail_upper = (
        10 * max(emitted, 0.0) * omitted_fraction * scale_area * peak_acceptance
        * spectral_peak
        * np.exp(-float(np.min(sample["absorption_per_m"])) * tail_distance)
        / (4 * np.pi * tail_distance ** 2))
    unsafe = (omitted_fraction > 1e-14) & (tail_upper >= kernel.config.threshold_pe)
    if np.any(unsafe):
        raise ValueError(
            f"{int(unsafe.sum())} OMs have track segments outside the validated "
            "directional radial range with a conservative estimate above threshold")

    fastest_group = float(np.min(sample["group_index"]))
    origins = source.earliest_arrival_ns(positions, fastest_group)

    charge = np.zeros((len(positions), 3))
    zero_source = replace(
        compiled, channels=np.ascontiguousarray(compiled.channels[:, :, :1]))
    caches = []
    cache_seconds = 0.0
    prepass_seconds = 0.0
    zero_prepare_seconds = 0.0
    zero_apply_seconds = 0.0
    ballistic_charge_seconds = 0.0

    for wavelength, weight, factor in zip(
            wavelengths, weights, spectral_factor, strict=True):
        before = perf_counter()
        cache = (kernel._directional_cache_for(float(wavelength))
                 if folded is None else None)
        cache_seconds += perf_counter() - before
        caches.append(cache)
        medium = cache.medium if cache is not None else kernel.medium.band(float(wavelength))

        before = perf_counter()
        detector_scale = (
            weight * scale_area
            * float(kernel.detector.spectral_weight(float(wavelength))))
        ballistic_charge = ballistic_directional_fast(
            field, positions, looks, medium, field.cone_cosine, alpha)[0]
        charge[:, 0] += detector_scale * factor * ballistic_charge
        ballistic_charge_seconds += perf_counter() - before

        if len(work) and folded is None:
            prepare_start = perf_counter()
            zero = PreparedDirectionalKernel.from_cache(
                cache, degree=degree, frequency_indices=[0])
            zero_prepare_seconds += perf_counter() - prepare_start
            zero_apply_start = perf_counter()
            scattered = zero.apply(
                zero_source, positions[work], looks[work], alpha,
                source_omega_per_ns=omega[:1],
                receiver_block=kernel.config.receiver_block,
                cell_begin=cell_begin[work], cell_end=cell_end[work])[0].real
            zero_apply_seconds += perf_counter() - zero_apply_start
            charge[work, 1:] += (
                detector_scale[work, None] * factor * scattered)
        prepass_seconds += perf_counter() - before

    if len(work) and folded is not None:
        zero_apply_start = perf_counter()
        def apply_zero(prepared):
            return prepared.apply(
                zero_source, positions[work], looks[work], alpha,
                source_omega_per_ns=omega[:1],
                receiver_block=kernel.config.receiver_block,
                cell_begin=cell_begin[work], cell_end=cell_end[work])[0].real
        if source.beta == 1.0:
            scattered = apply_zero(folded.zero_frequency("track_beta1"))
        else:
            scattered = (apply_zero(folded.zero_frequency("field0"))
                         - apply_zero(folded.zero_frequency("field2"))
                         / source.beta ** 2)
        zero_apply_seconds += perf_counter() - zero_apply_start
        prepass_seconds += perf_counter() - zero_apply_start
        charge[work, 1:] += scale_area[work, None] * scattered

    computable = has_window | (np.abs(charge[:, 0]) > 0)
    active = computable & (charge.sum(axis=1) >= kernel.config.threshold_pe)
    if kernel.config.threshold_pe == 0:
        active = computable

    spectrum = np.zeros((len(omega), len(positions), 3), complex)
    components = np.zeros((
        len(positions), len(kernel.config.relative_time_edges_ns) - 1, 3))
    spectrum[0] = charge
    active_index = np.flatnonzero(active)
    scattered_index = np.flatnonzero(active & has_window)

    apply_start = perf_counter()
    full_prepare_seconds = 0.0
    full_prepare_wait_seconds = 0.0
    full_scatter_seconds = 0.0
    ballistic_bins_seconds = 0.0
    full_receiver_block = 1  # one independent receiver per Numba task

    def prepare_full(cache):
        started = perf_counter()
        prepared = PreparedDirectionalKernel.from_cache(cache, degree=degree)
        return prepared, perf_counter() - started

    if len(active_index) and folded is None:
        # Build the next wavelength's spline while Numba applies the current
        # one. Keeping only one look-ahead spline bounds the extra memory, even
        # for the 321-frequency production grid.
        with ThreadPoolExecutor(max_workers=1) as executor:
            pending = (executor.submit(prepare_full, caches[0])
                       if len(scattered_index) else None)
            for index, (wavelength, weight, factor, cache) in enumerate(zip(
                    wavelengths, weights, spectral_factor, caches, strict=True)):
                detector_scale = (
                    weight * scale_area
                    * float(kernel.detector.spectral_weight(float(wavelength))))

                if pending is not None:
                    wait_start = perf_counter()
                    full, prepare_time = pending.result()
                    full_prepare_wait_seconds += perf_counter() - wait_start
                    full_prepare_seconds += prepare_time
                    pending = (executor.submit(prepare_full, caches[index + 1])
                               if index + 1 < len(caches) else None)
                    scatter_start = perf_counter()
                    scattered = full.apply(
                        compiled, positions[scattered_index], looks[scattered_index],
                        alpha, source_omega_per_ns=omega,
                        receiver_block=full_receiver_block,
                        cell_begin=cell_begin[scattered_index],
                        cell_end=cell_end[scattered_index])
                    full_scatter_seconds += perf_counter() - scatter_start
                    spectrum[:, scattered_index, 1:] += (
                        detector_scale[scattered_index][None, :, None]
                        * factor * scattered)
                    del full

                ballistic_start = perf_counter()
                _, ballistic_bins = ballistic_directional_fast(
                    field, positions[active_index], looks[active_index], cache.medium,
                    field.cone_cosine, alpha,
                    time_origin_ns=origins[active_index],
                    relative_edges_ns=kernel.config.relative_time_edges_ns)
                ballistic_bins_seconds += perf_counter() - ballistic_start
                components[active_index, :, 0] += (
                    detector_scale[active_index, None] * factor * ballistic_bins)

    if len(active_index) and folded is not None:
        if len(scattered_index):
            scatter_start = perf_counter()
            def apply_full(prepared):
                return prepared.apply(
                    compiled, positions[scattered_index],
                    looks[scattered_index], alpha, source_omega_per_ns=omega,
                    receiver_block=full_receiver_block,
                    cell_begin=cell_begin[scattered_index],
                    cell_end=cell_end[scattered_index])
            if source.beta == 1.0:
                scattered = apply_full(folded.track_beta1)
            else:
                scattered = (apply_full(folded.field0)
                             - apply_full(folded.field2) / source.beta ** 2)
            full_scatter_seconds += perf_counter() - scatter_start
            spectrum[:, scattered_index, 1:] += (
                scale_area[scattered_index][None, :, None] * scattered)
        for wavelength, weight, factor in zip(
                wavelengths, weights, spectral_factor, strict=True):
            detector_scale = (
                weight * scale_area[active_index]
                * float(kernel.detector.spectral_weight(float(wavelength))))
            ballistic_start = perf_counter()
            _, ballistic_bins = ballistic_directional_fast(
                field, positions[active_index], looks[active_index],
                kernel.medium.band(float(wavelength)), field.cone_cosine,
                alpha, time_origin_ns=origins[active_index],
                relative_edges_ns=kernel.config.relative_time_edges_ns)
            ballistic_bins_seconds += perf_counter() - ballistic_start
            components[active_index, :, 0] += (
                detector_scale[:, None] * factor * ballistic_bins)

    readout_start = perf_counter()
    spectrum[0] = charge
    relative = (
        spectrum[:, active_index, 1:]
        * np.exp(-1j * omega[:, None] * origins[None, active_index])[:, :, None])
    if len(omega) > 1 and len(active_index):
        for order in range(2):
            components[active_index, :, order + 1] = inverse_bins(
                omega, relative[:, :, order], kernel.config.relative_time_edges_ns)
    apply_seconds = perf_counter() - apply_start
    readout_seconds = perf_counter() - readout_start

    counts = cell_end - cell_begin
    used_counts = counts[has_window]
    return kernel._response(
        spectrum, components, charge, origins, active, method,
        {
            "elapsed_seconds": perf_counter() - began,
            "compile_seconds": compile_seconds,
            "apply_seconds": apply_seconds,
            "cache_seconds": cache_seconds,
            "folded_load_seconds": folded_load_seconds,
            "spectral_folded_cache": folded is not None,
            "spectral_folded_cache_path": str(folded.path) if folded is not None else None,
            "prepass_seconds": prepass_seconds,
            "zero_prepare_seconds": zero_prepare_seconds,
            "zero_apply_seconds": zero_apply_seconds,
            "ballistic_charge_seconds": ballistic_charge_seconds,
            "full_prepare_seconds": full_prepare_seconds,
            "full_prepare_wait_seconds": full_prepare_wait_seconds,
            "full_scatter_seconds": full_scatter_seconds,
            "full_receiver_block": full_receiver_block,
            "ballistic_bins_seconds": ballistic_bins_seconds,
            "readout_seconds": readout_seconds,
            "backend": ("numba axial m=0 track compile + spectral-folded directional"
                        if folded is not None else
                        "numba axial m=0 track compile + numba directional windowed"),
            "wavelength_nodes": len(wavelengths),
            "spectral_source": "lambda^-2 * (1 - (beta*n_phase)^-2)",
            "source_fields": 1,
            "reference_phase_index": elements.reference_phase_index,
            "source_elements": len(elements),
            "source_elements_dropped_at_spectral_threshold": 0,
            "source_coefficient_fraction_dropped": 0.0,
            "compiled_source_cells": int(len(compiled.z_m)),
            "compiled_source_channels": int(compiled.channels.shape[1]),
            "active_modules": int(active.sum()),
            "threshold_pe": kernel.config.threshold_pe,
            "fourier_inverted_orders": [1, 2] if len(omega) > 1 else [],
            "radial_range_m": list(kernel.config.radial_range_m),
            "omitted_outside_range": int(np.count_nonzero(omitted_fraction > 1e-14)),
            "radial_windowed_modules": int(np.count_nonzero(has_window)),
            "radial_cells_used_min": int(used_counts.min()) if len(used_counts) else 0,
            "radial_cells_used_max": int(used_counts.max()) if len(used_counts) else 0,
            "radial_cells_used_mean": float(used_counts.mean()) if len(used_counts) else 0.0,
            "radial_cells_skipped": int(
                np.sum(len(compiled.z_m) - counts[has_window])) if len(used_counts) else 0,
            "radial_tail_fraction_max": float(omitted_fraction.max()),
            "track_fast_path": True,
            "time_origin": "earliest straight-track emission and group flight",
            "detector_angular_model": "exact_m_blocks",
            "detector_acceptance_degree": response["degree"],
            "detector_acceptance_residual_above_degree":
                response["residual_above_degree"],
            "azimuthal_degree": 0,
            "ballistic_angular_model": "exact per-element arrival direction",
            "ballistic_spectrum":
                "charge and exact bins stored; nonzero-frequency column omitted",
        })
