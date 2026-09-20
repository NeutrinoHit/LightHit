#!/usr/bin/env python3
"""Run local G4 files and a point laser through the fastest LightHit route.

The shower route is the validated axial representation with the exact same
source/cache discretisation as chapter 9, evaluated by the fused Numba source
compiler and fused two-order apply.  Order 0 is evaluated on the original G4
elements and placed at its exact arrival time.  Results remain local under the
chosen output directory and are assembled into one source-agnostic viewer.
"""
from pathlib import Path
import argparse
import gc
import json
import platform
import sys
from dataclasses import replace
from time import perf_counter

import numpy as np
from scipy.spatial.transform import Rotation

from lighthit import SolverSettings, __version__, synthetic_medium
from lighthit.cache import CacheGrid, ResponseCache
from lighthit.readout import inverse_bins
from lighthit.viewer import SCHEMA, write_event_viewer
from lighthit.experimental.axial_source import AxisFrame
from lighthit.experimental.axial_fast import PreparedAxialKernel, compile_axial_source_fast
from lighthit.experimental.ballistic_fast import (vectorised_ballistic_fast,
                                                   vectorised_ballistic_bins_fast)
from lighthit.experimental.g4_source import SourceContract, load_event
from lighthit.experimental.signal_screen import (isotropic_axial_proxy,
                                                  screening_candidates)


def cluster_centres(pitch_m=250.0):
    return np.array([[0.0, 0.0, 0.0], [pitch_m, 0.0, 0.0],
                     [pitch_m / 2, pitch_m * np.sqrt(3) / 2, 0.0]])


def detector_array(clusters=3, strings=8, modules=36, radius_m=60.0,
                   spacing_m=15.0, pitch_m=250.0):
    """Idealised triangular three-cluster array."""
    centres = cluster_centres(pitch_m)[:clusters]
    offsets = np.zeros((strings, 3))
    azimuth = np.arange(strings - 1) * 2 * np.pi / (strings - 1)
    offsets[1:, :2] = radius_m * np.column_stack((np.cos(azimuth), np.sin(azimuth)))
    heights = (np.arange(modules) - (modules - 1) / 2) * spacing_m
    positions = (centres[:, None, None, :] + offsets[None, :, None, :]
                 + np.stack((np.zeros_like(heights), np.zeros_like(heights), heights),
                            axis=-1)[None, None, :, :]).reshape(-1, 3)
    cluster_id = np.repeat(np.arange(clusters), strings * modules)
    string_id = np.tile(np.repeat(np.arange(strings), modules), clusters)
    module_id = np.tile(np.arange(modules), clusters * strings)
    return positions, cluster_id, string_id, module_id


def event_label(path, event):
    words = {"sim_e_100GeV_10": "Электрон 100 ГэВ",
             "sim_e_1TeV_10": "Электрон 1 ТэВ",
             "sim_mu_100GeV_10": "Мюон 100 ГэВ",
             "sim_mu_1TeV_10": "Мюон 1 ТэВ"}
    return f"{words.get(path.stem, path.stem)} · событие {event}"


def pose_event(path, elements):
    """Place showers near one cluster and long muons across the array."""
    centres = cluster_centres()
    if path.stem == "sim_e_100GeV_10":
        target, axis = centres[0] + [35.0, 25.0, -35.0], np.array([.45, .20, .87])
    elif path.stem == "sim_e_1TeV_10":
        target, axis = centres[2] + [-35.0, -25.0, 20.0], np.array([-.40, -.35, .85])
    elif path.stem == "sim_mu_100GeV_10":
        target, axis = centres.mean(axis=0) + [-30.0, -15.0, -35.0], np.array([.83, .45, .32])
    else:
        target, axis = centres.mean(axis=0) + [25.0, 20.0, 15.0], np.array([-.72, .62, .31])
    axis = axis / np.linalg.norm(axis)
    original = AxisFrame.of(elements).axis
    rotation = Rotation.align_vectors(axis[None, :], original[None, :])[0].as_matrix()
    translation = target - rotation @ elements.centroid_m
    return elements.moved(rotation=rotation, translation=translation), {
        "target_centroid_m": target.tolist(), "target_axis": axis.tolist(),
        "rotation": rotation.tolist(), "translation_m": translation.tolist()}


def cache_matches(cache, medium, omega, degree, low, high):
    return (cache.medium == medium and cache.degree >= degree
            and np.array_equal(cache.grid.omega_per_ns, omega)
            and cache.radius_range_m[0] <= low and cache.radius_range_m[1] >= high)


def make_cache(path, medium, omega, degree, low, high, radii):
    if path.exists():
        cache = ResponseCache.load(path)
        if cache_matches(cache, medium, omega, degree, low, high):
            return cache, {"loaded": True, "path": str(path)}
    settings = SolverSettings(24, degree, 8.0, 0.04, 10)
    began = perf_counter()
    cache = ResponseCache.build(
        medium, settings, CacheGrid.geometric(low, high, radii, omega),
        angular_backend="numba", radial_phase="flight")
    path.parent.mkdir(parents=True, exist_ok=True)
    cache.save(path)
    return cache, {"loaded": False, "path": str(path),
                   "build_seconds": perf_counter() - began,
                   "stages": cache.timings_s}


def source_front(elements, receivers, medium):
    """Stable per-OM time origin from the earliest stored emitting step."""
    index = int(np.argmin(elements.start_ns))
    start = elements.start_m[index]
    return (float(elements.start_ns[index])
            + np.linalg.norm(receivers - start[None, :], axis=1)
            / medium.speed_m_per_ns)


def source_display(elements, frame, label, shape):
    return [{"type": "axis", "position_m": frame.centre_m.tolist(),
             "direction": frame.axis.tolist(), "extent_m": float(elements.extent_m),
             "shape": shape, "label": label}]


def run_shower(path, event_index, elements, cache, prepared, zero_prepared,
               receivers, medium,
               omega, edges, degree, azimuthal_degree, cell_m, receiver_block,
               output, *, threshold_pe, effective_area_m2, efficiency,
               safety_factor, proxy_bins, audit_screen=False, pose=None):
    began = perf_counter()
    frame = AxisFrame.of(elements)
    source = compile_axial_source_fast(
        elements, degree, omega, azimuthal_degree=azimuthal_degree,
        cell_m=cell_m, element_order=2, frame=frame)
    compile_seconds = perf_counter() - began

    # Screening never trusts the proxy alone. Exact ballistic charge is cheap,
    # and a one-frequency axial guard certifies every rejected OM before the
    # full spectrum is omitted.
    began = perf_counter()
    ballistic_all = vectorised_ballistic_fast(
        elements, receivers, medium, elements.cone_cosine)
    proxy = isotropic_axial_proxy(elements, receivers, medium, bins=proxy_bins,
                                  frame=frame)
    candidates, estimate_pe = screening_candidates(
        ballistic_all, proxy, threshold_pe=threshold_pe,
        effective_area_m2=effective_area_m2, efficiency=efficiency,
        safety_factor=safety_factor)
    exposure = effective_area_m2 * efficiency
    charge_guard = np.zeros((len(receivers), 2), float)
    promoted = np.zeros(len(receivers), bool)
    if threshold_pe > 0 and np.any(~candidates):
        zero_source = replace(
            source, channels=np.ascontiguousarray(source.channels[:, :, :1]))
        rejected = np.flatnonzero(~candidates)
        guarded = zero_prepared.apply(
            zero_source, receivers[rejected], source_omega_per_ns=omega[:1],
            receiver_block=receiver_block)[0].real
        charge_guard[rejected] = guarded
        exact_pe = exposure * (ballistic_all[rejected] + guarded.sum(axis=1))
        promoted[rejected] = exact_pe >= threshold_pe
        del zero_source
    active = candidates | promoted if threshold_pe > 0 else np.ones(len(receivers), bool)
    screening_seconds = perf_counter() - began

    began = perf_counter()
    scattered_active = (prepared.apply(
        source, receivers[active], source_omega_per_ns=omega,
        receiver_block=receiver_block) if np.any(active)
        else np.zeros((len(omega), 0, 2), complex))
    apply_seconds = perf_counter() - began
    origins = source_front(elements, receivers, medium)
    began = perf_counter()
    _, ballistic_bins = vectorised_ballistic_bins_fast(
        elements, receivers[active], medium, elements.cone_cosine,
        origins[active], edges)
    ballistic_seconds = perf_counter() - began
    relative = (scattered_active
                * np.exp(-1j * omega[:, None] * origins[None, active])[:, :, None])
    began = perf_counter()
    components = np.zeros((len(receivers), len(edges) - 1, 3), float)
    components[active, :, 0] = ballistic_bins
    for order in range(2):
        components[active, :, order + 1] = inverse_bins(
            omega, relative[:, :, order], edges)
    readout_seconds = perf_counter() - began
    charge = np.column_stack((ballistic_all, charge_guard))
    charge[active, 1:] = scattered_active[0].real
    components *= exposure
    charge *= exposure

    audit = None
    if audit_screen and np.any(~active):
        began = perf_counter()
        omitted = prepared.apply(source, receivers[~active], source_omega_per_ns=omega,
                                 receiver_block=receiver_block)
        omitted_seconds = perf_counter() - began
        delta = np.max(np.abs(omitted[0].real * exposure - charge[~active, 1:]))
        omitted_total_pe = exposure * (ballistic_all[~active] + omitted[0].real.sum(axis=1))
        audit = {"omitted_full_spectrum_seconds": omitted_seconds,
                 "charge_guard_max_absolute_difference_pe": float(delta),
                 "omitted_above_threshold": int(np.sum(omitted_total_pe >= threshold_pe)),
                 "maximum_omitted_charge_pe": float(np.max(omitted_total_pe)),
                 "unscreened_apply_equivalent_seconds": apply_seconds + omitted_seconds}
        del omitted
    label = event_label(path, event_index)
    total_seconds = (compile_seconds + screening_seconds + apply_seconds
                     + ballistic_seconds + readout_seconds)
    negative = np.abs(components[components < 0]).sum()
    mass = np.abs(components).sum()
    record = {
        "event_id": f"{path.stem}-event-{event_index}", "label": label,
        "source_kind": "g4_shower",
        "sources": source_display(elements, frame, label,
                                  "track" if path.stem.startswith("sim_mu") else "spindle"),
        "time_origin_ns": origins.tolist(), "components": components.tolist(),
        "charge_components": charge.tolist(), "active": active.tolist(),
        "source_summary": elements.summary(), "pose": pose or {},
        "diagnostics": {
            "elapsed_seconds": total_seconds, "compile_seconds": compile_seconds,
            "screening_seconds": screening_seconds,
            "apply_both_scattered_orders_seconds": apply_seconds,
            "ballistic_and_bins_seconds": ballistic_seconds,
            "readout_seconds": readout_seconds,
            "source_cells": int(source.summary["cells"]),
            "source_channels": int(source.summary["channels"]),
            "source_megabytes": source.channels.nbytes / 2**20,
            "screening": {"threshold_pe": threshold_pe,
                          "effective_area_m2": effective_area_m2,
                          "efficiency": efficiency, "safety_factor": safety_factor,
                          "proxy_bins": proxy_bins,
                          "initial_candidates": int(candidates.sum()),
                          "promoted_by_exact_charge_guard": int(promoted.sum()),
                          "full_spectra_computed": int(active.sum()),
                          "full_spectra_skipped": int((~active).sum()),
                          "skipped_integrated_charge_pe": float(charge[~active].sum()),
                          "skipped_fraction_of_array_charge": float(
                              charge[~active].sum() / max(charge.sum(), 1e-300)),
                          "largest_rejected_proxy_pe": float(
                              estimate_pe[~candidates].max()) if np.any(~candidates) else 0.0,
                          "audit": audit},
            "negative_bin_absolute_fraction": float(negative / max(mass, 1e-300)),
            "note": ("Axial fast: Lq=32, M=4, CIC 0.12 m; both scattered orders "
                     "in one Numba pass. Ballistic light uses original G4 elements "
                     "and exact arrival times. Screening uses an isotropic proxy "
                     "followed by an exact omega=0 guard. Signed bins are preserved."),
        },
    }
    np.savez_compressed(
        output / f"{path.stem}-event-{event_index}.npz", omega_per_ns=omega,
        receivers_m=receivers, time_origin_ns=origins, charge_components=charge,
        components=components, relative_edges_ns=edges)
    del source, scattered_active, relative, components
    gc.collect()
    return record


def run_laser(cache, receivers, medium, omega, edges, photons, exposure):
    began = perf_counter()
    position = cluster_centres().mean(axis=0)
    displacement = receivers - position
    radii = np.linalg.norm(displacement, axis=1)
    spectrum = (cache.acceptance_spectrum(
        radii, np.zeros(len(radii)), np.array([4 * np.pi]))
        * photons * exposure)
    origins = radii / medium.speed_m_per_ns
    relative = spectrum * np.exp(-1j * omega[:, None] * origins[None, :])[:, :, None]
    components = np.zeros((len(receivers), len(edges) - 1, 3), float)
    for order in (1, 2):
        components[:, :, order] = inverse_bins(omega, relative[:, :, order], edges)
    zero = np.searchsorted(edges, 0.0, side="right") - 1
    if 0 <= zero < len(edges) - 1:
        components[:, zero, 0] = spectrum[0, :, 0].real
    charge = spectrum[0].real
    return {
        "event_id": "isotropic-laser", "label": "Изотропный лазер",
        "source_kind": "point_flash",
        "sources": [{"type": "point", "position_m": position.tolist(),
                     "label": "Изотропный лазер"}],
        "time_origin_ns": origins.tolist(), "components": components.tolist(),
        "charge_components": charge.tolist(), "active": np.ones(len(receivers), bool).tolist(),
        "source_summary": {"photons": float(photons), "position_m": position.tolist()},
        "diagnostics": {
            "elapsed_seconds": perf_counter() - began,
            "note": ("Isotropic point flash queried from the same cache. Order 0 is "
                     "placed analytically at the front; orders 1 and >=2 are the "
                     "finite-L cached multipoles. Signed Fourier bins are preserved."),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="*", default=None,
                        help="G4 HDF5 files; default: all g4_data/*.h5")
    parser.add_argument("--event", type=int, default=5)
    parser.add_argument("--output", default=".build/event-viewer")
    parser.add_argument("--cache", default=None)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--frequencies", type=int, default=41)
    parser.add_argument("--omega-max", type=float, default=0.15,
                        help="Lower cutoff keeps a long reconstructible window at fixed cost")
    parser.add_argument("--degree", type=int, default=32)
    parser.add_argument("--azimuthal-degree", type=int, default=4)
    parser.add_argument("--cell-m", type=float, default=0.12)
    parser.add_argument("--receiver-block", type=int, default=8)
    parser.add_argument("--radii", type=int, default=180)
    parser.add_argument("--time-start", type=float, default=-60.0)
    parser.add_argument("--time-stop", type=float, default=740.0)
    parser.add_argument("--bin-ns", type=float, default=20.0)
    parser.add_argument("--laser-photons", type=float, default=1e10)
    parser.add_argument("--threshold-pe", type=float, default=0.01,
                        help="Skip full spectra below this expected integrated charge")
    parser.add_argument("--effective-area-m2", type=float, default=0.053)
    parser.add_argument("--efficiency", type=float, default=0.20)
    parser.add_argument("--screen-safety-factor", type=float, default=10.0)
    parser.add_argument("--screen-proxy-bins", type=int, default=64)
    parser.add_argument("--audit-screen", action="store_true",
                        help="Also compute omitted spectra to measure screening speed/error")
    args = parser.parse_args()
    import numba
    numba.set_num_threads(args.threads)
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        from contextlib import nullcontext
        threadpool_limits = lambda **_kwargs: nullcontext()
    root = Path(__file__).resolve().parents[1]
    inputs = [Path(value) for value in args.input] if args.input else sorted((root / "g4_data").glob("*.h5"))
    if not inputs or any(not path.is_file() for path in inputs):
        raise SystemExit("No readable G4 input files")
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    omega = np.linspace(0.0, args.omega_max, args.frequencies)
    edges = np.arange(args.time_start, args.time_stop + 0.5 * args.bin_ns, args.bin_ns)
    medium = synthetic_medium()
    receivers, cluster, string, module = detector_array()
    loaded = []
    for path in inputs:
        print(f"read {path.name}, event {args.event}", flush=True)
        raw = load_event(path, args.event, SourceContract())
        elements, pose = pose_event(path, raw)
        loaded.append((path, elements, pose))
    high = max(float(np.linalg.norm(receivers - elements.centroid_m, axis=1).max()
                     + 1.5 * elements.extent_m) for _, elements, _ in loaded)
    laser_radius = np.linalg.norm(receivers - cluster_centres().mean(axis=0), axis=1)
    high = max(high, float(laser_radius.max())) * 1.03
    low = 0.5
    cache_path = Path(args.cache).expanduser().resolve() if args.cache else output / "transport.npz"
    with threadpool_limits(limits=1):
        cache, cache_report = make_cache(cache_path, medium, omega, args.degree,
                                         low, high, args.radii)
        prepared = PreparedAxialKernel.from_cache(cache, degree=args.degree)
        zero_prepared = PreparedAxialKernel.from_cache(
            cache, degree=args.degree, frequency_indices=[0])
        events = []
        for path, elements, pose in loaded:
            print(f"run {path.name}: {len(elements):,} light elements", flush=True)
            record = run_shower(
                path, args.event, elements, cache, prepared, zero_prepared,
                receivers, medium,
                omega, edges, args.degree, args.azimuthal_degree, args.cell_m,
                args.receiver_block, output, threshold_pe=args.threshold_pe,
                effective_area_m2=args.effective_area_m2,
                efficiency=args.efficiency, safety_factor=args.screen_safety_factor,
                proxy_bins=args.screen_proxy_bins, audit_screen=args.audit_screen,
                pose=pose)
            events.append(record)
            print(f"  {record['diagnostics']['elapsed_seconds']:.2f} s; "
                  f"Q={np.asarray(record['charge_components']).sum():.6g}", flush=True)
        print("run isotropic laser", flush=True)
        events.append(run_laser(
            cache, receivers, medium, omega, edges, args.laser_photons,
            args.effective_area_m2 * args.efficiency))
    payload = {
        "schema": SCHEMA, "package_version": __version__,
        "units": {"position": "m", "time": "ns",
                  "signal": "photoelectrons"},
        "detector": {"label": ("Idealised triangular 3-cluster array; point isotropic "
                                f"receivers; A_eff={args.effective_area_m2:g} m²; "
                                f"efficiency={args.efficiency:g}"),
                     "positions_m": receivers.tolist(), "cluster_id": cluster.tolist(),
                     "string_id": string.tolist(), "module_id": module.tolist()},
        "medium": {"absorption_per_m": medium.absorption_per_m,
                   "scattering_per_m": medium.scattering_per_m, "g": medium.g,
                   "group_index": medium.group_index, "wavelength_nm": medium.wavelength_nm,
                   "provenance": medium.provenance},
        "readout": {"relative_time_edges_ns": edges.tolist(),
                    "frequency_nodes": len(omega), "omega_max_per_ns": float(omega[-1]),
                    "note": "Per-OM time zero; no detector smearing."},
        "calculation": {"method": "axial_fast", "cache": cache_report,
                        "settings": vars(args), "python": sys.version,
                        "platform": platform.platform(), "numba": numba.__version__},
        "events": events,
    }
    viewer = write_event_viewer(payload, output / "viewer.html", json_path=output / "viewer.json")
    summary = {}
    for event in events:
        row = {"label": event["label"],
               "integrated_charge": float(np.asarray(event["charge_components"]).sum()),
               "window_charge": float(np.asarray(event["components"]).sum()),
               "seconds": event["diagnostics"].get("elapsed_seconds", 0.0)}
        screening = event["diagnostics"].get("screening")
        if screening:
            row.update(full_spectra=screening["full_spectra_computed"],
                       skipped_spectra=screening["full_spectra_skipped"])
            audit = screening.get("audit")
            if audit:
                row["measured_apply_speedup"] = (
                    audit["unscreened_apply_equivalent_seconds"]
                    / event["diagnostics"]["apply_both_scattered_orders_seconds"])
        summary[event["event_id"]] = row
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    print(f"viewer: {viewer}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
