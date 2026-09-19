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
from time import perf_counter

import numpy as np

from lighthit import SolverSettings, __version__, synthetic_medium
from lighthit.cache import CacheGrid, ResponseCache
from lighthit.readout import inverse_bins
from lighthit.viewer import SCHEMA, write_event_viewer
from lighthit.experimental.axial_source import AxisFrame
from lighthit.experimental.axial_fast import PreparedAxialKernel, compile_axial_source_fast
from lighthit.experimental.ballistic_fast import vectorised_ballistic_bins_fast
from lighthit.experimental.g4_source import SourceContract, load_event


def detector_array(clusters=2, strings=8, modules=36, radius_m=60.0,
                   spacing_m=15.0, pitch_m=110.0):
    """Idealised two-cluster array used by the shower studies."""
    centres = np.array([[pitch_m * index, 0.0, 0.0] for index in range(clusters)])
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


def source_display(elements, frame, label):
    return [{"type": "axis", "position_m": frame.centre_m.tolist(),
             "direction": frame.axis.tolist(), "extent_m": float(elements.extent_m),
             "label": label}]


def run_shower(path, event_index, elements, cache, prepared, receivers, medium,
               omega, edges, degree, azimuthal_degree, cell_m, receiver_block,
               output):
    began = perf_counter()
    frame = AxisFrame.of(elements)
    source = compile_axial_source_fast(
        elements, degree, omega, azimuthal_degree=azimuthal_degree,
        cell_m=cell_m, element_order=2, frame=frame)
    compile_seconds = perf_counter() - began
    began = perf_counter()
    scattered = prepared.apply(source, receivers, source_omega_per_ns=omega,
                               receiver_block=receiver_block)
    apply_seconds = perf_counter() - began
    origins = source_front(elements, receivers, medium)
    began = perf_counter()
    ballistic, ballistic_bins = vectorised_ballistic_bins_fast(
        elements, receivers, medium, elements.cone_cosine, origins, edges)
    ballistic_seconds = perf_counter() - began
    relative = scattered * np.exp(-1j * omega[:, None] * origins[None, :])[:, :, None]
    began = perf_counter()
    components = np.zeros((len(receivers), len(edges) - 1, 3), float)
    components[:, :, 0] = ballistic_bins
    for order in range(2):
        components[:, :, order + 1] = inverse_bins(omega, relative[:, :, order], edges)
    readout_seconds = perf_counter() - began
    charge = np.column_stack((ballistic, scattered[0].real))
    label = event_label(path, event_index)
    total_seconds = compile_seconds + apply_seconds + ballistic_seconds + readout_seconds
    negative = np.abs(components[components < 0]).sum()
    mass = np.abs(components).sum()
    record = {
        "event_id": f"{path.stem}-event-{event_index}", "label": label,
        "source_kind": "g4_shower", "sources": source_display(elements, frame, label),
        "time_origin_ns": origins.tolist(), "components": components.tolist(),
        "charge_components": charge.tolist(), "source_summary": elements.summary(),
        "diagnostics": {
            "elapsed_seconds": total_seconds, "compile_seconds": compile_seconds,
            "apply_both_scattered_orders_seconds": apply_seconds,
            "ballistic_and_bins_seconds": ballistic_seconds,
            "readout_seconds": readout_seconds,
            "source_cells": int(source.summary["cells"]),
            "source_channels": int(source.summary["channels"]),
            "source_megabytes": source.channels.nbytes / 2**20,
            "negative_bin_absolute_fraction": float(negative / max(mass, 1e-300)),
            "note": ("Axial fast: Lq=32, M=4, CIC 0.12 m; both scattered orders "
                     "in one Numba pass. Ballistic light uses original G4 elements "
                     "and exact arrival times. Signed Fourier bins are preserved."),
        },
    }
    np.savez_compressed(
        output / f"{path.stem}-event-{event_index}.npz", omega_per_ns=omega,
        receivers_m=receivers, time_origin_ns=origins, charge_components=charge,
        components=components, relative_edges_ns=edges)
    del source, scattered, relative, components
    gc.collect()
    return record


def run_laser(cache, receivers, medium, omega, edges, photons):
    began = perf_counter()
    position = np.array([55.0, 30.0, 0.0])
    displacement = receivers - position
    radii = np.linalg.norm(displacement, axis=1)
    spectrum = cache.acceptance_spectrum(
        radii, np.zeros(len(radii)), np.array([4 * np.pi])) * photons
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
        "charge_components": charge.tolist(),
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
    pose = np.array([20.0, 15.0, -20.0])
    loaded = []
    for path in inputs:
        print(f"read {path.name}, event {args.event}", flush=True)
        loaded.append((path, load_event(path, args.event, SourceContract()).moved(translation=pose)))
    high = max(float(np.linalg.norm(receivers - elements.centroid_m, axis=1).max()
                     + 1.5 * elements.extent_m) for _, elements in loaded)
    laser_radius = np.linalg.norm(receivers - np.array([55.0, 30.0, 0.0]), axis=1)
    high = max(high, float(laser_radius.max())) * 1.03
    low = 0.5
    cache_path = Path(args.cache).expanduser().resolve() if args.cache else output / "transport.npz"
    with threadpool_limits(limits=1):
        cache, cache_report = make_cache(cache_path, medium, omega, args.degree,
                                         low, high, args.radii)
        prepared = PreparedAxialKernel.from_cache(cache, degree=args.degree)
        events = []
        for path, elements in loaded:
            print(f"run {path.name}: {len(elements):,} light elements", flush=True)
            record = run_shower(
                path, args.event, elements, cache, prepared, receivers, medium,
                omega, edges, args.degree, args.azimuthal_degree, args.cell_m,
                args.receiver_block, output)
            events.append(record)
            print(f"  {record['diagnostics']['elapsed_seconds']:.2f} s; "
                  f"Q={np.asarray(record['charge_components']).sum():.6g}", flush=True)
        print("run isotropic laser", flush=True)
        events.append(run_laser(cache, receivers, medium, omega, edges, args.laser_photons))
    payload = {
        "schema": SCHEMA, "package_version": __version__,
        "units": {"position": "m", "time": "ns",
                  "signal": "photons / m² effective area"},
        "detector": {"label": "Idealised 2-cluster array; point isotropic receivers",
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
    summary = {event["event_id"]: {"label": event["label"],
               "integrated_charge": float(np.asarray(event["charge_components"]).sum()),
               "window_charge": float(np.asarray(event["components"]).sum()),
               "seconds": event["diagnostics"].get("elapsed_seconds", 0.0)} for event in events}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    print(f"viewer: {viewer}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
