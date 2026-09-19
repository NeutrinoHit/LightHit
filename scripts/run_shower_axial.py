"""A whole stored shower, every module, every component, timed end to end.

This is the production run of the sparse axial representation: one G4 event,
the full array, the full frequency window, charges and time bins, a second
pose chosen after the tables were built, and a held-out event with the settings
frozen. Nothing here is a scan for its own sake; the scans are the refinement
checks that precede the run and are reported beside it.

What is compared, and against what:

* the ``>=2`` and finite-``L`` first components of the compact source are
  measured against the **same kernel summed element by element** at a handful
  of control modules. That isolates the error of the representation from the
  cache's own truncations, which are measured elsewhere, and it is a control
  sum rather than physics -- an independent physical control of ``>=2``
  remains a separate obligation;
* the unscattered order is not represented at all. It has a closed form per
  element (@eq-ballistic-fluence) and is evaluated that way for every module,
  so its cost enters the event budget without any approximation;
* time bins are produced from the full frequency window rather than quoted at
  a few frequencies, because a spectrum that agrees at four points can still
  misplace the leading edge.

Errors are reported per control module -- each signal, the maximum and the
early window -- and never averaged into a single number that would then be
called the accuracy of all 576.

Run:
  python scripts/run_shower_axial.py --input /path/sim_e_100GeV_10.h5 \
      --event 5 --held-out 6 --output .build/review/shower-axial
"""
from pathlib import Path
import argparse
import json
import platform
import resource
import sys
from time import perf_counter

import numpy as np
from scipy.spatial.transform import Rotation

from lighthit import SolverSettings, synthetic_medium
from lighthit.cache import CacheGrid, ResponseCache
from lighthit.readout import inverse_bins
from lighthit.experimental.axial_source import AxisFrame, AxialSource, axial_response
from lighthit.experimental.event_moments import KernelChannels, direct_response
from lighthit.experimental.g4_source import SourceContract, load_event

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_shower_moments import array_positions, serial, vectorised_ballistic  # noqa: E402

try:
    from lighthit.experimental.spline_fast import PreparedMultipoles
    SPLINE_BACKEND = "numba (spline_fast.PreparedMultipoles)"
except ImportError:
    PreparedMultipoles = None
    SPLINE_BACKEND = "scipy (ResponseCache.moments_at)"

try:
    from lighthit.experimental.ballistic_fast import vectorised_ballistic_fast as _fast_ballistic
    BALLISTIC_BACKEND = "numba (ballistic_fast.vectorised_ballistic_fast)"
except ImportError:
    _fast_ballistic = vectorised_ballistic
    BALLISTIC_BACKEND = "numpy (vectorised_ballistic)"

ORDERS = {"first": 0, "two_or_more": 1}


def peak_megabytes():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def errors_against(value, reference):
    """Per-module relative error, plus the summary statistics that are honest."""
    error = np.abs(value - reference) / np.maximum(np.abs(reference), 1e-300)
    return {"per_receiver": error.tolist(),
            "median": float(np.median(error)),
            "max": float(error.max()),
            "on_the_summed_signal": float(
                np.abs(value.sum() - reference.sum()) / max(abs(reference.sum()), 1e-300))}


def build_cache(medium, arguments, omega, low, high):
    # The output degree only ever has to reach L_q: this route never asks the
    # cache for a higher multipole, and building to k_max * r_max instead would
    # cost two orders of magnitude for coefficients that are then discarded.
    # The scattering degree L is untouched, and the equivalence of the retained
    # degrees is checked separately.
    settings = SolverSettings(arguments.scattering_degree, arguments.angular_degree,
                              arguments.k_max, 0.04, 10)
    grid = CacheGrid.geometric(low, high, arguments.radii, omega)
    began = perf_counter()
    cache = ResponseCache.build(medium, settings, grid)
    return cache, perf_counter() - began


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--event", type=int, default=5)
    parser.add_argument("--held-out", type=int, default=6)
    parser.add_argument("--output", default=".build/review/shower-axial")
    parser.add_argument("--angular-degree", type=int, default=32,
                        help="L_q: the source and output multipole degree")
    parser.add_argument("--scattering-degree", type=int, default=24,
                        help="L: the scattering operator degree, not L_q")
    parser.add_argument("--azimuthal-degree", type=int, default=4, help="M")
    parser.add_argument("--cell-m", type=float, default=0.06)
    parser.add_argument("--transverse-m", type=float, default=None)
    parser.add_argument("--frequencies", type=int, default=41)
    parser.add_argument("--omega-max", type=float, default=0.5)
    parser.add_argument("--k-max", type=float, default=8.0)
    parser.add_argument("--radii", type=int, default=160)
    parser.add_argument("--controls", type=int, default=6)
    parser.add_argument("--bin-ns", type=float, default=5.0)
    parser.add_argument("--bins", type=int, default=60)
    parser.add_argument("--early-ns", type=float, default=20.0)
    parser.add_argument("--refine", action="store_true",
                        help="also run the one-at-a-time refinement checks")
    parser.add_argument("--cache", default=None)
    arguments = parser.parse_args()
    out = Path(arguments.output)
    out.mkdir(parents=True, exist_ok=True)

    medium = synthetic_medium()
    omega = np.linspace(0.0, arguments.omega_max, arguments.frequencies)
    report = {"environment": {"python": sys.version, "platform": platform.platform(),
                              "numpy": np.__version__,
                              "cpu_count": __import__("os").cpu_count()},
              "settings": vars(arguments),
              "omega_per_ns": omega.tolist(),
              "medium": {"absorption_per_m": medium.absorption_per_m,
                         "scattering_per_m": medium.scattering_per_m,
                         "g": medium.g, "group_index": medium.group_index,
                         "speed_m_per_ns": medium.speed_m_per_ns,
                         "provenance": medium.provenance},
              "note": ("Control errors are per module. The median is a median "
                       "over the control modules only, never over 576.")}

    began = perf_counter()
    raw = load_event(arguments.input, arguments.event, SourceContract())
    read_seconds = perf_counter() - began
    pose = np.array([20.0, 15.0, -20.0])
    elements = raw.moved(translation=pose)
    report["source"] = {**elements.summary(), "read_seconds": read_seconds,
                        "pose_translation_m": pose.tolist()}

    receivers = array_positions()
    distance = np.linalg.norm(receivers - elements.centroid_m, axis=1)
    order = np.argsort(distance)
    picks = order[np.linspace(0, len(order) - 1, arguments.controls).astype(int)]
    control = receivers[picks]
    report["array"] = {"receivers": len(receivers),
                       "geometry": "2 clusters x 8 strings x 36 modules, idealised",
                       "distance_range_m": [float(distance.min()), float(distance.max())],
                       "control_index": picks.tolist(),
                       "control_distance_m": distance[picks].tolist()}

    # The cache has to cover every pose this run will ask for, including the
    # one chosen after the tables are built and the held-out event, or the
    # second half of the run would need a second cache and the claim that one
    # transport table serves any pose would be untested.
    rotation = Rotation.from_rotvec([0.3, -0.7, 0.4]).as_matrix()
    moved = raw.moved(rotation=rotation, translation=np.array([-30.0, 40.0, 10.0]),
                      delay_ns=25.0)
    poses = [elements, moved]
    if arguments.held_out is not None:
        held = load_event(arguments.input, arguments.held_out,
                          SourceContract()).moved(translation=pose)
        poses.append(held)
    spans = [np.linalg.norm(receivers - event.centroid_m, axis=1) for event in poses]
    reach = 1.5 * max(event.extent_m for event in poses)
    low = max(3.0, (min(span.min() for span in spans) - reach) * 0.95)
    high = (max(span.max() for span in spans) + reach) * 1.05
    if arguments.cache and Path(arguments.cache).exists():
        began = perf_counter()
        cache = ResponseCache.load(arguments.cache)
        report["cache"] = {"loaded_from": arguments.cache,
                           "load_seconds": perf_counter() - began}
    else:
        cache, seconds = build_cache(medium, arguments, omega, low, high)
        report["cache"] = {"build_seconds": seconds, "stages": cache.timings_s}
        if arguments.cache:
            began = perf_counter()
            cache.save(arguments.cache)
            report["cache"]["save_seconds"] = perf_counter() - began
    report["cache"].update({"radii": len(cache.grid.radii_m),
                            "degree": int(cache.degree),
                            "frequencies": len(cache.grid.omega_per_ns),
                            "megabytes": cache.moments.nbytes / 2 ** 20,
                            "radius_range_m": list(cache.radius_range_m)})
    report["cache"]["spline_backend"] = SPLINE_BACKEND
    omega = cache.grid.omega_per_ns
    cache_fast = PreparedMultipoles.of(cache) if PreparedMultipoles is not None else cache
    kernels = {name: KernelChannels.of(cache_fast, index, arguments.angular_degree)
               for name, index in ORDERS.items()}

    frame = AxisFrame.of(elements)

    def project(event, axis_frame, degree=None, azimuthal=None, cell=None,
                transverse=None, deposit="linear", element_order=2):
        return AxialSource.of(event, degree or arguments.angular_degree, omega,
                              azimuthal_degree=(arguments.azimuthal_degree
                                                if azimuthal is None else azimuthal),
                              cell_m=arguments.cell_m if cell is None else cell,
                              transverse_m=(arguments.transverse_m if transverse is None
                                            else transverse),
                              deposit=deposit, element_order=element_order,
                              frame=axis_frame)

    # -- the run itself ------------------------------------------------------
    began = perf_counter()
    source = project(elements, frame)
    compile_seconds = perf_counter() - began
    report["compact_source"] = {**source.summary, "seconds": compile_seconds,
                                "megabytes": source.channels.nbytes / 2 ** 20}
    print(f"source: {source.summary['cells']} cells x "
          f"{source.summary['channels']} channels x {len(omega)} frequencies "
          f"in {compile_seconds:.1f} s", flush=True)

    spectra, timings = {}, {}
    for name, kernel in kernels.items():
        began = perf_counter()
        spectra[name] = axial_response(kernel, source, receivers)
        timings[name] = perf_counter() - began
        print(f"apply {name} to {len(receivers)} modules: {timings[name]:.1f} s",
              flush=True)

    began = perf_counter()
    reference = {name: direct_response(kernel, elements, control)
                 for name, kernel in kernels.items()}
    reference_seconds = perf_counter() - began
    report["control"] = {
        "element_sum_seconds": reference_seconds,
        "element_sum_seconds_per_module": reference_seconds / len(control) / 2,
        "charge": {name: errors_against(spectra[name][0, picks].real,
                                        reference[name][0].real)
                   for name in kernels},
        "spectrum_at": {}}
    for index in (0, len(omega) // 4, len(omega) // 2, len(omega) - 1):
        report["control"]["spectrum_at"][f"{omega[index]:.3f}"] = {
            name: errors_against(spectra[name][index, picks], reference[name][index])
            for name in kernels}

    # -- the order no representation touches ---------------------------------
    began = perf_counter()
    ballistic = _fast_ballistic(elements, receivers, medium, elements.cone_cosine)
    ballistic_seconds = perf_counter() - began
    report["ballistic"] = {
        "seconds": ballistic_seconds, "total": float(ballistic.sum()),
        "backend": BALLISTIC_BACKEND,
        "note": "Closed cone formula per element; exact, and no compression applies."}

    # -- time bins, from the whole window ------------------------------------
    edges = np.arange(arguments.bins + 1) * arguments.bin_ns
    began = perf_counter()
    bins = {name: inverse_bins(omega, spectra[name], edges) for name in kernels}
    readout_seconds = perf_counter() - began
    control_bins = {name: inverse_bins(omega, reference[name], edges) for name in kernels}
    early = edges[:-1] < arguments.early_ns
    report["time_bins"] = {
        "edges_ns": edges.tolist(), "seconds": readout_seconds,
        "sigma_ns": 0.0,
        "note": ("No detector smearing. Negative bins are kept as computed: "
                 "they are the diagnostic that says the frequency window is "
                 "narrow, and clipping them would hide it."),
        "control": {}}
    for name in kernels:
        got, want = bins[name][picks], control_bins[name]
        scale = np.abs(want).max(axis=1)
        report["time_bins"]["control"][name] = {
            "per_receiver_max_bin_error": (np.abs(got - want).max(axis=1) / scale).tolist(),
            "per_receiver_early_window_error": [
                float(abs(got[i, early].sum() - want[i, early].sum())
                      / max(abs(want[i, early].sum()), 1e-300))
                for i in range(len(picks))],
            "negative_bin_fraction_of_mass": float(
                np.abs(got[got < 0]).sum() / max(np.abs(got).sum(), 1e-300))}
        np.save(out / f"bins-{name}.npy", bins[name])
    np.save(out / "edges-ns.npy", edges)
    for name in kernels:
        np.save(out / f"spectrum-{name}.npy", spectra[name])
    np.save(out / "ballistic.npy", ballistic)
    np.save(out / "receivers-m.npy", receivers)

    report["budget_seconds"] = {
        "read_and_contract_source": read_seconds,
        "compile_compact_source": compile_seconds,
        "apply_first": timings["first"], "apply_two_or_more": timings["two_or_more"],
        "ballistic_all_modules": ballistic_seconds,
        "readout": readout_seconds,
        "total_per_event_after_cache": (read_seconds + compile_seconds
                                        + sum(timings.values()) + ballistic_seconds
                                        + readout_seconds),
        "note": ("The cache is built once for a medium and is not in the "
                 "per-event budget; its own cost is reported under 'cache'.")}

    # -- a pose chosen after the tables were built ---------------------------
    # One compact source at a time: at 41 frequencies each is most of a
    # gigabyte, and holding two of them is what this container cannot do.
    del source
    began = perf_counter()
    moved_source = project(moved, AxisFrame.of(moved))
    moved_compile = perf_counter() - began
    moved_spectra, moved_timings = {}, {}
    for name, kernel in kernels.items():
        began = perf_counter()
        moved_spectra[name] = axial_response(kernel, moved_source, receivers)
        moved_timings[name] = perf_counter() - began
    moved_distance = np.linalg.norm(receivers - moved.centroid_m, axis=1)
    moved_order = np.argsort(moved_distance)
    moved_picks = moved_order[np.linspace(0, len(moved_order) - 1,
                                          arguments.controls).astype(int)]
    moved_reference = {name: direct_response(kernel, moved, receivers[moved_picks])
                       for name, kernel in kernels.items()}
    report["new_pose"] = {
        "rotation": rotation.tolist(), "translation_m": [-30.0, 40.0, 10.0],
        "delay_ns": 25.0, "same_cache": True,
        "cells": moved_source.summary["cells"],
        "compile_seconds": moved_compile, "apply_seconds": moved_timings,
        "control_distance_m": moved_distance[moved_picks].tolist(),
        "charge": {name: errors_against(moved_spectra[name][0, moved_picks].real,
                                        moved_reference[name][0].real)
                   for name in kernels}}
    print(f"new pose: {moved_source.summary['cells']} cells, "
          f"compile {moved_compile:.1f} s", flush=True)

    # -- the held-out event, settings frozen ---------------------------------
    del moved_source
    if arguments.held_out is not None:
        other = held
        began = perf_counter()
        other_source = project(other, AxisFrame.of(other))
        other_compile = perf_counter() - began
        other_distance = np.linalg.norm(receivers - other.centroid_m, axis=1)
        other_picks = np.argsort(other_distance)[
            np.linspace(0, len(other_distance) - 1, arguments.controls).astype(int)]
        other_spectra = {name: axial_response(kernel, other_source,
                                              receivers[other_picks])
                         for name, kernel in kernels.items()}
        other_reference = {name: direct_response(kernel, other, receivers[other_picks])
                           for name, kernel in kernels.items()}
        report["held_out_event"] = {
            "event": arguments.held_out, "settings": "frozen from the run above",
            "elements": len(other), "cells": other_source.summary["cells"],
            "compile_seconds": other_compile,
            "control_distance_m": other_distance[other_picks].tolist(),
            "charge": {name: errors_against(other_spectra[name][0].real,
                                            other_reference[name][0].real)
                       for name in kernels}}
        print(f"held-out event {arguments.held_out}: "
              f"{other_source.summary['cells']} cells", flush=True)

    # -- one-at-a-time refinement -------------------------------------------
    if arguments.held_out is not None:
        del other_source
    if arguments.refine:
        # The refinement asks which knob the error answers to, which a handful
        # of frequencies across the window settles as well as all of them, and
        # a hundred times cheaper.
        sample = sorted({0, len(omega) // 4, len(omega) // 2, len(omega) - 1})
        few = omega[sample]
        few_kernels = {name: KernelChannels.of(cache_fast, index, arguments.angular_degree,
                                               frequencies=few)
                       for name, index in ORDERS.items()}
        few_reference = {name: direct_response(kernel, elements, control)
                         for name, kernel in few_kernels.items()}
        refinement = {"frequencies": few.tolist(), "rows": []}
        base = dict(azimuthal=arguments.azimuthal_degree, cell=arguments.cell_m,
                    deposit="linear", element_order=2)
        variants = ([{**base, "azimuthal": m} for m in (2, 8)]
                    + [{**base, "cell": c} for c in (0.12, 0.03)]
                    + [{**base, "transverse": t} for t in (0.12, 0.03)]
                    + [{**base, "element_order": 4}, {**base, "deposit": "nearest"}])
        variants = [base] + variants
        for variant in variants:
            began = perf_counter()
            probe = AxialSource.of(
                elements, arguments.angular_degree, few,
                azimuthal_degree=variant.get("azimuthal", arguments.azimuthal_degree),
                cell_m=variant.get("cell", arguments.cell_m),
                transverse_m=variant.get("transverse", arguments.transverse_m),
                deposit=variant.get("deposit", "linear"),
                element_order=variant.get("element_order", 2), frame=frame)
            seconds = perf_counter() - began
            value = {name: axial_response(kernel, probe, control)
                     for name, kernel in few_kernels.items()}
            row = {"variant": dict(variant), "cells": probe.summary["cells"],
                   "channels": probe.summary["channels"], "compile_seconds": seconds,
                   "at_frequency": {}}
            for index, w in enumerate(few):
                row["at_frequency"][f"{w:.3f}"] = {
                    name: errors_against(value[name][index], few_reference[name][index])
                    for name in few_kernels}
            refinement["rows"].append(row)
            print(f"refine {variant}: cells {probe.summary['cells']}, >=2 median at "
                  f"w={few[-1]:.2f} "
                  f"{row['at_frequency'][f'{few[-1]:.3f}']['two_or_more']['median']:.2e}",
                  flush=True)
        report["refinement"] = refinement

    report["peak_megabytes"] = peak_megabytes()
    (out / "shower-axial.json").write_text(json.dumps(report, default=serial, indent=2),
                                           encoding="utf-8")
    print(f"written: {out / 'shower-axial.json'}")


if __name__ == "__main__":
    main()
