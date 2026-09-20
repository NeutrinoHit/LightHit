"""Every method that produces light transport, on one real event, side by
side -- except ``direct_response``, kept out because it is the slowest,
most honest method and the whole point of this script is to compare the
methods people actually run.

Four route families are computed on the same event, the same array, the same
cache, and the same frequency grid:

* the ballistic (order-0, unscattered) term, exact and closed-form,
  identical for every method below because none of them touch it;
* the moment route (@sec-event-moments): block-and-fit, applied with
  ``evaluate_moments``;
* the axial/needle route (@sec-axial-source): both its conservative sparse
  3-D lattice and a pure 1-D harmonic source along the shower axis;
* method 7, the effective-segments reduction
  (``lighthit.experimental.effective_segments``): the event refit to a
  chosen number of straight pieces, applied with ``full_response``.

``direct_response`` -- the element-by-element sum -- is still used, but only
at a handful of control modules, to score the compact routes above; it is
never run over the full array, which is the expense the other three exist
to avoid. Fast paths (``spline_fast.PreparedMultipoles``,
``ballistic_fast.vectorised_ballistic_fast``) are used wherever available,
same as every other production script in this repository.

Run:
  python scripts/run_method_survey.py --input /path/sim_e_100GeV_10.h5 \
      --event 5 --output .build/review/method-survey
"""
from pathlib import Path
import argparse
import json
import platform
import sys
from time import perf_counter

import numpy as np

from lighthit import SolverSettings, synthetic_medium
from lighthit.cache import CacheGrid, ResponseCache
from lighthit.experimental.axial_source import AxisFrame, AxialSource, axial_response
from lighthit.experimental.effective_segments import fit_effective_segments, full_response
from lighthit.experimental.event_moments import (BlockPartition, KernelChannels,
                                                 compile_joint_moments,
                                                 direct_response, evaluate_moments)
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

try:
    from lighthit.experimental.axial_fast import (PreparedAxialKernel,
                                                   compile_axial_source_fast)
    AXIAL_BACKEND = "numba (axial_fast; both orders in one pass)"
except ImportError:
    PreparedAxialKernel = None
    compile_axial_source_fast = None
    AXIAL_BACKEND = "numpy (AxialSource.of + axial_response)"

ORDERS = {"first": 0, "two_or_more": 1}


def errors_against(value, reference):
    difference = value - reference
    error = np.abs(difference) / np.maximum(np.abs(reference), 1e-300)
    zero = np.abs(difference[0]) / np.maximum(np.abs(reference[0]), 1e-300)
    return {"median": float(np.median(error)),
            "p95": float(np.quantile(error, 0.95)),
            "max": float(error.max()),
            "relative_l2": float(np.linalg.norm(difference)
                                 / max(np.linalg.norm(reference), 1e-300)),
            "zero_frequency": {"median": float(np.median(zero)),
                               "max": float(zero.max())}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="stored G4 steps (HDF5)")
    parser.add_argument("--event", type=int, default=5)
    parser.add_argument("--output", default=".build/review/method-survey")
    parser.add_argument("--angular-degree", type=int, default=32)
    parser.add_argument("--frequencies", type=int, default=41)
    parser.add_argument("--omega-max", type=float, default=0.5)
    parser.add_argument("--k-max", type=float, default=8.0)
    parser.add_argument("--radii", type=int, default=160)
    parser.add_argument("--controls", type=int, default=6)
    parser.add_argument("--blocks", type=int, default=8,
                        help="moment route: block count (must be a power of two)")
    parser.add_argument("--spatial-degree", type=int, default=3,
                        help="moment route: block polynomial degree")
    parser.add_argument("--cell-m", type=float, default=0.08,
                        help="3-D axial route: cell size")
    parser.add_argument("--axial-1d-bins", type=int, default=64,
                        help="pure 1-D harmonic route: longitudinal bins")
    parser.add_argument("--azimuthal-degree", type=int, default=4,
                        help="3-D axial route: M")
    parser.add_argument("--axial-1d-azimuthal-degree", type=int, default=8,
                        help="pure 1-D harmonic route: M")
    parser.add_argument("--segments", type=int, default=16,
                        help="method 7: number of effective segments K")
    parser.add_argument("--cache", default=None)
    arguments = parser.parse_args()
    out = Path(arguments.output)
    out.mkdir(parents=True, exist_ok=True)

    medium = synthetic_medium()
    omega = np.linspace(0.0, arguments.omega_max, arguments.frequencies)
    report = {"environment": {"python": sys.version, "platform": platform.platform(),
                              "numpy": np.__version__},
              "settings": vars(arguments),
              "note": ("Every method on one event and the full array, side by side. "
                       "direct_response -- the slowest, most honest method -- is "
                       "used only at the control modules below, to score the others, "
                       "and is deliberately never run over the full array here.")}

    began = perf_counter()
    elements = load_event(arguments.input, arguments.event,
                          SourceContract()).moved(translation=np.array([20.0, 15.0, -20.0]))
    report["source"] = {**elements.summary(), "read_seconds": perf_counter() - began}

    receivers = array_positions()
    distance = np.linalg.norm(receivers - elements.centroid_m, axis=1)
    order = np.argsort(distance)
    picks = order[np.linspace(0, len(order) - 1, arguments.controls).astype(int)]
    control = receivers[picks]
    report["array"] = {"receivers": len(receivers),
                       "geometry": "2 clusters x 8 strings x 36 modules, idealised",
                       "control_index": picks.tolist(),
                       "control_distance_m": distance[picks].tolist()}
    print(f"event: {len(elements)} elements, control at "
          + " ".join(f"{d:.0f}" for d in distance[picks]) + " m", flush=True)

    reach = 1.5 * elements.extent_m
    low = max(3.0, (distance.min() - reach) * 0.95)
    high = (distance.max() + reach) * 1.05
    if arguments.cache and Path(arguments.cache).exists():
        began = perf_counter()
        cache = ResponseCache.load(arguments.cache)
        report["cache"] = {"loaded_from": arguments.cache,
                           "load_seconds": perf_counter() - began}
    else:
        began = perf_counter()
        settings = SolverSettings(24, arguments.angular_degree, arguments.k_max, 0.04, 10)
        cache = ResponseCache.build(medium, settings,
                                    CacheGrid.geometric(low, high, arguments.radii, omega))
        report["cache"] = {"build_seconds": perf_counter() - began, "stages": cache.timings_s}
        if arguments.cache:
            cache.save(arguments.cache)
    report["cache"].update({"radii": len(cache.grid.radii_m), "degree": int(cache.degree),
                            "megabytes": cache.moments.nbytes / 2 ** 20,
                            "radius_range_m": list(cache.radius_range_m),
                            "spline_backend": SPLINE_BACKEND})
    omega = cache.grid.omega_per_ns
    cache_fast = PreparedMultipoles.of(cache) if PreparedMultipoles is not None else cache
    kernels = {name: KernelChannels.of(cache_fast, index, arguments.angular_degree,
                                       frequencies=omega)
               for name, index in ORDERS.items()}

    # -- the one term every method shares, computed once ----------------------
    began = perf_counter()
    ballistic_control = _fast_ballistic(elements, control, medium, elements.cone_cosine)
    ballistic_all = _fast_ballistic(elements, receivers, medium, elements.cone_cosine)
    ballistic_seconds = perf_counter() - began
    report["ballistic"] = {"seconds": ballistic_seconds, "backend": BALLISTIC_BACKEND,
                           "all_receivers_total": float(ballistic_all.sum()),
                           "note": ("Closed cone formula per element; exact, and shared "
                                    "unchanged by every method below -- none of them "
                                    "touch this order.")}
    print(f"ballistic ({BALLISTIC_BACKEND}): {ballistic_seconds:.2f} s for "
          f"{len(receivers)} receivers", flush=True)

    # -- the honest reference, at control modules only -------------------------
    reference = {}
    for name, kernel in kernels.items():
        began = perf_counter()
        reference[name] = direct_response(kernel, elements, control)
        report.setdefault("direct_reference", {})[name] = {
            "seconds": perf_counter() - began}
    print("direct reference (control only): "
          + " ".join(f"{name} {report['direct_reference'][name]['seconds']:.1f}s"
                     for name in kernels), flush=True)

    methods = {}

    # -- moment route -----------------------------------------------------------
    began = perf_counter()
    partition = BlockPartition.split(elements, arguments.blocks)
    moments, powers = compile_joint_moments(elements, partition, arguments.angular_degree,
                                            arguments.spatial_degree, omega)
    row = {"compile_seconds": perf_counter() - began, "blocks": arguments.blocks,
          "spatial_degree": arguments.spatial_degree}
    scattered_total = 0.0
    for name, kernel in kernels.items():
        began = perf_counter()
        value_control = evaluate_moments(kernel, moments, powers, partition, control)
        apply_control = perf_counter() - began
        began = perf_counter()
        value_all = evaluate_moments(kernel, moments, powers, partition, receivers)
        apply_all = perf_counter() - began
        row[name] = {"apply_control_seconds": apply_control, "apply_all_seconds": apply_all,
                    "error": errors_against(value_control, reference[name]),
                    "all_receivers_total": float(value_all[0].real.sum())}
        scattered_total += row[name]["all_receivers_total"]
    row["scattered_all_receivers_total"] = scattered_total
    methods["moments"] = row
    print(f"moments: first {row['first']['error']['median']:.2e}, "
          f">=2 {row['two_or_more']['error']['median']:.2e} median error, "
          f"apply {row['first']['apply_all_seconds'] + row['two_or_more']['apply_all_seconds']:.1f} s",
          flush=True)

    # -- harmonic/axial routes ----------------------------------------------------
    # The 3-D sparse lattice is the conservative representation already used by
    # the axial study.  The pure 1-D variant is the useful successor to method 7:
    # it still collapses the shower onto its axis, but retains the true angular
    # harmonics and emission phases instead of replacing every slab by one cone.
    frame = AxisFrame.of(elements)
    prepared_axial = (PreparedAxialKernel.from_cache(cache, degree=arguments.angular_degree)
                      if PreparedAxialKernel is not None else None)
    axial_factory = (compile_axial_source_fast
                     if compile_axial_source_fast is not None else AxialSource.of)

    def survey_axial(source, compile_seconds):
        row = {"backend": AXIAL_BACKEND, "compile_seconds": compile_seconds,
               "cells": source.summary["cells"],
               "source_megabytes": source.channels.nbytes / 2 ** 20}
        if prepared_axial is not None:
            began = perf_counter()
            control_both = prepared_axial.apply(
                source, control, source_omega_per_ns=omega)
            row["apply_control_seconds"] = perf_counter() - began
            began = perf_counter()
            all_both = prepared_axial.apply(
                source, receivers, source_omega_per_ns=omega)
            row["apply_all_seconds"] = perf_counter() - began
            row["apply_note"] = "both scattered orders share this one pass"
            values = {name: (control_both[:, :, index], all_both[:, :, index])
                      for index, name in enumerate(kernels)}
        else:
            values = {}
            control_seconds = all_seconds = 0.0
            for name, kernel in kernels.items():
                began = perf_counter()
                value_control = axial_response(kernel, source, control)
                control_seconds += perf_counter() - began
                began = perf_counter()
                value_all = axial_response(kernel, source, receivers)
                all_seconds += perf_counter() - began
                values[name] = (value_control, value_all)
            row["apply_control_seconds"] = control_seconds
            row["apply_all_seconds"] = all_seconds
            row["apply_note"] = "sum of the two separate scattered-order passes"
        scattered_total = 0.0
        for name, (value_control, value_all) in values.items():
            row[name] = {"error": errors_against(value_control, reference[name]),
                         "all_receivers_total": float(value_all[0].real.sum())}
            scattered_total += row[name]["all_receivers_total"]
        row["scattered_all_receivers_total"] = scattered_total
        return row

    began = perf_counter()
    source = axial_factory(elements, arguments.angular_degree, omega,
                           azimuthal_degree=arguments.azimuthal_degree,
                           cell_m=arguments.cell_m, deposit="linear",
                           element_order=2, frame=frame)
    row = survey_axial(source, perf_counter() - began)
    methods["axial"] = row
    print(f"axial 3-D: first {row['first']['error']['median']:.2e}, "
          f">=2 {row['two_or_more']['error']['median']:.2e} median error, "
          f"apply {row['apply_all_seconds']:.2f} s ({AXIAL_BACKEND})", flush=True)
    del source

    began = perf_counter()
    source = axial_factory(elements, arguments.angular_degree, omega,
                           azimuthal_degree=arguments.axial_1d_azimuthal_degree,
                           cell_m=None, bins=arguments.axial_1d_bins,
                           transverse_m=None, deposit="linear",
                           element_order=2, frame=frame)
    row = survey_axial(source, perf_counter() - began)
    row["longitudinal_bins_requested"] = arguments.axial_1d_bins
    row["azimuthal_degree"] = arguments.axial_1d_azimuthal_degree
    methods["axial_1d"] = row
    print(f"axial 1-D ({arguments.axial_1d_bins} bins, "
          f"M={arguments.axial_1d_azimuthal_degree}): first "
          f"{row['first']['error']['median']:.2e}, >=2 "
          f"{row['two_or_more']['error']['median']:.2e} median error, "
          f"apply {row['apply_all_seconds']:.3f} s ({AXIAL_BACKEND})", flush=True)
    del source

    # -- method 7: effective segments ----------------------------------------------
    began = perf_counter()
    effective = fit_effective_segments(elements, arguments.segments)
    fit_seconds = perf_counter() - began
    began = perf_counter()
    full_control = full_response(cache_fast, elements, medium, effective, control,
                                 longitudinal_order=arguments.angular_degree,
                                 ballistic=ballistic_control)
    apply_control_seconds = perf_counter() - began
    began = perf_counter()
    full_all = full_response(cache_fast, elements, medium, effective, receivers,
                             longitudinal_order=arguments.angular_degree,
                             ballistic=ballistic_all)
    apply_all_seconds = perf_counter() - began
    ballistic_match = float(np.max(np.abs(full_control[:, :, 0] - ballistic_control[None, :])))
    row = {"segments": arguments.segments, "elements": len(elements),
          "empty_bins": effective.empty_bins,
          "fit_seconds": fit_seconds,
          "first": {"apply_control_seconds": apply_control_seconds,
                    "apply_all_seconds": apply_all_seconds,
                    "error": errors_against(full_control[:, :, 1], reference["first"]),
                    "all_receivers_total": float(full_all[0, :, 1].real.sum())},
          "two_or_more": {"error": errors_against(full_control[:, :, 2],
                                                  reference["two_or_more"]),
                          "all_receivers_total": float(full_all[0, :, 2].real.sum())},
          "ballistic_max_abs_deviation": ballistic_match,
          "ballistic_note": ("0 by construction: full_response never takes order 0 "
                             "from the segment fit, only the closed form above.")}
    row["scattered_all_receivers_total"] = (row["first"]["all_receivers_total"]
                                            + row["two_or_more"]["all_receivers_total"])
    methods["effective_segments"] = row
    print(f"effective_segments (K={arguments.segments}): first "
          f"{row['first']['error']['median']:.2e}, >=2 "
          f"{row['two_or_more']['error']['median']:.2e} median error, ballistic "
          f"deviation {ballistic_match:.1e}, apply {apply_all_seconds:.2f} s", flush=True)

    report["methods"] = methods
    report["excluded"] = {
        "direct_response": ("the element-by-element sum -- the most honest and "
                            "slowest method. Used above only to score the compact "
                            "methods at the control modules; deliberately not run "
                            "over the full array, which is the cost the methods "
                            "above exist to avoid.")}
    (out / "method-survey.json").write_text(json.dumps(report, default=serial, indent=2),
                                            encoding="utf-8")
    print(f"written: {out / 'method-survey.json'}")


if __name__ == "__main__":
    main()
