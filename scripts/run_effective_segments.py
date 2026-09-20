"""Method 7 on a real stored shower: fit it to K straight segments, K adjustable.

Run:
  python scripts/run_effective_segments.py --input /path/sim_e_100GeV_10.h5 \
      --event 5 --segments 1 2 4 8 16 32 64 --output .build/review/effective-segments

What this measures, in the order it measures it:

1. how ``fit_effective_segments`` reduces the real, stored shower to each
   requested segment count K: pieces built, empty bins skipped, photon share
   per piece;
2. orders 1 and >=2 from ``effective_segment_response`` against the same
   element-by-element ``direct_response`` reference ``run_shower_moments.py``
   uses, at a handful of control receivers spread across the array -- this
   tests whether the error shrinks as K grows; axial refinement alone does
   not guarantee that for a shower with broad directions at every depth;
3. the ballistic order from ``full_response``, which never comes from the
   segment fit at any K: it is ``ballistic_fast``'s exact closed form on the
   real elements, so this section checks that it is identical across K, not
   that it converges;
4. wall-clock cost of the fit and of applying it, at every K, both at the
   control receivers and across the full detector array, so the K vs.
   accuracy vs. cost trade-off is visible in one table.

Each fitted segment is evaluated exactly, but replacing all elements in one
axial slab by that segment is an approximation. In particular, the mean
direction and mean cone cosine do not preserve the slab's higher angular
moments and need not preserve even its dipole. See ``effective_segments.py``.
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
from lighthit.experimental.event_moments import KernelChannels, direct_response
from lighthit.experimental.effective_segments import (fit_effective_segments,
                                                       effective_segment_response,
                                                       full_response)
from lighthit.experimental.g4_source import SourceContract, load_event

try:
    from lighthit.experimental.ballistic_fast import vectorised_ballistic_fast as vectorised_ballistic
    BALLISTIC_BACKEND = "numba (ballistic_fast)"
except ImportError:
    from run_shower_moments import vectorised_ballistic
    BALLISTIC_BACKEND = "numpy (run_shower_moments.vectorised_ballistic)"


def array_positions(clusters=2, strings=8, modules=36, radius_m=60.0,
                    spacing_m=15.0, pitch_m=110.0):
    """The same idealised two-cluster array run_shower_moments.py surveys."""
    centres = np.array([[pitch_m * index, 0.0, 0.0] for index in range(clusters)])
    offsets = np.zeros((strings, 3))
    azimuth = np.arange(strings - 1) * 2 * np.pi / (strings - 1)
    offsets[1:, :2] = radius_m * np.column_stack((np.cos(azimuth), np.sin(azimuth)))
    heights = (np.arange(modules) - (modules - 1) / 2) * spacing_m
    grid = (centres[:, None, None, :] + offsets[None, :, None, :]
            + np.stack((np.zeros_like(heights), np.zeros_like(heights), heights),
                       axis=-1)[None, None, :, :])
    return grid.reshape(-1, 3)


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="stored G4 steps (HDF5)")
    parser.add_argument("--event", type=int, default=5)
    parser.add_argument("--output", default=".build/review/effective-segments")
    parser.add_argument("--segments", type=int, nargs="*",
                        default=[1, 2, 4, 8, 16, 32, 64])
    parser.add_argument("--angular-degree", type=int, default=64)
    parser.add_argument("--longitudinal-order", type=int, default=16)
    parser.add_argument("--frequencies", type=float, nargs="*", default=[0.0])
    parser.add_argument("--k-max", type=float, default=2.0)
    parser.add_argument("--spatial-degree", type=int, default=480)
    parser.add_argument("--radii", type=int, default=60)
    parser.add_argument("--max-distance", type=float, default=None)
    parser.add_argument("--controls", type=int, default=6)
    parser.add_argument("--cache", default=None, help="reuse a saved cache")
    arguments = parser.parse_args()
    out = Path(arguments.output)
    out.mkdir(parents=True, exist_ok=True)
    medium = synthetic_medium()
    report = {"environment": {"python": sys.version, "platform": platform.platform(),
                              "numpy": np.__version__, "cpu_count": __import__("os").cpu_count(),
                              "ballistic_backend": BALLISTIC_BACKEND},
              "medium": {"absorption_per_m": medium.absorption_per_m,
                         "scattering_per_m": medium.scattering_per_m,
                         "g": medium.g, "group_index": medium.group_index,
                         "provenance": medium.provenance},
              "settings": vars(arguments)}

    began = perf_counter()
    elements = load_event(arguments.input, arguments.event, SourceContract())
    read_seconds = perf_counter() - began
    pose = np.array([20.0, 15.0, -20.0])
    elements = elements.moved(translation=pose)
    report["source"] = {**elements.summary(), "read_seconds": read_seconds,
                        "pose_translation_m": pose.tolist()}

    receivers = array_positions()
    distance = np.linalg.norm(receivers - elements.centroid_m, axis=1)
    if arguments.max_distance:
        keep = distance <= arguments.max_distance
        receivers, distance = receivers[keep], distance[keep]
    order = np.argsort(distance)
    picks = order[np.linspace(0, len(order) - 1, arguments.controls).astype(int)]
    control = receivers[picks]
    report["array"] = {"receivers": len(receivers),
                       "distance_range_m": [float(distance.min()), float(distance.max())],
                       "control_index": picks.tolist(),
                       "control_distance_m": distance[picks].tolist()}

    if arguments.cache and Path(arguments.cache).exists():
        began = perf_counter()
        cache = ResponseCache.load(arguments.cache)
        report["cache"] = {"loaded_from": arguments.cache,
                           "seconds": perf_counter() - began}
        if (len(cache.grid.radii_m) != arguments.radii
                or list(cache.grid.omega_per_ns) != list(arguments.frequencies)):
            report["cache"]["ignored_request"] = {
                "radii": arguments.radii, "frequencies": list(arguments.frequencies)}
            print("warning: the loaded cache does not match --radii/--frequencies",
                  flush=True)
    else:
        settings = SolverSettings(24, arguments.spatial_degree, arguments.k_max, 0.04, 10)
        reach = 1.5 * elements.extent_m
        grid = CacheGrid.geometric(max(3.0, (distance.min() - reach) * 0.95),
                                   (distance.max() + reach) * 1.05,
                                   arguments.radii, arguments.frequencies)
        began = perf_counter()
        cache = ResponseCache.build(medium, settings, grid)
        report["cache"] = {"build_seconds": perf_counter() - began,
                           "stages": cache.timings_s,
                           "megabytes": cache.moments.nbytes / 2 ** 20}
        if arguments.cache:
            cache.save(arguments.cache)
    report["cache"]["radius_range_m"] = list(cache.radius_range_m)
    report["cache"]["radii"] = len(cache.grid.radii_m)
    report["cache"]["degree"] = int(cache.degree)
    report["cache"]["frequencies"] = cache.grid.omega_per_ns.tolist()
    arguments.frequencies = cache.grid.omega_per_ns.tolist()

    # -- element-by-element reference at the control receivers --------------
    k1 = KernelChannels.of(cache, 0, arguments.angular_degree, frequencies=arguments.frequencies)
    k2 = KernelChannels.of(cache, 1, arguments.angular_degree, frequencies=arguments.frequencies)
    began = perf_counter()
    ref1 = direct_response(k1, elements, control)
    ref2 = direct_response(k2, elements, control)
    direct_seconds = perf_counter() - began
    scale1, scale2 = np.abs(ref1).max(), np.abs(ref2).max()
    report["direct_reference"] = {"seconds": direct_seconds,
                                  "note": "Element-by-element sum with the same kernel; "
                                          "the ground truth for orders 1 and >=2, never "
                                          "run over the full array here."}
    print(f"direct reference (control receivers): {direct_seconds:.1f} s", flush=True)

    # -- the exact ballistic term, computed once, independent of K ----------
    began = perf_counter()
    ballistic_control = vectorised_ballistic(elements, control, medium, elements.cone_cosine)
    ballistic_control_seconds = perf_counter() - began
    began = perf_counter()
    ballistic_all = vectorised_ballistic(elements, receivers, medium, elements.cone_cosine)
    ballistic_all_seconds = perf_counter() - began
    report["ballistic"] = {"control_seconds": ballistic_control_seconds,
                           "all_receivers_seconds": ballistic_all_seconds,
                           "control_value": ballistic_control.tolist(),
                           "all_receivers_total": float(ballistic_all.sum()),
                           "note": "Closed form on the real elements; full_response uses "
                                   "this at every K below, never the segment fit."}
    print(f"ballistic ({BALLISTIC_BACKEND}): {ballistic_all_seconds:.3f} s for "
          f"{len(receivers)} receivers", flush=True)

    # -- fit and apply at each requested segment count -----------------------
    scan = []
    for k in arguments.segments:
        began = perf_counter()
        effective = fit_effective_segments(elements, k)
        fit_seconds = perf_counter() - began

        began = perf_counter()
        approx_control = effective_segment_response(
            cache, effective, control, longitudinal_order=arguments.longitudinal_order)
        apply_control_seconds = perf_counter() - began

        began = perf_counter()
        approx_all = effective_segment_response(
            cache, effective, receivers, longitudinal_order=arguments.longitudinal_order)
        apply_all_seconds = perf_counter() - began

        error1 = np.max(np.abs(approx_control[:, :, 1] - ref1)) / max(scale1, 1e-300)
        error2 = np.max(np.abs(approx_control[:, :, 2] - ref2)) / max(scale2, 1e-300)

        full_all = approx_all.copy()
        full_all[:, :, 0] = ballistic_all[None, :]

        # Self-consistency check, not a convergence check: full_response's
        # ballistic column must equal the precomputed exact ballistic term
        # exactly, at every K, because it is never taken from the segment
        # fit (see effective_segments.py::full_response).
        full_control = full_response(cache, elements, medium, effective, control,
                                     longitudinal_order=arguments.longitudinal_order,
                                     ballistic=ballistic_control)
        ballistic_match = float(np.max(np.abs(full_control[:, :, 0] - ballistic_control[None, :])))

        row = {"segments_requested": k, "segments_built": len(effective),
              "empty_bins": effective.empty_bins,
              "photons_per_segment": effective.photons_per_segment.tolist(),
              "elements_per_segment": effective.elements_per_segment.tolist(),
              "fit_seconds": fit_seconds,
              "apply_control_seconds": apply_control_seconds,
              "apply_all_seconds": apply_all_seconds,
              "order1_relative_error": float(error1),
              "order2_relative_error": float(error2),
              "ballistic_max_abs_deviation": ballistic_match,
              "all_receivers_total_charge": float(full_all[0].real.sum())}
        scan.append(row)
        print(f"K={k:5d} pieces={len(effective):5d} empty={effective.empty_bins:4d} "
              f"fit={fit_seconds:.3f}s apply={apply_all_seconds:.3f}s "
              f"order1_err={error1:.3e} order2_err={error2:.3e} "
              f"ballistic_dev={ballistic_match:.1e}", flush=True)
    report["segment_scan"] = scan

    (out / "effective-segments.json").write_text(
        json.dumps(report, default=serial, indent=2), encoding="utf-8")
    print(f"written: {out / 'effective-segments.json'}")


if __name__ == "__main__":
    main()
