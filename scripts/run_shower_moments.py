"""Joint moments of a stored shower against an element-by-element reference.

Run:
  python scripts/run_shower_moments.py --input /path/sim_e_100GeV_10.h5 \
      --event 5 --output .build/review/shower-moments

What this measures, in the order it measures it:

1. the source contract applied to every stored step, with its own summary;
2. how far the angular channel index has to run before the *summed* event
   response stops moving, which is what sets the channel count;
3. the moment route against a direct sum over the same elements with the same
   kernel, scanning the polynomial degree and the number of spatial blocks;
4. the cost of the whole route at every receiver, separated into reading the
   input, compiling the moments and applying them;
5. the cost of the unscattered order, which no moment expansion touches, so
   that the floor of the total time is visible.

The reference is deliberately the *same kernel* summed element by element: that
isolates the error of the moment expansion from the cache's own truncations,
which are measured elsewhere. An independent physical control of the ``>=2``
component is a separate piece of work and is not claimed here.
"""
from pathlib import Path
import argparse
import json
import platform
import resource
import sys
from time import perf_counter

import numpy as np

from lighthit import SolverSettings, synthetic_medium
from lighthit.cache import CacheGrid, ResponseCache
from lighthit.experimental.event_moments import (BlockPartition, KernelChannels,
                                                 compile_joint_moments,
                                                 direct_response, evaluate_moments,
                                                 monomial_powers)
from lighthit.experimental.g4_source import SourceContract, load_event

try:
    from lighthit.experimental.spline_fast import PreparedMultipoles
    SPLINE_BACKEND = "numba (spline_fast.PreparedMultipoles)"
except ImportError:
    PreparedMultipoles = None
    SPLINE_BACKEND = "scipy (ResponseCache.moments_at)"

ORDERS = {"first": 0, "two_or_more": 1}


def peak_megabytes():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def array_positions(clusters=2, strings=8, modules=36, radius_m=60.0,
                    spacing_m=15.0, pitch_m=110.0):
    """An idealised two-cluster array, not surveyed coordinates."""
    centres = np.array([[pitch_m * index, 0.0, 0.0] for index in range(clusters)])
    offsets = np.zeros((strings, 3))
    azimuth = np.arange(strings - 1) * 2 * np.pi / (strings - 1)
    offsets[1:, :2] = radius_m * np.column_stack((np.cos(azimuth), np.sin(azimuth)))
    heights = (np.arange(modules) - (modules - 1) / 2) * spacing_m
    grid = (centres[:, None, None, :] + offsets[None, :, None, :]
            + np.stack((np.zeros_like(heights), np.zeros_like(heights), heights),
                       axis=-1)[None, None, :, :])
    return grid.reshape(-1, 3)


def vectorised_ballistic(elements, receivers, medium, cone_cosine):
    """The same unscattered sum, written once over all elements at a time.

    Kept as plain numpy on purpose: this is the numerical reference other
    modules and tests import by name (``ballistic_fast`` is checked against
    it, not the other way round). The fast path this script actually runs
    the "ballistic" report section with is ``_fast_ballistic`` below.
    """
    total = np.zeros(len(receivers))
    sine = np.sqrt(np.maximum(1 - cone_cosine ** 2, 0.0))
    for index, receiver in enumerate(receivers):
        relative = receiver[None, :] - elements.start_m
        along = np.einsum("ij,ij->i", relative, elements.direction)
        impact = np.sqrt(np.maximum(np.einsum("ij,ij->i", relative, relative)
                                    - along ** 2, 1e-300))
        root = along - impact * cone_cosine / sine
        distance = impact / sine
        lit = (root >= 0) & (root < elements.length_m)
        weight = np.where(lit, elements.photons / np.maximum(elements.length_m, 1e-300)
                          * np.exp(-medium.extinction_per_m * distance)
                          / (2 * np.pi * impact * sine), 0.0)
        total[index] = weight.sum()
    return total


try:
    from lighthit.experimental.ballistic_fast import vectorised_ballistic_fast as _fast_ballistic
    BALLISTIC_BACKEND = "numba (ballistic_fast.vectorised_ballistic_fast)"
except ImportError:
    _fast_ballistic = vectorised_ballistic
    BALLISTIC_BACKEND = "numpy (vectorised_ballistic)"


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.generic,)):
        return value.item()
    raise TypeError(type(value).__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="stored G4 steps (HDF5)")
    parser.add_argument("--event", type=int, default=5)
    parser.add_argument("--output", default=".build/review/shower-moments")
    parser.add_argument("--angular-degree", type=int, default=64)
    parser.add_argument("--reference-degree", type=int, default=128)
    parser.add_argument("--frequencies", type=float, nargs="*", default=[0.0])
    parser.add_argument("--k-max", type=float, default=2.0)
    parser.add_argument("--spatial-degree", type=int, default=480)
    parser.add_argument("--radii", type=int, default=60)
    parser.add_argument("--max-distance", type=float, default=None)
    parser.add_argument("--degrees", type=int, nargs="*", default=[0, 1, 2, 3])
    parser.add_argument("--blocks", type=int, nargs="*", default=[1, 2, 4, 8])
    parser.add_argument("--controls", type=int, default=6)
    parser.add_argument("--skip-ballistic", action="store_true")
    parser.add_argument("--cache", default=None, help="reuse a saved cache")
    arguments = parser.parse_args()
    out = Path(arguments.output)
    out.mkdir(parents=True, exist_ok=True)
    medium = synthetic_medium()
    report = {"environment": {"python": sys.version, "platform": platform.platform(),
                              "numpy": np.__version__, "cpu_count": __import__("os").cpu_count()},
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
        # A saved cache carries its own grid, which need not be the one asked
        # for on the command line. Say so rather than quietly reporting the
        # requested numbers next to somebody else's table.
        if (len(cache.grid.radii_m) != arguments.radii
                or list(cache.grid.omega_per_ns) != list(arguments.frequencies)):
            report["cache"]["ignored_request"] = {
                "radii": arguments.radii,
                "frequencies": list(arguments.frequencies)}
            print("warning: the loaded cache does not match --radii/--frequencies",
                  flush=True)
    else:
        settings = SolverSettings(24, arguments.spatial_degree, arguments.k_max, 0.04, 10)
        # Elements and stencil points sit up to one event extent away from the
        # centroid, so the table has to start closer in and end further out.
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

    # cache_fast wraps the same cache with a compiled Horner spline instead
    # of scipy's -- everything that only reads moments through KernelChannels
    # (angular convergence, the direct reference, the moment route) uses it;
    # cache.save/.moments/.timings_s/.grid below still go through the real
    # ResponseCache, which PreparedMultipoles wraps rather than replaces.
    cache_fast = PreparedMultipoles.of(cache) if PreparedMultipoles is not None else cache
    report["cache"]["spline_backend"] = SPLINE_BACKEND

    # -- how many angular channels the summed response needs ----------------
    convergence = {}
    for name, index in ORDERS.items():
        values = {}
        for degree in (8, 16, 32, arguments.angular_degree, arguments.reference_degree):
            if degree > cache.degree:
                continue
            kernel = KernelChannels.of(cache_fast, index, degree,
                                       frequencies=arguments.frequencies)
            began = perf_counter()
            values[degree] = {"value": direct_response(kernel, elements, control)[0].real,
                              "seconds": perf_counter() - began}
        reference = values[max(values)]["value"]
        convergence[name] = {
            "reference_degree": max(values),
            "value": {str(k): v["value"].tolist() for k, v in values.items()},
            "seconds": {str(k): v["seconds"] for k, v in values.items()},
            "relative_to_reference": {
                str(k): (np.abs(v["value"] - reference)
                         / np.maximum(np.abs(reference), 1e-300)).tolist()
                for k, v in values.items()}}
        print(f"angular convergence {name} done", flush=True)
    report["angular_convergence"] = convergence

    # -- the moment route against the element-by-element sum -----------------
    kernels = {name: KernelChannels.of(cache_fast, index, arguments.angular_degree,
                                       frequencies=arguments.frequencies)
               for name, index in ORDERS.items()}
    direct = {}
    for name, kernel in kernels.items():
        began = perf_counter()
        direct[name] = direct_response(kernel, elements, control)
        report.setdefault("direct_reference", {})[name] = {
            "seconds": perf_counter() - began,
            "value": direct[name].real.tolist()}
        print(f"direct reference {name}: {perf_counter() - began:.1f} s", flush=True)

    scan = []
    for blocks in arguments.blocks:
        partition = BlockPartition.split(elements, blocks)
        for degree in arguments.degrees:
            began = perf_counter()
            moments, powers = compile_joint_moments(elements, partition,
                                                    arguments.angular_degree, degree,
                                                    arguments.frequencies)
            build = perf_counter() - began
            row = {"blocks": blocks, "spatial_degree": degree,
                   "monomials": len(powers),
                   "channels": (arguments.angular_degree + 1) ** 2,
                   "coefficients": int(moments.size),
                   "megabytes": moments.nbytes / 2 ** 20,
                   "compile_seconds": build,
                   "block_half_sizes_m": partition.half_sizes_m.tolist(),
                   "orders": {}}
            for name, kernel in kernels.items():
                began = perf_counter()
                value = evaluate_moments(kernel, moments, powers, partition, control)
                apply_control = perf_counter() - began
                began = perf_counter()
                everything = evaluate_moments(kernel, moments, powers, partition, receivers)
                apply_all = perf_counter() - began
                error = (np.abs(value - direct[name])
                         / np.maximum(np.abs(direct[name]), 1e-300))[0]
                row["orders"][name] = {
                    "value": value[0].real.tolist(),
                    "relative_error": error.tolist(),
                    "median_relative_error": float(np.median(error)),
                    "apply_control_seconds": apply_control,
                    "apply_all_seconds": apply_all,
                    "all_receivers_total": float(everything[0].real.sum())}
            scan.append(row)
            print(f"blocks={blocks} p={degree} compile {build:.1f}s "
                  + " ".join(f"{name}:{row['orders'][name]['median_relative_error']:.2e}"
                             for name in kernels), flush=True)
    report["moment_scan"] = scan

    # -- the order no expansion touches --------------------------------------
    if not arguments.skip_ballistic:
        began = perf_counter()
        ballistic_control = _fast_ballistic(elements, control, medium,
                                            elements.cone_cosine)
        control_seconds = perf_counter() - began
        began = perf_counter()
        ballistic_all = _fast_ballistic(elements, receivers, medium,
                                        elements.cone_cosine)
        all_seconds = perf_counter() - began
        report["ballistic"] = {"control_seconds": control_seconds,
                               "all_receivers_seconds": all_seconds,
                               "control_value": ballistic_control.tolist(),
                               "all_receivers_total": float(ballistic_all.sum()),
                               "backend": BALLISTIC_BACKEND,
                               "note": "Closed cone formula per element; no cache, "
                                       "no expansion, and no moment route helps it. "
                                       "It is the main contribution and the fastest "
                                       "method by construction -- see 'backend'."}
        print(f"ballistic ({BALLISTIC_BACKEND}): {all_seconds:.1f} s for "
              f"{len(receivers)} receivers", flush=True)

    report["peak_megabytes"] = peak_megabytes()
    (out / "shower-moments.json").write_text(
        json.dumps(report, default=serial, indent=2), encoding="utf-8")
    print(f"written: {out / 'shower-moments.json'}")


if __name__ == "__main__":
    main()
