"""The moment route at nonzero frequency, where the block criterion changes.

At ``omega = 0`` the only thing that decides how large a block may be is how
far the kernel bends across it, so ``extent / distance`` is a fair guide. At
nonzero frequency a second, sharper scale appears: the arrival phase swings by

    delta = omega * (block half-size) / v

across the block, and once that approaches one radian a low-order polynomial
cannot follow it however far away the receiver is. This script measures that
directly: the same event, the same receivers, the same kernel, scanning
frequency against the number of blocks, and reporting ``delta`` beside the
error so the two can be compared rather than argued about.

Two settings have to follow the frequency rather than being left at their
``omega = 0`` values, and getting them wrong quietly produces a converged-looking
comparison on an unresolved kernel:

* the wavenumber cutoff, because the spectral kernel carries structure out to
  ``k ~ |mu_t - i*omega/v|``, which is ``omega/v`` once the frequency is
  appreciable. ``--k-margin`` sets how many times that the cutoff is placed at;
* the stored multipole degree, which has to follow ``k_max * r_max``.

Both are derived here from the top frequency and the distance range, and both
are written into the report. The distance range is deliberately restricted:
the cache cost grows with ``k_max * r_max``, so a frequency scan at array scale
is a different and much larger computation.

Run:
  python scripts/run_shower_frequency.py --input /path/sim_e_100GeV_10.h5 \
      --event 5 --output .build/review/shower-moments
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
                                                 direct_response, evaluate_moments)
from lighthit.experimental.g4_source import SourceContract, load_event

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_shower_moments import array_positions, serial  # noqa: E402

ORDERS = {"first": 0, "two_or_more": 1}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--event", type=int, default=5)
    parser.add_argument("--output", default=".build/review/shower-moments")
    parser.add_argument("--frequencies", type=float, nargs="*",
                        default=[0.0, 0.1, 0.3, 0.6])
    parser.add_argument("--angular-degree", type=int, default=16)
    parser.add_argument("--spatial-degree", type=int, default=2)
    parser.add_argument("--blocks", type=int, nargs="*", default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--max-distance", type=float, default=80.0)
    parser.add_argument("--controls", type=int, default=5)
    parser.add_argument("--radii", type=int, default=90)
    parser.add_argument("--k-margin", type=float, default=3.0,
                        help="k_max is this many times omega_max / v")
    parser.add_argument("--cache", default=None)
    arguments = parser.parse_args()
    out = Path(arguments.output)
    out.mkdir(parents=True, exist_ok=True)

    medium = synthetic_medium()
    speed = medium.speed_m_per_ns
    report = {"environment": {"python": sys.version, "platform": platform.platform(),
                              "numpy": np.__version__},
              "settings": vars(arguments), "speed_m_per_ns": speed}

    elements = load_event(arguments.input, arguments.event, SourceContract())
    elements = elements.moved(translation=np.array([20.0, 15.0, -20.0]))
    report["source"] = elements.summary()

    receivers = array_positions()
    distance = np.linalg.norm(receivers - elements.centroid_m, axis=1)
    keep = distance <= arguments.max_distance
    receivers, distance = receivers[keep], distance[keep]
    order = np.argsort(distance)
    picks = order[np.linspace(0, len(order) - 1, arguments.controls).astype(int)]
    control = receivers[picks]
    report["array"] = {"receivers": len(receivers),
                       "control_distance_m": distance[picks].tolist()}

    omega = sorted(set(float(value) for value in arguments.frequencies))
    reach = 1.5 * elements.extent_m
    low = max(3.0, (distance.min() - reach) * 0.95)
    high = (distance.max() + reach) * 1.05
    k_max = max(2.0, arguments.k_margin * max(omega) / speed)
    degree = int(np.ceil(k_max * high)) + 40
    report["derived"] = {"k_max_per_m": k_max, "spatial_degree": degree,
                         "radius_range_m": [low, high],
                         "omega_over_v_per_m": [value / speed for value in omega]}
    print(f"k_max {k_max:.2f} /m, spatial degree {degree}, radii {low:.1f}-{high:.1f} m",
          flush=True)

    if arguments.cache and Path(arguments.cache).exists():
        cache = ResponseCache.load(arguments.cache)
    else:
        settings = SolverSettings(24, degree, k_max, 0.04, 10)
        grid = CacheGrid.geometric(low, high, arguments.radii, omega)
        began = perf_counter()
        cache = ResponseCache.build(medium, settings, grid)
        report["cache"] = {"build_seconds": perf_counter() - began,
                           "stages": cache.timings_s}
        print(f"cache built in {perf_counter() - began:.0f} s", flush=True)
        if arguments.cache:
            cache.save(arguments.cache)
    report.setdefault("cache", {})["radii"] = len(cache.grid.radii_m)
    report["cache"]["degree"] = int(cache.degree)

    kernels = {name: KernelChannels.of(cache, index, arguments.angular_degree,
                                       frequencies=omega)
               for name, index in ORDERS.items()}
    direct = {}
    for name, kernel in kernels.items():
        began = perf_counter()
        direct[name] = direct_response(kernel, elements, control)
        report.setdefault("direct_reference", {})[name] = {
            "seconds": perf_counter() - began,
            "abs": np.abs(direct[name]).tolist()}
        print(f"direct {name}: {perf_counter() - began:.1f} s", flush=True)

    scan = []
    for blocks in arguments.blocks:
        partition = BlockPartition.split(elements, blocks)
        half = float(np.max(np.linalg.norm(partition.half_sizes_m, axis=1)))
        began = perf_counter()
        moments, powers = compile_joint_moments(elements, partition,
                                                arguments.angular_degree,
                                                arguments.spatial_degree, omega)
        build = perf_counter() - began
        row = {"blocks": blocks, "largest_half_size_m": half,
               "phase_spread_rad": {str(value): value * half / speed for value in omega},
               "compile_seconds": build, "orders": {}}
        for name, kernel in kernels.items():
            value = evaluate_moments(kernel, moments, powers, partition, control)
            error = (np.abs(value - direct[name])
                     / np.maximum(np.abs(direct[name]), 1e-300))
            row["orders"][name] = {
                "median_relative_error": {str(value): float(np.median(error[i]))
                                          for i, value in enumerate(omega)},
                "max_relative_error": {str(value): float(np.max(error[i]))
                                       for i, value in enumerate(omega)}}
        scan.append(row)
        summary = " ".join(
            f"w={value}:{row['orders']['two_or_more']['median_relative_error'][str(value)]:.2e}"
            for value in omega)
        print(f"blocks={blocks} half={half:.2f}m {summary}", flush=True)
    report["frequency_scan"] = scan
    report["peak_megabytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    (out / "shower-frequency.json").write_text(
        json.dumps(report, default=serial, indent=2), encoding="utf-8")
    print(f"written: {out / 'shower-frequency.json'}")


if __name__ == "__main__":
    main()
