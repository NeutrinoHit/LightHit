"""The moment route against the element sum, at equal accuracy, timed.

An error table says which settings are accurate; it does not say whether the
route is worth using. That depends on a comparison at *matched* accuracy, and
on the fact that the two sides scale differently:

    direct, per receiver        ~  N_elements * (L_q + 1)
    moment apply, per receiver  ~  N_blocks * S^3 * (L_q + 1)^2

with ``S`` the stencil order. The moment side also pays a one-off compile that
no receiver count changes. Because the number of blocks has to grow with the
frequency — the arrival phase swings by ``omega * h / v`` across a block — the
second expression grows with frequency while the first does not, and the two
can cross. This script measures where.

Run:
  python scripts/benchmark_moment_route.py --input /path/sim_e_100GeV_10.h5 \
      --event 5 --output .build/review/shower-moments
"""
from pathlib import Path
import argparse
import json
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_shower_moments import array_positions, serial  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--event", type=int, default=5)
    parser.add_argument("--output", default=".build/review/shower-moments")
    parser.add_argument("--frequencies", type=float, nargs="*", default=[0.0, 0.6])
    parser.add_argument("--angular-degree", type=int, default=16)
    parser.add_argument("--spatial-degree", type=int, default=2)
    parser.add_argument("--blocks", type=int, nargs="*", default=[4, 256])
    parser.add_argument("--max-distance", type=float, default=80.0)
    parser.add_argument("--radii", type=int, default=90)
    parser.add_argument("--k-margin", type=float, default=3.0)
    parser.add_argument("--cache", default=None)
    arguments = parser.parse_args()
    out = Path(arguments.output)
    out.mkdir(parents=True, exist_ok=True)

    medium = synthetic_medium()
    speed = medium.speed_m_per_ns
    elements = load_event(arguments.input, arguments.event, SourceContract())
    elements = elements.moved(translation=np.array([20.0, 15.0, -20.0]))
    receivers = array_positions()
    distance = np.linalg.norm(receivers - elements.centroid_m, axis=1)
    receivers = receivers[distance <= arguments.max_distance]
    distance = distance[distance <= arguments.max_distance]

    omega = sorted(set(float(value) for value in arguments.frequencies))
    reach = 1.5 * elements.extent_m
    low = max(3.0, (distance.min() - reach) * 0.95)
    high = (distance.max() + reach) * 1.05
    k_max = max(2.0, arguments.k_margin * max(omega) / speed)
    degree = int(np.ceil(k_max * high)) + 40
    if arguments.cache and Path(arguments.cache).exists():
        cache = ResponseCache.load(arguments.cache)
    else:
        cache = ResponseCache.build(medium, SolverSettings(24, degree, k_max, 0.04, 10),
                                    CacheGrid.geometric(low, high, arguments.radii, omega))
        if arguments.cache:
            cache.save(arguments.cache)

    kernel = KernelChannels.of(cache, 1, arguments.angular_degree, frequencies=omega)
    report = {"settings": vars(arguments), "elements": len(elements),
              "receivers": len(receivers),
              "distance_range_m": [float(distance.min()), float(distance.max())],
              "speed_m_per_ns": speed, "k_max_per_m": k_max, "cache_degree": int(cache.degree)}

    began = perf_counter()
    reference = direct_response(kernel, elements, receivers)
    direct_seconds = perf_counter() - began
    report["direct"] = {"seconds": direct_seconds,
                        "seconds_per_receiver": direct_seconds / len(receivers)}
    print(f"direct: {direct_seconds:.1f} s for {len(receivers)} receivers "
          f"({direct_seconds / len(receivers) * 1e3:.0f} ms each)", flush=True)

    stencil = max(3, arguments.spatial_degree + 2)
    rows = []
    for blocks in arguments.blocks:
        partition = BlockPartition.split(elements, blocks)
        half = float(np.max(np.linalg.norm(partition.half_sizes_m, axis=1)))
        began = perf_counter()
        moments, powers = compile_joint_moments(elements, partition,
                                                arguments.angular_degree,
                                                arguments.spatial_degree, omega)
        compile_seconds = perf_counter() - began
        began = perf_counter()
        value = evaluate_moments(kernel, moments, powers, partition, receivers)
        apply_seconds = perf_counter() - began
        error = np.abs(value - reference) / np.maximum(np.abs(reference), 1e-300)
        work = blocks * stencil ** 3 * (arguments.angular_degree + 1) ** 2
        rows.append({
            "blocks": blocks, "largest_half_size_m": half,
            "monomials": len(powers), "stencil_order": stencil,
            "compile_seconds": compile_seconds, "apply_seconds": apply_seconds,
            "apply_seconds_per_receiver": apply_seconds / len(receivers),
            "phase_spread_rad": {str(w): w * half / speed for w in omega},
            "median_relative_error": {str(w): float(np.median(error[i]))
                                      for i, w in enumerate(omega)},
            "break_even_receivers": {
                str(w): (compile_seconds
                         / max(direct_seconds / len(receivers)
                               - apply_seconds / len(receivers), 1e-12))
                for w in omega},
            "apply_work_units": work,
            "direct_work_units": len(elements) * (arguments.angular_degree + 1)})
        print(f"blocks={blocks:5d} half={half:.2f}m compile {compile_seconds:.1f}s "
              f"apply {apply_seconds:.1f}s  "
              + " ".join(f"w={w}:{rows[-1]['median_relative_error'][str(w)]:.1e}"
                         for w in omega), flush=True)
    report["moment_route"] = rows
    report["peak_megabytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    (out / "moment-route-cost.json").write_text(
        json.dumps(report, default=serial, indent=2), encoding="utf-8")
    print(f"written: {out / 'moment-route-cost.json'}")


if __name__ == "__main__":
    main()
