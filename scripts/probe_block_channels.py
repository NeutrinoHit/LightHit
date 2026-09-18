"""Why a higher polynomial degree stops helping: the block has to resolve
the harmonics.

The kernel channel of @eq-kernel-channels carries ``Y_lm`` of the direction to
the receiver. Across a block of half-size ``h`` at distance ``r`` that
direction swings by about ``h / r``, and the harmonic of degree ``l``
oscillates on an angular scale ``1 / l``. So the polynomial that is fitted
across the block has to follow roughly

    s = L_q * h / r

oscillations. Once ``s`` exceeds one, raising the polynomial degree buys
nothing, because the difficulty is not the smooth bending of the kernel but
the wiggle of the harmonic. This script measures the error against ``s``
directly, scanning the retained degree against the number of blocks at fixed
polynomial degree.

The element list is subsampled — the expansion error is a property of the
geometry, not of how many elements carry the light — so that the whole scan
costs minutes rather than hours. The photon count is scaled to compensate,
which keeps the event's brightness but is irrelevant to a relative error.

Run:
  python scripts/probe_block_channels.py --input /path/sim_e_100GeV_10.h5 \
      --event 5 --output .build/review/shower-moments
"""
from pathlib import Path
import argparse
import json
import sys
from time import perf_counter

import numpy as np

from lighthit import SolverSettings, synthetic_medium
from lighthit.cache import CacheGrid, ResponseCache
from lighthit.experimental.event_moments import (BlockPartition, KernelChannels,
                                                 compile_joint_moments,
                                                 direct_response, evaluate_moments)
from lighthit.experimental.g4_source import LightElements, SourceContract, load_event

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_shower_moments import array_positions, serial  # noqa: E402


def subsample(elements, stride):
    if stride <= 1:
        return elements
    piece = slice(None, None, stride)
    scale = len(elements) / len(elements.photons[piece])
    return LightElements(elements.start_m[piece], elements.direction[piece],
                         elements.length_m[piece], elements.photons[piece] * scale,
                         elements.cone_cosine[piece], elements.start_ns[piece],
                         elements.end_ns[piece], elements.row_index[piece],
                         elements.uid[piece], elements.contract,
                         {**elements.provenance, "subsample_stride": stride})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--event", type=int, default=5)
    parser.add_argument("--output", default=".build/review/shower-moments")
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--spatial-degree", type=int, default=3)
    parser.add_argument("--degrees", type=int, nargs="*", default=[8, 16, 32, 64])
    parser.add_argument("--blocks", type=int, nargs="*", default=[1, 4, 16, 64])
    parser.add_argument("--controls", type=int, default=5)
    parser.add_argument("--radii", type=int, default=160)
    parser.add_argument("--cache", default=None)
    arguments = parser.parse_args()
    out = Path(arguments.output)
    out.mkdir(parents=True, exist_ok=True)

    elements = load_event(arguments.input, arguments.event, SourceContract())
    elements = subsample(elements.moved(translation=np.array([20.0, 15.0, -20.0])),
                         arguments.stride)
    receivers = array_positions()
    distance = np.linalg.norm(receivers - elements.centroid_m, axis=1)
    order = np.argsort(distance)
    picks = order[np.linspace(0, len(order) - 1, arguments.controls).astype(int)]
    control, control_distance = receivers[picks], distance[picks]

    medium = synthetic_medium()
    report = {"settings": vars(arguments), "elements": len(elements),
              "control_distance_m": control_distance.tolist(),
              "event_extent_m": elements.extent_m}
    print(f"{len(elements)} elements, control at "
          f"{np.round(control_distance, 1).tolist()} m", flush=True)

    if arguments.cache and Path(arguments.cache).exists():
        cache = ResponseCache.load(arguments.cache)
    else:
        reach = 1.5 * elements.extent_m
        grid = CacheGrid.geometric(max(3.0, (distance.min() - reach) * 0.95),
                                   (distance.max() + reach) * 1.05,
                                   arguments.radii, [0.0])
        cache = ResponseCache.build(medium, SolverSettings(24, 480, 2.0, 0.04, 10), grid)
        if arguments.cache:
            cache.save(arguments.cache)
    report["cache"] = {"radii": len(cache.grid.radii_m), "degree": int(cache.degree)}

    rows = []
    for blocks in arguments.blocks:
        partition = BlockPartition.split(elements, blocks)
        half = np.linalg.norm(partition.half_sizes_m, axis=1).max()
        for degree in arguments.degrees:
            kernel = KernelChannels.of(cache, 1, degree, frequencies=[0.0])
            exact = direct_response(kernel, elements, control)[0]
            began = perf_counter()
            moments, powers = compile_joint_moments(elements, partition, degree,
                                                    arguments.spatial_degree, [0.0])
            value = evaluate_moments(kernel, moments, powers, partition, control)[0]
            error = np.abs(value - exact) / np.abs(exact)
            rows.append({"blocks": blocks, "angular_degree": degree,
                         "largest_half_size_m": float(half),
                         "oscillations": (degree * half / control_distance).tolist(),
                         "relative_error": error.tolist(),
                         "median_relative_error": float(np.median(error)),
                         "seconds": perf_counter() - began})
            print(f"blocks={blocks:3d} Lq={degree:3d} h={half:.2f}m "
                  f"s={degree * half / control_distance.min():6.2f} "
                  f"err={np.median(error):.2e}", flush=True)
    report["scan"] = rows
    (out / "block-channels.json").write_text(
        json.dumps(report, default=serial, indent=2), encoding="utf-8")
    print(f"written: {out / 'block-channels.json'}")


if __name__ == "__main__":
    main()
