"""Cost of the whole route: build, save, load, apply, reuse, read out.

Run: python scripts/benchmark_route.py --output .build/claude-review/timing

Every stage is timed separately, because they scale differently and only one
of them is paid per event. Peak resident memory is sampled around each stage
with :mod:`resource`, and the cache's own accounting of its Bessel block,
scratch files and output array is reported next to it: the first is the
process, the second is the part this code controls.

The last section compares two ways of evaluating the same small event at the
same physics and the same numerics: a plain loop over segments, and a batched
route that gathers every emission node of every segment into one interpolation
call. Both are run twice so the first-call cost of the spline construction is
visible rather than averaged away.
"""
from pathlib import Path
import argparse
import json
import platform
import resource
import sys
from time import perf_counter

import numpy as np

from lighthit import Medium, SolverSettings
from lighthit.cache import BandedResponseCache, CacheGrid, ResponseCache
from lighthit.readout import inverse_bins
from lighthit.experimental.cone_segment import (ConeSegment, ballistic_spectrum,
                                                segment_spectrum)

MEDIUM = Medium(0.04, 0.05, 0.7, 1.35, 450.0, "course-synthetic")


def peak_megabytes():
    """Peak resident size of this process so far, in MiB."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1024 if sys.platform.startswith("linux") else peak / 2 ** 20


def timed(function, *args, repeats=1, **kwargs):
    """Return (result, seconds per repeat as a list)."""
    seconds, result = [], None
    for _ in range(repeats):
        start = perf_counter()
        result = function(*args, **kwargs)
        seconds.append(perf_counter() - start)
    return result, seconds


def build_event(count):
    """A chain of straight segments with individual times and yields."""
    pieces, start, time = [], np.zeros(3), 0.0
    direction = np.array([0.1, 0.05, 1.0])
    direction /= np.linalg.norm(direction)
    for index in range(count):
        beta = 0.999 - 0.01 * index
        pieces.append(ConeSegment(tuple(start), tuple(direction), 1.5, beta,
                                  1.34, 5000.0 - 300.0 * index, time))
        start = start + 1.5 * np.asarray(direction)
        time += 1.5 / (beta * 0.299792458)
        turn = np.array([0.04 * (-1) ** index, 0.03, 0.0])
        direction = direction + turn
        direction /= np.linalg.norm(direction)
    return pieces


def batched_event_spectrum(cache, segments, receivers, *, longitudinal_order=48):
    """One interpolation call for every node of every segment.

    Same quadrature, same weights and same cache as the per-segment route; the
    only difference is that the radii and cosines of all segments are gathered
    before the moments are asked for.
    """
    from scipy.special import eval_legendre, roots_legendre
    band = cache.bands[0] if isinstance(cache, BandedResponseCache) else cache
    omega = band.grid.omega_per_ns
    receivers = np.atleast_2d(np.asarray(receivers, float))
    nodes, weights = roots_legendre(int(longitudinal_order))
    radii, cosines, phases, scales = [], [], [], []
    for piece in segments:
        start = np.asarray(piece.start_m, float)
        axis = np.asarray(piece.direction, float)
        half = piece.length_m / 2
        a = half * (nodes + 1)
        vectors = receivers[None, :, :] - (start[None, None, :] + a[:, None, None] * axis)
        distance = np.linalg.norm(vectors, axis=2)
        radii.append(distance.ravel())
        cosines.append(((vectors @ axis) / distance).ravel())
        time = piece.start_time_ns + a / (piece.beta * 0.299792458)
        phases.append(np.exp(1j * omega[:, None] * time[None, :]))
        scales.append(half * weights * piece.photons_per_m)
    moments = band.moments_at(np.concatenate(radii))
    cosines = np.concatenate(cosines)
    ell = np.arange(band.degree + 1)
    block_size = len(nodes) * len(receivers)
    out = np.zeros((len(omega), len(receivers), 3), complex)
    for index, piece in enumerate(segments):
        lo = index * block_size
        weight = (eval_legendre(ell, piece.cone_cosine)[None, :]
                  * eval_legendre(ell[None, :], cosines[lo:lo + block_size, None]))
        field = np.einsum("wrlo,rl->wro", moments[:, lo:lo + block_size], weight,
                          optimize=True)
        field = field.reshape(len(omega), len(nodes), len(receivers), 2)
        out[:, :, 1:] += np.einsum("wj,j,wjdo->wdo", phases[index], scales[index],
                                   field, optimize=True)
        out[:, :, 0] += ballistic_spectrum(omega, piece,
                                           receivers - np.asarray(piece.start_m),
                                           band.medium)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=".build/claude-review/timing")
    parser.add_argument("--frequencies", type=int, default=41)
    parser.add_argument("--segments", type=int, default=8)
    parser.add_argument("--receivers", type=int, default=14)
    arguments = parser.parse_args()
    out = Path(arguments.output)
    out.mkdir(parents=True, exist_ok=True)

    omega = np.linspace(0.0, 0.5, arguments.frequencies)
    settings = SolverSettings(32, 340, 6.0, 0.04, 10)
    grid = CacheGrid.geometric(12.0, 48.0, 44, omega)
    report = {"environment": {"python": sys.version, "platform": platform.platform(),
                              "numpy": np.__version__,
                              "threads": {name: __import__("os").environ.get(name)
                                          for name in ("OMP_NUM_THREADS",
                                                       "OPENBLAS_NUM_THREADS",
                                                       "MKL_NUM_THREADS")},
                              "cpu_count": __import__("os").cpu_count()},
               "settings": {"scattering_degree": settings.scattering_degree,
                            "spatial_degree": settings.spatial_degree,
                            "k_max_per_m": settings.k_max_per_m,
                            "k_panel_per_m": settings.k_panel_per_m,
                            "k_nodes": len(settings.quadrature()[0]),
                            "radii": len(grid.radii_m),
                            "frequencies": len(omega)},
               "memory_before_build_MiB": peak_megabytes()}

    cache, seconds = timed(ResponseCache.build, MEDIUM, settings, grid)
    report["build"] = {"seconds": seconds[0], "stages": cache.timings_s,
                       "peak_process_MiB": peak_megabytes(),
                       "moments_MiB": cache.moments.nbytes / 2 ** 20}

    streamed, seconds = timed(ResponseCache.build, MEDIUM, settings, grid,
                              radius_chunk=8)
    report["build_streamed"] = {"seconds": seconds[0], "stages": streamed.timings_s,
                                "peak_process_MiB": peak_megabytes(),
                                "identical": bool(np.array_equal(streamed.moments,
                                                                 cache.moments))}
    del streamed

    path = out / "benchmark-cache.npz"
    _, seconds = timed(cache.save, path)
    report["save"] = {"seconds": seconds[0],
                      "file_MiB": path.stat().st_size / 2 ** 20}
    loaded, seconds = timed(ResponseCache.load, path, repeats=3)
    report["load"] = {"seconds": seconds, "median": float(np.median(seconds))}

    banded = BandedResponseCache([loaded])
    segments = build_event(arguments.segments)
    heights = np.linspace(-9.0, 33.0, arguments.receivers // 2)
    receivers = np.array([[x, y, z] for x, y in ([24.0, 3.0], [-17.0, 14.0])
                          for z in heights])

    def per_segment():
        return sum(segment_spectrum(banded, piece, receivers, longitudinal_order=48)
                   for piece in segments)

    first, seconds = timed(per_segment)
    report["apply_first_call"] = {"seconds": seconds[0],
                                  "note": "includes building the interpolation spline"}
    _, seconds = timed(per_segment, repeats=3)
    report["apply_repeat"] = {"seconds": seconds, "median": float(np.median(seconds))}

    moved = [ConeSegment(tuple(np.asarray(p.start_m) + np.array([7.0, -4.0, 2.0])),
                         p.direction, p.length_m, p.beta, p.phase_index,
                         p.photons_per_m, p.start_time_ns) for p in segments]
    _, seconds = timed(lambda: sum(segment_spectrum(banded, piece, receivers,
                                                    longitudinal_order=48)
                                   for piece in moved), repeats=3)
    report["apply_new_pose"] = {"seconds": seconds, "median": float(np.median(seconds))}

    batched, seconds = timed(batched_event_spectrum, banded, segments, receivers)
    report["batched_first_call"] = {"seconds": seconds[0]}
    _, seconds = timed(batched_event_spectrum, banded, segments, receivers, repeats=3)
    scale = np.max(np.abs(first), axis=(0, 2))
    report["batched_repeat"] = {
        "seconds": seconds, "median": float(np.median(seconds)),
        "max_relative_difference_to_per_segment":
            float(np.max(np.abs(batched - first) / scale[None, :, None])),
        "note": "Same quadrature and cache; only the grouping of the "
                "interpolation calls differs."}

    edges = np.arange(0.0, 620.0, 5.0)
    _, seconds = timed(lambda: [inverse_bins(omega, first[:, :, order], edges)
                                for order in (1, 2)], repeats=3)
    report["readout"] = {"seconds": seconds, "median": float(np.median(seconds)),
                         "bins": len(edges) - 1}

    direct = None
    try:
        from lighthit.green import PointGreenSolver
        solver = PointGreenSolver(MEDIUM, settings)
        _, seconds = timed(solver.solve, omega, receivers[:1], direction=(0.0, 0.0, 1.0))
        direct = seconds[0]
    except Exception as error:  # pragma: no cover - reported, not raised
        report["direct_solve_error"] = str(error)
    report["direct_solve_one_receiver"] = direct

    try:
        import numba  # noqa: F401
        report["numba"] = "available; see scripts/benchmark_backends.py"
    except ImportError as error:
        report["numba"] = f"not installed here: {error}"

    report["peak_process_MiB_total"] = peak_megabytes()
    (out / "timing.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items()
                      if key not in ("environment",)}, indent=2)[:4000])


if __name__ == "__main__":
    main()
