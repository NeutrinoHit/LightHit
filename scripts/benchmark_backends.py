#!/usr/bin/env python3
"""Compare NumPy and Numba on the same complete public point-Green problem.

Run from the repository root, after installing .[accelerate]:
  python scripts/benchmark_backends.py --repeat 3

Workers run sequentially in separate processes with one numerical-library
thread. Numba gets a fresh temporary JIT cache. First-use costs, warm solves,
readout, and output writing are reported separately. No scientific cache,
geometry reduction, coarser grid, or private data is introduced by this script.
A caller-selected config is used verbatim: keep private-medium outputs local.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from time import perf_counter


def worker(args):
    began = perf_counter()
    import platform
    import numpy as np
    import scipy
    from lighthit import PointGreenSolver
    from lighthit.cli import configure, peak_rss_mib
    config, medium, settings = configure(args.config)
    if args.backend == "numba":
        import numba
        numba_version = numba.__version__
    else:
        numba_version = None
    source = config.get("source", {})
    frequency = config.get("frequency", {})
    omega = np.linspace(0., frequency.get("max_per_ns", 1.2), frequency.get("nodes", 241))
    obs = config.get("detector", {}).get("displacement_m", [17.32050807568877, 0., 10.])
    mode = source.get("kind", "directed")
    if mode not in ("directed", "isotropic"):
        raise ValueError("source.kind must be 'directed' or 'isotropic'")
    kwargs = dict(direction=source.get("direction", [0, 0, 1]) if mode == "directed" else None,
                  photons=source.get("photons", 1.), emission_time_ns=source.get("time_ns", 0.))
    solver = PointGreenSolver(medium, settings, angular_backend=args.backend,
                              single_backend=args.backend)
    init = perf_counter() - began
    first = solver.solve(omega, obs, **kwargs)
    first_timing = first.timings_s.copy()
    runs = []
    for _ in range(args.repeat):
        result = solver.solve(omega, obs, **kwargs)
        runs.append(result.timings_s.copy())
    readout = config.get("readout", {})
    width = float(readout.get("bin_width_ns", 2.))
    front = min(result.front_time_ns)
    edges = np.arange(front + readout.get("start_relative_front_ns", -30.),
                      front + readout.get("stop_relative_front_ns", 650.) + width * .5, width)
    raw = result.readout(edges, 0.)
    smooth = result.readout(edges, readout.get("sigma_ns", 3.))
    charge_times = []
    for _ in range(args.repeat):
        charge = solver.solve([0.], obs, **kwargs)
        charge_times.append(charge.timings_s.copy())
    from dataclasses import asdict
    report = dict(backend=args.backend, angular_backend=result.angular_backend,
                  single_backend=result.single_backend,
                  config=config, medium=asdict(medium),
                  settings=asdict(settings), initialization_s=init,
                  first_use_solve=first_timing, warm_solves=runs,
                  median_warm_solve_s=float(np.median([t["total"] for t in runs])),
                  charge_solves=charge_times,
                  median_charge_s=float(np.median([t["total"] for t in charge_times])),
                  readout_s=dict(raw=raw.elapsed_s, smooth=smooth.elapsed_s),
                  diagnostics=dict(raw=raw.diagnostics, smooth=smooth.diagnostics),
                  charge_per_m2=result.charge_per_m2.tolist(),
                  peak_process_rss_mib=peak_rss_mib(),
                  environment=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__,
                                   numba=numba_version, platform=platform.platform(),
                                   machine=platform.machine(),
                                   threads={key: os.environ.get(key) for key in THREAD_KEYS}),
                  note="First use includes any JIT compilation. Warm solves recompute transport. "
                       "Readout/output excluded from solve timings. RSS is whole-process high-water mark.")
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    io_start = perf_counter()
    np.savez_compressed(out / (args.backend + ".npz"), omega=omega, components=result.components,
                        raw=raw.components, smooth=smooth.components, edges=edges)
    report["npz_write_s"] = perf_counter() - io_start
    report["worker_s_before_json_write"] = perf_counter() - began
    (out / (args.backend + ".json")).write_text(json.dumps(report, indent=2) + "\n")


THREAD_KEYS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="examples/point-green.toml")
    p.add_argument("--output", default=".build/benchmark-backends")
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--backend", choices=["numpy", "numba"], help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.repeat < 1:
        p.error("--repeat must be positive")
    if args.backend:
        worker(args)
        return
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    elapsed = {}
    with tempfile.TemporaryDirectory(prefix="lighthit-jit-benchmark-") as jit_cache:
        env = dict(os.environ)
        env.update({key: "1" for key in THREAD_KEYS})
        env["NUMBA_CACHE_DIR"] = jit_cache
        for backend in ("numpy", "numba"):
            command = [sys.executable, str(Path(__file__).resolve()), "--backend", backend,
                       "--config", args.config, "--output", args.output, "--repeat", str(args.repeat)]
            start = perf_counter()
            with (out / (backend + ".log")).open("w") as log:
                subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            elapsed[backend] = perf_counter() - start
    import numpy as np
    reports = {b: json.loads((out / (b + ".json")).read_text()) for b in elapsed}
    with np.load(out / "numpy.npz") as ref, np.load(out / "numba.npz") as got:
        charge = np.abs(ref["components"][0].sum(axis=-1).real)
        if np.any(charge == 0):
            raise ValueError("Benchmark comparison requires nonzero charge at every point")
        diff = got["components"] - ref["components"]
        metrics = {
            "max_component_spectral_difference_over_charge": float(np.max(abs(diff) / charge[None, :, None])),
            "max_total_spectral_difference_over_charge": float(np.max(abs(diff.sum(axis=-1)) / charge[None, :])),
            "max_charge_relative_difference": float(np.max(abs(diff[0].sum(axis=-1).real) / charge)),
            "raw_bin_L1_difference_over_charge": (abs(got["raw"]-ref["raw"]).sum(axis=(1,2))/charge).tolist(),
            "smooth_bin_L1_difference_over_charge": (abs(got["smooth"]-ref["smooth"]).sum(axis=(1,2))/charge).tolist(),
        }
    summary = dict(median_warm_solve_s={b: reports[b]["median_warm_solve_s"] for b in elapsed},
                   speedup=reports["numpy"]["median_warm_solve_s"]/reports["numba"]["median_warm_solve_s"],
                   fresh_worker_total_s=elapsed, accuracy=metrics,
                   jit_cache="Fresh temporary directory, removed after measurements",
                   note="Comparison at identical grids; this checks backend equivalence, not grid convergence.")
    (out / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
