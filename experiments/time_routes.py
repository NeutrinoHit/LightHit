#!/usr/bin/env python
"""Cold, JIT and warm computation time of the default routes on one machine.

    python experiments/time_routes.py --output timing.json [--cases electron-100GeV ...]
          [--threads 1] [--calls 3]

Every case runs twice, each time in a fresh process with the thread count
fixed through NUMBA_NUM_THREADS, OMP_NUM_THREADS, OPENBLAS_NUM_THREADS,
VECLIB_MAXIMUM_THREADS and MKL_NUM_THREADS:

* ``empty-jit-cache``: NUMBA_CACHE_DIR is a new empty directory, so every
  compiled function is compiled in this process;
* ``filled-jit-cache``: the same directory, now filled by the first process.

A process records the import of lighthit, the setup (model, kernel and route
tables opened), then ``--calls`` event calls. Every call receives a new copy
of the source, so a route does not reuse what it compiled for an earlier
source object; tables, compiled functions and table-level caches are reused,
as for the second event of a run. Definitions written to the output:

* cold event = import + setup + first call in a fresh process;
* warm event = median of the later calls;
* JIT share of the first call = first call - warm event (empty cache), which
  also contains first touches of memory-mapped tables.

The data bundle is ``$LIGHTHIT_DATA`` (inputs as in experiments/run.py). The
operating-system file cache is not purged; whether the table files were read
from disk or from memory is therefore not controlled and is stated as such.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

THREAD_VARIABLES = ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS")
CASES = {"electron-100GeV": "default-shower-route/electron-100GeV",
         "muon-100GeV": "default-muon-route/muon-100GeV",
         "laser-532nm": "default-laser-route/laser-532nm"}


def _clock():
    return time.perf_counter(), time.process_time()


def _since(start):
    wall, cpu = _clock()
    return dict(wall_s=wall - start[0], cpu_s=cpu - start[1])


def child(name, calls):
    began = _clock()
    import lighthit  # noqa: F401  (the import itself is measured)
    import numba
    imported = _since(began)
    from dataclasses import replace
    from harness.context import Context
    from harness.inputs import laser_flash
    import numpy as np
    ctx = Context()
    start = _clock()
    kernel = ctx.kernel()
    if name == "electron-100GeV":
        from lighthit.shower import ShowerTransport
        route = ShowerTransport(kernel)
        route.tables()
    elif name == "muon-100GeV":
        route = lighthit.MuonTransport(kernel)
        if hasattr(route, "tables"):
            route.tables()
    else:
        from lighthit.event import EventTransport
        route = EventTransport(kernel, laser_kernel=ctx.laser_kernel())
    setup = _since(start)
    if name == "electron-100GeV":
        source = ctx.shower_source(name)
        run = lambda: route(replace(source))
    elif name == "muon-100GeV":
        steps, ancestry, uid = ctx.muon_source(name)
        run = lambda: route(replace(steps), dict(ancestry), primary_uid=uid)
    else:
        run = lambda: route(laser_flash())
    records = []
    for _ in range(calls):
        start = _clock()
        response = run()
        entry = _since(start)
        entry["charge_pe_by_order"] = [float(x) for x in
                                       np.asarray(response.charge_components_pe).sum(axis=0)]
        records.append(entry)
    return dict(case=name, import_=imported, setup=setup, calls=records,
                numba_threads=numba.get_num_threads(),
                lighthit=lighthit.__version__, inputs=ctx.inputs)


def summarise(name, runs):
    empty, filled = runs["empty-jit-cache"], runs["filled-jit-cache"]
    def warm(run):
        later = [c["wall_s"] for c in run["calls"][1:]]
        return statistics.median(later) if later else None
    def cold(run):
        return run["import_"]["wall_s"] + run["setup"]["wall_s"] + run["calls"][0]["wall_s"]
    charges = [c["charge_pe_by_order"] for r in (empty, filled) for c in r["calls"]]
    spread = max(max(abs(a - b) for a, b in zip(c, charges[0])) for c in charges)
    return dict(case=name, registry_case=CASES[name],
                cold_event_s=dict(empty_jit_cache=cold(empty), filled_jit_cache=cold(filled)),
                import_s=dict(empty_jit_cache=empty["import_"]["wall_s"],
                              filled_jit_cache=filled["import_"]["wall_s"]),
                setup_s=dict(empty_jit_cache=empty["setup"]["wall_s"],
                             filled_jit_cache=filled["setup"]["wall_s"]),
                first_call_s=dict(empty_jit_cache=empty["calls"][0]["wall_s"],
                                  filled_jit_cache=filled["calls"][0]["wall_s"]),
                warm_event_s=warm(filled), warm_event_empty_cache_process_s=warm(empty),
                jit_share_of_first_call_s=empty["calls"][0]["wall_s"] - warm(empty),
                warm_cpu_s=statistics.median(c["cpu_s"] for c in filled["calls"][1:]),
                charge_pe_by_order=charges[0], max_charge_spread_pe=spread,
                calls=len(empty["calls"]))


def machine():
    info = dict(platform=platform.platform(), python=platform.python_version(),
                cpu_count=os.cpu_count())
    try:
        info["cpu"] = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                     capture_output=True, text=True).stdout.strip() or None
    except OSError:
        pass
    for module in ("numpy", "scipy", "numba", "lighthit"):
        try:
            info[module] = __import__(module).__version__
        except Exception:
            info[module] = None
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command")
    one = sub.add_parser("_child")
    one.add_argument("case", choices=sorted(CASES))
    one.add_argument("--calls", type=int, default=3)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cases", nargs="+", choices=sorted(CASES), default=list(CASES))
    parser.add_argument("--threads", default="1")
    parser.add_argument("--calls", type=int, default=3)
    args = parser.parse_args()
    if args.command == "_child":
        print("RESULT " + json.dumps(child(args.case, args.calls)), flush=True)
        return
    if args.output is None or args.output.exists():
        parser.error("--output must name a new file")
    if args.calls < 2:
        parser.error("at least two calls are needed for a warm event")
    from harness.record import commit
    from harness.inputs import data_dir
    data_dir()
    env = dict(os.environ)
    for name in THREAD_VARIABLES:
        env[name] = args.threads
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    rows, raw = [], {}
    for name in args.cases:
        with tempfile.TemporaryDirectory(prefix="numba-cache-") as cache:
            env["NUMBA_CACHE_DIR"] = cache
            raw[name] = {}
            for mode in ("empty-jit-cache", "filled-jit-cache"):
                load = list(os.getloadavg())
                out = subprocess.run([sys.executable, __file__, "_child", name, "--calls", str(args.calls)],
                                     env=env, capture_output=True, text=True)
                line = [x for x in out.stdout.splitlines() if x.startswith("RESULT ")]
                if out.returncode or not line:
                    raise SystemExit(f"{name} {mode} failed:\n{out.stderr[-3000:]}")
                raw[name][mode] = dict(json.loads(line[-1][7:]), load_average_before=load)
                print(name, mode, json.dumps({k: raw[name][mode][k] for k in ("import_", "setup")}),
                      [round(c["wall_s"], 3) for c in raw[name][mode]["calls"]], flush=True)
        rows.append(summarise(name, raw[name]))
    receiver = "centre"
    from harness.inputs import bundle_path
    manifest = bundle_path("tables", "shower-data.json")
    if manifest.is_file():
        receiver = (json.loads(manifest.read_text()).get("kernel_settings") or {}).get(
            "scattered_receiver", "centre")
    result = dict(schema="lighthit/route-timing/1", started=started,
                  finished=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  source=commit(), machine=machine(), threads=args.threads,
                  thread_variables={k: env[k] for k in THREAD_VARIABLES}, receiver=receiver,
                  definitions=dict(
                      cold_event="import + setup + first call in a fresh process",
                      warm_event="median of the later calls, each on a new source copy",
                      jit_share="first call minus warm event with an empty Numba cache; "
                                "includes first touches of memory-mapped tables",
                      file_cache="operating-system file cache not purged"),
                  rows=rows, raw=raw)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
