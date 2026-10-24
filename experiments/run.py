#!/usr/bin/env python
"""Run measured cases of the idea registry.

    python experiments/run.py list [PATTERN]          # cases, their idea and saved result
    python experiments/run.py run CASE [CASE ...]      # each in its own process, one after another
          [--threads N | --threads default] [--force]

CASE is ``idea/case`` or a glob over it (``'shower-*/*'``). Every case runs in a
fresh process with the thread count fixed through NUMBA_NUM_THREADS,
OMP_NUM_THREADS, OPENBLAS_NUM_THREADS and VECLIB_MAXIMUM_THREADS (default 1;
``--threads default`` leaves them unset). A case is called once to warm up
(Numba compilation, table loading) unless it opts out, then timed; its response
is saved to experiments/results/<idea>/<case>.npz and compared with its
reference case's saved response. Results: experiments/results/<idea>/<case>.json.
Cases run strictly one at a time; nothing heavy runs in parallel.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

THREAD_VARIABLES = ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS")


def _select(cases, patterns):
    keys = []
    for pattern in patterns:
        found = [k for k in cases if fnmatch.fnmatchcase(k, pattern)]
        if not found:
            raise SystemExit(f"no case matches {pattern!r}")
        keys += [k for k in found if k not in keys]
    return keys


def _suffix(key, threads):
    """References are stored under their plain name at any thread count: their
    timing is not compared, only their response is used."""
    return "" if threads == "1" or key.startswith("reference-") else f"@threads-{threads}"


def _result_path(key, threads):
    from harness.record import RESULTS
    idea, name = key.split("/", 1)
    suffix = _suffix(key, threads)
    return RESULTS / idea / f"{name}{suffix}.json"


def cmd_list(args):
    from harness.cases import load_all
    cases = load_all()
    for key in sorted(cases):
        if args.pattern and not fnmatch.fnmatchcase(key, args.pattern):
            continue
        c = cases[key]
        done = _result_path(key, "1").is_file()
        print(f"{'*' if done else ' '} {key:60s} ref={c.reference or '-':40s}"
              f"{' heavy' if c.heavy else ''}")


def cmd_run(args):
    from harness.cases import load_all
    cases = load_all()
    keys = _select(cases, args.cases)
    for key in keys:
        path = _result_path(key, args.threads)
        if not args.force and path.is_file() and json.loads(path.read_text()).get("status") == "ok":
            print(f"skip {key} (result exists; --force to rerun)")
            continue
        env = dict(os.environ)
        for name in THREAD_VARIABLES:
            env.pop(name, None)
            if args.threads != "default":
                env[name] = args.threads
        print(f"== {key} threads={args.threads}", flush=True)
        status = subprocess.run([sys.executable, __file__, "_one", key, "--threads", args.threads],
                                env=env).returncode
        if status:
            print(f"   FAILED ({status})", flush=True)
            if not args.keep_going:
                raise SystemExit(status)


def cmd_one(args):
    """Run a single case in this process (called by ``run``)."""
    import traceback
    from harness import metrics
    from harness.cases import load_all
    from harness.record import RESULTS, load_response, save_response, timed, write_result
    case = load_all()[args.case]
    from harness.context import Context
    ctx = Context(case)
    payload = dict(threads=args.threads, kind=case.kind, reference=case.reference,
                   notes=case.notes, warmup=case.warmup)
    suffix = _suffix(args.case, args.threads)
    try:
        if case.warmup:
            with timed(payload, "warmup"):
                case.function(ctx)
        runs = []
        result = None
        for _ in range(max(1, case.repeat)):
            entry = {}
            with timed(entry):
                result = case.function(ctx)
            runs.append(entry["timing"])
        payload["timing"] = min(runs, key=lambda r: r["wall_s"])
        payload["timings"] = runs
        payload["inputs"] = ctx.inputs
        if case.kind == "response":
            path = save_response(result, RESULTS / case.idea / f"{case.name}{suffix}.npz")
            payload["response"] = str(path.relative_to(RESULTS))
            payload["summary"] = _summary(result)
            if case.reference:
                ref = RESULTS / f"{case.reference}.npz"
                if ref.is_file():
                    payload["accuracy"] = metrics.compare(result, load_response(ref))
                else:
                    payload["accuracy_missing"] = f"reference {case.reference} not computed"
        else:
            if isinstance(result, dict) and "timing" in result:
                payload["timing"] = result.pop("timing")      # a subprocess's own timing
                payload["timings"] = [payload["timing"]]
            payload["metrics"] = result
            if isinstance(result, dict) and result.get("exit_status"):
                raise RuntimeError(f"script exited with {result['exit_status']}: "
                                   f"{result.get('stderr_tail', '')[-600:]}")
        payload["status"] = "ok"
    except Exception as exc:  # recorded, not hidden
        payload["status"] = "error"
        payload["error"] = f"{type(exc).__name__}: {exc}"
        payload["traceback"] = traceback.format_exc()[-4000:]
    path = write_result(case.idea, case.name + suffix, payload)
    timing = payload.get("timing", {})
    print(f"   {payload['status']}: {os.path.relpath(path, HERE)} wall={timing.get('wall_s', float('nan')):.2f}s "
          f"cpu={timing.get('cpu_s', float('nan')):.2f}s rss={timing.get('peak_rss_mib', float('nan')):.0f}MiB"
          + (f" {payload.get('error')}" if payload["status"] != "ok" else ""), flush=True)
    if payload.get("accuracy"):
        print("   accuracy:", json.dumps(metrics.headline(payload["accuracy"])), flush=True)
    raise SystemExit(0 if payload["status"] == "ok" else 1)


def _summary(response):
    import numpy as np
    from harness.metrics import three_orders
    charge, _ = three_orders(response)
    meta = getattr(response, "metadata", {}) or {}
    return dict(active=int(np.asarray(response.active).sum()),
                charge_pe_by_order=[float(x) for x in charge.sum(axis=0)],
                route=meta.get("route"), method=meta.get("method"),
                route_fallbacks=meta.get("route_fallbacks"))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("list")
    p.add_argument("pattern", nargs="?")
    p.set_defaults(func=cmd_list)
    p = sub.add_parser("run")
    p.add_argument("cases", nargs="+")
    p.add_argument("--threads", default="1")
    p.add_argument("--force", action="store_true")
    p.add_argument("--keep-going", action="store_true")
    p.set_defaults(func=cmd_run)
    p = sub.add_parser("_one")
    p.add_argument("case")
    p.add_argument("--threads", default="1")
    p.set_defaults(func=cmd_one)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
