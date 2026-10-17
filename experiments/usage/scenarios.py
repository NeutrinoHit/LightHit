#!/usr/bin/env python
"""Run one default-route scenario on the standard inputs under the usage tracer.

    python experiments/usage/scenarios.py SCENARIO [--out DIR]

Scenarios: shower, muon, event, event-stopping-muon, laser, prompt, hybrid
(see SCENARIOS). One process per scenario, so every record holds only what
that route executes. The record goes to experiments/results/usage/<name>.json.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import usage_trace as trace  # noqa: E402


def _summary(response):
    import numpy as np
    from harness.metrics import three_orders
    charge, _ = three_orders(response)
    return dict(route=response.metadata.get("route"), method=response.metadata.get("method"),
                active=int(np.asarray(response.active).sum()),
                charge_pe_by_order=[float(x) for x in charge.sum(axis=0)],
                route_fallbacks=response.metadata.get("route_fallbacks"))


def shower():
    from harness import inputs as I
    from lighthit.shower import ShowerTransport
    kernel = I.route_kernel()
    return ShowerTransport(kernel)(I.electron(kernel))


def muon():
    import lighthit as lh
    from harness import inputs as I
    steps, ancestry, _ = I.muon_1tev()
    return lh.MuonTransport(I.route_kernel())(steps, ancestry, primary_uid=0)


def event():
    from harness import inputs as I
    from lighthit.event import EventTransport, SourceComponent
    kernel = I.route_kernel()
    steps, ancestry, _ = I.muon_1tev()
    parts = [SourceComponent("muon", steps, ancestry, primary_uid=0, label="muon"),
             SourceComponent("shower", I.electron(kernel), label="vertex shower")]
    return EventTransport(kernel)(parts)


def event_stopping_muon():
    from harness import inputs as I
    from lighthit.event import EventTransport
    steps, parts, _ = I.regenerated("muon-stopping")
    return EventTransport(I.route_kernel())(I.components(steps, parts))


def event_muon_near_om():
    from harness import inputs as I
    from lighthit.event import EventTransport
    steps, parts, _ = I.regenerated("muon-near-om")
    return EventTransport(I.route_kernel())(I.components(steps, parts))


def laser():
    from harness import inputs as I
    from lighthit.event import EventTransport
    return EventTransport(I.route_kernel(), laser_kernel=I.laser_kernel())(I.laser_flash())


def prompt():
    from harness import inputs as I
    kernel = I.route_kernel()
    return kernel.transport_prompt(I.electron(kernel))


def hybrid():
    from harness import inputs as I
    kernel = I.route_kernel()
    return kernel.transport_hybrid(I.electron(kernel))


SCENARIOS = {"shower": shower, "muon": muon, "event": event,
             "event-stopping-muon": event_stopping_muon,
             "event-muon-near-om": event_muon_near_om, "laser": laser,
             "prompt": prompt, "hybrid": hybrid}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scenario", choices=sorted(SCENARIOS))
    parser.add_argument("--out", type=Path, default=HERE.parent / "results" / "usage")
    args = parser.parse_args()
    import importlib.util
    trace.start(Path(importlib.util.find_spec("lighthit").origin).parent)
    began = time.perf_counter()
    try:
        result = _summary(SCENARIOS[args.scenario]())
    except Exception as exc:   # the record keeps what ran up to the error
        result = dict(error=f"{type(exc).__name__}: {exc}")
    elapsed = time.perf_counter() - began
    import numba
    record = trace.finish(args.out / f"{args.scenario}.json", args.scenario,
                          dict(result=result, scenario_seconds=elapsed,
                               numba_threads=numba.get_num_threads(),
                               numba_num_threads_env=os.environ.get("NUMBA_NUM_THREADS")))
    print(args.scenario, record["result"], f"{elapsed:.1f} s",
          len(record["python"]), "python functions,", len(record["numba_executed"]), "numba")


if __name__ == "__main__":
    main()
