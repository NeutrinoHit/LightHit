"""Shower-basis order >=2 versus the reference hybrid: accuracy and timing.

Reproducible entry point for ``research/documentation/research/shower-basis.md``.  Each
(kind, event, method) runs in a fresh process, so the peak RSS and the
first-call costs are those of that method alone.

Methods
-------
hybrid        ``transport_hybrid(source, frequency_nodes=65)`` (reference)
basis         ``transport_hybrid(source, frequency_nodes=65, ge2_basis=ShowerBasis(...))``

For ``basis`` one worker process runs all requested events: the first event
includes building (or memory-mapping, with ``--pack``) the order >=2 kernel
store, later events are warm.  ``--pack DIR`` writes the store after the
first event if DIR does not exist yet and memory-maps it otherwise.

Accuracy is per OM on the union of the active sets: integrated charge of
orders 0, 1, >=2 and total, early light (total within the first
``--early-ns`` after each OM's time origin), and L1 of the total and of the
order >=2 time bins before any hardware smearing.  Relative errors for
bright OMs, absolute errors (p.e.) for all.  Nothing is clipped, shifted or
renormalised.

    python scripts/shower_basis_benchmark.py --bgvd-model MODEL --cache CACHE65 \\
        --g4 e100=sim_e_100GeV_10.h5 --events e100:5,6,7 --output OUT
"""
import argparse
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

POSITION = [11.428243447430553, -249.74053298611113, 362.13987569444436]
DIRECTION = [0.4, 0.2, 0.894]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def kernel_for(args):
    import lighthit as lh
    model = lh.load_bgvd_model(args.bgvd_model, dataset="2021", clusters=args.cluster)
    config = lh.KernelConfig(
        omega_per_ns=np.linspace(0., args.omega_max, args.frequencies),
        readout_omega_per_ns=np.linspace(0., args.omega_max, args.readout_frequencies),
        relative_time_edges_ns=np.arange(-60., 745., 5.), cache_directory=args.cache,
        cache_policy="require")
    return model.kernel(config)


def worker(args):
    import lighthit as lh
    kernel = kernel_for(args)
    files = dict(item.split("=", 1) for item in args.g4)
    basis = lh.ShowerBasis(**json.loads(args.basis))
    report = {"method": args.method, "events": []}
    if args.method == "basis" and args.pack and Path(args.pack, "pack.json").exists():
        started = time.perf_counter()
        kernel.ge2_kernel_store(args.pack)
        report["pack_load_seconds"] = time.perf_counter() - started
    for spec in args.worker_events.split(","):
        kind, event = spec.split(":")
        source = lh.G4Shower.from_hdf5(files[kind], event=int(event)).placed(
            lh.SourcePose(args.position, args.direction, 0.0))
        started = time.perf_counter()
        if args.method == "hybrid":
            response = kernel.transport_hybrid(source, frequency_nodes=args.frequencies)
        else:
            response = kernel.transport_hybrid(source, frequency_nodes=args.frequencies,
                                               ge2_basis=basis)
        seconds = time.perf_counter() - started
        out = Path(args.output) / f"{args.method}_{kind}_{event}.npz"
        np.savez_compressed(out, charge=response.charge_components_pe,
                            bins=response.components_pe, active=response.active,
                            origin=response.time_origin_ns)
        meta = response.metadata
        report["events"].append({
            "kind": kind, "event": int(event), "seconds": seconds,
            "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "timings": {key: value for key, value in meta.items() if key.endswith("seconds")},
            "active": int(response.active.sum()),
            "basis": meta.get("ge2_basis_summary"),
            "store_mib": meta.get("ge2_kernel_store_bytes", 0) / 2 ** 20,
            "cells": meta.get("compiled_source_cells")})
        if args.method == "basis" and args.pack and not Path(args.pack, "pack.json").exists():
            started = time.perf_counter()
            kernel.ge2_kernel_store().save(args.pack)
            report["pack_save_seconds"] = time.perf_counter() - started
            report["pack_bytes"] = sum(p.stat().st_size for p in Path(args.pack).iterdir())
    Path(args.report).write_text(json.dumps(report, indent=1))


def compare(reference, test, *, early_ns, edges, bright=10):
    both = reference["active"] | test["active"]
    index = np.flatnonzero(both)
    relative_edges = edges
    early = (relative_edges[:-1] >= 0) & (relative_edges[1:] <= early_ns)
    total_ref = reference["charge"].sum(1)
    order = index[np.argsort(-total_ref[index])]
    out = {"active_reference": int(reference["active"].sum()),
           "active_test": int(test["active"].sum()),
           "active_lost": int((reference["active"] & ~test["active"]).sum()),
           "active_gained": int((test["active"] & ~reference["active"]).sum())}
    series = {"order0": [0], "order1": [1], "order_ge2": [2], "total": [0, 1, 2]}
    for name, columns in series.items():
        a = reference["charge"][:, columns].sum(1)
        b = test["charge"][:, columns].sum(1)
        rel = np.abs(b - a) / np.maximum(np.abs(a), 1e-300)
        out[f"{name}_charge_rel_max_bright"] = float(rel[order[:bright]].max())
        out[f"{name}_charge_rel_max"] = float(rel[index].max())
        out[f"{name}_charge_abs_max_pe"] = float(np.abs(b - a)[index].max())
        ba = reference["bins"][:, :, columns].sum(2)
        bb = test["bins"][:, :, columns].sum(2)
        l1 = np.abs(bb - ba).sum(1) / np.maximum(np.abs(ba).sum(1), 1e-300)
        out[f"{name}_L1_max_bright"] = float(l1[order[:bright]].max())
        out[f"{name}_L1_max"] = float(l1[index].max())
        out[f"{name}_bins_abs_max_pe"] = float(np.abs(bb - ba)[index].max())
    ea = reference["bins"][:, early].sum((1, 2))
    eb = test["bins"][:, early].sum((1, 2))
    out["early_light_abs_max_pe"] = float(np.abs(eb - ea)[index].max())
    out["early_light_rel_max_bright"] = float(
        (np.abs(eb - ea) / np.maximum(np.abs(ea), 1e-300))[order[:bright]].max())
    out["negative_ge2_bins_pe_reference"] = float(np.minimum(reference["bins"][:, :, 2], 0).sum())
    out["negative_ge2_bins_pe_test"] = float(np.minimum(test["bins"][:, :, 2], 0).sum())
    out["per_om"] = [{"om": int(i), "total_ref": float(total_ref[i]),
                      "ge2_ref": float(reference["charge"][i, 2]),
                      "ge2_test": float(test["charge"][i, 2])} for i in order]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bgvd-model", required=True)
    parser.add_argument("--cache", required=True, help="65-frequency directional cache")
    parser.add_argument("--g4", action="append", required=True, help="kind=path.h5")
    parser.add_argument("--events", default="e100:5", help="kind:event[,event...];...")
    parser.add_argument("--methods", default="hybrid,basis")
    parser.add_argument("--basis", default="{}", help="ShowerBasis keyword JSON")
    parser.add_argument("--pack", default=None, help="order >=2 kernel pack directory")
    parser.add_argument("--cluster", type=int, default=1)
    parser.add_argument("--frequencies", type=int, default=65)
    parser.add_argument("--readout-frequencies", type=int, default=321)
    parser.add_argument("--omega-max", type=float, default=1.2)
    parser.add_argument("--early-ns", type=float, default=20.0)
    parser.add_argument("--position", type=float, nargs=3, default=POSITION)
    parser.add_argument("--direction", type=float, nargs=3, default=DIRECTION)
    parser.add_argument("--reuse", action="store_true",
                        help="skip (method, event) outputs that already exist")
    parser.add_argument("--output", required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--method", help=argparse.SUPPRESS)
    parser.add_argument("--worker-events", help=argparse.SUPPRESS)
    parser.add_argument("--report", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        return worker(args)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    events = []
    for group in args.events.split(";"):
        kind, numbers = group.split(":")
        events += [f"{kind}:{n}" for n in numbers.split(",")]
    files = dict(item.split("=", 1) for item in args.g4)
    import lighthit
    import numba
    summary = {"schema": "lighthit/shower-basis-benchmark/1",
               "lighthit_version": lighthit.__version__, "python": platform.python_version(),
               "numpy": np.__version__, "numba": numba.__version__,
               "threads": numba.get_num_threads(), "cpus": os.cpu_count(),
               "platform": platform.platform(), "basis": json.loads(args.basis),
               "inputs": {kind: {"file": Path(path).name, "sha256": sha256(path)}
                          for kind, path in files.items()},
               "pose": {"position_m": args.position, "direction": args.direction},
               "cluster": args.cluster, "runs": {}, "comparisons": {}}
    common = [sys.executable, __file__, "--worker", "--bgvd-model", args.bgvd_model,
              "--cache", args.cache, "--basis", args.basis, "--cluster", str(args.cluster),
              "--frequencies", str(args.frequencies), "--readout-frequencies",
              str(args.readout_frequencies), "--omega-max", str(args.omega_max),
              "--output", str(output), "--position", *map(str, args.position),
              "--direction", *map(str, args.direction)]
    for item in args.g4:
        common += ["--g4", item]
    if args.pack:
        common += ["--pack", args.pack]
    for method in args.methods.split(","):
        todo = [e for e in events if not (args.reuse and (
            output / f"{method}_{e.replace(':', '_')}.npz").exists())]
        groups = [[e] for e in todo] if method == "hybrid" else ([todo] if todo else [])
        for group in groups:
            report = output / f"report_{method}_{'_'.join(g.replace(':', '-') for g in group)}.json"
            started = time.perf_counter()
            process = subprocess.run(common + ["--method", method, "--worker-events",
                                               ",".join(group), "--report", str(report)],
                                     capture_output=True, text=True)
            if process.returncode:
                print(process.stderr[-3000:], file=sys.stderr)
                raise SystemExit(f"{method} {group} failed")
            data = json.loads(report.read_text())
            data["process_seconds"] = time.perf_counter() - started
            summary["runs"][f"{method}:{','.join(group)}"] = data
            for entry in data["events"]:
                print(method, entry["kind"], entry["event"], f"{entry['seconds']:.1f} s",
                      f"peak {entry['peak_rss_mib']:.0f} MiB", flush=True)
    edges = np.arange(-60., 745., 5.)
    for spec in events:
        stem = spec.replace(":", "_")
        paths = [output / f"{m}_{stem}.npz" for m in ("hybrid", "basis")]
        if all(p.exists() for p in paths):
            summary["comparisons"][spec] = compare(dict(np.load(paths[0])), dict(np.load(paths[1])),
                                                   early_ns=args.early_ns, edges=edges)
    (output / "summary.json").write_text(json.dumps(summary, indent=1))
    print("summary:", output / "summary.json")


if __name__ == "__main__":
    main()
