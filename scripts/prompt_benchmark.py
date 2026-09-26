"""Prompt (orders 0 + 1) versus full LightHit transport: validation and timing.

Every (case, method) runs in a fresh Python process so that peak RSS and the
first-call (JIT / compile) cost are measured in isolation.  Each worker calls
the transport twice on the same kernel: the first call includes numba
compilation (or loading it from the on-disk cache) and any per-process
preparation; the second is the warm per-event cost.

Cases
-----
track          CherenkovTrack of ``--track-length`` m (default 120 m)
shower-N       SyntheticShower.gaussian with N elements (``--shower-sizes``)
electron       stored G4 event (``--g4-electron``, ``--event``)
muon           stored G4 event (``--g4-muon``, ``--event``)

Methods
-------
prompt         ``kernel.transport_prompt`` on the whole event
full           ``kernel.transport`` on the whole event (cache required)
full-chunked   ``kernel.transport`` on slabs of ``--chunk-m`` that share the
               whole event's time origin (see ``chunk_source``), summed
prompt-slabs,  both methods on the same plain slabs of ``--chunk-m``, each
full-slabs     with its own time origin; charges are compared summed over
               slabs and time bins slab by slab.  This is how the full path
               can be run on a long event whose whole-event compiled source
               does not fit in memory or fails its radial-range check.

All cases use the BGVD model (``--bgvd-model``) with the production
``KernelConfig`` (9 wavelengths, 5 ns bins from -60 to 740 ns, 0.01 p.e.
threshold); the full method reads the prepared directional caches from
``--cache`` with ``cache_policy="require"`` and never builds them.  Sources
are placed with the same explicit pose for both methods.

Usage::

    python scripts/prompt_benchmark.py --bgvd-model DIR --g4-electron E.h5 \\
        --g4-muon MU.h5 --cache CACHE --output OUT \\
        --cases track,shower-1000,shower-10000,electron,muon

Writes ``OUT/<case>-<method>.npz`` (responses), ``OUT/<case>-<method>.json``
(worker report) and ``OUT/summary.json`` (comparison of all cases).
"""
from __future__ import annotations

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

POSITION = [5.0, -269.5, 330.0]
DIRECTION = [0.35, -0.2, 0.915]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_sha256(directory):
    digest = hashlib.sha256()
    for path in sorted(Path(directory).rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts and path.name != ".DS_Store":
            digest.update(str(path.relative_to(directory)).encode())
            digest.update(sha256(path).encode())
    return digest.hexdigest()


def make_source(args, case):
    import lighthit as lh
    pose = lh.SourcePose(POSITION, DIRECTION, 0.0)
    if case == "track":
        return lh.CherenkovTrack.centered(POSITION, DIRECTION, args.track_length), {
            "type": "CherenkovTrack", "length_m": args.track_length, "beta": 1.0}
    if case.startswith("shower-"):
        count = int(case.split("-", 1)[1])
        source = lh.SyntheticShower.gaussian(pose, charged_track_length_m=500.0,
                                             elements=count, seed=1)
        return source, {"type": "SyntheticShower", "elements": count,
                        "charged_track_length_m": 500.0, "seed": 1}
    path = args.g4_electron if case == "electron" else args.g4_muon
    started = time.perf_counter()
    source = lh.G4Shower.from_hdf5(path, event=args.event).placed(pose)
    return source, {"type": "G4Shower", "file": str(path), "event": args.event,
                    "elements": len(source.elements),
                    "read_seconds": time.perf_counter() - started}


def slab_groups(elements, chunk_m):
    """Element indices of consecutive ``chunk_m`` slabs along the principal axis."""
    axis = elements.principal_axis
    z = (elements.midpoints_m - elements.centroid_m) @ axis
    edges = np.arange(z.min(), z.max() + chunk_m, chunk_m)
    index = np.clip(np.digitize(z, edges) - 1, 0, len(edges) - 2)
    groups = [np.flatnonzero(index == k) for k in range(len(edges) - 1)]
    return [g for g in groups if len(g)]


def slab_sources(source, chunk_m):
    """Plain slabs (each with its own time origin) whose sum is the source."""
    import lighthit as lh
    out = []
    for g in slab_groups(source.elements, chunk_m):
        mask = np.zeros(len(source.elements), bool)
        mask[g] = True
        out.append(lh.G4Shower(source.elements.subset(mask), source.input_path, source.event))
    return out


def chunk_source(source, chunk_m):
    """Slabs of ``chunk_m`` along the principal axis whose sum is the source.

    Every slab also carries the globally earliest element with ``1/n`` of its
    light, so that the full path assigns every slab exactly the same time
    origin (earliest element start and position) and the slabs add up to the
    original source by linearity.
    """
    from dataclasses import replace
    import lighthit as lh
    elements = source.elements
    groups = slab_groups(elements, chunk_m)
    first = int(np.argmin(elements.start_ns))
    count = len(groups)
    out = []
    for g in groups:
        g = np.union1d(g, [first])
        c0 = elements.coefficient0.copy()
        c2 = elements.coefficient2.copy()
        c0[first] /= count
        c2[first] /= count
        piece = replace(elements, coefficient0=c0, coefficient2=c2)
        mask = np.zeros(len(elements), bool)
        mask[g] = True
        out.append(lh.G4Shower(piece.subset(mask), source.input_path, source.event))
    return out


def transport_chunked(kernel, chunks):
    """Full transport of every slab, summed (charges and bins are linear)."""
    total = None
    for piece in chunks:
        response = kernel.transport(piece)
        if total is None:
            total = response
            charge = response.charge_components_pe.copy()
            bins = response.components_pe.copy()
            active = response.active.copy()
            origin = response.time_origin_ns
        else:
            if not np.array_equal(response.time_origin_ns, origin):
                raise ValueError("slab time origins differ")
            charge += response.charge_components_pe
            bins += response.components_pe
            active |= response.active
    total.charge_components_pe = charge
    total.components_pe = bins
    total.active = active & (charge.sum(axis=1) >= kernel.config.threshold_pe)
    total.metadata = {**total.metadata, "chunked_slabs": len(chunks)}
    return total


def worker(args):
    import numba
    import lighthit as lh
    if args.threads:
        numba.set_num_threads(args.threads)
    bgvd = lh.load_bgvd_model(args.bgvd_model, dataset="2021")
    source, description = make_source(args, args.case)
    report = {"case": args.case, "method": args.method, "source": description,
              "threads": numba.get_num_threads(), "calls": []}
    if args.method.startswith("full"):
        config = lh.KernelConfig(cache_directory=args.cache, cache_policy="require")
    else:
        config = lh.KernelConfig()
    kernel = bgvd.kernel(config)
    prompt_config = lh.PromptConfig(**json.loads(args.prompt_config or "{}"))
    response = None
    if args.method == "full-chunked":
        chunks = chunk_source(source, args.chunk_m)
        report["chunks"] = {"count": len(chunks), "chunk_m": args.chunk_m,
                            "elements": [len(c.elements) for c in chunks]}
    if args.method.endswith("-slabs"):
        chunks = slab_sources(source, args.chunk_m)
        report["chunks"] = {"count": len(chunks), "chunk_m": args.chunk_m,
                            "elements": [len(c.elements) for c in chunks]}
        return slab_worker(args, kernel, prompt_config, chunks, report)
    for call in range(args.repeat):
        started = time.perf_counter()
        if args.method == "full":
            response = kernel.transport(source)
        elif args.method == "full-chunked":
            response = transport_chunked(kernel, chunks)
        else:
            response = kernel.transport_prompt(source, prompt_config)
        elapsed = time.perf_counter() - started
        report["calls"].append({"seconds": elapsed,
                                "peak_rss_mib": resource.getrusage(
                                    resource.RUSAGE_SELF).ru_maxrss / 1024})
    meta = response.metadata
    if args.method.startswith("full"):
        charge = response.charge_components_pe
        bins = response.components_pe
    else:
        charge = np.zeros((len(response.detector), 3))
        charge[:, :2] = response.charge_orders_pe
        bins = np.zeros(response.bins_orders_pe.shape[:2] + (3,))
        bins[:, :, :2] = response.bins_orders_pe
        report["order1_level"] = meta.get("order1_level")
    np.savez_compressed(args.out_npz, charge=charge, bins=bins,
                        active=response.active, origin=response.time_origin_ns)
    report["metadata"] = {key: value for key, value in meta.items()
                          if key not in ("order1_level",)}
    report["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    Path(args.out_json).write_text(json.dumps(report, indent=1, default=_default))


def _arrays(response):
    """(charge (N, 3), bins (N, nb, 3)) of a full or prompt response."""
    if hasattr(response, "charge_components_pe"):
        return response.charge_components_pe, response.components_pe
    charge = np.zeros((len(response.detector), 3))
    charge[:, :2] = response.charge_orders_pe
    bins = np.zeros(response.bins_orders_pe.shape[:2] + (3,))
    bins[:, :, :2] = response.bins_orders_pe
    return charge, bins


def slab_worker(args, kernel, prompt_config, chunks, report):
    """Every slab separately; responses stacked along a leading slab axis."""
    full = args.method.startswith("full")
    for call in range(args.repeat):
        started = time.perf_counter()
        responses = [kernel.transport(piece) if full
                     else kernel.transport_prompt(piece, prompt_config) for piece in chunks]
        report["calls"].append({"seconds": time.perf_counter() - started,
                                "peak_rss_mib": resource.getrusage(
                                    resource.RUSAGE_SELF).ru_maxrss / 1024})
    arrays = [_arrays(r) for r in responses]
    np.savez_compressed(args.out_npz, charge=np.stack([a[0] for a in arrays]),
                        bins=np.stack([a[1] for a in arrays]),
                        active=np.stack([r.active for r in responses]),
                        origin=np.stack([r.time_origin_ns for r in responses]))
    report["metadata"] = {"slabs": len(chunks)}
    report["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    Path(args.out_json).write_text(json.dumps(report, indent=1, default=_default))


def compare_slabs(prompt, full, threshold=0.01):
    """Charges summed over slabs; time bins slab by slab (same origins)."""
    if not np.array_equal(prompt["origin"], full["origin"]):
        raise ValueError("prompt and full slab time origins differ")
    summed = compare({"charge": prompt["charge"].sum(axis=0), "bins": prompt["bins"][0] * 0,
                      "active": prompt["active"].any(axis=0)},
                     {"charge": full["charge"].sum(axis=0), "bins": full["bins"][0] * 0,
                      "active": full["active"].any(axis=0)}, threshold)
    summed.pop("representative_bins")
    ranking = np.argsort(-full["charge"][:, :, :2].sum(axis=(0, 2)))[:5]
    rows = []
    for i in ranking:
        entry = {"om": int(i)}
        for name, columns in (("order0", [0]), ("order1", [1]), ("0+1", [0, 1])):
            a = prompt["bins"][:, i][:, :, columns].sum(axis=2)
            b = full["bins"][:, i][:, :, columns].sum(axis=2)
            entry[name] = {"slabwise_l1_over_full_window": float(
                               np.abs(a - b).sum() / max(np.abs(b).sum(), 1e-300)),
                           "full_window_pe": float(b.sum()),
                           "prompt_window_pe": float(a.sum()),
                           "full_negative_bins_pe": float(b[b < 0].sum())}
        rows.append(entry)
    summed["representative_bins_slabwise"] = rows
    return summed


def _default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return str(value)


def compare(prompt, full, threshold=0.01):
    """Order 0, 1, 0+1 per OM on the modules both methods computed."""
    both = prompt["active"] | full["active"]
    out = {}
    for name, columns in (("order0", [0]), ("order1", [1]), ("prompt_0+1", [0, 1])):
        a = prompt["charge"][:, columns].sum(axis=1)
        b = full["charge"][:, columns].sum(axis=1)
        scale = np.maximum(np.abs(b), threshold)
        difference = (a - b) / scale
        out[name] = {
            "total_prompt_pe": float(a.sum()), "total_full_pe": float(b.sum()),
            "total_relative": float(a.sum() / b.sum() - 1) if b.sum() else None,
            "max_abs_diff_over_max(full,0.01)": float(np.abs(difference[both]).max())
            if both.any() else 0.0,
            "median_abs_diff_over_max(full,0.01)": float(np.median(np.abs(difference[both])))
            if both.any() else 0.0,
            "bright_om_relative": [
                {"om": int(i), "prompt_pe": float(a[i]), "full_pe": float(b[i]),
                 "relative": float(a[i] / b[i] - 1) if b[i] else None}
                for i in np.argsort(-b)[:8]],
        }
    # time bins for representative hit OMs
    ranking = np.argsort(-full["charge"][:, :2].sum(axis=1))[:5]
    rows = []
    for i in ranking:
        entry = {"om": int(i)}
        for name, columns in (("order0", [0]), ("order1", [1]), ("0+1", [0, 1])):
            a = prompt["bins"][i][:, columns].sum(axis=1)
            b = full["bins"][i][:, columns].sum(axis=1)
            entry[name] = {"l1_over_full_window": float(np.abs(a - b).sum()
                                                        / max(np.abs(b).sum(), 1e-300)),
                           "full_window_pe": float(b.sum()),
                           "prompt_window_pe": float(a.sum()),
                           "full_negative_bins_pe": float(b[b < 0].sum())}
        rows.append(entry)
    out["representative_bins"] = rows
    out["active_prompt"] = int(prompt["active"].sum())
    out["active_full"] = int(full["active"].sum())
    out["active_both"] = int((prompt["active"] & full["active"]).sum())
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bgvd-model", required=True)
    parser.add_argument("--g4-electron")
    parser.add_argument("--g4-muon")
    parser.add_argument("--event", type=int, default=0)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output")
    parser.add_argument("--cases", default="track,shower-1000,shower-10000,electron,muon")
    parser.add_argument("--methods", default="prompt,full")
    parser.add_argument("--track-length", type=float, default=120.0)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--chunk-m", type=float, default=10.0,
                        help="slab length for the full-chunked method")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--prompt-config", default="",
                        help="JSON dict of PromptConfig overrides")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--case", help=argparse.SUPPRESS)
    parser.add_argument("--method", help=argparse.SUPPRESS)
    parser.add_argument("--out-npz", help=argparse.SUPPRESS)
    parser.add_argument("--out-json", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        return worker(args)
    if not args.output:
        parser.error("--output is required")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    inputs = {"bgvd_model_tree_sha256": tree_sha256(args.bgvd_model)}
    for key in ("g4_electron", "g4_muon"):
        path = getattr(args, key)
        if path:
            inputs[key] = {"path": str(path), "sha256": sha256(path)}
    import lighthit
    import numba
    import scipy
    summary = {"schema": "lighthit/prompt-benchmark/1",
               "lighthit_version": lighthit.__version__,
               "python": platform.python_version(), "numpy": np.__version__,
               "scipy": scipy.__version__, "numba": numba.__version__,
               "machine": {"platform": platform.platform(), "cpus": os.cpu_count()},
               "inputs": inputs, "pose": {"position_m": POSITION, "direction": DIRECTION},
               "event": args.event, "prompt_config_overrides": args.prompt_config or "{}",
               "cases": {}}
    for case in args.cases.split(","):
        results = {}
        for method in args.methods.split(","):
            stem = output / f"{case}-{method}"
            command = [sys.executable, __file__, "--worker", "--case", case,
                       "--method", method, "--bgvd-model", args.bgvd_model,
                       "--cache", args.cache, "--event", str(args.event),
                       "--track-length", str(args.track_length),
                       "--repeat", str(args.repeat), "--threads", str(args.threads),
                       "--chunk-m", str(args.chunk_m),
                       "--out-npz", f"{stem}.npz", "--out-json", f"{stem}.json"]
            if args.prompt_config:
                command += ["--prompt-config", args.prompt_config]
            for key in ("g4_electron", "g4_muon"):
                if getattr(args, key):
                    command += [f"--{key.replace('_', '-')}", getattr(args, key)]
            started = time.perf_counter()
            process = subprocess.run(command, capture_output=True, text=True)
            wall = time.perf_counter() - started
            if process.returncode != 0:
                results[method] = {"error": process.stderr[-4000:], "wall_seconds": wall}
                print(f"{case} {method}: FAILED\n{process.stderr[-2000:]}", flush=True)
                continue
            report = json.loads(Path(f"{stem}.json").read_text())
            report["process_wall_seconds"] = wall
            results[method] = report
            calls = ", ".join(f"{c['seconds']:.2f}s" for c in report["calls"])
            print(f"{case} {method}: calls {calls}; peak RSS "
                  f"{report['peak_rss_mib']:.0f} MiB; process {wall:.1f}s", flush=True)
        entry = {"reports": {m: {k: v for k, v in r.items() if k != "metadata"}
                             | {"metadata_timings": {k: v for k, v in r.get("metadata", {}).items()
                                                     if "seconds" in k or k == "timings"}}
                             for m, r in results.items()}}
        for reference in ("full", "full-chunked"):
            if all(m in results and "error" not in results[m] for m in ("prompt", reference)):
                prompt = dict(np.load(output / f"{case}-prompt.npz"))
                full = dict(np.load(output / f"{case}-{reference}.npz"))
                entry[f"comparison_vs_{reference}"] = compare(prompt, full)
                entry[f"same_time_origin_vs_{reference}"] = bool(
                    np.array_equal(prompt["origin"], full["origin"]))
        if all(m in results and "error" not in results[m]
               for m in ("prompt-slabs", "full-slabs")):
            entry["comparison_slabs"] = compare_slabs(
                dict(np.load(output / f"{case}-prompt-slabs.npz")),
                dict(np.load(output / f"{case}-full-slabs.npz")))
        if all(m in results and "error" not in results[m] for m in ("prompt", "prompt-slabs")):
            whole = dict(np.load(output / f"{case}-prompt.npz"))
            slabs = dict(np.load(output / f"{case}-prompt-slabs.npz"))
            q_whole, q_slabs = whole["charge"][:, :2], slabs["charge"][:, :, :2].sum(axis=0)
            entry["prompt_slab_sum_vs_whole"] = {
                "max_abs_diff_over_max(q,0.01)": float(np.max(
                    np.abs(q_slabs - q_whole) / np.maximum(q_whole, 0.01))),
                "total_order1_relative": float(q_slabs[:, 1].sum() / q_whole[:, 1].sum() - 1)}
        if all(m in results and "error" not in results[m] for m in ("full", "full-chunked")):
            entry["comparison_chunked_vs_full"] = compare(
                dict(np.load(output / f"{case}-full-chunked.npz")),
                dict(np.load(output / f"{case}-full.npz")))
        summary["cases"][case] = entry
        (output / "summary.json").write_text(json.dumps(summary, indent=1, default=_default))
    print("summary:", output / "summary.json")


if __name__ == "__main__":
    main()
