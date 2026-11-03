"""Export a compact results report without running transport calculations."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import tomllib

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(f"{value:.8g}") if np.isfinite(value) else None
    if isinstance(value, np.generic):
        return value.item()
    return value


def read_result(path):
    v = json.loads(path.read_text())
    return {k: v.get(k) for k in ("idea", "case", "date", "source", "status", "kind", "reference", "timing", "metrics", "accuracy", "summary")}


def read_arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {k: archive[k] for k in archive.files if k != "metadata"}


def leaves(value, prefix="", limit=8):
    found = []
    def walk(v, key):
        if len(found) >= limit:
            return
        if isinstance(v, dict):
            for k, x in v.items():
                walk(x, f"{key}.{k}" if key else k)
        elif isinstance(v, (bool, int, float)) or v is None:
            found.append([key, v])
    walk(value, prefix)
    return found


def build(atlas_report=None):
    results_dir = ROOT / "experiments/results"
    sources = {}
    def track(path):
        sources[str(path.relative_to(ROOT))] = digest(path)
    records = {}
    for path in sorted(results_dir.glob("*/*.json")):
        try:
            record = read_result(path)
        except (ValueError, OSError):
            continue
        if record.get("idea") and record.get("case"):
            track(path)
            records.setdefault(record["idea"], []).append(record)
    ideas = []
    for path in sorted((ROOT / "experiments/ideas").glob("*.toml")):
        item = tomllib.loads(path.read_text())
        track(path)
        saved = records.get(item["id"], [])
        saved.sort(key=lambda r: (r.get("date") or "", r.get("case") or ""), reverse=True)
        ideas.append({
            **{k: item.get(k, "") for k in ("id", "title", "area", "kind", "status", "question", "inputs", "how-to-run", "code")},
            "source": str(path.relative_to(ROOT)), "sha256": digest(path),
            "runs": [{"case": r["case"], "date": r["date"], "source": r["source"], "status": r["status"],
                      "metrics": leaves(r.get("metrics") or r.get("accuracy") or r.get("summary") or {}),
                      "wall_s": (r.get("timing") or {}).get("wall_s")} for r in saved],
        })
    spec = importlib.util.spec_from_file_location("stored_response_metrics", ROOT / "experiments/harness/metrics.py")
    metric_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metric_module)
    track(ROOT / "experiments/harness/metrics.py")
    examples = []
    for idea, case, title in (
        ("default-shower-route", "electron-100GeV", "100 GeV electron"),
        ("default-muon-route", "muon-100GeV", "100 GeV muon"),
        ("default-laser-route", "laser-532nm", "532 nm calibration flash"),
    ):
        json_path = results_dir / idea / f"{case}.json"
        npz_path = json_path.with_suffix(".npz")
        record = read_result(json_path)
        arrays = read_arrays(npz_path)
        track(json_path); track(npz_path)
        q = arrays["charge_components_pe"]
        active = np.flatnonzero(arrays["active"])
        ranked = active[np.argsort(q[active].sum(1), kind="stable")[::-1]]
        sample_positions = np.unique(np.round(np.linspace(0, len(ranked) - 1, 6)).astype(int))
        selected = ranked[sample_positions]
        example = {
            "id": case, "title": title, "idea": idea, "date": record["date"], "source": record["source"],
            "route": (record.get("summary") or {}).get("route"),
            "record_path": str(json_path.relative_to(ROOT)), "response_path": str(npz_path.relative_to(ROOT)),
            "response_sha256": digest(npz_path), "timing": record["timing"],
            "total_by_order": q.sum(0), "active_count": int(len(active)), "module_count": len(q),
            "charge_ranked": [{"index": int(i), "charge": q[i]} for i in ranked],
            "time_edges_ns": arrays["relative_time_edges_ns"],
            "traces": [{"index": int(i), "origin_ns": float(arrays["time_origin_ns"][i]), "charge": q[i],
                        "bins": arrays["components_pe"][i]} for i in selected],
            "reference": record["reference"], "comparison": None,
        }
        if record["reference"]:
            reference_path = results_dir / (record["reference"] + ".npz")
            if reference_path.exists():
                track(reference_path)
                reference = read_arrays(reference_path)
                example["comparison"] = metric_module.compare(arrays, reference, floor_pe=0.01)
                compatible_bins = (example["comparison"]["same_clock"]
                                   and "components_pe" in reference
                                   and reference["components_pe"].shape == arrays["components_pe"].shape
                                   and np.array_equal(reference.get("relative_time_edges_ns"), arrays["relative_time_edges_ns"]))
                example["reference_bins_match"] = bool(compatible_bins)
                if compatible_bins:
                    covered = reference.get("covered", np.ones(len(q), dtype=bool))
                    for trace in example["traces"]:
                        i = trace["index"]
                        if covered[i] and reference["active"][i] and arrays["time_origin_ns"][i] == reference["time_origin_ns"][i]:
                            trace["reference_bins"] = reference["components_pe"][i]
        examples.append(example)
    matrix = []
    for path in sorted((results_dir / "photon-mc-matrix").glob("*.json")):
        record = read_result(path)
        if not record.get("metrics", {}).get("by_distance"):
            continue
        track(path)
        metrics = record["metrics"]
        matrix.append({"id": record["case"], "date": record["date"], "source": record["source"],
                       "path": str(path.relative_to(ROOT)), "sha256": digest(path),
                       "headline": metrics["headline"], "totals": metrics["totals"],
                       "rows": metrics["by_distance"], "same_steps": metrics["same_steps_in_all_runs"],
                       "runs": metrics["mc"]["runs"]})
    status_path = atlas_report
    time = None
    if status_path is not None:
        raw = json.loads(status_path.read_text())
        sources["checks/time-atlas.json"] = digest(status_path)
        time = {"passed": raw["passed"], "complete": raw["complete"],
                "rows": [{k: r[k] for k in ("radius_m", "passed", "max_scaled_error")} for r in raw["results"]]}
    all_dates = sorted(r["date"] for rr in records.values() for r in rr if r.get("date"))
    def result_set(title, items, description):
        dates = sorted(i["date"] for i in items if i.get("date"))
        commits = sorted({i["source"]["commit"] for i in items})
        dirty = any(i["source"].get("src_modified") for i in items)
        return {"title": title, "date_min": dates[0], "date_max": dates[-1],
                "code": ", ".join(commits) + (" + local edits" if dirty else ""),
                "description": description}
    sets = {"examples": result_set("Detector examples", examples, "Stored event responses"),
            "mc": result_set("Photon MC comparisons", matrix, "Independent photon Monte Carlo")}
    if time is not None:
        sets["r6"] = {"title": "Time atlas R6", "date_min": None, "date_max": None,
                      "code": "Recorded atlas radii", "description": "Atlas component checks"}
    return clean({"schema": "lighthit/results/1", "registry_count": len(ideas),
                  "registry_digest": hashlib.sha256(json.dumps(ideas, sort_keys=True, default=str).encode()).hexdigest(),
                  "registry_states": dict(Counter(i["status"].split(":")[0] for i in ideas)),
                  "record_count": sum(len(r) for r in records.values()), "last_recorded_date": all_dates[-1],
                  "reference_records": [{k: r.get(k) for k in ("idea", "case", "date", "source", "status")}
                                        for idea, items in records.items()
                                        if idea not in {i["id"] for i in ideas} for r in items],
                  "ideas": ideas, "examples": examples, "matrix": matrix, "time": time, "sets": sets,
                  "sources": sources})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--atlas-report", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; choose a new immutable report path")
    data = build(args.atlas_report)
    packed = json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(packed + "\n")
    print(f"Exported {data['registry_count']} ideas and {data['record_count']} case records")


if __name__ == "__main__":
    main()
