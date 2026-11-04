"""Export a compact results report without running transport calculations."""
from __future__ import annotations

import argparse
from collections import Counter
import datetime
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
    return {k: v.get(k) for k in ("idea", "case", "date", "source", "status", "kind", "reference", "timing",
                                  "metrics", "accuracy", "summary", "variant", "receiver", "centre_reference",
                                  "data_bundle", "machine", "threads", "warmup")}


#: receiver definition of the route tables -> public label
RECEIVERS = {"first-entry": "glass", "centre": "centre", None: "centre"}

#: stored detector examples: (idea, record case, page ID, title)
EXAMPLES = (
    ("default-shower-route", "electron-100GeV@glass", "electron-100GeV", "100 GeV electron"),
    ("default-muon-route", "muon-100GeV@glass", "muon-100GeV", "100 GeV muon"),
    ("default-laser-route", "laser-532nm@glass", "laser-532nm", "532 nm calibration flash"),
)


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


def build(atlas_report=None, kit_check=None, kit_bundle=None, halfspace_results=None, timing=(),
          publication=None, mc_check=None):
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
                      "wall_s": (r.get("timing") or {}).get("wall_s"),
                      **({"receiver": RECEIVERS.get(r.get("receiver"), r.get("receiver")),
                          "centre_reference": r.get("centre_reference")} if r.get("variant") else {})}
                     for r in saved],
        })
    spec = importlib.util.spec_from_file_location("stored_response_metrics", ROOT / "experiments/harness/metrics.py")
    metric_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metric_module)
    track(ROOT / "experiments/harness/metrics.py")
    examples = []
    for idea, case, page_id, title in EXAMPLES:
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
            "id": page_id, "record_case": case, "title": title, "idea": idea, "date": record["date"],
            "source": record["source"], "receiver": RECEIVERS.get(record.get("receiver"), record.get("receiver")),
            "route": (record.get("summary") or {}).get("route"),
            "record_path": str(json_path.relative_to(ROOT)), "response_path": str(npz_path.relative_to(ROOT)),
            "response_sha256": digest(npz_path), "timing": record["timing"],
            "total_by_order": q.sum(0), "active_count": int(len(active)), "module_count": len(q),
            "charge_ranked": [{"index": int(i), "charge": q[i]} for i in ranked],
            "time_edges_ns": arrays["relative_time_edges_ns"],
            "traces": [{"index": int(i), "origin_ns": float(arrays["time_origin_ns"][i]), "charge": q[i],
                        "bins": arrays["components_pe"][i]} for i in selected],
            "reference": record["reference"], "comparison": None,
            "centre_reference": record.get("centre_reference"), "centre_difference": None,
        }
        if record.get("centre_reference"):
            # the stored reference is centre-referenced: the difference is the receiver change
            centre_path = results_dir / (record["centre_reference"] + ".npz")
            if centre_path.exists():
                track(centre_path)
                centre = read_arrays(centre_path)
                computed = [str(x) for x in centre.get("orders_computed", ["0", "1", "2"])]
                kept = tuple(k for k in (0, 1, 2) if str(k) in computed or (k == 2 and ">=2" in computed))
                example["centre_difference"] = metric_module.compare(arrays, centre, orders=kept, floor_pe=0.01)
                if 2 not in kept:
                    example["centre_difference"]["orders"].pop("total", None)
        base_path = results_dir / idea / (case.split("@")[0] + ".npz")
        if base_path != npz_path and base_path.exists():
            track(base_path)
            base = read_arrays(base_path)
            example["centre_record"] = str(base_path.relative_to(ROOT))
            example["identical_to_centre_record"] = bool(
                np.array_equal(base["charge_components_pe"], q)
                and np.array_equal(base.get("components_pe"), arrays.get("components_pe")))
            example["centre_record_total_by_order"] = base["charge_components_pe"].sum(0)
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
                       "receiver": RECEIVERS.get(record.get("receiver"), record.get("receiver")),
                       "path": str(path.relative_to(ROOT)), "sha256": digest(path),
                       "headline": metrics["headline"], "totals": metrics["totals"],
                       "rows": metrics["by_distance"], "same_steps": metrics["same_steps_in_all_runs"],
                       "runs": metrics["mc"]["runs"],
                       **({"mc_timing": {k: metrics["mc"].get(k) for k in
                                         ("photons", "flash_photons", "wall_s", "cpu_s", "chunk_photons",
                                          "chunk_wall_s", "threads", "load_average")}}
                          if "chunk_wall_s" in metrics["mc"] else {})})
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
    receivers = sorted({x["receiver"] for x in examples})
    sets = {"examples": result_set("Detector examples", examples,
                                   "Stored event responses" + (" · " + "/".join(receivers) + " tables"
                                                               if receivers != ["centre"] else "")),
            "mc": result_set("Photon MC comparisons", matrix, "Independent photon Monte Carlo")}
    extra = {}
    if kit_check is not None:
        extra["kit"] = kit_acceptance(kit_check, kit_bundle, sources)
        for example in examples:
            # the kit's MC ratios belong to an example only when it is the same event
            match = next((c for c in extra["kit"]["cases"] if c["name"] == example["id"]), None)
            example["kit_fixture_same_event"] = bool(match) and bool(np.allclose(
                example["total_by_order"], match["numerical"]["charge_by_order"], rtol=1e-9, atol=0))
        sets["kit"] = {"title": "Glass kit acceptance", "date_min": extra["kit"]["date"],
                       "date_max": extra["kit"]["date"], "code": "LightHit " + extra["kit"]["code_version"],
                       "description": "check.py of the 0.3.0a6 glass kit and photon MC by distance"}
    if halfspace_results is not None:
        extra["halfspace"] = halfspace(halfspace_results, sources)
        sets["halfspace"] = {"title": "Half-space reference", "date_min": extra["halfspace"]["date"],
                             "date_max": extra["halfspace"]["date"], "code": "commit 93520b3",
                             "description": "Experimental collimated half-space transport and its checks"}
    if timing:
        extra["timing"] = route_timing(timing, matrix, sources, mc_check)
    if time is not None:
        sets["r6"] = {"title": "Time atlas R6", "date_min": None, "date_max": None,
                      "code": "Recorded atlas radii", "description": "Atlas component checks"}
    return clean({"schema": "lighthit/results/2" if extra or receivers != ["centre"] else "lighthit/results/1", "registry_count": len(ideas),
                  "registry_digest": hashlib.sha256(json.dumps(ideas, sort_keys=True, default=str).encode()).hexdigest(),
                  "registry_states": dict(Counter(i["status"].split(":")[0] for i in ideas)),
                  "record_count": sum(len(r) for r in records.values()), "last_recorded_date": all_dates[-1],
                  "reference_records": [{k: r.get(k) for k in ("idea", "case", "date", "source", "status")}
                                        for idea, items in records.items()
                                        if idea not in {i["id"] for i in ideas} for r in items],
                  "ideas": ideas, "examples": examples, "matrix": matrix, "time": time, "sets": sets,
                  **extra, "sources": sources,
                  **({"publication": {"id": publication, "date": datetime.date.today().isoformat(),
                                      "kind": "archived research results", "release_acceptance": False}}
                     if publication else {})})


ORDER_KEYS = ("0", "1", ">=2", "total")


def kit_acceptance(path, bundle, sources):
    """check.py of the delivered glass kit: numerical reproduction, timing and photon MC by
    distance, with the event totals recomputed from the kit's MC records."""
    raw = json.loads(path.read_text())
    sources["kit/" + path.name] = digest(path)
    cases = []
    for case in raw["cases"]:
        mc_record = mc_totals = None
        if bundle is not None:
            mc_path = bundle / "validation" / f"{case['name']}-mc_record.json"
            if mc_path.exists():
                sources[f"kit/validation/{mc_path.name}"] = digest(mc_path)
                metrics = json.loads(mc_path.read_text())["metrics"]
                mc_record, mc_totals = metrics["by_distance"], metrics["totals"]
        groups, totals, flags = [], None, []
        for j, row in enumerate(case["mc_by_distance"]):
            out = {"distance_m": row["distance_m"], "oms": row["oms"]}
            for key in ORDER_KEYS:
                value = row["orders"][key]
                out[key] = {k: value.get(k) for k in ("ratio", "ratio_sigma", "significant", "passed")}
                if mc_record is not None:
                    reference = mc_record[j][key]
                    out[key].update(mc_pe=reference["mc_pe"], mc_share=reference["mc_share"])
                if value["passed"] is False and value["ratio"] is not None:
                    kind = "flag" if value["significant"] else "weak group (MC share ≤ 0.1 %)"
                    if value["ratio"] == 0 and row["distance_m"][0] >= 300:
                        kind += "; no LightHit charge beyond the 300 m table range"
                    flags.append({"distance_m": row["distance_m"], "order": key, "ratio": value["ratio"],
                                  "ratio_sigma": value["ratio_sigma"], "kind": kind})
            groups.append(out)
        if mc_totals is not None:
            # event totals: this run's LightHit charge over the whole-event MC charge and its
            # run-to-run error (the groups of one MC run are correlated)
            charge = case["numerical"]["charge_by_order"]
            totals = {}
            for k, key in enumerate(ORDER_KEYS):
                lighthit = charge[k] if k < 3 else sum(charge)
                mc, sigma = mc_totals[key]["mc_pe"], mc_totals[key]["mc_sigma_pe"]
                ratio = lighthit / mc if mc else None
                totals[key] = {"lighthit_pe": lighthit, "mc_pe": mc, "ratio": ratio,
                               "ratio_sigma": ratio * sigma / mc if ratio else None}
                if ratio and abs(ratio - 1) > max(.03, 2 * totals[key]["ratio_sigma"]):
                    flags.append({"distance_m": None, "order": key, "ratio": ratio,
                                  "ratio_sigma": totals[key]["ratio_sigma"], "kind": "event total"})
        numerical = case["numerical"]
        cases.append({"name": case["name"], "steps_sha256": case["steps_sha256"],
                      "numerical": {k: numerical[k] for k in ("passed", "max_charge_absolute_pe", "max_bin_l1_pe",
                                                               "charge_by_order", "reference_by_order", "rtol", "atol_pe")},
                      "timings": case["timings"], "peak_rss_bytes": case.get("peak_rss_platform_units"),
                      "groups": groups, "totals": totals, "flags": flags})
    identity = None
    if bundle is not None and (bundle / "bundle.json").exists():
        data = json.loads((bundle / "bundle.json").read_text())
        sources["kit/bundle.json"] = digest(bundle / "bundle.json")
        identity = {"model_sha256": data["identity"]["model_sha256"], "files": len(data["files"]),
                    "bytes": sum(v["size"] for v in data["files"].values()), "complete": data["complete"],
                    "bundle_json_sha256": digest(bundle / "bundle.json"),
                    "validation_scope": data.get("validation_scope")}
    date = datetime.datetime.fromtimestamp(path.stat().st_mtime, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {"code_version": raw["code_version"], "complete": raw["complete"], "passed": raw["passed"],
            "threads": raw["threads"], "date": date, "bundle": identity, "cases": cases,
            "criteria": {"numerical": "charge and bin L1 within atol + rtol * reference, same active set and clock",
                         "mc": "a group with MC share > 0.1 % passes when |ratio - 1| <= max(3 %, 2 sigma)"},
            "notes": ["Fixture events are the kit's own saved steps, not the detector examples.",
                      "Numerical reproduction and the photon-MC criterion are separate checks; "
                      "passing the numerical check does not remove the MC flags.",
                      "Timing: first call warms the route, second call is the complete event including "
                      "source compilation; tables are opened before the timer."]}


def halfspace(directory, sources):
    """The verified run of experiments/reproduce_halfspace.py (reduced numbers only)."""
    manifest = json.loads((directory / "manifest.json").read_text())
    sources["halfspace/manifest.json"] = digest(directory / "manifest.json")
    unchanged = all((ROOT / name).exists() and digest(ROOT / name) == value
                    for name, value in manifest["source_sha256"].items())
    def load(name):
        path = directory / name / "validation.json"
        sources[f"halfspace/{name}/validation.json"] = digest(path)
        return json.loads(path.read_text())
    synthetic, materials, quadrature, timing = (load(n) for n in ("synthetic", "materials", "quadrature", "time"))
    synthetic_rows = []
    for case in synthetic["cases"]:
        mc_balance = case["monte_carlo_balance"]
        synthetic_rows.append({
            "n": case["n"], "photons": case["photons"],
            "balance": {k: case["photon_balance"][k] for k in ("specular", "absorbed", "escaped", "residual")},
            "mc_balance": {k: mc_balance[k] for k in ("specular", "absorbed", "escaped")},
            "adjoint_relative_error": case["adjoint_absorption"]["relative_error"],
            "radial_max_pull": case["radial"]["monte_carlo"]["max_pull"],
            "radial_cutoff_change": case["radial"]["cutoff_quadrature_change"],
            "banana_max_pull": case["banana"]["monte_carlo"]["max_pull"],
            "banana_angular_change": case["banana"]["angular_change_relative"],
            "banana_radial_change": case["banana"]["radial_change_relative"]})
    names = {p["id"]: p["name"] for p in materials["input"]["profiles"]}
    material_rows = [{"profile": names.get(c["profile"], c["profile"]), "mu_a_per_cm": c["mu_a_per_cm"],
                      "mu_s_per_cm": c["mu_s_per_cm"], "g": c["g"], "n_ratio": c["n_ratio"],
                      "specular": c["specular"], "total_reflectance": c["total_reflectance"],
                      "escaped_by_order": c["escaped_components"], "mc_by_order": c["monte_carlo"]["mean"],
                      "mc_stderr_by_order": c["monte_carlo"]["stderr"], "mc_pull_by_order": c["monte_carlo"]["pull"],
                      "mc_photons": c["monte_carlo"]["photons"],
                      "grid_change_by_order": c["grid_change_relative_per_order"], "passed": c["passed"]}
                     for c in materials["cases"]]
    profiles = [{k: p.get(k) for k in ("name", "g", "phase_index_inside", "wavelength_nm",
                                         "phase_index_status", "absorption_status", "model_status")}
                for p in materials["input"]["profiles"]]
    mc = timing["monte_carlo"]
    return {"date": manifest["created_utc"][:19] + "Z", "passed": manifest["passed"],
            "source_unchanged_during_run": manifest["source_unchanged"],
            "sources_match_checkout": unchanged, "wall_s": manifest["wall_seconds"],
            "runtime": manifest["runtime"],
            "steps": [{"name": x["name"], "returncode": x["returncode"], "wall_s": x["wall_seconds"]}
                      for x in manifest["steps"]],
            "synthetic": {"medium": synthetic["medium"], "criteria": synthetic["criteria"],
                          "passed": synthetic["passed"], "cases": synthetic_rows},
            "materials": {"criteria": materials["criteria"], "passed": materials["passed"],
                          "geometry": materials["geometry"], "profiles": profiles, "cases": material_rows},
            "quadrature": {"criteria": quadrature["criteria"], "passed": quadrature["passed"]},
            "time": {"parameters": timing["parameters"], "errors": timing["errors"],
                     "criteria": timing["criteria"], "passed": timing["passed"],
                     "mc_photons": mc["photons"], "max_resolved_pull": mc["max_resolved_pull"],
                     "max_statistical_only_pull": mc["max_statistical_only_pull"],
                     "unresolved_tails_within_bound": mc["unresolved_tails_within_absolute_bound"]},
            "scope": "Experimental module (lighthit.experimental.halfspace), not shipped in the wheel and not "
                     "part of the optical-module routes. Material profiles are illustrative model "
                     "specifications, not measured calibrations; no tissue data have been compared.",
            "reproduce": "python experiments/reproduce_halfspace.py --output-directory RESULTS"}


def route_timing(paths, matrix, sources, mc_check=None):
    """experiments/time_routes.py outputs and the laser photon-MC time to its precision."""
    runs = []
    for path in paths:
        raw = json.loads(path.read_text())
        sources["timing/" + path.name] = digest(path)
        loads = [m["load_average_before"][0] for case in raw["raw"].values() for m in case.values()]
        runs.append({**{k: raw[k] for k in ("started", "finished", "source", "machine", "threads",
                                            "thread_variables", "receiver", "definitions", "rows")},
                     "load_average_1min_range": [min(loads), max(loads)]})
    laser = None
    entry = next((m for m in matrix if m["id"].startswith("laser-532nm") and m.get("mc_timing")), None)
    if entry is not None:
        t = entry["mc_timing"]
        wall = sum(t["chunk_wall_s"])
        laser = {"mc_case": entry["id"], "mc_photons": t["photons"], "mc_wall_s": wall,
                 "mc_wall_per_photon_s": wall / t["photons"], "mc_threads": t["threads"],
                 "mc_load_average": t["load_average"], "rows": entry["rows"], "headline": entry["headline"]}
        if mc_check is not None:
            # the same photon chain alone on the machine: its time per photon, for the
            # Monte-Carlo time of the compared run without other jobs
            with np.load(mc_check, allow_pickle=False) as data:
                meta = json.loads(str(data["metadata"]))
            sources["timing/" + mc_check.name] = digest(mc_check)
            chunks = meta["chunks"][1:] or meta["chunks"]     # the first chunk loads the code
            per_photon = sum(c["wall_s"] for c in chunks) / sum(c["photons"] for c in chunks)
            laser["quiet_check"] = {"photons": meta["photons"], "wall_s": meta["wall_s"],
                                    "wall_per_photon_s": per_photon, "threads": meta["threads"],
                                    "load_average": meta["load_average"],
                                    "mc_wall_s_at_compared_photons": per_photon * t["photons"]}
    return {"runs": runs, "laser_mc": laser}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--atlas-report", type=Path)
    parser.add_argument("--kit-check", type=Path, help="check.py report of a delivered glass kit")
    parser.add_argument("--kit-bundle", type=Path, help="the kit's runtime bundle (validation MC records)")
    parser.add_argument("--halfspace", type=Path, help="a verified reproduce_halfspace.py output directory")
    parser.add_argument("--timing", type=Path, nargs="*", default=(), help="experiments/time_routes.py outputs")
    parser.add_argument("--laser-mc-check", type=Path,
                        help="hits file of a short photon_mc_laser.py run alone on the machine")
    parser.add_argument("--publication", help="report ID recorded in the report")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; choose a new immutable report path")
    data = build(args.atlas_report, args.kit_check, args.kit_bundle, args.halfspace, args.timing,
                 args.publication, args.laser_mc_check)
    packed = json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(packed + "\n")
    print(f"Exported {data['registry_count']} ideas and {data['record_count']} case records")


if __name__ == "__main__":
    main()
