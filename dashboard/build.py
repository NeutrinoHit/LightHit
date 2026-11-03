"""Validate the published report catalog and build the static results site."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil

HERE = Path(__file__).resolve().parent
ASSETS = ("index.html", "style.css", "app.js", "explore-results.ipynb", "package-versions.json")


def read_json(path):
    def invalid(value):
        raise ValueError(f"non-finite JSON value: {value}")
    return json.loads(path.read_text(), parse_constant=invalid)


def validate_report(data):
    ideas = data["ideas"]
    if len(ideas) != data["registry_count"] or len({i["id"] for i in ideas}) != len(ideas):
        raise ValueError("registry count or identity mismatch")
    if sum(len(i["runs"]) for i in ideas) + len(data["reference_records"]) != data["record_count"]:
        raise ValueError("case-record count mismatch")
    for case in data["examples"]:
        edges = case["time_edges_ns"]
        if any(a >= b for a, b in zip(edges, edges[1:])):
            raise ValueError("time-bin edges must increase")
        if len(case["total_by_order"]) != 3:
            raise ValueError("three collision orders are required")
        for trace in case["traces"]:
            if len(trace["bins"]) + 1 != len(edges):
                raise ValueError("time bins do not match their recorded edges")
            if any(len(row) != 3 for row in trace["bins"]):
                raise ValueError("time bins must retain all three collision orders")
    def inspect(value):
        if isinstance(value, dict):
            for key, item in value.items():
                inspect(key)
                inspect(item)
        elif isinstance(value, list):
            for item in value:
                inspect(item)
        elif isinstance(value, str):
            for token in value.split():
                candidate = token.strip("\"';,()")
                if "/" not in candidate or candidate == "/":
                    continue
                path = PurePosixPath(candidate)
                if path.is_absolute() or (path.parts and path.parts[0].startswith(".")
                                          and path.parts[0] not in (".", "..")):
                    raise ValueError("local source path in public report")
    inspect(data)


def catalog(source=HERE):
    index = read_json(source / "reports/index.json")
    seen = set()
    for snapshot in index["snapshots"]:
        name = snapshot["id"]
        if not re.fullmatch(r"[a-z0-9][a-z0-9.-]*", name) or name in seen:
            raise ValueError("invalid or repeated report ID")
        seen.add(name)
        if snapshot["file"] != name + ".json":
            raise ValueError("report file must match its immutable ID")
        path = source / "reports" / snapshot["file"]
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != snapshot["sha256"]:
            raise ValueError("published report checksum mismatch")
        validate_report(read_json(path))
    if index["latest"] not in seen:
        raise ValueError("latest report is absent")
    return index


def build(output, source=HERE):
    source, output = source.resolve(), output.resolve()
    if output == source or source in output.parents:
        raise ValueError("build output must be outside the source directory")
    index = catalog(source)
    if output.exists() and any(output.iterdir()):
        raise ValueError("build output must be an empty directory")
    output.mkdir(parents=True, exist_ok=True)
    reports = output / "reports"
    reports.mkdir(exist_ok=True)
    for item in index["snapshots"]:
        shutil.copyfile(source / "reports" / item["file"], reports / item["file"])
    targets = [(output, index["latest"], "")]
    targets += [(output / "versions" / item["id"], item["id"], "../../") for item in index["snapshots"]]
    for target, current, prefix in targets:
        target.mkdir(parents=True, exist_ok=True)
        for name in ASSETS:
            shutil.copyfile(source / name, target / name)
        shutil.copytree(source / "vendor", target / "vendor")
        manifest = {"schema": index["schema"], "current": current, "snapshots": []}
        for item in index["snapshots"]:
            manifest["snapshots"].append({k: v for k, v in item.items() if k != "file"})
            manifest["snapshots"][-1].update(path=prefix + "reports/" + item["file"],
                                             url=prefix + "versions/" + item["id"] + "/")
        (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        current_file = next(i["file"] for i in index["snapshots"] if i["id"] == current)
        shutil.copyfile(reports / current_file, target / "results.json")
        (target / ".nojekyll").write_text("")
    (output / "robots.txt").write_text("User-agent: *\nDisallow: /\n")
    print(f"Built {len(index['snapshots'])} published report(s); current: {index['latest']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("_site"))
    args = parser.parse_args()
    build(args.output)


if __name__ == "__main__":
    main()
