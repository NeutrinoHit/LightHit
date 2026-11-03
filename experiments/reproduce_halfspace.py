#!/usr/bin/env python
"""Reproduce the half-space tests, numerical checks and report figures.

Run from a checkout with the validation and test dependencies installed.
Each child process uses this checkout's source. Outputs go to an initially
empty directory; no installation, source rewriting or network access occurs.
The default run includes spatial, material and time-profile validation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version, PackageNotFoundError
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter


ROOT = Path(__file__).resolve().parents[1]
FILES = [
    "src/lighthit/medium.py", "src/lighthit/readout.py",
    "src/lighthit/experimental/halfspace.py",
    "src/lighthit/experimental/halfspace_boundary.py",
    "src/lighthit/experimental/halfspace_mc.py",
    "experiments/validate_halfspace.py", "experiments/validate_halfspace_materials.py",
    "experiments/halfspace_time_profile.py", "experiments/validate_halfspace_time.py",
    "experiments/audit_halfspace_quadrature.py", "experiments/plot_halfspace_report.py",
    "experiments/reproduce_halfspace.py", "experiments/fixtures/halfspace-materials.json",
    "tests/test_halfspace.py", "tests/test_halfspace_boundary.py",
    "tests/test_halfspace_mc.py", "tests/test_halfspace_time_example.py",
]


def source_hashes():
    return {path: hashlib.sha256((ROOT/path).read_bytes()).hexdigest() for path in FILES}


def repository_head():
    try:
        top = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "--show-toplevel"],
                                      text=True, stderr=subprocess.DEVNULL).strip()
        if Path(top).resolve() == ROOT:
            return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                           text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--no-figures", action="store_true",
                        help="run numerical checks and save arrays without rendering images")
    args = parser.parse_args()
    output = args.output_directory.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("output directory must be empty; use a new directory to preserve previous results")
    output.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join([str(ROOT/"src"), str(ROOT)])
    environment["MPLCONFIGDIR"] = str(output/".cache"/"matplotlib")
    environment["XDG_CACHE_HOME"] = str(output/".cache")
    environment.setdefault("OPENBLAS_NUM_THREADS", "1")
    packages = {}
    for name in ("numpy", "scipy", "numba", "matplotlib", "pytest"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    before = source_hashes()
    manifest = {"schema": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
                "repository_head": repository_head(),
                "runtime": {"python": sys.version.split()[0], "system": platform.system(),
                            "architecture": platform.machine(), "packages": packages},
                "source_sha256": before, "figures_requested": not args.no_figures,
                "steps": [], "passed": False}
    destination = output/"manifest.json"

    def save():
        destination.write_text(json.dumps(manifest, indent=2)+"\n")

    synthetic = ["experiments/validate_halfspace.py", "--output", str(output/"synthetic"/"validation.json")]
    if not args.no_figures:
        synthetic += ["--figures-directory", str(output/"synthetic"/"figures")]
    steps = [
        ("tests", ["-m", "pytest", "-q", "tests/test_halfspace.py", "tests/test_halfspace_boundary.py",
                   "tests/test_halfspace_mc.py", "tests/test_halfspace_time_example.py"]),
        ("synthetic", synthetic),
        ("materials", ["experiments/validate_halfspace_materials.py", "--profiles",
                       "experiments/fixtures/halfspace-materials.json", "--output",
                       str(output/"materials"/"validation.json")]),
        ("quadrature", ["experiments/audit_halfspace_quadrature.py", "--output",
                        str(output/"quadrature"/"validation.json")]),
        ("time-coarse", ["experiments/halfspace_time_profile.py", "compute", "--output-directory",
                         str(output/"time"/"coarse"), "--no-plot"]),
        ("time-fine", ["experiments/halfspace_time_profile.py", "compute", "--output-directory",
                       str(output/"time"/"fine"), "--polar-nodes", "64", "--phase-degree", "96",
                       "--omega-step-per-ns", "0.1", "--no-plot"]),
        ("time-validation", ["experiments/validate_halfspace_time.py", "--coarse-spectrum",
                             str(output/"time"/"coarse"/"spectrum.npz"), "--fine-spectrum",
                             str(output/"time"/"fine"/"spectrum.npz"), "--output",
                             str(output/"time"/"validation.json")]),
    ]
    readout = ["experiments/halfspace_time_profile.py", "plot", "--spectrum",
               str(output/"time"/"fine"/"spectrum.npz"), "--output-directory",
               str(output/"time"/"profile"), "--bin-width-ns", "0.01"]
    if args.no_figures:
        readout += ["--no-plot"]
    steps.append(("time-readout", readout))
    if not args.no_figures:
        steps.append(("report-figures", ["experiments/plot_halfspace_report.py", "--validation",
                                        str(output/"synthetic"/"validation.json"), "--materials",
                                        str(output/"materials"/"validation.json"), "--quadrature-audit",
                                        str(output/"quadrature"/"validation.json"), "--banana-data",
                                        str(output/"synthetic"/"figures"/"banana.npz"),
                                        "--output-directory", str(output/"report")]))
    started = perf_counter()
    save()
    for name, command in steps:
        print(f"[{name}] starting", flush=True)
        t = perf_counter()
        with (output/f"{name}.log").open("w") as log:
            process = subprocess.Popen([sys.executable, "-u", *command], cwd=ROOT,
                                       env=environment, text=True, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT)
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            returncode = process.wait()
        manifest["steps"].append({"name": name, "returncode": returncode,
                                  "wall_seconds": perf_counter()-t})
        save()
        if returncode:
            manifest["wall_seconds"] = perf_counter()-started
            save()
            print(f"FAIL: {name}; previous results and log are preserved", flush=True)
            return returncode
    manifest["source_unchanged"] = before == source_hashes()
    manifest["passed"] = bool(manifest["source_unchanged"])
    manifest["wall_seconds"] = perf_counter()-started
    save()
    print(f"{'PASS' if manifest['passed'] else 'FAIL'}: {len(steps)} steps, "
          f"{manifest['wall_seconds']:.2f} s", flush=True)
    return 0 if manifest["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
