#!/usr/bin/env python
"""Check sourced material profiles at q=omega=0 against continuous full-HG MC.

Read a JSON profile file with coefficients in cm^-1. Missing absorption
must have an explicitly labelled positive scan. No measured absorption,
wavelength calibration, or sample identity is inferred. The m=0 solver
computes the exact azimuthal mean of this laterally integrated problem.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))

from lighthit.medium import Medium
from lighthit.experimental.halfspace import HalfSpaceSolver, HalfSpaceSettings, SurfacePort
from lighthit.experimental.halfspace_boundary import HalfSpaceBoundary
from lighthit.experimental.halfspace_mc import sample_halfspace


def material_values(solver, source):
    solution = solver.solve(source)
    balance = solution.photon_balance()
    h = solver.half
    first = solution.first_field([0.])[0, h:]
    factor = solver.weight[h:]*(-solver.mu[h:])*solver.transmission
    first_escape = float((factor@first).real)
    return np.array([0., first_escape, balance["escaped"]-first_escape]), balance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--photons", type=int, default=250000)
    args = parser.parse_args()
    config = json.loads(args.profiles.read_text())
    if config.get("schema") != 1:
        raise ValueError("profile schema must be 1")
    start = perf_counter()
    source = SurfacePort.pencil()
    report = {"schema": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
              "profiles_sha256": hashlib.sha256(args.profiles.read_bytes()).hexdigest(),
              "input": config, "geometry": "normal pencil; q=0; full exterior hemisphere",
              "omega_per_ns": 0., "orders": [0, 1, ">=2"],
              "criteria": {"max_mc_pull": 4.5, "max_grid_relative_per_order": .001,
                           "max_balance_residual": 1e-9}, "cases": []}
    checks = []
    for index, profile in enumerate(config["profiles"]):
        mus_cm, g = float(profile["scattering_per_cm"]), float(profile["g"])
        absorption = ([float(profile["absorption_per_cm"])] if profile.get("absorption_per_cm") is not None
                      else list(map(float, profile["absorption_scan_per_cm"])))
        if not absorption or min(absorption) <= 0 or not np.isfinite(absorption).all():
            raise ValueError("stationary material cases need positive explicit absorption values")
        boundary = HalfSpaceBoundary(profile["phase_index_inside"]/config["outside_phase_index"])
        base = Medium(100*min(absorption), 100*mus_cm, g, profile["phase_index_inside"],
                      profile["wavelength_nm"], profile["id"]+"; see supplied profile provenance")
        max_events = max(10000, int(math.ceil(100*base.extinction_per_m/base.absorption_per_m)))
        histories = sample_halfspace(base, boundary, source, photons=args.photons, seed=20261130+index,
                                     max_events=max_events)
        settings = ([HalfSpaceSettings(32, 1, 0, plane_parallel=True),
                     HalfSpaceSettings(64, 1, 0, plane_parallel=True)] if g == 0 else
                    [HalfSpaceSettings(80, 1, 128, plane_parallel=True),
                     HalfSpaceSettings(112, 1, 192, plane_parallel=True)])
        for mua_cm in sorted(absorption):
            medium = Medium(100*mua_cm, 100*mus_cm, g, base.group_index, base.wavelength_nm, base.provenance)
            values, balances = [], []
            for setting in settings:
                solver = HalfSpaceSolver(medium, boundary, setting)
                value, balance = material_values(solver, source)
                values.append(value)
                balances.append(balance)
            estimate = histories.spectrum([[0., 0.]], [0.],
                                           absorption_offset_per_m=medium.absorption_per_m-base.absorption_per_m)
            mean = estimate.mean[0, 0].real
            stderr = estimate.stderr_real[0, 0]
            pull = abs(values[-1]-mean)/np.maximum(stderr, 1e-14)
            change = abs(values[-1][1:]-values[0][1:])/np.maximum(abs(values[-1][1:]), 1e-14)
            passed = (float(pull.max()) <= 4.5 and float(change.max()) <= .001
                      and abs(balances[-1]["residual"]) <= 1e-9)
            checks.append(passed)
            case = {"profile": profile["id"], "mu_a_per_cm": mua_cm, "mu_s_per_cm": mus_cm,
                    "mu_s_prime_per_cm": mus_cm*(1-g), "g": g, "n_ratio": boundary.n,
                    "settings": [asdict(s) for s in settings], "escaped_components": values[-1].tolist(),
                    "specular": balances[-1]["specular"],
                    "total_reflectance": balances[-1]["specular"]+values[-1].sum(),
                    "balance": balances[-1], "grid_change_relative_per_order": change.tolist(),
                    "monte_carlo": {"photons": args.photons, "seed": 20261130+index,
                                     "base_mu_a_per_cm": min(absorption), "max_events": max_events,
                                     "mean": mean.tolist(), "stderr": stderr.tolist(),
                                     "pull": pull.tolist()}, "passed": bool(passed)}
            report["cases"].append(case)
            print(f"{profile['id']} mu_a={mua_cm:g}/cm: R={case['total_reflectance']:.8f}; "
                  f"MC pull={pull.max():.3f}, grid={change.max():.3g}, "
                  f"balance={balances[-1]['residual']:.3g} {'PASS' if passed else 'FAIL'}", flush=True)
    files = ["src/lighthit/experimental/halfspace.py", "src/lighthit/experimental/halfspace_boundary.py",
             "src/lighthit/experimental/halfspace_mc.py", "experiments/validate_halfspace_materials.py"]
    report["source_sha256"] = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files}
    report["passed"] = bool(all(checks))
    report["wall_seconds"] = perf_counter()-start
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    print(f"{'PASS' if report['passed'] else 'FAIL'}: {len(checks)} material cases, {report['wall_seconds']:.2f} s")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
