#!/usr/bin/env python
"""Separate bulk polar convergence from exterior receiver integration.

The fixed synthetic medium is the one used by validate_halfspace.py.
Roundoff photon balance is not treated as an angular accuracy bound.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))

from lighthit.medium import Medium
from lighthit.experimental.halfspace import HalfSpaceSolver, HalfSpaceSettings, SurfacePort
from lighthit.experimental.halfspace_boundary import HalfSpaceBoundary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    medium = Medium(.2, 1., .3, 1.4)
    source = SurfacePort.pencil()
    report = {"schema": 1, "detector_quadrature": [], "polar_quadrature": [],
              "refined_q_zero": [], "criteria": {"balance_residual": 1e-9,
                                                    "refined_flux_difference": 1e-6}}
    checks = []
    for n in (1., 1.4):
        kernel = HalfSpaceSolver(medium, HalfSpaceBoundary(n),
                                 HalfSpaceSettings(8, 1, 10, plane_parallel=True))
        solution = kernel.solve(source)
        balance = solution.photon_balance()
        for angle in (.7, np.pi/2):
            for polar in (4, 8, 16, 32, 64, 128):
                receiver = SurfacePort.aperture(angle, polar_nodes=polar, azimuth_nodes=2)
                value = solution.response(receiver).real
                report["detector_quadrature"].append(
                    {"n": n, "half_angle_rad": angle, "polar_nodes": polar,
                     "components": value.tolist(), "escape": float(value.sum()),
                     "balance_escape": balance["escaped"]})
    for polar in (6, 8, 12, 16, 24, 32, 48, 64, 96, 128):
        kernel = HalfSpaceSolver(medium, HalfSpaceBoundary(1.4),
                                 HalfSpaceSettings(polar, 1, 10, plane_parallel=True))
        solution = kernel.solve(source)
        value = solution.response(SurfacePort.aperture(polar_nodes=128, azimuth_nodes=2)).real
        balance = solution.photon_balance()
        checks.append(abs(balance["residual"]) < 1e-9)
        report["polar_quadrature"].append(
            {"polar_nodes": polar, "phase_degree": 10,
             "balance": balance, "receiver128": value.tolist()})
    for n in (1., 1.4):
        kernel = HalfSpaceSolver(medium, HalfSpaceBoundary(n),
                                 HalfSpaceSettings(128, 1, 18, plane_parallel=True))
        solution = kernel.solve(source)
        first = solution.first_field([0.])[0, kernel.half:]
        factor = (kernel.weight[kernel.half:]*(-kernel.mu[kernel.half:])
                  * kernel.transmission)
        order1 = float((factor@first).real)
        balance = solution.photon_balance()
        readout = solution.response(SurfacePort.aperture(polar_nodes=160, azimuth_nodes=2)).real
        gap = abs(balance["escaped"]-readout.sum())
        checks += [abs(balance["residual"]) < 1e-9, gap < 1e-6]
        report["refined_q_zero"].append(
            {"n": n, "settings": {"polar_nodes": 128, "phase_degree": 18,
                                    "plane_parallel": True},
             "balance": balance,
             "internal_components": [0., order1, balance["escaped"]-order1],
             "receiver160_components": readout.tolist()})
        print(f"n={n:g}: refined escape={balance['escaped']:.10f}, receiver gap={gap:.3g}", flush=True)
    coarse, fine = report["polar_quadrature"][1], report["polar_quadrature"][-1]
    gap_coarse = abs(coarse["balance"]["escaped"]-sum(coarse["receiver128"]))
    gap_fine = abs(fine["balance"]["escaped"]-sum(fine["receiver128"]))
    checks.append(gap_fine < gap_coarse)
    report["passed"] = bool(all(checks))
    paths = ["src/lighthit/experimental/halfspace.py",
             "src/lighthit/experimental/halfspace_boundary.py",
             "experiments/audit_halfspace_quadrature.py"]
    report["source_sha256"] = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print("PASS" if report["passed"] else "FAIL", flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
