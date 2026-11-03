#!/usr/bin/env python
"""Validate the half-space reference on explicit synthetic optical inputs.

Example from the checkout:
  python experiments/validate_halfspace.py --output validation.json

The report preserves failures and includes independent full-HG Monte Carlo,
angular convergence, reciprocal port tests and absorption sensitivities.
All lengths are metres; these inputs are not a tissue calibration.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import numba
import scipy
from scipy.special import roots_legendre

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lighthit.medium import Medium
from lighthit.experimental.halfspace import HalfSpaceSolver, HalfSpaceSettings, SurfacePort
from lighthit.experimental.halfspace_boundary import HalfSpaceBoundary
from lighthit.experimental.halfspace_mc import sample_halfspace


def complex_array(value):
    value = np.asarray(value)
    return {"real": value.real.tolist(), "imag": value.imag.tolist()}


def pull_report(value, estimate):
    delta = np.asarray(value)-estimate.mean
    real = np.abs(delta.real)/np.maximum(estimate.stderr_real, 1e-14)
    imag = np.abs(delta.imag)/np.maximum(estimate.stderr_imag, 1e-14)
    maximum = max(float(real.max()), float(imag.max()))
    return {"max_pull": maximum, "passed": maximum <= 4.5,
            "estimate": complex_array(estimate.mean),
            "stderr_real": estimate.stderr_real.tolist(),
            "stderr_imag": estimate.stderr_imag.tolist()}


def cell_quadrature(cells, order=4):
    x, w = roots_legendre(order)
    grid = np.stack(np.meshgrid(x, x, x, indexing="ij"), axis=-1).reshape(-1, 3)
    weight = np.prod(np.stack(np.meshgrid(w, w, w, indexing="ij"), axis=-1), axis=-1).ravel()/8
    points = np.concatenate([(low+high)/2+grid*(high-low)/2 for low, high in cells])
    return points, weight


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--photons", type=int, default=1000000)
    parser.add_argument("--banana-photons", type=int, default=2000000)
    parser.add_argument("--figures-directory", type=Path)
    args = parser.parse_args()
    started = perf_counter()
    medium = Medium(.2, 1., .3, 1.4, provenance="synthetic half-space validation; not a tissue calibration")
    settings = [HalfSpaceSettings(4, 8, 6), HalfSpaceSettings(6, 12, 8), HalfSpaceSettings(8, 16, 10)]
    source = SurfacePort.pencil()
    q = np.array([[0., 0.], [.8, 0.], [2., 0.]])
    omega = np.array([0., .1])
    report = {"schema": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
              "runtime": {"python": sys.version.split()[0], "numpy": np.__version__,
                          "scipy": scipy.__version__, "numba": numba.__version__},
              "medium": asdict(medium), "q_per_m": q.tolist(), "omega_per_ns": omega.tolist(),
              "volume_orders": [0, 1, ">=2"], "source": "normal pencil beam; no isotropic replacement",
              "criteria": {"max_mc_pull": 4.5, "spectral_grid_change_over_dc": .005,
                           "reciprocity_relative": 1e-9, "adjoint_relative": 1e-6,
                           "banana_angular_relative": .02, "banana_radial_relative": .002},
              "cases": []}
    passed = []
    for index, n in enumerate((1., 1.4)):
        boundary = HalfSpaceBoundary(n)
        solvers = [HalfSpaceSolver(medium, boundary, config) for config in settings]
        histories = sample_halfspace(medium, boundary, source, photons=args.photons, seed=20261103+index)
        balance = solvers[-1].solve(source).photon_balance()
        case = {"n": n, "photon_balance": balance, "monte_carlo_balance": histories.photon_balance(),
                "seed": 20261103+index, "photons": args.photons, "apertures": []}
        passed.append(abs(balance["residual"]) < 1e-10)
        for angle in (.7, np.pi/2):
            detector = SurfacePort.aperture(angle, polar_nodes=8, azimuth_nodes=16)
            spectra = [np.array([[k.response(source, detector, q_per_m=wave, omega_per_ns=f)
                                  for wave in q] for f in omega]) for k in solvers]
            normalization = np.maximum(abs(spectra[-1][0, 0, 1:]), 1e-15)
            changes = [float(np.max(abs(a[:, :, 1:]-b[:, :, 1:])/normalization))
                       for a, b in zip(spectra[:-1], spectra[1:])]
            estimate = histories.spectrum(q, omega, half_angle_rad=angle)
            comparison = pull_report(spectra[-1], estimate)
            reverse = np.array([[solvers[-1].response(detector, source, q_per_m=-wave, omega_per_ns=f)
                                 for wave in q] for f in omega])
            reciprocity = float(np.max(abs(spectra[-1]/detector.collection_factor-reverse))
                                / max(abs(reverse).max(), 1e-15))
            passed += [comparison["passed"], changes[-1] <= .005, reciprocity <= 1e-9]
            case["apertures"].append({"half_angle_rad": angle, "settings": [asdict(c) for c in settings],
                                      "spectrum": complex_array(spectra[-1]),
                                      "grid_changes_over_dc": changes, "reciprocity_relative": reciprocity,
                                      "monte_carlo": comparison})
            print(f"n={n:g} aperture={angle:.6g}: MC max pull {comparison['max_pull']:.3f}, "
                  f"grid {changes[-1]:.3g}, reciprocity {reciprocity:.3g}", flush=True)

        detector = SurfacePort.aperture(.7, polar_nodes=4, azimuth_nodes=12)
        x, w = roots_legendre(120)
        z, weight = (x+1)*30, w*30
        derivative = weight @ solvers[1].depth_sensitivity(source, detector, z, q_per_m=(.8, .3), omega_per_ns=.1)
        epsilon = 1e-5
        plus = HalfSpaceSolver(replace(medium, absorption_per_m=.2+epsilon), boundary, settings[1])
        minus = HalfSpaceSolver(replace(medium, absorption_per_m=.2-epsilon), boundary, settings[1])
        finite = (plus.response(source, detector, q_per_m=(.8, .3), omega_per_ns=.1).sum()
                  - minus.response(source, detector, q_per_m=(.8, .3), omega_per_ns=.1).sum())/(2*epsilon)
        relative = float(abs(derivative-finite)/abs(finite))
        passed.append(relative <= 1e-6)
        case["adjoint_absorption"] = {"integrated_kernel": complex_array(derivative),
                                       "finite_difference": complex_array(finite), "relative_error": relative}

        cells = np.array([[[0., -.2, .1], [.4, .2, .4]],
                          [[.3, -.2, .5], [.7, .2, .8]],
                          [[.6, -.2, .1], [1., .2, .4]]])
        gaussian_histories = sample_halfspace(medium, boundary, source, photons=args.banana_photons,
                                             seed=20261113+index, source_sigma_m=.4,
                                             sensitivity_cells_m=cells)
        radius = np.array([0., .5, 1., 1.5, 2.])
        radial = solvers[1].radial_response(source, detector, radius, q_max_per_m=12., q_nodes=48,
                                            source_sigma_m=.4, detector_sigma_m=.4)
        radial_refined = solvers[1].radial_response(source, detector, radius, q_max_per_m=14., q_nodes=64,
                                                    source_sigma_m=.4, detector_sigma_m=.4)
        radial_estimate = gaussian_histories.radial_density(radius, detector_sigma_m=.4, half_angle_rad=.7)
        radial_comparison = pull_report(radial_refined, radial_estimate)
        radial_change = float(np.max(abs(radial-radial_refined))/max(abs(radial_refined).max(), 1e-15))
        passed += [radial_comparison["passed"], radial_change <= .002]
        case["radial"] = {"radii_m": radius.tolist(), "source_sigma_m": .4, "detector_sigma_m": .4,
                           "flux_density_per_m2": complex_array(radial_refined),
                           "cutoff_quadrature_change": radial_change, "monte_carlo": radial_comparison}

        points, cell_weight = cell_quadrature(cells)
        banana = []
        for k in solvers[:2]:
            values = k.banana(source, detector, points, detector_xy_m=(1., 0.), source_sigma_m=.4,
                              detector_sigma_m=.4, q_max_per_m=10., q_nodes=48)
            banana.append(values.reshape(len(cells), -1) @ cell_weight)
        values_refined = solvers[1].banana(source, detector, points, detector_xy_m=(1., 0.),
                                          source_sigma_m=.4, detector_sigma_m=.4, q_max_per_m=12., q_nodes=64)
        banana_refined = values_refined.reshape(len(cells), -1) @ cell_weight
        banana_change = float(np.max(abs(banana[1]-banana[0])/abs(banana[1])))
        banana_radial = float(np.max(abs(banana_refined-banana[1])/abs(banana_refined)))
        banana_estimate = gaussian_histories.cell_absorption_sensitivity(detector_sigma_m=.4, half_angle_rad=.7)
        banana_comparison = pull_report(banana_refined, banana_estimate)
        passed += [banana_comparison["passed"], banana_change <= .02, banana_radial <= .002]
        case["banana"] = {"cells_m": cells.tolist(), "cell_average_per_m4": complex_array(banana_refined),
                           "angular_change_relative": banana_change, "radial_change_relative": banana_radial,
                           "monte_carlo": banana_comparison, "photons": args.banana_photons,
                           "seed": 20261113+index, "source_sigma_m": .4, "detector_sigma_m": .4,
                           "detector_xy_m": [1., 0.]}
        print(f"n={n:g} spatial: radial MC pull {radial_comparison['max_pull']:.3f}; "
              f"banana MC pull {banana_comparison['max_pull']:.3f}, "
              f"angular {banana_change:.3g}, radial {banana_radial:.3g}", flush=True)
        report["cases"].append(case)

        if args.figures_directory and n == 1.4:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.colors import LogNorm
            args.figures_directory.mkdir(parents=True, exist_ok=True)
            xs, zs = np.linspace(-.6, 1.6, 55), np.linspace(.025, 2., 50)
            xx, zz = np.meshgrid(xs, zs)
            points = np.column_stack([xx.ravel(), np.zeros(xx.size), zz.ravel()])
            sensitivity = -solvers[1].banana(source, detector, points, detector_xy_m=(1., 0.),
                                             source_sigma_m=.4, detector_sigma_m=.4,
                                             q_max_per_m=12., q_nodes=64).real.reshape(xx.shape)
            fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
            positive = sensitivity[sensitivity > 0]
            im = ax.pcolormesh(xs, zs, np.ma.masked_less_equal(sensitivity, 0), shading="auto",
                               norm=LogNorm(vmin=positive.max()/300, vmax=positive.max()), cmap="magma")
            ax.invert_yaxis()
            ax.set(xlabel="x [m]", ylabel="depth z [m]", title="Synthetic half-space: absorption sensitivity, n=1.4")
            ax.plot([0., 1.], [0., 0.], "cv", markersize=8, clip_on=False)
            ax.text(0., -.045, "normal beam", ha="center", fontsize=9)
            ax.text(1., -.045, "aperture receiver", ha="center", fontsize=9)
            fig.colorbar(im, ax=ax, label="-dD/dmu_a(r) [m^-4 per incident photon]")
            fig.savefig(args.figures_directory/"banana.png", dpi=180)
            plt.close(fig)
            np.savez(args.figures_directory/"banana.npz", x_m=xs, z_m=zs, sensitivity_per_m4=sensitivity)

    files = ["src/lighthit/experimental/halfspace.py", "src/lighthit/experimental/halfspace_boundary.py",
             "src/lighthit/experimental/halfspace_mc.py", "experiments/validate_halfspace.py"]
    report["source_sha256"] = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files}
    report["passed"] = bool(all(passed))
    report["wall_seconds"] = perf_counter()-started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    print(f"{'PASS' if report['passed'] else 'FAIL'}: {len(passed)} checks, {report['wall_seconds']:.2f} s", flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
