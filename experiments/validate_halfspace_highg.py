#!/usr/bin/env python
"""Validate the spatial half-space reference at high anisotropy and q != 0.

Example from the checkout:
  python experiments/validate_halfspace_highg.py --output highg.json --workers 3

Materials come from the sourced profile file (coefficients in cm^-1); the
optional g=0.95 case is explicitly synthetic. Geometry is scaled by the
transport length l* = 1/(mu_s (1-g) + mu_a): a normal pencil source, an
exterior aperture receiver of half-angle 0.7 rad, Gaussian footprints of
0.4 l* and a receiver offset of 1 l*. The volume field uses azimuthal cosine
moments. Every collision order is compared separately with independent
full-HG histories, and the polar/phase, harmonic, readout and transverse
quadratures are each refined. Each job runs single-threaded; jobs may run
in parallel processes. Failures are preserved in the report.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS")
CELLS = [[[0., -.2, .1], [.4, .2, .4]], [[.3, -.2, .5], [.7, .2, .8]], [[.6, -.2, .1], [1., .2, .4]]]
RADII = [0., .5, 1., 1.5, 2.]
WAVES = [0., .5, 2., 5.]
SIGMA, OFFSET, ANGLE = .4, 1., .7
CRITERIA = {"max_mc_pull": 4.5, "spectral_grid_change_over_dc": .005,
            "banana_angular_relative": .02, "banana_radial_relative": .002,
            "radial_quadrature_relative": .002, "eigen_residual": 1e-10,
            "min_effective_histories": 100}


def _imports():
    sys.path.insert(0, str(ROOT / "src"))
    import numpy as np
    from lighthit.medium import Medium
    from lighthit.experimental.halfspace import HalfSpaceSolver, HalfSpaceSettings, SurfacePort
    from lighthit.experimental.halfspace_boundary import HalfSpaceBoundary
    from lighthit.experimental.halfspace_mc import sample_halfspace
    return np, Medium, HalfSpaceSolver, HalfSpaceSettings, SurfacePort, HalfSpaceBoundary, sample_halfspace


def cases(profiles):
    config = json.loads(Path(profiles).read_text())
    by_id = {p["id"]: p for p in config["profiles"]}
    milk, skin = by_id["milk_dn"], by_id["skin_dermis_660_hg"]
    outside = config["outside_phase_index"]
    return {
        "milk": {"profile": "milk_dn", "mu_a_per_cm": max(milk["absorption_scan_per_cm"]),
                 "mu_s_per_cm": milk["scattering_per_cm"], "g": milk["g"],
                 "n_inside": milk["phase_index_inside"], "n_outside": outside,
                 "base": [97, 192, 8, 32], "polar": [128, 254], "harmonics": 12,
                 "note": "largest value of the explicit illustrative absorption scan"},
        "dermis": {"profile": "skin_dermis_660_hg", "mu_a_per_cm": skin["absorption_per_cm"],
                   "mu_s_per_cm": skin["scattering_per_cm"], "g": skin["g"],
                   "n_inside": skin["phase_index_inside"], "n_outside": outside,
                   "base": [97, 192, 8, 32], "polar": [128, 254], "harmonics": 12,
                   "note": "published-range midpoints, pure-HG option"},
        "g95": {"profile": None, "mu_a_per_cm": .135, "mu_s_per_cm": 220., "g": .95,
                "n_inside": 1.4, "n_outside": 1., "base": [161, 320, 8, 32], "polar": [200, 398],
                "harmonics": 12, "note": "synthetic: dermis-like mu_s'=11/cm and mu_a at g=0.95; not a material"},
    }


def _setup(case):
    np, Medium, HalfSpaceSolver, HalfSpaceSettings, SurfacePort, HalfSpaceBoundary, _ = _imports()
    medium = Medium(100*case["mu_a_per_cm"], 100*case["mu_s_per_cm"], case["g"], case["n_inside"],
                    provenance=case["note"])
    boundary = HalfSpaceBoundary(case["n_inside"]/case["n_outside"])
    lstar = 1/(medium.scattering_per_m*(1-medium.g)+medium.absorption_per_m)
    return np, medium, boundary, lstar, HalfSpaceSolver, HalfSpaceSettings, SurfacePort


def run_job(job):
    """One independent single-threaded calculation; returns arrays as lists."""
    for name in THREADS:
        os.environ[name] = "1"
    np, medium, boundary, lstar, HalfSpaceSolver, HalfSpaceSettings, SurfacePort = _setup(job["case"])
    polar, degree, harmonics, readout = job["settings"]
    settings = HalfSpaceSettings(polar, readout, degree, mode_cache_size=1, azimuth_harmonics=harmonics)
    solver = HalfSpaceSolver(medium, boundary, settings)
    source = SurfacePort.pencil()
    detector = SurfacePort.aperture(ANGLE, polar_nodes=8, azimuth_nodes=16)
    start = perf_counter()
    result = {"name": job["name"], "settings": asdict(settings), "unknowns": len(solver.mu),
              "phase_normalization_error": solver.phase_normalization_error}
    qmax, nodes = job.get("q", [12., 48])
    if job["kind"] == "spectral":
        values, residual = [], 0.
        for wave in WAVES:
            solution = solver.solve(source, q_per_m=(wave/lstar, 0.))
            residual = max(residual, solution.eigen_residual)
            values.append(solution.response(detector))
        values = np.array(values)
        result.update(real=values.real.tolist(), imag=values.imag.tolist(), eigen_residual=residual)
    elif job["kind"] == "radial":
        values = solver.radial_response(source, detector, np.array(RADII)*lstar, q_max_per_m=qmax/lstar,
                                        q_nodes=nodes, source_sigma_m=SIGMA*lstar, detector_sigma_m=SIGMA*lstar)
        result.update(real=values.real.tolist(), imag=values.imag.tolist())
    else:
        points, weight = cell_points(np, np.array(CELLS)*lstar)
        values = solver.banana(source, detector, points, detector_xy_m=(OFFSET*lstar, 0.),
                               source_sigma_m=SIGMA*lstar, detector_sigma_m=SIGMA*lstar,
                               q_max_per_m=qmax/lstar, q_nodes=nodes, by_order=True)
        cells = np.einsum('cpo,p->co', values.reshape(len(CELLS), -1, 3), weight)
        result.update(real=cells.real.tolist(), imag=cells.imag.tolist())
    result["q"] = [qmax, nodes]
    result["wall_seconds"] = perf_counter()-start
    return result


def run_monte_carlo(job):
    for name in THREADS:
        os.environ[name] = "1"
    np, medium, boundary, lstar, _, _, SurfacePort = _setup(job["case"])
    from lighthit.experimental.halfspace_mc import sample_halfspace
    source = SurfacePort.pencil()
    start = perf_counter()
    max_events = max(10000, int(math.ceil(100*medium.extinction_per_m/medium.absorption_per_m)))
    result = {"name": job["name"], "photons": job["photons"], "seed": job["seed"], "max_events": max_events}
    if job["kind"] == "spectral":
        histories = sample_halfspace(medium, boundary, source, photons=job["photons"], seed=job["seed"],
                                     max_events=max_events)
        estimate = histories.spectrum(np.column_stack([np.array(WAVES)/lstar, np.zeros(len(WAVES))]), [0.],
                                      half_angle_rad=ANGLE)
        result.update(mean_real=estimate.mean[0].real.tolist(), mean_imag=estimate.mean[0].imag.tolist(),
                      stderr_real=estimate.stderr_real[0].tolist(), stderr_imag=estimate.stderr_imag[0].tolist())
    else:
        histories = sample_halfspace(medium, boundary, source, photons=job["photons"], seed=job["seed"],
                                     max_events=max_events, source_sigma_m=SIGMA*lstar,
                                     sensitivity_cells_m=np.array(CELLS)*lstar)
        radial = histories.radial_density(np.array(RADII)*lstar, detector_sigma_m=SIGMA*lstar, half_angle_rad=ANGLE)
        cells = histories.cell_absorption_sensitivity(detector_xy_m=(OFFSET*lstar, 0.), detector_sigma_m=SIGMA*lstar,
                                                      half_angle_rad=ANGLE, by_order=True)
        # Effective number of contributing histories, (sum v)^2/sum v^2.
        gaussian = np.exp(-((histories.x_m-OFFSET*lstar)**2+histories.y_m**2)/(2*(SIGMA*lstar)**2))
        weight = histories._weights(ANGLE)*gaussian
        total, square = np.zeros((len(CELLS), 3)), np.zeros((len(CELLS), 3))
        for order in range(3):
            value = histories.cell_path_m*(weight*(histories.order == order))[:, None]
            total[:, order], square[:, order] = value.sum(axis=0), (value*value).sum(axis=0)
        result.update(radial_mean=radial.mean.tolist(), radial_stderr=radial.stderr_real.tolist(),
                      cell_mean=cells.mean.tolist(), cell_stderr=cells.stderr_real.tolist(),
                      cell_path_sum=total.tolist(), cell_path_square=square.tolist())
    result["balance"] = histories.photon_balance()
    result["wall_seconds"] = perf_counter()-start
    return result


def merge_chunks(np, chunks):
    """Exact mean/stderr of the union of independent equal-estimator chunks."""
    count = np.array([c["photons"] for c in chunks], float)
    merged = {"name": chunks[0]["name"].rsplit("#", 1)[0], "photons": int(count.sum()),
              "seeds": [c["seed"] for c in chunks], "max_events": chunks[0]["max_events"],
              "wall_seconds": float(sum(c["wall_seconds"] for c in chunks)),
              "balance": {k: float(np.dot(count, [c["balance"][k] for c in chunks])/count.sum())
                          for k in chunks[0]["balance"]}}
    pairs = [("mean_real", "stderr_real"), ("mean_imag", "stderr_imag"),
             ("radial_mean", "radial_stderr"), ("cell_mean", "cell_stderr")]
    for mean_key, error_key in pairs:
        if mean_key not in chunks[0]:
            continue
        mean = np.array([c[mean_key] for c in chunks])
        error = np.array([c[error_key] for c in chunks])
        shape = (-1,) + (1,)*(mean.ndim-1)
        second = (error**2*(count-1).reshape(shape) + mean**2)
        total_mean = np.tensordot(count, mean, 1)/count.sum()
        total_second = np.tensordot(count, second, 1)/count.sum()
        merged[mean_key] = total_mean.tolist()
        merged[error_key] = np.sqrt(np.maximum(0., (total_second-total_mean**2)/(count.sum()-1))).tolist()
    if "cell_path_sum" in chunks[0]:
        total = np.sum([c["cell_path_sum"] for c in chunks], axis=0)
        square = np.sum([c["cell_path_square"] for c in chunks], axis=0)
        merged["cell_effective_histories"] = np.divide(total*total, square, out=np.zeros_like(total),
                                                       where=square > 0).tolist()
    return merged


def cell_points(np, cells, order=4):
    x, w = np.polynomial.legendre.leggauss(order)
    grid = np.stack(np.meshgrid(x, x, x, indexing="ij"), axis=-1).reshape(-1, 3)
    weight = np.prod(np.stack(np.meshgrid(w, w, w, indexing="ij"), axis=-1), axis=-1).ravel()/8
    return np.concatenate([(low+high)/2+grid*(high-low)/2 for low, high in cells]), weight


def jobs_for(name, case, full):
    p, l, m, a = case["base"]
    pp, lp = case["polar"]
    base = {"case": case, "settings": [p, l, m, a]}
    jobs = [dict(base, name=f"{name}/spectral/base", kind="spectral"),
            dict(base, name=f"{name}/spectral/polar", kind="spectral", settings=[pp, lp, m, a]),
            dict(base, name=f"{name}/spectral/harmonics", kind="spectral", settings=[p, l, case["harmonics"], 2*(case["harmonics"]+1)]),
            dict(base, name=f"{name}/banana/base", kind="banana"),
            dict(base, name=f"{name}/banana/polar", kind="banana", settings=[pp, lp, m, a])]
    if full:
        jobs += [dict(base, name=f"{name}/radial/base", kind="radial"),
                 dict(base, name=f"{name}/radial/q", kind="radial", q=[14., 64]),
                 dict(base, name=f"{name}/banana/harmonics", kind="banana",
                      settings=[p, l, case["harmonics"], max(a, 2*(case["harmonics"]+1))]),
                 dict(base, name=f"{name}/banana/readout", kind="banana", settings=[p, l, m, 2*a]),
                 dict(base, name=f"{name}/banana/q", kind="banana", q=[14., 64])]
    return jobs


def summarize(np, name, case, results, mc):
    def value(key):
        r = results[f"{name}/{key}"]
        return np.array(r["real"])+1j*np.array(r["imag"])

    def pull(model, mean, stderr):
        stderr = np.maximum(np.asarray(stderr), 1e-14)
        return np.abs(np.real(model)-np.asarray(mean))/stderr

    def change(a, b, scale):
        return (np.max(np.abs(a-b), axis=0)/scale).tolist()

    out = {"case": case, "checks": {}}
    spectral = value("spectral/base")
    dc = np.maximum(np.abs(spectral[0]), 1e-300)
    dc[0] = max(np.abs(spectral[0]).max(), 1e-300)
    mcs = mc[f"{name}/mc/spectral"]
    mean = np.array(mcs["mean_real"])+1j*np.array(mcs["mean_imag"])
    pr = pull(spectral, mean.real, mcs["stderr_real"])
    pi = np.abs(spectral.imag-mean.imag)/np.maximum(np.array(mcs["stderr_imag"]), 1e-14)
    out["spectral"] = {"q_per_lstar": WAVES, "model": [spectral.real.tolist(), spectral.imag.tolist()],
                       "mc_mean": [mean.real.tolist(), mean.imag.tolist()],
                       "mc_stderr": [mcs["stderr_real"], mcs["stderr_imag"]],
                       "pull_per_order": np.maximum(pr, pi).max(axis=0).tolist(),
                       "polar_change_over_dc_per_order": change(spectral, value("spectral/polar"), dc),
                       "harmonic_change_over_dc_per_order": change(spectral, value("spectral/harmonics"), dc)}
    out["checks"]["spectral_mc"] = float(np.maximum(pr, pi).max()) <= CRITERIA["max_mc_pull"]
    out["checks"]["spectral_grid"] = max(max(out["spectral"]["polar_change_over_dc_per_order"][1:]),
                                         max(out["spectral"]["harmonic_change_over_dc_per_order"][1:])) <= CRITERIA["spectral_grid_change_over_dc"]
    mcg = mc[f"{name}/mc/gaussian"]
    banana = value("banana/base")
    scale = np.maximum(np.abs(banana).max(axis=0), 1e-300)
    resolved = np.array(mcg["cell_effective_histories"]) >= CRITERIA["min_effective_histories"]
    resolved[:, 0] = True  # order 0 is identically zero in both references
    pull_cells = np.where(resolved, pull(banana, mcg["cell_mean"], mcg["cell_stderr"]), 0.)
    out["banana"] = {"cells_per_lstar": CELLS, "model_per_m4": [banana.real.tolist(), banana.imag.tolist()],
                     "mc_mean": mcg["cell_mean"], "mc_stderr": mcg["cell_stderr"],
                     "pull_per_order": pull_cells.max(axis=0).tolist(),
                     "mc_effective_histories": mcg["cell_effective_histories"],
                     "unresolved_by_mc": np.argwhere(~resolved).tolist(),
                     "relative_to_mc_per_cell_order": (np.real(banana)/np.where(np.array(mcg["cell_mean"]) == 0, np.nan,
                                                                               np.array(mcg["cell_mean"]))).tolist(),
                     "polar_change_per_order": change(banana, value("banana/polar"), scale)}
    out["checks"]["banana_mc"] = float(pull_cells.max()) <= CRITERIA["max_mc_pull"]
    out["checks"]["banana_polar"] = max(out["banana"]["polar_change_per_order"][1:]) <= CRITERIA["banana_angular_relative"]
    if f"{name}/radial/base" in results:
        for key, label in (("banana/harmonics", "harmonic_change_per_order"), ("banana/readout", "readout_change_per_order"),
                           ("banana/q", "q_change_per_order")):
            out["banana"][label] = change(banana, value(key), scale)
        out["checks"]["banana_harmonics"] = max(out["banana"]["harmonic_change_per_order"][1:]) <= CRITERIA["banana_angular_relative"]
        out["checks"]["banana_readout"] = max(out["banana"]["readout_change_per_order"][1:]) <= CRITERIA["banana_angular_relative"]
        out["checks"]["banana_q"] = max(out["banana"]["q_change_per_order"][1:]) <= CRITERIA["banana_radial_relative"]
        radial = value("radial/base")
        rscale = np.maximum(np.abs(radial).max(axis=0), 1e-300)
        pull_radial = pull(radial, mcg["radial_mean"], mcg["radial_stderr"])
        out["radial"] = {"radii_per_lstar": RADII, "model_per_m2": [radial.real.tolist(), radial.imag.tolist()],
                         "mc_mean": mcg["radial_mean"], "mc_stderr": mcg["radial_stderr"],
                         "pull_per_order": pull_radial.max(axis=0).tolist(),
                         "q_change_per_order": change(radial, value("radial/q"), rscale)}
        out["checks"]["radial_mc"] = float(pull_radial.max()) <= CRITERIA["max_mc_pull"]
        out["checks"]["radial_q"] = max(out["radial"]["q_change_per_order"][1:]) <= CRITERIA["radial_quadrature_relative"]
    out["eigen_residual"] = results[f"{name}/spectral/base"]["eigen_residual"]
    out["checks"]["eigen_residual"] = out["eigen_residual"] <= CRITERIA["eigen_residual"]
    out["timings_seconds"] = {k.split("/", 1)[1]: v["wall_seconds"] for k, v in {**results, **mc}.items()
                              if k.startswith(name + "/")}
    out["checks"] = {key: bool(flag) for key, flag in out["checks"].items()}
    out["passed"] = bool(all(out["checks"].values()))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, default=ROOT/"experiments"/"fixtures"/"halfspace-materials.json")
    parser.add_argument("--cases", default="milk,dermis,g95")
    parser.add_argument("--reduced", default="g95",
                        help="cases without the harmonic/readout/q refinements and radial checks")
    parser.add_argument("--photons", type=int, default=20_000_000)
    parser.add_argument("--banana-photons", type=int, default=100_000_000)
    parser.add_argument("--chunk", type=int, default=5_000_000, help="histories per independent MC chunk")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--smoke", action="store_true",
                        help="tiny resolutions and statistics to exercise the pipeline; not a validation")
    args = parser.parse_args()
    started = perf_counter()
    for name in THREADS:
        os.environ[name] = "1"
    np = _imports()[0]
    import numba
    import scipy
    table = cases(args.profiles)
    if args.smoke:
        for case in table.values():
            case.update(base=[40, 78, 3, 8], polar=[44, 86], harmonics=4)
    names = [n for n in args.cases.split(",") if n]
    reduced = set(args.reduced.split(","))
    jobs, mc_jobs = [], []
    for index, name in enumerate(names):
        jobs += jobs_for(name, table[name], name not in reduced)
        for kind, photons, offset in (("spectral", args.photons, 0), ("gaussian", args.banana_photons, 1)):
            chunks = max(1, math.ceil(photons/args.chunk))
            for chunk in range(chunks):
                mc_jobs.append({"case": table[name], "name": f"{name}/mc/{kind}#{chunk}", "kind": kind,
                                "photons": min(args.chunk, photons-chunk*args.chunk),
                                "seed": 20261104 + 1000*(2*index+offset) + chunk})
    # Longest first: banana refinements dominate.
    if args.smoke:
        for job in jobs:
            job["q"] = [12., 6] if "/q" not in job["name"] else [14., 8]
    jobs.sort(key=lambda j: -(j["settings"][0]**3*(j["settings"][2]+1)**3*(j.get("q", [0, 48])[1])
                              * (1 if j["kind"] != "spectral" else .1)))
    results, chunks = {}, {}
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
        futures = [(j["name"], pool.submit(run_job, j)) for j in jobs]
        futures += [(j["name"], pool.submit(run_monte_carlo, j)) for j in mc_jobs]
        for job_name, future in futures:
            result = future.result()
            if "/mc/" in job_name:
                chunks.setdefault(job_name.rsplit("#", 1)[0], []).append(result)
            else:
                results[job_name] = result
            print(f"{job_name}: {result['wall_seconds']:.1f} s", flush=True)
    mc = {key: merge_chunks(np, value) for key, value in chunks.items()}
    report = {"schema": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
              "runtime": {"python": sys.version.split()[0], "numpy": np.__version__,
                          "scipy": scipy.__version__, "numba": numba.__version__, "threads_per_job": 1},
              "geometry": {"source": "normal pencil", "detector": f"aperture {ANGLE} rad, 8x16 exterior nodes",
                           "footprint_sigma_lstar": SIGMA, "detector_offset_lstar": OFFSET,
                           "lengths": "per transport length l* = 1/(mu_s(1-g)+mu_a)"},
              "volume_orders": [0, 1, ">=2"], "criteria": CRITERIA,
              "profiles_sha256": hashlib.sha256(args.profiles.read_bytes()).hexdigest(),
              "jobs": results, "monte_carlo": mc, "cases": {}}
    for name in names:
        report["cases"][name] = summarize(np, name, table[name], results, mc)
        c = report["cases"][name]
        print(f"{name}: spectral pull/order {np.round(c['spectral']['pull_per_order'], 3).tolist()}, "
              f"banana pull/order {np.round(c['banana']['pull_per_order'], 3).tolist()}, "
              f"{'PASS' if c['passed'] else 'FAIL'}", flush=True)
    files = ["src/lighthit/experimental/halfspace.py", "src/lighthit/experimental/halfspace_boundary.py",
             "src/lighthit/experimental/halfspace_mc.py", "experiments/validate_halfspace_highg.py"]
    report["source_sha256"] = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files}
    report["passed"] = bool(all(c["passed"] for c in report["cases"].values()))
    report["wall_seconds"] = perf_counter()-started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    print(f"{'PASS' if report['passed'] else 'FAIL'}: {len(names)} cases, {report['wall_seconds']:.1f} s", flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
