#!/usr/bin/env python
"""Validate the half-space frequency and time response at high g and q != 0.

Example from the checkout:
  python experiments/validate_halfspace_time_highg.py --output time-highg.json --workers 3

Media, ports, footprints and cosine-moment settings are those of
validate_halfspace_highg.py, now at nonzero frequency. Lengths are in l*,
times in l*/c (c in the medium) and frequencies in c/l*. Every collision
order (0, 1, >=2) is compared separately with independent full-HG histories:

- spectral: escape spectrum at q l* = 0.5, 2, 5 and omega l*/c = 0.3, 3,
  with polar/phase and azimuthal-moment refinement;
- time: the q l* = 1 component of the escaping flux in time with a Gaussian
  timing response of sigma = 2 l*/c, from 49 frequencies (period 100 l*/c);
  every second frequency (period 50 l*/c) and the band below 2 c/l* are
  compared with it as in validate_halfspace_time.py;
- banana: absorption sensitivity of the three cells at omega l*/c = 0.3.

Each job runs single-threaded; jobs may run in parallel processes.
Failures are preserved in the report.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import multiprocessing
import os
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("validate_halfspace_highg", ROOT/"experiments"/"validate_halfspace_highg.py")
highg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(highg)
THREADS, CELLS, SIGMA, OFFSET, ANGLE = highg.THREADS, highg.CELLS, highg.SIGMA, highg.OFFSET, highg.ANGLE

SPECTRAL_Q = [.5, 2., 5.]
SPECTRAL_OMEGA = [.3, 3.]
TIME_Q = 1.
TIME_PERIOD = 100.
TIME_FREQUENCIES = 49
TIME_SIGMA = 2.
TIME_BAND = 2.
TIME_EDGES = [-6. + .5*k for k in range(101)]
MC_TIME_EDGES = [-6., -2., 0., 1., 2., 3., 4., 6., 8., 11., 15., 20., 27., 35., 44.]
BANANA_OMEGA = .3
CRITERIA = {"max_mc_pull": 4.5, "spectral_grid_change": .005, "time_relative_l1": .001,
            "eigen_residual": 1e-10, "min_effective_histories": 100}


def time_frequencies():
    return [2*math.pi*k/TIME_PERIOD for k in range(TIME_FREQUENCIES)]


def run_job(job):
    """One independent single-threaded calculation; returns arrays as lists."""
    for name in THREADS:
        os.environ[name] = "1"
    np, medium, boundary, lstar, HalfSpaceSolver, HalfSpaceSettings, SurfacePort = highg._setup(job["case"])
    unit = medium.speed_m_per_ns/lstar
    polar, degree, harmonics, readout = job["settings"]
    settings = HalfSpaceSettings(polar, readout, degree, mode_cache_size=1, azimuth_harmonics=harmonics)
    solver = HalfSpaceSolver(medium, boundary, settings)
    source = SurfacePort.pencil()
    detector = SurfacePort.aperture(ANGLE, polar_nodes=8, azimuth_nodes=16)
    start = perf_counter()
    result = {"name": job["name"], "settings": asdict(settings), "unknowns": len(solver.mu)}
    residual = 0.
    if job["kind"] == "spectral":
        values = []
        for omega in SPECTRAL_OMEGA:
            row = []
            for wave in SPECTRAL_Q:
                solution = solver.solve(source, q_per_m=(wave/lstar, 0.), omega_per_ns=omega*unit)
                residual = max(residual, solution.eigen_residual)
                row.append(solution.response(detector))
            values.append(row)
        values = np.array(values)
    elif job["kind"] == "time":
        values = []
        for omega in time_frequencies():
            solution = solver.solve(source, q_per_m=(TIME_Q/lstar, 0.), omega_per_ns=omega*unit)
            residual = max(residual, solution.eigen_residual)
            values.append(solution.response(detector))
        values = np.array(values)
    else:
        qmax, nodes = job.get("q", [12., 48])
        points, weight = highg.cell_points(np, np.array(CELLS)*lstar)
        values = solver.banana(source, detector, points, detector_xy_m=(OFFSET*lstar, 0.),
                               source_sigma_m=SIGMA*lstar, detector_sigma_m=SIGMA*lstar,
                               q_max_per_m=qmax/lstar, q_nodes=nodes, omega_per_ns=BANANA_OMEGA*unit,
                               by_order=True)
        values = np.einsum('cpo,p->co', values.reshape(len(CELLS), -1, 3), weight)
        result["q"] = [qmax, nodes]
    result.update(real=values.real.tolist(), imag=values.imag.tolist(), eigen_residual=residual,
                  wall_seconds=perf_counter()-start)
    return result


def _moments(np, rows, shape, order):
    """Per-order sums for exact chunk merging; ``rows`` yields one estimator at a time."""
    sums = {key: np.zeros(shape + (3,)) for key in ("re", "re2", "im", "im2", "abs", "abs2")}
    masks = [order == k for k in range(3)]
    for index, value in zip(np.ndindex(*shape), rows):
        for k, mask in enumerate(masks):
            x = value[mask]
            magnitude = np.abs(x)
            sums["re"][index + (k,)], sums["re2"][index + (k,)] = x.real.sum(), (x.real**2).sum()
            sums["im"][index + (k,)], sums["im2"][index + (k,)] = x.imag.sum(), (x.imag**2).sum()
            sums["abs"][index + (k,)], sums["abs2"][index + (k,)] = magnitude.sum(), (magnitude**2).sum()
    return {key: value.tolist() for key, value in sums.items()}


def run_monte_carlo(job):
    for name in THREADS:
        os.environ[name] = "1"
    np, medium, boundary, lstar, _, _, SurfacePort = highg._setup(job["case"])
    from scipy.special import ndtr
    from lighthit.experimental.halfspace_mc import sample_halfspace
    unit = medium.speed_m_per_ns/lstar
    tau = lstar/medium.speed_m_per_ns
    source = SurfacePort.pencil()
    start = perf_counter()
    max_events = max(10000, int(math.ceil(100*medium.extinction_per_m/medium.absorption_per_m)))
    result = {"name": job["name"], "photons": job["photons"], "seed": job["seed"], "max_events": max_events,
              "estimators": {}}
    if job["kind"] == "spectral":
        h = sample_halfspace(medium, boundary, source, photons=job["photons"], seed=job["seed"],
                             max_events=max_events)
        weight = h._weights(ANGLE)
        time = h.path_m/medium.speed_m_per_ns
        rows = (weight*np.exp(1j*(omega*unit*time - wave/lstar*h.x_m))
                for omega in SPECTRAL_OMEGA for wave in SPECTRAL_Q)
        result["estimators"]["spectral"] = _moments(np, rows, (len(SPECTRAL_OMEGA), len(SPECTRAL_Q)), h.order)
        edges = np.array(MC_TIME_EDGES)*tau
        sigma = TIME_SIGMA*tau
        profile = weight*np.cos(TIME_Q/lstar*h.x_m)
        rows = (profile*(ndtr((right-time)/sigma)-ndtr((left-time)/sigma))
                for left, right in zip(edges[:-1], edges[1:]))
        result["estimators"]["time"] = _moments(np, rows, (len(edges)-1,), h.order)
    else:
        h = sample_halfspace(medium, boundary, source, photons=job["photons"], seed=job["seed"],
                             max_events=max_events, source_sigma_m=SIGMA*lstar,
                             sensitivity_cells_m=np.array(CELLS)*lstar)
        gaussian = np.exp(-((h.x_m-OFFSET*lstar)**2+h.y_m**2)/(2*(SIGMA*lstar)**2))/(2*np.pi*(SIGMA*lstar)**2)
        weight = h._weights(ANGLE)*gaussian*np.exp(1j*BANANA_OMEGA*unit*h.path_m/medium.speed_m_per_ns)
        volume = np.prod(h.cells_m[:, 1]-h.cells_m[:, 0], axis=1)
        rows = (-h.cell_path_m[:, c]*weight/volume[c] for c in range(len(CELLS)))
        result["estimators"]["banana"] = _moments(np, rows, (len(CELLS),), h.order)
    result["count"] = int(job["photons"])
    result["balance"] = h.photon_balance()
    result["wall_seconds"] = perf_counter()-start
    return result


def merge_chunks(np, chunks):
    """Exact mean, standard error and effective histories of the union of chunks."""
    count = sum(c["count"] for c in chunks)
    merged = {"name": chunks[0]["name"].rsplit("#", 1)[0], "photons": count,
              "seeds": [c["seed"] for c in chunks], "max_events": chunks[0]["max_events"],
              "wall_seconds": float(sum(c["wall_seconds"] for c in chunks)),
              "balance": {k: float(sum(c["count"]*c["balance"][k] for c in chunks)/count)
                          for k in chunks[0]["balance"]}}
    for key in chunks[0]["estimators"]:
        total = {part: np.sum([c["estimators"][key][part] for c in chunks], axis=0)
                 for part in chunks[0]["estimators"][key]}
        out = {}
        for part in ("re", "im"):
            mean = total[part]/count
            out[f"mean_{part}"] = mean.tolist()
            out[f"stderr_{part}"] = np.sqrt(np.maximum(0., (total[part + "2"]/count-mean**2)/(count-1))).tolist()
        # Effective number of contributing histories, (sum |v|)^2/sum |v|^2.
        out["effective_histories"] = np.divide(total["abs"]**2, total["abs2"], out=np.zeros_like(total["abs"]),
                                               where=total["abs2"] > 0).tolist()
        merged[key] = out
    return merged


def jobs_for(name, case, settings_only=False):
    p, l, m, a = case["base"]
    pp, lp = case["polar"]
    base = {"case": case, "settings": [p, l, m, a]}
    return [dict(base, name=f"{name}/time/base", kind="time"),
            dict(base, name=f"{name}/banana/base", kind="banana"),
            dict(base, name=f"{name}/spectral/base", kind="spectral"),
            dict(base, name=f"{name}/spectral/polar", kind="spectral", settings=[pp, lp, m, a]),
            dict(base, name=f"{name}/spectral/harmonics", kind="spectral",
                 settings=[p, l, case["harmonics"], 2*(case["harmonics"]+1)])]


def summarize(np, name, case, results, mc):
    from lighthit.readout import inverse_bins
    from scipy.special import erfc

    def value(key):
        r = results[f"{name}/{key}"]
        return np.array(r["real"])+1j*np.array(r["imag"])

    def compare(model, estimate, extra_resolved=None):
        mean = np.array(estimate["mean_re"])+1j*np.array(estimate["mean_im"])
        resolved = np.array(estimate["effective_histories"]) >= CRITERIA["min_effective_histories"]
        resolved[..., 0] = True  # order 0 is identically zero in both references
        if extra_resolved is not None:
            resolved &= extra_resolved
        pulls = np.maximum(np.abs(model.real-mean.real)/np.maximum(np.array(estimate["stderr_re"]), 1e-300),
                           np.abs(model.imag-mean.imag)/np.maximum(np.array(estimate["stderr_im"]), 1e-300))
        pulls = np.where(resolved, np.where(np.abs(model-mean) == 0, 0., pulls), 0.)
        axes = tuple(range(pulls.ndim-1))
        return mean, resolved, pulls.max(axis=axes)

    out = {"case": case, "checks": {}}
    mcs, mcg = mc[f"{name}/mc/spectral"], mc[f"{name}/mc/gaussian"]
    spectral = value("spectral/base")
    scale = np.maximum(np.abs(spectral).max(axis=(0, 1)), 1e-300)
    mean, resolved, pulls = compare(spectral, mcs["spectral"])
    polar = (np.abs(value("spectral/polar")-spectral).max(axis=(0, 1))/scale).tolist()
    harmonic = (np.abs(value("spectral/harmonics")-spectral).max(axis=(0, 1))/scale).tolist()
    out["spectral"] = {"q_per_lstar": SPECTRAL_Q, "omega_per_c_over_lstar": SPECTRAL_OMEGA,
                       "model": [spectral.real.tolist(), spectral.imag.tolist()],
                       "mc_mean": [mean.real.tolist(), mean.imag.tolist()],
                       "mc_stderr": [mcs["spectral"]["stderr_re"], mcs["spectral"]["stderr_im"]],
                       "unresolved_by_mc": np.argwhere(~resolved).tolist(),
                       "pull_per_order": pulls.tolist(),
                       "polar_change_per_order": polar, "harmonic_change_per_order": harmonic}
    out["checks"]["spectral_mc"] = float(pulls.max()) <= CRITERIA["max_mc_pull"]
    out["checks"]["spectral_grid"] = max(polar[1:]+harmonic[1:]) <= CRITERIA["spectral_grid_change"]

    _, medium, _, lstar, *_ = highg._setup(case)
    tau = lstar/medium.speed_m_per_ns
    omega = np.array(time_frequencies())/tau
    spectrum = value("time/base")
    sigma = TIME_SIGMA*tau
    edges = np.array(TIME_EDGES)*tau
    fine = inverse_bins(omega, spectrum, edges, sigma).T
    coarse = inverse_bins(omega[::2], spectrum[::2], edges, sigma).T
    band = omega <= TIME_BAND/tau + 1e-12
    low = inverse_bins(omega[band], spectrum[band], edges, sigma).T
    dc = float(spectrum[0].sum().real)
    errors = {"frequency_step_relative_l1": float(np.abs(fine-coarse).sum()/dc),
              "band_relative_l1": float(np.abs(fine-low).sum()/dc)}
    mc_edges = np.array(MC_TIME_EDGES)*tau
    theory = inverse_bins(omega, spectrum, mc_edges, sigma).T
    bound = (np.diff(mc_edges)[:, None]*np.maximum(spectrum[0].real, 0)[None, :]
             / (sigma*np.sqrt(2*np.pi))*erfc(omega[-1]*sigma/np.sqrt(2)))
    estimate = mcs["time"]
    tmean = np.array(estimate["mean_re"])
    above = np.array(estimate["stderr_re"]) > bound
    _, tresolved, tpulls = compare(theory.astype(complex), estimate, above)
    effective = np.array(estimate["effective_histories"]) >= CRITERIA["min_effective_histories"]
    tails = ~above & effective
    tail_ok = bool(np.all(np.abs(theory-tmean)[tails] <= bound[tails]))
    out["time"] = {"q_per_lstar": TIME_Q, "sigma_lstar_over_c": TIME_SIGMA, "period_lstar_over_c": TIME_PERIOD,
                   "frequencies": len(omega), "band_per_c_over_lstar": TIME_BAND,
                   "time_edges_lstar_over_c": TIME_EDGES, "errors": errors,
                   "mc_time_edges_lstar_over_c": MC_TIME_EDGES, "model": theory.tolist(),
                   "mc_mean": tmean.tolist(), "mc_stderr": estimate["stderr_re"],
                   "mc_effective_histories": estimate["effective_histories"],
                   "unresolved_by_mc": np.argwhere(~effective).tolist(),
                   "gaussian_band_bin_bounds": bound.tolist(), "pull_per_order": tpulls.tolist(),
                   "unresolved_tails_within_absolute_bound": tail_ok,
                   "eigen_residual": results[f"{name}/time/base"]["eigen_residual"]}
    out["checks"]["time_grid"] = max(errors.values()) < CRITERIA["time_relative_l1"]
    out["checks"]["time_mc"] = float(tpulls.max()) <= CRITERIA["max_mc_pull"] and tail_ok

    banana = value("banana/base")
    mean, resolved, pulls = compare(banana, mcg["banana"])
    out["banana"] = {"cells_per_lstar": CELLS, "omega_per_c_over_lstar": BANANA_OMEGA,
                     "model_per_m4": [banana.real.tolist(), banana.imag.tolist()],
                     "mc_mean": [mean.real.tolist(), mean.imag.tolist()],
                     "mc_stderr": [mcg["banana"]["stderr_re"], mcg["banana"]["stderr_im"]],
                     "mc_effective_histories": mcg["banana"]["effective_histories"],
                     "unresolved_by_mc": np.argwhere(~resolved).tolist(),
                     "pull_per_order": pulls.tolist()}
    out["checks"]["banana_mc"] = float(pulls.max()) <= CRITERIA["max_mc_pull"]
    out["eigen_residual"] = max(results[f"{name}/{k}"]["eigen_residual"]
                                for k in ("time/base", "spectral/base", "spectral/polar", "spectral/harmonics"))
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
    parser.add_argument("--cases", default="milk,dermis")
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
    np = highg._imports()[0]
    import numba
    import scipy
    table = highg.cases(args.profiles)
    if args.smoke:
        for case in table.values():
            case.update(base=[40, 78, 3, 8], polar=[44, 86], harmonics=4)
    names = [n for n in args.cases.split(",") if n]
    jobs, mc_jobs = [], []
    for index, name in enumerate(names):
        jobs += jobs_for(name, table[name])
        for kind, photons, offset in (("spectral", args.photons, 0), ("gaussian", args.banana_photons, 1)):
            for chunk in range(max(1, math.ceil(photons/args.chunk))):
                mc_jobs.append({"case": table[name], "name": f"{name}/mc/{kind}#{chunk}", "kind": kind,
                                "photons": min(args.chunk, photons-chunk*args.chunk),
                                "seed": 20261105 + 1000*(2*index+offset) + chunk})
    if args.smoke:
        for job in jobs:
            job["q"] = [12., 6]
    cost = {"time": TIME_FREQUENCIES, "banana": 48, "spectral": len(SPECTRAL_Q)*len(SPECTRAL_OMEGA)}
    jobs.sort(key=lambda j: -(j["settings"][0]*(j["settings"][2]+1))**3*cost[j["kind"]])
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
                           "units": "lengths l* = 1/(mu_s(1-g)+mu_a), times l*/c, frequencies c/l*",
                           "fourier": "spectrum = integral of flux(t) exp(i omega t) dt"},
              "volume_orders": [0, 1, ">=2"], "criteria": CRITERIA,
              "profiles_sha256": hashlib.sha256(args.profiles.read_bytes()).hexdigest(),
              "jobs": results, "monte_carlo": mc, "cases": {}}
    for name in names:
        report["cases"][name] = c = summarize(np, name, table[name], results, mc)
        print(f"{name}: spectral pull/order {np.round(c['spectral']['pull_per_order'], 3).tolist()}, "
              f"time pull/order {np.round(c['time']['pull_per_order'], 3).tolist()}, "
              f"banana pull/order {np.round(c['banana']['pull_per_order'], 3).tolist()}, "
              f"time L1 {c['time']['errors']}, {'PASS' if c['passed'] else 'FAIL'}", flush=True)
    files = ["src/lighthit/experimental/halfspace.py", "src/lighthit/experimental/halfspace_boundary.py",
             "src/lighthit/experimental/halfspace_mc.py", "src/lighthit/readout.py",
             "experiments/validate_halfspace_highg.py", "experiments/validate_halfspace_time_highg.py"]
    report["source_sha256"] = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files}
    report["passed"] = bool(all(c["passed"] for c in report["cases"].values()))
    report["wall_seconds"] = perf_counter()-started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    print(f"{'PASS' if report['passed'] else 'FAIL'}: {len(names)} cases, {report['wall_seconds']:.1f} s", flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
