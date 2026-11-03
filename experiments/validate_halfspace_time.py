#!/usr/bin/env python
"""Check paired time spectra, frequency convergence and continuous full-HG MC.

The two spectra must describe the same axial normal-pencil experiment.
The fine frequency grid must refine the coarse grid by two. The default
time window and Gaussian response reproduce the illustrative milk check.
Near-zero bins use the analytic omitted-band bound as an absolute check.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from scipy.special import ndtr, erfc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))

from lighthit.medium import Medium
from lighthit.readout import inverse_bins
from lighthit.experimental.halfspace import SurfacePort
from lighthit.experimental.halfspace_boundary import HalfSpaceBoundary
from lighthit.experimental.halfspace_mc import sample_halfspace


def load_spectrum(path):
    with np.load(path, allow_pickle=False) as data:
        return (data["omega_per_ns"].copy(), data["spectrum"].copy(),
                json.loads(str(data["metadata_json"])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coarse-spectrum", type=Path, required=True)
    parser.add_argument("--fine-spectrum", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--photons", type=int, default=300000)
    args = parser.parse_args()
    wc, hc, mc = load_spectrum(args.coarse_spectrum)
    wf, hf, mf = load_spectrum(args.fine_spectrum)
    np.testing.assert_allclose(wc, wf[::2], rtol=0, atol=1e-13)
    physical = ("mu_a_per_cm", "mu_s_per_cm", "g", "phase_index_ratio",
                "group_index", "wavelength_nm", "aperture_deg")
    if any(mc["parameters"][key] != mf["parameters"][key] for key in physical):
        raise ValueError("coarse and fine spectra must have identical physical parameters")
    if wf[-1] < 60:
        raise ValueError("this time-band check requires a cutoff of at least 60 rad/ns")
    parameters = mf["parameters"]
    edges = np.linspace(-.5, 10., 211)
    sigma = .1
    if 2*np.pi/np.max(np.diff(wc)) <= edges[-1]-edges[0]:
        raise ValueError("the coarse frequency period is shorter than the test window")
    bc = inverse_bins(wc, hc, edges, sigma).T
    bf = inverse_bins(wf, hf, edges, sigma).T
    same_step = inverse_bins(wf[::2], hf[::2], edges, sigma).T
    low = wf <= 40.+1e-12
    low_band = inverse_bins(wf[low], hf[low], edges, sigma).T
    dc = float(hf[0].sum().real)
    if dc <= 0:
        raise ValueError("the test requires positive scattered charge")

    def relative_l1(a, b):
        return float(np.abs(a-b).sum()/dc)

    errors = {"angular_relative_l1": relative_l1(bc, same_step),
              "frequency_step_relative_l1": relative_l1(same_step, bf),
              "band_40_to_60_relative_l1": relative_l1(low_band, bf),
              "combined_default_to_fine_relative_l1": relative_l1(bc, bf)}
    tail_bound = (dc*(edges[-1]-edges[0])/(sigma*np.sqrt(2*np.pi))
                  * erfc(wf[-1]*sigma/np.sqrt(2)))
    print("time convergence:", errors, flush=True)
    started = perf_counter()
    medium = Medium(100*parameters["mu_a_per_cm"], 100*parameters["mu_s_per_cm"],
                    parameters["g"], parameters["group_index"], parameters["wavelength_nm"],
                    "specified time-profile validation model")
    boundary = HalfSpaceBoundary(parameters["phase_index_ratio"])
    max_events = max(10000, int(math.ceil(100*medium.extinction_per_m/medium.absorption_per_m)))
    histories = sample_halfspace(medium, boundary, SurfacePort.pencil(), photons=args.photons,
                                 seed=20261201, max_events=max_events)
    mc_edges = np.array([-.5, -.2, -.1, 0., .1, .2, .4, .8, 1.5, 3., 5., 10.])
    theory = inverse_bins(wf, hf, mc_edges, sigma).T
    time = histories.path_m/medium.speed_m_per_ns
    angle = np.deg2rad(parameters["aperture_deg"])
    weight = histories.entered_weight*((histories.order >= 0)
                                       & (histories.cosine_external >= np.cos(angle)))
    mean, se = np.zeros_like(theory), np.zeros_like(theory)
    for i, (left, right) in enumerate(zip(mc_edges[:-1], mc_edges[1:])):
        value = weight*(ndtr((right-time)/sigma)-ndtr((left-time)/sigma))
        for order in (1, 2):
            x = value*(histories.order == order)
            mean[i, order] = x.mean()
            se[i, order] = np.sqrt(max(0., (np.mean(x*x)-x.mean()**2)/(args.photons-1)))
    delta = np.abs(theory-mean)
    raw_pull = delta/np.maximum(se, 1e-14)
    per_bin_bound = (np.diff(mc_edges)[:, None]*np.maximum(hf[0].real, 0)[None, :]
                     /(sigma*np.sqrt(2*np.pi))*erfc(wf[-1]*sigma/np.sqrt(2)))
    resolved = se > per_bin_bound
    max_resolved = float(np.max(np.where(resolved, raw_pull, 0)))
    tail_ok = bool(np.all(delta[~resolved] <= per_bin_bound[~resolved]))
    passed = all(v < .001 for v in errors.values()) and max_resolved < 4.5 and tail_ok
    report = {"schema": 1, "parameters": parameters, "sigma_ns": sigma,
              "time_edges_ns": edges.tolist(), "errors": errors,
              "gaussian_omitted_band_l1_bound": float(tail_bound),
              "monte_carlo": {"photons": args.photons, "seed": 20261201,
                               "max_events": max_events, "time_edges_ns": mc_edges.tolist(),
                               "rte": theory.tolist(), "mean": mean.tolist(), "stderr": se.tolist(),
                               "max_statistical_only_pull": float(raw_pull.max()),
                               "max_resolved_pull": max_resolved,
                               "gaussian_band_bin_bounds": per_bin_bound.tolist(),
                               "unresolved_tails_within_absolute_bound": tail_ok,
                               "wall_seconds": perf_counter()-started},
              "criteria": {"relative_l1": .001, "max_mc_pull": 4.5,
                           "unresolved_tails": "absolute omitted Gaussian-band bound"},
              "passed": bool(passed)}
    files = ["src/lighthit/experimental/halfspace.py", "src/lighthit/experimental/halfspace_mc.py",
             "src/lighthit/readout.py", "experiments/validate_halfspace_time.py"]
    report["source_sha256"] = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files}
    report["spectrum_sha256"] = {"coarse": hashlib.sha256(args.coarse_spectrum.read_bytes()).hexdigest(),
                                  "fine": hashlib.sha256(args.fine_spectrum.read_bytes()).hexdigest()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(f"{'PASS' if passed else 'FAIL'}: resolved MC pull={max_resolved:.3f}, absolute tails={tail_ok}", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
