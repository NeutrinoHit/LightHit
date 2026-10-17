"""Order 0 on the finite OM sphere (``BallisticIntervalConfig.finite_receiver_m``).

* ``finite-receiver-order0/muon-near-om-mc``: order 0 of the 6 GeV muon passing
  OM 267 at 1.76 m against the photon Monte Carlo hits without scattering
  (``photon_mc.py --orders``, all runs found in ``inputs/mc``). LightHit with
  the point receiver, the default (sphere for elements within 20 m), the sphere
  everywhere, and the default on photon yields of the chord length (what NTSim's
  CherenkovGenerator emits) instead of the true Geant4 step length.
* ``finite-receiver-order0/far-<input>``: point receiver against the sphere
  everywhere on all 1440 OMs, by the distance of the OM to the nearest step
  point, with the wall time of each.
"""
from __future__ import annotations

import json
import time
from dataclasses import replace

import numpy as np

from .cases import case
from .inputs import inputs_dir, load_steps

FINE_STRUCTURE = 7.2973525693e-3
NEAR_OMS = (266, 267, 268, 269)
FAR_INPUTS = ("electron-100GeV", "pion-100GeV", "nue-cc", "muon-100GeV")
BINS_M = ((0., 5.), (5., 10.), (10., 20.), (20., 50.), (50., 100.), (100., np.inf))


def _order0(kernel, steps, reach):
    from lighthit.ballistic_interval import ballistic_interval, BallisticIntervalConfig
    config = BallisticIntervalConfig() if reach is None else BallisticIntervalConfig(finite_receiver_m=reach)
    return np.asarray(ballistic_interval(kernel, steps, config=config).charge_pe, float)


def _chord_yields(steps):
    """The same steps emitting along the chord length, as NTSim's CherenkovGenerator does."""
    c0 = 2 * np.pi * FINE_STRUCTURE * 1e9 * np.asarray(steps.length_m, float)
    return replace(steps, coefficient0=c0, coefficient2=-c0 / np.asarray(steps.beta) ** 2)


def _near_mc(ctx, name="muon-near-om"):
    base = inputs_dir() / "mc"
    runs = sorted(base.glob(f"{name}-run*-hits.npz"), key=lambda p: int(p.stem.split("-run")[1].split("-")[0]))
    hits = []
    for path in runs:
        with np.load(path, allow_pickle=False) as data:
            if "order" in data.files:
                hits.append((path, data["uid"], data["phe"], data["order"]))
    if not hits:
        raise FileNotFoundError(f"no {name} hits with scattering orders in {base}: "
                                f"run experiments/photon_mc.py {name} --orders --run N")
    steps_path = hits[0][0].with_name(hits[0][0].name.replace("-hits", "-steps"))
    ctx._note(steps_path, *[h[0] for h in hits])
    steps, _ = load_steps(steps_path)
    kernel = ctx.kernel()
    uids = np.asarray(kernel.detector.identifiers["uid"]).astype(np.int64)
    row = {int(u): i for i, u in enumerate(uids)}
    n = len(uids)
    mc = np.zeros((len(hits), n)); var = np.zeros((len(hits), n))
    for k, (_, uid, phe, order) in enumerate(hits):
        index = np.array([row[int(u)] for u in uid], int)
        direct = order == 0
        mc[k] = np.bincount(index[direct], phe[direct], n)
        var[k] = np.bincount(index[direct], phe[direct] ** 2, n)
    mean, sigma = mc.mean(0), np.sqrt(var.sum(0)) / len(hits)
    variants = dict(point=_order0(kernel, steps, 0.), default=_order0(kernel, steps, None),
                    sphere_everywhere=_order0(kernel, steps, np.inf),
                    default_chord_yields=_order0(kernel, _chord_yields(steps), None))
    per_om = {str(om): dict(mc_pe=float(mean[om]), mc_sigma_pe=float(sigma[om]),
                            **{k: float(v[om]) for k, v in variants.items()}) for om in NEAR_OMS}
    sel = (mean > 0.01) | (variants["default"] > 0.01)
    pulls = {k: ((v - mean) / np.maximum(sigma, 1e-12))[sel & (sigma > 0)] for k, v in variants.items()}
    return dict(
        headline=dict(om267_point_pe=per_om["267"]["point"], om267_default_pe=per_om["267"]["default"],
                      om267_chord_pe=per_om["267"]["default_chord_yields"], om267_mc_pe=per_om["267"]["mc_pe"],
                      om267_mc_sigma_pe=per_om["267"]["mc_sigma_pe"], runs=len(hits)),
        per_om=per_om,
        pulls_rms={k: float(np.sqrt(np.mean(p ** 2))) for k, p in pulls.items()},
        oms_compared=int((sel & (sigma > 0)).sum()),
        true_over_chord_path=float(np.sum(steps.coefficient0) / np.sum(_chord_yields(steps).coefficient0)))


def _far(ctx, name):
    from scipy.spatial import cKDTree
    if name == "electron-100GeV":
        steps = ctx.electron()
    else:
        steps = ctx.regenerated(name)[0]
    kernel = ctx.kernel()
    timings = {}
    charges = {}
    for label, reach in (("point", 0.), ("default", None), ("sphere_everywhere", np.inf)):
        began = time.perf_counter()
        charges[label] = _order0(kernel, steps, reach)
        timings[label] = time.perf_counter() - began
    a = np.asarray(steps.start_m)
    b = a + np.asarray(steps.length_m)[:, None] * np.asarray(steps.direction)
    distance = cKDTree(np.r_[a, b, .5 * (a + b)]).query(kernel.detector.positions_m)[0]
    point, sphere = charges["point"], charges["sphere_everywhere"]
    rows = []
    for lo, hi in BINS_M:
        m = (distance >= lo) & (distance < hi) & (np.maximum(point, sphere) >= 1e-3)
        if m.any():
            rel = (point[m] - sphere[m]) / sphere[m]
            rows.append(dict(distance_m=[lo, hi], oms=int(m.sum()), point_pe=float(point[m].sum()),
                             sphere_pe=float(sphere[m].sum()), median_abs_rel=float(np.median(np.abs(rel))),
                             p90_abs_rel=float(np.quantile(np.abs(rel), .9)), max_abs_rel=float(np.abs(rel).max())))
    return dict(
        headline=dict(point_pe=float(point.sum()), sphere_pe=float(sphere.sum()),
                      default_equals_point=bool(np.array_equal(charges["default"], point)),
                      nearest_om_m=float(distance.min()),
                      sphere_over_point_seconds=timings["sphere_everywhere"] / timings["point"]),
        seconds=timings, by_distance=rows)


case("finite-receiver-order0", "muon-near-om-mc", kind="metrics",
     notes="order 0 per OM: point / default (20 m) / sphere everywhere / chord yields vs MC hits without scattering")(
    lambda ctx: _near_mc(ctx))
for _name in FAR_INPUTS:
    case("finite-receiver-order0", f"far-{_name}", kind="metrics", warmup=True,
         notes="order 0 of the point receiver vs the sphere everywhere, by OM distance; wall time of each")(
        lambda ctx, n=_name: _far(ctx, n))
