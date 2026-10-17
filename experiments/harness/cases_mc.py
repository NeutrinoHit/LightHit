"""Default routes against the photon Monte Carlo of experiments/photon_mc.py.

The LightHit side runs EventTransport with defaults on the charged steps saved
by the Monte-Carlo process itself (the same Geant4 event). Metrics, all over
OMs (the Monte-Carlo error of an OM is sqrt(sum of squared hit weights)):

* total charge of both and the Monte-Carlo standard error of its total;
* pulls (LightHit - MC)/sigma_MC over OMs where either gives >= 0.1 pe (OMs
  without MC hits get sigma from the LightHit expectation, Poisson-like);
* bright-OM time profiles: L1 of normalised 5 ns histograms on LightHit's
  clock for OMs with >= 1 pe in both.
"""
from __future__ import annotations

import json

import numpy as np

from .cases import case
from .inputs import inputs_dir

RUNS = {"electron-100GeV": (0,), "pion-100GeV": (0,), "nue-cc": (0,), "muon-100GeV": (0,)}


def _compare(ctx, name, run):
    from lighthit.event import EventTransport
    from .inputs import components, load_steps
    base = inputs_dir() / "mc"
    stem = f"{name}-run{run}"
    ctx._note(base / f"{stem}-steps.npz", base / f"{stem}-hits.npz")
    steps, meta = load_steps(base / f"{stem}-steps.npz")
    parts = []
    with np.load(base / f"{stem}-parts.npz", allow_pickle=False) as data:
        for index, record in enumerate(meta["parts"]):
            mask = np.zeros(len(steps.length_m), bool)
            mask[data[f"mask_{index}"]] = True
            ancestry = ({k[len(f"ancestry_{index}_"):]: data[k] for k in data.files
                         if k.startswith(f"ancestry_{index}_")} if record["kind"] == "muon" else None)
            parts.append(dict(record, mask=mask, ancestry=ancestry))
    kernel = ctx.kernel()
    route = ctx.cached("event-route", lambda: EventTransport(kernel))
    response = route(components(steps, parts))
    with np.load(base / f"{stem}-hits.npz", allow_pickle=False) as data:
        uid, t, phe = data["uid"], data["t_ns"], data["phe"]
        mc_meta = json.loads(str(data["metadata"]))
    uids = np.asarray(kernel.detector.identifiers["uid"]).astype(np.int64)
    row = {int(u): i for i, u in enumerate(uids)}
    index = np.array([row[int(u)] for u in uid], int)
    n = len(uids)
    mc = np.bincount(index, phe, n)
    mc_var = np.bincount(index, phe ** 2, n)
    lh_q = np.asarray(response.charge_pe, float)
    sel = (lh_q >= 0.1) | (mc >= 0.1)
    sigma = np.sqrt(np.where(mc_var > 0, mc_var, np.maximum(lh_q, 1e-12)))
    pulls = (lh_q[sel] - mc[sel]) / sigma[sel]
    edges = np.asarray(response.relative_time_edges_ns)
    l1 = []
    for om in np.flatnonzero((lh_q >= 1.0) & (mc >= 1.0) & np.asarray(response.active)):
        mine = index == om
        hist, _ = np.histogram(t[mine] - response.time_origin_ns[om], bins=edges, weights=phe[mine])
        model = np.asarray(response.bins_pe[om], float)
        if hist.sum() > 0 and model.sum() > 0:
            l1.append(float(np.abs(hist / hist.sum() - model / model.sum()).sum()))
    return dict(
        headline=dict(lighthit_pe=float(lh_q.sum()), mc_pe=float(mc.sum()),
                      mc_total_sigma_pe=float(np.sqrt(mc_var.sum())),
                      total_ratio=float(lh_q.sum() / mc.sum()) if mc.sum() else None,
                      pull_rms=float(np.sqrt(np.mean(pulls ** 2))) if len(pulls) else None,
                      oms_compared=int(sel.sum()),
                      bright_profile_l1_median=float(np.median(l1)) if l1 else None),
        pulls_mean=float(np.mean(pulls)) if len(pulls) else None,
        fraction_abs_pull_gt_3=float(np.mean(np.abs(pulls) > 3)) if len(pulls) else None,
        bright_profile_l1=l1, active=int(np.asarray(response.active).sum()),
        route=response.metadata.get("route"), mc=mc_meta)


for _name, _runs in RUNS.items():
    for _run in _runs:
        case("photon-mc-comparison", f"{_name}-run{_run}", kind="metrics",
             notes="EventTransport defaults vs NTSim photon MC (30 scatterings) on the same event")(
            lambda ctx, n=_name, r=_run: _compare(ctx, n, r))
