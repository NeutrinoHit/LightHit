"""Default routes against the photon Monte Carlo over energies, particles and geometries.

``photon-mc-matrix/<event>`` runs EventTransport with defaults once on the steps
of the first Monte-Carlo run of ``<event>`` and compares it with every run of
``experiments/photon_mc.py <event> --orders`` found in ``inputs/mc`` (all runs
share the Geant4 event; the step files are hashed and whether they are all
identical is recorded as ``same_steps_in_all_runs``, not enforced).

OMs are grouped by the distance of the OM centre to the nearest point of the
emitting steps (start, middle, end). Per distance bin and per scattering order
(0, 1, >=2 and their total; the Monte-Carlo order of a hit is the number of
scatterings before it) the case reports both charges, their ratio, the error of
the ratio and the share of the event's charge. The Monte-Carlo error of a bin is
the spread of the bin charge over the runs divided by sqrt(runs) (with one run,
the square root of the summed squared hit weights). A bin with more than 0.1 %
of the Monte-Carlo charge passes when |ratio - 1| <= max(2 sigma, 3 %).
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

from .cases import case
from .inputs import inputs_dir, load_steps

BINS_M = (0., 20., 50., 100., 150., 200., 300., np.inf)
ORDERS = ("0", "1", ">=2", "total")
SHARE_MIN = 1e-3
TOLERANCE = 0.03

#: event -> (runs, photon_suppression) of photon_mc.py
MATRIX = {
    "electron-10GeV": (32, 1), "electron-100GeV": (16, 1), "electron-1TeV": (8, 1),
    "electron-10TeV": (4, 10),
    "pion-10GeV": (32, 1), "pion-100GeV": (16, 1), "pion-1TeV": (8, 1),
    "muon-100GeV": (16, 1), "muon-1TeV-check": (8, 1), "muon-10TeV": (4, 10),
    "nue-cc": (16, 1), "numu-cc": (8, 1),
    "electron-100GeV-outside": (16, 1), "electron-100GeV-down": (16, 1),
}


def _runs(name):
    base = inputs_dir() / "mc"
    paths = sorted(base.glob(f"{name}-run*-hits.npz"),
                   key=lambda p: int(p.stem.split("-run")[1].split("-")[0]))
    if not paths:
        raise FileNotFoundError(f"no Monte-Carlo runs of {name} in {base}: "
                                f"run experiments/photon_mc.py {name} --orders --run N")
    return paths


def _digest(path):
    with np.load(path, allow_pickle=False) as data:
        h = hashlib.sha256()
        for key in ("start_m", "direction", "length_m", "coefficient0"):
            h.update(np.ascontiguousarray(data[key]).tobytes())
    return h.hexdigest()[:16]


def _components(steps_path):
    from .inputs import components
    steps, meta = load_steps(steps_path)
    parts = []
    with np.load(steps_path.with_name(steps_path.name.replace("-steps", "-parts")), allow_pickle=False) as data:
        for index, record in enumerate(meta["parts"]):
            mask = np.zeros(len(steps.length_m), bool)
            mask[data[f"mask_{index}"]] = True
            ancestry = ({k[len(f"ancestry_{index}_"):]: data[k] for k in data.files
                         if k.startswith(f"ancestry_{index}_")} if record["kind"] == "muon" else None)
            parts.append(dict(record, mask=mask, ancestry=ancestry))
    return steps, components(steps, parts)


def _nearest(steps, positions):
    from scipy.spatial import cKDTree
    a = np.asarray(steps.start_m)
    b = a + np.asarray(steps.length_m)[:, None] * np.asarray(steps.direction)
    return cKDTree(np.r_[a, b, .5 * (a + b)]).query(positions)[0]


def _matrix(ctx, name):
    from lighthit.event import EventTransport
    hits_paths = _runs(name)
    steps_paths = [p.with_name(p.name.replace("-hits", "-steps")) for p in hits_paths]
    ctx._note(*steps_paths, *hits_paths)
    digests = [_digest(p) for p in steps_paths]
    steps, sources = _components(steps_paths[0])
    kernel = ctx.kernel()
    response = ctx.cached("event-route", lambda: EventTransport(kernel))(sources)
    lh = np.asarray(response.charge_components_pe, float)          # (OM, 3)
    lh = np.c_[lh, lh.sum(1)]
    uids = np.asarray(kernel.detector.identifiers["uid"]).astype(np.int64)
    row = {int(u): i for i, u in enumerate(uids)}
    n = len(uids)
    runs, w2, meta = [], [], []
    for path in hits_paths:
        with np.load(path, allow_pickle=False) as data:
            if "order" not in data.files:
                raise ValueError(f"{path.name} has no scattering orders: rerun photon_mc.py with --orders")
            index = np.array([row[int(u)] for u in data["uid"]], int)
            order = np.minimum(np.asarray(data["order"], int), 2)
            phe = np.asarray(data["phe"], float)
            per = np.zeros((n, 4)); var = np.zeros((n, 4))
            for k in range(3):
                m = order == k
                per[:, k] = np.bincount(index[m], phe[m], n)
                var[:, k] = np.bincount(index[m], phe[m] ** 2, n)
            per[:, 3] = per[:, :3].sum(1); var[:, 3] = var[:, :3].sum(1)
            runs.append(per); w2.append(var)
            meta.append(json.loads(str(data["metadata"])))
    runs = np.array(runs); w2 = np.array(w2)                       # (run, OM, 4)
    mc = runs.mean(0)
    distance = _nearest(steps, kernel.detector.positions_m)
    total_mc, total_lh = mc[:, 3].sum(), lh[:, 3].sum()
    rows, failed = [], []
    for lo, hi in zip(BINS_M[:-1], BINS_M[1:]):
        m = (distance >= lo) & (distance < hi)
        entry = dict(distance_m=[lo, None if np.isinf(hi) else hi], oms=int(m.sum()),
                     oms_with_mc_hits=int((mc[m, 3] > 0).sum()))
        for k, label in enumerate(ORDERS):
            q_mc, q_lh = mc[m, k].sum(), lh[m, k].sum()
            per_run = runs[:, m, k].sum(1)
            if len(runs) > 1:
                sigma_mc = per_run.std(ddof=1) / np.sqrt(len(runs))
            else:
                sigma_mc = np.sqrt(w2[0, m, k].sum())
            ratio = q_lh / q_mc if q_mc > 0 else None
            sigma = sigma_mc / q_mc * ratio if q_mc > 0 and ratio else None
            entry[label] = dict(lighthit_pe=float(q_lh), mc_pe=float(q_mc), mc_sigma_pe=float(sigma_mc),
                                ratio=ratio, ratio_sigma=sigma,
                                mc_share=float(q_mc / total_mc) if total_mc else None,
                                lighthit_share=float(q_lh / total_lh) if total_lh else None)
        tot = entry["total"]
        tested = tot["mc_share"] is not None and tot["mc_share"] > SHARE_MIN and tot["ratio"] is not None
        entry["tested"] = bool(tested)
        if tested:
            entry["passed"] = bool(abs(tot["ratio"] - 1) <= max(2 * tot["ratio_sigma"], TOLERANCE))
            if not entry["passed"]:
                failed.append(f"{lo:g}-{hi:g} m")
        rows.append(entry)
    totals = {}
    for k, label in enumerate(ORDERS):
        per_run = runs[:, :, k].sum(1)
        q_mc = mc[:, k].sum()
        s = per_run.std(ddof=1) / np.sqrt(len(runs)) if len(runs) > 1 else np.sqrt(w2[0, :, k].sum())
        totals[label] = dict(lighthit_pe=float(lh[:, k].sum()), mc_pe=float(q_mc), mc_sigma_pe=float(s),
                             ratio=float(lh[:, k].sum() / q_mc) if q_mc > 0 else None)
    return dict(
        headline=dict(lighthit_pe=float(total_lh), mc_pe=float(total_mc),
                      ratio=totals["total"]["ratio"],
                      ratio_sigma=(totals["total"]["mc_sigma_pe"] / total_mc * totals["total"]["ratio"]
                                   if total_mc and totals["total"]["ratio"] else None),
                      order0_ratio=totals["0"]["ratio"], order1_ratio=totals["1"]["ratio"],
                      order2_ratio=totals[">=2"]["ratio"], runs=len(runs),
                      bins_failed=", ".join(failed) or "none"),
        totals=totals, by_distance=rows, route=response.metadata.get("route"),
        same_steps_in_all_runs=len(set(digests)) == 1,
        mc=dict(runs=len(runs), photon_suppression=sorted({m.get("photon_suppression", 1) for m in meta}),
                seeds=[m.get("seed") for m in meta], n_scatterings=sorted({m.get("n_scatterings") for m in meta}),
                wall_s=[m.get("wall_s") for m in meta]))


for _name in MATRIX:
    case("photon-mc-matrix", _name, kind="metrics",
         notes="EventTransport defaults vs all photon_mc.py --orders runs, by OM distance and scattering order")(
        lambda ctx, n=_name: _matrix(ctx, n))
