"""The wavelength quadrature of the scattered orders, against photon Monte Carlo.

* ``spectral-integral``: the detected beta = 1 Cherenkov spectrum after r metres
  of attenuation, Gauss-Legendre with N nodes against a fine trapezoid on the
  bundle model (what order 1 and the RTE tables integrate);
* ``order1-<event>``: order 1 of the prompt solver (no tables) with 9, 32 and
  64 nodes on the steps of the photon Monte-Carlo events;
* ``mc-<event>-nodes<N>``: EventTransport with N nodes against the photon
  Monte Carlo of photon_mc.py, per scattering order and distance;
* ``point-mc-nodes<N>``: the same for the isotropic point source of
  photon_mc_point.py (four runs).

The nodes-9 cases need a bundle built for nine nodes (LightHit <= 0.3.0a4),
the nodes-32 cases one built for 32; the >=2 pack refuses the other one.
"""
from __future__ import annotations

import json
from functools import lru_cache

import numpy as np

from .cases import case
from .inputs import inputs_dir

EVENTS = ("electron-100GeV", "pion-100GeV", "nue-cc", "muon-100GeV")
SHOWERS = EVENTS[:3]
EVENT_EDGES_M = (0., 20., 40., 60., 100., np.inf)
POINT_EDGES_M = (0., 15., 30., 50., 60., 70., 80., 100., 130., 170., 250.)
POINT_SEEDS = (1, 2, 3, 4)
RADII_M = (0., 5., 20., 50., 80., 120., 160., 200.)


def _spectral_integral(ctx):
    from lighthit.prompt import _spectral_tables
    kernel = ctx.kernel()
    wavelength = np.linspace(*kernel.medium.wavelength_range_nm, 260001)
    band = kernel.medium.sample(wavelength)
    attenuation = band["absorption_per_m"] + band["scattering_per_m"]
    field = (kernel.detector.spectral_weight(wavelength) / wavelength ** 2
             * (1 - 1 / band["phase_index"] ** 2))
    exact = [np.trapezoid(field * np.exp(-attenuation * r), wavelength) for r in RADII_M]
    out = {}
    for nodes in (9, 17, 32, 48):
        tables = _spectral_tables(ctx.kernel(wavelength_nodes=nodes))
        s0 = tables["s0w"] * (1 - 1 / tables["phase_index"] ** 2)
        out[f"nodes{nodes}"] = [float(np.sum(s0 * np.exp(-tables["mut"] * r)) / e - 1)
                                for r, e in zip(RADII_M, exact)]
    headline = {f"error_{n}_at_{int(r)}m": out[f"nodes{n}"][RADII_M.index(r)]
                for n in (9, 32) for r in (0., 50., 120.)}
    return dict(headline=headline, radii_m=list(RADII_M), relative_error=out,
                absorption_minimum_per_m=float(np.min(band["absorption_per_m"])),
                absorption_minimum_nm=float(wavelength[np.argmin(band["absorption_per_m"])]),
                nine_node_absorption_per_m=np.asarray(
                    _spectral_tables(ctx.kernel(wavelength_nodes=9))["mua"]).tolist())


def _order1(ctx, name):
    from .inputs import load_steps
    path = inputs_dir() / "mc" / f"{name}-run0-steps.npz"
    ctx._note(path)
    steps, _ = load_steps(path)
    charge = {}
    for nodes in (9, 32, 64):
        response = ctx.kernel(wavelength_nodes=nodes).transport_prompt(steps)
        charge[nodes] = np.asarray(response.charge_orders_pe, float).sum(axis=0)
    return dict(headline=dict(order1_nodes9_pe=float(charge[9][1]),
                              order1_nodes32_pe=float(charge[32][1]),
                              order1_nodes64_pe=float(charge[64][1]),
                              ratio_9_to_64=float(charge[9][1] / charge[64][1]),
                              ratio_32_to_64=float(charge[32][1] / charge[64][1])),
                order0_quadrature_pe={str(k): float(v[0]) for k, v in charge.items()})


def _distance(positions, steps, parts):
    """OM distance to the light-weighted shower centre, or to the muon's track piece."""
    mid = steps.start_m + 0.5 * steps.length_m[:, None] * steps.direction
    weight = np.asarray(steps.coefficient0, float)
    muon = [p for p in parts if p["kind"] == "muon"]
    if muon:
        mask = muon[0]["mask"]
        centre = np.average(mid[mask], axis=0, weights=weight[mask])
        rel = mid[mask] - centre
        axis = np.linalg.svd(rel * np.sqrt(weight[mask])[:, None], full_matrices=False)[2][0]
        along = rel @ axis
        offset = positions - centre
        closest = np.clip(offset @ axis, along.min(), along.max())
        return np.linalg.norm(offset - np.outer(closest, axis), axis=1)
    centre = np.average(mid, axis=0, weights=weight)
    return np.linalg.norm(positions - centre, axis=1)


def _by_order(kernel, response, hits, distance, edges):
    uids = np.asarray(kernel.detector.identifiers["uid"]).astype(np.int64)
    row = {int(u): i for i, u in enumerate(uids)}
    n = len(uids)
    mc = np.zeros((n, 3))
    variance = np.zeros((n, 3))
    for uid, phe, order in hits:
        index = np.array([row[int(u)] for u in uid], int)
        order = np.minimum(order, 2)
        for k in range(3):
            mc[:, k] += np.bincount(index[order == k], phe[order == k], n)
            variance[:, k] += np.bincount(index[order == k], phe[order == k] ** 2, n)
    lighthit = np.asarray(response.charge_components_pe, float)
    rows = []
    for low, high in zip(edges[:-1], edges[1:]):
        sel = (distance >= low) & (distance < high)
        m, s, q = mc[sel].sum(0), np.sqrt(variance[sel].sum(0)), lighthit[sel].sum(0)
        if m.sum() <= 0:
            continue
        rows.append(dict(distance_m=[float(low), float(high)], oms=int(sel.sum()),
                         lighthit_pe=q.tolist(), mc_pe=m.tolist(), mc_sigma_pe=s.tolist(),
                         ratio=[float(a / b) if b > 0 else None for a, b in zip(q, m)],
                         total_ratio=float(q.sum() / m.sum()),
                         total_sigma=float(np.sqrt((s ** 2).sum()) / m.sum())))
    total_m, total_s, total_q = mc.sum(0), np.sqrt(variance.sum(0)), lighthit.sum(0)
    headline = dict(lighthit_pe=float(total_q.sum()), mc_pe=float(total_m.sum()),
                    mc_sigma_pe=float(np.sqrt((total_s ** 2).sum())),
                    total_ratio=float(total_q.sum() / total_m.sum()))
    for k, label in enumerate(("order0", "order1", "order2plus")):
        headline[f"{label}_ratio"] = float(total_q[k] / total_m[k])
        headline[f"{label}_sigma"] = float(total_s[k] / total_m[k])
    return dict(headline=headline, lighthit_by_order_pe=total_q.tolist(),
                mc_by_order_pe=total_m.tolist(), mc_sigma_by_order_pe=total_s.tolist(),
                by_distance=rows)


def _event(ctx, name, nodes):
    from lighthit.event import EventTransport
    from .inputs import components, load_steps
    base = inputs_dir() / "mc"
    stem = f"{name}-run0"
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
    # The Monte Carlo emits only inside NTSim's world cylinder with a 100 m
    # margin (CherenkovGenerator volume); the muon starts outside it.
    inside = _mc_volume(steps)
    outside_light = float(steps.coefficient0[~inside].sum() / steps.coefficient0.sum())
    if not inside.all():
        keep = np.flatnonzero(inside)
        steps = steps.subset(inside)
        for part in parts:
            part["mask"] = part["mask"][keep]
            if part["ancestry"] is not None:
                part["ancestry"] = dict(part["ancestry"],
                                        element_family=part["ancestry"]["element_family"][keep])
    kernel = ctx.kernel(wavelength_nodes=nodes)
    route = ctx.cached(("event-route", nodes), lambda: EventTransport(kernel))
    response = route(components(steps, parts))
    with np.load(base / f"{stem}-hits.npz", allow_pickle=False) as data:
        hits = [(data["uid"], data["phe"], data["order"])]
    distance = _distance(kernel.detector.positions_m, steps, parts)
    out = _by_order(kernel, response, hits, distance, EVENT_EDGES_M)
    out["light_outside_mc_volume"] = outside_light
    return out


@lru_cache(maxsize=1)
def _mc_emission_volume():
    """NTSim's world cylinder with the 100 m margin of photon_mc.py (one telescope per process)."""
    from ntsim import bgvd
    from regenerate_inputs import CLUSTERS
    return bgvd.Telescope(n_clusters=CLUSTERS).world.with_margin(100)


def _mc_volume(steps):
    """Steps whose midpoint lies in the emission volume of the photon Monte Carlo."""
    midpoints = steps.start_m + 0.5 * steps.length_m[:, None] * steps.direction
    return np.asarray(_mc_emission_volume().contains(midpoints), bool)


def _point(ctx, nodes):
    from lighthit.event import EventTransport, SourceComponent
    from .inputs import load_steps
    base = inputs_dir() / "mc"
    kernel = ctx.kernel(wavelength_nodes=nodes)
    route = ctx.cached(("event-route", nodes), lambda: EventTransport(kernel))
    total, hits, point, photons = None, [], None, 0
    for seed in POINT_SEEDS:
        stem = f"point-seed{seed}"
        ctx._note(base / f"{stem}-steps.npz", base / f"{stem}-hits.npz")
        steps, meta = load_steps(base / f"{stem}-steps.npz")
        response = route([SourceComponent("shower", steps, label=stem)])
        total = response if total is None else _Sum(total, response)
        with np.load(base / f"{stem}-hits.npz", allow_pickle=False) as data:
            hits.append((data["uid"], data["phe"], data["order"]))
        point = np.asarray(meta["point_m"], float)
        photons += json.loads(str(np.load(base / f"{stem}-hits.npz")["metadata"]))["photons"]
    distance = np.linalg.norm(kernel.detector.positions_m - point, axis=1)
    out = _by_order(kernel, total, hits, distance, POINT_EDGES_M)
    out["photons"] = int(photons)
    return out


class _Sum:
    def __init__(self, a, b):
        self.charge_components_pe = (np.asarray(a.charge_components_pe, float)
                                     + np.asarray(b.charge_components_pe, float))


case("wavelength-quadrature", "spectral-integral", kind="metrics",
     notes="Gauss-Legendre wavelength rules against a fine trapezoid on the bundle model")(
    _spectral_integral)
for _name in SHOWERS:
    case("wavelength-quadrature", f"order1-{_name}", kind="metrics", warmup=False,
         notes="prompt order 1 with 9, 32 and 64 wavelength nodes")(
        lambda ctx, n=_name: _order1(ctx, n))
for _nodes in (9, 32):
    for _name in EVENTS:
        case("wavelength-quadrature", f"mc-{_name}-nodes{_nodes}", kind="metrics",
             notes=f"EventTransport with {_nodes} wavelength nodes vs photon MC, by order and distance")(
            lambda ctx, n=_name, k=_nodes: _event(ctx, n, k))
    case("wavelength-quadrature", f"point-mc-nodes{_nodes}", kind="metrics",
         notes=f"isotropic point source, EventTransport with {_nodes} nodes vs photon MC (4 runs)")(
        lambda ctx, k=_nodes: _point(ctx, k))
