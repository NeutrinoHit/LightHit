"""NTSim's former check scripts of the LightHit hybrid, recomputed with LightHit.

* ``ntsim-spectral-basis`` (ntsim ``run/check_spectral_basis.sh``): the script
  ran ``run/check_lighthit.sh hybrid`` -> ``run_lighthit.py --mode hybrid`` on
  cluster 1 for a 100 GeV electron or muon from (5, -269.5, 330) m along
  (0.35, -0.20, 0.915), seed 42, with ``--hybrid-ge2-basis default`` (route
  ``reference``) or ``shower`` (``basis``; ``track`` adds the primary as the
  track channel; ``fast-track`` adds spectral fusion and 33 frequencies).
  ``LightHitTransport._transport`` then calls
  ``kernel.transport_hybrid(source, frequency_nodes=..., ge2_basis=...)`` with
  the default prompt; the cases make the same call on the cluster-1 OMs of the
  bundle model and the regenerated events (``electron-100GeV``,
  ``muon-100GeV-cluster1``).
* ``ntsim-compare-expected-pe`` (ntsim ``run/compare_expected_pe.py``): the
  script compared the charge of orders 0 and 0+1 of the photon Monte Carlo
  (the stored Geant4 tracks replayed with one and two photon-flight segments)
  with LightHit's prompt orders 0 and 1 on the same event. The cases take the
  Monte-Carlo hits with their scattering order (``photon_mc.py --orders``,
  ``inputs/mc``) and the prompt route on the Monte Carlo's own steps.
"""
from __future__ import annotations

import json

import numpy as np

from .cases import case
from .inputs import inputs_dir, load_steps

ROUTES = {"reference": dict(basis=False, nodes=65),
          "basis": dict(basis=True, nodes=65),
          "track": dict(basis=True, track=True, nodes=65),
          "fast-track": dict(basis=True, track=True, fusion=True, nodes=33)}
#: route -> the route it is compared with; the legacy per-band route refuses the muon
#: (253 OMs outside the validated directional radial range), so its routes are
#: compared with its shower basis
INPUTS = {"electron-100GeV": {"reference": None, "basis": "reference"},
          "muon-100GeV-cluster1": {"basis": None, "track": "basis", "fast-track": "basis"}}


def _cluster1_kernel(ctx):
    """The bundle kernel on the OMs of cluster 1, with the prepared >=2 pack."""
    from . import references as R
    from .inputs import bundle_path
    kernel = ctx.kernel()
    cluster = np.asarray(kernel.detector.identifiers["uid"]) // 10000   # NTSim's cluster number
    kernel = R.subset_kernel(kernel, np.flatnonzero(cluster == 1))
    kernel.ge2_kernel_store(str(bundle_path("tables", "ge2-pack")))
    return kernel


def _spectral_basis(name, basis=False, track=False, fusion=False, nodes=65):
    def run(ctx):
        import lighthit as lh
        steps, parts, _ = ctx.regenerated(name)
        kernel = ctx.cached("cluster1-kernel", lambda: _cluster1_kernel(ctx))
        options = {}
        if basis:
            uid = None
            if track:
                muons = [p for p in parts if p["kind"] == "muon"]
                uid = int(muons[0]["primary_uid"])
            options["ge2_basis"] = lh.ShowerBasis(track_uid=uid, spectral_fusion=fusion)
        return kernel.transport_hybrid(steps, frequency_nodes=nodes, **options)
    return run


for _name, _routes in INPUTS.items():
    for _route, _against in _routes.items():
        case("ntsim-spectral-basis", f"{_name}-{_route}",
             reference=_against and f"ntsim-spectral-basis/{_name}-{_against}",
             notes=f"check_spectral_basis.sh {'11' if _name.startswith('electron') else '13'} {_route}: "
                   f"transport_hybrid on cluster 1, {ROUTES[_route]}")(
            _spectral_basis(_name, **ROUTES[_route]))


def _compare_expected(name, run_index=0):
    def run(ctx):
        base = inputs_dir() / "mc"
        stem = f"{name}-run{run_index}"
        ctx._note(base / f"{stem}-steps.npz", base / f"{stem}-hits.npz")
        steps, _ = load_steps(base / f"{stem}-steps.npz")
        with np.load(base / f"{stem}-hits.npz", allow_pickle=False) as data:
            phe, order = data["phe"], data["order"]
            mc_meta = json.loads(str(data["metadata"]))
        mc = {}
        for label, mask in (("order0", order == 0), ("order0+1", order <= 1), ("all", order >= 0)):
            mc[label] = dict(pe=float(phe[mask].sum()), sigma_pe=float(np.sqrt(np.square(phe[mask]).sum())),
                             hits=int(mask.sum()))
        response = ctx.kernel().transport_prompt(steps)
        q = np.asarray(response.charge_orders_pe, float).sum(axis=0)
        lighthit = {"order0": float(q[0]), "order0+1": float(q[0] + q[1])}
        ratios = {k: lighthit[k] / mc[k]["pe"] for k in lighthit}
        sigmas = {k: ratios[k] * mc[k]["sigma_pe"] / mc[k]["pe"] for k in lighthit}
        return dict(headline=dict(order0_ratio=ratios["order0"], order0_sigma=sigmas["order0"],
                                  order01_ratio=ratios["order0+1"], order01_sigma=sigmas["order0+1"],
                                  lighthit_order0_pe=lighthit["order0"],
                                  lighthit_order01_pe=lighthit["order0+1"],
                                  mc_order0_pe=mc["order0"]["pe"], mc_order01_pe=mc["order0+1"]["pe"]),
                    mc=mc, lighthit_prompt_pe=lighthit, mc_run=mc_meta)
    return run


for _name in ("electron-100GeV", "muon-100GeV"):
    case("ntsim-compare-expected-pe", f"{_name}-run0", kind="metrics", warmup=False,
         notes="compare_expected_pe.py: photon-MC orders 0 and 0+1 against the default prompt "
               "route on the Monte Carlo's own steps")(_compare_expected(_name))
