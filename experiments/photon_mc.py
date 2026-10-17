#!/usr/bin/env python
"""Photon Monte-Carlo reference for one regenerated event (NTSim).

    python experiments/photon_mc.py NAME [--run N]

Runs, in one process, the Geant4 event NAME of experiments/regenerate_inputs.py
(same generator and seed) followed by NTSim's photon chain
CherenkovGenerator (photon_suppression=1, i.e. every photon) ->
MCPhotonPropagator (Henyey-Greenstein, n_scatterings=30) -> SmartRayTracer on
the five clusters, as in section 2 of NTSim's examples/LightHit notebook.
Writes to experiments/inputs/mc/:

* ``NAME-run<N>-steps.npz``/``-parts.npz``: the charged steps of this very
  event (the same arrays as the regenerated input when Geant4 reproduces it);
* ``NAME-run<N>-hits.npz``: the photon hits (OM uid, time, weight ``phe``) and
  the run metadata (wall/CPU time, photons, run number); with ``--orders`` also
  ``order``, the number of scatterings before each hit (the index of the
  photon path segment that reaches the OM, found from the hit time).

The Monte-Carlo stages draw from their own random streams, so runs differ only
in photon statistics. Geant4 starts once per process: one event per call.
"""
from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("name")
    parser.add_argument("--run", type=int, default=0)
    parser.add_argument("--scatterings", type=int, default=30)
    parser.add_argument("--orders", action="store_true",
                        help="record the scattering order of every hit")
    args = parser.parse_args()
    import ntsim
    from ntsim import bgvd
    from regenerate_inputs import CLUSTERS, EVENTS, generate
    from harness.inputs import inputs_dir
    if args.name not in EVENTS:
        parser.error(f"unknown event {args.name}")
    orders = []
    telescope = bgvd.Telescope(n_clusters=CLUSTERS)
    water = bgvd.Water()
    stages = [
        ntsim.modules.CherenkovGenerator(volume=telescope.world.with_margin(100),
                                         phase_wavelength_nm=water.wavelength_nm,
                                         phase_refraction_index=water.phase_refraction_index,
                                         photon_suppression=1, chunk_size=100_000),
        ntsim.modules.MCPhotonPropagator(medium_prop=water,
                                         medium_scat=ntsim.scattering.HenyeyGreenstein(),
                                         n_scatterings=args.scatterings),
        (_order_tracer(orders) if args.orders else ntsim.modules.SmartRayTracer)(telescope=telescope),
    ]
    out = inputs_dir() / "mc"
    stem = f"{args.name}-run{args.run}"
    usage0 = resource.getrusage(resource.RUSAGE_SELF)
    began = time.perf_counter()
    event = generate(args.name, out, extra=stages, stem=stem, telescope=telescope)
    wall = time.perf_counter() - began
    usage1 = resource.getrusage(resource.RUSAGE_SELF)
    hits = event.hits["Hits"]
    meta = dict(name=args.name, run=args.run, n_scatterings=args.scatterings,
                photon_suppression=1, wall_s=wall,
                cpu_s=(usage1.ru_utime - usage0.ru_utime) + (usage1.ru_stime - usage0.ru_stime),
                hits=int(len(hits)), total_phe=float(np.sum(hits.phe)),
                note="wall and CPU include Geant4 and the photon chain")
    extra = {}
    if args.orders:
        extra["order"] = np.concatenate(orders) if orders else np.zeros(0, int)
        assert len(extra["order"]) == len(hits)
    np.savez_compressed(out / f"{stem}-hits.npz", uid=np.asarray(hits.uid, np.int64),
                        t_ns=np.asarray(hits.t_ns, float), phe=np.asarray(hits.phe, float),
                        metadata=json.dumps(meta), **extra)
    print(json.dumps(meta))


def _order_tracer(orders):
    """SmartRayTracer that appends the scattering order of its hits to ``orders``."""
    from ntsim.IO import gHits
    from ntsim.Propagators.RayTracers.SmartRayTracer import SmartRayTracer
    from ntsim.Propagators.RayTracers.rt_utils import ray_tracer

    class OrderRayTracer(SmartRayTracer):
        def propagate(self, photons):
            hits_list = ray_tracer(photons.pos_m, photons.t_ns, photons.wl_nm, photons.weight,
                                   photons.ta_ns, self.bboxes_depth, self.detectors_depth,
                                   self.effects, self.effects_options)
            if len(hits_list):
                uid, t_ns, x, y, z, photon_id, *metadata, phe = np.array(hits_list).T
                photon_id = np.asarray(photon_id, int)
                vertex_t = photons.t_ns[photon_id]
                orders.append(np.array([np.searchsorted(v, t, side="right") - 1
                                        for v, t in zip(vertex_t, t_ns)], int))
                return gHits(size=len(uid), uid=uid, t_ns=t_ns, phe=phe,
                             track_uid=photons.track_uid[photon_id])
    return OrderRayTracer


if __name__ == "__main__":
    main()
