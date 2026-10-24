#!/usr/bin/env python
"""Photon Monte-Carlo reference for one regenerated event (NTSim).

    python experiments/photon_mc.py NAME [--run N] [--seed S] [--suppression K] [--orders]

Runs, in one process, the Geant4 event NAME of experiments/regenerate_inputs.py
(same generator and seed) followed by NTSim's photon chain
CherenkovGenerator (photon_suppression=1, i.e. every photon; ``--suppression K``
draws a Poisson 1/K of the photons with weight K, unbiased, for very bright
events) ->
MCPhotonPropagator (Henyey-Greenstein, n_scatterings=30) -> SmartRayTracer on
the five clusters, as in section 2 of NTSim's examples/LightHit notebook.
Writes to experiments/inputs/mc/:

* ``NAME-run<N>-steps.npz``/``-parts.npz``: the charged steps of this very
  event (the same arrays as the regenerated input when Geant4 reproduces it);
* ``NAME-run<N>-hits.npz``: the photon hits (OM uid, time, weight ``phe``,
  emitting particle ``track_uid``) and
  the run metadata (wall/CPU time, photons, run number); with ``--orders`` also
  ``order``, the number of scatterings before each hit (the index of the
  photon path segment that reaches the OM, found from the hit time).

The photon stages draw from NTSim's random state, seeded with ``--seed``
(default ``SEED_BASE + run``, recorded in the metadata); Geant4 keeps the event's
own seed, so runs differ only in photon statistics. Geant4 starts once per
process: one event per call.
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

SEED_BASE = 20261021


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("name")
    parser.add_argument("--run", type=int, default=0)
    parser.add_argument("--scatterings", type=int, default=30)
    parser.add_argument("--orders", action="store_true",
                        help="record the scattering order of every hit")
    parser.add_argument("--seed", type=int, default=None,
                        help="seed of the photon stages (default SEED_BASE + run)")
    parser.add_argument("--suppression", type=int, default=1,
                        help="photon_suppression of the CherenkovGenerator (weight per photon)")
    args = parser.parse_args()
    seed = SEED_BASE + args.run if args.seed is None else args.seed
    import ntsim
    import ntsim.random
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
                                         photon_suppression=args.suppression, chunk_size=100_000),
        ntsim.modules.MCPhotonPropagator(medium_prop=water,
                                         medium_scat=ntsim.scattering.HenyeyGreenstein(),
                                         n_scatterings=args.scatterings),
        (_order_tracer(orders) if args.orders else ntsim.modules.SmartRayTracer)(telescope=telescope),
    ]
    out = inputs_dir() / "mc"
    stem = f"{args.name}-run{args.run}"
    ntsim.random.set_seed(seed)
    usage0 = resource.getrusage(resource.RUSAGE_SELF)
    began = time.perf_counter()
    event = generate(args.name, out, extra=stages, stem=stem, telescope=telescope)
    wall = time.perf_counter() - began
    usage1 = resource.getrusage(resource.RUSAGE_SELF)
    hits = event.hits["Hits"]
    meta = dict(name=args.name, run=args.run, n_scatterings=args.scatterings,
                photon_suppression=args.suppression, seed=seed, wall_s=wall,
                cpu_s=(usage1.ru_utime - usage0.ru_utime) + (usage1.ru_stime - usage0.ru_stime),
                hits=int(len(hits)), total_phe=float(np.sum(hits.phe)),
                note="wall and CPU include Geant4 and the photon chain")
    _save_hits(out / f"{stem}-hits.npz", hits, meta, orders if args.orders else None)
    print(json.dumps(meta))


def _save_hits(path, hits, metadata, orders=None):
    """Keep emitting-particle identities aligned with every recorded hit."""
    arrays = dict(uid=np.asarray(hits.uid, np.int64),
                  track_uid=np.asarray(hits.track_uid, np.int64),
                  t_ns=np.asarray(hits.t_ns, float), phe=np.asarray(hits.phe, float))
    if orders is not None:
        arrays["order"] = np.concatenate(orders) if orders else np.zeros(0, np.int64)
    if any(value.shape != (len(hits),) for value in arrays.values()):
        raise ValueError("hit columns and scattering orders must have one entry per hit")
    np.savez_compressed(path, **arrays, metadata=json.dumps(metadata))


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
