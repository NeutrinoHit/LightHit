#!/usr/bin/env python
"""Photon Monte-Carlo reference for the calibration laser flash (NTSim).

    python experiments/photon_mc_laser.py [--seed S] [--photons N] [--chunk M]

The flash of the default laser route (``default-laser-route/laser-532nm``):
an isotropic monochromatic 532 nm point flash at (-62, -385, 361) m, between
clusters 1, 2 and 4, at time 0. NTSim's photon chain as in photon_mc.py, without
a Cherenkov generator: N photons with isotropic random directions and unit
weight -> MCPhotonPropagator (Henyey-Greenstein, 30 scatterings) ->
SmartRayTracer on the five clusters, with the scattering order of every hit.
A LightHit charge for F flash photons is compared with ``phe * F / N``.

The photon chain runs in chunks; the wall and CPU time of every chunk are
saved, so the time to reach a stated statistical precision can be read from
the cumulative record. Writes ``experiments/inputs/mc/laser-532nm-seed<S>-hits.npz``
(uid, t_ns, phe, order, chunk, and the run metadata).
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def isotropic(n, rng):
    direction = rng.normal(size=(n, 3))
    return direction / np.linalg.norm(direction, axis=1)[:, None]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--photons", type=float, default=1e7)
    parser.add_argument("--chunk", type=int, default=500_000)
    parser.add_argument("--scatterings", type=int, default=30)
    args = parser.parse_args()
    import ntsim
    from numba import njit
    from ntsim import bgvd
    from ntsim.IO import gPhotons
    from ntsim.random import set_seed
    from regenerate_inputs import CLUSTERS
    from photon_mc import _order_tracer
    from harness.inputs import LASER, inputs_dir

    @njit
    def seed_numba(value):
        np.random.seed(value)

    # Three random streams: numpy (directions), numba (free paths) and
    # ntsim.random (scattering angles).
    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)
    seed_numba(args.seed)
    set_seed(args.seed)
    telescope = bgvd.Telescope(n_clusters=CLUSTERS)
    water = bgvd.Water()
    propagator = ntsim.modules.MCPhotonPropagator(
        medium_prop=water, medium_scat=ntsim.scattering.HenyeyGreenstein(),
        n_scatterings=args.scatterings)
    orders = []
    tracer = _order_tracer(orders)(telescope=telescope)
    total = int(args.photons)
    point = np.asarray(LASER["position_m"], float)
    uid, t_ns, phe, chunk_id, chunks = [], [], [], [], []
    usage0 = resource.getrusage(resource.RUSAGE_SELF)
    began = time.perf_counter()
    done = 0
    while done < total:
        n = min(args.chunk, total - done)
        wall0, cpu0 = time.perf_counter(), time.process_time()
        photons = gPhotons(size=n, pos_m=np.broadcast_to(point, (n, 3)), t_ns=0.0,
                           direction=isotropic(n, rng), wl_nm=float(LASER["wavelength_nm"]),
                           weight=1.0)
        before = len(orders)
        hits = tracer.propagate(propagator.propagate(photons))
        if hits is not None:
            uid.append(np.asarray(hits.uid, np.int64))
            t_ns.append(np.asarray(hits.t_ns, float))
            phe.append(np.asarray(hits.phe, float))
            chunk_id.append(np.full(len(hits.uid), len(chunks), np.int32))
        if len(orders) == before:
            orders.append(np.zeros(0, int))
        done += n
        chunks.append(dict(photons=n, wall_s=time.perf_counter() - wall0,
                           cpu_s=time.process_time() - cpu0,
                           hits=int(len(hits.uid)) if hits is not None else 0))
        print(json.dumps(dict(chunk=len(chunks) - 1, done=done, **chunks[-1])), flush=True)
    wall = time.perf_counter() - began
    usage1 = resource.getrusage(resource.RUSAGE_SELF)
    out = inputs_dir() / "mc"
    out.mkdir(parents=True, exist_ok=True)
    meta = dict(source="isotropic monochromatic flash", seed=args.seed, photons=total,
                position_m=list(LASER["position_m"]), wavelength_nm=LASER["wavelength_nm"],
                flash_photons=LASER["photons"], n_scatterings=args.scatterings, chunks=chunks,
                wall_s=wall, cpu_s=(usage1.ru_utime - usage0.ru_utime)
                + (usage1.ru_stime - usage0.ru_stime),
                peak_rss_mib=usage1.ru_maxrss / (2 ** 20 if sys.platform == "darwin" else 2 ** 10),
                threads={k: os.environ.get(k) for k in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS")},
                load_average=list(os.getloadavg()))
    empty = (np.zeros(0, np.int64),)
    np.savez_compressed(out / f"laser-532nm-seed{args.seed}-hits.npz",
                        uid=np.concatenate(uid or empty), t_ns=np.concatenate(t_ns or empty),
                        phe=np.concatenate(phe or empty), order=np.concatenate(orders),
                        chunk=np.concatenate(chunk_id or empty), metadata=json.dumps(meta))
    print(json.dumps({k: v for k, v in meta.items() if k != "chunks"}))


if __name__ == "__main__":
    main()
