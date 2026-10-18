#!/usr/bin/env python
"""Photon Monte-Carlo reference for an isotropic Cherenkov point source (NTSim).

    python experiments/photon_mc_point.py [--seed S] [--segments N] [--point X Y Z]

The source is N straight 5 cm steps of a 100 GeV muon (beta = 1), centred on
one point with isotropic random directions (seed S): the light of every
wavelength leaves the point in all directions, so every OM of the cluster is
reached directly, once and many times scattered, at 8-250 m. No Geant4 runs.
The default point lies 8 m from the central string of cluster 1, between two
of its OMs.

NTSim's photon chain as in photon_mc.py: CherenkovTracksGenerator
(photon_suppression=1, photon yield per true step length) ->
MCPhotonPropagator (Henyey-Greenstein, 30 scatterings) -> SmartRayTracer on
the five clusters, with the scattering order of every hit. Writes to
experiments/inputs/mc/:

* ``point-seed<S>-steps.npz``: the steps as LightHit SpectralLightElements
  (``source_from_event``, spectral cones), as regenerate_inputs.py saves them;
* ``point-seed<S>-hits.npz``: uid, t_ns, phe, order and the run metadata.
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

POINT = (-12.0, -269.5, 337.0)
LENGTH_M = 0.05


def point_tracks(segments, seed, point, length_m=LENGTH_M):
    """NTSim gTracks: one two-row 100 GeV muon track per isotropic step."""
    from ntsim.IO import gTracks
    rng = np.random.default_rng(seed)
    direction = rng.normal(size=(segments, 3))
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    point = np.asarray(point, float)
    tracks = gTracks(size=2 * segments)
    tracks.uid = np.repeat(np.arange(segments), 2) + 1
    tracks.pdgid = np.full(2 * segments, 13)
    position = np.empty((2 * segments, 3))
    position[0::2] = point - 0.5 * length_m * direction
    position[1::2] = point + 0.5 * length_m * direction
    tracks.pos_m = position
    clock = np.zeros(2 * segments)
    clock[1::2] = length_m / 0.299792458
    tracks.t_ns = clock
    tracks.Etot_GeV = np.full(2 * segments, 100.0)
    step = np.zeros(2 * segments)
    step[1::2] = length_m
    tracks.step_length_m = step
    return tracks


class _Event:
    def __init__(self, tracks):
        self.tracks = {"point": tracks}
        self.id = 0


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--segments", type=int, default=50_000)
    parser.add_argument("--point", type=float, nargs=3, default=list(POINT))
    parser.add_argument("--scatterings", type=int, default=30)
    args = parser.parse_args()
    import ntsim
    from numba import njit
    from ntsim import bgvd
    from ntsim.CherenkovGenerators import CherenkovParameters, CherenkovTracksGenerator
    from ntsim.LightTransport import bgvd_model_from_ntsim
    from ntsim.LightTransport.LightHitTransport import source_from_event
    from ntsim.random import set_seed
    from regenerate_inputs import CLUSTERS
    from photon_mc import _order_tracer
    from harness.inputs import inputs_dir

    @njit
    def seed_numba(value):
        np.random.seed(value)

    # Three random streams: numpy (photon yields, positions, wavelengths,
    # directions), numba (free paths) and ntsim.random (scattering angles).
    np.random.seed(args.seed)
    seed_numba(args.seed)
    set_seed(args.seed)
    telescope = bgvd.Telescope(n_clusters=CLUSTERS)
    water = bgvd.Water()
    generator = CherenkovTracksGenerator(config=CherenkovParameters(
        volume=telescope.world.with_margin(100), phase_wavelength_nm=water.wavelength_nm,
        phase_refraction_index=water.phase_refraction_index, photon_suppression=1,
        chunk_size=200_000))
    propagator = ntsim.modules.MCPhotonPropagator(
        medium_prop=water, medium_scat=ntsim.scattering.HenyeyGreenstein(),
        n_scatterings=args.scatterings)
    orders = []
    tracer = _order_tracer(orders)(telescope=telescope)
    tracks = point_tracks(args.segments, args.seed, args.point)
    usage0 = resource.getrusage(resource.RUSAGE_SELF)
    began = time.perf_counter()
    uid, t_ns, phe, photons = [], [], [], 0
    for chunk in generator.generate_photons(tracks):
        photons += len(chunk)
        hits = tracer.propagate(propagator.propagate(chunk))
        if hits is not None:
            uid.append(np.asarray(hits.uid, np.int64))
            t_ns.append(np.asarray(hits.t_ns, float))
            phe.append(np.asarray(hits.phe, float))
    wall = time.perf_counter() - began
    usage1 = resource.getrusage(resource.RUSAGE_SELF)
    out = inputs_dir() / "mc"
    out.mkdir(parents=True, exist_ok=True)
    stem = f"point-seed{args.seed}"
    model = bgvd_model_from_ntsim(telescope, water)
    phase_max = float(np.max(model.medium.phase_index))
    elements = source_from_event(_Event(tracks), cone_model="spectral",
                                 max_phase_index=phase_max).elements
    fields = ("start_m", "direction", "length_m", "coefficient0", "coefficient2", "beta",
              "start_ns", "end_ns", "row_index", "uid")
    meta = dict(source="isotropic point", seed=args.seed, segments=args.segments,
                point_m=list(args.point), step_m=LENGTH_M, energy_GeV=100.0, pdg=13,
                n_scatterings=args.scatterings, photon_suppression=1, photons=int(photons),
                wall_s=wall, cpu_s=(usage1.ru_utime - usage0.ru_utime)
                + (usage1.ru_stime - usage0.ru_stime), max_phase_index=phase_max)
    np.savez_compressed(out / f"{stem}-steps.npz",
                        **{k: np.asarray(getattr(elements, k)) for k in fields},
                        metadata=json.dumps(meta))
    np.savez_compressed(out / f"{stem}-hits.npz", uid=np.concatenate(uid),
                        t_ns=np.concatenate(t_ns), phe=np.concatenate(phe),
                        order=np.concatenate(orders), metadata=json.dumps(meta))
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
