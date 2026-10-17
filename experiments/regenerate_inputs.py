#!/usr/bin/env python
"""Regenerate the NTSim/Geant4 events of the experiments into experiments/inputs/.

Each event runs in its own process (Geant4 is started once per process):

    python experiments/regenerate_inputs.py            # all events
    python experiments/regenerate_inputs.py nue-cc     # one event

Writes, per event, ``<name>-steps.npz`` (the fields of SpectralLightElements
with spectral cones, as ``source_from_event`` builds them for the 1440 OMs of
clusters 1-5, plus a JSON ``metadata`` string) and ``<name>-parts.npz`` (the
split of NTSim's ``event_components``: element indices per source and, for
muon sources, the ``muon_ancestry`` table). Needs the NTSim environment with
Geant4 and the bgvd_model package.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

ELECTRON_VERTEX = [5.0, -269.5, 330.0]          # inside cluster 1, as the bundle electron
ELECTRON_DIRECTION = [0.35, -0.20, 0.915]
NEAR_OM = [-19.98255, -269.661, 328.555]         # OM 267 (cluster 1, string 7)
MUON_START = [-100.0, -950.0, 350.0]


def _unit(v):
    v = np.asarray(v, float)
    return (v / np.linalg.norm(v)).tolist()


def _aimed_start(distance_m):
    """Start ``distance_m`` before OM 267 on a line passing 1.5 m from it."""
    d = np.asarray(_unit(ELECTRON_DIRECTION))
    u = np.cross(d, [0.0, 0.0, 1.0])
    u /= np.linalg.norm(u)
    return (np.asarray(NEAR_OM) - distance_m * d + 1.5 * u).tolist()


#: name -> (vertex, [(pdg, energy GeV, direction), ...][, seed])
EVENTS = {
    "electron-100GeV": (ELECTRON_VERTEX, [(11, 100.0, ELECTRON_DIRECTION)]),
    "electron-1TeV": (ELECTRON_VERTEX, [(11, 1000.0, ELECTRON_DIRECTION)]),
    "pion-100GeV": (ELECTRON_VERTEX, [(211, 100.0, ELECTRON_DIRECTION)]),
    "nue-cc": (ELECTRON_VERTEX, [(11, 70.0, ELECTRON_DIRECTION),
                                 (211, 25.0, [0.55, -0.10, 0.83]),
                                 (2212, 5.0, [0.20, -0.40, 0.89])]),
    "numu-cc": ([5.0, -269.5, 200.0], [(13, 800.0, [0.30, -0.15, 0.94]),
                                        (211, 150.0, [0.50, -0.30, 0.81]),
                                        (2212, 50.0, [0.10, 0.20, 0.97])]),
    "muon-100GeV": ([-100.0, -950.0, 350.0], [(13, 100.0, [0.0, 1.0, 0.0])]),
    # 6 GeV muon (about 25-30 m of track) aimed 1.5 m past OM 267:
    # muon-near-om starts 20 m before the OM and passes it at 1.5 m;
    # muon-stopping starts 40 m before it and stops short of it.
    "muon-near-om": (_aimed_start(20.0), [(13, 6.0, ELECTRON_DIRECTION)]),
    "muon-stopping": (_aimed_start(40.0), [(13, 6.0, ELECTRON_DIRECTION)]),
    # The research muons of the muon-route scripts (benchmark_long_muon.py generate:
    # ToyGen mu-, seed 42, start (-100, -950, 350) m, direction +y). muon-1TeV-check
    # regenerates the bundle event to test that Geant4 reproduces it bit for bit.
    "muon-300GeV": (MUON_START, [(13, 300.0, [0.0, 1.0, 0.0])]),
    "muon-3000GeV": (MUON_START, [(13, 3000.0, [0.0, 1.0, 0.0])]),
    "muon-1TeV-check": (MUON_START, [(13, 1000.0, [0.0, 1.0, 0.0])]),
    # independent holdout of the primary atlases (generate_primary_holdout.py defaults)
    "muon-1TeV-seed43": ([-120.0, -900.0, 330.0], [(13, 1000.0, [0.02, 1.0, 0.01])], 43),
}
SEED = 42
CLUSTERS = (1, 2, 3, 4, 5)


def chain_modules(name: str):
    """(generator, Geant4 propagator) of an event in EVENTS."""
    import ntsim
    vertex, particles, *rest = EVENTS[name]
    seed = rest[0] if rest else SEED
    if len(particles) == 1:
        pdg, energy, direction = particles[0]
        generator = ntsim.primary.ToyGen(energy=energy, pdg=pdg, position=tuple(vertex),
                                         direction=tuple(_unit(direction)), time=0.0)
    else:
        generator = ntsim.primary.ParticleList(
            pdg=tuple(p[0] for p in particles), energy=tuple(p[1] for p in particles),
            direction=tuple(tuple(_unit(p[2])) for p in particles),
            position=(tuple(vertex),), time=(0.0,))
    propagator = ntsim.modules.ParticlePropagator(
        random_seed=seed, enable_cherenkov=False, cascade_bounds_e=(0.0, 0.0),
        save_process_name=True)
    return generator << "primary_gen", propagator << "g4prop"


def generate(name: str, out: Path, *, extra=(), stem: str | None = None, telescope=None):
    """Run one event (Geant4 and any ``extra`` chain stages) and save its steps.

    ``telescope``: the NTSim telescope the ``extra`` stages already use (NTSim
    allows one telescope per process). Returns the processed NTSim event.
    """
    from ntsim.event_chain import Chain, EmptyEventSource
    began = time.perf_counter()
    event = next(iter(Chain(*chain_modules(name), *extra)(EmptyEventSource())))
    save_event(name, event, out, stem=stem or name, generated_seconds=time.perf_counter() - began,
               telescope=telescope)
    return event


def save_event(name: str, event, out: Path, *, stem: str, generated_seconds: float, telescope=None):
    """<stem>-steps.npz and <stem>-parts.npz of a processed NTSim event."""
    import ntsim
    import lighthit as lh
    from ntsim.LightTransport import bgvd_model_from_ntsim
    from ntsim.LightTransport.LightHitTransport import (event_components, muon_ancestry,
                                                        source_from_event)
    vertex, particles, *rest = EVENTS[name]
    seed = rest[0] if rest else SEED
    model = bgvd_model_from_ntsim(telescope or ntsim.bgvd.Telescope(n_clusters=CLUSTERS), ntsim.bgvd.Water())
    phase_max = float(np.max(model.medium.phase_index))
    source = source_from_event(event, cone_model="spectral", max_phase_index=phase_max)
    elements = source.elements
    parts = event_components(event, source)
    arrays, records = {}, []
    for index, part in enumerate(parts):
        arrays[f"mask_{index}"] = np.flatnonzero(part.mask)
        record = dict(kind=part.kind, label=part.label, primary_uid=part.primary_uid,
                      roots=list(part.roots), elements=int(part.mask.sum()))
        if part.kind == "muon":
            subset = elements if np.all(part.mask) else elements.subset(part.mask)
            ancestry, summary = muon_ancestry(event, subset, part.primary_uid)
            for key, value in ancestry.items():
                arrays[f"ancestry_{index}_{key}"] = np.asarray(value)
            record["ancestry_summary"] = summary
        records.append(record)
    metadata = dict(name=name, vertex_m=vertex, particles=[dict(pdg=p[0], energy_GeV=p[1],
                    direction=_unit(p[2])) for p in particles], seed=seed, clusters=list(CLUSTERS),
                    ntsim_version=getattr(ntsim, "__version__", "unknown"),
                    lighthit_version=lh.__version__, elements=int(len(elements.length_m)),
                    max_phase_index=phase_max, parts=records,
                    generated_seconds=generated_seconds, provenance=elements.provenance)
    fields = ("start_m", "direction", "length_m", "coefficient0", "coefficient2", "beta",
              "start_ns", "end_ns", "row_index", "uid")
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"{stem}-steps.npz",
                        **{k: np.asarray(getattr(elements, k)) for k in fields},
                        metadata=json.dumps(metadata))
    np.savez_compressed(out / f"{stem}-parts.npz", **arrays)
    print(json.dumps({k: metadata[k] for k in ("name", "elements", "generated_seconds")}
                     | {"parts": [(r["kind"], r["label"], r["elements"]) for r in records]}))


def main():
    from harness.inputs import inputs_dir
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="*", help=f"events among {sorted(EVENTS)}")
    parser.add_argument("--one", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    out = inputs_dir()
    names = args.names or list(EVENTS)
    unknown = sorted(set(names) - set(EVENTS))
    if unknown:
        parser.error(f"unknown events {unknown}")
    if args.one:
        generate(names[0], out)
        return
    for name in names:
        subprocess.run([sys.executable, __file__, "--one", name], check=True)


if __name__ == "__main__":
    main()
