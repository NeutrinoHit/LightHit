"""Shared setup for the shower-basis study.

Detector: cluster 1 of the bundle model (288 OMs; the study first used
``load_bgvd_model(dataset="2021", clusters=1)``, whose water and OM response
equal the bundle's). Tables: the bundle's 65-frequency RTE cache. Events (see
scripts/shower_basis/paths.py): "e100" the bundle Geant4 100 GeV electron
(events 0-9); "e1000" the regenerated 1 TeV electron, "mu100" the regenerated
100 GeV muon and "mu1000" the bundle 1 TeV muon (one event each, index 0).
The study's own files (ref_*.npz, cell_*.npz, exp*.npz, ...) are read from and
written to the current directory: run the scripts from paths.work("shower-basis").
"""
import json
from dataclasses import replace

import numpy as np
import lighthit as lh

import paths

WORK_DIR = str(paths.work("shower-basis"))
CACHE = paths.RTE_CACHE
POSITION = [11.428243447430553, -249.74053298611113, 362.13987569444436]
DIRECTION = [0.4, 0.2, 0.894]
_STEPS = {"e1000": lambda: paths.inputs() / "electron-1TeV-steps.npz",
          "mu100": lambda: paths.inputs() / "muon-100GeV-steps.npz",
          "mu1000": lambda: paths.muon_source(1000)}


def model():
    full = lh.load_model(paths.MODEL.path())
    d = full.detector
    keep = np.asarray(d.identifiers["cluster_id"]) == 0
    area = np.asarray(d.effective_area_m2)
    detector = replace(d, positions_m=d.positions_m[keep], orientations=d.orientations[keep],
                       effective_area_m2=area[keep] if area.ndim else area,
                       identifiers={k: np.asarray(v)[keep] for k, v in d.identifiers.items()})
    return replace(full, detector=detector)


def config(**kw):
    base = dict(omega_per_ns=np.linspace(0., 1.2, 65),
                readout_omega_per_ns=np.linspace(0., 1.2, 321),
                relative_time_edges_ns=np.arange(-60., 745., 5.),
                cache_directory=str(CACHE), cache_policy="require")
    base.update(kw)
    return lh.KernelConfig(**base)


def shower(kind="e100", event=5, position=POSITION, direction=DIRECTION):
    pose = lh.SourcePose(position, direction, 0.0)
    if kind == "e100":
        return lh.G4Shower.from_hdf5(paths.ELECTRON_100GEV.path(), event=event).placed(pose)
    if event != 0:
        raise ValueError(f"{kind} has one event (index 0)")
    with np.load(_STEPS[kind](), allow_pickle=False) as z:
        fields = {k: z[k] for k in z.files if k != "metadata"}
        meta = json.loads(str(z["metadata"]))
    elements = lh.SpectralLightElements(**fields, provenance={"source": str(meta.get("name", kind))})
    return lh.G4Shower(elements, kind, 0).placed(pose)
