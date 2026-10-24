"""Standard inputs of the experiments.

Everything comes from the data bundle (``$LIGHTHIT_DATA``) or from events
regenerated with NTSim by ``experiments/regenerate_inputs.py`` into
``experiments/inputs/`` (``$LIGHTHIT_EXPERIMENT_INPUTS`` overrides it).
No other path is read.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]

#: Pose of the bundle electron in the stand-alone notebook (inside cluster 1).
ELECTRON_POSE = dict(position_m=[5.0, -269.5, 330.0], direction=[0.35, -0.20, 0.915])
ELECTRON_EVENT = 5
#: The regenerated events (name -> NTSim primaries); see regenerate_inputs.py.
REGENERATED = ("electron-100GeV", "electron-1TeV", "pion-100GeV", "nue-cc", "numu-cc", "muon-100GeV", "muon-near-om", "muon-stopping",
               "muon-300GeV", "muon-3000GeV", "muon-1TeV-check", "muon-1TeV-seed43", "muon-100GeV-cluster1")


def data_dir() -> Path:
    value = os.environ.get("LIGHTHIT_DATA")
    if not value:
        raise SystemExit("set LIGHTHIT_DATA to the lighthit-bgvd-data bundle")
    path = Path(value).expanduser()
    if not (path / "tables" / "muon-data.json").is_file():
        raise SystemExit(f"{path} is not a LightHit data bundle (tables/muon-data.json missing)")
    return path


def inputs_dir() -> Path:
    return Path(os.environ.get("LIGHTHIT_EXPERIMENT_INPUTS", HERE / "inputs")).expanduser()


def sha256(path: Path, chunk: int = 1 << 22) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while block := stream.read(chunk):
            digest.update(block)
    return digest.hexdigest()


@lru_cache(maxsize=None)
def file_hash(path: str) -> str:
    """sha256 of an input file (cached per process)."""
    return sha256(Path(path))


def bundle_path(*parts) -> Path:
    return data_dir().joinpath(*parts)


@lru_cache(maxsize=1)
def model():
    import lighthit as lh
    return lh.load_model(bundle_path("model", "bgvd-model.npz"))


def route_config(**overrides):
    """The kernel configuration of the stand-alone notebook (65 table, 321 readout frequencies)."""
    import lighthit as lh
    base = dict(omega_per_ns=np.linspace(0.0, 1.2, 65),
                readout_omega_per_ns=np.linspace(0.0, 1.2, 321),
                relative_time_edges_ns=np.arange(-60.0, 745.0, 5.0),
                cache_directory=bundle_path("tables", "rte-cache"), cache_policy="require")
    base.update(overrides)
    return lh.KernelConfig(**base)


def route_kernel(**overrides):
    return model().kernel(route_config(**overrides))


def subset_kernel(kernel, indices, **overrides):
    """A kernel over the OMs ``indices`` only (same medium and tables)."""
    import lighthit as lh
    detector = kernel.detector
    indices = np.asarray(indices, int)
    sub = detector.subset(indices)
    config = replace(kernel.config, **overrides) if overrides else kernel.config
    return lh.TransportKernel(kernel.medium, sub, config)


def max_phase_index(kernel) -> float:
    return float(np.max(kernel.medium.sample(kernel.wavelength.wavelength_nm)["phase_index"]))


def electron(kernel, event: int = ELECTRON_EVENT, pose: dict | None = None):
    """Bundle Geant4 100 GeV electron, spectral cones, placed like the notebook."""
    import lighthit as lh
    pose = pose or ELECTRON_POSE
    g4 = lh.G4Shower.from_hdf5(bundle_path("events", "geant4-electron-100GeV.h5"), event=event,
                               reference_phase_index=max_phase_index(kernel))
    return replace(g4.elements, cone_model="spectral").placed(lh.SourcePose(**pose))


def load_steps(path: Path):
    import lighthit as lh
    with np.load(path, allow_pickle=False) as data:
        fields = {k: data[k] for k in data.files if k != "metadata"}
        meta = json.loads(str(data["metadata"])) if "metadata" in data.files else {}
    return lh.SpectralLightElements(**fields, cone_model="spectral"), meta


def load_ancestry(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files if k != "metadata"}


def muon_1tev():
    """Bundle 1 TeV muon: (steps, ancestry, metadata)."""
    steps, meta = load_steps(bundle_path("events", "muon-1TeV-steps.npz"))
    return steps, load_ancestry(bundle_path("events", "muon-1TeV-ancestry.npz")), meta


def regenerated(name: str):
    """A regenerated NTSim event: (steps, parts, metadata).

    ``parts`` is a list of dicts (kind, label, mask, primary_uid, ancestry) in
    the split NTSim's ``event_components`` makes.
    """
    if name not in REGENERATED:
        raise KeyError(name)
    base = inputs_dir()
    steps_path = base / f"{name}-steps.npz"
    if not steps_path.is_file():
        raise FileNotFoundError(f"{steps_path} missing: run experiments/regenerate_inputs.py")
    steps, meta = load_steps(steps_path)
    parts = []
    for index, record in enumerate(meta["parts"]):
        mask = np.zeros(len(steps.length_m), bool)
        with np.load(base / f"{name}-parts.npz", allow_pickle=False) as data:
            mask[data[f"mask_{index}"]] = True
            ancestry = ({k[len(f"ancestry_{index}_"):]: data[k] for k in data.files
                         if k.startswith(f"ancestry_{index}_")} if record["kind"] == "muon" else None)
        parts.append(dict(record, mask=mask, ancestry=ancestry))
    return steps, parts, meta


def components(steps, parts):
    """lighthit.event SourceComponents of a regenerated event."""
    from lighthit.event import SourceComponent
    out = []
    for part in parts:
        elements = steps if part["mask"].all() else steps.subset(part["mask"])
        if part["kind"] == "muon":
            out.append(SourceComponent("muon", elements, part["ancestry"], part["primary_uid"],
                                       part["label"]))
        else:
            out.append(SourceComponent("shower", elements, label=part["label"]))
    return out


def laser_kernel(**options):
    from lighthit.event import laser_kernel as make
    m = model()
    base = dict(cache_directory=bundle_path("tables", "laser-cache"), bin_ns=5.0,
                omega_per_ns=np.linspace(0.0, 1.2, 321), cache_policy="require")
    base.update(options)
    return make(m.medium, m.detector, **base)


#: Laser flash of the stand-alone notebook (between clusters 1, 2 and 4).
LASER = dict(position_m=[-62.0, -385.0, 361.0], photons=1e15, wavelength_nm=532.0)


def laser_flash():
    import lighthit as lh
    return lh.IsotropicFlash.monochromatic(LASER["position_m"], LASER["photons"],
                                           LASER["wavelength_nm"])


def input_record(*paths) -> dict:
    """{file name: sha256} of the input files used."""
    return {Path(p).name: file_hash(str(Path(p).resolve())) for p in paths}
