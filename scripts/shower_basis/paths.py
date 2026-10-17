"""Where the research scripts find their inputs and put their outputs.

The scripts once read and wrote fixed folders of a local ``.build`` directory
that no longer exists. Their inputs now come from

* the data bundle ``$LIGHTHIT_DATA`` (tables, the BGVD model, the 1 TeV muon and
  the 100 GeV Geant4 electron);
* events regenerated with NTSim by ``experiments/regenerate_inputs.py`` into
  ``$LIGHTHIT_EXPERIMENT_INPUTS`` (default ``experiments/inputs``);
* responses computed by the experiment harness (``experiments/run.py``) in
  ``experiments/results``;

and their outputs and intermediate files go below ``$LIGHTHIT_WORK`` (default
``experiments/results/work``), under the name of the folder they used before.

    python scripts/shower_basis/paths.py prepare [ENERGY ...]

writes the muon sources in the layout the scripts expect (``muon<E>.npz`` with
the metadata of ``benchmark_long_muon.py generate``) and their particle
ancestry (``analyze_muon_secondaries.py --capture-source``, which regenerates
the Geant4 event and checks that the steps are identical).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def _data() -> Path:
    value = os.environ.get("LIGHTHIT_DATA")
    if not value:
        raise SystemExit("set LIGHTHIT_DATA to the lighthit-bgvd-data bundle")
    return Path(value).expanduser()


class _Lazy(os.PathLike):
    """A bundle path resolved when used (LIGHTHIT_DATA may be set after import)."""

    def __init__(self, *parts):
        self.parts = parts

    def path(self) -> Path:
        return _data().joinpath(*self.parts)

    def __fspath__(self):
        return str(self.path())

    def __truediv__(self, other):
        return self.path() / other

    def __str__(self):
        return str(self.path())

    def __getattr__(self, name):
        return getattr(self.path(), name)


RTE_CACHE = _Lazy("tables", "rte-cache")          # was .build/ntsim-check/lighthit-cache-standard-freq65
GE2_PACK = _Lazy("tables", "ge2-pack")            # was ge2-pack-shower-basis-review and ge2-latency-20261010/compact-pack
Q1_ATLAS = _Lazy("tables", "muon", "q1-atlas")    # was primary-universal-20261009/q1/refined
GE2_ATLAS = _Lazy("tables", "muon", "ge2-atlas")  # was primary-universal-20261009/ge2/atlas
Q1_BETA = _Lazy("tables", "muon", "first-beta-charge.npz")   # was first-charge-beta-20261010/first-beta-charge.npz
GE2_BETA = _Lazy("tables", "muon", "ge2-beta-charge.npz")    # was ge2-charge-beta-20261010/beta-charge.npz
PREVIEW_ATLAS = _Lazy("tables", "muon", "preview-atlas.npz")  # was positive-map-20261009/atlas-seed81-fine.npz
MODEL = _Lazy("model", "bgvd-model.npz")
ELECTRON_100GEV = _Lazy("events", "geant4-electron-100GeV.h5")


def inputs() -> Path:
    return Path(os.environ.get("LIGHTHIT_EXPERIMENT_INPUTS", REPO / "experiments" / "inputs")).expanduser()


def results() -> Path:
    return Path(os.environ.get("LIGHTHIT_EXPERIMENT_RESULTS", REPO / "experiments" / "results")).expanduser()


def work(*parts) -> Path:
    """Output or intermediate location; ``parts`` repeat the old .build sub-path."""
    root = Path(os.environ.get("LIGHTHIT_WORK", results() / "work")).expanduser()
    return root.joinpath(*map(str, parts))


#: energy (GeV) -> regenerated event name
_REGENERATED = {100: "muon-100GeV", 300: "muon-300GeV", 3000: "muon-3000GeV"}


def _energy(value) -> int:
    value = float(value)
    if not value.is_integer():
        raise ValueError(f"no standard muon of {value} GeV")
    return int(value)


def muon_source(energy) -> Path:
    """muon<E>.npz of benchmark_long_muon.py generate (all steps, seed 42, start
    (-100, -950, 350) m along +y): the bundle event for 1 TeV, else the
    regenerated event converted to that layout."""
    energy = _energy(energy)
    if energy == 1000:
        return _data() / "events" / "muon-1TeV-steps.npz"
    target = work("muon-five-cluster-20261008", f"muon{energy}.npz")
    if target.is_file():
        return target
    name = _REGENERATED.get(energy)
    source = inputs() / f"{name}-steps.npz" if name else None
    if source is None or not source.is_file():
        raise FileNotFoundError(f"no {energy} GeV muon: run experiments/regenerate_inputs.py "
                                f"{name or ''} first")
    import numpy as np
    with np.load(source, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files if k != "metadata"}
        meta = json.loads(str(z["metadata"]))
    (particle,) = meta["particles"]
    primary = arrays["uid"] == 0
    old = dict(energy_GeV=float(particle["energy_GeV"]), seed=meta["seed"], clusters=meta["clusters"],
               position_m=meta["vertex_m"], direction=particle["direction"],
               generated_seconds=meta["generated_seconds"], elements=int(len(arrays["uid"])),
               primary_elements=int(primary.sum()),
               primary_chord_length_m=float(arrays["length_m"][primary].sum()),
               source_bounds_m=[arrays["start_m"].min(0).tolist(), arrays["start_m"].max(0).tolist()],
               provenance=meta.get("provenance"), model_fingerprint=_fingerprint(),
               regenerated_from=source.name)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, **arrays, metadata=json.dumps(old))
    return target


def _fingerprint():
    import lighthit as lh
    return lh.load_model(MODEL.path()).fingerprint


def muon_ancestry(energy) -> Path:
    """Particle ancestry of muon_source(energy) as written by
    analyze_muon_secondaries.py --capture-source (bundle file for 1 TeV)."""
    energy = _energy(energy)
    if energy == 1000:
        return _data() / "events" / "muon-1TeV-ancestry.npz"
    target = work("muon-segments-20261009", "structure", f"muon{energy}-ancestry.npz")
    if not target.is_file():
        raise FileNotFoundError(f"{target} missing: run python scripts/shower_basis/paths.py prepare {energy}")
    return target


def muon_response(energy) -> Path:
    """Full-detector response of the standard muon, the replacement of the old
    corrected<E>/response.npz: the default MuonTransport response computed by
    experiments/run.py (case default-muon-route/muon-<E>). It is not the same
    calculation as the old corrected hybrid response; scripts use it for the
    per-OM clock, the active set and, where they compare charges, as the event's
    default-route response."""
    energy = _energy(energy)
    name = {1000: "muon-1TeV"}.get(energy, f"muon-{energy}GeV")
    path = results() / "default-muon-route" / f"{name}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"{path} missing: run python experiments/run.py run default-muon-route/{name}")
    return path


def ntsim_hybrid_response(energy, components="off", stencil="axis") -> Path:
    """The NTSim expert hybrid muon run of ntsim/run/check_muon_fast.sh (muon-fast
    prompt profile, apply and batch tolerances 1e-6, 65-frequency pack;
    MUON_COMPONENTS and MUON_TRACK_STENCIL as given), recomputed on the same
    Geant4 event by experiments/run.py (idea ntsim-hybrid-muon-fast). It replaces
    the old ntsim<E>-corrected, ntsim<E>-all and ntsim1000-all-centroid outputs."""
    energy = _energy(energy)
    name = {1000: "muon-1TeV"}.get(energy, f"muon-{energy}GeV")
    variant = components + ("-centroid" if stencil == "centroid" else "")
    path = results() / "ntsim-hybrid-muon-fast" / f"{name}-{variant}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"{path} missing: run python experiments/run.py run "
                                f"ntsim-hybrid-muon-fast/{name}-{variant}")
    return path


def holdout_primary() -> Path:
    """Primary-only steps of the independent 1 TeV holdout (seed 43, start
    (-120, -900, 330) m, direction (0.02, 1, 0.01)), the input of
    generate_primary_holdout.py's consumers."""
    target = work("primary-universal", "holdout", "muon1000-seed43-primary.npz")
    if target.is_file():
        return target
    import numpy as np
    source = inputs() / "muon-1TeV-seed43-steps.npz"
    if not source.is_file():
        raise FileNotFoundError("run experiments/regenerate_inputs.py muon-1TeV-seed43 first")
    with np.load(source, allow_pickle=False) as z:
        primary = z["uid"] == 0
        arrays = {k: z[k][primary] for k in z.files if k != "metadata"}
        meta = json.loads(str(z["metadata"]))
    (particle,) = meta["particles"]
    meta = dict(energy_GeV=float(particle["energy_GeV"]), seed=meta["seed"], clusters=meta["clusters"],
                position_m=meta["vertex_m"], direction=particle["direction"],
                generated_seconds=meta["generated_seconds"], primary_only=True,
                elements=int(primary.sum()), primary_elements=int(primary.sum()),
                primary_chord_length_m=float(arrays["length_m"].sum()),
                provenance=meta.get("provenance"), model_fingerprint=_fingerprint(),
                regenerated_from=source.name)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, **arrays, metadata=json.dumps(meta))
    return target


def prepare(energies):
    """Write muon<E>.npz and capture its ancestry (one Geant4 process per event)."""
    for energy in energies:
        source = muon_source(energy)
        if energy == 1000:
            # analyze_muon_secondaries.py looks for <output>/<source stem>-ancestry.npz
            import shutil
            copy = work("muon-segments-20261009", "structure", "muon-1TeV-steps-ancestry.npz")
            copy.parent.mkdir(parents=True, exist_ok=True)
            if not copy.is_file():
                shutil.copyfile(muon_ancestry(1000), copy)
            continue
        target = work("muon-segments-20261009", "structure", f"muon{energy}-ancestry.npz")
        if target.is_file():
            continue
        subprocess.run([sys.executable, str(HERE / "analyze_muon_secondaries.py"),
                        "--capture-source", str(source), "--output", str(target.parent)], check=True)
        produced = target.parent / f"{source.stem}-ancestry.npz"
        if produced != target:
            produced.rename(target)


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "prepare":
        prepare([int(e) for e in sys.argv[2:]] or [300, 1000, 3000])
    else:
        raise SystemExit(__doc__)
