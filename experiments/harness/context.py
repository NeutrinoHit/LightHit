"""Per-process context of a case: cached kernels, routes and inputs, input hashes.

The warm-up call and the timed call of a case share this context, so the timed
call sees what a second event of the same run sees: compiled Numba code, loaded
tables, memory-mapped packs and prepared band folds.
"""
from __future__ import annotations

from pathlib import Path

from . import inputs as I


class Context:
    def __init__(self, case=None):
        self.case = case
        self._cache = {}
        self.inputs = {}

    def cached(self, key, factory):
        if key not in self._cache:
            self._cache[key] = factory()
        return self._cache[key]

    def _note(self, *paths):
        for p in paths:
            p = Path(p)
            self.inputs[str(p.name)] = I.file_hash(str(p.resolve()))

    # kernels -------------------------------------------------------------
    def _tables(self, *directories):
        """Table files are named by their physics key; record the names (not hashes)."""
        for directory in directories:
            for p in sorted(Path(directory).rglob("*")):
                if p.suffix == ".json" and p.stat().st_size < 1 << 20:
                    self._note(p)
            self.inputs[f"{Path(directory).name}/files"] = ",".join(
                sorted(p.name for p in Path(directory).iterdir() if p.suffix in (".npz", ".npy")))[:4000]

    def kernel(self, **overrides):
        key = ("kernel", tuple(sorted((k, repr(v)) for k, v in overrides.items())))
        def make():
            self._note(I.bundle_path("model", "bgvd-model.npz"),
                       I.bundle_path("tables", "muon-data.json"))
            self._tables(I.bundle_path("tables", "rte-cache"), I.bundle_path("tables", "ge2-pack"),
                         I.bundle_path("tables", "muon"))
            return I.route_kernel(**overrides)
        return self.cached(key, make)

    def laser_kernel(self, **options):
        def make():
            self._tables(I.bundle_path("tables", "laser-cache"))
            return I.laser_kernel(**options)
        return self.cached(("laser", tuple(sorted(options.items()))), make)

    # inputs --------------------------------------------------------------
    def electron(self, event=I.ELECTRON_EVENT):
        def make():
            self._note(I.bundle_path("events", "geant4-electron-100GeV.h5"))
            return I.electron(self.kernel(), event)
        return self.cached(("electron", event), make)

    def muon_1tev(self):
        def make():
            self._note(I.bundle_path("events", "muon-1TeV-steps.npz"),
                       I.bundle_path("events", "muon-1TeV-ancestry.npz"))
            return I.muon_1tev()
        return self.cached("muon-1TeV", make)

    def regenerated(self, name):
        def make():
            base = I.inputs_dir()
            self._note(base / f"{name}-steps.npz", base / f"{name}-parts.npz")
            return I.regenerated(name)
        return self.cached(("regenerated", name), make)

    def shower_source(self, name):
        """A shower input by name: electron-100GeV or a single-shower regenerated event."""
        if name == "electron-100GeV":
            return self.electron()
        steps, parts, _ = self.regenerated(name)
        if len(parts) != 1 or parts[0]["kind"] != "shower":
            raise ValueError(f"{name} is not a single shower")
        return steps

    def muon_source(self, name):
        """(steps, ancestry, primary_uid) of a muon input: muon-1TeV or a regenerated muon."""
        if name == "muon-1TeV":
            steps, ancestry, _ = self.muon_1tev()
            return steps, ancestry, 0
        steps, parts, _ = self.regenerated(name)
        muons = [p for p in parts if p["kind"] == "muon"]
        if len(muons) != 1:
            raise ValueError(f"{name} has {len(muons)} muon sources")
        part = muons[0]
        elements = steps if part["mask"].all() else steps.subset(part["mask"])
        return elements, part["ancestry"], part["primary_uid"]

    def components(self, name):
        if name == "numu-like":     # the stand-alone notebook's 1 TeV muon + 100 GeV electron
            from lighthit.event import SourceComponent
            steps, ancestry, _ = self.muon_1tev()
            return [SourceComponent("muon", steps, ancestry, primary_uid=0, label="muon"),
                    SourceComponent("shower", self.electron(), label="vertex shower")]
        steps, parts, _ = self.regenerated(name)
        return I.components(steps, parts)
