"""Prepared inputs of the shower transport and how they are found.

The shower route needs the RTE cache (the kernel's) and the order >=2 kernel
pack of :class:`lighthit.shower_basis.Ge2KernelStore`. ``shower-data.json``
names them, optionally with a prepared band-folded pack::

    {"rte_cache": "lighthit-cache-standard-freq65",
     "ge2_pack": "ge2-pack-shower-basis-review",
     "ge2_band_fold": "ge2-band-fold"}

Without that file the ``rte_cache`` and ``ge2_pack`` entries of the muon
manifest ``muon-data.json`` are used (same tables). A missing folded pack, or
one of another rank, subdivision or frequency set, is built in memory from the
>=2 pack on first use.
"""
from dataclasses import dataclass
import json
import os
from pathlib import Path

MANIFEST = "shower-data.json"
ENVIRONMENT = "LIGHTHIT_SHOWER_DATA"


@dataclass(frozen=True)
class ShowerData:
    """Paths of the tables used by :class:`lighthit.shower.ShowerTransport`."""
    rte_cache: Path
    ge2_pack: Path
    ge2_band_fold: Path | None = None
    manifest: Path | None = None

    @classmethod
    def of(cls, rte_cache, ge2_pack, ge2_band_fold=None):
        """Tables given directly (no manifest file); paths must exist."""
        paths = [Path(item).expanduser().resolve() for item in (rte_cache, ge2_pack)]
        absent = [str(item) for item in paths if not item.exists()]
        if absent:
            raise FileNotFoundError("shower data incomplete: " + ", ".join(absent))
        fold = None if ge2_band_fold is None else Path(ge2_band_fold).expanduser().resolve()
        return cls(paths[0], paths[1], fold)

    @classmethod
    def load(cls, path):
        """Read ``shower-data.json`` (or ``muon-data.json``) or a directory holding one."""
        path = Path(path).expanduser()
        if path.is_dir():
            path = path / MANIFEST if (path / MANIFEST).is_file() else path / "muon-data.json"
        entries = json.loads(path.read_text())
        missing = [name for name in ("rte_cache", "ge2_pack") if name not in entries]
        if missing:
            raise ValueError(f"{path}: missing entries {missing}")
        base = path.parent
        fold = entries.get("ge2_band_fold")
        data = cls((base / entries["rte_cache"]).resolve(), (base / entries["ge2_pack"]).resolve(),
                   None if fold is None else (base / fold).resolve(), path.resolve())
        absent = [str(item) for item in (data.rte_cache, data.ge2_pack) if not item.exists()]
        if absent:
            raise FileNotFoundError("shower data incomplete: " + ", ".join(absent))
        return data

    @classmethod
    def find(cls, cache_directory=None):
        """Environment variable, then next to the RTE cache (shower, then muon manifest)."""
        candidates = []
        if os.environ.get(ENVIRONMENT):
            candidates.append(Path(os.environ[ENVIRONMENT]))
        if cache_directory is not None:
            cache = Path(cache_directory).expanduser()
            for name in (MANIFEST, "muon-data.json"):
                candidates += [cache / name, cache.parent / name]
        for candidate in candidates:
            if candidate.is_file():
                return cls.load(candidate)
            if candidate.is_dir() and ((candidate / MANIFEST).is_file()
                                       or (candidate / "muon-data.json").is_file()):
                return cls.load(candidate)
        searched = ", ".join(str(item) for item in candidates) or "nothing (no cache directory)"
        raise FileNotFoundError(f"no {MANIFEST} or muon-data.json found; searched {searched}")
