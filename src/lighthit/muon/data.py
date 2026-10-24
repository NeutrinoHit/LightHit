"""Prepared inputs of the muon transport and how they are found.

One JSON file, ``muon-data.json``, names the prepared tables; relative paths
are taken from the file's directory::

    {"rte_cache": "lighthit-cache-standard-freq65",
     "ge2_pack": "../ge2-latency-20261010/compact-pack",
     "primary_first_atlas": "../primary-universal-20261009/q1/refined",
     "primary_ge2_atlas": "../primary-universal-20261009/ge2/atlas",
     "primary_first_beta": "../first-charge-beta-20261010/first-beta-charge.npz",
     "primary_ge2_beta": "../ge2-charge-beta-20261010/beta-charge.npz",
     "preview_atlas": "../positive-map-20261009/atlas-seed81-fine.npz",
     "ge2_band_fold": "ge2-band-fold"}

``ge2_band_fold`` is optional: the folded >=2 kernels of the shower route,
which the primary's direct >=2 fallback then uses for nearly relativistic
pieces (``universal_primary.DirectGe2``).

All tables depend on the water optics and the OM response, not on the
detector geometry. They are runtime inputs and are never part of a wheel.
"""
from dataclasses import dataclass, fields
import json
import os
from pathlib import Path

MANIFEST = 'muon-data.json'
ENVIRONMENT = 'LIGHTHIT_MUON_DATA'


@dataclass(frozen=True)
class MuonData:
    """Paths of the prepared tables used by :class:`MuonTransport`."""
    rte_cache: Path
    ge2_pack: Path
    primary_first_atlas: Path
    primary_ge2_atlas: Path
    primary_first_beta: Path
    primary_ge2_beta: Path
    preview_atlas: Path
    manifest: Path | None = None
    ge2_band_fold: Path | None = None

    @classmethod
    def load(cls, path):
        """Read a manifest file, or ``muon-data.json`` inside a directory."""
        path = Path(path).expanduser()
        if path.is_dir():
            path = path / MANIFEST
        entries = json.loads(path.read_text())
        optional = ('manifest', 'ge2_band_fold')
        names = [item.name for item in fields(cls) if item.name not in optional]
        missing = [name for name in names if name not in entries]
        if missing:
            raise ValueError(f'{path}: missing entries {missing}')
        base = path.parent
        fold = entries.get('ge2_band_fold')
        data = cls(**{name: (base / entries[name]).resolve() for name in names},
                   manifest=path.resolve(),
                   ge2_band_fold=None if fold is None else (base / fold).resolve())
        data.check()
        return data

    @classmethod
    def find(cls, cache_directory=None):
        """Locate the manifest: environment variable, then next to the RTE cache.

        ``LIGHTHIT_MUON_DATA`` may name the file or its directory. Without it
        the cache directory and its parent are searched, so a kernel built on
        the prepared cache finds the rest of the tables by itself.
        """
        candidates = []
        if os.environ.get(ENVIRONMENT):
            candidates.append(Path(os.environ[ENVIRONMENT]))
        if cache_directory is not None:
            cache = Path(cache_directory).expanduser()
            candidates += [cache / MANIFEST, cache.parent / MANIFEST]
        for candidate in candidates:
            if candidate.is_file() or (candidate / MANIFEST).is_file():
                return cls.load(candidate)
        searched = ', '.join(str(item) for item in candidates) or 'nothing (no cache directory)'
        raise FileNotFoundError(f'no {MANIFEST} found; searched {searched}. '
                                f'Set {ENVIRONMENT} or place {MANIFEST} next to the RTE cache.')

    def check(self):
        """Fail early when a named table does not exist."""
        absent = [f'{item.name}={getattr(self, item.name)}' for item in fields(self)
                  if item.name != 'manifest' and getattr(self, item.name) is not None
                  and not Path(getattr(self, item.name)).exists()]
        if absent:
            raise FileNotFoundError('muon data incomplete: ' + ', '.join(absent))

    def paths(self):
        """Every file whose content defines the result (for provenance)."""
        files = [self.primary_first_atlas / 'manifest.json', self.primary_ge2_atlas / 'manifest.json',
                 self.primary_first_beta, self.primary_ge2_beta, self.preview_atlas,
                 self.ge2_pack / 'pack.json']
        if self.manifest is not None:
            files.insert(0, self.manifest)
        return files


def write_manifest(path, **entries):
    """Write ``muon-data.json`` with paths relative to its directory."""
    path = Path(path)
    base = path.parent.resolve()
    relative = {name: os.path.relpath(Path(value).resolve(), base) for name, value in entries.items()}
    path.write_text(json.dumps(relative, indent=2) + '\n')
    return MuonData.load(path)
