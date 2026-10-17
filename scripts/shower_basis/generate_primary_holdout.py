#!/usr/bin/env python3
"""Generate an independent, uncompressed primary-only G4 validation source.

Uses already installed Geant4 data, and fails rather than downloading data.
No transport cache is read or created. Geometry and all source arrays are saved
under the requested private output directory; existing sources are preserved.
"""
from __future__ import annotations

import paths  # inputs and outputs: scripts/shower_basis/paths.py
import argparse
from dataclasses import fields
import hashlib
from importlib.metadata import version, PackageNotFoundError
import json
import os
from pathlib import Path
import sys
from time import perf_counter, process_time

import numpy as np


def installed_g4_data(directory):
    mapping = {
        'G4ABLADATA': 'G4ABLA', 'G4LEDATA': 'G4EMLOW',
        'G4ENSDFSTATEDATA': 'G4ENSDFSTATE', 'G4INCLDATA': 'G4INCL',
        'G4NEUTRONHPDATA': 'G4NDL', 'G4PARTICLEXSDATA': 'G4PARTICLEXS',
        'G4PIIDATA': 'G4PII', 'G4SAIDXSDATA': 'G4SAIDDATA',
        'G4LEVELGAMMADATA': 'PhotonEvaporation',
        'G4RADIOACTIVEDATA': 'RadioactiveDecay',
        'G4REALSURFACEDATA': 'RealSurface', 'G4CHANNELINGDATA': 'G4CHANNELING',
    }
    resolved = {}
    for name, prefix in mapping.items():
        explicit = os.environ.get(name)
        candidates = ([Path(explicit)] if explicit else
                      sorted(p for p in directory.glob(prefix+'*') if p.is_dir()))
        if not candidates or not candidates[-1].is_dir() or not any(candidates[-1].iterdir()):
            raise FileNotFoundError(f'{name}: installed data directory missing in {directory}')
        path = candidates[-1].resolve()
        os.environ[name] = str(path)
        resolved[name] = str(path)
    return resolved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--energy', type=float, default=1000.)
    parser.add_argument('--seed', type=int, default=43)
    parser.add_argument('--position', type=float, nargs=3, default=(-120., -900., 330.))
    parser.add_argument('--direction', type=float, nargs=3, default=(.02, 1., .01))
    parser.add_argument('--g4-data', type=Path, default=Path.home()/'.geant4_pybind')
    parser.add_argument('--output', type=Path,
                        default=paths.work('primary-universal-20261009/holdout'))
    args = parser.parse_args()
    if args.energy <= 0 or not np.isfinite(args.energy):
        parser.error('energy must be finite and positive')
    direction = np.asarray(args.direction, float)
    if not np.isfinite(direction).all() or np.linalg.norm(direction) <= 0:
        parser.error('finite nonzero direction required')
    direction /= np.linalg.norm(direction)
    target = args.output/f'muon{args.energy:g}-seed{args.seed}-primary.npz'
    if target.exists():
        parser.error(f'preserve existing source: {target}')
    g4data = installed_g4_data(args.g4_data)
    # A missing dataset must never trigger an interactive download prompt.
    sys.stdin = open(os.devnull, 'r')
    import ntsim
    import lighthit as lh
    from ntsim.LightTransport import bgvd_model_from_ntsim, source_from_event
    from ntsim.event_chain import Chain, EmptyEventSource
    model = bgvd_model_from_ntsim(ntsim.bgvd.Telescope(n_clusters=(1, 2, 3, 4, 5)),
                                 ntsim.bgvd.Water())
    chain = Chain(
        ntsim.primary.ToyGen(energy=args.energy, pdg=13, position=tuple(args.position),
                            direction=tuple(direction), time=0.) << 'primary_gen',
        ntsim.modules.ParticlePropagator(random_seed=args.seed, enable_cherenkov=False,
                                        cascade_bounds_e=(0., 0.)) << 'g4prop')
    wall, cpu = perf_counter(), process_time()
    event = next(chain(EmptyEventSource()))
    generation_wall, generation_cpu = perf_counter()-wall, process_time()-cpu
    wall, cpu = perf_counter(), process_time()
    all_source = source_from_event(event, cone_model='spectral',
                         max_phase_index=float(np.max(model.medium.phase_index))).elements
    source = all_source.subset(all_source.uid == 0)
    conversion_wall, conversion_cpu = perf_counter()-wall, process_time()-cpu
    if np.any(np.diff(source.start_ns) < 0):
        raise ValueError('primary chronology is not ordered')
    arrays = {f.name: np.asarray(getattr(source, f.name)) for f in fields(source)
              if isinstance(getattr(source, f.name), np.ndarray)}
    packages = {}
    for name in ('lighthit', 'ntsim', 'g4camp', 'geant4_pybind'):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = 'distribution metadata unavailable'
    meta = dict(
        energy_GeV=args.energy, seed=args.seed, clusters=[1, 2, 3, 4, 5], pdg=13,
        position_m=list(args.position), direction=direction.tolist(),
        requested_direction=list(args.direction),
        model_fingerprint=model.fingerprint, purpose='independent holdout, no source fitting or transport',
        source_scope='original primary-only SpectralLightElements, UID 0',
        original_elements_all_particles=len(all_source), primary_elements=len(source),
        primary_chord_length_m=float(source.length_m.sum()),
        original_first_point_m=source.start_m[0].tolist(),
        original_last_point_m=(source.start_m[-1]+source.length_m[-1]*source.direction[-1]).tolist(),
        start_ns=float(source.start_ns[0]), end_ns=float(source.end_ns[-1]),
        beta_min=float(source.beta.min()), beta_max=float(source.beta.max()),
        generated_seconds=generation_wall, generated_cpu_seconds=generation_cpu,
        source_conversion_seconds=conversion_wall, source_conversion_cpu_seconds=conversion_cpu,
        provenance=source.provenance, cone_model='spectral', packages=packages,
        g4_data_environment=g4data, original_secondary_arrays_saved=False,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    d = model.detector
    geometry_path = target.with_name(target.stem+'-detector.npz')
    np.savez_compressed(geometry_path, positions_m=d.positions_m,
                        orientations=d.orientations, effective_area_m2=d.effective_area_m2,
                        **{f'id_{key}': value for key, value in d.identifiers.items()})
    meta['detector_file'] = str(geometry_path)
    meta['detector_sha256'] = hashlib.sha256(geometry_path.read_bytes()).hexdigest()
    np.savez_compressed(target, **arrays, metadata=json.dumps(meta))
    meta['source_sha256'] = hashlib.sha256(target.read_bytes()).hexdigest()
    meta['source_file'] = str(target)
    target.with_suffix('.json').write_text(json.dumps(meta, indent=2))
    with np.load(target, allow_pickle=False) as saved:
        check = lh.SpectralLightElements(**{k: saved[k] for k in arrays}, cone_model='spectral')
        for key, value in arrays.items():
            np.testing.assert_array_equal(getattr(check, key), value)
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == '__main__':
    main()
