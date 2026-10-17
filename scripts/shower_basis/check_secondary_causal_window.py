#!/usr/bin/env python3
"""Measure >=2 stationary light emitted too late for the current readout window.

This is a diagnostic of physical causality versus frequency sampling. It does
not change transport, remove delayed particles or validate the RTE kernel.
"""
import paths  # inputs and outputs: scripts/shower_basis/paths.py
import argparse
import hashlib
import json
from pathlib import Path
from time import process_time

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--oms', type=int, nargs='+', default=[197, 1241])
    parser.add_argument('--chord-order', type=int, choices=[2,4,8],
                        help='explicit provenance for older reference reports')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('fresh output directory required')
    import lighthit as lh
    import numba
    import ntsim
    from ntsim.LightTransport import bgvd_model_from_ntsim
    from lighthit._shower_basis import _cone_harmonics
    from lighthit.experimental.axial_fast import _layout
    from lighthit.experimental.axial_source import AxisFrame
    from lighthit.experimental.point_ge2 import apply_point_spectral
    from lighthit.shower_basis import Ge2KernelStore

    if numba.get_num_threads() != 1:
        parser.error('one thread required')
    reference = json.loads((args.reference / 'report.json').read_text())
    if not reference.get('direct_complete'):
        parser.error('complete direct reference required')
    chord_order = reference.get('chord_gauss_order', args.chord_order)
    if chord_order is None:
        parser.error('older reference requires explicit --chord-order')
    if args.chord_order is not None and chord_order != args.chord_order:
        parser.error('chord order differs from reference')
    path = Path(reference['source'])
    if hashlib.sha256(path.read_bytes()).hexdigest() != reference['source_sha256']:
        parser.error('source changed')
    with np.load(path) as data:
        source = lh.SpectralLightElements(
            **{key: data[key] for key in data.files if key != 'metadata'},
            cone_model='spectral')
    model = bgvd_model_from_ntsim(
        ntsim.bgvd.Telescope(n_clusters=(1, 2, 3, 4, 5)), ntsim.bgvd.Water())
    if model.fingerprint != reference['model_fingerprint']:
        parser.error('model changed')
    kernel = model.kernel(lh.KernelConfig(
        omega_per_ns=np.linspace(0, 1.2, 65), cache_policy='require'))
    wavelengths = kernel.wavelength.wavelength_nm
    sample = kernel.medium.sample(wavelengths)
    phase = sample['phase_index']
    source = source.subset((source.uid != 0) & (source.beta * phase.max() > 1))
    frame = AxisFrame.of(source.field(0, phase_index=float(phase.max())))
    bundle = Ge2KernelStore.load(str(paths.GE2_PACK.path())).bundle(
        kernel, np.arange(65))
    bundle = dict(bundle, omega=bundle['omega'][:1],
                  coefficients=bundle['coefficients'][..., :1])
    degree, max_m = 32, 4
    starts, aa, bb, kept, _ = _layout(degree, max_m)
    alpha = np.ascontiguousarray(kernel.acceptance()['alpha'], float)
    ids = np.asarray(args.oms, np.int64)
    columns = [reference['receiver_indices'].index(int(om)) for om in ids]
    positions = np.ascontiguousarray(model.detector.positions_m[ids])
    looks = np.ascontiguousarray(-model.detector.orientations[ids])
    scales = (kernel.wavelength.weight_nm[:, None]
              * np.asarray(model.detector.spectral_weight(wavelengths))[:, None]
              * np.asarray(model.detector.effective_area_m2)[None, ids])
    with np.load(args.reference / 'direct.npz') as data:
        origins = data['time_origin_ns'][columns]
        total_charge = data['spectrum'][0, columns].real
    edges = kernel.config.relative_time_edges_ns
    late_charge = np.zeros(len(ids))
    late_counts = np.zeros((len(ids), len(wavelengths)), np.int64)
    cpu = process_time()
    gx, gw = np.polynomial.legendre.leggauss(chord_order)
    for fraction, weight in zip((gx + 1) / 2, gw / 2):
        points = source.start_m + fraction * source.length_m[:, None] * source.direction
        times = source.start_ns + fraction * (source.end_ns - source.start_ns)
        for receiver, position in enumerate(positions):
            radii = np.linalg.norm(points - position, axis=1)
            # No >=2 trajectory can arrive before the straight photon flight.
            # This selects only provably late contributions, not all late light.
            late = (times[None, :] + radii[None, :] / bundle['speed'][:, None]
                    > origins[receiver] + edges[-1])
            late &= (source.beta[None, :] * phase[:, None] > 1)
            late &= (radii[None, :] >= 3) & (radii[None, :] <= 300)
            late_counts[receiver] += late.sum(axis=1)
            selected = np.flatnonzero(late.any(axis=0))
            for begin in range(0, len(selected), 1024):
                take = selected[begin:begin + 1024]
                vectors = np.ascontiguousarray(frame.rotate(source.direction[take]))
                angular = np.empty((len(wavelengths), len(take), len(kept)))
                for lam, wavelength in enumerate(wavelengths):
                    active = late[lam, take]
                    cones = np.where(active, 1 / (source.beta[take] * phase[lam]), 1.)
                    _cone_harmonics(vectors, np.ascontiguousarray(cones), starts,
                                    aa, bb, max_m, angular[lam])
                    photons = np.where(active, source.coefficient0[take]
                                       + source.coefficient2[take] / phase[lam]**2, 0.)
                    angular[lam] *= (photons * weight / wavelength**2)[:, None]
                late_charge[receiver] += apply_point_spectral(
                    bundle, frame, points[take], angular, times[take],
                    positions[receiver:receiver + 1], looks[receiver:receiver + 1],
                    alpha, scales[:, receiver:receiver + 1])[0, 0].real
    report = dict(energy_GeV=reference['energy_GeV'], source_sha256=reference['source_sha256'],
                  receiver_indices=ids.tolist(), threads=1, chord_gauss_order=chord_order,
                  readout_edges_ns=edges.tolist(),
                  all_time_ge2_charge_pe=total_charge.tolist(),
                  provably_after_window_ge2_charge_pe=late_charge.tolist(),
                  late_fraction=(late_charge / np.maximum(abs(total_charge), 1e-30)).tolist(),
                  late_chord_nodes_per_band=late_counts.tolist(), cpu_s=process_time() - cpu,
                  scope='stationary same-RTE charge of causally late source; not an alias estimate or waveform correction')
    args.output.mkdir(parents=True)
    (args.output / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
