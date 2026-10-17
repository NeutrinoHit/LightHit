#!/usr/bin/env python3
"""Bounded real-G4 comparison of separable-point and complex-field >=2 APIs.

Both backends use identical points, L/M, real wavelength bands, actual emission
clocks and cached radial kernels. This checks implementation and local CPU cost;
it does not establish full-event speed or independent RTE accuracy.
"""
import paths  # inputs and outputs: scripts/shower_basis/paths.py
import argparse
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter, process_time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path,
                        default=paths.muon_source(1000))
    parser.add_argument('--pack', type=Path,
                        default=paths.GE2_PACK.path())
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--points', type=int, default=512)
    parser.add_argument('--oms', type=int, nargs='+', default=[197, 232])
    parser.add_argument('--band-range', type=int, nargs=2, default=[3, 6],
                        metavar=('BEGIN', 'END'))
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('fresh output directory required')
    if args.points < 2 or args.repeats < 1 or not args.oms:
        parser.error('positive repeats and at least two sample points required')
    args.output.mkdir(parents=True)
    # Set these before importing either numerical runtime. Keep new caches in
    # this benchmark's output directory while other calculations are running.
    thread_variables = ('NUMBA_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                        'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                        'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS')
    for name in thread_variables:
        os.environ[name] = '1'
    os.environ['NUMBA_CACHE_DIR'] = str((args.output / 'numba-cache').resolve())
    import numpy as np
    import numba
    import lighthit as lh
    import ntsim
    from ntsim.LightTransport import bgvd_model_from_ntsim
    from lighthit._shower_basis import _cone_harmonics, apply_multi_spectral
    from lighthit.experimental.axial_fast import _layout
    from lighthit.experimental.axial_source import AxisFrame, AxialSource
    from lighthit.experimental.point_ge2 import apply_point_spectral
    from lighthit.shower_basis import Ge2KernelStore

    numba.set_num_threads(1)
    with np.load(args.source) as data:
        source = lh.SpectralLightElements(
            **{key: data[key] for key in data.files if key != 'metadata'},
            cone_model='spectral')
        provenance = json.loads(str(data['metadata']))
    model = bgvd_model_from_ntsim(
        ntsim.bgvd.Telescope(n_clusters=(1, 2, 3, 4, 5)), ntsim.bgvd.Water())
    if provenance['model_fingerprint'] != model.fingerprint:
        raise ValueError('model/source fingerprint mismatch')
    omega = np.linspace(0., 1.2, 65)
    kernel = model.kernel(lh.KernelConfig(omega_per_ns=omega, cache_policy='require'))
    wavelengths = kernel.wavelength.wavelength_nm
    sample = kernel.medium.sample(wavelengths)
    phase = sample['phase_index']
    source = source.subset((source.uid != 0) & (source.beta * phase.max() > 1.))
    frame = AxisFrame.of(source.field(0, phase_index=float(phase.max())))
    ids = np.asarray(args.oms, np.int64)
    if np.any(ids < 0) or np.any(ids >= len(model.detector.positions_m)):
        raise ValueError('receiver index outside detector')
    positions = np.ascontiguousarray(model.detector.positions_m[ids])
    looks = np.ascontiguousarray(-model.detector.orientations[ids])
    lo, hi = args.band_range
    if not 0 <= lo < hi <= len(wavelengths):
        raise ValueError('band range outside wavelength grid')
    full_bundle = Ge2KernelStore.load(args.pack).bundle(kernel, np.arange(len(omega)))
    bundle = dict(full_bundle, coefficients=full_bundle['coefficients'][lo:hi],
                  absorption=full_bundle['absorption'][lo:hi],
                  speed=full_bundle['speed'][lo:hi])
    degree, max_m = int(bundle['degree']), 4
    starts, aa, bb, kept, channel_degree = _layout(degree, max_m)
    alpha = np.ascontiguousarray(kernel.acceptance()['alpha'], float)
    scale = np.ascontiguousarray(
        kernel.wavelength.weight_nm[lo:hi, None]
        * np.asarray(model.detector.spectral_weight(wavelengths[lo:hi]))[:, None]
        * np.asarray(model.detector.effective_area_m2)[None, ids])

    # Choose actual two-node chord quadrature points supported by the first OM.
    # Uniform ranks in emission-time order include real early and late clocks.
    gx, gw = np.polynomial.legendre.leggauss(2)
    fractions, weights = (gx + 1.) / 2., gw / 2.
    candidates, node_ids, clocks = [], [], []
    for node, fraction in enumerate(fractions):
        points = source.start_m + fraction * source.length_m[:, None] * source.direction
        radius = np.linalg.norm(points - positions[0], axis=1)
        rows = np.flatnonzero((radius >= 3.) & (radius <= 300.))
        candidates.append(rows)
        node_ids.append(np.full(len(rows), node, np.int64))
        clocks.append(source.start_ns[rows]
                      + fraction * (source.end_ns[rows] - source.start_ns[rows]))
    candidates, node_ids, clocks = map(np.concatenate, (candidates, node_ids, clocks))
    if len(candidates) < args.points:
        raise ValueError('fewer supported quadrature points than requested')
    ordered = np.argsort(clocks, kind='stable')
    chosen = ordered[np.linspace(0, len(ordered) - 1, args.points).astype(np.int64)]
    rows, nodes = candidates[chosen], node_ids[chosen]
    times = np.ascontiguousarray(clocks[chosen])
    points = np.ascontiguousarray(source.start_m[rows]
                                 + (fractions[nodes] * source.length_m[rows])[:, None]
                                 * source.direction[rows])
    vectors = np.ascontiguousarray(frame.rotate(source.direction[rows]))
    angular = np.empty((hi - lo, len(rows), len(kept)), np.float64)
    for band, lam in enumerate(range(lo, hi)):
        active = source.beta[rows] * phase[lam] > 1.
        cones = np.where(active, 1. / (source.beta[rows] * phase[lam]), 1.)
        _cone_harmonics(vectors, np.ascontiguousarray(cones), starts, aa, bb, max_m,
                        angular[band])
        photons = np.where(active, source.coefficient0[rows]
                           + source.coefficient2[rows] / phase[lam] ** 2, 0.)
        angular[band] *= (photons * weights[nodes] / wavelengths[lam] ** 2)[:, None]
    local = np.ascontiguousarray(frame.rotate(points - frame.centre_m))
    reference = float(times.min())
    temporal = np.exp(1j * (times - reference)[:, None] * omega[None, :])
    sources = [AxialSource(
        frame, local[:, 2], local[:, :2],
        np.ascontiguousarray(values[:, :, None] * temporal[:, None, :]),
        kept, channel_degree, degree, max_m, reference, {}) for values in angular]

    def point():
        return apply_point_spectral(bundle, frame, points, angular, times,
                                    positions, looks, alpha, scale,
                                    azimuthal_degree=max_m, radial_window_m=(3., 300.))

    def ordinary():
        return apply_multi_spectral(bundle, sources, positions, looks, alpha, scale,
                                    channel_dtype=np.complex128,
                                    radial_window_m=(3., 300.))

    methods = {'point': point, 'ordinary': ordinary}
    warm = {}
    for name, method in methods.items():
        cpu, wall = process_time(), perf_counter()
        method()
        warm[name] = dict(cpu_s=process_time() - cpu, wall_s=perf_counter() - wall)
        print('warm', name, warm[name], flush=True)
    measurements = {name: [] for name in methods}
    outputs = {}
    for repeat in range(args.repeats):
        order = ('point', 'ordinary') if repeat % 2 == 0 else ('ordinary', 'point')
        for name in order:
            cpu, wall = process_time(), perf_counter()
            outputs[name] = methods[name]()
            row = dict(repeat=repeat, cpu_s=process_time() - cpu,
                       wall_s=perf_counter() - wall)
            measurements[name].append(row)
            print(name, row, flush=True)
    expected, actual = outputs['ordinary'], outputs['point']
    delta = np.abs(actual - expected)
    relative = delta.max(axis=0) / np.maximum(np.abs(expected).max(axis=0), 1e-300)
    numeric = dict(max_absolute_spectral_error_pe=float(delta.max()),
                   max_relative_spectral_error=float(relative.max()),
                   relative_error_per_receiver=relative.tolist(),
                   relative_spectral_l2=float(np.linalg.norm(delta)
                                             / max(np.linalg.norm(expected), 1e-300)),
                   charge_absolute_error_pe=delta[0].tolist(),
                   ordinary_charge_pe=expected[0].real.tolist(),
                   point_charge_pe=actual[0].real.tolist(),
                   normalized_max_error_limit=1e-8,
                   passed=bool(relative.max() <= 1e-8))
    medians = {name: float(np.median([row['cpu_s'] for row in measurements[name]]))
               for name in methods}
    complex_bytes = sum(item.channels.nbytes for item in sources)
    memory = dict(point_angular_bytes=angular.nbytes,
                  shared_point_geometry_and_clocks_bytes=points.nbytes + times.nbytes,
                  ordinary_input_channels_bytes=complex_bytes,
                  ordinary_api_stacked_channels_bytes=complex_bytes,
                  source_payload_ratio=complex_bytes / angular.nbytes,
                  shared_band_kernel_logical_bytes=bundle['coefficients'].nbytes,
                  kernel_dtype=str(bundle['coefficients'].dtype),
                  note='Array sizes, not measured resident memory; the ordinary API '
                       'allocates a same-sized complex128 channel stack. The kernel '
                       'is a shared memory-mapped view; per-call scratch is excluded.')
    distances = np.linalg.norm(points[:, None, :] - positions[None, :, :], axis=2)
    code_paths = {
        'point_backend': Path(lh.__file__).parent / 'experimental/point_ge2.py',
        'ordinary_backend': Path(lh.__file__).parent / '_shower_basis.py',
        'benchmark': Path(__file__),
    }
    report = dict(source=str(args.source),
                  source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
                  model_fingerprint=model.fingerprint, kernel_pack=str(args.pack),
                  kernel_pack_manifest_sha256=hashlib.sha256(
                      (args.pack / 'pack.json').read_bytes()).hexdigest(),
                  code_sha256={name: hashlib.sha256(path.read_bytes()).hexdigest()
                               for name, path in code_paths.items()},
                  numpy_version=np.__version__, numba_version=numba.__version__,
                  numba_threads=numba.get_num_threads(),
                  thread_environment={name: os.environ[name] for name in thread_variables},
                  sample_points=len(rows), available_points=len(candidates),
                  selection='Uniform emission-time ranks of both Gauss nodes within '
                            '3–300 m of the first receiver; secondary source only',
                  receiver_indices=ids.tolist(), wavelength_indices=list(range(lo, hi)),
                  wavelength_nm=wavelengths[lo:hi].tolist(), omega_per_ns=omega.tolist(),
                  source_degree=degree, source_azimuthal_degree=max_m,
                  acceptance_degree=int(bundle['acceptance_degree']),
                  radial_window_m=[3., 300.],
                  frame='C0 at maximum phase index, entire threshold-selected secondary source',
                  emission_time_range_ns=[float(times.min()), float(times.max())],
                  emission_time_span_ns=float(np.ptp(times)),
                  radius_ranges_m=[[float(x.min()), float(x.max())] for x in distances.T],
                  supported_points_per_receiver=np.sum(
                      (distances >= 3.) & (distances <= 300.), axis=0).tolist(),
                  warmup=warm, repeats=args.repeats, measurements=measurements,
                  median_cpu_s=medians,
                  median_cpu_speedup=medians['ordinary'] / medians['point'],
                  numerical=numeric, memory=memory,
                  limitations='Shared machine with other accuracy calculations running; '
                              'CPU timings are reported separately from wall time. '
                              'Bounded sampled points, no full-event performance claim. '
                              'Identical finite L/M, radial window, complex64 cached '
                              'kernel and 65 frequencies; no independent physics validation.')
    np.savez_compressed(args.output / 'spectra.npz', **outputs, omega_per_ns=omega,
                        receivers_m=positions, looks=looks)
    np.savez_compressed(args.output / 'sample.npz', points_m=points, times_ns=times,
                        source_row_index=source.row_index[rows], source_uid=source.uid[rows],
                        quadrature_node=nodes, quadrature_fraction=fractions[nodes],
                        angular=angular, output_scale=scale)
    (args.output / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(dict(median_cpu_s=medians,
                         median_cpu_speedup=report['median_cpu_speedup'],
                         numerical=numeric, memory=memory), indent=2), flush=True)
    if not numeric['passed']:
        raise AssertionError('point/ordinary normalized spectral error exceeds 1e-8')


if __name__ == '__main__':
    main()
