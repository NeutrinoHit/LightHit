#!/usr/bin/env python3
"""Bounded real-packet joint-bin >=2 check against points AND current basis.

This is source-representation agreement with the same finite L32/M4 radial
kernel and sampled/readout spectrum. It does not certify late-time Nyquist
accuracy, full physical RTE accuracy, or a full-event speedup.
"""
import paths  # inputs and outputs: scripts/shower_basis/paths.py
import argparse, gc, hashlib, json
from dataclasses import replace
from pathlib import Path
from time import process_time, perf_counter
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--energy', type=int, choices=[300, 1000, 3000], default=1000)
    parser.add_argument('--families', type=int, default=1)
    parser.add_argument('--kinds', nargs='+', choices=['shower', 'soft', 'general', 'late'], default=['shower'])
    parser.add_argument('--ranking', choices=['visible', 'emitted'], default='visible',
                        help='visible is a positive cheap guide only, never an OM omission bound')
    parser.add_argument('--family-uid', type=int, nargs='+', help='filter the selected packets by actual family UID')
    parser.add_argument('--resolutions', nargs='+', default=['.25,.25,inf,2', '.5,.5,inf,2', '1,1,inf,2'],
                        help='z metres,tau ns,transverse metres,charged mu width')
    parser.add_argument('--scan', action='store_true', help='27 combinations z/tau/transverse; charged mu merged width2')
    parser.add_argument('--chunk', type=int, default=1024)
    parser.add_argument('--reuse-reference', type=Path, help='same complete packet/receiver/source/operator direct reference')
    parser.add_argument('--replay-measurements', type=Path,
                        help='remeasure source compiler/application on existing identical source packets; no expensive point reference')
    parser.add_argument('--compiler-baseline', type=Path, help='optional frozen previous source compiler for paired coefficient checks')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): parser.error('fresh output required')
    if args.families < 1 or args.chunk < 1: parser.error('positive families/chunk required')
    resolutions = [(z, t, x, 2.) for z in [.25, .5, 1.] for t in [.25, .5, 1.] for x in [.1, .2, np.inf]] if args.scan else [tuple(map(float, row.split(','))) for row in args.resolutions]
    if any(len(row) != 4 or min(row) <= 0 or not np.isfinite([row[0], row[1], row[3]]).all() for row in resolutions):
        parser.error('resolution requires positive z,tau,transverse,mu')
    import lighthit as lh, ntsim, numba
    from ntsim.LightTransport import bgvd_model_from_ntsim
    from lighthit.experimental.axial_source import AxisFrame, VACUUM_M_PER_NS
    from lighthit.experimental.axial_fast import _layout
    from lighthit.experimental.muon_packets import partition_muon
    from lighthit.experimental.joint_source import compile_joint_source
    from lighthit.experimental.point_ge2 import apply_point_spectral
    from lighthit._shower_basis import _cone_harmonics, compile_axial_slices, apply_multi, earliest_element_arrival
    from lighthit.shower_basis import Ge2KernelStore
    from lighthit.readout import inverse_bins
    from scipy.interpolate import CubicSpline
    if numba.get_num_threads() != 1: parser.error('one Numba thread required')
    source_path = paths.muon_source(args.energy)
    ancestry_path = paths.muon_ancestry(args.energy)
    with np.load(source_path) as data:
        source = lh.SpectralLightElements(**{k: data[k] for k in data.files if k != 'metadata'}, cone_model='spectral')
        metadata = json.loads(str(data['metadata']))
    with np.load(ancestry_path) as data:
        ancestry = {k: data[k] for k in data.files if k != 'metadata'}
    packets = partition_muon(source, ancestry)
    model = bgvd_model_from_ntsim(ntsim.bgvd.Telescope(n_clusters=(1, 2, 3, 4, 5)), ntsim.bgvd.Water())
    if metadata['model_fingerprint'] != model.fingerprint: parser.error('source/model mismatch')
    kernel = model.kernel(lh.KernelConfig(omega_per_ns=np.linspace(0, 1.2, 65), cache_policy='require'))
    wl = kernel.wavelength.wavelength_nm; wave_weights = kernel.wavelength.weight_nm
    sampled = kernel.medium.sample(wl); phase = sampled['phase_index']; omega = kernel.config.omega_per_ns
    pack_path = paths.GE2_PACK.path() / 'pack.json'
    bundle = Ge2KernelStore.load(pack_path.parent).bundle(kernel, np.arange(65))
    pack_bytes = pack_path.read_bytes(); operator_hash = hashlib.sha256(pack_bytes)
    for entry in json.loads(pack_bytes)['bundles']:
        if len(entry['key'][2]) != len(omega): continue
        with (pack_path.parent/entry['file']).resolve().open('rb') as stream:
            while block := stream.read(16*1024*1024): operator_hash.update(block)
    alpha = np.ascontiguousarray(kernel.acceptance()['alpha'], float)
    starts, aa, bb, kept, _ = _layout(32, 4)
    gx, gw = np.polynomial.legendre.leggauss(2); gx, gw = (gx+1)/2, gw/2
    outomega = np.linspace(0, 1.2, 321); edges = kernel.config.relative_time_edges_ns
    guide_phase = float(phase[len(phase)//2]); guide_wavelength = float(wl[len(wl)//2])
    attenuation = float(bundle['absorption'][len(wl)//2])
    all_positions = np.asarray(model.detector.positions_m)
    all_midpoints = source.midpoints_m
    guide_scores = {}
    for packet in packets:
        if packet.kind not in args.kinds: continue
        indices = packet.indices; weights = source.coefficient0[indices]
        centre = np.average(all_midpoints[indices], axis=0, weights=weights)
        direction = np.average(source.direction[indices], axis=0, weights=weights)
        norm = np.linalg.norm(direction); direction = direction/max(norm, 1e-30)
        displacement = all_positions-centre; radius = np.linalg.norm(displacement, axis=1)
        radius = np.maximum(radius, 3.)
        cosine = displacement @ direction/radius
        angular_guide = .2+.8*(1+np.clip(cosine, -1., 1.))/2
        emitted = np.sum(np.where(source.beta[indices]*guide_phase > 1,
                         np.maximum(0., weights+source.coefficient2[indices]/guide_phase**2), 0.))/guide_wavelength**2
        guide_scores[id(packet)] = float(np.max(emitted*np.exp(-attenuation*radius)*angular_guide/radius**2))
    chosen = []
    for kind in args.kinds:
        available = [packet for packet in packets if packet.kind == kind]
        available.sort(key=(lambda packet: source.coefficient0[packet.indices].sum())
                       if args.ranking == 'emitted' or args.replay_measurements else (lambda packet: guide_scores[id(packet)]), reverse=True)
        chosen += available[:args.families if kind == 'shower' else 1]
    if args.family_uid:
        chosen = [packet for packet in chosen if packet.family_uid in args.family_uid]
        if not chosen: parser.error('requested family is absent from selected packets')
    args.output.mkdir(parents=True)
    report = dict(energy_GeV=args.energy, threads=1, clusters=5, wavelength_nm=wl.tolist(),
                  model_fingerprint=model.fingerprint, source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
                  ancestry_sha256=hashlib.sha256(ancestry_path.read_bytes()).hexdigest(),
                  operator_sha256=operator_hash.hexdigest(),
                  source_l=32, source_m=4, radial_window_m=[3., 300.], chord_order=2,
                  scope='selected real packets only; all nine wavelengths; >=2; shared frequency/readout approximations',
                  source_frequency_per_ns=omega.tolist(), readout_frequency_per_ns=outomega.tolist(),
                  relative_time_edges_ns=edges.tolist(),
                  complete=False, packets=[])
    report.update(packet_ranking='emitted historical replay' if args.replay_measurements else args.ranking,
                  ranking_guide='mid-band photon yield * exp(-absorption*r)/r^2 * positive forward guide; not a physical bound')
    frozen = None
    if args.compiler_baseline:
        import importlib.util, sys
        spec = importlib.util.spec_from_file_location('lighthit.experimental._joint_source_frozen', args.compiler_baseline)
        frozen = importlib.util.module_from_spec(spec); sys.modules[spec.name] = frozen; spec.loader.exec_module(frozen)
        report['frozen_compiler_sha256'] = hashlib.sha256(args.compiler_baseline.read_bytes()).hexdigest()
    previous = None
    if args.reuse_reference:
        previous = json.loads((args.reuse_reference/'report.json').read_text())
        for key in ['energy_GeV', 'model_fingerprint', 'source_sha256', 'ancestry_sha256', 'source_l', 'source_m',
                    'radial_window_m', 'chord_order', 'wavelength_nm', 'source_frequency_per_ns', 'readout_frequency_per_ns',
                    'operator_sha256', 'relative_time_edges_ns']:
            if previous.get(key) != report[key]: parser.error(f'reused reference differs: {key}')
        if not previous.get('complete'): parser.error('reused reference incomplete')
        report['direct_reused_from'] = str(args.reuse_reference)
    replay = None
    if args.replay_measurements:
        replay = json.loads((args.replay_measurements/'report.json').read_text())
        for key in ['energy_GeV', 'model_fingerprint', 'source_sha256', 'ancestry_sha256', 'source_l', 'source_m',
                    'radial_window_m', 'chord_order', 'wavelength_nm', 'source_frequency_per_ns', 'readout_frequency_per_ns']:
            if replay.get(key) != report[key]: parser.error(f'replayed source differs: {key}')
        if not replay.get('complete'): parser.error('replayed measurement incomplete')
        report.update(replayed_from=str(args.replay_measurements),
                      accuracy_provenance='historical shared-operator point reference; current frozen-source spectrum replay tested per case',
                      original_reference_coefficient_hash_recorded='operator_sha256' in replay)
    def save(): (args.output/'report.json').write_text(json.dumps(report, indent=2, default=str))
    for packet in chosen:
        part = packet.source(source); part = part.subset(part.beta*phase.max() > 1)
        print('START', args.energy, packet.kind, packet.family_uid, 'rows', len(part), flush=True)
        geometry = part.field(0, phase_index=float(phase.max())); frame = AxisFrame.of(geometry)
        near = np.argsort(np.linalg.norm(model.detector.positions_m-part.centroid_m, axis=1))[:2]
        ids = np.array(list(dict.fromkeys([197, 232, 342, 666, 1062, 1241]+near.tolist())), np.int64)
        positions = np.ascontiguousarray(model.detector.positions_m[ids]); looks = np.ascontiguousarray(-model.detector.orientations[ids])
        areas = np.asarray(model.detector.effective_area_m2)[ids]
        scales = wave_weights[:, None]*np.asarray(model.detector.spectral_weight(wl))[:, None]*areas[None, :]
        origins = earliest_element_arrival(positions, part.start_m, part.direction, part.length_m, part.start_ns, float(sampled['group_index'].min()))
        def bins_of(spectrum):
            relative = spectrum*np.exp(-1j*omega[:, None]*origins[None, :])
            return inverse_bins(outomega, CubicSpline(omega, relative, axis=0)(outomega), edges)
        def band(lam):
            return dict(bundle, coefficients=bundle['coefficients'][lam:lam+1], absorption=bundle['absorption'][lam:lam+1], speed=bundle['speed'][lam:lam+1])
        # Both backends are warmed before CPU measurement.
        tiny = compile_joint_source(part.subset(np.arange(len(part)) == 0), wl, phase, frame=frame)
        apply_point_spectral(bundle, frame, tiny.points_m, tiny.angular, tiny.times_ns, positions[:1], looks[:1], alpha, scales[:, :1])
        options = dict(degree=32, azimuthal_degree=4, cell_m=.5, axial_order=3, transverse='stencil',
                       stencil_m=.2, stencil_threshold_m=.03, frame=frame, time_step_ns=.1, time_deposit='cubic',
                       channel_dtype=np.complex64, stencil_center='centroid', geometry_reference=geometry,
                       axial_clock_ns_per_m=1/VACUUM_M_PER_NS)
        warm = compile_axial_slices(geometry, omega_per_ns=omega, **options)
        apply_multi(band(0), warm, warm, positions[:1], looks[:1], alpha, np.ones(1), np.zeros(1), scales[:1, :1], radial_window_m=(3., 300.))
        del warm, tiny; gc.collect()
        row = dict(kind=packet.kind, family_uid=packet.family_uid, root_energy_GeV=packet.root_energy_GeV,
                   source_elements=len(part), receiver_indices=ids.tolist(), cases={}, visible_guide_score=guide_scores[id(packet)])
        prefix = f'{packet.kind}-{packet.family_uid if packet.family_uid is not None else "soft"}'
        if replay is not None:
            history = next((old for old in replay['packets'] if old['kind'] == packet.kind and old['family_uid'] == packet.family_uid), None)
            if history is None or history['source_elements'] != len(part) or history['receiver_indices'] != ids.tolist():
                parser.error('historical packet selection differs')
            with np.load(args.replay_measurements/f'{prefix}-reference.npz') as data:
                np.testing.assert_array_equal(origins, data['time_origin_ns'])
                direct = data['direct'].copy()
            row.update(direct=history['direct'], current_axial_basis=history['current_axial_basis'],
                       timing_scope='new histogram compiler+application measured; historical current-basis timings retained')
            for z_step, tau_step, transverse_step, mu_step in resolutions:
                name = f'z{z_step:g}-t{tau_step:g}-x{transverse_step:g}-mu{mu_step:g}'
                if name not in history['cases']: parser.error('replayed histogram resolution is missing')
                opts = dict(frame=frame, z_step_m=z_step, tau_step_ns=tau_step,
                            transverse_step_m=transverse_step, mu_step=mu_step)
                tiny_part = part.subset(np.arange(len(part)) == 0)
                compile_joint_source(tiny_part, wl, phase, **opts)
                if frozen is not None: frozen.compile_joint_source(tiny_part, wl, phase, **opts)
                comparisons = {}; old_compile_cpu = None
                if frozen is not None:
                    cpu = process_time(); old_hist = frozen.compile_joint_source(part, wl, phase, **opts); old_compile_cpu = process_time()-cpu
                cpu = process_time(); hist = compile_joint_source(part, wl, phase, **opts); compile_cpu = process_time()-cpu
                if frozen is not None:
                    for attribute in ['keys', 'points_m', 'times_ns', 'angular', 'photons_per_nm']:
                        comparisons[attribute+'_bitwise_equal'] = bool(np.array_equal(getattr(hist, attribute), getattr(old_hist, attribute)))
                        if not comparisons[attribute+'_bitwise_equal']: parser.error(f'compiler optimisation changed {attribute}')
                    del old_hist
                cpu = process_time(); spectrum = apply_point_spectral(bundle, frame, hist.points_m, hist.angular, hist.times_ns, positions, looks, alpha, scales); apply_cpu = process_time()-cpu
                with np.load(args.replay_measurements/f'{prefix}-{name}.npz') as data: previous_spectrum = data['spectrum']
                replay_error = float(np.max(abs(spectrum-previous_spectrum))/max(float(np.max(abs(previous_spectrum))), 1e-30))
                if replay_error > 1e-11: parser.error('current source/kernel replay differs from the historical spectrum')
                total = compile_cpu+apply_cpu
                case = dict(history['cases'][name], compile_cpu_s=compile_cpu, apply_cpu_s=apply_cpu,
                            total_cpu_s=total, current_basis_speedup=row['current_axial_basis']['cpu_s']/max(total, 1e-30),
                            previous_compile_cpu_s=old_compile_cpu, compiler_cpu_speedup=old_compile_cpu/compile_cpu if old_compile_cpu else None,
                            source_arrays_comparison=comparisons, historical_spectrum_normalised_difference=replay_error,
                            histogram_total_speedup=history['cases'][name]['total_cpu_s']/max(total, 1e-30))
                row['cases'][name] = case
                np.savez_compressed(args.output/f'{prefix}-{name}.npz', spectrum=spectrum)
                print('REPLAY', args.energy, packet.kind, packet.family_uid, name, 'compiler',compile_cpu,
                      'apply',apply_cpu,'source equality',comparisons,'current speedup',case['current_basis_speedup'],flush=True)
                del hist; gc.collect()
            report['packets'].append(row); save(); continue
        direct = np.zeros((65, len(ids)), complex); total_points = 0
        reused = None
        if previous is not None:
            reused = next((old for old in previous['packets'] if old['kind'] == packet.kind and old['family_uid'] == packet.family_uid), None)
            if reused is not None and (reused['receiver_indices'] != ids.tolist() or reused['source_elements'] != len(part)):
                parser.error('reused packet or receiver set differs')
            if reused is not None:
                with np.load(args.reuse_reference/f'{prefix}-reference.npz') as data:
                    np.testing.assert_allclose(data['time_origin_ns'], origins, rtol=0, atol=0)
                    direct = data['direct'].copy()
        cpu, wall = process_time(), perf_counter()
        for fraction, weight in ([] if reused is not None else zip(gx, gw)):
            points = part.start_m+fraction*part.length_m[:, None]*part.direction
            reachable = np.zeros(len(part), bool)
            for receiver in positions:
                d2 = np.sum((points-receiver)**2, axis=1); reachable |= (d2 >= 9)&(d2 <= 90000)
            chosen_rows = np.flatnonzero(reachable)
            for start in range(0, len(chosen_rows), args.chunk):
                take = chosen_rows[start:start+args.chunk]
                vectors = np.ascontiguousarray(frame.rotate(part.direction[take])); angular = np.empty((len(wl), len(take), len(kept)))
                for lam in range(len(wl)):
                    active = part.beta[take]*phase[lam] > 1
                    cones = np.where(active, 1/(part.beta[take]*phase[lam]), 1.)
                    _cone_harmonics(vectors, np.ascontiguousarray(cones), starts, aa, bb, 4, angular[lam])
                    mass = np.where(active, np.maximum(0., part.coefficient0[take]+part.coefficient2[take]/phase[lam]**2), 0.)*weight/wl[lam]**2
                    angular[lam] *= mass[:, None]
                clocks = part.start_ns[take]+fraction*(part.end_ns[take]-part.start_ns[take])
                direct += apply_point_spectral(bundle, frame, points[take], angular, clocks, positions, looks, alpha, scales)
                total_points += len(take)
        row['direct'] = (dict(reused['direct'], reused=True) if reused is not None
                         else dict(cpu_s=process_time()-cpu, wall_s=perf_counter()-wall, points=total_points))
        reference_bins = bins_of(direct); reference_l1 = abs(reference_bins).sum(1)
        bright = reference_l1 >= max(1e-4, float(reference_l1.max())*1e-3)
        def metric(spectrum):
            bins = bins_of(spectrum); err = abs(bins-reference_bins).sum(1)
            relative = err/np.maximum(reference_l1, 1e-30)
            return dict(charge_pe=spectrum[0].real.tolist(), charge_relative_error=(abs(spectrum[0]-direct[0])/np.maximum(abs(direct[0]), 1e-30)).tolist(),
                        charge_absolute_error_pe=abs(spectrum[0]-direct[0]).tolist(), bins_relative_l1=relative.tolist(),
                        bins_absolute_l1_pe=err.tolist(), bright_om_indices=ids[bright].tolist(),
                        bright_control_count=int(bright.sum()),
                        bright_control_available=bool(bright.any()),
                        max_bright_bins_relative_l1=float(relative[bright].max()) if bright.any() else 0.,
                        max_weak_bins_absolute_l1_pe=float(err[~bright].max()) if (~bright).any() else 0.)
        row['direct'].update(charge_pe=direct[0].real.tolist(), bins_l1_pe=reference_l1.tolist())
        baseline = np.zeros_like(direct); measurements=[]
        for lam in range(len(wl)):
            active = part.beta*phase[lam] > 1
            if not active.any(): continue
            s = part.subset(active); f0, f2 = s.field(0, phase_index=float(phase[lam])), s.field(2, phase_index=float(phase[lam]))
            field = replace(f0, photons=np.ascontiguousarray(np.maximum(0., f0.photons-f2.photons/phase[lam]**2)/wl[lam]**2))
            cpu = process_time(); compiled = compile_axial_slices(field, omega_per_ns=omega, **options); compile_cpu = process_time()-cpu
            cpu = process_time(); baseline += apply_multi(band(lam), compiled, compiled, positions, looks, alpha, np.ones(1), np.zeros(1), scales[lam:lam+1], radial_window_m=(3., 300.)); apply_cpu = process_time()-cpu
            measurements.append(dict(band=lam, compile_cpu_s=compile_cpu, apply_cpu_s=apply_cpu, points=len(compiled.z_m), payload_bytes=compiled.channels.nbytes))
            del compiled; gc.collect()
        baseline_cpu = sum(x['compile_cpu_s']+x['apply_cpu_s'] for x in measurements)
        row['current_axial_basis'] = dict(cpu_s=baseline_cpu, measurements=measurements, **metric(baseline))
        report['packets'].append(row); save()
        np.savez_compressed(args.output/f'{prefix}-reference.npz', direct=direct, baseline=baseline, time_origin_ns=origins, receiver_indices=ids)
        print('REFERENCE', args.energy, packet.kind, packet.family_uid, row['direct'], 'baseline CPU', baseline_cpu, flush=True)
        for z_step, tau_step, transverse_step, mu_step in resolutions:
            name = f'z{z_step:g}-t{tau_step:g}-x{transverse_step:g}-mu{mu_step:g}'
            try:
                cpu = process_time(); hist = compile_joint_source(part, wl, phase, frame=frame, z_step_m=z_step,
                    tau_step_ns=tau_step, transverse_step_m=transverse_step, mu_step=mu_step); compile_cpu = process_time()-cpu
                cpu = process_time(); spectrum = apply_point_spectral(bundle, frame, hist.points_m, hist.angular, hist.times_ns, positions, looks, alpha, scales); apply_cpu = process_time()-cpu
                total = compile_cpu+apply_cpu
                case = dict(compile_cpu_s=compile_cpu, apply_cpu_s=apply_cpu, total_cpu_s=total,
                            direct_speedup=row['direct']['cpu_s']/max(total, 1e-30), current_basis_speedup=baseline_cpu/max(total, 1e-30),
                            summary=hist.summary, **metric(spectrum))
                row['cases'][name] = case
                np.savez_compressed(args.output/f'{prefix}-{name}.npz', spectrum=spectrum)
                print('CASE', args.energy, packet.kind, packet.family_uid, name, 'cells',len(hist.keys),'CPU',total,
                      'current speedup',case['current_basis_speedup'],'brightL1',case['max_bright_bins_relative_l1'],flush=True)
                del hist; gc.collect()
            except ValueError as error:
                row['cases'][name] = dict(error=str(error)); print('SKIP', name, str(error), flush=True)
            save()
    report['complete'] = True; save()


if __name__ == '__main__': main()
