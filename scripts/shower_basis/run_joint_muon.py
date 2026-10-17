#!/usr/bin/env python3
"""One-core complete muon event: independent screening and all collision orders.

The joint route uses reusable optical atlases and numerical local sources. The
reference route reruns the current original-source smart transport. Previous
fine results can be read for comparison only after the timed calculation ends.
Saved G4 production and optical cache construction are excluded from timing.
"""
import paths  # inputs and outputs: scripts/shower_basis/paths.py
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import resource
import sys
from time import perf_counter, process_time
from types import SimpleNamespace

import numpy as np


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(16 * 2**20), b''):
            checksum.update(block)
    return checksum.hexdigest()


def compare_response(response, reference_path, global_ids):
    """Diagnostic agreement with another calculation, never physical truth."""
    with np.load(reference_path, allow_pickle=False) as archive:
        saved_ids = archive['receiver_indices'] if 'receiver_indices' in archive else np.arange(len(archive['active']))
        lookup = {int(key): row for row, key in enumerate(saved_ids)}
        try:
            take = np.array([lookup[int(key)] for key in global_ids], np.int64)
        except KeyError as error:
            raise ValueError('comparison does not contain every requested receiver') from error
        charge = archive['charge_components_pe'][take]
        bins = archive['components_pe'][take]
        active = archive['active'][take]
        origins = archive['time_origin_ns'][take]
        edges = archive['relative_time_edges_ns']
        metadata = json.loads(str(archive['metadata'])) if 'metadata' in archive else {}
    if not np.allclose(origins, response.time_origin_ns, rtol=0, atol=1e-9):
        raise ValueError('comparison clocks differ; bin differences would be misleading')
    if not np.array_equal(edges, response.relative_time_edges_ns):
        raise ValueError('comparison time-bin edges differ')
    candidate_fine = np.asarray(response.metadata.get('fine_charge_mask', np.ones(len(global_ids))), bool)
    old_fine = np.asarray(metadata.get('fine_charge_mask', np.ones(len(saved_ids))), bool)[take]
    compared = candidate_fine & old_fine
    bright = compared & (active | response.active)
    result = dict(reference=str(reference_path), fine_reference_loaded_after_transport=True,
                  reference_is_physical_truth=False, fine_common_receivers=global_ids[compared].tolist(),
                  lost_active=global_ids[active & ~response.active].tolist(),
                  gained_active=global_ids[~active & response.active].tolist(), orders={})
    for label, columns in [('0', [0]), ('1', [1]), ('ge2', [2]), ('total', [0, 1, 2])]:
        oldq = charge[:, columns].sum(1)
        newq = response.charge_components_pe[:, columns].sum(1)
        oldb = bins[:, :, columns].sum(2)
        newb = response.components_pe[:, :, columns].sum(2)
        absolute = np.abs(newb-oldb).sum(1)
        l1 = np.abs(oldb).sum(1)
        result['orders'][label] = dict(
            active_common_count=int(bright.sum()),
            max_charge_absolute_error_pe=float(np.max(np.abs(newq[bright]-oldq[bright]), initial=0)),
            max_charge_relative_error=float(np.max(np.abs(newq[bright]-oldq[bright])/np.maximum(np.abs(oldq[bright]), 1e-30), initial=0)),
            active_common_bins_relative_l1=float(absolute[bright].sum()/max(float(l1[bright].sum()), 1e-30)),
            per_receiver_charge_absolute_error_pe=np.abs(newq-oldq).tolist(),
            per_receiver_bins_absolute_l1_pe=absolute.tolist(),
            per_receiver_bins_relative_l1=(absolute/np.maximum(l1, 1e-30)).tolist())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--energy', type=int, choices=(300, 1000, 3000), default=1000)
    parser.add_argument('--mode', choices=('joint', 'reference'), default='joint')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--receiver-indices', type=int, nargs='+', help='Global OM indices; bounded debug only')
    parser.add_argument('--source', type=Path)
    parser.add_argument('--ancestry', type=Path)
    parser.add_argument('--q1-atlas', type=Path, default=paths.Q1_ATLAS.path())
    parser.add_argument('--ge2-atlas', type=Path, default=paths.GE2_ATLAS.path())
    parser.add_argument('--q1-beta', type=Path, default=paths.Q1_BETA.path())
    parser.add_argument('--ge2-beta', type=Path, default=paths.GE2_BETA.path())
    parser.add_argument('--ge2-pack', type=Path, default=paths.GE2_PACK.path())
    parser.add_argument('--first-cdf-atlas', type=Path, default=paths.work('joint-first-20261011/atlas-test/production-grid-atlas.npz'))
    parser.add_argument('--preview-atlas', type=Path, default=paths.PREVIEW_ATLAS.path())
    parser.add_argument('--ge2-backend', choices=('basis', 'joint'), default='basis')
    parser.add_argument('--ge2-operator',choices=('scalar','blas'),default='scalar')
    parser.add_argument('--first-backend', choices=('atlas', 'grouped'), default='atlas')
    parser.add_argument('--first-skip-epsilon', type=float, default=0.,
                        help='grouped secondary Q1: certified omitted-charge fraction per band and OM')
    parser.add_argument('--first-skip-seed-fraction', type=float, default=.5)
    parser.add_argument('--first-fuse-bands', action='store_true',
                        help='grouped secondary Q1: one fused all-band unit with finer direction pixels')
    parser.add_argument('--ge2-skip-epsilon', type=float, default=0.,
                        help='secondary >=2: predicted omitted-charge fraction per OM (not certified)')
    parser.add_argument('--ge2-skip-safety', type=float, default=3.)
    parser.add_argument('--ge2-batch-points', type=int, default=0,
                        help='secondary >=2: apply packets together in batches of this many points (0: per packet)')
    parser.add_argument('--ge2-blas-precision', choices=('float64', 'float32'), default='float64')
    parser.add_argument('--ge2-z-step-m', type=float, default=.25)
    parser.add_argument('--ge2-tau-step-ns', type=float, default=.5)
    parser.add_argument('--ge2-transverse-step-m', type=float, default=.1)
    parser.add_argument('--reference-ballistic-method',choices=('interval','quadrature'),default='interval')
    parser.add_argument('--ring-nodes', type=int, default=96)
    parser.add_argument('--first-chunk-elements', type=int, default=4096)
    parser.add_argument('--threshold-pe', type=float, default=.01)
    parser.add_argument('--audit-all-rejected', action='store_true')
    parser.add_argument('--compare', type=Path, help='Read this result only after the event calculation completes')
    parser.add_argument('--viewer', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('choose a fresh output directory')
    if not np.isfinite(args.threshold_pe) or args.threshold_pe < 0:
        parser.error('threshold must be finite and nonnegative')
    import numba
    import lighthit as lh
    import ntsim
    from ntsim.LightTransport import bgvd_model_from_ntsim
    from lighthit.experimental.positive_fluence import PositiveFluenceAtlas
    from lighthit.experimental.adaptive_screen import AdaptiveScreenConfig, transport_smart_muon
    from lighthit.experimental.joint_muon import JointMuonConfig, transport_joint_muon
    from lighthit.experimental.joint_first_atlas import UniversalFirstCDFAtlas, evaluate_photon_histogram_atlas
    from lighthit.experimental.joint_source import compile_photon_histogram
    from lighthit.experimental.track_first_primitive import TrackFirstAtlasStore
    from lighthit.experimental.track_ge2_primitive import DiskTrackGe2Atlas
    from lighthit.experimental.first_charge_beta_atlas import FirstChargeBetaAtlas
    from lighthit.experimental.ge2_charge_beta_atlas import Ge2ChargeBetaAtlas
    from lighthit.experimental.universal_primary import subset_kernel
    from lighthit.ballistic_interval import ballistic_interval
    from probe_positive_map_screen import group_elements, query_groups, preview_ballistic
    from run_positive_map_fine import acceptance_maximum
    if numba.get_num_threads() != 1:
        parser.error('NUMBA_NUM_THREADS=1 required for comparable timings')
    if any(__import__('os').environ.get(name) != '1' for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS')):
        parser.error('OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=1 required')
    source_path = args.source or paths.muon_source(args.energy)
    ancestry_path = args.ancestry or paths.muon_ancestry(args.energy)
    with np.load(source_path, allow_pickle=False) as archive:
        source = lh.SpectralLightElements(**{key: archive[key] for key in archive.files if key != 'metadata'}, cone_model='spectral')
        source_metadata = json.loads(str(archive['metadata']))
    with np.load(ancestry_path, allow_pickle=False) as archive:
        ancestry = {key: archive[key].copy() for key in archive.files if key != 'metadata'}
    model = bgvd_model_from_ntsim(ntsim.bgvd.Telescope(n_clusters=(1, 2, 3, 4, 5)), ntsim.bgvd.Water())
    atlas = PositiveFluenceAtlas.load(args.preview_atlas)
    if source_metadata['model_fingerprint'] != model.fingerprint or atlas.metadata['model_fingerprint'] != model.fingerprint:
        raise ValueError('source/preview-atlas/five-cluster model fingerprints differ')
    complete_model_fingerprint = model.fingerprint
    global_ids = np.arange(len(model.detector.positions_m))
    if args.receiver_indices is not None:
        ids = np.asarray(args.receiver_indices, np.int64)
        if len(np.unique(ids)) != len(ids) or np.any(ids < 0) or np.any(ids >= len(global_ids)):
            parser.error('receiver indices must be unique global OM indices')
        global_ids = ids
        detector = model.detector
        model = replace(model, detector=replace(detector, positions_m=detector.positions_m[ids],
            orientations=detector.orientations[ids], effective_area_m2=detector.effective_area_m2[ids],
            identifiers={key: value[ids] for key, value in detector.identifiers.items()}))
    kernel = model.kernel(lh.KernelConfig(omega_per_ns=np.linspace(0, 1.2, 65),
        readout_omega_per_ns=np.linspace(0, 1.2, 321), threshold_pe=args.threshold_pe,
        cache_policy='require', cache_directory=str(paths.RTE_CACHE.path())))
    ge2_store = kernel.ge2_kernel_store(args.ge2_pack)
    load_start = perf_counter()
    first = TrackFirstAtlasStore(args.q1_atlas)
    ge2 = DiskTrackGe2Atlas(args.ge2_atlas)
    first_beta = FirstChargeBetaAtlas(args.q1_beta, first)
    ge2_beta = Ge2ChargeBetaAtlas(args.ge2_beta, ge2)
    first_cdf = UniversalFirstCDFAtlas.load(args.first_cdf_atlas)
    cache_load_seconds = perf_counter()-load_start
    kernel.acceptance()
    input_paths = [source_path, ancestry_path, args.preview_atlas, args.first_cdf_atlas,
                   args.q1_atlas/'manifest.json', args.ge2_atlas/'manifest.json',
                   args.q1_beta, args.ge2_beta, args.ge2_pack/'pack.json']
    manifest = json.loads((args.ge2_pack/'pack.json').read_text())
    input_paths.extend((args.ge2_pack / bundle['file']).resolve() for bundle in manifest['bundles'])
    provenance = {str(path): digest(path) for path in dict.fromkeys(input_paths)}
    # Warm tiny physical queries only, without computing any previous event response.
    warm_start = perf_counter()
    tiny = source.subset(np.arange(len(source)) == int(np.flatnonzero(source.uid == 0)[0]))
    local = subset_kernel(kernel, [0])
    ballistic_interval(local, tiny, time_origin_ns=[0.], relative_edges_ns=kernel.config.relative_time_edges_ns)
    phase = float(kernel.medium.sample(np.array([480.]))['phase_index'][0])
    histogram = compile_photon_histogram(tiny, 480., phase, ring_nodes=args.ring_nodes, retain_samples=True)
    evaluate_photon_histogram_atlas(first_cdf, local, histogram, np.array([0]),
        kernel.config.relative_time_edges_ns, np.array([0.]))
    from lighthit.experimental.filon_ge2 import warm_direct_rte
    warm_direct_rte()
    from lighthit.experimental.first_charge_only import first_scattering_charge_only
    from lighthit.experimental.muon_integral import order1_integrals
    warm_prompt_kernel=subset_kernel(kernel,[0])
    warm_prompt_kernel.detector=replace(warm_prompt_kernel.detector,
        positions_m=tiny.start_m[:1]+np.array([[20.,0.,30.]]))
    first_scattering_charge_only(warm_prompt_kernel,tiny,gauss_order=6)
    order1_integrals(warm_prompt_kernel,tiny,time_origin_ns=np.array([0.]),
        config=lh.PromptConfig(element_method='segment',screen=False))
    if args.mode=='joint' and args.first_backend=='grouped':
        from lighthit.experimental.secondary_first import evaluate_secondary_first
        from lighthit.experimental.muon_packets import SourcePacket
        warm_source=replace(tiny,uid=np.ones(len(tiny),np.int64))
        warm_kernel=subset_kernel(kernel,[0])
        warm_kernel.detector=replace(warm_kernel.detector,
            positions_m=warm_source.start_m[:1]+np.array([[20.,0.,30.]]))
        evaluate_secondary_first(warm_kernel,warm_source,
            [SourcePacket('soft',np.arange(len(warm_source)))],np.array([0]),
            time_origin_ns=np.array([0.]))
        if args.first_skip_epsilon or args.first_fuse_bands:
            evaluate_secondary_first(warm_kernel,warm_source,
                [SourcePacket('soft',np.arange(len(warm_source)))],np.array([0]),
                time_origin_ns=np.array([0.]),skip_epsilon=args.first_skip_epsilon,
                skip_seed_fraction=args.first_skip_seed_fraction,
                fuse_bands=args.first_fuse_bands)
    if args.mode=='joint' and args.ge2_backend=='joint':
        from lighthit.experimental.joint_source import compile_joint_source
        from lighthit.experimental.point_ge2 import apply_point_spectral
        sample=kernel.medium.sample(kernel.wavelength.wavelength_nm)
        bundle=ge2_store.bundle(kernel,np.arange(len(kernel.config.omega_per_ns)))
        joint=compile_joint_source(tiny,kernel.wavelength.wavelength_nm,sample['phase_index'],
            degree=int(bundle['degree']),azimuthal_degree=kernel.config.azimuthal_degree)
        # One tiny source and an explicitly dark scale specialize the same
        # mapped coefficient signature without computing an event response.
        apply_point_spectral(bundle,joint.frame,joint.points_m,joint.angular,joint.times_ns,
            kernel.detector.positions_m[:1],-kernel.detector.orientations[:1],
            kernel.acceptance()['alpha'],np.zeros((len(kernel.wavelength.wavelength_nm),1)),
            azimuthal_degree=kernel.config.azimuthal_degree,radial_window_m=kernel.config.radial_range_m,
            backend=args.ge2_operator)
    first.query_charge_many([float(first.radii_m[len(first.radii_m)//2])], [[0., 0., 1.]], [0.], [1.], coefficient0_per_m=[1.])
    ge2.segment_charge_many([float(ge2.impact_nodes_m[len(ge2.impact_nodes_m)//2])], [[0., 0., 1.]], [1.], [0.], coefficient0_per_m=[1.])
    warm_seconds = perf_counter()-warm_start
    args.output.mkdir(parents=True)
    progress_path = args.output/'progress.jsonl'
    def progress(stage, details):
        record = dict(stage=stage, details=details)
        with progress_path.open('a') as stream:
            stream.write(json.dumps(record, default=str)+'\n')
        print(stage, json.dumps(details, default=str), flush=True)
    start, cpu = perf_counter(), process_time()
    preview_start, preview_cpu = perf_counter(), process_time()
    groups = group_elements(source)
    estimate = query_groups(atlas, groups, kernel.detector.positions_m)
    factors = kernel.detector.effective_area_m2*acceptance_maximum(kernel)
    scattered = estimate['charge']*factors[:, None]
    sigma = estimate['sigma_sum']*factors[:, None]
    # Joint replaces this exact direct-charge column itself. Legacy selection needs it now.
    if args.mode=='joint':
        q0=np.zeros(len(global_ids))
    elif args.reference_ballistic_method=='interval':
        q0=ballistic_interval(kernel,source).charge_pe
    else:
        q0=preview_ballistic(kernel,source,'spectral-fused')
    coarse_charge = np.column_stack((q0, scattered))
    forced = (estimate['nearest_source_m'] < 3) | ((sigma.sum(1) > .25*scattered.sum(1))
        & (q0+scattered.sum(1)+4*sigma.sum(1) >= args.threshold_pe/8))
    preview = SimpleNamespace(charge_components_pe=coarse_charge, charge_pe=coarse_charge.sum(1),
        unsafe_mask=forced, metadata=dict(independent=True, inactivity_certified=False,
        fine_event_data_loaded=False, groups=len(groups.points)))
    preview_seconds, preview_cpu_seconds = perf_counter()-preview_start, process_time()-preview_cpu
    np.savez_compressed(args.output/'independent-preview.npz', charge=coarse_charge, sigma=sigma, forced=forced)
    progress('independent preview', dict(seconds=preview_seconds, cpu_seconds=preview_cpu_seconds, groups=len(groups.points)))
    screen = AdaptiveScreenConfig(scatter_margin=2., audit_all_rejected=args.audit_all_rejected)
    joint_config = JointMuonConfig(ring_nodes=args.ring_nodes, first_chunk_elements=args.first_chunk_elements,
                                  ge2_backend=args.ge2_backend,first_backend=args.first_backend,
                                  first_skip_epsilon=args.first_skip_epsilon,
                                  first_skip_seed_fraction=args.first_skip_seed_fraction,
                                  ge2_skip_epsilon=args.ge2_skip_epsilon,ge2_skip_safety=args.ge2_skip_safety,
                                  first_fuse_bands=args.first_fuse_bands,
                                  ge2_batch_points=args.ge2_batch_points,ge2_blas_precision=args.ge2_blas_precision,
                                  ge2_operator=args.ge2_operator,
                                  ge2_z_step_m=args.ge2_z_step_m,ge2_tau_step_ns=args.ge2_tau_step_ns,
                                  ge2_transverse_step_m=args.ge2_transverse_step_m)
    if args.mode == 'joint':
        response = transport_joint_muon(kernel, source, ancestry, preview=preview,
            primary_first_store=first, primary_ge2_store=ge2, ge2_store=ge2_store,
            joint_first_atlas=first_cdf, config=joint_config, screen_config=screen, progress=progress,
            primary_options=dict(ge2_beta_atlas=ge2_beta, first_beta_atlas=first_beta,
                direct_ge2_quadrature='filon', q1_charge_gauss_order=6,
                fallback_pack=args.ge2_pack))
    else:
        response = transport_smart_muon(kernel, source, ancestry, preview=preview,
            first_pack_directory=None, frequency_nodes=65, screen_config=screen,
            prompt_config=lh.PromptConfig(screen=False,batch_charge_tolerance_pe=1e-6,
                                         ballistic_method=args.reference_ballistic_method))
    elapsed, cpu_seconds = perf_counter()-start, process_time()-cpu
    response.save(args.output/'response.npz')
    # Supplemental global OM IDs make restricted debug results comparable to full runs.
    with np.load(args.output/'response.npz', allow_pickle=False) as archive:
        payload = {key: archive[key].copy() for key in archive.files}
    payload['receiver_indices'] = global_ids
    payload['positions_m'] = kernel.detector.positions_m
    payload['orientations'] = kernel.detector.orientations
    np.savez_compressed(args.output/'response.npz', **payload)
    report = dict(complete=True, energy_GeV=args.energy, mode=args.mode,
        source=str(source_path), source_metadata=source_metadata,
        full_model_fingerprint=complete_model_fingerprint, receiver_indices=global_ids.tolist(),
        geometry_scope='five clusters' if args.receiver_indices is None else 'restricted debug OMs from five clusters',
        source_elements=len(source), primary_elements=int((source.uid == 0).sum()),
        secondary_elements=int((source.uid != 0).sum()), modules=len(global_ids),
        active=int(response.active.sum()), threads=numba.get_num_threads(),
        elapsed_seconds=elapsed, cpu_seconds=cpu_seconds, preview_seconds=preview_seconds,
        preview_cpu_seconds=preview_cpu_seconds, cache_load_seconds=cache_load_seconds,
        jit_warm_seconds=warm_seconds, cache_build_included=False, g4_generation_included=False,
        source_and_cache_sha256=provenance,
        process_peak_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(2**20 if sys.platform == 'darwin' else 2**10),
        joint_config=asdict(joint_config) if args.mode == 'joint' else None,
        reference_ballistic_method=args.reference_ballistic_method if args.mode=='reference' else None,
        metadata=response.metadata, fine_reference_loaded_after_transport=args.compare is not None,
        comparison=compare_response(response, args.compare, global_ids) if args.compare else None)
    (args.output/'report.json').write_text(json.dumps(report, indent=2, default=str))
    if args.viewer:
        from lighthit.viewer import viewer_payload, write_event_viewer
        payload = viewer_payload(response, source=source, event_id=f'muon-{args.energy}-{args.mode}',
            label=f'{args.energy} GeV muon, {args.mode}, five clusters')
        write_event_viewer(payload, args.output/'viewer.html', json_path=args.output/'viewer.json')
    progress('complete', dict(energy_GeV=args.energy, mode=args.mode, cpu_seconds=cpu_seconds,
        elapsed_seconds=elapsed, active=int(response.active.sum()), process_peak_mib=report['process_peak_mib']))


if __name__ == '__main__':
    main()
