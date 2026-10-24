"""Earlier reference muon route: local muon components on adaptively chosen OMs.

Kept for comparisons with the current route (:mod:`lighthit.muon`). It uses the
primary-track integrals and packet bases of
:mod:`lighthit.experimental.muon_components` for every selected OM.
"""
from copy import copy
from dataclasses import replace, asdict
from time import perf_counter

import numpy as np

from ..muon.adaptive_screen import AdaptiveScreenConfig, adaptive_candidates


def transport_smart_muon(kernel, source, ancestry, *, first_pack_directory,
                          screen_config=None, preview_config=None, preview=None,
                          frequency_nodes=33, primary_uid=0,prompt_config=None):
    """Explicit prototype: independent coarse charges, then original fine physics.

    Unselected OMs retain labelled coarse integrated q1/qge2 estimates and zero
    uncomputed time bins. Fine results are cached per OM through all reopenings.
    No previous fine result is accepted by the selector.
    """
    from .charge_preview import charge_preview, ChargePreviewConfig
    from .muon_components import transport_muon_components
    from ..prompt import PromptConfig
    from ..transport import HybridTransportResponse
    from .._shower_basis import earliest_element_arrival
    began = perf_counter()
    supplied_preview = preview is not None
    preview = preview or charge_preview(kernel, source, ge2_store=kernel.ge2_kernel_store(),
        first_pack_directory=first_pack_directory,
        config=preview_config or ChargePreviewConfig(track_uid=primary_uid))
    batches = []
    detector = kernel.detector
    def fine(indices):
        local = copy(kernel)
        local.detector = detector.subset(indices)
        response = transport_muon_components(local, source, ancestry,
            primary_uid=primary_uid,
            frequency_nodes=frequency_nodes, partition_ge2=True, track_stencil_center='centroid',
            prompt_config=(prompt_config or PromptConfig(screen=False, batch_charge_tolerance_pe=1e-6)))
        batches.append((indices.copy(), response))
        return response.charge_components_pe
    charge, evaluated, audit = adaptive_candidates(preview, detector, kernel.config.threshold_pe,
                                                    fine, screen_config)
    if not batches:
        # Always provide at least one fully evaluated control receiver; this
        # also supplies the standard output grids without inventing metadata.
        i = np.array([int(np.argmax(preview.charge_pe))])
        charge[i] = fine(i); evaluated[i] = True
        audit['fine_receivers'] = int(evaluated.sum())
        audit['coarse_only_receivers'] = int((~evaluated).sum())
    example = batches[0][1]
    n = len(detector.positions_m)
    components = np.zeros((n, len(example.relative_time_edges_ns)-1, 3))
    spectrum = np.zeros((len(example.omega_per_ns), n), complex)
    spectrum[0] = charge[:, 2]
    elements = kernel._source_elements(source)
    sample = kernel.medium.sample(kernel.wavelength.wavelength_nm)
    valid = elements.beta*np.max(sample['phase_index']) > 1
    if not np.all(valid): elements=elements.subset(valid)
    fastest = float(np.min(sample['group_index']))
    if (np.linalg.norm(np.ptp(elements.midpoints_m, axis=0)) > kernel.config.radial_range_m[1]
            or np.max(elements.end_ns)-np.min(elements.start_ns) > .5*np.ptp(example.relative_time_edges_ns)):
        origins = earliest_element_arrival(np.ascontiguousarray(detector.positions_m),
            np.ascontiguousarray(elements.start_m), np.ascontiguousarray(elements.direction),
            np.ascontiguousarray(elements.length_m), np.ascontiguousarray(elements.start_ns), fastest)
    else:
        first = int(np.argmin(elements.start_ns))
        origins = elements.start_ns[first]+np.linalg.norm(detector.positions_m-elements.start_m[first], axis=1)*fastest/.299792458
    active = np.zeros(n, bool)
    levels = np.ones(n, np.int8)
    for indices, response in batches:
        np.testing.assert_allclose(origins[indices], response.time_origin_ns, rtol=0, atol=1e-10)
        components[indices] = response.components_pe
        spectrum[:, indices] = response.spectrum_ge2_pe
        active[indices] = response.active
        levels[indices] = np.asarray(response.metadata['prompt_order1_level'], np.int8)
    details = []
    for ids, response in batches:
        details.append(dict(receivers=len(ids), active=int(response.active.sum()),
            stages={k:v for k,v in response.metadata.items() if k.endswith('_seconds')}))
    metadata = dict(example.metadata)
    metadata['active_modules'] = int(active.sum())
    metadata['prompt_order1_level'] = levels.tolist()
    metadata['inactive_order1_charge_may_be_coarse'] = bool(np.any(~evaluated))
    metadata['inactive_ge2_charge_may_be_coarse'] = bool(np.any(~evaluated))
    metadata['fine_charge_mask'] = evaluated.tolist()
    metadata['smart_screen'] = dict(**audit, preview=preview.metadata,
        supplied_preview=supplied_preview, fine_batches=details,
        elapsed_seconds=perf_counter()-began,
        coarse_only_values='integrated preview charges; no fine waveform or activity certificate')
    return HybridTransportResponse(detector, example.omega_per_ns.copy(),
        example.relative_time_edges_ns.copy(), spectrum, components, charge,
        origins, active, 'hybrid-smart-screen-experimental', metadata)
