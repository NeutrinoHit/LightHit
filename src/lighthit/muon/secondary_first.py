"""First scattering (Q1) of all secondary packets of a muon event.

Each packet is compressed into pencils and grouped per OM by the prompt solver
(physical time, no frequency inversion). Packet support, spectral weights and
omission budgets follow the primary/secondary split of the muon route; direct
light and >=2 are computed elsewhere.
"""
from dataclasses import asdict, replace
from time import perf_counter, process_time

import numpy as np


def evaluate_secondary_first(kernel, full_source, packets, receiver_indices,
                             time_origin_ns, *, radius_m=400., config=None,
                             progress=None, skip_epsilon=0., skip_seed_fraction=.5,
                             fuse_bands=False, fused_pixel_face=64, fused_ring_nodes=192):
    """Return secondary Q1 charge, bins and measurements on supplied receivers.

    The spectral budget denominator includes all emitting original elements,
    including the primary, exactly as in MuonFineStrategy. Each band applies
    Cherenkov threshold before constructing the packet support sphere.
    Original coefficients go to the prompt evaluator; its spectral tables
    apply lambda^-2, wavelength quadrature, QE and detector area once.

    ``fuse_bands=True`` evaluates elements above threshold in every band in one
    grouped call for all bands: shared pencils with one frozen cone at the
    s0w-weighted mean phase index, built with ``fused_pixel_face`` and
    ``fused_ring_nodes``; each band keeps its own optics, yield (phase index in
    s2w) and clocks in the batch evaluation. Elements above threshold in only
    some bands are evaluated per band with the ordinary settings. This changes
    the approximation; it was accepted against original-step controls.

    ``skip_epsilon > 0`` enables certified omission of weak packet-receiver
    pairs (``packet_skip``). Per evaluation unit (band or fused band group),
    each receiver's seed packets (largest rigorous Q1 bounds covering
    ``skip_seed_fraction`` of its bound sum) are computed first; further
    packets are omitted only while their summed bounds stay <= skip_epsilon
    times that receiver's computed seed Q1. Computed pairs are unchanged;
    unit-outer accumulation changes only summation order.
    """
    from .. import prompt

    if full_source.cone_model != 'spectral':
        raise ValueError('secondary first requires spectral cones')
    if not np.isfinite(radius_m) or radius_m <= 0.:
        raise ValueError('prompt packet radius must be finite and positive')
    if not (np.isfinite(skip_epsilon) and 0. <= skip_epsilon < 1.):
        raise ValueError('skip_epsilon must be finite and in [0, 1)')
    if not (np.isfinite(skip_seed_fraction) and 0. < skip_seed_fraction <= 1.):
        raise ValueError('skip_seed_fraction must be in (0, 1]')
    ids = np.asarray(receiver_indices)
    origins = np.ascontiguousarray(time_origin_ns, float)
    if (ids.ndim != 1 or ids.dtype.kind not in 'iu' or
            origins.shape != ids.shape or not np.isfinite(origins).all() or
            np.any(ids < 0) or np.any(ids >= len(kernel.detector.positions_m)) or
            len(np.unique(ids)) != len(ids)):
        raise ValueError('unique receiver indices and one finite origin per receiver required')
    positions = np.ascontiguousarray(kernel.detector.positions_m[ids], float)
    config = config or prompt.PromptConfig(screen=False, batch_charge_tolerance_pe=1e-6)
    if config.element_method not in ('auto', 'grouped'):
        raise ValueError('secondary first requires grouped prompt settings')
    config = replace(config, screen=False, element_method='grouped')
    fused_config = replace(config, pixel_face=int(fused_pixel_face), ring_nodes=int(fused_ring_nodes))
    looks = np.ascontiguousarray(-kernel.detector.orientations[ids], float)
    areas = np.ascontiguousarray(kernel.detector.effective_area_m2[ids], float)
    edges = np.asarray(kernel.config.relative_time_edges_ns, float)
    tables = prompt._spectral_tables(kernel)
    coef, ipow, jpow, degree = prompt.acceptance_monomials(kernel.acceptance()['alpha'])
    g = float(kernel.medium.g)
    nband = len(tables['wavelength_nm'])
    common_valid = full_source.beta * float(np.max(tables['phase_index'])) > 1.
    band_valid = [common_valid & (full_source.beta * phase > 1.) for phase in tables['phase_index']]
    denominators = [float(full_source.coefficient0[valid].sum()) for valid in band_valid]
    units = _evaluation_units(tables, band_valid, fuse_bands)
    q = np.zeros(len(ids)); bins = np.zeros((len(ids), len(edges)-1))
    omissions = np.zeros((len(ids), 2)); details = []
    wavelength = [dict(wavelength_nm=float(wave), source_elements=int(valid.sum()),
                       secondary_elements=0, charge_pe_per_om=np.zeros(len(ids)),
                       first_compile_cpu_s=0., first_apply_cpu_s=0.)
                  for wave, valid in zip(tables['wavelength_nm'], band_valid)]
    fused_row = dict(bands=units[0]['bands'], cone_phase_index=units[0]['cone'],
                     pixel_face=int(fused_pixel_face), ring_nodes=int(fused_ring_nodes),
                     secondary_elements=0, charge_pe_per_om=np.zeros(len(ids)),
                     first_compile_cpu_s=0., first_apply_cpu_s=0.) if fuse_bands else None

    def prepare(packet, unit, selected):
        """Unit source part, support receivers, segments and settings."""
        from .muon_packets import SourcePacket
        part = SourcePacket(packet.kind, selected).source(full_source)
        part = replace(part, cone_model='frozen', reference_phase_index=float(unit['cone']))
        ends = part.start_m + part.length_m[:, None]*part.direction
        centre = .5*(np.minimum(part.start_m.min(0), ends.min(0))+
                     np.maximum(part.start_m.max(0), ends.max(0)))
        reach = max(np.linalg.norm(part.start_m-centre, axis=1).max(),
                    np.linalg.norm(ends-centre, axis=1).max())
        work = np.flatnonzero(np.linalg.norm(positions-centre, axis=1)-reach <= radius_m)
        segments = prompt._element_segments(part)
        budget = 0.
        for band in unit['bands']:
            budget += config.batch_charge_tolerance_pe/nband*float(part.coefficient0.sum()/denominators[band])
        settings = replace(fused_config if unit['fused'] else config, batch_charge_tolerance_pe=budget)
        return part, work, segments, budget, settings

    def account(unit, key, value):
        """Add a timing or charge entry to the band or fused-unit record."""
        target = fused_row if unit['fused'] else wavelength[unit['bands'][0]]
        target[key] += value

    def apply(unit, part, segments, settings, targets):
        """Grouped Q1 of one unit part on ``targets``; accumulates the result."""
        qp, bp, metadata = prompt._order1('grouped', kernel, part, part, segments,
            positions, looks, areas, origins, targets, unit['tables'],
            coef, ipow, jpow, degree, g, edges, settings, np.zeros(len(ids)))
        q[:] += qp; bins[:] += bp; omissions[:] += metadata.get('order1_batch_omission', 0.)
        account(unit, 'charge_pe_per_om', qp)
        return qp, {key: value.tolist() if isinstance(value, np.ndarray) else value
                    for key, value in metadata.items()}

    began, cpu = perf_counter(), process_time()
    if skip_epsilon > 0.:
        skip_meta = _skipping_units(kernel, full_source, packets, ids, positions, areas,
            tables, units, common_valid, prepare, apply, account, details,
            float(skip_epsilon), float(skip_seed_fraction), progress)
    for number, packet in (() if skip_epsilon > 0. else enumerate(packets)):
        if packet.kind == 'primary':
            continue
        row = dict(packet=number, kind=packet.kind, family_uid=packet.family_uid,
                   elements=len(packet.indices), prompt_receivers=0,
                   first_compile_cpu_s=0., first_apply_cpu_s=0., wavelengths=[])
        touched = np.zeros(len(ids), bool)
        for unit in units:
            selected = np.sort(np.asarray(packet.indices)[unit['valid'][packet.indices]])
            if not len(selected):
                continue
            started = process_time()
            part, work, segments, budget, settings = prepare(packet, unit, selected)
            compile_cpu = process_time()-started
            row['first_compile_cpu_s'] += compile_cpu
            account(unit, 'first_compile_cpu_s', compile_cpu)
            account(unit, 'secondary_elements', len(part))
            band_row = dict(wavelength_nm=[float(tables['wavelength_nm'][b]) for b in unit['bands']]
                            if unit['fused'] else float(tables['wavelength_nm'][unit['bands'][0]]),
                            source_elements=len(part), receivers=len(work),
                            batch_charge_tolerance_pe=budget, compile_cpu_s=compile_cpu,
                            apply_cpu_s=0.)
            if len(work):
                started = process_time()
                _, metadata = apply(unit, part, segments, settings, work)
                apply_cpu = process_time()-started
                touched[work] = True
                row['first_apply_cpu_s'] += apply_cpu
                account(unit, 'first_apply_cpu_s', apply_cpu)
                band_row.update(apply_cpu_s=apply_cpu, prompt_metadata=metadata)
            row['wavelengths'].append(band_row)
        row['prompt_receivers'] = int(touched.sum())
        details.append(row)
        if progress:
            progress('secondary first packet', row)
    for row in wavelength + ([fused_row] if fused_row else []):
        row['charge_pe_per_om'] = row['charge_pe_per_om'].tolist()
    return q, bins, dict(cpu_s=process_time()-cpu, wall_s=perf_counter()-began,
        packets=details, wavelengths=wavelength, config=asdict(config),
        orders=[1], first_backend='legacy-grouped', prompt_radius_m=float(radius_m),
        receiver_indices=ids.tolist(), order1_batch_omission=omissions.tolist(),
        compression_included_in_apply_cpu=True,
        source_representation=('original spectral secondary elements, grouped separately by packet; '
                               + ('one fused all-band unit plus per-band threshold remainders'
                                  if fuse_bands else 'one unit per band')),
        ballistic_computed=False, primary_computed=False,
        wavelength_yield_applied_twice=False,
        limitation='inherits legacy grouping, quadrature and empirical batch omission estimates',
        **({'fused_bands': fused_row} if fuse_bands else {}),
        **({'packet_skip': skip_meta} if skip_epsilon > 0. else {}))


def _evaluation_units(tables, band_valid, fuse_bands):
    """Bands evaluated together, their element masks, cone phase and tables.

    Without fusion every band is its own unit (the original computation).
    With fusion the first unit holds elements above threshold in all bands and
    every band keeps a remainder unit for elements emitting only in some bands.
    """
    nband = len(tables['wavelength_nm'])
    phase = np.asarray(tables['phase_index'], float)
    single = [dict(bands=[band], valid=band_valid[band], cone=float(phase[band]), fused=False,
                   tables={key: value[band:band+1] for key, value in tables.items()})
              for band in range(nband)]
    if not fuse_bands:
        return single
    everywhere = np.logical_and.reduce(band_valid)
    weight = np.asarray(tables['s0w'], float)
    fused = dict(bands=list(range(nband)), valid=everywhere, fused=True,
                 cone=float(np.sum(weight*phase)/np.sum(weight)), tables=tables)
    for unit in single:
        unit['valid'] = unit['valid'] & ~everywhere
    return [fused] + single


def _skipping_units(kernel, full_source, packets, ids, positions, areas, tables,
                    units, common_valid, prepare, apply, account, details,
                    epsilon, seed_fraction, progress):
    """Unit-outer secondary Q1 with certified packet-receiver omission.

    Accumulates through ``apply``/``account`` and returns the omission
    certificate. Packet support, selection, budgets and the grouped solver are
    those of the ordinary loop. A unit's bound is the sum of its bands' bounds
    over the unit's elements, so the certificate holds per unit and receiver.
    """
    from .muon_packets import SourcePacket
    from .packet_skip import (bound_constants, packet_first_bounds,
                              seed_packets, select_skips)
    secondary = [number for number, packet in enumerate(packets) if packet.kind != 'primary']
    n_packet, n_receiver, n_unit = len(secondary), len(ids), len(units)
    rows = [dict(packet=number, kind=packets[number].kind,
                 family_uid=packets[number].family_uid,
                 elements=len(packets[number].indices), prompt_receivers=0,
                 first_compile_cpu_s=0., first_apply_cpu_s=0., wavelengths=[])
            for number in secondary]
    selections = [[None]*n_unit for _ in secondary]
    reach = np.zeros((n_unit, n_packet, n_receiver), bool)
    bounds = np.zeros((n_unit, n_packet, n_receiver))
    cost = np.zeros((n_unit, n_packet))
    touched = np.zeros((n_packet, n_receiver), bool)
    constants = bound_constants(kernel)
    plan_cpu = process_time()
    for k, number in enumerate(secondary):
        packet = packets[number]
        for u, unit in enumerate(units):
            selected = np.sort(np.asarray(packet.indices)[unit['valid'][packet.indices]])
            if not len(selected):
                continue
            _, work, _, _, _ = prepare(packet, unit, selected)
            selections[k][u] = selected
            reach[u, k, work] = True
            cost[u, k] = len(selected)
            account(unit, 'secondary_elements', len(selected))
        targets = np.flatnonzero(reach[:, k].any(0))
        if len(targets):
            # One bound evaluation per packet: the kernel applies each band's
            # own threshold to the common-threshold elements, which contain
            # every unit's elements, so each unit's band sum stays an upper bound.
            common = np.sort(np.asarray(packet.indices)[common_valid[packet.indices]])
            part = SourcePacket(packet.kind, common).source(full_source)
            per_band = packet_first_bounds(part, positions[targets], areas[targets], tables, constants)
            for u, unit in enumerate(units):
                bounds[u, k, targets] = per_band[unit['bands']].sum(0)
    bounds[~reach] = 0.
    plan_cpu = process_time() - plan_cpu

    def run(k, u, targets, stage):
        """Compute one packet of one unit on ``targets`` and record its cost."""
        packet = packets[secondary[k]]; unit = units[u]
        started = process_time()
        part, _, segments, budget, settings = prepare(packet, unit, selections[k][u])
        compile_cpu = process_time()-started
        started = process_time()
        qp, metadata = apply(unit, part, segments, settings, targets)
        apply_cpu = process_time()-started
        touched[k, targets] = True
        row = rows[k]
        row['first_compile_cpu_s'] += compile_cpu; row['first_apply_cpu_s'] += apply_cpu
        account(unit, 'first_compile_cpu_s', compile_cpu)
        account(unit, 'first_apply_cpu_s', apply_cpu)
        row['wavelengths'].append(dict(
            wavelength_nm=[float(tables['wavelength_nm'][b]) for b in unit['bands']],
            fused=unit['fused'], stage=stage, source_elements=len(part), receivers=len(targets),
            batch_charge_tolerance_pe=budget, compile_cpu_s=compile_cpu,
            apply_cpu_s=apply_cpu, prompt_metadata=metadata))
        return qp

    certificate = []
    for u, unit in enumerate(units):
        if not reach[u].any():
            continue
        seed = seed_packets(bounds[u], seed_fraction) & reach[u]
        seed_charge = np.zeros(n_receiver)
        for k in np.flatnonzero(seed.any(1)):
            seed_charge += run(k, u, np.flatnonzero(seed[k]), 'seed')
        skip = select_skips(bounds[u], cost[u], seed, seed_charge, epsilon) & reach[u]
        rest = reach[u] & ~seed & ~skip
        for k in np.flatnonzero(rest.any(1)):
            run(k, u, np.flatnonzero(rest[k]), 'rest')
        omitted = (bounds[u]*skip).sum(0)
        ratio = np.divide(omitted, seed_charge, out=np.zeros(n_receiver), where=seed_charge > 0)
        if np.any((omitted > 0) & ~(seed_charge > 0)) or ratio.max(initial=0.) > epsilon*(1+1e-9):
            raise RuntimeError('packet omission certificate violated')
        computed = reach[u] & ~skip
        row = dict(wavelength_nm=[float(tables['wavelength_nm'][b]) for b in unit['bands']],
                   fused=unit['fused'],
                   pairs=int(reach[u].sum()), seed_pairs=int(seed.sum()),
                   skipped_pairs=int(skip.sum()),
                   skipped_elements_x_receivers=float((cost[u][:, None]*skip).sum()),
                   elements_x_receivers=float((cost[u][:, None]*reach[u]).sum()),
                   packets_fully_skipped=int((reach[u].any(1) & ~computed.any(1)).sum()),
                   max_omitted_bound_over_seed_charge=float(ratio.max(initial=0.)),
                   omitted_bound_pe=omitted.tolist(), seed_charge_pe=seed_charge.tolist())
        certificate.append(row)
        if progress:
            progress('secondary first unit', {key: value for key, value in row.items()
                                              if not isinstance(value, list) or key == 'wavelength_nm'})
    for k, row in enumerate(rows):
        row['prompt_receivers'] = int(touched[k].sum())
        details.append(row)
    return dict(epsilon=epsilon, seed_fraction=seed_fraction, plan_and_bound_cpu_s=plan_cpu,
                bands=certificate,
                certificate=('per evaluation unit and receiver, omitted all-time Q1 <= sum of '
                             'rigorous packet bounds <= epsilon * computed seed Q1 of that '
                             'receiver, for the supplied positive spectral quadrature'),
                bound='min(finite-ray, infinite-line) per element and band; track_first_bounds')
