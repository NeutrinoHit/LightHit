"""Experimental all-time first-scattering charge with receiver-local grouping.

Keep the production spatial/angular pencils and their acceptance moments, but
remove the arrival-time grouping axis. This is an approximation of stationary
transport, not an upper bound and not a waveform method. Production is unchanged.
"""
from dataclasses import replace
from time import perf_counter, process_time
import numpy as np


def stationary_pencils_charge(kernel, elements, *, config=None, merge_time=False):
    """Return q1 and diagnostics, preserving physical beta and spectral weights.

    ``merge_time=True`` is a timing control retaining ordinary arrival groups.
    The candidate uses a receiver-local auxiliary clock placing every pencil in
    the middle of the same arrival bucket. It never creates an output waveform.
    Radial and angular tolerances still need detector-response convergence tests.
    """
    from .. import prompt, _prompt_numba as kernels
    from ..sources import SpectralLightElements
    if not isinstance(elements, SpectralLightElements) or elements.cone_model != 'spectral':
        raise ValueError('requires a spectral element source')
    settings = config or prompt.PromptConfig(element_method='grouped', screen=False)
    if settings.element_method != 'grouped' or settings.screen:
        raise ValueError('requires grouped, screen=False')
    tables = prompt._spectral_tables(kernel)
    alpha = np.ascontiguousarray(kernel.acceptance()['alpha'], float)
    coef, ipow, jpow, degree = prompt.acceptance_monomials(alpha)
    positions = np.ascontiguousarray(kernel.detector.positions_m, float)
    looks = np.ascontiguousarray(-kernel.detector.orientations, float)
    areas = np.asarray(kernel.detector.effective_area_m2, float)
    edges = np.array([0., 1.])
    charge = np.zeros(len(positions)); details = []
    wall, cpu = perf_counter(), process_time()
    for lam, phase in enumerate(tables['phase_index']):
        if tables['mus'][lam] == 0 or (tables['s0w'][lam] == 0 and tables['s2w'][lam] == 0):
            continue
        active = elements.beta * phase > 1.
        if not np.any(active):
            continue
        source = elements if np.all(active) else elements.subset(active)
        source = replace(source, reference_phase_index=float(phase), cone_model='frozen')
        if not merge_time:
            # Compression may accumulate clock moments; make independence exact.
            source = replace(source, start_ns=np.zeros(len(source)), end_ns=np.zeros(len(source)))
        start = process_time()
        pencils = prompt.compress(source, prompt._element_segments(source), settings)
        compression_cpu = process_time() - start
        single = {key: value[lam:lam+1] for key, value in tables.items()}
        kappa = -single['mut'] * single['speed']
        eedge = np.exp(np.outer(kappa, edges))
        itab = kernels.interval_table(edges, kappa)
        start = process_time(); groups = []; batches = []; omitted = []
        for d in range(len(positions)):
            r = np.linalg.norm(pencils['position'] - positions[d], axis=1)
            speed_ref = float(single['speed'][0])
            # Half-bucket offset avoids floating-point splitting at tau=0.
            clock = (pencils['time'] if merge_time else
                     .5 * settings.merge_tau_ns - r / speed_ref)
            omission = np.zeros((1, 2))
            q, bins, ng, nb = kernels.grouped_pencils_order1(
                positions[d:d+1], looks[d:d+1], areas[d:d+1], np.zeros(1), np.array([0], np.int64),
                pencils['position'], pencils['direction'], pencils['w0'], pencils['w2'],
                np.ascontiguousarray(clock), pencils['major'], pencils['var_major'],
                pencils['var_minor'], pencils['pos_cov'], kernels._GAUSS_TAU, kernels._GAUSS_W,
                coef, ipow, jpow, degree, float(kernel.medium.g),
                (1.-abs(kernel.medium.g))/(1.+abs(kernel.medium.g)),
                settings.path_step, settings.path_step, settings.tail_exponent,
                single['mus'], single['mut'], single['speed'], single['s0w'], single['s2w'],
                edges, eedge, itab, settings.direct_bins, settings.max_path_nodes,
                settings.smoothing_factor, settings.merge_theta_min, settings.merge_theta_ratio,
                settings.merge_tau_ns, settings.merge_log_r, speed_ref, 1, False,
                settings.batch_charge_tolerance_pe / len(tables['phase_index']), omission)
            if bins.shape != (1, 0):
                raise RuntimeError('stationary route unexpectedly allocated time bins')
            charge[d] += q[0]; groups.append(int(ng[0])); batches.append(int(nb[0]))
            omitted.append(float(omission[0, 1]))
        details.append(dict(wavelength_nm=float(tables['wavelength_nm'][lam]),
            pencils=len(pencils['position']), compression_cpu_s=compression_cpu,
            evaluation_cpu_s=process_time()-start, groups=groups, batches=batches,
            omitted_profile_estimate_pe=omitted))
    return charge, dict(cpu_seconds=process_time()-cpu, wall_seconds=perf_counter()-wall,
        elements=len(elements), receivers=len(positions), arrival_time_grouping=bool(merge_time),
        computed_bins=False, certified_upper_bound=False, wavelengths=details)
