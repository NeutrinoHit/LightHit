"""Charge-only coarse order 1 with receiver-local angular refinement.

The inverse-angle-weighted Gaussian measure is integrated with two positive
radial nodes. Broad parent pixels are replaced by their children near the
receiver direction. All angular moments come from the same deposited rings.
Charge batches depend on distance and angle, never on emission time.
The detailed solver's phase, acceptance and optical-path integrals are shared.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import math

import numpy as np
from numba import njit

from ._prompt_numba import (
    _accumulate_cubic_moments_pair_prepared,
    _batch_add_moments,
    _batch_deposit,
    _batch_grid,
    _batch_sigma,
    _exact_pencil_geometry_ids,
    _fill_moments_averaged_pair,
    _finalize,
    _float64_bits,
    _grouped_batch_charge_estimate,
    _grow,
    _hash_slot,
    _is_cubic_moment_layout,
    _pencil_frame_from_ray_cached_look,
    _pencil_shape_factor_from_ray,
    _prepare_common_pencil_look,
    _prepare_gauss_inverse,
    _prepare_pencil_look_powers,
    _prepare_pencil_shape,
    build_theta_bit_lut,
    build_theta_thresholds,
    scalar_theta_index,
)


_FIELDS = ("position", "direction", "w0", "w2", "time", "major",
           "var_major", "var_minor", "pos_cov")


def parent_keys(keys, face, parent_face):
    """Exact parent indices of nested equi-angular cube pixels."""
    if face % parent_face:
        raise ValueError("angular refinement must be a multiple of its parent")
    factor = face // parent_face
    cell, pixel = np.divmod(keys, 6 * face * face)
    cube_face, ij = np.divmod(pixel, face * face)
    i, j = np.divmod(ij, face)
    return (cell * (6 * parent_face * parent_face)
            + (cube_face * parent_face + i // factor) * parent_face + j // factor)


def pack_pencils(parts, face, parent_face):
    """Preorder layout lets an accepted parent skip all of its children."""
    fine = {key: np.concatenate([part[0][i] for part in parts])
            for i, key in enumerate(_FIELDS)}
    coarse = {key: np.concatenate([part[1][i] for part in parts])
              for i, key in enumerate(_FIELDS)}
    fine_keys = np.concatenate([part[2] for part in parts])
    coarse_keys = np.concatenate([part[3] for part in parts])
    parents = np.searchsorted(coarse_keys, parent_keys(fine_keys, face, parent_face))
    n = len(coarse_keys)
    if np.any(parents >= n) or not np.array_equal(
            coarse_keys[parents], parent_keys(fine_keys, face, parent_face)):
        raise ValueError("inconsistent coarse angular hierarchy")
    total = np.bincount(parents, weights=fine["w0"], minlength=n)
    if not np.allclose(total, coarse["w0"], rtol=1e-10, atol=0.):
        raise ValueError("coarse angular hierarchy does not conserve emission")
    order = np.argsort(np.r_[np.arange(n), parents], kind="stable")
    inverse = np.empty(len(order), np.int64)
    inverse[order] = np.arange(len(order))
    result = {key: np.ascontiguousarray(np.concatenate([coarse[key], fine[key]])[order])
              for key in _FIELDS}
    parent = np.r_[np.full(n, -2), parents].astype(np.int64)[order]
    child = parent >= 0
    parent[child] = inverse[parent[child]]
    next_node = np.arange(1, len(order) + 1, dtype=np.int64)
    roots = np.flatnonzero(parent == -2)
    next_node[roots] = np.r_[roots[1:], len(order)] if len(roots) else []
    result["hierarchy_parent"] = parent
    result["next_node"] = next_node
    return result


def compress_screen(elements, segments, config):
    from .prompt import compress
    coarse = replace(config, cell_m=config.screen_cell_m,
                     pixel_face=config.screen_pixel_face * config.screen_refine_factor,
                     ring_nodes=config.screen_ring_nodes)
    return compress(elements, segments, coarse,
                    _parent_pixel_face=int(config.screen_pixel_face))


@njit(cache=True, nogil=True)
def inverse_radius_rule(u, v, var_u, var_v, tangent, weighted_sec2):
    """Two-node Gauss rule for dnu = dP(Gaussian)/radius.

    Its first four moments are E[1/R], 1, E[R], E[R**2]. The same Laplace
    integral computes both inverse and ordinary mean radius; tilted Gaussian
    second moments supply E[R]. The Jacobi matrix then gives positive nodes
    and weights. The constant and linear regular profiles are preserved,
    while their curvature is no longer discarded at a harmonic-mean angle.
    """
    scale = 2.0 * (var_u + var_v) + u*u + v*v
    if scale <= 0.0:
        return 0., 0., 0., 0.
    s0 = 1.0 / math.sqrt(scale)
    inverse = radial = 0.0
    for k in range(tangent.shape[0]):
        t = s0 * tangent[k]
        a = 1.0 + 2.0*t*t*var_u
        b = 1.0 + 2.0*t*t*var_v
        mass = weighted_sec2[k] * math.exp(-t*t*(u*u/a + v*v/b)) / math.sqrt(a*b)
        inverse += mass
        radial += mass * (var_u/a + var_v/b + u*u/(a*a) + v*v/(b*b))
    multiplier = 2.0 / math.sqrt(math.pi) * s0
    inverse *= multiplier
    radial *= multiplier
    if inverse <= 0.0:
        return 0., 0., 0., 0.
    mean = 1.0 / inverse
    second = radial / inverse
    variance = second - mean*mean
    if variance <= 0.0:
        return inverse, mean, 0., mean
    third = (u*u + v*v + var_u + var_v) / inverse
    other = (third - 2.0*mean*second + mean**3) / variance
    disc = math.sqrt((mean-other)**2 + 4.0*variance)
    low = .5 * (mean + other - disc)
    high = .5 * (mean + other + disc)
    low_weight = inverse * (high-mean) / (high-low)
    return low_weight, low, inverse-low_weight, high


@njit(cache=True, nogil=True)
def pencil_radial_rule(p, r, theta, sine, hx, hy, hz, major, vmajor, vminor,
                       prepared, smoothing, tangent, weights):
    """Angular nodes and singular weights of the apparent forward pencil."""
    m11 = vmajor[p] + prepared[p, 3] / (r*r)
    m22 = vminor[p] + prepared[p, 4] / (r*r)
    m12 = prepared[p, 5] / (r*r)
    half = .5 * (m11+m22)
    disc = math.sqrt(max(.25*(m11-m22)**2 + m12*m12, 0.))
    mu1 = half + disc
    mu2 = max(half-disc, 1e-16*mu1 + 1e-300)
    extent = math.sqrt(max(mu1, 0.))
    if extent > 0. and theta < smoothing*extent:
        u = hx*major[p, 0] + hy*major[p, 1] + hz*major[p, 2]
        v = hx*prepared[p, 0] + hy*prepared[p, 1] + hz*prepared[p, 2]
        # Direction covariances live in the angular tangent plane.
        if sine > 0.:
            u *= theta/sine
            v *= theta/sine
        angle = .5 * math.atan2(2*m12, m11-m22)
        ca, sa = math.cos(angle), math.sin(angle)
        w0, t0, w1, t1 = inverse_radius_rule(ca*u + sa*v, -sa*u + ca*v,
                                            mu1, mu2, tangent, weights)
        if t0 > 0. and t1 < .5*math.pi and w0 > 0. and w1 >= 0.:
            return (w0*t0/math.sin(t0), t0, w1*t1/math.sin(t1), t1,
                    theta < extent, True)
    factor, angle, averaged, ok = _pencil_shape_factor_from_ray(
        p, r, theta, sine, hx, hy, hz, major, vmajor, vminor, prepared,
        smoothing, tangent, weights)
    return factor, angle, 0., angle, averaged, ok


@njit(cache=True, nogil=True)
def _screen_block(block, receivers, looks, areas, origins, work, pencil_pos, pencil_dir, pencil_w0, pencil_w2, pencil_t, pencil_major, pencil_var_major, pencil_var_minor, pencil_pos_cov, gauss_tau, gauss_w, coef, ipow, jpow, max_power, g, y_g, h, h_tail, tail_exp, mus, mut, speed, s0w, s2w, edges, eedge, itab, direct_bins, max_nodes, smoothing_factor, theta_lo, theta_ratio, tau_step, log_r_step, speed_ref, n_blocks, n_recv, n_pen, n_lambda, nb, n_mono, charge_out, bins_out, groups_out, batches_out, mut_min, speed_max, n_values, log_theta_ratio, theta_bins, n_work, compute_bins, batch_tolerance_pe, omission, geometry_ids, representatives, prepared_shape, gauss_tangent, gauss_weighted_sec2, cubic_layout, theta_thresholds, theta_lut, theta_first, common_look, shared_look_powers, hierarchy_parent, next_node):
    """Private receiver scratch with read-only shared source preparation."""
    # Copies restore private ownership in hot inlined helpers. The source
    # mathematics and exact hash construction still run once per invocation.
    prepared_shape = prepared_shape.copy()
    gauss_tangent = gauss_tangent.copy()
    gauss_weighted_sec2 = gauss_weighted_sec2.copy()
    theta_thresholds = theta_thresholds.copy()
    theta_lut = theta_lut.copy()
    pre = np.zeros((n_lambda, 3))
    acc = np.zeros((n_lambda, nb if compute_bins else 0, 4))
    charge = np.zeros(n_lambda)
    s_buf = np.empty(max_nodes)
    t_buf = np.empty(max_nodes)
    h_buf = np.empty(max_nodes)
    g0 = np.empty(max_nodes)
    g2 = np.empty(max_nodes)
    cpow = np.empty(max_power + 1)
    spow = np.empty(max_power + 1)
    mom0 = np.empty(n_mono)
    mom2 = np.empty(n_mono)
    keys = np.full(1 << 12, -1, dtype=np.int64)
    values = np.empty((1 << 12, n_values))
    geometry = np.empty((representatives.shape[0], 7))
    geometry_base_keys = np.empty(representatives.shape[0], dtype=np.int64)
    if common_look:
        # Copy shared powers once; all receiver-dependent writes stay local.
        look_powers = shared_look_powers.copy()
    else:
        look_powers = np.empty((n_pen, 3))
    look_bits = looks.view(np.uint64)
    cached_look = np.empty(3, dtype=np.uint64)
    look_ready = False
    for index in range(block, n_work, n_blocks):
        d = work[index]
        pre[:, :] = 0.0
        acc[:, :, :] = 0.0
        charge[:] = 0.0
        keys[:] = -1
        used = 0
        rx = receivers[d, 0]
        ry = receivers[d, 1]
        rz = receivers[d, 2]
        if (not common_look and
                (not look_ready or cached_look[0] != look_bits[d, 0]
                 or cached_look[1] != look_bits[d, 1]
                 or cached_look[2] != look_bits[d, 2])):
            _prepare_pencil_look_powers(pencil_dir, looks[d, 0], looks[d, 1],
                                        looks[d, 2], look_powers)
            cached_look[0] = look_bits[d, 0]
            cached_look[1] = look_bits[d, 1]
            cached_look[2] = look_bits[d, 2]
            look_ready = True
        geometry_ready=np.zeros(representatives.shape[0],np.bool_)
        omission[d, 0] = 0.0
        omission[d, 1] = 0.0
        cursor=0
        while cursor<n_pen:
            p=cursor
            cursor+=1
            parent=hierarchy_parent[p]
            geometry_id = geometry_ids[p]
            if not geometry_ready[geometry_id]:
                rep=representatives[geometry_id]
                dx=rx-pencil_pos[rep,0]
                dy=ry-pencil_pos[rep,1]
                dz=rz-pencil_pos[rep,2]
                radius=math.sqrt(dx*dx+dy*dy+dz*dz)
                geometry[geometry_id,0]=radius
                geometry[geometry_id,1]=dx/radius
                geometry[geometry_id,2]=dy/radius
                geometry[geometry_id,3]=dz/radius
                geometry[geometry_id,4]=0.
                geometry[geometry_id,5]=0.
                geometry[geometry_id,6]=math.log(radius)
                geometry_base_keys[geometry_id]=(int(math.floor(math.log(radius)/log_r_step))+(1<<20))*2*theta_bins
                geometry_ready[geometry_id]=True
            ray = geometry[geometry_id]
            r = ray[0]
            r, theta, sine, cosine, c1, c3 = _pencil_frame_from_ray_cached_look(
                r, ray[1], ray[2], ray[3], pencil_dir[p, 0],
                pencil_dir[p, 1], pencil_dir[p, 2],
                looks[d, 0], looks[d, 1], looks[d, 2], look_powers[p, 0])
            if parent == -2:
                extent2=pencil_var_major[p]+prepared_shape[p,6]/(r*r)
                if theta<.5*math.pi and theta*theta<smoothing_factor*smoothing_factor*extent2:
                    continue
                cursor=next_node[p]
            # Apparent angular shape combines direction covariance with
            # the projected position covariance / r**2.
            factor2 = 0.0
            weight2 = 0.0
            theta2 = theta
            backward = theta > 0.5 * math.pi
            bound = 0.0
            if not backward:
                bound = (pencil_var_major[p] + pencil_var_minor[p]
                         + prepared_shape[p, 6] / (r * r))
            if backward:
                # The whole backward profile is smooth. Averaging its
                # isolated 1/sin(theta) factor creates a false antipodal pole.
                # Keep finite moments; covariance remains an O(C) truncation.
                weight = 1.0 / r
                averaged = False
            elif sine > 1e-300 and bound < 1e-4 * sine * sine:
                # far from the module direction: the shape correction is
                # below 1e-4 of the point value
                factor = 1.0 / sine
                weight = factor / r
                weight2 = factor2 / r
                averaged = False
            else:
                factor, theta, factor2, theta2, averaged, ok = pencil_radial_rule(
                    p, r, theta, sine, ray[1], ray[2], ray[3],
                    pencil_major, pencil_var_major, pencil_var_minor, prepared_shape,
                    smoothing_factor, gauss_tangent, gauss_weighted_sec2)
                if not ok:
                    continue
                weight = factor / r
                weight2 = factor2 / r
            first_weight, first_theta = weight, theta
            for node in range(2 if weight2 > 0.0 else 1):
                weight = first_weight if node == 0 else weight2
                theta = first_theta if node == 0 else theta2
                if (theta_thresholds.shape[0] == 0 or not math.isfinite(theta)
                        or theta > math.pi):
                    i_theta = scalar_theta_index(theta, theta_lo, log_theta_ratio)
                elif theta < theta_lo:
                    i_theta = 0
                elif theta_lut.shape[0] > 0:
                    bucket = np.int64(_float64_bits(theta) >> np.uint64(47)) - theta_first
                    i_theta = theta_lut[bucket]
                    while (i_theta < theta_thresholds.shape[0]
                           and theta >= theta_thresholds[i_theta]):
                        i_theta += 1
                else:
                    low = 0
                    high = theta_thresholds.shape[0]
                    while low < high:
                        middle = (low + high) // 2
                        if theta < theta_thresholds[middle]:
                            high = middle
                        else:
                            low = middle + 1
                    i_theta = low
                tau = ray[5]
                # Twenty-one radial bits cover the float64 log-radius range at
                # useful refinement steps. The former 11-bit field silently
                # merged every r>exp(1023*.005)~=166 m when log_r_step=.005.
                # Hemispheres use different moment normalizations and must not
                # share either a theta group or a (tau,r) quadrature batch.
                key = geometry_base_keys[geometry_id] + int(backward) * theta_bins + i_theta
                if 2 * (used + 1) > keys.shape[0]:
                    keys, values = _grow(keys, values, used)
                slot = _hash_slot(keys, key)
                if keys[slot] == -1:
                    keys[slot] = key
                    used += 1
                    # Only occupied rows are read or moved by _grow. Reset a row
                    # on insertion, preserving the original +0 initialization.
                    values[slot, :] = 0.0
                row = values[slot]
                if cubic_layout:
                    omega = _accumulate_cubic_moments_pair_prepared(
                        weight * pencil_w0[p], weight * pencil_w2[p],
                        look_powers[p, 0], look_powers[p, 1], look_powers[p, 2],
                        c3, averaged, row)
                else:
                    _fill_moments_averaged_pair(weight * pencil_w0[p], weight * pencil_w2[p],
                                                c1, c3, averaged, ipow, jpow, max_power,
                                                mom0, mom2, cpow, spow)
                    for m in range(n_mono):
                        row[m] += mom0[m]
                        row[n_mono + m] += mom2[m]
                    omega = abs(mom0[0])
                row[2 * n_mono] += omega
                row[2 * n_mono + 1] += omega * theta
                row[2 * n_mono + 2] += omega * r
                row[2 * n_mono + 3] += omega * ray[4]
                row[2 * n_mono + 4] += omega * tau
                row[2 * n_mono + 5] += omega * tau * tau
        # occupied slots, ordered so that one (tau, r) batch is contiguous
        order_keys = np.empty(used, dtype=np.int64)
        order_slots = np.empty(used, dtype=np.int64)
        k = 0
        for slot in range(keys.shape[0]):
            if keys[slot] != -1:
                order_keys[k] = keys[slot]
                order_slots[k] = slot
                k += 1
        order = np.argsort(order_keys)
        start = 0
        batches = 0
        while start < used:
            batch_key = order_keys[order[start]] // theta_bins
            backward = batch_key % 2 == 1
            stop = start
            omega_sum = 0.0
            r_sum = 0.0
            t_sum = 0.0
            tau_sum = 0.0
            tau2_sum = 0.0
            theta_small = math.pi
            while stop < used and order_keys[order[stop]] // theta_bins == batch_key:
                row = values[order_slots[order[stop]]]
                omega = row[2 * n_mono]
                if omega > 0.0:
                    omega_sum += omega
                    r_sum += row[2 * n_mono + 2]
                    t_sum += row[2 * n_mono + 3]
                    tau_sum += row[2 * n_mono + 4]
                    tau2_sum += row[2 * n_mono + 5]
                    theta_small = min(theta_small, row[2 * n_mono + 1] / omega)
                stop += 1
            if omega_sum > 0.0:
                r_b = r_sum / omega_sum
                if batch_tolerance_pe > 0.0:
                    estimate = _grouped_batch_charge_estimate(
                        values, order_slots, order, start, stop, n_mono, coef,
                        r_b, areas[d], g, mus, mut, s0w, s2w, backward)
                    # There cannot be more batches than occupied groups.
                    if estimate <= batch_tolerance_pe / used:
                        omission[d, 0] += 1.0
                        omission[d, 1] += estimate
                        start = stop
                        continue
                t_b = t_sum / omega_sum
                tau_mean = tau_sum / omega_sum
                tau_var = max(tau2_sum / omega_sum - tau_mean * tau_mean, 0.0)
                theta_small = max(theta_small, 1e-9)
                if backward:
                    sigma = max(0.5 * r_b * math.sin(theta_small) * y_g,
                                0.25 * r_b * (1.0 - math.cos(theta_small)))
                else:
                    sigma = _batch_sigma(r_b, math.sin(theta_small),
                                         math.tan(0.5 * theta_small), y_g)
                n = _batch_grid(r_b, sigma, 0., h, h, tail_exp, mut_min,
                                speed_max, math.inf, s_buf)
                g0[:n] = 0.0
                g2[:n] = 0.0
                for q in range(start, stop):
                    row = values[order_slots[order[q]]]
                    omega = row[2 * n_mono]
                    if omega <= 0.0:
                        continue
                    theta = max(row[2 * n_mono + 1] / omega, 1e-9)
                    for m in range(n_mono):
                        mom0[m] = row[m]
                        mom2[m] = row[n_mono + m]
                    sine = math.sin(theta)
                    _batch_add_moments(n, s_buf, r_b, sine, math.cos(theta), sine,
                                       mom0, mom2, coef, ipow, jpow, max_power, g,
                                       g0, g2, cpow, spow, backward)
                _batch_deposit(n, s_buf, g0, g2, r_b, t_b, tail_exp, mus, mut,
                               speed, s0w, s2w, edges, eedge, pre, acc, charge,
                               direct_bins, t_buf, h_buf,
                               math.sqrt(12.0 * tau_var), 0.0, 0.0, 0.0,
                               compute_bins)
                batches += 1
            start = stop
        if compute_bins:
            charge_out[d] = _finalize(edges, eedge, itab, pre, acc, charge,
                                      areas[d], bins_out[d])
        else:
            total_charge = 0.0
            for lam in range(n_lambda):
                total_charge += charge[lam]
            charge_out[d] = areas[d] * total_charge
        groups_out[d] = used
        batches_out[d] = batches


def screen_order1(receivers, looks, areas, origins, work,
                           pencil_pos, pencil_dir, pencil_w0, pencil_w2,
                           pencil_t, pencil_major, pencil_var_major,
                           pencil_var_minor, pencil_pos_cov, gauss_tau, gauss_w,
                           coef, ipow, jpow, max_power,
                           g, y_g, h, h_tail, tail_exp,
                           mus, mut, speed, s0w, s2w, edges, eedge, itab,
                           direct_bins, max_nodes, smoothing_factor,
                           theta_lo, theta_ratio, tau_step, log_r_step, speed_ref,
                           n_blocks, compute_bins=True, batch_tolerance_pe=0.0,
                           omission=None, *, hierarchy_parent, next_node):
    """Charge-only batches; selected parent or child represents each emission."""
    if compute_bins:
        raise ValueError("the coarse screen computes integrated charge only")
    n_recv = receivers.shape[0]
    n_pen = pencil_pos.shape[0]
    n_lambda = mus.shape[0]
    nb = edges.shape[0] - 1
    n_mono = coef.shape[0]
    charge_out = np.zeros(n_recv)
    bins_out = np.zeros((n_recv, nb if compute_bins else 0))
    groups_out = np.zeros(n_recv, dtype=np.int64)
    batches_out = np.zeros(n_recv, dtype=np.int64)
    mut_min = mut.min()
    speed_max = speed.max()
    n_values = 2 * n_mono + 6
    log_theta_ratio = math.log(1.0 + theta_ratio)
    theta_bins = 2 + int(math.log(math.pi / theta_lo) / log_theta_ratio)
    # Reject unsupported key resolutions instead of clamping/overflowing.
    if (log_r_step < 710.0 / ((1 << 20) - 2)
            or theta_bins > ((1 << 63) - 1) // (2 * (1 << 42))):
        raise ValueError('grouped radius/angular resolution exceeds integer key range')
    n_work = work.shape[0]
    omitted = np.zeros((n_recv, 2)) if omission is None else omission
    if n_work == 0 or n_blocks <= 0:
        return charge_out, bins_out, groups_out, batches_out
    from .prompt import _thread_count
    # One scratch allocation per worker. Static receiver strides preserve
    # per-OM arithmetic while avoiding repeated copies on a single thread.
    n_blocks = min(n_blocks, _thread_count())
    # Source-dependent preparation is identical across receiver blocks.
    # Build once, retain ownership here, and let all blocks read it.
    geometry_ids, representatives = _exact_pencil_geometry_ids(pencil_pos, pencil_t)
    prepared_shape = _prepare_pencil_shape(pencil_dir, pencil_major, pencil_pos_cov)
    gauss_tangent, gauss_weighted_sec2 = _prepare_gauss_inverse(gauss_tau, gauss_w)
    cubic_layout = _is_cubic_moment_layout(ipow, jpow, max_power)
    theta_thresholds = build_theta_thresholds(theta_lo, log_theta_ratio, theta_bins)
    theta_lut, theta_first = build_theta_bit_lut(theta_lo, theta_thresholds)
    common_look, shared_look_powers = _prepare_common_pencil_look(pencil_dir, looks, work)
    def run(block):
        _screen_block(block, receivers, looks, areas, origins, work, pencil_pos, pencil_dir, pencil_w0, pencil_w2, pencil_t, pencil_major, pencil_var_major, pencil_var_minor, pencil_pos_cov, gauss_tau, gauss_w, coef, ipow, jpow, max_power, g, y_g, h, h_tail, tail_exp, mus, mut, speed, s0w, s2w, edges, eedge, itab, direct_bins, max_nodes, smoothing_factor, theta_lo, theta_ratio, tau_step, log_r_step, speed_ref, n_blocks, n_recv, n_pen, n_lambda, nb, n_mono, charge_out, bins_out, groups_out, batches_out, mut_min, speed_max, n_values, log_theta_ratio, theta_bins, n_work, compute_bins, batch_tolerance_pe, omitted, geometry_ids, representatives, prepared_shape, gauss_tangent, gauss_weighted_sec2, cubic_layout, theta_thresholds, theta_lut, theta_first, common_look, shared_look_powers, hierarchy_parent, next_node)
    workers = n_blocks
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(run, range(n_blocks)))
    else:
        for block in range(n_blocks):
            run(block)
    return charge_out, bins_out, groups_out, batches_out
