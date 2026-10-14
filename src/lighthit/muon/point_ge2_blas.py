"""Bounded batched BLAS contraction of the original >=2 point operator.

Only floating-point reassociation changes: spatial points, harmonics, radial
cubic coefficients, wavelengths and all Fourier frequencies are retained.
BLAS thread count is controlled by the caller/environment.
"""
from collections import OrderedDict
import numpy as np
from numba import njit, prange
from .._directional_numba import (_wigner_into, _harmonics_into as _detector_harmonics,
                                  _start_norm_table, _parity)

_MATRIX_CACHE = OrderedDict()
_MATRIX_CACHE_BYTES = 0
_MATRIX_CACHE_LIMIT = 64 * 2**20


def clear_matrix_cache():
    """Drop the bounded cache, useful for reproducible cold measurements."""
    global _MATRIX_CACHE_BYTES
    _MATRIX_CACHE.clear()
    _MATRIX_CACHE_BYTES = 0


def matrix_cache_info():
    """Entries and bytes of the persistent coefficient-matrix cache."""
    return dict(entries=len(_MATRIX_CACHE), bytes=_MATRIX_CACHE_BYTES,
                limit_bytes=_MATRIX_CACHE_LIMIT)


def _immutable(array):
    # A readonly view of a writable base is still mutable through that base.
    """True when the array and all its bases are read-only."""
    current = array
    while isinstance(current, np.ndarray):
        if current.flags.writeable:
            return False
        current = current.base
    return True


def _coefficient_matrix(coefficients, band, interval, local_cache):
    """Real (re, im interleaved) matrix of one band and radial interval, cached."""
    global _MATRIX_CACHE_BYTES
    persistent = _immutable(coefficients)
    key = (coefficients.__array_interface__['data'][0],
           coefficients.shape, coefficients.strides, coefficients.dtype.str,
           int(band), int(interval))
    cache = _MATRIX_CACHE if persistent else local_cache
    if key in cache:
        item = cache.pop(key)
        cache[key] = item
        return item[1]
    slab = coefficients[band, interval]
    matrix = np.empty((np.prod(slab.shape[:-1]), 2 * slab.shape[-1]), np.float64)
    flat = slab.reshape(-1, slab.shape[-1])
    matrix[:, 0::2] = flat.real
    matrix[:, 1::2] = flat.imag
    # Retaining the source object prevents stale entries after allocator reuse.
    cache[key] = (coefficients, matrix)
    if persistent:
        _MATRIX_CACHE_BYTES += matrix.nbytes
        while _MATRIX_CACHE_BYTES > _MATRIX_CACHE_LIMIT and cache:
            _, (_, removed) = cache.popitem(last=False)
            _MATRIX_CACHE_BYTES -= removed.nbytes
    else:
        # Mutable packs may be reused only during this call, bounded likewise.
        while sum(item[1].nbytes for item in cache.values()) > _MATRIX_CACHE_LIMIT:
            cache.popitem(last=False)
    return matrix


@njit(cache=True, nogil=True)
def _supported_pairs(points, receivers, emitting, out_scale, low, high, log_r, begin, end):
    """Use the scalar kernel's sqrt/log and interval selection at cache knots."""
    capacity = (end-begin)*len(receivers)
    cells = np.empty(capacity, np.int64)
    receiver_ids = np.empty(capacity, np.int64)
    radii = np.empty(capacity, np.float64)
    intervals = np.empty(capacity, np.int64)
    offsets = np.empty(capacity, np.float64)
    count = 0
    for receiver in range(len(receivers)):
        for cell in range(begin, end):
            dx = receivers[receiver, 0] - points[cell, 0]
            dy = receivers[receiver, 1] - points[cell, 1]
            dz = receivers[receiver, 2] - points[cell, 2]
            radius = np.sqrt(dx*dx + dy*dy + dz*dz)
            if radius < low or radius > high:
                continue
            active = False
            for band in range(len(emitting)):
                active |= emitting[band, cell] and out_scale[band, receiver] != 0.
            if not active:
                continue
            logr = np.log(radius)
            interval = np.searchsorted(log_r, logr, side='right') - 1
            if interval < 0:
                interval = 0
            if interval > len(log_r)-2:
                interval = len(log_r)-2
            cells[count] = cell
            receiver_ids[count] = receiver
            radii[count] = radius
            intervals[count] = interval
            offsets[count] = logr-log_r[interval]
            count += 1
    return cells[:count], receiver_ids[:count], radii[:count], intervals[:count], offsets[:count]


@njit(cache=True, nogil=True, parallel=True)
def _angular_weights(points, receivers, looks, angular, cells, receiver_ids,
                     emitting, out_scale, starts, degree, max_m, max_out,
                     alpha, pair_lambda, pair_mu, start_norm, jmax):
    """Rotated source harmonics contracted with the OM acceptance, per pair and band;
    one row of the design matrix per point-OM pair.
    """
    nlam = angular.shape[0]
    npair = len(pair_lambda)
    width = 2 * max_out + 1
    weights = np.zeros((nlam, len(cells), (degree + 1) * npair), np.float64)
    root2 = np.sqrt(2.0)
    # Pairs are independent: chunks run in parallel, each with its own scratch.
    nchunk = max(1, min(len(cells), 64))
    for chunk in prange(nchunk):
        wigner = np.empty((degree + 1, width, max_m + 1), np.float64)
        detector = np.empty((max_out + 1) * (max_out + 1), np.float64)
        erow = np.empty((width, 2 * max_m + 1), np.float64)
        rotated = np.empty((nlam, width), np.float64)
        cosines = np.empty(max_m + 1, np.float64)
        sines = np.empty(max_m + 1, np.float64)
        active_bands = np.empty(nlam, np.bool_)
        for index in range(chunk * len(cells) // nchunk, (chunk + 1) * len(cells) // nchunk):
            cell = cells[index]
            receiver = receiver_ids[index]
            dx = receivers[receiver, 0] - points[cell, 0]
            dy = receivers[receiver, 1] - points[cell, 1]
            dz = receivers[receiver, 2] - points[cell, 2]
            radius = np.sqrt(dx * dx + dy * dy + dz * dz)
            for lam in range(nlam):
                active_bands[lam] = emitting[lam, cell] and out_scale[lam, receiver] != 0.
            cosine = dz / radius
            if cosine > 1.0:
                cosine = 1.0
            elif cosine < -1.0:
                cosine = -1.0
            phi = np.arctan2(dy, dx)
            transverse = np.sqrt(dx * dx + dy * dy)
            sine = transverse / radius
            cp = 1.0 if transverse == 0.0 else dx / transverse
            sp = 0.0 if transverse == 0.0 else dy / transverse
            for n in range(max_m + 1):
                cosines[n] = np.cos(n * phi)
                sines[n] = np.sin(n * phi)
            lx = looks[receiver, 0]
            ly = looks[receiver, 1]
            lz = looks[receiver, 2]
            a1 = cosine * cp * lx + cosine * sp * ly - sine * lz
            a2 = -sp * lx + cp * ly
            a3 = sine * cp * lx + sine * sp * ly + cosine * lz
            _detector_harmonics(a1, a2, a3, max_out, detector)
            _wigner_into(degree, max_out, max_m, cosine, start_norm, jmax, wigner)
            for ell in range(degree + 1):
                for a in range(width):
                    for b in range(2 * max_m + 1):
                        erow[a, b] = 0.0
                erow[max_out, max_m] = wigner[ell, max_out, 0]
                top_n = min(ell, max_m)
                top_p = min(ell, max_out)
                for n in range(1, top_n + 1):
                    base = root2 * _parity(n) * wigner[ell, max_out, n]
                    erow[max_out, max_m + n] = base * cosines[n]
                    erow[max_out, max_m - n] = base * sines[n]
                for p in range(1, top_p + 1):
                    erow[max_out + p, max_m] = (root2 * _parity(p)
                                                * wigner[ell, max_out + p, 0])
                    for n in range(1, top_n + 1):
                        cross = _parity(p) * wigner[ell, max_out + p, n]
                        straight = wigner[ell, max_out - p, n]
                        cc = _parity(n) * (cross + straight)
                        ss = _parity(n) * (cross - straight)
                        erow[max_out + p, max_m + n] = cc * cosines[n]
                        erow[max_out + p, max_m - n] = cc * sines[n]
                        erow[max_out - p, max_m + n] = -ss * sines[n]
                        erow[max_out - p, max_m - n] = ss * cosines[n]
                keep = min(ell, max_m)
                centre = starts[ell] + min(ell, max_m)
                for lam in range(nlam):
                    if not active_bands[lam]:
                        continue
                    for a in range(width):
                        rotated[lam, a] = 0.0
                for a in range(max_out - top_p, max_out + top_p + 1):
                    for b in range(max_m - keep, max_m + keep + 1):
                        factor = erow[a, b]
                        if factor == 0.0:
                            continue
                        column = centre + (b - max_m)
                        for lam in range(nlam):
                            if not active_bands[lam]:
                                continue
                            rotated[lam, a] += factor * angular[lam, cell, column]
                for lam in range(nlam):
                    if not active_bands[lam]:
                        continue
                    for p in range(npair):
                        lam_detector = pair_lambda[p]
                        mu_pair = pair_mu[p]
                        if mu_pair > top_p:
                            continue
                        detector_centre = lam_detector * lam_detector + lam_detector
                        positive = alpha[lam_detector] * detector[detector_centre + mu_pair]
                        negative = (alpha[lam_detector] * detector[detector_centre - mu_pair]
                                    if mu_pair else 0.0)
                        angular_weight = (positive * rotated[lam, max_out + mu_pair]
                                          + negative * rotated[lam, max_out - mu_pair])
                        weights[lam, index, ell * npair + p] = angular_weight
    return weights


@njit(cache=True, nogil=True)
def _accumulate(result, values, cells, receiver_ids, radius, times, out_scale,
                absorption, velocity, omega, flight, band):
    """Add attenuation, 1/(4 pi r^2) and the emission and flight phase of each pair."""
    for row in range(len(cells)):
        receiver = receiver_ids[row]
        scale = (out_scale[band, receiver] * np.exp(-absorption[band] * radius[row])
                 / (4 * np.pi * radius[row] * radius[row]))
        for w in range(len(omega)):
            angle = omega[w] * (times[cells[row]] + radius[row] / velocity[band]) if flight else omega[w] * times[cells[row]]
            response = values[row, 2*w] + 1j * values[row, 2*w+1]
            phase = scale * (np.cos(angle) + 1j * np.sin(angle))
            result[receiver, w] += response * phase


def apply_point_spectral_blas(bundle, frame, points_m, angular, times_ns,
                              receivers_m, looks, alpha, out_scale, *,
                              azimuthal_degree=4, radial_window_m=(3., 300.),
                              block_size=128, pair_budget_bytes=32 * 2**20,
                              precision='float64'):
    """Apply the original point operator using bounded real BLAS matrices.

    The coefficient LRU is limited to 64 MiB. Readonly arrays whose ndarray
    bases are also readonly may persist across calls; mutable arrays use a
    fresh per-call LRU, so modifications are always observed. No double copy
    of the entire spectral pack is formed. Pair inventory temporaries are
    bounded by approximately twice ``pair_budget_bytes``; block temporaries
    are ``8*block_size*((bands+4)*(degree+1)*pairs + 2*frequencies)`` bytes.
    No array with shape (all points, all OMs, all frequencies) is formed.

    ``precision='float32'`` multiplies in single precision against a zero-copy
    float32 view of complex64 coefficients (interleaved real/imaginary parts
    are already the matrix layout), so no coefficient matrix is rebuilt. Pair
    geometry, angular weights and accumulation stay float64. The pack itself
    is single precision; the default 'float64' path is unchanged.
    """
    if precision not in ('float64', 'float32'):
        raise ValueError("precision must be 'float64' or 'float32'")
    if np.iscomplexobj(angular):
        raise ValueError('point angular coefficients must be real')
    if (not isinstance(azimuthal_degree, (int, np.integer))
            or isinstance(azimuthal_degree, (bool, np.bool_)) or azimuthal_degree < 0):
        raise ValueError('azimuthal_degree must be a nonnegative integer')
    for value, name, minimum in [(block_size, 'block_size', 1),
                                 (pair_budget_bytes, 'pair_budget_bytes', 1024)]:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f'{name} must be an integer >= {minimum}')
    values = np.ascontiguousarray(angular, dtype=float)
    points = np.asarray(points_m, float)
    times = np.asarray(times_ns, float)
    receivers = np.atleast_2d(np.asarray(receivers_m, float))
    normals = np.atleast_2d(np.asarray(looks, float))
    scale = np.ascontiguousarray(out_scale, dtype=float)
    if (times.ndim != 1 or points.ndim != 2 or points.shape[1] != 3
            or receivers.ndim != 2 or receivers.shape[1] != 3):
        raise ValueError('points and receivers require three coordinates; clocks must be one-dimensional')
    degree = int(bundle['degree'])
    max_m = min(degree, int(azimuthal_degree))
    starts = np.r_[0, np.cumsum([2*min(l, max_m)+1 for l in range(degree+1)])].astype(np.int64)
    nlam = len(bundle['absorption'])
    nfreq = len(bundle['omega'])
    if (values.shape != (nlam, len(points), int(starts[-1]))
            or points.shape != (len(times), 3) or normals.shape != receivers.shape
            or scale.shape != (nlam, len(receivers))):
        raise ValueError('incompatible point/spectral/angular/detector shapes')
    if not all(np.isfinite(a).all() for a in [values, points, times, receivers, normals, scale]):
        raise ValueError('point contraction requires finite values')
    if not len(points):
        return np.zeros((nfreq, len(receivers)), complex)
    low, high = map(float, radial_window_m)
    log_r = np.asarray(bundle['log_r'])
    if (not np.isfinite([low, high]).all() or low < np.exp(log_r[0])-1e-9
            or high > np.exp(log_r[-1])+1e-9 or not low < high):
        raise ValueError('radial window outside cache')
    max_out = int(bundle['acceptance_degree'])
    jmax = max(max_m, max_out)
    alpha = np.ascontiguousarray(np.asarray(alpha, float)[:max_out+1])
    if alpha.shape != (max_out+1,) or not np.isfinite(alpha).all():
        raise ValueError('finite acceptance coefficients must cover the detector degree')
    centre = np.asarray(frame.centre_m)
    reference = float(times.min())
    points = np.ascontiguousarray(frame.rotate(points-centre))
    receivers = np.ascontiguousarray(frame.rotate(receivers-centre))
    normals = np.ascontiguousarray(frame.rotate(normals))
    times = np.ascontiguousarray(times-reference)
    emitting = np.ascontiguousarray(np.any(values != 0, axis=2))
    coefficients = np.asarray(bundle['coefficients'])
    single = precision == 'float32'
    if single and coefficients.dtype != np.complex64:
        raise ValueError('float32 precision requires complex64 coefficients')
    pair_lambda = np.asarray(bundle['pair_lambda'])
    pair_mu = np.asarray(bundle['pair_mu'])
    start_norm = _start_norm_table(jmax)
    result = np.zeros((len(receivers), nfreq), complex)
    local_cache = OrderedDict()
    # All receivers share each interval matrix while each source chunk is live.
    chunk = min(8192, max(1, int(pair_budget_bytes)//max(64*len(receivers), 1)))
    for begin in range(0, len(points), chunk):
        end = min(begin+chunk, len(points))
        cells, receiver_ids, radius, intervals, offsets = _supported_pairs(
            points, receivers, emitting, scale, low, high, log_r, begin, end)
        if not len(cells):
            continue
        order = np.argsort(intervals, kind='stable')
        boundaries = np.r_[0, np.flatnonzero(np.diff(intervals[order]))+1, len(order)]
        for lo, hi in zip(boundaries[:-1], boundaries[1:]):
            interval = int(intervals[order[lo]])
            for row in range(int(lo), int(hi), block_size):
                take = order[row:min(row+block_size, int(hi))]
                bcells = np.ascontiguousarray(cells[take])
                breceivers = np.ascontiguousarray(receiver_ids[take])
                bradius = np.ascontiguousarray(radius[take])
                t = offsets[take]
                powers = np.column_stack((t*t*t, t*t, t, np.ones(len(t))))
                weights = _angular_weights(points, receivers, normals, values, bcells,
                    breceivers, emitting, scale, starts, degree, max_m, max_out, alpha,
                    pair_lambda, pair_mu, start_norm, jmax)
                for band in range(nlam):
                    if not np.any(weights[band]):
                        continue
                    design = (weights[band, :, :, None]*powers[:, None, :]).reshape(len(take), -1)
                    if single:
                        matrix = coefficients[band, interval].view(np.float32).reshape(design.shape[1], -1)
                        product = (design.astype(np.float32) @ matrix).astype(np.float64)
                    else:
                        matrix = _coefficient_matrix(coefficients, band, interval, local_cache)
                        product = design @ matrix
                    _accumulate(result, product, bcells, breceivers, bradius, times,
                        scale, bundle['absorption'], bundle['speed'], bundle['omega'],
                        bundle['phase'] == 'flight', band)
    return result.T * np.exp(1j*np.asarray(bundle['omega'])*reference)[:, None]


def apply_point_spectral_blas_multi(bundle, items, receivers_m, looks, alpha, out_scale, *,
                                    azimuthal_degree=4, radial_window_m=(3., 300.),
                                    block_size=2048, precision='float32'):
    """Sum of ``apply_point_spectral_blas`` over several sources in one pass.

    ``items`` holds ``(frame, points_m, angular, times_ns, targets)`` per
    source, with ``targets`` indexing ``receivers_m``/``looks``/``out_scale``
    columns. Each source keeps its own frame, clock reference and pairs; pairs
    of all sources are grouped by radial interval, so each (band, interval)
    multiplication covers many sources. The result equals the sum of single
    calls up to summation order (and single precision when requested).
    Returns the (frequency, receiver) spectrum for all ``receivers_m``.
    """
    if precision not in ('float64', 'float32'):
        raise ValueError("precision must be 'float64' or 'float32'")
    receivers_all = np.atleast_2d(np.asarray(receivers_m, float))
    normals_all = np.atleast_2d(np.asarray(looks, float))
    scale_all = np.ascontiguousarray(out_scale, dtype=float)
    degree = int(bundle['degree'])
    max_m = min(degree, int(azimuthal_degree))
    starts = np.r_[0, np.cumsum([2*min(l, max_m)+1 for l in range(degree+1)])].astype(np.int64)
    nlam = len(bundle['absorption'])
    omega = np.asarray(bundle['omega'])
    nfreq = len(omega)
    low, high = map(float, radial_window_m)
    log_r = np.asarray(bundle['log_r'])
    if (not np.isfinite([low, high]).all() or low < np.exp(log_r[0])-1e-9
            or high > np.exp(log_r[-1])+1e-9 or not low < high):
        raise ValueError('radial window outside cache')
    max_out = int(bundle['acceptance_degree'])
    jmax = max(max_m, max_out)
    alpha = np.ascontiguousarray(np.asarray(alpha, float)[:max_out+1])
    coefficients = np.asarray(bundle['coefficients'])
    single = precision == 'float32'
    if single and coefficients.dtype != np.complex64:
        raise ValueError('float32 precision requires complex64 coefficients')
    pair_lambda = np.asarray(bundle['pair_lambda'])
    pair_mu = np.asarray(bundle['pair_mu'])
    start_norm = _start_norm_table(jmax)
    points, values, times, rows_rec, rows_norm, rows_scale = [], [], [], [], [], []
    row_target, row_reference = [], []
    pair_parts = []
    point_offset = row_offset = 0
    for frame, points_m, angular, times_ns, targets in items:
        targets = np.asarray(targets, np.int64)
        local = np.ascontiguousarray(angular, dtype=float)
        if not len(targets) or not len(local[0]):
            continue
        if local.shape[:2] != (nlam, len(points_m)) or local.shape[2] != int(starts[-1]):
            raise ValueError('incompatible point/spectral/angular shapes')
        centre = np.asarray(frame.centre_m)
        clock = np.asarray(times_ns, float)
        reference = float(clock.min())
        pts = np.ascontiguousarray(frame.rotate(np.asarray(points_m, float)-centre))
        rec = np.ascontiguousarray(frame.rotate(receivers_all[targets]-centre))
        nor = np.ascontiguousarray(frame.rotate(normals_all[targets]))
        scl = np.ascontiguousarray(scale_all[:, targets])
        emitting = np.ascontiguousarray(np.any(local != 0, axis=2))
        cells, receiver_ids, radius, intervals, offsets = _supported_pairs(
            pts, rec, emitting, scl, low, high, log_r, 0, len(pts))
        if len(cells):
            pair_parts.append((cells+point_offset, receiver_ids+row_offset, radius, intervals, offsets))
        points.append(pts); values.append(local); times.append(clock-reference)
        rows_rec.append(rec); rows_norm.append(nor); rows_scale.append(scl)
        row_target.append(targets); row_reference.append(np.full(len(targets), reference))
        point_offset += len(pts); row_offset += len(targets)
    out = np.zeros((nfreq, len(receivers_all)), complex)
    if not pair_parts:
        return out
    points = np.ascontiguousarray(np.concatenate(points))
    values = np.ascontiguousarray(np.concatenate(values, axis=1))
    times = np.ascontiguousarray(np.concatenate(times))
    receivers = np.ascontiguousarray(np.concatenate(rows_rec))
    normals = np.ascontiguousarray(np.concatenate(rows_norm))
    scale = np.ascontiguousarray(np.concatenate(rows_scale, axis=1))
    row_target = np.concatenate(row_target); row_reference = np.concatenate(row_reference)
    emitting = np.ascontiguousarray(np.any(values != 0, axis=2))
    cells, receiver_ids, radius, intervals, offsets = (np.concatenate(part) for part in zip(*pair_parts))
    result = np.zeros((len(receivers), nfreq), complex)
    local_cache = OrderedDict()
    order = np.argsort(intervals, kind='stable')
    boundaries = np.r_[0, np.flatnonzero(np.diff(intervals[order]))+1, len(order)]
    for lo, hi in zip(boundaries[:-1], boundaries[1:]):
        interval = int(intervals[order[lo]])
        for row in range(int(lo), int(hi), block_size):
            take = order[row:min(row+block_size, int(hi))]
            bcells = np.ascontiguousarray(cells[take])
            breceivers = np.ascontiguousarray(receiver_ids[take])
            bradius = np.ascontiguousarray(radius[take])
            t = offsets[take]
            powers = np.column_stack((t*t*t, t*t, t, np.ones(len(t))))
            weights = _angular_weights(points, receivers, normals, values, bcells,
                breceivers, emitting, scale, starts, degree, max_m, max_out, alpha,
                pair_lambda, pair_mu, start_norm, jmax)
            for band in range(nlam):
                if not np.any(weights[band]):
                    continue
                design = (weights[band, :, :, None]*powers[:, None, :]).reshape(len(take), -1)
                if single:
                    matrix = coefficients[band, interval].view(np.float32).reshape(design.shape[1], -1)
                    product = (design.astype(np.float32) @ matrix).astype(np.float64)
                else:
                    matrix = _coefficient_matrix(coefficients, band, interval, local_cache)
                    product = design @ matrix
                _accumulate(result, product, bcells, breceivers, bradius, times,
                    scale, bundle['absorption'], bundle['speed'], omega,
                    bundle['phase'] == 'flight', band)
    result *= np.exp(1j*omega[None, :]*row_reference[:, None])
    np.add.at(out.T, row_target, result)
    return out
