"""Axial-slice shower basis for scattering order >=2.

The source of a shower is written in its own axial frame (``AxisFrame``:
photon-weighted centre and principal axis) as

    S(x, u, omega) = sum_p sum_(l,m) c_(p,lm)(omega) delta(x - x_p) Y_lm(u),

on a small set of points ``x_p``: slices ``z_r = z_0 + r h`` on the axis and,
where the slice has transverse extent, four more points at ``+-s`` along the
two transverse frame vectors.  ``Y_lm`` are real harmonics in the axial frame
with ``l <= L`` and ``|m| <= M``.  The response of module ``d`` is

    N_d^(>=2)(omega) = sum_(p,lm) c_(p,lm)(omega) K_(d,p,lm)^(>=2)(omega),

where ``K`` is the exact directional RTE Green function of the existing cache
(:class:`lighthit._directional_numba.PreparedDirectionalKernel`).  Nothing
about the medium, the detector, the angular variable or the frequency grid is
approximated.  The coefficients are moments of the G4 chords,

    c_(r,lm)^(k)(omega) = sum_i Q_i P_l(mu_C,i) Y_lm(u_i) w_r(z_i) m_k(x_i) e^{i omega t_i},

with ``w_r`` the linear (two-slice) or cubic Lagrange (four-slice) weights of
the chord point on the axis and ``m_k = 1, x, y, x^2, y^2`` its transverse
moments.  The five stencil weights reproduce all five moments of every
``(slice, l, m, omega)`` coefficient exactly; slices whose rms transverse
extent is below ``stencil_threshold_m`` keep a single point at their photon
centroid (zeroth and first moments of the total light).  The error
terms are the axial interpolation (O(h^2) linear, O(h^4) cubic), the
unmatched third and mixed second transverse moments, and the time lattice
used for ``e^{i omega t}``.  They are measured in
``docs/research/shower-basis.md``.
"""
from __future__ import annotations

import numpy as np

try:
    from numba import get_num_threads, njit, prange
except ImportError as exc:  # pragma: no cover
    raise ImportError("_shower_basis requires the optional 'accelerate' extra") from exc

from .experimental.axial_fast import _harmonics_into, _layout
from .experimental.axial_source import AxialSource, AxisFrame

__all__ = ["compile_axial_slices"]


@njit(cache=True, nogil=True, parallel=True)
def earliest_element_arrival(receivers, starts, directions, lengths, start_ns,
                             group_index):
    """Per-OM lower bound on arrival from any moving source segment.

    The emission time is at least ``start_ns`` and the optical path is at
    least the distance to the segment.  This bound keeps a long source's
    readout origin near its locally relevant light instead of tying every OM
    to the first element of the entire event.
    """
    result = np.empty(len(receivers), np.float64)
    flight = float(group_index) / 0.299792458
    for receiver in prange(len(receivers)):
        earliest = np.inf
        for element in range(len(lengths)):
            dx = receivers[receiver, 0] - starts[element, 0]
            dy = receivers[receiver, 1] - starts[element, 1]
            dz = receivers[receiver, 2] - starts[element, 2]
            along = (dx * directions[element, 0] + dy * directions[element, 1]
                     + dz * directions[element, 2])
            along = min(max(along, 0.0), lengths[element])
            dx -= along * directions[element, 0]
            dy -= along * directions[element, 1]
            dz -= along * directions[element, 2]
            arrival = start_ns[element] + flight * np.sqrt(dx * dx + dy * dy + dz * dz)
            if arrival < earliest:
                earliest = arrival
        result[receiver] = earliest
    return result


@njit(cache=True, nogil=True, parallel=True)
def _cone_harmonics(vectors, cones, starts, aa, bb, max_m, out):
    """out[e, c] = P_l(mu_C,e) Y_lm(u_e), parallel over elements."""
    degree = len(starts) - 2
    for e in prange(vectors.shape[0]):
        _harmonics_into(vectors[e, 0], vectors[e, 1], vectors[e, 2], degree, max_m,
                        starts, aa, bb, out[e])
        prev = 1.0
        current = cones[e]
        for ell in range(degree + 1):
            if ell == 0:
                p = 1.0
            elif ell == 1:
                p = current
            else:
                p = ((2 * ell - 1) * cones[e] * current - (ell - 1) * prev) / ell
                prev = current
                current = p
            for c in range(starts[ell], starts[ell + 1]):
                out[e, c] *= p


@njit(cache=True, nogil=True, parallel=True)
def _deposit_elements(rows, weights, factors, harmonics, out, blocks):
    """out[row, k, c] += weights[e, j] * factors[e, k] * harmonics[e, c].

    Parallel over channel blocks, so threads never write the same entry.
    """
    count, links = rows.shape
    nk = factors.shape[1]
    nch = out.shape[2]
    for block in prange(blocks):
        c_lo = (nch * block) // blocks
        c_hi = (nch * (block + 1)) // blocks
        for e in range(count):
            for j in range(links):
                row = rows[e, j]
                w = weights[e, j]
                if w == 0.0:
                    continue
                for k in range(nk):
                    f = w * factors[e, k]
                    if f == 0.0:
                        continue
                    for c in range(c_lo, c_hi):
                        out[row, k, c] += f * harmonics[e, c]


def _axial_weights(r, order):
    """Slice shifts and weights of a point at fractional lattice offset ``r``."""
    if order == 1:
        return (0, 1), (1 - r, r)
    return ((-1, 0, 1, 2),
            (-r * (r - 1) * (r - 2) / 6, (r + 1) * (r - 1) * (r - 2) / 2,
             -(r + 1) * r * (r - 2) / 2, (r + 1) * r * (r - 1) / 6))


def compile_axial_slices(fields, degree, omega_per_ns, *, azimuthal_degree=4,
                         cell_m=0.25, axial_order=3, transverse="stencil",
                         stencil_m=0.2, stencil_threshold_m=0.0,
                         time_step_ns=0.02, time_deposit="linear", element_order="auto",
                         frame=None, chunk=65536, deposit_method="gemm", min_group=32):
    """Compile the axial-slice basis coefficients of one or more light fields.

    ``fields`` is one element set or a sequence of element sets that differ
    only in their photon weights (the two Cherenkov fields).

    ``transverse="collapse"`` moves every emission point onto the axis;
    ``"stencil"`` keeps the zeroth, first and diagonal second transverse
    moments exactly with five points per slice (see the module docstring).
    ``axial_order`` is 1 (linear, O(h^2)) or 3 (cubic Lagrange, O(h^4);
    weights may be negative and are kept).  ``e^{i omega t}`` is deposited on
    a uniform lattice of ``time_step_ns``: ``time_deposit="linear"`` on the
    two neighbouring ticks (relative error at most ``(omega dt)**2 / 8``) or
    ``"nearest"`` on one (half the links; the mean phase error over a tick
    is ``(omega dt)**2 / 24``).  ``element_order="auto"`` uses the
    chord midpoint when every chord is shorter than ``cell_m / 10`` and two
    Gauss nodes otherwise.

    Returns one :class:`AxialSource` per field (a single one for a single
    field), directly usable by :meth:`PreparedDirectionalKernel.apply`.
    """
    single = not isinstance(fields, (list, tuple))
    fields = [fields] if single else list(fields)
    elements = fields[0]
    if not (cell_m > 0 and time_step_ns > 0 and stencil_m > 0 and stencil_threshold_m >= 0):
        raise ValueError("cell_m, time_step_ns and stencil_m must be positive")
    if axial_order not in (1, 3):
        raise ValueError("axial_order must be 1 or 3")
    if time_deposit not in ("linear", "nearest"):
        raise ValueError("time_deposit must be 'linear' or 'nearest'")
    if transverse not in ("collapse", "stencil"):
        raise ValueError("transverse must be 'collapse' or 'stencil'")
    for other in fields[1:]:
        if (len(other) != len(elements)
                or not np.array_equal(other.start_m, elements.start_m)
                or not np.array_equal(other.direction, elements.direction)):
            raise ValueError("fields must share their elements")
    frame = frame or AxisFrame.of(elements)
    omega = np.atleast_1d(np.asarray(omega_per_ns, float))
    photons = np.stack([np.asarray(f.photons, float) for f in fields])
    count = len(elements)
    max_m = min(int(degree), int(azimuthal_degree))
    starts, aa, bb, kept, channel_degree = _layout(int(degree), max_m)
    nch = len(kept)
    if element_order == "auto":
        element_order = 1 if float(np.max(elements.length_m)) <= cell_m / 10 else 2
    nodes, node_weights = np.polynomial.legendre.leggauss(int(element_order))
    nodes, node_weights = (nodes + 1) / 2, node_weights / 2
    reference = float(np.asarray(elements.start_ns).min())
    margin = 1 if axial_order == 1 else 2
    z_origin = float(frame.coordinates(elements)[0].min()) - margin * cell_m
    moments = 1 if transverse == "collapse" else 5
    nf = len(fields)
    coordinates = [frame.coordinates(elements, fraction) for fraction in nodes]
    time_max = max(float(np.max(c[3])) for c in coordinates)
    kmax = int(np.floor((time_max - reference) / time_step_ns)) + 3
    shifts_count = 2 if axial_order == 1 else 4
    ticks = 2 if time_deposit == "linear" else 1
    links = ticks * shifts_count
    keys, link_weights, factors = [], [], []
    for (z, offset, _, time), node_weight in zip(coordinates, node_weights):
        if moments == 1:
            mom = np.ones((count, 1))
        else:
            x = offset @ frame.first
            y = offset @ frame.second
            mom = np.stack([np.ones(count), x, y, x * x, y * y], axis=1)
        factors.append((photons.T[:, :, None] * mom[:, None, :]).reshape(count, -1)
                       * node_weight)
        zl = (z - z_origin) / cell_m
        zb = np.floor(zl).astype(np.int64)
        tl = (np.asarray(time, float) - reference) / time_step_ns
        if ticks == 2:
            tb = np.floor(tl).astype(np.int64)
            tr = tl - tb
            tick_weights = ((0, 1 - tr), (1, tr))
        else:
            tb = np.rint(tl).astype(np.int64)
            tick_weights = ((0, 1.0),)
        shifts, zw = _axial_weights(zl - zb, axial_order)
        key = np.empty((count, links), np.int64)
        weight = np.empty((count, links))
        j = 0
        for shift, wz in zip(shifts, zw):
            for dt, wt in tick_weights:
                key[:, j] = (zb + shift) * kmax + tb + dt
                weight[:, j] = wz * wt
                j += 1
        keys.append(key)
        link_weights.append(weight)
    keys = np.concatenate(keys, axis=1)
    span_keys = int(keys.max()) + 1
    if span_keys <= 1 << 28 and int(keys.min()) >= 0:
        # dense key space (slices x ticks): occupancy and prefix sum, O(N)
        occupied = np.zeros(span_keys, bool)
        occupied[keys.ravel()] = True
        unique = np.flatnonzero(occupied)
        position = np.cumsum(occupied) - 1
        rows = position[keys]
        del occupied, position
    else:
        unique, inverse = np.unique(keys, return_inverse=True)
        rows = inverse.reshape(count, -1).astype(np.int64)
    cell_of = unique // kmax
    tick_of = unique % kmax
    slices, row_start = np.unique(cell_of, return_index=True)
    row_start = np.append(row_start, len(unique)).astype(np.int64)
    deposit = np.zeros((len(unique), nf * moments, nch))
    vectors = np.ascontiguousarray(frame.rotate(elements.direction))
    cones = np.ascontiguousarray(elements.cone_cosine, dtype=float)
    blocks = max(1, int(get_num_threads()))
    nk = nf * moments
    for n in range(len(nodes)):
        span = slice(n * links, (n + 1) * links)
        node_rows = rows[:, span]
        if deposit_method == "loop":
            harmonics = np.empty((min(count, chunk), nch))
            for begin in range(0, count, chunk):
                end = min(count, begin + chunk)
                _cone_harmonics(vectors[begin:end], cones[begin:end], starts, aa, bb,
                                max_m, harmonics[:end - begin])
                _deposit_elements(np.ascontiguousarray(node_rows[begin:end]),
                                  np.ascontiguousarray(link_weights[n][begin:end]),
                                  np.ascontiguousarray(factors[n][begin:end]),
                                  harmonics[:end - begin], deposit, blocks)
            continue
        # Chord points with the same base slice and time tick share all their
        # links: deposit each such group as one dense product
        #   (links*k, n_group) @ (n_group, channels).
        order = np.argsort(node_rows[:, 0], kind="stable")
        group_key = node_rows[order, 0]
        edges = np.flatnonzero(np.diff(group_key)) + 1
        group_start = np.concatenate(([0], edges))
        group_end = np.concatenate((edges, [count]))
        begin_group = 0
        while begin_group < len(group_start):
            # whole groups, about ``chunk`` chord points at a time
            first = group_start[begin_group]
            last_group = int(np.searchsorted(group_end, first + chunk, side="right"))
            last_group = max(last_group, begin_group + 1)
            last = group_end[last_group - 1]
            ids = order[first:last]
            harmonics = np.empty((last - first, nch))
            _cone_harmonics(np.ascontiguousarray(vectors[ids]),
                            np.ascontiguousarray(cones[ids]), starts, aa, bb, max_m,
                            harmonics)
            weight = link_weights[n][ids]
            factor = factors[n][ids]
            sizes = group_end[begin_group:last_group] - group_start[begin_group:last_group]
            small = np.repeat(sizes < min_group, sizes)
            if np.any(small):
                # many tiny groups (a muon's sparse secondaries): element loop
                _deposit_elements(np.ascontiguousarray(node_rows[ids[small]]),
                                  np.ascontiguousarray(weight[small]),
                                  np.ascontiguousarray(factor[small]),
                                  np.ascontiguousarray(harmonics[small]), deposit, blocks)
            for g in np.flatnonzero(sizes >= min_group) + begin_group:
                lo = group_start[g] - first
                hi = group_end[g] - first
                block = (weight[lo:hi, :, None] * factor[lo:hi, None, :]).reshape(hi - lo, -1)
                product = (block.T @ harmonics[lo:hi]).reshape(links, nk, nch)
                deposit[node_rows[ids[lo]]] += product  # distinct rows per point
            begin_group = last_group
    # omega = 0 transverse moments of the total light (field 0, l = m = 0) per
    # slice, straight from the deposit: they decide which slices need the
    # stencil and place the thin ones at their centroid
    zero = np.add.reduceat(deposit[:, :moments, 0], row_start[:-1], axis=0)
    if moments == 1:
        wide = np.zeros(len(slices), bool)
        needed = np.ones((len(slices), nf), bool)
    else:
        mass = zero[:, 0]
        safe = np.where(np.abs(mass) > 0, mass, 1.0)
        spread = np.sqrt(np.abs(zero[:, 3] + zero[:, 4]) / np.abs(safe))
        wide = spread > stencil_threshold_m
        centroid = np.stack([np.where(wide, 0.0, zero[:, 1] / safe),
                             np.where(wide, 0.0, zero[:, 2] / safe)], axis=1)
        needed = np.zeros((len(slices), nf, moments), bool)
        needed[:, :, 0] = True
        needed[wide] = True
        needed = needed.reshape(len(slices), -1)
    z_centres = z_origin + slices * cell_m
    h = float(stencil_m)
    stencil_slices = int(wide.sum())
    # point layout: one axis/centroid point per slice, then the +x, -x, +y, -y
    # stencil points of the wide slices, each group in slice order
    wide_index = np.cumsum(wide) - 1
    count_wide = stencil_slices
    npoints = len(slices) + 4 * count_wide
    tables = [np.zeros((npoints, nch, len(omega)), np.complex128) for _ in range(nf)]
    times = tick_of * time_step_ns
    for cell in range(len(slices)):
        span = slice(row_start[cell], row_start[cell + 1])
        keep = np.flatnonzero(needed[cell])
        phase = np.outer(times[span], omega)
        block = deposit[span][:, keep].reshape(-1, len(keep) * nch).T
        values = ((block @ np.cos(phase)) + 1j * (block @ np.sin(phase))).reshape(
            len(keep), nch, len(omega))
        if moments == 1 or not wide[cell]:
            for f in range(nf):
                tables[f][cell] = values[f if moments == 1 else list(keep).index(f * moments)]
            continue
        values = values.reshape(nf, moments, nch, len(omega))
        slot = wide_index[cell]
        for f in range(nf):
            m0, mx, my, mxx, myy = values[f]
            tables[f][cell] = m0 - (mxx + myy) / h ** 2
            tables[f][len(slices) + slot] = 0.5 * (mxx / h ** 2 + mx / h)
            tables[f][len(slices) + count_wide + slot] = 0.5 * (mxx / h ** 2 - mx / h)
            tables[f][len(slices) + 2 * count_wide + slot] = 0.5 * (myy / h ** 2 + my / h)
            tables[f][len(slices) + 3 * count_wide + slot] = 0.5 * (myy / h ** 2 - my / h)
    if moments == 1:
        points_z = z_centres
        points_t = np.zeros((len(slices), 2))
    else:
        # A thin slice keeps one point, placed at its photon centroid rather
        # than on the axis: this keeps the first transverse moment (a muon
        # track is not exactly on the principal axis); the neglected second
        # moment about the centroid is below stencil_threshold_m**2.
        offsets = [(h, 0.0), (-h, 0.0), (0.0, h), (0.0, -h)]
        points_z = np.concatenate([z_centres] + [z_centres[wide]] * 4)
        points_t = np.concatenate([centroid] + [np.tile(o, (count_wide, 1)) for o in offsets])
    summary = dict(elements=count, degree=int(degree), cells=len(points_z),
                   slices=len(slices), stencil_slices=stencil_slices, cell_m=float(cell_m),
                   transverse=transverse, axial_order=int(axial_order),
                   stencil_m=float(stencil_m) if moments > 1 else None,
                   stencil_threshold_m=float(stencil_threshold_m),
                   azimuthal_degree=int(azimuthal_degree), channels=nch,
                   time_step_ns=float(time_step_ns), time_deposit=time_deposit,
                   time_rows=len(unique),
                   element_order=int(element_order), threads=blocks,
                   compiler="axial_slices")
    out = [AxialSource(frame, points_z, points_t, tab, kept, channel_degree, int(degree),
                       int(azimuthal_degree), reference,
                       {**summary, "photons": float(photons[f].sum())})
           for f, tab in enumerate(tables)]
    return out[0] if single else out


# ---------------------------------------------------------------------------
# Wavelength-merged order >=2 apply
#
# Equation (*) of lighthit.directional, as in
# lighthit._directional_numba._apply with ge2_only=True, but for all
# wavelengths at once: the geometry, the detector harmonics, the Wigner
# rotation and the rotation of both source fields into the r-hat frame depend
# on neither the wavelength nor the medium and are computed once per
# (receiver, point); only the radial spline, the attenuation and the flight
# phase are evaluated per wavelength.  The arithmetic of every term is that
# of the per-wavelength kernel.

from ._directional_numba import _harmonics_into as _detector_harmonics  # noqa: E402
from ._directional_numba import _parity, _start_norm_table, _wigner_into  # noqa: E402


@njit(cache=True, nogil=True, parallel=True)
def _apply_multi(points, receivers, looks, channels0, channels2, scale0, scale2,
                 out_scale, starts, degree, max_m, max_out, alpha, pair_lambda, pair_mu,
                 log_r, coefficients, absorption, velocity, omega, flight, start_norm,
                 jmax, radial_low, radial_high):
    """sum_lambda out_scale[lambda, d] * (s0 K field0 - s2 K field2)_lambda, order >=2.

    ``coefficients``: (lambda, radial interval, degree, pair, cubic power,
    frequency).  Returns (receiver, frequency) complex.
    """
    nfreq = len(omega)
    nrec = receivers.shape[0]
    nlam = coefficients.shape[0]
    npair = len(pair_lambda)
    width = 2 * max_out + 1
    result = np.zeros((nrec, nfreq), np.complex128)
    root2 = np.sqrt(2.0)
    for index in prange(nrec):
        wigner = np.empty((degree + 1, width, max_m + 1), np.float64)
        detector = np.empty((max_out + 1) * (max_out + 1), np.float64)
        erow = np.empty((width, 2 * max_m + 1), np.float64)
        rotated0 = np.empty((width, nfreq), np.complex128)
        rotated2 = np.empty((width, nfreq), np.complex128)
        channel_sum = np.empty((width, nfreq), np.complex128)
        cosines = np.empty(max_m + 1, np.float64)
        sines = np.empty(max_m + 1, np.float64)
        scale_phase = np.empty((nlam, nfreq), np.complex128)
        for cell in range(points.shape[0]):
            dx = receivers[index, 0] - points[cell, 0]
            dy = receivers[index, 1] - points[cell, 1]
            dz = receivers[index, 2] - points[cell, 2]
            radius = np.sqrt(dx * dx + dy * dy + dz * dz)
            if radius < radial_low or radius > radial_high:
                continue
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
            lx = looks[index, 0]
            ly = looks[index, 1]
            lz = looks[index, 2]
            a1 = cosine * cp * lx + cosine * sp * ly - sine * lz
            a2 = -sp * lx + cp * ly
            a3 = sine * cp * lx + sine * sp * ly + cosine * lz
            _detector_harmonics(a1, a2, a3, max_out, detector)
            _wigner_into(degree, max_out, max_m, cosine, start_norm, jmax, wigner)
            logr = np.log(radius)
            j = np.searchsorted(log_r, logr, side='right') - 1
            if j < 0:
                j = 0
            if j > len(log_r) - 2:
                j = len(log_r) - 2
            t = logr - log_r[j]
            for lam in range(nlam):
                scale = (out_scale[lam, index] * np.exp(-absorption[lam] * radius)
                         / (4 * np.pi * radius * radius))
                for w in range(nfreq):
                    if flight:
                        angle = omega[w] * radius / velocity[lam]
                        scale_phase[lam, w] = scale * (np.cos(angle) + 1j * np.sin(angle))
                    else:
                        scale_phase[lam, w] = scale
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
                for a in range(width):
                    for w in range(nfreq):
                        rotated0[a, w] = 0.0
                        rotated2[a, w] = 0.0
                for a in range(max_out - top_p, max_out + top_p + 1):
                    for b in range(max_m - keep, max_m + keep + 1):
                        factor = erow[a, b]
                        if factor == 0.0:
                            continue
                        column = centre + (b - max_m)
                        for w in range(nfreq):
                            rotated0[a, w] += factor * channels0[cell, column, w]
                            rotated2[a, w] += factor * channels2[cell, column, w]
                for lam in range(nlam):
                    for a in range(width):
                        for w in range(nfreq):
                            channel_sum[a, w] = 0.0
                    for p in range(npair):
                        lam_detector = pair_lambda[p]
                        mu_pair = pair_mu[p]
                        if mu_pair > top_p:
                            continue
                        detector_centre = lam_detector * lam_detector + lam_detector
                        positive = alpha[lam_detector] * detector[detector_centre + mu_pair]
                        negative = (alpha[lam_detector] * detector[detector_centre - mu_pair]
                                    if mu_pair else 0.0)
                        if positive == 0.0 and negative == 0.0:
                            continue
                        for w in range(nfreq):
                            c0 = coefficients[lam, j, ell, p, 0, w]
                            c1 = coefficients[lam, j, ell, p, 1, w]
                            c2 = coefficients[lam, j, ell, p, 2, w]
                            c3 = coefficients[lam, j, ell, p, 3, w]
                            radial = ((c0 * t + c1) * t + c2) * t + c3
                            if negative != 0.0:
                                channel_sum[max_out - mu_pair, w] += negative * radial
                            if positive != 0.0:
                                channel_sum[max_out + mu_pair, w] += positive * radial
                    s0 = scale0[lam]
                    s2 = scale2[lam]
                    for a in range(max_out - top_p, max_out + top_p + 1):
                        for w in range(nfreq):
                            value = s0 * rotated0[a, w] - s2 * rotated2[a, w]
                            if value == 0.0:
                                continue
                            result[index, w] += value * channel_sum[a, w] * scale_phase[lam, w]
    return result



@njit(cache=True, nogil=True, parallel=True)
def _apply_multi_spectral(points, receivers, looks, channels, out_scale, starts, degree, max_m, max_out, alpha, pair_lambda, pair_mu,
                 log_r, coefficients, absorption, velocity, omega, flight, start_norm,
                 jmax, radial_low, radial_high):
    """Fused wavelength-dependent sources on a shared spatial point set, order >=2.

    ``coefficients``: (lambda, radial interval, degree, pair, cubic power,
    frequency).  Returns (receiver, frequency) complex.
    """
    nfreq = len(omega)
    nrec = receivers.shape[0]
    nlam = coefficients.shape[0]
    npair = len(pair_lambda)
    width = 2 * max_out + 1
    result = np.zeros((nrec, nfreq), np.complex128)
    root2 = np.sqrt(2.0)
    for index in prange(nrec):
        wigner = np.empty((degree + 1, width, max_m + 1), np.float64)
        detector = np.empty((max_out + 1) * (max_out + 1), np.float64)
        erow = np.empty((width, 2 * max_m + 1), np.float64)
        rotated = np.empty((nlam, width, nfreq), np.complex128)
        channel_sum = np.empty((width, nfreq), np.complex128)
        cosines = np.empty(max_m + 1, np.float64)
        sines = np.empty(max_m + 1, np.float64)
        scale_phase = np.empty((nlam, nfreq), np.complex128)
        for cell in range(points.shape[0]):
            dx = receivers[index, 0] - points[cell, 0]
            dy = receivers[index, 1] - points[cell, 1]
            dz = receivers[index, 2] - points[cell, 2]
            radius = np.sqrt(dx * dx + dy * dy + dz * dz)
            if radius < radial_low or radius > radial_high:
                continue
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
            lx = looks[index, 0]
            ly = looks[index, 1]
            lz = looks[index, 2]
            a1 = cosine * cp * lx + cosine * sp * ly - sine * lz
            a2 = -sp * lx + cp * ly
            a3 = sine * cp * lx + sine * sp * ly + cosine * lz
            _detector_harmonics(a1, a2, a3, max_out, detector)
            _wigner_into(degree, max_out, max_m, cosine, start_norm, jmax, wigner)
            logr = np.log(radius)
            j = np.searchsorted(log_r, logr, side='right') - 1
            if j < 0:
                j = 0
            if j > len(log_r) - 2:
                j = len(log_r) - 2
            t = logr - log_r[j]
            for lam in range(nlam):
                scale = (out_scale[lam, index] * np.exp(-absorption[lam] * radius)
                         / (4 * np.pi * radius * radius))
                for w in range(nfreq):
                    if flight:
                        angle = omega[w] * radius / velocity[lam]
                        scale_phase[lam, w] = scale * (np.cos(angle) + 1j * np.sin(angle))
                    else:
                        scale_phase[lam, w] = scale
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
                    for a in range(width):
                        for w in range(nfreq):
                            rotated[lam, a, w] = 0.0
                for a in range(max_out - top_p, max_out + top_p + 1):
                    for b in range(max_m - keep, max_m + keep + 1):
                        factor = erow[a, b]
                        if factor == 0.0:
                            continue
                        column = centre + (b - max_m)
                        for lam in range(nlam):
                            for w in range(nfreq):
                                rotated[lam, a, w] += factor * channels[lam, cell, column, w]
                for lam in range(nlam):
                    for a in range(width):
                        for w in range(nfreq):
                            channel_sum[a, w] = 0.0
                    for p in range(npair):
                        lam_detector = pair_lambda[p]
                        mu_pair = pair_mu[p]
                        if mu_pair > top_p:
                            continue
                        detector_centre = lam_detector * lam_detector + lam_detector
                        positive = alpha[lam_detector] * detector[detector_centre + mu_pair]
                        negative = (alpha[lam_detector] * detector[detector_centre - mu_pair]
                                    if mu_pair else 0.0)
                        if positive == 0.0 and negative == 0.0:
                            continue
                        for w in range(nfreq):
                            c0 = coefficients[lam, j, ell, p, 0, w]
                            c1 = coefficients[lam, j, ell, p, 1, w]
                            c2 = coefficients[lam, j, ell, p, 2, w]
                            c3 = coefficients[lam, j, ell, p, 3, w]
                            radial = ((c0 * t + c1) * t + c2) * t + c3
                            if negative != 0.0:
                                channel_sum[max_out - mu_pair, w] += negative * radial
                            if positive != 0.0:
                                channel_sum[max_out + mu_pair, w] += positive * radial
                    for a in range(max_out - top_p, max_out + top_p + 1):
                        for w in range(nfreq):
                            value = rotated[lam, a, w]
                            if value == 0.0:
                                continue
                            result[index, w] += value * channel_sum[a, w] * scale_phase[lam, w]
    return result

def apply_multi(bundle, source0, source2, receivers_m, look_directions, alpha, scale0,
                scale2, out_scale, *, frequencies=None, radial_window_m=None):
    """Order >=2 response summed over wavelengths; returns (frequency, receiver).

    ``bundle`` is a :class:`lighthit.shower_basis.Ge2KernelStore` bundle,
    ``scale0/scale2`` the two field weights and ``out_scale`` (lambda,
    receiver) the detector scale of every wavelength.  ``radial_window_m``
    skips source points outside a per-receiver radial window inside the cache;
    callers must check the omitted contribution before using this option.
    """
    coefficients = bundle["coefficients"]
    omega = bundle["omega"]
    if frequencies is not None:
        coefficients = coefficients[..., frequencies]
        omega = omega[frequencies]
    if source0.channels.shape[2] != len(omega):
        raise ValueError("source and kernel frequency grids differ")
    if bundle["degree"] > source0.degree:
        raise ValueError("source degree is insufficient")
    max_m = min(int(source0.azimuthal_degree), int(bundle["degree"]))
    starts = [0]
    for ell in range(source0.degree + 1):
        starts.append(starts[-1] + 2 * min(ell, source0.azimuthal_degree) + 1)
    starts = np.asarray(starts, np.int64)
    points = np.asarray(source0.points_m(), float)
    if not np.array_equal(points, np.asarray(source2.points_m(), float)):
        raise ValueError("the two fields must share their points")
    centre = np.asarray(source0.frame.centre_m, float)
    local_points = np.ascontiguousarray(source0.frame.rotate(points - centre))
    receivers = np.atleast_2d(np.asarray(receivers_m, float))
    local_receivers = np.ascontiguousarray(source0.frame.rotate(receivers - centre))
    local_looks = np.ascontiguousarray(source0.frame.rotate(
        np.atleast_2d(np.asarray(look_directions, float))))
    radial_low = np.exp(bundle["log_r"][0])
    radial_high = np.exp(bundle["log_r"][-1])
    if radial_window_m is None:
        for receiver in local_receivers:
            radius = np.linalg.norm(local_points - receiver, axis=1)
            if np.any(radius < radial_low) or np.any(radius > radial_high):
                raise ValueError("Source/receiver distances outside the cache")
    else:
        low, high = map(float, radial_window_m)
        if (not np.isfinite(low) or not np.isfinite(high) or low < radial_low - 1e-9
                or high > radial_high + 1e-9 or low >= high):
            raise ValueError("radial window must lie inside the kernel cache")
        radial_low, radial_high = low, high
    acceptance_degree = int(bundle["acceptance_degree"])
    alpha = np.ascontiguousarray(np.asarray(alpha, float)[:acceptance_degree + 1])
    jmax = max(max_m, acceptance_degree)
    result = _apply_multi(
        local_points, local_receivers, local_looks,
        np.ascontiguousarray(source0.channels), np.ascontiguousarray(source2.channels),
        np.asarray(scale0, float), np.asarray(scale2, float),
        np.ascontiguousarray(out_scale, dtype=float), starts, int(bundle["degree"]),
        int(max_m), acceptance_degree, alpha, bundle["pair_lambda"], bundle["pair_mu"],
        bundle["log_r"], coefficients, bundle["absorption"], bundle["speed"], omega,
        bundle["phase"] == "flight", _start_norm_table(jmax), int(jmax),
        radial_low, radial_high)
    carrier = np.exp(1j * omega * source0.reference_ns)
    return result.T * carrier[:, None]


def apply_multi_spectral(bundle, sources, receivers_m, look_directions, alpha,
                         out_scale, *, radial_window_m=None,
                         channel_dtype=np.complex64):
    """Apply multiple spectral source bands sharing exactly the same points.

    The expensive receiver-to-source geometry is evaluated once per point.
    Wavelength-dependent source harmonics and RTE splines are then contracted
    inside the same Numba pass.  Each band's emission-time reference is aligned
    before stacking; the returned spectrum uses absolute event time.
    """
    sources = list(sources)
    if not sources or len(sources) != len(bundle["absorption"]):
        raise ValueError("one spectral source is needed for each bundled wavelength")
    first = sources[0]
    omega = np.asarray(bundle["omega"])
    points = np.asarray(first.points_m(), float)
    for item in sources:
        if (not np.array_equal(item.points_m(), points)
                or item.channels.shape != first.channels.shape
                or item.azimuthal_degree != first.azimuthal_degree
                or not np.array_equal(item.frame.axis, first.frame.axis)):
            raise ValueError("fused spectral sources must share points and angular layout")
    reference = min(float(item.reference_ns) for item in sources)
    channels = np.empty((len(sources),) + first.channels.shape, channel_dtype)
    for index, item in enumerate(sources):
        phase = np.exp(1j * omega * (item.reference_ns - reference))
        channels[index] = item.channels * phase[None, None, :]
    starts = [0]
    for ell in range(first.degree + 1):
        starts.append(starts[-1] + 2 * min(ell, first.azimuthal_degree) + 1)
    starts = np.asarray(starts, np.int64)
    centre = np.asarray(first.frame.centre_m, float)
    local_points = np.ascontiguousarray(first.frame.rotate(points - centre))
    receivers = np.atleast_2d(np.asarray(receivers_m, float))
    local_receivers = np.ascontiguousarray(first.frame.rotate(receivers - centre))
    local_looks = np.ascontiguousarray(first.frame.rotate(
        np.atleast_2d(np.asarray(look_directions, float))))
    radial_low = float(np.exp(bundle["log_r"][0]))
    radial_high = float(np.exp(bundle["log_r"][-1]))
    if radial_window_m is None:
        for receiver in local_receivers:
            radius = np.linalg.norm(local_points - receiver, axis=1)
            if np.any(radius < radial_low) or np.any(radius > radial_high):
                raise ValueError("Source/receiver distances outside the cache")
    else:
        low, high = map(float, radial_window_m)
        if (not np.isfinite(low) or not np.isfinite(high) or low < radial_low - 1e-9
                or high > radial_high + 1e-9 or low >= high):
            raise ValueError("radial window must lie inside the kernel cache")
        radial_low, radial_high = low, high
    acceptance_degree = int(bundle["acceptance_degree"])
    alpha = np.ascontiguousarray(np.asarray(alpha, float)[:acceptance_degree + 1])
    max_m = min(int(first.azimuthal_degree), int(bundle["degree"]))
    jmax = max(max_m, acceptance_degree)
    result = _apply_multi_spectral(
        local_points, local_receivers, local_looks, np.ascontiguousarray(channels),
        np.ascontiguousarray(out_scale, float), starts, int(bundle["degree"]),
        int(max_m), acceptance_degree, alpha, bundle["pair_lambda"], bundle["pair_mu"],
        bundle["log_r"], bundle["coefficients"], bundle["absorption"],
        bundle["speed"], omega, bundle["phase"] == "flight",
        _start_norm_table(jmax), int(jmax), radial_low, radial_high)
    return result.T * np.exp(1j * omega * reference)[:, None]
