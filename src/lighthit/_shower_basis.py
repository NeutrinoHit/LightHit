"""Axial-slice shower basis for scattering order >=2.

The source of a shower is written in its own axial frame (``AxisFrame``:
photon-weighted centre and principal axis) as

    S(x, u, omega) = sum_p sum_(l,m) c_(p,lm)(omega) delta(x - x_p) Y_lm(u),

on a small set of points ``x_p``: slices ``z_r = z_0 + r h`` on the axis and,
where the slice has transverse extent, four more points at ``+-s`` along the
two transverse frame vectors.  ``Y_lm`` are real harmonics in the axial frame
with ``l <= L`` and ``|m| <= M``.  The response of module ``d`` is

    N_d^(>=2)(omega) = sum_(p,lm) c_(p,lm)(omega) K_(d,p,lm)^(>=2)(omega),

where ``K`` is the directional RTE kernel of the existing cache
(:class:`lighthit._directional_numba.PreparedDirectionalKernel`). Scattering
rank, retained source harmonics, spatial quadrature, radial interpolation,
stored precision and frequency sampling remain numerical approximations.
The coefficients are moments of the charged-step chords,

    c_(r,lm)^(k)(omega) = sum_i Q_i P_l(mu_C,i) Y_lm(u_i) w_r(z_i) m_k(x_i) e^{i omega t_i},

with ``w_r`` the linear (two-slice) or cubic Lagrange (four-slice) weights of
the chord point on the axis and ``m_k = 1, x, y, x^2, y^2`` its transverse
moments.  The five stencil weights reproduce all five moments of every
``(slice, l, m, omega)`` coefficient exactly; slices whose rms transverse
extent is below ``stencil_threshold_m`` keep a single point at their photon
centroid (zeroth and first moments of the total light).  The error
terms are the axial interpolation (O(h^2) linear, O(h^4) cubic), the
unmatched third and mixed second transverse moments, and the time lattice
used for ``e^{i omega t}``.  Accuracy is summarised in README.md.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

try:
    from numba import get_num_threads, njit, prange
except ImportError as exc:  # pragma: no cover
    raise ImportError("_shower_basis requires the optional 'accelerate' extra") from exc

from .axial_fast import _harmonics_into, _layout
from .axial_source import AxialSource, AxisFrame

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


def _align_geometry_reference(fields, reference):
    """Pad spectral subsets on one exact geometry without adding light.

    Row identity alone is insufficient: every physical chord and emission
    clock is verified before replacing its storage by the shared arrays.
    Missing rows receive exactly zero photon weight and a finite dummy cone.
    """
    reference_rows = np.asarray(reference.row_index)
    count = len(reference)
    if (reference_rows.shape != (count,) or not np.issubdtype(reference_rows.dtype, np.integer)
            or len(np.unique(reference_rows)) != count):
        raise ValueError("geometry_reference must have unique integer row_index values")
    geometry_weight = np.asarray(reference.photons, float)
    if (geometry_weight.shape != (count,) or not np.isfinite(geometry_weight).all()
            or np.any(geometry_weight < 0.) or not np.any(geometry_weight > 0.)):
        raise ValueError("geometry_reference photon weights must be finite, nonnegative and nonzero")
    order = np.argsort(reference_rows)
    sorted_rows = reference_rows[order]
    geometry_names = ("start_m", "direction", "length_m", "start_ns", "end_ns", "uid")
    aligned = []
    for field in fields:
        rows = np.asarray(field.row_index)
        if (rows.shape != (len(field),) or not np.issubdtype(rows.dtype, np.integer)
                or len(np.unique(rows)) != len(rows)):
            raise ValueError("source fields must have unique integer row_index values")
        positions = np.searchsorted(sorted_rows, rows)
        if np.any(positions >= count) or not np.array_equal(sorted_rows[positions], rows):
            raise ValueError("source field rows are absent from geometry_reference")
        indices = order[positions]
        for name in geometry_names:
            actual = np.asarray(getattr(field, name))
            expected = np.asarray(getattr(reference, name))[indices]
            # SpectralLightElements.subset normalizes an already unit vector
            # again. Permit only its machine-roundoff scale, not an altered
            # track direction or a displacement/time approximation.
            matches = (np.allclose(actual, expected, rtol=0., atol=1e-14)
                       if name == "direction" else np.array_equal(actual, expected))
            if not matches:
                raise ValueError(f"source field {name} differs from geometry_reference")
        photons = np.zeros(count)
        cones = np.ones(count)
        photons[indices] = field.photons
        cones[indices] = field.cone_cosine
        if np.any((photons != 0.) & (geometry_weight == 0.)):
            raise ValueError("geometry_reference must cover every emitting source row")
        aligned.append(replace(field, **{name: getattr(reference, name) for name in geometry_names},
                               row_index=reference.row_index, photons=photons, cone_cosine=cones))
    return aligned, geometry_weight


def compile_axial_slices(fields, degree, omega_per_ns, *, azimuthal_degree=4,
                         cell_m=0.25, axial_order=3, transverse="stencil",
                         stencil_m=0.2, stencil_threshold_m=0.0,
                         time_step_ns=0.02, time_deposit="linear", element_order="auto",
                         frame=None, chunk=65536, deposit_method="gemm", min_group=32,
                         point_selection=None, channel_dtype=np.complex128,
                         stencil_center="axis", geometry_reference=None, prune_selected=True,
                         axial_clock_ns_per_m=0.0):
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
    is ``(omega dt)**2 / 24``). ``"cubic"`` uses four ticks with pointwise
    phase error at most ``3 * (omega dt)**4 / 128``. Its weights can be
    negative, just like the cubic axial weights; all are retained.
    ``element_order="auto"`` uses the
    chord midpoint when every chord is shorter than ``cell_m / 10`` and two
    Gauss nodes otherwise.

    Optional ``axial_clock_ns_per_m`` interpolates the demodulated source
    clock ``t - slope * (z-z_origin)`` and restores the travelling carrier at
    each output slice. This interpolates a different, slower adjoint amplitude;
    it is not a change to the recorded source clocks. Zero retains the original
    spatial interpolation. Receiver-response convergence must be checked.

    ``stencil_center="axis"`` retains the original layout and arithmetic.
    ``"centroid"`` places thin slices at their signed field-0 centroid. Wide
    stencils use a physical centre weighted by absolute axial weights, then
    translate every angular/frequency moment to that centre. A near-cancelled
    cubic slice must not place a virtual source far outside its real support.
    A slice is then thin only when *every contributing chord quadrature node*
    lies within ``stencil_threshold_m`` of the centroid transversely. This
    conservative support test cannot be fooled by cancellations of signed
    cubic axial weights. Thin slices retain all zeroth angular/frequency
    coefficients but approximate their transverse dependence by one point.

    With ``geometry_reference=<LightElements>`` the centroid, support test and
    spatial layout instead use its fixed nonnegative weights (typically C0).
    Every wavelength field is aligned by unique ``row_index`` and checked for
    identical chord/UID/time data; threshold-removed rows are padded with zero
    photons. This keeps one frame, emission clock and bitwise identical points
    across wavelengths without rounding coordinates or changing their light.
    Reference photons never enter the emitted harmonic/frequency coefficients.
    This option requires ``stencil_center="centroid"`` and a stencil, and leaves
    the old path unchanged when omitted.

    Returns one :class:`AxialSource` per field (a single one for a single
    field), directly usable by :meth:`PreparedDirectionalKernel.apply`.
    """
    single = not isinstance(fields, (list, tuple))
    channel_dtype = np.dtype(channel_dtype)
    if channel_dtype not in (np.dtype(np.complex64), np.dtype(np.complex128)):
        raise ValueError("channel_dtype must be complex64 or complex128")
    if not np.isfinite(axial_clock_ns_per_m):
        raise ValueError("axial_clock_ns_per_m must be finite")
    fields = [fields] if single else list(fields)
    physical_counts = [len(item) for item in fields]
    geometry_weight = None
    if geometry_reference is not None:
        if stencil_center != "centroid" or transverse != "stencil":
            raise ValueError("geometry_reference requires stencil_center='centroid' and transverse='stencil'")
        fields, geometry_weight = _align_geometry_reference(fields, geometry_reference)
    elements = fields[0]
    if not (cell_m > 0 and time_step_ns > 0 and stencil_m > 0 and stencil_threshold_m >= 0):
        raise ValueError("cell_m, time_step_ns and stencil_m must be positive")
    if axial_order not in (1, 3):
        raise ValueError("axial_order must be 1 or 3")
    if time_deposit not in ("linear", "nearest", "cubic"):
        raise ValueError("time_deposit must be 'linear', 'nearest' or 'cubic'")
    if transverse not in ("collapse", "stencil"):
        raise ValueError("transverse must be 'collapse' or 'stencil'")
    if stencil_center not in ("axis", "centroid"):
        raise ValueError("stencil_center must be 'axis' or 'centroid'")
    for other in fields[1:]:
        if (len(other) != len(elements)
                or not np.array_equal(other.start_m, elements.start_m)
                or not np.array_equal(other.direction, elements.direction)):
            raise ValueError("fields must share their elements")
    frame = frame or AxisFrame.of(elements if geometry_reference is None else geometry_reference)
    omega = np.atleast_1d(np.asarray(omega_per_ns, float))
    charge_only = len(omega) == 1 and omega[0] == 0
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
    if time_deposit == "cubic" and not charge_only:
        # Leave one tick to the left of the earliest emission. Otherwise
        # tb-1 wraps into the preceding axial slice in the combined key.
        reference -= time_step_ns
    margin = 1 if axial_order == 1 else 2
    z_origin = float(frame.coordinates(elements)[0].min()) - margin * cell_m
    moments = 1 if transverse == "collapse" else 5
    nf = len(fields)
    coordinates = [frame.coordinates(elements, fraction) for fraction in nodes]
    clock_times = ([c[3] - axial_clock_ns_per_m * (c[0]-z_origin) for c in coordinates]
                   if axial_clock_ns_per_m else [c[3] for c in coordinates])
    if axial_clock_ns_per_m:
        reference = min(float(np.min(t)) for t in clock_times)
        if time_deposit == "cubic" and not charge_only:
            reference -= time_step_ns
    time_max = max(float(np.max(t)) for t in clock_times)
    # At omega=0 emission time drops out exactly. Collapsing the time key
    # avoids building a sparse time lattice just to sum it back to one value.
    kmax = 1 if charge_only else int(np.floor((time_max - reference) / time_step_ns)) + 3
    shifts_count = 2 if axial_order == 1 else 4
    ticks = (4 if time_deposit == "cubic" else 2 if time_deposit == "linear" else 1) if not charge_only else 1
    links = ticks * shifts_count
    keys, link_weights, factors, geometry_links = [], [], [], []
    reference_factors = []
    for (z, offset, _, _), time, node_weight in zip(coordinates, clock_times, node_weights):
        if moments == 1:
            mom = np.ones((count, 1))
        else:
            x = offset @ frame.first
            y = offset @ frame.second
            mom = np.stack([np.ones(count), x, y, x * x, y * y], axis=1)
        factors.append((photons.T[:, :, None] * mom[:, None, :]).reshape(count, -1)
                       * node_weight)
        if geometry_weight is not None:
            reference_factors.append(geometry_weight[:, None] * mom * node_weight)
        zl = (z - z_origin) / cell_m
        zb = np.floor(zl).astype(np.int64)
        tl = (np.zeros(count) if charge_only else
              (np.asarray(time, float) - reference) / time_step_ns)
        if ticks == 4:
            tb = np.floor(tl).astype(np.int64)
            tr = tl - tb
            tick_shifts, weights_t = _axial_weights(tr, 3)
            tick_weights = tuple(zip(tick_shifts, weights_t))
        elif ticks == 2:
            tb = np.floor(tl).astype(np.int64)
            tr = tl - tb
            tick_weights = ((0, 1 - tr), (1, tr))
        else:
            tb = np.rint(tl).astype(np.int64)
            tick_weights = ((0, 1.0),)
        shifts, zw = _axial_weights(zl - zb, axial_order)
        geometry_links.append([(zb + shift, wz) for shift, wz in zip(shifts, zw)])
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
    if span_keys <= min(1 << 28, 8 * keys.size) and int(keys.min()) >= 0:
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
    # Compute the spatial layout before any angular/time deposition. The
    # total-light geometry is independent of harmonic and frequency channels.
    # Define spatial points without the time lattice, identically in the
    # charge and spectrum passes. In particular a slice exactly at the
    # stencil threshold must not change layout because time sums round
    # differently. Y_00 is constant and cancels in these moment ratios.
    zero = np.zeros((len(slices), moments))
    geometry_factors = factors if geometry_weight is None else reference_factors
    for links_n, factor in zip(geometry_links, geometry_factors):
        for cell_ids, weight in links_n:
            rows_n = np.searchsorted(slices, cell_ids)
            for moment in range(moments):
                zero[:, moment] += np.bincount(
                    rows_n, weights=weight * factor[:, moment], minlength=len(slices))
    if moments == 1:
        wide = np.zeros(len(slices), bool)
        needed = np.ones((len(slices), nf), bool)
    else:
        mass = zero[:, 0]
        safe = np.where(np.abs(mass) > 0, mass, 1.0)
        if stencil_center == "axis":
            spread = np.sqrt(np.abs(zero[:, 3] + zero[:, 4]) / np.abs(safe))
            wide = spread > stencil_threshold_m
            centroid = np.stack([np.where(wide, 0.0, zero[:, 1] / safe),
                                 np.where(wide, 0.0, zero[:, 2] / safe)], axis=1)
        else:
            centroid = zero[:, 1:3] / safe[:, None]
            # A signed cubic slice can have zero net mass but nonzero higher
            # moments. Its centre is arbitrary for a full stencil; a stable
            # absolute-weight centre avoids division by zero without deleting
            # those moments. It still undergoes the conservative support test.
            cancelled = mass == 0
            if np.any(cancelled):
                absolute = np.zeros((len(slices), 3))
                for links_n, factor in zip(geometry_links, geometry_factors):
                    for cell_ids, weight in links_n:
                        rows_n = np.searchsorted(slices, cell_ids)
                        for moment in range(3):
                            absolute[:, moment] += np.bincount(
                                rows_n, weights=np.abs(weight) * factor[:, moment],
                                minlength=len(slices))
                denominator = np.where(absolute[:, 0] > 0, absolute[:, 0], 1.)
                centroid[cancelled] = absolute[cancelled, 1:3] / denominator[cancelled, None]
            support_radius2 = np.zeros(len(slices))
            emitting = (np.any(photons != 0., axis=0) if geometry_weight is None
                        else geometry_weight > 0.)
            for links_n, coordinate in zip(geometry_links, coordinates):
                offset = coordinate[1]
                x, y = offset @ frame.first, offset @ frame.second
                for cell_ids, weight in links_n:
                    rows_n = np.searchsorted(slices, cell_ids)
                    # Include any node emitting in any field. This also
                    # protects a field-0 zero with a nonzero second field.
                    contributes = (weight != 0.) & emitting
                    chosen = rows_n[contributes]
                    radius2 = ((x[contributes] - centroid[chosen, 0]) ** 2
                               + (y[contributes] - centroid[chosen, 1]) ** 2)
                    np.maximum.at(support_radius2, chosen, radius2)
            wide = support_radius2 > stencil_threshold_m ** 2
            if np.any(wide):
                # Signed cubic masses can nearly cancel: mx/m0 may lie far
                # outside the emitting cloud. Wide stencils can be centred
                # anywhere while retaining all five moments after translation.
                # Use a convex combination of actual nodes for stability.
                absolute = np.zeros((len(slices), 3))
                for links_n, factor in zip(geometry_links, geometry_factors):
                    for cell_ids, weight in links_n:
                        rows_n = np.searchsorted(slices, cell_ids)
                        for moment in range(3):
                            absolute[:, moment] += np.bincount(
                                rows_n, weights=np.abs(weight) * factor[:, moment],
                                minlength=len(slices))
                denominator = np.where(absolute[:, 0] > 0, absolute[:, 0], 1.)
                centroid[wide] = absolute[wide, 1:3] / denominator[wide, None]
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
    selected = (np.ones(npoints, bool) if point_selection is None else
                np.asarray(point_selection, bool))
    if selected.shape != (npoints,):
        raise ValueError("point_selection must match the complete spatial point layout")
    output_row = np.full(npoints, -1, np.int64)
    output_row[selected] = np.arange(int(selected.sum()))
    unpruned_time_rows = len(unique)
    pruning = point_selection is not None and prune_selected and not np.all(selected)
    time_row_selection = np.ones(len(unique), bool)
    if pruning:
        slice_selected = selected[:len(slices)].copy()
        for offset in range(4 if moments > 1 else 0):
            slice_selected[wide] |= selected[len(slices)+offset*count_wide:
                                             len(slices)+(offset+1)*count_wide]
        time_row_selection = slice_selected[np.searchsorted(slices, cell_of)]
        row_map = np.zeros(len(unique), np.int64)
        row_map[time_row_selection] = np.arange(int(time_row_selection.sum()))
        # Keep the complete slice layout, but allocate only retained time rows.
        retained_counts = np.bincount(np.searchsorted(slices, cell_of[time_row_selection]),
                                      minlength=len(slices))
        row_start = np.r_[0, np.cumsum(retained_counts)].astype(np.int64)
        unique = unique[time_row_selection]
        tick_of = tick_of[time_row_selection]
    angular_elements_per_node = []
    deposit = np.zeros((len(unique), nf * moments, nch))
    vectors = np.ascontiguousarray(frame.rotate(elements.direction))
    cones = np.ascontiguousarray(elements.cone_cosine, dtype=float)
    blocks = max(1, int(get_num_threads()))
    nk = nf * moments
    for n in range(len(nodes)):
        span = slice(n * links, (n + 1) * links)
        old_rows = rows[:, span]
        if pruning:
            live = time_row_selection[old_rows]
            element_ids = np.flatnonzero(np.any(live & (link_weights[n] != 0.), axis=1)
                                        & np.any(photons != 0., axis=0))
            node_key = old_rows[element_ids, 0]
            node_live = live[element_ids]
            node_rows = row_map[old_rows[element_ids]]
            node_weights = link_weights[n][element_ids] * node_live
            node_factors = factors[n][element_ids]
            node_vectors = np.ascontiguousarray(vectors[element_ids])
            node_cones = np.ascontiguousarray(cones[element_ids])
        else:
            node_rows = old_rows
            node_key = old_rows[:, 0]
            node_weights = link_weights[n]
            node_factors = factors[n]
            node_vectors, node_cones = vectors, cones
        node_count = len(node_rows)
        angular_elements_per_node.append(node_count)
        if not node_count:
            continue
        if deposit_method == "loop":
            harmonics = np.empty((min(node_count, chunk), nch))
            for begin in range(0, node_count, chunk):
                end = min(node_count, begin + chunk)
                _cone_harmonics(node_vectors[begin:end], node_cones[begin:end], starts, aa, bb,
                                max_m, harmonics[:end - begin])
                _deposit_elements(np.ascontiguousarray(node_rows[begin:end]),
                                  np.ascontiguousarray(node_weights[begin:end]),
                                  np.ascontiguousarray(node_factors[begin:end]),
                                  harmonics[:end - begin], deposit, blocks)
            continue
        # Chord points with the same base slice and time tick share all their
        # links: deposit each such group as one dense product
        #   (links*k, n_group) @ (n_group, channels).
        order = np.argsort(node_key, kind="stable")
        group_key = node_key[order]
        edges = np.flatnonzero(np.diff(group_key)) + 1
        group_start = np.concatenate(([0], edges))
        group_end = np.concatenate((edges, [node_count]))
        begin_group = 0
        while begin_group < len(group_start):
            # whole groups, about ``chunk`` chord points at a time
            first = group_start[begin_group]
            last_group = int(np.searchsorted(group_end, first + chunk, side="right"))
            last_group = max(last_group, begin_group + 1)
            last = group_end[last_group - 1]
            ids = order[first:last]
            harmonics = np.empty((last - first, nch))
            _cone_harmonics(np.ascontiguousarray(node_vectors[ids]),
                            np.ascontiguousarray(node_cones[ids]), starts, aa, bb, max_m,
                            harmonics)
            weight = node_weights[ids]
            factor = node_factors[ids]
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
                if pruning:
                    # Inactive links use dummy row 0. Never include them in
                    # advanced-index +=: duplicate dummy indices could undo a
                    # real contribution to row 0 even when their weights are zero.
                    live_links = node_live[ids[lo]]
                    deposit[node_rows[ids[lo], live_links]] += product[live_links]
                else:
                    deposit[node_rows[ids[lo]]] += product  # distinct rows per point
            begin_group = last_group
    # Spectral fusion already stores complex64 channels. Writing the final
    # double-precision moments directly into that dtype avoids a full-size
    # complex128 table plus a second conversion allocation; the rounding is
    # exactly the same as compiling in complex128 and casting afterwards.
    tables = [np.zeros((int(selected.sum()), nch, len(omega)), channel_dtype) for _ in range(nf)]
    times = tick_of * time_step_ns
    for cell in range(len(slices)):
        ids = [cell]
        if moments > 1 and wide[cell]:
            ids += [len(slices) + offset * count_wide + wide_index[cell]
                    for offset in range(4)]
        if not np.any(selected[ids]):
            continue
        span = slice(row_start[cell], row_start[cell + 1])
        keep = np.flatnonzero(needed[cell])
        phase = np.outer(times[span], omega)
        block = deposit[span][:, keep].reshape(-1, len(keep) * nch).T
        values = ((block @ np.cos(phase)) + 1j * (block @ np.sin(phase))).reshape(
            len(keep), nch, len(omega))
        if moments == 1 or not wide[cell]:
            for f in range(nf):
                tables[f][output_row[cell]] = values[f if moments == 1 else list(keep).index(f * moments)]
            continue
        values = values.reshape(nf, moments, nch, len(omega))
        slot = wide_index[cell]
        for f in range(nf):
            m0, mx, my, mxx, myy = values[f]
            if stencil_center == "centroid":
                # Translate each (field, harmonic, frequency) coefficient,
                # not only the integrated light used to choose the centre.
                cx, cy = centroid[cell]
                mxx = mxx - 2 * cx * mx + cx * cx * m0
                myy = myy - 2 * cy * my + cy * cy * m0
                mx = mx - cx * m0
                my = my - cy * m0
            values_at_points = [m0 - (mxx + myy) / h ** 2,
                                0.5 * (mxx / h ** 2 + mx / h),
                                0.5 * (mxx / h ** 2 - mx / h),
                                0.5 * (myy / h ** 2 + my / h),
                                0.5 * (myy / h ** 2 - my / h)]
            for point, value in zip(ids, values_at_points):
                if selected[point]:
                    tables[f][output_row[point]] = value
    if moments == 1:
        points_z = z_centres
        points_t = np.zeros((len(slices), 2))
    else:
        # A thin slice keeps one point at its field-0 centroid. The optional
        # centroid mode also moves wide stencils there, preserving every
        # zeroth, first and diagonal second moment after translation.
        offsets = [(h, 0.0), (-h, 0.0), (0.0, h), (0.0, -h)]
        points_z = np.concatenate([z_centres] + [z_centres[wide]] * 4)
        if stencil_center == "centroid":
            points_t = np.concatenate([centroid] + [centroid[wide] + o for o in offsets])
        else:
            points_t = np.concatenate([centroid] + [np.tile(o, (count_wide, 1)) for o in offsets])
    if axial_clock_ns_per_m and not charge_only:
        carrier = np.exp(1j * axial_clock_ns_per_m *
                         np.outer(points_z[selected]-z_origin, omega))
        for table in tables:
            table *= carrier[:, None, :]
    summary = dict(elements=count, degree=int(degree), cells=int(selected.sum()),
                   unselected_cells=int((~selected).sum()),
                   slices=len(slices), stencil_slices=stencil_slices, cell_m=float(cell_m),
                   transverse=transverse, axial_order=int(axial_order),
                   stencil_m=float(stencil_m) if moments > 1 else None,
                   stencil_threshold_m=float(stencil_threshold_m),
                   azimuthal_degree=int(azimuthal_degree), channels=nch,
                   time_step_ns=float(time_step_ns), time_deposit=time_deposit,
                   time_rows=len(unique), unpruned_time_rows=unpruned_time_rows,
                   angular_elements_per_node=angular_elements_per_node,
                   early_point_selection=bool(pruning),
                   axial_clock_ns_per_m=float(axial_clock_ns_per_m),
                   element_order=int(element_order), threads=blocks,
                   compiler="axial_slices")
    if stencil_center == "centroid":
        summary.update(stencil_center="centroid", thin_criterion="contributing_node_radius")
    if geometry_reference is not None:
        summary.update(geometry_reference="fixed photon-weighted chords", physical_field_elements=physical_counts)
    out = [AxialSource(frame, points_z[selected], points_t[selected], tab, kept, channel_degree, int(degree),
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


def _source_norm_bound(source):
    """Sum of degree-wise harmonic norms, bounded over all supplied frequencies."""
    result = np.zeros(len(source.z_m))
    for ell in range(source.degree + 1):
        channels = np.asarray(source.channels[:, source.channel_degree == ell], np.complex128)
        result += np.sqrt(np.sum(np.abs(channels) ** 2, axis=1)).max(axis=1)
    return result


def _radial_norm_bound(bundle, coefficients, alpha):
    """Interval upper bound for the *implemented* spline/angular operator.

    A real harmonic obeys |Y_am| <= sqrt((2a+1)/(4*pi)); rotations are
    orthogonal. Bound the cubic by the absolute power sum on its interval,
    sum detector terms, then maximise over source degree and frequency.
    This bounds the numerical cache, not its error against the physical RTE.
    """
    cache = bundle.get("_norm_cache") if not coefficients.flags.writeable else None
    key = None
    if cache is not None:
        key = (coefficients.__array_interface__["data"][0], coefficients.shape, coefficients.strides,
               np.asarray(alpha).tobytes(), bundle["log_r"].tobytes(),
               bundle["pair_lambda"].tobytes(), bundle["pair_mu"].tobytes())
        if key in cache:
            return cache[key]
    widths = np.diff(bundle["log_r"])
    detector_degree = np.asarray(bundle["pair_lambda"])
    factors = (np.abs(alpha[detector_degree])
               * np.sqrt((2 * detector_degree + 1) / (4 * np.pi))
               * np.where(np.asarray(bundle["pair_mu"]) == 0, 1., 2.))
    out = np.empty(coefficients.shape[:2])
    for lam in range(len(coefficients)):
        # Loop over polynomial powers to bound temporary memory.
        bound = np.zeros(coefficients[lam, :, :, :, 0, :].shape)
        for power in range(4):
            bound += (np.abs(np.asarray(coefficients[lam, :, :, :, power, :], np.complex128))
                      * widths[:, None, None, None] ** (3 - power))
        out[lam] = np.sum(bound * factors[None, None, :, None], axis=2).max(axis=(1, 2))
    # Guard normal floating-point evaluation of the norm, harmonic recurrence
    # and cubic. This is not an interval-arithmetic certification of the RTE.
    out = np.ascontiguousarray(out * (1.0 + 1e-10))
    if cache is not None:
        cache[key] = out
    return out


@njit(cache=True, inline="always")
def _active_pair_bands(radius, cell, receiver, log_r, amplitude, envelope,
                       absorption, out_scale, tolerance_per_point, active_bands):
    if tolerance_per_point == 0.0:
        active_bands[:] = True
        return True
    j = min(max(np.searchsorted(log_r, np.log(radius), side="right") - 1, 0), len(log_r)-2)
    any_active = False
    for lam in range(len(absorption)):
        bound = (amplitude[lam, cell] * envelope[lam, j]
                  * abs(out_scale[lam, receiver]) * np.exp(-absorption[lam] * radius)
                  / (4 * np.pi * radius * radius))
        active_bands[lam] = bound > tolerance_per_point / len(absorption)
        any_active |= active_bands[lam]
    return any_active


@njit(cache=True, nogil=True, parallel=True)
def _apply_multi(points, receivers, looks, channels0, channels2, scale0, scale2,
                 out_scale, starts, degree, max_m, max_out, alpha, pair_lambda, pair_mu,
                 log_r, coefficients, absorption, velocity, omega, flight, start_norm,
                 jmax, radial_low, radial_high, amplitude, envelope, tolerance_per_point):
    """sum_lambda out_scale[lambda, d] * (s0 K field0 - s2 K field2)_lambda, order >=2.

    ``coefficients``: (lambda, radial interval, degree, pair, cubic power,
    frequency).  Returns (receiver, frequency) complex.
    """
    nfreq = len(omega)
    nrec = receivers.shape[0]
    nlam = coefficients.shape[0]
    has_second_field = np.any(scale2 != 0.0)
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
        response = np.empty((nlam, nfreq), np.complex128)
        active_bands = np.empty(nlam, np.bool_)
        for cell in range(points.shape[0]):
            dx = receivers[index, 0] - points[cell, 0]
            dy = receivers[index, 1] - points[cell, 1]
            dz = receivers[index, 2] - points[cell, 2]
            radius = np.sqrt(dx * dx + dy * dy + dz * dz)
            if radius < radial_low or radius > radial_high:
                continue
            if not _active_pair_bands(radius, cell, index, log_r, amplitude, envelope,
                                      absorption, out_scale, tolerance_per_point, active_bands):
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
                if not active_bands[lam]:
                    continue
                scale = (out_scale[lam, index] * np.exp(-absorption[lam] * radius)
                         / (4 * np.pi * radius * radius))
                for w in range(nfreq):
                    response[lam, w] = 0.0
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
                        if has_second_field:
                            rotated2[a, w] = 0.0
                for a in range(max_out - top_p, max_out + top_p + 1):
                    for b in range(max_m - keep, max_m + keep + 1):
                        factor = erow[a, b]
                        if factor == 0.0:
                            continue
                        column = centre + (b - max_m)
                        for w in range(nfreq):
                            rotated0[a, w] += factor * channels0[cell, column, w]
                            if has_second_field:
                                rotated2[a, w] += factor * channels2[cell, column, w]
                for lam in range(nlam):
                    if not active_bands[lam]:
                        continue
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
                            value = s0 * rotated0[a, w]
                            if has_second_field:
                                value -= s2 * rotated2[a, w]
                            if value == 0.0:
                                continue
                            response[lam, w] += value * channel_sum[a, w]
            # The wavelength's flight/attenuation factor is common to every
            # angular degree. Apply it once after their sum, not per channel.
            for lam in range(nlam):
                if active_bands[lam]:
                    for w in range(nfreq):
                        result[index, w] += response[lam, w] * scale_phase[lam, w]
    return result



@njit(cache=True, nogil=True, parallel=True)
def _apply_multi_spectral(points, receivers, looks, channels, out_scale, starts, degree, max_m, max_out, alpha, pair_lambda, pair_mu,
                 log_r, coefficients, absorption, velocity, omega, flight, start_norm,
                 jmax, radial_low, radial_high, amplitude, envelope, tolerance_per_point):
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
        response = np.empty((nlam, nfreq), np.complex128)
        active_bands = np.empty(nlam, np.bool_)
        for cell in range(points.shape[0]):
            dx = receivers[index, 0] - points[cell, 0]
            dy = receivers[index, 1] - points[cell, 1]
            dz = receivers[index, 2] - points[cell, 2]
            radius = np.sqrt(dx * dx + dy * dy + dz * dz)
            if radius < radial_low or radius > radial_high:
                continue
            if not _active_pair_bands(radius, cell, index, log_r, amplitude, envelope,
                                      absorption, out_scale, tolerance_per_point, active_bands):
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
                if not active_bands[lam]:
                    continue
                scale = (out_scale[lam, index] * np.exp(-absorption[lam] * radius)
                         / (4 * np.pi * radius * radius))
                for w in range(nfreq):
                    response[lam, w] = 0.0
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
                    if not active_bands[lam]:
                        continue
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
                            if not active_bands[lam]:
                                continue
                            for w in range(nfreq):
                                rotated[lam, a, w] += factor * channels[lam, cell, column, w]
                for lam in range(nlam):
                    if not active_bands[lam]:
                        continue
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
                            response[lam, w] += value * channel_sum[a, w]
            for lam in range(nlam):
                if active_bands[lam]:
                    for w in range(nfreq):
                        result[index, w] += response[lam, w] * scale_phase[lam, w]
    return result

def apply_multi(bundle, source0, source2, receivers_m, look_directions, alpha, scale0,
                scale2, out_scale, *, frequencies=None, radial_window_m=None,
                absolute_tolerance_pe=0.0):
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
    if not np.isfinite(absolute_tolerance_pe) or absolute_tolerance_pe < 0:
        raise ValueError("absolute_tolerance_pe must be finite and nonnegative")
    amplitude = np.zeros((len(scale0), 0))
    envelope = np.zeros((len(scale0), 0))
    if absolute_tolerance_pe:
        amplitude = np.abs(scale0)[:, None] * _source_norm_bound(source0)
        if np.any(np.asarray(scale2) != 0.0):
            amplitude += np.abs(scale2)[:, None] * _source_norm_bound(source2)
        amplitude = np.ascontiguousarray(amplitude)
        envelope = _radial_norm_bound(bundle, coefficients, alpha)
    result = _apply_multi(
        local_points, local_receivers, local_looks,
        np.ascontiguousarray(source0.channels), np.ascontiguousarray(source2.channels),
        np.asarray(scale0, float), np.asarray(scale2, float),
        np.ascontiguousarray(out_scale, dtype=float), starts, int(bundle["degree"]),
        int(max_m), acceptance_degree, alpha, bundle["pair_lambda"], bundle["pair_mu"],
        bundle["log_r"], coefficients, bundle["absorption"], bundle["speed"], omega,
        bundle["phase"] == "flight", _start_norm_table(jmax), int(jmax),
        radial_low, radial_high, amplitude, envelope,
        float(absolute_tolerance_pe) / max(len(points), 1))
    carrier = np.exp(1j * omega * source0.reference_ns)
    return result.T * carrier[:, None]


def apply_multi_spectral(bundle, sources, receivers_m, look_directions, alpha,
                         out_scale, *, radial_window_m=None,
                         channel_dtype=np.complex64, absolute_tolerance_pe=0.0):
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
    if not np.isfinite(absolute_tolerance_pe) or absolute_tolerance_pe < 0:
        raise ValueError("absolute_tolerance_pe must be finite and nonnegative")
    amplitude = np.zeros((len(sources), 0))
    envelope = np.zeros((len(sources), 0))
    if absolute_tolerance_pe:
        # Bound the actual rounded channels supplied to the compiled kernel.
        amplitude = np.ascontiguousarray(np.stack([
            _source_norm_bound(replace(source, channels=channels[i]))
            for i, source in enumerate(sources)]))
        envelope = _radial_norm_bound(bundle, bundle["coefficients"], alpha)
    result = _apply_multi_spectral(
        local_points, local_receivers, local_looks, np.ascontiguousarray(channels),
        np.ascontiguousarray(out_scale, float), starts, int(bundle["degree"]),
        int(max_m), acceptance_degree, alpha, bundle["pair_lambda"], bundle["pair_mu"],
        bundle["log_r"], bundle["coefficients"], bundle["absorption"],
        bundle["speed"], omega, bundle["phase"] == "flight",
        _start_norm_table(jmax), int(jmax), radial_low, radial_high,
        amplitude, envelope, float(absolute_tolerance_pe) / max(len(points), 1))
    return result.T * np.exp(1j * omega * reference)[:, None]
