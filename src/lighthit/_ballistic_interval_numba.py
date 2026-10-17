"""Compiled positive ballistic quadrature on exact spectral support intervals.

All spectral tables share ``panel_edges``; efficiency coefficients are ordinary
powers of q=(lambda-left)/(right-left). Geometry and time extrema split panels
before quadrature, so narrow emission intervals or time bins cannot be skipped.

A receiver of radius ``R > 0`` is the OM sphere, not its centre. At one
wavelength the photons reaching the sphere leave the element from a window of
half-width ``R/sin(theta)`` around the emission point of the ray through the
centre, weighted by the semicircle law of the sphere's chords across the cone
surface. The point receiver's hard emission window ``[0, L)`` becomes the part
of that window inside the element. This is the leading order in ``R/rho``;
attenuation, acceptance and arrival time stay those of the centre ray.
"""
import numpy as np
from numba import njit, prange

_C = 0.299792458


@njit(cache=True)
def _acceptance(alpha, x):
    """OM acceptance from its Legendre coefficients at cosine ``x``."""
    result = alpha[0] / (4 * np.pi)
    previous, current = 1., x
    if len(alpha) > 1:
        result += 3 * alpha[1] * current / (4 * np.pi)
    for ell in range(2, len(alpha)):
        value = ((2 * ell - 1) * x * current - (ell - 1) * previous) / ell
        result += (2 * ell + 1) * alpha[ell] * value / (4 * np.pi)
        previous, current = current, value
    if result < -1e-12 * max(1., np.sum(np.abs(alpha))):
        raise ValueError('negative ballistic angular response')
    return max(0., result)


@njit(cache=True)
def _arrival(lam, left, width, n0, dn, g0, dg, beta, z, rho, tau, t0):
    """Arrival time of the light emitted at wavelength ``lam`` (phase and group index linear in a panel)."""
    q = (lam - left) / width
    n, group = n0 + q * dn, g0 + q * dg
    v = np.sqrt((beta * n - 1.) * (beta * n + 1.))
    root, distance = z - rho / v, rho * beta * n / v
    return t0 + tau * root + distance * group / _C


@njit(cache=True)
def _sort_small(values, count):
    """Insertion sort of ``values[:count]`` in place."""
    for k in range(1, count):
        value, m = values[k], k - 1
        while m >= 0 and values[m] > value:
            values[m + 1] = values[m]
            m -= 1
        values[m + 1] = value


@njit(cache=True)
def _stationary_cuts_into(cuts, a, b, left, width, n0, dn, g0, dg, beta, tau):
    """Write the sorted endpoints and internal arrival extrema to ``cuts``; return their count."""
    cuts[0], count = a, 1
    if dn != 0.:
        k = dg / dn
        h = g0 - k * n0
        aa = beta ** 3 * k / _C
        bb = tau * beta ** 2 - 2 * beta * k / _C
        cc = -beta * h / _C
        scale = max(abs(aa), abs(bb), abs(cc), 1e-300)
        aa, bb, cc = aa / scale, bb / scale, cc / scale
        x1 = n0 + (a - left) / width * dn
        x2 = n0 + (b - left) / width * dn
        lo, hi = min(x1, x2), max(x1, x2)
        middle, nb = hi, 2
        if aa != 0. and -bb / (3 * aa) > 0.:
            turning = np.sqrt(-bb / (3 * aa))
            if lo < turning < hi:
                middle, nb = turning, 3
        for index in range(nb - 1):
            l = lo if index == 0 else middle
            r = (middle if nb == 3 else hi) if index == 0 else hi
            fl = (aa * l * l + bb) * l + cc
            fr = (aa * r * r + bb) * r + cc
            if fl == 0.:
                root = l
            elif fr == 0.:
                root = r
            elif (fl < 0.) == (fr < 0.):
                continue
            else:
                for iteration in range(64):
                    mid = (l + r) * .5
                    if mid == l or mid == r:
                        break
                    fm = (aa * mid * mid + bb) * mid + cc
                    if (fl < 0.) == (fm < 0.):
                        l, fl = mid, fm
                    else:
                        r = mid
                root = (l + r) * .5
            lam = left + width * (root - n0) / dn
            if a < lam < b:
                duplicate = False
                for old in range(count):
                    duplicate = duplicate or abs(cuts[old] - lam) < 1e-12
                if not duplicate:
                    cuts[count], count = lam, count + 1
    cuts[count], count = b, count + 1
    _sort_small(cuts, count)
    return count


@njit(cache=True)
def _stationary_cuts(a, b, left, width, n0, dn, g0, dg, beta, tau):
    """Sorted wavelength endpoints and every internal arrival extremum."""
    cuts = np.empty(5)
    count = _stationary_cuts_into(cuts, a, b, left, width, n0, dn, g0, dg, beta, tau)
    return cuts[:count].copy()


@njit(cache=True)
def _time_root(target, a, b, left, width, n0, dn, g0, dg,
               beta, z, rho, tau, t0):
    """Wavelength in [a, b] whose arrival time equals ``target`` (bisection-secant)."""
    fa = _arrival(a, left, width, n0, dn, g0, dg, beta, z, rho, tau, t0) - target
    for iteration in range(64):
        mid = (a + b) * .5
        if mid == a or mid == b:
            break
        fm = _arrival(mid, left, width, n0, dn, g0, dg,
                      beta, z, rho, tau, t0) - target
        if (fa < 0.) == (fm < 0.):
            a, fa = mid, fm
        else:
            b = mid
    return (a + b) * .5


@njit(cache=True)
def _semicircle_cdf(x):
    """Distribution function of the semicircle law on [-1, 1]."""
    if x <= -1.:
        return 0.
    if x >= 1.:
        return 1.
    return .5 + (x * np.sqrt(1. - x * x) + np.arcsin(x)) / np.pi


@njit(cache=True)
def _support(z, rho, length, beta, radius):
    """Phase-index support of one element and the two inner cuts of its window.

    ``radius == 0``: the emission point of the centre ray lies in [0, L].
    ``radius > 0``: the window of half-width R/sin(theta) overlaps [0, L]; the
    inner cuts are the indices where the window starts to leave the element at
    either end (kinks of the window fraction). ``inf`` means no cut.
    """
    if radius == 0.:
        if z <= 0.:
            return False, 0., 0., np.inf, np.inf
        nmin = np.hypot(z, rho) / (beta * z)
        nmax = np.inf
        if z > length:
            residual_z = z - length
            nmax = np.hypot(residual_z, rho) / (beta * residual_z)
        return True, nmin, nmax, np.inf, np.inf
    # Window edge at s: sin(theta - psi(s)) = +-R/|w(s)|, psi the angle of the
    # centre seen from s.
    quarter = .5 * np.pi
    psi0, psi1 = np.arctan2(rho, z), np.arctan2(rho, z - length)
    a0 = np.arcsin(min(1., radius / np.hypot(z, rho)))
    a1 = np.arcsin(min(1., radius / np.hypot(z - length, rho)))
    low, high = psi0 - a0, psi1 + a1
    if low >= quarter:
        return False, 0., 0., np.inf, np.inf
    # A window reaching the cone axis starts just above the threshold, where the
    # arrival time is still finite and the light ~ sin^2(theta) vanishes.
    nmin = (1. + 1e-9) / beta if low <= 0. else max((1. + 1e-9) / beta, 1. / (beta * np.cos(low)))
    nmax = np.inf if high >= quarter else 1. / (beta * np.cos(high))
    inner0, inner1 = psi0 + a0, psi1 - a1
    cut0 = np.inf if inner0 >= quarter else 1. / (beta * np.cos(inner0))
    cut1 = np.inf if (inner1 <= 0. or inner1 >= quarter) else 1. / (beta * np.cos(inner1))
    return True, nmin, nmax, cut0, cut1


@njit(cache=True)
def _integral(a, b, left, width, n0, dn, e0, de, efficiency,
              beta, rho, transverse_look, axial_look, coefficient0,
              coefficient2, length, area, alpha, gx, gw, z, radius):
    """Detected charge from the wavelength interval [a, b] (Gauss quadrature)."""
    total = 0.
    half, middle = (b - a) * .5, (b + a) * .5
    for j in range(len(gx)):
        lam = middle + half * gx[j]
        q = (lam - left) / width
        n, extinction = n0 + dn * q, e0 + de * q
        bn = beta * n
        if radius > 0. and bn <= 1.:
            continue
        v = np.sqrt((bn - 1.) * (bn + 1.))
        sine, cosine = v / bn, 1. / bn
        window = 1.
        if radius > 0.:
            # Emission point of the centre ray and the window half-width R/sin(theta).
            centre, spread = z - rho / v, radius / sine
            window = _semicircle_cdf((length - centre) / spread) - _semicircle_cdf(-centre / spread)
            if window <= 0.:
                continue
        distance = rho / sine
        angle = min(1., max(-1., transverse_look * sine + axial_look * cosine))
        response = ((efficiency[3] * q + efficiency[2]) * q + efficiency[1]) * q + efficiency[0]
        photons = coefficient0 + coefficient2 / (n * n)
        if response < -1e-12 or photons < -1e-12 * max(1., coefficient0):
            raise ValueError('negative ballistic spectral weight')
        total += (gw[j] * max(0., response) * max(0., photons) / (lam * lam)
                  * np.exp(-extinction * distance) * _acceptance(alpha, angle)
                  / (2 * np.pi * rho * sine) * window)
    return total * half * area / length


@njit(cache=True)
def _precheck(om, i, receivers, radii, reach, start, direction, length, coefficient0, beta,
              global_min, global_max):
    """Geometry for a pair whose direct spectrum can be nonempty, else ok=False.

    The receiver is the OM sphere of radius ``radii[om]`` when the element
    comes closer to the OM centre than ``reach``, else the centre itself.
    """
    if coefficient0[i] == 0. or beta[i] * global_max <= 1.:
        return False, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.
    dx = receivers[om, 0] - start[i, 0]
    dy = receivers[om, 1] - start[i, 1]
    dz = receivers[om, 2] - start[i, 2]
    ux, uy, uz = direction[i, 0], direction[i, 1], direction[i, 2]
    z = dx * ux + dy * uy + dz * uz
    # Behind the element only a sphere within reach can be lit.
    if z <= 0. and (radii[om] == 0. or z <= -reach):
        return False, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.
    cx, cy, cz = dy * uz - dz * uy, dz * ux - dx * uz, dx * uy - dy * ux
    rho = np.sqrt(cx * cx + cy * cy + cz * cz)
    radius = 0.
    if radii[om] > 0. and rho < reach:
        along = z if z < 0. else (z - length[i] if z > length[i] else 0.)
        if along * along + rho * rho < reach * reach:
            radius = radii[om]
    if z <= 0. and radius == 0.:
        return False, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.
    if rho == 0.:
        if 0. < z < length[i]:
            raise ValueError('point receiver lies on emitting track: singular ballistic fluence')
        return False, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.
    ok, nmin, nmax, cut0, cut1 = _support(z, rho, length[i], beta[i], radius)
    if not ok or nmin > global_max or nmax <= global_min:
        return False, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.
    return True, dx, dy, dz, ux, uy, uz, z, rho, nmin, nmax, cut0, cut1, radius


@njit(cache=True)
def _insert_cuts(cuts, count, left, width, n0, dn, cut0, cut1):
    """Add the wavelengths of the phase indices cut0, cut1 inside the cuts; return the count."""
    if dn == 0.:
        return count
    low, high = cuts[0], cuts[count - 1]
    for value in (cut0, cut1):
        if np.isfinite(value):
            lam = left + width * (value - n0) / dn
            if low < lam < high:
                cuts[count], count = lam, count + 1
    _sort_small(cuts, count)
    return count


@njit(cache=True)
def _pair_contribution(om, i, row, receivers, looks, areas, radius, start, direction, length,
                       coefficient0, coefficient2, beta, start_ns, end_ns,
                       panel_edges, phase_nodes, group_nodes, extinction_nodes,
                       efficiency_coefficients, alpha, gauss_x, gauss_w,
                       origins, bin_edges, want_bins, nbin, global_min, monotonic,
                       nondecreasing, increasing_phase, dx, dy, dz, ux, uy, uz, z, rho,
                       nmin, nmax, cut0, cut1, charge, bins, diagnostics):
    """Spectral and time pieces of one element for one receiver (output row)."""
    diagnostics[0] += 1
    cuts = np.empty(7)
    pieces = np.empty(nbin + 3)
    lx, ly, lz = looks[om, 0], looks[om, 1], looks[om, 2]
    transverse_look = ((dx-z*ux)*lx + (dy-z*uy)*ly + (dz-z*uz)*lz) / rho
    axial_look = ux*lx + uy*ly + uz*lz
    tau = (end_ns[i] - start_ns[i]) / length[i]
    first_panel, last_panel = 0, len(panel_edges) - 1
    if monotonic:
        lower = nmin if nondecreasing else -nmax
        upper = nmax if nondecreasing else -nmin
        # Include the bracketing panel and one guard on either side.
        # searchsorted handles long plateaus without dropping support.
        first_panel = max(0, np.searchsorted(increasing_phase, lower, side='left') - 2)
        last_panel = min(last_panel, np.searchsorted(increasing_phase, upper, side='right') + 1)
    for panel in range(first_panel, last_panel):
        left, right = panel_edges[panel], panel_edges[panel + 1]
        width = right - left
        n0, dn = phase_nodes[panel], phase_nodes[panel + 1] - phase_nodes[panel]
        g0, dg = group_nodes[panel], group_nodes[panel + 1] - group_nodes[panel]
        e0, de = extinction_nodes[panel], extinction_nodes[panel + 1] - extinction_nodes[panel]
        a, b = left, right
        if dn == 0.:
            if n0 < nmin or n0 >= nmax or beta[i] * n0 <= 1.:
                continue
        else:
            q1, q2 = (nmin - n0) / dn, (nmax - n0) / dn
            a = left + width * max(0., min(q1, q2))
            b = left + width * min(1., max(q1, q2))
            if b <= a:
                continue
        # Physical geometry excludes the Cherenkov threshold for a point at rho>0;
        # a sphere's support may start at the threshold, where the light vanishes.
        if radius == 0. and min(n0 + (a-left)/width*dn, n0 + (b-left)/width*dn) * beta[i] <= 1.:
            raise ValueError('unresolved near-axis Cherenkov threshold')
        if want_bins:
            ncut = _stationary_cuts_into(cuts, a, b, left, width, n0, dn, g0, dg, beta[i], tau)
        else:
            cuts[0], cuts[1], ncut = a, b, 2
        if radius > 0.:
            ncut = _insert_cuts(cuts, ncut, left, width, n0, dn, cut0, cut1)
        for segment in range(ncut - 1):
            a, b = cuts[segment], cuts[segment + 1]
            pieces[0], count = a, 1
            if want_bins:
                ta = _arrival(a, left, width, n0, dn, g0, dg, beta[i], z, rho, tau, start_ns[i]) - origins[row]
                tb = _arrival(b, left, width, n0, dn, g0, dg, beta[i], z, rho, tau, start_ns[i]) - origins[row]
                first = np.searchsorted(bin_edges, min(ta, tb), side='right')
                last = np.searchsorted(bin_edges, max(ta, tb), side='left')
                for j in range(first, last):
                    edge_index = j if tb > ta else last - 1 - (j - first)
                    root = _time_root(bin_edges[edge_index] + origins[row], a, b,
                                      left, width, n0, dn, g0, dg,
                                      beta[i], z, rho, tau, start_ns[i])
                    pieces[count], count = root, count + 1
            pieces[count], count = b, count + 1
            for piece in range(count - 1):
                p, q = pieces[piece], pieces[piece + 1]
                value = _integral(p, q, left, width, n0, dn, e0, de,
                                  efficiency_coefficients[panel], beta[i], rho,
                                  transverse_look, axial_look, coefficient0[i], coefficient2[i],
                                  length[i], areas[om], alpha, gauss_x, gauss_w, z, radius)
                charge[row] += value
                diagnostics[1] += 1
                diagnostics[2] += len(gauss_x)
                if want_bins:
                    time = _arrival((p+q)*.5, left, width, n0, dn, g0, dg,
                                    beta[i], z, rho, tau, start_ns[i]) - origins[row]
                    slot = np.searchsorted(bin_edges, time, side='right') - 1
                    if 0 <= slot < nbin:
                        bins[row, slot] += value


@njit(cache=True)
def _phase_order(phase_nodes):
    """Global range and monotonicity of the phase-index nodes."""
    nondecreasing, nonincreasing = True, True
    for index in range(len(phase_nodes) - 1):
        nondecreasing = nondecreasing and phase_nodes[index + 1] >= phase_nodes[index]
        nonincreasing = nonincreasing and phase_nodes[index + 1] <= phase_nodes[index]
    increasing_phase = phase_nodes if nondecreasing else -phase_nodes
    return (np.min(phase_nodes), np.max(phase_nodes), nondecreasing or nonincreasing,
            nondecreasing, increasing_phase)


@njit(cache=True)
def interval_kernel(receivers, looks, areas, start, direction, length,
                    coefficient0, coefficient2, beta, start_ns, end_ns,
                    panel_edges, phase_nodes, group_nodes, extinction_nodes,
                    efficiency_coefficients, alpha, gauss_x, gauss_w,
                    origins, bin_edges, want_bins, radii, reach):
    """Return charge, bins and counts (geometry pairs, panels, evaluations).

    ``looks`` points along accepted photon propagation (minus OM orientation).
    ``radii`` are the OM sphere radii, used for elements closer than ``reach``
    to the OM centre (0 or ``reach`` = 0: the point receiver at the centre).
    This low-level routine assumes finite validated arrays and positive lengths.
    Bins own their left endpoint; emission segments own [0,length).
    """
    nbin = len(bin_edges) - 1 if want_bins else 0
    charge = np.zeros(len(receivers))
    bins = np.zeros((len(receivers), nbin))
    diagnostics = np.zeros(3, dtype=np.int64)
    global_min, global_max, monotonic, nondecreasing, increasing_phase = _phase_order(phase_nodes)
    for om in range(len(receivers)):
        for i in range(len(start)):
            ok, dx, dy, dz, ux, uy, uz, z, rho, nmin, nmax, cut0, cut1, radius = _precheck(
                om, i, receivers, radii, reach, start, direction, length, coefficient0, beta,
                global_min, global_max)
            if not ok:
                continue
            _pair_contribution(om, i, om, receivers, looks, areas, radius, start, direction, length,
                coefficient0, coefficient2, beta, start_ns, end_ns, panel_edges,
                phase_nodes, group_nodes, extinction_nodes, efficiency_coefficients,
                alpha, gauss_x, gauss_w, origins, bin_edges, want_bins, nbin, global_min,
                monotonic, nondecreasing, increasing_phase, dx, dy, dz, ux, uy, uz, z,
                rho, nmin, nmax, cut0, cut1, charge, bins, diagnostics)
    return charge, bins, diagnostics


@njit(cache=True)
def interval_candidates(receivers, radii, reach, start, direction, length, coefficient0, beta, phase_nodes):
    """Receiver/element pairs passing the order-independent geometry precheck.

    Pairs are listed receiver-major in element order, the loop order of
    ``interval_kernel``, so a pair-list evaluation sums identically.
    """
    global_min, global_max = np.min(phase_nodes), np.max(phase_nodes)
    capacity = 1024
    oms = np.empty(capacity, np.int64)
    elements = np.empty(capacity, np.int64)
    count = 0
    for om in range(len(receivers)):
        sphere = radii[om] > 0.
        for i in range(len(start)):
            # The test of _precheck, written out: this loop visits every pair.
            if coefficient0[i] == 0. or beta[i] * global_max <= 1.:
                continue
            dx = receivers[om, 0] - start[i, 0]
            dy = receivers[om, 1] - start[i, 1]
            dz = receivers[om, 2] - start[i, 2]
            ux, uy, uz = direction[i, 0], direction[i, 1], direction[i, 2]
            z = dx * ux + dy * uy + dz * uz
            if z <= 0. and (not sphere or z <= -reach):
                continue
            cx, cy, cz = dy * uz - dz * uy, dz * ux - dx * uz, dx * uy - dy * ux
            rho = np.sqrt(cx * cx + cy * cy + cz * cz)
            radius = 0.
            if sphere and rho < reach:
                along = z if z < 0. else (z - length[i] if z > length[i] else 0.)
                if along * along + rho * rho < reach * reach:
                    radius = radii[om]
            if z <= 0. and radius == 0.:
                continue
            if rho == 0.:
                if 0. < z < length[i]:
                    raise ValueError('point receiver lies on emitting track: singular ballistic fluence')
                continue
            if radius == 0.:
                nmin = np.hypot(z, rho) / (beta[i] * z)
                nmax = np.inf
                if z > length[i]:
                    residual_z = z - length[i]
                    nmax = np.hypot(residual_z, rho) / (beta[i] * residual_z)
                ok = True
            else:
                ok, nmin, nmax, cut0, cut1 = _support(z, rho, length[i], beta[i], radius)
            if not ok or nmin > global_max or nmax <= global_min:
                continue
            if count == capacity:
                capacity *= 2
                grown = np.empty(capacity, np.int64); grown[:count] = oms[:count]; oms = grown
                grown = np.empty(capacity, np.int64); grown[:count] = elements[:count]; elements = grown
            oms[count] = om
            elements[count] = i
            count += 1
    return oms[:count], elements[:count]


@njit(cache=True, parallel=True)
def interval_kernel_pairs(pair_om, pair_element, pair_row, n_rows, receivers, looks, areas,
                          start, direction, length, coefficient0, coefficient2, beta,
                          start_ns, end_ns, panel_edges, phase_nodes, group_nodes,
                          extinction_nodes, efficiency_coefficients, alpha, gauss_x,
                          gauss_w, row_origins, bin_edges, want_bins, radii, reach):
    """``interval_kernel`` restricted to listed pairs; output row per pair."""
    nbin = len(bin_edges) - 1 if want_bins else 0
    charge = np.zeros(n_rows)
    bins = np.zeros((n_rows, nbin))
    diagnostics = np.zeros(3, dtype=np.int64)
    global_min, global_max, monotonic, nondecreasing, increasing_phase = _phase_order(phase_nodes)
    # Pairs come row-contiguous; every row is summed by one thread in the
    # original pair order, so the result does not depend on the thread count.
    runs = [0]
    for k in range(1, len(pair_row)):
        if pair_row[k] != pair_row[k - 1]:
            runs.append(k)
    runs.append(len(pair_row))
    bounds = np.array(runs, np.int64)
    counters = np.zeros((max(len(bounds) - 1, 0), 3), np.int64)
    for r in prange(len(bounds) - 1):
        local = counters[r]
        for k in range(bounds[r], bounds[r + 1]):
            om, i, row = pair_om[k], pair_element[k], pair_row[k]
            ok, dx, dy, dz, ux, uy, uz, z, rho, nmin, nmax, cut0, cut1, radius = _precheck(
                om, i, receivers, radii, reach, start, direction, length, coefficient0, beta,
                global_min, global_max)
            if not ok:
                continue
            _pair_contribution(om, i, row, receivers, looks, areas, radius, start, direction, length,
                coefficient0, coefficient2, beta, start_ns, end_ns, panel_edges,
                phase_nodes, group_nodes, extinction_nodes, efficiency_coefficients,
                alpha, gauss_x, gauss_w, row_origins, bin_edges, want_bins, nbin, global_min,
                monotonic, nondecreasing, increasing_phase, dx, dy, dz, ux, uy, uz, z,
                rho, nmin, nmax, cut0, cut1, charge, bins, local)
    for r in range(counters.shape[0]):
        for c in range(3):
            diagnostics[c] += counters[r, c]
    return charge, bins, diagnostics
