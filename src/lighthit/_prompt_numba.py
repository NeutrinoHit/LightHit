"""Compiled kernels for the first-order prompt contribution.

See :mod:`lighthit.prompt` for the physics contract and
:mod:`lighthit.prompt_reference` for the slow independent reference.

Pencil
    One directed point emitter ``(x, s0, t_emit)``.  For a module at distance
    ``r`` with ``theta = angle(s0, R - x)``, parametrise the scatter point by
    ``y = tan(chi / 2)`` (``chi`` the scattering angle).  Exactly,

        S(y)  = r cos(theta) + r sin(theta) y                    (total path)
        cos chi = (1 - y^2)/(1 + y^2),  sin chi = 2y/(1 + y^2)
        x(y)  = cos chi c1 + sin chi c3                  (acceptance argument)
        dN1/dS = mu_s exp(-mu_t S) * G(S),
        G(S)  = W / (r sin theta) * p(cos chi) A(x) * 2 / ((1 + y^2) r sin theta)

    with ``c1 = s0 . n`` and ``c3 = e_perp . n``.  ``G`` does not depend on the
    wavelength.

Batch
    Pencils that share the emission point and time share ``r``, the front
    ``S = r`` and the arrival time ``t = t_emit + S / v``.  Their ``G`` are
    summed on one path grid, uniform in ``asinh((S - r) / sigma)`` (linear at
    the front, geometric in the power-law tail), and deposited once per
    wavelength.  All azimuth nodes of one ring point of a track are a batch.

Deposit
    For one wavelength the time density is ``exp(kappa t) H(t)`` with the
    shared ``kappa = -mu_t v`` and ``H`` quadratic in ``t`` between grid nodes;
    the exponential is exact.  Short panels are integrated over the actual
    bin edges directly; long panels become start/stop events of global
    quadratics that one prefix pass per module and wavelength integrates
    over the bins.  No batch loops over all bins.

Front
    The density jumps at the front ``t_f``.  A quadrature cell of emission
    points spreads its fronts over ``W``; the step ``H(t_f) exp(kappa (t -
    t_f))`` is replaced by a ramp of width ``W`` normalised to the exact step
    mass, the continuous remainder is deposited as is.  ``W -> 0`` under
    refinement.

Acceptance
    ``A`` is a polynomial of degree ``L_A``.  A single pencil evaluates it by
    Horner; a group carries the exact moments ``sum w c1^i c3^j``.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

__all__ = []

TWO_PI = 2.0 * math.pi


# ---------------------------------------------------------------- utilities

@njit(cache=True, nogil=True, inline="always")
def _hg(cosine, g):
    denominator = 1.0 + g * g - 2.0 * g * cosine
    return (1.0 - g * g) / (4.0 * math.pi * denominator * math.sqrt(denominator))


_GL48 = np.polynomial.legendre.leggauss(48)
_GAUSS_TAU = 0.25 * math.pi * (_GL48[0] + 1.0)
_GAUSS_W = 0.25 * math.pi * _GL48[1]


@njit(cache=True, nogil=True)
def gauss_inverse_mean(u, v, var_u, var_v, tau, weight):
    """Mean of ``1/|y - p|`` for ``y ~ N(0, diag(var_u, var_v))`` in 2D and
    ``p = (u, v)``.

    From ``1/rho = (2/sqrt(pi)) int_0^inf exp(-s^2 rho^2) ds`` and the
    Gaussian moment generating function,

        E[1/rho] = (2/sqrt(pi)) int_0^inf ds prod_i (1 + 2 s^2 var_i)^(-1/2)
                   exp(-s^2 sum_i p_i^2 / (1 + 2 s^2 var_i)),

    evaluated with ``s = s0 tan(tau)`` and 48-point Gauss--Legendre in
    ``tau`` (``tau``/``weight`` are those nodes on [0, pi/2]).
    """
    scale = 2.0 * (var_u + var_v) + u * u + v * v
    if scale <= 0.0:
        return 1e300
    s0 = 1.0 / math.sqrt(scale)
    total = 0.0
    for k in range(tau.shape[0]):
        t = math.tan(tau[k])
        sec2 = 1.0 + t * t
        s = s0 * t
        s2 = s * s
        a = 1.0 + 2.0 * s2 * var_u
        b = 1.0 + 2.0 * s2 * var_v
        total += weight[k] * sec2 * math.exp(-s2 * (u * u / a + v * v / b)) / math.sqrt(a * b)
    return 2.0 / math.sqrt(math.pi) * s0 * total


@njit(cache=True, nogil=True)
def _ifuncs(z, ez):
    """``I_k(z) = int_0^1 x^k exp(z x) dx`` for k = 0, 1, 2 (stable)."""
    if abs(z) < 1.0:
        term = 1.0
        i0 = 0.0
        i1 = 0.0
        i2 = 0.0
        for n in range(24):
            i0 += term / (n + 1.0)
            i1 += term / (n + 2.0)
            i2 += term / (n + 3.0)
            term *= z / (n + 1.0)
        return i0, i1, i2
    return ((ez - 1.0) / z,
            (ez * (z - 1.0) + 1.0) / (z * z),
            (ez * (z * z - 2.0 * z + 2.0) - 2.0) / (z * z * z))


@njit(cache=True, nogil=True, inline="always")
def _quad_integral(e_start, alpha, beta, gamma, width, kappa, e_end):
    """``int_0^width exp(kappa (t0 + s)) (alpha + beta s + gamma s^2) ds``."""
    z = kappa * width
    ez = e_end / e_start if e_start > 0.0 else math.exp(z)
    i0, i1, i2 = _ifuncs(z, ez)
    return e_start * width * (alpha * i0 + width * (beta * i1 + width * gamma * i2))


@njit(cache=True, nogil=True)
def _find_bin(edges, t):
    """Index k with edges[k] <= t < edges[k+1]; -1 before, nb after."""
    nb = edges.shape[0] - 1
    if t < edges[0]:
        return -1
    if t >= edges[nb]:
        return nb
    low = 0
    high = nb
    while high - low > 1:
        middle = (low + high) // 2
        if t < edges[middle]:
            high = middle
        else:
            low = middle
    return low


@njit(cache=True, nogil=True)
def _deposit_panel(lam, t_a, t_b, e_a, e_b, alpha, beta, gamma, kappa,
                   edges, eedge, pre, acc, direct_bins):
    """Put one quadratic panel ``[t_a, t_b]`` of ``exp(kappa t) q(t)`` into bins.

    ``q(t) = alpha + beta s + gamma s^2`` with ``s = t - t_a``.  Panels that
    cover at most ``direct_bins`` bins are integrated bin by bin in the local
    variable; longer panels become start/stop events of a global quadratic
    in ``t`` and are summed by the prefix pass of :func:`_finalize`.
    """
    nb = edges.shape[0] - 1
    lo = t_a if t_a > edges[0] else edges[0]
    hi = t_b if t_b < edges[nb] else edges[nb]
    if hi <= lo:
        return
    k_lo = _find_bin(edges, lo)
    k_hi = _find_bin(edges, hi)
    if k_hi >= nb or (k_hi > k_lo and hi <= edges[k_hi]):
        k_hi -= 1
    if k_hi - k_lo + 1 <= direct_bins:
        for k in range(k_lo, k_hi + 1):
            s_lo = lo if lo > edges[k] else edges[k]
            s_hi = hi if hi < edges[k + 1] else edges[k + 1]
            if s_hi <= s_lo:
                continue
            e_lo = e_a if s_lo == t_a else eedge[lam, k]
            if s_lo != t_a and s_lo != edges[k]:
                e_lo = math.exp(kappa * s_lo)
            e_hi = e_b if s_hi == t_b else eedge[lam, k + 1]
            if s_hi != t_b and s_hi != edges[k + 1]:
                e_hi = math.exp(kappa * s_hi)
            shift = s_lo - t_a
            a_loc = alpha + shift * (beta + shift * gamma)
            b_loc = beta + 2.0 * gamma * shift
            acc[lam, k, 3] += _quad_integral(e_lo, a_loc, b_loc, gamma,
                                             s_hi - s_lo, kappa, e_hi)
        return
    # long panel: global coefficients in t
    big_a = alpha - t_a * (beta - gamma * t_a)
    big_b = beta - 2.0 * gamma * t_a
    big_c = gamma
    if t_a < edges[0]:
        pre[lam, 0] += big_a
        pre[lam, 1] += big_b
        pre[lam, 2] += big_c
    else:
        k = k_lo
        acc[lam, k, 0] += big_a
        acc[lam, k, 1] += big_b
        acc[lam, k, 2] += big_c
        # part of bin k before t_a is not covered: finalize integrates the
        # whole bin with the *previous* prefix; this panel's share of bin k
        # is added here explicitly.
        acc[lam, k, 3] += _quad_integral(e_a, alpha, beta, gamma,
                                         edges[k + 1] - t_a, kappa, eedge[lam, k + 1])
    if t_b < edges[nb]:
        k = _find_bin(edges, t_b)
        acc[lam, k, 0] -= big_a
        acc[lam, k, 1] -= big_b
        acc[lam, k, 2] -= big_c
        shift = t_b - t_a
        a_loc = alpha + shift * (beta + shift * gamma)
        b_loc = beta + 2.0 * gamma * shift
        # the prefix of bin k (> start bin) includes this panel over the
        # whole bin; remove the part after t_b.
        acc[lam, k, 3] -= _quad_integral(e_b, a_loc, b_loc, gamma,
                                         edges[k + 1] - t_b, kappa, eedge[lam, k + 1])



# -------------------------------------------------------------- batch core

@njit(cache=True, nogil=True)
def _batch_grid(r, sigma, t_emit_rel, h, h_tail, tail_exp, mut_min, speed_max,
                edge_end, s_buf):
    """Path nodes ``S = r + sigma sinh(u)``, step ``h`` in ``u`` inside the
    time window and ``h_tail`` beyond it, up to ``mu_t,min (S - r) =
    tail_exp``.  Returns the node count (the first node is ``S = r``)."""
    capacity = s_buf.shape[0]
    ext_end = tail_exp / mut_min
    ext_win = speed_max * (edge_end - t_emit_rel) - r
    if ext_win < 0.0:
        ext_win = 0.0
    if ext_win > ext_end:
        ext_win = ext_end
    u_end = math.asinh(ext_end / sigma)
    u_win = math.asinh(ext_win / sigma)
    n_in = int(math.ceil(u_win / h)) if u_win > 0.0 else 0
    n_out = int(math.ceil((u_end - u_win) / h_tail)) if u_end > u_win else 0
    if n_in + n_out + 1 > capacity:
        scale = (capacity - 1.0) / (n_in + n_out)
        n_in = int(n_in * scale)
        n_out = capacity - 1 - n_in
    s_buf[0] = r
    n = 1
    for i in range(1, n_in + 1):
        s_buf[n] = r + sigma * math.sinh(u_win * i / n_in)
        n += 1
    for i in range(1, n_out + 1):
        s_buf[n] = r + sigma * math.sinh(u_win + (u_end - u_win) * i / n_out)
        n += 1
    return n


@njit(cache=True, nogil=True, inline="always")
def _batch_sigma(r, sin_min, tan_half_min, y_g):
    """Smallest path scale resolved by a batch grid."""
    return max(0.5 * r * sin_min * max(y_g, 0.5 * tan_half_min), 1e-9 * r)


@njit(cache=True, nogil=True)
def _batch_add_pencil(n, s_buf, r, sin_t, cos_t, pref, jac_sin, w0, w2, c1, c3,
                      apoly, g, g0, g2):
    """Add one pencil with polynomial acceptance ``apoly`` (monomial
    coefficients in ``x``) to the batch profile ``g0, g2``.  ``pref`` is the
    singular weight ``1/(r sin theta)`` (or its disk mean), ``jac_sin`` the
    sine used in the Jacobian ``dchi/dS``."""
    rs = r * sin_t
    y_theta = sin_t / (1.0 + cos_t)          # tan(theta/2)
    jac = 2.0 * pref / (r * jac_sin)
    la = apoly.shape[0] - 1
    g1 = 1.0 + g * g
    gg = 2.0 * g
    norm = (1.0 - g * g) / (4.0 * math.pi)
    for k in range(n):
        y = (s_buf[k] - r) / rs + y_theta
        inv = 1.0 / (1.0 + y * y)
        c = (1.0 - y * y) * inv
        s = 2.0 * y * inv
        x = c * c1 + s * c3
        acc = apoly[la]
        for q in range(la - 1, -1, -1):
            acc = acc * x + apoly[q]
        den = g1 - gg * c
        value = jac * inv * norm * acc / (den * math.sqrt(den))
        g0[k] += w0 * value
        g2[k] += w2 * value


@njit(cache=True, nogil=True)
def _batch_add_moments(n, s_buf, r, sin_t, cos_t, jac_sin, mom0, mom2,
                       coef, ipow, jpow, max_power, g, g0, g2, cpow, spow):
    """Add a group with singular-weighted acceptance moments ``mom0/mom2``
    (``mom*[0]`` is ``sum w f / r``) to the batch profile."""
    rs = r * sin_t
    y_theta = sin_t / (1.0 + cos_t)
    jac = 2.0 / (r * jac_sin)
    n_mono = coef.shape[0]
    g1 = 1.0 + g * g
    gg = 2.0 * g
    norm = (1.0 - g * g) / (4.0 * math.pi)
    for k in range(n):
        y = (s_buf[k] - r) / rs + y_theta
        inv = 1.0 / (1.0 + y * y)
        c = (1.0 - y * y) * inv
        s = 2.0 * y * inv
        cpow[0] = 1.0
        spow[0] = 1.0
        for p in range(1, max_power + 1):
            cpow[p] = cpow[p - 1] * c
            spow[p] = spow[p - 1] * s
        a0 = 0.0
        a2 = 0.0
        for m in range(n_mono):
            basis = coef[m] * cpow[ipow[m]] * spow[jpow[m]]
            a0 += basis * mom0[m]
            a2 += basis * mom2[m]
        den = g1 - gg * c
        value = jac * inv * norm / (den * math.sqrt(den))
        g0[k] += a0 * value
        g2[k] += a2 * value


@njit(cache=True, nogil=True)
def _batch_deposit(n, s_buf, g0, g2, r, t_emit_rel, tail_exp,
                   mus, mut, speed, s0w, s2w, edges, eedge, pre, acc, charge,
                   direct_bins, t_buf, h_buf,
                   front_extra, front_da, front_dtda, front_cos):
    """Deposit a batch profile for every wavelength (see module docstring)."""
    n_lambda = mus.shape[0]
    for lam in range(n_lambda):
        if mus[lam] == 0.0:
            continue
        v = speed[lam]
        kappa = -mut[lam] * v
        e_emit = math.exp(kappa * t_emit_rel)
        h_scale = mus[lam] * v / e_emit
        w0 = s0w[lam] * h_scale
        w2 = s2w[lam] * h_scale
        last = n - 1
        limit = r + tail_exp / mut[lam]
        while last > 2 and s_buf[last - 1] > limit:
            last -= 1
        if last < 2:
            continue
        for j in range(last + 1):
            t_buf[j] = t_emit_rel + s_buf[j] / v
            h_buf[j] = w0 * g0[j] + w2 * g2[j]
        h_front = h_buf[0]
        for j in range(last + 1):
            h_buf[j] -= h_front
        total = 0.0
        e_first = e_emit * math.exp(-mut[lam] * s_buf[0])
        e_a = e_first
        e_last = e_first
        j = 0
        while j < last:
            if j + 2 <= last:
                t_a = t_buf[j]
                t_m = t_buf[j + 1]
                t_b = t_buf[j + 2]
                u_m = t_m - t_a
                u_b = t_b - t_a
                d1 = (h_buf[j + 1] - h_buf[j]) / u_m
                d2 = ((h_buf[j + 2] - h_buf[j + 1]) / (t_b - t_m) - d1) / u_b
                alpha = h_buf[j]
                beta = d1 - d2 * u_m
                gamma = d2
                e_b = e_emit * math.exp(-mut[lam] * s_buf[j + 2])
                step = 2
            else:
                t_p = t_buf[j - 1]
                t_a = t_buf[j]
                t_b = t_buf[j + 1]
                u_m = t_a - t_p
                u_b = t_b - t_p
                d1 = (h_buf[j] - h_buf[j - 1]) / u_m
                d2 = ((h_buf[j + 1] - h_buf[j]) / (t_b - t_a) - d1) / u_b
                alpha = h_buf[j]
                beta = d1 - d2 * u_m + 2.0 * d2 * u_m
                gamma = d2
                e_b = e_emit * math.exp(-mut[lam] * s_buf[j + 1])
                step = 1
            width = t_b - t_a
            total += _quad_integral(e_a, alpha, beta, gamma, width, kappa, e_b)
            _deposit_panel(lam, t_a, t_b, e_a, e_b, alpha, beta, gamma, kappa,
                           edges, eedge, pre, acc, direct_bins)
            e_a = e_b
            e_last = e_b
            j += step
        # front step as a mass-normalised ramp of width W
        t_f = t_buf[0]
        t_end = t_buf[last]
        spread = front_extra + abs(front_dtda - front_cos / v) * front_da
        if spread > t_end - t_f:
            spread = t_end - t_f
        exact = _quad_integral(e_first, h_front, 0.0, 0.0, t_end - t_f, kappa, e_last)
        total += exact
        if spread > 0.0:
            t_1 = t_f - 0.5 * spread
            t_2 = t_f + 0.5 * spread
            e_1 = math.exp(kappa * t_1)
            e_2 = math.exp(kappa * t_2)
            ramp = _quad_integral(e_1, 0.0, 1.0 / spread, 0.0, spread, kappa, e_2)
            flat = (_quad_integral(e_2, 1.0, 0.0, 0.0, t_end - t_2, kappa, e_last)
                    if t_end > t_2 else 0.0)
            height = exact / (ramp + flat) if ramp + flat != 0.0 else 0.0
            _deposit_panel(lam, t_1, t_2, e_1, e_2, 0.0, height / spread, 0.0, kappa,
                           edges, eedge, pre, acc, direct_bins)
            if t_end > t_2:
                _deposit_panel(lam, t_2, t_end, e_2, e_last, height, 0.0, 0.0, kappa,
                               edges, eedge, pre, acc, direct_bins)
        elif t_end > t_f:
            _deposit_panel(lam, t_f, t_end, e_first, e_last, h_front, 0.0, 0.0, kappa,
                           edges, eedge, pre, acc, direct_bins)
        charge[lam] += total


@njit(cache=True, nogil=True, inline="always")
def _pencil_frame(dx, dy, dz, sx, sy, sz, nx, ny, nz):
    """(r, theta, sin, cos, c1, c3) for one pencil and one module."""
    r = math.sqrt(dx * dx + dy * dy + dz * dz)
    hx = dx / r
    hy = dy / r
    hz = dz / r
    cosine = hx * sx + hy * sy + hz * sz
    px = hx - cosine * sx
    py = hy - cosine * sy
    pz = hz - cosine * sz
    sine = math.sqrt(px * px + py * py + pz * pz)
    c1 = sx * nx + sy * ny + sz * nz
    if sine > 1e-300:
        c3 = (px * nx + py * ny + pz * nz) / sine
    else:
        c3 = 0.0
    theta = math.atan2(sine, cosine)
    return r, theta, sine, cosine, c1, c3


@njit(cache=True, nogil=True)
def _finalize(edges, eedge, itab, pre, acc, charge, area, bins_out):
    """Prefix pass over bins: global quadratic lines plus local corrections."""
    n_lambda = pre.shape[0]
    nb = edges.shape[0] - 1
    total_charge = 0.0
    for lam in range(n_lambda):
        a = pre[lam, 0]
        b = pre[lam, 1]
        c = pre[lam, 2]
        for k in range(nb):
            e_k = edges[k]
            width = edges[k + 1] - e_k
            a_loc = a + e_k * (b + c * e_k)
            b_loc = b + 2.0 * c * e_k
            value = eedge[lam, k] * width * (
                a_loc * itab[lam, k, 0]
                + width * (b_loc * itab[lam, k, 1] + width * c * itab[lam, k, 2]))
            bins_out[k] += area * (value + acc[lam, k, 3])
            a += acc[lam, k, 0]
            b += acc[lam, k, 1]
            c += acc[lam, k, 2]
        total_charge += charge[lam]
    return area * total_charge


def interval_table(edges, kappa):
    """``I_k(kappa * width)`` for every bin and wavelength, shape (L, nb, 3)."""
    edges = np.asarray(edges, float)
    width = np.diff(edges)
    out = np.empty((len(kappa), len(width), 3))
    for lam, value in enumerate(kappa):
        for k, w in enumerate(width):
            z = value * w
            out[lam, k] = _ifuncs(z, math.exp(z))
    return out


@njit(cache=True)
def _gauss_legendre(n):
    """Gauss--Legendre nodes and weights on [-1, 1] (Newton on P_n)."""
    x = np.empty(n)
    w = np.empty(n)
    for i in range(n):
        z = math.cos(math.pi * (i + 0.75) / (n + 0.5))
        for _ in range(100):
            p1 = 1.0
            p2 = 0.0
            for j in range(1, n + 1):
                p3 = p2
                p2 = p1
                p1 = ((2.0 * j - 1.0) * z * p2 - (j - 1.0) * p3) / j
            pp = n * (z * p1 - p2) / (z * z - 1.0)
            z1 = z
            z = z1 - p1 / pp
            if abs(z - z1) < 1e-15:
                break
        x[i] = -z
        w[i] = 2.0 / ((1.0 - z * z) * pp * pp)
    return x, w


@njit(cache=True, nogil=True)
def _phi_rule(theta_min, kphi, w_panel, n_gauss, gx, gw, n_uniform, out_phi, out_w, start):
    """Azimuth rule on (-pi, pi]; sinh-graded about phi = 0 when the ring
    passes within 0.5 rad of the module direction, uniform otherwise.
    Weights include ``1/(2 pi)``.  Returns the new count."""
    count = start
    capacity = out_phi.shape[0]
    if kphi < 1e-9 or theta_min > 0.5:
        for q in range(n_uniform):
            if count >= capacity:
                return count
            out_phi[count] = -math.pi + TWO_PI * (q + 0.5) / n_uniform
            out_w[count] = 1.0 / n_uniform
            count += 1
        return count
    th = max(theta_min, 1e-12)
    w_max = math.asinh(math.pi * kphi / th)
    n_panels = max(2, int(math.ceil(2.0 * w_max / w_panel)))
    scale = th / kphi
    for panel in range(n_panels):
        w_lo = -w_max + 2.0 * w_max * panel / n_panels
        w_hi = -w_max + 2.0 * w_max * (panel + 1) / n_panels
        for node in range(n_gauss):
            if count >= capacity:
                return count
            w = w_lo + 0.5 * (w_hi - w_lo) * (gx[node] + 1.0)
            out_phi[count] = scale * math.sinh(w)
            out_w[count] = scale * math.cosh(w) * 0.5 * (w_hi - w_lo) * gw[node] / TWO_PI
            count += 1
    return count


@njit(cache=True, nogil=True)
def _segment_a_breaks(b, z_r, length, cone_cos, cone_sin, dtda, inv_v_lo, inv_v_hi,
                      w_front, grade_ratio, grade_min, max_rel, out):
    """Panel boundaries along a segment.

    Geometric grading towards the Cherenkov root (log singularity of the
    azimuth-integrated integrand), then subdivision so that every panel spans
    at most ``w_front`` ns of front time ``t_f(a) = t0 + a dtda + r(a)/v``
    for every group speed, and at most ``max_rel * r(a)`` metres.
    """
    capacity = out.shape[0]
    root = z_r - b * cone_cos / cone_sin
    centre = min(max(root, 0.0), length)
    r_star = math.sqrt(b * b + (z_r - centre) ** 2)
    delta = 0.3 * r_star * r_star / b
    delta_min = max(delta * grade_min, 0.25 * abs(root - centre))
    raw = np.empty(capacity)
    n = 0
    raw[n] = 0.0
    n += 1
    raw[n] = length
    n += 1
    raw[n] = centre
    n += 1
    step = delta
    while step > delta_min and n < capacity - 2:
        raw[n] = centre - step
        n += 1
        raw[n] = centre + step
        n += 1
        step *= grade_ratio
    # clip and sort
    for i in range(n):
        raw[i] = min(max(raw[i], 0.0), length)
    raw[:n] = np.sort(raw[:n])
    count = 0
    for i in range(n):
        if count == 0 or raw[i] > out[count - 1] + 1e-12 * max(length, 1.0):
            if count >= capacity:
                break
            out[count] = raw[i]
            count += 1
    # subdivide for front spread and geometry
    final = np.empty(capacity)
    m = 0
    for i in range(count - 1):
        lo = out[i]
        hi = out[i + 1]
        slope = 0.0
        r_min = 1e300
        for k in range(3):
            a = lo + 0.5 * k * (hi - lo)
            dz = z_r - a
            r = math.sqrt(b * b + dz * dz)
            cos_g = dz / r
            slope = max(slope, abs(dtda - cos_g * inv_v_lo), abs(dtda - cos_g * inv_v_hi))
            r_min = min(r_min, r)
        pieces = max(1, int(math.ceil((hi - lo) * slope / w_front)),
                     int(math.ceil((hi - lo) / (max_rel * r_min))))
        for k in range(pieces):
            if m >= capacity - 1:
                break
            final[m] = lo + (hi - lo) * k / pieces
            m += 1
    final[m] = out[count - 1]
    m += 1
    out[:m] = final[:m]
    return m



def _scratch_note():
    """Every driver allocates, per thread block: accumulators ``pre (L, 3)``,
    ``acc (L, nb, 4)``, ``charge (L)`` and path buffers of ``max_nodes``."""


@njit(cache=True, nogil=True)
def _segment_order1_block(block, receivers, looks, areas, origins, work, seg_start, seg_dir, seg_length, seg_q0, seg_q2, seg_cone, seg_t0, seg_dtda, apoly, g, y_g, h, h_tail, tail_exp, mus, mut, speed, s0w, s2w, edges, eedge, itab, direct_bins, a_gauss, w_front, grade_ratio, grade_min, max_rel, phi_w_panel, phi_gauss, phi_uniform, max_breaks, max_nodes, min_impact, n_blocks, n_recv, n_seg, n_lambda, nb, charge_out, bins_out, nodes_out, batch_out, close_out, mut_min, speed_max, inv_v_lo, inv_v_hi, ax, aw, gx, gw, n_work, max_phi):
    """One thread block of :func:`segment_order1` (no hoisting)."""
    pre = np.zeros((n_lambda, 3))
    acc = np.zeros((n_lambda, nb, 4))
    charge = np.zeros(n_lambda)
    s_buf = np.empty(max_nodes)
    t_buf = np.empty(max_nodes)
    h_buf = np.empty(max_nodes)
    g0 = np.empty(max_nodes)
    g2 = np.empty(max_nodes)
    breaks = np.empty(max_breaks)
    phis = np.empty(max_phi)
    phiw = np.empty(max_phi)
    for index in range(block, n_work, n_blocks):
        d = work[index]
        pre[:, :] = 0.0
        acc[:, :, :] = 0.0
        charge[:] = 0.0
        rx = receivers[d, 0]
        ry = receivers[d, 1]
        rz = receivers[d, 2]
        nx = looks[d, 0]
        ny = looks[d, 1]
        nz = looks[d, 2]
        used = 0
        batches = 0
        for i in range(n_seg):
            ux = seg_dir[i, 0]
            uy = seg_dir[i, 1]
            uz = seg_dir[i, 2]
            ox = rx - seg_start[i, 0]
            oy = ry - seg_start[i, 1]
            oz = rz - seg_start[i, 2]
            z_r = ox * ux + oy * uy + oz * uz
            bx = ox - z_r * ux
            by = oy - z_r * uy
            bz = oz - z_r * uz
            b = math.sqrt(bx * bx + by * by + bz * bz)
            closest = min(max(z_r, 0.0), seg_length[i])
            cxm = ox - closest * ux
            cym = oy - closest * uy
            czm = oz - closest * uz
            if math.sqrt(cxm * cxm + cym * cym + czm * czm) < min_impact or b < 1e-9:
                close_out[d] += 1
                continue
            e1x = bx / b
            e1y = by / b
            e1z = bz / b
            e2x = uy * e1z - uz * e1y
            e2y = uz * e1x - ux * e1z
            e2z = ux * e1y - uy * e1x
            cone_cos = seg_cone[i]
            cone_sin = math.sqrt(1.0 - cone_cos * cone_cos)
            theta_c = math.acos(cone_cos)
            n_breaks = _segment_a_breaks(b, z_r, seg_length[i], cone_cos, cone_sin,
                                         seg_dtda[i], inv_v_lo, inv_v_hi, w_front,
                                         grade_ratio, grade_min, max_rel, breaks)
            for panel in range(n_breaks - 1):
                a_lo = breaks[panel]
                a_hi = breaks[panel + 1]
                for ia in range(a_gauss):
                    a = a_lo + 0.5 * (a_hi - a_lo) * (ax[ia] + 1.0)
                    wa = 0.5 * (a_hi - a_lo) * aw[ia]
                    dz = z_r - a
                    r_a = math.sqrt(b * b + dz * dz)
                    gamma_angle = math.atan2(b, dz)
                    theta_min = abs(gamma_angle - theta_c)
                    kphi = math.sqrt(max(math.sin(gamma_angle) * cone_sin, 0.0))
                    n_phi = _phi_rule(theta_min, kphi, phi_w_panel, phi_gauss, gx, gw,
                                      phi_uniform, phis, phiw, 0)
                    dx = ox - a * ux
                    dy = oy - a * uy
                    dz3 = oz - a * uz
                    t_emit = seg_t0[i] + a * seg_dtda[i] - origins[d]
                    sin_min = math.sin(max(theta_min, 1e-9))
                    sigma = _batch_sigma(r_a, sin_min, math.tan(0.5 * max(theta_min, 1e-9)), y_g)
                    n = _batch_grid(r_a, sigma, t_emit, h, h_tail, tail_exp, mut_min,
                                    speed_max, edges[nb], s_buf)
                    g0[:n] = 0.0
                    g2[:n] = 0.0
                    for q in range(n_phi):
                        phi = phis[q]
                        cph = math.cos(phi)
                        sph = math.sin(phi)
                        sx = cone_cos * ux + cone_sin * (cph * e1x + sph * e2x)
                        sy = cone_cos * uy + cone_sin * (cph * e1y + sph * e2y)
                        sz = cone_cos * uz + cone_sin * (cph * e1z + sph * e2z)
                        r, theta, sine, cosine, c1, c3 = _pencil_frame(
                            dx, dy, dz3, sx, sy, sz, nx, ny, nz)
                        if sine <= 1e-300:
                            continue
                        weight = wa * phiw[q]
                        _batch_add_pencil(n, s_buf, r, sine, cosine, 1.0 / (r * sine),
                                          sine, weight * seg_q0[i],
                                          weight * seg_q2[i], c1, c3, apoly, g,
                                          g0, g2)
                        used += 1
                    _batch_deposit(n, s_buf, g0, g2, r_a, t_emit, tail_exp,
                                   mus, mut, speed, s0w, s2w, edges, eedge, pre, acc,
                                   charge, direct_bins, t_buf, h_buf,
                                   0.0, wa, seg_dtda[i], dz / r_a)
                    batches += 1
        charge_out[d] = _finalize(edges, eedge, itab, pre, acc, charge,
                                  areas[d], bins_out[d])
        nodes_out[d] = used
        batch_out[d] = batches


@njit(cache=True, parallel=True)
def segment_order1(receivers, looks, areas, origins, work,
                   seg_start, seg_dir, seg_length, seg_q0, seg_q2, seg_cone,
                   seg_t0, seg_dtda, apoly,
                   g, y_g, h, h_tail, tail_exp,
                   mus, mut, speed, s0w, s2w,
                   edges, eedge, itab, direct_bins,
                   a_gauss, w_front, grade_ratio, grade_min, max_rel,
                   phi_w_panel, phi_gauss, phi_uniform,
                   max_breaks, max_nodes, min_impact, n_blocks):
    """First-order contribution of straight Cherenkov segments at the listed modules.

    Tensor rule per segment and module: Gauss--Legendre panels in ``a``
    (graded towards the Cherenkov root, front spread per panel at most
    ``w_front`` ns) and, at every ``a``, the azimuth rule of
    :func:`_phi_rule` about the ring's closest approach to the module.  All
    azimuth nodes of one ``a`` form one batch.  ``seg_q0/seg_q2`` are photons
    per metre of the two spectral fields; emission time is
    ``seg_t0 + a * seg_dtda``.
    Returns ``(charge, bins, pencils_used, batches_used, too_close)``.
    """
    n_recv = receivers.shape[0]
    n_seg = seg_start.shape[0]
    n_lambda = mus.shape[0]
    nb = edges.shape[0] - 1
    charge_out = np.zeros(n_recv)
    bins_out = np.zeros((n_recv, nb))
    nodes_out = np.zeros(n_recv, dtype=np.int64)
    batch_out = np.zeros(n_recv, dtype=np.int64)
    close_out = np.zeros(n_recv, dtype=np.int64)
    mut_min = mut.min()
    speed_max = speed.max()
    inv_v_lo = 1.0 / speed.max()
    inv_v_hi = 1.0 / speed.min()
    ax, aw = _gauss_legendre(a_gauss)
    gx, gw = _gauss_legendre(phi_gauss)
    n_work = work.shape[0]
    max_phi = 8 * phi_uniform + 4096
    for block in prange(n_blocks):
        _segment_order1_block(block, receivers, looks, areas, origins, work, seg_start, seg_dir, seg_length, seg_q0, seg_q2, seg_cone, seg_t0, seg_dtda, apoly, g, y_g, h, h_tail, tail_exp, mus, mut, speed, s0w, s2w, edges, eedge, itab, direct_bins, a_gauss, w_front, grade_ratio, grade_min, max_rel, phi_w_panel, phi_gauss, phi_uniform, max_breaks, max_nodes, min_impact, n_blocks, n_recv, n_seg, n_lambda, nb, charge_out, bins_out, nodes_out, batch_out, close_out, mut_min, speed_max, inv_v_lo, inv_v_hi, ax, aw, gx, gw, n_work, max_phi)
    return charge_out, bins_out, nodes_out, batch_out, close_out


@njit(cache=True, parallel=True)
def pencils_order1(receivers, looks, areas, origins, pencil_pos, pencil_dir,
                   pencil_w0, pencil_w2, pencil_t, apoly,
                   g, y_g, h, h_tail, tail_exp,
                   mus, mut, speed, s0w, s2w, edges, eedge, itab, direct_bins,
                   max_nodes):
    """First order of explicit delta-direction pencils, one batch each."""
    n_recv = receivers.shape[0]
    n_pen = pencil_pos.shape[0]
    n_lambda = mus.shape[0]
    nb = edges.shape[0] - 1
    charge_out = np.zeros(n_recv)
    bins_out = np.zeros((n_recv, nb))
    mut_min = mut.min()
    speed_max = speed.max()
    for d in prange(n_recv):
        pre = np.zeros((n_lambda, 3))
        acc = np.zeros((n_lambda, nb, 4))
        charge = np.zeros(n_lambda)
        s_buf = np.empty(max_nodes)
        t_buf = np.empty(max_nodes)
        h_buf = np.empty(max_nodes)
        g0 = np.empty(max_nodes)
        g2 = np.empty(max_nodes)
        for p in range(n_pen):
            r, theta, sine, cosine, c1, c3 = _pencil_frame(
                receivers[d, 0] - pencil_pos[p, 0], receivers[d, 1] - pencil_pos[p, 1],
                receivers[d, 2] - pencil_pos[p, 2],
                pencil_dir[p, 0], pencil_dir[p, 1], pencil_dir[p, 2],
                looks[d, 0], looks[d, 1], looks[d, 2])
            if sine <= 1e-300:
                continue
            t_emit = pencil_t[p] - origins[d]
            sigma = _batch_sigma(r, sine, sine / (1.0 + cosine), y_g)
            n = _batch_grid(r, sigma, t_emit, h, h_tail, tail_exp, mut_min, speed_max,
                            edges[nb], s_buf)
            g0[:n] = 0.0
            g2[:n] = 0.0
            _batch_add_pencil(n, s_buf, r, sine, cosine, 1.0 / (r * sine), sine,
                              pencil_w0[p], pencil_w2[p], c1, c3, apoly, g, g0, g2)
            _batch_deposit(n, s_buf, g0, g2, r, t_emit, tail_exp, mus, mut, speed,
                           s0w, s2w, edges, eedge, pre, acc, charge, direct_bins,
                           t_buf, h_buf, 0.0, 0.0, 0.0, 0.0)
        charge_out[d] = _finalize(edges, eedge, itab, pre, acc, charge,
                                  areas[d], bins_out[d])
    return charge_out, bins_out


# ----------------------------------------------------- element compression

@njit(cache=True, nogil=True, inline="always")
def _pixel(sx, sy, sz, n_face):
    """Equi-angular cube-sphere pixel index of a unit vector."""
    ax = abs(sx)
    ay = abs(sy)
    az = abs(sz)
    if ax >= ay and ax >= az:
        face = 0 if sx > 0 else 1
        u = sy / ax
        v = sz / ax
    elif ay >= az:
        face = 2 if sy > 0 else 3
        u = sx / ay
        v = sz / ay
    else:
        face = 4 if sz > 0 else 5
        u = sx / az
        v = sy / az
    quarter = 0.25 * math.pi
    i = int((math.atan(u) + quarter) / (2.0 * quarter) * n_face)
    j = int((math.atan(v) + quarter) / (2.0 * quarter) * n_face)
    if i >= n_face:
        i = n_face - 1
    if j >= n_face:
        j = n_face - 1
    if i < 0:
        i = 0
    if j < 0:
        j = 0
    return (face * n_face + i) * n_face + j


@njit(cache=True, nogil=True)
def _hash_slot(keys, key):
    mask = keys.shape[0] - 1
    mixed = key ^ (key >> 31)
    mixed = (mixed & 0x7FFFFFFF) * 0x5BD1E995 + ((mixed >> 31) & 0x7FFFFFFF) * 0x1B873593
    slot = (mixed ^ (mixed >> 17)) & mask
    while True:
        current = keys[slot]
        if current == key or current == -1:
            return slot
        slot = (slot + 1) & mask


@njit(cache=True)
def _grow(keys, values, used):
    capacity = keys.shape[0] * 2
    new_keys = np.full(capacity, -1, dtype=np.int64)
    new_values = np.zeros((capacity, values.shape[1]))
    for slot in range(keys.shape[0]):
        if keys[slot] != -1:
            target = _hash_slot(new_keys, keys[slot])
            new_keys[target] = keys[slot]
            new_values[target] = values[slot]
    return new_keys, new_values


N_PENCIL_MOMENTS = 23  # w0 w2 wx wy wz wt wsx wsy wsz wxx wtt sxx syy szz sxy sxz syz xx yy zz xy xz yz


@njit(cache=True, nogil=True)
def compress_elements(start, direction, length, c0, c2, cone_cos, start_ns,
                      end_ns, cell_m, n_face, ring_nodes, a_step, origin_m, dims,
                      first, stop):
    """Deposit every element's Cherenkov ring into (cell, direction pixel) bins.

    Each element is split into ``ceil(length / a_step)`` sub-segments; each
    sub-segment's cone is sampled with ``ring_nodes`` azimuths (a per-element
    golden-ratio phase offset avoids aliasing between elements).  Every node
    carries ``c / (n_a ring_nodes)`` of both spectral fields and is added to
    the bin of its position cell and direction pixel.  Returns the moment
    table (``N_PENCIL_MOMENTS`` columns) of the occupied bins and their keys,
    for the elements ``first <= i < stop`` (chunks are merged by the caller).
    """
    capacity = 1 << 16
    keys = np.full(capacity, -1, dtype=np.int64)
    values = np.zeros((capacity, N_PENCIL_MOMENTS))
    used = 0
    golden = 0.6180339887498949
    n_pix = 6 * n_face * n_face
    step_c = math.cos(TWO_PI / ring_nodes)
    step_s = math.sin(TWO_PI / ring_nodes)
    for i in range(first, stop):
        ux = direction[i, 0]
        uy = direction[i, 1]
        uz = direction[i, 2]
        if abs(uz) < 0.9:
            hx, hy, hz = 0.0, 0.0, 1.0
        else:
            hx, hy, hz = 1.0, 0.0, 0.0
        e1x = uy * hz - uz * hy
        e1y = uz * hx - ux * hz
        e1z = ux * hy - uy * hx
        norm = math.sqrt(e1x * e1x + e1y * e1y + e1z * e1z)
        e1x /= norm
        e1y /= norm
        e1z /= norm
        e2x = uy * e1z - uz * e1y
        e2y = uz * e1x - ux * e1z
        e2z = ux * e1y - uy * e1x
        cc = cone_cos[i]
        cs = math.sqrt(max(1.0 - cc * cc, 0.0))
        n_a = max(1, int(math.ceil(length[i] / a_step)))
        share = 1.0 / (n_a * ring_nodes)
        arc2 = (TWO_PI * cs / ring_nodes) ** 2 / 12.0
        sub2 = (length[i] / n_a) ** 2 / 12.0
        w0 = c0[i] * share
        w2 = c2[i] * share
        offset = ((i * golden) % 1.0) * TWO_PI / ring_nodes
        for ia in range(n_a):
            frac = (ia + 0.5) / n_a
            px = start[i, 0] + frac * length[i] * ux
            py = start[i, 1] + frac * length[i] * uy
            pz = start[i, 2] + frac * length[i] * uz
            t = start_ns[i] + frac * (end_ns[i] - start_ns[i])
            cx = min(max(int(math.floor((px - origin_m[0]) / cell_m)), 0), dims[0] - 1)
            cy = min(max(int(math.floor((py - origin_m[1]) / cell_m)), 0), dims[1] - 1)
            cz = min(max(int(math.floor((pz - origin_m[2]) / cell_m)), 0), dims[2] - 1)
            cell_key = (cx * dims[1] + cy) * dims[2] + cz
            cp = math.cos(offset)
            sp = math.sin(offset)
            for q in range(ring_nodes):
                if q > 0:
                    cp, sp = cp * step_c - sp * step_s, sp * step_c + cp * step_s
                sx = cc * ux + cs * (cp * e1x + sp * e2x)
                sy = cc * uy + cs * (cp * e1y + sp * e2y)
                sz = cc * uz + cs * (cp * e1z + sp * e2z)
                key = cell_key * n_pix + _pixel(sx, sy, sz, n_face)
                if 2 * (used + 1) > keys.shape[0]:
                    keys, values = _grow(keys, values, used)
                slot = _hash_slot(keys, key)
                if keys[slot] == -1:
                    keys[slot] = key
                    used += 1
                row = values[slot]
                row[0] += w0
                row[1] += w2
                row[2] += w0 * px
                row[3] += w0 * py
                row[4] += w0 * pz
                row[5] += w0 * t
                row[6] += w0 * sx
                row[7] += w0 * sy
                row[8] += w0 * sz
                row[9] += w0 * (px * px + py * py + pz * pz + sub2)
                row[10] += w0 * t * t
                # direction second moments, with the exact variance of the
                # arc this node stands for along the ring tangent
                tx = -sp * e1x + cp * e2x
                ty = -sp * e1y + cp * e2y
                tz = -sp * e1z + cp * e2z
                row[11] += w0 * (sx * sx + arc2 * tx * tx)
                row[12] += w0 * (sy * sy + arc2 * ty * ty)
                row[13] += w0 * (sz * sz + arc2 * tz * tz)
                row[14] += w0 * (sx * sy + arc2 * tx * ty)
                row[15] += w0 * (sx * sz + arc2 * tx * tz)
                row[16] += w0 * (sy * sz + arc2 * ty * tz)
                # position second moments with the sub-segment's own spread
                row[17] += w0 * (px * px + sub2 * ux * ux)
                row[18] += w0 * (py * py + sub2 * uy * uy)
                row[19] += w0 * (pz * pz + sub2 * uz * uz)
                row[20] += w0 * (px * py + sub2 * ux * uy)
                row[21] += w0 * (px * pz + sub2 * ux * uz)
                row[22] += w0 * (py * pz + sub2 * uy * uz)
    out = np.empty((used, N_PENCIL_MOMENTS))
    out_keys = np.empty(used, dtype=np.int64)
    k = 0
    for slot in range(keys.shape[0]):
        if keys[slot] != -1:
            out[k] = values[slot]
            out_keys[k] = keys[slot]
            k += 1
    return out_keys, out



# ------------------------------------------------ grouped pencils per module

@njit(cache=True, nogil=True, inline="always")
def _fill_moments_averaged(weight, c1, c3, averaged, ipow, jpow, max_power,
                           out, cp, sp):
    """Moments ``w c1^i c3^j``; with ``averaged`` the module direction lies
    inside the pencil's disk and ``c3^j`` is replaced by its azimuthal mean
    ``|n_perp|^j (j-1)!!/j!!`` (zero for odd ``j``)."""
    cp[0] = 1.0
    sp[0] = 1.0
    perp2 = max(1.0 - c1 * c1, 0.0)
    for p in range(1, max_power + 1):
        cp[p] = cp[p - 1] * c1
        if averaged:
            if p % 2 == 1:
                sp[p] = 0.0
            else:
                # (p-1)!!/p!! * perp^p, built from the previous even power
                sp[p] = (sp[p - 2] if p >= 2 else 1.0) * perp2 * (p - 1.0) / p
        else:
            sp[p] = sp[p - 1] * c3
    for m in range(out.shape[0]):
        out[m] = weight * cp[ipow[m]] * sp[jpow[m]]



@njit(cache=True, nogil=True)
def _pencil_shape_factor(p, r, theta, sine, rx, ry, rz, pencil_pos, pencil_dir,
                         pencil_major, pencil_var_major, pencil_var_minor,
                         pencil_pos_cov, smoothing_factor, gauss_tau, gauss_w):
    """Mean of ``1/sin(theta)`` over one pencil's apparent angular shape.

    The shape seen from the module is the direction covariance (major and
    minor axes in the tangent plane) plus the position covariance projected
    on that plane over ``r^2``.  Within ``smoothing_factor`` extents the exact
    2D-Gaussian mean of ``1/theta`` is used and the pencil is evaluated at the
    harmonic-mean angle ``1/<1/theta>`` (exact for a regular part linear in
    theta); beyond, the second-order correction
    ``(1/2) C_ij (3 p_i p_j - d^2 delta_ij) / d^5``.
    Returns ``(factor, theta_eff, azimuth_averaged, ok)``.
    """
    ax_ = pencil_major[p, 0]
    ay_ = pencil_major[p, 1]
    az_ = pencil_major[p, 2]
    sx_ = pencil_dir[p, 0]
    sy_ = pencil_dir[p, 1]
    sz_ = pencil_dir[p, 2]
    bx_ = sy_ * az_ - sz_ * ay_
    by_ = sz_ * ax_ - sx_ * az_
    bz_ = sx_ * ay_ - sy_ * ax_
    cov = pencil_pos_cov[p]
    inv_r2 = 1.0 / (r * r)
    c_aa = (ax_ * (cov[0] * ax_ + cov[3] * ay_ + cov[4] * az_)
            + ay_ * (cov[3] * ax_ + cov[1] * ay_ + cov[5] * az_)
            + az_ * (cov[4] * ax_ + cov[5] * ay_ + cov[2] * az_)) * inv_r2
    c_bb = (bx_ * (cov[0] * bx_ + cov[3] * by_ + cov[4] * bz_)
            + by_ * (cov[3] * bx_ + cov[1] * by_ + cov[5] * bz_)
            + bz_ * (cov[4] * bx_ + cov[5] * by_ + cov[2] * bz_)) * inv_r2
    c_ab = (ax_ * (cov[0] * bx_ + cov[3] * by_ + cov[4] * bz_)
            + ay_ * (cov[3] * bx_ + cov[1] * by_ + cov[5] * bz_)
            + az_ * (cov[4] * bx_ + cov[5] * by_ + cov[2] * bz_)) * inv_r2
    m11 = pencil_var_major[p] + c_aa
    m22 = pencil_var_minor[p] + c_bb
    m12 = c_ab
    half_tr = 0.5 * (m11 + m22)
    disc = math.sqrt(max(0.25 * (m11 - m22) ** 2 + m12 * m12, 0.0))
    mu1 = half_tr + disc
    mu2 = max(half_tr - disc, 1e-16 * mu1 + 1e-300)
    extent = math.sqrt(max(mu1, 0.0))
    hx_ = (rx - pencil_pos[p, 0]) / r
    hy_ = (ry - pencil_pos[p, 1]) / r
    hz_ = (rz - pencil_pos[p, 2]) / r
    u_ = hx_ * ax_ + hy_ * ay_ + hz_ * az_
    v_ = hx_ * bx_ + hy_ * by_ + hz_ * bz_
    if extent > 0.0 and theta < smoothing_factor * extent:
        angle = 0.5 * math.atan2(2.0 * m12, m11 - m22)
        ca = math.cos(angle)
        sa = math.sin(angle)
        mean = gauss_inverse_mean(ca * u_ + sa * v_, -sa * u_ + ca * v_,
                                  mu1, mu2, gauss_tau, gauss_w)
        factor = mean * (theta / sine if sine > 1e-300 else 1.0)
        theta_eff = math.asin(min(1.0, 1.0 / factor)) if factor > 1.0 else theta
        return factor, theta_eff, theta < extent, True
    if sine <= 1e-300:
        return 0.0, theta, False, False
    d2 = u_ * u_ + v_ * v_
    factor = 1.0 / sine
    if d2 > 0.0:
        quad_form = m11 * u_ * u_ + 2.0 * m12 * u_ * v_ + m22 * v_ * v_
        factor *= 1.0 + 0.5 * (3.0 * quad_form / d2 - (m11 + m22)) / d2
    return factor, theta, False, True


@njit(cache=True, nogil=True)
def _grouped_pencils_order1_block(block, receivers, looks, areas, origins, work, pencil_pos, pencil_dir, pencil_w0, pencil_w2, pencil_t, pencil_major, pencil_var_major, pencil_var_minor, pencil_pos_cov, gauss_tau, gauss_w, coef, ipow, jpow, max_power, g, y_g, h, h_tail, tail_exp, mus, mut, speed, s0w, s2w, edges, eedge, itab, direct_bins, max_nodes, smoothing_factor, theta_lo, theta_ratio, tau_step, log_r_step, speed_ref, n_blocks, n_recv, n_pen, n_lambda, nb, n_mono, charge_out, bins_out, groups_out, batches_out, mut_min, speed_max, n_values, log_theta_ratio, theta_bins, n_work):
    """One thread block of :func:`grouped_pencils_order1` (no hoisting)."""
    pre = np.zeros((n_lambda, 3))
    acc = np.zeros((n_lambda, nb, 4))
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
    values = np.zeros((1 << 12, n_values))
    for index in range(block, n_work, n_blocks):
        d = work[index]
        pre[:, :] = 0.0
        acc[:, :, :] = 0.0
        charge[:] = 0.0
        keys[:] = -1
        values[:, :] = 0.0
        used = 0
        rx = receivers[d, 0]
        ry = receivers[d, 1]
        rz = receivers[d, 2]
        origin = origins[d]
        for p in range(n_pen):
            r, theta, sine, cosine, c1, c3 = _pencil_frame(
                rx - pencil_pos[p, 0], ry - pencil_pos[p, 1], rz - pencil_pos[p, 2],
                pencil_dir[p, 0], pencil_dir[p, 1], pencil_dir[p, 2],
                looks[d, 0], looks[d, 1], looks[d, 2])
            # apparent angular shape of the pencil seen from this module:
            # direction covariance (major/minor in the tangent plane) plus the
            # position covariance projected on that plane over r^2; the mean
            # of 1/theta is the exact 2D-Gaussian mean with that covariance
            cov = pencil_pos_cov[p]
            bound = (pencil_var_major[p] + pencil_var_minor[p]
                     + (cov[0] + cov[1] + cov[2]) / (r * r))
            if sine > 1e-300 and bound < 1e-4 * sine * sine:
                # far from the module direction: the shape correction is
                # below 1e-4 of the point value
                factor = 1.0 / sine
                weight = factor / r
                averaged = False
            else:
                factor, theta, averaged, ok = _pencil_shape_factor(
                    p, r, theta, sine, rx, ry, rz, pencil_pos, pencil_dir,
                    pencil_major, pencil_var_major, pencil_var_minor, pencil_pos_cov,
                    smoothing_factor, gauss_tau, gauss_w)
                if not ok:
                    continue
                weight = factor / r
            if theta < theta_lo:
                i_theta = 0
            else:
                i_theta = 1 + int(math.log(theta / theta_lo) / log_theta_ratio)
            tau = pencil_t[p] - origin + r / speed_ref
            i_tau = min(max(int(math.floor(tau / tau_step)) + (1 << 20), 0),
                        (1 << 21) - 1)
            i_r = min(max(int(math.floor(math.log(r) / log_r_step)) + (1 << 10), 0),
                      (1 << 11) - 1)
            key = ((i_tau * (1 << 11) + i_r) * theta_bins) + i_theta
            if 2 * (used + 1) > keys.shape[0]:
                keys, values = _grow(keys, values, used)
            slot = _hash_slot(keys, key)
            if keys[slot] == -1:
                keys[slot] = key
                used += 1
            _fill_moments_averaged(weight * pencil_w0[p], c1, c3, averaged,
                                   ipow, jpow, max_power, mom0, cpow, spow)
            _fill_moments_averaged(weight * pencil_w2[p], c1, c3, averaged,
                                   ipow, jpow, max_power, mom2, cpow, spow)
            row = values[slot]
            for m in range(n_mono):
                row[m] += mom0[m]
                row[n_mono + m] += mom2[m]
            omega = abs(mom0[0])
            row[2 * n_mono] += omega
            row[2 * n_mono + 1] += omega * theta
            row[2 * n_mono + 2] += omega * r
            row[2 * n_mono + 3] += omega * (pencil_t[p] - origin)
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
                t_b = t_sum / omega_sum
                tau_mean = tau_sum / omega_sum
                tau_var = max(tau2_sum / omega_sum - tau_mean * tau_mean, 0.0)
                theta_small = max(theta_small, 1e-9)
                sigma = _batch_sigma(r_b, math.sin(theta_small),
                                     math.tan(0.5 * theta_small), y_g)
                n = _batch_grid(r_b, sigma, t_b, h, h_tail, tail_exp, mut_min,
                                speed_max, edges[nb], s_buf)
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
                                       g0, g2, cpow, spow)
                _batch_deposit(n, s_buf, g0, g2, r_b, t_b, tail_exp, mus, mut,
                               speed, s0w, s2w, edges, eedge, pre, acc, charge,
                               direct_bins, t_buf, h_buf,
                               math.sqrt(12.0 * tau_var), 0.0, 0.0, 0.0)
                batches += 1
            start = stop
        charge_out[d] = _finalize(edges, eedge, itab, pre, acc, charge,
                                  areas[d], bins_out[d])
        groups_out[d] = used
        batches_out[d] = batches


@njit(cache=True, parallel=True)
def grouped_pencils_order1(receivers, looks, areas, origins, work,
                           pencil_pos, pencil_dir, pencil_w0, pencil_w2,
                           pencil_t, pencil_major, pencil_var_major,
                           pencil_var_minor, pencil_pos_cov, gauss_tau, gauss_w,
                           coef, ipow, jpow, max_power,
                           g, y_g, h, h_tail, tail_exp,
                           mus, mut, speed, s0w, s2w, edges, eedge, itab,
                           direct_bins, max_nodes, smoothing_factor,
                           theta_lo, theta_ratio, tau_step, log_r_step, speed_ref,
                           n_blocks):
    """Per-module merge of pencils into (tau, r, theta) groups; one batch per
    (tau, r) with one profile term per theta group.

    For module ``d`` every pencil ``p`` has distance ``r_p``, angle
    ``theta_p`` to the module direction, arrival proxy
    ``tau_p = t_p + r_p / speed_ref`` and acceptance cosines ``c1, c3``.  The
    singular factor ``1/sin(theta)`` is smoothed using the 2D Gaussian mean
    of ``1/theta`` from its direction and position covariances near the cone;
    near the cone the ``c3`` moments are azimuth averaged. Groups carry exact acceptance
    moments of both fields and singular-weighted means of theta, r, t and
    tau.  ``work`` lists the modules to evaluate.
    Returns ``(charge, bins, groups, batches)``.
    """
    n_recv = receivers.shape[0]
    n_pen = pencil_pos.shape[0]
    n_lambda = mus.shape[0]
    nb = edges.shape[0] - 1
    n_mono = coef.shape[0]
    charge_out = np.zeros(n_recv)
    bins_out = np.zeros((n_recv, nb))
    groups_out = np.zeros(n_recv, dtype=np.int64)
    batches_out = np.zeros(n_recv, dtype=np.int64)
    mut_min = mut.min()
    speed_max = speed.max()
    n_values = 2 * n_mono + 6
    log_theta_ratio = math.log(1.0 + theta_ratio)
    theta_bins = 2 + int(math.log(math.pi / theta_lo) / log_theta_ratio)
    n_work = work.shape[0]
    for block in prange(n_blocks):
        _grouped_pencils_order1_block(block, receivers, looks, areas, origins, work, pencil_pos, pencil_dir, pencil_w0, pencil_w2, pencil_t, pencil_major, pencil_var_major, pencil_var_minor, pencil_pos_cov, gauss_tau, gauss_w, coef, ipow, jpow, max_power, g, y_g, h, h_tail, tail_exp, mus, mut, speed, s0w, s2w, edges, eedge, itab, direct_bins, max_nodes, smoothing_factor, theta_lo, theta_ratio, tau_step, log_r_step, speed_ref, n_blocks, n_recv, n_pen, n_lambda, nb, n_mono, charge_out, bins_out, groups_out, batches_out, mut_min, speed_max, n_values, log_theta_ratio, theta_bins, n_work)
    return charge_out, bins_out, groups_out, batches_out


# ------------------------------------------ uncompressed point-cone reference

@njit(cache=True, nogil=True)
def _rings_order1_block(block, receivers, looks, areas, origins, work, start, direction, length, c0, c2, cone_cos, start_ns, end_ns, a_step, apoly, g, y_g, h, h_tail, tail_exp, mus, mut, speed, s0w, s2w, edges, eedge, itab, direct_bins, max_nodes, phi_w_panel, phi_gauss, phi_uniform, n_blocks, n_recv, n_el, n_lambda, nb, charge_out, bins_out, nodes_out, mut_min, speed_max, gx, gw, max_phi, n_work):
    """One thread block of :func:`rings_order1` (no hoisting)."""
    pre = np.zeros((n_lambda, 3))
    acc = np.zeros((n_lambda, nb, 4))
    charge = np.zeros(n_lambda)
    s_buf = np.empty(max_nodes)
    t_buf = np.empty(max_nodes)
    h_buf = np.empty(max_nodes)
    g0 = np.empty(max_nodes)
    g2 = np.empty(max_nodes)
    phis = np.empty(max_phi)
    phiw = np.empty(max_phi)
    for index in range(block, n_work, n_blocks):
        d = work[index]
        pre[:, :] = 0.0
        acc[:, :, :] = 0.0
        charge[:] = 0.0
        nodes = 0
        rx = receivers[d, 0]
        ry = receivers[d, 1]
        rz = receivers[d, 2]
        for i in range(n_el):
            ux = direction[i, 0]
            uy = direction[i, 1]
            uz = direction[i, 2]
            cc = cone_cos[i]
            cs = math.sqrt(max(1.0 - cc * cc, 0.0))
            theta_c = math.acos(min(max(cc, -1.0), 1.0))
            n_a = max(1, int(math.ceil(length[i] / a_step)))
            dtda = (end_ns[i] - start_ns[i]) / length[i] if length[i] > 0 else 0.0
            for ia in range(n_a):
                frac = (ia + 0.5) / n_a
                px = start[i, 0] + frac * length[i] * ux
                py = start[i, 1] + frac * length[i] * uy
                pz = start[i, 2] + frac * length[i] * uz
                t_emit = start_ns[i] + frac * (end_ns[i] - start_ns[i]) - origins[d]
                ox = rx - px
                oy = ry - py
                oz = rz - pz
                dist = math.sqrt(ox * ox + oy * oy + oz * oz)
                along = (ox * ux + oy * uy + oz * uz) / dist
                bx = ox / dist - along * ux
                by = oy / dist - along * uy
                bz = oz / dist - along * uz
                sin_gamma = math.sqrt(bx * bx + by * by + bz * bz)
                if sin_gamma > 1e-12:
                    e1x = bx / sin_gamma
                    e1y = by / sin_gamma
                    e1z = bz / sin_gamma
                else:
                    if abs(uz) < 0.9:
                        e1x, e1y, e1z = uy, -ux, 0.0
                    else:
                        e1x, e1y, e1z = 0.0, uz, -uy
                    norm = math.sqrt(e1x * e1x + e1y * e1y + e1z * e1z)
                    e1x /= norm
                    e1y /= norm
                    e1z /= norm
                e2x = uy * e1z - uz * e1y
                e2y = uz * e1x - ux * e1z
                e2z = ux * e1y - uy * e1x
                gamma_angle = math.atan2(sin_gamma, along)
                theta_min = abs(gamma_angle - theta_c)
                kphi = math.sqrt(max(sin_gamma * cs, 0.0))
                n_phi = _phi_rule(theta_min, kphi, phi_w_panel, phi_gauss, gx, gw,
                                  phi_uniform, phis, phiw, 0)
                sigma = _batch_sigma(dist, math.sin(max(theta_min, 1e-9)),
                                     math.tan(0.5 * max(theta_min, 1e-9)), y_g)
                n = _batch_grid(dist, sigma, t_emit, h, h_tail, tail_exp, mut_min,
                                speed_max, edges[nb], s_buf)
                g0[:n] = 0.0
                g2[:n] = 0.0
                share = 1.0 / n_a
                for q in range(n_phi):
                    phi = phis[q]
                    cp = math.cos(phi)
                    sp = math.sin(phi)
                    sx = cc * ux + cs * (cp * e1x + sp * e2x)
                    sy = cc * uy + cs * (cp * e1y + sp * e2y)
                    sz = cc * uz + cs * (cp * e1z + sp * e2z)
                    r, theta, sine, cosine, c1, c3 = _pencil_frame(
                        ox, oy, oz, sx, sy, sz, looks[d, 0], looks[d, 1], looks[d, 2])
                    if sine <= 1e-300:
                        continue
                    weight = share * phiw[q]
                    _batch_add_pencil(n, s_buf, r, sine, cosine, 1.0 / (r * sine),
                                      sine, weight * c0[i], weight * c2[i], c1, c3,
                                      apoly, g, g0, g2)
                    nodes += 1
                _batch_deposit(n, s_buf, g0, g2, dist, t_emit, tail_exp, mus, mut,
                               speed, s0w, s2w, edges, eedge, pre, acc, charge,
                               direct_bins, t_buf, h_buf,
                               0.0, length[i] / n_a, dtda, along)
        charge_out[d] = _finalize(edges, eedge, itab, pre, acc, charge,
                                  areas[d], bins_out[d])
        nodes_out[d] = nodes


@njit(cache=True, parallel=True)
def rings_order1(receivers, looks, areas, origins, work,
                 start, direction, length, c0, c2, cone_cos, start_ns, end_ns,
                 a_step, apoly, g, y_g, h, h_tail, tail_exp,
                 mus, mut, speed, s0w, s2w, edges, eedge, itab,
                 direct_bins, max_nodes, phi_w_panel, phi_gauss, phi_uniform,
                 n_blocks):
    """Every element as ``ceil(length/a_step)`` point cones; no grouping.

    Each point cone is one batch whose azimuth nodes follow
    :func:`_phi_rule` about the ring's closest approach to the module; the
    front ramp width is the sub-cone's own front spread.  Intended as the
    element-level reference for the grouped algorithm.
    Returns ``(charge, bins, pencils)``.
    """
    n_recv = receivers.shape[0]
    n_el = start.shape[0]
    n_lambda = mus.shape[0]
    nb = edges.shape[0] - 1
    charge_out = np.zeros(n_recv)
    bins_out = np.zeros((n_recv, nb))
    nodes_out = np.zeros(n_recv, dtype=np.int64)
    mut_min = mut.min()
    speed_max = speed.max()
    gx, gw = _gauss_legendre(phi_gauss)
    max_phi = 8 * phi_uniform + 4096
    n_work = work.shape[0]
    for block in prange(n_blocks):
        _rings_order1_block(block, receivers, looks, areas, origins, work, start, direction, length, c0, c2, cone_cos, start_ns, end_ns, a_step, apoly, g, y_g, h, h_tail, tail_exp, mus, mut, speed, s0w, s2w, edges, eedge, itab, direct_bins, max_nodes, phi_w_panel, phi_gauss, phi_uniform, n_blocks, n_recv, n_el, n_lambda, nb, charge_out, bins_out, nodes_out, mut_min, speed_max, gx, gw, max_phi, n_work)
    return charge_out, bins_out, nodes_out


@njit(cache=True, parallel=True)
def min_distance_to_segments(receivers, start, direction, length):
    """Distance from every module to the nearest point of any segment."""
    n_recv = receivers.shape[0]
    out = np.empty(n_recv)
    for d in prange(n_recv):
        best = 1e300
        for i in range(start.shape[0]):
            ox = receivers[d, 0] - start[i, 0]
            oy = receivers[d, 1] - start[i, 1]
            oz = receivers[d, 2] - start[i, 2]
            along = ox * direction[i, 0] + oy * direction[i, 1] + oz * direction[i, 2]
            if along < 0.0:
                along = 0.0
            elif along > length[i]:
                along = length[i]
            dx = ox - along * direction[i, 0]
            dy = oy - along * direction[i, 1]
            dz = oz - along * direction[i, 2]
            value = dx * dx + dy * dy + dz * dz
            if value < best:
                best = value
        out[d] = math.sqrt(best)
    return out


@njit(cache=True)
def finish_pencils(keys, rows, order):
    """Merge chunk tables (sorted by ``order``) and turn moments into pencils.

    Returns ``(position (P,3), direction (P,3), w0, w2, time, major (P,3),
    var_major, var_minor, pos_cov (P,6))``.  The direction covariance is
    projected on the tangent plane of the mean direction and diagonalised
    there in closed form; ``pos_cov`` is ``xx yy zz xy xz yz``.
    """
    n = keys.shape[0]
    count = 0
    for i in range(n):
        if i == 0 or keys[order[i]] != keys[order[i - 1]]:
            count += 1
    acc = np.zeros((count, rows.shape[1]))
    k = -1
    for i in range(n):
        if i == 0 or keys[order[i]] != keys[order[i - 1]]:
            k += 1
        row = rows[order[i]]
        for c in range(rows.shape[1]):
            acc[k, c] += row[c]
    keep = 0
    for p in range(count):
        if acc[p, 0] > 0.0:
            keep += 1
    position = np.empty((keep, 3))
    direction = np.empty((keep, 3))
    w0 = np.empty(keep)
    w2 = np.empty(keep)
    time = np.empty(keep)
    major = np.empty((keep, 3))
    var_major = np.empty(keep)
    var_minor = np.empty(keep)
    pos_cov = np.empty((keep, 6))
    q = 0
    for p in range(count):
        a = acc[p]
        w = a[0]
        if w <= 0.0:
            continue
        mx = a[2] / w
        my = a[3] / w
        mz = a[4] / w
        position[q, 0] = mx
        position[q, 1] = my
        position[q, 2] = mz
        time[q] = a[5] / w
        w0[q] = w
        w2[q] = a[1]
        rx = a[6] / w
        ry = a[7] / w
        rz = a[8] / w
        norm = math.sqrt(rx * rx + ry * ry + rz * rz)
        dx = rx / norm
        dy = ry / norm
        dz = rz / norm
        direction[q, 0] = dx
        direction[q, 1] = dy
        direction[q, 2] = dz
        # direction covariance about the resultant
        cxx = a[11] / w - rx * rx
        cyy = a[12] / w - ry * ry
        czz = a[13] / w - rz * rz
        cxy = a[14] / w - rx * ry
        cxz = a[15] / w - rx * rz
        cyz = a[16] / w - ry * rz
        # tangent basis (e1, e2) at d
        if abs(dz) < 0.9:
            hx, hy, hz = 0.0, 0.0, 1.0
        else:
            hx, hy, hz = 1.0, 0.0, 0.0
        e1x = dy * hz - dz * hy
        e1y = dz * hx - dx * hz
        e1z = dx * hy - dy * hx
        nn = math.sqrt(e1x * e1x + e1y * e1y + e1z * e1z)
        e1x /= nn
        e1y /= nn
        e1z /= nn
        e2x = dy * e1z - dz * e1y
        e2y = dz * e1x - dx * e1z
        e2z = dx * e1y - dy * e1x
        c11 = (e1x * (cxx * e1x + cxy * e1y + cxz * e1z)
               + e1y * (cxy * e1x + cyy * e1y + cyz * e1z)
               + e1z * (cxz * e1x + cyz * e1y + czz * e1z))
        c22 = (e2x * (cxx * e2x + cxy * e2y + cxz * e2z)
               + e2y * (cxy * e2x + cyy * e2y + cyz * e2z)
               + e2z * (cxz * e2x + cyz * e2y + czz * e2z))
        c12 = (e1x * (cxx * e2x + cxy * e2y + cxz * e2z)
               + e1y * (cxy * e2x + cyy * e2y + cyz * e2z)
               + e1z * (cxz * e2x + cyz * e2y + czz * e2z))
        half = 0.5 * (c11 + c22)
        disc = math.sqrt(max(0.25 * (c11 - c22) ** 2 + c12 * c12, 0.0))
        var_major[q] = max(half + disc, 0.0)
        var_minor[q] = max(half - disc, 0.0)
        angle = 0.5 * math.atan2(2.0 * c12, c11 - c22)
        ca = math.cos(angle)
        sa = math.sin(angle)
        major[q, 0] = ca * e1x + sa * e2x
        major[q, 1] = ca * e1y + sa * e2y
        major[q, 2] = ca * e1z + sa * e2z
        pos_cov[q, 0] = max(a[17] / w - mx * mx, 0.0)
        pos_cov[q, 1] = max(a[18] / w - my * my, 0.0)
        pos_cov[q, 2] = max(a[19] / w - mz * mz, 0.0)
        pos_cov[q, 3] = a[20] / w - mx * my
        pos_cov[q, 4] = a[21] / w - mx * mz
        pos_cov[q, 5] = a[22] / w - my * mz
        q += 1
    return position, direction, w0, w2, time, major, var_major, var_minor, pos_cov
