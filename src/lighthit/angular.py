"""m=0 adjoint Green coefficients, with an exact *free* angular tail.

The medium scattering operator is truncated at L; streaming is not.
The output degree J controls the subsequent spatial angular inversion.
No discretized intensity I(x,s,t) is constructed.
"""
import numpy as np
from scipy.special import eval_legendre, roots_legendre


def _degree(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def _free_moments_and_ratios_numpy(k, d0, degree):
    r"""Return free moments and all successive normalized tail ratios.

    p_l = sqrt((2*l+1)/2) P_l, l=0..N. The tail ratio is computed
    without division of underflowing moments in the Miller branch.
    k: (K,), b: (K,N+1), ratios: (K,N+1); Re(d0)>0.
    """
    N = _degree(degree, "degree")
    k = np.asarray(k, dtype=float)
    d0 = complex(d0)
    if k.ndim != 1 or not len(k) or not np.isfinite(k).all() or np.any(k < 0):
        raise ValueError("k must be a nonempty 1-D array of finite nonnegative wave numbers")
    if not np.isfinite(d0) or d0.real <= 0:
        raise ValueError("d0 must be finite with positive real part")
    b = np.zeros((len(k), N + 1), dtype=complex, order="F")
    ratios_all = np.zeros((len(k), N + 1), dtype=complex, order="F")
    nz = k > 0
    b[~nz, 0] = np.sqrt(2) / d0
    if not np.any(nz):
        return b, ratios_all
    kk = k[nz]
    z = 1j * d0 / kk
    root = np.sqrt(z - 1) * np.sqrt(z + 1)
    decay = 1 / (z + root)
    # The solution decaying with Legendre degree is required.
    decay = np.where(abs(decay) > 1, 1 / decay, decay)
    eta = -np.log(np.abs(decay))
    forward = eta * (N + 3) < 3
    moments = np.zeros((len(kk), N + 2), dtype=complex, order="F")
    moments[:, 0] = 2 * np.arctan(kk / d0) / kk
    ratios_normalized = np.zeros((len(kk), N + 1), dtype=complex)
    if np.any(forward):
        zz = z[forward]
        moments[forward, 1] = zz * moments[forward, 0] + 2 / (1j * kk[forward])
        for ell in range(1, N + 1):
            moments[forward, ell + 1] = (
                (2 * ell + 1) * zz * moments[forward, ell]
                - ell * moments[forward, ell - 1]
            ) / (ell + 1)
        ratios_normalized[forward] = moments[forward, 1:] / moments[forward, :-1]
    if np.any(~forward):
        zz = z[~forward]
        # Each node gets exactly the Miller start depth its own decay rate
        # requires, N+1+max(32, ceil(28/eta_i)); the constants are unchanged.
        # Taking one batch-wide maximum instead would charge every node the
        # margin of the single slowest-decaying node, which sits at the
        # forward/backward threshold eta=3/(N+3) and needs about 9.3*N extra
        # sweeps. Sorting by depth makes the still-inactive nodes a suffix,
        # so one descending loop visits each node over its own depth only.
        depth = N + 1 + np.maximum(32, np.ceil(28 / eta[~forward]).astype(np.int64))
        order = np.argsort(-depth, kind="stable")
        depth_sorted = depth[order]
        top = int(depth_sorted[0])
        zz = zz[order]
        ratio = decay[~forward][order] * (1 - 0.5 / (depth_sorted + 1))
        ratios = np.empty((len(zz), N + 1), dtype=complex)
        # active[ell] = number of leading nodes whose depth is at least ell.
        # Every node is active once ell <= N+1, since the smallest depth is
        # N+33, so the stored block needs no masking.
        active = np.searchsorted(-depth_sorted, -np.arange(top + 2), side="right")
        for ell in range(top, N + 1, -1):
            view = slice(0, active[ell])
            ratio[view] = ell / ((2 * ell + 1) * zz[view] - (ell + 1) * ratio[view])
        for ell in range(N + 1, 0, -1):
            ratio = ell / ((2 * ell + 1) * zz - (ell + 1) * ratio)
            ratios[:, ell - 1] = ratio
        # Results are in depth-sorted order; scatter them back in one pass.
        destination = np.flatnonzero(~forward)[order]
        moments[destination, 1:] = (moments[destination, :1]
                                    * np.cumprod(ratios, axis=1))
        ratios_normalized[destination] = ratios
    b[nz] = moments[:, :N + 1] * np.sqrt((2 * np.arange(N + 1) + 1) / 2)
    j = np.arange(N + 1)
    ratios_all[nz] = np.sqrt((2 * j + 3) / (2 * j + 1)) * ratios_normalized
    return b, ratios_all



def _check_backend(backend):
    if backend not in ("numpy", "numba"):
        raise ValueError("angular_backend must be 'numpy' or 'numba'")
    return backend


def _free_moments_and_ratios(k, d0, degree, *, backend="numpy"):
    """Dispatch the same free-tail calculation to an explicit backend.

    The NumPy implementation remains the default and does not import Numba.
    Selecting Numba requires the optional ``accelerate`` extra; a missing
    dependency raises an error rather than silently changing the backend.
    """
    _check_backend(backend)
    if backend == "numpy":
        return _free_moments_and_ratios_numpy(k, d0, degree)
    N = _degree(degree, "degree")
    k = np.asarray(k, dtype=float)
    d0 = complex(d0)
    if k.ndim != 1 or not len(k) or not np.isfinite(k).all() or np.any(k < 0):
        raise ValueError("k must be a nonempty 1-D array of finite nonnegative wave numbers")
    if not np.isfinite(d0) or d0.real <= 0:
        raise ValueError("d0 must be finite with positive real part")
    try:
        from ._angular_numba import free_moments_and_ratios
    except ModuleNotFoundError as exc:
        if exc.name == "numba":
            raise ImportError(
                "The numba backend requires the optional accelerate extra: "
                "python -m pip install 'lighthit[accelerate]'"
            ) from exc
        raise
    return free_moments_and_ratios(k, d0, N)

def free_moments_and_tail(k, d0, degree, *, backend="numpy"):
    """Free moments (K,N+1) and exact normalized tail ratio (K,)."""
    b, ratios = _free_moments_and_ratios(k, d0, degree, backend=backend)
    return b, ratios[:, -1]


def solve_tail_system(k, d0, tail, gamma, rhs):
    r"""Batched tridiagonal solve A h = rhs, one system per k.

    A_ll=d0-gamma_l; A_(l-1,l)=A_(l,l-1)=i*k*l/sqrt(4*l^2-1).
    The last diagonal also contains i*k*a_(N+1)*tail, the exact Schur
    complement of all free harmonics above N. rhs has no support above N.
    """
    k = np.asarray(k, dtype=float)
    rhs = np.asarray(rhs, dtype=complex)
    gamma = np.asarray(gamma, dtype=float)
    tail = np.asarray(tail, dtype=complex)
    if rhs.ndim != 2 or rhs.shape != (len(k), len(gamma)) or tail.shape != k.shape:
        raise ValueError("Incompatible k, gamma, tail, rhs shapes")
    if not np.isfinite(rhs).all() or not np.isfinite(gamma).all():
        raise ValueError("Nonfinite linear-system data")
    N = rhs.shape[1] - 1
    j = np.arange(1, N + 2)
    a = j / np.sqrt(4 * j * j - 1)
    off = np.asfortranarray(1j * k[:, None] * a[None, :-1])
    diagonal = np.array(np.broadcast_to(d0 - gamma[None, :], rhs.shape), order="F", copy=True)
    diagonal[:, -1] += 1j * k * a[-1] * tail
    # Columns must be contiguous: the Thomas sweep runs over degree,
    # updating the whole k batch. C-order is much slower for large batches.
    r = np.array(rhs, order="F", copy=True)
    for ell in range(1, N + 1):
        multiplier = off[:, ell - 1] / diagonal[:, ell - 1]
        diagonal[:, ell] -= multiplier * off[:, ell - 1]
        r[:, ell] -= multiplier * r[:, ell - 1]
    x = np.empty_like(r, order="F")
    x[:, -1] = r[:, -1] / diagonal[:, -1]
    for ell in range(N - 1, -1, -1):
        x[:, ell] = (r[:, ell] - off[:, ell] * x[:, ell + 1]) / diagonal[:, ell]
    if not np.isfinite(x).all():
        raise FloatingPointError("Nonfinite angular solution")
    return x


def angular_components(k, omega_per_ns, medium, scattering_degree, output_degree,
                       *, backend="numpy"):
    """Return (free, one, >=2) coefficients of the truncated-scattering RTE.

    Each array has shape (K, max(L,J)+1). The >=2 part is obtained
    from its own right-hand side, without subtracting nearly equal fields.
    All quantities use the +i*omega*t transform convention.
    """
    L = _degree(scattering_degree, "scattering_degree")
    J = _degree(output_degree, "output_degree")
    if not np.isfinite(omega_per_ns):
        raise ValueError("Frequency must be finite")
    N = max(L, J)
    d0 = medium.extinction_per_m - 1j * omega_per_ns / medium.speed_m_per_ns
    b, ratios = _free_moments_and_ratios(k, d0, N, backend=backend)
    gamma = medium.scattering_per_m * medium.g ** np.arange(L + 1)
    # Eliminate the free tail immediately above L, independent of output degree J.
    first_low = solve_tail_system(k, d0, ratios[:, L], np.zeros(L + 1), b[:, :L + 1] * gamma)
    multiple_low = solve_tail_system(k, d0, ratios[:, L], gamma, first_low * gamma)
    if N == L:
        return b, first_low, multiple_low
    first = np.empty_like(b, order="F")
    multiple = np.empty_like(b, order="F")
    first[:, :L + 1] = first_low
    multiple[:, :L + 1] = multiple_low
    # Above L all right-hand sides vanish: extend by exact free ratios.
    extension = np.cumprod(ratios[:, L:N], axis=1)
    first[:, L + 1:] = first_low[:, -1:] * extension
    multiple[:, L + 1:] = multiple_low[:, -1:] * extension
    return b, first, multiple


def dense_finite_rank_reference(k, omega_per_ns, medium, scattering_degree,
                                output_degree, quadrature_order=1024):
    """Independent dense B-matrix implementation, for small validation cases.

    It uses direct Gauss angular integration, not the tail recurrence.
    Returns projected full response h_l for l=0..output_degree.
    Large k/extinction may need a larger quadrature_order.
    """
    L = _degree(scattering_degree, "scattering_degree")
    J = _degree(output_degree, "output_degree")
    mu, w = roots_legendre(quadrature_order)
    degrees = np.arange(max(L, J) + 1)
    p = eval_legendre(degrees[:, None], mu) * np.sqrt((2 * degrees + 1) / 2)[:, None]
    D = medium.extinction_per_m - 1j * omega_per_ns / medium.speed_m_per_ns + 1j * k * mu
    b = p[:L + 1] @ (w / D)
    B = (p[:L + 1] * (w / D)) @ p[:L + 1].T
    gamma = medium.scattering_per_m * medium.g ** np.arange(L + 1)
    a = np.linalg.solve(np.eye(L + 1) - gamma[:, None] * B, gamma * b)
    field = (1 + a @ p[:L + 1]) / D
    return p[:J + 1] @ (w * field)


# ---------------------------------------------------------------------------
# Azimuthal blocks m != 0.
#
# In the k-frame the streaming operator is multiplication by mu = s.k and the
# Henyey-Greenstein kernel is diagonal in (l, m), so the angular system splits
# into independent blocks, one per m:
#
#     (d0 - gamma_l) h_lm + i k a_lm h_(l-1)m + i k a_(l+1)m h_(l+1)m = A_lm,
#     a_lm = sqrt((l^2 - m^2) / (4 l^2 - 1)),        l = |m| ... .
#
# Three properties are used downstream and are derived, not assumed:
#   * the block starts at l = |m| because a_|m|,m = 0;
#   * the matrix is complex symmetric, so a column of the resolvent is also a
#     row -- one solve serves both the source and the detector index;
#   * a_lm depends on m^2, hence G^(m) = G^(-m).
# See docs/research/directional-om-plan.md, sections 3 and 4.
# ---------------------------------------------------------------------------


def coupling_coefficients(degree, m):
    """``a_{l,m}`` for ``l = 0 .. degree``; zero where ``l < |m|``."""
    degree = _degree(degree, "degree")
    m = abs(int(m))
    ell = np.arange(degree + 1)
    value = np.zeros(degree + 1)
    high = ell >= max(m, 1)
    value[high] = np.sqrt((ell[high] ** 2 - m * m)
                          / (4 * ell[high] ** 2 - 1))
    return value


def _decay_rate(z):
    """|rho| and eta = -log|rho| of the minimal free solution."""
    root = np.sqrt(z - 1) * np.sqrt(z + 1)
    decay = 1 / (z + root)
    decay = np.where(abs(decay) > 1, 1 / decay, decay)
    return decay, -np.log(np.abs(decay))


def tail_ratios_continued_fraction(k, d0, degree, m, *, depth=None,
                                   depth_cap=200000):
    """``R^(m)_l = y_(l+1)/y_l`` of the minimal free solution, ``l = |m|..degree``.

    Downward continued fraction

        R_(l-1) = a_lm / (z - a_(l+1)m R_l),       z = i d0 / k,

    started from ``R = 0`` at ``depth``. The minimal solution is the stable
    fixed point of that map (Pincherle), so the branch is selected by the
    direction of the iteration rather than by a choice of sign, and the
    starting guess is damped as ``|rho|^(2(depth-l))``.

    This routine is the *reference* closure: it is independent of the m = 0
    machinery and of the raising relation used by :func:`free_tail_ratios_m`.
    Entries below ``|m|`` are zero.
    """
    N = _degree(degree, "degree")
    m = abs(int(m))
    k = np.asarray(k, dtype=float)
    d0 = complex(d0)
    if k.ndim != 1 or not len(k) or not np.isfinite(k).all() or np.any(k < 0):
        raise ValueError("k must be a nonempty 1-D array of finite nonnegative wave numbers")
    if not np.isfinite(d0) or d0.real <= 0:
        raise ValueError("d0 must be finite with positive real part")
    out = np.zeros((len(k), N + 1), dtype=complex)
    nz = k > 0
    if not np.any(nz):
        return out
    z = 1j * d0 / k[nz]
    _, eta = _decay_rate(z)
    if depth is None:
        need = N + 1 + np.maximum(32, np.ceil(28 / np.maximum(eta, 1e-300)))
        top = int(min(depth_cap, np.max(need)))
    else:
        top = int(depth)
    if top <= N:
        raise ValueError("continued-fraction depth must exceed the degree")
    a = coupling_coefficients(top + 1, m)
    ratio = np.zeros(len(z), dtype=complex)
    block = np.zeros((len(z), N + 1), dtype=complex)
    for ell in range(top, m, -1):
        ratio = a[ell] / (z - a[ell + 1] * ratio)
        if ell - 1 <= N:
            block[:, ell - 1] = ratio
    out[nz] = block
    return out


def free_tail_ratios_m(k, d0, degree, max_m, *, backend="numpy",
                       guard=1e-6, report=None):
    r"""Free tail ratios ``R^(m)_l`` for ``|m| <= max_m``, shape (K, max_m+1, degree+1).

    ``m = 0`` is the existing, validated Miller/forward scheme. Higher orders
    are obtained by the associated-Legendre raising relation written purely in
    terms of ratios,

        v^(m)_l   = (l-m) z - (l+m) b_l / R^(m)_(l-1),
        R^(m+1)_l = sqrt((l-m)(l+m+1) / ((l+1-m)(l+m+2)))
                    * v^(m)_(l+1) / v^(m)_l * R^(m)_l,

    with ``b_l = sqrt((2l+1)(l-m) / ((2l-1)(l+m)))``. No two underflowing
    moments are ever divided: ratios go in, ratios come out, and the
    ``(z^2-1)^(-1/2)`` of the textbook relation cancels and is never formed.

    The raising step is singular at a zero of the minimal solution. Nodes where
    ``|v|`` falls below ``guard`` times the larger of its two terms are recomputed
    with :func:`tail_ratios_continued_fraction`; ``report`` (a dict) receives the
    count.
    """
    N = _degree(degree, "degree")
    max_m = _degree(max_m, "max_m")
    k = np.asarray(k, dtype=float)
    d0 = complex(d0)
    if k.ndim != 1 or not len(k) or not np.isfinite(k).all() or np.any(k < 0):
        raise ValueError("k must be a nonempty 1-D array of finite nonnegative wave numbers")
    if not np.isfinite(d0) or d0.real <= 0:
        raise ValueError("d0 must be finite with positive real part")
    if not np.isfinite(guard) or guard < 0:
        raise ValueError("guard must be finite and nonnegative")
    extended = N + max_m + 1
    _, ratios = _free_moments_and_ratios(k, d0, extended, backend=backend)
    out = np.zeros((len(k), max_m + 1, extended + 1), dtype=complex)
    out[:, 0, :ratios.shape[1]] = ratios
    nz = k > 0
    fallbacks = 0
    if np.any(nz) and max_m:
        z = (1j * d0 / k[nz])[:, None]
        for m in range(max_m):
            previous = out[nz, m]
            ell = np.arange(extended + 1)
            usable = ell >= m + 1
            b = np.zeros(extended + 1)
            b[usable] = np.sqrt((2 * ell[usable] + 1) * (ell[usable] - m)
                                / ((2 * ell[usable] - 1) * (ell[usable] + m)))
            first = (ell[None, :] - m) * z
            second = np.zeros_like(first)
            with np.errstate(divide="ignore", invalid="ignore"):
                second[:, usable] = ((ell[usable] + m) * b[usable])[None, :] \
                    / previous[:, np.asarray(usable).nonzero()[0] - 1]
            v = first - second
            reference = np.maximum(np.abs(first), np.abs(second))
            bad = usable[None, :] & (np.abs(v) <= guard * np.maximum(reference, 1e-300))
            step = np.zeros(extended + 1)
            take = ell <= extended - 1
            valid = usable & take
            index = np.asarray(valid).nonzero()[0]
            step[index] = np.sqrt((index - m) * (index + m + 1)
                                  / ((index + 1 - m) * (index + m + 2)))
            raised = np.zeros_like(previous)
            with np.errstate(divide="ignore", invalid="ignore"):
                raised[:, index] = (step[index][None, :] * v[:, index + 1]
                                    / v[:, index] * previous[:, index])
            raised[:, :m + 1] = 0.0
            if np.any(bad) or not np.isfinite(raised[:, m + 1:extended]).all():
                fallbacks += int(np.count_nonzero(bad))
                reference_ratio = tail_ratios_continued_fraction(
                    k[nz], d0, extended, m + 1)
                rows = np.any(bad, axis=1) | ~np.isfinite(raised).all(axis=1)
                raised[rows] = reference_ratio[rows]
            out[nz, m + 1] = raised
    if report is not None:
        report["tail_raise_fallback_nodes"] = fallbacks
    return np.ascontiguousarray(out[:, :, :N + 1])


def solve_tail_system_m(k, d0, tail, gamma, rhs, m):
    r"""Batched tridiagonal solve for one azimuthal block, ``l = |m| .. |m|+n-1``.

    ``rhs`` has shape (K, n) and covers ``l = |m| .. |m| + n - 1``; ``gamma`` has
    length n and is the scattering eigenvalue at those same degrees. The last
    diagonal carries ``i k a_(|m|+n, m) * tail``, the exact Schur complement of
    every free harmonic above the truncation.
    """
    m = abs(int(m))
    k = np.asarray(k, dtype=float)
    rhs = np.asarray(rhs, dtype=complex)
    gamma = np.asarray(gamma, dtype=float)
    tail = np.asarray(tail, dtype=complex)
    if rhs.ndim != 2 or rhs.shape != (len(k), len(gamma)) or tail.shape != k.shape:
        raise ValueError("Incompatible k, gamma, tail, rhs shapes")
    if not np.isfinite(rhs).all() or not np.isfinite(gamma).all():
        raise ValueError("Nonfinite linear-system data")
    n = rhs.shape[1]
    a = coupling_coefficients(m + n, m)[m:]           # a_{m+i, m}, i = 0..n
    off = np.asfortranarray(1j * k[:, None] * a[None, 1:n])
    diagonal = np.array(np.broadcast_to(d0 - gamma[None, :], rhs.shape),
                        order="F", copy=True)
    diagonal[:, -1] += 1j * k * a[n] * tail
    r = np.array(rhs, order="F", copy=True)
    for i in range(1, n):
        multiplier = off[:, i - 1] / diagonal[:, i - 1]
        diagonal[:, i] -= multiplier * off[:, i - 1]
        r[:, i] -= multiplier * r[:, i - 1]
    x = np.empty_like(r, order="F")
    x[:, -1] = r[:, -1] / diagonal[:, -1]
    for i in range(n - 2, -1, -1):
        x[:, i] = (r[:, i] - off[:, i] * x[:, i + 1]) / diagonal[:, i]
    if not np.isfinite(x).all():
        raise FloatingPointError("Nonfinite angular solution")
    return x


def resolvent_rows(k, omega_per_ns, medium, scattering_degree, output_degree,
                   acceptance_degree, *, backend="numpy", report=None):
    r"""Rows ``lambda <= acceptance_degree`` of the scattered-order resolvents.

    Returns ``(first, multiple)``; each is a dict keyed by ``(m, lam)`` with
    ``0 <= m <= lam <= acceptance_degree`` holding an array of shape
    ``(K, output_degree + 1)`` indexed by the *source* degree ``l``. Entries with
    ``l < m`` are zero, because the block does not carry them.

    ``first`` is row ``lambda`` of ``G0 Gamma G0`` and ``multiple`` is row
    ``lambda`` of ``G Gamma G0 Gamma G0``; both operators are symmetric, so a
    row is obtained from the solve with right-hand side ``e_lambda``. With
    ``acceptance_degree = 0`` this is the m = 0 content of
    :func:`angular_components` up to the source normalisation ``sqrt(2)``.
    """
    L = _degree(scattering_degree, "scattering_degree")
    J = _degree(output_degree, "output_degree")
    A = _degree(acceptance_degree, "acceptance_degree")
    if A > L:
        raise ValueError("acceptance_degree must not exceed scattering_degree")
    if not np.isfinite(omega_per_ns):
        raise ValueError("Frequency must be finite")
    N = max(L, J)
    k = np.asarray(k, dtype=float)
    d0 = medium.extinction_per_m - 1j * omega_per_ns / medium.speed_m_per_ns
    ratios = free_tail_ratios_m(k, d0, N, A, backend=backend, report=report)
    gamma_full = medium.scattering_per_m * medium.g ** np.arange(L + 1)
    first, multiple = {}, {}
    for m in range(A + 1):
        size = L + 1 - m
        gamma = gamma_full[m:]
        tail = ratios[:, m, L]
        unit = np.zeros((len(k), size), dtype=complex)
        for lam in range(m, A + 1):
            unit[:] = 0.0
            unit[:, lam - m] = 1.0
            free = solve_tail_system_m(k, d0, tail, np.zeros(size), unit, m)
            low_first = solve_tail_system_m(k, d0, tail, np.zeros(size),
                                            free * gamma, m)
            low_multiple = solve_tail_system_m(k, d0, tail, gamma,
                                               low_first * gamma, m)
            for name, low in (("first", low_first), ("multiple", low_multiple)):
                full = np.zeros((len(k), N + 1), dtype=complex)
                full[:, m:L + 1] = low
                if N > L:
                    extension = np.cumprod(ratios[:, m, L:N], axis=1)
                    full[:, L + 1:] = low[:, -1:] * extension
                (first if name == "first" else multiple)[(m, lam)] = full
    return first, multiple


def dense_block_reference(k, omega_per_ns, medium, scattering_degree,
                          output_degree, m):
    """Dense inverse of one azimuthal block, hard truncation, no tail closure.

    Independent of the Thomas sweep and of the tail machinery, for small cases.
    Returns ``G^(m)`` of shape ``(K, n, n)`` over ``l = |m| .. output_degree``.
    """
    L = _degree(scattering_degree, "scattering_degree")
    J = _degree(output_degree, "output_degree")
    m = abs(int(m))
    k = np.asarray(k, dtype=float)
    d0 = medium.extinction_per_m - 1j * omega_per_ns / medium.speed_m_per_ns
    ell = np.arange(m, J + 1)
    a = coupling_coefficients(J + 1, m)
    gamma = np.where(ell <= L, medium.scattering_per_m * medium.g ** ell, 0.0)
    n = len(ell)
    out = np.empty((len(k), n, n), dtype=complex)
    for index, value in enumerate(k):
        matrix = np.diag(d0 - gamma).astype(complex)
        for i in range(1, n):
            matrix[i, i - 1] = 1j * value * a[ell[i]]
            matrix[i - 1, i] = 1j * value * a[ell[i]]
        out[index] = np.linalg.inv(matrix)
    return out
