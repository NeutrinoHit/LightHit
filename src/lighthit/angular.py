"""m=0 adjoint Green coefficients, with an exact *free* angular tail.

The medium scattering operator is truncated at L; streaming is not.
The output degree J controls the subsequent spatial angular inversion.
No discretized intensity I(x,s,t) is constructed.
"""
from functools import lru_cache
import numpy as np
from scipy.special import eval_legendre, roots_legendre


def _degree(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def _free_moments_and_ratios(k, d0, degree):
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
        top = N + 1 + max(32, int(np.ceil(28 / eta[~forward].min())))
        ratio = decay[~forward] * (1 - 0.5 / (top + 1))
        ratios = np.empty((len(zz), N + 1), dtype=complex)
        for ell in range(top, 0, -1):
            ratio = ell / ((2 * ell + 1) * zz - (ell + 1) * ratio)
            if ell <= N + 1:
                ratios[:, ell - 1] = ratio
        moments[~forward, 1:] = moments[~forward, :1] * np.cumprod(ratios, axis=1)
        ratios_normalized[~forward] = ratios
    b[nz] = moments[:, :N + 1] * np.sqrt((2 * np.arange(N + 1) + 1) / 2)
    j = np.arange(N + 1)
    ratios_all[nz] = np.sqrt((2 * j + 3) / (2 * j + 1)) * ratios_normalized
    return b, ratios_all


def free_moments_and_tail(k, d0, degree):
    """Free moments (K,N+1) and exact normalized tail ratio (K,)."""
    b, ratios = _free_moments_and_ratios(k, d0, degree)
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


def angular_components(k, omega_per_ns, medium, scattering_degree, output_degree):
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
    b, ratios = _free_moments_and_ratios(k, d0, N)
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
