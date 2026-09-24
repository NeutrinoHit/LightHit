"""Fused kernel for the directional-OM apply.

The arithmetic is the one written out in
:func:`lighthit.directional.rotation_rows_explicit` and in equation (*) of
:mod:`lighthit.directional`; nothing is approximated here that is not
approximated there. What changes is only what is kept in memory: the running
sum for one (receiver, cell) pair lives in registers, and no array indexed by
(OM, cell, source channel, detector channel, frequency) is ever formed.

Numba is optional until this module is imported. No fastmath is used.

Every routine has a numpy twin that is exercised by the test suite with or
without numba: :func:`lighthit.directional.rotation_rows_explicit` for the
rotation and :func:`lighthit.directional.directional_response` for the whole
apply. ``tests/test_directional_numba.py`` asserts that the two agree.
"""
from dataclasses import dataclass
from math import exp, lgamma

import numpy as np
from scipy.interpolate import CubicSpline

try:
    from numba import njit, prange
except ImportError as exc:  # pragma: no cover - exercised only without numba
    raise ImportError("_directional_numba requires the optional 'accelerate' extra") from exc

__all__ = ["PreparedDirectionalKernel"]


@njit(cache=True, nogil=True, inline="always")
def _parity(n):
    return 1.0 if n % 2 == 0 else -1.0


@njit(cache=True, nogil=True)
def _start_value(j, mu, nu, half_cos, half_sin, start_norm, jmax):
    """``d^j_(mu nu)`` at ``j = max(|mu|, |nu|)``, from a precomputed norm table."""
    sign = 1.0
    if abs(nu) == j and abs(mu) != j:
        mu, nu = nu, mu
        sign *= _parity(mu - nu)
    if mu == -j:
        mu = j
        nu = -nu
        sign *= _parity(j + nu)
    return (sign * start_norm[j, nu + jmax]
            * half_cos ** (j + nu) * (-half_sin) ** (j - nu))


@njit(cache=True, nogil=True)
def _wigner_into(degree, max_out, max_in, cosine, start_norm, jmax, out):
    """Fill ``out[l, nu + max_out, mu]`` with ``d^l_(mu nu)(theta)``.

    ``mu`` runs over ``0 .. max_in`` only: the explicit real rotation needs no
    negative input index. The recurrence in ``l`` at fixed magnetic indices is
    the stable direction.
    """
    for a in range(out.shape[0]):
        for b in range(out.shape[1]):
            for c in range(out.shape[2]):
                out[a, b, c] = 0.0
    half_cos = np.sqrt((1.0 + cosine) / 2.0)
    half_sin = np.sqrt((1.0 - cosine) / 2.0)
    for nu in range(-max_out, max_out + 1):
        for mu in range(max_in + 1):
            start = max(abs(nu), mu)
            if start > degree:
                continue
            previous = 0.0
            current = _start_value(start, mu, nu, half_cos, half_sin,
                                   start_norm, jmax)
            out[start, nu + max_out, mu] = current
            for ell in range(start, degree):
                if ell == 0:
                    value = cosine * current
                else:
                    lower = (ell * ell - mu * mu) * (ell * ell - nu * nu)
                    lower = np.sqrt(lower) if lower > 0.0 else 0.0
                    upper = np.sqrt(((ell + 1) ** 2 - mu * mu)
                                    * ((ell + 1) ** 2 - nu * nu))
                    value = (((2 * ell + 1) * (ell * (ell + 1) * cosine - mu * nu)
                              * current - (ell + 1) * lower * previous)
                             / (ell * upper))
                previous = current
                current = value
                out[ell + 1, nu + max_out, mu] = current


@njit(cache=True, nogil=True)
def _harmonics_into(x, y, z, degree, out):
    """Real orthonormal harmonics of a unit vector, all orders, flat ``l*l+l+m``."""
    radius = np.sqrt(x * x + y * y + z * z)
    mu = z / radius
    transverse = np.sqrt(x * x + y * y)
    sine = transverse / radius
    cp = 1.0 if transverse == 0.0 else x / transverse
    sp = 0.0 if transverse == 0.0 else y / transverse
    cm = 1.0
    sm = 0.0
    diagonal = 1.0 / np.sqrt(4 * np.pi)
    v0 = 0.0
    v1 = 0.0
    for m in range(degree + 1):
        if m:
            diagonal *= np.sqrt((2.0 * m + 1) / (2.0 * m)) * sine
            cm, sm = cm * cp - sm * sp, sm * cp + cm * sp
        v0 = diagonal
        for ell in range(m, degree + 1):
            if ell == m:
                value = v0
            elif ell == m + 1:
                value = np.sqrt(2 * m + 3) * mu * v0
                v1 = value
            else:
                aa = np.sqrt((4.0 * ell * ell - 1) / (ell * ell - m * m))
                bb = np.sqrt(((2.0 * ell + 1) * ((ell - 1) ** 2 - m * m))
                             / ((2.0 * ell - 3) * (ell * ell - m * m)))
                value = aa * mu * v1 - bb * v0
                v0 = v1
                v1 = value
            centre = ell * ell + ell
            if m == 0:
                out[centre] = value
            else:
                out[centre - m] = np.sqrt(2) * value * sm
                out[centre + m] = np.sqrt(2) * value * cm
    return radius


@njit(cache=True, nogil=True, parallel=True)
def _apply_m0(points, receivers, looks, channels, second_channels,
              first_scale, second_scale, starts, degree, max_out,
              alpha, pair_lambda, pair_mu, log_r, coefficients, absorption,
              velocity, omega, flight, start_norm, jmax, receiver_block,
              cell_begin, cell_end):
    """Specialization of equation (*) for an exactly axisymmetric m=0 source."""
    nfreq = len(omega)
    nrec = receivers.shape[0]
    npair = len(pair_lambda)
    result = np.zeros((nrec, nfreq, 2), np.complex128)
    groups = (nrec + receiver_block - 1) // receiver_block
    width = 2 * max_out + 1
    root2 = np.sqrt(2.0)
    for group in prange(groups):
        group_begin = group * receiver_block
        count = min(receiver_block, nrec - group_begin)
        wigner = np.empty((degree + 1, width, 1), np.float64)
        detector = np.empty((max_out + 1) * (max_out + 1), np.float64)
        scale_phase = np.empty(nfreq, np.complex128)
        radial_sum = np.empty((nfreq, 2), np.complex128)
        for d in range(count):
            index = group_begin + d
            for cell in range(cell_begin[index], cell_end[index]):
                dx = receivers[index, 0] - points[cell, 0]
                dy = receivers[index, 1] - points[cell, 1]
                dz = receivers[index, 2] - points[cell, 2]
                radius = np.sqrt(dx * dx + dy * dy + dz * dz)
                cosine = dz / radius
                if cosine > 1.0:
                    cosine = 1.0
                elif cosine < -1.0:
                    cosine = -1.0
                transverse = np.sqrt(dx * dx + dy * dy)
                sine = transverse / radius
                cp = 1.0 if transverse == 0.0 else dx / transverse
                sp = 0.0 if transverse == 0.0 else dy / transverse

                lx = looks[index, 0]
                ly = looks[index, 1]
                lz = looks[index, 2]
                a1 = cosine * cp * lx + cosine * sp * ly - sine * lz
                a2 = -sp * lx + cp * ly
                a3 = sine * cp * lx + sine * sp * ly + cosine * lz
                _harmonics_into(a1, a2, a3, max_out, detector)
                _wigner_into(degree, max_out, 0, cosine, start_norm, jmax, wigner)

                logr = np.log(radius)
                j = np.searchsorted(log_r, logr, side="right") - 1
                if j < 0:
                    j = 0
                if j > len(log_r) - 2:
                    j = len(log_r) - 2
                t = logr - log_r[j]
                scale = np.exp(-absorption * radius) / (4 * np.pi * radius * radius)
                for w in range(nfreq):
                    if flight:
                        angle = omega[w] * radius / velocity
                        scale_phase[w] = scale * (np.cos(angle) + 1j * np.sin(angle))
                    else:
                        scale_phase[w] = scale

                for ell in range(degree + 1):
                    source_column = starts[ell]
                    for w in range(nfreq):
                        radial_sum[w, 0] = 0.0
                        radial_sum[w, 1] = 0.0
                    for p in range(npair):
                        lam = pair_lambda[p]
                        mu = pair_mu[p]
                        if mu > min(ell, max_out):
                            continue
                        if mu == 0:
                            rotation = wigner[ell, max_out, 0]
                        else:
                            rotation = (root2 * _parity(mu)
                                        * wigner[ell, max_out + mu, 0])
                        if rotation == 0.0:
                            continue
                        detector_column = lam * lam + lam + mu
                        weight = alpha[lam] * detector[detector_column] * rotation
                        if weight == 0.0:
                            continue
                        for order in range(2):
                            for w in range(nfreq):
                                c0 = coefficients[j, ell, p, 0, order, w]
                                c1 = coefficients[j, ell, p, 1, order, w]
                                c2 = coefficients[j, ell, p, 2, order, w]
                                c3 = coefficients[j, ell, p, 3, order, w]
                                radial = ((c0 * t + c1) * t + c2) * t + c3
                                radial_sum[w, order] += weight * radial
                    for w in range(nfreq):
                        if second_scale == 0.0:
                            value = channels[cell, source_column, w]
                        else:
                            value = (first_scale * channels[cell, source_column, w]
                                     - second_scale * second_channels[cell, source_column, w])
                        value *= scale_phase[w]
                        result[index, w, 0] += value * radial_sum[w, 0]
                        result[index, w, 1] += value * radial_sum[w, 1]
    return result


@njit(cache=True, nogil=True, parallel=True)
def _apply(points, receivers, looks, channels, second_channels,
           first_scale, second_scale, starts, degree, max_m, max_out,
           alpha, pair_lambda, pair_mu, log_r, coefficients, absorption,
           velocity, omega, flight, start_norm, jmax, receiver_block,
           cell_begin, cell_end):
    """Equation (*) for the requested (receiver, cell-window) pairs."""
    nfreq = len(omega)
    nrec = receivers.shape[0]
    npair = len(pair_lambda)
    width = 2 * max_out + 1
    result = np.zeros((nrec, nfreq, 2), np.complex128)
    groups = (nrec + receiver_block - 1) // receiver_block
    for group in prange(groups):
        begin = group * receiver_block
        count = min(receiver_block, nrec - begin)
        wigner = np.empty((degree + 1, width, max_m + 1), np.float64)
        detector = np.empty((max_out + 1) * (max_out + 1), np.float64)
        erow = np.empty((width, 2 * max_m + 1), np.float64)
        rotated = np.empty((width, nfreq), np.complex128)
        channel_sum = np.empty((width, nfreq, 2), np.complex128)
        cosines = np.empty(max_m + 1, np.float64)
        sines = np.empty(max_m + 1, np.float64)
        scale_phase = np.empty(nfreq, np.complex128)
        root2 = np.sqrt(2.0)
        for d in range(count):
            index = begin + d
            for cell in range(cell_begin[index], cell_end[index]):
                dx = receivers[index, 0] - points[cell, 0]
                dy = receivers[index, 1] - points[cell, 1]
                dz = receivers[index, 2] - points[cell, 2]
                radius = np.sqrt(dx * dx + dy * dy + dz * dz)
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
                # module axis in the r-hat frame: e1 = (c cp, c sp, -s),
                # e2 = (-sp, cp, 0), e3 = (s cp, s sp, c)
                lx = looks[index, 0]
                ly = looks[index, 1]
                lz = looks[index, 2]
                a1 = cosine * cp * lx + cosine * sp * ly - sine * lz
                a2 = -sp * lx + cp * ly
                a3 = sine * cp * lx + sine * sp * ly + cosine * lz
                _harmonics_into(a1, a2, a3, max_out, detector)
                _wigner_into(degree, max_out, max_m, cosine, start_norm, jmax,
                             wigner)
                logr = np.log(radius)
                j = np.searchsorted(log_r, logr, side='right') - 1
                if j < 0:
                    j = 0
                if j > len(log_r) - 2:
                    j = len(log_r) - 2
                t = logr - log_r[j]
                scale = np.exp(-absorption * radius) / (4 * np.pi * radius * radius)
                for w in range(nfreq):
                    if flight:
                        angle = omega[w] * radius / velocity
                        scale_phase[w] = scale * (np.cos(angle) + 1j * np.sin(angle))
                    else:
                        scale_phase[w] = scale
                for ell in range(degree + 1):
                    # ---- rotation rows for this degree, explicit real form
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
                    # ---- source channels rotated into the r-hat frame
                    keep = min(ell, max_m)
                    centre = starts[ell] + min(ell, max_m)
                    for a in range(width):
                        for w in range(nfreq):
                            rotated[a, w] = 0.0
                    for a in range(max_out - min(ell, max_out),
                                   max_out + min(ell, max_out) + 1):
                        for b in range(max_m - keep, max_m + keep + 1):
                            factor = erow[a, b]
                            if factor == 0.0:
                                continue
                            column = centre + (b - max_m)
                            for w in range(nfreq):
                                if second_scale == 0.0:
                                    value = channels[cell, column, w]
                                else:
                                    value = (first_scale * channels[cell, column, w]
                                             - second_scale * second_channels[cell, column, w])
                                rotated[a, w] += factor * value
                    # ---- radial coefficients contracted over lambda
                    for a in range(width):
                        for w in range(nfreq):
                            channel_sum[a, w, 0] = 0.0
                            channel_sum[a, w, 1] = 0.0
                    for p in range(npair):
                        lam = pair_lambda[p]
                        mu_pair = pair_mu[p]
                        if mu_pair > min(ell, max_out):
                            continue
                        for sign in range(-1, 2, 2):
                            nu = sign * mu_pair
                            if mu_pair == 0 and sign == -1:
                                continue
                            column = lam * lam + lam + nu
                            weight = alpha[lam] * detector[column]
                            if weight == 0.0:
                                continue
                            slot = nu + max_out
                            for order in range(2):
                                for w in range(nfreq):
                                    c0 = coefficients[j, ell, p, 0, order, w]
                                    c1 = coefficients[j, ell, p, 1, order, w]
                                    c2 = coefficients[j, ell, p, 2, order, w]
                                    c3 = coefficients[j, ell, p, 3, order, w]
                                    radial = ((c0 * t + c1) * t + c2) * t + c3
                                    channel_sum[slot, w, order] += weight * radial
                    # ---- contract
                    for a in range(width):
                        for w in range(nfreq):
                            value = rotated[a, w]
                            if value == 0.0:
                                continue
                            phase = scale_phase[w]
                            for order in range(2):
                                result[index, w, order] += (
                                    value * channel_sum[a, w, order] * phase)
    return result


def _start_norm_table(jmax):
    """``sqrt(binom(2j, j-nu))``, in log form so that large ``j`` cannot overflow."""
    table = np.zeros((jmax + 1, 2 * jmax + 1))
    for j in range(jmax + 1):
        for nu in range(-j, j + 1):
            table[j, nu + jmax] = exp(0.5 * (lgamma(2 * j + 1)
                                             - lgamma(j + nu + 1)
                                             - lgamma(j - nu + 1)))
    return table


@dataclass
class PreparedDirectionalKernel:
    """The same radial spline as :class:`~lighthit.directional.DirectionalCache`.

    Built once per cache and retained degree; mutating the cache afterwards is
    not reflected here.
    """
    degree: int
    acceptance_degree: int
    pair_lambda: np.ndarray
    pair_mu: np.ndarray
    omega_per_ns: np.ndarray
    log_r: np.ndarray
    coefficients: np.ndarray
    absorption_per_m: float
    speed_m_per_ns: float
    radial_phase: str

    @classmethod
    def from_cache(cls, cache, degree=None, frequency_indices=None):
        degree = cache.degree if degree is None else int(degree)
        if not 0 <= degree <= cache.degree:
            raise ValueError("degree must be an integer within the cache")
        axis = np.asarray(cache.grid.omega_per_ns, float)
        if frequency_indices is None:
            index = np.arange(len(axis))
        else:
            index = np.asarray(frequency_indices)
            if index.ndim != 1 or not len(index) or index.dtype.kind not in "iu":
                raise ValueError("frequency_indices must be a nonempty integer vector")
            if np.any(index < 0) or np.any(index >= len(axis)):
                raise ValueError("frequency index out of range")
        omega = axis[index].copy()
        radii = np.asarray(cache.grid.radii_m, float)
        scale = np.exp(-cache.medium.absorption_per_m * radii) / (4 * np.pi * radii ** 2)
        if np.any(scale == 0) or not np.isfinite(scale).all():
            raise ValueError("Radial scale underflow/nonfinite")
        values = cache.coefficients[index][:, :, :degree + 1] / scale[None, :, None, None, None]
        phase = getattr(cache, "radial_phase", "none")
        if phase not in ("none", "flight"):
            raise ValueError("Unknown radial_phase")
        if phase == "flight":
            values = values * np.exp(
                -1j * omega[:, None] * radii[None, :] / cache.medium.speed_m_per_ns
            )[:, :, None, None, None]
        spline = CubicSpline(np.log(radii), values, axis=1)
        # spline.c: (power, interval, frequency, degree, pair, order)
        coefficients = np.ascontiguousarray(
            np.transpose(spline.c, (1, 3, 4, 0, 5, 2)))
        return cls(int(degree), cache.table.acceptance_degree,
                   np.ascontiguousarray(cache.table.pair_lambda),
                   np.ascontiguousarray(cache.table.pair_mu),
                   omega, np.log(radii), coefficients,
                   cache.medium.absorption_per_m, cache.medium.speed_m_per_ns,
                   phase)

    def apply(self, source, receivers_m, look_directions, alpha, *,
              source_omega_per_ns, receiver_block=4,
              cell_begin=None, cell_end=None, second_source=None,
              first_scale=1.0, second_scale=0.0):
        """Return both scattered orders, shape (frequency, receiver, 2).

        ``cell_begin``/``cell_end`` optionally select one contiguous source-cell
        window per receiver.  The default is the complete source, preserving the
        previous API and arithmetic.
        """
        axis = np.asarray(source_omega_per_ns, float)
        if axis.shape != self.omega_per_ns.shape or not np.array_equal(axis, self.omega_per_ns):
            raise ValueError("Source and kernel frequency grids differ")
        if self.degree > source.degree:
            raise ValueError("Source degree is insufficient")
        if not isinstance(receiver_block, (int, np.integer)) or receiver_block < 1:
            raise ValueError("receiver_block must be a positive integer")
        receivers = np.atleast_2d(np.asarray(receivers_m, float))
        looks = np.atleast_2d(np.asarray(look_directions, float))
        if receivers.ndim != 2 or receivers.shape[1] != 3 or not len(receivers):
            raise ValueError("receivers must be a nonempty (D,3) array")
        if looks.shape != receivers.shape:
            raise ValueError("look_directions must match receivers")
        if not np.allclose(np.linalg.norm(looks, axis=1), 1.0, rtol=0, atol=1e-9):
            raise ValueError("look_directions must be unit vectors")
        alpha = np.asarray(alpha, float)
        if len(alpha) < self.acceptance_degree + 1:
            raise ValueError("alpha must cover the acceptance degree of the cache")
        alpha = np.ascontiguousarray(alpha[:self.acceptance_degree + 1])
        max_m = min(int(source.azimuthal_degree), self.degree)
        starts = [0]
        for ell in range(source.degree + 1):
            starts.append(starts[-1] + 2 * min(ell, source.azimuthal_degree) + 1)
        starts = np.asarray(starts, np.int64)
        data = np.ascontiguousarray(source.channels)
        if data.dtype != np.complex128 or data.ndim != 3 or data.shape[2] != len(axis):
            raise ValueError("Expected complex128 (cell,channel,frequency) source")
        if second_source is None:
            second_data = data
            if second_scale != 0.0 or first_scale != 1.0:
                raise ValueError("field scales require a second source")
        else:
            second_data = np.ascontiguousarray(second_source.channels)
            if (second_data.shape != data.shape
                    or second_source.reference_ns != source.reference_ns
                    or not np.array_equal(second_source.kept, source.kept)
                    or not np.array_equal(second_source.points_m(), source.points_m())):
                raise ValueError("two source fields must share cells, channels and time origin")
            if (not np.isfinite(first_scale) or not np.isfinite(second_scale)):
                raise ValueError("source field scales must be finite")
        points = np.asarray(source.points_m(), float)
        centre = np.asarray(source.frame.centre_m, float)
        local_points = np.ascontiguousarray(source.frame.rotate(points - centre))
        local_receivers = np.ascontiguousarray(source.frame.rotate(receivers - centre))
        local_looks = np.ascontiguousarray(source.frame.rotate(looks))

        if (cell_begin is None) != (cell_end is None):
            raise ValueError("cell_begin and cell_end must be provided together")
        if cell_begin is None:
            begin = np.zeros(len(receivers), dtype=np.int64)
            end = np.full(len(receivers), len(local_points), dtype=np.int64)
        else:
            begin = np.ascontiguousarray(cell_begin, dtype=np.int64)
            end = np.ascontiguousarray(cell_end, dtype=np.int64)
            if begin.shape != (len(receivers),) or end.shape != (len(receivers),):
                raise ValueError("cell windows must contain one pair per receiver")
            if (np.any(begin < 0) or np.any(end < begin)
                    or np.any(end > len(local_points))):
                raise ValueError("invalid source-cell window")

        radial_low = np.exp(self.log_r[0])
        radial_high = np.exp(self.log_r[-1])
        for index, receiver in enumerate(local_receivers):
            subset = local_points[begin[index]:end[index]]
            if not len(subset):
                continue
            radius = np.linalg.norm(subset - receiver, axis=1)
            if np.any(radius < radial_low) or np.any(radius > radial_high):
                raise ValueError("Source/receiver distances outside the cache")
        jmax = max(max_m, self.acceptance_degree)
        common = (
            alpha, self.pair_lambda, self.pair_mu, self.log_r,
            self.coefficients, self.absorption_per_m, self.speed_m_per_ns,
            self.omega_per_ns, self.radial_phase == "flight",
            _start_norm_table(jmax), int(jmax), int(receiver_block), begin, end)
        if max_m == 0:
            result = _apply_m0(
                local_points, local_receivers, local_looks, data, second_data,
                float(first_scale), float(second_scale), starts,
                int(self.degree), int(self.acceptance_degree), *common)
        else:
            result = _apply(
                local_points, local_receivers, local_looks, data, second_data,
                float(first_scale), float(second_scale), starts,
                int(self.degree), int(max_m), int(self.acceptance_degree), *common)
        carrier = np.exp(1j * self.omega_per_ns * source.reference_ns)
        return np.transpose(result, (1, 0, 2)) * carrier[:, None, None]
