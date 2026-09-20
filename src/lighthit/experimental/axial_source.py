"""The axial reduction of an event, and the response it implies.

An electron shower is a needle. Measured on the stored 100 GeV sample, the
photon-weighted longitudinal spread is 1.2 m against 0.065 m across, the
azimuth of the emission direction about the axis is uniform to a few per cent,
and the residual emission time -- what is left of the emission time after the
flight of the front is subtracted, ``tau = t - t0 - z/c0`` -- reaches only
0.26 ns at the 99th percentile. Those three facts are what make the reduction
below worth doing, and each of them is a measurable error rather than an
assumption.

The reduction is not a fit and not a statistical model of a shower. Put the
axis along ``z`` and write the kernel's angular factor with the addition
theorem in that frame:

    P_l(r_hat . u) = 4 pi / (2l + 1) sum_m Y_lm(r_hat) Y_lm(u).

Every quantity that belongs to the source then separates from every quantity
that belongs to the receiver, and the response of a source lying on the axis is
*exactly*

    N(omega) = sum_l 4 pi / (2l+1) sum_m int dz A_lm(z, omega)
               e^{i omega z / c0} M_l(r(z), omega) Y_lm(r_hat(z)),

    A_lm(z, omega) = sum_i Q_i P_l(mu_C,i) Y_lm(u_i) e^{i omega tau_i},

with ``r(z)`` the distance from the axis point to the receiver and all
directions expressed in the axis frame. Nothing is approximated by this step.

The compression is in where the two sums are cut. Keeping only ``m = 0`` is
exactly averaging over the azimuth of the emission direction about the axis,
which reduces the source to ``L+1`` channels instead of ``(L+1)^2``; keeping
``|m| <= M`` restores the leading azimuthal correlation at ``(2M+1)`` channels
per degree. A shower is azimuthally symmetric in distribution but not in any
one realisation, so ``M`` is a measurable choice rather than a matter of
principle -- see ``scripts/compare_event_bases.py``.

Two things follow that are the point of the whole construction. The source
needs a number of channels linear rather than quadratic in the angular degree.
And the longitudinal integral is carried out against the true kernel, so the
arrival phase ``e^{i omega z / c0}`` over the length of the shower is never
approximated by a polynomial -- which is exactly what limited the moment route
at nonzero frequency.

What is given up is stated where it is taken: the transverse offset of each
element, the azimuthal channels above ``M``, and the longitudinal binning.
Each is measured on its own rather than blamed collectively.
"""
from dataclasses import dataclass
from itertools import product
import numpy as np
from scipy.sparse import coo_matrix
from scipy.special import eval_legendre

from .event_moments import real_spherical_harmonics

__all__ = ["VACUUM_M_PER_NS", "AxisFrame", "AxialSource", "axial_response",
           "channel_index"]

VACUUM_M_PER_NS = 0.299792458


def channel_index(degree, azimuthal_degree):
    """The ``(l, m)`` pairs kept, and where each sits in the full harmonic list.

    Returns the flat positions inside the ``(degree + 1) ** 2`` ordering used by
    :func:`~lighthit.experimental.event_moments.real_spherical_harmonics`, and
    the degree of each kept channel.
    """
    keep, degrees = [], []
    for l in range(degree + 1):
        limit = min(l, int(azimuthal_degree))
        for m in range(-limit, limit + 1):
            keep.append(l * l + l + m)
            degrees.append(l)
    return np.array(keep, dtype=np.int64), np.array(degrees, dtype=np.int64)


@dataclass(frozen=True)
class AxisFrame:
    """Where the event's axis is, which way it points, and its transverse pair."""
    centre_m: np.ndarray
    axis: np.ndarray
    first: np.ndarray
    second: np.ndarray

    @classmethod
    def of(cls, elements):
        """The photon-weighted centre and principal axis of the emission points."""
        points = elements.midpoints_m
        weights = np.asarray(elements.photons, float)
        share = weights / weights.sum()
        centre = (share[:, None] * points).sum(axis=0)
        offset = points - centre
        covariance = np.einsum("i,ij,ik->jk", share, offset, offset)
        values, vectors = np.linalg.eigh(covariance)
        axis = vectors[:, int(np.argmax(values))]
        axis = axis / np.linalg.norm(axis)
        # The sign of an eigenvector is free. Point the axis the way the light
        # is going, so that z increases along the front.
        if float((share * (elements.direction @ axis)).sum()) < 0:
            axis = -axis
        helper = np.array([0.0, 0.0, 1.0])
        if abs(float(axis @ helper)) > 0.9:
            helper = np.array([1.0, 0.0, 0.0])
        first = np.cross(axis, helper)
        first /= np.linalg.norm(first)
        return cls(centre, axis, first, np.cross(axis, first))

    def rotate(self, vectors):
        """Return ``(first, second, axis)`` components, with the axis as z."""
        vectors = np.asarray(vectors, float)
        return np.stack((vectors @ self.first, vectors @ self.second,
                         vectors @ self.axis), axis=-1)

    def coordinates(self, elements, fraction=0.5):
        """``z`` along the axis, transverse offset, direction cosine, time.

        ``fraction`` picks the point along each chord, so that the projection
        can use the same quadrature as the element sum it is measured against.
        """
        points = (elements.start_m
                  + (fraction * elements.length_m)[:, None] * elements.direction)
        offset = points - self.centre_m
        z = offset @ self.axis
        transverse = offset - z[:, None] * self.axis
        cosine = elements.direction @ self.axis
        time = (np.asarray(elements.start_ns, float)
                + fraction * (np.asarray(elements.end_ns, float)
                              - np.asarray(elements.start_ns, float)))
        return z, transverse, cosine, time


@dataclass(frozen=True)
class AxialSource:
    """``A_lm(z, omega)`` on a longitudinal grid, with the frame that made it."""
    frame: AxisFrame
    z_m: np.ndarray                 # cell centres along the axis
    transverse_m: np.ndarray        # cell centres across it, in the frame pair
    channels: np.ndarray            # (cells, kept channels, frequency), complex
    kept: np.ndarray                # flat harmonic positions of those channels
    channel_degree: np.ndarray      # l of each kept channel
    degree: int
    azimuthal_degree: int
    reference_ns: float             # emission time the phases are measured from
    summary: dict

    def points_m(self):
        return (self.frame.centre_m[None, :]
                + self.z_m[:, None] * self.frame.axis[None, :]
                + self.transverse_m[:, 0:1] * self.frame.first[None, :]
                + self.transverse_m[:, 1:2] * self.frame.second[None, :])

    @classmethod
    def of(cls, elements, degree, omega_per_ns, *, azimuthal_degree=0,
           cell_m=None, bins=512, transverse_m=None, deposit="linear",
           element_order=2, frame=None, chunk=None,
           budget_bytes=192 * 2 ** 20):
        """Bin an event into cells of the axis frame and project the angles.

        The source is cut into boxes of the axis frame and each box keeps its
        own angular channels. Two sizes control it, and they are not the same
        physics:

        ``cell_m``
            the size along the axis. At zero frequency it only has to follow
            the bending of the kernel, which is slow; at frequency it has to
            keep the arrival phase inside a cell small, and the phase across a
            cell is ``omega * cell / v`` with ``v`` the group speed. A tenth of
            a radian at 0.6 rad/ns is about 4 cm.
        ``transverse_m``
            the size across the axis, defaulting to ``cell_m``. ``None`` with
            ``cell_m`` also ``None`` falls back to ``bins`` slabs along the
            axis and no transverse resolution at all -- the pure axial
            reduction, where every element is moved onto the axis. That is
            nearly free at zero frequency, where the move changes only an
            amplitude, and not free at all once ``omega * rho / v`` is a
            radian.

        Cells are sparse: only the occupied ones are kept, so a needle costs a
        needle's worth of cells however far the grid nominally reaches. Nothing
        is clipped -- the few elements far off the axis get their own cells
        rather than being folded into the outermost one, which is worth the
        handful of extra cells because a mislaid element carries a phase error
        of several radians.

        ``chunk`` defaults to whatever keeps one chunk's payload inside
        ``budget_bytes``; it changes the arithmetic order and nothing else.

        ``deposit`` is ``"linear"`` (cloud in cell) or ``"nearest"``. The linear
        rule spreads each element over the surrounding cell centres with weights
        that satisfy

            sum_j W_j = 1,      sum_j W_j X_j = x_i,

        so both the photon count and its first spatial moment survive the move
        onto the grid, and the leading error of a smooth kernel cancels.

        The emission phase is stored as ``exp(i omega (t_i - t_ref))`` with the
        element's own time, never rebuilt from the cell's position. That
        distinction is not cosmetic: giving a cell the front phase of its own
        centre while keeping the element's residual time silently moves the
        emission by ``(z_j - z_i) / c0``, which at 0.6 rad/ns is two radians per
        metre. The only thing the move approximates is the propagation.
        """
        if deposit not in ("linear", "nearest"):
            raise ValueError("deposit must be 'linear' or 'nearest'")
        frame = frame or AxisFrame.of(elements)
        omega = np.atleast_1d(np.asarray(omega_per_ns, float))
        kept, channel_degree = channel_index(degree, azimuthal_degree)
        photons = np.asarray(elements.photons, float)
        share = photons / photons.sum()
        # The same rule along the chord that the element sum uses, so that a
        # comparison between them measures the grid and not the quadrature.
        nodes, node_weights = np.polynomial.legendre.leggauss(max(1, int(element_order)))
        nodes, node_weights = (nodes + 1) / 2, node_weights / 2
        z, transverse, cosine, _ = frame.coordinates(elements)
        local_transverse = np.stack((transverse @ frame.first,
                                     transverse @ frame.second), axis=-1)

        if cell_m is None:
            along = (z.max() - z.min() + 1e-9) / int(max(1, bins))
        else:
            along = float(cell_m)
        across = along if transverse_m is None else float(transverse_m)
        flat = cell_m is None and transverse_m is None
        if flat:
            across = np.inf

        # Lattice coordinates, in units of the cell, with a node at every
        # integer. The origin sits one cell below the source so that rounding
        # alone never produces a negative index.
        spacing = np.array([along, 1.0 if flat else across, 1.0 if flat else across])
        origin = np.array([z.min() - along,
                           0.0 if flat else local_transverse[:, 0].min() - across,
                           0.0 if flat else local_transverse[:, 1].min() - across])
        axes = (0,) if flat else (0, 1, 2)
        steps = list(product(*[(0, 1) if a in axes else (0,) for a in range(3)]))

        # One pass over the chord rule to find the cells, a second to fill them.
        # The two have to see the same points, so the geometry is computed once
        # per node and kept.
        geometry = []
        for fraction, node_weight in zip(nodes, node_weights):
            zf, transverse_f, _, time_f = frame.coordinates(elements, fraction)
            position = np.stack((zf,
                                 transverse_f @ frame.first if not flat else np.zeros(len(zf)),
                                 transverse_f @ frame.second if not flat else np.zeros(len(zf))),
                                axis=-1)
            lattice = (position - origin) / spacing
            if deposit == "nearest":
                corners = [(np.rint(lattice).astype(np.int64),
                            np.full(len(zf), float(node_weight)))]
            else:
                base = np.floor(lattice).astype(np.int64)
                remainder = lattice - base
                corners = []
                for step in steps:
                    shift = np.array(step, dtype=np.int64)
                    weight = np.full(len(zf), float(node_weight))
                    for a in axes:
                        weight = weight * (remainder[:, a] if shift[a]
                                           else 1 - remainder[:, a])
                    corners.append((base + shift[None, :], weight))
            geometry.append((corners, time_f))

        keys = np.concatenate([key for corners, _ in geometry for key, _ in corners],
                              axis=0)
        unique, flat_inverse = np.unique(keys, axis=0, return_inverse=True)
        flat_inverse = flat_inverse.reshape(-1, len(z))
        node_centres = origin[None, :] + unique * spacing[None, :]
        centres = node_centres[:, 0]
        cell_offset = np.zeros((len(unique), 2)) if flat else node_centres[:, 1:]

        table = np.zeros((len(unique), len(kept), len(omega)), complex)
        reference = float(np.asarray(elements.start_ns).min())
        # The payload of one chunk is (elements x channels x frequencies) and
        # is by far the largest thing here: at 133 channels and 41 frequencies
        # a fixed chunk of 20000 elements asks for 1.7 GB. Size it instead.
        if chunk is None:
            per_element = max(len(kept) * len(omega) * 16, 1)
            chunk = int(np.clip(budget_bytes // per_element, 256, 20000))
        row = 0
        for corners, time_f in geometry:
            targets = flat_inverse[row:row + len(corners)]
            row += len(corners)
            for begin in range(0, len(elements), chunk):
                piece = slice(begin, begin + chunk)
                size = len(z[piece])
                local = frame.rotate(elements.direction[piece])
                harmonics = real_spherical_harmonics(degree, local, azimuthal_degree)
                cone = eval_legendre(channel_degree[None, :],
                                     elements.cone_cosine[piece][:, None])
                angular = cone * harmonics
                # The element's own emission time at this node, never rebuilt
                # from the cell: giving a cell the front phase of its own centre
                # while keeping the element's residual time moves the emission
                # by (z_cell - z_i) / c0, two radians per metre at 0.6 rad/ns.
                phase = np.exp(1j * omega[None, :]
                               * (time_f[piece] - reference)[:, None])
                payload = (angular[:, :, None] * phase[:, None, :]).reshape(size, -1)
                rows = np.concatenate([target[piece] for target in targets])
                columns = np.tile(np.arange(size), len(corners))
                data = np.concatenate([photons[piece] * weight[piece]
                                       for _, weight in corners])
                # One sparse product per chunk: np.add.at is unbuffered and
                # costs an order of magnitude more for the same sum. The
                # product is built only over the cells this chunk touches --
                # spreading it over every cell would allocate a second copy of
                # the whole table on every chunk, which for a wide frequency
                # grid is gigabytes of pure transient.
                touched, local_rows = np.unique(rows, return_inverse=True)
                spread = coo_matrix((data, (local_rows, columns)),
                                    shape=(len(touched), size)).tocsr()
                table[touched] += (spread @ payload).reshape(
                    len(touched), len(kept), len(omega))

        share = photons / photons.sum()
        radius = np.linalg.norm(transverse, axis=1)
        # Reported for what it says about the event, never used as a bound.
        tau = (np.asarray(elements.start_ns, float) - reference
               - (z - z.min()) / VACUUM_M_PER_NS)
        summary = {
            "elements": len(elements), "photons": float(photons.sum()),
            "degree": int(degree), "cells": int(len(centres)),
            "cell_m": float(along),
            "transverse_m": None if not np.isfinite(across) else float(across),
            "azimuthal_degree": int(azimuthal_degree), "channels": len(kept),
            "channels_if_full": (degree + 1) ** 2,

            "z_range_m": [float(z.min()), float(z.max())],
            "transverse_rms_m": float(np.sqrt((share * radius ** 2).sum())),
            "transverse_q99_m": float(_weighted_quantile(radius, share, 0.99)),
            "residual_time_rms_ns": float(np.sqrt((share * tau ** 2).sum()
                                                  - (share * tau).sum() ** 2)),
            "residual_time_q99_ns": float(_weighted_quantile(tau, share, 0.99)),
            "axis": frame.axis.tolist(), "centre_m": frame.centre_m.tolist(),
            "deposit": deposit, "element_order": int(element_order),
            "coefficients": int(table.size)}
        return cls(frame, centres, cell_offset, table, kept, channel_degree,
                   int(degree), int(azimuthal_degree), reference, summary)

    def truncated(self, azimuthal_degree):
        """The same source with fewer azimuthal channels, without reprojecting."""
        if azimuthal_degree > self.azimuthal_degree:
            raise ValueError("cannot add azimuthal channels after the projection")
        kept, channel_degree = channel_index(self.degree, azimuthal_degree)
        select = np.searchsorted(self.kept, kept)
        return AxialSource(self.frame, self.z_m, self.transverse_m,
                           self.channels[:, select, :], kept,
                           channel_degree, self.degree, int(azimuthal_degree),
                           self.reference_ns,
                           {**self.summary, "azimuthal_degree": int(azimuthal_degree),
                            "channels": len(kept),
                            "coefficients": int(self.channels[:, select, :].size)})


def _weighted_quantile(values, share, quantile):
    order = np.argsort(values)
    return float(np.interp(quantile, np.cumsum(share[order]), values[order]))


def axial_response(kernel, source, receivers_m, *, budget_bytes=128 * 2 ** 20):
    """Contract a compact source with the cached kernel. Shape ``(freq, receivers)``.

    The longitudinal and transverse integrals are sums over the cells against
    the true multipoles, so the phase across the source is carried by the
    kernel itself rather than by an expansion of it. Nothing is fitted.

    The work is ``receivers * cells * channels * frequencies``, and the largest
    intermediate would be the multipoles at every (receiver, cell, degree,
    frequency). That does not fit for a full array, so both the receivers and
    the frequencies are blocked to stay inside ``budget_bytes``; the blocking
    changes the arithmetic order and nothing else.
    """
    receivers = np.atleast_2d(np.asarray(receivers_m, float))
    omega = kernel.omega_per_ns
    if kernel.degree > source.degree:
        raise ValueError("the axial source carries fewer degrees than the kernel")
    keep = source.channel_degree <= kernel.degree
    # Selecting every channel would still copy the table, which for a wide
    # frequency grid is gigabytes duplicated for nothing.
    if keep.all():
        channels, kept, degrees = source.channels, source.kept, source.channel_degree
    else:
        channels = np.ascontiguousarray(source.channels[:, keep, :])
        kept, degrees = source.kept[keep], source.channel_degree[keep]
    weight = 4 * np.pi / (2 * degrees + 1)
    points = source.points_m()
    carrier = np.exp(1j * omega * source.reference_ns)
    total = np.zeros((len(omega), len(receivers)), complex)

    cells, orders = len(points), len(kept)
    # Two intermediates have to fit, not one: the harmonics at every
    # (receiver, cell, channel), which do not depend on frequency, and the
    # multipoles at every (receiver, cell, degree, frequency), which do.
    # Budgeting only the second is how this ran out of memory the first time.
    half = max(budget_bytes // 2, 1)
    receiver_chunk = int(np.clip(half // max(cells * orders * 8, 1), 1, len(receivers)))
    frequency_chunk = int(np.clip(
        half // max(receiver_chunk * cells * (kernel.degree + 1) * 16, 1),
        1, len(omega)))

    for begin in range(0, len(receivers), receiver_chunk):
        here = receivers[begin:begin + receiver_chunk]
        vectors = here[:, None, :] - points[None, :, :]
        radii = np.linalg.norm(vectors, axis=2)
        local = source.frame.rotate(vectors.reshape(-1, 3))
        harmonics = real_spherical_harmonics(kernel.degree, local,
                                             source.azimuthal_degree)
        harmonics = harmonics.reshape(len(here), cells, -1)[:, :, keep] * weight
        flat = radii.reshape(-1)
        for low in range(0, len(omega), frequency_chunk):
            window = slice(low, low + frequency_chunk)
            multipoles = kernel.multipoles(flat, frequency_slice=window)
            multipoles = multipoles.reshape(len(here), cells, kernel.degree + 1, -1)
            total[window, begin:begin + len(here)] = np.einsum(
                "rbcw,rbc,bcw->wr", multipoles[:, :, degrees, :], harmonics,
                channels[:, :, window], optimize=True)
    return total * carrier[:, None]
