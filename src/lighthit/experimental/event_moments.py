"""Joint moments of an event, contracted against a local kernel expansion.

The response of a detector to an event is a linear functional of the source,

    N_d(omega) = int d3x dt dOmega  q(x, s, t) e^{i omega t} K(R_d - x, s, omega),

so the only question is how to avoid walking every emitting element once per
receiver. This module does it by expanding the kernel over a spatial block and
moving the sum over elements inside:

    N_d(omega) ~ sum_B sum_{|a|<=p} sum_{lm} c^{(d,B)}_{a,lm}(omega)
                 M_{B,a,lm}(omega),

    M_{B,a,lm}(omega) = sum_i int da  (x_i(a) - x_B)^a y_i(a) e^{i omega t_i(a)}
                        P_l(mu_C,i) Y^R_{lm}(u_i).

The moments carry no receiver index: position, direction and time of one
element are multiplied *before* they are summed, so no marginal distribution is
formed and no correlation is averaged away. The Cherenkov cone is integrated
analytically by the addition theorem, which is why no optical photons are
generated.

Two things about the kernel matter here. First, for an isotropic point
receiver the cached multipoles already give every angular channel in closed
form: with ``K(r, s) = sum_l M_l(r) P_l(s.r_hat)`` and the real-harmonic
addition theorem,

    K_{lm}(r, omega) = 4 pi / (2l+1) M_l(r, omega) Y^R_{lm}(r_hat),

so the angular bandwidth of the *source* and the multipole index of the
*kernel* are the same index, and no separate solve is needed. Second, the
coefficients ``c`` are obtained by fitting the kernel over the block rather
than by differentiating an interpolated table: a least-squares fit on a
stencil has the same content as a Taylor expansion, moves the derivative of
the spline out of the calculation, and reuses one pseudo-inverse for every
receiver, frequency and channel.

Everything is truncated somewhere: the polynomial degree ``p``, the angular
degree ``L_q``, the number of blocks, and the cache's own settings. Each has
its own control, and :func:`direct_response` evaluates the same kernel element
by element so that the approximation error can be separated from all of them.
"""
from dataclasses import dataclass
from itertools import product
import numpy as np
from scipy.special import eval_legendre, sph_harm_y_all

from ..cache import BandedResponseCache, ResponseCache

__all__ = ["real_spherical_harmonics", "monomial_powers", "BlockPartition",
           "KernelChannels", "compile_joint_moments", "evaluate_moments",
           "direct_response"]


def real_spherical_harmonics(degree, vectors, azimuthal_degree=None):
    """Real orthonormal harmonics up to ``degree``, shape (points, (L+1)**2).

    The channel order is ``l = 0, 1, ...`` and, inside each ``l``,
    ``m = -l ... +l``.

    ``azimuthal_degree`` restricts the orders actually computed to
    ``|m| <= M``. The returned array then has one column per kept ``(l, m)``,
    in the same relative order, and :func:`channel_index` says which flat
    positions those are. Nothing is computed and discarded, which matters: at
    ``L = 32`` the full set is 1089 columns against 159 for ``M = 2``.
    """
    unit = np.asarray(vectors, float)
    unit = unit / np.linalg.norm(unit, axis=-1, keepdims=True)
    theta = np.arccos(np.clip(unit[..., 2], -1.0, 1.0))
    phi = np.arctan2(unit[..., 1], unit[..., 0])
    limit = degree if azimuthal_degree is None else min(int(azimuthal_degree), degree)
    # One call gives every order up to the limit; looping over them costs
    # seconds per chunk.
    table = sph_harm_y_all(degree, limit, theta, phi)
    orders = [(l, m) for l in range(degree + 1)
              for m in range(-min(l, limit), min(l, limit) + 1)]
    out = np.empty(unit.shape[:-1] + (len(orders),), float)
    for index, (l, m) in enumerate(orders):
        value = table[l, abs(m)]
        if m == 0:
            out[..., index] = value.real
        elif m > 0:
            out[..., index] = np.sqrt(2) * (-1) ** m * value.real
        else:
            out[..., index] = np.sqrt(2) * (-1) ** m * value.imag
    return out


def monomial_powers(degree):
    """Exponent triples of every monomial of total degree at most ``degree``."""
    powers = [(i, j, k) for total in range(degree + 1)
              for i, j, k in product(range(total + 1), repeat=3) if i + j + k == total]
    return np.array(powers, dtype=np.int64)


def _monomials(displacements, powers):
    d = np.asarray(displacements, float)
    return np.prod(d[:, None, :] ** powers[None, :, :], axis=2)


@dataclass(frozen=True)
class BlockPartition:
    """Spatial blocks of an event, with their centres and local half-sizes."""
    labels: np.ndarray
    centres_m: np.ndarray
    half_sizes_m: np.ndarray

    @property
    def count(self):
        return len(self.centres_m)

    @staticmethod
    def _reach(elements, inside, centre):
        """Half-sizes that contain the whole chord of every element in a block.

        Blocks are chosen from midpoints, but what is integrated is the chord,
        whose nodes lie up to half a length away from its midpoint. Bounding by
        the midpoints alone lets the fit be evaluated outside the box it was
        fitted on -- invisible for a 0.5 mm shower step and not at all invisible
        for a metre-long track segment.
        """
        start = elements.start_m[inside]
        finish = start + elements.length_m[inside][:, None] * elements.direction[inside]
        span = np.maximum(np.abs(start - centre), np.abs(finish - centre))
        return np.maximum(np.max(span, axis=0), 1e-6)

    @classmethod
    def single(cls, elements):
        centre = elements.centroid_m
        half = cls._reach(elements, np.ones(len(elements), bool), centre)
        return cls(np.zeros(len(elements), np.int64), centre[None, :], half[None, :])

    @classmethod
    def split(cls, elements, blocks=1, strategy="extent"):
        """Recursive cuts along the widest axis of a block.

        What has to shrink is the largest block, because the stencil, the
        polynomial and the phase swing all answer to the block's half-size.
        The two strategies differ in what they equalise, and the difference is
        not cosmetic for a shower:

        ``"photons"``
            cut at the photon-weighted median, splitting every existing block
            at each round. A shower is a bright core inside a sparse halo, so
            the median sits inside the core and one block keeps almost the
            whole halo: asking for 32 blocks can leave the largest of them
            nearly the size of the event.
        ``"extent"``
            repeatedly take the block with the largest half-diagonal and cut
            its widest axis at the midpoint of its bounding box. Each cut
            halves that axis of that block, so the largest half-size falls
            with the block count by construction. This is the default.

        Block centres are photon weighted under both strategies; only the
        partition differs.
        """
        if blocks < 1 or (blocks & (blocks - 1)):
            raise ValueError("blocks must be a power of two")
        if strategy not in ("extent", "photons"):
            raise ValueError("strategy must be 'extent' or 'photons'")
        points = elements.midpoints_m
        weights = np.asarray(elements.photons, float)
        labels = np.zeros(len(points), np.int64)

        def half_diagonal(inside):
            cloud = points[inside]
            return float(np.linalg.norm(cloud.max(axis=0) - cloud.min(axis=0)) / 2)

        if strategy == "photons":
            while labels.max() + 1 < blocks:
                for label in range(labels.max() + 1):
                    inside = np.flatnonzero(labels == label)
                    cloud = points[inside]
                    axis = int(np.argmax(cloud.max(axis=0) - cloud.min(axis=0)))
                    order = np.argsort(cloud[:, axis])
                    share = np.cumsum(weights[inside][order])
                    cut = inside[order][np.searchsorted(share, share[-1] / 2):]
                    labels[cut] = labels.max() + 1
        else:
            while labels.max() + 1 < blocks:
                groups = [np.flatnonzero(labels == label)
                          for label in range(labels.max() + 1)]
                sizes = [half_diagonal(inside) for inside in groups]
                inside = groups[int(np.argmax(sizes))]
                cloud = points[inside]
                low, high = cloud.min(axis=0), cloud.max(axis=0)
                axis = int(np.argmax(high - low))
                if high[axis] - low[axis] <= 0:
                    break           # a block of coincident points cannot shrink
                cut = inside[cloud[:, axis] > 0.5 * (low[axis] + high[axis])]
                if not len(cut) or len(cut) == len(inside):
                    break
                labels[cut] = labels.max() + 1

        centres, halves = [], []
        for label in range(labels.max() + 1):
            inside = labels == label
            weight = weights[inside] / weights[inside].sum()
            centre = (weight[:, None] * points[inside]).sum(axis=0)
            centres.append(centre)
            halves.append(cls._reach(elements, inside, centre))
        return cls(labels, np.array(centres), np.array(halves))

    def summary(self):
        return {"blocks": self.count,
                "centres_m": self.centres_m.tolist(),
                "half_sizes_m": self.half_sizes_m.tolist(),
                "elements_per_block": np.bincount(self.labels).tolist()}


@dataclass(frozen=True)
class KernelChannels:
    """Angular channels of the cached response, for one scattering order."""
    cache: ResponseCache
    order: int                      # 0 -> finite-L first order, 1 -> >=2
    degree: int                     # L_q, the retained angular degree
    frequency_index: np.ndarray = None

    @classmethod
    def of(cls, cache, order, degree, frequencies=None):
        band = cache.bands[0] if isinstance(cache, BandedResponseCache) else cache
        if degree > band.degree:
            raise ValueError("The cache does not store that many output degrees")
        if isinstance(cache, BandedResponseCache) and len(cache.bands) > 1:
            # Picking band 0 quietly would answer a query about the other bands'
            # radii with the wrong table. Refuse instead.
            raise ValueError("a multi-band cache is not dispatched here; pass one band")
        index = None
        if frequencies is not None:
            axis = band.grid.omega_per_ns
            index = np.array([int(np.argmin(np.abs(axis - value))) for value in frequencies])
            # rtol=0: a frequency near a cached one is not the cached one, and
            # the default relative tolerance would accept 0.600001 as 0.6.
            if not np.allclose(axis[index], frequencies, rtol=0.0, atol=1e-12):
                raise ValueError("Requested frequencies are not on the cached axis")
        return cls(band, order, degree, index)

    @property
    def omega_per_ns(self):
        axis = self.cache.grid.omega_per_ns
        return axis if self.frequency_index is None else axis[self.frequency_index]

    def multipoles(self, radii, frequency_slice=None):
        """``M_l(r, omega)`` for the retained degrees, shape (points, L+1, freq).

        ``frequency_slice`` narrows the frequency axis before anything large is
        built, which is what makes a full array affordable: the interpolation
        is per frequency anyway, and the caller rarely wants all of them at
        once.
        """
        index = self.frequency_index
        if index is None:
            index = np.arange(len(self.cache.grid.omega_per_ns))
        if frequency_slice is not None:
            index = index[frequency_slice]
        radii = np.asarray(radii, float)
        if frequency_slice is None and len(index) == len(self.cache.grid.omega_per_ns):
            block = self.cache.moments_at(radii, degrees=self.degree + 1)[..., self.order]
        else:
            # One frequency at a time: the interpolation is per frequency in any
            # case, and asking for all of them builds an array that a full array
            # of modules cannot hold.
            block = np.concatenate(
                [self.cache.moments_at(radii, degrees=self.degree + 1,
                                       frequency_index=int(one))[..., self.order]
                 for one in index], axis=0)
        return np.transpose(block, (1, 2, 0))

    def channels(self, displacements):
        """``K_{lm}(r, omega)`` at the given source-to-receiver vectors.

        Shape ``(points, (L+1)**2, freq)``.
        """
        vectors = np.atleast_2d(np.asarray(displacements, float))
        radii = np.linalg.norm(vectors, axis=1)
        multipoles = self.multipoles(radii)
        harmonics = real_spherical_harmonics(self.degree, vectors)
        ell = np.concatenate([[l] * (2 * l + 1) for l in range(self.degree + 1)])
        weight = 4 * np.pi / (2 * ell + 1)
        return multipoles[:, ell, :] * (harmonics * weight)[:, :, None]


def compile_joint_moments(elements, partition, kernel_degree, spatial_degree,
                          omega_per_ns, *, element_order=2, chunk=8192):
    """Build ``M_{B,a,lm}(omega)`` once, with no receiver in sight.

    The integral along each element is a Gauss rule in the chord parameter, so
    the position, the emission time and the cone angle of that element all
    enter the same product before anything is summed. ``element_order=1`` is
    the midpoint rule; the default 2 is exact for the linear time law times a
    quadratic monomial and is checked by raising it.
    """
    powers = monomial_powers(spatial_degree)
    omega = np.atleast_1d(np.asarray(omega_per_ns, float))
    # order 1 really is the midpoint rule; it used to be silently promoted to 2.
    nodes, weights = np.polynomial.legendre.leggauss(max(1, int(element_order)))
    nodes, weights = (nodes + 1) / 2, weights / 2
    channels = (kernel_degree + 1) ** 2
    # Built as (block, monomial, frequency, channel) so that one matrix product
    # fills it; returned in the documented (block, monomial, channel, frequency).
    moments = np.zeros((partition.count, len(powers), len(omega), channels), complex)
    ell = np.concatenate([[l] * (2 * l + 1) for l in range(kernel_degree + 1)])
    flat = moments.reshape(partition.count, len(powers) * len(omega), channels)

    for begin in range(0, len(elements), chunk):
        piece = slice(begin, begin + chunk)
        start = elements.start_m[piece]
        direction = elements.direction[piece]
        length = elements.length_m[piece]
        photons = elements.photons[piece]
        labels = partition.labels[piece]
        cone = elements.cone_cosine[piece]
        harmonics = real_spherical_harmonics(kernel_degree, direction)
        angular = harmonics * eval_legendre(ell[None, :], cone[:, None])
        for node, weight in zip(nodes, weights):
            point = start + (node * length)[:, None] * direction
            time = elements.start_ns[piece] + node * (elements.end_ns[piece]
                                                      - elements.start_ns[piece])
            phase = np.exp(1j * omega[None, :] * time[:, None])
            for label in np.unique(labels):
                inside = labels == label
                local = _monomials(point[inside] - partition.centres_m[label], powers)
                carrier = (weight * photons[inside])[:, None]
                # (elements, monomials x frequencies), then one matrix product
                # against the angular channels. Two real products cost half of
                # the complex one, and this is the whole cost of the build.
                mixed = ((local * carrier)[:, :, None] * phase[inside][:, None, :])
                mixed = mixed.reshape(len(local), -1)
                block = angular[inside]
                flat[label] += (mixed.real.T @ block) + 1j * (mixed.imag.T @ block)
    return np.ascontiguousarray(np.transpose(moments, (0, 1, 3, 2))), powers


def evaluate_moments(kernel, moments, powers, partition, receivers_m, *,
                     stencil_order=None, margin=1.0, receiver_chunk=None,
                     budget_bytes=32 * 2 ** 20):
    """Contract the moments with a local polynomial fit of the kernel.

    For each block the kernel channels are evaluated on a small stencil that
    covers the block, and one pseudo-inverse turns those values into the
    polynomial coefficients the contraction needs. The result has shape
    ``(freq, receivers)``.

    The receiver never enters the pseudo-inverse or the moments, only the
    stencil values, so the two receiver-independent factors are contracted
    first::

        N(w, R) = sum_{s,c} K(R, s, c, w) * G(s, c, w),
        G(s, c, w) = sum_a inverse[a, s] * M[a, c, w].

    That leaves one matrix product per block over receivers and stencil
    points together, instead of a pseudo-inverse per receiver. Receivers are
    processed in chunks so that the stencil values stay inside
    ``budget_bytes``; the chunk changes nothing but peak memory.
    """
    receivers = np.atleast_2d(np.asarray(receivers_m, float))
    frequencies = len(kernel.omega_per_ns)
    total = np.zeros((frequencies, len(receivers)), complex)
    # The stencil must be able to support the polynomial: a 3x3x3 grid carries
    # 27 samples, which is already tight for the 20 monomials of degree three.
    if stencil_order is None:
        stencil_order = max(3, int(powers.sum(axis=1).max()) + 2)
    grid = np.linspace(-1.0, 1.0, int(stencil_order))
    offsets = np.array(list(product(grid, grid, grid)))
    channels = (kernel.degree + 1) ** 2
    # The design matrix is built on the dimensionless offsets, never on metres.
    # A block with half-sizes spanning orders of magnitude -- a thin needle is
    # exactly that -- otherwise loses rank to the scaling alone: at (1e-6, 1, 1)
    # metres the cubic design drops from rank 20 to 19.
    design = _monomials(offsets, powers)
    rank = np.linalg.matrix_rank(design)
    if rank < len(powers):
        # {-1, 0, 1} is the standard trap: x**3 == x there, so 27 samples carry
        # rank 17 against the 20 cubic monomials, not 20.
        raise ValueError(
            f"a {stencil_order}-node stencil supports only {rank} of the "
            f"{len(powers)} monomials of degree {int(powers.sum(axis=1).max())}; "
            "raise stencil_order")
    inverse = np.linalg.pinv(design)
    if receiver_chunk is None:
        per_receiver = max(len(offsets) * channels * frequencies * 16, 1)
        receiver_chunk = max(1, int(budget_bytes // per_receiver))
    for label in range(partition.count):
        half = margin * partition.half_sizes_m[label]
        local = offsets * half[None, :]
        # Coefficients come out in the dimensionless basis, so the physical
        # moments are scaled to match rather than the matrix being unscaled.
        scale = np.prod(half[None, :] ** powers, axis=1)
        # (stencil, channels, frequency): the whole receiver-independent part.
        weights = np.einsum("as,acw->scw", inverse,
                            moments[label] / scale[:, None, None], optimize=True)
        base = partition.centres_m[label][None, :] + local
        for lo in range(0, len(receivers), receiver_chunk):
            here = receivers[lo:lo + receiver_chunk]
            vectors = (here[:, None, :] - base[None, :, :]).reshape(-1, 3)
            values = kernel.channels(vectors).reshape(len(here), len(base),
                                                      channels, frequencies)
            total[:, lo:lo + len(here)] += np.einsum("rscw,scw->wr", values,
                                                     weights, optimize=True)
    return total


def direct_response(kernel, elements, receivers_m, *, chunk=None, element_order=2,
                    budget_bytes=96 * 2 ** 20):
    """The same kernel, summed element by element. Shape ``(freq, receivers)``.

    This is the reference the compact routes are measured against: identical
    medium, identical cache, identical angular truncation, no expansion.

    ``chunk`` defaults to whatever keeps the multipoles of one chunk inside
    ``budget_bytes``. A fixed chunk is what a wide frequency grid cannot
    afford: 20000 elements at 33 degrees and 81 frequencies is 855 MB for that
    one array.
    """
    receivers = np.atleast_2d(np.asarray(receivers_m, float))
    omega = kernel.omega_per_ns
    if chunk is None:
        per_element = max((kernel.degree + 1) * len(omega) * 16, 1)
        chunk = int(np.clip(budget_bytes // per_element, 256, 20000))
    out = np.zeros((len(omega), len(receivers)), complex)
    nodes, weights = np.polynomial.legendre.leggauss(max(1, int(element_order)))
    nodes, weights = (nodes + 1) / 2, weights / 2
    degrees = np.arange(kernel.degree + 1)
    for index, receiver in enumerate(receivers):
        for begin in range(0, len(elements), chunk):
            piece = slice(begin, begin + chunk)
            start = elements.start_m[piece]
            direction = elements.direction[piece]
            length = elements.length_m[piece]
            photons = elements.photons[piece]
            cone = eval_legendre(degrees[None, :], elements.cone_cosine[piece][:, None])
            for node, weight in zip(nodes, weights):
                point = start + (node * length)[:, None] * direction
                vectors = receiver[None, :] - point
                radii = np.linalg.norm(vectors, axis=1)
                cosine = np.einsum("ij,ij->i", vectors, direction) / radii
                angular = cone * eval_legendre(degrees[None, :], cosine[:, None])
                multipoles = kernel.multipoles(radii)          # (points, L+1, freq)
                time = elements.start_ns[piece] + node * (elements.end_ns[piece]
                                                          - elements.start_ns[piece])
                phase = np.exp(1j * omega[None, :] * time[:, None])
                carrier = weight * photons
                out[:, index] += np.einsum("i,il,ilw,iw->w", carrier, angular,
                                           multipoles, phase, optimize=True)
    return out
