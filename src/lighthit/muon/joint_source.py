"""Sparse joint numerical source, with exact spectral Cherenkov moments.

The bins label axial position z, *charged-particle* cosine mu, and residual
emission clock tau=t-reference-z/c. Photon angles are not sampled: each step
contributes P_l(1/(beta*n_lambda)) Y_lm(u) analytically, including its own beta
threshold. Thus angular binning never substitutes a bin-centre direction.
Nonzero azimuthal moments and physical transverse centroids are retained.

Moving points and clocks to cell centroids is the approximation of this
source; its z, tau and transverse steps are set by the route and were checked
on the OM response, not only on charge conservation.
"""
from dataclasses import dataclass
import numpy as np
from numba import njit, prange

from ..axial_source import AxisFrame, VACUUM_M_PER_NS
from ..axial_fast import _layout, _harmonics_into


@njit(cache=True, nogil=True, parallel=True)
def _charged_harmonics(vectors, starts, aa, bb, max_m):
    """Wavelength-independent charged harmonics, computed only once."""
    out = np.empty((len(vectors), starts[-1]))
    for row in prange(len(vectors)):
        _harmonics_into(vectors[row, 0], vectors[row, 1], vectors[row, 2],
                        len(starts)-2, max_m, starts, aa, bb, out[row])
    return out


@njit(cache=True, nogil=True, parallel=True)
def _accumulate_spectral_cones(harmonics, cones, physical, inverse, starts, angular, photons):
    """Same ordered sums as row-major scatter, without scalar NumPy indexing.

    Wavelength bands are independent. The order of additions within every
    channel is the original quadrature order; neither rows nor cell sums are
    reordered. No angular, beta or time approximation is introduced.
    """
    for band in prange(len(cones)):
        for row in range(len(inverse)):
            target = inverse[row]; weight = physical[band, row]
            photons[band, target] += weight
            if weight == 0.: continue
            prev = 1.; current = cones[band, row]
            for ell in range(len(starts)-1):
                if ell == 0: value = 1.
                elif ell == 1: value = current
                else:
                    value = ((2*ell-1)*cones[band, row]*current-(ell-1)*prev)/ell
                    prev = current; current = value
                for channel in range(starts[ell], starts[ell+1]):
                    angular[band, target, channel] += (harmonics[row, channel]*value)*weight


@dataclass(frozen=True)
class JointSource:
    """Occupied (z, mu, tau, x, y) cells with their centroid, clock and angular moments per band."""
    frame: AxisFrame
    keys: np.ndarray               # z, charged mu, tau, transverse x, y
    points_m: np.ndarray           # shared, positive C0-weighted centroid
    times_ns: np.ndarray           # same centroid weights as points
    angular: np.ndarray            # wavelength, occupied cell, harmonic
    photons_per_nm: np.ndarray     # wavelength, occupied cell
    local_covariance_m2: np.ndarray
    residual_variance_ns2: np.ndarray
    coefficient0_weight: np.ndarray
    wavelength_nm: np.ndarray
    phase_index: np.ndarray
    kept: np.ndarray
    degree: int
    azimuthal_degree: int
    reference_ns: float
    summary: dict

    def spectrum(self, omega_per_ns):
        """Angular source spectrum, for source-only convergence comparisons."""
        omega = np.atleast_1d(np.asarray(omega_per_ns, float))
        if omega.ndim != 1 or not np.isfinite(omega).all():
            raise ValueError('finite one-dimensional frequencies required')
        return np.einsum('lpc,pw->lcw', self.angular,
                         np.exp(1j * (self.times_ns-self.reference_ns)[:, None]
                                * omega[None, :]))


@dataclass(frozen=True)
class PhotonHistogram:
    """Positive photon pencils, with geometry shared across residual clocks.

    A single wavelength avoids merging physically different spectral cones.
    Directions and points are positive photon centroids of each geometry bin.
    That move is an approximation, independent of transport quadrature.
    """
    frame: AxisFrame
    position: np.ndarray
    direction: np.ndarray
    geometry_id: np.ndarray
    emission_times_ns: np.ndarray
    photon_weights: np.ndarray
    wavelength_nm: np.ndarray
    geometry_keys: np.ndarray
    clock_keys: np.ndarray
    summary: dict
    original_position: np.ndarray | None = None
    original_direction: np.ndarray | None = None
    original_times_ns: np.ndarray | None = None
    original_photons_per_nm: np.ndarray | None = None
    original_geometry_id: np.ndarray | None = None
    geometry_position_covariance_m2: np.ndarray | None = None
    geometry_direction_covariance: np.ndarray | None = None
    geometry_position_direction_covariance_m: np.ndarray | None = None
    geometry_position_rms_m: np.ndarray | None = None
    geometry_angular_rms_upper_rad: np.ndarray | None = None
    geometry_sample_angular_max_rad: np.ndarray | None = None
    geometry_sample_position_max_m: np.ndarray | None = None
    geometry_direction_mean: np.ndarray | None = None


def _compile_photon_histogram_streaming(source, wavelength_nm, phase_index, *, frame=None,
                             reference_ns=None, z_step_m=.5, mu_step=.05,
                             phi_step_rad=.1, tau_step_ns=.1,
                             transverse_step_m=.05, ring_nodes=32,
                             chord_order=2, chunk=256, retain_samples=False):
    """Positive (z, photon mu, photon phi, tau) histogram at one wavelength.

    Actual Cherenkov rings are sampled with a deterministic phase offset per
    charged step. Each row keeps its beta and threshold; no mean cone is used.
    Nonzero azimuth and transverse coordinates are explicitly binned. Geometry
    keys omit tau so a first-scattering CDF is computed once per geometry and
    reused for all its emission-time bins. There is no universal-axisymmetric
    or separable angle/time assumption. Ring and bin convergence are required.
    Streaming accumulation limits temporary ring arrays to ``chunk`` steps;
    the output still scales with the number of occupied cells.
    """
    wavelength = _positive(wavelength_nm, 'wavelength_nm')
    index = _positive(phase_index, 'phase_index')
    z_step = _positive(z_step_m, 'z_step_m')
    mu_step = _positive(mu_step, 'mu_step')
    phi_step = _positive(phi_step_rad, 'phi_step_rad')
    tau_step = _positive(tau_step_ns, 'tau_step_ns')
    transverse = _positive(transverse_step_m, 'transverse_step_m', True)
    for value, name in [(ring_nodes, 'ring_nodes'), (chord_order, 'chord_order'), (chunk, 'chunk')]:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    physical = np.where(source.beta*index > 1, np.maximum(0., source.coefficient0
                        + source.coefficient2/index**2), 0.)/wavelength**2
    rows = np.flatnonzero(physical > 0)
    reference = float(source.start_ns.min() if reference_ns is None else reference_ns)
    if not np.isfinite(reference): raise ValueError('reference_ns must be finite')
    if frame is None:
        if not len(rows): raise ValueError('explicit frame required for a dark source')
        frame = AxisFrame.of(source.subset(physical > 0).field(0, phase_index=index))
    basis = np.stack((frame.first, frame.second, frame.axis))
    if not np.isfinite(basis).all() or not np.allclose(basis @ basis.T, np.eye(3), atol=1e-12) or not np.isclose(np.linalg.det(basis), 1., atol=1e-12):
        raise ValueError('frame must be right-handed and orthonormal')
    nodes, weights = np.polynomial.legendre.leggauss(chord_order)
    nodes, weights = (nodes+1)/2, weights/2
    phi_count = int(np.ceil(2*np.pi/phi_step)); phi_width = 2*np.pi/phi_count
    clock_cells = {}; original = []
    def position_anchor(keys):
        """Cell anchor position from integer keys."""
        across = transverse if np.isfinite(transverse) else 0.
        local_anchor = np.column_stack((keys[:, 3]*across, keys[:, 4]*across,
                                       keys[:, 0]*z_step))
        return frame.centre_m+local_anchor @ basis
    def clock_anchor(keys):
        """Cell anchor clock from integer keys."""
        return reference+keys[:, 0]*z_step/VACUUM_M_PER_NS+keys[:, 5]*tau_step
    # Each accumulated value is mass, weighted position, direction, time,
    # squared radius and squared time. These track centroid moves explicitly.
    for lo in range(0, len(rows), chunk):
        take = rows[lo:lo+chunk]; u = np.asarray(source.direction[take])
        helper = np.tile([0., 0., 1.], (len(take), 1)); helper[np.abs(u[:, 2]) >= .9] = [1., 0., 0.]
        first = np.cross(u, helper); first /= np.linalg.norm(first, axis=1)[:, None]
        second = np.cross(u, first)
        angle = 2*np.pi*(np.arange(ring_nodes)[None, :]
                        + ((take*.6180339887498949) % 1)[:, None])/ring_nodes
        cone = 1/(source.beta[take]*index); sine = np.sqrt(np.maximum(0., 1-cone**2))
        directions = (cone[:, None, None]*u[:, None, :] + sine[:, None, None]
                      *(np.cos(angle)[:, :, None]*first[:, None, :]
                        + np.sin(angle)[:, :, None]*second[:, None, :]))
        local_dirs = frame.rotate(directions)
        for fraction, weight in zip(nodes, weights):
            point = source.start_m[take]+(fraction*source.length_m[take])[:, None]*u
            local = frame.rotate(point-frame.centre_m)
            time = source.start_ns[take]+fraction*(source.end_ns[take]-source.start_ns[take])
            tau = time-reference-local[:, 2]/VACUUM_M_PER_NS
            key = np.empty((len(take), ring_nodes, 6), np.int64)
            key[:, :, 0] = np.floor(local[:, 2]/z_step+.5).astype(np.int64)[:, None]
            key[:, :, 1] = np.floor(local_dirs[:, :, 2]/mu_step+.5).astype(np.int64)
            key[:, :, 2] = np.minimum(phi_count-1, np.floor((np.arctan2(local_dirs[:, :, 1], local_dirs[:, :, 0])+np.pi)/phi_width)).astype(np.int64)
            key[:, :, 3] = np.floor(local[:, 0]/transverse+.5).astype(np.int64)[:, None]
            key[:, :, 4] = np.floor(local[:, 1]/transverse+.5).astype(np.int64)[:, None]
            key[:, :, 5] = np.floor(tau/tau_step+.5).astype(np.int64)[:, None]
            unique, inverse = np.unique(key.reshape(-1, 6), axis=0, return_inverse=True)
            mass = np.repeat(physical[take]*weight/ring_nodes, ring_nodes)
            world = np.repeat(point, ring_nodes, axis=0); direction = directions.reshape(-1, 3)
            clock = np.repeat(time, ring_nodes)
            if retain_samples:
                original.append((world.copy(), direction.copy(), clock.copy(), mass.copy()))
            # Sum offsets from the bin anchor to keep variances stable even
            # for kilometre positions and very late/global shifted clocks.
            node_keys = key.reshape(-1, 6)
            relative_point = world-position_anchor(node_keys)
            relative_clock = clock-clock_anchor(node_keys)
            values = np.column_stack((mass, mass[:, None]*relative_point, mass[:, None]*direction,
                                      mass*relative_clock, mass*np.sum(relative_point**2, axis=1), mass*relative_clock**2))
            grouped = np.zeros((len(unique), 10)); np.add.at(grouped, inverse, values)
            for label, value in zip(unique, grouped):
                item = tuple(label)
                if item in clock_cells: clock_cells[item] += value
                else: clock_cells[item] = value
    keys = np.asarray(sorted(clock_cells), np.int64).reshape(-1, 6)
    values = np.asarray([clock_cells[tuple(label)] for label in keys], float).reshape(-1, 10)
    geometry_keys, geometry_id = np.unique(keys[:, :5], axis=0, return_inverse=True)
    geometry = np.zeros((len(geometry_keys), 10)); np.add.at(geometry, geometry_id, values)
    if len(geometry):
        relative_position = geometry[:, 1:4]/geometry[:, :1]
        position = relative_position+position_anchor(geometry_keys)
        raw_direction = geometry[:, 4:7]/geometry[:, :1]
        norm = np.linalg.norm(raw_direction, axis=1)
        if np.any(norm < 1e-12): raise ValueError('angular bins cancel; refine mu/phi bins')
        direction = raw_direction/norm[:, None]
        spatial_var = np.maximum(0., geometry[:, 8]/geometry[:, 0]-np.sum(relative_position**2, axis=1))
        angle_spread = np.arccos(np.clip(norm, 0., 1.))
    else:
        position = direction = np.empty((0, 3)); spatial_var = angle_spread = np.empty(0)
    relative_clocks = values[:, 7]/values[:, 0] if len(values) else np.empty(0)
    clocks = relative_clocks+clock_anchor(keys)
    clock_var = np.maximum(0., values[:, 9]/values[:, 0]-relative_clocks**2) if len(values) else np.empty(0)
    summary = dict(source_rows=int(len(source)), emitting_rows=int(len(rows)),
                   deposited_ring_nodes=int(len(rows)*chord_order*ring_nodes),
                   geometry_cells=int(len(geometry_keys)), joint_clock_cells=int(len(keys)),
                   clock_reuse_ratio=float(len(keys)/max(1, len(geometry_keys))),
                   ring_nodes=int(ring_nodes), chord_order=int(chord_order),
                   max_spatial_rms_m=float(np.sqrt(spatial_var.max())) if len(spatial_var) else 0.,
                   max_direction_spread_rad=float(angle_spread.max()) if len(angle_spread) else 0.,
                   max_clock_rms_ns=float(np.sqrt(clock_var.max())) if len(clock_var) else 0.,
                   photons_per_nm=float(values[:, 0].sum()),
                   source_approximation='positive geometry centroids shared across time bins',
                   azimuth_symmetry_assumed=False, photon_mu_bins=True,
                   z_step_m=z_step, photon_mu_step=mu_step, photon_phi_step_rad=phi_width,
                   tau_step_ns=tau_step, transverse_step_m=transverse)
    return PhotonHistogram(frame, position, direction, geometry_id, clocks,
                           values[:, 0][None, :], np.array([wavelength]), geometry_keys, keys, summary,
                           *([np.concatenate([part[i] for part in original]) for i in range(4)]
                             if original else [None]*4))


def compile_photon_histogram(source, wavelength_nm, phase_index, *, frame=None,
                             reference_ns=None, z_step_m=.5, mu_step=.05,
                             phi_step_rad=.1, tau_step_ns=.1,
                             transverse_step_m=.05, ring_nodes=32,
                             chord_order=2, chunk=256, retain_samples=False):
    """Positive joint photon histogram with bounded vectorised raw fallback.

    ``retain_samples=True`` uses vectorised unique/bincount accumulation and
    exposes raw geometry memberships and covariance tensors. At most two
    million photon quadrature samples are admitted in one such call; split
    larger families into source chunks and sum their responses. Nothing is
    clipped. The streaming path remains available without retained samples.

    Direction covariance is around the *unnormalised* mean E[u]. The angular
    RMS upper bound follows theta <= pi/2*|u-unit(E[u])|, and is valid for the
    sampled source. Maximum bounds refer to these sampled ring/chord nodes;
    ring/chord convergence is still a separate requirement.
    """
    if not retain_samples:
        return _compile_photon_histogram_streaming(source, wavelength_nm, phase_index,
            frame=frame, reference_ns=reference_ns, z_step_m=z_step_m, mu_step=mu_step,
            phi_step_rad=phi_step_rad, tau_step_ns=tau_step_ns, transverse_step_m=transverse_step_m,
            ring_nodes=ring_nodes, chord_order=chord_order, chunk=chunk, retain_samples=False)
    wavelength = _positive(wavelength_nm, 'wavelength_nm'); index = _positive(phase_index, 'phase_index')
    z_step = _positive(z_step_m, 'z_step_m'); mu_step = _positive(mu_step, 'mu_step')
    phi_step = _positive(phi_step_rad, 'phi_step_rad'); tau_step = _positive(tau_step_ns, 'tau_step_ns')
    transverse = _positive(transverse_step_m, 'transverse_step_m', True)
    for value, name in [(ring_nodes, 'ring_nodes'), (chord_order, 'chord_order'), (chunk, 'chunk')]:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    physical = np.where(source.beta*index > 1, np.maximum(0., source.coefficient0
                        + source.coefficient2/index**2), 0.)/wavelength**2
    rows = np.flatnonzero(physical > 0)
    count = len(rows)*ring_nodes*chord_order
    if count > 2_000_000:
        raise ValueError('retained photon histogram exceeds two million samples; chunk the source, without discarding tails')
    reference = float(source.start_ns.min() if reference_ns is None else reference_ns)
    if not np.isfinite(reference): raise ValueError('reference_ns must be finite')
    if frame is None:
        if not len(rows): raise ValueError('explicit frame required for a dark source')
        frame = AxisFrame.of(source.subset(physical > 0).field(0, phase_index=index))
    basis = np.stack((frame.first, frame.second, frame.axis))
    if not np.isfinite(basis).all() or not np.allclose(basis @ basis.T, np.eye(3), atol=1e-12) or not np.isclose(np.linalg.det(basis), 1., atol=1e-12):
        raise ValueError('frame must be right-handed and orthonormal')
    nodes, quadrature = np.polynomial.legendre.leggauss(chord_order)
    nodes, quadrature = (nodes+1)/2, quadrature/2
    phi_count = int(np.ceil(2*np.pi/phi_step)); phi_width = 2*np.pi/phi_count
    original_position = np.empty((count, 3)); original_direction = np.empty((count, 3))
    original_time = np.empty(count); original_mass = np.empty(count)
    keys = np.empty((count, 6), np.int64); offset = 0
    for lo in range(0, len(rows), chunk):
        take = rows[lo:lo+chunk]; u = source.direction[take]
        helper = np.tile([0., 0., 1.], (len(take), 1)); helper[np.abs(u[:, 2]) >= .9] = [1., 0., 0.]
        first = np.cross(u, helper); first /= np.linalg.norm(first, axis=1)[:, None]
        second = np.cross(u, first)
        angle = 2*np.pi*(np.arange(ring_nodes)[None, :]+((take*.6180339887498949) % 1)[:, None])/ring_nodes
        cone = 1/(source.beta[take]*index); sine = np.sqrt(np.maximum(0., 1-cone**2))
        directions = (cone[:, None, None]*u[:, None, :]+sine[:, None, None]
                      *(np.cos(angle)[:, :, None]*first[:, None, :]+np.sin(angle)[:, :, None]*second[:, None, :]))
        local_dirs = frame.rotate(directions)
        for fraction, weight in zip(nodes, quadrature):
            stop = offset+len(take)*ring_nodes
            point = source.start_m[take]+(fraction*source.length_m[take])[:, None]*u
            local = frame.rotate(point-frame.centre_m)
            time = source.start_ns[take]+fraction*(source.end_ns[take]-source.start_ns[take])
            tau = time-reference-local[:, 2]/VACUUM_M_PER_NS
            label = keys[offset:stop].reshape(len(take), ring_nodes, 6)
            label[:, :, 0] = np.floor(local[:, 2]/z_step+.5).astype(np.int64)[:, None]
            label[:, :, 1] = np.floor(local_dirs[:, :, 2]/mu_step+.5).astype(np.int64)
            label[:, :, 2] = np.minimum(phi_count-1, np.floor((np.arctan2(local_dirs[:, :, 1], local_dirs[:, :, 0])+np.pi)/phi_width)).astype(np.int64)
            label[:, :, 3] = np.floor(local[:, 0]/transverse+.5).astype(np.int64)[:, None]
            label[:, :, 4] = np.floor(local[:, 1]/transverse+.5).astype(np.int64)[:, None]
            label[:, :, 5] = np.floor(tau/tau_step+.5).astype(np.int64)[:, None]
            original_position[offset:stop] = np.repeat(point, ring_nodes, axis=0)
            original_direction[offset:stop] = directions.reshape(-1, 3)
            original_time[offset:stop] = np.repeat(time, ring_nodes)
            original_mass[offset:stop] = np.repeat(physical[take]*weight/ring_nodes, ring_nodes)
            offset = stop
    clock_keys, clock_inverse = _unique_integer_rows(keys)
    del keys
    geometry_keys, geometry_id = _unique_integer_rows(clock_keys[:, :5])
    original_geometry_id = geometry_id[clock_inverse]
    ng = len(geometry_keys); nc = len(clock_keys)
    geom_mass = np.bincount(original_geometry_id, weights=original_mass, minlength=ng)
    clock_mass = np.bincount(clock_inverse, weights=original_mass, minlength=nc)
    across = transverse if np.isfinite(transverse) else 0.
    anchor = frame.centre_m+np.column_stack((geometry_keys[:, 3]*across, geometry_keys[:, 4]*across,
                                            geometry_keys[:, 0]*z_step)) @ basis
    relative_position = original_position-anchor[original_geometry_id]
    position_mean = np.zeros((ng, 3)); raw_mean = np.zeros((ng, 3))
    for component in range(3):
        position_mean[:, component] = np.bincount(original_geometry_id, weights=original_mass*relative_position[:, component], minlength=ng)/geom_mass
        raw_mean[:, component] = np.bincount(original_geometry_id, weights=original_mass*original_direction[:, component], minlength=ng)/geom_mass
    position = anchor+position_mean; norm = np.linalg.norm(raw_mean, axis=1)
    if np.any(norm < 1e-12): raise ValueError('angular bins cancel; refine mu/phi bins')
    direction = raw_mean/norm[:, None]
    clock_anchor = reference+clock_keys[:, 0]*z_step/VACUUM_M_PER_NS+clock_keys[:, 5]*tau_step
    relative_clock = original_time-clock_anchor[clock_inverse]
    clock_mean = np.bincount(clock_inverse, weights=original_mass*relative_clock, minlength=nc)/clock_mass
    clocks = clock_anchor+clock_mean
    clock_delta = relative_clock-clock_mean[clock_inverse]
    clock_variance = np.bincount(clock_inverse, weights=original_mass*clock_delta**2, minlength=nc)/clock_mass
    position_delta = relative_position-position_mean[original_geometry_id]
    direction_delta = original_direction-raw_mean[original_geometry_id]
    pcov = np.zeros((ng, 3, 3)); dcov = np.zeros_like(pcov); crosscov = np.zeros_like(pcov)
    for first in range(3):
        for second in range(3):
            pcov[:, first, second] = np.bincount(original_geometry_id, weights=original_mass*position_delta[:, first]*position_delta[:, second], minlength=ng)/geom_mass
            dcov[:, first, second] = np.bincount(original_geometry_id, weights=original_mass*direction_delta[:, first]*direction_delta[:, second], minlength=ng)/geom_mass
            crosscov[:, first, second] = np.bincount(original_geometry_id, weights=original_mass*position_delta[:, first]*direction_delta[:, second], minlength=ng)/geom_mass
    position_rms = np.sqrt(np.maximum(0., np.trace(pcov, axis1=1, axis2=2)))
    angular_rms_upper = np.pi/2*np.sqrt(np.maximum(0., 2*(1-np.minimum(norm, 1.))))
    angular_max = np.zeros(ng); spatial_max = np.zeros(ng)
    np.maximum.at(angular_max, original_geometry_id, np.arccos(np.clip(np.sum(original_direction*direction[original_geometry_id], axis=1), -1., 1.)))
    np.maximum.at(spatial_max, original_geometry_id, np.linalg.norm(position_delta, axis=1))
    summary = dict(source_rows=int(len(source)), emitting_rows=int(len(rows)), deposited_ring_nodes=count,
        geometry_cells=ng, joint_clock_cells=nc, clock_reuse_ratio=float(nc/max(1, ng)),
        ring_nodes=int(ring_nodes), chord_order=int(chord_order), photons_per_nm=float(original_mass.sum()),
        max_spatial_rms_m=float(position_rms.max()) if ng else 0.,
        max_direction_spread_rad=float(np.arccos(np.clip(norm, 0., 1.)).max()) if ng else 0.,
        max_clock_rms_ns=float(np.sqrt(clock_variance.max())) if nc else 0.,
        source_approximation='positive geometry centroids shared across time bins', azimuth_symmetry_assumed=False,
        photon_mu_bins=True, z_step_m=z_step, photon_mu_step=mu_step, photon_phi_step_rad=phi_width,
        tau_step_ns=tau_step, transverse_step_m=transverse, accumulation='vectorised unique/bincount',
        covariance_reference='direction covariance about E[u], representative direction unit(E[u])',
        sample_support_bounds=True, continuous_ring_convergence_certified=False)
    return PhotonHistogram(frame, position, direction, geometry_id, clocks, clock_mass[None, :],
        np.array([wavelength]), geometry_keys, clock_keys, summary, original_position, original_direction,
        original_time, original_mass, original_geometry_id, pcov, dcov, crosscov, position_rms,
        angular_rms_upper, angular_max, spatial_max, raw_mean)


def _unique_integer_rows(keys):
    """Lexicographic unique rows via scalar packed keys when capacity permits.

    Packing changes the sorting implementation only. Overflow is checked with
    Python integer products; very wide sparse grids retain the tuple path.
    """
    if not len(keys): return np.unique(keys, axis=0, return_inverse=True)
    minimum = keys.min(axis=0); maximum = keys.max(axis=0)
    widths = [int(high)-int(low)+1 for high, low in zip(maximum, minimum)]
    product = 1; strides = []
    for width in reversed(widths):
        strides.append(product); product *= width
    if product > np.iinfo(np.int64).max:
        return np.unique(keys, axis=0, return_inverse=True)
    encoded = np.zeros(len(keys), np.int64)
    for component, stride in enumerate(reversed(strides)):
        encoded += (keys[:, component]-minimum[component])*stride
    _, first, inverse = np.unique(encoded, return_index=True, return_inverse=True)
    return keys[first], inverse


def _positive(value, name, allow_inf=False):
    """Positive (optionally infinite) float, or a ValueError naming the setting."""
    value = float(value)
    if value <= 0 or np.isnan(value) or (not allow_inf and not np.isfinite(value)):
        raise ValueError(f'{name} must be positive and finite')
    return value


def compile_joint_source(source, wavelength_nm, phase_index, *, z_step_m=0.5,
                         mu_step=0.1, tau_step_ns=0.1, transverse_step_m=0.05,
                         degree=32, azimuthal_degree=4, chord_order=2,
                         frame=None, reference_ns=None, chunk=4096,
                         max_payload_bytes=512*2**20):
    """Compile sparse (z,mu,tau) bins and exact cone harmonic moments.

    Mu labels the charged direction, whereas ``angular`` contains the photon
    direction moments. No beta grid or mean-beta cone is used. Transverse bins
    provide an optional guard for broad families: infinity disables those
    labels but still retains each cell's actual transverse centroid/variance.
    No source tail is clipped. Bins are nearest nodes anchored at local zero.
    The shared spatial/temporal centroid uses positive C0 quadrature weights,
    independent of wavelength; all photons and harmonics are accumulated with
    their physical wavelength-specific weights. Optical QE/quadrature weights
    are not included; output photon density is per nm.
    """
    steps = np.array([_positive(z_step_m, 'z_step_m'),
                      _positive(mu_step, 'mu_step'),
                      _positive(tau_step_ns, 'tau_step_ns'),
                      _positive(transverse_step_m, 'transverse_step_m', True),
                      _positive(transverse_step_m, 'transverse_step_m', True)])
    for value, name, lower in [(degree, 'degree', 0),
                               (azimuthal_degree, 'azimuthal_degree', 0),
                               (chord_order, 'chord_order', 1), (chunk, 'chunk', 1),
                               (max_payload_bytes, 'max_payload_bytes', 1)]:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < lower:
            raise ValueError(f'{name} must be an integer >= {lower}')
    wavelengths = np.atleast_1d(np.asarray(wavelength_nm, float))
    phase = np.atleast_1d(np.asarray(phase_index, float))
    if (wavelengths.ndim != 1 or not len(wavelengths) or phase.shape != wavelengths.shape
            or not np.isfinite(wavelengths).all() or not np.isfinite(phase).all()
            or np.any(wavelengths <= 0) or np.any(phase <= 0)):
        raise ValueError('positive finite wavelength and phase-index vectors required')
    reference = float(np.min(source.start_ns) if reference_ns is None else reference_ns)
    if not np.isfinite(reference):
        raise ValueError('reference_ns must be finite')
    m = min(degree, azimuthal_degree)
    starts, aa, bb, kept, _ = _layout(degree, m)
    emitting = (np.asarray(source.beta) * phase.max() > 1) & (np.asarray(source.coefficient0) > 0)
    rows = np.flatnonzero(emitting)
    if frame is None:
        if not len(rows):
            raise ValueError('an explicit frame is required for an empty/dark source')
        frame = AxisFrame.of(source.subset(emitting).field(0, phase_index=float(phase.max())))
    basis = np.stack((frame.first, frame.second, frame.axis))
    if (np.asarray(frame.centre_m).shape != (3,) or not np.isfinite(frame.centre_m).all()
            or not np.isfinite(basis).all() or not np.allclose(basis @ basis.T, np.eye(3), atol=1e-12)
            or not np.isclose(np.linalg.det(basis), 1., atol=1e-12)):
        raise ValueError('frame must be a finite right-handed orthonormal basis')
    nodes, weights = np.polynomial.legendre.leggauss(chord_order)
    nodes, weights = (nodes + 1)/2, weights/2
    # Geometry is bounded by the original quadrature count, not a dense 3D grid.
    node_rows = np.repeat(rows, chord_order)
    fractions = np.tile(nodes, len(rows))
    quad_weight = np.tile(weights, len(rows))
    points = (source.start_m[node_rows]
              + (fractions * source.length_m[node_rows])[:, None] * source.direction[node_rows])
    local = frame.rotate(points-frame.centre_m)
    clocks = (source.start_ns[node_rows] + fractions
              * (source.end_ns[node_rows]-source.start_ns[node_rows]))
    tau = clocks-reference-local[:, 2]/VACUUM_M_PER_NS
    directions = np.ascontiguousarray(frame.rotate(source.direction[node_rows]))
    labels = np.stack((local[:, 2], directions[:, 2], tau, local[:, 0], local[:, 1]), axis=1)
    quotient = labels/steps
    if np.any(np.abs(quotient) > np.iinfo(np.int64).max/4):
        raise ValueError('bin coordinates exceed integer capacity')
    keys, inverse = np.unique(np.floor(quotient+0.5).astype(np.int64), axis=0, return_inverse=True)
    count, bands = len(keys), len(wavelengths)
    payload = count*(bands*(len(kept)+1)*8 + 5*8 + 3*8 + 8 + 9*8 + 8 + 8)
    if payload > max_payload_bytes:
        raise ValueError(f'joint source payload {payload} exceeds max_payload_bytes={max_payload_bytes}; '
                         'use local families or coarser bins')
    angular = np.zeros((bands, count, len(kept)))
    photons = np.zeros((bands, count))
    geom_weight = source.coefficient0[node_rows]*quad_weight
    total_weight = np.bincount(inverse, weights=geom_weight, minlength=count)
    local_mean = np.zeros((count, 3)); clock_mean = np.zeros(count); tau_mean = np.zeros(count)
    if count:
        for component in range(3):
            local_mean[:, component] = np.bincount(inverse, weights=geom_weight*local[:, component], minlength=count)/total_weight
        clock_mean = np.bincount(inverse, weights=geom_weight*clocks, minlength=count)/total_weight
        tau_mean = np.bincount(inverse, weights=geom_weight*tau, minlength=count)/total_weight
    covariance = np.zeros((count, 3, 3)); tau_variance = np.zeros(count)
    offset = local-local_mean[inverse]
    if count:
        for first in range(3):
            for second in range(first, 3):
                value = np.bincount(inverse, weights=geom_weight*offset[:, first]*offset[:, second], minlength=count)/total_weight
                covariance[:, first, second] = covariance[:, second, first] = value
        tau_variance = np.bincount(inverse, weights=geom_weight*(tau-tau_mean[inverse])**2, minlength=count)/total_weight
    for lo in range(0, len(node_rows), chunk):
        hi = min(lo+chunk, len(node_rows)); take = node_rows[lo:hi]
        values = _charged_harmonics(np.ascontiguousarray(directions[lo:hi]), starts, aa, bb, m)
        all_cones = np.empty((bands, hi-lo)); all_physical = np.empty_like(all_cones)
        for band, (wavelength, index) in enumerate(zip(wavelengths, phase)):
            beta = source.beta[take]
            active = beta*index > 1
            cone = np.where(active, 1/(beta*index), 1.)
            physical = np.where(active, np.maximum(0., source.coefficient0[take]
                                 + source.coefficient2[take]/index**2), 0.)
            physical *= quad_weight[lo:hi]/wavelength**2
            all_cones[band] = cone; all_physical[band] = physical
        _accumulate_spectral_cones(values, all_cones, all_physical,
                                  np.ascontiguousarray(inverse[lo:hi]), starts, angular, photons)
    world = frame.centre_m + local_mean @ basis
    summary = dict(source_rows=int(len(source)), emitting_rows=int(len(rows)),
                   quadrature_points=int(len(node_rows)), occupied_cells=int(count),
                   point_compression_ratio=float(len(node_rows)/max(count, 1)),
                   z_step_m=float(steps[0]), charged_mu_step=float(steps[1]),
                   tau_step_ns=float(steps[2]), transverse_step_m=float(steps[3]),
                   transverse_rms_max_m=float(np.sqrt(np.max(covariance[:, 0, 0]+covariance[:, 1, 1]))) if count else 0.,
                   residual_rms_max_ns=float(np.sqrt(tau_variance.max())) if count else 0.,
                   angular_binning='exact cone harmonics conditioned on charged-mu bins',
                   geometry_approximation='positive C0 centroids; within-cell position-angle-time covariance not applied',
                   azimuth_symmetry_assumed=False, spectral_beta_threshold_exact=True,
                   payload_bytes=int(angular.nbytes+photons.nbytes+keys.nbytes+world.nbytes+clock_mean.nbytes))
    return JointSource(frame, keys, world, clock_mean, angular, photons, covariance,
                       tau_variance, total_weight, wavelengths, phase, kept, degree, m, reference, summary)
