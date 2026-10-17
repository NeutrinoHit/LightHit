"""Order >=2 light of source points closer to an OM than the RTE tables reach.

The prepared order >=2 kernels are tabulated from the radial minimum of the
cache (3 m) outwards. A track that passes an OM closer than that has source
points the tables do not cover. The routes leave those points out of their
own order >=2 sums and this module adds them back with an explicit
approximation, the *near-field clamp*:

    G>=2(r, directions) ~= G>=2(r_min, same directions),   r < r_min,

with the flight phase (arrival time) of the actual distance ``r``. Directions
(source cone, OM orientation) are exact. The clamp rests on the bounded
behaviour of multiply scattered light near its source: the probability of two
scatterings within ``r`` grows as ``r**2`` while the geometric dilution falls
as ``r**-2``, so the order >=2 Green function tends to a finite, slowly
varying limit as ``r -> 0`` instead of the ``r**-2`` growth of direct light.
Its error is the variation of ``G>=2`` between ``r`` and ``r_min``.

Each OM that uses the clamp is reported (never silently): the route metadata
gets ``near_field`` (OM indices, the clamped charge) and a ``RuntimeWarning``
names the OMs. Orders 0 and 1 are not affected: they are computed in physical
space without the radial table.
"""
from dataclasses import dataclass, field
import warnings

import numpy as np


@dataclass
class NearFieldGe2:
    """Clamped order >=2 of the near source portions, per OM."""
    oms: np.ndarray                 # OM indices that have near source points
    charge_pe: np.ndarray           # (n,) clamped order >=2 charge
    spectrum_pe: np.ndarray         # (frequency, n) relative to time_origin_ns
    metadata: dict = field(default_factory=dict)


def near_portions(positions_m, source, radius_m):
    """Element portions within ``radius_m`` of each OM.

    Returns {om: [(element, s0, s1), ...]} with ``s`` measured along the
    element from its start.
    """
    from scipy.spatial import cKDTree
    positions = np.asarray(positions_m, float)
    start = np.asarray(source.start_m, float)
    direction = np.asarray(source.direction, float)
    length = np.asarray(source.length_m, float)
    middle = start + 0.5 * length[:, None] * direction
    nearest, _ = cKDTree(positions).query(middle, k=1)
    candidates = np.flatnonzero(nearest < radius_m + 0.5 * length)
    portions = {}
    if not len(candidates):
        return portions
    tree = cKDTree(positions)
    for index in candidates:
        reach = radius_m + 0.5 * length[index]
        for om in tree.query_ball_point(middle[index], reach):
            delta = positions[om] - start[index]
            z = float(delta @ direction[index])
            rho2 = max(float(delta @ delta) - z * z, 0.)
            if rho2 >= radius_m * radius_m:
                continue
            half = np.sqrt(radius_m * radius_m - rho2)
            s0, s1 = max(z - half, 0.), min(z + half, float(length[index]))
            if s1 > s0:
                portions.setdefault(int(om), []).append((int(index), s0, s1))
    return portions


def _harmonics(source, index, wavelength_nm, phase_index, degree):
    """Spectral cone harmonics of one element per unit coefficient0."""
    from scipy.special import eval_legendre
    ell = np.arange(degree + 1)
    norm = np.sqrt((2 * ell + 1) / (4 * np.pi))
    beta = float(source.beta[index])
    ratio = float(source.coefficient2[index] / source.coefficient0[index]) \
        if source.coefficient0[index] > 0 else 0.
    frozen = source.cone_model == "frozen"
    out = np.zeros((len(wavelength_nm), degree + 1))
    for band, (wavelength, phase) in enumerate(zip(wavelength_nm, phase_index)):
        cone_phase = source.reference_phase_index if frozen else phase
        yield_ = (1 + ratio / cone_phase ** 2) / wavelength ** 2
        if beta * cone_phase <= 1 or yield_ <= 0:
            continue
        out[band] = norm * eval_legendre(ell, 1 / (beta * cone_phase)) * yield_
    return out


def near_field_ge2(kernel, source, bundle, *, time_origin_ns, radius_m=None,
                   panel_m=0.05, gauss_order=6):
    """Clamped order >=2 of every source portion closer than ``radius_m``.

    ``bundle`` is the order >=2 spline of the route (``Ge2KernelStore.bundle``
    on the kernel's frequencies, flight-referenced); ``radius_m`` defaults to
    its radial minimum. ``time_origin_ns`` is the per-OM clock of the
    response the result is added to.
    """
    from .muon.track_ge2_primitive import cone_point_evaluator
    low = float(np.exp(bundle["log_r"][0]))
    radius = low if radius_m is None else float(radius_m)
    omega = np.asarray(bundle["omega"], float)
    detector = kernel.detector
    positions = np.asarray(detector.positions_m, float)
    n = len(positions)
    origins = np.asarray(time_origin_ns, float)
    if origins.shape != (n,) or not np.isfinite(origins).all():
        raise ValueError("one finite time origin per OM required")
    wave = kernel.wavelength
    sample = kernel.medium.sample(wave.wavelength_nm)
    efficiency = kernel.detector.spectral_weight(wave.wavelength_nm)
    alpha = kernel.acceptance()["alpha"]
    area = np.broadcast_to(np.asarray(detector.effective_area_m2, float), (n,))
    nodes, weights = np.polynomial.legendre.leggauss(int(gauss_order))
    charge = np.zeros(n)
    spectrum = np.zeros((len(omega), n), complex)
    portions = near_portions(positions, source, radius)
    harmonics = {}
    elements = 0
    for om, parts in portions.items():
        position = positions[om]
        normal = np.asarray(detector.orientations[om], float)
        for index, s0, s1 in parts:
            axis = source.direction[index]
            delta = position - source.start_m[index]
            z = float(delta @ axis)
            radial = delta - z * axis
            rho = float(np.linalg.norm(radial))
            if rho < 1e-9:
                # The line goes through the OM centre: any transverse axis.
                trial = np.array([1., 0., 0.]) if abs(axis[0]) < .9 else np.array([0., 1., 0.])
                radial = trial - (trial @ axis) * axis
                rho = 1e-9
            ex = radial / np.linalg.norm(radial)
            ey = np.cross(axis, ex)
            look = -np.array([normal @ ex, normal @ ey, normal @ axis])
            if index not in harmonics:
                harmonics[index] = _harmonics(source, index, wave.wavelength_nm,
                                              sample["phase_index"], int(bundle["degree"]))
            if not np.any(harmonics[index]):
                continue
            # Points as u = z - s (OM minus source along the axis).
            count = max(1, int(np.ceil((s1 - s0) / panel_m)))
            edges = np.linspace(z - s1, z - s0, count + 1)
            half = np.diff(edges) / 2
            u = (edges[:-1, None] + half[:, None] * (nodes + 1)).ravel()
            w = (half[:, None] * weights).ravel()
            evaluate = cone_point_evaluator(
                bundle, impact_m=rho, look=look, beta=float(source.beta[index]), alpha=alpha,
                wavelength_nm=wave.wavelength_nm, wavelength_weight_nm=wave.weight_nm,
                phase_index=sample["phase_index"], efficiency=efficiency,
                effective_area_m2=float(area[om]), backend="fused-looks",
                source_harmonics=harmonics[index], near_field=True)
            values = evaluate(u)
            length = float(source.length_m[index])
            clock = float(source.end_ns[index] - source.start_ns[index]) / length
            density = float(source.coefficient0[index]) / length
            # Emission at s = z - u, time start + s * clock.
            closest = float(source.start_ns[index]) + z * clock - origins[om]
            part = density * np.exp(1j * omega * closest) * (
                (w[:, None] * values * np.exp(-1j * u[:, None] * clock * omega)).sum(0))
            spectrum[:, om] += part
            charge[om] += part[0].real
            elements += 1
    oms = np.array(sorted(portions), int)
    metadata = dict(
        radius_m=radius, table_radial_minimum_m=low, oms=oms.tolist(),
        charge_pe=[float(charge[om]) for om in oms], elements=elements,
        approximation="order >=2 Green function clamped at the table radial minimum "
                      "along the same directions; flight time of the actual distance",
        panel_m=float(panel_m), gauss_order=int(gauss_order))
    return NearFieldGe2(oms, charge, spectrum, metadata)


def add_near_field(response, kernel, source, bundle, *, label=None):
    """Add the clamped near-field order >=2 to a ``HybridTransportResponse``.

    Charges of the near OMs receive it where the response computed them in
    detail (``metadata['fine_charge_mask']`` when present, else every OM);
    time bins only where it is ``active``. The OMs are recorded in
    ``response.metadata['near_field']`` and named in a ``RuntimeWarning``.
    Returns the response (changed in place).
    """
    from .readout import readout_bins
    near = near_field_ge2(kernel, source, bundle, time_origin_ns=response.time_origin_ns)
    if not len(near.oms):
        return response
    omega = np.asarray(response.omega_per_ns, float)
    if not np.allclose(np.asarray(bundle["omega"], float), omega, rtol=0, atol=1e-12):
        raise ValueError("near-field frequencies differ from the response")
    fine = response.metadata.get("fine_charge_mask")
    fine = np.ones(len(response.active), bool) if fine is None else np.asarray(fine, bool)
    oms = near.oms[fine[near.oms]]
    response.charge_components_pe[oms, 2] += near.charge_pe[oms]
    active = np.asarray(response.active, bool)
    detailed = oms[active[oms]]
    if len(detailed):
        response.components_pe[detailed, :, 2] += readout_bins(
            omega, near.spectrum_pe[:, detailed], response.relative_time_edges_ns,
            getattr(kernel.config, "readout_omega_per_ns", None))
    if response.spectrum_ge2_pe is not None:
        response.spectrum_ge2_pe[:, oms] += (
            near.spectrum_pe[:, oms]
            * np.exp(1j * omega[:, None] * np.asarray(response.time_origin_ns)[None, oms]))
    record = dict(near.metadata, label=label, charge_added_oms=oms.tolist(),
                  bins_added_oms=detailed.tolist())
    response.metadata.setdefault("near_field", []).append(record)
    ids = kernel.detector.identifiers.get("uid") if kernel.detector.identifiers else None
    named = near.oms.tolist() if ids is None else [int(np.asarray(ids)[i]) for i in near.oms]
    warnings.warn(
        f"{len(near.oms)} OMs have source points closer than the {near.metadata['radius_m']:g} m "
        f"radial minimum of the order >=2 tables (OM {'uid' if ids is not None else 'index'} "
        f"{named}); their order >=2 near part uses the near-field clamp "
        f"({sum(near.metadata['charge_pe']):.3g} pe), see metadata['near_field']",
        RuntimeWarning, stacklevel=2)
    return response
