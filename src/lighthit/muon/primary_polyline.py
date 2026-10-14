"""Connected polyline approximation of the primary track.

Tolerances constrain geometry, emission clock and spectrum, not the error of
the detected signal. The long chords are meant for the primary order-one and
atlas routes only; the generic >=2 point compiler must not receive them.
"""
from dataclasses import dataclass
import numpy as np
from ..sources import SpectralLightElements


@dataclass(frozen=True)
class PrimaryPolylineConfig:
    """Tolerances of the connected polyline fitted to the primary."""
    max_length_m: float = 250.
    position_tolerance_m: float = .01
    time_tolerance_ns: float = .01
    direction_rms_tolerance_rad: float = .003
    beta_tolerance: float = .0002
    yield_relative_tolerance: float = .001

    def __post_init__(self):
        """All tolerances must be finite and positive."""
        values = tuple(vars(self).values())
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError('polyline tolerances must be finite and positive')


@dataclass(frozen=True)
class PrimaryPolylineFit:
    """Polyline chords as elements, their step ranges and the fit diagnostics."""
    elements: SpectralLightElements
    ranges: np.ndarray
    diagnostics: dict


def fit_primary_polyline(source, *, config=None, phase_indices=()):
    """Fit adjacent endpoint chords, splitting at measured local deviations.

    Both spectral fields are conserved per chord; beta_eff=sqrt(-C0/C2).
    Original spatial and temporal gaps are never bridged. Endpoint times are
    retained exactly. Threshold crossings, speed changes and nonuniform light
    yield force local refinement, so a stopping tail receives finer pieces
    without an energy-dependent policy. Single original steps are immutable.
    """
    config = config or PrimaryPolylineConfig()
    phases = np.asarray(phase_indices, float)
    if phases.ndim != 1 or not np.isfinite(phases).all() or np.any(phases <= 0):
        raise ValueError('phase_indices must be a vector of positive finite indices')
    if len(np.unique(source.uid)) != 1:
        raise ValueError('provide one primary trajectory UID')
    if np.any(np.diff(source.start_ns) < 0) or np.any(source.end_ns < source.start_ns):
        raise ValueError('primary times must be ordered with nonnegative duration')
    ends = source.start_m + source.length_m[:, None] * source.direction
    n = len(source)
    eps = 64 * np.finfo(float).eps
    position_scale = np.maximum(1., np.maximum(np.max(abs(ends[:-1]), axis=1),
                                              np.max(abs(source.start_m[1:]), axis=1)))
    time_scale = np.maximum(1., np.maximum(abs(source.end_ns[:-1]), abs(source.start_ns[1:])))
    gaps = (np.max(abs(ends[:-1]-source.start_m[1:]), axis=1) > eps*position_scale)
    gaps |= abs(source.end_ns[:-1]-source.start_ns[1:]) > eps*time_scale
    boundaries = np.r_[0, np.flatnonzero(gaps)+1, n]
    arc = np.r_[0., np.cumsum(source.length_m)]
    accepted = []
    constraints = dict(position_m=[], time_ns=[], direction_rms_rad=[], beta_range=[])

    def recurse(lo, hi):
        """Accept steps lo..hi as one chord when all tolerances hold, else split."""
        c0 = source.coefficient0[lo:hi].sum()
        c2 = source.coefficient2[lo:hi].sum()
        delta = ends[hi-1] - source.start_m[lo]
        length = np.linalg.norm(delta)
        if hi-lo == 1:
            accepted.append((lo, hi, source.direction[lo], source.length_m[lo], source.beta[lo]))
            for values in constraints.values():
                values.append(0.)
            return
        if length > 0 and c0 > 0 and c2 < 0:
            direction = delta / length
            weights = source.coefficient0[lo:hi] / c0
            position_error = time_error = 0.
            for positions, times in ((source.start_m[lo:hi], source.start_ns[lo:hi]),
                                     (ends[lo:hi], source.end_ns[lo:hi])):
                rel = positions - source.start_m[lo]
                along = np.clip(rel @ direction, 0., length)
                position_error = max(position_error, float(np.max(np.linalg.norm(
                    rel-along[:, None]*direction, axis=1))))
                fitted_time = source.start_ns[lo] + along/length*(source.end_ns[hi-1]-source.start_ns[lo])
                time_error = max(time_error, float(np.max(abs(times-fitted_time))))
            angles = np.arccos(np.clip(source.direction[lo:hi] @ direction, -1., 1.))
            rms = float(np.sqrt(weights @ (angles*angles)))
            beta_min, beta_max = source.beta[lo:hi].min(), source.beta[lo:hi].max()
            density = source.coefficient0[lo:hi] / source.length_m[lo:hi]
            threshold_crossing = np.any((beta_min*phases <= 1.) & (beta_max*phases > 1.))
            beta_eff = float(np.sqrt(-c0/c2))
            # Generic SpectralLightElements need not encode C2=-C0/beta^2.
            # Retain such records rather than silently inventing a new cone.
            beta_consistent = np.allclose(-source.coefficient2[lo:hi]*source.beta[lo:hi]**2,
                                         source.coefficient0[lo:hi], rtol=1e-10, atol=0.)
            good = (length <= config.max_length_m and position_error <= config.position_tolerance_m
                    and time_error <= config.time_tolerance_ns and rms <= config.direction_rms_tolerance_rad
                    and beta_max-beta_min <= config.beta_tolerance and not threshold_crossing
                    and np.ptp(density) <= config.yield_relative_tolerance*(weights @ density)
                    and beta_consistent)
            if good:
                accepted.append((lo, hi, direction, length, beta_eff))
                for key, value in zip(constraints, (position_error, time_error, rms, beta_max-beta_min)):
                    constraints[key].append(value)
                return
        split = int(np.searchsorted(arc, .5*(arc[lo]+arc[hi])))
        split = min(hi-1, max(lo+1, split))
        recurse(lo, split)
        recurse(split, hi)

    for lo, hi in zip(boundaries[:-1], boundaries[1:]):
        recurse(int(lo), int(hi))
    ranges = np.asarray([(a[0], a[1]) for a in accepted], dtype=np.int64)
    lo, hi = ranges.T
    elements = SpectralLightElements(
        start_m=source.start_m[lo].copy(), direction=np.asarray([a[2] for a in accepted]),
        length_m=np.asarray([a[3] for a in accepted]),
        coefficient0=np.asarray([source.coefficient0[a:b].sum() for a,b in ranges]),
        coefficient2=np.asarray([source.coefficient2[a:b].sum() for a,b in ranges]),
        beta=np.asarray([a[4] for a in accepted]), start_ns=source.start_ns[lo].copy(),
        end_ns=source.end_ns[hi-1].copy(), row_index=source.row_index[lo], uid=source.uid[lo],
        reference_phase_index=source.reference_phase_index, cone_model=source.cone_model,
        provenance={**(source.provenance or {}), 'primary_polyline': vars(config)})
    diagnostics = dict(original_elements=n, pieces=len(elements), disconnected_runs=len(boundaries)-1,
                       max_errors={k: float(max(v)) for k,v in constraints.items()},
                       per_piece_errors=constraints, config=vars(config),
                       certified_transport_error=False)
    return PrimaryPolylineFit(elements, ranges, diagnostics)


def order1_primary_polyline(kernel, source, *, time_origin_ns, fit_config=None, prompt_config=None):
    """Explicit primary-only order-one route; no q0 or >=2 mutation."""
    from .muon_integral import order1_integrals
    if prompt_config is not None and prompt_config.element_method != 'segment':
        raise ValueError('primary polyline requires physical-time segment integration')
    phases = kernel.medium.sample(kernel.wavelength.wavelength_nm)['phase_index']
    fit = fit_primary_polyline(source, config=fit_config, phase_indices=phases)
    charge, bins, diagnostics = order1_integrals(kernel, fit.elements,
        time_origin_ns=time_origin_ns, config=prompt_config)
    return charge, bins, {'polyline': fit.diagnostics, 'transport': diagnostics}


def primary_spectral_quadrature(medium, detector, *, order=2, efficiency_knots_nm=None, wavelength_range_nm=None):
    """Positive composite Gauss rule over known optical-table breakpoints.

    Detector knots must be supplied explicitly or advertised as
    ``detector.spectral_efficiency.wavelength_knots_nm``. An explicitly empty
    vector means the caller asserts a smooth efficiency over the medium range.
    This resolves known interpolation kinks; it does not certify wavelength
    convergence of the receiver response or unknown callable structure.
    """
    from ..model import WavelengthQuadrature
    if isinstance(order, bool) or not isinstance(order, (int, np.integer)) or order < 1:
        raise ValueError('order must be a positive integer')
    if efficiency_knots_nm is None:
        efficiency_knots_nm = getattr(detector.spectral_efficiency, 'wavelength_knots_nm', None)
    if efficiency_knots_nm is None:
        raise ValueError('provide known efficiency wavelength knots; no smoothness assumption is implicit')
    knots = np.asarray(efficiency_knots_nm, float)
    if knots.ndim != 1 or not np.isfinite(knots).all() or np.any(knots <= 0):
        raise ValueError('efficiency knots must be a finite positive vector')
    low, high = wavelength_range_nm or medium.wavelength_range_nm
    if not medium.wavelength_range_nm[0] <= low < high <= medium.wavelength_range_nm[1]:
        raise ValueError('quadrature range must lie within medium table')
    water = medium.wavelength_nm
    edges = np.unique(np.r_[low, high, water[(water > low) & (water < high)], knots[(knots > low) & (knots < high)]])
    nodes, weights = np.polynomial.legendre.leggauss(order)
    half = np.diff(edges) / 2
    wavelengths = edges[:-1, None] + half[:, None]*(nodes[None, :]+1)
    integration_weights = half[:, None]*weights[None, :]
    return WavelengthQuadrature(wavelengths.ravel(), integration_weights.ravel())
