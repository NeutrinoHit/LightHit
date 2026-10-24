"""Continuous-spectrum direct light, split at geometric and time-bin roots.

The medium is exactly its public piecewise-linear table. A supplied tabulated
OM response uses its exact linear-times-exponential law; an opaque detector
spectral callable is approximated by locally verified cubic panels. Known
callable breakpoints can be supplied via ``wavelength_knots_nm``; a callable
with unresolved hidden features cannot be certified from finitely many probes.
This explicit API does not rebuild or alter a scattered-light RTE cache.
"""
from dataclasses import dataclass
from time import perf_counter, process_time

import numpy as np


@dataclass(frozen=True)
class BallisticIntervalConfig:
    """Quadrature orders, tolerances and efficiency panels of the direct light.

    ``pair_list`` evaluates every convergence pass only on the element-OM
    pairs passing the geometry test (bitwise equal to the full loop).
    ``receiver_mode="auto"`` uses the supplied receiver surface at every
    distance and source size; a detector without a surface is a point model.
    ``"surface"`` applies the supplied physical sphere everywhere; ``"point"``
    evaluates the centre everywhere. Effective collection area never supplies
    a geometric radius. The default finite cone window retains centre-ray
    attenuation and time. ``first_entry_ballistic`` is the explicit
    first-entry evaluator, currently more expensive at the same tolerance.
    """
    receiver_mode: str = "auto"
    order: int = 8
    max_order: int = 64
    relative_tolerance: float = 2e-6
    absolute_tolerance_pe: float = 1e-10
    efficiency_relative_tolerance: float = 1e-8
    efficiency_absolute_tolerance: float = 1e-12
    efficiency_max_depth: int = 16
    pair_list: bool = True

    def __post_init__(self):
        """Integer orders and positive tolerances."""
        for name in ('order', 'max_order', 'efficiency_max_depth'):
            if isinstance(getattr(self, name), bool) or not isinstance(getattr(self, name), (int, np.integer)):
                raise ValueError(f'{name} must be an integer')
        if self.order < 2 or self.max_order < 2*self.order or self.efficiency_max_depth < 1:
            raise ValueError('require order>=2, max_order>=2*order and positive panel depth')
        for name in ('relative_tolerance', 'absolute_tolerance_pe', 'efficiency_relative_tolerance',
                     'efficiency_absolute_tolerance'):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if self.receiver_mode not in ("auto", "surface", "point"):
            raise ValueError("receiver_mode must be auto, surface or point")


@dataclass
class BallisticIntervalResult:
    """Direct charge per OM, optional bins, and the quadrature history."""
    charge_pe: np.ndarray
    bins_pe: np.ndarray | None
    metadata: dict


def prepare_optics(kernel, config=None, *, spectral_knots_nm=None, exact_response=False):
    """Construct numerical panels and check their spectral-efficiency error.

    Cubics use ordinary ascending powers of normalized panel coordinate.
    With ``exact_response=True``, a callable advertising
    ``linear_exponential_tables`` supplies its exact linear-QE times
    log-linear-transmission law. Its six coefficients are the four ordinary
    polynomial coefficients followed by the constant and slope of the log
    transmission. The default four-column representation is unchanged.
    Water knots are always preserved; extra subdivisions are exact for the
    water's linear tables. Efficiency interpolation is checked at seven points
    independent of its four interpolation nodes and at polynomial minima.
    """
    config = config or BallisticIntervalConfig()
    medium = kernel.medium
    low, high = kernel.config.wavelength_range_nm or medium.wavelength_range_nm
    if not medium.wavelength_range_nm[0] <= low < high <= medium.wavelength_range_nm[1]:
        raise ValueError('spectral range lies outside medium table')
    supplied = spectral_knots_nm
    if supplied is None:
        supplied = getattr(kernel.detector.spectral_efficiency, 'wavelength_knots_nm', None)
    known_knots = supplied is not None
    extra = np.asarray([] if supplied is None else supplied, float)
    if extra.ndim != 1 or not np.isfinite(extra).all():
        raise ValueError('spectral response knots must be a finite vector')
    response_tables = (getattr(kernel.detector.spectral_efficiency,
                               'linear_exponential_tables', None) if exact_response else None)
    if response_tables is not None:
        ew, ev, tw, log_t = [np.asarray(value, float) for value in response_tables]
        if (any(value.ndim != 1 or not np.isfinite(value).all() for value in (ew, ev, tw, log_t))
                or len(ew) < 2 or len(tw) < 2 or ev.shape != ew.shape or log_t.shape != tw.shape
                or np.any(np.diff(ew) <= 0) or np.any(np.diff(tw) <= 0) or np.any(ev < 0)):
            raise ValueError('invalid linear-exponential response tables')
        extra = np.r_[extra, ew, tw]
        known_knots = True
    edges = np.unique(np.r_[low, high, medium.wavelength_nm, extra,
                            np.linspace(low, high, max(2, int(np.ceil((high-low)/20))+1))])
    edges = edges[(edges >= low) & (edges <= high)]
    nodes = np.array([0., 1/3, 2/3, 1.])
    probes = np.array([1/8, 1/6, 3/8, .5, 5/8, 5/6, 7/8])
    inverse = np.linalg.inv(np.polynomial.polynomial.polyvander(nodes, 3))
    checks = []
    if response_tables is not None:
        widths = np.diff(edges)
        qe = np.interp(edges, ew, ev)
        transmission = np.interp(edges, tw, log_t)
        coefficients = np.zeros((len(widths), 6))
        coefficients[:, 0], coefficients[:, 1] = qe[:-1], np.diff(qe)
        coefficients[:, 4], coefficients[:, 5] = transmission[:-1], np.diff(transmission)
        wavelength_probe = edges[:-1, None]+widths[:, None]*probes
        exact = kernel.detector.spectral_weight(wavelength_probe)
        predicted = ((qe[:-1, None]+np.diff(qe)[:, None]*probes)
                     * np.exp(transmission[:-1, None]+np.diff(transmission)[:, None]*probes))
        if not np.allclose(predicted, exact, rtol=1e-13, atol=config.efficiency_absolute_tolerance):
            raise ValueError('linear-exponential tables differ from the spectral callable')
        sample = medium.sample(edges)
        return dict(panel_edges=np.ascontiguousarray(edges),
                    phase_nodes=np.ascontiguousarray(sample['phase_index']),
                    group_nodes=np.ascontiguousarray(sample['group_index']),
                    extinction_nodes=np.ascontiguousarray(sample['absorption_per_m']+sample['scattering_per_m']),
                    efficiency_coefficients=np.ascontiguousarray(coefficients),
                    metadata=dict(panels=len(widths), supplied_response_knots=True,
                                  efficiency_representation='linear QE times exponential linear transmission',
                                  efficiency_interpolation_error=False))
    for depth in range(config.efficiency_max_depth+1):
        widths = np.diff(edges)
        wavelength = edges[:-1, None]+widths[:, None]*nodes
        values = np.asarray(kernel.detector.spectral_weight(wavelength), float)
        if values.shape != wavelength.shape:
            raise ValueError('spectral response must preserve input shape')
        coefficients = values@inverse.T
        wavelength_probe = edges[:-1, None]+widths[:, None]*probes
        exact = np.asarray(kernel.detector.spectral_weight(wavelength_probe), float)
        predicted = coefficients@np.polynomial.polynomial.polyvander(probes, 3).T
        tolerance = config.efficiency_absolute_tolerance+config.efficiency_relative_tolerance*np.abs(exact)
        error_ratio = np.max(np.abs(predicted-exact)/tolerance, axis=1)
        bad = error_ratio > 1
        # Cubic extrema, including endpoints, must be nonnegative to roundoff.
        for i, coef in enumerate(coefficients):
            roots = np.polynomial.polynomial.polyroots(np.array([coef[1], 2*coef[2], 3*coef[3]]))
            internal = [r.real for r in roots if abs(r.imag) < 1e-12 and 0 < r.real < 1]
            minimum = min(np.polynomial.polynomial.polyval(np.r_[0., internal, 1.], coef))
            bad[i] |= minimum < -config.efficiency_absolute_tolerance
        checks.append(dict(depth=depth, panels=len(widths), failed=int(bad.sum()),
                           maximum_tolerance_ratio=float(error_ratio.max())))
        if not bad.any():
            break
        if depth == config.efficiency_max_depth:
            raise ValueError('spectral-efficiency interpolation did not converge; supply its breakpoints')
        edges = np.sort(np.r_[edges, (edges[:-1]+widths/2)[bad]])
    sample = medium.sample(edges)
    return dict(panel_edges=np.ascontiguousarray(edges),
                phase_nodes=np.ascontiguousarray(sample['phase_index']),
                group_nodes=np.ascontiguousarray(sample['group_index']),
                extinction_nodes=np.ascontiguousarray(sample['absorption_per_m']+sample['scattering_per_m']),
                efficiency_coefficients=np.ascontiguousarray(coefficients),
                metadata=dict(panels=len(edges)-1, supplied_response_knots=known_knots,
                    efficiency_checks=checks, efficiency_error_is_sampled_not_rigorous=True))


def _ballistic_interval_leading(kernel, source, *, config=None, time_origin_ns=None,
                       relative_edges_ns=None, spectral_knots_nm=None, _tables=None, _pairs=None):
    """Compute integrated charge and optional time bins on original elements.

    No source compression is performed. Quadrature orders are compared per OM;
    unconverged OMs alone are recomputed at higher order. Failure at max_order is
    explicit. The error test controls charge and bin L1 relative to total charge,
    including light outside the requested time window. It is a numerical
    convergence test, not a proof for arbitrary unknown response functions.
    """
    from .sources import SpectralLightElements
    from ._ballistic_interval_numba import (interval_kernel, interval_candidates,
                                            interval_kernel_pairs)
    config = config or BallisticIntervalConfig()
    if not isinstance(source, SpectralLightElements) or source.cone_model != 'spectral':
        raise ValueError('continuous ballistic spectrum requires spectral LightElements')
    if not np.allclose(source.coefficient2, -source.coefficient0/source.beta**2, rtol=1e-10, atol=1e-20):
        raise ValueError('source must have the physical Cherenkov C0/C2/beta relation')
    began, cpu = perf_counter(), process_time()
    tables = (prepare_optics(kernel, config, spectral_knots_nm=spectral_knots_nm, exact_response=True)
              if _tables is None else _tables)
    prepare_seconds = perf_counter()-began
    detector = kernel.detector
    positions = np.ascontiguousarray(detector.positions_m, float)
    looks = np.ascontiguousarray(-detector.orientations, float)
    area = np.ascontiguousarray(detector.effective_area_m2, float)
    from .receiver import SphereSurface
    if isinstance(detector.receiver_surface, SphereSurface):
        radii = np.ascontiguousarray(detector.receiver_surface.radii(len(detector)), float)
    elif config.receiver_mode == "surface":
        raise ValueError('surface mode needs detector.receiver_surface geometry')
    else:
        radii = np.zeros(len(detector))
    reach = 0. if config.receiver_mode == "point" else np.inf
    alpha = np.ascontiguousarray(kernel.acceptance()['alpha'], float)
    want_bins = time_origin_ns is not None or relative_edges_ns is not None
    if want_bins:
        if time_origin_ns is None or relative_edges_ns is None:
            raise ValueError('time origins and bin edges must be provided together')
        origins = np.ascontiguousarray(time_origin_ns, float)
        edges = np.ascontiguousarray(relative_edges_ns, float)
        if (origins.shape != (len(positions),) or not np.isfinite(origins).all()
                or edges.ndim != 1 or len(edges) < 2 or not np.isfinite(edges).all()
                or np.any(np.diff(edges) <= 0)):
            raise ValueError('invalid time origins or edges')
    else:
        origins, edges = np.zeros(len(positions)), np.array([0., 1.])
    arrays = [np.ascontiguousarray(getattr(source, name), float) for name in
              ('start_m', 'direction', 'length_m', 'coefficient0', 'coefficient2', 'beta', 'start_ns', 'end_ns')]
    optics = [tables[key] for key in ('panel_edges', 'phase_nodes', 'group_nodes',
                                      'extinction_nodes', 'efficiency_coefficients')]
    if config.pair_list:
        # The geometry precheck does not depend on the quadrature order: find
        # its pairs once, then every convergence pass visits only those pairs,
        # receiver-major in element order as in interval_kernel.
        pair_om, pair_element = interval_candidates(positions, radii, reach, arrays[0], arrays[1], arrays[2],
                                                    arrays[3], arrays[5], tables['phase_nodes']) if _pairs is None else _pairs
    def evaluate(ids, order):
        """Charge and bins of the OMs ``ids`` at Gauss order ``order``."""
        gx, gw = np.polynomial.legendre.leggauss(order)
        if not config.pair_list:
            return interval_kernel(positions[ids], looks[ids], area[ids], *arrays, *optics,
                alpha, gx, gw, origins[ids], edges, want_bins, radii[ids], reach)
        row_of = np.full(len(positions), -1, np.int64)
        row_of[ids] = np.arange(len(ids))
        keep = row_of[pair_om] >= 0
        return interval_kernel_pairs(pair_om[keep], pair_element[keep], row_of[pair_om[keep]],
            len(ids), positions, looks, area, *arrays, *optics, alpha, gx, gw,
            origins[ids], edges, want_bins, radii, reach)
    ids = np.arange(len(positions))
    order = config.order
    charge, bins, counts = evaluate(ids, order)
    history = [dict(order=order, receivers=len(ids), counters=counts.tolist())]
    while len(ids):
        order = min(config.max_order, 2*order)
        q, b, counts = evaluate(ids, order)
        delta = np.abs(q-charge[ids])
        if want_bins:
            delta = np.maximum(delta, np.abs(b-bins[ids]).sum(axis=1))
        tolerance = config.absolute_tolerance_pe+config.relative_tolerance*np.abs(q)
        bad = delta > tolerance
        charge[ids], bins[ids] = q, b
        history.append(dict(order=order, receivers=len(ids), failed=int(bad.sum()),
                            maximum_change_pe=float(delta.max(initial=0)), counters=counts.tolist()))
        ids = ids[bad]
        if len(ids) and order == config.max_order:
            raise ValueError(f'ballistic spectral quadrature failed for {len(ids)} OMs at order {order}')
    return BallisticIntervalResult(charge, bins if want_bins else None,
        dict(method='continuous-spectrum-intervals', source_elements=len(source),
             optical_panels=tables['metadata'], quadrature=history,
             prepare_seconds=prepare_seconds, elapsed_seconds=perf_counter()-began,
             cpu_seconds=process_time()-cpu, source_compressed=False,
             time_bins_split_at_arrival_roots=want_bins, pair_list=config.pair_list,
             receiver_mode=config.receiver_mode,
             receiver_surface=("sphere" if isinstance(detector.receiver_surface, SphereSurface) else None),
             relative_tolerance=config.relative_tolerance, absolute_tolerance_pe=config.absolute_tolerance_pe))


def _prepare_ballistic_interval(kernel, source, *, config=None, spectral_knots_nm=None):
    """Prepare charge now and defer arrival bins until receiver selection."""
    from ._surface_ballistic import PreparedBallistic
    return PreparedBallistic(kernel, source, config or BallisticIntervalConfig(), spectral_knots_nm)


def first_entry_ballistic(kernel, source, *, config=None, time_origin_ns=None,
                       relative_edges_ns=None, spectral_knots_nm=None):
    """Direct light to the first glass entry, or to an explicit point receiver.

    Finite receivers use the analytic cone/disk response plus its geometric
    residual. Positive residual bounds permit omission within the requested
    per-OM tolerance. Charge and bin L1 undergo separate convergence checks;
    no event normalization or fixed distance/time correction is applied.
    """
    began = perf_counter()
    if (time_origin_ns is None) != (relative_edges_ns is None):
        raise ValueError('time origins and bin edges must be provided together')
    prepared = _prepare_ballistic_interval(kernel, source, config=config,
                                          spectral_knots_nm=spectral_knots_nm)
    if time_origin_ns is None and relative_edges_ns is None:
        result = prepared.charge()
    else:
        result = prepared.bins(np.arange(len(kernel.detector)), time_origin_ns, relative_edges_ns)
    result.metadata['elapsed_seconds'] = perf_counter()-began
    return result


def ballistic_interval(kernel, source, *, config=None, time_origin_ns=None,
                       relative_edges_ns=None, spectral_knots_nm=None):
    """Fast finite-cone response with centre-ray extinction and arrival clock.

    The supplied surface sets the finite cone window at every distance. This
    preserves the existing fast route. ``first_entry_ballistic`` evaluates the
    full first-entry correction explicitly; its event cost is still higher.
    """
    return _ballistic_interval_leading(kernel, source, config=config,
        time_origin_ns=time_origin_ns, relative_edges_ns=relative_edges_ns,
        spectral_knots_nm=spectral_knots_nm)
