"""Continuous-spectrum direct light, split at geometric and time-bin roots.

The medium is exactly its public piecewise-linear table. The detector's
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
    ``finite_receiver_m``: elements closer than this to an OM centre light
    the OM sphere whose cross-section is its ``effective_area_m2`` (the OM of
    the photon Monte Carlo), farther ones its centre. ``0`` is the point
    receiver everywhere, ``inf`` the sphere everywhere.
    """
    order: int = 8
    max_order: int = 64
    relative_tolerance: float = 2e-6
    absolute_tolerance_pe: float = 1e-10
    efficiency_relative_tolerance: float = 1e-8
    efficiency_absolute_tolerance: float = 1e-12
    efficiency_max_depth: int = 16
    pair_list: bool = True
    finite_receiver_m: float = 20.

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
        if not self.finite_receiver_m >= 0:
            raise ValueError('finite_receiver_m must be nonnegative')


@dataclass
class BallisticIntervalResult:
    """Direct charge per OM, optional bins, and the quadrature history."""
    charge_pe: np.ndarray
    bins_pe: np.ndarray | None
    metadata: dict


def prepare_optics(kernel, config=None, *, spectral_knots_nm=None):
    """Construct numerical panels and check their spectral-efficiency error.

    Cubics use ordinary ascending powers of normalized panel coordinate.
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
    edges = np.unique(np.r_[low, high, medium.wavelength_nm, extra,
                            np.linspace(low, high, max(2, int(np.ceil((high-low)/20))+1))])
    edges = edges[(edges >= low) & (edges <= high)]
    nodes = np.array([0., 1/3, 2/3, 1.])
    probes = np.array([1/8, 1/6, 3/8, .5, 5/8, 5/6, 7/8])
    inverse = np.linalg.inv(np.polynomial.polynomial.polyvander(nodes, 3))
    checks = []
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


def ballistic_interval(kernel, source, *, config=None, time_origin_ns=None,
                       relative_edges_ns=None, spectral_knots_nm=None):
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
    tables = prepare_optics(kernel, config, spectral_knots_nm=spectral_knots_nm)
    prepare_seconds = perf_counter()-began
    detector = kernel.detector
    positions = np.ascontiguousarray(detector.positions_m, float)
    looks = np.ascontiguousarray(-detector.orientations, float)
    area = np.ascontiguousarray(detector.effective_area_m2, float)
    radii = np.sqrt(area / np.pi)
    reach = float(config.finite_receiver_m)
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
                                                    arrays[3], arrays[5], tables['phase_nodes'])
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
             finite_receiver_m=reach,
             relative_tolerance=config.relative_tolerance, absolute_tolerance_pe=config.absolute_tolerance_pe))
