"""Prepared first-entry direct response with bounded geometric residuals.

Analytic path moments for charge and a finite-disk arrival CDF are control variates.
Quarter of the requested tolerance is allocated to their quadrature, quarter
to omission and reused-panel error estimates, and half to residual quadrature.
These budgets are independent of event normalization and receiver selection.
"""
from copy import copy
from time import perf_counter

import numpy as np

from .ballistic_interval import (
    BallisticIntervalResult, _ballistic_interval_leading, prepare_optics,
)
from .receiver import SphereSurface
from .sources import SpectralLightElements


class PreparedBallistic:
    def __init__(self, kernel, source, config, spectral_knots_nm=None):
        began = perf_counter()
        if not isinstance(source, SpectralLightElements) or source.cone_model != 'spectral':
            raise ValueError('continuous ballistic spectrum requires spectral LightElements')
        if not np.allclose(source.coefficient2, -source.coefficient0/source.beta**2,
                           rtol=1e-10, atol=1e-20):
            raise ValueError('source must have the physical Cherenkov C0/C2/beta relation')
        self.kernel, self.source, self.config = kernel, source, config
        self.knots = spectral_knots_nm
        self.surface = (config.receiver_mode != 'point'
                        and isinstance(kernel.detector.receiver_surface, SphereSurface))
        if config.receiver_mode == 'surface' and not self.surface:
            raise ValueError('surface mode needs detector.receiver_surface geometry')
        self.cached_charge = None
        self.panel_charge = self.panel_error = self.panel_support = None
        if not self.surface:
            return
        from ._ballistic_interval_numba import interval_candidates
        from ._surface_ballistic_bounds import angular_bounds, efficiency_upper, full_charge_bounds
        self.tables = prepare_optics(kernel, config, spectral_knots_nm=spectral_knots_nm,
                                     exact_response=True)
        d = kernel.detector
        self.radii = d.receiver_surface.radii(len(d))
        self.pairs = interval_candidates(d.positions_m, self.radii, np.inf,
            source.start_m, source.direction, source.length_m, source.coefficient0,
            source.beta, self.tables['phase_nodes'], True)
        po, pe = self.pairs
        w = d.positions_m[po] - source.start_m[pe]
        z = np.einsum('ij,ij->i', w, source.direction[pe])
        rho = np.linalg.norm(w-z[:, None]*source.direction[pe], axis=1)
        along = np.maximum(np.maximum(-z, z-source.length_m[pe]), 0.)
        if np.any(np.hypot(rho, along) <= self.radii[po]):
            raise ValueError('emitting segment intersects the glass boundary')
        self.regular = rho > self.radii[po]
        if np.any(np.diff(self.tables['phase_nodes']) == 0.):
            self.regular[:] = False
        self.angular_bounds = angular_bounds(kernel.acceptance()['alpha'])
        self.eta_upper = efficiency_upper(self.tables['efficiency_coefficients'])
        self.candidate_count = len(po)
        upper = full_charge_bounds(po, pe, d.positions_m, d.effective_area_m2,
            self.radii, source.start_m, source.direction, source.length_m,
            source.coefficient0, source.beta, self.tables['panel_edges'],
            self.tables['phase_nodes'], self.tables['extinction_nodes'],
            self.eta_upper, self.angular_bounds[0])
        keep = np.ones(len(po), bool)
        self.omitted_full = np.zeros(len(d))
        # interval_candidates orders pairs by receiver; do not scan the
        # entire list once for each of thousands of receivers.
        counts = np.bincount(po, minlength=len(d))
        boundaries = np.r_[0, np.cumsum(counts)]
        for om in np.flatnonzero(counts):
            ix = np.arange(boundaries[om], boundaries[om+1])
            ix = ix[np.argsort(upper[ix])]
            skip = ix[np.cumsum(upper[ix]) <= config.absolute_tolerance_pe/8]
            keep[skip] = False
            self.omitted_full[om] = upper[skip].sum()
        self.pairs = po[keep], pe[keep]
        self.regular = self.regular[keep]
        from ._memory import available_bytes
        panel_count = len(self.tables['panel_edges'])-1
        # Live charge/error matrices, refinement copies and masks. Disable
        # reuse when the required working storage exceeds available memory.
        self.cache_bytes = len(self.pairs[0])*panel_count*(
            10*np.dtype(float).itemsize+2*np.dtype(bool).itemsize)
        self.cache_panels = self.cache_bytes <= available_bytes()
        self.prepare_seconds = perf_counter()-began

    def tolerance(self, q):
        return self.config.absolute_tolerance_pe + self.config.relative_tolerance*np.abs(q)

    def charge(self):
        if self.cached_charge is None:
            if self.surface:
                self.cached_charge = self._calculate(None, None, None)
            else:
                self.cached_charge = _ballistic_interval_leading(self.kernel, self.source,
                    config=self.config, spectral_knots_nm=self.knots)
        return self.cached_charge

    def bins(self, ids, origins, edges):
        ids = np.asarray(ids, dtype=np.int64)
        origins, edges = np.asarray(origins, float), np.asarray(edges, float)
        if (origins.shape != (len(ids),) or not np.isfinite(origins).all()
                or edges.ndim != 1 or len(edges) < 2 or not np.isfinite(edges).all()
                or np.any(np.diff(edges) <= 0)):
            raise ValueError('invalid time origins or edges')
        if not self.surface:
            local = copy(self.kernel)
            local.detector = self.kernel.detector.subset(ids)
            return _ballistic_interval_leading(local, self.source, config=self.config,
                spectral_knots_nm=self.knots, time_origin_ns=origins, relative_edges_ns=edges)
        return self._calculate(ids, origins, edges)

    def _calculate(self, ids, origins, edges):
        from . import _surface_ballistic_charge as correction_charge
        from . import _surface_ballistic_bins as correction_bins
        from . import _surface_ballistic_panels as disk
        from ._surface_ballistic_bounds import bound_pairs, single_arrival_bins
        from ._surface_ballistic_rays import evaluate as rays
        began = perf_counter()
        want_bins = ids is not None
        kernel = copy(self.kernel)
        if ids is None:
            ids = np.arange(len(kernel.detector))
        else:
            kernel.detector = kernel.detector.subset(ids)
        n = len(ids)
        rows = np.full(len(self.kernel.detector), -1, np.int64)
        rows[ids] = np.arange(n)
        selected = rows[self.pairs[0]] >= 0
        selected_indices = np.flatnonzero(selected)
        fixed_q = np.zeros(n)
        fixed_b = np.zeros((n, len(edges)-1)) if want_bins else None
        reused_error = np.zeros(n)
        reused_count = 0
        reused_panels = 0
        remaining_panels = None
        if want_bins and self.panel_charge is not None:
            global_po, global_pe = self.pairs[0][selected], self.pairs[1][selected]
            slots = single_arrival_bins(rows[global_po], global_pe,
                kernel.detector.positions_m, self.radii[ids], self.source.start_m,
                self.source.direction, self.source.length_m, self.source.beta,
                self.source.start_ns, self.source.end_ns, self.tables['panel_edges'],
                self.tables['phase_nodes'], self.tables['group_nodes'], origins, edges)
            remaining_panels = self.panel_support.copy()
            available = self.tolerance(np.maximum(0., self.cached_charge.charge_pe[ids]
                -self.tolerance(self.cached_charge.charge_pe[ids])))/8
            for om in range(n):
                pair, panel = np.nonzero((rows[global_po] == om)[:, None] & (slots >= 0)
                    & self.panel_support[selected_indices]
                    & (self.panel_charge[selected_indices] >= 0.))
                cached = selected_indices[pair]
                error = self.panel_error[cached, panel]
                order = np.argsort(error)
                order = order[np.cumsum(error[order]) <= available[om]]
                pair, panel, cached = pair[order], panel[order], cached[order]
                remaining_panels[cached, panel] = False
                np.add.at(fixed_b[om], slots[pair, panel], self.panel_charge[cached, panel])
                fixed_q[om] = self.panel_charge[cached, panel].sum()
                reused_error[om] = self.panel_error[cached, panel].sum()
                reused_panels += len(cached)
            complete = ~remaining_panels[selected_indices].any(1)
            selected[selected_indices[complete]] = False
            reused_count = int(complete.sum())
            selected_indices = np.flatnonzero(selected)
        po = rows[self.pairs[0][selected_indices]]
        pe = self.pairs[1][selected_indices]
        regular = self.regular[selected_indices]
        ordering = np.argsort(po, kind='stable')
        po, pe, regular = po[ordering], pe[ordering], regular[ordering]
        selected_indices = selected_indices[ordering]
        regular_pairs = po[regular], pe[regular]
        panel_mask = (remaining_panels[selected_indices[regular]]
                      if remaining_panels is not None else None)
        d, s, t = kernel.detector, self.source, self.tables
        radii = self.radii[ids]
        history = []

        def adaptive(evaluate, fraction, label):
            order = self.config.order
            q, b = evaluate(order, np.ones(n, bool))
            pending = np.ones(n, bool)
            while pending.any():
                order = min(2*order, self.config.max_order)
                x, y = evaluate(order, pending)
                error = abs(x-q)
                if want_bins:
                    error = np.maximum(error, abs(y-b).sum(1))
                bad = pending & (error > self.tolerance(x)*fraction)
                history.append(dict(part=label, order=order, receivers=int(pending.sum()),
                                    failed=int(bad.sum()), maximum_change_pe=float(error[pending].max(initial=0.))))
                q[pending] = x[pending]
                if want_bins:
                    b[pending] = y[pending]
                if bad.any() and order == self.config.max_order:
                    raise ValueError(f'ballistic {label} quadrature failed for {bad.sum()} OMs at order {order}')
                pending = bad
            return q, b

        if want_bins:
            prepared = disk.prepare(kernel, s, origins, edges, tables=t, pairs=regular_pairs,
                                    panel_mask=panel_mask)
            def baseline(order, mask):
                q, b, _ = disk.response(kernel, s, origins, edges, order,
                                        prepared=prepared, receiver_mask=mask)
                return q+fixed_q, b+fixed_b
            q, b = adaptive(baseline, .25, 'disk')
        else:
            from ._surface_ballistic_linear import response as linear_charge, prepare as prepare_linear
            prepared = prepare_linear(kernel, s, t, regular_pairs)
            columns = len(t['panel_edges'])-1 if self.cache_panels else 1
            linear_values = np.zeros((len(regular_pairs[0]),columns))
            linear_errors = np.zeros_like(linear_values)
            def baseline(order, mask):
                previous = linear_values.copy()
                result = linear_charge(kernel, s, t, regular_pairs, order,
                    prepared=prepared, receiver_mask=mask, panel_charge=linear_values)
                use = mask[regular_pairs[0]]
                linear_errors[use] = abs(linear_values[use]-previous[use])
                return result, None
            q, b = adaptive(baseline, .25, 'linear-path-moments')

        # The noninvertible geometry is integrated directly, with identical
        # entry distances and positive wavelength/source/azimuth quadrature.
        fallback = ~regular
        fallback_origins = origins if want_bins else np.zeros(n)
        fallback_edges = edges if want_bins else np.array([-1e100, 1e100])
        def direct(order, mask, pair_mask=fallback):
            use = pair_mask & mask[po]
            return rays(po[use], pe[use], d.positions_m, -d.orientations,
                d.effective_area_m2, radii, s.start_m, s.direction, s.length_m,
                s.coefficient0, s.coefficient2, s.beta, s.start_ns, s.end_ns,
                t['panel_edges'], t['phase_nodes'], t['group_nodes'], t['extinction_nodes'],
                t['efficiency_coefficients'], kernel.acceptance()['alpha'],
                *np.polynomial.legendre.leggauss(order), fallback_origins, fallback_edges)
        if fallback.any():
            x, y = adaptive(direct, .25, 'rays')
            q += x
            if want_bins:
                b += y

        bound = bound_pairs(*regular_pairs, d.positions_m, -d.orientations,
            d.effective_area_m2, radii, s.start_m, s.direction, s.length_m, s.coefficient0,
            s.beta, t['panel_edges'], t['phase_nodes'], t['extinction_nodes'],
            self.eta_upper, self.angular_bounds, want_bins, not want_bins)
        rp = regular_pairs[0]
        keep = np.ones(len(rp), bool)
        omitted = np.zeros(n)
        residual_values = np.zeros_like(linear_values) if not want_bins else None
        residual_errors = np.zeros(len(rp))
        if not want_bins:residual_errors=np.zeros_like(linear_values)
        for om in range(n):
            ix = np.arange(np.searchsorted(rp, om), np.searchsorted(rp, om, side='right'))
            ix = ix[np.argsort(bound[ix])]
            skip = ix[np.cumsum(bound[ix]) <= self.tolerance(q[om])/4
                      -self.omitted_full[ids[om]]-reused_error[om]]
            keep[skip] = False
            omitted[om] = bound[skip].sum()

        def correction(mask, source_order, impact_order, phi_order, measure):
            pairs = regular_pairs[0][mask], regular_pairs[1][mask]
            if want_bins:
                x, y, _ = correction_bins.response(kernel, s, origins, edges,
                    source_order, impact_order, phi_order, measure, tables=t, pairs=pairs,
                    panel_mask=panel_mask[mask] if panel_mask is not None else None)
                return x, y
            pair_values = np.zeros((int(mask.sum()),columns))
            x, _ = correction_charge.boundary_charge(kernel, s, source_order,
                impact_order, phi_order, measure_source=measure, subtract_leading=True,
                tables=t, pairs=pairs, linear_control=True, panel_charge=pair_values)
            residual_errors[mask] = abs(pair_values-residual_values[mask])
            residual_values[mask] = pair_values
            return x, None

        dq, db = correction(keep, 2, 4, 2, True)
        while True:
            # Use a lower bound after applying the correction: omission must
            # satisfy the tolerance of the corrected charge, not the baseline.
            allowed = (self.tolerance(np.maximum(0., q+dq-omitted))/4
                       -self.omitted_full[ids]-reused_error)
            reopen = np.zeros(len(keep), bool)
            for om in np.flatnonzero(omitted > allowed):
                ix = np.flatnonzero((rp == om) & ~keep)
                ix = ix[np.argsort(-bound[ix])]
                for pair in ix:
                    if omitted[om] <= allowed[om]:
                        break
                    reopen[pair] = True
                    omitted[om] -= bound[pair]
            if not reopen.any():
                break
            x, y = correction(reopen, 2, 4, 2, True)
            dq += x
            if want_bins:
                db += y
            keep |= reopen

        source_order, impact_order, phi_order = 3, min(6, self.config.max_order), 3
        pending = np.ones(n, bool)
        while pending.any():
            x, y = correction(keep & pending[rp], source_order, impact_order, phi_order, False)
            error = abs(x-dq)
            if want_bins:
                error = np.maximum(error, abs(y-db).sum(1))
            bad = pending & (error > self.tolerance(q+x)/2)
            history.append(dict(part='residual', source_order=source_order, impact_order=impact_order,
                azimuth_order=phi_order, receivers=int(pending.sum()), failed=int(bad.sum()),
                maximum_change_pe=float(error[pending].max(initial=0.))))
            dq[pending] = x[pending]
            if want_bins:
                db[pending] = y[pending]
            if bad.any() and impact_order >= self.config.max_order:
                # A difficult residual must not force the whole detector into
                # detailed ray integration. Recompute only the failed OMs,
                # using an independent integration order and the same tolerance.
                ray_mask = bad[po]
                x, y = adaptive(lambda order, mask: direct(order, mask, ray_mask), 1., 'rays-refinement')
                q[bad], dq[bad], omitted[bad] = x[bad]+fixed_q[bad], 0., 0.
                if want_bins:
                    b[bad], db[bad] = y[bad]+fixed_b[bad], 0.
                fallback |= ray_mask
                bad[:] = False
            source_order *= 2
            impact_order = min(impact_order*2, self.config.max_order)
            phi_order *= 2
            pending = bad
        q += dq
        if want_bins:
            b += db
        elif self.cache_panels:
            shape = (len(self.pairs[0]), columns)
            self.panel_charge = np.zeros(shape)
            self.panel_error = np.full(shape, np.inf)
            self.panel_support = np.ones(shape,bool)
            indices = selected_indices[regular]
            self.panel_charge[indices] = linear_values+residual_values
            self.panel_error[indices] = linear_errors+residual_errors+np.where(keep, 0., bound)[:,None]
            self.panel_support[indices] = False
            self.panel_support[indices[prepared['record_pair']],prepared['record_panel']] = True
            self.panel_error[selected_indices[fallback]] = np.inf
        return BallisticIntervalResult(q, b, dict(method='first-entry-control-variate',
            receiver_mode=self.config.receiver_mode, receiver_surface='sphere',
            reference_point='first-entry', source_elements=len(s), source_compressed=False,
            charge_control=('finite-disk' if want_bins else 'linear-path-moments'),
            optical_panels=t['metadata'], quadrature=history, candidate_pairs=self.candidate_count,
            integrated_pairs=len(po), omitted_full_bound_pe=self.omitted_full[ids].tolist(),
            reused_charge_pairs=reused_count, reused_charge_panels=reused_panels,
            reused_panel_error_estimate_pe=reused_error.tolist(), panel_cache_bytes=self.cache_bytes,
            residual_pairs=int(keep.sum()), ray_pairs=int(fallback.sum()),
            prepare_seconds=self.prepare_seconds,
            omitted_residual_bound_pe=omitted.tolist(), elapsed_seconds=perf_counter()-began,
            relative_tolerance=self.config.relative_tolerance,
            absolute_tolerance_pe=self.config.absolute_tolerance_pe))
