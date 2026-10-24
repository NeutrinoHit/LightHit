"""Analytic first path/curvature moments on each finite emission window.

At fixed wavelength the uniform arc has mean depth pi/4 times its
meridional half chord. Integrating that depth and the odd curvature term
over the actual source window gives the three elementary moments below.
Only the spectral coordinate remains numerical. The exact entry response
is recovered by adding its bounded residual, without a fixed path shift.
"""
import math

import numpy as np
from numba import njit

from ._ballistic_interval_numba import _acceptance, _insert_cuts, _phase_order, _precheck


from ._surface_ballistic_moments import window_moments


@njit(cache=True)
def spectral_integral(a, b, left, width, n0, dn, e0, de, efficiency,
                      beta, rho, transverse_look, axial_look, coefficient0,
                      coefficient2, length, area, alpha, gx, gw, z, radius):
    total = 0.
    for j in range(len(gx)):
        lam = (a+b)/2+(b-a)/2*gx[j]
        q = (lam-left)/width
        n, mu = n0+dn*q, e0+de*q
        bn = beta*n
        if bn <= 1.:
            continue
        v = math.sqrt((bn-1)*(bn+1))
        sine, cosine = v/bn, 1/bn
        centre, spread = z-rho/v, radius/sine
        m0, m1, m2 = window_moments(-centre/spread, (length-centre)/spread)
        if m0 == 0.:
            continue
        # The odd moment depends on the actual partial emission window.
        # No symmetric-window mean depth is substituted for these moments.
        mass = m0+(mu*radius*cosine/sine+cosine*radius/(2*rho))*m1+mu*radius*math.pi/4*m2
        response = ((efficiency[3]*q+efficiency[2])*q+efficiency[1])*q+efficiency[0]
        photons = coefficient0+coefficient2/(n*n)
        if response < -1e-12 or photons < -1e-12*max(1., coefficient0):
            raise ValueError('negative ballistic spectral weight')
        exponent = -mu*rho/sine
        if len(efficiency) > 4:
            exponent += efficiency[4]+efficiency[5]*q
        angle = min(1., max(-1., transverse_look*sine+axial_look*cosine))
        total += (gw[j]*max(0., response)*max(0., photons)/(lam*lam)
                  *math.exp(exponent)*_acceptance(alpha, angle)
                  /(2*math.pi*rho*sine)*(2/math.pi)*mass)
    return total*(b-a)/2*area/length


@njit(cache=True)
def prepare_panels(po, pe, receivers, looks, radii, start, direction, length,
                   coefficient0, beta, panel_edges, phase_nodes):
    """Order-independent geometry and exact optical-window branch cuts."""
    geometry = np.empty((len(po), 4))
    records = []
    low_n, high_n, monotonic, increasing, ordered = _phase_order(phase_nodes)
    for pair in range(len(po)):
        om, i = po[pair], pe[pair]
        ok, dx, dy, dz, ux, uy, uz, z, rho, nmin, nmax, cut0, cut1, radius = _precheck(
            om, i, receivers, radii, np.inf, start, direction, length,
            coefficient0, beta, low_n, high_n)
        if not ok:
            continue
        across = ((dx-z*ux)*looks[om, 0]+(dy-z*uy)*looks[om, 1]
                  +(dz-z*uz)*looks[om, 2])/rho
        axial = ux*looks[om, 0]+uy*looks[om, 1]+uz*looks[om, 2]
        geometry[pair] = np.array([rho, z, across, axial])
        first, last = 0, len(panel_edges)-1
        if monotonic:
            lower, upper = (nmin, nmax) if increasing else (-nmax, -nmin)
            first = max(0, np.searchsorted(ordered, lower, side='left')-2)
            last = min(last, np.searchsorted(ordered, upper, side='right')+1)
        for panel in range(first, last):
            left, right = panel_edges[panel], panel_edges[panel+1]
            width = right-left
            n0, dn = phase_nodes[panel], phase_nodes[panel+1]-phase_nodes[panel]
            a, b = left, right
            if dn == 0.:
                if n0 < nmin or n0 >= nmax or beta[i]*n0 <= 1.:
                    continue
            else:
                p, q = (nmin-n0)/dn, (nmax-n0)/dn
                a, b = left+width*max(0., min(p, q)), left+width*min(1., max(p, q))
                if a >= b:
                    continue
            cuts = np.empty(4)
            cuts[0], cuts[1] = a, b
            count = _insert_cuts(cuts, 2, left, width, n0, dn, cut0, cut1)
            for piece in range(count-1):
                if cuts[piece] < cuts[piece+1]:
                    records.append((pair, panel, cuts[piece], cuts[piece+1]))
    pairs = np.empty(len(records), np.int64)
    panels = np.empty(len(records), np.int64)
    bounds = np.empty((len(records), 2))
    for j in range(len(records)):
        pairs[j], panels[j] = records[j][0], records[j][1]
        bounds[j, 0], bounds[j, 1] = records[j][2], records[j][3]
    return geometry, pairs, panels, bounds


@njit(cache=True)
def evaluate(po, pe, geometry, record_pair, record_panel, bounds, count_om,
             areas, radii, length, coefficient0, coefficient2, beta, panel_edges,
             phase_nodes, extinction_nodes, efficiency_coefficients, alpha, gx, gw, panel_charge):
    charge = np.zeros(count_om)
    for record in range(len(record_pair)):
        pair, panel = record_pair[record], record_panel[record]
        om, i = po[pair], pe[pair]
        rho, z, across, axial = geometry[pair]
        left, right = panel_edges[panel], panel_edges[panel+1]
        value = spectral_integral(bounds[record, 0], bounds[record, 1], left, right-left,
            phase_nodes[panel], phase_nodes[panel+1]-phase_nodes[panel],
            extinction_nodes[panel], extinction_nodes[panel+1]-extinction_nodes[panel],
            efficiency_coefficients[panel], beta[i], rho, across, axial,
            coefficient0[i], coefficient2[i], length[i], areas[om], alpha, gx, gw, z, radii[om])
        charge[om] += value
        if len(panel_charge):
            column = panel if panel_charge.shape[1] > 1 else 0
            panel_charge[pair, column] += value
    return charge


def prepare(kernel, source, tables, pairs):
    d = kernel.detector
    geometry, rows, panels, bounds = prepare_panels(*pairs, d.positions_m, -d.orientations,
        d.receiver_surface.radii(len(d)), source.start_m, source.direction, source.length_m,
        source.coefficient0, source.beta, tables['panel_edges'], tables['phase_nodes'])
    return dict(om=pairs[0], element=pairs[1], geometry=geometry,
                record_pair=rows, record_panel=panels, bounds=bounds)


def response(kernel, source, tables, pairs, order, *, prepared=None, receiver_mask=None, panel_charge=None):
    d = kernel.detector
    p = prepare(kernel, source, tables, pairs) if prepared is None else prepared
    rows, panels, bounds = p['record_pair'], p['record_panel'], p['bounds']
    if receiver_mask is not None:
        keep = np.asarray(receiver_mask)[p['om'][rows]]
        rows, panels, bounds = rows[keep], panels[keep], bounds[keep]
    if panel_charge is None:
        panel_charge = np.empty((0, 0))
    elif receiver_mask is None:
        panel_charge[:] = 0.
    else:
        panel_charge[np.asarray(receiver_mask)[p['om']]] = 0.
    return evaluate(p['om'], p['element'], p['geometry'], rows, panels, bounds, len(d),
        d.effective_area_m2, d.receiver_surface.radii(len(d)), source.length_m,
        source.coefficient0, source.coefficient2, source.beta, tables['panel_edges'],
        tables['phase_nodes'], tables['extinction_nodes'], tables['efficiency_coefficients'],
        kernel.acceptance()['alpha'], *np.polynomial.legendre.leggauss(order), panel_charge)
