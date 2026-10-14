"""Exact integrated ballistic charge with geometry shared across wavelengths.

Experimental only. Original elements, spectral Cherenkov cones, endpoint
ownership and directional acceptance are retained. Conservative geometric
rejections precede the identical per-wavelength cone-root calculation.
"""
import numpy as np
from numba import njit

from ..ballistic_fast import _acceptance


@njit(cache=True, fastmath=False)
def _cone_bounds(beta, phase):
    lower = np.full(len(beta), np.inf)
    upper = np.zeros(len(beta))
    for i in range(len(beta)):
        for lam in range(len(phase)):
            if beta[i]*phase[lam] > 1.:
                mu = 1./(beta[i]*phase[lam])
                sine = np.sqrt(max(0., 1.-mu*mu))
                cot = mu/sine
                lower[i] = min(lower[i], cot)
                upper[i] = max(upper[i], cot)
    return lower, upper


@njit(cache=True, fastmath=False)
def _spectral_kernel(receivers, looks, start, direction, length, coefficient0,
                     coefficient2, beta, phase, wavelength, extinction, alpha,
                     lower_cot, upper_cot):
    result = np.zeros((len(receivers), len(phase)))
    eps = 64.*np.finfo(np.float64).eps
    for r in range(len(receivers)):
        for i in range(len(start)):
            if not np.isfinite(lower_cot[i]):
                continue  # Below every spectral Cherenkov threshold.
            dx = receivers[r, 0]-start[i, 0]
            dy = receivers[r, 1]-start[i, 1]
            dz = receivers[r, 2]-start[i, 2]
            ux, uy, uz = direction[i, 0], direction[i, 1], direction[i, 2]
            along = dx*ux+dy*uy+dz*uz
            if along < 0.:
                continue  # Every physical cone cosine is positive.
            dot = dx*dx+dy*dy+dz*dz
            impact2 = max(dot-along*along, 1e-300)
            # Necessary conditions for some root in [0,L). Guard the broad
            # comparisons: actual endpoint ownership is decided below only.
            near_cone2 = impact2*lower_cot[i]*lower_cot[i]
            along2 = along*along
            if along2 < near_cone2-eps*(along2+near_cone2+1.):
                continue
            if along > length[i]:
                behind2 = (along-length[i])**2
                far_cone2 = impact2*upper_cot[i]*upper_cot[i]
                if behind2 > far_cone2+eps*(behind2+far_cone2+1.):
                    continue
            impact = np.sqrt(impact2)
            safe_length = max(length[i], 1e-300)
            for lam in range(len(phase)):
                if beta[i]*phase[lam] <= 1.:
                    continue
                mu = 1./(beta[i]*phase[lam])
                sine = np.sqrt(max(0., 1.-mu*mu))
                # Keep precisely the reference operation order at boundaries.
                root = along-impact*mu/sine
                if root < 0. or root >= length[i]:
                    continue
                distance = impact/sine
                sx = (dx-root*ux)/distance
                sy = (dy-root*uy)/distance
                sz = (dz-root*uz)/distance
                cosine = sx*looks[r, 0]+sy*looks[r, 1]+sz*looks[r, 2]
                cosine = min(1., max(-1., cosine))
                s0 = 1./wavelength[lam]**2
                s2 = s0/phase[lam]**2
                photons = s0*coefficient0[i]+s2*coefficient2[i]
                result[r, lam] += (photons/safe_length
                    *np.exp(-extinction[lam]*distance)/(2.*np.pi*impact*sine)
                    *_acceptance(alpha, cosine))
    return result


def spectral_ballistic_charge(kernel, source):
    """Expected integrated PE, same spectral order-zero formula as prompt.

    Requires SpectralLightElements with wavelength-dependent cones. No time
    bins or source compression are performed. Deliberately uses one compiled
    thread; acceleration comes from geometry reuse and exact root exclusions.
    """
    from ..sources import SpectralLightElements
    from ..prompt import _spectral_tables
    if not isinstance(source, SpectralLightElements) or source.cone_model != 'spectral':
        raise ValueError('requires SpectralLightElements with cone_model="spectral"')
    tables = _spectral_tables(kernel)
    phase = np.ascontiguousarray(tables['phase_index'])
    lower, upper = _cone_bounds(np.ascontiguousarray(source.beta), phase)
    detector = kernel.detector
    raw = _spectral_kernel(
        np.ascontiguousarray(detector.positions_m), np.ascontiguousarray(-detector.orientations),
        np.ascontiguousarray(source.start_m), np.ascontiguousarray(source.direction),
        np.ascontiguousarray(source.length_m), np.ascontiguousarray(source.coefficient0),
        np.ascontiguousarray(source.coefficient2), np.ascontiguousarray(source.beta),
        phase, np.ascontiguousarray(tables['wavelength_nm']), np.ascontiguousarray(tables['mut']),
        np.ascontiguousarray(kernel.acceptance()['alpha']), lower, upper)
    charge = np.zeros(len(detector.positions_m))
    for lam in range(len(phase)):
        charge += (tables['weight_nm'][lam]*tables['efficiency'][lam]
                   *detector.effective_area_m2*raw[:, lam])
    return charge
