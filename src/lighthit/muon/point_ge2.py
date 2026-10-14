"""Exact separable >=2 contraction of point sources.

Every point has real angular coefficients times exp(i omega t). The angular
vector is rotated once, keeping the radial RTE kernel and frequencies; there is
no truncation beyond the supplied L/M and no new spatial basis. The BLAS
backend (:mod:`.point_ge2_blas`) gives the same result up to summation order.
"""
import numpy as np
from numba import njit,prange
from .._directional_numba import _wigner_into,_harmonics_into as _detector_harmonics,_start_norm_table,_parity

@njit(cache=True,nogil=True,parallel=True)
def _apply_points(points, receivers, looks, angular, times, emitting, out_scale, starts, degree, max_m, max_out, alpha, pair_lambda, pair_mu,
                 log_r, coefficients, absorption, velocity, omega, flight, start_norm,
                 jmax, radial_low, radial_high):
    """Fused wavelength-dependent sources on a shared spatial point set, order >=2.

    ``coefficients``: (lambda, radial interval, degree, pair, cubic power,
    frequency).  Returns (receiver, frequency) complex.
    """
    nfreq = len(omega)
    nrec = receivers.shape[0]
    nlam = coefficients.shape[0]
    npair = len(pair_lambda)
    width = 2 * max_out + 1
    result = np.zeros((nrec, nfreq), np.complex128)
    root2 = np.sqrt(2.0)
    for index in prange(nrec):
        wigner = np.empty((degree + 1, width, max_m + 1), np.float64)
        detector = np.empty((max_out + 1) * (max_out + 1), np.float64)
        erow = np.empty((width, 2 * max_m + 1), np.float64)
        rotated = np.empty((nlam, width), np.float64)
        cosines = np.empty(max_m + 1, np.float64)
        sines = np.empty(max_m + 1, np.float64)
        scale_phase = np.empty((nlam, nfreq), np.complex128)
        response = np.empty((nlam, nfreq), np.complex128)
        active_bands = np.empty(nlam, np.bool_)
        for cell in range(points.shape[0]):
            dx = receivers[index, 0] - points[cell, 0]
            dy = receivers[index, 1] - points[cell, 1]
            dz = receivers[index, 2] - points[cell, 2]
            radius = np.sqrt(dx * dx + dy * dy + dz * dz)
            if radius < radial_low or radius > radial_high:
                continue
            any_active = False
            for lam in range(nlam):
                active_bands[lam] = emitting[lam, cell] and out_scale[lam, index] != 0.
                any_active |= active_bands[lam]
            if not any_active:
                continue
            cosine = dz / radius
            if cosine > 1.0:
                cosine = 1.0
            elif cosine < -1.0:
                cosine = -1.0
            phi = np.arctan2(dy, dx)
            transverse = np.sqrt(dx * dx + dy * dy)
            sine = transverse / radius
            cp = 1.0 if transverse == 0.0 else dx / transverse
            sp = 0.0 if transverse == 0.0 else dy / transverse
            for n in range(max_m + 1):
                cosines[n] = np.cos(n * phi)
                sines[n] = np.sin(n * phi)
            lx = looks[index, 0]
            ly = looks[index, 1]
            lz = looks[index, 2]
            a1 = cosine * cp * lx + cosine * sp * ly - sine * lz
            a2 = -sp * lx + cp * ly
            a3 = sine * cp * lx + sine * sp * ly + cosine * lz
            _detector_harmonics(a1, a2, a3, max_out, detector)
            _wigner_into(degree, max_out, max_m, cosine, start_norm, jmax, wigner)
            logr = np.log(radius)
            j = np.searchsorted(log_r, logr, side='right') - 1
            if j < 0:
                j = 0
            if j > len(log_r) - 2:
                j = len(log_r) - 2
            t = logr - log_r[j]
            for lam in range(nlam):
                if not active_bands[lam]:
                    continue
                scale = (out_scale[lam, index] * np.exp(-absorption[lam] * radius)
                         / (4 * np.pi * radius * radius))
                for w in range(nfreq):
                    response[lam, w] = 0.0
                    if flight:
                        angle = omega[w] * (times[cell] + radius / velocity[lam])
                        scale_phase[lam, w] = scale * (np.cos(angle) + 1j * np.sin(angle))
                    else:
                        scale_phase[lam, w] = scale * (np.cos(omega[w]*times[cell]) + 1j*np.sin(omega[w]*times[cell]))
            for ell in range(degree + 1):
                for a in range(width):
                    for b in range(2 * max_m + 1):
                        erow[a, b] = 0.0
                erow[max_out, max_m] = wigner[ell, max_out, 0]
                top_n = min(ell, max_m)
                top_p = min(ell, max_out)
                for n in range(1, top_n + 1):
                    base = root2 * _parity(n) * wigner[ell, max_out, n]
                    erow[max_out, max_m + n] = base * cosines[n]
                    erow[max_out, max_m - n] = base * sines[n]
                for p in range(1, top_p + 1):
                    erow[max_out + p, max_m] = (root2 * _parity(p)
                                                * wigner[ell, max_out + p, 0])
                    for n in range(1, top_n + 1):
                        cross = _parity(p) * wigner[ell, max_out + p, n]
                        straight = wigner[ell, max_out - p, n]
                        cc = _parity(n) * (cross + straight)
                        ss = _parity(n) * (cross - straight)
                        erow[max_out + p, max_m + n] = cc * cosines[n]
                        erow[max_out + p, max_m - n] = cc * sines[n]
                        erow[max_out - p, max_m + n] = -ss * sines[n]
                        erow[max_out - p, max_m - n] = ss * cosines[n]
                keep = min(ell, max_m)
                centre = starts[ell] + min(ell, max_m)
                for lam in range(nlam):
                    if not active_bands[lam]:
                        continue
                    for a in range(width):
                        rotated[lam, a] = 0.0
                for a in range(max_out - top_p, max_out + top_p + 1):
                    for b in range(max_m - keep, max_m + keep + 1):
                        factor = erow[a, b]
                        if factor == 0.0:
                            continue
                        column = centre + (b - max_m)
                        for lam in range(nlam):
                            if not active_bands[lam]:
                                continue
                            rotated[lam, a] += factor * angular[lam, cell, column]
                for lam in range(nlam):
                    if not active_bands[lam]:
                        continue
                    for p in range(npair):
                        lam_detector = pair_lambda[p]
                        mu_pair = pair_mu[p]
                        if mu_pair > top_p:
                            continue
                        detector_centre = lam_detector * lam_detector + lam_detector
                        positive = alpha[lam_detector] * detector[detector_centre + mu_pair]
                        negative = (alpha[lam_detector] * detector[detector_centre - mu_pair]
                                    if mu_pair else 0.0)
                        angular_weight = (positive * rotated[lam, max_out + mu_pair]
                                          + negative * rotated[lam, max_out - mu_pair])
                        if angular_weight == 0.0:
                            continue
                        for w in range(nfreq):
                            c0 = coefficients[lam, j, ell, p, 0, w]
                            c1 = coefficients[lam, j, ell, p, 1, w]
                            c2 = coefficients[lam, j, ell, p, 2, w]
                            c3 = coefficients[lam, j, ell, p, 3, w]
                            radial = ((c0 * t + c1) * t + c2) * t + c3
                            response[lam, w] += angular_weight * radial
            # Flight/emission phase is common to all angular degrees/pairs.
            for lam in range(nlam):
                if active_bands[lam]:
                    for w in range(nfreq):
                        result[index, w] += response[lam, w] * scale_phase[lam, w]
    return result


def apply_point_spectral(bundle,frame,points_m,angular,times_ns,receivers_m,looks,alpha,out_scale,
                         *,azimuthal_degree=4,radial_window_m=(3.,300.),backend='scalar',
                         blas_precision='float64'):
    """All-time complex spectrum for real angular point weights and clocks.

    Angular shape is (wavelength, point, retained real-harmonic channel).
    The emission clock is shared by wavelengths, as in the G4 source contract.
    Frame/L/M/kernel/radial approximations are exactly those supplied by caller.
    """
    if backend=='blas':
        from .point_ge2_blas import apply_point_spectral_blas
        return apply_point_spectral_blas(bundle,frame,points_m,angular,times_ns,
            receivers_m,looks,alpha,out_scale,azimuthal_degree=azimuthal_degree,
            radial_window_m=radial_window_m,precision=blas_precision)
    if backend!='scalar':
        raise ValueError('point backend must be scalar or blas')
    if np.iscomplexobj(angular):
        raise ValueError('point angular coefficients must be real')
    if (not isinstance(azimuthal_degree, (int, np.integer))
            or isinstance(azimuthal_degree, (bool, np.bool_)) or azimuthal_degree < 0):
        raise ValueError('azimuthal_degree must be a nonnegative integer')
    values=np.ascontiguousarray(angular,dtype=float);points=np.asarray(points_m,float)
    times=np.asarray(times_ns,float);receivers=np.atleast_2d(np.asarray(receivers_m,float))
    normals=np.atleast_2d(np.asarray(looks,float));scale=np.ascontiguousarray(out_scale,dtype=float)
    if (times.ndim!=1 or points.ndim!=2 or points.shape[1]!=3
            or receivers.ndim!=2 or receivers.shape[1]!=3):
        raise ValueError('points and receivers require three coordinates; clocks must be one-dimensional')
    degree=int(bundle['degree']);max_m=min(degree,int(azimuthal_degree))
    starts=np.r_[0,np.cumsum([2*min(l,max_m)+1 for l in range(degree+1)])].astype(np.int64)
    nlam=len(bundle['absorption']);nfreq=len(bundle['omega'])
    if (values.shape!=(nlam,len(points),int(starts[-1])) or points.shape!=(len(times),3)
            or normals.shape!=receivers.shape or scale.shape!=(nlam,len(receivers))):
        raise ValueError('incompatible point/spectral/angular/detector shapes')
    if not all(np.isfinite(a).all() for a in [values,points,times,receivers,normals,scale]):
        raise ValueError('point contraction requires finite values')
    if not len(points):return np.zeros((nfreq,len(receivers)),complex)
    low,high=map(float,radial_window_m)
    if (not np.isfinite([low,high]).all() or low<np.exp(bundle['log_r'][0])-1e-9
            or high>np.exp(bundle['log_r'][-1])+1e-9 or not low<high):
        raise ValueError('radial window outside cache')
    max_out=int(bundle['acceptance_degree']);jmax=max(max_m,max_out)
    alpha=np.ascontiguousarray(np.asarray(alpha,float)[:max_out+1])
    if alpha.shape!=(max_out+1,) or not np.isfinite(alpha).all():
        raise ValueError('finite acceptance coefficients must cover the detector degree')
    centre=np.asarray(frame.centre_m);reference=float(times.min())
    response=_apply_points(np.ascontiguousarray(frame.rotate(points-centre)),
        np.ascontiguousarray(frame.rotate(receivers-centre)),np.ascontiguousarray(frame.rotate(normals)),
        values,np.ascontiguousarray(times-reference),np.ascontiguousarray(np.any(values!=0,axis=2)),
        scale,starts,degree,max_m,max_out,alpha,bundle['pair_lambda'],bundle['pair_mu'],bundle['log_r'],
        bundle['coefficients'],bundle['absorption'],bundle['speed'],bundle['omega'],bundle['phase']=='flight',
        _start_norm_table(jmax),jmax,low,high)
    return response.T*np.exp(1j*np.asarray(bundle['omega'])*reference)[:,None]
