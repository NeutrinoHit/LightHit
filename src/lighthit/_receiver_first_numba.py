"""First-entry angular moments used to build finite-receiver tables."""
import math

import numpy as np
from numba import njit


@njit(cache=True)
def _elliptic(m, complementary_m):
    """Complete elliptic K and E by the arithmetic-geometric mean."""
    a, b = 1., math.sqrt(complementary_m)
    weighted, power = m/2, 1.
    while True:
        c = (a-b)/2
        weighted += power*c*c
        a, b = (a+b)/2, math.sqrt(a*b)
        power *= 2
        if abs(c) <= 4*np.finfo(np.float64).eps*a:
            break
    K = math.pi/(2*a)
    return K, K*(1-weighted)


@njit(cache=True)
def _cosine_average(power):
    if power % 2:
        return 0.
    value = 1.
    for j in range(1, power//2+1):
        value *= (2*j-1)/(2*j)
    return value


@njit(cache=True)
def _hg_azimuth(gap, b):
    """Integrals cos(phi)^k/(a-b*cos(phi))^1.5, a=gap+|b|, k=0..3."""
    # Retain the positive minimum denominator separately: a-|b| may round
    # to zero for a narrowly forward-peaked phase function.
    a = gap+abs(b)
    eta = abs(b/a)
    out = np.zeros(4)
    if eta < .2:
        # The elliptic expressions subtract nearly equal numbers here.
        term, n = 1., 0
        while True:
            for k in range(4):
                out[k] += term*_cosine_average(n+k)
            if n > 4 and abs(term) < np.finfo(np.float64).eps:
                break
            term *= eta*(n+1.5)/(n+1)
            n += 1
        out *= math.pi/a**1.5
    else:
        complementary_m = gap/(gap+2*abs(b))
        m = 1-complementary_m
        K, E = _elliptic(m, complementary_m)
        root = math.sqrt(1+eta)
        J0 = 2*E/((gap/a)*root)
        J1 = 2*K/root
        J2 = 2*root*E
        J3 = 2*root**3*(2*(2-m)*E-(1-m)*K)/3
        out[0] = J0
        out[1] = (J0-J1)/eta
        out[2] = (J0-2*J1+J2)/eta**2
        out[3] = (J0-3*J1+3*J2-J3)/eta**3
        out /= a**1.5
    if b < 0:
        out[1] *= -1
        out[3] *= -1
    return out


@njit(cache=True)
def _density(r, cosine, path, g, gx, gw):
    """Ten direction moments per path and collection area; receiver radius 1.

    The pencil points along z; the receiver centre lies in the xz plane.
    Wavelength-dependent scattering and exp(-mu_t*path) are folded later.
    Quadrature nodes must avoid the integrable logarithmic singularity at
    the pencil's direct entry path.
    """
    result = np.zeros(10)
    if path <= r-1:
        return result
    z = r*cosine
    rho = r*math.sqrt((1-cosine)*(1+cosine))
    hit = z > 0 and rho <= 1
    if hit and path >= math.sqrt((r-1)*(r+1)):
        return result
    shift = path-z
    if hit:
        root = math.sqrt(max(0., (1-rho)*(1+rho)))
        entry, exit_path = z-root, z+root
        numerator = (path-entry)*(path-exit_path)
    else:
        numerator = shift*shift+(rho-1)*(rho+1)
    # Integrate the outgoing length directly. Subtracting the first-flight
    # upper limit from path loses its positive remainder near direct entry.
    last_lo = max(0., numerator/(2*(path-(z-1))))
    last_hi = path
    if hit:
        last_lo = max(last_lo, path-entry)
    if shift > 0:
        last_hi = min(last_hi, numerator/(2*shift))
    elif shift < 0:
        last_lo = max(last_lo, numerator/(2*shift))
    elif numerator <= 0:
        return result
    if last_hi <= last_lo:
        return result
    qa, qb = math.log(last_lo), math.log(last_hi)
    panels = max(1, int(math.ceil(qb-qa)))
    for p in range(panels):
        left = qa+(qb-qa)*p/panels
        right = qa+(qb-qa)*(p+1)/panels
        for j in range(len(gx)):
            q = (left+right)/2+(right-left)/2*gx[j]
            last = math.exp(q)
            axial = last-shift
            distance = math.hypot(rho, axial)
            entry_jacobian = numerator-2*shift*last
            if entry_jacobian <= 0:
                continue
            jac = entry_jacobian/(2*distance*last*last)
            delta = (1-(distance-last)**2)/(2*distance*last)
            delta = min(1., max(0., delta))
            c, s = 1-delta, math.sqrt(delta*(2-delta))
            ux, uz = rho/distance, axial/distance
            x0, x1 = c*ux, -s*uz
            z0, z1 = c*uz, s*ux
            angular_gap = (max(0., 1-z0-abs(z1)) if g >= 0
                           else max(0., 1+z0-abs(z1)))
            gap = (1-abs(g))**2+2*abs(g)*angular_gap
            I = _hg_azimuth(gap, 2*g*z1)
            weight = (right-left)/2*gw[j]*last*jac*(1-g*g)/(2*math.pi**2)
            result[0] += weight*I[0]
            result[1] += weight*(x0*I[0]+x1*I[1])
            result[2] += weight*(z0*I[0]+z1*I[1])
            result[3] += weight*(x0*x0*I[0]+2*x0*x1*I[1]+x1*x1*I[2])
            result[4] += weight*(x0*z0*I[0]+(x0*z1+x1*z0)*I[1]+x1*z1*I[2])
            result[5] += weight*(z0*z0*I[0]+2*z0*z1*I[1]+z1*z1*I[2])
            result[6] += weight*(x0**3*I[0]+3*x0*x0*x1*I[1]
                                 +3*x0*x1*x1*I[2]+x1**3*I[3])
            result[7] += weight*(x0*x0*z0*I[0]+(x0*x0*z1+2*x0*x1*z0)*I[1]
                                 +(2*x0*x1*z1+x1*x1*z0)*I[2]+x1*x1*z1*I[3])
            result[8] += weight*(x0*z0*z0*I[0]+(x1*z0*z0+2*x0*z0*z1)*I[1]
                                 +(2*x1*z0*z1+x0*z1*z1)*I[2]+x1*z1*z1*I[3])
            result[9] += weight*(z0**3*I[0]+3*z0*z0*z1*I[1]
                                 +3*z0*z1*z1*I[2]+z1**3*I[3])
    return result


@njit(cache=True)
def cumulative(r, cosine, paths, g, sx, sw, ex, ew):
    """Positive path integration between topology-aware grid endpoints."""
    output = np.zeros((len(paths), 10))
    for k in range(1, len(paths)):
        lo, hi = paths[k-1], paths[k]
        value = np.zeros(10)
        for j in range(len(sx)):
            angle = math.pi/2*sx[j]
            path = (lo+hi)/2+(hi-lo)/2*math.sin(angle)
            if path <= lo:
                path = np.nextafter(lo, hi)
            elif path >= hi:
                path = np.nextafter(hi, lo)
            if path <= lo or path >= hi:
                # There is no representable interior quadrature node.
                continue
            weight = (hi-lo)/2*math.pi/2*math.cos(angle)*sw[j]
            value += weight*_density(r, cosine, path, g, ex, ew)
        output[k] = output[k-1]+value
    return output
