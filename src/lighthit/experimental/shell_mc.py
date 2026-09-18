"""Independent positive track-length Monte Carlo for an isotropic flash.

Measures volume-averaged scalar fluence/current in thin spherical shells,
NOT a first-entry probability and NOT a point estimate. All recrossings count.
Absorption is an implicit weight exp(-mu_a*path), so one set of trajectories
can test several absorption coefficients. No RTE matrices or Bessel integrals.
Numba is optional for the package, required only for this experimental module.
"""
import numpy as np
from scipy.special import roots_legendre
try:
    from numba import njit
except ImportError as exc:
    raise ImportError("shell_mc requires pip install -e '.[accelerate]'") from exc


@njit(cache=True)
def _walk(nphotons, seed, mu_s, g, absorptions, radii, widths, max_path, node, weight):
    np.random.seed(seed)
    out = np.zeros((len(absorptions), len(radii), 3, 5))
    for photon in range(nphotons):
        x = np.zeros(3)
        # Shell-integrated observables are rotationally invariant. Fixing
        # the initial direction is equivalent to averaging isotropic emission.
        s = np.array([0.0, 0.0, 1.0])
        path = 0.0
        order = 0
        while path < max_path:
            flight = min(-np.log(max(np.random.random(), 1e-300))/mu_s, max_path-path)
            dot = x[0]*s[0]+x[1]*s[1]+x[2]*s[2]
            impact2 = max(x[0]**2+x[1]**2+x[2]**2-dot**2, 0.0)
            for ir in range(len(radii)):
                inner = radii[ir]-widths[ir]/2
                outer = radii[ir]+widths[ir]/2
                if impact2 >= outer*outer:
                    continue
                root = np.sqrt(outer*outer-impact2)
                lo = max(0.0, -dot-root)
                hi = min(flight, -dot+root)
                if hi <= lo:
                    continue
                bounds = np.empty((2, 2))
                if impact2 < inner*inner:
                    root_in = np.sqrt(inner*inner-impact2)
                    bounds[0, 0], bounds[0, 1] = lo, min(hi, -dot-root_in)
                    bounds[1, 0], bounds[1, 1] = max(lo, -dot+root_in), hi
                else:
                    bounds[0, 0], bounds[0, 1] = lo, hi
                    bounds[1, 0], bounds[1, 1] = 0.0, 0.0
                for interval in range(2):
                    left, right = bounds[interval, 0], bounds[interval, 1]
                    if right <= left:
                        continue
                    for q in range(len(node)):
                        a = left+(right-left)*node[q]
                        length = path+a
                        mu = (dot+a)/np.sqrt(impact2+(dot+a)**2)
                        for ia in range(len(absorptions)):
                            w = (right-left)*weight[q]*np.exp(-absorptions[ia]*length)
                            o = min(order, 2)
                            out[ia, ir, o, 0] += w
                            out[ia, ir, o, 1] += w*mu
                            out[ia, ir, o, 2] += w*mu*mu
                            out[ia, ir, o, 3] += w*length
                            out[ia, ir, o, 4] += w*mu*length
            x += flight*s
            path += flight
            if path >= max_path:
                break
            u = np.random.random()
            if abs(g) < 1e-12:
                c = 2*u-1
            else:
                z = (1-g*g)/(1-g+2*g*u)
                c = (1+g*g-z*z)/(2*g)
            c = min(1.0, max(-1.0, c))
            st = np.sqrt(max(0.0, 1-c*c))
            phi = 2*np.pi*np.random.random()
            # Orthonormal tangent frame, robust at the poles.
            if abs(s[2]) < 0.9:
                e1 = np.array([-s[1], s[0], 0.0])
            else:
                e1 = np.array([0.0, -s[2], s[1]])
            e1 /= np.sqrt((e1*e1).sum())
            e2 = np.array([s[1]*e1[2]-s[2]*e1[1], s[2]*e1[0]-s[0]*e1[2],
                           s[0]*e1[1]-s[1]*e1[0]])
            s = c*s+st*(np.cos(phi)*e1+np.sin(phi)*e2)
            order += 1
    for ir in range(len(radii)):
        volume = 4*np.pi/3*((radii[ir]+widths[ir]/2)**3-(radii[ir]-widths[ir]/2)**3)
        out[:, ir, :, :] /= nphotons*volume
    return out


def shell_estimate(*, scattering_per_m=0.022, g=0.9, absorptions=(0.02, 0.07),
                   radii_m=(20.0, 80.0), widths_m=0.5, photons_per_batch=4000,
                   batches=16, max_path_m=1000.0, seed=1731, quadrature_order=8):
    """Return batch estimates. Axes: batch, absorption, radius, order, observable.

    Observables: (1, radial_cosine, radial_cosine**2, path, cosine*path).
    Missing tracks beyond max_path must be checked by a second max_path run.
    """
    aa=np.atleast_1d(np.asarray(absorptions,float)); rr=np.atleast_1d(np.asarray(radii_m,float))
    ww=np.broadcast_to(np.asarray(widths_m,float), rr.shape).copy()
    if (not np.isfinite(aa).all() or not np.isfinite(rr).all() or not np.isfinite(ww).all()
        or np.any(aa<=0) or np.any(rr<=0) or np.any(ww<=0) or np.any(ww>=2*rr)):
        raise ValueError("Invalid absorption or shell geometry")
    if not np.isfinite(scattering_per_m) or scattering_per_m<=0 or not -1<g<1:
        raise ValueError("Require positive scattering and -1<g<1")
    if not np.isfinite(max_path_m) or max_path_m<=np.max(rr+ww/2):
        raise ValueError("max_path_m must extend beyond every shell")
    for value in (photons_per_batch,batches,quadrature_order):
        if not isinstance(value,(int,np.integer)) or value<2:
            raise ValueError("Batch sizes and quadrature order must be integers >=2")
    x,w=roots_legendre(quadrature_order); x=(x+1)/2; w=w/2
    return np.array([_walk(photons_per_batch,seed+b,scattering_per_m,g,aa,rr,ww,
                           max_path_m,x,w) for b in range(batches)])


def ratio_with_standard_error(batch_data, numerator=1):
    """Ratio of ensemble means, with paired-batch delta-method standard error."""
    q=batch_data[...,0]; f=batch_data[...,numerator]
    qm=q.mean(axis=0); fm=f.mean(axis=0)
    ratio=fm/qm
    residual=f-ratio[None,...]*q
    se=residual.std(axis=0,ddof=1)/(np.sqrt(len(q))*qm)
    return ratio,se
