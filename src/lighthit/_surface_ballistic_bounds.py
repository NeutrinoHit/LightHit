"""Geometry bounds for the glass-minus-leading direct-charge residual."""
import math
import numpy as np
from numba import njit
from ._ballistic_interval_numba import _support


def angular_bounds(alpha):
    ell=np.arange(len(alpha))
    weight=(2*ell+1)*np.asarray(alpha)/(4*np.pi)
    absolute=np.abs(weight)
    peak=float(absolute.sum())
    first=float((absolute*ell*(ell+1)/2).sum())
    second=float((absolute*(ell-1)*ell*(ell+1)*(ell+2)/8).sum())
    p=np.polynomial.legendre.leg2poly(weight)
    roots=np.polynomial.polynomial.polyroots(np.polynomial.polynomial.polyder(p)) if len(p)>1 else []
    nodes=[-1.,1.]+[float(v.real) for v in roots if abs(v.imag)<1e-12 and -1<v.real<1]
    negative=max(0.,-float(np.polynomial.polynomial.polyval(nodes,p).min()))
    negative+=32*np.finfo(float).eps*max(1.,peak)
    return np.array([peak,first,second,negative])


@njit(cache=True)
def bound_pairs(po,pe,positions,looks,areas,radii,starts,axes,lengths,c0,beta,
                panels,phases,extinction,eta_upper,angular,time_bins=False,linear_control=False):
    out=np.zeros(len(po))
    peak,first,second,negative=angular
    nlo,nhi=phases.min(),phases.max()
    for p in range(len(po)):
        om=po[p];i=pe[p];R=radii[om];L=lengths[i]
        w=positions[om]-starts[i];z=np.dot(w,axes[i]);perp=w-z*axes[i]
        rho=np.linalg.norm(perp)
        if rho<=R:out[p]=np.inf;continue
        valid,nmin,nmax,_,_=_support(z,rho,L,beta[i],R)
        if not valid or nmin>nhi or nmax<=nlo:continue
        e=R/rho;root=math.sqrt((1-e)*(1+e))
        delta_cos=e*e/(1+root)
        jacmax=math.sqrt(2/(1+root))
        ratio2=(2*e-delta_cos)/(2*(1-e))
        shape_tv=2*min(1.,ratio2/(1+math.sqrt(1+ratio2)))
        across=np.dot(perp/rho,looks[om])
        side=np.dot(np.cross(axes[i],perp/rho),looks[om])
        value=0.
        for k in range(len(panels)-1):
            left=panels[k];width=panels[k+1]-left
            n0=phases[k];dn=phases[k+1]-n0
            qa,qb=0.,1.
            if dn==0:
                if n0<nmin or n0>=nmax:continue
            else:
                x0=(nmin-n0)/dn;x1=(nmax-n0)/dn
                qa=max(0.,min(x0,x1));qb=min(1.,max(x0,x1))
                if qa>=qb:continue
            na,nb=n0+dn*qa,n0+dn*qb
            small,large=min(na,nb),max(na,nb)
            cmin=1/(beta[i]*large);cmax=1/(beta[i]*small)
            if cmin>=1:continue
            smax=math.sqrt(max(0.,1-cmin*cmin))
            smin=math.sqrt(max(0.,1-cmax*cmax))
            if smin==0:out[p]=np.inf;break
            ea=extinction[k]+qa*(extinction[k+1]-extinction[k])
            eb=extinction[k]+qb*(extinction[k+1]-extinction[k])
            emin,emax=min(ea,eb),max(ea,eb)
            chi=emax*R/smin
            if chi>math.log(np.finfo(np.float64).max)/2:
                out[p]=np.inf;break
            if e<cmin:
                gmax=1/math.sqrt(1-cmax*e)
            else:
                gmax=math.sqrt(2/((1-cmax*e)+smin*root))
            gmin=1/math.sqrt(1+cmax*e)
            upper=gmax*jacmax*math.exp(chi)
            lower=gmin*math.exp(-chi)
            relative=max(upper-1,1-lower)
            dx=smax*abs(across)*delta_cos;dy=smax*abs(side)*e
            # The two azimuth signs have the same path and Jacobian.
            # Their linear side-direction terms cancel before bounding A.
            response=first*dx+second*(dx*dx+dy*dy)/2+2*negative
            difference=peak*relative+response
            if linear_control:
                if time_bins:
                    raise ValueError('linear path-moment bound is charge only')
                fmin=1-cmax*e
                jac_remainder=(3/8*(cmax*e)**2/fmin**2.5
                    +1/math.sqrt(fmin)*(2/(1+root)-1))
                delta_path=R*smax*e/(fmin*(1+root))
                amplitude=(math.exp(chi)*jac_remainder+cmax*e/2*math.expm1(chi)
                    +math.exp(chi)*(emax*delta_path+chi*chi/2))
                linear_difference=peak*amplitude+(1+cmax*e/2+chi)*response
                # The derivative remainder can be loose close to the glass.
                # The old bound plus the known control increment is another
                # valid bound, so use the tighter one without a distance cut.
                difference=min(linear_difference, difference+peak*(chi+cmax*e/2))
            if time_bins:difference+=peak*shape_tv
            fraction=min(1.,2*R/(L*smin))
            low=left+qa*width;high=left+qb*width
            value+=((1/low-1/high)*(1-cmin*cmin)*eta_upper[k]*fraction
                    *math.exp(-emin*rho/smax)*difference)
        if np.isfinite(out[p]):
            out[p]=c0[i]*areas[om]/(math.pi*R*R)*(R/(math.pi*rho))*value
    return out


def efficiency_upper(coefficients):
    result=[]
    for row in coefficients:
        value=row[0]+sum(max(0.,x) for x in row[1:4])
        if len(row)>4:value*=np.exp(max(row[4],row[4]+row[5]))
        result.append(max(0.,value))
    return np.asarray(result)


@njit(cache=True)
def full_charge_bounds(po, pe, positions, areas, radii, starts, axes, lengths,
                       c0, beta, panels, phases, extinction, eta_upper, peak):
    """Positive first-entry charge bound, also bounding any bin's L1 mass.

    For transverse distance rho>R, every hitting cone has azimuthal
    half-width at most asin(R/rho). Its emission interval has length at
    most 2R/sin(theta). All paths exceed the closest segment distance-R.
    Optical extrema are taken on the actual supported part of each panel.
    """
    out = np.zeros(len(po))
    for pair in range(len(po)):
        om, i = po[pair], pe[pair]
        R, L = radii[om], lengths[i]
        w = positions[om]-starts[i]
        z = np.dot(w, axes[i])
        rho = np.linalg.norm(w-z*axes[i])
        if rho <= R:
            out[pair] = np.inf
            continue
        valid, nmin, nmax, _, _ = _support(z, rho, L, beta[i], R)
        if not valid:
            continue
        axial = max(0., max(-z, z-L))
        minimum_path = math.hypot(rho, axial)-R
        value = 0.
        for panel in range(len(panels)-1):
            left, width = panels[panel], panels[panel+1]-panels[panel]
            n0, dn = phases[panel], phases[panel+1]-phases[panel]
            qa, qb = 0., 1.
            if dn == 0.:
                if n0 < nmin or n0 >= nmax:
                    continue
            else:
                x0, x1 = (nmin-n0)/dn, (nmax-n0)/dn
                qa, qb = max(0., min(x0, x1)), min(1., max(x0, x1))
                if qa >= qb:
                    continue
            na, nb = n0+qa*dn, n0+qb*dn
            small, large = min(na, nb), max(na, nb)
            cmin = 1/(beta[i]*large)
            if cmin >= 1.:
                continue
            smin = math.sqrt(max(0., 1-1/(beta[i]*small)**2))
            fraction = min(1., 2*R/(L*smin)) if smin > 0. else 1.
            ea = extinction[panel]+qa*(extinction[panel+1]-extinction[panel])
            eb = extinction[panel]+qb*(extinction[panel+1]-extinction[panel])
            low, high = left+qa*width, left+qb*width
            value += ((1/low-1/high)*(1-cmin*cmin)*eta_upper[panel]*fraction
                      *math.exp(-min(ea, eb)*minimum_path))
        out[pair] = (c0[i]*areas[om]/(math.pi*R*R)*peak
                     *math.asin(R/rho)/math.pi*value)
    return out


@njit(cache=True)
def single_arrival_bins(po, pe, positions, radii, starts, axes, lengths, beta,
                       t0, t1, panels, phases, groups, origins, edges):
    """Return a bin only when conservative bounds contain every entry ray."""
    slots = np.full((len(po),len(panels)-1), -1, np.int64)
    for pair in range(len(po)):
        om, i = po[pair], pe[pair]
        R, L = radii[om], lengths[i]
        w = positions[om]-starts[i]
        z = np.dot(w, axes[i])
        rho = np.linalg.norm(w-z*axes[i])
        if rho <= R:
            continue
        valid, nmin, nmax, _, _ = _support(z, rho, L, beta[i], R)
        if not valid:
            continue
        axial = max(0., max(-z, z-L))
        nearest = math.hypot(rho, axial)
        farthest = max(math.hypot(rho, z), math.hypot(rho, z-L))
        for k in range(len(panels)-1):
            n0, dn = phases[k], phases[k+1]-phases[k]
            qa, qb = 0., 1.
            if dn == 0.:
                if n0 < nmin or n0 >= nmax:
                    continue
            else:
                a, b = (nmin-n0)/dn, (nmax-n0)/dn
                qa, qb = max(0., min(a, b)), min(1., max(a, b))
                if qa >= qb:
                    continue
            ga = groups[k]+qa*(groups[k+1]-groups[k])
            gb = groups[k]+qb*(groups[k+1]-groups[k])
            earliest = min(t0[i], t1[i])+min(ga,gb)/.299792458*(nearest-R)-origins[om]
            latest = max(t0[i], t1[i])+max(ga,gb)/.299792458*farthest-origins[om]
            slot = np.searchsorted(edges, earliest, side='right')-1
            if 0 <= slot < len(edges)-1 and latest <= edges[slot+1]:
                slots[pair,k] = slot
    return slots
