"""Positive point-cone quadrature in normalized impact, with real entry rays."""
import math
import numpy as np
from numba import njit
from ._ballistic_interval_numba import (
    _acceptance, _sort_small, interval_candidates, _phase_order, _support, _supported_panels,
)
from .ballistic_interval import prepare_optics
from ._surface_ballistic_moments import root_moments


@njit(cache=True)
def evaluate(po, pe, positions, looks, areas, radii, starts, axes, lengths, c0, c2, beta,
             panels, phases, extinction, efficiency, alpha, sx, sw, hx, hw, px, pw,
             measure_source, mx, mw, subtract_leading, linear_control, panel_charge):
    output=np.zeros(len(positions));count=0
    cuts=np.empty(6)
    _,_,monotonic,increasing,ordered=_phase_order(phases)
    for pair in range(len(po)):
        if len(panel_charge):panel_charge[pair,:]=0.
        om=po[pair];i=pe[pair];R=radii[om];L=lengths[i]
        w=positions[om]-starts[i];u=axes[i]
        z=np.dot(w,u);transverse=w-z*u;rho=np.linalg.norm(transverse)
        if rho<=R:raise ValueError('impact quadrature requires rho > R')
        er=transverse/rho;ey=np.cross(u,er)
        axial=np.dot(u,looks[om]);across=np.dot(er,looks[om]);side=np.dot(ey,looks[om])
        valid,nmin,nmax,_,_=_support(z,rho,L,beta[i],R)
        if not valid:continue
        first,last=_supported_panels(nmin,nmax,monotonic,increasing,ordered)
        for panel in range(first,last):
            left=panels[panel];width=panels[panel+1]-left
            n0=phases[panel];dn=phases[panel+1]-n0
            if dn==0:raise ValueError('impact quadrature requires nonconstant phase')
            nlo=max(min(n0,n0+dn),1/beta[i]);nhi=max(n0,n0+dn)
            if nhi<=nlo:continue
            clo=1/(beta[i]*nlo);chi=1/(beta[i]*nhi)
            slo=math.sqrt(max(0.,1-clo*clo));shi=math.sqrt(1-chi*chi)
            amin=max(0.,z-(rho*clo+R)/slo) if slo>0 else 0.
            amax=min(L,z-(rho*chi-R)/shi)
            if amin>=amax:continue
            cuts[0]=amin;cuts[1]=amax;ncut=2
            for c,s in ((clo,slo),(chi,shi)):
                if s==0:continue
                for sign in (-1.,1.):
                    a=z-(rho*c+sign*R)/s
                    if amin<a<amax:cuts[ncut]=a;ncut+=1
            _sort_small(cuts,ncut)
            subtotal=0.
            for part in range(ncut-1):
                low,high=cuts[part],cuts[part+1]
                source_x=sx;source_w=sw
                if measure_source:
                    mom=np.zeros(4)
                    for j in range(len(mx)):
                        a=(low+high)/2+(high-low)/2*mx[j]
                        h0=max(-1.,(rho*chi-(z-a)*shi)/R)
                        h1=min(1.,(rho*clo-(z-a)*slo)/R)
                        m=(h1-h0)*root_moments(h0,h1)[0] if h1>h0 else 0.
                        value=m*mw[j]
                        for power in range(4):mom[power]+=value;value*=mx[j]
                    if mom[0]<=0:continue
                    mean=mom[1]/mom[0];var=mom[2]/mom[0]-mean*mean
                    diagonal=(mom[3]/mom[0]-2*mean*mom[2]/mom[0]+mean**3)/var
                    discr=math.sqrt((mean-diagonal)**2+4*var)
                    x0=(mean+diagonal-discr)/2;x1=(mean+diagonal+discr)/2
                    w1=(mean-x0)/(x1-x0);w0=1-w1
                    source_x=np.array([x0,x1]);source_w=np.array([w0,w1])*mom[0]
                for j in range(len(sx)):
                    source=(low+high)/2+(high-low)/2*source_x[j]
                    along=z-source;r=math.hypot(rho,along)
                    ha=max(-1.,(rho*chi-along*shi)/R)
                    hb=min(1.,(rho*clo-along*slo)/R)
                    if ha>=hb:continue
                    # Known geometric measure: sqrt(1-h*h) dh.
                    mass=(hb-ha)*root_moments(ha,hb)[0]
                    source_weight=source_w[j]/mass if measure_source else source_w[j]
                    kind=(1 if hb==1. else 0)+(2 if ha==-1. else 0)
                    norm=0.;half=(hb-ha)/2;mid=(ha+hb)/2
                    for kh in range(hx.shape[1]):
                        x=hx[kind,kh];h=mid+half*x
                        shape=1.
                        if hb!=1.:shape*=1-h
                        if ha!=-1.:shape*=1+h
                        norm+=hw[kind,kh]*math.sqrt(shape)
                    for kh in range(hx.shape[1]):
                        x=hx[kind,kh];h=mid+half*x;root=math.sqrt((1-h)*(1+h))
                        shape=1.
                        if hb!=1.:shape*=1-h
                        if ha!=-1.:shape*=1+h
                        weight=hw[kind,kh]*math.sqrt(shape)/norm
                        eps=R/r;ca=math.sqrt(1-(eps*h)**2)
                        c=along/r*ca+rho/r*eps*h
                        s=rho/r*ca-along/r*eps*h
                        if c<=0:continue
                        n=1/(beta[i]*c);q=(n-n0)/dn;lam=left+width*q
                        jac=n*s/c*R/(r*ca)*width/abs(dn)
                        mu=extinction[panel]+q*(extinction[panel+1]-extinction[panel])
                        eta=((efficiency[panel,3]*q+efficiency[panel,2])*q+efficiency[panel,1])*q+efficiency[panel,0]
                        if efficiency.shape[1]>4:eta*=math.exp(efficiency[panel,4]+q*efficiency[panel,5])
                        spectral=(c0[i]+c2[i]/n**2)/L/lam**2*eta*jac
                        k0=math.sqrt(r*r-(R*h)**2);tangent=math.sqrt(r*r-R*R)
                        v=R/math.sqrt(2*(k0+tangent)*s*rho)
                        small=v*root
                        delta_max=R*R*root*root/(k0+tangent)
                        value=0.
                        for jp in range(len(px)):
                            y=px[jp];sinhalf=small*math.sqrt(1-y*y)
                            cp=1-2*sinhalf*sinhalf;sp=2*sinhalf*math.sqrt(1-sinhalf*sinhalf)
                            shift=delta_max*y*y
                            distance=tangent+shift-y*math.sqrt(delta_max*(2*tangent+shift))
                            accept=(_acceptance(alpha,c*axial+s*(cp*across+sp*side))
                                    +_acceptance(alpha,c*axial+s*(cp*across-sp*side)))
                            value+=pw[jp]/math.sqrt(1-sinhalf*sinhalf)*accept*math.exp(-mu*distance)
                            count+=1
                        angular=2*v*value
                        if subtract_leading:
                            control=1.
                            if linear_control:
                                control+=(mu*R*c/s+c*R/(2*rho))*h+mu*R*math.pi/4*root
                            angular-=2*R/rho*_acceptance(alpha,c*axial+s*across)*math.exp(-mu*rho/s)*control
                        subtotal+=((high-low)/2*source_weight*mass*weight
                                   *spectral*angular/(2*math.pi))
            value=subtotal*areas[om]/(math.pi*R*R)
            output[om]+=value
            if len(panel_charge):
                column=panel if panel_charge.shape[1]>1 else 0
                panel_charge[pair,column]+=value
    return output,count


def disk_rule(count):
    if count > 3:
        x, w = np.polynomial.legendre.leggauss(count)
        theta = (x+1)*np.pi/4
        return np.cos(theta), w*np.pi/4*np.cos(theta)
    from scipy.linalg import eigh
    from scipy.special import beta
    moment=np.array([beta((k+2)/2,.5)/2 for k in range(2*count)])
    hankel=np.array([[moment[i+j] for j in range(count)] for i in range(count)])
    shift=np.array([[moment[i+j+1] for j in range(count)] for i in range(count)])
    x,v=eigh(shift,hankel)
    w=(moment[:count]@v)**2
    return x,w


def boundary_charge(kernel,source,source_order=2,impact_order=4,phi_order=3,*,source_sine=False,measure_source=False,
                    subtract_leading=False, tables=None, pairs=None, linear_control=False, panel_charge=None):
    from scipy.special import roots_jacobi
    t=prepare_optics(kernel,exact_response=True) if tables is None else tables
    d=kernel.detector;radii=d.receiver_surface.radii(len(d))
    if pairs is None: pairs=interval_candidates(d.positions_m,radii,np.inf,source.start_m,source.direction,
                             source.length_m,source.coefficient0,source.beta,t['phase_nodes'])
    sx,sw=np.polynomial.legendre.leggauss(source_order)
    if source_sine:
        sw=sw*np.pi/2*np.cos(np.pi/2*sx);sw*=2/sw.sum()
        sx=np.sin(np.pi/2*sx)
    hrules=[roots_jacobi(impact_order,.5*(kind%2),.5*(kind//2)) for kind in range(4)]
    hx=np.array([rule[0] for rule in hrules]);hw=np.array([rule[1] for rule in hrules])
    mx,mw=np.polynomial.legendre.leggauss(12)
    mw=mw*np.pi/2*np.cos(np.pi/2*mx);mx=np.sin(np.pi/2*mx)
    if panel_charge is None:panel_charge=np.empty((0,0))
    return evaluate(*pairs,d.positions_m,-d.orientations,d.effective_area_m2,radii,
                    source.start_m,source.direction,source.length_m,source.coefficient0,source.coefficient2,
                    source.beta,t['panel_edges'],t['phase_nodes'],t['extinction_nodes'],
                    t['efficiency_coefficients'],kernel.acceptance()['alpha'],
                    sx,sw,hx,hw,*disk_rule(phi_order),measure_source,mx,mw,subtract_leading,linear_control,panel_charge)
