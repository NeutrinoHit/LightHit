"""Finite-receiver bin residual in point-source impact coordinates."""
import math
import numpy as np
from numba import njit
from scipy.special import roots_jacobi
from ._ballistic_interval_numba import (
    _sort_small, interval_candidates, _phase_order, _support, _supported_panels,
)
from .ballistic_interval import prepare_optics
from ._surface_ballistic_moments import root_moments
from ._surface_ballistic_charge import disk_rule
from ._surface_ballistic_time import time_cuts,deposit_h,source_time_cuts


@njit(cache=True)
def evaluate(po,pe,positions,looks,areas,radii,starts,axes,lengths,c0,c2,beta,t0,t1,
             panels,phases,groups,extinction,efficiency,alpha,sx,sw,hx,hw,gx,gw,px,pw,tx,tw,
             measure_source,mx,mw,origins,edges,panel_mask):
    output=np.zeros(len(positions));bins=np.zeros((len(positions),len(edges)-1));count=0
    _,_,monotonic,increasing,ordered=_phase_order(phases)
    for pair in range(len(po)):
        om=po[pair];i=pe[pair];R=radii[om];L=lengths[i];tau=(t1[i]-t0[i])/L
        w=positions[om]-starts[i];u=axes[i]
        z=np.dot(w,u);transverse=w-z*u;rho=np.linalg.norm(transverse)
        if rho<=R:raise ValueError('impact quadrature requires rho > R')
        er=transverse/rho;ey=np.cross(u,er)
        axial=np.dot(u,looks[om]);across=np.dot(er,looks[om]);side=np.dot(ey,looks[om])
        valid,nmin,nmax,_,_=_support(z,rho,L,beta[i],R)
        if not valid:continue
        first,last=_supported_panels(nmin,nmax,monotonic,increasing,ordered)
        for panel in range(first,last):
            if len(panel_mask) and not panel_mask[pair,panel]:continue
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
            extra=source_time_cuts(amin,amax,z,rho,R,beta[i],t0[i],tau,origins[om],n0,dn,
                groups[panel],groups[panel+1]-groups[panel],edges)
            cuts=np.empty(len(extra)+4);cuts[:len(extra)]=extra;ncut=len(extra)
            for c,s in ((clo,slo),(chi,shi)):
                if s==0:continue
                for sign in (-1.,1.):
                    a=z-(rho*c+sign*R)/s
                    if amin<a<amax:cuts[ncut]=a;ncut+=1
            _sort_small(cuts,ncut)
            for part in range(ncut-1):
                low,high=cuts[part],cuts[part+1]
                source_x=sx;source_w=sw
                if measure_source:
                    mom=np.zeros(4)
                    for j in range(len(mx)):
                        a=(low+high)/2+(high-low)/2*mx[j]
                        h0=max(-1.,(rho*chi-(z-a)*shi)/R)
                        h1=min(1.,(rho*clo-(z-a)*slo)/R)
                        mass=(h1-h0)*root_moments(h0,h1)[0] if h1>h0 else 0.
                        value=mass*mw[j]
                        for power in range(4):mom[power]+=value;value*=mx[j]
                    if mom[0]<=0:continue
                    mean=mom[1]/mom[0];var=mom[2]/mom[0]-mean*mean
                    diagonal=(mom[3]/mom[0]-2*mean*mom[2]/mom[0]+mean**3)/var
                    discr=math.sqrt((mean-diagonal)**2+4*var)
                    x0=(mean+diagonal-discr)/2;x1=(mean+diagonal+discr)/2
                    w1=(mean-x0)/(x1-x0);w0=1-w1
                    source_x=np.array([x0,x1]);source_w=np.array([w0,w1])*mom[0]
                for js in range(len(source_x)):
                    source=(low+high)/2+(high-low)/2*source_x[js]
                    along=z-source;r=math.hypot(rho,along);emit=t0[i]+tau*source
                    ha=max(-1.,(rho*chi-along*shi)/R)
                    hb=min(1.,(rho*clo-along*slo)/R)
                    if ha>=hb:continue
                    source_mass=(hb-ha)*root_moments(ha,hb)[0]
                    source_weight=source_w[js]/source_mass if measure_source else source_w[js]
                    hcuts=time_cuts(ha,hb,rho,along,R,beta[i],emit,origins[om],n0,dn,
                                    groups[panel],groups[panel+1]-groups[panel],edges)
                    split=len(hcuts)>2
                    for kh in range(len(hcuts)-1):
                        ah,bh=hcuts[kh],hcuts[kh+1]
                        if ah>=bh:continue
                        half=(bh-ah)/2;mid=(ah+bh)/2
                        mass=(bh-ah)*root_moments(ah,bh)[0]
                        kind=(1 if bh==1. else 0)+(2 if ah==-1. else 0)
                        norm=0.
                        for j in range(len(gx)):
                            if split:
                                h=mid+half*math.sin(math.pi/2*gx[j])
                                wj=gw[j]*math.cos(math.pi/2*gx[j])*math.sqrt(max(0.,(1-h)*(1+h)))
                            else:
                                h=mid+half*hx[kind,j];shape=1.
                                if bh!=1.:shape*=1-h
                                if ah!=-1.:shape*=1+h
                                wj=hw[kind,j]*math.sqrt(shape)
                            norm+=wj
                        if norm==0:continue
                        for j in range(len(gx)):
                            if split:
                                h=mid+half*math.sin(math.pi/2*gx[j])
                                wj=gw[j]*math.cos(math.pi/2*gx[j])*math.sqrt(max(0.,(1-h)*(1+h)))
                            else:
                                h=mid+half*hx[kind,j];shape=1.
                                if bh!=1.:shape*=1-h
                                if ah!=-1.:shape*=1+h
                                wj=hw[kind,j]*math.sqrt(shape)
                            if wj==0:continue
                            weight=((high-low)/2*source_weight*mass*wj/norm*areas[om]/(math.pi*R*R))
                            q,used=deposit_h(h,rho,along,r,R,beta[i],emit,origins[om],left,width,n0,dn,
                                groups[panel],groups[panel+1]-groups[panel],extinction[panel],
                                extinction[panel+1]-extinction[panel],efficiency[panel],c0[i],c2[i],L,
                                alpha,axial,across,side,weight,px,pw,tx,tw,edges,bins[om])
                            output[om]+=q;count+=used
    return output,bins,count


def response(kernel,source,origins,edges,source_order=2,impact_order=4,phi_order=2,measure_source=True, *, tables=None, pairs=None,panel_mask=None):
    t=prepare_optics(kernel,exact_response=True) if tables is None else tables
    d=kernel.detector
    radii=d.receiver_surface.radii(len(d))
    po,pe=interval_candidates(d.positions_m,radii,np.inf,source.start_m,source.direction,
                             source.length_m,source.coefficient0,source.beta,t['phase_nodes']) if pairs is None else pairs
    hrules=[roots_jacobi(impact_order,.5*(kind%2),.5*(kind//2)) for kind in range(4)]
    hx=np.array([v[0] for v in hrules]);hw=np.array([v[1] for v in hrules])
    mx,mw=np.polynomial.legendre.leggauss(12)
    mw=mw*np.pi/2*np.cos(np.pi/2*mx);mx=np.sin(np.pi/2*mx)
    alpha=kernel.acceptance()['alpha'];ell=np.arange(len(alpha))
    apoly=np.polynomial.legendre.leg2poly((2*ell+1)*alpha/(4*np.pi))
    if len(apoly)<4:apoly=np.pad(apoly,(0,4-len(apoly)))
    if panel_mask is None:panel_mask=np.empty((0,0),bool)
    return evaluate(po,pe,d.positions_m,-d.orientations,d.effective_area_m2,radii,
        source.start_m,source.direction,source.length_m,source.coefficient0,source.coefficient2,source.beta,
        source.start_ns,source.end_ns,t['panel_edges'],t['phase_nodes'],t['group_nodes'],
        t['extinction_nodes'],t['efficiency_coefficients'],apoly,
        *np.polynomial.legendre.leggauss(source_order),hx,hw,*np.polynomial.legendre.leggauss(impact_order),
        *disk_rule(phi_order),*np.polynomial.legendre.leggauss(max(8,2*phi_order)),measure_source,mx,mw,
        np.asarray(origins),np.asarray(edges),panel_mask)
