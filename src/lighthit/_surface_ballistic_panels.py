"""Reuse exact receiver/spectral branch cuts across convergence passes."""
import math
import numpy as np
from numba import njit
from ._ballistic_interval_numba import (
    _support,_acceptance,_insert_cuts,_sort_small,interval_candidates,_phase_order,_supported_panels,
)
from .ballistic_interval import prepare_optics
from ._surface_ballistic_disk import disk_time_cuts,disk_area,disk_cdf


@njit(cache=True)
def prepare_panels(po,pe,positions,looks,radii,starts,axes,lengths,beta,t0,t1,
                   panels,phases,groups,origins,edges,panel_mask):
    records=[];geometry=np.empty((len(po),5))
    _,_,monotonic,increasing,ordered=_phase_order(phases)
    for pair in range(len(po)):
        om=po[pair];i=pe[pair];R=radii[om];L=lengths[i]
        w=positions[om]-starts[i];z=np.dot(w,axes[i]);transverse=w-z*axes[i];rho=np.linalg.norm(transverse)
        if rho<=R:raise ValueError('disk preparation requires rho > R')
        axial=np.dot(axes[i],looks[om]);across=np.dot(transverse/rho,looks[om]);tau=(t1[i]-t0[i])/L
        geometry[pair]=np.array([z,rho,axial,across,tau])
        ok,nmin,nmax,cut0,cut1=_support(z,rho,L,beta[i],R)
        if not ok:continue
        first,last=_supported_panels(nmin,nmax,monotonic,increasing,ordered)
        for panel in range(first,last):
            if len(panel_mask) and not panel_mask[pair,panel]:continue
            left=panels[panel];width=panels[panel+1]-left
            n0=phases[panel];dn=phases[panel+1]-n0
            low=0.;high=1.
            if dn==0:
                if n0<nmin or n0>=nmax:continue
            else:
                x0=(nmin-n0)/dn;x1=(nmax-n0)/dn
                low=max(0.,min(x0,x1));high=min(1.,max(x0,x1))
            if low>=high:continue
            extra=disk_time_cuts(z,rho,R,L,beta[i],t0[i],t1[i],origins[om],n0,dn,
                                 groups[panel],groups[panel+1]-groups[panel],low,high,edges)
            cuts=np.empty(len(extra)+4);cuts[:len(extra)]=left+width*extra
            count=_insert_cuts(cuts,len(extra),left,width,n0,dn,cut0,cut1)
            _sort_small(cuts,count)
            for j in range(count-1):
                a,b=cuts[j],cuts[j+1]
                if a<b:records.append((pair,panel,a,b))
    pairs=np.empty(len(records),np.int64);which=np.empty(len(records),np.int64);limits=np.empty((len(records),2))
    for j in range(len(records)):
        pairs[j]=records[j][0];which[j]=records[j][1];limits[j,0]=records[j][2];limits[j,1]=records[j][3]
    return geometry,pairs,which,limits


@njit(cache=True)
def evaluate(po,pe,geometry,pairs,which,limits,count_om,areas,radii,lengths,c0,c2,beta,t0,
             panels,phases,groups,extinction,efficiency,alpha,lx,lw,origins,edges):
    output=np.zeros(count_om);bins=np.zeros((count_om,len(edges)-1));nodes=0
    for record in range(len(pairs)):
        pair=pairs[record];panel=which[record];om=po[pair];i=pe[pair];R=radii[om];L=lengths[i]
        z,rho,axial,across,tau=geometry[pair]
        left=panels[panel];width=panels[panel+1]-left;n0=phases[panel];dn=phases[panel+1]-n0
        a,b=limits[record]
        for j in range(len(lx)):
            lam=(a+b)/2+(b-a)/2*lx[j];q=(lam-left)/width;n=n0+dn*q
            c=1/(beta[i]*n);s=math.sqrt((1-c)*(1+c));D=rho/s
            sc=z-rho*c/s;spread=R/s
            ha=max(-1.,-sc/spread);hb=min(1.,(L-sc)/spread)
            if ha>=hb:continue
            mu=extinction[panel]+q*(extinction[panel+1]-extinction[panel])
            eta=((efficiency[panel,3]*q+efficiency[panel,2])*q+efficiency[panel,1])*q+efficiency[panel,0]
            if efficiency.shape[1]>4:eta*=math.exp(efficiency[panel,4]+q*efficiency[panel,5])
            factor=((b-a)/2*lw[j]*(c0[i]+c2[i]/n**2)/L/lam**2*eta
                    *math.exp(-mu*D)*_acceptance(alpha,c*axial+s*across)*areas[om]/(2*math.pi**2*rho*s))
            mass=disk_area(ha,hb);output[om]+=factor*mass;nodes+=1
            G=(groups[panel]+q*(groups[panel+1]-groups[panel]))/.299792458
            tc=t0[i]+tau*sc+G*D-origins[om];A=(tau-G*c)*spread;B=G*R
            star=max(ha,min(hb,-A/math.sqrt(A*A+B*B)))
            earliest=tc+A*star-B*math.sqrt(max(0.,1-star*star));latest=tc+max(A*ha,A*hb)
            first=max(0,np.searchsorted(edges,earliest,side='right')-1)
            last=min(len(edges)-2,np.searchsorted(edges,latest,side='left')-1)
            if first>last:continue
            if first==last and edges[first]<=earliest and latest<=edges[last+1]:
                bins[om,first]+=factor*mass;continue
            previous=0. if edges[first]<=earliest else disk_cdf(ha,hb,A,B,edges[first]-tc)
            for slot in range(first,last+1):
                current=mass if edges[slot+1]>=latest else disk_cdf(ha,hb,A,B,edges[slot+1]-tc)
                bins[om,slot]+=factor*(current-previous);previous=current
    return output,bins,nodes


def prepare(kernel,source,origins,edges,*,tables=None,pairs=None,panel_mask=None):
    t=prepare_optics(kernel,exact_response=True) if tables is None else tables
    d=kernel.detector;radii=d.receiver_surface.radii(len(d))
    po,pe=interval_candidates(d.positions_m,radii,np.inf,source.start_m,source.direction,source.length_m,
                             source.coefficient0,source.beta,t['phase_nodes']) if pairs is None else pairs
    if panel_mask is None:panel_mask=np.empty((0,0),bool)
    geometry,pairs,which,limits=prepare_panels(po,pe,d.positions_m,-d.orientations,radii,source.start_m,
        source.direction,source.length_m,source.beta,source.start_ns,source.end_ns,t['panel_edges'],
        t['phase_nodes'],t['group_nodes'],np.asarray(origins),np.asarray(edges),panel_mask)
    return dict(tables=t,om=po,element=pe,geometry=geometry,pairs=pairs,which=which,limits=limits)


def response(kernel,source,origins,edges,order=8,*,prepared=None,receiver_mask=None):
    p=prepare(kernel,source,origins,edges) if prepared is None else prepared
    t=p['tables'];d=kernel.detector;radii=d.receiver_surface.radii(len(d))
    pairs,which,limits=p['pairs'],p['which'],p['limits']
    if receiver_mask is not None:
        keep=np.asarray(receiver_mask)[p['om'][pairs]]
        pairs=pairs[keep];which=which[keep];limits=limits[keep]
    x,w=np.polynomial.legendre.leggauss(order);w=w*np.pi/2*np.cos(np.pi/2*x);x=np.sin(np.pi/2*x)
    return evaluate(p['om'],p['element'],p['geometry'],pairs,which,limits,len(d),d.effective_area_m2,radii,
        source.length_m,source.coefficient0,source.coefficient2,source.beta,source.start_ns,
        t['panel_edges'],t['phase_nodes'],t['group_nodes'],t['extinction_nodes'],t['efficiency_coefficients'],
        kernel.acceptance()['alpha'],x,w,np.asarray(origins),np.asarray(edges))
