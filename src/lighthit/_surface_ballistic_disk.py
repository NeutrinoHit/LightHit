"""Analytic finite-disk time CDF as a control variate for real glass rays."""
import math
import numpy as np
from numba import njit
from ._surface_ballistic_moments import window_moments
from ._ballistic_interval_numba import _support,_acceptance,_insert_cuts,_sort_small,interval_candidates
from .ballistic_interval import prepare_optics
from ._surface_ballistic_roots import time_cuts,add_roots,plus,physical_roots


@njit(cache=True)
def disk_area(a,b):
    a=max(-1.,a);b=min(1.,b)
    return 2*window_moments(a,b)[0] if a<b else 0.


@njit(cache=True)
def disk_cdf(a,b,A,B,time):
    """Area in a unit disk with h in [a,b] and A*h-B*sqrt(1-h*h-y*y)<=time."""
    a=max(-1.,a);b=min(1.,b)
    if a>=b:return 0.
    full=0.;pa=a;pb=b
    if A>0:
        cut=time/A;full=disk_area(a,min(b,cut));pa=max(a,cut)
    elif A<0:
        cut=time/A;full=disk_area(max(a,cut),b);pb=min(b,cut)
    elif time>=0:return disk_area(a,b)
    if pa>=pb:return full
    slope=A/B;delta=time/B;K=1+slope*slope
    discriminant=K-delta*delta
    if discriminant<=0:return full
    centre=slope*delta/K;radius=math.sqrt(discriminant)/K
    ua=max(-1.,(pa-centre)/radius);ub=min(1.,(pb-centre)/radius)
    return full+math.sqrt(K)*radius*radius*disk_area(ua,ub)


@njit(cache=True)
def disk_time_cuts(z,rho,R,L,beta,t0,t1,origin,n0,dn,g0,dg,low,high,edges):
    existing=time_cuts(z,rho,R,L,beta,t0,t1,origin,n0,dn,g0,dg,low,high,edges)
    # Include the endpoint projection time of the finite disk, which differs
    # from the exact sphere's tangent time at O(R/rho).
    tau=(t1-t0)/L;X=beta*np.array([n0,dn]);G=np.array([g0,dg])/.299792458
    S=np.convolve(X,X);S[0]-=1
    rmin=math.hypot(rho,z if z<0 else (z-L if z>L else 0.))
    rmax=max(math.hypot(rho,z),math.hypot(rho,z-L))
    earliest=min(t0,t1)+min(g0,g0+dg)/.299792458*(rmin-R)-origin
    latest=max(t0,t1)+max(g0+dg*low,g0+dg*high)/.299792458*rmax-origin
    first=np.searchsorted(edges,earliest,side='right');last=np.searchsorted(edges,latest,side='left')
    if first>=last:return existing
    cuts=np.empty(len(existing)+20*(last-first));cuts[:len(existing)]=existing;count=len(existing)
    for j in range(first,last):
        target=origin+edges[j]
        for source in (0.,L):
            dt=target-t0-tau*source
            Q=plus(dt*X,G,-(z-source))
            polynomial=plus(np.convolve(Q,Q),np.convolve(np.convolve(G,G),S),-rho*rho)
            start=count;count=add_roots(polynomial,low,high,cuts,count)
            count=physical_roots(cuts,start,count,5,source,z,rho,R,L,beta,t0,tau,target,n0,dn,g0,dg)
    return cuts[:count]
