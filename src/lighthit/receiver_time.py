"""Positive local-time deposition with an explicit CDF variation bound."""
from functools import lru_cache
import numpy as np
from numba import njit


@lru_cache(maxsize=16)
def quadratic_rule(nodes,maximum):
    """Quadratic CDF panels and exact extrema of their asinh-CDF error.

    Each panel matches its endpoints and physical-time midpoint. The CDF's
    derivative error has a quartic equation in the normalized time x.
    Summing differences at its real extrema gives total variation, so the
    bound survives arbitrary bin edges and positive clock convolution.
    """
    u=np.linspace(0.,maximum,nodes);s=np.sinh(u);a=np.empty(nodes-1);b=np.empty(nodes-1);tv=np.empty(nodes-1)
    valid=True
    for j in range(nodes-1):
        du=u[j+1]-u[j];ds=s[j+1]-s[j]
        midpoint=(np.arcsinh((s[j]+s[j+1])/2)-u[j])/du
        A=4*midpoint-1;B=2-4*midpoint;a[j]=A;b[j]=B
        valid=valid and A>=0 and A+2*B>=0
        coefficients=np.polynomial.polynomial.polymul([A*A,4*A*B,4*B*B],
            [1+s[j]*s[j],2*s[j]*ds,ds*ds])
        coefficients[0]-=(ds/du)**2
        roots=np.polynomial.polynomial.polyroots(coefficients)
        interior=sorted(float(x.real) for x in roots if abs(x.imag)<1e-8 and 0<x.real<1 and A+2*B*x.real>0)
        x=np.array([0.,*interior,1.]);error=(np.arcsinh(s[j]+ds*x)-u[j])/du-(A*x+B*x*x)
        tv[j]=abs(np.diff(error)).sum()+128*np.finfo(float).eps*(1+abs(u[j])+abs(u[j+1]))/du
    return s,a,b,tv,bool(valid and np.isfinite(tv).all())


def time_levels(coarse,fine,budget):
    """0: coarse quadratic; 1: fine quadratic; 2: exact fine CDF lookup."""
    levels=np.zeros(len(coarse),np.int8);remaining=float(np.sum(coarse))
    if budget==0:return np.full(len(coarse),2,np.int8),0.
    if remaining>budget:
        for j in np.argsort(coarse-fine)[::-1]:
            if coarse[j]<=fine[j]:continue
            levels[j]=1;remaining-=coarse[j]-fine[j]
            if remaining<=budget:break
    if remaining>budget:
        current=np.where(levels==0,coarse,fine)
        for j in np.argsort(current)[::-1]:
            levels[j]=2;remaining-=current[j]
            if remaining<=budget:break
    return levels,max(0.,remaining)


@njit(cache=True,inline='always')
def _panel_cdf(t,start,h,spread,A,B):
    x=t-start
    if spread==0.:
        y=min(1.,max(0.,x/h));return A*y+B*y*y
    lo=x-spread/2;hi=x+spread/2
    if hi<=0:return 0.
    if lo>=h:return 1.
    if lo>=0 and hi<=h:
        y=x/h;return A*y+B*(y*y+spread*spread/(12*h*h))
    low=max(lo,0.);high=min(hi,h);width=max(0.,high-low)
    value=width/spread*(A/h*(low+width/2)+B/(h*h)*(low*low+low*width+width*width/3))
    return value+max(0.,hi-max(lo,h))/spread


@njit(cache=True,nogil=True)
def deposit_cdf(cdf,clock,spread,front,width,edges,sinh_u,A,B,bins):
    """Deposit each positive CDF increment only into bins it overlaps.

    The all-time mass is unchanged. No tail threshold, clipping or event
    normalization is used. Most front panels lie wholly inside one bin.
    """
    nb=len(bins);left=0;right=0
    for k in range(len(cdf)-1):
        mass=cdf[k+1]-cdf[k]
        if mass==0:continue
        start=clock+front+width*sinh_u[k];h=width*(sinh_u[k+1]-sinh_u[k])
        lower=start-spread/2;upper=start+h+spread/2
        if upper<=edges[0]:continue
        if lower>=edges[-1]:break
        while left<nb-1 and edges[left+1]<=lower:left+=1
        right=max(right,left)
        while right<nb-1 and edges[right+1]<upper:right+=1
        if left==right and edges[left]<=lower and upper<=edges[right+1]:
            bins[left]+=mass;continue
        previous=_panel_cdf(edges[left],start,h,spread,A[k],B[k])
        for j in range(left,right+1):
            value=_panel_cdf(edges[j+1],start,h,spread,A[k],B[k])
            bins[j]+=mass*(value-previous);previous=value
