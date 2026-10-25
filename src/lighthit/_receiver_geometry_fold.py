"""Analytic extinction and clock folding of a cubic geometry CDF.

The path density is quadratic on each panel. Exponential moments integrate
it without a wavelength quadrature or a ray trace. Uniform emission clocks
use positive local ramps, avoiding differences of large antiderivatives.
"""
import numpy as np
from numba import njit


@njit(cache=True)
def _phi(z):
    result=np.empty(4)
    if z<1.:
        for n in range(4):
            term=1.;value=1./(n+1)
            for k in range(1,32):
                term*=-z/k;increment=term/(n+k+1);value+=increment
                if abs(increment)<2e-16*abs(value):break
            result[n]=value
    else:
        decay=np.exp(-z);result[0]=-np.expm1(-z)/z
        for n in range(1,4):result[n]=(n*result[n-1]-decay)/z
    return result


@njit(cache=True)
def _panel(coeff,index,offset,width,mu,path_start,ramp_top,ramp_width):
    """Integral of the local density, optionally times a decreasing ramp."""
    p=_phi(mu*width);scale=np.exp(-mu*(path_start+offset));out=np.empty(coeff.shape[2])
    for m in range(len(out)):
        a=3*coeff[0,index,m];b=2*coeff[1,index,m];c=coeff[2,index,m]
        c+=b*offset+a*offset*offset;b+=2*a*offset
        value=width*(c*p[0]+b*width*p[1]+a*width*width*p[2])
        if ramp_width>0:
            first=width*width*(c*p[1]+b*width*p[2]+a*width*width*p[3])
            value=(ramp_top*value-first)/ramp_width
        out[m]=scale*value
    return out


@njit(cache=True)
def fold_coefficients(path,coeff,mu,scatter,speed,edges,clock_width):
    prefix=np.zeros((len(path),coeff.shape[2]))
    for i in range(len(path)-1):
        prefix[i+1]=prefix[i]+_panel(coeff,i,0.,path[i+1]-path[i],mu,path[i],0.,0.)
    cdf=np.zeros((len(edges),coeff.shape[2]));window=clock_width*speed
    for it in range(len(edges)):
        high=edges[it]*speed+window/2;low=high-window
        stop=min(max(low,path[0]),path[-1]);index=min(np.searchsorted(path,stop,side='right')-1,len(path)-2)
        cdf[it]=prefix[index]+_panel(coeff,index,0.,stop-path[index],mu,path[index],0.,0.)
        if window>0 and high>path[0] and low<path[-1]:
            start=max(low,path[0]);end=min(high,path[-1])
            index=min(max(np.searchsorted(path,start,side='right')-1,0),len(path)-2)
            while index<len(path)-1 and start<end:
                finish=min(end,path[index+1])
                cdf[it]+=_panel(coeff,index,start-path[index],finish-start,mu,path[index],high-start,window)
                start=finish;index+=1
    return scatter*prefix[-1],scatter*cdf
