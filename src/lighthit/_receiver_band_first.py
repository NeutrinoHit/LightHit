"""Spectral path-CDF preparation and bounded coarse-time queries."""
import math
import numpy as np
from numba import njit
from ._receiver_first_lookup import _geometry,_scalar_edge,_outside_time_window
from .receiver_time import deposit_cdf


@njit(cache=True)
def node_paths(r,R,hit,j,count,g,maximum,nodes,excess):
    x=j/(count-1);cap=math.asin(R/r)
    theta=(math.asin(R/r*math.sin(math.pi*x/2)) if hit else
           cap+R/r*math.sinh(x*math.asinh(r/R*(math.pi-cap))))
    width=R+.5*r*abs(math.sin(theta))*(1-abs(g))/(1+abs(g))+.25*r*(1-math.cos(theta))
    extent=math.sqrt((r-R)*(r+R)) if hit else r+R*excess
    paths=np.empty(nodes)
    for k in range(nodes):paths[k]=min(extent,r-R+width*math.sinh(maximum*k/(nodes-1)))
    return paths


@njit(cache=True,nogil=True)
def fold_row(raw,paths,mu,front):
    """Normalized band CDFs, factoring mus*exp(-mut*(r-R))/r outside."""
    result=np.zeros((len(mu),len(paths),10))
    for band in range(len(mu)):
        for k in range(1,len(paths)):
            step=paths[k]-paths[k-1];z=mu[band]*step
            factor=math.exp(-mu[band]*(paths[k-1]-front))*(-math.expm1(-z)/z if z>0 else 1.)
            for m in range(10):result[band,k,m]=result[band,k-1,m]+factor*(raw[k,m]-raw[k-1,m])
    return result


@njit(cache=True,nogil=True)
def charge_only(radii,angles,moments,R,g,log_radii,mu,ms,speed,s0,s2,hit_charge,miss_charge):
    charges=np.zeros(len(radii))
    for group in range(len(radii)):
        r=radii[group]
        if r<=R or math.log(r)<log_radii[0] or math.log(r)>log_radii[-1]:raise ValueError('pencil outside spectral Q1 atlas')
        hit,i,j,fr,fa,_,_=_geometry(r,angles[group],R,log_radii,hit_charge.shape[1],miss_charge.shape[1],g,speed)
        q=hit_charge if hit else miss_charge;scale=ms*math.exp(-mu*(r-R))/r
        for m in range(10):
            weight=(s0*moments[group,0,m]+s2*moments[group,1,m])*scale
            value=(1-fr)*((1-fa)*q[i,j,m]+fa*q[i,j+1,m])+fr*((1-fa)*q[i+1,j,m]+fa*q[i+1,j+1,m])
            charges[group]+=weight*value
    return charges


@njit(cache=True,nogil=True)
def charge_bounds(radii,angles,moments,R,g,log_radii,mu,ms,speed,s0,s2,hit_charge,miss_charge,hit_error,miss_error):
    charges=np.zeros(len(radii));bounds=np.zeros(len(radii))
    for group in range(len(radii)):
        r=radii[group]
        if r<=R or math.log(r)<log_radii[0] or math.log(r)>log_radii[-1]:raise ValueError('pencil outside spectral Q1 atlas')
        hit,i,j,fr,fa,_,_=_geometry(r,angles[group],R,log_radii,hit_charge.shape[1],miss_charge.shape[1],g,speed)
        q=hit_charge if hit else miss_charge;e=hit_error if hit else miss_error
        scale=ms*math.exp(-mu*(r-R))/r
        for m in range(10):
            weight=(s0*moments[group,0,m]+s2*moments[group,1,m])*scale
            value=(1-fr)*((1-fa)*q[i,j,m]+fa*q[i,j+1,m])+fr*((1-fa)*q[i+1,j,m]+fa*q[i+1,j+1,m])
            error=(1-fr)*((1-fa)*e[i,j,m]+fa*e[i,j+1,m])+fr*((1-fa)*e[i+1,j,m]+fa*e[i+1,j+1,m])
            charges[group]+=weight*value;bounds[group]+=abs(weight)*error
    return charges,bounds


@njit(cache=True,nogil=True)
def sum_band_groups(radii,angles,clocks,spreads,moments,edges,fine,charges,R,g,log_radii,
                    mu,ms,speed,s0,s2,maximum,hit_cdf,miss_cdf,geometry_indices,raw_hit,raw_miss,
                    raw_radii,raw_excess,coarse_sinh,coarse_linear,fine_sinh,fine_linear,
                    quadratic,coarse_u,coarse_a,coarse_b,fine_u,fine_a,fine_b,coarse_tv,fine_tv):
    bins=np.zeros(max(0,len(edges)-1));coarse_nodes=hit_cdf.shape[2];raw_nodes=raw_hit.shape[2];guards=0
    scalar=np.empty(raw_nodes);primitive=np.empty(raw_nodes)
    for group in range(len(radii)):
        r=radii[group]
        hit,i,j,fr,fa,front,width=_geometry(r,angles[group],R,log_radii,hit_cdf.shape[1],miss_cdf.shape[1],g,speed)
        if _outside_time_window(clocks[group],spreads[group],front,width,maximum,edges):continue
        data=hit_cdf if hit else miss_cdf;raw=raw_hit if hit else raw_miss
        scale=ms*math.exp(-mu*(r-R))/r;weights=(moments[group,0]*s0+moments[group,1]*s2)*scale
        nodes=raw_nodes if fine[group] else coarse_nodes
        scalar[:nodes]=0.
        for ri,wr in [(i,1-fr),(i+1,fr)]:
            for aj,wa in [(j,1-fa),(j+1,fa)]:
                if wr*wa==0:continue
                if fine[group]:
                    original=geometry_indices[ri];physical_r=raw_radii[original]*R
                    paths=node_paths(physical_r,R,hit,aj,data.shape[1],g,maximum,raw_nodes,raw_excess)
                    folded=fold_row(raw[original,aj],paths,np.array([mu]),physical_r-R)[0]
                else:folded=data[ri,aj]
                for k in range(nodes):
                    for m in range(10):scalar[k]+=wr*wa*weights[m]*folded[k,m]
        # The last CDF value is the separately stored, full-resolution charge.
        scalar[0]=0.;scalar[nodes-1]=charges[group];primitive[0]=0.
        if quadratic[group]:
            u=fine_u if fine[group] else coarse_u;a=fine_a if fine[group] else coarse_a;b=fine_b if fine[group] else coarse_b
            tv=fine_tv if fine[group] else coarse_tv;error=0.
            for k in range(nodes-1):error+=abs(scalar[k+1]-scalar[k])*tv[k]
            if error<=np.max(tv)*abs(charges[group]):
                deposit_cdf(scalar[:nodes],clocks[group],spreads[group],front,width,edges,u,a,b,bins)
                continue
            guards+=1
        ds=fine_sinh if fine[group] else coarse_sinh;dl=fine_linear if fine[group] else coarse_linear
        for k in range(nodes-1):
            primitive[k+1]=primitive[k]+scalar[k]*ds[k]+(scalar[k+1]-scalar[k])*dl[k]
        previous=_scalar_edge(edges[0],clocks[group],spreads[group],front,width,maximum,scalar[:nodes],primitive[:nodes])
        for b in range(len(bins)):
            value=_scalar_edge(edges[b+1],clocks[group],spreads[group],front,width,maximum,scalar[:nodes],primitive[:nodes])
            bins[b]+=value-previous;previous=value
    return bins,guards
