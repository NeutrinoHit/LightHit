"""Compiled single-band folding for partial spectra and primary table builds."""
import math
import numpy as np
from numba import njit
from ._receiver_first_lookup import _geometry
from ._receiver_geometry_fold import fold_coefficients


@njit(cache=True)
def _pchip(path,values):
    """Scalar shape-preserving cubic CDF coefficients on a nonuniform grid."""
    h=np.diff(path);delta=np.diff(values)/h;n=len(values);slope=np.zeros(n)
    for i in range(1,n-1):
        if delta[i-1]*delta[i]>0:
            w1=2*h[i]+h[i-1];w2=h[i]+2*h[i-1]
            slope[i]=(w1+w2)/(w1/delta[i-1]+w2/delta[i])
    if n==2:slope[:]=delta[0]
    else:
        for first,second,k in [(0,1,0),(n-2,n-3,n-1)]:
            d=((2*h[first]+h[second])*delta[first]-h[first]*delta[second])/(h[first]+h[second])
            if d*delta[first]<=0:d=0.
            elif delta[first]*delta[second]<=0 and abs(d)>3*abs(delta[first]):d=3*delta[first]
            slope[k]=d
    coeff=np.empty((4,n-1,1))
    for i in range(n-1):
        coeff[0,i,0]=(slope[i]+slope[i+1]-2*delta[i])/(h[i]*h[i])
        coeff[1,i,0]=(3*delta[i]-2*slope[i]-slope[i+1])/h[i]
        coeff[2,i,0]=slope[i];coeff[3,i,0]=values[i]
    return coeff


@njit(cache=True,inline='always')
def _scalar(data,i,j,fr,fa,moments,factor):
    value=0.
    for m in range(10):
        value+=moments[m]*((1-fr)*((1-fa)*data[i,j,m]+fa*data[i,j+1,m])
                           +fr*((1-fa)*data[i+1,j,m]+fa*data[i+1,j+1,m]))
    return value*factor


@njit(cache=True)
def sum_geometry_groups(radii,angles,clocks,spreads,moments,edges,R,g,log_radii,maximum,max_excess,
                       hit_cdf,hit_total,hit_entry,miss_cdf,miss_total,mu,mus,speed,s0w,s2w):
    charge=0.;bins=np.zeros(max(0,len(edges)-1));tail=0.
    nodes=hit_cdf.shape[2];storage=np.empty(nodes+2);values=np.empty(nodes+2)
    for group in range(len(radii)):
        r=radii[group];theta=angles[group]
        if r<=R or math.log(r)<log_radii[0] or math.log(r)>log_radii[-1]:
            raise ValueError('pencil outside receiver geometry atlas')
        hit,i,j,fr,fa,_,_= _geometry(r,theta,R,log_radii,hit_total.shape[1],miss_total.shape[1],g,1.)
        rn=r/R;extent=math.sqrt((rn-1)*(rn+1)) if hit else rn+max_excess
        width=1+.5*rn*abs(math.sin(theta))*(1-abs(g))/(1+abs(g))+.25*rn*(1-math.cos(theta))
        cdf=hit_cdf if hit else miss_cdf;total=hit_total if hit else miss_total
        impact=rn*math.sin(theta)
        entry=rn*math.cos(theta)-math.sqrt(max(0.,1-impact*impact)) if hit else -1.
        for band in range(len(mu)):
            weights=moments[group,0]*s0w[band]+moments[group,1]*s2w[band]
            scale=1/(rn*R);used=0;inserted=False
            for k in range(nodes):
                path=rn-1+width*math.sinh(maximum*k/(nodes-1))
                if path>=extent:break
                if hit and not inserted and path>entry>rn-1:
                    storage[used]=entry*R
                    values[used]=_scalar(hit_entry,i,j,fr,fa,weights,scale)
                    used+=1;inserted=True
                storage[used]=path*R
                values[used]=_scalar(cdf[:,:,k,:],i,j,fr,fa,weights,scale);used+=1
            storage[used]=extent*R;values[used]=_scalar(total,i,j,fr,fa,weights,scale);used+=1
            # Retain the last accumulated mass when physical ticks coincide.
            keep=1
            for k in range(1,used):
                if storage[k]>storage[keep-1]:
                    storage[keep]=storage[k];values[keep]=values[k];keep+=1
                elif keep>1:values[keep-1]=values[k]
            path=storage[:keep];curve=values[:keep];coeff=_pchip(path,curve)
            q,f=fold_coefficients(path,coeff,mu[band],mus[band],speed[band],edges-clocks[group],spreads[group])
            charge+=q[0]
            for b in range(len(bins)):bins[b]+=f[b+1,0]-f[b,0]
            if not hit:
                phase_max=(1+abs(g))/(4*math.pi*(1-abs(g))**2)
                tail+=np.sum(np.abs(weights))*mus[band]*2*math.pi*phase_max/R*math.exp(-mu[band]*path[-1])
    return charge,bins,tail


@njit(cache=True)
def _moment_profile(r,theta,R,g,log_radii,maximum,max_excess,hit_cdf,hit_total,miss_cdf,miss_total):
    hit,i,j,fr,fa,_,_=_geometry(r,theta,R,log_radii,hit_total.shape[1],miss_total.shape[1],g,1.)
    rn=r/R;extent=math.sqrt((rn-1)*(rn+1)) if hit else rn+max_excess
    width=1+.5*rn*abs(math.sin(theta))*(1-abs(g))/(1+abs(g))+.25*rn*(1-math.cos(theta))
    cdf=hit_cdf if hit else miss_cdf;total=hit_total if hit else miss_total
    nodes=cdf.shape[2];paths=np.empty(nodes+1);moments=np.empty((nodes+1,10));used=0
    scale=1/(rn*R)
    for k in range(nodes+1):
        path=rn-1+width*math.sinh(maximum*k/(nodes-1)) if k<nodes else extent
        final=path>=extent
        if final:path=extent
        paths[used]=path*R
        a=total if final else cdf[:,:,k,:]
        for m in range(10):
            moments[used,m]=scale*((1-fr)*((1-fa)*a[i,j,m]+fa*a[i,j+1,m])
                                       +fr*((1-fa)*a[i+1,j,m]+fa*a[i+1,j+1,m]))
        if used>0 and paths[used]==paths[used-1]:
            if used>1:moments[used-1]=moments[used]
        else:used+=1
        if final:break
    return paths[:used],moments[:used],hit


@njit(cache=True)
def _weighted_profile(r,theta,R,g,log_radii,maximum,max_excess,hit_cdf,hit_total,miss_cdf,miss_total,weights):
    hit,i,j,fr,fa,_,_=_geometry(r,theta,R,log_radii,hit_total.shape[1],miss_total.shape[1],g,1.)
    rn=r/R;extent=math.sqrt((rn-1)*(rn+1)) if hit else rn+max_excess
    width=1+.5*rn*abs(math.sin(theta))*(1-abs(g))/(1+abs(g))+.25*rn*(1-math.cos(theta))
    cdf=hit_cdf if hit else miss_cdf;total=hit_total if hit else miss_total
    nodes=cdf.shape[2];paths=np.empty(nodes+1);values=np.empty((nodes+1,1));used=0;scale=1/(rn*R)
    for k in range(nodes+1):
        path=rn-1+width*math.sinh(maximum*k/(nodes-1)) if k<nodes else extent
        final=path>=extent
        if final:path=extent
        paths[used]=path*R;value=0.
        for m in range(10):
            if final:
                v=((1-fr)*((1-fa)*total[i,j,m]+fa*total[i,j+1,m])
                    +fr*((1-fa)*total[i+1,j,m]+fa*total[i+1,j+1,m]))
            else:
                v=((1-fr)*((1-fa)*cdf[i,j,k,m]+fa*cdf[i,j+1,k,m])
                    +fr*((1-fa)*cdf[i+1,j,k,m]+fa*cdf[i+1,j+1,k,m]))
            value+=weights[m]*scale*v
        values[used,0]=value
        if used>0 and paths[used]==paths[used-1]:
            if used>1:values[used-1,0]=value
        else:used+=1
        if final:break
    return paths[:used],values[:used]


@njit(cache=True,nogil=True)
def fold_linear_moments(path,moments,mu,scatter,speed,edges):
    """Exact extinction fold of a piecewise linear geometric CDF.

    Its panel density is constant. A direct exponential integral avoids the
    general cubic/ramp machinery; this changes neither paths nor quadrature.
    """
    prefix=np.zeros_like(moments)
    for i in range(len(path)-1):
        width=path[i+1]-path[i];z=mu*width
        weight=math.exp(-mu*path[i])*(-math.expm1(-z)/z if z>0 else 1.)
        for m in range(moments.shape[1]):prefix[i+1,m]=prefix[i,m]+weight*(moments[i+1,m]-moments[i,m])
    cdf=np.empty((len(edges),moments.shape[1]))
    for j in range(len(edges)):
        stop=edges[j]*speed
        if stop<=path[0]:
            cdf[j,:]=0.
            continue
        if stop>=path[-1]:
            cdf[j,:]=scatter*prefix[-1]
            continue
        i=min(np.searchsorted(path,stop,side='right')-1,len(path)-2)
        width=stop-path[i];z=mu*width
        weight=math.exp(-mu*path[i])*(-math.expm1(-z)/mu if mu>0 else width)/(path[i+1]-path[i])
        for m in range(moments.shape[1]):cdf[j,m]=scatter*(prefix[i,m]+weight*(moments[i+1,m]-moments[i,m]))
    return scatter*prefix[-1],cdf


@njit(cache=True,nogil=True)
def sum_linear_pencils(radii,angles,clocks,source_weight,direction,basis_x,looks,polynomial,edges,
                       R,g,log_radii,maximum,max_excess,hit_cdf,hit_total,miss_cdf,miss_total,mu,mus,speed):
    """Positive, orientation-linear fold for offline continuous-track tables.

    All looks share a receiver position and clock. Linear path-CDF panels
    preserve positive measures and the exact cubic acceptance dependence.
    Numerical path resolution is checked against refined geometry tables.
    """
    from_weights=np.empty((len(looks),10));charge=np.zeros(len(looks));bins=np.zeros((len(looks),max(0,len(edges)-1)))
    for ray in range(len(radii)):
        r=radii[ray];theta=angles[ray]
        if r<=R or math.log(r)<log_radii[0] or math.log(r)>log_radii[-1]:
            raise ValueError('quadrature pencil outside receiver geometry atlas')
        a0,a1,a2,a3=polynomial
        for j in range(len(looks)):
            x=np.dot(looks[j],basis_x[ray]);z=np.dot(looks[j],direction[ray]);y2=max(0.,1-x*x-z*z)
            from_weights[j]=np.array([a0+a2*y2,a1*x+3*a3*x*y2,a1*z+3*a3*z*y2,
                a2*(x*x-y2),2*a2*x*z,a2*(z*z-y2),a3*(x*x*x-3*x*y2),
                3*a3*z*(x*x-y2),3*a3*x*(z*z-y2),a3*(z*z*z-3*z*y2)])*source_weight[ray]
        if len(looks)==1:
            # A runtime fallback usually has one OM. Contract before the
            # path fold instead of building ten unused charge/time arrays.
            path,scalar=_weighted_profile(r,theta,R,g,log_radii,maximum,max_excess,
                hit_cdf,hit_total,miss_cdf,miss_total,from_weights[0])
            q,f=fold_linear_moments(path,scalar,mu,mus,speed,edges-clocks[ray])
            charge[0]+=q[0]
            for t in range(len(edges)-1):bins[0,t]+=f[t+1,0]-f[t,0]
            continue
        path,moments,_=_moment_profile(r,theta,R,g,log_radii,maximum,max_excess,hit_cdf,hit_total,miss_cdf,miss_total)
        q,f=fold_linear_moments(path,moments,mu,mus,speed,edges-clocks[ray])
        for j in range(len(looks)):
            for m in range(10):charge[j]+=from_weights[j,m]*q[m]
            for t in range(len(edges)-1):
                for m in range(10):bins[j,t]+=from_weights[j,m]*(f[t+1,m]-f[t,m])
    return charge,bins
