"""Compiled Q1 atlas queries, with no optical-path integration."""
import math
import numpy as np
from numba import njit
from .receiver_time import deposit_cdf


@njit(cache=True)
def _geometry(r,theta,R,log_radii,hit_nodes,miss_nodes,g,speed_max):
    log_r=math.log(r)
    i=np.searchsorted(log_radii,log_r,side='right')-1
    i=min(max(i,0),len(log_radii)-2)
    fr=(log_r-log_radii[i])/(log_radii[i+1]-log_radii[i])
    fr=min(1.,max(0.,fr))
    cap=math.asin(R/r);hit=theta<=cap
    if hit:
        x=2/math.pi*math.asin(min(1.,max(0.,r*math.sin(theta)/R)))
        coordinate=x*(hit_nodes-1);size=hit_nodes
    else:
        x=math.asinh(r/R*(theta-cap))/math.asinh(r/R*(math.pi-cap))
        coordinate=x*(miss_nodes-1);size=miss_nodes
    j=min(max(int(coordinate),0),size-2);fa=coordinate-j
    fa=min(1.,max(0.,fa))
    front=(r-R)/speed_max
    hg_width=(1-abs(g))/(1+abs(g))
    width=(R+.5*r*abs(math.sin(theta))*hg_width+.25*r*(1-math.cos(theta)))/speed_max
    return hit,i,j,fr,fa,front,width


@njit(cache=True)
def _cdf_indices(time,front,width,maximum,nodes):
    u=math.asinh(max(0.,time-front)/width)
    clipped=min(u,maximum);step=maximum/(nodes-1)
    k=min(int(clipped/step),nodes-2);fraction=clipped/step-k
    lo=k*step
    second=((clipped-lo)*math.sinh(clipped)-math.cosh(clipped)+math.cosh(lo))/step
    first=math.sinh(clipped)-math.sinh(lo)
    tail=math.sinh(u)-math.sinh(clipped)
    return k,fraction,first-second,second,tail


@njit(cache=True,inline='always')
def _mix(data,i,j,k,field,moment,fr,fa):
    return ((1-fr)*((1-fa)*data[i,j,k,field,moment]+fa*data[i,j+1,k,field,moment])
            +fr*((1-fa)*data[i+1,j,k,field,moment]+fa*data[i+1,j+1,k,field,moment]))


@njit(cache=True)
def _edge_value(edge,clock,spread,front,width,maximum,i,j,fr,fa,scale,moments,cdf,integral):
    total=0.;time=edge-clock;nodes=cdf.shape[2]
    if spread==0.:
        k,f,_,_,_=_cdf_indices(time,front,width,maximum,nodes)
        for field in range(2):
            for m in range(10):
                coefficient=moments[field,m]*scale[field]
                if coefficient!=0.:
                    value=(1-f)*_mix(cdf,i,j,k,field,m,fr,fa)+f*_mix(cdf,i,j,k+1,field,m,fr,fa)
                    total+=coefficient*value
        return total
    lo=_cdf_indices(time-spread/2,front,width,maximum,nodes)
    hi=_cdf_indices(time+spread/2,front,width,maximum,nodes)
    for field in range(2):
        for m in range(10):
            coefficient=moments[field,m]*scale[field]
            if coefficient!=0.:
                k,_,a,b,tail=hi
                value=_mix(integral,i,j,k,field,m,fr,fa)
                value+=a*_mix(cdf,i,j,k,field,m,fr,fa)+b*_mix(cdf,i,j,k+1,field,m,fr,fa)
                value+=tail*_mix(cdf,i,j,nodes-1,field,m,fr,fa)
                k,_,a,b,tail=lo
                value-=_mix(integral,i,j,k,field,m,fr,fa)
                value-=a*_mix(cdf,i,j,k,field,m,fr,fa)+b*_mix(cdf,i,j,k+1,field,m,fr,fa)
                value-=tail*_mix(cdf,i,j,nodes-1,field,m,fr,fa)
                total+=coefficient*width/spread*value
    return total


@njit(cache=True,inline='always')
def _local_cdf_average(lo,hi,front,width,maximum,cdf):
    """Positive local integral of a CDF linear in asinh time.

    Differences of large cumulative primitives lose precision for tiny
    source-clock widths. Integrating at most a few local intervals avoids
    that cancellation and preserves a constant CDF exactly.
    """
    if hi==lo:
        u=min(math.asinh(max(0.,lo-front)/width),maximum)
        k=min(int(u/maximum*(len(cdf)-1)),len(cdf)-2)
        f=u/maximum*(len(cdf)-1)-k
        return (1-f)*cdf[k]+f*cdf[k+1]
    length=hi-lo;step=maximum/(len(cdf)-1)
    if hi<=front:return 0.
    a=max(lo,front);u=math.asinh((a-front)/width)
    if u>=maximum:return cdf[-1]
    initial=min(int(u/step),len(cdf)-2)
    fraction=u/step-initial
    baseline=(1-fraction)*cdf[initial]+fraction*cdf[initial+1]
    total=baseline*length
    if lo<front:total-=baseline*(min(hi,front)-lo)
    end_u=math.asinh((hi-front)/width)
    while u<min(end_u,maximum):
        k=min(int(u/step),len(cdf)-2)
        # Roundoff at an exact knot must advance to its following interval.
        if (k+1)*step<=u and k<len(cdf)-2:k+=1
        stop=min((k+1)*step,end_u,maximum)
        delta=stop-u
        if delta<=0:break
        dt=width*2*math.sinh(delta/2)*math.cosh(u+delta/2)
        slope=(cdf[k+1]-cdf[k])/step;first=cdf[k]+slope*(u-k*step)
        if delta<.125:
            d2=delta*delta
            A=d2*(.5+d2*(.125+d2*(1/144+d2*(1/5760+d2/403200))))
            B=delta*d2*(1/3+d2*(1/30+d2*(1/840+d2*(1/45360+d2/3991680))))
        else:
            A=delta*math.sinh(delta)-math.cosh(delta)+1
            B=delta*math.cosh(delta)-math.sinh(delta)
        total+=(first-baseline)*dt+slope*width*(math.cosh(u)*A+math.sinh(u)*B)
        u=stop
    if end_u>maximum:
        total+=(cdf[-1]-baseline)*(hi-(front+width*math.sinh(maximum)))
    return total/length


@njit(cache=True)
def _scalar_edge(time,clock,spread,front,width,maximum,cdf,integral):
    t=time-clock;step=maximum/(len(cdf)-1)
    if spread==0.:
        u=min(math.asinh(max(0.,t-front)/width),maximum)
        k=min(int(u/step),len(cdf)-2);f=u/step-k
        return (1-f)*cdf[k]+f*cdf[k+1]
    lo=t-spread/2;hi=t+spread/2
    lower=math.asinh(max(0.,lo-front)/width);upper=math.asinh(max(0.,hi-front)/width)
    k_lo=min(int(min(lower,maximum)/step),len(cdf)-2)
    k_hi=min(int(min(upper,maximum)/step),len(cdf)-2)
    if cdf[k_lo]==cdf[k_hi+1]:return cdf[k_lo]
    if k_hi-k_lo<=2:return _local_cdf_average(lo,hi,front,width,maximum,cdf)
    answer=0.
    for sign in (-1.,1.):
        offset=t+sign*spread/2
        z=max(0.,offset-front)/width;u=math.asinh(z)
        clipped=min(u,maximum);k=min(int(clipped/step),len(cdf)-2);lo=k*step
        sinh_clip=z if u<=maximum else math.sinh(maximum)
        cosh_clip=math.sqrt(1+z*z) if u<=maximum else math.cosh(maximum)
        second=((clipped-lo)*sinh_clip-cosh_clip+math.cosh(lo))/step
        first=sinh_clip-math.sinh(lo)
        value=integral[k]+(first-second)*cdf[k]+second*cdf[k+1]
        value+=cdf[-1]*(z-sinh_clip)
        answer+=sign*value
    return width/spread*answer


@njit(cache=True,inline='always')
def _spectral_scale(r,R,mu,mus,s0w,s2w):
    first=0.;second=0.
    for band in range(len(mu)):
        factor=mus[band]*math.exp(-mu[band]*(r-R))/r
        first+=s0w[band]*factor;second+=s2w[band]*factor
    return first,second


@njit(cache=True,inline='always')
def _outside_time_window(clock,spread,front,width,maximum,edges):
    """The existing stored CDF and clock interval cannot reach any bin."""
    return (clock+front-spread/2>=edges[-1]
        or clock+front+width*math.sinh(maximum)+spread/2<=edges[0])


@njit(cache=True,nogil=True)
def time_cdf_rows(radii,angles,clocks,spreads,log_radii,R,g,speed,maximum,edges,prepared=None):
    """The same radial brackets and hit branch as the time interpolator."""
    needed=np.zeros((2,len(log_radii)),np.bool_)
    hg_width=(1-abs(g))/(1+abs(g))
    for p in range(len(radii)):
        if prepared is not None:
            indices,values=prepared
            if _outside_time_window(clocks[p],spreads[p],values[p,2],values[p,3],maximum,edges):continue
            branch=0 if indices[p,0] else 1;i=indices[p,1]
            needed[branch,i]=True;needed[branch,i+1]=True
            continue
        r=radii[p];theta=angles[p]
        if not math.isfinite(r) or r<=R or not 0<=theta<=math.pi:continue
        front=(r-R)/speed
        width=(R+.5*r*abs(math.sin(theta))*hg_width+.25*r*(1-math.cos(theta)))/speed
        if _outside_time_window(clocks[p],spreads[p],front,width,maximum,edges):continue
        i=min(max(np.searchsorted(log_radii,math.log(r),side='right')-1,0),len(log_radii)-2)
        branch=0 if theta<=math.asin(R/r) else 1
        needed[branch,i]=True;needed[branch,i+1]=True
    return needed


@njit(cache=True,nogil=True)
def sum_groups(radii,angles,clocks,spreads,moments,edges,log_radii,R,g,
               mu,mus,s0w,s2w,speed_max,maximum,
               hit_charge,hit_cdf,hit_integral,miss_charge,miss_cdf,miss_integral,
               quadratic_time,sinh_u,quadratic_a,quadratic_b,quadratic_tv,time_major=False,prepared=None):
    """Sum groups after contracting their fixed angular moments once.

    Time edges then query scalar CDFs instead of repeatedly contracting ten
    moments, two fields and four spatial neighbors for each edge.
    """
    charge=0.;bins=np.zeros(max(0,len(edges)-1));scale=np.empty(2);guards=0
    nodes=hit_cdf.shape[-1] if time_major else hit_cdf.shape[2]
    scalar_cdf=np.empty(nodes);scalar_integral=np.empty(nodes);coefficients=np.empty((2,10))
    for group in range(len(radii)):
        r=radii[group];theta=angles[group]
        if (not math.isfinite(r) or r<=R or not 0<=theta<=math.pi
                or (prepared is None and (math.log(r)<log_radii[0] or math.log(r)>log_radii[-1]))):
            raise ValueError('pencil geometry outside the first-entry Q1 atlas')
        if prepared is None:
            hit,i,j,fr,fa,front,width=_geometry(r,theta,R,log_radii,hit_charge.shape[1],miss_charge.shape[1],g,speed_max)
            scale[:]=_spectral_scale(r,R,mu,mus,s0w,s2w)
        else:
            indices,values=prepared
            hit=indices[group,0]!=0;i=indices[group,1];j=indices[group,2]
            fr=values[group,0];fa=values[group,1];front=values[group,2];width=values[group,3]
            scale[0]=values[group,4];scale[1]=values[group,5]
        q=hit_charge if hit else miss_charge;c=hit_cdf if hit else miss_cdf
        primitive=hit_integral if hit else miss_integral
        scalar_cdf[:]=0.;scalar_integral[:]=0.
        local_charge=0.
        for field in range(2):
            for m in range(10):
                coefficient=moments[group,field,m]*scale[field]
                coefficients[field,m]=coefficient
                if coefficient==0.:continue
                value=((1-fr)*((1-fa)*q[i,j,field,m]+fa*q[i,j+1,field,m])
                      +fr*((1-fa)*q[i+1,j,field,m]+fa*q[i+1,j+1,field,m]))
                charge+=coefficient*value;local_charge+=coefficient*value
        if len(edges):
            if _outside_time_window(clocks[group],spreads[group],front,width,maximum,edges):continue
            need_integral=not quadratic_time and spreads[group]>width*maximum/(len(scalar_cdf)-1)
            if time_major:
                # Independent time lanes permit SIMD without reassociating
                # the sum over moments. Each lane keeps the original order.
                for field in range(2):
                    for m in range(10):
                        coefficient=coefficients[field,m]
                        if coefficient==0.:continue
                        for k in range(nodes):
                            scalar_cdf[k]+=coefficient*((1-fr)*((1-fa)*c[i,j,field,m,k]+fa*c[i,j+1,field,m,k])
                                +fr*((1-fa)*c[i+1,j,field,m,k]+fa*c[i+1,j+1,field,m,k]))
                if need_integral:
                    for k in range(nodes):
                        value=0.
                        for field in range(2):
                            for m in range(10):
                                coefficient=coefficients[field,m]
                                if coefficient!=0.:value+=coefficient*_mix(primitive,i,j,k,field,m,fr,fa)
                        scalar_integral[k]=value
            else:
                for k in range(nodes):
                    value=0.;antiderivative=0.
                    for field in range(2):
                        for m in range(10):
                            coefficient=coefficients[field,m]
                            if coefficient!=0.:
                                value+=coefficient*_mix(c,i,j,k,field,m,fr,fa)
                                if need_integral:antiderivative+=coefficient*_mix(primitive,i,j,k,field,m,fr,fa)
                    scalar_cdf[k]=value;scalar_integral[k]=antiderivative
            if quadratic_time:
                error=0.
                for k in range(len(scalar_cdf)-1):error+=abs(scalar_cdf[k+1]-scalar_cdf[k])*quadratic_tv[k]
                if error<=np.max(quadratic_tv)*abs(local_charge):
                    deposit_cdf(scalar_cdf,clocks[group],spreads[group],front,width,edges,sinh_u,quadratic_a,quadratic_b,bins)
                    continue
                guards+=1
                for k in range(len(scalar_cdf)):
                    value=0.
                    for field in range(2):
                        for m in range(10):value+=coefficients[field,m]*_mix(primitive,i,j,k,field,m,fr,fa)
                    scalar_integral[k]=value
            previous=_scalar_edge(edges[0],clocks[group],spreads[group],front,width,maximum,scalar_cdf,scalar_integral)
            for b in range(len(bins)):
                value=_scalar_edge(edges[b+1],clocks[group],spreads[group],front,width,maximum,scalar_cdf,scalar_integral)
                bins[b]+=value-previous;previous=value
    return charge,bins,guards


@njit(cache=True,nogil=True)
def charge_groups(radii,angles,moments,log_radii,R,g,mu,mus,s0w,s2w,speed_max,hit_charge,miss_charge):
    """Charge-only pass: do not read time-error tables for OM screening."""
    charges=np.zeros(len(radii))
    for group in range(len(radii)):
        r=radii[group];theta=angles[group]
        if r<=R or math.log(r)<log_radii[0] or math.log(r)>log_radii[-1]:
            raise ValueError('pencil geometry outside the first-entry Q1 atlas')
        hit,i,j,fr,fa,_,_=_geometry(r,theta,R,log_radii,hit_charge.shape[1],miss_charge.shape[1],g,speed_max)
        q=hit_charge if hit else miss_charge
        scale=_spectral_scale(r,R,mu,mus,s0w,s2w)
        for field in range(2):
            for m in range(10):
                coefficient=moments[group,field,m]*scale[field]
                value=((1-fr)*((1-fa)*q[i,j,field,m]+fa*q[i,j+1,field,m])
                        +fr*((1-fa)*q[i+1,j,field,m]+fa*q[i+1,j+1,field,m]))
                charges[group]+=coefficient*value
    return charges


@njit(cache=True,nogil=True)
def charge_and_time_bounds(radii,angles,moments,log_radii,R,g,mu,mus,s0w,s2w,speed_max,
                           hit_charge,miss_charge,hit_error,miss_error):
    """Charge and rigorous coarse/fine CDF total-variation bounds per group."""
    charges=np.zeros(len(radii));bounds=np.zeros(len(radii))
    indices=np.empty((len(radii),3),np.int64);values=np.empty((len(radii),6))
    for group in range(len(radii)):
        r=radii[group];theta=angles[group]
        if r<=R or math.log(r)<log_radii[0] or math.log(r)>log_radii[-1]:
            raise ValueError('pencil geometry outside the first-entry Q1 atlas')
        hit,i,j,fr,fa,front,width=_geometry(r,theta,R,log_radii,hit_charge.shape[1],miss_charge.shape[1],g,speed_max)
        q=hit_charge if hit else miss_charge;e=hit_error if hit else miss_error
        scale=_spectral_scale(r,R,mu,mus,s0w,s2w)
        indices[group,0]=int(hit);indices[group,1]=i;indices[group,2]=j
        values[group,0]=fr;values[group,1]=fa;values[group,2]=front;values[group,3]=width
        values[group,4]=scale[0];values[group,5]=scale[1]
        for field in range(2):
            for m in range(10):
                coefficient=moments[group,field,m]*scale[field]
                value=((1-fr)*((1-fa)*q[i,j,field,m]+fa*q[i,j+1,field,m])
                        +fr*((1-fa)*q[i+1,j,field,m]+fa*q[i+1,j+1,field,m]))
                bound=((1-fr)*((1-fa)*e[i,j,field,m]+fa*e[i,j+1,field,m])
                        +fr*((1-fa)*e[i+1,j,field,m]+fa*e[i+1,j+1,field,m]))
                charges[group]+=coefficient*value;bounds[group]+=abs(coefficient)*bound
    return charges,bounds,(indices,values)
