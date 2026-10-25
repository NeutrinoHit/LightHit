"""Positive Neumann-series reference with an analytic one-scattering tail.

The geometric prefix proposal has unbounded collision order. Every prefix
is killed at an earlier glass entry. Only the final one-scattering integral
uses a table; no event, angular source truncation or response fitting enters.
"""
import math
import numpy as np
from numba import njit
from .._receiver_first_lookup import _geometry
from .._receiver_first_groups import _second,_third


@njit(cache=True,inline='always')
def _rotate(x,y,z,c,s,cp,sp):
    radial=c*x+s*z
    return radial*cp-y*sp,radial*sp+y*cp,-s*x+c*z


@njit(cache=True,nogil=True)
def first_tail(dx,dy,dz,sx,sy,sz,band,arguments,out):
    R,g,log_r,hit,miss,mu,ms,speed=arguments
    r=math.sqrt(dx*dx+dy*dy+dz*dz)
    if r<math.exp(log_r[0]) or r>math.exp(log_r[-1]):raise ValueError('positive reference needs a wider Q1 charge table')
    cosine=min(1.,max(-1.,(dx*sx+dy*sy+dz*sz)/r))
    ex=dx/r-cosine*sx;ey=dy/r-cosine*sy;ez=dz/r-cosine*sz
    sine=math.sqrt(ex*ex+ey*ey+ez*ez)
    if sine>0:ex/=sine;ey/=sine;ez/=sine
    else:
        ex=1.-sx*sx;ey=-sx*sy;ez=-sx*sz
        norm=math.sqrt(ex*ex+ey*ey+ez*ez)
        if norm==0.:ex=0.;ey=1.;ez=0.
        else:ex/=norm;ey/=norm;ez/=norm
    X=np.array([ex,sy*ez-sz*ey,sx]);Z=np.array([ez,sx*ey-sy*ex,sz])
    branch,i,j,fr,fa,_,_=_geometry(r,math.atan2(sine,cosine),R,log_r,hit.shape[2],miss.shape[2],g,speed[band])
    table=hit if branch else miss;q=np.empty(10)
    for m in range(10):q[m]=((1-fr)*((1-fa)*table[band,i,j,m]+fa*table[band,i,j+1,m])
        +fr*((1-fa)*table[band,i+1,j,m]+fa*table[band,i+1,j+1,m]))
    raw=np.zeros(20);raw[0]=q[0];raw[1]=q[1];raw[3]=q[2];raw[4]=q[3];raw[6]=q[4]
    raw[7]=q[0]-q[3]-q[5];raw[9]=q[5];raw[10]=q[6];raw[12]=q[7]
    raw[13]=q[1]-q[6]-q[8];raw[15]=q[8];raw[17]=q[2]-q[7]-q[9];raw[19]=q[9]
    out[0]=q[0];out[1]=X[0]*q[1]+X[2]*q[2];out[2]=Z[0]*q[1]+Z[2]*q[2]
    out[3]=_second(raw,X,X);out[4]=_second(raw,X,Z);out[5]=_second(raw,Z,Z)
    out[6]=_third(raw,X,X,X);out[7]=_third(raw,X,X,Z);out[8]=_third(raw,X,Z,Z);out[9]=_third(raw,Z,Z,Z)
    out*=ms[band]*math.exp(-mu[band]*(r-R))/r


@njit(cache=True,nogil=True)
def positive_multiple_cell(samples,seed,rho_bounds,z_bounds,R,g,mu,ms,source_weights,cones,
                           evaluate,arguments,proposal=.5,fixed_prefix=0,radial_high=np.inf):
    np.random.seed(seed);total=np.zeros(10);square=np.zeros(10);tail=np.empty(10)
    for sample in range(samples):
        rho=math.sqrt(rho_bounds[0]**2+np.random.random()*(rho_bounds[1]**2-rho_bounds[0]**2))
        z=z_bounds[0]+np.random.random()*(z_bounds[1]-z_bounds[0]);radius=math.hypot(rho,z)
        if radius<=R:raise ValueError('source cell intersects the glass interior')
        if radius>radial_high:continue
        count=fixed_prefix if fixed_prefix>0 else 1+int(math.log1p(-np.random.random())/math.log(proposal))
        probability=1. if fixed_prefix>0 else (1-proposal)*proposal**(count-1)
        path=np.zeros((count+1,3));directions=np.zeros((count+1,3));directions[0,2]=1.;flights=np.empty(count)
        for step in range(count):
            length=-math.log1p(-np.random.random());flights[step]=length
            path[step+1]=path[step]+length*directions[step]
            x,y,zz=directions[step];u=np.random.random()
            c=2*u-1 if abs(g)<1e-12 else (1+g*g-((1-g*g)/(1-g+2*g*u))**2)/(2*g)
            sn=math.sqrt(max(0.,1-c*c));phi=2*math.pi*np.random.random()
            if abs(zz)<.9:ax=-y;ay=x;az=0.
            else:ax=0.;ay=-zz;az=y
            norm=math.sqrt(ax*ax+ay*ay+az*az);ax/=norm;ay/=norm;az/=norm
            bx=y*az-zz*ay;by=zz*ax-x*az;bz=x*ay-y*ax
            cp=math.cos(phi);sp=math.sin(phi)
            directions[step+1,0]=c*x+sn*(cp*ax+sp*bx)
            directions[step+1,1]=c*y+sn*(cp*ay+sp*by)
            directions[step+1,2]=c*zz+sn*(cp*az+sp*bz)
        phi=2*math.pi*np.random.random();cp=math.cos(phi);sp=math.sin(phi);value=np.zeros(10)
        for band in range(len(mu)):
            if source_weights[band]==0 or ms[band]==0:continue
            c=cones[band];s=math.sqrt(max(0.,1-c*c));alive=True
            for step in range(count):
                x,y,zz=_rotate(path[step,0],path[step,1],path[step,2],c,s,cp,sp)
                x=x/mu[band]-rho;y/=mu[band];zz=zz/mu[band]-z
                sx,sy,sz=_rotate(directions[step,0],directions[step,1],directions[step,2],c,s,cp,sp)
                dot=x*sx+y*sy+zz*sz;disc=dot*dot-(x*x+y*y+zz*zz-R*R)
                if R>0 and disc>=0:
                    entry=-dot-math.sqrt(disc)
                    if 0<=entry<=flights[step]/mu[band]:alive=False;break
            if not alive:continue
            x,y,zz=_rotate(path[count,0],path[count,1],path[count,2],c,s,cp,sp)
            sx,sy,sz=_rotate(directions[count,0],directions[count,1],directions[count,2],c,s,cp,sp)
            evaluate(rho-x/mu[band],-y/mu[band],z-zz/mu[band],sx,sy,sz,band,arguments,tail)
            weight=source_weights[band]*(ms[band]/mu[band])**count/probability
            value+=weight*tail
        total+=value;square+=value*value
    mean=total/samples
    return mean,np.sqrt(np.maximum(0.,(square/samples-mean*mean)/(samples-1)))
