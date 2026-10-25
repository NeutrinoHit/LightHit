"""Stationary cone Q1 arrival moments on prepared first-entry pencil kernels."""
import math
import numpy as np
from numba import njit
from .._receiver_first_lookup import _geometry
from .._receiver_first_groups import _second,_third


@njit(cache=True,nogil=True)
def cone_first_moments(rho,z,beta,R,g,log_r,hit,miss,mu,ms,speed,s0,s2,phase,gx,gw,panel_width):
    result=np.zeros((len(z),10));raw=np.zeros(20);q=np.empty(10)
    for iz in range(len(z)):
        r=math.hypot(rho,z[iz])
        # The interior is outside this exterior kernel's domain. These cells
        # must force detailed evaluation through the preview's near-source guard.
        if r<=math.exp(log_r[0]):continue
        if r>math.exp(log_r[-1]):raise ValueError('preview point outside prepared Q1 range')
        hx=rho/r;hz=z[iz]/r;gamma=math.atan2(rho,z[iz]);cap=math.asin(R/r)
        for band in range(len(mu)):
            if beta*phase[band]<=1:continue
            cs=1/(beta*phase[band]);sn=math.sqrt(1-cs*cs)
            physical=(s0[band]-s2[band]/(beta*beta))*ms[band]*math.exp(-mu[band]*(r-R))/r
            if physical==0:continue
            kp=math.sqrt(max(0.,hx*sn));gap=abs(gamma-math.acos(cs))
            scale=math.sqrt(gap*gap+cap*cap)/kp if kp>0 else math.pi
            maximum=math.asinh(math.pi/scale);cross=0.
            if kp>0:
                c=(math.sqrt((1-R/r)*(1+R/r))-hz*cs)/(kp*kp)
                if -1<c<1:cross=math.asinh(math.acos(c)/scale)
            breaks=np.array([0.,cross,maximum])
            for interval in range(2):
                low,high=breaks[interval],breaks[interval+1]
                if high<=low:continue
                panels=max(1,int(math.ceil((high-low)/panel_width)))
                for p in range(panels):
                    a=low+(high-low)*p/panels;b=low+(high-low)*(p+1)/panels
                    for n in range(len(gx)):
                        w=(a+b)/2+(b-a)/2*gx[n];phi=scale*math.sinh(w)
                        weight=physical*scale*math.cosh(w)*(b-a)/2*gw[n]/math.pi
                        dx=sn*math.cos(phi);dy=sn*math.sin(phi);dz=cs
                        cosine=min(1.,max(-1.,hx*dx+hz*dz))
                        ex=hx-cosine*dx;ey=-cosine*dy;ez=hz-cosine*dz
                        sine=math.sqrt(ex*ex+ey*ey+ez*ez);theta=math.atan2(sine,cosine)
                        if sine>0:ex/=sine;ey/=sine;ez/=sine
                        else:ex=cs;ey=0.;ez=-sn
                        X=np.array([ex,dy*ez-dz*ey,dx]);Z=np.array([ez,dx*ey-dy*ex,dz])
                        branch,i,j,fr,fa,_,_=_geometry(r,theta,R,log_r,hit.shape[2],miss.shape[2],g,speed[band])
                        table=hit if branch else miss
                        for k in range(10):
                            q[k]=((1-fr)*((1-fa)*table[band,i,j,k]+fa*table[band,i,j+1,k])
                                +fr*((1-fa)*table[band,i+1,j,k]+fa*table[band,i+1,j+1,k]))
                        raw[0]=q[0];raw[1]=q[1];raw[3]=q[2];raw[4]=q[3];raw[6]=q[4]
                        raw[7]=q[0]-q[3]-q[5];raw[9]=q[5];raw[10]=q[6];raw[12]=q[7]
                        raw[13]=q[1]-q[6]-q[8];raw[15]=q[8];raw[17]=q[2]-q[7]-q[9];raw[19]=q[9]
                        result[iz,0]+=weight*q[0]
                        result[iz,1]+=weight*(X[0]*q[1]+X[2]*q[2])
                        result[iz,2]+=weight*(Z[0]*q[1]+Z[2]*q[2])
                        result[iz,3]+=weight*_second(raw,X,X);result[iz,4]+=weight*_second(raw,X,Z)
                        result[iz,5]+=weight*_second(raw,Z,Z);result[iz,6]+=weight*_third(raw,X,X,X)
                        result[iz,7]+=weight*_third(raw,X,X,Z);result[iz,8]+=weight*_third(raw,X,Z,Z)
                        result[iz,9]+=weight*_third(raw,Z,Z,Z)
    return result
