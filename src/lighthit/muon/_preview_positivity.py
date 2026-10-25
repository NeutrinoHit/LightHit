"""Bernstein certificate for cubic OM responses over every orientation.

The propagation look has (x,z) in the unit disk. Subdivision bounds the whole
polynomial, including between sampled looks. An unresolved cell requests an
independent positive reference; it is never repaired by clipping a value.
"""
import math
import numpy as np
from numba import njit


def observer_bernstein_map(polynomial):
    a0,a1,a2,a3=np.pad(np.asarray(polynomial,float),(0,4-len(polynomial)))
    response=np.zeros((10,10))
    for j in range(10):
        m=np.eye(10)[j];m0,x,z,xx,xz,zz,xxx,xxz,xzz,zzz=m
        response[:,j]=[a0*m0+a2*(m0-xx-zz),a1*x+3*a3*(x-xxx-xzz),
            a1*z+3*a3*(z-xxz-zzz),a2*(2*xx+zz-m0),2*a2*xz,a2*(xx+2*zz-m0),
            a3*(4*xxx+3*xzz-3*x),3*a3*(2*xxz+zzz-z),3*a3*(xxx+2*xzz-x),a3*(3*xxz+4*zzz-3*z)]
    powers=[(0,0),(1,0),(0,1),(2,0),(1,1),(0,2),(3,0),(2,1),(1,2),(0,3)]
    transform=np.zeros((16,10))
    for column,(i,j) in enumerate(powers):
        unit=np.zeros((4,4))
        for p in range(i+1):
            for q in range(j+1):unit[p,q]=math.comb(i,p)*math.comb(j,q)*2**(p+q)*(-1)**(i+j-p-q)
        for k in range(4):
            for l in range(4):
                transform[4*k+l,column]=sum(unit[p,q]*math.comb(k,p)/math.comb(3,p)*math.comb(l,q)/math.comb(3,q)
                    for p in range(k+1) for q in range(l+1))
    return np.ascontiguousarray(transform@response),np.ascontiguousarray(transform)


@njit(cache=True)
def _split(b,axis):
    low=np.empty((4,4));high=np.empty((4,4))
    for j in range(4):
        v=b[:,j] if axis==0 else b[j,:]
        a=(v[0]+v[1])/2;c=(v[1]+v[2])/2;d=(v[2]+v[3])/2
        e=(a+c)/2;f=(c+d)/2;middle=(e+f)/2
        if axis==0:
            low[:,j]=np.array([v[0],a,e,middle]);high[:,j]=np.array([middle,f,d,v[3]])
        else:
            low[j,:]=np.array([v[0],a,e,middle]);high[j,:]=np.array([middle,f,d,v[3]])
    return low,high


@njit(cache=True,nogil=True)
def positive_on_disk(coefficients):
    scale=np.max(np.abs(coefficients))
    if scale==0:return True
    eps=np.finfo(np.float64).eps;roundoff=512*eps
    depth_limit=2*int(math.ceil(-math.log2(math.sqrt(eps))))
    stack=np.empty((depth_limit+2,4,4));boxes=np.empty((depth_limit+2,4));depth=np.empty(depth_limit+2,np.int64)
    stack[0]=coefficients.reshape(4,4)/scale;boxes[0]=np.array([-1.,1.,-1.,1.]);depth[0]=0;used=1
    weights=np.array([1.,3.,3.,1.])/8
    while used:
        used-=1;b=stack[used].copy();x0,x1,z0,z1=boxes[used];level=depth[used]
        cx=0. if x0<=0<=x1 else min(abs(x0),abs(x1))
        cz=0. if z0<=0<=z1 else min(abs(z0),abs(z1))
        if cx*cx+cz*cz>1+64*eps:continue
        if np.min(b)>roundoff:continue
        if np.max(b)<-roundoff:return False
        x=(x0+x1)/2;z=(z0+z1)/2;value=0.
        for i in range(4):
            for j in range(4):value+=weights[i]*weights[j]*b[i,j]
        if x*x+z*z<=1 and value<-roundoff:return False
        if level>=depth_limit:return False
        axis=0 if x1-x0>=z1-z0 else 1;left,right=_split(b,axis)
        stack[used]=left;stack[used+1]=right;depth[used:used+2]=level+1
        if axis==0:
            boxes[used]=np.array([x0,x,z0,z1]);boxes[used+1]=np.array([x,x1,z0,z1])
        else:
            boxes[used]=np.array([x0,x1,z0,z]);boxes[used+1]=np.array([x0,x1,z,z1])
        used+=2
    return True


@njit(cache=True,nogil=True)
def certified_moments(moments,transform):
    result=np.empty(len(moments),np.bool_)
    for i in range(len(moments)):result[i]=positive_on_disk(transform@moments[i])
    return result
