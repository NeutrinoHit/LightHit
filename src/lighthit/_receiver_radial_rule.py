"""Positive two-node Gaussian radial rule for a finite angular receiver.

Use dnu=dP/sqrt(theta^2+cap^2) for a well-conditioned importance measure.
Its first and third moments are known exactly; the same Laplace rule as the
point screen supplies the other two. Multiplying its weights by the node's
regularized radius gives ordinary photon probabilities of total mass one.
The detector kernel itself is never regularized or rescaled.
"""
import math
from numba import njit


@njit(cache=True)
def radial_rule(u,v,var_u,var_v,cap,tangent,weighted_sec2):
    if var_u==0 and var_v==0:return 1.,math.hypot(u,v),0.,0.
    scale=2*(var_u+var_v)+u*u+v*v+cap*cap
    if scale<=0:return 1.,0.,0.,0.
    s0=1/math.sqrt(scale);inverse=0.;radial=0.
    for k in range(len(tangent)):
        t=s0*tangent[k];a=1+2*t*t*var_u;b=1+2*t*t*var_v
        mass=weighted_sec2[k]*math.exp(-t*t*(u*u/a+v*v/b+cap*cap))/math.sqrt(a*b)
        inverse+=mass
        radial+=mass*(var_u/a+var_v/b+u*u/(a*a)+v*v/(b*b)+cap*cap)
    multiplier=2/math.sqrt(math.pi)*s0;inverse*=multiplier;radial*=multiplier
    if inverse<=0:raise ValueError('finite radial quadrature has no positive mass')
    mean=1/inverse;second=radial/inverse;variance=second-mean*mean
    if variance<=0:
        if mean<cap:raise ValueError('finite radial quadrature has invalid support')
        return 1.,math.sqrt((mean-cap)*(mean+cap)),0.,0.
    third=(u*u+v*v+var_u+var_v+cap*cap)/inverse
    other=(third-2*mean*second+mean**3)/variance
    disc=math.sqrt((mean-other)**2+4*variance)
    low=(mean+other-disc)/2;high=(mean+other+disc)/2
    if low<cap or high<=low:raise ValueError('finite radial quadrature has invalid support')
    first=inverse*(high-mean)/(high-low);second_weight=inverse-first
    p0=first*low;p1=second_weight*high
    if p0<=0 or p1<0:raise ValueError('finite radial quadrature has invalid weights')
    t0=math.sqrt((low-cap)*(low+cap));t1=math.sqrt((high-cap)*(high+cap))
    if t1>math.pi:raise ValueError('finite radial quadrature exceeds angular support')
    return p0,t0,p1,t1
