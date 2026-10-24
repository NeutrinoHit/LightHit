"""Eight precomputed CDF moments, with scalar density coefficients."""
import math
import numpy as np
from numba import njit
def polynomial_lower(poly):
    derivative=np.polynomial.polynomial.polyder(poly)
    roots=np.polynomial.polynomial.polyroots(derivative)
    nodes=[-1.,1.]+[float(z.real) for z in roots if abs(z.imag)<1e-12 and -1<z.real<1]
    return float(np.polynomial.polynomial.polyval(nodes,poly).min()-64*np.finfo(float).eps*np.abs(poly).sum())



def make_basis(poly, relative_density_tolerance, max_degree=64):
    poly=np.asarray(poly,float)
    if poly.ndim!=1 or len(poly)!=4 or not np.isfinite(poly).all():
        raise ValueError('require four finite cubic acceptance coefficients')
    if not np.isfinite(relative_density_tolerance) or relative_density_tolerance<=0:
        raise ValueError('require a finite positive density tolerance')
    minimum=polynomial_lower(poly)
    expected=tuple(float(x) for x in poly)
    requested=float(relative_density_tolerance)
    zero=(0.,0.,0.,0.,0.,0.,0.,0.)

    @njit(cache=True)
    def prepare(c,s,axial,across,side,mu,tangent,delta,a2,alpha):
        if len(alpha)!=4 or minimum<=0 or tangent<=0 or a2<0 or a2>=1 or max_degree<7:
            return zero,-1.,np.inf
        for index in range(4):
            if alpha[index]!=expected[index]:return zero,-1.,np.inf
        if mu<0 or delta<0:return zero,-1.,np.inf
        m=c*axial+s*across;mt=-2*s*across*a2
        v1=4*s*s*side*side*a2;v2=-v1*a2
        p0,p1,p2,p3=alpha
        derivative=p1+2*p2*m+3*p3*m*m;curvature=p2+3*p3*m
        A0=2*(((p3*m+p2)*m+p1)*m+p0)
        A1=2*(derivative*mt+curvature*v1)
        A2=2*(curvature*(mt*mt+v2)+3*p3*mt*v1)
        A3=2*p3*(mt*mt*mt+3*mt*v2)
        j1=.5*a2;j2=.375*a2*a2
        B0=A0;B1=A1+j1*A0;B2=A2+j1*A1+j2*A0
        omitted=(abs(A3+j1*A2+j2*A1)+abs(j1*A3+j2*A2)+abs(j2*A3)
            +(abs(A0)+abs(A1)+abs(A2)+abs(A3))*.3125*a2**3/(1-a2))
        chi=mu*math.sqrt(2*tangent*delta);e=mu*delta;d=delta/(2*tangent);f=chi*d/2
        exponent=chi+f+e
        if exponent>math.log(np.finfo(np.float64).max):return zero,-1.,np.inf
        amplification=(1/(1-exponent) if exponent<1 else math.exp(exponent))/(2*minimum)
        weight=abs(B0)+abs(B1)+abs(B2)
        remainder=e**3/6+.5*f*f*(1+e+e*e/2)+chi*d*d/8
        rounding=256*np.finfo(np.float64).eps*(abs(A0)+abs(A1)+abs(A2)+abs(A3)+weight)
        bound=amplification*(omitted+weight*remainder+rounding)
        if bound>=requested:return zero,-1.,bound
        b0=B0+B1+B2;b2=-B1-2*B2;b4=B2
        g0=b0;g2=b2-e*b0;g4=b4-e*b2+e*e*b0/2
        g6=-e*b4+e*e*b2/2;g8=e*e*b4/2
        q1=chi;q2=q1*chi/2;q3=q2*chi/3;q4=q3*chi/4
        q5=q4*chi/5;q6=q5*chi/6;q7=q6*chi/7
        T7=q7;T6=q6+T7;T5=q5+T6;T4=q4+T5
        T3=q3+T4;T2=q2+T3;T1=q1+T2;T0=1+T1
        tail=(abs(g2)*T6+abs(g4)*T4+abs(g6)*T2+abs(g8)*T0
            +f*(abs(g0)*T5+abs(g2)*T3+abs(g4)*T1+(abs(g6)+abs(g8))*T0))
        bound+=amplification*(tail+weight*q7*chi/8)
        if bound>requested:return zero,-1.,bound
        C0=g0
        C1=q1*g0
        C2=q2*g0+g2
        C3=q3*g0+q1*g2+f*g0
        C4=q4*g0+q2*g2+g4+q1*f*g0
        C5=q5*g0+q3*g2+q1*g4+f*g2+q2*f*g0
        C6=q6*g0+q4*g2+q2*g4+g6+q1*f*g2+q3*f*g0
        C7=q7*g0+q5*g2+q3*g4+q1*g6+f*g4+q2*f*g2+q4*f*g0
        return (C0,C1,C2,C3,C4,C5,C6,C7),math.exp(-mu*tangent),bound

    @njit(cache=True)
    def cdf(upper,a,scale):
        if upper<=0:return 0.
        if upper>=1:
            return scale*(a[0]+a[1]*math.pi/4+a[2]*2/3+a[3]*3*math.pi/16
                +a[4]*8/15+a[5]*5*math.pi/32+a[6]*16/35+a[7]*35*math.pi/256)
        y=math.sqrt((1-upper)*(1+upper));y2=y*y;y3=y2*y
        F0=upper;F1=(math.asin(upper)+upper*y)/2
        F2=(2*F0+upper*y2)/3;F3=(3*F1+upper*y3)/4
        F4=(4*F2+upper*y2*y2)/5;F5=(5*F3+upper*y3*y2)/6
        F6=(6*F4+upper*y3*y3)/7;F7=(7*F5+upper*y3*y3*y)/8
        return scale*(a[0]*F0+a[1]*F1+a[2]*F2+a[3]*F3+a[4]*F4+a[5]*F5+a[6]*F6+a[7]*F7)

    return prepare,cdf,dict(minimum_acceptance=minimum,relative_density_tolerance=requested,
        maximum_degree=7,full_moment_bytes=8*8,representation='scalar eight-moment density')
