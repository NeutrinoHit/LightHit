"""Real polynomial cuts by Bernstein sign isolation, without dense eigensolves."""
import math
import numpy as np
from numba import njit

CHOOSE=np.zeros((10,10))
for i in range(10):
    for j in range(i+1):CHOOSE[i,j]=math.comb(i,j)


@njit(cache=True)
def value_derivative(p,x):
    f=p[-1];d=0.
    for j in range(len(p)-2,-1,-1):d=d*x+f;f=f*x+p[j]
    return f,d


@njit(cache=True)
def add_roots(p,a,b,cuts,count):
    degree=len(p)-1
    while degree>0 and p[degree]==0.:degree-=1
    if degree==0:return count
    scale=np.max(np.abs(p))
    p=p[:degree+1]/scale
    power=np.zeros(degree+1)
    for j in range(degree,-1,-1):
        for k in range(degree,0,-1):power[k]=a*power[k]+(b-a)*power[k-1]
        power[0]=a*power[0]+p[j]
    bern=np.zeros(degree+1)
    for i in range(degree+1):
        for j in range(i+1):bern[i]+=power[j]*CHOOSE[i,j]/CHOOSE[degree,j]
    tolerance=32*np.finfo(np.float64).eps*max(1.,abs(a),abs(b))
    capacity=degree*(2+int(math.ceil(math.log2((b-a)/tolerance))))
    capacity=max(2,capacity)
    stack=np.empty((capacity,degree+1));left=np.empty(capacity);right=np.empty(capacity)
    stack[0]=bern;left[0]=a;right[0]=b;pending=1
    while pending:
        pending-=1;co=stack[pending].copy();lo=left[pending];hi=right[pending]
        previous=0.;variation=0
        for v in co:
            if v!=0:
                if previous!=0 and (v>0)!=(previous>0):variation+=1
                previous=v
        if co[0]==0:cuts[count]=lo;count+=1
        if co[-1]==0:cuts[count]=hi;count+=1
        if variation==0:continue
        if hi-lo<=tolerance:
            cuts[count]=(lo+hi)/2;count+=1;continue
        if variation==1:
            fl=value_derivative(p,lo)[0];x=(lo+hi)/2
            while True:
                f,d=value_derivative(p,x)
                if f==0:break
                if (f>0)==(fl>0):lo=x;fl=f
                else:hi=x
                if hi-lo<=tolerance:x=(lo+hi)/2;break
                y=x-f/d if d!=0 else (lo+hi)/2
                if y<=lo or y>=hi:y=(lo+hi)/2
                if y==x:break
                x=y
            cuts[count]=x;count+=1;continue
        work=co.copy();bl=np.empty(degree+1);br=np.empty(degree+1)
        bl[0]=work[0];br[-1]=work[-1]
        for j in range(1,degree+1):
            for k in range(degree+1-j):work[k]=(work[k]+work[k+1])/2
            bl[j]=work[0];br[degree-j]=work[degree-j]
        midpoint=(lo+hi)/2
        stack[pending]=br;left[pending]=midpoint;right[pending]=hi;pending+=1
        stack[pending]=bl;left[pending]=lo;right[pending]=midpoint;pending+=1
    return count


def roots(p,a=0.,b=1.):
    cuts=np.empty(256)
    count=add_roots(np.asarray(p,float),a,b,cuts,0)
    return np.unique(cuts[:count])


@njit(cache=True)
def plus(a,b,factor=1.):
    out=np.zeros(max(len(a),len(b)))
    out[:len(a)]+=a;out[:len(b)]+=factor*b
    return out


@njit(cache=True)
def physical_roots(cuts,start,count,kind,arg,z,rho,R,L,beta,t0,tau,target,n0,dn,g0,dg):
    keep=start
    for j in range(start,count):
        q=cuts[j];n=n0+dn*q;G=(g0+dg*q)/.299792458
        if beta*n<=1:continue
        c=1/(beta*n);s=math.sqrt((1-c)*(1+c));D=rho/s;sc=z-rho*c/s;spread=R/s
        tol_a=np.sqrt(np.finfo(np.float64).eps)*(R+L+abs(sc))
        value=np.inf;valid=False
        if kind==0:
            A=(tau-G*c)*spread;B=G*R;h=-A/math.sqrt(A*A+B*B)
            a=sc+spread*h;valid=-tol_a<=a<=L+tol_a
            value=t0+tau*sc+G*D-math.sqrt(A*A+B*B)
        elif kind==1:
            if G*G>tau*tau:
                a=z-tau*math.sqrt((rho*rho-R*R)/(G*G-tau*tau))
                valid=-tol_a<=a<=L+tol_a and abs((a-sc)/spread)<=1+tol_a/spread
                value=t0+tau*a+G*math.sqrt(rho*rho+(z-a)**2-R*R)
        elif kind==2:
            a=sc+arg*spread;valid=-tol_a<=a<=L+tol_a
            value=t0+tau*a+G*(D-c*(a-sc))
        else:
            a=arg;h=(a-sc)/spread
            valid=abs(h)<=1+tol_a/spread
            if kind==3:
                value=t0+tau*a+G*(D-c*(a-sc)-R*math.sqrt(max(0.,1-h*h)))
            elif kind==4:
                value=t0+tau*a+G*math.sqrt(rho*rho+(z-a)**2-R*R)
            else:value=t0+tau*a+G*(D-c*(a-sc))
        tolerance=np.sqrt(np.finfo(np.float64).eps)*(1+abs(target)+abs(t0)+(abs(tau)+G)*(abs(z)+rho))
        if valid and abs(value-target)<=tolerance:
            cuts[keep]=q;keep+=1
    return keep


@njit(cache=True)
def time_cuts(z,rho,R,L,beta,t0,t1,origin,n0,dn,g0,dg,low,high,edges):
    tau=(t1-t0)/L
    near=math.hypot(rho,z if z<0 else (z-L if z>L else 0.))
    far=max(math.hypot(rho,z),math.hypot(rho,z-L))
    minimum_group=min(g0+dg*low,g0+dg*high)/.299792458
    maximum_group=max(g0+dg*low,g0+dg*high)/.299792458
    first=np.searchsorted(edges,min(t0,t1)+minimum_group*(near-R)-origin,side='right')
    last=np.searchsorted(edges,max(t0,t1)+maximum_group*far-origin,side='left')
    if first>=last:return np.array([low,high])
    X=beta*np.array([n0,dn]);G=np.array([g0,dg])/.299792458
    XX=np.convolve(X,X);GG=np.convolve(G,G)
    S=XX.copy();S[0]-=1
    A=rho*np.convolve(X,G);A[0]-=rho*tau
    AA=np.convolve(A,A)
    D=plus(np.convolve(XX,plus(GG,np.array([tau*tau]))),np.convolve(X,G),-2*tau)
    rmin=math.hypot(rho,z if z<0 else (z-L if z>L else 0.))
    rmax=max(math.hypot(rho,z),math.hypot(rho,z-L))
    earliest=min(t0,t1)+min(g0,g0+dg)/.299792458*(rmin-R)-origin
    latest=max(t0,t1)+max(g0,g0+dg)/.299792458*rmax-origin
    first=np.searchsorted(edges,earliest,side='right');last=np.searchsorted(edges,latest,side='left')
    cuts=np.empty(2+84*(last-first));cuts[0]=low;cuts[1]=high;count=2
    for j in range(first,last):
        target=origin+edges[j];delta=target-t0-tau*z
        tmp=plus(plus(AA,S,delta*delta),D,-R*R)
        polynomial=plus(np.convolve(tmp,tmp),np.convolve(AA,S),-4*delta*delta)
        start=count;count=add_roots(polynomial,low,high,cuts,count)
        count=physical_roots(cuts,start,count,0,0.,z,rho,R,L,beta,t0,tau,target,n0,dn,g0,dg)
        polynomial=-(rho*rho-R*R)*plus(GG,np.array([-tau*tau]))
        polynomial[0]+=delta*delta
        start=count;count=add_roots(polynomial,low,high,cuts,count)
        count=physical_roots(cuts,start,count,1,0.,z,rho,R,L,beta,t0,tau,target,n0,dn,g0,dg)
        for sign in (-1.,1.):
            E=plus(A,plus(tau*X,G,-1.),sign*R)
            polynomial=plus(delta*delta*S,np.convolve(E,E),-1.)
            start=count;count=add_roots(polynomial,low,high,cuts,count)
            count=physical_roots(cuts,start,count,2,sign,z,rho,R,L,beta,t0,tau,target,n0,dn,g0,dg)
        for a in (0.,L):
            zz=z-a;B=zz*zz+rho*rho-R*R;dt=target-t0-tau*a
            Q=np.convolve(X,plus(B*GG,np.array([dt*dt])))
            Q=plus(Q,G,-2*dt*zz)
            polynomial=plus(np.convolve(Q,Q),np.convolve(GG,S),-4*dt*dt*rho*rho)
            start=count;count=add_roots(polynomial,low,high,cuts,count)
            count=physical_roots(cuts,start,count,3,a,z,rho,R,L,beta,t0,tau,target,n0,dn,g0,dg)
            polynomial=-B*GG;polynomial[0]+=dt*dt
            start=count;count=add_roots(polynomial,low,high,cuts,count)
            count=physical_roots(cuts,start,count,4,a,z,rho,R,L,beta,t0,tau,target,n0,dn,g0,dg)
    return cuts[:count]


if __name__=='__main__':
    from numpy.polynomial import Polynomial as P
    rng=np.random.default_rng(312)
    maximum=0.
    for n in range(1,9):
        known=np.sort(rng.uniform(-.5,1.5,n))
        p=P.fromroots(known)
        got=roots(p.coef)
        expected=known[(known>=0)&(known<=1)]
        if len(got)!=len(expected):raise AssertionError((known,got))
        maximum=max(maximum,np.max(abs(got-expected),initial=0))
    print(dict(maximum_root_error=maximum))
