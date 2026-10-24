"""Exact visible-arc time limits for a point Cherenkov source."""
import math
import numpy as np
from numba import njit
from ._ballistic_interval_numba import _acceptance,_sort_small
from ._surface_ballistic_roots import add_roots,plus,physical_roots


@njit(cache=True,inline='always')
def acceptance(apoly,x):
    value=apoly[-1]
    for j in range(len(apoly)-2,-1,-1):value=value*x+apoly[j]
    return max(0.,value)


@njit(cache=True,inline='always')
def acceptance_pair(apoly,mean,side):
    if len(apoly)==4:
        side2=side*side;mean2=mean*mean
        even=apoly[0]+apoly[1]*mean+apoly[2]*(mean2+side2)+apoly[3]*mean*(mean2+3*side2)
        odd=side*(apoly[1]+2*apoly[2]*mean+apoly[3]*(3*mean2+side2))
        return max(0.,even+odd)+max(0.,even-odd)
    return acceptance(apoly,mean+side)+acceptance(apoly,mean-side)


@njit(cache=True,inline='always')
def phase_at_h(h,rho,along,R,beta):
    r=math.hypot(rho,along);eps=R/r
    ca=math.sqrt(1-(eps*h)**2)
    c=along/r*ca+rho/r*eps*h
    return 1/(beta*c)


@njit(cache=True,inline='always')
def h_at_phase(n,rho,along,R,beta):
    c=1/(beta*n)
    return (rho*c-along*math.sqrt(max(0.,(1-c)*(1+c))))/R


@njit(cache=True)
def insert_source_quadratic(a,b,c,low,high,cuts,count):
    scale=max(abs(a),abs(b),abs(c))
    if scale==0:return count
    a,b,c=a/scale,b/scale,c/scale
    if a==0:
        if b!=0:
            root=-c/b
            if low<root<high:cuts[count]=root;count+=1
        return count
    disc=b*b-4*a*c
    if disc<0:return count
    q=-.5*(b+math.copysign(math.sqrt(disc),b))
    if q==0:
        root=-b/(2*a)
        if low<root<high:cuts[count]=root;count+=1
        return count
    for root in (q/a,c/q):
        if low<root<high:cuts[count]=root;count+=1
    return count


@njit(cache=True)
def source_time_cuts(low,high,z,rho,R,beta,t0,tau,origin,n0,dn,g0,dg,edges):
    near=math.hypot(rho,z-low if z<low else (z-high if z>high else 0.))
    far=max(math.hypot(rho,z-low),math.hypot(rho,z-high))
    Glo=min(g0,g0+dg)/.299792458;Ghi=max(g0,g0+dg)/.299792458
    earliest=min(t0+tau*low,t0+tau*high)+Glo*(near-R)-origin
    latest=max(t0+tau*low,t0+tau*high)+Ghi*far-origin
    first=np.searchsorted(edges,earliest,side='right');last=np.searchsorted(edges,latest,side='left')
    if first>=last:return np.array([low,high])
    cuts=np.empty(2+10*(last-first));cuts[0]=low;cuts[1]=high;count=2
    for endpoint in (0.,1.):
        n=n0+dn*endpoint
        if beta*n<=1:continue
        c=1/(beta*n);s=math.sqrt((1-c)*(1+c));G=(g0+dg*endpoint)/.299792458
        D=rho/s;sc=z-rho*c/s;spread=R/s
        tc=t0+tau*sc+G*D-origin;A=(tau-G*c)*spread;B=G*R;norm=A*A+B*B
        for j in range(first,last):
            delta=edges[j]-tc;disc=norm-delta*delta
            if disc>=0:
                root=B*math.sqrt(disc)
                for h in ((A*delta-root)/norm,(A*delta+root)/norm):
                    a=sc+spread*h
                    if -1<h<1 and low<a<high:cuts[count]=a;count+=1
            slope=tau-G*c
            if slope!=0:
                a=(edges[j]+origin-t0-G*(rho*s+c*z))/slope
                if low<a<high:cuts[count]=a;count+=1
            dt=edges[j]+origin-t0
            count=insert_source_quadratic(tau*tau-G*G,-2*dt*tau+2*G*G*z,
                dt*dt-G*G*(z*z+rho*rho-R*R),low,high,cuts,count)
    _sort_small(cuts,count)
    return cuts[:count]


@njit(cache=True)
def polynomial_time_cuts(ha,hb,rho,along,R,beta,emit,origin,n0,dn,g0,dg,edges):
    qa=(phase_at_h(ha,rho,along,R,beta)-n0)/dn
    qb=(phase_at_h(hb,rho,along,R,beta)-n0)/dn
    low=max(0.,min(qa,qb));high=min(1.,max(qa,qb))
    r=math.hypot(rho,along)
    ga=(g0+low*dg)/.299792458;gb=(g0+high*dg)/.299792458
    earliest=emit+min(ga,gb)*(r-R)-origin
    latest=emit+max(ga,gb)*r-origin
    first=np.searchsorted(edges,earliest,side='right');last=np.searchsorted(edges,latest,side='left')
    if first>=last:return np.array([ha,hb])
    cuts=np.empty(2+32*(last-first));cuts[0]=ha;cuts[1]=hb;count=2
    X=beta*np.array([n0,dn]);G=np.array([g0,dg])/.299792458
    GG=np.convolve(G,G);S=np.convolve(X,X);S[0]-=1
    B=r*r-R*R;temporary=np.empty(32)
    for j in range(first,last):
        target=edges[j]+origin;dt=target-emit
        Q=np.convolve(X,plus(B*GG,np.array([dt*dt])))
        Q=plus(Q,G,-2*dt*along)
        p=plus(np.convolve(Q,Q),np.convolve(GG,S),-4*dt*dt*rho*rho)
        number=add_roots(p,low,high,temporary,0)
        number=physical_roots(temporary,0,number,3,0.,along,rho,R,0.,beta,emit,0.,target,n0,dn,g0,dg)
        for k in range(number):
            h=h_at_phase(n0+dn*temporary[k],rho,along,R,beta)
            if ha<h<hb:cuts[count]=h;count+=1
        Q=plus(dt*X,G,-along)
        p=plus(np.convolve(Q,Q),np.convolve(GG,S),-rho*rho)
        number=add_roots(p,low,high,temporary,0)
        number=physical_roots(temporary,0,number,5,0.,along,rho,R,0.,beta,emit,0.,target,n0,dn,g0,dg)
        for k in range(number):
            h=h_at_phase(n0+dn*temporary[k],rho,along,R,beta)
            if ha<h<hb:cuts[count]=h;count+=1
        if dg!=0:
            q=((target-emit)/math.sqrt(B)*.299792458-g0)/dg
            if low<q<high:
                h=h_at_phase(n0+dn*q,rho,along,R,beta)
                if ha<h<hb:cuts[count]=h;count+=1
    _sort_small(cuts,count)
    return cuts[:count]


@njit(cache=True,inline='always')
def clock(v,rho,along,R,beta,n0,dn,g0,dg,base,front):
    r=math.hypot(rho,along);eps=R/r;h=math.sin(v);cv=math.cos(v)
    ca=math.sqrt(1-eps*eps*h*h)
    c=along/r*ca+rho/r*eps*h
    dc=-along/r*eps*eps*h*cv/ca+rho/r*eps*cv
    ddca=-eps*eps*((cv*cv-h*h)/ca+eps*eps*h*h*cv*cv/ca**3)
    ddc=along/r*ddca-rho/r*eps*h
    n=1/(beta*c);dnv=-n*dc/c;ddnv=n*(2*(dc/c)**2-ddc/c)
    slope=dg/(dn*.299792458)
    G=(g0+dg*(n-n0)/dn)/.299792458;dG=slope*dnv;ddG=slope*ddnv
    path=r*ca;dp=-r*eps*eps*h*cv/ca;ddp=r*ddca
    if front:path-=R*cv;dp+=R*h;ddp+=R*cv
    return base+G*path,dG*path+G*dp,ddG*path+2*dG*dp+G*ddp


@njit(cache=True)
def clock_root(lo,hi,target,rho,along,R,beta,n0,dn,g0,dg,base,front,minimum):
    left=clock(lo,rho,along,R,beta,n0,dn,g0,dg,base,front)
    fl=left[1] if minimum else left[0]-target
    x=(lo+hi)/2
    while True:
        value,derivative,second=clock(x,rho,along,R,beta,n0,dn,g0,dg,base,front)
        f=derivative if minimum else value-target
        d=second if minimum else derivative
        if f==0:return x
        if (f>0)==(fl>0):lo=x;fl=f
        else:hi=x
        mid=(lo+hi)/2
        if mid==lo or mid==hi:return mid
        y=x-f/d if d!=0 else mid
        if y<=lo or y>=hi:y=mid
        if y==x:return x
        x=y


@njit(cache=True)
def time_cuts(ha,hb,rho,along,R,beta,emit,origin,n0,dn,g0,dg,edges):
    r=math.hypot(rho,along);eps=R/r;maxh=max(abs(ha),abs(hb))
    ca=math.sqrt(1-(eps*maxh)**2)
    na=phase_at_h(ha,rho,along,R,beta);nb=phase_at_h(hb,rho,along,R,beta)
    nsmall,nlarge=min(na,nb),max(na,nb)
    G0=(g0+dg*(na-n0)/dn)/.299792458;G1=(g0+dg*(nb-n0)/dn)/.299792458
    Gmin,Gmax=min(G0,G1),max(G0,G1);slope=dg/(dn*.299792458)
    earliest=emit+Gmin*(r-R)-origin;latest=emit+Gmax*r-origin
    first=np.searchsorted(edges,earliest,side='right');last=np.searchsorted(edges,latest,side='left')
    if first>=last:return np.array([ha,hb])
    dcmin=rho/r*eps-abs(along)/r*eps*eps*maxh/ca
    dcmax=rho/r*eps+abs(along)/r*eps*eps*maxh/ca
    cmin=1/(beta*nlarge);cmax=1/(beta*nsmall)
    convex=(along>=0 and slope>=0 and 2*slope*nlarge*dcmax/cmin<Gmin)
    monotone=(slope*nsmall*dcmin/cmax*r*ca>Gmax*R*R*max(0.,-ha)/(r*ca))
    if not convex or (not monotone and slope!=0):
        return polynomial_time_cuts(ha,hb,rho,along,R,beta,emit,origin,n0,dn,g0,dg,edges)
    base=emit-origin;va=math.asin(ha);vb=math.asin(hb)
    da=clock(va,rho,along,R,beta,n0,dn,g0,dg,base,True)[1]
    db=clock(vb,rho,along,R,beta,n0,dn,g0,dg,base,True)[1]
    vm=va if da>=0 else (vb if db<=0 else clock_root(va,vb,0.,rho,along,R,beta,n0,dn,g0,dg,base,True,True))
    cuts=np.empty(3+6*(last-first));cuts[0]=ha;cuts[1]=hb;count=2
    if va<vm<vb:cuts[count]=math.sin(vm);count+=1
    for ie in range(first,last):
        target=edges[ie]
        for branch in range(2):
            lo=va if branch==0 else vm;hi=vm if branch==0 else vb
            if lo>=hi:continue
            fl=clock(lo,rho,along,R,beta,n0,dn,g0,dg,base,True)[0]-target
            fr=clock(hi,rho,along,R,beta,n0,dn,g0,dg,base,True)[0]-target
            if (fl>0)!=(fr>0):
                root=clock_root(lo,hi,target,rho,along,R,beta,n0,dn,g0,dg,base,True,False)
                if va<root<vb:cuts[count]=math.sin(root);count+=1
        split=0. if va<0<vb and slope==0 else vb
        for branch in range(2 if split!=vb else 1):
            lo=va if branch==0 else split;hi=split if branch==0 else vb
            fl=clock(lo,rho,along,R,beta,n0,dn,g0,dg,base,False)[0]-target
            fr=clock(hi,rho,along,R,beta,n0,dn,g0,dg,base,False)[0]-target
            if (fl>0)!=(fr>0):
                root=clock_root(lo,hi,target,rho,along,R,beta,n0,dn,g0,dg,base,False,False)
                if va<root<vb:cuts[count]=math.sin(root);count+=1
        if dg!=0:
            q=((target-base)/math.sqrt(r*r-R*R)*.299792458-g0)/dg
            h=h_at_phase(n0+dn*q,rho,along,R,beta) if 0<q<1 else np.inf
            if ha<h<hb:cuts[count]=h;count+=1
    _sort_small(cuts,count)
    return cuts[:count]


@njit(cache=True,inline='always')
def angular_value(y,c,s,axial,across,side,mu,tangent,delta,a2,alpha):
    u2=max(0.,1-y*y);sinhalf2=a2*u2
    cp=1-2*sinhalf2;sp=2*math.sqrt(max(0.,sinhalf2*(1-sinhalf2)))
    shift=delta*y*y
    distance=tangent+shift-y*math.sqrt(delta*(2*tangent+shift))
    accept=acceptance_pair(alpha,c*axial+s*cp*across,s*sp*side)
    return accept*math.exp(-mu*distance)/math.sqrt(1-sinhalf2)


@njit(cache=True)
def angular_integral(umax,c,s,axial,across,side,mu,tangent,delta,a2,alpha,px,pw,tx,tw):
    if umax<=0:return 0.,0
    value=0.
    if umax>=1:
        for j in range(len(px)):
            value+=pw[j]*angular_value(px[j],c,s,axial,across,side,mu,tangent,delta,a2,alpha)
        return value,len(px)
    if len(px)>3:
        for j in range(len(tx)):
            theta=(tx[j]+1)*math.pi/4
            u=umax*math.sin(theta);y=math.sqrt(max(0.,(1-u)*(1+u)))
            value+=tw[j]*math.pi/4*math.cos(theta)*angular_value(y,c,s,axial,across,side,mu,tangent,delta,a2,alpha)
        return umax*value,len(tx)
    ymin=math.sqrt(max(0.,(1-umax)*(1+umax)))
    width=1-ymin
    thin=width*width<math.sqrt(np.finfo(np.float64).eps)
    if len(px)>=3 and width>=.05:
        moments=np.ones(6);moments[1]=(umax*ymin+math.asin(umax))/(2*umax)
        for n in range(2,6):moments[n]=(ymin**n+n*moments[n-2])/(n+1)
        mean=moments[1];variance=moments[2]-mean*mean
        diagonal1=(moments[3]-2*mean*moments[2]+mean**3)/variance
        S=mean+diagonal1;P=mean*diagonal1-variance
        norm=(moments[4]-2*S*moments[3]+(S*S+2*P)*moments[2]-2*S*P*mean+P*P)
        if norm>math.sqrt(np.finfo(np.float64).eps):
            diagonal2=(moments[5]-2*S*moments[4]+(S*S+2*P)*moments[3]
                       -2*S*P*moments[2]+P*P*mean)/norm
            beta2=norm/variance;centre=(mean+diagonal1+diagonal2)/3
            radius2=((mean-centre)**2+(diagonal1-centre)**2+(diagonal2-centre)**2+2*(variance+beta2))/6
            radius=math.sqrt(radius2)
            determinant=((mean-centre)*(diagonal1-centre)*(diagonal2-centre)
                         -variance*(diagonal2-centre)-beta2*(mean-centre))
            angle=math.acos(max(-1.,min(1.,determinant/(2*radius**3))))/3
            y2=centre+2*radius*math.cos(angle)
            y0=centre+2*radius*math.cos(angle+2*math.pi/3)
            y1=3*centre-y0-y2
            w0=(moments[2]-(y1+y2)*mean+y1*y2)/((y0-y1)*(y0-y2))
            w1=(moments[2]-(y0+y2)*mean+y0*y2)/((y1-y0)*(y1-y2))
            w2=1-w0-w1
            if y0>=ymin and y2<=1 and min(w0,w1,w2)>=0:
                value=(w0*angular_value(y0,c,s,axial,across,side,mu,tangent,delta,a2,alpha)
                       +w1*angular_value(y1,c,s,axial,across,side,mu,tangent,delta,a2,alpha)
                       +w2*angular_value(y2,c,s,axial,across,side,mu,tangent,delta,a2,alpha))
                return umax*value,3
        thin=True
    elif len(px)>=3:thin=True
    if not thin:
        angle=math.asin(umax)
        mean=(umax*ymin+angle)/(2*umax)
        second=1-umax*umax/3
        third=(3*angle/umax+ymin*(5-2*umax*umax))/8
        var=second-mean*mean
        if var<=0:thin=True
        else:
            diagonal=(third-2*mean*second+mean**3)/var
            discr=math.sqrt((mean-diagonal)**2+4*var)
            y0=(mean+diagonal-discr)/2;y1=(mean+diagonal+discr)/2
            w1=(mean-y0)/(y1-y0);w0=1-w1
            if y0<ymin or y1>1 or w0<0 or w1<0:thin=True
            else:
                value=(w0*angular_value(y0,c,s,axial,across,side,mu,tangent,delta,a2,alpha)
                       +w1*angular_value(y1,c,s,axial,across,side,mu,tangent,delta,a2,alpha))
                return umax*value,2
    for j in range(len(tx)):
        u=umax*(1+tx[j])/2;y=math.sqrt(max(0.,(1-u)*(1+u)))
        value+=tw[j]*angular_value(y,c,s,axial,across,side,mu,tangent,delta,a2,alpha)/2
    return umax*value,len(tx)


@njit(cache=True)
def deposit_h(h,rho,along,r,R,beta,emit,origin,left,width,n0,dn,g0,dg,e0,de,
              efficiency,c0,c2,L,alpha,axial,across,side,weight,px,pw,tx,tw,edges,bins):
    eps=R/r;ca=math.sqrt(1-(eps*h)**2)
    c=along/r*ca+rho/r*eps*h;s=rho/r*ca-along/r*eps*h
    n=1/(beta*c);q=(n-n0)/dn;lam=left+width*q
    jac=n*s/c*R/(r*ca)*width/abs(dn)
    mu=e0+de*q;G=(g0+dg*q)/.299792458
    eta=((efficiency[3]*q+efficiency[2])*q+efficiency[1])*q+efficiency[0]
    if len(efficiency)>4:eta*=math.exp(efficiency[4]+q*efficiency[5])
    spectral=(c0+c2/(n*n))/L/(lam*lam)*eta*jac
    root=math.sqrt(max(0.,(1-h)*(1+h)))
    if root==0:return 0.,0
    k0=math.sqrt(r*r-(R*h)**2);tangent=math.sqrt(r*r-R*R)
    delta=R*R*root*root/(k0+tangent)
    v=R/math.sqrt(2*(k0+tangent)*s*rho);a2=(v*root)**2
    integral,nodes=angular_integral(1.,c,s,axial,across,side,mu,tangent,delta,a2,alpha,px,pw,tx,tw)
    full=2*v*integral
    control=2*R/rho*acceptance(alpha,c*axial+s*across)*math.exp(-mu*rho/s)
    result=weight*spectral*(full-control)/(2*math.pi)
    first_path=(r*r-R*R)/(k0+R*root);exit_path=k0+R*root
    earliest=emit+G*first_path-origin;latest=emit+G*k0-origin
    first=max(0,np.searchsorted(edges,earliest,side='right')-1)
    last=min(len(bins)-1,np.searchsorted(edges,latest,side='left')-1)
    if first>last:return result,nodes
    if first==last and edges[first]<=earliest and latest<=edges[last+1]:
        bins[first]+=result;return result,nodes
    previous=0.
    for edge_index in range(first,last+2):
        edge=edges[edge_index]
        path=(edge+origin-emit)/G
        true_cdf=0.
        if path>=tangent:true_cdf=full
        elif path>first_path:
            u2=(path-first_path)*(exit_path-path)/(2*path*delta)
            value,used=angular_integral(math.sqrt(min(1.,max(0.,u2))),c,s,axial,across,side,
                                       mu,tangent,delta,a2,alpha,px,pw,tx,tw)
            true_cdf=2*v*value;nodes+=used
        chord=(emit+G*k0-origin-edge)/(G*R*root)
        disk_cdf=control*(1. if chord<=0 else (0. if chord>=1 else math.sqrt((1-chord)*(1+chord))))
        current=true_cdf-disk_cdf
        if edge_index>first:bins[edge_index-1]+=weight*spectral*(current-previous)/(2*math.pi)
        previous=current
    return result,nodes
