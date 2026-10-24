"""Direct positive ray quadrature for axial and constant-phase geometries.

This handles pairs outside the invertible wavelength/impact transform; it
uses the same first-entry boundary and time-edge cuts, without a centre-ray
approximation. Emitting segments must lie outside the glass.
"""
import numpy as np
from numba import njit
from ._ballistic_interval_numba import _support,_acceptance,_insert_cuts
from .ballistic_interval import prepare_optics

C=.299792458


@njit(cache=True)
def _quadratic_cuts(cuts,count,a,b,c,low,high):
    if a==0.:
        if b!=0.:
            root=-c/b
            if low<root<high: cuts[count]=root;count+=1
        return count
    discriminant=b*b-4*a*c
    if discriminant<0.: return count
    q=-.5*(b+np.copysign(np.sqrt(discriminant),b))
    if q==0.:
        root=-b/(2*a)
        if low<root<high:cuts[count]=root;count+=1
    else:
        for root in (q/a,c/q):
            if low<root<high:cuts[count]=root;count+=1
    return count


@njit(cache=True)
def _entry(B,dot):
    return B/(dot+np.sqrt(max(0.,dot*dot-B)))


@njit(cache=True)
def evaluate(po,pe,positions,looks,areas,radii,starts,axes,lengths,c0,c2,beta,t0,t1,
              panels,phases,groups,extinction,efficiency,alpha,gx,gw,origins,edges):
    charge=np.zeros(len(positions));bins=np.zeros((len(positions),len(edges)-1))
    phicuts=np.empty(len(edges)+2)
    scuts=np.empty(6*len(edges)+2)
    for pair in range(len(po)):
        om=po[pair];radius=radii[om]
        for i in (pe[pair],):
            w=positions[om]-starts[i];u=axes[i];z=np.dot(w,u)
            transverse=w-z*u;rho=np.linalg.norm(transverse)
            ok,nmin,nmax,cut0,cut1=_support(z,rho,lengths[i],beta[i],radius)
            if not ok or nmin>phases.max() or nmax<=phases.min():continue
            along_min=z if z<0. else (z-lengths[i] if z>lengths[i] else 0.)
            if rho*rho+along_min*along_min<=radius*radius:
                raise ValueError('emitting segment intersects the glass boundary')
            if rho>0.: e1=transverse/rho
            else:
                auxiliary=np.array([1.,0.,0.]) if abs(u[0])<.9 else np.array([0.,1.,0.])
                e1=np.cross(u,auxiliary);e1/=np.linalg.norm(e1)
            e2=np.cross(u,e1)
            axial=np.dot(u,looks[om]);across=np.dot(e1,looks[om]);side=np.dot(e2,looks[om])
            tau=(t1[i]-t0[i])/lengths[i]
            for panel in range(len(panels)-1):
                left=panels[panel];width=panels[panel+1]-left
                n0=phases[panel];dn=phases[panel+1]-n0
                a=left;b=left+width
                if dn==0.:
                    if n0<nmin or n0>=nmax:continue
                else:
                    x0=(nmin-n0)/dn;x1=(nmax-n0)/dn
                    a=left+width*max(0.,min(x0,x1));b=left+width*min(1.,max(x0,x1))
                if a>=b:continue
                cuts=np.empty(7);cuts[0]=a;cuts[1]=b
                count=_insert_cuts(cuts,2,left,width,n0,dn,cut0,cut1)
                for piece in range(count-1):
                    low=cuts[piece];high=cuts[piece+1]
                    for ilam in range(len(gx)):
                        lam=(low+high)/2+(high-low)/2*gx[ilam]
                        q=(lam-left)/width;n=n0+dn*q;bn=beta[i]*n
                        if bn<=1.:continue
                        cosine=1/bn;sine=np.sqrt(1-cosine*cosine)
                        centre=z-rho*cosine/sine;spread=radius/sine
                        slo=max(0.,centre-spread);shi=min(lengths[i],centre+spread)
                        if slo>=shi:continue
                        absorption=extinction[panel]+q*(extinction[panel+1]-extinction[panel])
                        group=groups[panel]+q*(groups[panel+1]-groups[panel])
                        velocity=C/group
                        eta=((efficiency[panel,3]*q+efficiency[panel,2])*q+efficiency[panel,1])*q+efficiency[panel,0]
                        if efficiency.shape[1]>4:eta*=np.exp(efficiency[panel,4]+q*efficiency[panel,5])
                        spectral=(c0[i]+c2[i]/n**2)/lengths[i]/lam**2*eta
                        scale=(high-low)/2*gw[ilam]*spectral*areas[om]/(np.pi*radius*radius)
                        # Split s at all possible appearances/disappearances
                        # of a time edge in the azimuth integral: phi=0, pi,
                        # and the tangent ray. Extraneous quadratic roots only
                        # add harmless integration panels.
                        scuts[0]=slo;scuts[1]=shi;ns=2
                        rmax=np.sqrt(rho*rho+max((z-slo)**2,(z-shi)**2))
                        rmin=np.sqrt(rho*rho+(0. if slo<z<shi else min((z-slo)**2,(z-shi)**2)))
                        earliest=min(t0[i]+tau*slo,t0[i]+tau*shi)+(rmin-radius)/velocity-origins[om]
                        latest=max(t0[i]+tau*slo,t0[i]+tau*shi)+rmax/velocity-origins[om]
                        first_edge=np.searchsorted(edges,earliest,side='right')
                        last_edge=np.searchsorted(edges,latest,side='left')
                        for edge in range(first_edge,last_edge):
                            A=(origins[om]+edges[edge]-t0[i])*velocity;B=tau*velocity
                            for sign in (-1.,1.):
                                D=cosine*z+sign*sine*rho
                                ns=_quadratic_cuts(scuts,ns,1+B*B-2*B*cosine,
                                    -2*z-2*A*B+2*(A*cosine+B*D),
                                    z*z+rho*rho-radius*radius+A*A-2*A*D,slo,shi)
                            ns=_quadratic_cuts(scuts,ns,B*B-1.,-2*A*B+2*z,
                                A*A-z*z-rho*rho+radius*radius,slo,shi)
                        sorted_s=np.sort(scuts[:ns])
                        for spiece in range(ns-1):
                            sa=sorted_s[spiece];sb=sorted_s[spiece+1]
                            if sa>=sb:continue
                            for j in range(len(gx)):
                                psi=np.pi/2*gx[j]
                                s=(sa+sb)/2+(sb-sa)/2*np.sin(psi)
                                ds=(sb-sa)/2*np.cos(psi)*np.pi/2*gw[j]
                                along=z-s;B=rho*rho+along*along-radius*radius
                                if rho>0.:
                                    half=np.arccos(min(1.,max(-1.,(np.sqrt(B)-cosine*along)/(sine*rho))))
                                else:half=np.pi if cosine*along>=np.sqrt(B) else 0.
                                if half==0.:continue
                                emission=t0[i]+tau*s-origins[om]
                                tmin=emission+_entry(B,cosine*along+sine*rho)/velocity
                                tmax=emission+_entry(B,cosine*along+sine*rho*np.cos(half))/velocity
                                phicuts[0]=0.;nphi=1
                                if rho>0.:
                                    first=np.searchsorted(edges,tmin,side='right')
                                    last=np.searchsorted(edges,tmax,side='left')
                                    for edge in range(first,last):
                                        distance=(edges[edge]-emission)*velocity
                                        dot=(B+distance*distance)/(2*distance)
                                        phi=np.arccos(min(1.,max(-1.,(dot-cosine*along)/(sine*rho))))
                                        phicuts[nphi]=np.arcsin(min(1.,phi/half));nphi+=1
                                phicuts[nphi]=np.pi/2;nphi+=1
                                for ppiece in range(nphi-1):
                                    pa=phicuts[ppiece];pb=phicuts[ppiece+1]
                                    if pa>=pb:continue
                                    integral=0.
                                    for jphi in range(len(gx)):
                                        eta_phi=(pa+pb)/2+(pb-pa)/2*gx[jphi]
                                        phi=half*np.sin(eta_phi)
                                        dphi=half*np.cos(eta_phi)*(pb-pa)/2*gw[jphi]
                                        cp=np.cos(phi);ss=np.sin(phi)
                                        distance=_entry(B,cosine*along+sine*rho*cp)
                                        accept=(_acceptance(alpha,cosine*axial+sine*(cp*across+ss*side))
                                                +_acceptance(alpha,cosine*axial+sine*(cp*across-ss*side)))
                                        integral+=dphi*accept*np.exp(-absorption*distance)/(2*np.pi)
                                    value=scale*ds*integral
                                    charge[om]+=value
                                    phi_mid=half*np.sin((pa+pb)/2)
                                    arrival=emission+_entry(B,cosine*along+sine*rho*np.cos(phi_mid))/velocity
                                    slot=np.searchsorted(edges,arrival,side='right')-1
                                    if 0<=slot<len(edges)-1:bins[om,slot]+=value
    return charge,bins
