"""Continuous-spectrum next-event reference for a finite OM group.

This independently samples the collision expansion. Exactly two forced
scattering collisions carry albedo squared. Further collisions use roulette
with probability twice the albedo and compensating weight one half; this
reference requires albedo below one half. There is no collision-order cutoff.

Importance proposals aim the source azimuth, free flights and later
scattering directions towards the selected OM. Every proposal is divided
out of its score. Receiver-allocation probabilities likewise affect variance
only. First sphere entry stops that receiver's history. Other OMs are treated
independently, as in the transport Green function.

The azimuth of the outgoing HG cap is integrated analytically; positive
polar quadrature is refined to convergence. The optical wavelength is
sampled continuously from the exact tabulated detector response.
"""
import math
import numpy as np
from numba import njit
from scipy.special import roots_legendre

@njit(cache=True)
def rotate(v,mu,phi):
    helper=np.array([1.,0.,0.]) if abs(v[0])<.9 else np.array([0.,1.,0.])
    a=np.cross(v,helper);a/=np.linalg.norm(a);b=np.cross(v,a)
    return mu*v+math.sqrt(max(0.,(1-mu)*(1+mu)))*(math.cos(phi)*a+math.sin(phi)*b)

@njit(cache=True)
def angular(p,x):
    value=p[-1]
    for j in range(len(p)-2,-1,-1):value=value*x+p[j]
    return max(0.,value)

@njit(cache=True)
def phase(cosine,g):
    return (1-g*g)/(4*math.pi*((1-g)**2+2*g*(1-cosine))**1.5)

from lighthit._receiver_first_numba import _hg_azimuth
_N=np.array([4,8,16,32,64,128,256,512,1024,2048],np.int64)
_QX=np.zeros((len(_N),int(_N[-1])));_QW=_QX.copy()
for _i,_n in enumerate(_N):
    _x,_w=roots_legendre(int(_n));_QX[_i,:_n]=(_x+1)/2;_QW[_i,:_n]=_w/2

@njit(cache=True)
def cap_score(w,incoming,look,R,mu,g,p,qx,qw,phix,phiw):
    r=np.linalg.norm(w)
    if r<=R:return 0.
    er=w/r;cosine=max(-1.,min(1.,np.dot(er,incoming)))
    transverse=incoming-cosine*er;sine=np.linalg.norm(transverse)
    if sine>1e-12:e1=transverse/sine
    else:
        helper=np.array([1.,0.,0.]) if abs(er[0])<.9 else np.array([0.,1.,0.])
        e1=np.cross(er,helper);e1/=np.linalg.norm(e1)
    e2=np.cross(er,e1);Lr=np.dot(er,look);L1=np.dot(e1,look);L2=np.dot(e2,look)
    outside=(r-R)*(r+R);previous=0.
    for level in range(len(_N)):
        score=0.
        for j in range(_N[level]):
            t=_QX[level,j]
            root=math.sqrt(outside)
            cap_gap=R*R/(r*(r+root));cap_cosine=root/r
            # Squared polar coordinate resolves the grazing path root while
            # keeping a regular solid-angle measure even next to the glass.
            one_minus=cap_gap*(1-t*t)
            ca=1-one_minus;sa=math.sqrt(max(0.,one_minus*(2-one_minus)))
            chord=t*math.sqrt(cap_gap*(ca+cap_cosine))
            entry=outside/(r*(ca+chord))
            gap=(1-g)**2+g*((ca-cosine)**2+(sa-sine)**2)
            moments=_hg_azimuth(gap,2*g*sa*sine)
            a,b,c=ca*Lr,sa*L1,sa*L2
            c0=p[0]+p[1]*a+p[2]*(a*a+c*c)+p[3]*(a*a*a+3*a*c*c)
            c1=p[1]*b+2*p[2]*a*b+3*p[3]*b*(a*a+c*c)
            c2=(p[2]+3*p[3]*a)*(b*b-c*c)
            c3=p[3]*b*(b*b-3*c*c)
            arc=(1-g*g)/(2*math.pi)*(c0*moments[0]+c1*moments[1]+c2*moments[2]+c3*moments[3])
            score+=_QW[level,j]*2*cap_gap*t*math.exp(-mu*(entry-(r-R)))*arc
        if level>0 and abs(score-previous)<=1e-6*abs(score):return score*math.exp(-mu*(r-R))
        previous=score
    raise ValueError('HG-cap polar quadrature did not converge')

@njit(cache=True)
def collision_batch(count,seed,start,axis,length,beta,source_cdf,edges,phase_index,extinction,scattering,efficiency,eta_max,max_sin2,g,
          receivers,looks,radii,areas,apoly,qx,qw,phix,phiw,target_prob,importance=3):
    np.random.seed(seed);total=np.zeros((2,2));square=total.copy();inside=np.zeros(2,np.int64)
    norm_total=np.zeros(2);norm_square=np.zeros(2)
    target_cdf=np.cumsum(target_prob)
    target_cdf[-1]=1.
    for sample in range(count):
        target_index=np.searchsorted(target_cdf,np.random.random(),side='right')
        receiver=receivers[target_index];look=looks[target_index]
        R=radii[target_index];area=areas[target_index]
        while True:
            wavelength=1/(1/edges[0]-np.random.random()*(1/edges[0]-1/edges[-1]))
            panel=np.searchsorted(edges,wavelength,side='right')-1
            fraction=(wavelength-edges[panel])/(edges[panel+1]-edges[panel])
            e=efficiency[panel]
            eta=((e[3]*fraction+e[2])*fraction+e[1])*fraction+e[0]
            if len(e)>4:eta*=math.exp(e[4]+fraction*e[5])
            if np.random.random()*eta_max<eta:break
        i=np.searchsorted(source_cdf,np.random.random(),side='right')
        index=phase_index[panel]+fraction*(phase_index[panel+1]-phase_index[panel])
        if beta[i]*index<=1:continue
        spectral_factor=(1-1/(beta[i]*index)**2)/max_sin2[i]
        point=start[i]+np.random.random()*length[i]*axis[i]
        source_factor=1.
        if importance>=2:
            target=receiver-point;z=np.dot(target,axis[i]);transverse=target-z*axis[i];rho=np.linalg.norm(transverse)
            c=1/(beta[i]*index);ss=math.sqrt((1-c)*(1+c))
            if rho>0:e1=transverse/rho
            else:
                helper=np.array([1.,0.,0.]) if abs(axis[i,0])<.9 else np.array([0.,1.,0.])
                e1=np.cross(axis[i],helper);e1/=np.linalg.norm(e1)
            e2=np.cross(axis[i],e1)
            bmin=abs(rho*c-z*ss);dot0=z*c+rho*ss
            width=math.atan2(max(R,bmin),math.sqrt(max(0.,dot0*rho*ss)))
            parameter=math.exp(-width)
            if np.random.random()<.5:phi=2*math.pi*(np.random.random()-.5)
            else:phi=2*math.atan((1-parameter)/(1+parameter)*math.tan(math.pi*(np.random.random()-.5)))
            relative=(1-parameter*parameter)/((1-parameter)**2+4*parameter*math.sin(phi/2)**2)
            source_factor=2/(1+relative)
            d=c*axis[i]+ss*(math.cos(phi)*e1+math.sin(phi)*e2)
        else:
            d=rotate(axis[i],1/(beta[i]*index),2*math.pi*np.random.random())
        mu=extinction[panel]+fraction*(extinction[panel+1]-extinction[panel])
        mus=scattering[panel]+fraction*(scattering[panel+1]-scattering[panel])
        a=mus/mu;weight=a*source_factor*spectral_factor/target_prob[target_index];alive=True;order=0
        score=np.zeros((2,2));norm_score=np.zeros(2)
        while True:
            order+=1
            if order==2:weight*=a
            elif order>2:
                continuation=2*a if importance>=4 else a
                if np.random.random()>=continuation:break
                weight*=a/continuation
            w=receiver-point
            if importance:
                along=np.dot(w,d)
                impact=math.sqrt(max(0.,np.dot(w,w)-along*along))
                scale=max(impact,R)
                lower=math.atan2(-along,scale)
                norm=math.pi/2-lower
                if np.random.random()<.5:
                    flight=-math.log(1-np.random.random())/mu
                else:
                    flight=max(0.,along+scale*math.tan(lower+norm*np.random.random()))
                physical=mu*math.exp(-mu*flight)
                proposed=.5*physical+.5*scale/((flight-along)**2+scale*scale)/norm
                weight*=physical/proposed
            else:
                flight=-math.log(1-np.random.random())/mu
            if alive:
                dot=np.dot(w,d);outside=np.dot(w,w)-R*R;disc=dot*dot-outside
                if disc>=0 and dot>0:
                    hit=outside/(dot+math.sqrt(max(0.,disc)))
                    if 0<=hit<=flight:alive=False
            if weight==0.:break
            point=point+flight*d;w=receiver-point;r=np.linalg.norm(w);er=w/r
            column=0 if order==1 else 1
            norm_score[column]+=weight
            score[column,0]+=weight*area*phase(max(-1.,min(1.,np.dot(d,er))),g)*math.exp(-mu*r)/r**2*angular(apoly,np.dot(er,look))
            if r<R:inside[column]+=1
            if alive:
                score[column,1]+=weight*area/(math.pi*R*R)*cap_score(w,d,look,R,mu,g,apoly,qx,qw,phix,phiw)
            u=2*np.random.random()-1
            if importance>=4 or (importance>=3 and order==1):
                gp=max(g,(r-R)/(r+R))
                aimed_branch=np.random.random()>=.5
                proposal_g=gp if aimed_branch else g
                proposal_axis=er if aimed_branch else d
                scatter=(1+proposal_g*proposal_g-((1-proposal_g*proposal_g)/(1+proposal_g*u))**2)/(2*proposal_g) if proposal_g!=0 else u
                outgoing=rotate(proposal_axis,max(-1.,min(1.,scatter)),2*math.pi*np.random.random())
                physical=phase(max(-1.,min(1.,np.dot(d,outgoing))),g)
                aimed=phase(max(-1.,min(1.,np.dot(er,outgoing))),gp)
                weight*=2*physical/(physical+aimed)
                d=outgoing
            else:
                scatter=(1+g*g-((1-g*g)/(1+g*u))**2)/(2*g) if g!=0 else u
                d=rotate(d,max(-1.,min(1.,scatter)),2*math.pi*np.random.random())
        total+=score;square+=score*score
        norm_total+=norm_score;norm_square+=norm_score*norm_score
    return total,square,inside,norm_total,norm_square


def estimate(kernel, source, receivers, *, samples=1_000_000, seed=582317,
             allocation=None, batch_size=100_000):
    """Return order 1 and >=2 first-entry charge and photon-sampling error.

    ``allocation`` optionally supplies positive receiver proposal weights.
    The angular response must be a nonnegative cubic and the medium must
    have nonnegative HG g and albedo strictly below one half. Point scores
    are returned for comparison; their second moment need not be finite.
    """
    from lighthit.ballistic_interval import prepare_optics
    from lighthit._surface_ballistic_bounds import efficiency_upper
    if (isinstance(samples,bool) or not isinstance(samples,(int,np.integer))
            or isinstance(batch_size,bool) or not isinstance(batch_size,(int,np.integer))):
        raise ValueError('sample and batch counts must be integers')
    ids=np.asarray(receivers,int)
    if not len(ids) or samples<2 or batch_size<1:
        raise ValueError('nonempty receivers and at least two samples required')
    if not 0<=kernel.medium.g<1:
        raise ValueError('this reference requires nonnegative HG g')
    if not np.allclose(source.coefficient2,-source.coefficient0/source.beta**2,
                       rtol=1e-10,atol=1e-20):
        raise ValueError('reference requires physical Cherenkov C0/C2/beta')
    alpha=np.asarray(kernel.acceptance()['alpha'])
    if len(alpha)>4:
        raise ValueError('this reference requires cubic angular acceptance')
    ell=np.arange(len(alpha));poly=np.polynomial.legendre.leg2poly((2*ell+1)*alpha/(4*np.pi))
    poly=np.pad(poly,(0,4-len(poly)))
    roots=np.polynomial.polynomial.polyroots(np.polynomial.polynomial.polyder(poly))
    extrema=[-1.,1.]+[float(v.real) for v in roots if abs(v.imag)<1e-12 and -1<v.real<1]
    if np.polynomial.polynomial.polyval(extrema,poly).min() < -1e-12*max(1.,abs(poly).sum()):
        raise ValueError('HG-cap reference requires a nonnegative angular polynomial')
    t=prepare_optics(kernel,exact_response=True);edges=t['panel_edges']
    optics=kernel.medium.sample(edges)
    albedo=optics['scattering_per_m']/(optics['absorption_per_m']+optics['scattering_per_m'])
    if np.max(albedo)>=.5:
        raise ValueError('this reference requires albedo below one half')
    x,w=np.polynomial.legendre.leggauss(16)
    lam=((edges[:-1]+edges[1:])[:,None]/2+np.diff(edges)[:,None]/2*x).ravel()
    eta_integral=np.sum((np.diff(edges)[:,None]/2*w).ravel()*kernel.detector.spectral_weight(lam)/lam**2)
    max_sin2=np.maximum(0.,1-1/(source.beta*optics['phase_index'].max())**2)
    weights=source.coefficient0*max_sin2
    if weights.sum()<=0 or eta_integral<=0:
        raise ValueError('source must emit in the detected spectral band')
    cdf=np.cumsum(weights)/weights.sum();cdf[-1]=1.
    yield_scale=weights.sum()*eta_integral
    proposal=np.ones(len(ids)) if allocation is None else np.asarray(allocation,float)
    if proposal.shape!=(len(ids),) or np.any(proposal<=0) or not np.isfinite(proposal).all():
        raise ValueError('receiver proposal weights must be finite and positive')
    proposal=proposal/proposal.sum()
    d=kernel.detector
    radii=d.receiver_surface.radii(len(d))[ids]
    for position,radius in zip(d.positions_m[ids],radii):
        w=position-source.start_m
        along=np.clip(np.einsum('ij,ij->i',w,source.direction),0.,source.length_m)
        closest=w-along[:,None]*source.direction
        if np.any(np.einsum('ij,ij->i',closest,closest)<=radius*radius):
            raise ValueError('reference source intersects the receiver')
    base=(source.start_m,source.direction,source.length_m,source.beta,cdf,edges,
          t['phase_nodes'],t['extinction_nodes'],optics['scattering_per_m'],
          t['efficiency_coefficients'],float(efficiency_upper(t['efficiency_coefficients']).max()),
          max_sin2,kernel.medium.g,d.positions_m[ids],-d.orientations[ids],
          radii,d.effective_area_m2[ids],poly)
    empty=np.empty(0)
    total=np.zeros((2,2));square=total.copy();norm=np.zeros(2);norm2=norm.copy();inside=np.zeros(2,np.int64)
    count=0;batch_index=0
    while count<samples:
        n=min(int(batch_size),int(samples-count))
        q,v,within,nt,nv=collision_batch(n,int(seed+batch_index),*base,
            empty,empty,empty,empty,proposal,4)
        total+=q;square+=v;inside+=within;norm+=nt;norm2+=nv
        count+=n;batch_index+=1
    mean=yield_scale*total/count
    error=yield_scale*np.sqrt(np.maximum(0.,square-total*total/count)/(count-1)/count)
    return dict(charge_pe=mean[:,1].tolist(),mc_sigma_pe=error[:,1].tolist(),
                point_charge_pe=mean[:,0].tolist(),samples=count,seed=int(seed),
                receivers=ids.tolist(),spectral_sampling='continuous',
                no_collision_order_cutoff=True,normalization_yield=float(yield_scale))
