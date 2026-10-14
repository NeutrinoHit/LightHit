"""Universal dimensionless HG first-scatter CDF primitives.

The stored array contains no source, geometry, energy, emission time or saved
event response. Its variables are kappa=(mu_a+mu_s)*r, theta and transformed
excess path u. Arbitrary OM polynomial acceptance is contracted at lookup.
"""
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
from numba import njit


@njit(cache=True, nogil=True)
def _coordinate(values, x):
    lo, hi = 0, values.size - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if values[mid] <= x:lo = mid
        else:hi = mid
    return lo, (x-values[lo])/(values[lo+1]-values[lo])


@njit(cache=True, nogil=True)
def _lagrange4(grid, x):
    lo, fraction = _coordinate(grid,x)
    start=max(0,min(lo-1,grid.size-4))
    weights=np.ones(4)
    for i in range(4):
        for j in range(4):
            if j!=i:weights[i]*=(x-grid[start+j])/(grid[start+i]-grid[start+j])
    return start,weights


@njit(cache=True,nogil=True)
def _prepare(displacement,direction,look,apoly,mus,mut,g,log_kappa,log_tangent,u_grid,modes,powers, interpolation,charge_only=False):
    n=displacement.shape[0];nu=1 if charge_only else u_grid.size;nm=modes.shape[-1]
    curves=np.zeros((n,nu));prefactors=np.zeros(n);sigmas=np.zeros(n);radii=np.zeros(n)
    supported=np.ones(n,np.bool_)
    for i in range(n):
        r=np.sqrt(np.sum(displacement[i]**2));radii[i]=r
        if r<=0.:supported[i]=False;continue
        rh=displacement[i]/r
        q=np.dot(rh,direction[i]);q=min(1.,max(-1.,q))
        perp=rh-q*direction[i];sine=np.sqrt(np.sum(perp**2))
        theta=np.arctan2(sine,q);kappa=mut*r
        if theta<=0. or theta>=np.pi or kappa<=0.:
            supported[i]=False;continue
        tk=np.log(kappa);tt=np.log(np.tan(theta/2.))
        if tk<log_kappa[0] or tk>log_kappa[-1] or tt<log_tangent[0] or tt>log_tangent[-1]:
            supported[i]=False;continue
        c1=np.dot(direction[i],look);c3=np.dot(perp,look)/sine
        orient=np.empty(nm)
        for m in range(nm):
            degree,j=powers[m,0],powers[m,1]
            coefficient=1.
            for a in range(j):coefficient*=float(degree-a)/float(a+1)
            orient[m]=apoly[degree]*coefficient*c1**(degree-j)*c3**j
        rho=sine+1.-q
        prefactors[i]=(mus/r)*np.exp(-kappa)/rho
        sigma0=.5*sine*(1.-abs(g))/(1.+abs(g))+.25*(1.-q)
        sigmas[i]=r*sigma0
        # The optical tail beyond kappa*excess/r=30 is already negligible
        # relative to the cache's finite-tail contract. Avoid contracting all
        # global u nodes for ordinary geometries whose useful tail is short.
        tail_index,tail_fraction=_coordinate(u_grid,min(u_grid[-1],np.arcsinh(30./(kappa*sigma0))))
        last_needed=min(nu-2,tail_index+2)
        if interpolation==1:
            ik,wk=_lagrange4(log_kappa,tk);it,wt=_lagrange4(log_tangent,tt)
            for a in range(4):
                for b in range(4):
                    w=wk[a]*wt[b]
                    for v in range(nu):
                        if v>last_needed and v!=nu-1:continue
                        table_v=u_grid.size-1 if charge_only else v
                        value=0.
                        for m in range(nm):value+=orient[m]*modes[ik+a,it+b,table_v,m]
                        curves[i,v]+=w*value
        elif interpolation==3:
            ik,ak=_coordinate(log_kappa,tk);it,wt=_lagrange4(log_tangent,tt)
            for v in range(0 if charge_only else 1,nu):
                if v>last_needed and v!=nu-1:continue
                table_v=u_grid.size-1 if charge_only else v
                interpolated=0.
                for a in range(2):
                    for b in range(4):
                        w=(ak if a else 1-ak)*wt[b]
                        value=0.
                        for m in range(nm):value+=orient[m]*modes[ik+a,it+b,table_v,m]
                        if value<=0.:supported[i]=False;continue
                        interpolated+=w*np.log(value)
                curves[i,v]=np.exp(interpolated)
        else:
            ik,ak=_coordinate(log_kappa,tk);it,at=_coordinate(log_tangent,tt)
            for v in range(0 if charge_only else 1,nu):
                if v>last_needed and v!=nu-1:continue
                table_v=u_grid.size-1 if charge_only else v
                interpolated=0.
                for a in range(2):
                    for b in range(2):
                        w=(ak if a else 1-ak)*(at if b else 1-at)
                        value=0.
                        for m in range(nm):value+=orient[m]*modes[ik+a,it+b,table_v,m]
                        if interpolation==2:
                            if value<=0.:supported[i]=False;continue
                            interpolated+=w*np.log(value)
                        else:interpolated+=w*value
                curves[i,v]=np.exp(interpolated) if interpolation==2 else interpolated
        for v in range(last_needed+1,nu-1):curves[i,v]=curves[i,-1]
    return curves,prefactors,sigmas,radii,supported


@njit(cache=True,nogil=True)
def _pchip_slopes(curves,u_grid):
    slopes=np.zeros_like(curves);bad=np.zeros(curves.shape[0],np.bool_)
    for i in range(curves.shape[0]):
        neg=0.
        for j in range(u_grid.size-1):neg+=max(curves[i,j]-curves[i,j+1],0.)
        if neg>1e-6*abs(curves[i,-1]) or curves[i,-1]<0.:
            bad[i]=True
        for j in range(1,u_grid.size-1):
            hm=u_grid[j]-u_grid[j-1];hp=u_grid[j+1]-u_grid[j]
            dm=(curves[i,j]-curves[i,j-1])/hm;dp=(curves[i,j+1]-curves[i,j])/hp
            if dm*dp>0.:
                w1=2*hp+hm;w2=hp+2*hm
                slopes[i,j]=(w1+w2)/(w1/dm+w2/dp)
        # Shape-preserving one-sided endpoint derivatives (PCHIP scheme).
        h0=u_grid[1]-u_grid[0];h1=u_grid[2]-u_grid[1]
        d0=(curves[i,1]-curves[i,0])/h0;d1=(curves[i,2]-curves[i,1])/h1
        derivative=((2*h0+h1)*d0-h0*d1)/(h0+h1)
        if derivative*d0<=0.:derivative=0.
        elif d0*d1<0. and abs(derivative)>3*abs(d0):derivative=3*d0
        slopes[i,0]=derivative
        # The integral reaches a constant attenuated tail at the final knot.
        slopes[i,-1]=0.
    return slopes,bad


@njit(cache=True,nogil=True)
def _sample_curve(excess,sigma,u_grid,curve,slopes,prefactor):
    if excess<=0.:return 0.
    u=np.arcsinh(excess/sigma)
    if u>=u_grid[-1]:return prefactor*curve[-1]
    j,f=_coordinate(u_grid,u)
    # Fritsch-Carlson/PCHIP cubic Hermite interpolation preserves increasing
    # CDF intervals; it changes derivative interpolation, never clips masses.
    h=u_grid[j+1]-u_grid[j]
    f2=f*f;f3=f2*f
    value=(2*f3-3*f2+1)*curve[j]+(f3-2*f2+f)*h*slopes[j]+(-2*f3+3*f2)*curve[j+1]+(f3-f2)*h*slopes[j+1]
    return prefactor*value


@njit(cache=True,nogil=True)
def _contract(curves,slopes,prefactors,sigmas,radii,supported,u_grid,ids,times,weights,edges,origin,speed):
    bins=np.zeros(edges.size-1);charge=0.
    for row in range(ids.size):
        geom=ids[row]
        if not supported[geom]:continue
        weight=weights[row];charge+=weight*prefactors[geom]*curves[geom,-1]
        if speed*(edges[-1]+origin-times[row])<=radii[geom]:continue
        previous=_sample_curve(speed*(edges[0]+origin-times[row])-radii[geom],sigmas[geom],
            u_grid,curves[geom],slopes[geom],prefactors[geom])
        for b in range(bins.size):
            current=_sample_curve(speed*(edges[b+1]+origin-times[row])-radii[geom],sigmas[geom],
                u_grid,curves[geom],slopes[geom],prefactors[geom])
            bins[b]+=weight*(current-previous);previous=current
    return charge,bins


@njit(cache=True,nogil=True)
def _stationary_direct(displacement,direction,look,apoly,mus,mut,g,gx,gw):
    """Positive Gauss stationary fallback, without constructing time splines."""
    result=np.zeros(displacement.shape[0])
    for row in range(len(result)):
        r=np.sqrt(np.sum(displacement[row]**2))
        if r<=0.:raise ValueError('point detector cannot coincide with source')
        rh=displacement[row]/r;q=np.dot(rh,direction[row]);q=min(1.,max(-1.,q))
        sine=np.sqrt(np.sum(np.cross(rh,direction[row])**2))
        if sine<1e-8 and q>0.:raise ValueError('forward collinear pencil requires finite angular-bin integration')
        if mus==0.:continue
        sigma=r*(.5*sine*(1.-abs(g))/(1.+abs(g))+.25*(1.-q))
        end=np.arcsinh(25./(mut*sigma));lo=0.;value=0.
        while lo<end:
            hi=min(end,lo+.5,np.arcsinh(np.sinh(lo)+4./(mut*sigma)))
            if hi<=lo:raise ValueError('stationary path cannot advance')
            half=(hi-lo)/2.;centre=(hi+lo)/2.
            for j in range(len(gx)):
                u=centre+half*gx[j];excess=sigma*np.sinh(u);path=r+excess
                denominator=path-r*q
                a=excess*(2*r+excess)/(2*denominator);b=path-a
                outgoing=(displacement[row]-a*direction[row])/b
                cosine=min(1.,max(-1.,np.dot(outgoing,direction[row])))
                x=np.dot(outgoing,look);acceptance=apoly[-1]
                for k in range(len(apoly)-2,-1,-1):acceptance=acceptance*x+apoly[k]
                hg=(1.-g*g)/(4.*np.pi*(1.+g*g-2*g*cosine)**1.5)
                value+=half*gw[j]*mus*hg*acceptance*np.exp(-mut*path)*sigma*np.cosh(u)/(b*denominator)
            lo=hi
        result[row]=value
    return result


def _acceptance_extrema(apoly):
    derivative=np.polynomial.polynomial.polyder(apoly)
    candidates=[-1.,1.]
    if len(derivative)>1:
        for root in np.polynomial.polynomial.polyroots(derivative):
            if abs(root.imag)<1e-12 and -1.<root.real<1.:candidates.append(float(root.real))
    values=np.polynomial.polynomial.polyval(candidates,apoly)
    return float(np.min(values)),float(np.max(values))


def _positive_charge_bound(displacement,direction,photon_mass,*,mus,mut,g,acceptance_max,area):
    """Analytic all-time Q1 upper bound for positive point-photon masses.

    Removes additional path attenuation and bounds HG/acceptance inside the
    exact scattering-angle integral. This is independent of atlas interpolation.
    """
    radius=np.linalg.norm(displacement,axis=1);rh=displacement/radius[:,None]
    cosine=np.clip(np.einsum('ij,ij->i',rh,direction),-1.,1.)
    sine=np.linalg.norm(np.cross(rh,direction),axis=1)
    # atan2(sin(theta),-cos(theta)) evaluates pi-theta without losing the
    # antipodal angle to subtraction; a lost angle could underestimate a bound.
    backward_span=np.arctan2(sine,-cosine)
    span_ratio=np.divide(backward_span,sine,out=np.full(len(radius),np.inf),where=sine>0.)
    span_ratio[(sine<=1e-15)&(cosine<0.)]=1.
    hg_cosine=cosine if g>=0. else -np.ones(len(radius))
    hg_max=(1-g*g)/(4*np.pi*(1+g*g-2*g*hg_cosine)**1.5)
    mass=np.asarray(photon_mass,float)
    bound=np.zeros(len(radius));active=mass>0.
    if mus>0. and area>0. and acceptance_max>0.:
        with np.errstate(divide='ignore',invalid='ignore'):
            logbound=(np.log(mus)-np.log(radius[active])-mut*radius[active]+
                np.log(acceptance_max)+np.log(hg_max[active])+np.log(span_ratio[active])+
                np.log(area)+np.log(mass[active]))
        # Keep a representable positive upper bound at double underflow rather
        # than turning a mathematically positive bound into exact zero.
        bound[active]=np.exp(np.maximum(logbound,np.log(np.nextafter(0.,1.))))
    return bound


@dataclass
class PreparedFirstCDF:
    atlas: object
    curves: np.ndarray
    slopes: np.ndarray
    prefactors: np.ndarray
    sigmas: np.ndarray
    radii: np.ndarray
    supported: np.ndarray
    fallback: dict
    speed: float

    def contract(self,geometry_id,times_ns,weights,edges_ns,*,origin_ns=0.,area_m2=1.):
        ids=np.ascontiguousarray(geometry_id,np.int64);times=np.ascontiguousarray(times_ns,float)
        weights=np.ascontiguousarray(weights,float);edges=np.ascontiguousarray(edges_ns,float)
        if (ids.ndim!=1 or times.shape!=ids.shape or weights.shape!=ids.shape or
                edges.ndim!=1 or len(edges)<2 or np.any(np.diff(edges)<=0.) or
                any(not np.all(np.isfinite(x)) for x in (times,weights,edges)) or
                (len(ids) and (ids.min()<0 or ids.max()>=len(self.supported))) or
                not np.isfinite(origin_ns) or not np.isfinite(area_m2) or area_m2<0.):
            raise ValueError('invalid temporal contraction')
        q,b=_contract(self.curves,self.slopes,self.prefactors,self.sigmas,self.radii,self.supported,
            self.atlas.u_grid,ids,times,weights,edges,float(origin_ns),self.speed)
        for geom,response in self.fallback.items():
            rows=ids==geom
            if np.any(rows):
                fq,fb=response.contract(times[rows],weights[rows],edges,time_origin_ns=origin_ns)
                q+=fq;b+=fb
        return q*area_m2,b*area_m2


@dataclass
class UniversalFirstCDFAtlas:
    g: float
    degree: int
    log_kappa: np.ndarray
    log_tangent: np.ndarray
    u_grid: np.ndarray
    powers: np.ndarray
    modes: np.ndarray

    @classmethod
    def build(cls,*,g=.9,degree=3,kappa_range=(.02,80.),theta_range=(1e-4,np.pi-1e-4),
              kappa_nodes=48,theta_nodes=128,path_nodes=512,tail_exponent=30.):
        if (abs(g)>=1 or not np.isfinite(g) or degree<0 or degree>8 or
                kappa_range[0]<=0 or kappa_range[1]<=kappa_range[0] or
                theta_range[0]<=0 or theta_range[1]>=np.pi or theta_range[1]<=theta_range[0] or
                min(kappa_nodes,theta_nodes,path_nodes)<4):raise ValueError('invalid atlas grid')
        lk=np.linspace(np.log(kappa_range[0]),np.log(kappa_range[1]),kappa_nodes)
        lt=np.linspace(np.log(np.tan(theta_range[0]/2)),np.log(np.tan(theta_range[1]/2)),theta_nodes)
        theta=2*np.arctan(np.exp(lt));sine=np.sin(theta);cosine=np.cos(theta)
        # Smooth forward/backward path scale: max(...) introduces a parameter
        # kink near the HG angular width and harms geometry interpolation.
        sigma=.5*sine*(1-abs(g))/(1+abs(g))+.25*(1-cosine)
        rho=sine+1-cosine
        u=np.linspace(0,np.arcsinh(tail_exponent/(kappa_range[0]*sigma.min())),path_nodes)
        # Positive Gauss panel quadrature creates each CDF increment before
        # summation; cumulative Simpson's odd half-panels may be negative in
        # rapidly attenuated tails, despite a positive physical integrand.
        gx,gw=np.polynomial.legendre.leggauss(3)
        uq=.5*(u[1:]+u[:-1])[:,None]+.5*np.diff(u)[:,None]*gx
        excess=sigma[:,None,None]*np.sinh(uq)[None,:,:]
        y=np.tan(theta[:,None,None]/2)+excess/sine[:,None,None]
        inv=1/(1+y*y);c=(1-y*y)*inv;s=2*y*inv
        jac=2*sigma[:,None,None]*np.cosh(uq)[None,:,:]*rho[:,None,None]*inv/(sine[:,None,None]**2)
        hg=(1-g*g)/(4*np.pi*(1+g*g-2*g*c)**1.5)
        powers=np.array([(k,j) for k in range(degree+1) for j in range(k+1)],np.int64)
        angular=np.stack([c**(k-j)*s**j for k,j in powers],axis=-1)
        modes=np.empty((kappa_nodes,theta_nodes,path_nodes,len(powers)),np.float32)
        for i,kappa in enumerate(np.exp(lk)):
            density=(hg*jac*np.exp(-kappa*excess))[:,:,:,None]*angular
            increments=(density*gw[None,None,:,None]).sum(axis=2)*(.5*np.diff(u)[None,:,None])
            modes[i,:,0,:]=0.
            modes[i,:,1:,:]=np.cumsum(increments,axis=1)
        return cls(float(g),int(degree),lk,lt,u,powers,modes)

    def save(self,path):
        path=Path(path)
        if path.exists():raise ValueError('refuse to overwrite atlas')
        path.parent.mkdir(parents=True,exist_ok=True)
        meta=dict(kind='universal-first-dimensionless-cdf-v2',g=self.g,degree=self.degree,
            dimensions=['log((mu_a+mu_s)*r)','log(tan(theta/2))','u=asinh((S-r)/(r*sigma0))','acceptance_monomial'],
            source_or_geometry_stored=False,path_scale='smooth-sum-v1',integration='positive-Gauss3-panels')
        np.savez_compressed(path,metadata=json.dumps(meta),log_kappa=self.log_kappa,
            log_tangent=self.log_tangent,u_grid=self.u_grid,powers=self.powers,modes=self.modes)

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as z:
            meta=json.loads(str(z['metadata']))
            if meta['kind']!='universal-first-dimensionless-cdf-v2':raise ValueError('invalid atlas kind')
            return cls(meta['g'],meta['degree'],z['log_kappa'],z['log_tangent'],z['u_grid'],z['powers'],z['modes'])

    def prepare(self,displacement_m,direction,look,acceptance_polynomial,*,scattering_per_m,
                absorption_per_m,g,speed_m_per_ns,cubic=True,direct_fallback=True,interpolation=None):
        from .joint_first import PencilFirstCDF
        delta=np.ascontiguousarray(displacement_m,float);direction=np.ascontiguousarray(direction,float)
        look=np.ascontiguousarray(look,float);apoly=np.ascontiguousarray(acceptance_polynomial,float)
        mus=float(scattering_per_m);mua=float(absorption_per_m);mut=mus+mua
        if (delta.ndim!=2 or delta.shape[1]!=3 or direction.shape!=delta.shape or look.shape!=(3,) or
                len(apoly)>self.degree+1 or apoly.ndim!=1 or
                any(not np.all(np.isfinite(x)) for x in (delta,direction,look,apoly)) or
                not np.allclose(np.linalg.norm(direction,axis=1),1.) or not np.isclose(np.linalg.norm(look),1.) or
                not np.all(np.isfinite([mus,mua,g,speed_m_per_ns])) or
                mus<0. or mua<0. or mut<=0. or speed_m_per_ns<=0. or not np.isclose(g,self.g,rtol=0,atol=1e-14)):
            raise ValueError('atlas optical/geometry contract mismatch')
        padded=np.pad(apoly,(0,self.degree+1-len(apoly)))
        method=('logcubic' if cubic else 'linear') if interpolation is None else interpolation
        if method not in ('cubic','linear','loglinear','logcubic'):raise ValueError('invalid interpolation method')
        mode={'linear':0,'cubic':1,'loglinear':2,'logcubic':3}[method]
        curves,prefactors,sigmas,radii,supported=_prepare(delta,direction,look,padded,mus,mut,g,
            self.log_kappa,self.log_tangent,self.u_grid,self.modes,self.powers,mode)
        slopes,nonmonotone=_pchip_slopes(curves,self.u_grid)
        supported &= ~nonmonotone
        fallback={}
        if not np.all(supported):
            if not direct_fallback:raise ValueError('geometry outside atlas support')
            for i in np.flatnonzero(~supported):
                fallback[int(i)]=PencilFirstCDF.build(delta[i],direction[i],look,apoly,
                    scattering_per_m=mus,absorption_per_m=mut-mus,g=g,speed_m_per_ns=speed_m_per_ns)
        return PreparedFirstCDF(self,curves,slopes,prefactors,sigmas,radii,supported,fallback,float(speed_m_per_ns))

    def charge_coefficients(self,displacement_m,direction,look,acceptance_polynomial,*,
                            scattering_per_m,absorption_per_m,g,direct_fallback=True):
        """All-time accepted Q1/m²/photon, with no clock/time-CDF allocation.

        Optical and geometric checks match prepare. Unsupported geometries
        use positive Gauss integration of the physical HG path, with no spline
        construction. Waveform guards are unnecessary because no interior
        atlas u is used for stationary charge.
        """
        delta=np.ascontiguousarray(displacement_m,float);direction=np.ascontiguousarray(direction,float)
        look=np.ascontiguousarray(look,float);apoly=np.ascontiguousarray(acceptance_polynomial,float)
        mus=float(scattering_per_m);mua=float(absorption_per_m);mut=mus+mua
        if (delta.ndim!=2 or delta.shape[1]!=3 or direction.shape!=delta.shape or look.shape!=(3,) or
                apoly.ndim!=1 or len(apoly)>self.degree+1 or
                any(not np.all(np.isfinite(x)) for x in (delta,direction,look,apoly)) or
                not np.allclose(np.linalg.norm(direction,axis=1),1.) or not np.isclose(np.linalg.norm(look),1.) or
                not np.all(np.isfinite([mus,mua,g])) or mus<0. or mua<0. or mut<=0. or
                not np.isclose(g,self.g,rtol=0,atol=1e-14)):
            raise ValueError('atlas optical/geometry contract mismatch')
        padded=np.pad(apoly,(0,self.degree+1-len(apoly)))
        curves,prefactors,sigmas,radii,supported=_prepare(delta,direction,look,padded,mus,mut,g,
            self.log_kappa,self.log_tangent,self.u_grid,self.modes,self.powers,3,True)
        charge=curves[:,0]*prefactors
        if not np.all(supported):
            if not direct_fallback:raise ValueError('geometry outside atlas support')
            indices=np.flatnonzero(~supported)
            gx,gw=np.polynomial.legendre.leggauss(8)
            charge[indices]=_stationary_direct(delta[indices],direction[indices],look,apoly,mus,mut,g,gx,gw)
        return charge,supported


def photon_source_guard(histogram,receiver_position_m,*,factor=8.):
    """Geometry bins that need original photon members near a beam root.

    The weighted covariance rule is supplemented by retained sample support,
    so an arbitrarily light outlier crossing the receiver ray cannot escape
    merely because its RMS contribution is small. This is a source-bin guard,
    distinct from the cache's interpolation/domain fallback.
    """
    if not np.isfinite(factor) or factor<0.:raise ValueError('invalid source guard factor')
    if factor==0.:return np.zeros(len(histogram.position),bool)
    pcov=getattr(histogram,'geometry_position_covariance_m2',None)
    dcov=getattr(histogram,'geometry_direction_covariance',None)
    if pcov is None or dcov is None:raise ValueError('source guard requires source covariance')
    delta=np.asarray(receiver_position_m,float)-histogram.position;radius=np.linalg.norm(delta,axis=1)
    if not np.all(np.isfinite(pcov)) or not np.all(np.isfinite(dcov)):
        raise ValueError('source covariance must be finite')
    spatial=np.sqrt(np.maximum(np.trace(pcov,axis1=1,axis2=2),0.))
    angular=np.sqrt(np.maximum(np.trace(dcov,axis1=1,axis2=2),0.))
    with np.errstate(divide='ignore',invalid='ignore'):
        rhat=delta/radius[:,None]
        cosine=np.einsum('ij,ij->i',rhat,histogram.direction)
        sine=np.linalg.norm(np.cross(rhat,histogram.direction),axis=1)
        theta=np.arctan2(sine,cosine)
        unsafe=(theta<=factor*(angular+spatial/radius))|(radius<=factor*spatial)
        angular_max=getattr(histogram,'geometry_sample_angular_max_rad',None)
        spatial_max=getattr(histogram,'geometry_sample_position_max_m',None)
        if angular_max is not None and spatial_max is not None:
            unsafe |= theta<=angular_max+np.arcsin(np.minimum(spatial_max/radius,1.))
    return unsafe


def evaluate_photon_histogram_atlas(atlas,kernel,histogram,receiver_indices,time_edges_ns,
                                  time_origin_ns,*,wavelength_weight_nm=1.,compute_bins=True,
                                  source_guard_factor=8.,raw_chunk=2048,bins_tolerance_pe=0.):
    """Q1 charge and optionally pruned physical-time profiles of a positive band.

    Every stationary contribution is retained, including original members of
    source-guarded bins. CDFs are built only for in-window contributors whose
    omission would exceed the per-receiver ``bins_tolerance_pe`` budget. An
    independent positive HG/acceptance bound, not approximate atlas charge,
    limits omitted bin L1. This bounds the finite photon model's omission;
    source discretization and numerical interpolation errors remain separate.
    """
    from time import process_time
    from ..prompt import acceptance_polynomial,acceptance_monomials,C_M_PER_NS
    if not np.isfinite(source_guard_factor) or source_guard_factor<0. or raw_chunk<1:
        raise ValueError('invalid source guard/chunk controls')
    ids=np.asarray(receiver_indices)
    edges=np.asarray(time_edges_ns,float);origins=np.asarray(time_origin_ns,float)
    wave=np.asarray(histogram.wavelength_nm,float)
    if (ids.ndim!=1 or ids.dtype.kind not in 'iu' or origins.shape!=ids.shape or
            wave.shape!=(1,) or edges.ndim!=1 or len(edges)<2 or np.any(np.diff(edges)<=0.) or
            np.any(ids<0) or np.any(ids>=len(kernel.detector.positions_m)) or
            not np.isfinite(wavelength_weight_nm) or wavelength_weight_nm<0. or
            not np.all(np.isfinite(edges)) or not np.all(np.isfinite(origins))):
        raise ValueError('invalid receiver/time/wavelength arguments')
    tolerance=np.asarray(bins_tolerance_pe,float)
    if tolerance.ndim==0:tolerance=np.full(len(ids),float(tolerance))
    if tolerance.shape!=ids.shape or not np.all(np.isfinite(tolerance)) or np.any(tolerance<0.):
        raise ValueError('bins_tolerance_pe must be a nonnegative scalar or one value per receiver')
    sample=kernel.medium.sample(wave)
    mus=float(sample['scattering_per_m'][0]);mua=float(sample['absorption_per_m'][0])
    speed=float(C_M_PER_NS/sample['group_index'][0]);g=float(kernel.medium.g);mut=mus+mua
    apoly=acceptance_polynomial(*acceptance_monomials(kernel.acceptance()['alpha']))
    acceptance_min,acceptance_max=_acceptance_extrema(apoly)
    spectral=wavelength_weight_nm*float(kernel.detector.spectral_weight(float(wave[0])))
    photon=np.asarray(histogram.photon_weights[0],float)*spectral
    geometry=np.asarray(histogram.geometry_id,np.int64)
    if photon.shape!=geometry.shape or not np.all(np.isfinite(photon)):
        raise ValueError('finite photon mass per geometry/clock row required')
    geometry_mass=np.bincount(geometry,weights=photon,minlength=len(histogram.position))
    geometry_abs_mass=np.bincount(geometry,weights=abs(photon),minlength=len(histogram.position))
    raw_geometry=getattr(histogram,'original_geometry_id',None)
    angular_max=getattr(histogram,'geometry_sample_angular_max_rad',None)
    spatial_max=getattr(histogram,'geometry_sample_position_max_m',None)
    if source_guard_factor>0.:
        required=('geometry_position_covariance_m2','geometry_direction_covariance','original_geometry_id',
                  'original_position','original_direction','original_times_ns','original_photons_per_nm')
        if any(getattr(histogram,name,None) is None for name in required):
            raise ValueError('source guard requires covariance and retained original photon members')
        raw_geometry=np.asarray(raw_geometry,np.int64)
        raw_photon_all=np.asarray(histogram.original_photons_per_nm,float)*spectral
        if not np.all(np.isfinite(raw_photon_all)):raise ValueError('finite original photon masses required')
    else:raw_photon_all=np.empty(0)
    if np.any(tolerance>0.) and (acceptance_min<0. or np.any(photon<0.) or np.any(raw_photon_all<0.)):
        raise ValueError('positive photon weights and nonnegative acceptance required for a bin omission bound')
    charge=np.zeros(len(ids));bins=np.zeros((len(ids),len(edges)-1 if compute_bins else 0))
    omitted_bounds=np.zeros(len(ids));clock_geometries=clock_members=budget_geometries=budget_members=0
    time_geometries=time_members=0;skipped_profiles=[]
    started=process_time();fallback_count=0;preparation_cpu=application_cpu=charge_cpu=0.
    source_guard_geometries=source_guard_members=0;guard_charge=0.;guard_cpu=0.
    earliest=np.full(len(histogram.position),np.inf)
    np.minimum.at(earliest,geometry[photon!=0.],histogram.emission_times_ns[photon!=0.])
    if source_guard_factor>0.:
        # Retained sample minima/support make the geometry-level clock gate
        # conservative for original source members as well as its centroid.
        np.minimum.at(earliest,raw_geometry[raw_photon_all!=0.],histogram.original_times_ns[raw_photon_all!=0.])
    for row,om in enumerate(ids):
        delta=kernel.detector.positions_m[om]-histogram.position
        look=-kernel.detector.orientations[om];area=float(kernel.detector.effective_area_m2[om])
        if area==0. or mus==0. or not np.any(geometry_abs_mass):
            if compute_bins:skipped_profiles.append(int(om))
            continue
        unsafe=photon_source_guard(histogram,kernel.detector.positions_m[om],factor=source_guard_factor)
        safe=np.flatnonzero((~unsafe)&(geometry_abs_mass>0.))
        members=(np.flatnonzero(unsafe[raw_geometry]&(raw_photon_all!=0.))
                 if source_guard_factor>0. and np.any(unsafe) else np.empty(0,np.int64))
        source_guard_geometries+=int(unsafe.sum());source_guard_members+=len(members)
        cpu=process_time()
        values,supported=atlas.charge_coefficients(delta[safe],histogram.direction[safe],look,apoly,
            scattering_per_m=mus,absorption_per_m=mua,g=g)
        charge[row]=float(values@geometry_mass[safe])*area
        fallback_count+=int((~supported).sum())
        charge_cpu+=process_time()-cpu;application_cpu+=process_time()-cpu
        raw_charge=np.empty(len(members))
        guard_started=process_time()
        for start in range(0,len(members),int(raw_chunk)):
            selected=members[start:start+int(raw_chunk)]
            raw_delta=kernel.detector.positions_m[om]-histogram.original_position[selected]
            cpu=process_time()
            rv,rs=atlas.charge_coefficients(raw_delta,histogram.original_direction[selected],look,apoly,
                scattering_per_m=mus,absorption_per_m=mua,g=g)
            raw_charge[start:start+len(selected)]=rv*raw_photon_all[selected]*area
            fallback_count+=int((~rs).sum());charge_cpu+=process_time()-cpu;application_cpu+=process_time()-cpu
        rq=float(raw_charge.sum());charge[row]+=rq;guard_charge+=rq
        guard_cpu+=process_time()-guard_started
        if not compute_bins:continue
        radius=np.linalg.norm(delta[safe],axis=1)
        lower_radius=np.maximum(radius-(spatial_max[safe] if spatial_max is not None else 0.),0.)
        safe_live=earliest[safe]+lower_radius/speed<edges[-1]+origins[row]
        if len(members):
            raw_delta=kernel.detector.positions_m[om]-histogram.original_position[members]
            raw_radius=np.linalg.norm(raw_delta,axis=1)
            raw_live=histogram.original_times_ns[members]+raw_radius/speed<edges[-1]+origins[row]
        else:raw_delta=np.empty((0,3));raw_live=np.empty(0,bool)
        clock_geometries+=int((~safe_live).sum());clock_members+=int((~raw_live).sum())
        if tolerance[row]>0.:
            safe_bounds=_positive_charge_bound(delta[safe],histogram.direction[safe],geometry_mass[safe],
                mus=mus,mut=mut,g=g,acceptance_max=acceptance_max,area=area)
            raw_bounds=(_positive_charge_bound(raw_delta,histogram.original_direction[members],raw_photon_all[members],
                mus=mus,mut=mut,g=g,acceptance_max=acceptance_max,area=area) if len(members) else np.empty(0))
            live=np.r_[safe_live,raw_live];bounds=np.r_[safe_bounds,raw_bounds]
            candidates=np.flatnonzero(live)
            order=candidates[np.argsort(bounds[candidates],kind='stable')]
            cumulative=np.cumsum(bounds[order]);omitted=order[cumulative<=tolerance[row]]
            omitted_bounds[row]=float(bounds[omitted].sum())
            safe_omitted=omitted[omitted<len(safe)];raw_omitted=omitted[omitted>=len(safe)]-len(safe)
            safe_live[safe_omitted]=False;raw_live[raw_omitted]=False
            budget_geometries+=len(safe_omitted);budget_members+=len(raw_omitted)
        selected_safe=safe[safe_live]
        if len(selected_safe):
            mapper=np.full(len(unsafe),-1,np.int64);mapper[selected_safe]=np.arange(len(selected_safe))
            source_rows=(mapper[geometry]>=0)&(photon!=0.)
            cpu=process_time();prepared=atlas.prepare(delta[selected_safe],histogram.direction[selected_safe],look,apoly,
                scattering_per_m=mus,absorption_per_m=mua,g=g,speed_m_per_ns=speed)
            preparation_cpu+=process_time()-cpu;fallback_count+=len(prepared.fallback)
            cpu=process_time();_,sb=prepared.contract(mapper[geometry[source_rows]],histogram.emission_times_ns[source_rows],
                photon[source_rows],edges,origin_ns=float(origins[row]),area_m2=area)
            bins[row]+=sb;application_cpu+=process_time()-cpu;time_geometries+=len(selected_safe)
        selected_members=members[raw_live]
        guard_started=process_time()
        for start in range(0,len(selected_members),int(raw_chunk)):
            selected=selected_members[start:start+int(raw_chunk)]
            raw_delta=kernel.detector.positions_m[om]-histogram.original_position[selected]
            cpu=process_time();prepared=atlas.prepare(raw_delta,histogram.original_direction[selected],look,apoly,
                scattering_per_m=mus,absorption_per_m=mua,g=g,speed_m_per_ns=speed)
            preparation_cpu+=process_time()-cpu;fallback_count+=len(prepared.fallback)
            cpu=process_time();_,rb=prepared.contract(np.arange(len(selected),dtype=np.int64),histogram.original_times_ns[selected],
                raw_photon_all[selected],edges,origin_ns=float(origins[row]),area_m2=area)
            bins[row]+=rb;application_cpu+=process_time()-cpu;time_members+=len(selected)
        guard_cpu+=process_time()-guard_started
        if not len(selected_safe) and not len(selected_members):skipped_profiles.append(int(om))
    return charge,bins,dict(cpu_s=process_time()-started,prepare_cpu_s=preparation_cpu,
        apply_cpu_s=application_cpu,charge_cpu_s=charge_cpu,fallback_geometries=fallback_count,wavelength_nm=float(wave[0]),
        geometry_cells=len(histogram.position),clock_cells=len(geometry),computed_order=1,
        bins_computed=bool(compute_bins),photon_wavelength_yield_applied_twice=False,
        event_specific_response_saved=False,source_guard_factor=float(source_guard_factor),
        source_guard_fallback_geometries=source_guard_geometries,source_guard_fallback_members=source_guard_members,
        source_guard_cpu_s=guard_cpu,source_guard_charge_pe=guard_charge,
        source_guard_charge_fraction=float(guard_charge/max(float(charge.sum()),1e-300)),
        source_guard_sample_support_used=bool(angular_max is not None and spatial_max is not None),
        bins_tolerance_per_om_pe=tolerance.tolist(),omitted_bins_l1_bound_pe=float(omitted_bounds.sum()),
        omitted_bins_l1_bound_per_om_pe=omitted_bounds.tolist(),
        skipped_clock_geometries=clock_geometries,skipped_clock_members=clock_members,
        skipped_budget_geometries=budget_geometries,skipped_budget_members=budget_members,
        time_cdf_geometries=time_geometries,time_cdf_raw_members=time_members,profile_skipped_oms=skipped_profiles,
        omission_bound_scope='positive finite photon source; excludes source discretization and interpolation error')
