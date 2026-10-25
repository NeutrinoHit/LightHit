"""Exterior-sphere adjoint RTE for reusable first-entry response tables.

The photon stops at the first glass intersection. No water collision is
propagated inside the sphere and no returning photon is counted twice.
All lengths use the supplied radius; area and spectral efficiency are folded
by the consumer. Output multipoles have the ordinary DirectionalCache shape.

This table builder has its own angular and radial discretisation. Those
resolutions require convergence checks, independently of event-source and
spatial Fourier truncation in the point-receiver caches.
"""
from dataclasses import dataclass, asdict
from functools import lru_cache
from time import perf_counter
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu
from scipy.linalg import eigh
from numpy.polynomial.legendre import leggauss


def _basis(mu, m, degrees):
    """Normalized positive-diagonal associated Legendre recurrence.

    Every requested degree shares its preceding values. Evaluating lpmv
    independently at each degree would repeat the recurrence quadratically
    in the large angular grids used for the finite glass boundary.
    """
    mu=np.asarray(mu);degrees=np.asarray(degrees,int);top=int(degrees.max())
    values=np.empty((len(mu),top-m+1))
    diagonal=np.full(len(mu),1/np.sqrt(2.))
    sine=np.sqrt(np.maximum(0.,(1-mu)*(1+mu)))
    for j in range(1,m+1):diagonal*=np.sqrt((2*j+1)/(2*j))*sine
    values[:,0]=diagonal
    if top>m:values[:,1]=np.sqrt(2*m+3)*mu*diagonal
    for j in range(m+2,top+1):
        a=np.sqrt((4*j*j-1)/(j*j-m*m))
        b=np.sqrt((2*j+1)*((j-1)**2-m*m)/((2*j-3)*(j*j-m*m)))
        values[:,j-m]=a*mu*values[:,j-m-1]-b*values[:,j-m-2]
    return values[:,degrees-m]


@lru_cache(maxsize=16)
def _operators(lam, pairs):
    # Even-sized angular blocks have no zero streaming characteristic.
    modes=[(m,j) for m in range(lam+1) for j in range(m,m+2*pairs)]
    n=len(modes);mu,w=leggauss(4*pairs+2*lam+4)
    q={m:_basis(mu,m,np.arange(m,m+2*pairs)) for m in range(lam+1)}
    dq={}
    for m in range(lam+1):
        # d/dmu P_j^m = ((j+m)P_(j-1)^m-j*mu*P_j^m)/(1-mu^2).
        j=np.arange(m,m+2*pairs)
        prev=np.zeros_like(q[m]);valid=j>m
        used=j[valid]
        ratio=np.sqrt((2*used+1)/(2*used-1)*(used-m)/(used+m))
        prev[:,valid]=q[m][:,:-1]*ratio
        dq[m]=((j+m)*prev-j*mu[:,None]*q[m])/(1-mu[:,None]**2)
    rotation=np.zeros((lam+1,lam+1))
    for m in range(lam):
        a=.5*np.sqrt((lam-m)*(lam+m+1))*(np.sqrt(2) if m==0 else 1)
        rotation[m,m+1]=a;rotation[m+1,m]=-a
    M=np.zeros((n,n));B=np.zeros_like(M)
    for m in range(lam+1):
        sl=slice(m*2*pairs,(m+1)*2*pairs)
        M[sl,sl]=q[m].T@((w*mu)[:,None]*q[m])
        B[sl,sl]=q[m].T@((w*(1-mu**2))[:,None]*dq[m])
        for t in range(lam+1):
            if rotation[t,m]:
                sr=slice(t*2*pairs,(t+1)*2*pairs)
                B[sl,sr]+=rotation[t,m]*(q[m].T@((w*np.sqrt(1-mu**2))[:,None]*q[t]))
    mode=np.asarray(modes)
    adjacent=abs(mode[:,1,None]-mode[None,:,1])==1
    M[~(adjacent & (mode[:,0,None]==mode[None,:,0]))]=0
    B[~(adjacent & (abs(mode[:,0,None]-mode[None,:,0])<=1))]=0
    M[abs(M)<1e-12]=0;B[abs(B)<1e-12]=0
    return np.array(modes),M,B


def _ballistic_moments(r,R,d,lam,modes,order=64,carrier=None):
    x,w=leggauss(order);v=(x+1)/2;w=w/2
    root=np.sqrt(1-(R/r[:,None])**2*(1-v*v))
    mu=-root
    entry=(r[:,None]**2-R**2)/(r[:,None]*root+R*v)
    carrier=d if carrier is None else carrier
    measure=(w*v)[None,:]/root/(np.pi*r[:,None]**2)*np.exp(-d*entry+carrier*(r-R)[:,None])
    result=np.zeros((len(r),len(modes)),complex)
    alpha=4*np.pi/(2*lam+1)
    for m in range(lam+1):
        pick=np.flatnonzero(modes[:,0]==m);js=modes[pick,1]
        shape=_basis(mu.ravel(),m,js).reshape(len(r),len(v),-1)
        boundary=_basis(mu.ravel(),m,np.array([lam])).reshape(len(r),len(v))
        factor=alpha/(np.sqrt(2*np.pi) if m==0 else np.sqrt(np.pi))
        result[:,pick]=factor*np.einsum('rt,rt,rtn->rn',measure,boundary,shape)
    return result


def _solve(medium,R,lam,*,pairs,points,high,omega,tail,ballistic_order=64,factored=False,
           collision_relative_bound=1e-10,collision_solver='auto',diagnostics=None,
           linear_ordering='COLAMD',scattering_degree=None,radial_grading=1.):
    modes,M,B=_operators(lam,pairs);n=len(modes)
    d=medium.extinction_per_m-1j*omega/medium.speed_m_per_ns
    carrier=medium.absorption_per_m-1j*omega/medium.speed_m_per_ns
    # A path from r<=high that reaches outer before reaching the receiver
    # has length at least 2*outer-high-R. Absorption alone bounds its
    # contribution, independently of HG scattering and the collision count.
    # This extent bounds the error relative to exp(-mu_a*high)/(4*pi*high^2).
    outer=high+R/2+np.log(4*high*high/(R*R*tail))/(2*medium.absorption_per_m)
    r=R*np.exp(np.log(outer/R)*np.linspace(0.,1.,points)**radial_grading)
    gamma=medium.scattering_per_m*medium.g**modes[:,1]
    if scattering_degree is not None:
        gamma=np.where(modes[:,1]<=scattering_degree,gamma,0.)
    E,U=eigh(M)
    # Incoming characteristics have prescribed first-entry data at R;
    # outgoing characteristics vanish at the exterior boundary.
    incoming=U[:,E<0].T;outgoing=U[:,E>0].T
    factor_reports=[]
    def matrix(scatter):
        started=perf_counter()
        # Absorption, rather than total extinction, factors out the slowest
        # possible decay. This avoids exponentially large multiple-scattering
        # envelopes in high-albedo media.
        A= -carrier*M-d*np.eye(n)+np.diag(gamma if scatter else np.zeros(n))
        blocks=[]
        for i in range(points-1):
            mid=(r[i]+r[i+1])/2;step=r[i+1]-r[i]
            lo=-M/step+(B/mid+A)/2
            hi=M/step+(B/mid+A)/2
            blocks.append((csc_matrix(lo),csc_matrix(hi)))
        # COO assembly keeps only the small moment stencil.
        from scipy.sparse import coo_matrix
        rows=[];cols=[];values=[]
        for i,(lo,hi) in enumerate(blocks):
            for offset,a in [(i,lo),(i+1,hi)]:
                a=a.tocoo();rows.extend(i*n+a.row);cols.extend(offset*n+a.col);values.extend(a.data)
        for offset,a in [(0,incoming),((points-1)*n,outgoing)]:
            rr,cc=np.nonzero(abs(a)>1e-13)
            start=(points-1)*n+(0 if offset==0 else len(incoming))
            rows.extend(start+rr);cols.extend(offset+cc);values.extend(a[rr,cc])
        sparse=coo_matrix((values,(rows,cols)),shape=(points*n,points*n)).tocsc()
        assembled=perf_counter()
        factor=splu(sparse,permc_spec=linear_ordering)
        factor_reports.append(dict(assembly_seconds=assembled-started,
            factor_seconds=perf_counter()-assembled,input_nonzeros=sparse.nnz,
            factor_nonzeros=factor.L.nnz+factor.U.nnz,ordering=linear_ordering))
        return factor
    h0=_ballistic_moments(r,R,d,lam,modes,order=ballistic_order,carrier=carrier)
    free=matrix(False)
    def rhs(h):
        a=np.zeros((points,n),complex)
        a[:-1]=-(h[:-1]+h[1:])/2*gamma
        return a.ravel()
    h1=free.solve(rhs(h0)).reshape(points,n)
    # Split absorption into two independent halves. One half bounds the
    # minimum path r-R; the other bounds the probability of >=K collisions
    # by p^K, p=mu_s/(mu_s+mu_a/2). Projecting a bounded angular response
    # contributes at most sqrt(8*pi) to each normalized moment. This gives
    # a collision-tail budget independent of the angular/radial error.
    p=medium.scattering_per_m/(medium.scattering_per_m+medium.absorption_per_m/2)
    if 0<p<1:
        log_bound=(medium.absorption_per_m*(high-R)/2
            +np.log(4*np.sqrt(8*np.pi)*high*high/(R*R*collision_relative_bound)))
        terms=max(2,int(np.ceil(log_bound/(-np.log(p)))))
    else:
        terms=None
    method=collision_solver
    if method=='auto':
        # Operation-count estimate: one extra factorization versus repeated
        # triangular solves. This is a numerical solver choice, not a
        # physical collision-order cutoff.
        method='series' if terms is not None and terms<n else 'direct'
    if method=='series':
        if terms is None:raise FloatingPointError('collision-tail bound is not representable; use direct solver')
        h2=np.zeros_like(h1);previous=h1
        for _ in range(2,terms):
            previous=free.solve(rhs(previous)).reshape(points,n)
            h2+=previous
    elif method=='direct':
        h2=matrix(True).solve(rhs(h1)).reshape(points,n)
    else:
        raise ValueError('collision_solver must be auto, series or direct')
    if diagnostics is not None:
        diagnostics.update(collision_solver=method,first_omitted_collision=terms if method=='series' else None,
                           collision_relative_bound=collision_relative_bound if method=='series' else 0.,
                           factorizations=factor_reports)
    result=np.stack([h0,h1,h2],axis=-1)
    if not factored:
        result*=np.exp(-carrier*(r-R))[:,None,None]
    return r,modes,result



@dataclass(frozen=True)
class SphereTransportConfig:
    """Numerical resolutions, not physical distance or collision cutoffs.

    ``angular_pairs`` gives an even block of 2*p moments for each |m|.
    The highest degree is m+2*p-1. All retained HG eigenvalues are used.
    ``exterior_relative_bound`` determines the exterior boundary from an
    absorption/path-length bound; it does not truncate collision order.
    """
    angular_pairs: int = 96
    radial_nodes: int = 600
    ballistic_quadrature: int = 64
    exterior_relative_bound: float = 1e-10
    collision_relative_bound: float = 1e-10
    collision_solver: str = 'auto'

    def __post_init__(self):
        for name, minimum in [('angular_pairs',2),('radial_nodes',16),('ballistic_quadrature',4)]:
            value=getattr(self,name)
            if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)) or value<minimum:
                raise ValueError(f'{name} must be an integer >= {minimum}')
        if not np.isfinite(self.exterior_relative_bound) or not 0<self.exterior_relative_bound<1:
            raise ValueError('exterior_relative_bound must be in (0,1)')
        if not np.isfinite(self.collision_relative_bound) or not 0<self.collision_relative_bound<1:
            raise ValueError('collision_relative_bound must be in (0,1)')
        if self.collision_solver not in ('auto','series','direct'):
            raise ValueError('collision_solver must be auto, series or direct')


def first_entry_cache(medium, settings, grid, surface, acceptance_degree, *,
                      config=None, receiver=0, radial_phase='flight', progress=None):
    """Build Q1 and >=2 in the same multipole layout as the fast point apply.

    The calculation uses the physical glass radius independently of area.
    Source/receiver harmonics and spectral folding can consume the resulting
    coefficients unchanged. Q0 is used only as the exact forcing of Q1; the
    event's direct evaluator remains separate.

    This constructor does not select or replace any event tables. Its finite
    angular/radial resolutions and convergence need to be recorded with the
    table before enabling it for event transport.

    ``settings.spatial_degree`` selects the stored source bandwidth. The
    Fourier k options are unused; angular streaming and HG scattering use
    ``config`` instead of a spatial Fourier inversion.
    """
    from .receiver import SphereSurface
    from .directional import DirectionalCache, CouplingTable
    from scipy.interpolate import CubicSpline
    if not isinstance(surface,SphereSurface):
        raise ValueError('first-entry tables require SphereSurface geometry')
    config=config or SphereTransportConfig()
    if not isinstance(config,SphereTransportConfig):
        raise ValueError('config must be SphereTransportConfig')
    radius=float(surface.radius_m if surface.radius_m.ndim==0 else surface.radius_m[receiver])
    if grid.radii_m[0]<=radius:
        raise ValueError('table sources must lie outside the glass')
    if isinstance(acceptance_degree,(bool,np.bool_)) or not isinstance(acceptance_degree,(int,np.integer)) or acceptance_degree<0:
        raise ValueError('acceptance_degree must be a nonnegative integer')
    if settings.spatial_degree>=2*config.angular_pairs:
        raise ValueError('angular_pairs must cover the stored source degree')
    if radial_phase not in ('none','flight'):
        raise ValueError('radial_phase must be none or flight')
    began=perf_counter()
    table=CouplingTable.build(settings.spatial_degree,acceptance_degree)
    values=np.zeros((len(grid.omega_per_ns),len(grid.radii_m),settings.spatial_degree+1,table.pairs,2),complex)
    solver_diagnostics=[]
    for iw,omega in enumerate(grid.omega_per_ns):
        for lam in range(acceptance_degree+1):
            if medium.scattering_per_m==0:
                continue
            row_diagnostics=dict(frequency_per_ns=float(omega),acceptance_degree=lam)
            r,modes,components=_solve(medium,radius,lam,pairs=config.angular_pairs,
                points=config.radial_nodes,high=float(grid.radii_m[-1]),omega=float(omega),
                tail=config.exterior_relative_bound,ballistic_order=config.ballistic_quadrature,
                factored=True,collision_relative_bound=config.collision_relative_bound,
                collision_solver=config.collision_solver,diagnostics=row_diagnostics)
            solver_diagnostics.append(row_diagnostics)
            # Interpolate a slow envelope, retaining the exact optical phase
            # and attenuation factor at the requested radii.
            d=medium.absorption_per_m-1j*omega/medium.speed_m_per_ns
            envelope=components[...,1:]
            sampled=CubicSpline(np.log(r),envelope,axis=0)(np.log(grid.radii_m))
            sampled*=np.exp(-d*(grid.radii_m-radius))[:,None,None]
            alpha=4*np.pi/(2*lam+1)
            for p in np.flatnonzero(table.pair_lambda==lam):
                m=int(table.pair_mu[p]);normal=1/np.sqrt(2*np.pi if m==0 else np.pi)
                for n in range(m,settings.spatial_degree+1):
                    index=np.flatnonzero((modes[:,0]==m)&(modes[:,1]==n))[0]
                    # Radial construction uses OM->source; the fast kernel
                    # uses source->OM. Both real cosine/sine products acquire
                    # the same parity (-1)^(source degree+receiver degree).
                    values[iw,:,n,p]=(-1.)**(n+lam)*sampled[:,index]/(alpha*normal)
        if progress is not None:
            progress(iw+1,len(grid.omega_per_ns))
    contract=dict(schema='lighthit/first-entry-sphere/1',radius_m=radius,
                  solver='exterior-adjoint-galerkin',numerics=asdict(config),
                  collision_orders=['1','>=2'],angular_phase='full-HG-to-Galerkin-degree',
                  convergence_verified=False)
    return DirectionalCache(grid,medium,settings,table,values,
        dict(total=perf_counter()-began,frequencies=len(grid.omega_per_ns),radii=len(grid.radii_m),
             collision_solvers=solver_diagnostics),
        'spherical-adjoint',radial_phase=radial_phase,receiver_contract=contract)
