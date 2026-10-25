"""Finite first-entry receiver vertices for the exact free-tail RTE solver.

An absorbing sphere is represented by its boundary jump: inward positive
flux minus outward first-return response. The vertex is local to the glass;
propagation to distant sources retains the existing infinite free angular
tail. Collision orders are kept separate before applying the resolvent.
"""
from dataclasses import dataclass,asdict
from functools import lru_cache
import numpy as np
from scipy.special import spherical_jn
from .receiver import SphereSurface
from .receiver_adjoint import _basis,_solve
from .angular import free_tail_ratios_m,solve_tail_system_m


@dataclass(frozen=True)
class ReceiverVertexConfig:
    angular_pairs: int = 64
    radial_nodes: int = 400
    exterior_relative_bound: float = 1e-10
    collision_relative_bound: float = 1e-10
    plane_wave_tolerance: float = 1e-12
    radial_grading: float = 2.
    return_angle_nodes: int = 64
    return_path_nodes: int = 256

    def __post_init__(self):
        for key in ('angular_pairs','radial_nodes','return_angle_nodes','return_path_nodes'):
            value=getattr(self,key)
            if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)) or value<2:
                raise ValueError(f'{key} must be an integer >=2')
        for key in ('exterior_relative_bound','collision_relative_bound','plane_wave_tolerance'):
            value=getattr(self,key)
            if not np.isfinite(value) or not 0<value<1:raise ValueError(f'{key} must lie in (0,1)')
        if not np.isfinite(self.radial_grading) or self.radial_grading<1:
            raise ValueError('radial_grading must be finite and >=1')


def _plane_degree(z,tolerance):
    # |j_l(z)| <= z^l/(2*l+1)!! bounds the remaining plane-wave expansion.
    degree=0;term=1.
    while True:
        following=term*z/(2*degree+1)
        ratio=z/(2*degree+3)
        if ratio<1 and following/(1-ratio)<=tolerance:return degree
        degree+=1;term=following


def _project(k,R,acceptance_degree,spatial_degree,local):
    """Boundary-frame coefficients -> k-frame weighted receiver rows."""
    from .directional import CouplingTable
    velocity_degree=spatial_degree+acceptance_degree
    coupling=CouplingTable.build(velocity_degree,acceptance_degree)
    bessel=spherical_jn(np.arange(spatial_degree+1)[None,:],np.asarray(k)[:,None]*R)
    output=np.zeros((len(k),velocity_degree+1,coupling.pairs,local.shape[-1]),complex)
    for t,(lam,j,J) in enumerate(zip(coupling.triple_lambda,coupling.triple_l,coupling.triple_J)):
        if J>spatial_degree:continue
        pick=np.flatnonzero(coupling.pair_lambda==lam)
        coefficient=np.zeros(local.shape[-1],complex)
        for p in pick:
            m=coupling.pair_mu[p]
            coefficient+=coupling.nu_weight[t,m]*local[j,p]
        for p in pick:
            output[:,j,p]+=4*np.pi*R*R*bessel[:,J,None]*coupling.pair_weight[t,p]*coefficient[None,:]
    return coupling,output


def incoming_vertex(k,R,acceptance_degree,*,spatial_degree):
    """Exact inward finite-sphere flux, without first-return subtraction.

    Returned rows act on normalized velocity harmonics. This positive part
    alone counts returning photons and is not the first-entry detector.
    """
    from .directional import CouplingTable
    velocity_degree=spatial_degree+acceptance_degree
    table=CouplingTable.build(velocity_degree,acceptance_degree)
    order=(velocity_degree+acceptance_degree+3)//2
    x,w=np.polynomial.legendre.leggauss(max(2,order));mu=(x-1)/2;w=w/2
    local=np.zeros((velocity_degree+1,table.pairs,3),complex)
    for p,(lam,m) in enumerate(zip(table.pair_lambda,table.pair_mu)):
        degrees=np.arange(m,velocity_degree+1)
        q=_basis(mu,m,degrees);a=_basis(mu,m,np.array([lam]))[:,0]
        local[degrees,p,0]=q.T@(w*(-mu)*a)/(np.pi*R*R)
    return _project(k,R,acceptance_degree,spatial_degree,local)


def _outgoing_flux(returning,m,degree):
    """Low streaming moments of a field with zero incoming characteristics."""
    if returning.shape[0]<=degree+1-m:
        raise ValueError('boundary angular grid must cover vertex streaming moments')
    result=np.empty((degree+1-m,returning.shape[1]),complex)
    for j in range(m,degree+1):
        flux=np.sqrt(((j+1)**2-m*m)/((2*j+1)*(2*j+3)))*returning[j+1-m]
        if j>m:flux+=np.sqrt((j*j-m*m)/((2*j-1)*(2*j+1)))*returning[j-1-m]
        result[j-m]=flux
    return result


@lru_cache(maxsize=2)
def _return_geometry(g,extent,angle_nodes,path_nodes):
    """Geometry of one return from a genuinely outgoing boundary pencil."""
    from .receiver_first import first_entry_profile,MOMENT_POWERS
    from scipy.interpolate import PchipInterpolator
    from .event_moments import real_spherical_harmonics
    x,w=np.polynomial.legendre.leggauss(angle_nodes);mu=(x+1)/2;weights=w/2
    qx,qw=np.polynomial.legendre.leggauss(8)
    # Cubic harmonic polynomials in the pencil's reflection-even moments.
    lat,_=np.polynomial.legendre.leggauss(6);phi=np.arange(8)*2*np.pi/8
    z=np.repeat(lat,8);az=np.tile(phi,6)
    directions=np.stack((np.sqrt(1-z*z)*np.cos(az),np.sqrt(1-z*z)*np.sin(az),z),axis=-1)
    design=np.stack([directions[:,0]**i*directions[:,2]**j for i,j in MOMENT_POWERS],axis=-1)
    paths=[];density=[];polynomials=[]
    for cosine in mu:
        profile=first_entry_profile(SphereSurface(1.),distance_m=1.,direction_cosine=-cosine,g=g,
            max_path_m=extent,path_nodes=path_nodes,outgoing_boundary=True)
        h=np.diff(profile.path_m)/2
        s=(profile.path_m[1:]+profile.path_m[:-1])[:,None]/2+h[:,None]*qx
        paths.append(s.ravel())
        derivative=PchipInterpolator(profile.path_m,profile.cumulative_m_inv,axis=0).derivative()
        density.append((h[:,None,None]*qw[None,:,None]*derivative(s)).reshape(-1,10))
        sine=np.sqrt((1-cosine)*(1+cosine))
        rotated=np.stack((cosine*directions[:,0]+sine*directions[:,2],directions[:,1],
                          -sine*directions[:,0]+cosine*directions[:,2]),axis=-1)
        harmonics=real_spherical_harmonics(3,rotated)
        polynomials.append(np.linalg.lstsq(design,harmonics,rcond=None)[0])
    return mu,weights,np.asarray(paths),np.asarray(density),np.asarray(polynomials)


def _first_return_local(omega,medium,R,table,velocity_degree,config):
    """Exact full-HG one-return flux, independently of PN boundary closure.

    A half-range ballistic jump needs very many PN modes to resolve its
    outgoing first collision. A positive geometric Q1 integral avoids that
    closure error before the unchanged free-tail propagation is applied.
    """
    if table.acceptance_degree>3:raise ValueError('geometric return requires acceptance degree <=3')
    extent=-np.log(config.exterior_relative_bound)/(medium.extinction_per_m*R)
    mu,w,paths,density,polynomials=_return_geometry(medium.g,extent,
        config.return_angle_nodes,config.return_path_nodes)
    folded=medium.scattering_per_m/R*np.einsum('un,unm->um',
        np.exp(-(medium.extinction_per_m-1j*omega/medium.speed_m_per_ns)*R*paths),density)
    result=np.zeros((velocity_degree+1,table.pairs),complex)
    for p,(lam,m) in enumerate(zip(table.pair_lambda,table.pair_mu)):
        norm=1/np.sqrt(2*np.pi if m==0 else np.pi)
        response=np.sum(folded*polynomials[:,:,lam*lam+lam+m],axis=1)/norm
        result[m:,p]=_basis(mu,m,np.arange(m,velocity_degree+1)).T@(w*mu*response)
    return result


def first_entry_vertex(k,omega,medium,surface,scattering_degree,acceptance_degree,*,config=None,report=None,
                       _minimum_return_degree=0):
    config=config or ReceiverVertexConfig()
    if not isinstance(surface,SphereSurface) or surface.radius_m.ndim!=0:
        raise ValueError('one explicit spherical receiver radius required')
    R=float(surface.radius_m);k=np.asarray(k,float)
    if k.ndim!=1 or not len(k) or not np.isfinite(k).all() or np.any(k<0):
        raise ValueError('finite nonnegative wave numbers required')
    if 2*config.angular_pairs-1<scattering_degree:
        raise ValueError('boundary angular grid must cover scattering degree')
    degree=_plane_degree(float(np.max(k)*R),config.plane_wave_tolerance)
    table,positive=incoming_vertex(k,R,acceptance_degree,spatial_degree=degree)
    velocity_degree=degree+acceptance_degree
    local=np.zeros((velocity_degree+1,table.pairs,3),complex);diagnostics=[]
    if medium.scattering_per_m:
        for lam in range(_minimum_return_degree,acceptance_degree+1):
            row={}
            _,modes,field=_solve(medium,R,lam,pairs=config.angular_pairs,points=config.radial_nodes,
                high=R,omega=omega,tail=config.exterior_relative_bound,factored=True,
                collision_relative_bound=config.collision_relative_bound,scattering_degree=scattering_degree,
                diagnostics=row,radial_grading=config.radial_grading)
            diagnostics.append(row)
            alpha=4*np.pi/(2*lam+1)
            for p in np.flatnonzero(table.pair_lambda==lam):
                m=int(table.pair_mu[p]);pick=np.flatnonzero(modes[:,0]==m)
                norm=alpha/np.sqrt(2*np.pi if m==0 else np.pi)
                # The solved incoming characteristics vanish at the glass.
                # Thus its outgoing flux is the full streaming moment M*h.
                # Integrating a reconstructed PN polynomial on a half-range
                # instead invents incoming leakage between the angular nodes;
                # that breaks boundary flux conservation and converges slowly.
                local[m:,p,1:]=_outgoing_flux(field[0,pick,1:],m,velocity_degree)/norm
        local[:,:,1]=_first_return_local(omega,medium,R,table,velocity_degree,config)
    _,negative=_project(k,R,acceptance_degree,degree,local)
    vertex=positive+negative
    if report is not None:
        report.update(receiver_vertex_degree=degree,receiver_velocity_degree=velocity_degree,
            boundary_solvers=diagnostics,receiver_contract=dict(schema='lighthit/first-entry-sphere/1',
            radius_m=R,solver='boundary-jump-exact-free-tail',numerics=asdict(config),
            scattering_degree=int(scattering_degree),convergence_verified=False))
    return table,vertex


def receiver_rows(k,omega,medium,scattering_degree,source_degree,acceptance_degree,vertex,table,*,backend='numpy'):
    """D G1 - B1 G0 and D G>=2 - B1 G>=1 - B>=2 G.

    Factored form: first=G0*(Gamma*G0*D-B1), multiple=G*(Gamma*first-B>=2).
    The same tridiagonal systems and exact free tails serve the point kernel.
    """
    L=int(scattering_degree);J=int(source_degree);A=int(acceptance_degree)
    N=max(L,vertex.shape[1]-1);output=max(N,J)
    d0=medium.extinction_per_m-1j*omega/medium.speed_m_per_ns
    ratios=free_tail_ratios_m(k,d0,output,A,backend=backend)
    gamma=medium.scattering_per_m*medium.g**np.arange(N+1)
    gamma[np.arange(N+1)>L]=0.
    first={};multiple={}
    for p,(lam,m) in enumerate(zip(table.pair_lambda,table.pair_mu)):
        m=int(m);lam=int(lam);size=N+1-m
        rhs=np.zeros((len(k),size,3),complex)
        upper=min(vertex.shape[1],N+1)
        rhs[:,:upper-m]=vertex[:,m:upper,p]
        tail=ratios[:,m,N];g=gamma[m:]
        free=solve_tail_system_m(k,d0,tail,np.zeros(size),rhs[:,:,0],m)
        one=solve_tail_system_m(k,d0,tail,np.zeros(size),free*g-rhs[:,:,1],m)
        many=solve_tail_system_m(k,d0,tail,g,one*g-rhs[:,:,2],m)
        for target,low in ((first,one),(multiple,many)):
            full=np.zeros((len(k),output+1),complex);full[:,m:N+1]=low
            if output>N:full[:,N+1:]=low[:,-1:]*np.cumprod(ratios[:,m,N:output],axis=1)
            target[(m,lam)]=full
    return first,multiple
