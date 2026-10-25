"""Spectral, physical-time folding of first-entry direction-moment profiles.

The two stored source fields are the usual lambda^-2 and
lambda^-2*n_phase^-2 weights. Physical paths end on the supplied glass.
No area, source yield, source clock or event normalization is included.
"""
from dataclasses import dataclass
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.special import roots_legendre


def acceptance_moment_weights(look, polynomial):
    """Contract a cubic acceptance with the ten first-entry x/z moments.

    The omitted odd-y moments vanish by reflection. Even-y moments follow
    from x*x+y*y+z*z=1, retaining a tilted receiver's full response.
    ``look`` points along the accepted photon propagation direction, in the
    pencil's local frame (z=initial ray, x=towards receiver).
    """
    look=np.asarray(look,float);polynomial=np.asarray(polynomial,float)
    if (look.shape!=(3,) or not np.isfinite(look).all()
            or not np.isclose(np.linalg.norm(look),1.,rtol=0,atol=1e-12)):
        raise ValueError('one finite unit receiver look required')
    if (polynomial.ndim!=1 or not 1<=len(polynomial)<=4
            or not np.isfinite(polynomial).all()):
        raise ValueError('finite monomial acceptance of degree <=3 required')
    a=np.pad(polynomial,(0,4-len(polynomial)))
    x,y,z=look;y2=y*y
    return np.array([a[0]+a[2]*y2,
        a[1]*x+3*a[3]*x*y2,a[1]*z+3*a[3]*z*y2,
        a[2]*(x*x-y2),2*a[2]*x*z,a[2]*(z*z-y2),
        a[3]*(x*x*x-3*x*y2),3*a[3]*z*(x*x-y2),
        3*a[3]*x*(z*z-y2),a[3]*(z*z*z-3*z*y2)])


@dataclass(frozen=True)
class FoldedFirstEntry:
    """Truncated charge/CDF moments, with a conservative spectral tail bound.

    ``charge`` is (two source fields, ten moments); ``cumulative`` is
    (time edge, two fields, ten moments). ``tail_bound`` bounds the absolute
    omitted value of *each* moment for each field, including all-time charge.
    After acceptance contraction multiply the bound by the sum of absolute
    moment weights; area and source coefficients remain consumer factors.
    """
    time_edges_ns: np.ndarray
    charge: np.ndarray
    cumulative: np.ndarray
    tail_bound: np.ndarray
    path_extent_m: float
    quadrature_order: int

    @property
    def bins(self):
        return np.diff(self.cumulative,axis=0)


def fold_first_entry(profile, *, extinction_per_m, scattering_per_m,
                     speed_m_per_ns, spectral_weights, time_edges_ns=(),
                     quadrature_order=8):
    """Fold a geometry row with wavelength weights and actual glass clocks.

    Spectral weights have shape (wavelength,2), without area or source yield.
    Finite profile extent is explicit: omitted tails are never assumed zero.
    ``time_edges_ns`` are measured from the photon's emission clock.
    Increasing the quadrature order checks numerical folding independently
    of the profile's path and scatter quadratures.
    """
    mu=np.asarray(extinction_per_m,float);mus=np.asarray(scattering_per_m,float)
    speed=np.asarray(speed_m_per_ns,float);weights=np.asarray(spectral_weights,float)
    times=np.asarray(time_edges_ns,float)
    if (mu.ndim!=1 or not len(mu) or mus.shape!=mu.shape or speed.shape!=mu.shape
            or weights.shape!=(len(mu),2) or not np.isfinite(mu).all()
            or not np.isfinite(mus).all() or not np.isfinite(speed).all()
            or not np.isfinite(weights).all() or np.any(mu<=0) or np.any(mus<0)
            or np.any(mus>mu) or np.any(speed<=0) or np.any(weights<0)):
        raise ValueError('positive extinction/speed, 0<=scattering<=extinction and nonnegative two-field weights required')
    if times.ndim!=1 or not np.isfinite(times).all() or (len(times) and (len(times)<2 or np.any(np.diff(times)<=0))):
        raise ValueError('provide no edges or increasing finite time edges')
    if isinstance(quadrature_order,(bool,np.bool_)) or not isinstance(quadrature_order,(int,np.integer)) or quadrature_order<2:
        raise ValueError('quadrature_order must be an integer >=2')
    path=np.asarray(profile.path_m,float)
    curve=PchipInterpolator(path,profile.cumulative_m_inv,axis=0).derivative()
    x,w=roots_legendre(quadrature_order)
    half=np.diff(path)/2
    nodes=(path[1:]+path[:-1])[:,None]/2+half[:,None]*x
    density=curve(nodes)
    charge=np.zeros((2,10));cumulative=np.zeros((len(times),2,10))
    for band,(attenuation,scatter,velocity) in enumerate(zip(mu,mus,speed)):
        increments=np.einsum('pq,pqm->pm',half[:,None]*w*np.exp(-attenuation*nodes),density)
        prefix=np.concatenate((np.zeros((1,10)),np.cumsum(increments,axis=0)))
        charge+=weights[band,:,None]*scatter*prefix[-1]
        if len(times):
            stop=np.clip(times*velocity,path[0],path[-1])
            index=np.clip(np.searchsorted(path,stop,side='right')-1,0,len(path)-2)
            width=(stop-path[index])/2
            partial_nodes=(stop+path[index])[:,None]/2+width[:,None]*x
            partial=np.einsum('tq,tqm->tm',width[:,None]*w*np.exp(-attenuation*partial_nodes),curve(partial_nodes))
            value=prefix[index]+partial
            cumulative+=scatter*value[:,None,:]*weights[band,None,:,None]
    # Cap solid angle / projected area <=2/distance^2. Integrating along
    # an outside pencil up to its first hit is <=pi/R. The HG maximum and
    # |x^i*z^j|<=1 bound every remaining moment, independent of wavelength.
    g=abs(profile.g);phase_max=(1+g)/(4*np.pi*(1-g)**2)
    geometry_bound=2*np.pi*phase_max/profile.receiver_radius_m
    tail=np.sum(weights*mus[:,None]*np.exp(-mu[:,None]*path[-1])*geometry_bound,axis=0)
    R=profile.receiver_radius_m;r=profile.distance_m/R;c=profile.direction_cosine
    if c>0 and r*np.sqrt((1-c)*(1+c))<=1 and path[-1]/R>=np.sqrt((r-1)*(r+1)):
        tail[:]=0.  # Hitting pencils have no Q1 after the tangent path.
    return FoldedFirstEntry(times.copy(),charge,cumulative,tail,float(path[-1]),int(quadrature_order))
