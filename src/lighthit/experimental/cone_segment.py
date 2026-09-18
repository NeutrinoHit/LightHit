"""Finite straight Cherenkov-cone segment, POINT ISOTROPIC receiver.

Teaching/reference implementation, not a fast G4 event evaluator. The cone,
orientation, endpoints and times are explicit. Scattered orders use the finite-L
moments already in ResponseCache (including finite-L order 1). Ballistic light
is evaluated analytically. All angular degrees required by the cone must be
converged; the two-moment cache for isotropic sources is not sufficient.
"""
from dataclasses import dataclass
import numpy as np
from scipy.special import roots_legendre, eval_legendre
from ..medium import C_VACUUM_M_PER_NS
from ..cache import BandedResponseCache


@dataclass(frozen=True)
class ConeSegment:
    start_m: tuple
    direction: tuple
    length_m: float
    beta: float
    phase_index: float
    photons_per_m: float
    start_time_ns: float = 0.0

    def __post_init__(self):
        x=np.asarray(self.start_m,float); u=np.asarray(self.direction,float)
        if x.shape!=(3,) or u.shape!=(3,) or not np.isfinite(x).all() or not np.isfinite(u).all():
            raise ValueError("start and direction must be finite 3-vectors")
        if not np.isclose(np.linalg.norm(u),1,rtol=0,atol=1e-12):
            raise ValueError("direction must be a unit vector")
        vals=[self.length_m,self.beta,self.phase_index,self.photons_per_m,self.start_time_ns]
        if not np.isfinite(vals).all() or self.length_m<=0 or not 0<self.beta<=1 or self.phase_index<=0 or self.photons_per_m<0:
            raise ValueError("Invalid segment length, beta, phase index, yield or time")
        if self.beta*self.phase_index<=1:
            raise ValueError("This cone segment requires beta*n_phase>1; below threshold yield is zero")

    @property
    def speed_m_per_ns(self):
        return self.beta*C_VACUUM_M_PER_NS

    @property
    def cone_cosine(self):
        return 1/(self.beta*self.phase_index)


def ballistic_spectrum(omega, segment: ConeSegment, displacement_from_start, medium):
    """Analytic cone-segment ballistic spectrum, per unit effective area.

    The emission root is a*=z-b*mu_C/sin(theta_C); the segment is half-open
    [0,length), so exact endpoint roots are not counted twice after splitting.
    Positive impact distance is required. No source crosses a point receiver.
    """
    w=np.atleast_1d(np.asarray(omega,float)); r=np.atleast_2d(np.asarray(displacement_from_start,float))
    if w.ndim!=1 or not np.isfinite(w).all() or r.ndim!=2 or r.shape[1]!=3 or not np.isfinite(r).all():
        raise ValueError("Invalid frequency or displacement array")
    u=np.asarray(segment.direction,float); z=r@u
    b=np.linalg.norm(r-z[:,None]*u,axis=1)
    if np.any(b<=1e-12):
        raise ValueError("On-axis cone/point-detector geometry needs separate regularization")
    mu=segment.cone_cosine; st=np.sqrt(1-mu*mu)
    a=z-b*mu/st; distance=b/st
    inside=(a>=0)&(a<segment.length_m)
    charge=segment.photons_per_m*np.exp(-medium.extinction_per_m*distance)/(2*np.pi*b*st)
    charge=np.where(inside,charge,0.0)
    t=segment.start_time_ns+a/segment.speed_m_per_ns+distance/medium.speed_m_per_ns
    return charge[None,:]*np.exp(1j*w[:,None]*t[None,:])


def _scatter(cache, radii, cosines, cone_cosine):
    """Integrate each incident-cone azimuth analytically by the addition theorem."""
    if isinstance(cache,BandedResponseCache):
        bands=cache.bands; omega=bands[0].grid.omega_per_ns
    else:
        bands=[cache];omega=cache.grid.omega_per_ns
    low,high=bands[0].radius_range_m[0],bands[-1].radius_range_m[1]
    if np.any(radii<low) or np.any(radii>high):
        raise ValueError("Some emission nodes are outside the cached radial range")
    ends=[b.radius_range_m[1] for b in bands[:-1]]
    which=np.searchsorted(ends,radii,side='left')
    out=np.zeros((len(omega),len(radii),2),complex)
    for ib,band in enumerate(bands):
        mask=which==ib
        if not np.any(mask):continue
        ell=np.arange(band.degree+1)
        angular=(eval_legendre(ell,cone_cosine)[None,:]
                 *eval_legendre(ell[None,:],cosines[mask,None]))
        out[:,mask]=np.einsum('wrlo,rl->wro',band.moments_at(radii[mask]),angular,optimize=True)
    return out


def segment_spectrum(cache, segment: ConeSegment, detector_positions_m,
                     *, longitudinal_order=16, detector_chunk=64):
    """Return (frequency, detector, 3) for orders (0, finite-L 1, >=2).

    y is photons per metre in the cache's monochromatic band. Phase and group
    indices are distinct. Nodes follow the exact straight segment, not an
    averaged shower axis. Refine longitudinal_order and cache independently.
    """
    if not isinstance(longitudinal_order,(int,np.integer)) or longitudinal_order<2:
        raise ValueError("longitudinal_order must be an integer >=2")
    if not isinstance(detector_chunk,(int,np.integer)) or detector_chunk<1:
        raise ValueError("detector_chunk must be a positive integer")
    pos=np.atleast_2d(np.asarray(detector_positions_m,float))
    if pos.ndim!=2 or pos.shape[1]!=3 or not np.isfinite(pos).all():
        raise ValueError("detector_positions_m must have shape (D,3)")
    bands=cache.bands if isinstance(cache,BandedResponseCache) else [cache]
    omega=bands[0].grid.omega_per_ns;medium=bands[0].medium
    r0=pos-np.asarray(segment.start_m,float)
    out=np.zeros((len(omega),len(pos),3),complex)
    out[:,:,0]=ballistic_spectrum(omega,segment,r0,medium)
    nodes,weights=roots_legendre(longitudinal_order)
    a=(nodes+1)*segment.length_m/2
    weights=weights*segment.length_m/2*segment.photons_per_m
    time=segment.start_time_ns+a/segment.speed_m_per_ns
    phase=np.exp(1j*omega[:,None]*time[None,:])
    u=np.asarray(segment.direction,float)
    for lo in range(0,len(pos),detector_chunk):
        vec=r0[None,lo:lo+detector_chunk,:]-a[:,None,None]*u
        radii=np.linalg.norm(vec,axis=2)
        if np.any(radii<=0):raise ValueError("An emission node coincides with a receiver")
        cosine=(vec@u)/radii
        field=_scatter(cache,radii.ravel(),cosine.ravel(),segment.cone_cosine)
        field=field.reshape(len(omega),len(a),len(radii[0]),2)
        out[:,lo:lo+detector_chunk,1:]=np.einsum('wj,j,wjdo->wdo',phase,weights,field,optimize=True)
    return out
