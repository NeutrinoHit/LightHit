"""Positive q1 charge bounds including segments crossing the Cherenkov cone.

These certify omitted charge for the supplied positive spectral quadrature,
not its error against the continuous spectrum. The finite source is extended
into an infinite uniform straight track with the same nonnegative yield per
metre. No signal is set to zero simply because it lies outside a cache.
"""
import numpy as np


def hg_angle_integral_over_sine_upper(theta,g):
    """Bound int_theta^pi p_HG(cos chi) dchi / sin(theta), HG 0<=g<1.

    Concavity bounds sin(chi/2) below by its chord on [theta,pi]. The
    resulting inverse-cube integral is elementary. Rationalization avoids
    subtracting almost equal endpoints near pi; the antipodal limit is finite.
    At theta=0 the pencil integral is singular and returns infinity.
    """
    theta=np.asarray(theta,float)
    if not np.isfinite(theta).all() or np.any(theta<0) or np.any(theta>np.pi) or not 0<=g<1:
        raise ValueError('theta in [0,pi] and HG 0<=g<1 required')
    backward=np.pi-theta
    ratio=np.ones_like(theta)
    regular=(theta>0)&(backward>1e-5)
    np.divide(backward,np.sin(theta),out=ratio,where=regular)
    small=backward<=1e-5
    ratio[small]=1+backward[small]**2/6+7*backward[small]**4/360
    v=np.sin(theta/2);d0=np.sqrt((1-g)**2+4*g*v*v);d1=1+g
    norm=(1-g*g)/(4*np.pi)
    value=norm*ratio*(1+v)/(d1*d0*(d0+v*d1))
    value=np.nextafter(value*(1+128*np.finfo(float).eps),np.inf)
    return np.where(theta>0,value,np.inf)


def yukawa_convolution_upper(impact_m, first_extinction_per_m,
                             extinction_per_m, *, nodes=32):
    """Upper bound for the 2D convolution (exp(-a*r)/r)*(exp(-b*r)/r).

    For a>=b>0 and rho>0 its exact value is
      4 int_0^(pi/2) K0(rho*sqrt(a²*sin²(t)+b²*cos²(t))) dt.
    The integrand decreases: positive LEFT rectangles bound the integral.
    K0(x)<=sqrt(pi/(2*x))*exp(-x) follows from cosh(t)>=1+t²/2.
    The axis rho=0 is singular and returns infinity, never a finite bound.
    """
    if isinstance(nodes, bool) or not isinstance(nodes, (int, np.integer)) or nodes < 1:
        raise ValueError('nodes must be a positive integer')
    rho,a,b=np.broadcast_arrays(np.asarray(impact_m,float),
                               np.asarray(first_extinction_per_m,float),
                               np.asarray(extinction_per_m,float))
    if (not all(np.isfinite(x).all() for x in (rho,a,b)) or np.any(rho<0)
            or np.any(b<=0) or np.any(a<b)):
        raise ValueError('finite rho>=0 and a>=b>0 required')
    positive=rho>0
    result=np.zeros_like(rho)
    safe_rho=np.where(positive,rho,1.)
    for index in range(nodes):
        angle=(np.pi/2)*(index/nodes)
        scale=np.hypot(a*np.sin(angle),b*np.cos(angle))
        x=safe_rho*scale
        result+=np.sqrt(np.pi/(2*x))*np.exp(-x)
    result*=2*np.pi/nodes
    # Conservative numerical padding; this is not interval arithmetic.
    result=np.nextafter(result*(1+128*np.finfo(float).eps),np.inf)
    return np.where(positive,result,np.inf)



def yukawa_convolution_elementary_upper(impact_m, first_extinction_per_m,
                                        extinction_per_m):
    """O(1) positive upper bound, no angular sum or special functions.

    Elliptic coordinates give I_ab=2*pi*I0(d)*K0(c), with
    c=rho*(a+b)/2 and d=rho*(a-b)/2. The elementary inequalities
      exp(-d)*I0(d) <= min(1,sqrt(pi/(8*d))),
      exp(c)*K0(c) <= sqrt(pi/(2*c))
    imply the result. The first follows from sin(t/2)>=t/pi on [0,pi]
    and extending the resulting Gaussian integral to infinity. At d=0 its
    limiting first bound is one. This also covers a=b without division by zero.
    """
    rho,a,b=np.broadcast_arrays(np.asarray(impact_m,float),
                               np.asarray(first_extinction_per_m,float),
                               np.asarray(extinction_per_m,float))
    if (not all(np.isfinite(x).all() for x in (rho,a,b)) or np.any(rho<0)
            or np.any(b<=0) or np.any(a<b)):
        raise ValueError('finite rho>=0 and a>=b>0 required')
    radius=np.where(rho>0,rho,1.)
    c=radius*(a+b)/2;d=radius*(a-b)/2
    angular=np.minimum(1.,np.sqrt(np.pi/(8*np.maximum(d,np.finfo(float).tiny))))
    result=2*np.pi*np.exp(-radius*b)*np.sqrt(np.pi/(2*c))*angular
    result=np.nextafter(result*(1+128*np.finfo(float).eps),np.inf)
    return np.where(rho>0,result,np.inf)

def infinite_track_first_scattering_bounds(kernel,elements,receiver_m,*,
                         effective_area_m2=None,quadrature_nodes=32,convolution_method='elementary'):
    """All-time PE upper bound for every supplied uniform source segment.

    Infinite-track direct fluence is Y*exp(-mut*rho/sin(theta))/(2*pi*rho*sin(theta)).
    Replacing HG by pmax and integrating the last longitudinal flight gives
      q1 <= Y*mus*pmax/(2*sin(theta)) * convolution_upper(rho,mut/sin(theta),mut).
    The cone-crossing singularity of the finite-ray bound is absent for rho>0.
    Y is the actual Cherenkov spectral photon yield per metre, including this
    kernel's spectral quadrature/QE weights. Detector area and angular maximum
    are applied once. A zero-yield or zero-scattering band contributes zero.
    """
    from ..prompt import _spectral_tables
    if convolution_method not in ('elementary','left-rectangle'):
        raise ValueError('choose elementary or left-rectangle convolution bound')
    if elements.cone_model!='spectral':
        raise ValueError('bounds require wavelength-dependent Cherenkov cones')
    if not np.isfinite(elements.length_m).all() or np.any(elements.length_m<=0):
        raise ValueError('positive finite segment lengths required')
    g=float(kernel.medium.g)
    if not np.isfinite(g) or abs(g)>=1:
        raise ValueError('HG bound requires finite |g|<1')
    receiver=np.asarray(receiver_m,float)
    if receiver.shape!=(3,) or not np.isfinite(receiver).all():
        raise ValueError('one finite receiver required')
    area=float(kernel.detector.effective_area_m2[0]
               if effective_area_m2 is None else effective_area_m2)
    if not np.isfinite(area) or area<0:
        raise ValueError('finite nonnegative detector area required')
    tables=_spectral_tables(kernel)
    if (np.any(tables['mus']<0) or np.any(tables['mut']<tables['mus'])
            or np.any(tables['weight_nm']<0) or np.any(tables['efficiency']<0)):
        raise ValueError('nonnegative physical rates and spectral weights required')
    alpha=np.asarray(kernel.acceptance()['alpha'],float)
    amax=float(np.sum(abs(alpha)*(2*np.arange(len(alpha))+1))/(4*np.pi))
    pmax=(1+abs(g))/(4*np.pi*(1-abs(g))**2)
    delta=receiver-elements.start_m
    z=np.sum(delta*elements.direction,axis=1)
    rho=np.linalg.norm(delta-z[:,None]*elements.direction,axis=1)
    phase=tables['phase_index'][None,:]
    active=(elements.beta[:,None]*phase>1)&(tables['mus'][None,:]>0)
    c0=elements.coefficient0[:,None]/elements.length_m[:,None]
    c2=elements.coefficient2[:,None]/elements.length_m[:,None]
    photons=c0*tables['s0w'][None,:]+c2*tables['s2w'][None,:]
    active &= photons>0
    response=np.zeros_like(photons)
    if np.any(active) and area>0 and amax>0:
        cosine=1/(elements.beta[:,None]*phase)
        sine=np.sqrt(np.maximum(0.,1-cosine*cosine))
        ii,jj=np.nonzero(active)
        b=tables['mut'][jj]
        integral=(yukawa_convolution_elementary_upper(rho[ii],b/sine[ii,jj],b)
                  if convolution_method=='elementary' else
                  yukawa_convolution_upper(rho[ii],b/sine[ii,jj],b,nodes=quadrature_nodes))
        response[ii,jj]=photons[ii,jj]*tables['mus'][jj]*pmax/(2*sine[ii,jj])*integral
    bounds=area*amax*response.sum(axis=1)
    # Preserve exact zero for physically zero area/yield/scattering.
    positive=bounds>0
    bounds[positive]=np.nextafter(bounds[positive]*(1+64*np.finfo(float).eps),np.inf)
    return bounds,dict(bound_scope='all-time q1 for supplied positive spectral quadrature',
        continuum_spectral_error_included=False,infinite_uniform_track_extension=True,
        convolution_method=convolution_method,
        convolution_left_rectangles=int(quadrature_nodes) if convolution_method=='left-rectangle' else None,K0_bound='exp(-x)*sqrt(pi/(2*x))',
        spectral_nodes=len(tables['wavelength_nm']),acceptance_upper=amax,HG_upper=pmax,
        unsupported_pieces=int(np.count_nonzero(~np.isfinite(bounds))))


def combined_first_scattering_segment_bounds(kernel,elements,receiver_m,*,
                         effective_area_m2=None,quadrature_nodes=32,convolution_method='elementary'):
    """Minimum of independent finite-ray and infinite-line upper bounds."""
    from .track_first_primitive import first_scattering_segment_bounds
    # Legacy finite-ray formula may encounter 0/0 in an exactly transparent
    # band. An undefined candidate cannot sharpen a valid upper bound.
    with np.errstate(divide='ignore',invalid='ignore'):
        ray,ray_meta=first_scattering_segment_bounds(kernel,elements,receiver_m,
                                                  effective_area_m2=effective_area_m2)
    ray=np.where(np.isnan(ray),np.inf,ray)
    line,line_meta=infinite_track_first_scattering_bounds(kernel,elements,receiver_m,
        effective_area_m2=effective_area_m2,quadrature_nodes=quadrature_nodes,convolution_method=convolution_method)
    value=np.minimum(ray,line)
    return value,dict(bound_scope=line_meta['bound_scope'],continuum_spectral_error_included=False,
        ray=ray_meta,infinite_line=line_meta,line_tighter_pieces=int(np.count_nonzero(line<ray)),
        unsupported_pieces=int(np.count_nonzero(~np.isfinite(value))))
