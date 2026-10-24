"""Backward-source >=2 integral with exact flight phase and Filon panels.

The same finite RTE coefficients, spectral cones and detector acceptance are
used. Only a smooth amplitude is interpolated; oscillations are integrated
using the Legendre plane-wave identity. No event response is a build input.
"""
from functools import lru_cache
from contextlib import contextmanager
import numpy as np
from scipy.special import spherical_jn


@lru_cache(maxsize=8)
def _projection(order):
    """Gauss nodes and the inverse Vandermonde matrix that maps node values to Legendre coefficients."""
    x,_=np.polynomial.legendre.leggauss(order)
    inverse=np.linalg.inv(np.polynomial.legendre.legvander(x,order-1))
    return x,inverse


def backward_panels(bundle,rho,lo,hi,*,angular_step=None):
    """Include every radial spline knot and bound source/OM angular change."""
    if not np.isfinite([rho,lo,hi]).all() or rho<=0 or not lo<hi<=0:
        raise ValueError('Filon requires a positive impact and a backward interval')
    angle_step=1/(bundle['degree']+bundle['acceptance_degree']+1) if angular_step is None else angular_step
    if not np.isfinite(angle_step) or angle_step<=0:raise ValueError('positive angular step required')
    radii=np.exp(bundle['log_r'])
    rlo=np.hypot(rho,hi);rhi=np.hypot(rho,lo)
    knots=radii[(radii>rlo)&(radii<rhi)]
    u=-np.sqrt(np.maximum(0.,knots*knots-rho*rho))
    angles=np.arctan2(np.array([hi,lo]),rho)
    count=max(1,int(np.ceil(abs(angles[1]-angles[0])/angle_step)))
    angular=rho*np.tan(np.linspace(angles[1],angles[0],count+1))
    result=np.unique(np.clip(np.r_[lo,hi,u,angular[1:-1]],lo,hi))[::-1]
    return result


def retarded_inverse(tau,rho,speed,clock):
    """Stable backward root of tau=hypot(rho,u)/speed-clock*u."""
    T=np.asarray(tau)*speed;B=clock*speed
    root=np.sqrt(np.maximum(0.,T*T-(1-B*B)*rho*rho))
    return (rho*rho-T*T)/(B*T+root)


def warm_direct_rte():
    """Warm both coefficient layouts on a tiny synthetic pack, not event data."""
    from .track_ge2_primitive import cone_point_evaluator
    coefficients=np.zeros((1,2,3,3,4,3),np.complex64);coefficients.setflags(write=False)
    bundle=dict(degree=2,acceptance_degree=1,omega=np.array([0.,.3,.6]),
        pair_lambda=np.array([0,1,1]),pair_mu=np.array([0,0,1]),
        log_r=np.log([3.,20.,300.]),coefficients=coefficients,absorption=np.array([.03]),
        speed=np.array([.22]),phase='flight')
    kwargs=dict(impact_m=8.,look=np.array([0.,0.,1.]),alpha=np.array([4*np.pi,1.]),
        wavelength_nm=np.array([480.]),wavelength_weight_nm=np.ones(1),phase_index=np.array([1.35]),
        efficiency=np.ones(1),effective_area_m2=.05,backend='fused-looks')
    for local in [bundle,dict(bundle,omega=bundle['omega'][:1],coefficients=coefficients[...,:1])]:
        cone_point_evaluator(local,**kwargs)(np.array([-2.,1.]))


@contextmanager
def wavelength_plane(coefficients,band,*,bounded_mapping=True):
    """Map one C-layout spectral plane; close it after its synchronous apply.

    The full pack's mapping remains untouched by this path. Ordinary arrays
    and noncontiguous packs retain a zero-copy slice instead.
    """
    mapped=None
    if bounded_mapping and isinstance(coefficients,np.memmap) and coefficients.flags.c_contiguous:
        size=int(np.prod(coefficients.shape[1:]))*coefficients.dtype.itemsize
        mapped=np.memmap(coefficients.filename,mode='r',dtype=coefficients.dtype,
            offset=coefficients.offset+band*size,shape=(1,)+coefficients.shape[1:])
        plane=mapped
    else:plane=coefficients[band:band+1]
    try:yield plane
    finally:
        if mapped is not None:mapped._mmap.close()


def backward_ge2_filon_many(kernel,bundle,queries,*,order=6,bounded_mapping=True,allow_gauss=False,
                            spectral=None):
    """Queries are rho,look,beta,clock,lo,hi,density,closest tuples.

    ``spectral`` (wavelength_nm, weight_nm, phase_index, efficiency per band of
    ``bundle``) replaces the kernel's wavelength nodes; it is how a folded
    bundle (``band_fold.fold_bundle``: node bands with a common attenuation)
    is applied. Without it the bundle must hold the kernel's nodes.

    All wavelength planes are processed one at a time over all queries. No
    mmap-backed arrays leave the plane scope. Positive clocks of at least
    0.5/min(speed) and a flight-referenced pack avoid turning-point singularities.
    Returns (query-by-frequency spectra, numerical-work metadata).
    """
    from .track_ge2_primitive import cone_point_evaluator
    if order not in (4,6,8,10,12):raise ValueError('unsupported Filon order')
    if bundle['phase']!='flight':raise ValueError('Filon needs flight-referenced RTE coefficients')
    if not allow_gauss and any(q[3]*np.min(bundle['speed'])<.5 for q in queries):
        raise ValueError('Filon clock too small for the backward branch')
    if spectral is None:
        wave=kernel.wavelength;sample=kernel.medium.sample(wave.wavelength_nm)
        from .._portable import RTOL
        if not np.allclose(sample['absorption_per_m'],bundle['absorption'],rtol=RTOL,atol=0):
            raise ValueError('RTE optics differ')
        wavelength_nm=np.asarray(wave.wavelength_nm,float);weight_nm=np.asarray(wave.weight_nm,float)
        phase_index=np.asarray(sample['phase_index'],float)
        eff=np.asarray(kernel.detector.spectral_weight(wave.wavelength_nm))
        eff=np.broadcast_to(eff,wavelength_nm.shape)
    else:
        wavelength_nm,weight_nm,phase_index,eff=(np.asarray(spectral[key],float) for key in
            ('wavelength_nm','weight_nm','phase_index','efficiency'))
        if not len(wavelength_nm)==len(bundle['speed'])==len(weight_nm)==len(phase_index)==len(eff):
            raise ValueError('spectral arrays do not match the bundle bands')
    omega=bundle['omega'];x,projection=_projection(order);ell=np.arange(order)
    output=np.zeros((len(queries),len(omega)),complex)
    plans=[]
    for q in queries:
        rho,look,beta,clock,lo,hi,density,closest=q
        if not np.isfinite([rho,beta,clock,lo,hi,density,closest]).all() or not 0<beta<=1 or not lo<hi:
            raise ValueError('invalid source integral')
        if allow_gauss:
            from .universal_primary import _direct_ge2_order,_legendre_rule
            edges=np.linspace(lo,hi,max(1,int(np.ceil((hi-lo)/2)))+1);half=np.diff(edges)/2
            gaussian=_direct_ge2_order(float(2*half.max()));gx,gw=_legendre_rule(gaussian)
            uu=(edges[:-1,None]+half[:,None]*(gx+1)).ravel();ww=(half[:,None]*gw).ravel()
            backward=(hi<=0 and clock*np.min(bundle['speed'])>=.5 and hi-lo>=8)
            p=backward_panels(bundle,rho,lo,hi) if backward else None
            use_filon=backward and order*(len(p)-1)<=.65*len(uu)
            plans.append(dict(filon=bool(use_filon),edges=p if use_filon else None,u=uu,w=ww))
        else:plans.append(dict(filon=True,edges=backward_panels(bundle,rho,lo,hi)))
    band_points=0;plane_bytes=0
    harmonics={};source_ell=np.arange(bundle['degree']+1);norm=np.sqrt((2*source_ell+1)/(4*np.pi))
    alpha=kernel.acceptance()['alpha']
    for band,speed in enumerate(bundle['speed']):
        if eff[band]==0 or weight_nm[band]==0:continue
        with wavelength_plane(bundle['coefficients'],band,bounded_mapping=bounded_mapping) as coefficients:
            plane_bytes=max(plane_bytes,coefficients.nbytes)
            local=dict(bundle,coefficients=coefficients,absorption=bundle['absorption'][band:band+1],
                speed=bundle['speed'][band:band+1],phase='unphased')
            for index,(rho,look,beta,clock,lo,hi,density,closest) in enumerate(queries):
                if beta*phase_index[band]<=1:continue
                plan=plans[index]
                if plan['filon']:
                    uedges=plan['edges'];tau=np.hypot(rho,uedges)/speed-clock*uedges
                    half=np.diff(tau)/2;centre=(tau[1:]+tau[:-1])/2
                    nodes=centre[:,None]+half[:,None]*x
                    u=retarded_inverse(nodes,rho,speed,clock)
                    radii=np.hypot(rho,u);jacobian=1/(clock-u/(speed*radii))
                else:u=plan['u']
                key=(band,beta)
                if key not in harmonics:
                    from scipy.special import eval_legendre
                    phase=phase_index[band];wavelength=wavelength_nm[band]
                    coefficient=(1-1/(beta*phase)**2)/wavelength**2
                    harmonics[key]=(norm*eval_legendre(source_ell,1/(beta*phase))*coefficient)[None,:]
                evaluator=cone_point_evaluator(local,impact_m=rho,look=look,beta=beta,
                    alpha=alpha,wavelength_nm=wavelength_nm[band:band+1],
                    wavelength_weight_nm=weight_nm[band:band+1],
                    phase_index=phase_index[band:band+1],efficiency=eff[band:band+1],
                    effective_area_m2=float(kernel.detector.effective_area_m2[0]),backend='fused-looks',
                    source_harmonics=harmonics[key])
                values=evaluator(u.ravel())
                if plan['filon']:
                    values=values.reshape(len(half),order,len(omega))*jacobian[:,:,None]
                    amplitude=np.einsum('kn,pnf->pkf',projection,values)
                    k=half[:,None]*omega[None,:]
                    moments=2*(1j)**ell[None,:,None]*spherical_jn(ell[None,:,None],k[:,None,:])
                    integral=np.einsum('pnf,pnf->pf',amplitude,moments)
                    output[index]+=(half[:,None]*np.exp(1j*centre[:,None]*omega)*integral).sum(0)
                    del amplitude
                else:
                    tau=np.hypot(rho,u)/speed-clock*u
                    output[index]+=(plan['w'][:,None]*values*np.exp(1j*tau[:,None]*omega)).sum(0)
                band_points+=u.size
                del evaluator,values
    for i,q in enumerate(queries):output[i]*=q[6]*np.exp(1j*q[7]*omega)
    return output,dict(filon_order=order,panels=sum(len(p['edges'])-1 for p in plans if p['filon']),
        filon_queries=sum(p['filon'] for p in plans),gauss_queries=sum(not p['filon'] for p in plans),
        spectral_point_evaluations=int(band_points),max_coefficient_plane_bytes=int(plane_bytes),
        bounded_mapping=bool(bounded_mapping),clock_actual=True,beta_actual=True)
