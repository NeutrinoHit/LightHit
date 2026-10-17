"""Independent stationary local check of a causal three-spatial-moment ansatz.

Research only. This is NOT the marginal of DN's seven-parameter joint maxent.
Fit stable collision-conditioned moments inside the causal ball at each age;
then integrate absorption and the >=2 Poisson probability over path length.
Compare volume-integrated cells with independent absorption-endpoint MC.

Run with OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 using ntsim_env Python.
"""
import paths  # inputs and outputs: scripts/shower_basis/paths.py
import argparse
import json
import math
from pathlib import Path
from time import process_time, perf_counter

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize
from scipy.special import gammainc, logsumexp

def conditional_spatial(n, g):
    """(E z, E z², E rho²) for y=x/(v t), given exactly n HG collisions."""
    powers = np.power(float(g), np.arange(n+1))
    geometric = np.cumsum(powers)
    szz = (1.+2.*np.power(float(g*g), np.arange(n+1)))/3.
    sperp = 2.*(1.-np.power(float(g*g), np.arange(n+1)))/3.
    factor = 2./((n+1)*(n+2))
    return np.array([geometric[-1]/(n+1),
                     factor*np.dot(szz, geometric[::-1]),
                     factor*np.dot(sperp, geometric[::-1])])


def residual_spatial(t, g=.9, tolerance=2e-16):
    """Positive conditional Poisson mixture, scaled so its n=2 weight is one."""
    eta = t/(1.-g)
    if not (0 <= t and 0 <= g < 1) or eta > 100:
        raise ValueError("prototype supports T>=0, 0<=g<1 and eta<=100")
    terms, targets = [], []
    term = 1.
    for n in range(2, 1000):
        terms.append(term)
        targets.append(conditional_spatial(n, g))
        total = math.fsum(terms)
        next_term = term*eta/(n+1)
        ratio_bound = eta/(n+2)
        if ratio_bound < 1:
            tail = next_term/(1-ratio_bound)
            if tail <= tolerance*total:
                break
        term = next_term
    else:
        raise RuntimeError("conditional collision-count sum did not converge")
    target = np.array([math.fsum(w*float(row[k]) for w,row in zip(terms,targets))/total for k in range(3)])
    mass = math.exp(-eta)*eta*eta*.5*total
    return target, dict(eta=eta, mass_ge2=mass, last_collision_order=n,
                        conditional_tail_error_bound=2*tail/total)




def quadrature(order):
    """Normalized uniform volume on the unit ball, azimuth integrated exactly."""
    z,wz = leggauss(order)
    a,wa = leggauss(order)
    a,wa = (a+1)/2,wa/2
    zz,aa = np.meshgrid(z,a,indexing='ij')
    perpendicular = (1-zz*zz)*aa
    features = np.stack((zz,zz*zz,perpendicular),axis=-1).reshape(-1,3)
    weights = (.75*wz[:,None]*(1-z[:,None]**2)*wa[None,:]).ravel()
    assert abs(weights.sum()-1.) < 2e-14
    return features,np.log(weights)


def statistics(parameters, features, logweights):
    value = logweights+features@parameters
    logz = logsumexp(value)
    probabilities = np.exp(value-logz)
    moments = probabilities@features
    centred = features-moments
    covariance = centred.T@(probabilities[:,None]*centred)
    return float(logz),moments,covariance


def fit(target, features, logweights, start=None):
    scale = np.array([.4,.3,.3])
    shifted = (features-target)/scale
    initial = np.zeros(3) if start is None else np.asarray(start)*scale
    memo = {}

    def terms(parameters):
        key = parameters.tobytes()
        if memo.get('key') != key:
            memo['key'] = key
            memo['value'] = statistics(parameters,shifted,logweights)
        return memo['value']

    result = minimize(lambda x:terms(x)[0], initial,
                      jac=lambda x:terms(x)[1], hess=lambda x:terms(x)[2],
                      method='trust-exact',options=dict(gtol=3e-12,maxiter=150))
    parameters = result.x/scale
    # Near the optimum the dual objective changes below one ulp before its
    # gradient vanishes. Refine the moment equations directly with their exact
    # covariance Jacobian; accept steps only when the residual norm decreases.
    refinements = 0
    for _ in range(12):
        _,actual,covariance = statistics(parameters,features,logweights)
        residual = actual-target
        if np.max(abs(residual)) < 2e-13:
            break
        step = np.linalg.solve(covariance,residual)
        old_norm = np.linalg.norm(residual)
        for backtrack in range(12):
            proposed = parameters-step*2.**(-backtrack)
            _,trial,_ = statistics(proposed,features,logweights)
            if np.linalg.norm(trial-target) < old_norm:
                parameters = proposed
                refinements += 1
                break
        else:
            break
    logz,actual,covariance = statistics(parameters,features,logweights)
    return parameters,logz,dict(iterations=int(result.nit),optimizer_success=bool(result.success),
                                optimizer_message=str(result.message),
                                residual_newton_refinements=refinements,
                                moment_error=float(np.max(abs(actual-target))),
                                hessian_condition_number=float(np.linalg.cond(covariance)))



def make_parameter_table(order=48, size=181, eta_max=20.):
    # Dense near eta=0, but no singular residual subtraction occurs there.
    eta = np.expm1(np.linspace(0., np.log1p(eta_max), size))
    features, lw = quadrature(order)
    rows, errors = [], []
    warm = None
    for e in eta:
        target, _ = residual_spatial(e * .1, .9)
        par, logz, info = fit(target, features, lw, warm)
        if info['moment_error'] > 2e-11:
            raise RuntimeError(info)
        rows.append(np.r_[par, logz]); errors.append(info['moment_error'])
        warm = par
    spline = CubicSpline(eta, np.asarray(rows), axis=0)
    # Independent mass/moment checks at interleaved ages and a finer volume rule.
    features2, lw2 = quadrature(80)
    checks = []
    for e in (eta[:-1]+eta[1:])[::9]/2:
        target, _ = residual_spatial(e*.1, .9)
        par, logz = spline(e)[:3], spline(e)[3]
        z2, moments, _ = statistics(par, features2, lw2)
        checks.append(max(abs(np.expm1(z2-logz)), np.max(abs(moments-target))))
    return spline, dict(knots=size, fit_max_moment_error=float(max(errors)),
                        interpolated_mass_or_moment_error=float(max(checks)),
                        eta_max=eta_max, fit_quadrature_order=order)


def integrate_cells(redges, medges, spline, spatial_order, age_order, extra_length=600.):
    a,s=.0545,.0151
    x,w=leggauss(spatial_order)
    r=(redges[1:,None]+redges[:-1,None])/2 + np.diff(redges)[:,None]*x/2
    wr=np.diff(redges)[:,None]*w/2
    mu=(medges[1:,None]+medges[:-1,None])/2 + np.diff(medges)[:,None]*x/2
    wm=np.diff(medges)[:,None]*w/2
    age_x,age_w=leggauss(age_order)
    out=np.zeros((len(redges)-1,len(medges)-1))
    # Blocks by radial bin keep peak temporary storage far below 100 MB.
    for ir in range(len(redges)-1):
        rr=r[ir,:,None]
        vmax=np.log1p(extra_length/rr)
        v=(age_x[None,:]+1)*vmax/2
        lv=rr*np.exp(v)
        weight=age_w[None,:]*vmax/2
        eta=s*lv
        if eta.max() > spline.x[-1]:
            raise ValueError('age interpolation domain exceeded')
        parameters=spline(eta)
        unit_r=np.exp(-v)
        zz=unit_r[None,None,:,:]*mu[:,:,None,None]
        perp=unit_r[None,None,:,:]**2 * (1-mu[:,:,None,None]**2)
        exponent=(parameters[None,None,:,:,0]*zz
                  +parameters[None,None,:,:,1]*zz**2
                  +parameters[None,None,:,:,2]*perp
                  -parameters[None,None,:,:,3])
        # dl/l^3 = dv/l^2; W(eta)=gammainc(2,eta).
        scalar=(3/(4*np.pi)*np.exp(-a*lv)*gammainc(2,eta)/lv**2*weight)
        density=np.sum(np.exp(exponent)*scalar[None,None,:,:],axis=-1)
        out[ir]=np.sum(density * wm[:,:,None] * (2*np.pi*r[ir]**2*wr[ir])[None,None,:],axis=(1,2))
    return out


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--output',default=str(paths.work('positive-map-20261009/moments')))
    p.add_argument('--mc',default=str(paths.work('green-map-20261009/stationary_mc/histogram.npz')))
    args=p.parse_args()
    cpu,wall=process_time(),perf_counter()
    data=np.load(args.mc)
    redges,medges=data['r_edges_m'],data['mu_edges']
    observed=data['F_ge2']*data['volume_m3']
    se=data['F_standard_error']*data['volume_m3']
    counts=data['counts']
    mass=(.0151/(.0545+.0151))**2/.0545
    spline,fit_info=make_parameter_table()
    rules=[(4,64),(8,128),(8,192)]
    models={}
    timings={}
    for spatial,age in rules:
        begin=process_time()
        models[f'{spatial}x{age}']=integrate_cells(redges,medges,spline,spatial,age)
        timings[f'{spatial}x{age}']=process_time()-begin
    model=models['8x192']
    assert fit_info['interpolated_mass_or_moment_error'] < 1e-7
    assert np.all(model >= 0)
    assert abs(models['8x128']-model).sum()/mass < 1e-7
    eligible=counts>=25
    absolute=abs(model-observed)
    rel=absolute[eligible]/observed[eligible]
    sigma=absolute[eligible]/se[eligible]
    forward=medges[:-1]>=.96875-1e-12
    backward=medges[1:]<=0
    result=dict(model='three-spatial-moment causal maxent integrated over age',
        source_photons=1.,mu_a=.0545,mu_s=.0151,g=.9,
        exact_full_space_mass_m=mass,model_map_mass_m=float(model.sum()),
        mc_map_mass_m=float(observed.sum()),mass_l1_error_over_exact_total=float(absolute.sum()/mass),
        negative_cells=int((model<0).sum()),eligible_cells=int(eligible.sum()),
        median_relative_error_eligible=float(np.median(rel)),p95_relative_error_eligible=float(np.quantile(rel,.95)),
        fraction_above_3sigma_eligible=float(np.mean(sigma>3)),
        fraction_above_5sigma_eligible=float(np.mean(sigma>5)),
        quadrature_l1_over_full_mass={key:float(abs(value-model).sum()/mass) for key,value in models.items()},
        forward_cap=dict(model=float(model[:,forward].sum()/mass),mc=float(observed[:,forward].sum()/mass)),
        backward_hemisphere=dict(model=float(model[:,backward].sum()/mass),mc=float(observed[:,backward].sum()/mass)),
        parameter_table=fit_info,quadrature_cpu_seconds=timings,
        cpu_seconds=process_time()-cpu,wall_seconds=perf_counter()-wall,
        age_truncation_extra_length_m=600.,
        omitted_global_fluence_upper_bound_m=math.exp(-.0545*600)/.0545,
        scope='spatial scalar density only; not joint position-arrival maxent and not OM response')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    (out/'causal-vs-mc.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(out/'causal-vs-mc.npz',r_edges_m=redges,mu_edges=medges,counts=counts,
                       model_mass_m=model,mc_mass_m=observed,standard_error_mass_m=se,
                       quadrature_8x128_mass_m=models['8x128'],quadrature_4x64_mass_m=models['4x64'])
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    main()
