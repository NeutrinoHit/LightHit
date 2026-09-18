"""Open, independent collimation checks. No private optics; no publication.
Run: python scripts/verify_absorption.py --output .build/absorption
"""
from pathlib import Path
import argparse,json,time,sys,platform
import numpy as np
from scipy.special import roots_legendre
from lighthit import Medium,synthetic_medium,SolverSettings
from lighthit.cache import CacheGrid,ResponseCache
from lighthit.experimental.stationary_modes import StationaryModes,collision_components
from lighthit.experimental.shell_mc import shell_estimate,ratio_with_standard_error


def serial(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    raise TypeError(type(x).__name__)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='.build/absorption')
    p.add_argument('--photons-per-batch',type=int,default=4000)
    p.add_argument('--batches',type=int,default=32);args=p.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    m=synthetic_medium();r=np.array([11.,20.,38.,69.,80.,128.,225.,400.,800.])
    models=[StationaryModes.build(m,n) for n in [64,128,256,512]]
    data=[collision_components(r,e) for e in models]
    q,f=data[2]
    ref=q.sum(1)
    convergence={str(e.degree):float(np.max(np.abs(d[0].sum(1)-ref)/ref)) for e,d in zip(models,data)}
    timings=[]
    for _ in range(5):
        start=time.perf_counter();model=StationaryModes.build(m,128);timings.append(time.perf_counter()-start)
    rq=np.linspace(11,400,864);ts=[]
    for _ in range(5):
        start=time.perf_counter();model.scalar_current(rq);ts.append(time.perf_counter()-start)
    # Full Fourier route, low output degree but high scattering degree.
    rm=np.array([11.,20.,40.,80.,128.]);fourier={}
    for km in [4.,6.,8.]:
        settings=SolverSettings(128,1,km,.025,10)
        c=ResponseCache.build(m,settings,CacheGrid(rm,np.array([0.])))
        b=np.exp(-m.extinction_per_m*rm)/(4*np.pi*rm*rm)
        qf=b+c.moments[0,:,0,:].real.sum(1)
        jf=b+c.moments[0,:,1,:].real.sum(1)/3
        qe,je=models[2].scalar_current(rm)
        fourier[str(km)]={'radii':rm,'relative_Q_error':(qf-qe)/qe,
                          'mean_cosine_error':jf/qf-je/qe,'build_s':c.timings_s['total']}
    # Independent positive path Monte Carlo. All reweightings share paths.
    absvals=np.array([.02,.07,.14]);mr=np.array([20.,80.,128.]);width=.5
    start=time.perf_counter()
    batch=shell_estimate(absorptions=absvals,radii_m=mr,widths_m=width,
                         photons_per_batch=args.photons_per_batch,batches=args.batches,
                         max_path_m=1000.,seed=20260917)
    mc_time=time.perf_counter()-start
    mu,se=ratio_with_standard_error(batch);average=batch.mean(0)
    theory=[]
    xx,ww=roots_legendre(4)
    for a in absvals:
        medium=Medium(a,.022,.9,1.36)
        radii=(mr[:,None]+width/2*xx[None,:]).ravel()
        sq,sf=collision_components(radii,StationaryModes.build(medium,256))
        volume_factor=((mr+width/2)**3-(mr-width/2)**3)/3
        weights=width/2*ww[None,:]*(radii.reshape(len(mr),4)**2)/volume_factor[:,None]
        qq=(sq.reshape(len(mr),4,3)*weights[:,:,None]).sum(1)
        ff=(sf.reshape(len(mr),4,3)*weights[:,:,None]).sum(1)
        theory.append((qq,ff))
    thq=np.array([z[0] for z in theory]);thf=np.array([z[1] for z in theory])
    thmu=thf/thq
    # Raw values and seeds retained. Standard errors are empirical, not guarantees.
    np.savez_compressed(out/'mc-batches.npz',batches=batch,absorptions=absvals,radii_m=mr,
                        width_m=width,max_path_m=1000.)
    pole_q,pole_f=models[2].scalar_current(r,leading_only=True)
    report={'base_commit':'3ba25b711408f65ed2511f4abe5fbb0ee0d4457c',
            'medium':{'mu_a':.07,'mu_s':.022,'g':.9,'group_index':1.36,'provenance':'synthetic'},
            'stationary':{'radii_m':r,'Q_by_order':q,'F_by_order':f,'mean_cosine_by_order':f/q,
                          'mean_cosine_total':f.sum(1)/q.sum(1),'relative_Q_error_vs_N256':convergence,
                          'kappa_per_m':models[2].leading_attenuation_per_m,
                          'limit_mean_cosine':models[2].leading_mean_cosine,
                          'leading_mode_fraction':pole_q/q.sum(1)},
            'timings':{'eigen_build_s':timings,'query_864_s':ts,'mc_s_including_possible_JIT':mc_time},
            'real_k_checks':fourier,
            'mc':{'photons_per_batch':args.photons_per_batch,'batches':args.batches,
                  'radii_m':mr,'width_m':width,'absorption_per_m':absvals,'max_path_m':1000.,
                  'mean_cosine':mu,'standard_error':se,'shell_theory_mean_cosine':thmu,
                  'z_multiple':(mu[:,:,2]-thmu[:,:,2])/se[:,:,2],
                  'relative_Q_error':average[:,:,:,0]/thq-1},
            'limitations':['Synthetic HG only; not measured Baikal optics.',
                           'Finite angular eigenproblem: convergence measured, no exact free-tail claim.',
                           'MC shell averages; finite path cutoff and empirical batch errors.',
                           'No independent full time-profile or finite-OM validation.',
                           'Leading mode is an asymptotic proposal, not a certified finite-radius switch.'],
            'environment':{'python':sys.version,'platform':platform.platform(),'numpy':np.__version__}}
    (out/'report.json').write_text(json.dumps(report,default=serial,indent=2))
    print(json.dumps({'eigen_build_median':np.median(timings),'query_864_median':np.median(ts),
                      'mc_s':mc_time,'multiple_z':report['mc']['z_multiple']},default=serial,indent=2))

if __name__=='__main__':main()
