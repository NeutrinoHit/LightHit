#!/usr/bin/env python3
"""Primary-only geometry and exact ballistic atom comparison, single CPU.

Fit endpoints and emission chronology; preserve C0/C2. This is a physics
probe, not a production compression policy. Full directional acceptance,
all wavelength nodes and every receiver are retained.
"""
import paths  # inputs and outputs: scripts/shower_basis/paths.py
import argparse,json
from pathlib import Path
from time import perf_counter
import numpy as np
from numba import njit
from scipy.stats import wasserstein_distance
from lighthit.experimental.ballistic_fast import _acceptance
from lighthit.experimental.ballistic_spectral import _cone_bounds

@njit(cache=True)
def roots(pos,look,area,x,u,L,c0,c2,beta,t0,t1,phase,mut,speed,s0w,s2w,alpha,lower,upper):
    ri=[];ti=[];wi=[];ei=[];li=[]
    for j in range(len(pos)):
        for i in range(len(x)):
            d=pos[j]-x[i];along=np.dot(d,u[i])
            if along<0:continue
            b2=max(np.dot(d,d)-along*along,1e-300)
            if along*along < b2*lower[i]**2-1e-8:continue
            if along>L[i] and (along-L[i])**2>b2*upper[i]**2+1e-8:continue
            b=np.sqrt(b2)
            for k in range(len(phase)):
                if beta[i]*phase[k]<=1:continue
                mu=1/(beta[i]*phase[k]);sn=np.sqrt(1-mu*mu)
                a=along-b*mu/sn
                if a<0 or a>=L[i]:continue
                dist=b/sn;direction=(d-a*u[i])/dist
                acc=_acceptance(alpha,max(-1.,min(1.,np.dot(direction,look[j]))))
                weight=(s0w[k]*c0[i]+s2w[k]*c2[i])/L[i]*np.exp(-mut[k]*dist)/(2*np.pi*b*sn)*acc*area[j]
                if weight<=0:continue
                ri.append(j);ti.append(t0[i]+a/L[i]*(t1[i]-t0[i])+dist/speed[k]);wi.append(weight);ei.append(i);li.append(k)
    return np.asarray(ri),np.asarray(ti),np.asarray(wi),np.asarray(ei),np.asarray(li)

def atoms(kernel,source):
    from lighthit.prompt import _spectral_tables
    t=_spectral_tables(kernel);lower,upper=_cone_bounds(source.beta,t['phase_index'])
    return roots(kernel.detector.positions_m,-kernel.detector.orientations,kernel.detector.effective_area_m2,
        source.start_m,source.direction,source.length_m,source.coefficient0,source.coefficient2,source.beta,
        source.start_ns,source.end_ns,t['phase_index'],t['mut'],t['speed'],t['s0w'],t['s2w'],
        kernel.acceptance()['alpha'],lower,upper)

def fit_pieces(source,n,centroid=False,edges=None):
    from lighthit import SpectralLightElements
    arclength=np.r_[0,np.cumsum(source.length_m)]
    cuts=np.unique(np.r_[0,np.searchsorted(arclength,np.linspace(0,arclength[-1],n+1)[1:-1]),len(source)]) if edges is None else edges
    ends=source.start_m+source.length_m[:,None]*source.direction
    entries=[];details=[]
    for lo,hi in zip(cuts[:-1],cuts[1:]):
        c0=source.coefficient0[lo:hi].sum();c2=source.coefficient2[lo:hi].sum()
        start=source.start_m[lo].copy();end=ends[hi-1];u=end-start;length=np.linalg.norm(u);u/=length
        if centroid:
            centre=np.average(source.midpoints_m[lo:hi],axis=0,weights=source.coefficient0[lo:hi]);shift=centre-start-u*np.dot(centre-start,u);start+=shift
        displacement=source.midpoints_m[lo:hi]-start;proj=displacement@u
        transverse=np.linalg.norm(displacement-proj[:,None]*u,axis=1)
        angles=np.arccos(np.clip(source.direction[lo:hi]@u,-1,1))
        clock=source.start_ns[lo]+proj/length*(source.end_ns[hi-1]-source.start_ns[lo])
        timeerr=(source.start_ns[lo:hi]+source.end_ns[lo:hi])/2-clock
        details.append(dict(lo=int(lo),hi=int(hi),length_m=float(length),max_transverse_m=float(transverse.max()),
            rms_transverse_m=float(np.sqrt(np.average(transverse**2,weights=source.coefficient0[lo:hi]))),
            rms_angle_mrad=float(1000*np.sqrt(np.average(angles**2,weights=source.coefficient0[lo:hi]))),max_clock_error_ns=float(np.max(np.abs(timeerr)))))
        entries.append((start,u,length,c0,c2,np.sqrt(-c0/c2),source.start_ns[lo],source.end_ns[hi-1],source.row_index[lo],source.uid[lo]))
    cols=list(zip(*entries))
    return SpectralLightElements(**{k:np.asarray(v) for k,v in zip(('start_m','direction','length_m','coefficient0','coefficient2','beta','start_ns','end_ns','row_index','uid'),cols)},cone_model='spectral'),details

def compare(reference,candidate,nrecv):
    r,t,w,_,_=reference;rr,tt,ww,_,_=candidate
    q=np.bincount(r,weights=w,minlength=nrecv);qc=np.bincount(rr,weights=ww,minlength=nrecv)
    active=q>=.01;both=active&(qc>0)
    per=[]
    for j in np.flatnonzero(active|(qc>=.01)):
        a=r==j;b=rr==j
        if np.any(a) and np.any(b):
            origin=np.floor(np.min(t[a])/5)*5
            bin_errors=[]
            for phase in np.arange(0.,5.,.5):
                edges=origin+phase+np.arange(-100,1005,5)
                hist=np.histogram(t[a],edges,weights=w[a])[0];hc=np.histogram(tt[b],edges,weights=ww[b])[0]
                bin_errors.append(float(np.sum(np.abs(hc-hist))/max(q[j],1e-300)))
            def quantiles(times,weights):
                order=np.argsort(times)
                return np.interp([.05,.5,.95],np.cumsum(weights[order])/weights.sum(),times[order])
            quantile_shift=quantiles(tt[b],ww[b])-quantiles(t[a],w[a])
            per.append(dict(om=int(j),q=float(q[j]),candidate_q=float(qc[j]),relative_charge=float((qc[j]-q[j])/max(q[j],1e-300)),
                mean_shift_ns=float(np.average(tt[b],weights=ww[b])-np.average(t[a],weights=w[a])),
                wasserstein_ns=float(wasserstein_distance(t[a],tt[b],w[a],ww[b])),
                bin_L1=bin_errors[0],bin_L1_phase_min=min(bin_errors),bin_L1_phase_max=max(bin_errors),quantile_shifts_ns=quantile_shift.tolist()))
    return dict(total_charge=float(qc.sum()),active=int(active.sum()),lost=int(np.sum(active&(qc<.01))),gained=int(np.sum(~active&(qc>=.01))),
        active_charge_L1=float(np.sum(np.abs(qc[active]-q[active]))/np.sum(q[active])),
        max_active_relative_charge=float(np.max(np.abs(qc[active]-q[active])/q[active])),
        max_weak_abs_pe=float(np.max(np.abs(qc[~active]-q[~active]))),
        max_wasserstein_ns=max([p['wasserstein_ns'] for p in per if p['q']>=.01],default=0),
        max_mean_shift_ns=max([abs(p['mean_shift_ns']) for p in per if p['q']>=.01],default=0),per_om=per)

def main():
    p=argparse.ArgumentParser();p.add_argument('--energy',type=int,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--interval-spectrum',action='store_true');args=p.parse_args()
    import ntsim,lighthit as lh
    from ntsim.LightTransport import bgvd_model_from_ntsim
    from probe_positive_map_screen import read_source
    source,provenance,_=read_source(args.energy);source=source.subset(source.uid==0)
    model=bgvd_model_from_ntsim(ntsim.bgvd.Telescope(n_clusters=(1,2,3,4,5)),ntsim.bgvd.Water())
    kernel=model.kernel(lh.KernelConfig(omega_per_ns=np.linspace(0,1.2,65),cache_policy='require',cache_directory=str(paths.RTE_CACHE.path())))
    atoms(kernel,source.subset(np.arange(len(source))==0))
    start=perf_counter();reference=atoms(kernel,source);elapsed=perf_counter()-start
    from lighthit.experimental.ballistic_spectral import spectral_ballistic_charge
    q=np.bincount(reference[0],weights=reference[2],minlength=1440)
    qc=spectral_ballistic_charge(kernel,source)
    assert np.allclose(q,qc,rtol=2e-12,atol=1e-14),(q-qc).max()
    report=dict(energy_GeV=args.energy,n_steps=len(source),length_m=float(source.length_m.sum()),original_seconds=elapsed,
        original_total_q=float(q.sum()),original_active=int(np.sum(q>=.01)),atom_vs_production_max_pe=float(np.max(np.abs(q-qc))),cases=[])
    args.output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output/'reference.npz',receiver=reference[0],arrival_ns=reference[1],weight=reference[2],element=reference[3],wavelength=reference[4])
    for centre in (False,True):
        for n in (1,2,4,8,16,32,64,128,256):
            fitted,details=fit_pieces(source,n,centre)
            start=perf_counter();candidate=atoms(kernel,fitted);elapsed=perf_counter()-start
            out=compare(reference,candidate,1440);out.update(n=len(fitted),centroid=centre,seconds=elapsed,
                geometry=details,c0_relative_error=float(fitted.coefficient0.sum()/source.coefficient0.sum()-1),
                c2_relative_error=float(fitted.coefficient2.sum()/source.coefficient2.sum()-1))
            report['cases'].append(out)
            print(args.energy,centre,n,'qL1',out['active_charge_L1'],'W1',out['max_wasserstein_ns'],'lost',out['lost'],flush=True)
    (args.output/'report.json').write_text(json.dumps(report,indent=2))
    if args.interval_spectrum:
        audit=spectrum_audit(model,source,reference)
        (args.output/'interval-spectrum.json').write_text(json.dumps(audit,indent=2))



def interval_atoms(kernel, source, order=16):
    """Independent positive ballistic quadrature split at cone/step crossings.

    Restricted to monotonically decreasing piecewise-linear phase index.
    Every emitting wavelength interval is split at medium interpolation knots;
    detector response is sampled directly. Compare quadrature orders for its
    unlisted interpolation knots. No global wavelength-node aliasing remains.
    """
    from lighthit.ballistic import acceptance_from_coefficients
    medium=kernel.medium;detector=kernel.detector
    assert np.all(np.diff(medium.phase_index)<0)
    low,high=medium.wavelength_range_nm
    gx,gw=np.polynomial.legendre.leggauss(order)
    output=[[],[],[],[],[]]
    for j,pos in enumerate(detector.positions_m):
        d=pos-source.start_m;along=np.einsum('ij,ij->i',d,source.direction)
        impact=np.sqrt(np.maximum(np.einsum('ij,ij->i',d,d)-along**2,1e-300))
        nlo=np.full(len(source),np.inf);nhi=np.full(len(source),np.inf)
        forward=along>0;nlo[forward]=np.sqrt(1+(impact[forward]/along[forward])**2)/source.beta[forward]
        before=along>source.length_m;nhi[before]=np.sqrt(1+(impact[before]/(along[before]-source.length_m[before]))**2)/source.beta[before]
        valid=forward&(nlo<medium.phase_index[0])&(nhi>medium.phase_index[-1])
        for i in np.flatnonzero(valid):
            a=np.interp(nhi[i],medium.phase_index[::-1],medium.wavelength_nm[::-1])
            b=np.interp(nlo[i],medium.phase_index[::-1],medium.wavelength_nm[::-1])
            if b<=a:continue
            cuts=np.r_[a,medium.wavelength_nm[(medium.wavelength_nm>a)&(medium.wavelength_nm<b)],b]
            half=np.diff(cuts)/2;lam=((cuts[:-1]+half)[:,None]+half[:,None]*gx).ravel()
            quad=(half[:,None]*gw).ravel();optics=medium.sample(lam)
            mu=1/(source.beta[i]*optics['phase_index']);sine=np.sqrt(1-mu*mu)
            root=along[i]-impact[i]*mu/sine;distance=impact[i]/sine
            direction=(d[i]-root[:,None]*source.direction[i])/distance[:,None]
            acc=acceptance_from_coefficients(kernel.acceptance()['alpha'],np.clip(direction@(-detector.orientations[j]),-1,1))
            photons=(source.coefficient0[i]+source.coefficient2[i]/optics['phase_index']**2)/lam**2
            weight=quad*detector.spectral_weight(lam)*detector.effective_area_m2[j]*photons/source.length_m[i]*np.exp(-(optics['absorption_per_m']+optics['scattering_per_m'])*distance)/(2*np.pi*impact[i]*sine)*acc
            time=source.start_ns[i]+root/source.length_m[i]*(source.end_ns[i]-source.start_ns[i])+distance*optics['group_index']/.299792458
            good=weight>0
            for bucket,value in zip(output,(np.full(good.sum(),j),time[good],weight[good],np.full(good.sum(),i),lam[good])):bucket.extend(value)
    return tuple(np.asarray(v,dtype=int if k in(0,3) else float) for k,v in enumerate(output))


def spectrum_audit(model,source,reference):
    """Converged local spectral control on all OMs with nine-node Q0>=.005 PE."""
    from dataclasses import replace
    import lighthit as lh
    q=np.bincount(reference[0],weights=reference[2],minlength=len(model.detector))
    ids=np.flatnonzero(q>=.005);d=model.detector
    d=replace(d,positions_m=d.positions_m[ids],orientations=d.orientations[ids],effective_area_m2=d.effective_area_m2[ids],identifiers={k:v[ids] for k,v in d.identifiers.items()})
    kernel=replace(model,detector=d).kernel(lh.KernelConfig(omega_per_ns=np.array([0.]),cache_policy='require'))
    convergence=[]
    for order in (8,16,32):
        exact=interval_atoms(kernel,source,order)
        convergence.append(dict(order=order,q=np.bincount(exact[0],weights=exact[2],minlength=len(ids)).tolist()))
    result=dict(ids=ids.tolist(),convergence=convergence,nine_node_original=compare(exact,atoms(kernel,source),len(ids)),cases=[])
    for count in (1,2,4,8,16,32,64,128,256):
        fitted,_=fit_pieces(source,count)
        item=compare(exact,interval_atoms(kernel,fitted,32),len(ids));item['n']=count;result['cases'].append(item)
    return result


if __name__=='__main__':main()
