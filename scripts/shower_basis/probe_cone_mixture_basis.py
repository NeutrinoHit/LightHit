#!/usr/bin/env python3
"""Positive physical cone mixture fitted to actual spectral-angular marginals.

This is a source fit, not a held-out predictive model and not transport validation.
Cone integration uses stable log-vMF and scaled Bessel evaluation. No production
code, cache, or defaults are modified.
"""
import paths  # inputs and outputs: scripts/shower_basis/paths.py
import json,time
from pathlib import Path
import numpy as np
from scipy.special import i0e
from scipy.optimize import nnls
from numpy.polynomial.legendre import leggauss

KAPPA=np.array([0.,1.,3.,10.,30.,100.,300.,1000.,3000.,10000.,np.inf])
BETA=np.array([.76,.8,.9,.97,1.])

def basis(wavelength,phase,weight,ng=16):
    x,w=leggauss(ng);edges=np.linspace(-1,1,301)
    mu=(edges[:-1,None]+edges[1:,None])/2+x[None,:]/300
    out=[];labels=[]
    for beta in BETA:
        cosine=1/(beta*phase);spectral=weight*np.maximum(0.,1-cosine**2)/wavelength**2
        if spectral.sum()==0:continue
        spectral/=spectral.sum()
        for k in KAPPA:
            hist=np.zeros((9,300))
            for j,c in enumerate(cosine):
                if c>=1:continue
                if np.isinf(k):hist[j,np.clip(np.searchsorted(edges,c,side='right')-1,0,299)]=1.
                elif k==0:hist[j]=1/300
                else:
                    z=k*np.sqrt(np.maximum(0.,(1-mu*mu)*(1-c*c)))
                    lognorm=np.log(k)-k-np.log1p(-np.exp(-2*k))
                    density=np.exp(lognorm+k*mu*c+z)*i0e(z)
                    hist[j]=(density*w).sum(1)/300
            hist*=spectral[:,None];out.append(hist.ravel());labels.append((float(beta),float(k)))
    result=np.asarray(out)
    return result,np.asarray(labels)

def fit(B,target):
    # Append total mass=1 to L2 objective, then enforce exact yield by normalization.
    G=B@B.T+1.; eig,V=np.linalg.eigh(G); eig=np.maximum(eig,eig.max()*1e-14)
    C=np.sqrt(eig)[:,None]*V.T
    rhs=(target@B.T+1.)@V/np.sqrt(eig)[None,:]
    coef=np.asarray([nnls(C,row,maxiter=2000)[0] for row in rhs]);coef/=coef.sum(1)[:,None]
    return coef,coef@B

def main():
    start=time.process_time()
    import lighthit as lh,ntsim
    from ntsim.LightTransport import bgvd_model_from_ntsim
    model=bgvd_model_from_ntsim(ntsim.bgvd.Telescope(n_clusters=(1,2,3,4,5)),ntsim.bgvd.Water())
    q=lh.WavelengthQuadrature.gauss_legendre(*model.medium.wavelength_range_nm,nodes=9)
    phase=model.medium.sample(q.wavelength_nm)['phase_index']
    B,labels=basis(q.wavelength_nm,phase,q.weight_nm,16)
    B8,_=basis(q.wavelength_nm,phase,q.weight_nm,8)
    assert np.max(abs(B.sum(1)-1))<1e-8
    groups={'10':np.isin(labels[:,0],[.8,1.])&np.isin(labels[:,1],[0,10,100,1000,np.inf]),
            '20':np.isin(labels[:,0],[.8,1.])&np.isin(labels[:,1],KAPPA[1:]),
            '55':np.ones(len(B),bool)}
    inp=paths.work('muon-physics-20261009/secondary/angular300');out=paths.work('muon-physics-20261009/secondary/cone-mixture');out.mkdir(parents=True,exist_ok=True)
    result={'basis':'Positive vMF charged-particle direction about local muon tangent, analytically convolved with spectral Cherenkov cone; effective beta bins.',
            'quadrature_8_vs_16_max_L1':float(np.max(abs(B-B8).sum(1))), 'basis_mass_maxerror':float(np.max(abs(B.sum(1)-1))), 'energies':{}}
    for energy in [300,1000,3000]:
        with np.load(inp/f'muon{energy}-hist.npz') as z:hist=z['hist']
        target=hist.reshape(-1,2700);wt=target.sum(1);valid=wt>0;target=target[valid];wt=wt[valid];target/=wt[:,None]
        cls=np.tile(np.arange(6),hist.shape[0])[valid];r={}
        for name,choice in groups.items():
            coef,pred=fit(B[choice],target);err=abs(pred-target).sum(1)
            r[name]={'basis_size':int(choice.sum()),'weighted_L1':float(err@wt/wt.sum()),'byclass_L1':{str(c):float(err[cls==c]@wt[cls==c]/wt[cls==c].sum()) for c in np.unique(cls)},
                'weighted_active_coefficients':float((coef>1e-5).sum(1)@wt/wt.sum())}
            if name=='55':
                np.savez_compressed(out/f'muon{energy}.npz',coeff=coef,weights=wt,classes=cls,labels=labels,basis=B)
                # Best-fit subset coefficient truncation is a diagnostic, no refitting.
                for count in [2,4,8]:
                    cc=coef.copy();order=np.argsort(cc,axis=1);np.put_along_axis(cc,order[:,:-count],0,axis=1);cc/=cc.sum(1)[:,None]
                    r[name][f'largest_{count}_coefficients_L1']=float((abs(cc@B-target).sum(1)*wt).sum()/wt.sum())
        result['energies'][str(energy)]=r;print(energy,r['55']['weighted_L1'],time.process_time()-start,flush=True)
    result['cpu_s_including_imports']=time.process_time()-start
    (out/'report.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
