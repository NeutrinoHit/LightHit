#!/usr/bin/env python3
"""Source-only physical diagnostics; exact Cherenkov-ring angular marginals.

No propagation is performed. Cross-energy template errors are source L1 errors,
not detector errors. Each event is seed 42; three energies are not an ensemble.
"""
from __future__ import annotations
import paths  # inputs and outputs: scripts/shower_basis/paths.py
import argparse,json,hashlib,time
from pathlib import Path
import numpy as np
from numba import njit
from numpy.polynomial.legendre import legvander
from analyze_muon_secondaries import quantiles
C=.299792458
NAMES=['root<10MeV','root10-100MeV','root0.1-1GeV','root>=1GeV','nuclear-family','delayed>10ns']
RADIUS=np.array([0.,.001,.003,.01,.03,.1,.3,1.,3.,10.,np.inf])
DELAY=np.array([-np.inf,-1.,0.,.1,.5,1.,2.,5.,10.,100.,np.inf])

@njit(cache=True)
def histograms(cosine,beta,spectral,phase,groups,delaybins,radiibins,ncell,nbins=40):
    # Exact ring azimuth CDF, integrated per equal cos(theta) bin.
    out=np.zeros((ncell,6,9,nbins)); joint=np.zeros((6,10,nbins)); radial=np.zeros((6,10,nbins))
    for i in range(len(beta)):
        cell=groups[i,0]; cat=groups[i,1]
        u=max(-1.,min(1.,cosine[i]))
        for j in range(9):
            w=spectral[i,j]
            if w<=0: continue
            v=min(1.,1/(beta[i]*phase[j])); a=u*v
            b=np.sqrt(max(0.,(1-u*u)*(1-v*v)))
            prev=0.
            for k in range(nbins):
                edge=-1.+(k+1)*2/nbins
                if b<1e-15: val=1. if edge>=a else 0.
                else:
                    x=(edge-a)/b
                    val=0. if x<=-1 else (1. if x>=1 else 1.-np.arccos(x)/np.pi)
                q=w*max(0.,val-prev); prev=val
                out[cell,cat,j,k]+=q; joint[cat,delaybins[i],k]+=q; radial[cat,radiibins[i],k]+=q
    return out,joint,radial

def wq(x,w,p=(.5,.9,.99)):
    return quantiles(x,w,p) if w.sum()>0 else []

def event(energy,model,quad,output,nbins):
    import lighthit as lh
    path=paths.muon_source(energy)
    ap=paths.muon_ancestry(energy)
    with np.load(path) as z:
        source=lh.SpectralLightElements(**{k:z[k] for k in z.files if k!='metadata'},cone_model='spectral')
    with np.load(ap) as z: anc={k:z[k] for k in z.files if k!='metadata'}
    phase=model.medium.sample(quad.wavelength_nm)['phase_index']
    spec=source.photon_density_per_nm(quad.wavelength_nm,phase)*quad.weight_nm
    spec[source.beta[:,None]*phase<=1]=0.; weight=spec.sum(1)
    primary=source.uid==0; mask=(~primary)&(weight>0)
    mid=source.midpoints_m; t=(source.start_ns+source.end_ns)/2
    # Ordered primary coordinate and nearest local direction by projected coordinate.
    origin=source.start_m[primary][0]; end=source.start_m[primary][-1]+source.length_m[primary][-1]*source.direction[primary][-1]
    axis=(end-origin)/np.linalg.norm(end-origin); s=(mid-origin)@axis
    porder=np.argsort(s[primary]); ps=s[primary][porder]; pm=mid[primary][porder]; pt=t[primary][porder]
    local=np.column_stack([np.interp(s,ps,pm[:,k]) for k in range(3)])
    pi=np.clip(np.searchsorted(ps,s),0,len(ps)-1)
    prior=np.maximum(pi-1,0); pi=np.where(abs(ps[pi]-s)<abs(ps[prior]-s),pi,prior)
    localaxis=source.direction[primary][porder][pi]
    rho=np.linalg.norm(mid-local,axis=1); delay=t-np.interp(s,ps,pt)
    roots=np.searchsorted(anc['uid'],anc['element_family'][mask])
    assert np.all(anc['uid'][roots]==anc['element_family'][mask])
    rootE=anc['energy_GeV'][roots]; process=anc['creator'][roots]
    cat=np.searchsorted([.01,.1,1.],rootE,side='right')
    cat[process=='muonNuclea']=4; cat[delay[mask]>10.]=5
    cells=np.maximum(0,np.floor(s[mask]/20).astype(int)); ncell=cells.max()+1
    cosine=np.sum(source.direction[mask]*localaxis[mask],axis=1)
    h,joint,radial=histograms(cosine,source.beta[mask],spec[mask],phase,np.column_stack([cells,cat]),np.clip(np.searchsorted(DELAY,delay[mask],side='right')-1,0,9),np.clip(np.searchsorted(RADIUS,rho[mask],side='right')-1,0,9),ncell,nbins)
    np.testing.assert_allclose(h.sum(),weight[mask].sum(),rtol=2e-12)
    q=weight[mask]; total=q.sum()
    # Local frame about primary tangent; exact photon transverse first/second moments.
    ax=localaxis[mask]; ex=np.cross(ax,np.tile([0.,0.,1.],(len(ax),1))); ex/=np.linalg.norm(ex,axis=1)[:,None]; ey=np.cross(ax,ex)
    dx=np.sum(source.direction[mask]*ex,axis=1); dy=np.sum(source.direction[mask]*ey,axis=1)
    cone=np.minimum(1.,1/(source.beta[mask,None]*phase)); c1=(spec[mask]*cone).sum(1); c2=(spec[mask]*(1.5*cone**2-.5)).sum(1)
    transverse1=c1*(dx+1j*dy); transverse2=c2*(dx+1j*dy)**2
    displacement=mid[mask]-local[mask]; posphi=np.arctan2(np.sum(displacement*ey,axis=1),np.sum(displacement*ex,axis=1))
    fam=anc['element_family'][mask]; unique,inv=np.unique(fam,return_inverse=True)
    fq=np.bincount(inv,weights=q); fdir=np.column_stack([np.bincount(inv,weights=q*source.direction[mask,k]) for k in range(3)])
    fdir/=np.maximum(np.linalg.norm(fdir,axis=1)[:,None],1e-30)
    fcos=np.sum(source.direction[mask]*fdir[inv],axis=1)
    def moments(choose,cos):
        raw=legvander(cos[choose],8); acc=np.zeros(9)
        for j in range(9):
            ring=legvander(np.minimum(1.,1/(source.beta[mask][choose]*phase[j])),8)
            acc+=(raw*ring*spec[mask][choose,j,None]).sum(0)
        return (acc/q[choose].sum()).tolist()
    result={'energy_GeV':energy,'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
      'elements':len(source),'secondary_elements':int(mask.sum()),'primary_photons':float(weight[primary].sum()),'secondary_photons':float(total),
      'secondary_fraction':float(total/weight.sum()),'length_m':float(source.length_m[primary].sum()),'category':{},
      'charged_pdg_fraction':{str(v):float(q[anc['element_pdgid'][mask]==v].sum()/total) for v in np.unique(anc['element_pdgid'][mask])},
      'after_last_OM_y_fraction':float(q[mid[mask,1]>model.detector.positions_m[:,1].max()].sum()/total),
      'last1000m_fraction':float(q[s[mask]>s[primary].max()-1000].sum()/total)}
    for c,name in enumerate(NAMES):
        choose=cat==c
        if not choose.any():continue
        theta=h[:,c].sum((0,1)); a=joint[c]/q[choose].sum(); independent=a.sum(1)[:,None]*a.sum(0)[None,:]
        ar=radial[c]/q[choose].sum(); rind=ar.sum(1)[:,None]*ar.sum(0)[None,:]
        def local_anisotropy(values):
            re=np.bincount(cells[choose],weights=values[choose].real,minlength=ncell)
            im=np.bincount(cells[choose],weights=values[choose].imag,minlength=ncell)
            return float(np.sqrt(re*re+im*im).sum()/q[choose].sum())
        result['category'][name]={'fraction':float(q[choose].sum()/total),'elements':int(choose.sum()),'families':len(np.unique(fam[choose])),
         'rho_m_quantiles':wq(rho[mask][choose],q[choose]),'delay_ns_quantiles':wq(delay[mask][choose],q[choose]),
         'local_muon_axis_moments':moments(choose,cosine),'family_axis_moments':moments(choose,fcos),
         'angular_isotropic_L1':float(np.abs(theta/theta.sum()-1/nbins).sum()),'angle_delay_factorization_L1':float(abs(a-independent).sum()),'angle_radius_factorization_L1':float(abs(ar-rind).sum()),
         'local20m_photon_transverse_dipole':local_anisotropy(transverse1),'local20m_photon_transverse_quadrupole':local_anisotropy(transverse2),
         'local20m_position_azimuth_m1':local_anisotropy(q*np.exp(1j*posphi))}
    # Longitudinal brightness and angular variation retain exact local total yield.
    longitudinal=h.sum((1,2,3)); rank=np.argsort(longitudinal)[::-1]
    result['longitudinal_20m']={'populated_cells':int((longitudinal>0).sum()),'brightest10pct_light_fraction':float(longitudinal[rank[:max(1,int(.1*(longitudinal>0).sum()))]].sum()/total),
        's_centers_m':((np.arange(ncell)+.5)*20).tolist(),'photons':longitudinal.tolist()}
    cellclass=h.sum((2,3)); result['longitudinal_20m']['components']={}
    for name,x in [('soft_root_below_100MeV',cellclass[:,:2].sum(1)),('EM_root_above_1GeV',cellclass[:,3])]:
        x=x[1:-1]  # Exclude incomplete endpoint cells from stationarity diagnostics.
        result['longitudinal_20m']['components'][name]={'CV':float(x.std()/x.mean()),'brightest10pct_fraction':float(np.sort(x)[-max(1,len(x)//10):].sum()/x.sum())}
    np.savez_compressed(output/f'muon{energy}-hist.npz',hist=h,joint=joint,radial=radial,radius_edges_m=RADIUS,delay_edges_ns=DELAY,cos_edges=np.linspace(-1,1,nbins+1))
    (output/f'muon{energy}.json').write_text(json.dumps(result,indent=2));return h,joint,result

def evaluate(data):
    results={}
    for held in data:
        train=[data[e][0] for e in data if e!=held]; h=data[held][0]; total=h.sum();nbins=h.shape[-1]
        # One and six physical templates, retaining actual yield per cell/category/lambda.
        accum=sum(x.sum(0) for x in train)
        template=np.divide(accum,accum.sum(-1,keepdims=True),out=np.zeros_like(accum),where=accum.sum(-1,keepdims=True)>0)
        pred=h.sum(-1,keepdims=True)*template[None]
        one=accum.sum(0);one/=one.sum(-1,keepdims=True)
        p1=h.sum((1,3))[:,:,None]*one[None]
        truth=h.sum(1)
        byclass={NAMES[c]:float(abs(pred[:,c]-h[:,c]).sum()/h[:,c].sum()) for c in range(6) if h[:,c].sum()>0}
        # POD correction on per-cell/category/spectral normalized shape; weighted row SVD.
        rows=np.concatenate([x.reshape(-1,nbins) for x in train]); rw=rows.sum(1);rows=rows[rw>0];rw=rw[rw>0]
        norm=rows/rw[:,None];_,sing,V=np.linalg.svd(norm*np.sqrt(rw[:,None]/rw.sum()),full_matrices=False)
        target=h.reshape(-1,nbins);tw=target.sum(1);valid=tw>0;target=target[valid];tw=tw[valid];target/=tw[:,None]
        pod={}; legendre={}
        from numpy.polynomial.legendre import legint,legval
        edges=np.linspace(-1,1,nbins+1)
        B=[]
        for l in range(16):
            coeff=np.zeros(l+1);coeff[-1]=1.;B.append(np.diff(legval(edges,legint(coeff))))
        B=np.asarray(B)
        for k in [2,4,6,8,12,16]:
            pr=target@np.linalg.pinv(B[:k])@B[:k];pr=np.maximum(pr,0);pr/=pr.sum(1)[:,None]
            legendre[str(k)]=float((abs(pr-target)*tw[:,None]).sum()/total)
        for k in [2,4,6,8,12,16]:
            pr=target@V[:k].T@V[:k];negative=np.minimum(pr,0);pr=np.maximum(pr,0);pr/=pr.sum(1)[:,None]
            pod[str(k)]={'weighted_L1':float((abs(pr-target)*tw[:,None]).sum()/total),'negative_mass_before_clip':float((-negative*tw[:,None]).sum()/total)}
        # Coupled wavelength-angle basis: one coefficient vector per cell/class.
        sr=np.concatenate([x.reshape(-1,9*nbins) for x in train]);sw=sr.sum(1);sr=sr[sw>0];sw=sw[sw>0];sr/=sw[:,None]
        _,_,VS=np.linalg.svd(sr*np.sqrt(sw[:,None]/sw.sum()),full_matrices=False)
        st=h.reshape(-1,9*nbins);wt=st.sum(1);st=st[wt>0];wt=wt[wt>0];st/=wt[:,None]
        coupled={}
        for k in [2,4,6,8,12,16,24,32,48]:
            pr=st@VS[:k].T@VS[:k];pr=np.maximum(pr,0);pr/=pr.sum(1)[:,None]
            coupled[str(k)]=float((abs(pr-st)*wt[:,None]).sum()/total)
        results[str(held)]={'one_template_L1':float(abs(p1-truth).sum()/total),'six_physical_templates_mixed_L1':float(abs(pred.sum(1)-truth).sum()/total),
             'six_templates_class_resolved_L1':float(abs(pred-h).sum()/total),'byclass_L1':byclass,'POD_oracle_projection':pod,'Legendre_oracle_projection_L1':legendre,'coupled_lambda_angle_POD_L1':coupled,
             'POD_note':f'Projection coefficients measured on held-out source; basis trained only other energies. Accuracy is {nbins}-bin angular spectral marginal, no spatial/time/azimuth compression.'}
    return results


def isolated_showers(data,model,quad,nbins):
    import lighthit as lh
    phase=model.medium.sample(quad.wavelength_nm)['phase_index'];result={}
    for energy in [100,1000]:
        path=paths.work(f'muon-segments-20261009/structure/electron-current-{energy}GeV.npz')
        with np.load(path) as z: source=lh.SpectralLightElements(**{k:z[k] for k in z.files},cone_model='spectral')
        spec=source.photon_density_per_nm(quad.wavelength_nm,phase)*quad.weight_nm
        spec[source.beta[:,None]*phase<=1]=0.;w=spec.sum(1)
        axis=np.sum(source.direction*w[:,None],axis=0);axis/=np.linalg.norm(axis)
        hist,_,_=histograms(source.direction@axis,source.beta,spec,phase,np.zeros((len(source),2),dtype=np.int64),np.zeros(len(source),dtype=np.int64),np.zeros(len(source),dtype=np.int64),1,nbins)
        truth=hist.sum((0,1));truth/=truth.sum(-1,keepdims=True)
        diffs={}
        for e,(h,j,r) in data.items():
            x=h[:,3].sum(0);weights=x.sum(1);x/=weights[:,None]
            diffs[str(e)]=float((abs(x-truth)*weights[:,None]).sum()/weights.sum())
        result[str(energy)]={'muon_EM_ge1GeV_marginal_L1':diffs,'axis':axis.tolist(),'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--angle-bins',type=int,default=40);p.add_argument('--output',type=Path,default=paths.work('muon-physics-20261009/secondary'));a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    import lighthit as lh,ntsim
    from ntsim.LightTransport import bgvd_model_from_ntsim
    model=bgvd_model_from_ntsim(ntsim.bgvd.Telescope(n_clusters=(1,2,3,4,5)),ntsim.bgvd.Water())
    quad=lh.WavelengthQuadrature.gauss_legendre(*model.medium.wavelength_range_nm,nodes=9)
    start=time.process_time();data={}
    for e in [300,1000,3000]:
        data[e]=event(e,model,quad,a.output,a.angle_bins);print(e,'done',time.process_time()-start,flush=True)
    out={'definition':'Emitted photons with actual 9 wavelength quadrature; no QE, acceptance, attenuation. Angular histogram integrates cone azimuth analytically. Family root energies are total energy. Delayed class overrides ancestry energy.',
         'energies':{str(e):v[2] for e,v in data.items()},'held_out':evaluate(data),'isolated_shower':isolated_showers(data,model,quad,a.angle_bins),'angle_bins':a.angle_bins,'cpu_s':time.process_time()-start}
    (a.output/'report.json').write_text(json.dumps(out,indent=2));print('CPU',out['cpu_s'],flush=True)
if __name__=='__main__': main()
