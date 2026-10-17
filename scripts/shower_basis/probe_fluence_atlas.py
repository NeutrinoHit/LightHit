#!/usr/bin/env python3
"""Stationary scalar fluence atlas for ONE photon along +z, at 480 nm.

Read-only prepared caches supply >=2 scattering. First scattering is a
positive full-HG quadrature, checked independently in scattering distance.
An exact specialization of the existing numerical isotropic-receiver cache
is also benchmarked; it avoids a stored angular axis entirely.
"""
import paths  # inputs and outputs: scripts/shower_basis/paths.py
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import signal
from time import perf_counter, process_time

import numpy as np
from numba import njit
from scipy.integrate import quad
from scipy.special import roots_legendre, exp1, eval_legendre


@njit(cache=True)
def _first_gauss(radius,theta,mu_a,mu_s,g,nodes,weights):
    out=np.empty(len(radius))
    mut=mu_a+mu_s
    for i in range(len(radius)):
        if theta[i]<=0 or theta[i]>np.pi or radius[i]<=0:
            raise ValueError("require r>0 and 0<theta<=pi; directed forward fluence is singular")
        delta=np.pi-theta[i]
        sd=np.sin(delta);cd=np.cos(delta)
        ratio=1. if delta<1e-8 else delta/sd
        value=0.
        for j in range(len(nodes)):
            t=nodes[j]
            angle=delta*(1.-t)
            scattering_cos=-np.cos(angle)
            phase=(1.-g*g)/(4.*np.pi*((1.-g)**2+2.*g*(1.-scattering_cos))**1.5)
            path=(1.+t)/(1.-t) if delta<1e-8 else -cd+sd/np.tan(angle/2.)
            value+=weights[j]*phase*np.exp(-mut*radius[i]*path)
        out[i]=mu_s/radius[i]*ratio*value
    return out


def first_scalar_gauss(radius_m,theta_rad,medium,order=128):
    radius,theta=np.broadcast_arrays(np.asarray(radius_m,float),np.asarray(theta_rad,float))
    x,w=roots_legendre(order)
    values=_first_gauss(radius.ravel(),theta.ravel(),medium['absorption_per_m'],
                       medium['scattering_per_m'],medium['g'],(x+1)/2,w/2)
    return values.reshape(radius.shape)


def first_scalar_quad(radius,theta,medium):
    """Independent integral in dimensionless initial flight u=l/r."""
    mu_s=medium['scattering_per_m'];mut=medium['absorption_per_m']+mu_s;g=medium['g']
    c,s=np.cos(theta),np.sin(theta)
    def integrand(u):
        distance=np.hypot(u-c,s)
        cosine=(c-u)/distance
        phase=(1-g*g)/(4*np.pi*((1-g)**2+2*g*(1-cosine))**1.5)
        return phase*np.exp(-mut*radius*(u+distance))/distance**2
    cuts=[0.]
    if c>0:
        cuts.extend([max(0.,c-10*s),c,c+10*s])
    cuts=sorted(set(cuts));cuts.append(np.inf)
    value=error=0.
    for lo,hi in zip(cuts[:-1],cuts[1:]):
        v,e=quad(integrand,lo,hi,epsabs=1e-13,epsrel=2e-10,limit=500)
        value+=v;error+=e
    return mu_s/radius*value,mu_s/radius*error


@njit(cache=True)
def scalar_cache_lookup(radius,theta,log_r,weighted_coefficients,absorption):
    """Exact numerical-cache specialization: cubic log-r and Legendre theta.

    weighted_coefficients[j,l,power] = sqrt(2*l+1) K[j,l,J=0,power].
    This is the same finite-L numerical cache, not a new physical approximation.
    """
    out=np.empty(len(radius));degree=weighted_coefficients.shape[1]-1
    for i in range(len(radius)):
        x=np.log(radius[i])
        if x<log_r[0]-1e-12 or x>log_r[-1]+1e-12:
            raise ValueError("radius outside prepared cache")
        j=min(max(np.searchsorted(log_r,x)-1,0),len(log_r)-2)
        t=x-log_r[j];mu=np.cos(theta[i]);p0=1.;p1=mu;value=0.
        for ell in range(degree+1):
            if ell==0:
                pl=p0
            elif ell==1:
                pl=p1
            else:
                pl=((2*ell-1)*mu*p1-(ell-1)*p0)/ell;p0=p1;p1=pl
            c=weighted_coefficients[j,ell]
            radial=((c[0]*t+c[1])*t+c[2])*t+c[3]
            value+=radial*pl
        out[i]=np.exp(-absorption*radius[i])/(4*np.pi*radius[i]**2)*value
    return out


@njit(cache=True)
def atlas_lookup(radius,theta,log_radius,angles,scaled,absorption,scattering,kind):
    """Bilinear atlas query, signed for >=2; no extrapolation or clipping."""
    out=np.empty(len(radius))
    for i in range(len(radius)):
        x=np.log(radius[i]);a=theta[i]
        if x<log_radius[0] or x>log_radius[-1] or a<angles[0] or a>angles[-1]:
            raise ValueError("lookup outside atlas")
        ir=min(max(np.searchsorted(log_radius,x)-1,0),len(log_radius)-2)
        it=min(max(np.searchsorted(angles,a)-1,0),len(angles)-2)
        u=(x-log_radius[ir])/(log_radius[ir+1]-log_radius[ir])
        v=(a-angles[it])/(angles[it+1]-angles[it])
        value=((1-u)*((1-v)*scaled[ir,it]+v*scaled[ir,it+1])
                   +u*((1-v)*scaled[ir+1,it]+v*scaled[ir+1,it+1]))
        if kind==1:
            out[i]=value*scattering*np.exp(-(absorption+scattering)*radius[i])/(radius[i]*a)
        else:
            out[i]=value*np.exp(-absorption*radius[i])/(4*np.pi*radius[i]**2)
    return out


def load_operators(pack,first_pack,cache_directory,wavelength):
    from lighthit.shower_basis import Ge2KernelStore
    store=Ge2KernelStore.load(pack)
    matched=None
    for key,bundle in store._bundles.items():
        if len(bundle['omega'])!=65:
            continue
        for lam,cachekey in enumerate(key[0]):
            with np.load(Path(cache_directory)/f'directional-{cachekey}.npz') as z:
                metadata=json.loads(str(z['metadata']))
            if metadata['medium']['wavelength_nm']==wavelength:
                matched=(key,bundle,lam,cachekey,metadata);break
        if matched is not None:
            break
    if matched is None:
        raise ValueError('requested wavelength is not an existing prepared node')
    key,full,lam,cachekey,metadata=matched
    pair=np.flatnonzero((full['pair_lambda']==0)&(full['pair_mu']==0))
    if len(pair)!=1:
        raise ValueError('isotropic receiver needs exactly one J=0 pair')
    pair=int(pair[0])
    common=dict(full,acceptance_degree=0,pair_lambda=np.array([0],np.int64),
                pair_mu=np.array([0],np.int64),omega=full['omega'][:1],
                absorption=full['absorption'][lam:lam+1],speed=full['speed'][lam:lam+1])
    second=dict(common,coefficients=np.ascontiguousarray(
        full['coefficients'][lam:lam+1,:,:,pair:pair+1,:,:1]))
    header=json.loads((Path(first_pack)/'first-collision.json').read_text())
    # The bundle ge2-pack names its tables by the pre-0.3.0a2 directional keys, a pack written
    # now by the current keys: the same tables are recognised by their absorption and speed.
    same_key=header.get('key')==[list(key[0]),key[1],list(key[2])]
    same_tables=(header.get('key',[None,None])[1]==key[1]
                 and np.allclose(header['meta']['absorption'],full['absorption'],rtol=1e-12,atol=0)
                 and np.allclose(header['meta']['speed'],full['speed'],rtol=1e-12,atol=0))
    if header.get('kind')!='experimental-first-collision-only-v1' or not (same_key or same_tables):
        raise ValueError('incompatible first-collision pack')
    one=np.load(Path(first_pack)/'first-collision.npy',mmap_mode='r')
    first=dict(common,coefficients=np.ascontiguousarray(one[lam:lam+1,:,:,pair:pair+1,:,:1]))
    for name in ['log_r','absorption','speed','omega','pair_lambda','pair_mu']:
        if not np.array_equal(np.asarray(header['meta'][name]),full[name]):
            raise ValueError('first/ge2 metadata mismatch: '+name)
    return first,second,metadata,cachekey


def point_source(degree,isotropic=False):
    from lighthit.experimental.axial_source import AxisFrame,AxialSource,channel_index
    frame=AxisFrame(np.zeros(3),np.array([0.,0.,1.]),np.array([1.,0.,0.]),np.array([0.,1.,0.]))
    kept,degrees=channel_index(degree,0)
    channels=np.sqrt((2*np.arange(degree+1)+1)/(4*np.pi)).astype(complex)[None,:,None]
    if isotropic:
        channels[:,1:]=0.
    return AxialSource(frame,np.zeros(1),np.zeros((1,2)),channels,kept,degrees,degree,0,0.,
                        dict(source_photons=1.,source_kind='isotropic' if isotropic else 'directed +z'))


def generic_query(bundle,radius,theta,isotropic=False):
    from lighthit._shower_basis import apply_multi
    r,t=np.broadcast_arrays(radius,theta);r=r.ravel();t=t.ravel()
    receivers=np.stack([r*np.sin(t),np.zeros(len(r)),r*np.cos(t)],axis=1)
    source=point_source(bundle['degree'],isotropic=isotropic)
    result=apply_multi(bundle,source,source,receivers,np.tile([0.,0.,1.],(len(r),1)),
                       np.array([4*np.pi]),np.ones(1),np.zeros(1),np.ones((1,len(r))),
                       radial_window_m=(3.-5e-10,300.+5e-10))
    return result[0].real.reshape(np.broadcast_arrays(radius,theta)[0].shape)


def metrics(candidate,reference):
    absolute=np.abs(candidate-reference)
    relative=absolute/np.maximum(np.abs(reference),1e-300)
    strong=np.abs(reference)>=np.max(np.abs(reference))*1e-4
    return dict(maximum_absolute_error=float(absolute.max()),median_relative_error=float(np.median(relative)),
                p95_relative_error=float(np.quantile(relative,.95)),maximum_relative_error=float(relative.max()),
                strong_cut_fraction_of_maximum=1e-4,strong_points=int(strong.sum()),
                strong_p95_relative_error=float(np.quantile(relative[strong],.95)),
                strong_maximum_relative_error=float(relative[strong].max()))


def cache_cell_averages(r_edges,mu_edges,log_r,radial_coefficients,absorption,
                        radial_order=12,angular_order=20):
    """Volume averages, integrating r^2 dr dmu; no centre-point approximation."""
    xr,wr=roots_legendre(radial_order);xm,wm=roots_legendre(angular_order)
    dr=np.diff(r_edges);dm=np.diff(mu_edges)
    r=(r_edges[:-1]+r_edges[1:])[:,None]/2+dr[:,None]*xr[None,:]/2
    mu=(mu_edges[:-1]+mu_edges[1:])[:,None]/2+dm[:,None]*xm[None,:]/2
    shape=(len(dr),len(dm),radial_order,angular_order)
    rr=np.broadcast_to(r[:,None,:,None],shape).ravel()
    theta=np.arccos(np.broadcast_to(mu[None,:,None,:],shape)).ravel()
    values=scalar_cache_lookup(rr,theta,log_r,radial_coefficients,absorption).reshape(shape)
    mass=(2*np.pi*dr[:,None]*dm[None,:]/4*np.einsum(
          'ijab,ia,a,b->ij',values,r*r,wr,wm))
    volume=2*np.pi/3*(r_edges[1:]**3-r_edges[:-1]**3)[:,None]*dm[None,:]
    return mass/volume,mass,volume


def compare_cone_mc(output):
    """Compare preserved volume averages with the independent cone MC only."""
    output=Path(output)
    mcfile=paths.work('green-map-20261009/stationary_mc/cone-histogram.npz')
    with np.load(output/'cherenkov_cone_average.npz') as z:
        cache=z['Fge2_cone_cell_average'];redges=z['r_edges_m'];muedges=z['mu_edges']
        cone_cosine=float(z['cone_cosine'])
    with np.load(mcfile) as z:
        if not np.array_equal(redges,z['r_edges_m']) or not np.array_equal(muedges,z['mu_edges']):
            raise ValueError('cone cache/MC bin edges differ')
        mc=z['F_ge2'];se=z['F_standard_error'];counts=z['counts'];volume=z['volume_m3']
    setup=json.loads(mcfile.with_name('cone-report.json').read_text())
    if not np.isclose(np.arccos(cone_cosine),setup['theta_C_rad'],rtol=0,atol=1e-12):
        raise ValueError('cone cache/MC opening angle differs')
    direct_report=json.loads(mcfile.with_name('report.json').read_text())
    full_mass=direct_report['fluence_mass_m'];delta=cache-mc;good=counts>=25;strong=counts>=100
    zscore=delta[good]/se[good]
    report=dict(reference_file=str(mcfile),samples=setup['count'],theta_C_rad=setup['theta_C_rad'],
        cache_mass_in_map_m=float(np.sum(cache*volume)),mc_mass_in_map_m=float(np.sum(mc*volume)),
        exact_full_space_fluence_mass_m=full_mass,negative_cache_cells=int((cache<0).sum()),
        negative_cache_mass_m=float(-np.sum(cache[cache<0]*volume[cache<0])),
        l1_cache_mc_mass_m=float(np.sum(np.abs(delta)*volume)),
        l1_fraction_of_exact_full_mass=float(np.sum(np.abs(delta)*volume)/full_mass),
        approximate_mc_sampling_l1_scale_m=float(np.sqrt(2/np.pi)*np.sum(se*volume)),
        bins_with_at_least_25_samples=int(good.sum()),bins_with_at_least_100_samples=int(strong.sum()),
        median_relative_error_counts_ge100=float(np.median(np.abs(delta[strong])/mc[strong])),
        p95_relative_error_counts_ge100=float(np.quantile(np.abs(delta[strong])/mc[strong],.95)),
        fraction_above_3sigma_counts_ge25=float(np.mean(np.abs(zscore)>3)),
        fraction_above_5sigma_counts_ge25=float(np.mean(np.abs(zscore)>5)),
        maximum_absolute_zscore_counts_ge25=float(np.max(np.abs(zscore))),
        comparison_scope='Same volume-integrated cone observable; no clipping, centre-point substitution or fitted normalization.')
    np.savez_compressed(output/'cone_mc_cell_comparison.npz',r_edges_m=redges,mu_edges=muedges,
        Fge2_cone_cache=cache,Fge2_cone_mc=mc,mc_counts=counts,mc_standard_error=se,volume_m3=volume)
    (output/'cone_mc_comparison.json').write_text(json.dumps(report,indent=2))
    if (output/'report.json').exists():
        original=json.loads((output/'report.json').read_text())
        original['independent_cone_mc']=report
        (output/'report.json').write_text(json.dumps(original,indent=2))
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=paths.work('green-map-20261009/numeric'))
    p.add_argument('--radial-nodes',type=int,default=48)
    p.add_argument('--angular-nodes',type=int,default=96)
    p.add_argument('--cone-phase-index',type=float,
                   help='Actual phase index at480nm for an optional beta=1 Cherenkov-cone average')
    p.add_argument('--compare-cone-only',action='store_true',help='Compare saved cone averages to saved MC; no transport calls')
    args=p.parse_args()
    if args.compare_cone_only:
        print(json.dumps(compare_cone_mc(args.output),indent=2));return
    import numba
    from lighthit.medium import Medium
    from lighthit.single import single_spectrum
    if numba.get_num_threads()!=1:
        p.error('requires one Numba thread')
    def timeout(signum,frame):
        raise TimeoutError('three-minute process CPU budget exceeded')
    signal.signal(signal.SIGPROF,timeout);signal.setitimer(signal.ITIMER_PROF,180.)
    began,cpu=perf_counter(),process_time()
    args.output.mkdir(parents=True,exist_ok=True)
    first,second,cache_metadata,key=load_operators(
        str(paths.GE2_PACK.path()),
        str(paths.work('muon-segments-20261009/first-cache/pack65')),
        str(paths.RTE_CACHE.path()),480.)
    medium=cache_metadata['medium'];mua=medium['absorption_per_m'];mus=medium['scattering_per_m'];g=medium['g']
    medium_object=Medium(**medium)
    degree=second['degree'];factor=np.sqrt(2*np.arange(degree+1)+1)[None,:,None]
    radial2=np.ascontiguousarray(second['coefficients'][0,:,:,0,:,0].real*factor)
    radial1=np.ascontiguousarray(first['coefficients'][0,:,:,0,:,0].real*factor)
    radii=np.geomspace(3.,300.,args.radial_nodes)
    angles=np.unique(np.r_[np.linspace(0.,np.pi,args.angular_nodes)[1:],np.geomspace(1e-4,.2,33)])
    rr,tt=np.meshgrid(radii,angles,indexing='ij')
    f2=scalar_cache_lookup(rr.ravel(),tt.ravel(),second['log_r'],radial2,mua).reshape(rr.shape)
    f1cache=scalar_cache_lookup(rr.ravel(),tt.ravel(),first['log_r'],radial1,mua).reshape(rr.shape)
    gauss={n:first_scalar_gauss(rr,tt,medium,n) for n in [64,128,256]}
    f1=gauss[256]
    quadrature={f'{n}_vs_256':metrics(gauss[n],f1) for n in [64,128]}
    checks=[]
    for r,theta in [(3.,1e-4),(3.,.02),(10.,.1),(30.,.5),(30.,1.5),(100.,2.5),(300.,np.pi)]:
        value,error=first_scalar_quad(r,theta,medium)
        direct=float(first_scalar_gauss(r,theta,medium,256))
        checks.append(dict(radius_m=r,theta_rad=theta,quadrature_m2=direct,
                           independent_quad_m2=value,quad_error_estimate_m2=error,
                           relative_difference=abs(direct-value)/max(abs(value),1e-300)))
    antipodal=[]
    pback=(1-g)/(4*np.pi*(1+g)**2)
    for r in [3.,30.,300.]:
        a=(mua+mus)*r
        exact=mus/r*pback*(np.exp(-a)-2*a*np.exp(a)*exp1(2*a))
        direct=float(first_scalar_gauss(r,np.pi,medium,256))
        antipodal.append(dict(radius_m=r,closed_form_m2=exact,quadrature_m2=direct,
                              relative_difference=abs(direct-exact)/exact))

    rng=np.random.default_rng(1942)
    held_r=np.exp(rng.uniform(np.log(3.001),np.log(299.99),512))
    held_theta=np.r_[rng.uniform(.02,np.pi,384),np.exp(rng.uniform(np.log(1e-4),np.log(.2),128))]
    generic2=generic_query(second,held_r,held_theta)
    fast2=scalar_cache_lookup(held_r,held_theta,second['log_r'],radial2,mua)
    generic1=generic_query(first,held_r,held_theta)
    fast1=scalar_cache_lookup(held_r,held_theta,first['log_r'],radial1,mua)
    direct1=first_scalar_gauss(held_r,held_theta,medium,256)
    scaled2=f2*(4*np.pi*rr**2)*np.exp(mua*rr)
    scaled1=f1*rr*tt*np.exp((mua+mus)*rr)/mus
    atlas2=atlas_lookup(held_r,held_theta,np.log(radii),angles,scaled2,mua,mus,2)
    atlas1=atlas_lookup(held_r,held_theta,np.log(radii),angles,scaled1,mua,mus,1)

    # Angular average of the directed cache must match one isotropic photon.
    mu,w=roots_legendre(64);norm=[]
    for r in [3.01,10.,30.,100.]:
        rvalues=np.full(len(mu),r);theta=np.arccos(mu)
        directed=scalar_cache_lookup(rvalues,theta,first['log_r'],radial1,mua)
        average=float(w@directed/2)
        isotropic=float(generic_query(first,np.array([r]),np.array([1.]),isotropic=True)[0])
        exact_iso=float(single_spectrum(np.array([0.]),r,None,medium_object,backend='numpy')[0][0].real)
        norm.append(dict(radius_m=r,directed_angular_average_m2=average,isotropic_cache_m2=isotropic,
                         exact_isotropic_m2=exact_iso,cache_over_exact_isotropic=isotropic/exact_iso,
                         angular_identity_relative_error=abs(average-isotropic)/max(abs(isotropic),1e-300)))

    # Warm both lookup kernels before measuring one million independent queries.
    qr=np.exp(rng.uniform(np.log(3.001),np.log(299.99),1_000_000))
    qt=rng.uniform(angles[0],angles[-1],len(qr))
    scalar_cache_lookup(qr[:1],qt[:1],second['log_r'],radial2,mua)
    atlas_lookup(qr[:1],qt[:1],np.log(radii),angles,scaled2,mua,mus,2)
    lookup={}
    for name,call in [('signed_bilinear_atlas',lambda:atlas_lookup(qr,qt,np.log(radii),angles,scaled2,mua,mus,2)),
                      ('native_radial_legendre',lambda:scalar_cache_lookup(qr,qt,second['log_r'],radial2,mua))]:
        t,c=perf_counter(),process_time();values=call();elapsed,used=perf_counter()-t,process_time()-c
        lookup[name]=dict(queries=len(qr),wall_seconds=elapsed,cpu_seconds=used,
                          queries_per_cpu_second=len(qr)/used,checksum=float(values.sum()))

    report=dict(medium=medium,source_photons=1.,source_direction=[0.,0.,1.],receiver_acceptance=1.,
        alpha_0=4*np.pi,receiver_effective_area_m2=1.,spectral_weight=1.,units='m^-2 per one directed photon',
        group_index_note='group velocity cancels from stationary direct first order',cache_key=key,
        cache_settings=cache_metadata['settings'],radial_nodes=len(radii),angular_nodes=len(angles),
        minimum_theta_rad=float(angles[0]),quadrature_convergence=quadrature,
        independent_first_order_checks=checks,antipodal_closed_form_checks=antipodal,
        normalization=norm,fast_cache_specialization_ge2=metrics(fast2,generic2),
        fast_cache_specialization_one=metrics(fast1,generic1),
        first_cached_vs_full_hg=metrics(fast1,direct1),
        heldout_interpolation_ge2=metrics(atlas2,generic2),heldout_interpolation_one=metrics(atlas1,direct1),
        negative_ge2_grid_points=int((f2<0).sum()),minimum_ge2_m2=float(f2.min()),maximum_ge2_m2=float(f2.max()),
        negative_cached_one_grid_points=int((f1cache<0).sum()),minimum_cached_one_m2=float(f1cache.min()),
        positive_exact_one_minimum_m2=float(f1.min()),lookup_benchmark=lookup,
        native_radial_legendre_bytes=int(radial2.nbytes+second['log_r'].nbytes),
        ge2_atlas_bytes=int(f2.nbytes+radii.nbytes+angles.nbytes),
        total_wall_seconds=perf_counter()-began,total_cpu_seconds=process_time()-cpu,
        note='Signed numerical >=2 cache is preserved; its physical accuracy is separate from interpolation accuracy.')
    mcfile=paths.work('green-map-20261009/stationary_mc/histogram.npz')
    if mcfile.exists():
        with np.load(mcfile) as z:
            redges=z['r_edges_m'];muedges=z['mu_edges'];counts=z['counts']
            mc=z['F_ge2'];standard_error=z['F_standard_error']
        cell12,mass12,volume=cache_cell_averages(redges,muedges,second['log_r'],radial2,mua,12,20)
        cell24,mass24,_=cache_cell_averages(redges,muedges,second['log_r'],radial2,mua,24,20)
        good=counts>=25;strong=counts>=100
        zscore=(cell24-mc)/np.maximum(standard_error,1e-300)
        mcreport=json.loads(mcfile.with_name('report.json').read_text())
        x,w=roots_legendre(12);lo=second['log_r'][:-1];hi=second['log_r'][1:]
        t=(x[None,:]+1)*(hi-lo)[:,None]/2;r=np.exp(lo[:,None]+t)
        integration_weight=w[None,:]*(hi-lo)[:,None]/2*r*np.exp(-mua*r)
        s=[((radial2[:,ell,0,None]*t+radial2[:,ell,1,None])*t
               +radial2[:,ell,2,None])*t+radial2[:,ell,3,None] for ell in range(3)]
        mass=float(np.sum(integration_weight*s[0]))
        mean_z=float(np.sum(integration_weight*r*s[1]/3)/mass)
        mean_z2=float(np.sum(integration_weight*r*r*(s[0]/3+2*s[2]/15))/mass)
        mean_rho2=float(np.sum(integration_weight*r*r*(2*s[0]/3-2*s[2]/15))/mass)
        report['independent_stationary_mc']=dict(
            reference_file=str(mcfile),counts=int(mcreport['count']),
            radial_bins=len(redges)-1,angular_mu_bins=len(muedges)-1,
            cell_average_radial_12_vs_24=metrics(cell12,cell24),
            cache_mass_in_map_m=float(mass24.sum()),mc_mass_in_map_m=float((mc*volume).sum()),
            exact_full_space_ge2_mass_m=(mus/(mua+mus))**2/mua,
            negative_cache_cells=int((cell24<0).sum()),
            negative_cache_mass_m=float(-mass24[mass24<0].sum()),
            absolute_cache_minus_mc_mass_m=float(np.abs((cell24-mc)*volume).sum()),
            bins_with_at_least_25_samples=int(good.sum()),
            bins_with_at_least_100_samples=int(strong.sum()),
            fraction_above_3sigma_counts_ge25=float(np.mean(np.abs(zscore[good])>3)),
            fraction_above_5sigma_counts_ge25=float(np.mean(np.abs(zscore[good])>5)),
            median_relative_error_counts_ge100=float(np.median(np.abs(cell24[strong]-mc[strong])/mc[strong])),
            p95_relative_error_counts_ge100=float(np.quantile(np.abs(cell24[strong]-mc[strong])/mc[strong],.95)),
            maximum_absolute_zscore_counts_ge25=float(np.max(np.abs(zscore[good]))),
            conditional_cache_moments_within_map=dict(mean_z_m=mean_z,mean_z2_m2=mean_z2,mean_rho2_m2=mean_rho2),
            full_space_exact_moments_from_mc_report=mcreport['mean_exact'][:3],
            comparison_scope='Volume-integrated cells, not map centres; MC noise and numerical-cache error remain distinct.')
        np.savez_compressed(args.output/'mc_cell_comparison.npz',r_edges_m=redges,mu_edges=muedges,
            Fge2_cache_cell_mean=cell24,Fge2_mc=mc,mc_standard_error=standard_error,
            mc_counts=counts,volume_m3=volume,cache_mass_m=mass24)
        truncations=[];truncated_maps={}
        for kept_degree in [8,16,24,32]:
            coeff=np.ascontiguousarray(radial2[:,:kept_degree+1])
            cells,cell_mass,_=cache_cell_averages(redges,muedges,second['log_r'],coeff,mua,24,20)
            truncated_maps[f'L{kept_degree}']=cells
            discrepancy=np.abs((cells-mc)*volume)
            residual=(cells-mc)/np.maximum(standard_error,1e-300)
            truncations.append(dict(source_degree=kept_degree,scattering_degree=cache_metadata['settings']['scattering_degree'],
                mass_in_map_m=float(cell_mass.sum()),negative_cells=int((cells<0).sum()),
                negative_mass_m=float(-cell_mass[cell_mass<0].sum()),
                l1_cache_mc_mass_m=float(discrepancy.sum()),
                l1_fraction_of_exact_full_mass=float(discrepancy.sum()/mcreport['fluence_mass_m']),
                median_relative_error_counts_ge100=float(np.median(np.abs(cells[strong]-mc[strong])/mc[strong])),
                fraction_above_3sigma_counts_ge25=float(np.mean(np.abs(residual[good])>3))))
        report['angular_source_truncation_same_operator']=truncations
        np.savez_compressed(args.output/'angular_truncation_cells.npz',r_edges_m=redges,mu_edges=muedges,
                             Fge2_mc=mc,mc_counts=counts,volume_m3=volume,**truncated_maps)
        if args.cone_phase_index is not None:
            if args.cone_phase_index<=1.:
                raise ValueError('beta=1 Cherenkov cone requires phase index>1')
            cosine=1./args.cone_phase_index
            cone_coeff=np.ascontiguousarray(radial2*eval_legendre(np.arange(degree+1),cosine)[None,:,None])
            cone=scalar_cache_lookup(rr.ravel(),tt.ravel(),second['log_r'],cone_coeff,mua).reshape(rr.shape)
            cells,mass_cone,_=cache_cell_averages(redges,muedges,second['log_r'],cone_coeff,mua,24,20)
            report['cherenkov_cone_average_same_signed_cache']=dict(phase_index=args.cone_phase_index,beta=1.,
                cone_cosine=cosine,cone_angle_rad=float(np.arccos(cosine)),source_degree=degree,
                negative_grid_points=int((cone<0).sum()),minimum_fluence_m2=float(cone.min()),
                maximum_fluence_m2=float(cone.max()),negative_cells=int((cells<0).sum()),
                negative_mass_m=float(-mass_cone[mass_cone<0].sum()),mass_in_map_m=float(mass_cone.sum()),
                negative_mass_fraction_of_exact_full=float(-mass_cone[mass_cone<0].sum()/mcreport['fluence_mass_m']),
                accuracy_scope='Positivity diagnostic only. No cone-convolved independent MC reference was used.')
            np.savez_compressed(args.output/'cherenkov_cone_average.npz',r_m=radii,theta_rad=angles,
                Fge2_cone_cached=cone,radial_cone_coefficients=cone_coeff,r_edges_m=redges,mu_edges=muedges,
                Fge2_cone_cell_average=cells,phase_index=args.cone_phase_index,cone_cosine=cosine)
            if paths.work('green-map-20261009/stationary_mc/cone-histogram.npz').exists():
                report['independent_cone_mc']=compare_cone_mc(args.output)
        report['total_wall_seconds']=perf_counter()-began
        report['total_cpu_seconds']=process_time()-cpu
    np.savez_compressed(args.output/'fluence_atlas_480nm.npz',r_m=radii,theta_rad=angles,
        F1_direct=f1,F1_cached=f1cache,Fge2=f2,scaled_F1=scaled1,scaled_Fge2=scaled2,
        log_r_native=second['log_r'],radial_ge2_coefficients=radial2,radial_one_coefficients=radial1,
        medium_json=json.dumps(medium),source_photons=np.array(1.),metadata=json.dumps(report))
    np.savez_compressed(args.output/'heldout.npz',r_m=held_r,theta_rad=held_theta,
        F1_direct=direct1,F1_cached=fast1,Fge2_direct=generic2,Fge2_atlas=atlas2,F1_atlas=atlas1)
    (args.output/'report.json').write_text(json.dumps(report,indent=2))
    signal.setitimer(signal.ITIMER_PROF,0)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
