"""Event-independent primary-only atlas application with explicit fallbacks.

Charge is accumulated over the complete source before selecting time profiles.
The q1 omitted-tail budget is rigorous for its positive spectral quadrature.
The >=2 radial window and numerical-kernel limitations remain explicit.
"""
from copy import copy
from dataclasses import dataclass, replace
from time import perf_counter, process_time
from functools import lru_cache
import numpy as np

from ..ballistic_interval import ballistic_interval
from ..prompt import PromptConfig
from .primary_polyline import fit_primary_polyline, primary_spectral_quadrature
from .muon_integral import order1_integrals
from .first_charge_only import first_scattering_charge_only
from .track_first_bounds import combined_first_scattering_segment_bounds


def subset_kernel(kernel, ids):
    """Shallow kernel copy restricted to the OMs ``ids``."""
    ids = np.asarray(ids, int)
    local = copy(kernel)
    d = kernel.detector
    local.detector = replace(d, positions_m=d.positions_m[ids], orientations=d.orientations[ids],
        effective_area_m2=d.effective_area_m2[ids], identifiers={k:np.asarray(v)[ids] for k,v in d.identifiers.items()})
    return local


def split_source(source, parts):
    """Split uniform fitted chords without changing either spectral field."""
    from ..sources import SpectralLightElements
    ids = np.array([p[0] for p in parts], int)
    lo = np.array([p[1] for p in parts]); hi = np.array([p[2] for p in parts])
    fraction = (hi-lo)/source.length_m[ids]
    dt = source.end_ns[ids]-source.start_ns[ids]
    return SpectralLightElements(source.start_m[ids]+lo[:,None]*source.direction[ids], source.direction[ids],
        hi-lo, source.coefficient0[ids]*fraction, source.coefficient2[ids]*fraction, source.beta[ids],
        source.start_ns[ids]+lo/source.length_m[ids]*dt,
        source.start_ns[ids]+hi/source.length_m[ids]*dt, np.arange(len(ids)), source.uid[ids], cone_model='spectral')


@lru_cache(maxsize=12)
def _legendre_rule(order):
    """Gauss-Legendre nodes and weights."""
    return np.polynomial.legendre.leggauss(order)


def _direct_ge2_order(span_m):
    """Candidate rule for short smooth >=2 intervals; independently checked."""
    for span,order in ((.03,2),(.1,3),(.25,4),(.6,6),(1.,8)):
        if span_m<=span:return order
    return 12


class DirectGe2:
    """Lazy existing-RTE fallback. Does not participate in atlas construction."""
    def __init__(self, kernel, directory, *, backend='fused-looks', quadrature='fixed12'):
        """Direct >=2 evaluator on the RTE pack in ``directory``; the pack is loaded lazily."""
        self.kernel = kernel; self.directory = directory; self.bundle = None
        if backend not in ('generic','fused-looks'):raise ValueError('unknown direct RTE backend')
        self.backend=backend
        if quadrature not in ('fixed12','fixed24','adaptive','filon'):raise ValueError('unknown direct quadrature')
        if quadrature=='filon' and backend!='fused-looks':raise ValueError('Filon batch requires fused-looks backend')
        self.quadrature=quadrature
        self.timings=dict(load_wall_seconds=0.,load_cpu_seconds=0.,
            full_eval_wall_seconds=0.,full_eval_cpu_seconds=0.,
            charge_eval_wall_seconds=0.,charge_eval_cpu_seconds=0.,full_calls=0,charge_calls=0,full_points=0,charge_points=0)

    def _load_bundle(self):
        """Load the pack once and keep its bundle."""
        from ..shower_basis import Ge2KernelStore
        if self.bundle is None:
            if self.directory is None:
                raise ValueError('direct >=2 fallback needs a pack: pass fallback_pack '
                                 '(MuonTransport passes the ge2_pack of muon-data.json)')
            wall,cpu=perf_counter(),process_time()
            store = Ge2KernelStore.load(self.directory)
            self.bundle = next(b for b in store._bundles.values() if len(b['omega']) == 65)
            self.charge_bundle=next((b for b in store._bundles.values() if len(b['omega'])==1 and b['omega'][0]==0),None)
            self.timings['load_wall_seconds']+=perf_counter()-wall
            self.timings['load_cpu_seconds']+=process_time()-cpu

    def many(self,queries):
        """Apply full-frequency fallback in bounded spectral-plane batches."""
        from .filon_ge2 import backward_ge2_filon_many
        self._load_bundle();wall,cpu=perf_counter(),process_time()
        values,meta=backward_ge2_filon_many(self.kernel,self.bundle,queries,order=6,allow_gauss=True)
        self.timings['full_batch_wall_seconds']=self.timings.get('full_batch_wall_seconds',0.)+perf_counter()-wall
        self.timings['full_batch_cpu_seconds']=self.timings.get('full_batch_cpu_seconds',0.)+process_time()-cpu
        self.timings['full_calls']+=len(queries);self.timings['full_batch_work']=meta
        return values

    def __call__(self, rho, look, beta, clock, lo, hi, density, closest, full):
        """>=2 spectrum (or charge) of one piece by direct quadrature over the track."""
        from .track_ge2_primitive import cone_point_evaluator
        if full and self.quadrature=='filon':
            return self.many([(rho,look,beta,clock,lo,hi,density,closest)])[0]
        self._load_bundle()
        bundle = self.bundle
        if not full:
            bundle = getattr(self,'charge_bundle',None) or dict(bundle,
                omega=bundle['omega'][:1], coefficients=bundle['coefficients'][...,:1])
        wave = self.kernel.wavelength
        sample = self.kernel.medium.sample(wave.wavelength_nm)
        if not np.allclose(sample['absorption_per_m'], bundle['absorption'], rtol=1e-12, atol=0):
            raise ValueError('fallback RTE optics differ')
        evaluator = cone_point_evaluator(bundle, impact_m=rho, look=look, beta=beta,
            alpha=self.kernel.acceptance()['alpha'], wavelength_nm=wave.wavelength_nm,
            wavelength_weight_nm=wave.weight_nm, phase_index=sample['phase_index'],
            efficiency=self.kernel.detector.spectral_weight(wave.wavelength_nm),
            effective_area_m2=float(self.kernel.detector.effective_area_m2[0]),backend=self.backend)
        edges = np.linspace(lo,hi,max(1,int(np.ceil((hi-lo)/2)))+1)
        half = np.diff(edges)/2
        order=(_direct_ge2_order(float(2*half.max())) if self.quadrature in ('adaptive','filon') else
            24 if self.quadrature=='fixed24' else 12)
        gx,gw = _legendre_rule(order)
        u = (edges[:-1,None]+half[:,None]*(gx+1)).ravel()
        wall,cpu=perf_counter(),process_time();point_values=evaluator(u)
        kind='full' if full else 'charge'
        self.timings[kind+'_eval_wall_seconds']+=perf_counter()-wall
        self.timings[kind+'_eval_cpu_seconds']+=process_time()-cpu
        self.timings[kind+'_calls']+=1;self.timings[kind+'_points']+=len(u)
        spectrum = ((half[:,None]*gw).ravel()[:,None]*point_values
                    *np.exp(-1j*u[:,None]*clock*bundle['omega'])).sum(0)
        return spectrum*density*np.exp(1j*closest*bundle['omega'])


@dataclass
class UniversalPrimaryResult:
    """Primary-only orders 0, 1 and >=2 on the supplied OMs, with metadata."""
    charge_orders_pe: np.ndarray
    bins_orders_pe: np.ndarray
    time_origin_ns: np.ndarray
    relative_time_edges_ns: np.ndarray
    active: np.ndarray
    metadata: dict
    spectrum_ge2_pe: np.ndarray | None = None


def transport_universal_primary(kernel, source, first_store, ge2_store, *, primary_uid=0,
        tail_budget_pe=1e-6, fallback_pack=None,
        selection_margin=1.05, time_origin_ns=None, progress=None,
        batch_beta_charge=True, beta_cell_m=.25, first_cubature=None, ge2_beta_atlas=None, first_beta_atlas=None,
        direct_ge2_quadrature='fixed12',q1_charge_gauss_order=0,q1_charge_panel_width=1.5,
        skip_ballistic=False):
    """Compute all primary charges, then time bins for selected OMs only.

    Stored beta=1/clock=c is used only below explicit cone/clock error guards.
    Unsupported or negative-reconstruction cases fall back to direct transport.
    No previous event response enters the selection or evaluation.
    """
    from .._shower_basis import earliest_element_arrival
    from ..readout import inverse_bins
    from scipy.interpolate import CubicSpline
    from .track_ge2_primitive import track_ge2_optics_metadata
    if not np.isfinite(tail_budget_pe) or tail_budget_pe < 0 or selection_margin < 1:
        raise ValueError('invalid omission budget or selection margin')
    begin,cpu = perf_counter(),process_time(); timings = {}
    if skip_ballistic and kernel.config.threshold_pe != 0:
        raise ValueError('skip_ballistic requires an externally selected receiver set and threshold_pe=0')
    first_store.validate_kernel(kernel)
    first_beta_requested=first_beta_atlas is not None
    first_beta_payload_bytes=0 if first_beta_atlas is None else first_beta_atlas.resident_bytes
    if first_beta_atlas is not None:first_beta_atlas.validate_kernel(kernel)
    if track_ge2_optics_metadata(kernel)['optics_fingerprint'] != ge2_store.metadata['optics_fingerprint']:
        raise ValueError('>=2 optical contract differs')
    beta_charge_requested=ge2_beta_atlas is not None
    beta_charge_payload_bytes=0 if ge2_beta_atlas is None else ge2_beta_atlas.resident_bytes
    if ge2_beta_atlas is not None:ge2_beta_atlas.validate_kernel(kernel)
    mask = source.uid == primary_uid
    if not mask.any(): raise ValueError('primary UID absent')
    original = source if mask.all() else source.subset(mask)
    fitted = fit_primary_polyline(original, phase_indices=kernel.medium.phase_index)
    pieces = fitted.elements; det=kernel.detector; n=len(det.positions_m)
    if first_beta_atlas is not None:
        phases=kernel.medium.phase_index
        above=pieces.beta[:,None]*phases[None,:]>1
        angles=np.max(abs(np.arccos(np.clip(1/(pieces.beta[:,None]*phases[None,:]),-1,1))-
            np.arccos(1/phases)[None,:]),axis=1)
        needs=((pieces.beta>=first_beta_atlas.metadata['beta_min'])&np.all(above,axis=1)&(angles>1e-6))
        if not needs.any():first_beta_atlas=None
    if ge2_beta_atlas is not None:
        needs_modes=((abs(pieces.beta-1)>1e-5)&(pieces.beta>=ge2_beta_atlas.metadata['beta_min'])
            &np.isclose(-pieces.coefficient2*pieces.beta**2,pieces.coefficient0,rtol=1e-10,atol=0.))
        if not needs_modes.any():ge2_beta_atlas=None
    if time_origin_ns is None:
        origins = earliest_element_arrival(np.ascontiguousarray(det.positions_m), original.start_m,
            original.direction, original.length_m, original.start_ns, float(kernel.medium.group_index.min()))
    else:
        origins=np.asarray(time_origin_ns,float)
        if origins.shape!=(n,) or not np.isfinite(origins).all():raise ValueError('invalid time origin')
    edges=np.asarray(kernel.config.relative_time_edges_ns); nb=len(edges)-1
    charge=np.zeros((n,3));bins=np.zeros((n,nb,3))
    stage=perf_counter()
    if not skip_ballistic:
        direct0=ballistic_interval(kernel,original,time_origin_ns=origins,relative_edges_ns=edges)
        charge[:,0]=direct0.charge_pe;bins[:,:,0]=direct0.bins_pe
    timings['ballistic_seconds']=perf_counter()-stage
    if progress:progress('ballistic complete',dict(seconds=timings['ballistic_seconds']))
    local=copy(kernel)
    local.wavelength=primary_spectral_quadrature(kernel.medium,kernel.detector,order=2,
        wavelength_range_nm=kernel.config.wavelength_range_nm)
    reduced=None
    if first_cubature is not None:
        first_cubature.validate_optics(kernel)
        reduced=copy(local);reduced.wavelength=first_cubature.quadrature
    direct_ge2=DirectGe2(kernel,fallback_pack,quadrature=direct_ge2_quadrature)
    prompt=PromptConfig(element_method='segment',screen=False)
    charge_options=(dict(gauss_order=q1_charge_gauss_order,panel_width=q1_charge_panel_width)
        if q1_charge_gauss_order else {})
    q1_pairs=[];ge2_pairs=[];q1_fallback=[];ge2_fallback=[]
    bound_sum=np.zeros(n)
    # Only the stationary pass is batched: on a few selected OMs direct
    # segment integration is faster than compiling all 65 frequencies.
    beta_tail=(abs(pieces.beta-1)>1e-5) if batch_beta_charge else np.zeros(len(pieces),bool)
    physical_beta=np.isclose(-pieces.coefficient2*pieces.beta**2,pieces.coefficient0,rtol=1e-10,atol=0.)
    if ge2_beta_atlas is not None:
        beta_tail &= (~physical_beta | (pieces.beta<ge2_beta_atlas.metadata['beta_min']))
    beta_batch_metadata=None
    phases=kernel.medium.phase_index; alpha0=1/first_store.metadata['speed_m_per_ns']
    alpha2=float(ge2_store.metadata['clock_ns_per_m'])
    high=float(ge2_store.metadata.get('radius_window_m',[3.,300.])[1])
    lengths=pieces.length_m
    clocks=(pieces.end_ns-pieces.start_ns)/lengths
    densities=pieces.coefficient0/lengths
    above=pieces.beta[:,None]*phases[None,:]>1
    cone_angles=np.max(abs(np.arccos(np.clip(1/(pieces.beta[:,None]*phases[None,:]),-1,1))-
        np.arccos(1/phases)[None,:]),axis=1)
    cone_angles[~np.all(above,axis=1)]=np.inf
    stage=perf_counter()
    for om in range(n):
        pos=det.positions_m[om]; normal=det.orientations[om]
        # Source cone/clock/yield do not depend on the OM. Evaluate geometric
        # frames for all pieces together instead of thousands of tiny calls.
        delta=pos[None,:]-pieces.start_m
        zs=np.einsum('ij,ij->i',delta,pieces.direction)
        radial=delta-zs[:,None]*pieces.direction
        rhos=np.linalg.norm(radial,axis=1)
        exs=np.divide(radial,rhos[:,None],out=np.zeros_like(radial),where=rhos[:,None]>0)
        eys=np.cross(pieces.direction,exs)
        orientations=np.column_stack((exs@normal,eys@normal,pieces.direction@normal))
        parts=[]
        for i in range(len(pieces)):
            z=float(zs[i]);rho=float(rhos[i]);L=float(lengths[i])
            clock=float(clocks[i]);density=float(densities[i])
            beta=float(pieces.beta[i]); closest=float(pieces.start_ns[i]+z*clock-origins[om])
            if rho==0:
                if z>=-high and z-L<=high:
                    raise ValueError('primary enters unvalidated inner >=2 radius; separate near-field solver required')
                parts.append((i,0.,L));continue
            orientation=orientations[i];angle=float(cone_angles[i])
            lo=max(z-L,float(first_store.x_edges[0]*rho));hi=min(z,float(first_store.x_edges[-1]*rho))
            good=(first_store.radii_m[0]<=rho<=first_store.radii_m[-1] and lo<hi and angle<=1e-6)
            if good:
                shift=closest+(alpha0-clock)*(lo+hi)/2
                q1_pairs.append((om,i,rho,orientation,lo,hi,density,shift))
                s0,s1=z-hi,z-lo
                if s0>0:parts.append((i,0.,s0))
                if s1<L:parts.append((i,s1,L))
            else:parts.append((i,0.,L))
            if rho>=high:continue
            reach=np.sqrt(high*high-rho*rho);a=min(z,reach);b=max(z-L,-reach)
            if a<=b:continue
            if rho<3.01:raise ValueError('primary enters unvalidated inner >=2 radius; separate near-field solver required')
            look=-orientation
            row=(om,i,rho,look,a,b,density,closest,clock,beta)
            if (ge2_store.impact_nodes_m[0]<=rho<=ge2_store.impact_nodes_m[-1]
                    and abs(beta-1)<=1e-5
                    and ge2_store.query_supported(rho,a,b,charge_only=True)):
                ge2_pairs.append(row)
            else:ge2_fallback.append(row)
        if parts:
            partial=split_source(pieces,parts)
            bounds,_=combined_first_scattering_segment_bounds(local,partial,pos,
                effective_area_m2=float(det.effective_area_m2[om]))
            ranked=np.argsort(bounds);acc=np.cumsum(bounds[ranked]);omit=np.zeros(len(parts),bool)
            omit[ranked[acc<=tail_budget_pe]]=True
            bound_sum[om]=float(bounds[omit].sum())
            if not omit.all():q1_fallback.append((om,partial.subset(~omit)))
    timings['geometry_bounds_seconds']=perf_counter()-stage
    partition_stage=perf_counter()
    first_beta_pairs=[];first_direct_charge=list(q1_fallback);first_fallback_charge=np.zeros(n)
    first_beta_failed=0;first_beta_refinement=np.zeros(n)
    if first_beta_atlas is not None:
        first_direct_charge=[]
        for om,part in q1_fallback:
            leftovers=[];pos=det.positions_m[om];normal=det.orientations[om]
            for i in range(len(part)):
                axis=part.direction[i];delta=pos-part.start_m[i];z=float(delta@axis)
                radial=delta-z*axis;rho=float(np.linalg.norm(radial));length=float(part.length_m[i])
                lo=max(z-length,first_store.x_edges[0]*rho);hi=min(z,first_store.x_edges[-1]*rho)
                physical=np.isclose(-part.coefficient2[i]*part.beta[i]**2,part.coefficient0[i],rtol=1e-10,atol=0.)
                if rho>0 and physical and first_beta_atlas.supports(rho,float(part.beta[i]),lo,hi):
                    ex=radial/rho;ey=np.cross(axis,ex)
                    orientation=np.array([normal@ex,normal@ey,normal@axis])
                    start,end=z-hi,z-lo
                    first_beta_pairs.append((om,rho,orientation,lo,hi,part.coefficient0[i]/length,part.beta[i],part,i,start,end))
                    if start>0:leftovers.append((i,0.,start))
                    if end<length:leftovers.append((i,end,length))
                else:leftovers.append((i,0.,length))
            if leftovers:first_direct_charge.append((om,split_source(part,leftovers)))
    timings['q1_beta_partition_seconds']=perf_counter()-partition_stage
    beta_charge_pairs=[];beta_charge_keys=set();negative_beta_reconstructions=0
    negative_beta_mass=np.zeros(n);beta_nonfinite_fallbacks=0
    stage=perf_counter()
    if q1_pairs:
        values=first_store.query_charge_many(np.array([p[2] for p in q1_pairs]),np.array([p[3] for p in q1_pairs]),
            np.array([p[4] for p in q1_pairs]),np.array([p[5] for p in q1_pairs]),
            coefficient0_per_m=np.array([p[6] for p in q1_pairs]))
        np.add.at(charge[:,1],np.array([p[0] for p in q1_pairs]),values)
    if first_beta_pairs:
        values=first_beta_atlas.query_charge_many(np.array([p[1] for p in first_beta_pairs]),
            np.array([p[2] for p in first_beta_pairs]),np.array([p[3] for p in first_beta_pairs]),
            np.array([p[4] for p in first_beta_pairs]),np.array([p[6] for p in first_beta_pairs]),
            coefficient0_per_m=np.array([p[5] for p in first_beta_pairs]))
        valid=np.isfinite(values)&(values>=0);first_beta_failed=int((~valid).sum())
        np.add.at(charge[:,1],np.array([p[0] for p in first_beta_pairs])[valid],values[valid])
        np.add.at(first_fallback_charge,np.array([p[0] for p in first_beta_pairs])[valid],values[valid])
        for row,good in zip(first_beta_pairs,valid):
            if not good:
                om,*_=row;first_direct_charge.append((om,split_source(row[7],[(row[8],row[9],row[10])])))
    if ge2_pairs:
        args=(np.array([p[2] for p in ge2_pairs]),np.array([p[3] for p in ge2_pairs]),
            np.array([p[4] for p in ge2_pairs]),np.array([p[5] for p in ge2_pairs]))
        density=np.array([p[6] for p in ge2_pairs])
        if ge2_beta_atlas is None:
            values=ge2_store.segment_charge_many(*args,coefficient0_per_m=density)
        else:
            values=ge2_beta_atlas.segment_charge_many(*args,np.array([p[9] for p in ge2_pairs]),coefficient0_per_m=density)
        np.add.at(charge[:,2],np.array([p[0] for p in ge2_pairs]),values)
    if ge2_beta_atlas is not None:
        beta_charge_pairs=[p for p in ge2_fallback if physical_beta[p[1]] and not beta_tail[p[1]]
            and ge2_beta_atlas.supports(p[2],p[9],p[4],p[5])]
        if beta_charge_pairs:
            values=ge2_beta_atlas.segment_charge_many(np.array([p[2] for p in beta_charge_pairs]),
                np.array([p[3] for p in beta_charge_pairs]),np.array([p[4] for p in beta_charge_pairs]),
                np.array([p[5] for p in beta_charge_pairs]),np.array([p[9] for p in beta_charge_pairs]),
                coefficient0_per_m=np.array([p[6] for p in beta_charge_pairs]))
            # The underlying finite RTE approximation can itself give signed
            # weak segment contributions. Preserve those values, as the base
            # atlas/direct solver do; sign alone is not a failed beta expansion.
            valid=np.isfinite(values)
            negative=valid&(values<0)
            negative_beta_reconstructions=int(negative.sum())
            beta_nonfinite_fallbacks=int((~valid).sum())
            np.add.at(negative_beta_mass,np.array([p[0] for p in beta_charge_pairs])[negative],abs(values[negative]))
            np.add.at(charge[:,2],np.array([p[0] for p in beta_charge_pairs])[valid],values[valid])
            beta_charge_keys={(p[0],p[1]) for p,good in zip(beta_charge_pairs,valid) if good}
    timings['atlas_charge_seconds']=perf_counter()-stage
    if progress:progress('charge atlas pass',dict(q1_pairs=len(q1_pairs),ge2_pairs=len(ge2_pairs),
        q1_fallback_oms=len(q1_fallback),ge2_fallback_pairs=len(ge2_fallback)))
    stage=perf_counter()
    for k,(om,part) in enumerate(first_direct_charge):
        if reduced is None:
            q,_=first_scattering_charge_only(subset_kernel(local,[om]),part,config=prompt,**charge_options)
        else:
            eligible=part.beta>=first_cubature.metadata['beta_min']
            eligible &= np.isclose(-part.coefficient2*part.beta**2,part.coefficient0,rtol=1e-10,atol=0.)
            q=np.zeros(1)
            if eligible.any():
                q+=first_scattering_charge_only(subset_kernel(reduced,[om]),part.subset(eligible),config=prompt,**charge_options)[0]
            if not eligible.all():
                q+=first_scattering_charge_only(subset_kernel(local,[om]),part.subset(~eligible),config=prompt,**charge_options)[0]
        charge[om,1]+=q[0]
        first_fallback_charge[om]+=q[0]
        if progress and k%20==0:progress('q1 fallback',dict(done=k+1,total=len(first_direct_charge)))
    for row in ge2_fallback:
        om,i,rho,look,a,b,density,closest,clock,beta=row
        if beta_tail[i] or (om,i) in beta_charge_keys:continue
        charge[om,2]+=direct_ge2(rho,look,beta,clock,b,a,density,closest,False)[0].real
    if beta_tail.any():
        from .ge2_beta_response import transport_ge2_tail_batch
        tail_result=transport_ge2_tail_batch(kernel,pieces.subset(beta_tail),fallback_pack,
            cell_m=beta_cell_m,time_origin_ns=origins)
        charge[:,2]+=tail_result.charge_pe
        beta_batch_metadata=tail_result.metadata
    timings['fallback_charge_seconds']=perf_counter()-stage
    score=np.maximum(charge,0).sum(1)
    active=score>=kernel.config.threshold_pe
    compute=score*selection_margin>=kernel.config.threshold_pe
    if progress:progress('time selection',dict(active=int(active.sum()),computed=int(compute.sum())))
    stage=perf_counter()
    substage=perf_counter()
    q1_fallback_time_oms=0
    for om,part in q1_fallback:
        if compute[om]:
            exact,b,_=order1_integrals(subset_kernel(local,[om]),part,time_origin_ns=origins[[om]],config=prompt)
            if first_beta_atlas is not None:
                first_beta_refinement[om]=exact[0]-first_fallback_charge[om]
                charge[om,1]+=first_beta_refinement[om]
            bins[om,:,1]+=b[0]
            q1_fallback_time_oms+=1
    timings['q1_direct_time_seconds']=perf_counter()-substage
    substage=perf_counter()
    # Sort by radial bracket to load each time row only when needed.
    selected=[p for p in q1_pairs if compute[p[0]]]
    selected.sort(key=lambda p:p[2])
    time_fallbacks=0
    for om,i,rho,orientation,lo,hi,density,shift in selected:
        try:
            clock=(pieces.end_ns[i]-pieces.start_ns[i])/pieces.length_m[i]
            if abs(clock-alpha0)*(hi-lo)/2>1e-3:
                raise ValueError('time profile requires actual source clock')
            _,b=first_store.query(rho,orientation,lo,hi,edges-shift,coefficient0_per_m=density)
            bins[om,:,1]+=b
        except ValueError:
            z=float((det.positions_m[om]-pieces.start_m[i])@pieces.direction[i])
            part=split_source(pieces,[(i,z-hi,z-lo)])
            q,b,_=order1_integrals(subset_kernel(local,[om]),part,time_origin_ns=origins[[om]],config=prompt)
            oldq=first_store.query_charge(rho,orientation,lo,hi,coefficient0_per_m=density)
            charge[om,1]+=q[0]-oldq
            bins[om,:,1]+=b[0];time_fallbacks+=1
    timings['q1_atlas_time_seconds']=perf_counter()-substage
    substage=perf_counter()
    omega=ge2_store.omega_per_ns;spectra=np.zeros((n,len(omega)),complex)
    ge2_time_fallbacks=0
    pending=[]
    guard_wall=guard_cpu=query_wall=query_cpu=0.
    for row in sorted([p for p in ge2_pairs if compute[p[0]]],key=lambda p:p[2]):
        om,i,rho,look,a,b,density,closest,clock,beta=row
        shift=closest+(alpha2-clock)*(a+b)/2
        if abs(clock-alpha2)*(a-b)/2<=1e-3 and ge2_store.query_supported(rho,a,b,charge_only=False):
            wall,used=perf_counter(),process_time()
            spectra[om]+=ge2_store.segment(rho,look,a,b,alignment='scaled',closest_time_ns=shift,coefficient0_per_m=density)
            query_wall+=perf_counter()-wall;query_cpu+=process_time()-used
        else:
            wall,used=perf_counter(),process_time()
            oldq=(ge2_store.segment_charge(rho,look,a,b,coefficient0_per_m=density) if ge2_beta_atlas is None else
                ge2_beta_atlas.segment_charge_many(np.array([rho]),np.array([look]),np.array([a]),np.array([b]),
                    np.array([beta]),coefficient0_per_m=density)[0])
            if direct_ge2_quadrature=='filon':
                pending.append((om,(rho,look,beta,clock,b,a,density,closest),oldq))
            else:
                direct=direct_ge2(rho,look,beta,clock,b,a,density,closest,True)
                charge[om,2]+=direct[0].real-oldq;spectra[om]+=direct
            ge2_time_fallbacks+=1
            guard_wall+=perf_counter()-wall;guard_cpu+=process_time()-used
    timings.update(ge2_cached_query_wall_seconds=query_wall,ge2_cached_query_cpu_seconds=query_cpu,
        ge2_guard_direct_wall_seconds=guard_wall,ge2_guard_direct_cpu_seconds=guard_cpu)
    timings['ge2_atlas_time_seconds']=perf_counter()-substage
    substage=perf_counter()
    ge2_direct_time_pairs=0
    for row in ge2_fallback:
        om,i,rho,look,a,b,density,closest,clock,beta=row
        if compute[om]:
            if direct_ge2_quadrature=='filon':
                pending.append((om,(rho,look,beta,clock,b,a,density,closest),None))
            else:spectra[om]+=direct_ge2(rho,look,beta,clock,b,a,density,closest,True)
            ge2_direct_time_pairs+=1
    timings['ge2_direct_time_seconds']=perf_counter()-substage
    if pending:
        wall,used=perf_counter(),process_time()
        values=direct_ge2.many([p[1] for p in pending])
        for (om,_,oldq),value in zip(pending,values):
            spectra[om]+=value
            if oldq is not None:charge[om,2]+=value[0].real-oldq
        timings['ge2_batch_wall_seconds']=perf_counter()-wall
        timings['ge2_batch_cpu_seconds']=process_time()-used
    substage=perf_counter()
    indices=np.flatnonzero(compute)
    outomega=getattr(kernel.config,'readout_omega_per_ns',None)
    outomega=omega if outomega is None else np.asarray(outomega)
    if len(indices):
        values=CubicSpline(omega,spectra[indices].T,axis=0)(outomega)
        bins[indices,:,2]=inverse_bins(outomega,values,edges)
    bins[~compute,:,1:]=0
    timings['ge2_readout_seconds']=perf_counter()-substage
    active=np.maximum(charge,0).sum(1)>=kernel.config.threshold_pe
    timings['atlas_time_and_fallback_seconds']=perf_counter()-stage
    metadata=dict(primary_only=True,computed_orders=[0,1,'>=2'],secondaries_computed=False,
        source_steps=len(original),source_pieces=len(pieces),modules=n,active=int(active.sum()),
        time_computed_mask=compute.tolist(),timings=timings,elapsed_seconds=perf_counter()-begin,cpu_seconds=process_time()-cpu,
        q1_cached_pairs=len(q1_pairs),ge2_cached_pairs=len(ge2_pairs),q1_fallback_oms=len(q1_fallback),
        ge2_fallback_pairs=len(ge2_fallback),q1_time_fallback_pairs=time_fallbacks,
        ge2_time_fallback_pairs=ge2_time_fallbacks,
        ge2_direct_time_pairs=ge2_direct_time_pairs,
        ge2_direct_backend=direct_ge2.backend,ge2_direct_quadrature=direct_ge2_quadrature,
        ge2_direct_timings=direct_ge2.timings,
        ge2_row_io_statistics=getattr(ge2_store,'io_statistics',{}),ge2_row_io_log=getattr(ge2_store,'row_io_log',[]),
        q1_fallback_time_oms=q1_fallback_time_oms,
        beta_charge_batch=beta_batch_metadata,
        q1_charge_response_cubature=(None if first_cubature is None else first_cubature.metadata),
        q1_charge_gauss_order=int(q1_charge_gauss_order),q1_charge_panel_width=float(q1_charge_panel_width),
        q1_beta_charge_metadata=None if first_beta_atlas is None else first_beta_atlas.metadata,
        q1_beta_charge_requested=first_beta_requested,q1_beta_charge_active=first_beta_atlas is not None,
        q1_beta_charge_pairs=len(first_beta_pairs)-first_beta_failed,q1_beta_failed_pairs=first_beta_failed,
        q1_direct_charge_oms=len(first_direct_charge),q1_beta_selected_charge_refinement_pe=first_beta_refinement.tolist(),
        ge2_beta_charge_metadata=(None if ge2_beta_atlas is None else ge2_beta_atlas.metadata),
        ge2_beta_charge_requested=beta_charge_requested,ge2_beta_charge_active=ge2_beta_atlas is not None,
        ge2_beta_charge_pairs=len(beta_charge_keys),ge2_beta_negative_reconstructions=negative_beta_reconstructions,
        ge2_beta_negative_pair_mass_pe=negative_beta_mass.tolist(),ge2_beta_nonfinite_fallbacks=beta_nonfinite_fallbacks,
        ge2_direct_charge_pairs=sum(not beta_tail[p[1]] and (p[0],p[1]) not in beta_charge_keys for p in ge2_fallback),
        q1_omitted_bound_pe=bound_sum.tolist(),q1_tail_budget_pe=tail_budget_pe,
        negative_charge_counts=(charge<0).sum(0).tolist(),selection_margin=selection_margin,
        q1_time_row_loads=first_store.time_row_loads,ge2_time_row_loads=ge2_store.full_row_loads,
        atlas_resident_bytes=first_store.resident_bytes+ge2_store.resident_bytes+beta_charge_payload_bytes+
            first_beta_payload_bytes,
        limits=['>=2 inherits finite radial window and original RTE spectral/angular errors',
                'atlas interpolation is validated numerically, not a certified transport error bound',
                'time profiles outside the computed mask are not evaluated'])
    # Keep the actual sampled >=2 spectrum for composition with secondary
    # sources. It is expressed on the global clock, as HybridTransportResponse.
    absolute_spectrum=spectra.T*np.exp(1j*omega[:,None]*origins[None,:])
    return UniversalPrimaryResult(charge,bins,origins,edges,active,metadata,absolute_spectrum)
