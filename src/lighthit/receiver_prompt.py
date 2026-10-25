"""First-entry Q1 on the compressed, screened shower route.

The common spectrum uses prepared time CDFs; partial spectra use a shared
geometry CDF with analytic extinction folding. Source compression, emission
mass and the coarse/fine OM selection are retained.
"""
from dataclasses import replace
from pathlib import Path
from time import perf_counter
import numpy as np
from .receiver import SphereSurface


class FirstEntryPrompt:
    def __init__(self,kernel):
        from .receiver_first_atlas import FirstEntryAtlas
        from .receiver_first_adaptive import AdaptiveFirstEntry
        from .receiver_geometry import ReceiverGeometry
        from .prompt import _spectral_tables
        directory=kernel.config.first_entry_tables
        if directory is None:raise FileNotFoundError('first-entry Q1 needs a prepared first_entry_tables directory')
        self.directory=Path(directory);self.surface=SphereSurface(kernel._receiver_contract()['radius_m'])
        self.fine=FirstEntryAtlas.load(self.directory/'q1-reference')
        self.fine.validate(tables=_spectral_tables(kernel),surface=self.surface,g=kernel.medium.g)
        self.adaptive=AdaptiveFirstEntry.load(self.directory/'q1-adaptive',fine=self.fine)
        self.geometry=ReceiverGeometry.load(self.directory/'geometry')
        self.spectral=None
        if (self.directory/'q1-bands/manifest.json').exists():
            from .receiver_first_spectral import SpectralFirstEntry,_signature
            self.spectral=SpectralFirstEntry.load(self.directory/'q1-bands')
            if self.spectral.radius_m!=self.fine.radius_m or self.spectral.g!=self.fine.g:
                raise ValueError('spectral Q1 tables belong to another receiver geometry')
            if _signature(self.geometry)!=self.spectral.metadata['geometry_signature']:
                raise ValueError('spectral Q1 needs its original geometry reference')
            self.spectral._geometry_id=id(self.geometry)
        if not np.isclose(self.geometry.g,kernel.medium.g,rtol=0,atol=1e-12):
            raise ValueError('geometry CDF belongs to another phase function')

    def screen(self,pencils,positions,looks,origins,targets,polynomial,config,tables):
        """Same charge as individual coarse queries, evaluated in OM batches."""
        from ._portable import same_values
        from ._prompt_numba import _prepare_gauss_inverse,_GAUSS_TAU,_GAUSS_W
        from ._receiver_screen import screen_batch
        from .receiver_first_atlas import _optics,_theta_rule
        from .prompt import _thread_count
        from concurrent.futures import ThreadPoolExecutor
        common=all(same_values(tables[key],self.fine.tables[key]) for key in self.fine.tables)
        band=None if self.spectral is None else self.spectral.band(tables)
        if not common and band is None:return None
        polynomial=np.pad(np.asarray(polynomial,float),(0,4-len(polynomial)))
        if (not all(np.isfinite(a).all() for a in (positions,looks,origins,polynomial))
                or not np.allclose(np.linalg.norm(looks,axis=1),1.,rtol=0,atol=1e-12)):
            raise ValueError('finite receiver geometry and unit propagation looks required')
        nodes,weights=_prepare_gauss_inverse(_GAUSS_TAU,_GAUSS_W)
        thresholds,theta_lut,theta_first=_theta_rule(float(config.merge_theta_min),.3)
        source=(pencils['position'],pencils['direction'],pencils['w0'],pencils['w2'],pencils['time'],
            pencils['major'],pencils['var_major'],pencils['var_minor'],pencils['_first_entry_shape'],polynomial,
            self.fine.radius_m,float(config.merge_log_r),1e300,float(config.merge_theta_min),.3,
            self.fine.config.hit_nodes,self.fine.config.miss_nodes,float(np.max(self.fine.tables['speed'])),
            nodes,weights,1/float(config.smoothing_factor)**2,None,True,
            pencils.get('hierarchy_parent'),pencils.get('next_node'),float(config.smoothing_factor),
            thresholds,theta_lut,theta_first)
        if common:
            from ._receiver_first_lookup import charge_groups
            a=self.adaptive.coarse;t=a.tables;evaluate=charge_groups
            lookup=(np.log(a.radii_m),a.radius_m,a.g,t['mut'],t['mus'],t['s0w'],t['s2w'],
                float(np.max(t['speed'])),a.hit_charge,a.miss_charge)
        else:
            from ._receiver_band_first import charge_only
            a=self.spectral;t=_optics(tables);evaluate=charge_only
            lookup=(a.radius_m,a.g,np.log(a.radii_m),float(a.tables['mut'][band]),float(a.tables['mus'][band]),
                float(a.tables['speed'][band]),float(t['s0w'][0]),float(t['s2w'][0]),a.hit_charge[band],a.miss_charge[band])
        def block(ids):return screen_batch(positions,looks,origins,ids,source,evaluate,lookup)
        count=min(len(targets),_thread_count())
        if count==0:return np.empty(0),np.empty(0,dtype=int),np.empty(0,dtype=int)
        chunks=np.array_split(np.asarray(targets,np.int64),count)
        try:
            if count==1:return block(chunks[0])
            with ThreadPoolExecutor(max_workers=count) as pool:parts=list(pool.map(block,chunks))
            return tuple(np.concatenate([p[j] for p in parts]) for j in range(3))
        except ValueError as error:
            # A rare ill-conditioned radial rule retains the existing full
            # cubature refinement, with its original geometry/error checks.
            if not str(error).startswith('finite radial quadrature'):raise
            return None

    def optical_unit(self,tables):
        """Validate this immutable evaluation unit once, before its OM loop."""
        from ._portable import same_values
        common=all(same_values(tables[key],self.fine.tables[key]) for key in self.fine.tables)
        band=None if common or self.spectral is None else self.spectral.band(tables)
        if band is None:return common,None
        from .receiver_first_atlas import _optics
        return common,(band,_optics(tables))

    def evaluate(self,pencils,receiver,look,origin,polynomial,config,tables,edges,*,coarse=False,absolute_time_budget=0.,_unit=None):
        merge=replace(config,merge_tau_ns=1e300,merge_theta_ratio=.3) if coarse else config
        stage=perf_counter()
        groups,metadata=self.fine.pencil_groups(pencils,receiver,look,polynomial,merge,origin_ns=origin,
            covariance_tolerance=1/(float(config.smoothing_factor)**2),radial_only=coarse)
        metadata['group_seconds']=perf_counter()-stage;stage=perf_counter()
        common,prepared=self.optical_unit(tables) if _unit is None else _unit
        if common:
            q,b,numerics=self.adaptive.sum_groups(*groups[:5],edges,relative_tolerance=.002)
            metadata.update(numerics)
            metadata['full_spectrum']=True
        elif prepared is not None:
            q,b,numerics=self.spectral.sum_groups(*groups[:5],edges,tables=tables,geometry=self.geometry,
                relative_tolerance=.003,absolute_tolerance=absolute_time_budget,_prepared=prepared)
            metadata.update(numerics,partial_spectrum=True,prepared_band=True)
        else:
            q,b,tail=self.geometry.sum_groups(*groups[:5],edges,surface=self.surface,tables=tables)
            metadata.update(partial_spectrum=True,absolute_tail_bound=tail)
        metadata['lookup_seconds']=perf_counter()-stage
        return q,b,metadata


def grouped_order1(kernel,elements,segments,positions,looks,areas,origins,work,tables,
                   polynomial,edges,config,order0_charge,*,coarse_only=False,selection_extra_charge=None,reuse_geometry=False):
    """Finite-kernel coarse/fine passes; no event normalization is fitted."""
    from .prompt import compress,_thread_count
    from ._prompt_screen import compress_screen
    from concurrent.futures import ThreadPoolExecutor
    backend=getattr(kernel,'_first_entry_prompt',None)
    if backend is None:backend=FirstEntryPrompt(kernel);kernel._first_entry_prompt=backend
    optical_unit=None if coarse_only else backend.optical_unit(tables)
    n=len(positions);charge=np.zeros(n);bins=np.zeros((n,len(edges)-1));level=np.zeros(n,np.int8)
    common_charge=getattr(kernel,'_first_entry_common_charge',None)
    if common_charge is None or len(common_charge)!=n:
        common_charge=np.zeros(n);kernel._first_entry_common_charge=common_charge
    timings={};statistics={};groups_total=0
    def apply(pencils,targets,coarse):
        from ._prompt_numba import _prepare_pencil_shape
        if '_first_entry_shape' not in pencils:
            pencils['_first_entry_shape']=_prepare_pencil_shape(pencils['direction'],pencils['major'],pencils['pos_cov'])
        if reuse_geometry and not coarse:
            from ._receiver_first_groups import share_pencil_geometry
            share_pencil_geometry(pencils)
        if coarse:
            start=perf_counter();result=backend.screen(pencils,positions,looks,origins,targets,polynomial,config,tables)
            if result is not None:
                timings['order1_screen_group_lookup_seconds']=perf_counter()-start
                q,counts,samples=result
                return [(int(index),q[j]*areas[index],np.empty(0),dict(groups=int(counts[j]),
                    covariance_samples=int(samples[j]),group_seconds=0.,lookup_seconds=0.,batched_screen=True))
                    for j,index in enumerate(targets)]
        def one(index):
            budget=(.001*abs(common_charge[index])/len(kernel.wavelength.wavelength_nm)/areas[index]
                    if not coarse and areas[index]>0 else 0.)
            q,b,meta=backend.evaluate(pencils,positions[index],looks[index],float(origins[index]),
                polynomial,config,tables,np.empty(0) if coarse else edges,coarse=coarse,absolute_time_budget=budget,_unit=optical_unit)
            if not coarse and meta.get('full_spectrum'):common_charge[index]=q*areas[index]
            if 'bin_l1_bound' in meta:
                meta['bin_l1_bound_pe']=meta['bin_l1_bound']*areas[index]
                meta['bin_l1_budget_pe']=meta.get('bin_l1_budget',0.)*areas[index]
            return int(index),q*areas[index],b*areas[index],meta
        workers=min(len(targets),_thread_count())
        if workers>1:
            with ThreadPoolExecutor(max_workers=workers) as pool:return list(pool.map(one,targets))
        return [one(index) for index in targets]
    stage=perf_counter();screen=coarse_only or (config.screen and kernel.config.threshold_pe>0 and len(work)>0)
    fine=work
    if screen:
        pencils=compress_screen(elements,segments,config)
        timings['order1_screen_compress_seconds']=perf_counter()-stage;stage=perf_counter()
        for index,q,_,meta in apply(pencils,work,True):
            charge[index]=q;level[index]=1;groups_total+=meta['groups']
            timings['order1_screen_group_seconds']=timings.get('order1_screen_group_seconds',0.)+meta['group_seconds']
            timings['order1_screen_lookup_seconds']=timings.get('order1_screen_lookup_seconds',0.)+meta['lookup_seconds']
        timings['order1_screen_eval_seconds']=perf_counter()-stage
        if not coarse_only:
            extra=np.zeros(n) if selection_extra_charge is None else np.asarray(selection_extra_charge,float)
            fine=work[(order0_charge[work]+extra[work]+config.screen_margin*np.maximum(charge[work],0.))>=kernel.config.threshold_pe]
    coarse_charge=charge.copy()
    if not coarse_only and len(fine):
        stage=perf_counter();pencils=compress(elements,segments,config)
        timings['order1_compress_seconds']=perf_counter()-stage;stage=perf_counter()
        for index,q,b,meta in apply(pencils,fine,False):
            charge[index]=q;bins[index]=b;level[index]=2;statistics[str(index)]=meta;groups_total+=meta['groups']
        timings['order1_eval_seconds']=perf_counter()-stage
        if screen:
            ratio=coarse_charge[fine]/np.maximum(charge[fine],1e-300);worst=float(ratio.min()) if len(ratio) else 1.
            if worst*config.screen_margin<1.:
                wider=1.5/max(worst,1e-3)
                candidates=work[(level[work]==1)&(order0_charge[work]+extra[work]+wider*np.maximum(charge[work],0.)>=kernel.config.threshold_pe)]
                stage=perf_counter()
                for index,q,b,meta in apply(pencils,candidates,False):
                    charge[index]=q;bins[index]=b;level[index]=2;statistics[str(index)]=meta;groups_total+=meta['groups']
                timings['order1_eval_seconds']+=perf_counter()-stage
    return charge,bins,dict(timings=timings,order1_level=level,order1_batch_omission=np.zeros((n,2)),
        order1_groups_total=groups_total,order1_receiver_reference='first-entry',order1_surface_numerics=statistics,
        order1_screen=dict(enabled=screen,fine_modules=int(np.sum(level==2)),coarse_only_modules=int(np.sum(level==1))),
        order1_backward_profile='finite sphere CDF with full arrival moments')
