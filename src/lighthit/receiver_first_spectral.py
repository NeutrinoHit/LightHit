"""Prepared one-band Q1 CDFs with a bounded, shared time-bin error budget.

Charges retain the full geometry path grid. Coarse time CDFs have measured
total-variation bounds against that grid; selected groups are folded on the
original geometry when the budget requires it. No photons are omitted.
"""
from dataclasses import dataclass,asdict
from functools import lru_cache
from pathlib import Path
import hashlib,json
import numpy as np
from .receiver_geometry import ReceiverGeometry
from .receiver_first_atlas import _optics


@dataclass(frozen=True)
class SpectralFirstEntryConfig:
    time_nodes: int = 65

    def __post_init__(self):
        if isinstance(self.time_nodes,(bool,np.bool_)) or not isinstance(self.time_nodes,(int,np.integer)) or self.time_nodes<2:
            raise ValueError('at least two integer time nodes required')


def _signature(geometry):
    data=dict(g=geometry.g,radii=geometry.radii_unit.tolist(),config=asdict(geometry.config),
              maximum=geometry.path_u_max,excess=geometry.max_excess_unit)
    return hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest()


@lru_cache(maxsize=16)
def _time_weights(nodes,maximum):
    u=np.linspace(0.,maximum,nodes);du=maximum/(nodes-1)
    return np.diff(np.sinh(u)),(du*np.sinh(u[1:])-np.diff(np.cosh(u)))/du


@dataclass
class SpectralFirstEntry:
    radii_m: np.ndarray
    geometry_indices: np.ndarray
    radius_m: float
    g: float
    tables: dict
    config: SpectralFirstEntryConfig
    hit_charge: np.ndarray
    hit_cdf: np.ndarray
    hit_error: np.ndarray
    miss_charge: np.ndarray
    miss_cdf: np.ndarray
    miss_error: np.ndarray
    metadata: dict

    @classmethod
    def build(cls,*,geometry,radius_m,tables,geometry_indices=None,config=None,progress=None):
        from ._receiver_band_first import node_paths,fold_row
        config=config or SpectralFirstEntryConfig();t=_optics(tables);R=float(radius_m)
        indices=(np.arange(len(geometry.radii_unit)) if geometry_indices is None else np.asarray(geometry_indices,np.int64))
        if (indices.ndim!=1 or len(indices)<2 or indices[0]<0 or indices[-1]>=len(geometry.radii_unit)
                or np.any(np.diff(indices)<=0) or not np.isfinite(R) or R<=0):raise ValueError('valid increasing geometry rows and radius required')
        reference=geometry.config.path_grid_nodes
        if (reference-1)%(config.time_nodes-1):raise ValueError('time grid must divide the geometry intervals')
        stride=(reference-1)//(config.time_nodes-1);radii=geometry.radii_unit[indices]*R
        position=np.arange(reference)/stride;low=np.minimum(position.astype(int),config.time_nodes-2);fraction=position-low
        data={};done=0
        for branch,count in [('hit',geometry.config.hit_nodes),('miss',geometry.config.miss_nodes)]:
            charge=np.empty((len(t['mut']),len(indices),count,10))
            cdf=np.empty((len(t['mut']),len(indices),count,config.time_nodes,10));error=np.empty_like(charge)
            raw=getattr(geometry,branch+'_cdf')
            for ir,original in enumerate(indices):
                r=radii[ir]
                for j in range(count):
                    path=node_paths(r,R,branch=='hit',j,count,geometry.g,geometry.path_u_max,reference,geometry.max_excess_unit)
                    fine=fold_row(raw[original,j],path,t['mut'],r-R)
                    coarse=fine[:,::stride]
                    reconstructed=np.take(coarse,low,axis=1)*(1-fraction)[None,:,None]+np.take(coarse,low+1,axis=1)*fraction[None,:,None]
                    charge[:,ir,j]=fine[:,-1];cdf[:,ir,j]=coarse
                    error[:,ir,j]=np.sum(abs(np.diff(fine-reconstructed,axis=1)),axis=1)
                    done+=1
                    if progress is not None:progress(done,len(indices)*(geometry.config.hit_nodes+geometry.config.miss_nodes))
            data[branch+'_charge']=charge;data[branch+'_cdf']=cdf;data[branch+'_error']=error
        metadata=dict(schema='lighthit/first-entry-spectral-q1/1',geometry_signature=_signature(geometry),
            time_u_max=geometry.path_u_max,reference_path_nodes=reference,convergence_verified=False,
            clock_interpolation='linear CDF in asinh path; positive uniform source-clock convolution')
        return cls(radii,indices,R,geometry.g,t,config,**data,metadata=metadata)

    def save(self,directory):
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
        for name in ('radii_m','geometry_indices','hit_charge','hit_cdf','hit_error','miss_charge','miss_cdf','miss_error'):
            np.save(directory/(name+'.npy'),getattr(self,name),allow_pickle=False)
        if hasattr(self,'_quad_coarse'):
            for branch in ('hit','miss'):
                for mode in ('coarse','fine'):np.save(directory/(branch+'_quad_'+mode+'.npy'),getattr(self,'_quad_'+mode)[branch],allow_pickle=False)
        self._save_metadata(directory)

    def _save_metadata(self,directory):
        record=dict(self.metadata,radius_m=self.radius_m,g=self.g,config=asdict(self.config),
                    tables={k:v.tolist() for k,v in self.tables.items()})
        (Path(directory)/'manifest.json').write_text(json.dumps(record,indent=2)+'\n')

    @classmethod
    def load(cls,directory):
        directory=Path(directory);m=json.loads((directory/'manifest.json').read_text())
        if m['schema']!='lighthit/first-entry-spectral-q1/1':raise ValueError('unknown spectral Q1 schema')
        arrays={name:np.load(directory/(name+'.npy'),mmap_mode='r',allow_pickle=False)
            for name in ('radii_m','geometry_indices','hit_charge','hit_cdf','hit_error','miss_charge','miss_cdf','miss_error')}
        result=cls(radius_m=m['radius_m'],g=m['g'],tables=_optics(m['tables']),
            config=SpectralFirstEntryConfig(**m['config']),metadata=m,**arrays)
        for branch in ('hit','miss'):
            q=getattr(result,branch+'_charge');cdf=getattr(result,branch+'_cdf');e=getattr(result,branch+'_error')
            if (q.ndim!=4 or q.shape[:2]!=(len(result.tables['mut']),len(result.radii_m)) or q.shape[-1]!=10
                    or cdf.shape!=q.shape[:3]+(result.config.time_nodes,10) or e.shape!=q.shape):
                raise ValueError('invalid spectral Q1 dimensions')
        if m.get('quadratic_time'):
            result._quad_coarse={};result._quad_fine={}
            for branch in ('hit','miss'):
                for mode in ('coarse','fine'):
                    value=np.load(directory/(branch+'_quad_'+mode+'.npy'),mmap_mode='r',allow_pickle=False)
                    if value.shape!=getattr(result,branch+'_charge').shape or not np.isfinite(value).all() or np.any(value<0):raise ValueError('invalid quadratic time bounds')
                    getattr(result,'_quad_'+mode)[branch]=value
        return result

    def prepare_quadratic(self,geometry,directory,*,progress=None):
        """Add deposition bounds without rewriting the prepared band CDFs."""
        from .receiver_time import quadratic_rule
        if _signature(geometry)!=self.metadata['geometry_signature']:raise ValueError('original geometry reference required')
        coarse=quadratic_rule(self.config.time_nodes,self.metadata['time_u_max'])
        fine=quadratic_rule(geometry.config.path_grid_nodes,self.metadata['time_u_max'])
        if not coarse[-1] or not fine[-1]:raise ValueError('time mesh is too coarse for positive quadratic panels')
        self._quad_coarse={};self._quad_fine={};done=0
        for branch in ('hit','miss'):
            cdf=getattr(self,branch+'_cdf');qc=np.empty_like(getattr(self,branch+'_charge'));qf=np.empty_like(qc)
            raw=getattr(geometry,branch+'_cdf')
            for ir,original in enumerate(self.geometry_indices):
                r=self.radii_m[ir]
                for j in range(cdf.shape[2]):
                    qc[:,ir,j]=np.sum(abs(np.diff(cdf[:,ir,j],axis=1))*coarse[3][None,:,None],axis=1)
                    # Every extinction panel multiplies a raw increment by
                    # a factor in [0,1]. Unattenuated geometry variation
                    # therefore bounds every band's fine deposition error.
                    qf[:,ir,j]=np.sum(abs(np.diff(raw[original,j],axis=0))*fine[3][:,None],axis=0)[None,:]
                done+=1
                if progress is not None:progress(done,2*len(self.radii_m))
            self._quad_coarse[branch]=qc;self._quad_fine[branch]=qf
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
        for branch in ('hit','miss'):
            for mode in ('coarse','fine'):np.save(directory/(branch+'_quad_'+mode+'.npy'),getattr(self,'_quad_'+mode)[branch],allow_pickle=False)
        self.metadata.update(quadratic_time=True,deposition_rule='positive quadratic CDF panels with total-variation bounds',
                             fine_deposition_bound='unattenuated geometry variation, conservative for every band')
        temporary=directory/'deposition-manifest'
        record=dict(self.metadata,radius_m=self.radius_m,g=self.g,config=asdict(self.config),tables={k:v.tolist() for k,v in self.tables.items()})
        temporary.write_text(json.dumps(record,indent=2)+'\n');temporary.replace(directory/'manifest.json')

    def band(self,tables):
        if len(tables['mut'])!=1:return None
        key=tuple(float(tables[k][0]) for k in ('mut','mus','speed'))
        cache=getattr(self,'_band_cache',None)
        if cache is None:self._band_cache={};cache=self._band_cache
        if key in cache:return cache[key]
        samples=np.stack([self.tables[k] for k in ('mut','mus','speed')],axis=1)
        found=np.flatnonzero(np.all(abs(samples-np.asarray(key))<=2e-9*abs(samples),axis=1))
        if len(found):cache[key]=int(found[0]);return int(found[0])
        return None

    def sum_groups(self,radii,angles,clocks,spreads,moments,time_edges_ns=(),*,tables,geometry,
                   relative_tolerance=.003,absolute_tolerance=0.,_prepared=None):
        from ._receiver_band_first import charge_bounds,sum_band_groups
        if getattr(self,'_geometry_id',None)!=id(geometry):
            if _signature(geometry)!=self.metadata['geometry_signature']:raise ValueError('spectral Q1 needs its original geometry reference')
            self._geometry_id=id(geometry)
        band=self.band(tables) if _prepared is None else _prepared[0]
        if band is None:raise ValueError('optical band is outside prepared spectral Q1 tables')
        t=_optics(tables) if _prepared is None else _prepared[1]
        r=np.asarray(radii,float);a=np.asarray(angles,float);c=np.asarray(clocks,float)
        w=np.asarray(spreads,float);m=np.asarray(moments,float);edges=np.asarray(time_edges_ns,float)
        if (r.ndim!=1 or a.shape!=r.shape or c.shape!=r.shape or w.shape!=r.shape or m.shape!=(len(r),2,10)
                or any(not np.isfinite(x).all() for x in (r,a,c,w,m)) or np.any(w<0) or np.any(a<0) or np.any(a>np.pi)):
            raise ValueError('finite group geometry, clocks and two-field moments required')
        if edges.ndim!=1 or not np.isfinite(edges).all() or (len(edges) and (len(edges)<2 or np.any(np.diff(edges)<=0))):raise ValueError('increasing finite time edges required')
        if not np.isfinite([relative_tolerance,absolute_tolerance]).all() or min(relative_tolerance,absolute_tolerance)<0:raise ValueError('nonnegative finite time tolerances required')
        mu,ms,v=[float(self.tables[k][band]) for k in ('mut','mus','speed')]
        s0,s2=[float(t[k][0]) for k in ('s0w','s2w')]
        has_quadratic=hasattr(self,'_quad_coarse')
        charges,bounds=charge_bounds(r,a,m,self.radius_m,self.g,np.log(self.radii_m),mu,ms,v,s0,s2,
            self.hit_charge[band],self.miss_charge[band],self.hit_error[band],self.miss_error[band])
        q=float(charges.sum());budget=absolute_tolerance+relative_tolerance*abs(q)
        if not len(edges):return q,np.empty(0),dict(fine_groups=0,coarse_groups=len(r),bin_l1_bound=0.,bin_l1_budget=budget)
        remaining=float(bounds.sum());fine=np.zeros(len(r),bool)
        quadratic=np.zeros(len(r),bool)
        if has_quadratic:
            from .receiver_time import time_levels,quadratic_rule
            cr=quadratic_rule(self.config.time_nodes,self.metadata['time_u_max'])
            fr=quadratic_rule(geometry.config.path_grid_nodes,self.metadata['time_u_max'])
            levels,remaining=time_levels(bounds+np.max(cr[3])*abs(charges),np.max(fr[3])*abs(charges),budget)
            fine=levels!=0;quadratic=levels!=2
        elif budget==0:fine[:]=True;remaining=0.
        elif remaining>budget:
            for j in np.argsort(bounds)[::-1]:
                fine[j]=True;remaining-=bounds[j]
                if remaining<=budget:break
        bins=np.empty(0);guards=0
        if len(edges):
            coarse_weights=_time_weights(self.config.time_nodes,self.metadata['time_u_max'])
            fine_weights=_time_weights(geometry.config.path_grid_nodes,self.metadata['time_u_max'])
            from .receiver_time import quadratic_rule
            if has_quadratic:
                cr=quadratic_rule(self.config.time_nodes,self.metadata['time_u_max']);fr=quadratic_rule(geometry.config.path_grid_nodes,self.metadata['time_u_max'])
                rule=(*cr[:3],*fr[:3],cr[3],fr[3])
            else:rule=(np.empty(0),)*8
            bins,guards=sum_band_groups(r,a,c,w,m,edges,fine,charges,self.radius_m,self.g,np.log(self.radii_m),mu,ms,v,s0,s2,
                self.metadata['time_u_max'],self.hit_cdf[band],self.miss_cdf[band],self.geometry_indices,
                geometry.hit_cdf,geometry.miss_cdf,geometry.radii_unit,geometry.max_excess_unit,*coarse_weights,*fine_weights,quadratic,*rule)
        return q,bins,dict(fine_groups=int(fine.sum()),coarse_groups=int((~fine).sum()),
            bin_l1_bound=max(0.,remaining) if len(edges) else 0.,bin_l1_budget=budget,
            reference_path_nodes=geometry.config.path_grid_nodes,coarse_time_nodes=self.config.time_nodes,
            quadratic_time=bool(has_quadratic),exact_groups=int(np.sum(fine&~quadratic))+guards,guard_refinements=int(guards))

    @classmethod
    def build_parallel(cls,*,geometry_directory,radius_m,tables,output,jobs=1,config=None,geometry_indices=None,progress=None,overwrite=False,keep_blocks=True):
        from concurrent.futures import ProcessPoolExecutor,as_completed
        from multiprocessing import get_context
        geometry=ReceiverGeometry.load(geometry_directory);config=config or SpectralFirstEntryConfig();tables=_optics(tables)
        indices=np.arange(len(geometry.radii_unit)) if geometry_indices is None else np.asarray(geometry_indices,np.int64)
        if indices.ndim!=1 or len(indices)<2 or np.any(np.diff(indices)<=0) or indices[0]<0 or indices[-1]>=len(geometry.radii_unit):raise ValueError('increasing geometry rows required')
        if isinstance(jobs,(bool,np.bool_)) or not isinstance(jobs,(int,np.integer)) or jobs<1:raise ValueError('positive integer jobs required')
        output=Path(output);output.mkdir(parents=True,exist_ok=True)
        if (output/'manifest.json').exists():
            if not overwrite:raise FileExistsError('completed spectral Q1 atlas exists')
            old=cls.load(output)
            from ._portable import same_values
            if (old.metadata['geometry_signature']!=_signature(geometry) or old.config!=config
                    or old.radius_m!=radius_m or not np.array_equal(old.geometry_indices,indices)
                    or not same_values(old.tables,tables)):
                raise ValueError('completed spectral Q1 has different inputs')
            return old
        from ._table_checkpoint import JoinedArrays,radial_blocks
        blocks=[indices[b] for b in radial_blocks(len(indices))];workers=min(jobs,len(blocks))
        arguments=[dict(geometry_directory=str(geometry_directory),radius_m=radius_m,tables=tables,
            geometry_indices=block,config=asdict(config),output=str(output/'blocks'/f'{i:04d}')) for i,block in enumerate(blocks)]
        shapes={}
        for branch,count in [('hit',geometry.config.hit_nodes),('miss',geometry.config.miss_nodes)]:
            shape=(len(tables['mut']),len(indices),count)
            for suffix,tail in [('charge',(10,)),('error',(10,)),('cdf',(config.time_nodes,10))]:shapes[branch+'_'+suffix]=shape+tail
        from . import _receiver_band_first
        builder=hashlib.sha256(Path(__file__).read_bytes()+Path(_receiver_band_first.__file__).read_bytes()).hexdigest()
        joined=JoinedArrays(output,shapes,dict(geometry=_signature(geometry),indices=indices.tolist(),radius=radius_m,
            config=asdict(config),tables={k:v.tolist() for k,v in tables.items()},builder=builder),axis=1)
        arrays=joined.arrays;completed=0;reference=None
        def merge(i):
            nonlocal completed,reference
            part=cls.load(arguments[i]['output']);locations=np.searchsorted(indices,part.geometry_indices)
            joined.merge(i,locations,{name:getattr(part,name) for name in arrays},metadata=part.metadata)
            reference=part.metadata;completed+=len(locations)
            if not keep_blocks:joined.discard(arguments[i]['output'])
            if progress is not None:progress(completed,len(indices))
        pending=[]
        for i,b in enumerate(blocks):
            locations=np.searchsorted(indices,b)
            if joined.contains(i,locations):reference=joined.metadata(i);completed+=len(b)
            else:pending.append(i)
        if workers==1:
            for i in pending:_spectral_block(arguments[i]);merge(i)
        else:
            with ProcessPoolExecutor(max_workers=workers,mp_context=get_context('spawn')) as pool:
                futures={pool.submit(_spectral_block,arguments[i]):i for i in pending}
                for f in as_completed(futures):f.result();merge(futures[f])
        for array in arrays.values():array.flush()
        np.save(output/'radii_m.npy',geometry.radii_unit[indices]*radius_m,allow_pickle=False)
        np.save(output/'geometry_indices.npy',indices,allow_pickle=False)
        atlas=cls(geometry.radii_unit[indices]*radius_m,indices,float(radius_m),geometry.g,tables,config,
                  **arrays,metadata=dict(reference,parallel_workers=workers))
        atlas._save_metadata(output)
        return cls.load(output)


def _spectral_block(arguments):
    from . import _receiver_band_first
    options=dict(arguments);output=Path(options.pop('output'));config=SpectralFirstEntryConfig(**options.pop('config'))
    geometry=ReceiverGeometry.load(options.pop('geometry_directory'))
    builder=hashlib.sha256(Path(__file__).read_bytes()+Path(_receiver_band_first.__file__).read_bytes()).hexdigest()
    if (output/'manifest.json').exists():
        old=SpectralFirstEntry.load(output)
        from ._portable import same_values
        if (old.metadata.get('builder_sha256')==builder and old.metadata['geometry_signature']==_signature(geometry)
                and np.array_equal(old.geometry_indices,options['geometry_indices']) and old.radius_m==options['radius_m']
                and old.config==config and same_values(old.tables,options['tables'])):return
    atlas=SpectralFirstEntry.build(geometry=geometry,config=config,**options)
    atlas.metadata['builder_sha256']=builder;atlas.save(output)
