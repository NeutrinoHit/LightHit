"""Unit-sphere Q1 geometry shared by wavelengths and physical OM radii.

This is a geometry CDF, not a spectrum-specific response. It retains the
initial-pencil hit/miss branches and the actual finite optical path. A
consumer folds scattering, extinction, yield, efficiency and clocks once.
"""
from dataclasses import dataclass,asdict
from pathlib import Path
import json,hashlib
import numpy as np
from scipy.interpolate import PchipInterpolator
from .receiver import SphereSurface
from .receiver_first import first_entry_profile
from .receiver_first_atlas import _geometry


@dataclass(frozen=True)
class ReceiverGeometryConfig:
    hit_nodes: int = 33
    miss_nodes: int = 97
    path_grid_nodes: int = 257
    path_nodes: int = 256
    path_order: int = 8
    scatter_order: int = 16

    def __post_init__(self):
        for key,value in asdict(self).items():
            if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)) or value<2:
                raise ValueError(f'{key} must be an integer >=2')


@dataclass
class ReceiverGeometry:
    radii_unit: np.ndarray
    g: float
    max_excess_unit: float
    config: ReceiverGeometryConfig
    path_u_max: float
    hit_cdf: np.ndarray
    hit_total: np.ndarray
    hit_entry: np.ndarray
    miss_cdf: np.ndarray
    miss_total: np.ndarray

    def __post_init__(self):
        r=np.asarray(self.radii_unit,float)
        if (r.ndim!=1 or len(r)<2 or not np.isfinite(r).all() or np.any(np.diff(r)<=0) or r[0]<=1
                or not np.isfinite(self.g) or abs(self.g)>=1
                or not np.isfinite(self.max_excess_unit) or self.max_excess_unit<=0
                or not np.isfinite(self.path_u_max)
                or self.path_u_max<np.arcsinh(r[-1]+self.max_excess_unit)):
            raise ValueError('invalid unit-sphere geometry contract')
        for branch,count in [('hit',self.config.hit_nodes),('miss',self.config.miss_nodes)]:
            for suffix,shape in [('cdf',(len(r),count,self.config.path_grid_nodes,10)),('total',(len(r),count,10))]:
                if getattr(self,branch+'_'+suffix).shape!=shape:
                    raise ValueError('invalid geometry array shape')
        if self.hit_entry.shape!=(len(r),self.config.hit_nodes,10):raise ValueError('invalid hit-entry shape')

    @classmethod
    def build(cls,*,radii_unit,g,max_excess_unit,config=None,path_u_max=None,progress=None):
        config=config or ReceiverGeometryConfig();radii=np.asarray(radii_unit,float)
        if (radii.ndim!=1 or len(radii)<2 or not np.isfinite(radii).all()
                or np.any(np.diff(radii)<=0) or radii[0]<=1):
            raise ValueError('increasing unit-sphere radii >1 required')
        if not np.isfinite(g) or abs(g)>=1 or not np.isfinite(max_excess_unit) or max_excess_unit<=0:
            raise ValueError('finite HG g and positive declared excess-path extent required')
        required=float(np.arcsinh(radii[-1]+max_excess_unit))
        maximum=required if path_u_max is None else float(path_u_max)
        if not np.isfinite(maximum) or maximum<required:raise ValueError('shared path grid too short')
        u=np.linspace(0.,maximum,config.path_grid_nodes);data={};done=0
        for branch,count in [('hit',config.hit_nodes),('miss',config.miss_nodes)]:
            cdf=np.zeros((len(radii),count,len(u),10));total=np.zeros((len(radii),count,10))
            entry=np.zeros_like(total)
            for ir,r in enumerate(radii):
                for j,x in enumerate(np.linspace(0.,1.,count)):
                    theta,cosine=_geometry(r,1.,branch,x)
                    extent=np.nextafter(np.sqrt((r-1)*(r+1)),np.inf) if branch=='hit' else r+max_excess_unit
                    profile=first_entry_profile(SphereSurface(1.),distance_m=r,direction_cosine=cosine,g=g,
                        max_path_m=extent,path_nodes=config.path_nodes,path_order=config.path_order,
                        scatter_order=config.scatter_order)
                    curve=PchipInterpolator(profile.path_m,profile.cumulative_m_inv,axis=0)
                    width=1+.5*r*abs(np.sin(theta))*(1-abs(g))/(1+abs(g))+.25*r*(1-np.cos(theta))
                    nodes=np.clip(r-1+width*np.sinh(u),profile.path_m[0],profile.path_m[-1])
                    cdf[ir,j]=curve(nodes)*r;total[ir,j]=profile.cumulative_m_inv[-1]*r
                    if branch=='hit':
                        impact=r*np.sqrt((1-cosine)*(1+cosine))
                        hit_path=r*cosine-np.sqrt(max(0.,1-impact*impact))
                        entry[ir,j]=curve(np.clip(hit_path,profile.path_m[0],profile.path_m[-1]))*r
                    done+=1
                    if progress is not None:progress(done,len(radii)*(config.hit_nodes+config.miss_nodes))
            data[branch+'_cdf']=cdf;data[branch+'_total']=total
            if branch=='hit':data['hit_entry']=entry
        return cls(radii.copy(),float(g),float(max_excess_unit),config,maximum,**data)

    def profile(self,distance_m,theta,surface,*,receiver=0):
        """Interpolated physical paths and ten cumulative moments in m^-1."""
        if not isinstance(surface,SphereSurface):raise ValueError('SphereSurface required')
        R=float(surface.radius_m if surface.radius_m.ndim==0 else surface.radius_m[receiver])
        r=float(distance_m)/R;theta=float(theta)
        if not self.radii_unit[0]<=r<=self.radii_unit[-1] or not 0<=theta<=np.pi:
            raise ValueError('source geometry outside unit-sphere atlas')
        cap=np.arcsin(1/r);hit=theta<=cap;branch='hit' if hit else 'miss'
        if hit:x=2/np.pi*np.arcsin(np.clip(r*np.sin(theta),0.,1.))
        else:x=np.arcsinh(r*(theta-cap))/np.arcsinh(r*(np.pi-cap))
        grid=np.log(self.radii_unit);i=np.clip(np.searchsorted(grid,np.log(r))-1,0,len(grid)-2)
        fr=np.clip((np.log(r)-grid[i])/(grid[i+1]-grid[i]),0.,1.)
        values=getattr(self,branch+'_cdf');z=x*(values.shape[1]-1);j=min(int(z),values.shape[1]-2);fa=z-j
        def mix(a):return ((1-fr)*((1-fa)*a[i,j]+fa*a[i,j+1])+fr*((1-fa)*a[i+1,j]+fa*a[i+1,j+1]))/r
        cumulative=mix(values);total=mix(getattr(self,branch+'_total'))
        width=1+.5*r*abs(np.sin(theta))*(1-abs(self.g))/(1+abs(self.g))+.25*r*(1-np.cos(theta))
        paths=r-1+width*np.sinh(np.linspace(0.,self.path_u_max,self.config.path_grid_nodes))
        extent=np.sqrt((r-1)*(r+1)) if hit else r+self.max_excess_unit
        interior=paths<extent;paths=paths[interior];cumulative=cumulative[interior]
        paths=np.r_[paths,extent];cumulative=np.concatenate((cumulative,total[None,:]),axis=0)
        if hit:
            impact=r*np.sin(theta);entry=r*np.cos(theta)-np.sqrt(max(0.,1-impact*impact))
            if paths[0]<entry<paths[-1]:
                index=np.searchsorted(paths,entry)
                paths=np.insert(paths,index,entry)
                cumulative=np.insert(cumulative,index,mix(self.hit_entry),axis=0)
        # Physical scaling can join neighbouring unit nodes into one float.
        physical=paths*R
        if np.any(np.diff(physical)==0):
            _,first=np.unique(physical,return_index=True);keep=np.r_[first[1:]-1,len(paths)-1];keep[0]=0
            physical,cumulative=physical[keep],cumulative[keep]
        return physical,cumulative/R

    def save(self,directory):
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
        for name in ('radii_unit','hit_cdf','hit_total','hit_entry','miss_cdf','miss_total'):
            np.save(directory/(name+'.npy'),getattr(self,name),allow_pickle=False)
        metadata=dict(schema='lighthit/first-entry-geometry/1',g=self.g,max_excess_unit=self.max_excess_unit,
            config=asdict(self.config),path_u_max=self.path_u_max,convergence_verified=False)
        (directory/'manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')

    def fold(self,distance_m,theta,surface,*,extinction_per_m,scattering_per_m,
             speed_m_per_ns,time_edges_ns=(),clock_width_ns=0.,receiver=0):
        """One wavelength's charge and CDF, before yield, acceptance and area.

        The explicit returned tail bound applies to each of the ten moments.
        This geometry atlas supports partial spectra and offline primary-track
        tables; the common shower spectrum uses a pre-folded time atlas.
        """
        from ._receiver_geometry_fold import fold_coefficients
        mu=float(extinction_per_m);ms=float(scattering_per_m);v=float(speed_m_per_ns)
        times=np.asarray(time_edges_ns,float);width=float(clock_width_ns)
        if (not np.isfinite([mu,ms,v,width]).all() or mu<=0 or not 0<=ms<=mu or v<=0 or width<0):
            raise ValueError('positive extinction/speed, valid scattering and nonnegative clock width required')
        if times.ndim!=1 or not np.isfinite(times).all() or (len(times) and (len(times)<2 or np.any(np.diff(times)<=0))):
            raise ValueError('increasing finite time edges required')
        path,cdf=self.profile(distance_m,theta,surface,receiver=receiver)
        coefficients=np.ascontiguousarray(PchipInterpolator(path,cdf,axis=0).c)
        q,f=fold_coefficients(path,coefficients,mu,ms,v,times,width)
        R=float(surface.radius_m if surface.radius_m.ndim==0 else surface.radius_m[receiver])
        cap=np.arcsin(R/distance_m)
        phase_max=(1+abs(self.g))/(4*np.pi*(1-abs(self.g))**2)
        tail=0. if theta<=cap else ms*2*np.pi*phase_max/R*np.exp(-mu*path[-1])
        return q,f,float(tail)

    def sum_groups(self,radii,angles,clocks,spreads,moments,time_edges_ns=(),*,surface,tables):
        """Fold partial spectra after scalar source/acceptance contraction.

        This preserves the positive scalar geometry CDF before cubic path
        interpolation. Its discretisation is checked separately from the
        moment-by-moment readable ``fold`` reference.
        """
        from ._receiver_geometry_lookup import sum_geometry_groups
        from .receiver_first_atlas import _optics
        t=_optics(tables);r=np.asarray(radii,float);a=np.asarray(angles,float)
        c=np.asarray(clocks,float);w=np.asarray(spreads,float);m=np.asarray(moments,float)
        edges=np.asarray(time_edges_ns,float)
        if (r.ndim!=1 or a.shape!=r.shape or c.shape!=r.shape or w.shape!=r.shape or m.shape!=(len(r),2,10)
                or any(not np.isfinite(x).all() for x in (r,a,c,w,m)) or np.any(w<0) or np.any(a<0) or np.any(a>np.pi)):
            raise ValueError('finite group geometry, clocks and two-field moments required')
        if not isinstance(surface,SphereSurface) or surface.radius_m.ndim!=0:
            raise ValueError('one explicit sphere radius required')
        if edges.ndim!=1 or not np.isfinite(edges).all() or (len(edges) and (len(edges)<2 or np.any(np.diff(edges)<=0))):
            raise ValueError('increasing finite time edges required')
        R=float(surface.radius_m)
        return sum_geometry_groups(r,a,c,w,m,edges,R,self.g,np.log(self.radii_unit*R),self.path_u_max,
            self.max_excess_unit,self.hit_cdf,self.hit_total,self.hit_entry,self.miss_cdf,self.miss_total,
            t['mut'],t['mus'],t['speed'],t['s0w'],t['s2w'])

    @classmethod
    def build_parallel(cls,*,radii_unit,g,max_excess_unit,output,config=None,jobs=1,overwrite=False,progress=None,keep_blocks=True):
        """Build independent radial blocks on a common declared path grid."""
        from concurrent.futures import ProcessPoolExecutor,as_completed
        from multiprocessing import get_context
        config=config or ReceiverGeometryConfig();r=np.asarray(radii_unit,float)
        if r.ndim!=1 or len(r)<2 or np.any(np.diff(r)<=0) or r[0]<=1 or not np.isfinite(r).all():
            raise ValueError('increasing unit-sphere radii >1 required')
        if isinstance(jobs,(bool,np.bool_)) or not isinstance(jobs,(int,np.integer)) or jobs<1:
            raise ValueError('positive integer worker count required')
        output=Path(output);output.mkdir(parents=True,exist_ok=True)
        manifest=output/'manifest.json'
        if manifest.exists():
            if not overwrite:raise FileExistsError(str(manifest))
            if not (output/'build-progress.json').exists():
                raise ValueError('legacy completed atlas needs a new output directory')
        from ._table_checkpoint import JoinedArrays,radial_blocks
        maximum=float(np.arcsinh(r[-1]+max_excess_unit));blocks=radial_blocks(len(r));workers=min(jobs,len(blocks))
        shapes={}
        for branch,count in [('hit',config.hit_nodes),('miss',config.miss_nodes)]:
            shapes[branch+'_cdf']=(len(r),count,config.path_grid_nodes,10)
            shapes[branch+'_total']=(len(r),count,10)
        shapes['hit_entry']=(len(r),config.hit_nodes,10)
        joined=JoinedArrays(output,shapes,dict(radii=r.tolist(),g=g,excess=max_excess_unit,
            config=asdict(config),maximum=maximum,builder=_fingerprint()))
        arrays=joined.arrays
        if manifest.exists():manifest.unlink()
        arguments=[dict(radii_unit=r[b],g=g,max_excess_unit=max_excess_unit,config=asdict(config),
            path_u_max=maximum,output=str(output/'blocks'/f'{i:04d}')) for i,b in enumerate(blocks)]
        complete=sum(len(blocks[i]) for i in range(len(blocks)) if joined.contains(i,blocks[i]))
        def merge(i):
            nonlocal complete
            block=cls.load(arguments[i]['output'])
            joined.merge(i,blocks[i],{name:getattr(block,name) for name in arrays})
            if not keep_blocks:joined.discard(arguments[i]['output'])
            complete+=len(block.radii_unit)
            if progress is not None:progress(complete,len(r))
        pending=[i for i in range(len(blocks)) if not joined.contains(i,blocks[i])]
        if workers==1:
            for i in pending:_geometry_block(arguments[i]);merge(i)
        else:
            with ProcessPoolExecutor(max_workers=workers,mp_context=get_context('spawn')) as pool:
                futures={pool.submit(_geometry_block,arguments[i]):i for i in pending}
                for f in as_completed(futures):f.result();merge(futures[f])
        for a in arrays.values():a.flush()
        # save() writes arrays too; only the owner writes the joined manifest.
        np.save(output/'radii_unit.npy',r,allow_pickle=False)
        metadata=dict(schema='lighthit/first-entry-geometry/1',g=float(g),max_excess_unit=float(max_excess_unit),
            config=asdict(config),path_u_max=maximum,convergence_verified=False,builder_sha256=_fingerprint())
        manifest.write_text(json.dumps(metadata,indent=2)+'\n')
        return cls.load(output)

    @classmethod
    def load(cls,directory,*,mmap=True):
        directory=Path(directory);metadata=json.loads((directory/'manifest.json').read_text())
        if metadata['schema']!='lighthit/first-entry-geometry/1':raise ValueError('unknown geometry atlas schema')
        arrays={name:np.load(directory/(name+'.npy'),allow_pickle=False,mmap_mode='r' if mmap else None)
                for name in ('radii_unit','hit_cdf','hit_total','hit_entry','miss_cdf','miss_total')}
        return cls(g=metadata['g'],max_excess_unit=metadata['max_excess_unit'],
            config=ReceiverGeometryConfig(**metadata['config']),path_u_max=metadata['path_u_max'],**arrays)


def _fingerprint():
    from . import receiver_first,_receiver_first_numba,receiver_first_atlas
    digest=hashlib.sha256(Path(__file__).read_bytes())
    for m in (receiver_first,_receiver_first_numba,receiver_first_atlas):digest.update(Path(m.__file__).read_bytes())
    return digest.hexdigest()


def _geometry_block(arguments):
    options=dict(arguments);output=Path(options.pop('output'));config=ReceiverGeometryConfig(**options.pop('config'))
    fingerprint=_fingerprint()
    if (output/'manifest.json').exists():
        metadata=json.loads((output/'manifest.json').read_text());stored=ReceiverGeometry.load(output)
        if (metadata.get('builder_sha256')==fingerprint and config==stored.config
                and np.array_equal(stored.radii_unit,options['radii_unit'])
                and stored.g==options['g'] and stored.max_excess_unit==options['max_excess_unit']
                and stored.path_u_max==options['path_u_max']):return
    atlas=ReceiverGeometry.build(**options,config=config);atlas.save(output)
    p=output/'manifest.json';metadata=json.loads(p.read_text());metadata['builder_sha256']=fingerprint
    p.write_text(json.dumps(metadata,indent=2)+'\n')
