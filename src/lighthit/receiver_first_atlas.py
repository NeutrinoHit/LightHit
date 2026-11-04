"""Reusable physical-time Q1 tables for a spherical first-entry receiver.

Geometry profiles are folded once over the optical spectrum. The two
Cherenkov fields, ten direction moments, charge and cumulative time mass
remain separate. Event queries interpolate; they launch no ray integral.
Hitting and missing initial pencils have separate interpolation branches.
"""
from dataclasses import dataclass,asdict
from pathlib import Path
from time import perf_counter
from functools import lru_cache
from threading import Lock
import hashlib,json,math
import numpy as np
from .receiver import SphereSurface
from .receiver_first import first_entry_profile
from .receiver_first_fold import fold_first_entry

_TIME_LAYOUT_LOCK=Lock()


@lru_cache(maxsize=8)
def _normal_rule(order):
    nodes,weights=np.polynomial.hermite.hermgauss(order)
    return nodes*np.sqrt(2.),weights/np.sqrt(np.pi)


@lru_cache(maxsize=8)
def _theta_rule(minimum,ratio):
    from ._prompt_numba import build_theta_thresholds,build_theta_bit_lut
    step=math.log1p(ratio);bins=2+int(math.log(math.pi/minimum)/step)
    thresholds=build_theta_thresholds(minimum,step,bins)
    lut,first=build_theta_bit_lut(minimum,thresholds)
    return thresholds,lut,first


@dataclass(frozen=True)
class FirstEntryAtlasConfig:
    hit_nodes: int = 33
    miss_nodes: int = 97
    time_nodes: int = 257
    path_nodes: int = 128
    path_order: int = 8
    scatter_order: int = 16
    spectral_order: int = 8
    tail_relative_bound: float = 1e-8

    def __post_init__(self):
        for key,value in asdict(self).items():
            if key=='tail_relative_bound':continue
            if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)) or value<2:
                raise ValueError(f'{key} must be an integer >=2')
        if not np.isfinite(self.tail_relative_bound) or not 0<self.tail_relative_bound<1:
            raise ValueError('tail_relative_bound must be in (0,1)')


def _optics(tables):
    keys=('mut','mus','speed','s0w','s2w')
    result={k:np.asarray(tables[k],float) for k in keys}
    n=len(result['mut'])
    if any(v.shape!=(n,) or not np.isfinite(v).all() for v in result.values()):
        raise ValueError('matching finite one-dimensional spectral tables required')
    if (not n or np.any(result['mut']<=0) or np.any(result['mus']<0)
            or np.any(result['mus']>result['mut']) or np.any(result['speed']<=0)
            or np.any(result['s0w']<0) or np.any(result['s2w']<0)):
        raise ValueError('invalid optical coefficients or two-field weights')
    return result


def _scale(r,R,tables):
    weights=np.stack((tables['s0w'],tables['s2w']),axis=-1)
    return np.sum(weights*tables['mus'][:,None]*np.exp(-tables['mut'][:,None]*(r-R)),axis=0)/r


def _path_extent(r,R,g,tables,relative):
    weights=np.stack((tables['s0w'],tables['s2w']),axis=-1)
    a=abs(g);phase_min=(1-a)/(4*np.pi*(1+a)**2)
    phase_max=(1+a)/(4*np.pi*(1-a)**2)
    # First scatter in [0,(r-R)/2] gives a positive lower bound: the ray has
    # not yet entered the glass, D<=1.5*r and total path<=2*r-R.
    lower=np.sum(weights*tables['mus'][:,None]*np.exp(-tables['mut'][:,None]*(2*r-R)),axis=0)
    lower*=phase_min*(r-R)/(4.5*r*r)
    upper=np.sum(weights*tables['mus'][:,None],axis=0)*(2*np.pi*phase_max/R)
    used=upper>0
    if not np.any(used):return r+R
    if np.any(lower[used]==0):
        raise FloatingPointError('Q1 tail lower bound underflows; narrow the atlas range or use logarithmic bounds')
    return max(r,np.max(np.log(upper[used]/(relative*lower[used])))/np.min(tables['mut']))


def _geometry(r,R,branch,x):
    cap=np.arcsin(R/r)
    if branch=='hit':
        b=np.sin(np.pi*x/2);theta=np.arcsin(b*R/r)
    else:
        theta=cap+R/r*np.sinh(x*np.arcsinh(r/R*(np.pi-cap)))
    cosine=float(np.cos(theta))
    if (branch=='hit' and x==1.) or (branch=='miss' and x==0.):
        # Keep the two one-sided limits separate, including roundoff in the
        # normalized geometry used by first_entry_profile.
        normalized=r/R
        impact=1. if branch=='hit' else np.nextafter(1.,np.inf)
        cosine=(np.sqrt(max(0.,(normalized-impact)*(normalized+impact)))/normalized
                if impact<normalized else 0.)
        while True:
            hit=cosine>0 and normalized*np.sqrt((1-cosine)*(1+cosine))<=1
            if hit==(branch=='hit'):break
            impact=np.nextafter(impact,-np.inf if branch=='hit' else np.inf)
            # At large r/R, millions of impact ulps can map to the same
            # cosine. Near r/R=1 the converse holds. Advance in whichever
            # floating coordinate resolves the required one-sided limit.
            candidate=(np.sqrt(max(0.,(normalized-impact)*(normalized+impact)))/normalized
                       if impact<normalized else 0.)
            angular=np.nextafter(cosine,np.inf if branch=='hit' else -np.inf)
            cosine=max(candidate,angular) if branch=='hit' else min(candidate,angular)
        theta=float(np.arccos(cosine))
    return theta,cosine


def _time_map(r,R,theta,extent,tables,g):
    fastest=float(np.max(tables['speed']));slowest=float(np.min(tables['speed']))
    front=(r-R)/fastest
    hg_width=(1-abs(g))/(1+abs(g))
    width=(R+.5*r*abs(np.sin(theta))*hg_width+.25*r*(1-np.cos(theta)))/fastest
    end=np.arcsinh((extent/slowest-front)/width)
    return front,width,end


def _coordinates(r,R,theta,branch):
    cap=np.arcsin(R/r)
    if branch=='hit':return 2/np.pi*np.arcsin(np.clip(r*np.sin(theta)/R,0.,1.))
    return np.arcsinh(r/R*(theta-cap))/np.arcsinh(r/R*(np.pi-cap))


@dataclass
class FirstEntryAtlas:
    radii_m: np.ndarray
    radius_m: float
    g: float
    tables: dict
    config: FirstEntryAtlasConfig
    hit_charge: np.ndarray
    hit_cdf: np.ndarray
    hit_integral: np.ndarray
    miss_charge: np.ndarray
    miss_cdf: np.ndarray
    miss_integral: np.ndarray
    metadata: dict

    @classmethod
    def build(cls,*,radii_m,surface,g,tables,config=None,receiver=0,progress=None,time_u_max=None):
        config=config or FirstEntryAtlasConfig();tables=_optics(tables)
        if not isinstance(surface,SphereSurface):raise ValueError('SphereSurface required')
        R=float(surface.radius_m if surface.radius_m.ndim==0 else surface.radius_m[receiver])
        radii=np.asarray(radii_m,float)
        if (radii.ndim!=1 or len(radii)<2 or not np.isfinite(radii).all()
                or np.any(np.diff(radii)<=0) or radii[0]<=R):
            raise ValueError('increasing radii outside the glass required')
        if not np.isfinite(g) or abs(g)>=1:raise ValueError('finite HG g in (-1,1) required')
        began=perf_counter();data={};max_tail=0.;total=len(radii)*(config.hit_nodes+config.miss_nodes);done=0
        weights=np.stack((tables['s0w'],tables['s2w']),axis=-1)
        maximum_extent=max(_path_extent(r,R,g,tables,config.tail_relative_bound) for r in radii)
        required=float(np.arcsinh(maximum_extent/np.min(tables['speed'])*np.max(tables['speed'])/R))
        u_max=required if time_u_max is None else float(time_u_max)
        if not np.isfinite(u_max) or u_max<required:
            raise ValueError('shared time coordinate does not cover the declared path extent')
        u_grid=np.linspace(0.,u_max,config.time_nodes)
        for branch,count in [('hit',config.hit_nodes),('miss',config.miss_nodes)]:
            charge=np.zeros((len(radii),count,2,10))
            cdf=np.zeros((len(radii),count,config.time_nodes,2,10))
            for ir,r in enumerate(radii):
                scale=_scale(r,R,tables)
                for ia,x in enumerate(np.linspace(0.,1.,count)):
                    theta,cosine=_geometry(r,R,branch,x)
                    extent=(np.nextafter(R*np.sqrt((r/R-1)*(r/R+1)),np.inf) if branch=='hit' else
                            _path_extent(r,R,g,tables,config.tail_relative_bound))
                    profile=first_entry_profile(surface,distance_m=r,direction_cosine=cosine,g=g,
                        receiver=receiver,max_path_m=extent,path_nodes=config.path_nodes,
                        path_order=config.path_order,scatter_order=config.scatter_order)
                    front,width,end=_time_map(r,R,theta,extent,tables,g)
                    time=front+width*np.sinh(u_grid)
                    folded=fold_first_entry(profile,extinction_per_m=tables['mut'],
                        scattering_per_m=tables['mus'],speed_m_per_ns=tables['speed'],
                        spectral_weights=weights,time_edges_ns=time,quadrature_order=config.spectral_order)
                    safe=np.where(scale>0,scale,1.)
                    charge[ir,ia]=folded.charge/safe[:,None]
                    cdf[ir,ia]=folded.cumulative/safe[None,:,None]
                    max_tail=max(max_tail,float(np.max(folded.tail_bound/safe)))
                    done+=1
                    if progress is not None:progress(done,total)
            # Integral in the common u coordinate: dt/width=cosh(u)du.
            # A uniform source-clock spread can then use differences of
            # this antiderivative, without another bin or ray quadrature.
            integral=np.zeros_like(cdf)
            du=np.diff(u_grid)
            slope=np.diff(cdf,axis=2)/du[None,None,:,None,None]
            first=np.diff(np.sinh(u_grid))
            second=du*np.sinh(u_grid[1:])-np.diff(np.cosh(u_grid))
            increments=cdf[:,:,:-1]*first[None,None,:,None,None]+slope*second[None,None,:,None,None]
            integral[:,:,1:]=np.cumsum(increments,axis=2)
            data[branch+'_charge']=charge;data[branch+'_cdf']=cdf;data[branch+'_integral']=integral
        return cls(radii.copy(),R,float(g),tables,config,**data,
            metadata=dict(schema='lighthit/first-entry-q1-atlas/1',seconds=perf_counter()-began,
                rows=total,maximum_scaled_tail_bound=max_tail,time_u_max=u_max,
                maximum_path_extent_m=float(maximum_extent),convergence_verified=False))

    def moments(self,r,theta,time_edges_ns=(),*,clock_width_ns=0.):
        """Two-field charge/CDF in the query pencil frame; no area or yield.

        Interpolation error requires separate convergence checks. Charge
        attenuation is restored from the actual spectrum at the query radius.
        """
        r=float(r);theta=float(theta);time=np.asarray(time_edges_ns,float)
        if not self.radii_m[0]<=r<=self.radii_m[-1] or not np.isfinite(theta) or not 0<=theta<=np.pi:
            raise ValueError('query outside atlas geometry')
        if time.ndim!=1 or not np.isfinite(time).all() or (len(time) and (len(time)<2 or np.any(np.diff(time)<=0))):
            raise ValueError('finite increasing time edges required')
        if not np.isfinite(clock_width_ns) or clock_width_ns<0:
            raise ValueError('finite nonnegative source clock width required')
        R=self.radius_m;cap=np.arcsin(R/r);branch='hit' if theta<=cap else 'miss'
        coordinate=np.clip(_coordinates(r,R,theta,branch),0.,1.)
        qa=getattr(self,branch+'_charge');ca=getattr(self,branch+'_cdf')
        u=coordinate*(qa.shape[1]-1);j=min(int(u),qa.shape[1]-2);f=u-j
        lr=np.log(self.radii_m);i=np.clip(np.searchsorted(lr,np.log(r))-1,0,len(lr)-2)
        h=(np.log(r)-lr[i])/(lr[i+1]-lr[i]);scale=_scale(r,R,self.tables)
        charge=((1-h)*((1-f)*qa[i,j]+f*qa[i,j+1])+h*((1-f)*qa[i+1,j]+f*qa[i+1,j+1]))*scale[:,None]
        out=np.zeros((len(time),2,10))
        if len(time):
            extent=np.sqrt((r-R)*(r+R)) if branch=='hit' else _path_extent(r,R,self.g,self.tables,self.config.tail_relative_bound)
            front,width,end=_time_map(r,R,theta,extent,self.tables,self.g)
            row=(1-h)*((1-f)*ca[i,j]+f*ca[i,j+1])+h*((1-f)*ca[i+1,j]+f*ca[i+1,j+1])
            maximum=self.metadata['time_u_max'];step=maximum/(self.config.time_nodes-1)
            def at(values):
                u=np.arcsinh(np.maximum(0.,values-front)/width)
                t=np.minimum(u,maximum)/step
                k=np.minimum(t.astype(int),self.config.time_nodes-2);a=t-k
                return (1-a[:,None,None])*row[k]+a[:,None,None]*row[k+1]
            if clock_width_ns==0:
                out=at(time)*scale[None,:,None]
            else:
                stored=getattr(self,branch+'_integral')
                prefix=(1-h)*((1-f)*stored[i,j]+f*stored[i,j+1])+h*((1-f)*stored[i+1,j]+f*stored[i+1,j+1])
                def primitive(values):
                    u=np.arcsinh(np.maximum(0.,values-front)/width)
                    clipped=np.minimum(u,maximum);k=np.minimum((clipped/step).astype(int),self.config.time_nodes-2)
                    lo=k*step;slope=(row[k+1]-row[k])/step
                    answer=prefix[k]+row[k]*(np.sinh(clipped)-np.sinh(lo))[:,None,None]
                    answer+=slope*((clipped-lo)*np.sinh(clipped)-np.cosh(clipped)+np.cosh(lo))[:,None,None]
                    answer+=row[-1]*(np.sinh(u)-np.sinh(clipped))[:,None,None]
                    return width*answer
                out=(primitive(time+clock_width_ns/2)-primitive(time-clock_width_ns/2))/clock_width_ns*scale[None,:,None]
        return charge,out

    def _time_layout(self,radii,angles,clocks,spreads,edges,log_radii,prepared=None):
        """Lazy, exact time-contiguous rows of immutable prepared CDFs.

        Allocation needs only the actual payload and currently available RAM.
        Writable tables and insufficient headroom retain the canonical path.
        A row is published only after copying, and is never changed afterward.
        """
        sources=(self.hit_cdf,self.miss_cdf)
        if any(a.flags.writeable for a in sources):return None
        identity=tuple((id(a),a.shape,a.strides) for a in sources)
        from ._receiver_first_lookup import time_cdf_rows
        needed=time_cdf_rows(radii,angles,clocks,spreads,log_radii,self.radius_m,self.g,
            float(np.max(self.tables['speed'])),self.metadata['time_u_max'],edges,prepared)
        if not np.any(needed):return None
        with _TIME_LAYOUT_LOCK:
            cache=getattr(self,'_time_layout_cache',None)
            if cache is None or cache[0]!=identity:
                self._time_layout_cache=(identity,None)
                from ._memory import available_bytes
                try:
                    if sum(a.nbytes for a in sources)>available_bytes():return None
                    arrays=tuple(np.empty(np.moveaxis(a,2,-1).shape,dtype=a.dtype) for a in sources)
                except (MemoryError,OSError,RuntimeError):return None
                views=tuple(a.view() for a in arrays)
                for a in views:a.setflags(write=False)
                cache=(identity,(arrays,views,np.zeros_like(needed)))
                self._time_layout_cache=cache
            if cache[1] is None:return None
            arrays,views,ready=cache[1]
            for branch in range(2):
                for row in np.flatnonzero(needed[branch]&~ready[branch]):
                    arrays[branch][row]=np.moveaxis(sources[branch][row],1,-1)
                    ready[branch,row]=True
            return views

    def sum_groups(self,radii,angles,clocks,spreads,moments,time_edges_ns=(),*,quadratic_time=False,report=None,contiguous_time=True,_prepared=None):
        """Compiled charge/bins of one receiver's prepared pencil groups.

        Source grouping and its covariance approximation remain separate;
        this method only interpolates the precomputed physical-time kernels.
        Effective collection area must be applied once by the consumer.
        """
        from ._receiver_first_lookup import sum_groups
        radii=np.asarray(radii,float);angles=np.asarray(angles,float)
        clocks=np.asarray(clocks,float);spreads=np.asarray(spreads,float)
        moments=np.asarray(moments,float);edges=np.asarray(time_edges_ns,float)
        if (radii.ndim!=1 or angles.shape!=radii.shape or clocks.shape!=radii.shape
                or spreads.shape!=radii.shape or moments.shape!=(len(radii),2,10)
                or any(not np.isfinite(a).all() for a in (radii,angles,clocks,spreads,moments))
                or np.any(spreads<0)):
            raise ValueError('one finite geometry/clock and two-field moment row per group required')
        if edges.ndim!=1 or not np.isfinite(edges).all() or (len(edges) and (len(edges)<2 or np.any(np.diff(edges)<=0))):
            raise ValueError('provide no edges or increasing finite time edges')
        t=self.tables
        from .receiver_time import quadratic_rule
        if quadratic_time:
            sinh_u,A,B,tv,valid=quadratic_rule(self.config.time_nodes,self.metadata['time_u_max'])
            if not valid:raise ValueError('quadratic time panels are not monotone; use exact CDF lookup')
        else:sinh_u=A=B=tv=np.empty(0)
        log_radii=np.log(self.radii_m)
        layout=self._time_layout(radii,angles,clocks,spreads,edges,log_radii,_prepared) if contiguous_time and len(edges) else None
        hit_cdf,miss_cdf=(self.hit_cdf,self.miss_cdf) if layout is None else layout
        q,b,guards=sum_groups(radii,angles,clocks,spreads,moments,edges,log_radii,
            self.radius_m,self.g,t['mut'],t['mus'],t['s0w'],t['s2w'],float(np.max(t['speed'])),
            self.metadata['time_u_max'],self.hit_charge,hit_cdf,self.hit_integral,
            self.miss_charge,miss_cdf,self.miss_integral,bool(quadratic_time),sinh_u,A,B,tv,layout is not None,_prepared)
        if report is not None:
            report['guard_refinements']=int(guards);report['contiguous_time']=layout is not None
        return q,b

    def pencil_groups(self,pencils,receiver,look,polynomial,config,*,origin_ns=0.,
                         covariance_order=3,covariance_tolerance=1e-4,selected=None,radial_only=False):
        """Retain prompt source compression and group on the finite kernel.

        The source Gaussian model is integrated by positive cubature where
        its variance exceeds the declared numerical tolerance relative to
        theta^2+(R/r)^2. Cubature and grouping require their own controls;
        they do not multiply the photon yield by a singular point factor.
        """
        from ._receiver_first_groups import prepare
        from ._prompt_numba import _prepare_pencil_shape
        receiver=np.asarray(receiver,float);look=np.asarray(look,float)
        polynomial=np.asarray(polynomial,float)
        if (receiver.shape!=(3,) or look.shape!=(3,) or not np.isfinite(receiver).all()
                or not np.isfinite(look).all() or not np.isclose(np.linalg.norm(look),1.,rtol=0,atol=1e-12)):
            raise ValueError('one finite receiver and unit propagation look required')
        if polynomial.ndim!=1 or not 1<=len(polynomial)<=4 or not np.isfinite(polynomial).all():
            raise ValueError('finite cubic acceptance required')
        if (isinstance(covariance_order,(bool,np.bool_)) or not isinstance(covariance_order,(int,np.integer))
                or covariance_order<1 or not np.isfinite(covariance_tolerance) or covariance_tolerance<0):
            raise ValueError('positive covariance quadrature order and nonnegative tolerance required')
        if not np.isfinite(origin_ns):raise ValueError('finite source clock origin required')
        polynomial=np.pad(polynomial,(0,4-len(polynomial)))
        shape=pencils.get('_first_entry_shape')
        if shape is None:
            shape=_prepare_pencil_shape(pencils['direction'],pencils['major'],pencils['pos_cov'])
            pencils['_first_entry_shape']=shape
        nodes,weights=_normal_rule(covariance_order)
        thresholds,theta_lut,theta_first=_theta_rule(float(config.merge_theta_min),float(config.merge_theta_ratio))
        selected_indices=None if selected is None else np.flatnonzero(np.asarray(selected,bool)).astype(np.int64)
        hierarchy=(pencils.get('hierarchy_parent') if selected is None else None)
        next_node=pencils.get('next_node') if hierarchy is not None else None
        representatives=pencils.get('_first_entry_geometry_representatives')
        geometry_count=0 if representatives is None else len(representatives)
        geometry_ids=None if geometry_count==0 else pencils['_first_entry_geometry_ids']
        if radial_only:
            from ._prompt_numba import _prepare_gauss_inverse,_GAUSS_TAU,_GAUSS_W
            nodes,weights=_prepare_gauss_inverse(_GAUSS_TAU,_GAUSS_W)
        try:
            groups=prepare(receiver,look,float(origin_ns),pencils['position'],pencils['direction'],
                pencils['w0'],pencils['w2'],pencils['time'],pencils['major'],pencils['var_major'],
                pencils['var_minor'],shape,polynomial,self.radius_m,float(config.merge_log_r),
                float(config.merge_tau_ns),float(config.merge_theta_min),float(config.merge_theta_ratio),
                self.config.hit_nodes,self.config.miss_nodes,float(np.max(self.tables['speed'])),
                nodes,weights,float(covariance_tolerance),selected_indices,bool(radial_only),
                hierarchy,next_node,float(config.smoothing_factor),thresholds,theta_lut,theta_first,geometry_ids,geometry_count)
        except ValueError as error:
            if not radial_only or not str(error).startswith('finite radial quadrature'):raise
            groups,metadata=self.pencil_groups(pencils,receiver,look,polynomial,config,origin_ns=origin_ns,
                covariance_order=covariance_order,covariance_tolerance=covariance_tolerance,selected=selected)
            metadata['radial_refined']=True
            return groups,metadata
        return groups,dict(groups=len(groups[0]),source_pencils=len(pencils['w0']),
                       cached_emission_geometries=geometry_count,
                       covariance_samples=int(groups[5]),covariance_order=int(covariance_order),
                       receiver_reference='first-entry',covariance_rule='finite radial two-node' if radial_only else 'positive tangent cubature')

    def evaluate_pencils(self,pencils,receiver,look,polynomial,config,*,origin_ns=0.,
                         time_edges_ns=(),covariance_order=3,covariance_tolerance=1e-4):
        groups,metadata=self.pencil_groups(pencils,receiver,look,polynomial,config,origin_ns=origin_ns,
            covariance_order=covariance_order,covariance_tolerance=covariance_tolerance)
        q,b=self.sum_groups(*groups[:5],time_edges_ns)
        return q,b,metadata

    def validate(self,*,tables,surface,g,receiver=0):
        """Reject another optical spectrum or physical glass radius."""
        from ._portable import same_values
        actual=_optics(tables)
        if not isinstance(surface,SphereSurface):raise ValueError('SphereSurface required')
        R=float(surface.radius_m if surface.radius_m.ndim==0 else surface.radius_m[receiver])
        if not same_values(actual,self.tables) or not same_values([R,float(g)],[self.radius_m,self.g]):
            raise ValueError('first-entry Q1 atlas belongs to another optical/receiver contract')

    def save(self,directory):
        """Separate mmap-compatible arrays; publish the manifest last."""
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
        for name in ('radii_m','hit_charge','hit_cdf','hit_integral','miss_charge','miss_cdf','miss_integral'):
            np.save(directory/(name+'.npy'),getattr(self,name),allow_pickle=False)
        self._save_metadata(directory)

    def _save_metadata(self,directory):
        directory=Path(directory)
        metadata=dict(self.metadata,radius_m=self.radius_m,g=self.g,
            tables={k:v.tolist() for k,v in self.tables.items()},config=asdict(self.config))
        encoded=json.dumps(metadata,sort_keys=True)
        metadata['contract_sha256']=hashlib.sha256(encoded.encode()).hexdigest()
        temporary=directory/'manifest.json.tmp'
        temporary.write_text(json.dumps(metadata,indent=2)+'\n')
        temporary.replace(directory/'manifest.json')

    @classmethod
    def load(cls,directory,*,mmap=True):
        directory=Path(directory);metadata=json.loads((directory/'manifest.json').read_text())
        digest=metadata.pop('contract_sha256')
        if (metadata.get('schema')!='lighthit/first-entry-q1-atlas/1' or
                hashlib.sha256(json.dumps(metadata,sort_keys=True).encode()).hexdigest()!=digest):
            raise ValueError('invalid first-entry Q1 atlas contract')
        config=FirstEntryAtlasConfig(**metadata['config']);tables=_optics(metadata['tables'])
        arrays={name:np.load(directory/(name+'.npy'),allow_pickle=False,mmap_mode='r' if mmap else None)
                for name in ('radii_m','hit_charge','hit_cdf','hit_integral','miss_charge','miss_cdf','miss_integral')}
        radii=arrays['radii_m'];R=float(metadata['radius_m'])
        if (radii.ndim!=1 or len(radii)<2 or not np.isfinite(radii).all()
                or np.any(np.diff(radii)<=0) or radii[0]<=R):
            raise ValueError('invalid first-entry Q1 radius grid')
        for branch,count in [('hit',config.hit_nodes),('miss',config.miss_nodes)]:
            if (arrays[branch+'_charge'].shape!=(len(radii),count,2,10)
                    or arrays[branch+'_cdf'].shape!=(len(radii),count,config.time_nodes,2,10)):
                raise ValueError('invalid first-entry Q1 payload dimensions')
            if arrays[branch+'_integral'].shape!=arrays[branch+'_cdf'].shape:
                raise ValueError('invalid first-entry Q1 time primitive')
        return cls(radius_m=R,g=float(metadata['g']),tables=tables,config=config,
                   metadata=metadata,**arrays)

    @classmethod
    def build_parallel(cls,*,radii_m,surface,g,tables,output,config=None,jobs=1,progress=None,overwrite=False,keep_blocks=True):
        """Independent radial blocks, joined into mmap-compatible tables.

        All workers use the same physical-time coordinate. Completed blocks
        can be reused only when their optics, geometry, resolution and
        builder fingerprint match. Only the owning process joins outputs.
        """
        from concurrent.futures import ProcessPoolExecutor,as_completed
        from multiprocessing import get_context
        radii=np.asarray(radii_m,float);config=config or FirstEntryAtlasConfig()
        tables=_optics(tables)
        if not isinstance(surface,SphereSurface) or surface.radius_m.ndim!=0:
            raise ValueError('one explicit SphereSurface radius required')
        R=float(surface.radius_m)
        if (radii.ndim!=1 or len(radii)<2 or not np.isfinite(radii).all()
                or np.any(np.diff(radii)<=0) or radii[0]<=R):
            raise ValueError('increasing radii outside the glass required')
        if isinstance(jobs,(bool,np.bool_)) or not isinstance(jobs,(int,np.integer)) or jobs<1:
            raise ValueError('jobs must be a positive integer')
        output=Path(output);output.mkdir(parents=True,exist_ok=True)
        manifest=output/'manifest.json'
        if manifest.exists():
            if not overwrite:raise FileExistsError(f'{manifest} already exists')
            if not (output/'build-progress.json').exists():
                raise ValueError('legacy completed atlas needs a new output directory')
        from ._table_checkpoint import JoinedArrays,radial_blocks
        began=perf_counter();blocks=radial_blocks(len(radii));workers=min(jobs,len(blocks))
        extent=max(_path_extent(r,R,g,tables,config.tail_relative_bound) for r in radii)
        maximum=float(np.arcsinh(extent/np.min(tables['speed'])*np.max(tables['speed'])/R))
        arguments=[dict(radii=radii[block],radius=R,g=g,tables=tables,config=asdict(config),
            maximum=maximum,output=str(output/'blocks'/f'{i:04d}')) for i,block in enumerate(blocks)]
        shapes=dict(hit_charge=(len(radii),config.hit_nodes,2,10),
            hit_cdf=(len(radii),config.hit_nodes,config.time_nodes,2,10),
            hit_integral=(len(radii),config.hit_nodes,config.time_nodes,2,10),
            miss_charge=(len(radii),config.miss_nodes,2,10),
            miss_cdf=(len(radii),config.miss_nodes,config.time_nodes,2,10),
            miss_integral=(len(radii),config.miss_nodes,config.time_nodes,2,10))
        joined=JoinedArrays(output,shapes,dict(radii=radii.tolist(),radius=R,g=g,config=asdict(config),
            maximum=maximum,tables={k:v.tolist() for k,v in tables.items()},builder=_radial_fingerprint()))
        arrays=joined.arrays
        if manifest.exists():manifest.unlink()
        maximum_tail=0.;completed=0;builder_signatures=set()
        def account(index,metadata):
            nonlocal maximum_tail,completed
            maximum_tail=max(maximum_tail,metadata['maximum_scaled_tail_bound'])
            builder_signatures.add(metadata['builder_sha256']);completed+=len(blocks[index])
            if len(builder_signatures)!=1:raise ValueError('radial blocks were built by different numerical sources')
        def merge(index,metadata):
            part=FirstEntryAtlas.load(arguments[index]['output'])
            joined.merge(index,blocks[index],{name:getattr(part,name) for name in arrays},metadata=metadata)
            account(index,metadata)
            if not keep_blocks:joined.discard(arguments[index]['output'])
            if progress is not None:progress(completed,len(radii))
        pending=[]
        for i in range(len(blocks)):
            if joined.contains(i,blocks[i]):account(i,joined.metadata(i))
            else:pending.append(i)
        if workers==1:
            for i in pending:merge(i,_build_radial_block(arguments[i]))
        else:
            with ProcessPoolExecutor(max_workers=workers,mp_context=get_context('spawn')) as pool:
                futures={pool.submit(_build_radial_block,arguments[i]):i for i in pending}
                for future in as_completed(futures):merge(futures[future],future.result())
        for array in arrays.values():array.flush()
        np.save(output/'radii_m.npy',radii,allow_pickle=False)
        atlas=cls(radii,R,float(g),tables,config,**arrays,metadata=dict(
            schema='lighthit/first-entry-q1-atlas/1',seconds=perf_counter()-began,
            rows=len(radii)*(config.hit_nodes+config.miss_nodes),parallel_workers=workers,
            maximum_scaled_tail_bound=maximum_tail,time_u_max=maximum,
            maximum_path_extent_m=float(extent),builder_sha256=next(iter(builder_signatures)),
            convergence_verified=False))
        atlas._save_metadata(output)
        return cls.load(output)


def _radial_fingerprint():
    from . import receiver_first,receiver_first_fold,_receiver_first_numba
    import inspect
    digest=hashlib.sha256()
    for module in (receiver_first,receiver_first_fold,_receiver_first_numba):
        digest.update(Path(module.__file__).read_bytes())
    for function in (FirstEntryAtlas.build,_geometry,_time_map,_path_extent,_scale):
        digest.update(inspect.getsource(function).encode())
    digest.update(np.__version__.encode())
    return digest.hexdigest()


def _build_radial_block(arguments):
    fingerprint=_radial_fingerprint();output=Path(arguments['output'])
    config=FirstEntryAtlasConfig(**arguments['config'])
    if (output/'manifest.json').exists():
        stored=FirstEntryAtlas.load(output)
        if (stored.metadata.get('builder_sha256')==fingerprint
                and stored.config==config and stored.metadata['time_u_max']==arguments['maximum']
                and np.array_equal(stored.radii_m,arguments['radii'])):
            stored.validate(tables=arguments['tables'],surface=SphereSurface(arguments['radius']),g=arguments['g'])
            return stored.metadata
    atlas=FirstEntryAtlas.build(radii_m=arguments['radii'],surface=SphereSurface(arguments['radius']),
        g=arguments['g'],tables=arguments['tables'],config=config,time_u_max=arguments['maximum'])
    atlas.metadata['builder_sha256']=fingerprint;atlas.save(output)
    return atlas.metadata
