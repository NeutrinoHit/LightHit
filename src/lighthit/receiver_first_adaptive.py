"""Time-grid refinement with an explicit bin-L1 budget for first-entry Q1.

Coarse and fine kernels share geometry, spectrum and total charge. The total
variation of their CDF difference bounds the error in any disjoint time bins.
Positive source-clock convolution cannot increase that bound. Source grouping,
angular/radial interpolation and physical-profile errors are separate.
"""
from dataclasses import dataclass,replace,asdict
from pathlib import Path
import hashlib,json
import numpy as np
from .receiver_first_atlas import FirstEntryAtlas


def _signature(atlas):
    record=dict(radius=atlas.radius_m,g=atlas.g,config=asdict(atlas.config),
        tables={k:v.tolist() for k,v in atlas.tables.items()},
        builder=atlas.metadata.get('builder_sha256'),time_u_max=atlas.metadata['time_u_max'])
    digest=hashlib.sha256(json.dumps(record,sort_keys=True).encode())
    digest.update(atlas.radii_m.tobytes())
    return digest.hexdigest()


def _primitive(cdf,maximum):
    u=np.linspace(0.,maximum,cdf.shape[2]);du=np.diff(u)
    result=np.zeros_like(cdf)
    for i in range(len(cdf)):
        slope=np.diff(cdf[i],axis=1)/du[None,:,None,None]
        increment=cdf[i,:,:-1]*np.diff(np.sinh(u))[None,:,None,None]
        increment+=slope*(du*np.sinh(u[1:])-np.diff(np.cosh(u)))[None,:,None,None]
        result[i,:,1:]=np.cumsum(increment,axis=1)
    return result


@dataclass
class AdaptiveFirstEntry:
    fine: FirstEntryAtlas
    coarse: FirstEntryAtlas
    hit_error: np.ndarray
    miss_error: np.ndarray
    hit_quad_coarse: np.ndarray | None = None
    hit_quad_fine: np.ndarray | None = None
    miss_quad_coarse: np.ndarray | None = None
    miss_quad_fine: np.ndarray | None = None

    @classmethod
    def of(cls,fine,*,stride=4):
        count=fine.config.time_nodes
        if (isinstance(stride,(bool,np.bool_)) or not isinstance(stride,(int,np.integer))
                or stride<1 or (count-1)%stride):
            raise ValueError('stride must divide the fine time grid intervals')
        data={};errors={};maximum=fine.metadata['time_u_max']
        positions=np.arange(count)/stride
        count_coarse=(count-1)//stride+1
        from .receiver_time import quadratic_rule
        tvc=quadratic_rule(count_coarse,maximum);tvf=quadratic_rule(count,maximum)
        quadratic=tvc[-1] and tvf[-1];quad={}
        low=np.minimum(positions.astype(int),count_coarse-2);fraction=positions-low
        for branch in ('hit','miss'):
            original=getattr(fine,branch+'_cdf')
            coarse=np.ascontiguousarray(original[:,:,::stride])
            error=np.empty(original.shape[:2]+original.shape[3:])
            if quadratic:qc=np.empty_like(error);qf=np.empty_like(error)
            for r in range(len(fine.radii_m)):
                reconstructed=(np.take(coarse[r],low,axis=1)*(1-fraction)[None,:,None,None]
                    +np.take(coarse[r],low+1,axis=1)*fraction[None,:,None,None])
                delta=original[r]-reconstructed
                error[r]=np.sum(abs(np.diff(delta,axis=1)),axis=1)
                if quadratic:
                    qc[r]=np.sum(abs(np.diff(coarse[r],axis=1))*tvc[3][None,:,None,None],axis=1)
                    qf[r]=np.sum(abs(np.diff(original[r],axis=1))*tvf[3][None,:,None,None],axis=1)
            if not np.isfinite(error).all():raise FloatingPointError('nonfinite time interpolation error')
            data[branch+'_charge']=getattr(fine,branch+'_charge')
            data[branch+'_cdf']=coarse;data[branch+'_integral']=_primitive(coarse,maximum)
            errors[branch+'_error']=error
            if quadratic:quad.update({branch+'_quad_coarse':qc,branch+'_quad_fine':qf})
        metadata=dict(fine.metadata,reference_time_nodes=count,time_stride=int(stride),
                      time_error_reference_sha256=_signature(fine),convergence_verified=False,quadratic_time=bool(quadratic))
        coarse=FirstEntryAtlas(fine.radii_m,fine.radius_m,fine.g,fine.tables,
            replace(fine.config,time_nodes=count_coarse),**data,metadata=metadata)
        return cls(fine,coarse,**errors,**quad)

    def save(self,directory):
        directory=Path(directory);self.coarse.save(directory)
        np.save(directory/'hit_time_error.npy',self.hit_error,allow_pickle=False)
        np.save(directory/'miss_time_error.npy',self.miss_error,allow_pickle=False)
        if self.hit_quad_coarse is not None:
            for branch in ('hit','miss'):
                for mode in ('coarse','fine'):
                    name=branch+'_quad_'+mode;np.save(directory/(name+'.npy'),getattr(self,name),allow_pickle=False)

    @classmethod
    def load(cls,directory,*,fine):
        directory=Path(directory);coarse=FirstEntryAtlas.load(directory)
        if coarse.metadata.get('time_error_reference_sha256')!=_signature(fine):
            raise ValueError('coarse time errors belong to another fine receiver table')
        errors={branch+'_error':np.load(directory/(branch+'_time_error.npy'),allow_pickle=False,mmap_mode='r')
                for branch in ('hit','miss')}
        for branch in ('hit','miss'):
            e=errors[branch+'_error']
            if e.shape!=getattr(coarse,branch+'_charge').shape or not np.isfinite(e).all() or np.any(e<0):
                raise ValueError('invalid time interpolation bounds')
            if coarse.metadata.get('quadratic_time'):
                for mode in ('coarse','fine'):
                    name=branch+'_quad_'+mode
                    value=np.load(directory/(name+'.npy'),allow_pickle=False,mmap_mode='r')
                    if value.shape!=e.shape or not np.isfinite(value).all() or np.any(value<0):raise ValueError('invalid deposition bounds')
                    errors[name]=value
        return cls(fine,coarse,**errors)

    def sum_groups(self,radii,angles,clocks,spreads,moments,time_edges_ns=(),*,
                   relative_tolerance=.003,absolute_tolerance=0.):
        """Refine largest error contributions until the declared L1 budget fits.

        Bounds and absolute tolerance have the charge's units (before area).
        The accepted error is relative to the stored fine time grid, not to
        an exact continuous transport solution. Charge is unchanged.
        """
        from ._receiver_first_lookup import charge_and_time_bounds
        radii=np.asarray(radii,float);angles=np.asarray(angles,float)
        clocks=np.asarray(clocks,float);spreads=np.asarray(spreads,float)
        moments=np.asarray(moments,float);edges=np.asarray(time_edges_ns,float)
        if (radii.ndim!=1 or angles.shape!=radii.shape or clocks.shape!=radii.shape
                or spreads.shape!=radii.shape or moments.shape!=(len(radii),2,10)
                or any(not np.isfinite(a).all() for a in (radii,angles,clocks,spreads,moments))
                or np.any(spreads<0)):
            raise ValueError('finite group geometry, clocks and two-field moments required')
        if edges.ndim!=1 or not np.isfinite(edges).all() or (len(edges) and (len(edges)<2 or np.any(np.diff(edges)<=0))):
            raise ValueError('increasing finite time edges required')
        if (not np.isfinite(relative_tolerance) or relative_tolerance<0
                or not np.isfinite(absolute_tolerance) or absolute_tolerance<0):
            raise ValueError('nonnegative finite time tolerances required')
        a=self.coarse;t=a.tables
        shared_geometry=(a.radius_m==self.fine.radius_m and a.g==self.fine.g
            and a.radii_m.dtype==self.fine.radii_m.dtype and np.array_equal(a.radii_m,self.fine.radii_m)
            and a.hit_charge.shape[1]==self.fine.hit_charge.shape[1]
            and a.miss_charge.shape[1]==self.fine.miss_charge.shape[1]
            and all(np.array_equal(t[k],self.fine.tables[k]) for k in ('mut','mus','speed','s0w','s2w')))
        if not len(edges):
            from ._receiver_first_lookup import charge_groups
            charges=charge_groups(radii,angles,moments,np.log(a.radii_m),a.radius_m,a.g,
                t['mut'],t['mus'],t['s0w'],t['s2w'],float(np.max(t['speed'])),a.hit_charge,a.miss_charge)
            return float(charges.sum()),np.empty(0),dict(fine_groups=0,coarse_groups=len(radii),bin_l1_bound=0.)
        if len(edges) and self.hit_quad_coarse is not None:
            from .receiver_time import time_levels,quadratic_rule
            charges,bounds,geometry=charge_and_time_bounds(radii,angles,moments,np.log(a.radii_m),a.radius_m,a.g,
                t['mut'],t['mus'],t['s0w'],t['s2w'],float(np.max(t['speed'])),a.hit_charge,a.miss_charge,self.hit_error,self.miss_error)
            charge=float(charges.sum());budget=absolute_tolerance+relative_tolerance*abs(charge)
            coarse_bound=bounds+np.max(quadratic_rule(a.config.time_nodes,a.metadata['time_u_max'])[3])*abs(charges)
            fine_bound=np.max(quadratic_rule(self.fine.config.time_nodes,self.fine.metadata['time_u_max'])[3])*abs(charges)
            levels,remaining=time_levels(coarse_bound,fine_bound,budget)
            result=np.zeros(len(edges)-1);guards=0
            for level,atlas,quadratic in [(0,a,True),(1,self.fine,True),(2,self.fine,False)]:
                mask=levels==level
                if np.any(mask):
                    report={}
                    prepared=tuple(v[mask] for v in geometry) if atlas is a or shared_geometry else None
                    _,b=atlas.sum_groups(radii[mask],angles[mask],clocks[mask],spreads[mask],moments[mask],edges,quadratic_time=quadratic,report=report,_prepared=prepared)
                    guards+=report['guard_refinements']
                    result+=b
            return charge,result,dict(fine_groups=int(np.sum(levels!=0)),coarse_groups=int(np.sum(levels==0)),
                exact_groups=int(np.sum(levels==2))+guards,guard_refinements=guards,bin_l1_bound=remaining,bin_l1_budget=budget,
                reference_time_nodes=self.fine.config.time_nodes,coarse_time_nodes=a.config.time_nodes,quadratic_time=True)
        charges,bounds,geometry=charge_and_time_bounds(radii,angles,moments,np.log(a.radii_m),
            a.radius_m,a.g,t['mut'],t['mus'],t['s0w'],t['s2w'],float(np.max(t['speed'])),
            a.hit_charge,a.miss_charge,self.hit_error,self.miss_error)
        charge=float(charges.sum())
        budget=absolute_tolerance+relative_tolerance*abs(charge)
        remaining=float(bounds.sum());fine=np.zeros(len(radii),bool)
        if budget==0:
            fine[:]=True;remaining=0.
        elif remaining>budget:
            for i in np.argsort(bounds)[::-1]:
                fine[i]=True;remaining-=bounds[i]
                if remaining<=budget:break
        result=np.zeros(len(edges)-1)
        for mask,atlas in [(fine,self.fine),(~fine,self.coarse)]:
            if not np.any(mask):continue
            prepared=tuple(v[mask] for v in geometry) if atlas is a or shared_geometry else None
            _,bins=atlas.sum_groups(radii[mask],angles[mask],clocks[mask],spreads[mask],moments[mask],edges,_prepared=prepared)
            result+=bins
        return charge,result,dict(fine_groups=int(fine.sum()),coarse_groups=int((~fine).sum()),
            bin_l1_bound=max(0.,remaining),bin_l1_budget=budget,
            reference_time_nodes=self.fine.config.time_nodes,coarse_time_nodes=a.config.time_nodes)
