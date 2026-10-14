"""Stationary Q1 shape correction with full-spectrum yield normalization.

The signed mode is fitted at a near-relativistic beta anchor. It removes a
smooth yield change before approximating the remaining cone/transport change.
This is a validated numerical approximation, not an exact angle transformation.
Time responses and near-threshold sources are deliberately outside this class.
"""
import hashlib,json
import numpy as np
from .track_first_primitive import _checked_source_bounds


class FirstChargeBetaAtlas:
    """Signed correction in (1/beta^2-1) to the beta=1 Q1 charge atlas."""
    def __init__(self,path,base,*,require_validated=True):
        """Load the correction and check that it belongs to ``base``."""
        self.base=base
        with np.load(path,allow_pickle=False) as data:
            if str(data['schema'])!='lighthit/first-charge-beta/1':raise ValueError('unsupported Q1 beta atlas')
            self.metadata=json.loads(str(data['metadata']))
            self.left=data['left'].copy();self.right=data['right'].copy()
        digest=hashlib.sha256((base.directory/'manifest.json').read_bytes()).hexdigest()
        if digest!=self.metadata['base_manifest_sha256']:raise ValueError('Q1 beta mode belongs to another base atlas')
        if require_validated and not self.metadata.get('synthetic_validation_passed',False):
            raise ValueError('Q1 beta mode has not passed validation')
        if (self.left.shape!=base.charge_left.shape or self.right.shape!=self.left.shape
                or not np.isfinite(self.left).all() or not np.isfinite(self.right).all()):
            raise ValueError('invalid Q1 beta mode payload')
        from scipy.interpolate import PchipInterpolator
        self._cubic={id(a):np.moveaxis(PchipInterpolator(base.x_edges,a,axis=1).c,2,0)
            for a in (self.left,self.right,base.charge_left,base.charge_right)}

    @property
    def resident_bytes(self):
        """Memory held by the correction and its cubic tables."""
        return self.left.nbytes+self.right.nbytes+sum(a.nbytes for a in self._cubic.values())

    def validate_kernel(self,kernel):
        """Same optics check as the base atlas."""
        self.base.validate_kernel(kernel)

    def supports(self,rho,beta,lo,hi):
        """True when radius, beta and span lie in the validated region."""
        return (self.metadata['beta_min']<=beta<=1 and self.base.radii_m[0]<=rho<=self.base.radii_m[-1]
            and (hi-lo)/rho>=self.metadata.get('minimum_span_over_rho',0.)
            and lo<hi and lo/rho>=self.base.x_edges[0] and hi/rho<=self.base.x_edges[-1])

    def query_charge_many(self,rho,look,lo,hi,beta,*,coefficient0_per_m=1.,enforce_span=True):
        """All-time Q1 charge of straight pieces with beta < 1, many pieces at once."""
        base=self.base;rho=np.asarray(rho,float);look=np.asarray(look,float)
        beta=np.broadcast_to(np.asarray(beta,float),rho.shape)
        if not np.isfinite(beta).all() or np.any(beta<self.metadata['beta_min']) or np.any(beta>1):
            raise ValueError('beta outside validated Q1 mode')
        density=np.broadcast_to(np.asarray(coefficient0_per_m,float),rho.shape)
        if (rho.ndim!=1 or look.shape!=(len(rho),3) or not np.isfinite(rho).all()
                or not np.isfinite(look).all() or not np.allclose(np.linalg.norm(look,axis=1),1.)
                or np.any(rho<base.radii_m[0]) or np.any(rho>base.radii_m[-1])
                or not np.isfinite(density).all() or np.any(density<0)):
            raise ValueError('invalid Q1 beta geometry/density')
        if enforce_span and np.any((np.asarray(hi)-np.asarray(lo))/rho<self.metadata.get('minimum_span_over_rho',0.)):
            raise ValueError('Q1 interval narrower than validated charge grid')
        bounds=_checked_source_bounds(np.column_stack((lo,hi))/rho[:,None],base.x_edges)
        index=np.clip(np.searchsorted(base.x_edges,bounds,side='right')-1,0,len(base.x_edges)-2)
        fraction=(bounds-base.x_edges[index])/(base.x_edges[index+1]-base.x_edges[index])
        features=look[:,0,None]**base.powers[:,0]*look[:,2,None]**base.powers[:,1]
        def values(j):
            """Charge of every piece for one radius node."""
            def at(data,k):
                """Cubic table value at one end of every piece."""
                i=index[:,k];t=(bounds[:,k]-base.x_edges[i])[:,None]
                coef=self._cubic[id(data)][j,:,i,:]
                value=((coef[:,0]*t+coef[:,1])*t+coef[:,2])*t+coef[:,3]
                return np.einsum('ij,ij->i',value,features)
            l0,l1=at(self.left,0),at(self.left,1);r0,r1=at(self.right,0),at(self.right,1)
            d=np.where(abs(l0)+abs(l1)<=abs(r0)+abs(r1),l1-l0,r0-r1)
            l0,l1=at(base.charge_left,0),at(base.charge_left,1)
            r0,r1=at(base.charge_right,0),at(base.charge_right,1)
            q=np.where(abs(l0)+abs(l1)<=abs(r0)+abs(r1),l1-l0,r0-r1)
            return q,d
        s=1/beta**2-1
        ratio=1-s*self.metadata['yield_slope']
        start=np.clip(np.searchsorted(base.radii_m,rho)-1,0,len(base.radii_m)-3)
        logarithm=np.zeros(len(rho));valid=np.ones(len(rho),bool)
        for offset in range(3):
            j=start+offset;radius=base.radii_m[j]
            weight=np.ones(len(rho))
            for other in range(3):
                if other!=offset:
                    node=base.radii_m[start+other];weight*=(rho-node)/(radius-node)
            q,d=values(j);actual=ratio*(q+s*d)
            used=abs(weight)>1e-14
            valid &= (~used)|(np.isfinite(actual)&(actual>0))
            logarithm+=weight*np.log(np.maximum(actual,np.finfo(float).tiny))
        return np.where(valid,density*np.exp(logarithm),np.nan)
