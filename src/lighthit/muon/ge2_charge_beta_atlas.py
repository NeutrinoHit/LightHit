"""Signed beta-response modes for stationary finite-track >=2 transport.

Base beta-one charges come from an existing frozen universal atlas. Two small
optics-only payloads supply Taylor corrections in delta=1/beta-1. Source clocks
do not enter omega=0. Threshold regions and unvalidated beta are rejected.
"""
from pathlib import Path
import hashlib,json
import numpy as np
from .track_ge2_primitive import _charge_pairs


class Ge2ChargeBetaAtlas:
    """Signed beta modes of the stationary >=2 charge of straight pieces."""
    def __init__(self,path,base,*,require_validated=True):
        """Load the modes and require a passed synthetic validation."""
        self.path=Path(path);self.base=base
        with np.load(path,allow_pickle=False) as data:
            if str(data['schema'])!='lighthit/ge2-charge-beta/1':raise ValueError('unsupported beta charge atlas')
            self.metadata=json.loads(str(data['metadata']))
            self.nodes=data['nodes_m'].copy();self.edge_offsets=data['edge_offsets'].copy()
            self.panel_offsets=data['panel_offsets'].copy();self.edges=data['edges_m'].copy()
            self.coefficients=data['coefficients'].copy();self.prefix=data['prefix'].copy()
            self.lognorm=data['lognorm'].copy()
        if require_validated and not self.metadata.get('synthetic_validation_passed',False):
            raise ValueError('beta response has not passed synthetic validation')
        digest=hashlib.sha256((base.directory/'manifest.json').read_bytes()).hexdigest()
        if digest!=self.metadata['base_manifest_sha256']:raise ValueError('beta modes belong to another base atlas')
        if not np.array_equal(self.nodes,base.impact_nodes_m):raise ValueError('beta/base radius nodes differ')
        if self.coefficients.shape[0]!=2 or self.prefix.shape[0]!=2:raise ValueError('two correction modes required')
        if (not np.isfinite(self.coefficients).all() or not np.isfinite(self.prefix).all()
                or self.prefix.shape[1:]!=(self.edge_offsets[-1],10)
                or self.coefficients.shape[1]!=self.panel_offsets[-1]
                or self.coefficients.shape[-1]!=10):raise ValueError('invalid beta response payload')
        checks=base.manifest.get('validation',[])
        self._validated=np.array([any(c['low_m']<=lo and c['high_m']>=hi
            and c.get('charge_passed',c['passed']) for c in checks)
            for lo,hi in zip(self.nodes[:-1],self.nodes[1:])],bool)
        # Prepare the base's packed stationary arrays once; no time rows load.
        base.segment_charge_many(np.empty(0),np.empty((0,3)),np.empty(0),np.empty(0))

    @property
    def resident_bytes(self):
        """Memory held by the tables."""
        return sum(x.nbytes for x in (self.nodes,self.edge_offsets,self.panel_offsets,self.edges,
            self.coefficients,self.prefix,self.lognorm))

    def validate_kernel(self,kernel):
        """Base atlas optics check plus consistency of the two tables."""
        self.base.validate_kernel(kernel)
        if self.metadata['optics_fingerprint']!=self.base.metadata['optics_fingerprint']:
            raise ValueError('beta/base optical contract differs')

    def supports(self,rho,beta,a,b):
        """True when beta and impact distance lie in the validated region."""
        if not self.metadata['beta_min']<=beta<=1. or not self.nodes[0]<=rho<=self.nodes[-1]:return False
        j=int(np.clip(np.searchsorted(self.nodes,rho)-1,0,len(self.nodes)-2))
        return bool(self._validated[j] or np.any(self.nodes==rho))

    def segment_charge_many(self,rho,looks,a,b,beta,*,coefficient0_per_m=1.):
        """All-time >=2 charge of straight pieces with beta < 1, many pieces at once."""
        rho,a,b,beta,density=np.broadcast_arrays(np.asarray(rho,float),np.asarray(a,float),
            np.asarray(b,float),np.asarray(beta,float),np.asarray(coefficient0_per_m,float))
        looks=np.asarray(looks,float)
        if looks.shape!=rho.shape+(3,) or not np.allclose(np.linalg.norm(looks,axis=-1),1.):
            raise ValueError('one unit look per beta pair required')
        if not np.isfinite(beta).all() or np.any(beta<self.metadata['beta_min']) or np.any(beta>1):
            raise ValueError('beta outside validated response region')
        if (not np.isfinite(rho).all() or np.any(rho<self.nodes[0]) or np.any(rho>self.nodes[-1])
                or not np.isfinite(a).all() or not np.isfinite(b).all() or np.any(a<b)
                or not np.isfinite(density).all() or np.any(density<0)):
            raise ValueError('invalid beta response geometry/density')
        j=np.clip(np.searchsorted(self.nodes,rho)-1,0,len(self.nodes)-2)
        exact=np.isin(rho,self.nodes)
        if not np.all(self._validated[j]|exact):raise ValueError('base radial interval failed charge validation')
        base=_charge_pairs(rho.ravel().copy(),looks.reshape(-1,3),a.ravel().copy(),b.ravel().copy(),
            density.ravel().copy(),self.nodes,*self.base._packed_charge).reshape(rho.shape)
        delta=1/beta-1;result=base.copy()
        for index in range(2):
            correction=_charge_pairs(rho.ravel().copy(),looks.reshape(-1,3),a.ravel().copy(),b.ravel().copy(),
                density.ravel().copy(),self.nodes,self.edge_offsets,self.panel_offsets,self.edges,
                self.coefficients[index],self.prefix[index],self.lognorm).reshape(rho.shape)
            result+=delta**(index+1)*correction
        return result
