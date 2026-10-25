"""Prepared physical-cone/clock Q1 time responses between beta anchors."""
from pathlib import Path
import json,hashlib
import numpy as np
from .track_first_primitive import TrackFirstAtlasStore

C_M_PER_NS=.299792458


class FirstTimeBetaAtlas:
    def __init__(self,path):
        path=Path(path);self.manifest=json.loads(path.read_text());self.path=path
        if self.manifest.get('schema')!='lighthit/first-time-beta/1':raise ValueError('unsupported beta-time atlas')
        if not self.manifest.get('validation_passed'):raise ValueError('beta-time interpolation requires validation')
        nodes=self.manifest['nodes'];self.beta=np.array([n['beta'] for n in nodes],float)
        if len(nodes)<2 or np.any(np.diff(self.beta)<=0):raise ValueError('ordered beta anchors required')
        self.stores=[]
        for node in nodes:
            root=(path.parent/node['directory']).resolve();data=root/'manifest.json'
            if hashlib.sha256(data.read_bytes()).hexdigest()!=node['manifest_sha256']:raise ValueError('beta-time anchor changed')
            store=TrackFirstAtlasStore(root)
            if store.metadata['beta']!=node['beta']:raise ValueError('beta-time anchor parameter differs')
            if not np.isclose(store.metadata['speed_m_per_ns'],node['beta']*C_M_PER_NS,rtol=64*np.finfo(float).eps,atol=0):
                raise ValueError('beta-time anchor has another emission clock')
            self.stores.append(store)
        self.radii_m=self.stores[0].radii_m
        if any(not np.array_equal(s.radii_m,self.radii_m) for s in self.stores):raise ValueError('beta-time radial axes differ')

    def validate_kernel(self,kernel):
        for store in self.stores:store.validate_kernel(kernel)

    def supports(self,rho,beta,lo,hi):
        return (self.beta[0]<=beta<=self.beta[-1] and self.radii_m[0]<=rho<=self.radii_m[-1]
            and lo<hi and all(lo>=s.x_edges[0]*rho and hi<=s.x_edges[-1]*rho for s in self.stores))

    def query(self,rho,orientation,lo,hi,edges,beta,*,coefficient0_per_m=1.):
        if not self.supports(rho,beta,lo,hi):raise ValueError('piece outside beta-time atlas')
        j=min(max(int(np.searchsorted(self.beta,beta,side='right')-1),0),len(self.beta)-2)
        s=1/beta**2-1;s0=1/self.beta[j]**2-1;s1=1/self.beta[j+1]**2-1
        w=(s-s0)/(s1-s0)
        def node(store):
            metadata=getattr(store,'metadata',{});boundary=metadata.get('receiver_contract')
            options=dict(coefficient0_per_m=coefficient0_per_m)
            if boundary is not None:
                options['radial_time_origin_ns']=boundary['radius_m']/max(metadata['spectral_tables']['speed'])
            return store.query(rho,orientation,lo,hi,edges,**options)
        if w==0:return node(self.stores[j])
        if w==1:return node(self.stores[j+1])
        q0,b0=node(self.stores[j]);q1,b1=node(self.stores[j+1])
        return (1-w)*q0+w*q1,(1-w)*b0+w*b1


def supported_time_part(kernel,part,position,normal,origin,edges,atlas):
    """Cached time contribution and remaining exact source intervals.

    The existing 1e-3 ns residual clock guard is retained. Cone and ideal
    beta-dependent clock are both contained in the prepared response.
    """
    charge=0.;bins=np.zeros(len(edges)-1);leftovers=[];cached=0
    for i in range(len(part)):
        axis=part.direction[i];delta=position-part.start_m[i];z=float(delta@axis)
        radial=delta-z*axis;rho=float(np.linalg.norm(radial));length=float(part.length_m[i]);beta=float(part.beta[i])
        if rho==0:leftovers.append((i,0.,length));continue
        lo=max(z-length,max(s.x_edges[0] for s in atlas.stores)*rho)
        hi=min(z,min(s.x_edges[-1] for s in atlas.stores)*rho)
        clock=float((part.end_ns[i]-part.start_ns[i])/length);ideal=1/(beta*C_M_PER_NS)
        physical=np.isclose(-part.coefficient2[i]*beta**2,part.coefficient0[i],rtol=1e-10,atol=0)
        if not physical or not atlas.supports(rho,beta,lo,hi) or abs(ideal-clock)*(hi-lo)/2>1e-3:
            leftovers.append((i,0.,length));continue
        ex=radial/rho;ey=np.cross(axis,ex);orientation=np.array([normal@ex,normal@ey,normal@axis])
        closest=float(part.start_ns[i]+z*clock-origin);shift=closest+(ideal-clock)*(lo+hi)/2
        try:q,b=atlas.query(rho,orientation,lo,hi,edges-shift,beta,coefficient0_per_m=part.coefficient0[i]/length)
        except ValueError:
            leftovers.append((i,0.,length));continue
        charge+=q;bins+=b;cached+=1;start=z-hi;end=z-lo
        if start>0:leftovers.append((i,0.,start))
        if end<length:leftovers.append((i,end,length))
    return charge,bins,leftovers,cached
