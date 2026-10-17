"""Finite-track >=2 primitives and the disk atlas built from them.

A row fixes impact distance, OM look in the track frame, beta, clock slope and
optical spectrum; directional acceptance and all stored frequencies are kept.
The radial window of the RTE cache is explicit and does not bound the light
outside it, and the cache keeps its own angular, spectral and frequency errors.
"""
from dataclasses import dataclass, field
import numpy as np
from time import perf_counter,process_time
from numpy.polynomial.legendre import leggauss, legvander, legint, legval
from ..medium import C_VACUUM_M_PER_NS


@dataclass
class SegmentPrimitive:
    """Panel polynomials of the >=2 spectrum along a track and their prefixes."""
    edges_m: np.ndarray
    integrated_coefficients: np.ndarray  # panel, polynomial degree, frequency
    prefix: np.ndarray                  # edge, frequency
    omega_per_ns: np.ndarray
    clock_ns_per_m: float
    metadata: dict

    @classmethod
    def build(cls, evaluator, edges_m, omega_per_ns, *, order=8,
              clock_ns_per_m=1/C_VACUUM_M_PER_NS, metadata=None):
        """Integrate a callable *unshifted* point response with positive exp(iωt).

        The argument u is receiver-minus-emitter axial displacement. Emission
        at u occurs at t_closest-clock*u, hence the NEGATIVE phase below.
        Gauss panels are integrated as local Legendre polynomials. Evaluating
        an endpoint requires one polynomial plus one cumulative-table entry.
        """
        edges = np.asarray(edges_m, float)
        w = np.asarray(omega_per_ns, float)
        if edges.ndim != 1 or len(edges)<2 or np.any(np.diff(edges)<=0) or not np.isfinite(edges).all():
            raise ValueError('edges must be finite and increasing')
        if w.ndim != 1 or not len(w) or w[0]!=0 or not np.isfinite(w).all() or np.any(np.diff(w)<=0):
            raise ValueError('omega must start at zero and increase')
        if not isinstance(order,int) or order<2 or clock_ns_per_m<=0 or not np.isfinite(clock_ns_per_m):
            raise ValueError('invalid quadrature order or clock slope')
        nodes,_ = leggauss(order)
        half = np.diff(edges)/2
        u = (edges[:-1,None]+half[:,None]*(nodes[None,:]+1)).ravel()
        response = np.array(evaluator(u),complex,copy=True)
        if response.shape != (len(u),len(w)) or not np.isfinite(response).all():
            raise ValueError('point evaluator must return finite (points, frequencies)')
        response *= np.exp(-1j*u[:,None]*clock_ns_per_m*w)
        values = response.reshape(len(half),order,len(w))
        coeff = np.einsum('ab,pbf->paf',np.linalg.inv(legvander(nodes,order-1)),values)
        integ = np.stack([legint(c,axis=0) for c in coeff]) * half[:,None,None]
        left = np.stack([legval(-1,c) for c in integ])
        integ[:,0,:] -= left
        mass = np.stack([legval(1,c) for c in integ])
        prefix = np.vstack([np.zeros((1,len(w)),complex),np.cumsum(mass,axis=0)])
        return cls(edges,integ,prefix,w,float(clock_ns_per_m),dict(metadata or {}))

    def primitive(self,u_m):
        """Spectrum integrated from the first edge to ``u_m`` (clamped)."""
        u=np.asarray(u_m,float)
        if not np.isfinite(u).all(): raise ValueError('nonfinite endpoint')
        flat=np.clip(u.ravel(),self.edges_m[0],self.edges_m[-1])
        panel=np.clip(np.searchsorted(self.edges_m,flat,side='right')-1,0,len(self.edges_m)-2)
        x=2*(flat-self.edges_m[panel])/(self.edges_m[panel+1]-self.edges_m[panel])-1
        # legval tensor=False broadcasts frequency and endpoint dimensions.
        coeff=self.integrated_coefficients[panel].transpose(1,2,0)
        value=legval(x[None,:],coeff,tensor=False).T+self.prefix[panel]
        return value.reshape(u.shape+(len(self.omega_per_ns),))

    def segment(self,u_start_m,u_end_m,*,closest_time_ns=0.,coefficient0_per_m=1.):
        """Response for increasing track coordinate: u_start >= u_end.

        Endpoints beyond the row domain are clipped to its radial window.
        This is a stated truncation, never an automatic proof of zero signal.
        """
        a,b=np.broadcast_arrays(np.asarray(u_start_m,float),np.asarray(u_end_m,float))
        if np.any(a<b): raise ValueError('u_start must not be below u_end')
        value=self.primitive(a)-self.primitive(b)
        return (value*np.asarray(coefficient0_per_m)[...,None]
                *np.exp(1j*np.asarray(closest_time_ns)[...,None]*self.omega_per_ns))


def cone_point_evaluator(bundle, *, impact_m, look, alpha, wavelength_nm,
                         wavelength_weight_nm, phase_index, efficiency, beta=1.,
                         effective_area_m2=1.,backend='generic',source_harmonics=None,
                         near_field=False):
    """Use existing full directional >=2 cache with spectral cone harmonics.

    Returns spectrum per unit source coefficient0 (not per emitted photon).
    All points lie in xz plane; look is expressed in that same local frame.
    No charged-track compiler or long-chord quadrature is involved.
    Explicit ``source_harmonics`` replaces the cone coefficients, including
    lambda^-2. It supports signed linear response modes, not a new physical
    cone; caller owns that normalization and approximation.
    ``near_field=True`` (fused-looks only) evaluates points closer than the
    table's radial minimum with the amplitude at that minimum along the same
    directions (see lighthit.near_field); otherwise such points raise.
    """
    from scipy.special import eval_legendre
    from .._shower_basis import apply_multi_spectral
    from ..axial_source import AxisFrame, AxialSource, channel_index
    low,high=np.exp(bundle['log_r'][[0,-1]])
    # An impact below the radial minimum is allowed for points that stay
    # outside it (a short piece far along its own line); evaluate() checks.
    if not 0 < impact_m < high:
        raise ValueError('impact must lie in validated radial interval')
    look=np.asarray(look,float)
    single_look=look.ndim==1
    looks=np.atleast_2d(look)
    if looks.shape[1:]!=(3,) or not np.allclose(np.linalg.norm(looks,axis=1),1.):
        raise ValueError('looks must be unit vectors')
    wavelengths=np.asarray(wavelength_nm,float)
    phases=np.asarray(phase_index,float)
    weights=np.asarray(wavelength_weight_nm,float)*np.asarray(efficiency,float)*effective_area_m2
    if len(wavelengths)!=len(bundle['absorption']) or phases.shape!=wavelengths.shape or weights.shape!=wavelengths.shape:
        raise ValueError('spectral arrays do not match cache')
    if not np.isfinite(beta) or not 0<beta<=1:raise ValueError('beta must be in (0,1]')
    degree=int(bundle['degree']); omega=bundle['omega']
    ell=np.arange(degree+1)
    supplied=None if source_harmonics is None else np.asarray(source_harmonics,float)
    if supplied is not None and (supplied.shape!=(len(wavelengths),degree+1) or not np.isfinite(supplied).all()):
        raise ValueError('source harmonics must be finite wavelength-by-degree coefficients')
    spectral_harmonics=np.empty((len(wavelengths),degree+1))
    for band,(wavelength,index) in enumerate(zip(wavelengths,phases)):
        if supplied is not None:
            harmonics=supplied[band]
        elif beta*index<=1:
            harmonics=np.zeros(degree+1)
        else:
            coefficient=(1-1/(beta*index)**2)/wavelength**2
            harmonics=np.sqrt((2*ell+1)/(4*np.pi))*eval_legendre(ell,1/(beta*index))*coefficient
        spectral_harmonics[band]=harmonics
    if backend not in ('generic','fused-looks'):raise ValueError('unknown cone backend')
    if near_field and backend!='fused-looks':raise ValueError('near_field requires the fused-looks backend')
    if backend=='fused-looks':
        from .._directional_numba import _start_norm_table
        norm=_start_norm_table(int(bundle['acceptance_degree']))
    else:
        frame=AxisFrame(np.zeros(3),np.array([0.,0.,1.]),np.array([1.,0.,0.]),np.array([0.,1.,0.]))
        kept,degrees=channel_index(degree,0)
        sources=[]
        for harmonics in spectral_harmonics:
            channels=np.broadcast_to(harmonics[None,:,None],(1,degree+1,len(omega))).astype(complex).copy()
            sources.append(AxialSource(frame,np.zeros(1),np.zeros((1,2)),channels,kept,degrees,degree,0,0.,{}))
    def evaluate(u):
        """>=2 spectrum of unit emission at track positions ``u``."""
        u=np.asarray(u,float)
        if not near_field and impact_m<low and np.any(np.hypot(impact_m,u)<low*(1-1e-12)):
            raise ValueError('track point closer than the validated radial minimum')
        if backend=='fused-looks':
            result=_cone_many_looks(float(impact_m),u,looks,spectral_harmonics,weights,np.asarray(alpha,float),
                bundle['pair_lambda'],bundle['pair_mu'],bundle['log_r'],bundle['coefficients'],
                bundle['absorption'],bundle['speed'],omega,bundle['phase']=='flight',norm,
                int(bundle['acceptance_degree']),float(low),float(high),0. if near_field else float(low))
            return result[0] if single_look else result
        receivers=np.column_stack([np.full(len(u),impact_m),np.zeros(len(u)),u])
        result=apply_multi_spectral(bundle,sources,np.tile(receivers,(len(looks),1)),np.repeat(looks,len(u),axis=0),alpha,
            np.broadcast_to(weights[:,None],(len(weights),len(u)*len(looks))).copy(),
            radial_window_m=(low,high),channel_dtype=np.complex128).T.reshape(len(looks),len(u),len(omega))
        return result[0] if single_look else result
    return evaluate


def look_polynomials(look):
    """Ten reflection-even angular polynomials for degree<=3 acceptance.

    Reflection flips the phi component, not radial/axial components. On the
    unit sphere phi²=1-radial²-axial²; any remaining degree<=3 response therefore
    belongs to these ten monomials. Signed coefficients are retained.
    """
    look=np.asarray(look,float)
    x,z=look[...,0],look[...,2]
    return np.stack([np.ones_like(x),x,z,x*x,x*z,z*z,x**3,x*x*z,x*z*z,z**3],axis=-1)


def orientation_basis_looks():
    """Pivoted QR chooses ten well-conditioned physical unit directions."""
    from scipy.linalg import qr
    zz=np.linspace(-1,1,21); phi=np.linspace(0,np.pi,25)
    z,ph=np.meshgrid(zz,phi,indexing='ij');r=np.sqrt(np.maximum(0,1-z*z))
    candidates=np.column_stack([(r*np.cos(ph)).ravel(),(r*np.sin(ph)).ravel(),z.ravel()])
    _,_,piv=qr(look_polynomials(candidates).T,pivoting=True,mode='economic')
    looks=candidates[piv[:10]]
    return looks


@dataclass
class TrackGe2MiniAtlas:
    """Reusable rho/orientation prototype at one beta and clock slope.

    Orientation is exact to input acceptance degree3; rho interpolation is a
    new approximation tested independently. No extrapolation is allowed.
    """
    impact_nodes_m: np.ndarray
    rows: list                    # one SegmentPrimitive holding (...,10*nfreq)
    basis_looks: np.ndarray
    inverse_orientation_matrix: np.ndarray
    omega_per_ns: np.ndarray
    front_ns_per_m: float
    metadata: dict = field(default_factory=dict)

    def save(self,path):
        """Persist reusable rows with their optical/source convention metadata."""
        import json
        np.savez_compressed(path,schema="track-ge2-mini-atlas-v1",impact_nodes_m=self.impact_nodes_m,
            edges_m=self.rows[0].edges_m,integrated_coefficients=np.stack([r.integrated_coefficients for r in self.rows]),
            prefix=np.stack([r.prefix for r in self.rows]),basis_looks=self.basis_looks,omega_per_ns=self.omega_per_ns,
            clock_ns_per_m=self.rows[0].clock_ns_per_m,front_ns_per_m=self.front_ns_per_m,metadata=json.dumps(self.metadata))

    @classmethod
    def load(cls,path):
        """Load a saved mini-atlas; caller must match requested optics/beta."""
        import json
        with np.load(path,allow_pickle=False) as data:
            if str(data["schema"])!="track-ge2-mini-atlas-v1":raise ValueError("unsupported atlas schema")
            omega=data["omega_per_ns"].copy();clock=float(data["clock_ns_per_m"])
            rows=[SegmentPrimitive(data["edges_m"].copy(),c.copy(),p.copy(),np.tile(omega,10),clock,{})
                  for c,p in zip(data["integrated_coefficients"],data["prefix"])]
            looks=data["basis_looks"].copy()
            return cls(data["impact_nodes_m"].copy(),rows,looks,np.linalg.inv(look_polynomials(looks)),omega,
                       float(data["front_ns_per_m"]),json.loads(str(data["metadata"])))

    @classmethod
    def build(cls,evaluator_factory,impact_nodes_m,edges_m,omega_per_ns,*,order=8,
              clock_ns_per_m=1/C_VACUUM_M_PER_NS,front_ns_per_m=0.):
        """Tabulate primitives on impact nodes from an evaluator factory."""
        impacts=np.asarray(impact_nodes_m,float);w=np.asarray(omega_per_ns,float)
        if impacts.ndim!=1 or len(impacts)<2 or np.any(np.diff(impacts)<=0) or impacts[0]<=0:
            raise ValueError('at least two increasing positive impact nodes required')
        looks=orientation_basis_looks(); inverse=np.linalg.inv(look_polynomials(looks)); rows=[]
        nodes,_=leggauss(order);edges=np.asarray(edges_m,float);half=np.diff(edges)/2
        u=(edges[:-1,None]+half[:,None]*(nodes+1)).ravel()
        for impact in impacts:
            evaluator=evaluator_factory(float(impact),looks)
            values=np.asarray(evaluator(u),complex)  # look,point,frequency
            if values.shape!=(10,len(u),len(w)):raise ValueError('batched evaluator shape mismatch')
            basis=np.einsum('ab,bpf->paf',inverse,values)
            # Integrate each frequency channel using its original omega phase.
            basis*=np.exp(-1j*u[:,None,None]*clock_ns_per_m*w[None,None,:])
            v=basis.reshape(len(half),order,10*len(w))
            coeff=np.einsum('ab,pbf->paf',np.linalg.inv(legvander(nodes,order-1)),v)
            integ=np.stack([legint(c,axis=0) for c in coeff])*half[:,None,None]
            integ[:,0,:]-=np.stack([legval(-1,c) for c in integ])
            mass=np.stack([legval(1,c) for c in integ])
            prefix=np.vstack([np.zeros((1,10*len(w)),complex),np.cumsum(mass,axis=0)])
            # Internal packed row: endpoints only, never its phase-aware segment().
            rows.append(SegmentPrimitive(edges.copy(),integ,prefix,np.tile(w,10),float(clock_ns_per_m),{}))
        return cls(impacts,rows,looks,inverse,w,float(front_ns_per_m))

    def segment(self,impact_m,look,u_start_m,u_end_m,*,closest_time_ns=0.,
                coefficient0_per_m=1.,alignment='raw'):
        """One geometry, vector endpoints supported; no radial extrapolation.

        front removes the infinite-track phase before interpolation. scaled
        also keeps u/rho fixed across neighboring rows. Both remain numerical
        interpolation strategies, not exact invariances in absorbing water.
        """
        rho=float(impact_m)
        if not self.impact_nodes_m[0]<=rho<=self.impact_nodes_m[-1]:raise ValueError('impact outside atlas')
        look=np.asarray(look,float)
        if look.shape!=(3,) or not np.isclose(np.linalg.norm(look),1.):raise ValueError('look must be unit')
        if alignment not in ('raw','front','scaled'):raise ValueError('unknown interpolation alignment')
        a,b=np.broadcast_arrays(np.asarray(u_start_m,float),np.asarray(u_end_m,float))
        if np.any(a<b):raise ValueError('reversed segment')
        # Local quadratic interpolation (linear only for a two-row atlas).
        begin=int(np.clip(np.searchsorted(self.impact_nodes_m,rho)-1,0,max(0,len(self.rows)-3)))
        selected=np.arange(begin,min(begin+3,len(self.rows)))
        nodes=self.impact_nodes_m[selected]
        rows=[self.rows[i] for i in selected]
        weights=[]
        for i,node in enumerate(nodes):
            weight=1.
            for j,other in enumerate(nodes):
                if i!=j:weight*=(rho-other)/(node-other)
            weights.append(weight)
        result=np.zeros(a.shape+(len(self.omega_per_ns),),complex)
        angular=look_polynomials(look)
        for i,(row,node,weight) in enumerate(zip(rows,nodes,weights)):
            scale=node/rho if alignment=='scaled' else 1.
            basis=(row.primitive(a*scale)-row.primitive(b*scale)).reshape(a.shape+(10,len(self.omega_per_ns)))
            value=np.einsum('...af,a->...f',basis,angular)
            if alignment!='raw':value*=np.exp(1j*self.omega_per_ns*self.front_ns_per_m*(rho-node))
            result+=weight*value
        return result*np.asarray(coefficient0_per_m)[...,None]*np.exp(1j*np.asarray(closest_time_ns)[...,None]*self.omega_per_ns)


def track_ge2_optics_metadata(kernel):
    """Optical/source signature independent of detector positions and events."""
    import hashlib,json
    wavelength=kernel.wavelength
    sample=kernel.medium.sample(wavelength.wavelength_nm)
    areas=np.asarray(kernel.detector.effective_area_m2,float)
    if not np.all(areas==areas[0]):raise ValueError('one shared OM area required')
    data=dict(wavelength_nm=wavelength.wavelength_nm.tolist(),wavelength_weight_nm=wavelength.weight_nm.tolist(),
              **{name:np.asarray(value).tolist() for name,value in sample.items()},
              g=float(kernel.medium.g),alpha=np.asarray(kernel.acceptance()['alpha']).tolist(),
              efficiency=np.asarray(kernel.detector.spectral_weight(wavelength.wavelength_nm)).tolist(),
              effective_area_m2=float(areas[0]),beta=1.,clock_ns_per_m=1/C_VACUUM_M_PER_NS)
    data['optics_fingerprint']=hashlib.sha256(json.dumps(data,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return data


def track_ge2_axial_edges(impact_m,radial_high_m=300.,*,max_panel_m=4.,asinh_step=.15):
    """Resolve near-cone angular structure while bounding far flight oscillations."""
    rho=float(impact_m);reach=np.sqrt(max(0.,radial_high_m**2-rho**2))
    if reach<=0:raise ValueError('impact must be below radial cutoff')
    stop=np.arcsinh(reach/rho)
    adaptive=rho*np.sinh(np.linspace(-stop,stop,max(2,int(np.ceil(2*stop/asinh_step))+1)))
    # Split each adaptive panel as needed, avoiding nearly duplicate merged grids.
    edges=[adaptive[0]]
    for left,right in zip(adaptive[:-1],adaptive[1:]):
        edges.extend(np.linspace(left,right,max(1,int(np.ceil((right-left)/max_panel_m)))+1)[1:])
    return np.asarray(edges)


def build_track_ge2_orientation_row(evaluator,edges_m,omega_per_ns,*,order=8,
                                   clock_ns_per_m=1/C_VACUUM_M_PER_NS):
    """Build10orientation-polynomial primitives in one batched physical apply.

    evaluator(u) must return (10selectedlooks,points,frequency), in the order
    from orientation_basis_looks(). No G4/source trajectory is accepted.
    """
    edges=np.asarray(edges_m,float);omega=np.asarray(omega_per_ns,float)
    nodes,_=leggauss(order);half=np.diff(edges)/2
    u=(edges[:-1,None]+half[:,None]*(nodes+1)).ravel()
    values=np.asarray(evaluator(u),complex)
    if values.shape!=(10,len(u),len(omega)):raise ValueError('invalid orientation evaluator shape')
    inverse=np.linalg.inv(look_polynomials(orientation_basis_looks()))
    basis=np.einsum('ab,bpf->paf',inverse,values)
    basis*=np.exp(-1j*u[:,None,None]*clock_ns_per_m*omega)
    v=basis.reshape(len(half),order,10*len(omega))
    coeff=np.einsum('ab,pbf->paf',np.linalg.inv(legvander(nodes,order-1)),v)
    integ=np.stack([legint(c,axis=0) for c in coeff])*half[:,None,None]
    integ[:,0,:]-=np.stack([legval(-1,c) for c in integ])
    mass=np.stack([legval(1,c) for c in integ])
    prefix=np.vstack([np.zeros((1,10*len(omega)),complex),np.cumsum(mass,axis=0)])
    return SegmentPrimitive(edges,integ,prefix,np.tile(omega,10),float(clock_ns_per_m),{})


class _ReadMeter:
    """Measure compressed-file reads separately from NumPy/zip decoding."""
    def __init__(self,stream):
        """Wrap a file object and count read time and bytes."""
        self.stream=stream;self.wall=0.;self.cpu=0.;self.bytes=0
    def read(self,*args):
        """Read and account the time and size."""
        wall,cpu=perf_counter(),process_time();data=self.stream.read(*args)
        self.wall+=perf_counter()-wall;self.cpu+=process_time()-cpu;self.bytes+=len(data)
        return data
    def __getattr__(self,name):
        """Everything else goes to the wrapped stream."""
        return getattr(self.stream,name)


class DiskTrackGe2Atlas:
    """Event-independent impact atlas with charge preload and bounded full-row LRU.

    A requested impact outside the stored domain raises instead of silently
    extrapolating. Radial truncation and validation scope are in metadata.
    Only the query's3neighbor rows enter RAM; allω0rows are preloaded.
    """
    def __init__(self,directory,*,max_loaded_rows=3,require_complete=True,require_validated=True):
        """Open an atlas directory; rows are read on demand and verified by hash."""
        from pathlib import Path
        from collections import OrderedDict
        import json
        self.directory=Path(directory)
        self.manifest=json.loads((self.directory/'manifest.json').read_text())
        if self.manifest.get('schema')!='lighthit/track-ge2-disk-atlas/1':raise ValueError('unsupported disk atlas')
        if require_complete and self.manifest.get('status')!='complete':raise ValueError('atlas manifest is incomplete')
        self.require_validated=bool(require_validated)
        self._verified={}
        self.metadata=self.manifest['metadata']
        self.omega_per_ns=np.asarray(self.manifest['omega_per_ns'],float)
        self.impact_nodes_m=np.asarray([item['impact_m'] for item in self.manifest['rows']],float)
        if (len(self.impact_nodes_m)<3 or not np.isfinite(self.impact_nodes_m).all()
                or np.any(np.diff(self.impact_nodes_m)<=0)):
            raise ValueError('atlas radial manifest is incomplete or unordered')
        self.front_ns_per_m=float(self.manifest['front_ns_per_m'])
        self.max_loaded_rows=int(max_loaded_rows)
        if self.max_loaded_rows<3:raise ValueError('LRU requires at least3rows')
        self._cache=OrderedDict();self._charge=[];self.full_row_loads=0
        self.io_statistics=dict(verify_wall_seconds=0.,verify_cpu_seconds=0.,verified_bytes=0,
            row_wall_seconds=0.,row_cpu_seconds=0.,read_wall_seconds=0.,read_cpu_seconds=0.,read_bytes=0)
        self.row_io_log=[]
        for item in self.manifest['rows']:
            self._verify_file(item,'charge_file','charge_sha256')
            with np.load(self.directory/item['charge_file'],allow_pickle=False) as z:
                self._charge.append(SegmentPrimitive(z['edges_m'].copy(),z['integrated_coefficients'].real.copy(),z['prefix'].real.copy(),
                       np.zeros(10),float(self.metadata['clock_ns_per_m']),{}))

    def _verify_file(self,item,file_key,hash_key):
        """Check one row file against its recorded SHA-256."""
        import hashlib
        path=self.directory/item[file_key]
        expected=item.get(hash_key)
        if not expected:
            if self.manifest.get('status')=='complete':raise ValueError('atlas row missing checksum')
            return
        stat=path.stat();stamp=(stat.st_size,stat.st_mtime_ns)
        if self._verified.get(item[file_key])==stamp:return
        wall,cpu=perf_counter(),process_time()
        digest=hashlib.sha256()
        with path.open('rb') as stream:
            while chunk:=stream.read(1024*1024):digest.update(chunk)
        if digest.hexdigest()!=expected:raise ValueError('atlas row checksum mismatch: '+item[file_key])
        self._verified[item[file_key]]=stamp
        self.io_statistics['verify_wall_seconds']+=perf_counter()-wall
        self.io_statistics['verify_cpu_seconds']+=process_time()-cpu
        self.io_statistics['verified_bytes']+=stat.st_size

    def validate_kernel(self,kernel):
        """Refuse a kernel with other optics."""
        actual=track_ge2_optics_metadata(kernel)['optics_fingerprint']
        if actual!=self.metadata.get('optics_fingerprint'):raise ValueError('atlas/kernel optics mismatch')
        return True

    def is_validated(self,impact_m,*,charge_only=False):
        """True when the impact distance lies in a validated interval."""
        rho=float(impact_m)
        if not self.impact_nodes_m[0]<=rho<=self.impact_nodes_m[-1]:return False
        if np.any(self.impact_nodes_m==rho):return True
        checks=self.manifest.get('validation',[])
        flag='charge_passed' if charge_only else 'front_passed'
        return any(c['low_m']<=rho<=c['high_m'] and c.get(flag,c['passed']) for c in checks)

    def query_supported(self,impact_m,u_start_m,u_end_m,*,charge_only=False):
        """True when the atlas may answer this piece; otherwise use the direct fallback."""
        rho=float(impact_m)
        if not self.is_validated(rho,charge_only=charge_only):return False
        if charge_only or np.any(self.impact_nodes_m==rho) or np.all(np.asarray(u_start_m)>0):return True
        return any(c['low_m']<=rho<=c['high_m'] and c.get('backward_passed',c['passed']) for c in self.manifest.get('validation',[]))

    def _row(self,index,charge_only):
        """Primitive row ``index`` (charge-only rows are always resident)."""
        if charge_only:return self._charge[index]
        if index in self._cache:
            self._cache.move_to_end(index);return self._cache[index]
        item=self.manifest['rows'][index]
        self._verify_file(item,'file','sha256')
        while len(self._cache)>=self.max_loaded_rows:self._cache.popitem(last=False)
        wall,cpu=perf_counter(),process_time()
        with (self.directory/item['file']).open('rb') as stream:
            meter=_ReadMeter(stream)
            with np.load(meter,allow_pickle=False) as z:
                row=SegmentPrimitive(z['edges_m'].copy(),z['integrated_coefficients'].copy(),z['prefix'].copy(),
                                     np.tile(self.omega_per_ns,10),float(self.metadata['clock_ns_per_m']),{})
        elapsed=perf_counter()-wall;used=process_time()-cpu
        self.io_statistics['row_wall_seconds']+=elapsed
        self.io_statistics['row_cpu_seconds']+=used
        self.io_statistics['read_wall_seconds']+=meter.wall
        self.io_statistics['read_cpu_seconds']+=meter.cpu
        self.io_statistics['read_bytes']+=meter.bytes
        self.row_io_log.append(dict(index=int(index),wall_seconds=elapsed,cpu_seconds=used,
            read_wall_seconds=meter.wall,read_cpu_seconds=meter.cpu,read_bytes=meter.bytes))
        self._cache[index]=row;self.full_row_loads+=1
        while len(self._cache)>self.max_loaded_rows:self._cache.popitem(last=False)
        return row

    @property
    def resident_bytes(self):
        """Memory held by loaded rows and packed charge tables."""
        return (sum(r.integrated_coefficients.nbytes+r.prefix.nbytes+r.edges_m.nbytes
                   for r in list(self._cache.values())+self._charge)
                +sum(a.nbytes for a in getattr(self,'_packed_charge',())))

    def _segment(self,rho,look,a,b,*,charge_only,alignment='scaled'):
        """Spectrum (or charge) of a piece, interpolated in impact distance."""
        rho=float(rho);look=np.asarray(look,float)
        if not self.impact_nodes_m[0]<=rho<=self.impact_nodes_m[-1]:raise ValueError('impact outside atlas; direct fallback required')
        if look.shape!=(3,) or not np.isclose(np.linalg.norm(look),1.):raise ValueError('look must be unit')
        if self.require_validated and not self.query_supported(rho,a,b,charge_only=charge_only):raise ValueError('impact/endpoint region failed validation; direct fallback required')
        if alignment not in ('raw','front','scaled'):raise ValueError('unknown alignment')
        a,b=np.broadcast_arrays(np.asarray(a,float),np.asarray(b,float))
        if np.any(a<b):raise ValueError('reversed endpoints')
        begin=int(np.clip(np.searchsorted(self.impact_nodes_m,rho)-1,0,max(0,len(self.impact_nodes_m)-3)))
        selected=np.arange(begin,min(begin+3,len(self.impact_nodes_m)));nodes=self.impact_nodes_m[selected]
        weights=np.array([np.prod([(rho-other)/(node-other) for j,other in enumerate(nodes) if j!=i]) for i,node in enumerate(nodes)])
        w=np.array([0.]) if charge_only else self.omega_per_ns
        result=np.zeros(a.shape+(len(w),),complex);angular=look_polynomials(look)
        lognorm=0.
        for index,node,weight in zip(selected,nodes,weights):
            row=self._row(int(index),charge_only);scale=node/rho if alignment=='scaled' else 1.
            pa=row.primitive(a*scale).reshape(a.shape+(10,len(w)))
            pb=row.primitive(b*scale).reshape(a.shape+(10,len(w)))
            va=np.einsum('...af,a->...f',pa,angular)
            vb=np.einsum('...af,a->...f',pb,angular)
            if alignment!='raw' and not charge_only and self.front_ns_per_m>0:
                # A prefix ending behind the unconstrained Cherenkov root has
                # its own earliest arrival. Infinite-track front alignment is
                # wrong there, even after scaling u/rho.
                clock=float(self.metadata['clock_ns_per_m'])
                optical_slowness=np.hypot(clock,self.front_ns_per_m)
                target_reach=np.sqrt(max(0.,float(self.metadata.get('radius_window_m',[0,300.])[1])**2-rho*rho))
                def front_at(u,radius,lo,hi):
                    """Arrival-front delay used to align neighbouring rows in time."""
                    u=np.clip(u,lo,hi)
                    stationary=radius*clock/self.front_ns_per_m
                    return np.where(u<stationary,optical_slowness*np.hypot(radius,u)-clock*u,radius*self.front_ns_per_m)
                for value,endpoint in ((va,a),(vb,b)):
                    target=front_at(endpoint,rho,-target_reach,target_reach)
                    native=front_at(endpoint*scale,node,row.edges_m[0],row.edges_m[-1])
                    value*=np.exp(1j*(target-native)[...,None]*w)
            result+=weight*(va-vb)
            lognorm+=weight*np.log(self.manifest['rows'][int(index)]['normalization'])
        return result*np.exp(lognorm)

    def segment(self,impact_m,look,u_start_m,u_end_m,*,alignment='scaled',closest_time_ns=0.,coefficient0_per_m=1.):
        """>=2 spectrum of a piece with its emission clock and C0 per metre."""
        value=self._segment(impact_m,look,u_start_m,u_end_m,charge_only=False,alignment=alignment)
        return value*np.asarray(coefficient0_per_m)[...,None]*np.exp(1j*np.asarray(closest_time_ns)[...,None]*self.omega_per_ns)

    def segment_charge_many(self,impact_m,look,u_start_m,u_end_m,*,coefficient0_per_m=1.,alignment='scaled'):
        """All-time >=2 charge of many pieces."""
        return _disk_charge_many(self,impact_m,look,u_start_m,u_end_m,coefficient0_per_m=coefficient0_per_m,alignment=alignment)

    def segment_charge(self,impact_m,look,u_start_m,u_end_m,*,alignment='scaled',coefficient0_per_m=1.,closest_time_ns=0.):
        """Integral charge without loading any full-frequency row; clockignored."""
        return (self._segment(impact_m,look,u_start_m,u_end_m,charge_only=True,alignment=alignment)[...,0].real
                *np.asarray(coefficient0_per_m))


def _disk_charge_many(self,impact_m,look,u_start_m,u_end_m,*,coefficient0_per_m=1.,alignment='scaled'):
    """Vector charge query for many pairs without any full-frequency row IO."""
    from numba import njit
    if alignment!='scaled':raise ValueError('batch charge currently uses scaled coordinates')
    rho,a,b,density=np.broadcast_arrays(np.asarray(impact_m,float),np.asarray(u_start_m,float),
        np.asarray(u_end_m,float),np.asarray(coefficient0_per_m,float))
    directions=np.asarray(look,float)
    if directions.shape!=rho.shape+(3,) or not np.allclose(np.linalg.norm(directions,axis=-1),1.):
        raise ValueError('one unit look per pair required')
    if not np.isfinite(rho).all() or np.any(rho<self.impact_nodes_m[0]) or np.any(rho>self.impact_nodes_m[-1]):
        raise ValueError('impact outside atlas; direct fallback required')
    if np.any(a<b) or not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('invalid endpoints')
    if self.require_validated and any(not self.is_validated(r,charge_only=True) for r in np.unique(rho)):raise ValueError('impact interval failed validation; direct fallback required')
    if not hasattr(self,'_packed_charge'):
        edge_offsets=np.r_[0,np.cumsum([len(r.edges_m) for r in self._charge])].astype(np.int64)
        panel_offsets=np.r_[0,np.cumsum([len(r.edges_m)-1 for r in self._charge])].astype(np.int64)
        degree=max(r.integrated_coefficients.shape[1] for r in self._charge)
        coef=np.zeros((panel_offsets[-1],degree,10))
        for i,row in enumerate(self._charge):coef[panel_offsets[i]:panel_offsets[i+1],:row.integrated_coefficients.shape[1]]=row.integrated_coefficients.real
        self._packed_charge=(edge_offsets,panel_offsets,np.concatenate([r.edges_m for r in self._charge]),coef,
            np.concatenate([r.prefix.real for r in self._charge]),np.log([item['normalization'] for item in self.manifest['rows']]))
    result=_charge_pairs(rho.ravel().copy(),directions.reshape(-1,3),a.ravel().copy(),b.ravel().copy(),density.ravel().copy(),self.impact_nodes_m,*self._packed_charge)
    return result.reshape(rho.shape)


from numba import njit as _ge2_njit


@_ge2_njit(cache=True)
def _charge_endpoint(u,angular,row,edge_offsets,panel_offsets,edges,coefficients,prefix):
    """Charge prefix of one row at track position ``u``."""
    start=edge_offsets[row];stop=edge_offsets[row+1]
    u=min(max(u,edges[start]),edges[stop-1])
    local=min(max(np.searchsorted(edges[start:stop],u,side='right')-1,0),stop-start-2)
    idx=panel_offsets[row]+local;eidx=start+local
    x=2*(u-edges[eidx])/(edges[eidx+1]-edges[eidx])-1
    total=0.
    for m in range(10):
        # Evaluate Legendre polynomial in ascending recurrence; degree<=10.
        p0=1.;p1=x
        value=coefficients[idx,0,m]
        if coefficients.shape[1]>1:value+=coefficients[idx,1,m]*x
        for n in range(2,coefficients.shape[1]):
            current=((2*n-1)*x*p1-(n-1)*p0)/n
            value+=coefficients[idx,n,m]*current;p0=p1;p1=current
        total+=angular[m]*(prefix[eidx,m]+value)
    return total


@_ge2_njit(cache=True)
def _charge_pairs(rho,looks,a,b,density,nodes,edge_offsets,panel_offsets,edges,coefficients,prefix,lognorm):
    """Charge of many pieces from the packed charge tables (interpolated in radius)."""
    output=np.zeros(len(rho))
    for pair in range(len(rho)):
        radius=rho[pair];x=looks[pair,0];z=looks[pair,2]
        angular=np.array([1.,x,z,x*x,x*z,z*z,x*x*x,x*x*z,x*z*z,z*z*z])
        begin=min(max(np.searchsorted(nodes,radius)-1,0),max(0,len(nodes)-3))
        count=min(3,len(nodes)-begin);value=0.;scale_log=0.
        for pos in range(count):
            row=begin+pos;node=nodes[row];weight=1.
            for other in range(count):
                if other!=pos:weight*=(radius-nodes[begin+other])/(node-nodes[begin+other])
            scale=node/radius
            difference=(_charge_endpoint(a[pair]*scale,angular,row,edge_offsets,panel_offsets,edges,coefficients,prefix)
                       -_charge_endpoint(b[pair]*scale,angular,row,edge_offsets,panel_offsets,edges,coefficients,prefix))
            value+=weight*difference;scale_log+=weight*lognorm[row]
        output[pair]=value*np.exp(scale_log)*density[pair]
    return output



from .._directional_numba import _wigner_into as _ge2_wigner, _harmonics_into as _ge2_detector_harmonics


@_ge2_njit(cache=True)
def _cone_many_looks(impact,u,looks,harmonics,weights,alpha,pair_lambda,pair_mu,log_r,coefficients,
                     absorption,velocity,omega,flight,start_norm,max_out,radial_low,radial_high,
                     clamp_low):
    """Contract spectral RTE once per point, then project every OM orientation.

    Points closer than ``clamp_low`` are skipped. Points between ``clamp_low``
    and ``radial_low`` take the radial amplitude at ``radial_low`` (same
    directions, flight phase at the actual radius); ``clamp_low=radial_low``
    disables this near-field clamp.
    """
    degree=harmonics.shape[1]-1;nfreq=len(omega);npair=len(pair_lambda);nlam=len(weights)
    result=np.zeros((len(looks),len(u),nfreq),np.complex128)
    wigner=np.empty((degree+1,2*max_out+1,1),np.float64)
    detector=np.empty((max_out+1)**2,np.float64)
    response=np.empty((npair,nfreq),np.complex128)
    phase=np.empty((nlam,nfreq),np.complex128)
    for point in range(len(u)):
        radius=np.hypot(impact,u[point])
        if radius<clamp_low or radius>radial_high or radius==0.:continue
        cosine=u[point]/radius;sine=impact/radius
        _ge2_wigner(degree,max_out,0,cosine,start_norm,max_out,wigner)
        amplitude_radius=max(radius,radial_low)
        j=min(max(np.searchsorted(log_r,np.log(amplitude_radius),side='right')-1,0),len(log_r)-2)
        t=np.log(amplitude_radius)-log_r[j]
        for lam in range(nlam):
            scale=weights[lam]*np.exp(-absorption[lam]*amplitude_radius)/(4*np.pi*amplitude_radius*amplitude_radius)
            for f in range(nfreq):
                angle=omega[f]*radius/velocity[lam] if flight else 0.
                phase[lam,f]=scale*(np.cos(angle)+1j*np.sin(angle))
        response[:]=0.
        for ell in range(degree+1):
            for p in range(npair):
                mu=pair_mu[p]
                if mu>ell:continue
                rotation=wigner[ell,max_out,0] if mu==0 else np.sqrt(2.)*((-1.)**mu)*wigner[ell,max_out+mu,0]
                for lam in range(nlam):
                    h=harmonics[lam,ell]*rotation
                    for f in range(nfreq):
                        radial=((coefficients[lam,j,ell,p,0,f]*t+coefficients[lam,j,ell,p,1,f])*t
                                +coefficients[lam,j,ell,p,2,f])*t+coefficients[lam,j,ell,p,3,f]
                        response[p,f]+=h*radial*phase[lam,f]
        for look in range(len(looks)):
            lx=looks[look,0];ly=looks[look,1];lz=looks[look,2]
            _ge2_detector_harmonics(cosine*lx-sine*lz,ly,sine*lx+cosine*lz,max_out,detector)
            for p in range(npair):
                degree_d=pair_lambda[p];mu=pair_mu[p]
                factor=alpha[degree_d]*detector[degree_d*degree_d+degree_d+mu]
                for f in range(nfreq):result[look,point,f]+=factor*response[p,f]
    return result
