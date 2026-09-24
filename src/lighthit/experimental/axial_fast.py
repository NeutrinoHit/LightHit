"""Exact-representation fast apply for the sparse axial source (1bb11dc).

No source compression, phase approximation or new transport discretisation.
Both cached scattering components are computed in one pass. A spline is
interpolated once per (cell,receiver,l,frequency), not once per m; harmonics,
geometry and source sums are shared between orders. Numba is optional until
this module is explicitly imported. No fastmath is used.

The public API returns (frequency, receiver, 2): finite-L one, >=2.
Ballistic light and readout remain separate and unchanged.
"""
from dataclasses import dataclass
import numpy as np
from scipy.interpolate import CubicSpline
try:
    from numba import njit, prange
except ImportError as exc:
    raise ImportError("axial_fast requires the optional 'accelerate' extra") from exc


@njit(cache=True, nogil=True)
def _harmonics_into(x, y, z, degree, max_m, starts, aa, bb, out):
    radius=np.sqrt(x*x+y*y+z*z)
    mu=z/radius
    # This form avoids cancellation for angles near the polar axis.
    transverse=np.sqrt(x*x+y*y)
    sine=transverse/radius
    cp=1.0 if transverse == 0.0 else x/transverse
    sp=0.0 if transverse == 0.0 else y/transverse
    cm=1.0; sm=0.0
    diagonal=1.0/np.sqrt(4*np.pi)
    for m in range(max_m+1):
        if m:
            diagonal *= np.sqrt((2.0*m+1)/(2.0*m))*sine
            cm,sm=cm*cp-sm*sp,sm*cp+cm*sp
        v0=diagonal
        for ell in range(m,degree+1):
            if ell == m:
                value=v0
            elif ell == m+1:
                value=np.sqrt(2*m+3)*mu*v0
                v1=value
            else:
                value=aa[ell,m]*mu*v1-bb[ell,m]*v0
                v0,v1=v1,value
            centre=starts[ell]+min(ell,max_m)
            if m == 0:
                out[centre]=value
            else:
                out[centre-m]=np.sqrt(2)*value*sm
                out[centre+m]=np.sqrt(2)*value*cm
    return radius


@njit(cache=True, nogil=True, parallel=True)
def _apply(points, receivers, source, starts, aa, bb, max_m, log_r,
           coefficients, absorption, velocity, omega, flight, receiver_block):
    # coefficients: (radial interval, degree, cubic power, order, frequency)
    nfreq=len(omega); degree=len(starts)-2
    nrec=len(receivers); nch=source.shape[1]
    result=np.zeros((nrec,nfreq,2),np.complex128)
    for group in prange((nrec+receiver_block-1)//receiver_block):
        begin=group*receiver_block
        count=min(receiver_block,nrec-begin)
        ylm=np.empty((receiver_block,nch),np.float64)
        interval=np.empty(receiver_block,np.int64)
        delta=np.empty(receiver_block,np.float64)
        scale_phase=np.empty((receiver_block,nfreq),np.complex128)
        angular=np.empty((receiver_block,nfreq),np.complex128)
        for cell in range(len(points)):
            for d in range(count):
                dx=receivers[begin+d,0]-points[cell,0]
                dy=receivers[begin+d,1]-points[cell,1]
                dz=receivers[begin+d,2]-points[cell,2]
                r=_harmonics_into(dx,dy,dz,degree,max_m,starts,aa,bb,ylm[d])
                logr=np.log(r)
                j=np.searchsorted(log_r,logr,side='right')-1
                j=min(j,len(log_r)-2)
                interval[d]=j
                delta[d]=logr-log_r[j]
                scale=np.exp(-absorption*r)/(4*np.pi*r*r)
                for w in range(nfreq):
                    scale_phase[d,w]=scale
                    if flight:
                        phase=omega[w]*r/velocity
                        scale_phase[d,w]*=np.cos(phase)+1j*np.sin(phase)
            for ell in range(degree+1):
                for d in range(count):
                    for w in range(nfreq):
                        angular[d,w]=0.0
                weight=4*np.pi/(2*ell+1)
                # Contract the azimuthal channels FIRST. The radial moment
                # has no m index; it must never be expanded to (l,m).
                for c in range(starts[ell],starts[ell+1]):
                    for d in range(count):
                        factor=weight*ylm[d,c]
                        for w in range(nfreq):
                            angular[d,w]+=factor*source[cell,c,w]
                for d in range(count):
                    j=interval[d]; t=delta[d]
                    for w in range(nfreq):
                        value=angular[d,w]*scale_phase[d,w]
                        for order in range(2):
                            coeff=coefficients[j,ell,:,order,w]
                            radial=((coeff[0]*t+coeff[1])*t+coeff[2])*t+coeff[3]
                            result[begin+d,w,order]+=value*radial
    return result


def _layout(degree,max_m):
    starts=[0]; kept=[]; ell_of=[]
    for ell in range(degree+1):
        for m in range(-min(ell,max_m),min(ell,max_m)+1):
            kept.append(ell*ell+ell+m); ell_of.append(ell)
        starts.append(len(kept))
    aa=np.zeros((degree+1,max_m+1)); bb=aa.copy()
    for m in range(max_m+1):
        for ell in range(m+2,degree+1):
            aa[ell,m]=np.sqrt((4*ell*ell-1)/(ell*ell-m*m))
            bb[ell,m]=np.sqrt(((2*ell+1)*((ell-1)**2-m*m))/((2*ell-3)*(ell*ell-m*m)))
    return np.asarray(starts,np.int64),aa,bb,np.asarray(kept,np.int64),np.asarray(ell_of,np.int64)


@dataclass
class PreparedAxialKernel:
    """A reusable copy of the SAME radial CubicSpline used by ResponseCache.

    Build once per cache and retained degree. The source frequency array must
    match omega_per_ns in both length AND values, explicitly supplied to apply.
    Mutating the original cache after preparation is not reflected here.
    """
    degree: int
    omega_per_ns: np.ndarray
    log_r: np.ndarray
    coefficients: np.ndarray
    absorption_per_m: float
    speed_m_per_ns: float
    radial_phase: str

    @classmethod
    def from_cache(cls,cache,degree=None,frequency_indices=None):
        if hasattr(cache,'bands'):
            raise ValueError("Pass one ResponseCache; multi-band routing is not implicit")
        degree=cache.degree if degree is None else degree
        if not isinstance(degree,(int,np.integer)) or not 0<=degree<=cache.degree:
            raise ValueError("degree must be an integer within the cache")
        axis=np.asarray(cache.grid.omega_per_ns,float)
        if frequency_indices is None:
            index=np.arange(len(axis))
        else:
            index=np.asarray(frequency_indices)
            if index.ndim!=1 or not len(index) or index.dtype.kind not in 'iu':
                raise ValueError('frequency_indices must be a nonempty integer vector')
            if np.any(index<0) or np.any(index>=len(axis)):
                raise ValueError('frequency index out of range')
        omega=axis[index].copy()
        radii=np.asarray(cache.grid.radii_m,float)
        scale=np.exp(-cache.medium.absorption_per_m*radii)/(4*np.pi*radii**2)
        if np.any(scale==0) or not np.isfinite(scale).all():
            raise ValueError('Radial scale underflow/nonfinite')
        values=cache.moments[index,:,:degree+1,:]/scale[None,:,None,None]
        phase=getattr(cache,'radial_phase','none')
        if phase not in ('none','flight'): raise ValueError('Unknown radial_phase')
        if phase=='flight':
            values*=np.exp(-1j*omega[:,None]*radii[None,:]/cache.medium.speed_m_per_ns)[:,:,None,None]
        spline=CubicSpline(np.log(radii),values,axis=1)
        coefficients=np.ascontiguousarray(np.transpose(spline.c,(1,3,0,4,2)))
        return cls(int(degree),omega,np.log(radii),coefficients,
                   cache.medium.absorption_per_m,cache.medium.speed_m_per_ns,phase)

    def apply(self,source,receivers_m,*,source_omega_per_ns,receiver_block=4):
        """Return both scattered orders, (frequency, receiver, order).

        source_omega_per_ns is mandatory because the present AxialSource does
        not record its own frequency grid. No source phases are reconstructed.
        """
        axis=np.asarray(source_omega_per_ns,float)
        if axis.shape!=self.omega_per_ns.shape or not np.array_equal(axis,self.omega_per_ns):
            raise ValueError('Source and kernel frequency grids differ')
        if self.degree>source.degree: raise ValueError('Source degree is insufficient')
        if not isinstance(receiver_block,(int,np.integer)) or receiver_block<1:
            raise ValueError('receiver_block must be positive integer')
        receivers=np.atleast_2d(np.asarray(receivers_m,float))
        if receivers.ndim!=2 or receivers.shape[1]!=3 or not len(receivers) or not np.isfinite(receivers).all():
            raise ValueError('receivers must be a nonempty finite (D,3) array')
        max_m=min(int(source.azimuthal_degree),self.degree)
        starts,aa,bb,kept,degrees=_layout(self.degree,max_m)
        if not np.array_equal(source.kept[:len(kept)],kept):
            raise ValueError('Unsupported source harmonic ordering')
        if not np.array_equal(source.channel_degree[:len(kept)],degrees):
            raise ValueError('Invalid source degree metadata')
        data=np.asarray(source.channels)
        if data.dtype!=np.complex128 or data.ndim!=3 or data.shape[2]!=len(axis):
            raise ValueError('Expected complex128 (cell,channel,frequency) source')
        if data.shape[1]<len(kept): raise ValueError('Missing source channels')
        points=np.asarray(source.points_m(),float)
        if len(points)!=len(data) or points.shape!=(len(data),3) or not np.isfinite(points).all():
            raise ValueError('Invalid source points')
        # Translate before the rotation to reduce cancellation in large world coordinates.
        centre=np.asarray(source.frame.centre_m,float)
        local_points=np.ascontiguousarray(source.frame.rotate(points-centre))
        local_receivers=np.ascontiguousarray(source.frame.rotate(receivers-centre))
        # Bounds checked once without constructing a detector x cell x channel array.
        for receiver in local_receivers:
            radius=np.linalg.norm(local_points-receiver,axis=1)
            if np.any(radius<np.exp(self.log_r[0])) or np.any(radius>np.exp(self.log_r[-1])):
                raise ValueError('Source/receiver distances outside the cache')
        result=_apply(local_points,local_receivers,data[:,:len(kept),:],starts,aa,bb,
                      max_m,self.log_r,self.coefficients,self.absorption_per_m,
                      self.speed_m_per_ns,self.omega_per_ns,self.radial_phase=='flight',
                      int(receiver_block))
        carrier=np.exp(1j*self.omega_per_ns*source.reference_ns)
        return np.transpose(result,(1,0,2))*carrier[:,None,None]


@njit(cache=True,nogil=True)
def _element_angular(vectors,cones,starts,aa,bb,max_m):
    degree=len(starts)-2
    result=np.empty((len(vectors),starts[-1]),np.float64)
    for i in range(len(vectors)):
        _harmonics_into(vectors[i,0],vectors[i,1],vectors[i,2],degree,max_m,
                        starts,aa,bb,result[i])
        prev=1.0; current=cones[i]
        for ell in range(degree+1):
            if ell==0: p=1.0
            elif ell==1: p=current
            else:
                p=((2*ell-1)*cones[i]*current-(ell-1)*prev)/ell
                prev,current=current,p
            for c in range(starts[ell],starts[ell+1]):
                result[i,c]*=p
    return result


def accumulate_cell_gemm(spread, angular, node_elements, times_ns, omega_per_ns,
                         reference_ns, out, *, link_chunk=8192):
    """Accumulate D[b,n] H[element(n),c] exp(i*w*t[n]) without n*c*w payload.

    D is a CSR deposit matrix with the photon counts AND chord quadrature
    weights. This is precisely the current cloud-in-cell sum, reordered into
    small dense matrix products per populated cell. No time binning or fit.
    ``out`` is updated in place. Caller may invoke this per element chunk.
    """
    omega=np.asarray(omega_per_ns,float)
    times=np.asarray(times_ns,float)
    phase=np.exp(1j*(times[:,None]-reference_ns)*omega[None,:])
    for cell in range(spread.shape[0]):
        lo,hi=spread.indptr[cell:cell+2]
        for begin in range(lo,hi,link_chunk):
            end=min(begin+link_chunk,hi)
            node=spread.indices[begin:end]
            weight=spread.data[begin:end,None]
            h=angular[node_elements[node]]
            out[cell].real += h.T @ (weight*phase[node].real)
            out[cell].imag += h.T @ (weight*phase[node].imag)


def compile_axial_source_fast(elements, degree, omega_per_ns, *, azimuthal_degree=0,
                              cell_m=None,bins=512,transverse_m=None,
                              deposit='linear',element_order=2,frame=None,
                              chunk=16384,link_chunk=8192,axial_only=False):
    """Compile the SAME lattice, chord rule, m band and absolute emission phases.

    This opt-in alternative changes only evaluation order. It avoids the
    element*channel*frequency payload; a small GEMM is made per touched cell.
    ``chunk`` and ``link_chunk`` limit work arrays, NOT the final dense source.
    The output still has the original possibly large (cell,channel,frequency)
    shape. Arrays produced from actual G4 must be compared with AxialSource.of.

    ``axial_only=True`` keeps the requested longitudinal ``cell_m`` but omits
    transverse CIC cells.  It is exact for a source known to lie on the chosen
    axis, in particular a straight Cherenkov track.
    """
    from itertools import product
    from scipy.sparse import coo_matrix
    from .axial_source import AxisFrame,AxialSource
    if deposit not in ('linear','nearest'): raise ValueError('Invalid deposit')
    if not isinstance(degree,(int,np.integer)) or degree<0: raise ValueError('Invalid degree')
    if not isinstance(azimuthal_degree,(int,np.integer)) or azimuthal_degree<0:
        raise ValueError('Invalid azimuthal degree')
    if not isinstance(element_order,(int,np.integer)) or element_order<1:
        raise ValueError('Invalid chord quadrature order')
    if not isinstance(chunk,(int,np.integer)) or chunk<1 or link_chunk<1:
        raise ValueError('Chunk sizes must be positive')
    if not isinstance(axial_only,(bool,np.bool_)):
        raise ValueError('axial_only must be boolean')
    frame=frame or AxisFrame.of(elements)
    omega=np.atleast_1d(np.asarray(omega_per_ns,float))
    if omega.ndim!=1 or not len(omega) or not np.isfinite(omega).all():
        raise ValueError('Invalid frequency array')
    photons=np.asarray(elements.photons,float)
    if not len(photons) or not np.isfinite(photons).all() or np.any(photons<0) or photons.sum()<=0:
        raise ValueError('Invalid photon weights')
    max_m=min(degree,int(azimuthal_degree))
    starts,aa,bb,kept,channel_degree=_layout(degree,max_m)
    nodes,node_weights=np.polynomial.legendre.leggauss(element_order)
    nodes,node_weights=(nodes+1)/2,node_weights/2
    z,transverse,_,_=frame.coordinates(elements)
    local_transverse=np.stack((transverse@frame.first,transverse@frame.second),axis=-1)
    along=(z.max()-z.min()+1e-9)/int(max(1,bins)) if cell_m is None else float(cell_m)
    flat=bool(axial_only) or (cell_m is None and transverse_m is None)
    across=np.inf if flat else (along if transverse_m is None else float(transverse_m))
    if not np.isfinite(along) or along<=0 or (not flat and (not np.isfinite(across) or across<=0)):
        raise ValueError('Grid spacing must be finite and positive')
    spacing=np.array([along,1.0 if flat else across,1.0 if flat else across])
    origin=np.array([z.min()-along,0.0 if flat else local_transverse[:,0].min()-across,
                     0.0 if flat else local_transverse[:,1].min()-across])
    axes=(0,) if flat else (0,1,2)
    steps=list(product(*[(0,1) if a in axes else (0,) for a in range(3)]))
    geometry=[]
    for fraction,node_weight in zip(nodes,node_weights):
        zf,transverse_f,_,time_f=frame.coordinates(elements,fraction)
        position=np.stack((zf,transverse_f@frame.first if not flat else np.zeros(len(zf)),
                            transverse_f@frame.second if not flat else np.zeros(len(zf))),axis=-1)
        lattice=(position-origin)/spacing
        if deposit=='nearest':
            corners=[(np.rint(lattice).astype(np.int64),np.full(len(zf),float(node_weight)))]
        else:
            base=np.floor(lattice).astype(np.int64); remainder=lattice-base
            corners=[]
            for step in steps:
                shift=np.array(step,dtype=np.int64)
                weight=np.full(len(zf),float(node_weight))
                for a in axes:
                    weight=weight*(remainder[:,a] if shift[a] else 1-remainder[:,a])
                corners.append((base+shift[None,:],weight))
        geometry.append((corners,time_f))
    keys=np.concatenate([key for corners,_ in geometry for key,_ in corners],axis=0)
    unique,flat_inverse=np.unique(keys,axis=0,return_inverse=True)
    del keys
    flat_inverse=flat_inverse.reshape(-1,len(elements))
    centres=origin[None,:]+unique*spacing[None,:]
    table=np.zeros((len(unique),len(kept),len(omega)),complex)
    reference=float(np.asarray(elements.start_ns).min())
    # Compute cone harmonics once per original element chunk, shared by all
    # chord nodes. Evaluate P_l only once per l, never once per m.
    for begin in range(0,len(elements),int(chunk)):
        piece=slice(begin,begin+int(chunk)); size=len(photons[piece])
        angular=_element_angular(np.ascontiguousarray(frame.rotate(elements.direction[piece])),
                                 np.asarray(elements.cone_cosine[piece]),starts,aa,bb,max_m)
        rows=[];cols=[];data=[];times=[];row=0
        for n,(corners,time_f) in enumerate(geometry):
            targets=flat_inverse[row:row+len(corners)];row+=len(corners)
            rows.extend([target[piece] for target in targets])
            cols.extend([n*size+np.arange(size) for _ in corners])
            data.extend([photons[piece]*weight[piece] for _,weight in corners])
            times.append(time_f[piece])
        spread=coo_matrix((np.concatenate(data),(np.concatenate(rows),np.concatenate(cols))),
                          shape=(len(unique),size*len(nodes))).tocsr()
        # The output is updated one cell at a time; there is never a second
        # dense source-shaped result from a sparse @ dense operation.
        accumulate_cell_gemm(spread,angular,np.tile(np.arange(size),len(nodes)),
                             np.concatenate(times),omega,reference,table,
                             link_chunk=int(link_chunk))
    summary=dict(elements=len(elements),photons=float(photons.sum()),degree=int(degree),
                 cells=len(unique),cell_m=along,transverse_m=None if flat else across,
                 azimuthal_degree=int(azimuthal_degree),channels=len(kept),
                 channels_if_full=(degree+1)**2,deposit=deposit,element_order=int(element_order),
                 axial_only=bool(axial_only),coefficients=int(table.size),
                 compiler='cell_gemm_no_new_approximation')
    return AxialSource(frame,centres[:,0],np.zeros((len(unique),2)) if flat else centres[:,1:],
                       table,kept,channel_degree,int(degree),int(azimuthal_degree),reference,summary)


def rigidly_moved_source(source, rotation=None, translation=None, delay_ns=0.0):
    """Rigidly move a compiled grid, without reading/reprojecting any G4 step.

    World transform: x' = rotation @ x + translation. Unlike reconstructing
    an AxisFrame from the moved event, this rotates the SAME lattice and its
    azimuth reference and therefore adds no grid/orientation approximation.
    """
    from dataclasses import replace
    rotation=np.eye(3) if rotation is None else np.asarray(rotation,float)
    translation=np.zeros(3) if translation is None else np.asarray(translation,float)
    if (rotation.shape!=(3,3) or not np.isfinite(rotation).all()
            or not np.allclose(rotation.T@rotation,np.eye(3),rtol=0,atol=1e-12)
            or not np.isclose(np.linalg.det(rotation),1,rtol=0,atol=1e-12)):
        raise ValueError('rotation must be proper orthogonal 3x3')
    if translation.shape!=(3,) or not np.isfinite(translation).all() or not np.isfinite(delay_ns):
        raise ValueError('Invalid translation or time delay')
    frame=replace(source.frame,centre_m=rotation@source.frame.centre_m+translation,
                  axis=rotation@source.frame.axis,first=rotation@source.frame.first,
                  second=rotation@source.frame.second)
    return replace(source,frame=frame,reference_ns=source.reference_ns+delay_ns,
                   summary={**source.summary,'pose':'rigidly_moved_compiled_source'})
