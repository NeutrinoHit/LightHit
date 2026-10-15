"""Finite-track first scattering from positive physical-time CDFs.

A row fixes transverse distance and OM orientation. The ten-component atlas
removes orientation dependence for cubic acceptance and interpolates radius.
TrackFirstAtlasStore loads only charge prefixes eagerly and two CDF rows lazily.
Optics, spectral weights, beta, clock speed and area are fixed cache properties;
source position, time and length are query inputs within declared coverage.

Canonical geometry: receiver=(rho,0,0), emitter=(0,0,-u), emission time=-u/v.
Each source cell is integrated by the prompt segment quadrature. No frequency
inversion is used. Unsupported late times and source ranges are rejected; a
separate analytic q1 bound controls omitted tails.
"""
from dataclasses import dataclass
import numpy as np


def _checked_source_bounds(bounds, edges):
    """Accept only coordinate-scaled roundoff from (edge*rho)/rho."""
    low,high=float(edges[0]),float(edges[-1])
    tolerance=64*np.finfo(float).eps*np.maximum(1.,np.abs([low,high]))
    if (not np.isfinite(bounds).all() or np.any(bounds[...,0]>bounds[...,1])
            or np.any(bounds[...,0]<low-tolerance[0])
            or np.any(bounds[...,1]>high+tolerance[1])):
        raise ValueError('source interval outside atlas')
    return np.clip(bounds,low,high)


@dataclass(frozen=True)
class TrackFirstPrimitive:
    """Q1 of a straight track for one row: charge and time-bin CDFs in source position."""
    u_edges_m: np.ndarray
    tau_edges_ns: np.ndarray
    charge_cells: np.ndarray
    bins_cells: np.ndarray
    speed_m_per_ns: float
    metadata: dict

    def __post_init__(self):
        """Check axes and that charge and bin prefixes are consistent."""
        u = np.asarray(self.u_edges_m, float)
        t = np.asarray(self.tau_edges_ns, float)
        q = np.asarray(self.charge_cells, float)
        b = np.asarray(self.bins_cells, float)
        if (u.ndim != 1 or t.ndim != 1 or len(u) < 2 or len(t) < 2
                or not np.isfinite(u).all() or not np.isfinite(t).all()
                or np.any(np.diff(u) <= 0) or np.any(np.diff(t) <= 0)):
            raise ValueError('strictly increasing finite source/time edges required')
        if (q.shape != (len(u)-1,) or b.shape != (len(u)-1,len(t)-1)
                or not np.isfinite(q).all() or not np.isfinite(b).all()
                or np.any(q < 0) or np.any(b < 0)):
            raise ValueError('nonnegative finite cell charge and time bins required')
        if not np.isfinite(self.speed_m_per_ns) or self.speed_m_per_ns <= 0:
            raise ValueError('positive finite emission speed required')
        if np.any(b.sum(axis=1) > q * (1+1e-8) + 1e-25):
            raise ValueError('time window charge exceeds all-time charge')
        # Left and right prefixes avoid subtraction of almost equal saturated
        # prefixes when querying the remote ends. Both include all stored mass.
        cdf = np.pad(np.cumsum(b, axis=1), ((0,0),(1,0)))
        left = np.pad(np.cumsum(cdf, axis=0), ((1,0),(0,0)))
        right = np.pad(np.cumsum(cdf[::-1], axis=0)[::-1], ((0,1),(0,0)))
        qleft = np.r_[0.,np.cumsum(q)]
        qright = np.r_[np.cumsum(q[::-1])[::-1],0.]
        for key,value in dict(u_edges_m=u.copy(),tau_edges_ns=t.copy(),
                charge_cells=q.copy(),bins_cells=b.copy(),_cdf_left=left,
                _cdf_right=right,_q_left=qleft,_q_right=qright).items():
            value.setflags(write=False)
            object.__setattr__(self,key,value)

    def _spatial_interval(self, low, high, left, right):
        """Difference of the left and right prefixes over [low, high]."""
        def at(x, data):
            """Linear interpolation of a prefix table at ``x``."""
            i = np.searchsorted(self.u_edges_m,x,side='right')-1
            i = np.clip(i,0,len(self.u_edges_m)-2)
            w = (x-self.u_edges_m[i])/(self.u_edges_m[i+1]-self.u_edges_m[i])
            return (1-w)*data[i]+w*data[i+1]
        # Select the smaller pair of prefix magnitudes. It reduces cancellation
        # but does not certify relative accuracy for extremely weak intervals.
        l0,l1 = at(low,left),at(high,left)
        r0,r1 = at(low,right),at(high,right)
        return np.where(abs(l0)+abs(l1) <= abs(r0)+abs(r1),l1-l0,r0-r1)

    def query(self, u_low_m, u_high_m, tau_edges_ns, *, coefficient0_per_m=1.):
        """Return all-time charge and bins for an interval and canonical clock.

        Query edges must remain inside the stored time window. This rejects
        unsupported requests instead of inventing zero late light. Source
        intervals outside the row likewise need an explicit caller fallback.
        """
        low,high = float(u_low_m),float(u_high_m)
        t = np.asarray(tau_edges_ns,float)
        density = float(coefficient0_per_m)
        if (not np.isfinite([low,high,density]).all() or density < 0 or high < low
                or low < self.u_edges_m[0] or high > self.u_edges_m[-1]):
            raise ValueError('source interval outside row or invalid density')
        if (t.ndim != 1 or len(t) < 2 or not np.isfinite(t).all()
                or np.any(np.diff(t) <= 0) or t[0] < self.tau_edges_ns[0]
                or t[-1] > self.tau_edges_ns[-1]):
            raise ValueError('time edges outside row or not strictly increasing')
        q = self._spatial_interval(low,high,self._q_left,self._q_right)
        cdf = self._spatial_interval(low,high,self._cdf_left,self._cdf_right)
        b = np.diff(np.interp(t,self.tau_edges_ns,cdf))
        # Only roundoff is clipped: cell positivity was checked at construction.
        return float(max(q,0.)*density), np.maximum(b,0.)*density

    def segment(self, z_receiver_from_start_m, length_m, start_ns, absolute_edges_ns,
                *, coefficient0_per_m=1.):
        """Query any translated finite segment with this row's geometry/speed."""
        z,length,t0 = map(float,(z_receiver_from_start_m,length_m,start_ns))
        if not np.isfinite([z,length,t0]).all() or length < 0:
            raise ValueError('finite segment geometry and nonnegative length required')
        tau = np.asarray(absolute_edges_ns,float)-t0-z/self.speed_m_per_ns
        return self.query(z-length,z,tau,coefficient0_per_m=coefficient0_per_m)


def build_track_first_primitive(kernel, *, rho_m, orientation, beta, speed_m_per_ns,
                                u_edges_m, tau_edges_ns, spectral_order=2,
                                prompt_config=None):
    """Build a single orientation-specific row; see build_track_first_rows."""
    return build_track_first_rows(kernel,rho_m=rho_m,orientations=np.asarray(orientation)[None,:],
        beta=beta,speed_m_per_ns=speed_m_per_ns,u_edges_m=u_edges_m,tau_edges_ns=tau_edges_ns,
        spectral_order=spectral_order,prompt_config=prompt_config)[0]


def build_track_first_rows(kernel, *, rho_m, orientations, beta, speed_m_per_ns,
                                u_edges_m, tau_edges_ns, spectral_order=2,
                                prompt_config=None):
    """Build one explicitly orientation-specific row with the direct integrator.

    ``kernel`` supplies medium, optical acceptance and detector properties; its
    first receiver's area is used. Charge normalization is C0 per metre=1,
    C2=-C0/beta². Pass the actual source density when querying. The all-time charge
    is independent of the stored time window, so window loss remains measurable.
    """
    from copy import copy
    from dataclasses import replace
    from time import perf_counter, process_time
    from ..sources import SpectralLightElements
    from ..prompt import PromptConfig
    from .muon_integral import order1_integrals
    from .primary_polyline import primary_spectral_quadrature
    u,t = np.asarray(u_edges_m,float),np.asarray(tau_edges_ns,float)
    rho,beta,v = map(float,(rho_m,beta,speed_m_per_ns))
    directions = np.asarray(orientations,float)
    if (not np.isfinite([rho,beta,v]).all() or rho <= 0 or not 0 < beta <= 1 or v <= 0
            or directions.ndim != 2 or directions.shape[1] != 3 or not len(directions)
            or not np.isfinite(directions).all()
            or not np.allclose(np.linalg.norm(directions,axis=1),1.)):
        raise ValueError('valid rho, beta, speed and unit OM orientation required')
    # Constructor validates array shapes before launching expensive evaluation.
    TrackFirstPrimitive(u,t,np.zeros(len(u)-1),np.zeros((len(u)-1,len(t)-1)),v,{})
    if t[0] > 0:
        raise ValueError('builder needs tau start <=0 to anchor its arrival CDF')
    # Guarantee no arrival precedes zero in the canonical clock. Slower moving
    # sources require a different CDF anchor and are intentionally unsupported.
    fastest_photon = .299792458 / kernel.medium.group_index.min()
    if v <= fastest_photon:
        raise ValueError('emission speed must exceed every photon group velocity')
    count = len(directions)
    local = copy(kernel)
    d = kernel.detector
    local.detector = replace(d,positions_m=np.tile([rho,0.,0.],(count,1)),
        orientations=directions,effective_area_m2=np.repeat(d.effective_area_m2[:1],count),
        identifiers={k:np.repeat(val[:1],count,axis=0) for k,val in d.identifiers.items()})
    local.config = replace(kernel.config,relative_time_edges_ns=t)
    local.wavelength = primary_spectral_quadrature(kernel.medium,d,order=spectral_order,
        wavelength_range_nm=kernel.config.wavelength_range_nm)
    config = prompt_config or PromptConfig(element_method='segment',screen=False)
    q,b = [],[]
    started,cpu = perf_counter(),process_time()
    for low,high in zip(u[:-1],u[1:]):
        length = high-low
        source = SpectralLightElements(np.array([[0.,0.,-high]]),np.array([[0.,0.,1.]]),
            np.array([length]),np.array([length]),np.array([-length/beta**2]),
            np.array([beta]),np.array([-high/v]),np.array([-low/v]),
            np.array([0]),np.array([0]),cone_model='spectral')
        charge,bins,_ = order1_integrals(local,source,time_origin_ns=np.zeros(count),config=config)
        q.append(charge);b.append(bins)
    metadata = dict(rho_m=rho,beta=beta,batch_orientations=count,
        spectral_nodes=len(local.wavelength.wavelength_nm),spectral_order=spectral_order,
        build_wall_s=perf_counter()-started,build_cpu_s=process_time()-cpu,
        orientation_specific=True,rho_interpolation=False,truncated_source=False,
        source_coverage_m=[float(u[0]),float(u[-1])],time_coverage_ns=[float(t[0]),float(t[-1])],
        source_interpolation='piecewise constant; cumulative linear',
        time_interpolation='piecewise linear cumulative',certified_accuracy=False)
    q,b=np.asarray(q),np.asarray(b)
    tolerance=512*np.finfo(float).eps*np.maximum(q[:,:,None],np.finfo(float).tiny)
    if np.any(b < -tolerance):
        bad=b < -tolerance
        raise ValueError('direct first-order quadrature has negative bins beyond roundoff: '
            f'min={b.min():.6g}, total negative={-np.minimum(b,0).sum():.6g}, '
            f'worst relative={np.min(b/np.maximum(q[:,:,None],np.finfo(float).tiny)):.6g}')
    metadata['negative_roundoff_mass_removed']=float(-np.minimum(b,0).sum())
    metadata['negative_roundoff_bins']=int(np.count_nonzero(b<0))
    b=np.maximum(b,0.)
    return [TrackFirstPrimitive(u,t,q[:,i],b[:,i,:],v,
        {**metadata,'orientation':directions[i].tolist()}) for i in range(count)]


def orientation_nodes():
    """Ten well-conditioned unit orientations for exact cubic acceptance.

    Reflection in the track/receiver plane eliminates odd transverse terms.
    Using n_y²=1-n_x²-n_z² leaves ten monomials n_x^i n_z^j, i+j<=3.
    """
    from scipy.linalg import qr
    powers=np.array([(i,degree-i) for degree in range(4) for i in range(degree+1)])
    angles=np.arange(24)*2*np.pi/24
    x=np.r_[0.,np.cos(angles),.6*np.cos(angles),.3*np.cos(angles)]
    z=np.r_[0.,np.sin(angles),.6*np.sin(angles),.3*np.sin(angles)]
    candidates=np.column_stack((x,np.sqrt(np.maximum(0.,1-x*x-z*z)),z))
    design=x[:,None]**powers[:,0]*z[:,None]**powers[:,1]
    _,_,pivot=qr(design.T,pivoting=True,mode='economic')
    nodes=candidates[pivot[:10]]
    matrix=nodes[:,0,None]**powers[:,0]*nodes[:,2,None]**powers[:,1]
    return nodes,powers,matrix


@dataclass(frozen=True)
class TrackFirstAtlas:
    """Experimental cubic-orientation/radius finite-track q1 atlas.

    Coordinates are x=u/rho and y=tau/rho. Ten angular coefficients are exact
    for a degree-three acceptance; radius and source/time interpolation remain
    approximations. beta, emission clock speed, area and all optical properties
    are fixed and recorded in metadata. Query failures require caller fallback.
    """
    radii_m: np.ndarray
    x_edges: np.ndarray
    time_over_rho_edges: np.ndarray
    powers: np.ndarray
    charge_left: np.ndarray
    charge_right: np.ndarray
    cdf_left: np.ndarray
    cdf_right: np.ndarray
    metadata: dict

    def __post_init__(self):
        """Check axes and table shapes."""
        r,x,t=map(np.asarray,(self.radii_m,self.x_edges,self.time_over_rho_edges))
        for axis in (r,x,t):
            if axis.ndim!=1 or len(axis)<2 or not np.isfinite(axis).all() or np.any(np.diff(axis)<=0):
                raise ValueError('atlas axes must be increasing finite vectors')
        if r[0]<=0: raise ValueError('radii must be positive')
        nq=(len(r),len(x),10);nc=(len(r),len(x),len(t),10)
        for key,shape in [('charge_left',nq),('charge_right',nq),('cdf_left',nc),('cdf_right',nc)]:
            values=np.asarray(getattr(self,key))
            if values.shape!=shape or not np.isfinite(values).all():
                raise ValueError('invalid angular coefficient array '+key)
        _,expected,_=orientation_nodes()
        if not np.array_equal(self.powers,expected):raise ValueError('unsupported orientation basis')

    def query(self,rho_m,orientation,u_low_m,u_high_m,tau_edges_ns,*,coefficient0_per_m=1.):
        """Q1 time bins of a straight piece at transverse distance ``rho_m``."""
        rho=float(rho_m);look=np.asarray(orientation,float);density=float(coefficient0_per_m)
        if (not np.isfinite([rho,density]).all() or density<0 or rho<self.radii_m[0] or rho>self.radii_m[-1]
                or look.shape!=(3,) or not np.isfinite(look).all() or not np.isclose(np.linalg.norm(look),1.)):
            raise ValueError('invalid rho/orientation/density or radius outside atlas')
        bounds=np.array([u_low_m,u_high_m],float)/rho
        time=np.asarray(tau_edges_ns,float)/rho
        bounds=_checked_source_bounds(bounds,self.x_edges)
        if (time.ndim!=1 or len(time)<2 or not np.isfinite(time).all() or np.any(np.diff(time)<=0)
                or (time[0]<self.time_over_rho_edges[0] and not (self.metadata.get('causal_zero_before_tau0',False) and self.time_over_rho_edges[0]<=0))
                or time[-1]>self.time_over_rho_edges[-1]):
            raise ValueError('time edges outside atlas')
        features=look[0]**self.powers[:,0]*look[2]**self.powers[:,1]
        indices=np.clip(np.searchsorted(self.x_edges,bounds,side='right')-1,0,len(self.x_edges)-2)
        fractions=(bounds-self.x_edges[indices])/(self.x_edges[indices+1]-self.x_edges[indices])
        ir=int(np.clip(np.searchsorted(self.radii_m,rho,side='right')-1,0,len(self.radii_m)-2))
        weight=(rho-self.radii_m[ir])/(self.radii_m[ir+1]-self.radii_m[ir])

        def evaluate(j):
            """Bins at one radius node."""
            def interval(left,right):
                """Difference of interpolated prefixes over the piece."""
                def endpoint(a,k):
                    """Prefix at one end, contracted with the acceptance features."""
                    i=indices[k];f=fractions[k]
                    return ((1-f)*a[j][i]+f*a[j][i+1])@features
                l0,l1=endpoint(left,0),endpoint(left,1)
                r0,r1=endpoint(right,0),endpoint(right,1)
                return np.where(abs(l0)+abs(l1)<=abs(r0)+abs(r1),l1-l0,r0-r1)
            q=float(interval(self.charge_left,self.charge_right))
            cdf=interval(self.cdf_left,self.cdf_right)
            bins=np.diff(np.interp(time,self.time_over_rho_edges,cdf))
            total=float(self.charge_left[j][-1]@features)
            tolerance=256*np.finfo(float).eps*max(total,1e-300)
            if total<=0 or q < -tolerance or np.min(bins)<-tolerance:
                raise ValueError('angular reconstruction produced a negative physical response; refine/fallback')
            # Only bounded floating-point roundoff can reach these clips.
            return max(q,0.),np.maximum(bins,0.),total
        q0,b0,a0=evaluate(ir);q1,b1,a1=evaluate(ir+1)
        amplitude=np.exp((1-weight)*np.log(a0)+weight*np.log(a1))
        q=amplitude*((1-weight)*q0/a0+weight*q1/a1)
        bins=amplitude*((1-weight)*b0/a0+weight*b1/a1)
        return float(q*density),bins*density

    def query_charge(self,rho_m,orientation,u_low_m,u_high_m,*,coefficient0_per_m=1.):
        """All-time charge only; no time CDF access (supports lazy stores)."""
        rho=float(rho_m);look=np.asarray(orientation,float);density=float(coefficient0_per_m)
        if (not np.isfinite([rho,density]).all() or density<0 or rho<self.radii_m[0] or rho>self.radii_m[-1]
                or look.shape!=(3,) or not np.isfinite(look).all() or not np.isclose(np.linalg.norm(look),1.)):
            raise ValueError('invalid rho/orientation/density or radius outside atlas')
        bounds=np.array([u_low_m,u_high_m],float)/rho
        bounds=_checked_source_bounds(bounds,self.x_edges)
        features=look[0]**self.powers[:,0]*look[2]**self.powers[:,1]
        indices=np.clip(np.searchsorted(self.x_edges,bounds,side='right')-1,0,len(self.x_edges)-2)
        fractions=(bounds-self.x_edges[indices])/(self.x_edges[indices+1]-self.x_edges[indices])
        ir=int(np.clip(np.searchsorted(self.radii_m,rho,side='right')-1,0,len(self.radii_m)-2))
        w=(rho-self.radii_m[ir])/(self.radii_m[ir+1]-self.radii_m[ir])
        def evaluate(j):
            """Charge at one radius node."""
            def endpoint(a,k):
                """Charge prefix at one end of the piece."""
                i=indices[k];f=fractions[k]
                return ((1-f)*a[j][i]+f*a[j][i+1])@features
            l0,l1=endpoint(self.charge_left,0),endpoint(self.charge_left,1)
            r0,r1=endpoint(self.charge_right,0),endpoint(self.charge_right,1)
            q=l1-l0 if abs(l0)+abs(l1)<=abs(r0)+abs(r1) else r0-r1
            total=float(self.charge_left[j][-1]@features)
            if total<=0 or q < -256*np.finfo(float).eps*max(total,1e-300):
                raise ValueError('angular reconstruction produced a negative physical charge')
            return max(q,0.),total
        q0,a0=evaluate(ir);q1,a1=evaluate(ir+1)
        amp=np.exp((1-w)*np.log(a0)+w*np.log(a1))
        return float(density*amp*((1-w)*q0/a0+w*q1/a1))

    def query_charge_many(self,rho_m,orientation,u_low_m,u_high_m,*,coefficient0_per_m=1.):
        """Vectorized all-time charge for N pairs; no Python loop over pairs/CDF IO."""
        rho=np.asarray(rho_m,float);look=np.asarray(orientation,float)
        if rho.ndim!=1 or look.shape!=(len(rho),3):raise ValueError('rho[N], orientations[N,3] required')
        n=len(rho)
        low=np.broadcast_to(np.asarray(u_low_m,float),(n,));high=np.broadcast_to(np.asarray(u_high_m,float),(n,))
        density=np.broadcast_to(np.asarray(coefficient0_per_m,float),(n,))
        if (not np.isfinite(rho).all() or not np.isfinite(look).all() or not np.isfinite(density).all()
                or np.any(density<0) or np.any(rho<self.radii_m[0]) or np.any(rho>self.radii_m[-1])
                or not np.allclose(np.linalg.norm(look,axis=1),1.)):
            raise ValueError('invalid rho/orientation/density or radius outside atlas')
        bounds=np.column_stack((low,high))/rho[:,None]
        bounds=_checked_source_bounds(bounds,self.x_edges)
        features=look[:,0,None]**self.powers[:,0]*look[:,2,None]**self.powers[:,1]
        index=np.clip(np.searchsorted(self.x_edges,bounds,side='right')-1,0,len(self.x_edges)-2)
        fraction=(bounds-self.x_edges[index])/(self.x_edges[index+1]-self.x_edges[index])
        ir=np.clip(np.searchsorted(self.radii_m,rho,side='right')-1,0,len(self.radii_m)-2)
        w=(rho-self.radii_m[ir])/(self.radii_m[ir+1]-self.radii_m[ir])
        def evaluate(j):
            """Charges of all pieces at one radius node."""
            def endpoint(a,k):
                """Charge prefixes at one end of every piece."""
                i=index[:,k];f=fraction[:,k,None]
                values=(1-f)*a[j,i]+f*a[j,i+1]
                return np.einsum('ij,ij->i',values,features)
            l0,l1=endpoint(self.charge_left,0),endpoint(self.charge_left,1)
            r0,r1=endpoint(self.charge_right,0),endpoint(self.charge_right,1)
            q=np.where(abs(l0)+abs(l1)<=abs(r0)+abs(r1),l1-l0,r0-r1)
            total=np.einsum('ij,ij->i',self.charge_left[j,-1],features)
            if np.any(total<=0) or np.any(q < -256*np.finfo(float).eps*np.maximum(total,1e-300)):
                raise ValueError('angular reconstruction produced a negative physical charge')
            return np.maximum(q,0.),total
        q0,a0=evaluate(ir);q1,a1=evaluate(ir+1)
        amp=np.exp((1-w)*np.log(a0)+w*np.log(a1))
        return density*amp*((1-w)*q0/a0+w*q1/a1)

    def segment_charge(self,rho_m,orientation,z_receiver_from_start_m,length_m,*,coefficient0_per_m=1.):
        """Charge of a piece given by the OM position along it and its length."""
        z,length=map(float,(z_receiver_from_start_m,length_m))
        if not np.isfinite([z,length]).all() or length<0:raise ValueError('invalid finite segment')
        return self.query_charge(rho_m,orientation,z-length,z,coefficient0_per_m=coefficient0_per_m)

    def segment(self,rho_m,orientation,z_receiver_from_start_m,length_m,start_ns,absolute_edges_ns,
                *,coefficient0_per_m=1.):
        """Finite segment with the fixed emission clock speed recorded at build."""
        z,length,start=map(float,(z_receiver_from_start_m,length_m,start_ns))
        speed=float(self.metadata['speed_m_per_ns'])
        if not np.isfinite([z,length,start,speed]).all() or length<0 or speed<=0:
            raise ValueError('finite geometry/clock and nonnegative length required')
        tau=np.asarray(absolute_edges_ns,float)-start-z/speed
        return self.query(rho_m,orientation,z-length,z,tau,coefficient0_per_m=coefficient0_per_m)

    def save(self,path):
        """Write all tables and metadata to one npz file."""
        import json
        np.savez_compressed(path,**{k:v for k,v in vars(self).items() if k!='metadata'},metadata=json.dumps(self.metadata))

    @classmethod
    def load(cls,path):
        """Read an atlas written by :meth:`save`."""
        import json
        with np.load(path,allow_pickle=False) as data:
            values={k:data[k] for k in data.files if k!='metadata'}
            return cls(**values,metadata=json.loads(str(data['metadata'])))


def build_track_first_atlas(kernel,*,radii_m,x_edges,time_over_rho_edges,beta,speed_m_per_ns,
                            spectral_order=2,prompt_config=None,progress=None):
    """Precompute ten acceptance coefficients per radius with batched receivers.

    The receiver's degree-three polynomial angular acceptance is required.
    Changing source geometry/OM orientation needs no rebuild; changing optics,
    area, beta or clock speed does. Radius outside the range requires fallback.
    """
    from time import perf_counter,process_time
    alpha=np.asarray(kernel.acceptance()['alpha'])
    if len(alpha)>4:raise ValueError('ten-component atlas requires acceptance degree <=3')
    radii=np.asarray(radii_m,float);x=np.asarray(x_edges,float);t=np.asarray(time_over_rho_edges,float)
    if radii.ndim!=1 or len(radii)<2 or np.any(np.diff(radii)<=0) or radii[0]<=0:
        raise ValueError('increasing radius axis required')
    nodes,powers,matrix=orientation_nodes();inverse=np.linalg.inv(matrix)
    ql,qr,cl,cr=[],[],[],[];per_radius=[]
    began,cpu=perf_counter(),process_time()
    for radius in radii:
        rows=build_track_first_rows(kernel,rho_m=radius,orientations=nodes,beta=beta,
            speed_m_per_ns=speed_m_per_ns,u_edges_m=radius*x,tau_edges_ns=radius*t,
            spectral_order=spectral_order,prompt_config=prompt_config)
        # Transform positive orientation samples into signed monomial coefficients.
        for destination,attribute in [(ql,'_q_left'),(qr,'_q_right'),(cl,'_cdf_left'),(cr,'_cdf_right')]:
            values=np.stack([getattr(row,attribute) for row in rows])
            destination.append(np.moveaxis(np.tensordot(inverse,values,axes=(1,0)),0,-1))
        info=dict(rho_m=float(radius),build_cpu_s=rows[0].metadata['build_cpu_s'],
                  build_wall_s=rows[0].metadata['build_wall_s'])
        per_radius.append(info)
        if progress is not None:progress(info)
    metadata=dict(format_version=1,beta=float(beta),speed_m_per_ns=float(speed_m_per_ns),
        acceptance_alpha=alpha.tolist(),effective_area_m2=float(kernel.detector.effective_area_m2[0]),
        spectral_order=spectral_order,spectral_nodes=rows[0].metadata['spectral_nodes'],
        orientation_nodes=nodes.tolist(),orientation_condition_number=float(np.linalg.cond(matrix)),
        radius_interpolation='positive normalized CDF mix with log global amplitude',
        source_and_time_variables=['u/rho','tau/rho'],build_wall_s=perf_counter()-began,
        build_cpu_s=process_time()-cpu,per_radius=per_radius,
        optical_tables=dict(wavelength_nm=kernel.medium.wavelength_nm.tolist(),
            absorption_per_m=kernel.medium.absorption_per_m.tolist(),
            scattering_per_m=kernel.medium.scattering_per_m.tolist(),
            phase_index=kernel.medium.phase_index.tolist(),group_index=kernel.medium.group_index.tolist(),
            g=float(kernel.medium.g)),certified_accuracy=False)
    return TrackFirstAtlas(radii,x,t,powers,np.array(ql),np.array(qr),np.array(cl),np.array(cr),metadata)


def first_scattering_segment_bounds(kernel,elements,receiver_m,*,effective_area_m2=None):
    """Conservative all-time q1 bounds for cone-separated finite segments.

    Bounds apply to the supplied positive spectral quadrature, not its continuum
    integration error. For each source segment, its full photon yield is bounded
    using minimal receiver distance and angular separation from the Cherenkov
    cone. Segments whose angular interval crosses a cone return infinity.

    For a ray separated by delta>0, final-flight distance d>=d_min and scattering
    cosine (r cos(delta)-ell)/d decreases with ell. HG with g>=0 is bounded by
    p_HG(cos(delta_min)). Convexity of ell+d gives

      ell+d >= r_min + (1-cos(delta_min))*ell,
      integral exp[-mu_t(ell+d)]/d² dell
        <= exp[-mu_t*r_min]/[mu_t*(1-cos(delta_min))*d_min²].

    No source tail is presumed zero. A caller may omit pieces only against the
    SUM of these charge bounds; any time-window/bin omission is bounded by the
    same positive all-time mass.
    """
    from ..prompt import _spectral_tables
    from .track_first_bounds import hg_angle_integral_over_sine_upper
    if elements.cone_model!='spectral':raise ValueError('bounds require a spectral cone')
    g=float(kernel.medium.g)
    if not 0<=g<1:raise ValueError('phase bound currently requires HG 0<=g<1')
    receiver=np.asarray(receiver_m,float)
    if receiver.shape!=(3,) or not np.isfinite(receiver).all():raise ValueError('finite receiver required')
    tables=_spectral_tables(kernel)
    alpha=np.asarray(kernel.acceptance()['alpha'])
    amax=float(np.sum(abs(alpha)*(2*np.arange(len(alpha))+1))/(4*np.pi))
    area=float(kernel.detector.effective_area_m2[0] if effective_area_m2 is None else effective_area_m2)
    if not np.isfinite(area) or area<0:raise ValueError('nonnegative finite detector area required')
    delta=receiver-elements.start_m
    high=np.sum(delta*elements.direction,axis=1);low=high-elements.length_m
    rho=np.linalg.norm(delta-high[:,None]*elements.direction,axis=1)
    nearest=np.where(low>0,low,np.where(high<0,-high,0.))
    rmin=np.hypot(rho,nearest)
    theta_low=np.arctan2(rho,high);theta_high=np.arctan2(rho,low)
    phase=tables['phase_index'][None,:]
    emitting=elements.beta[:,None]*phase>1
    cone=np.arccos(np.clip(1/(elements.beta[:,None]*phase),-1.,1.))
    separation=np.maximum(0.,np.maximum(theta_low[:,None]-cone,cone-theta_high[:,None]))
    distance=rmin[:,None]*np.sin(np.minimum(separation,np.pi/2))
    known=(separation>1e-12)&(rmin[:,None]>0)&emitting
    # Infinity means unsupported, not a numerical estimate of a dark piece.
    response=np.full_like(distance,np.inf)
    response[~emitting]=0.
    safe_distance=np.where(known,distance,1.)
    safe_radius=np.maximum(rmin[:,None],np.finfo(float).tiny)
    extinction=tables['mut'][None,:]
    kappa=np.where(known,2*np.sin(separation/2)**2,1.)
    phase_max=(1-g*g)/(4*np.pi*(1+g*g-2*g*np.cos(separation))**1.5)
    integral=np.exp(-extinction*safe_radius)/(extinction*kappa*safe_distance**2)
    bounded=tables['mus'][None,:]*phase_max*integral
    # Exact pencil geometry: S>=r and
    # q1<=mu_s*exp(-mu_t*r)/r * int_theta^pi p_HG(cos chi)dchi/sin(theta).
    # The last angular factor decreases with theta for HG g>=0. Substitute
    # the segment's minimum cone separation and an analytic integral bound,
    # rather than replacing p_HG everywhere by its endpoint maximum.
    angular=hg_angle_integral_over_sine_upper(np.where(known,separation,1.),g)
    angle_bound=tables['mus'][None,:]*np.exp(-extinction*safe_radius)/safe_radius*angular
    bounded=np.minimum(bounded,angle_bound)
    response[known]=bounded[known]
    photons=(elements.coefficient0[:,None]*tables['s0w'][None,:]
             +elements.coefficient2[:,None]*tables['s2w'][None,:])
    photons=np.maximum(photons,0.)
    weighted=np.zeros_like(response)
    np.multiply(photons,response,out=weighted,where=photons>0)
    bounds=area*amax*weighted.sum(axis=1)
    positive=bounds>0
    bounds[positive]=np.nextafter(bounds[positive]*(1+128*np.finfo(float).eps),np.inf)
    return bounds,dict(acceptance_upper=amax,spectral_nodes=len(tables['wavelength_nm']),
        angular_integral_bound='HG sine-half-angle concavity chord; monotone pencil response',
        bound_scope='all-time q1 for supplied positive spectral quadrature',
        continuum_spectral_error_included=False,phase_function='HG',
        unsupported_pieces=int(np.count_nonzero(~np.isfinite(bounds))))


class TrackFirstAtlasStore:
    """Read-only, checksum-verified row store with charge-only eager loading.

    All small charge prefixes are resident. Time CDFs load on demand through a
    bounded LRU; charge queries never open a time row. The frozen manifest ties
    every file to the same optical/grid contract. The atlas is built once per
    medium and detector response (``build_track_first_atlas``); the build
    driver is not part of the package.
    """
    def __init__(self,directory,*,max_resident_rows=2):
        """Open a row directory; charge prefixes are read at once, time rows on demand."""
        from pathlib import Path
        from collections import OrderedDict
        import json,hashlib
        if max_resident_rows<2:raise ValueError('at least two radial rows must fit')
        self.directory=Path(directory)
        self.manifest=json.loads((self.directory/'manifest.json').read_text())
        if self.manifest.get('status')!='complete':raise ValueError('atlas build incomplete')
        self.metadata=self.manifest['physics']
        self.max_resident_rows=int(max_resident_rows)
        self._rows=OrderedDict();self.time_row_loads=0
        self.radii_m=np.asarray(self.manifest['radii_m'],float)
        self.x_edges=np.asarray(self.manifest['x_edges'],float)
        self.time_over_rho_edges=np.asarray(self.manifest['time_over_rho_edges'],float)
        self.powers=np.asarray(self.manifest['powers'],int)
        path=self.directory/self.manifest['charge_file']['file']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=self.manifest['charge_file']['sha256']:
            raise ValueError('charge cache checksum mismatch')
        with np.load(path,allow_pickle=False) as data:
            self.charge_left=data['charge_left'];self.charge_right=data['charge_right']
        expected=(len(self.radii_m),len(self.x_edges),10)
        if self.charge_left.shape!=expected or self.charge_right.shape!=expected:
            raise ValueError('invalid charge cache dimensions')
        owner=self
        class Access:
            """Indexable view that loads time rows lazily."""
            def __init__(self,name):
                """Remember which table of a row to return."""
                self.name=name
            def __getitem__(self,j):
                """Table ``name`` of radial row ``j``."""
                return owner._load_row(int(j))[self.name]
        self.cdf_left=Access('cdf_left');self.cdf_right=Access('cdf_right')

    def _load_row(self,j):
        """Read and verify one radial row; keep a few rows resident."""
        import hashlib
        if j in self._rows:
            self._rows.move_to_end(j)
            return self._rows[j]
        record=self.manifest['rows'][j];path=self.directory/record['file']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=record['sha256']:
            raise ValueError('time row checksum mismatch')
        with np.load(path,allow_pickle=False) as data:
            row={key:data[key] for key in ('cdf_left','cdf_right')}
        expected=(len(self.x_edges),len(self.time_over_rho_edges),10)
        if any(a.shape!=expected or not np.isfinite(a).all() for a in row.values()):
            raise ValueError('invalid time row dimensions or nonfinite data')
        self._rows[j]=row;self.time_row_loads+=1
        while len(self._rows)>self.max_resident_rows:self._rows.popitem(last=False)
        return row

    def validate_kernel(self,kernel):
        """Refuse a kernel with other optics, QE, acceptance or OM area."""
        _,digest=track_first_optical_contract(kernel,spectral_order=self.metadata['spectral_order'])
        if digest!=self.metadata.get('optical_fingerprint'):
            raise ValueError('optical/QE/acceptance contract does not match atlas')
        if not np.all(np.asarray(kernel.detector.effective_area_m2)==self.metadata['effective_area_m2']):
            raise ValueError('atlas requires matching area for every supplied OM')
        return True

    @property
    def resident_bytes(self):
        """Memory held by the charge tables and resident rows."""
        return self.charge_left.nbytes+self.charge_right.nbytes+sum(a.nbytes for row in self._rows.values() for a in row.values())

    query=TrackFirstAtlas.query
    query_charge=TrackFirstAtlas.query_charge
    query_charge_many=TrackFirstAtlas.query_charge_many
    segment=TrackFirstAtlas.segment
    segment_charge=TrackFirstAtlas.segment_charge


def track_first_optical_contract(kernel,*,spectral_order=2):
    """Geometry-independent optical contract and SHA256 for universal q1 rows."""
    from copy import copy
    from hashlib import sha256
    import json
    from .primary_polyline import primary_spectral_quadrature
    from ..prompt import _spectral_tables
    local=copy(kernel)
    local.wavelength=primary_spectral_quadrature(kernel.medium,kernel.detector,order=spectral_order,
        wavelength_range_nm=kernel.config.wavelength_range_nm)
    tables=_spectral_tables(local)
    values=dict(acceptance_alpha=np.asarray(kernel.acceptance()['alpha']).tolist(),
        effective_area_m2=float(kernel.detector.effective_area_m2[0]),g=float(kernel.medium.g),
        spectral_order=int(spectral_order),spectral_nodes=len(local.wavelength.wavelength_nm),
        spectral_tables={key:np.asarray(value).tolist() for key,value in tables.items()})
    digest=sha256(json.dumps(values,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    return values,digest
