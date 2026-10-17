""">=2 of slow (beta < 1) primary pieces in batches, and near-one cone coefficients.

No event cache is created. Batches use the spectral axial compiler with the
retained azimuthal harmonics; its spatial and time grids are an approximation
checked against direct segment quadrature.
"""
from copy import copy
from dataclasses import dataclass,replace
from types import SimpleNamespace
from time import perf_counter,process_time
import numpy as np
from scipy.special import eval_legendre,eval_jacobi


@dataclass
class Ge2TailBatch:
    """Charge and optional spectrum and bins of a batch of slow (beta < 1) pieces."""
    charge_pe: np.ndarray
    spectrum_pe: np.ndarray | None       # frequency,OM; relative to time_origin_ns
    bins_pe: np.ndarray | None
    omega_per_ns: np.ndarray
    time_origin_ns: np.ndarray
    metadata: dict


def cone_beta_taylor(wavelength_nm,phase_index,degree,order=2):
    """Coefficients of delta=1/beta-1 through order2, per source C0.

    h_l = sqrt((2l+1)/(4pi))*(1-x²)*P_l(x)/lambda²,
    x=(1+delta)/n(lambda). Threshold crossings are NOT covered by this series.
    Returned shape: (order+1,wavelength,degree+1). Clock phases are separate.
    """
    if order not in (0,1,2):raise ValueError('implemented Taylor orders are0,1,2')
    wavelength=np.asarray(wavelength_nm,float);n=np.asarray(phase_index,float)
    if wavelength.shape!=n.shape or wavelength.ndim!=1 or np.any(n<=1):raise ValueError('invalid optical arrays')
    x=1/n;ell=np.arange(degree+1)
    p=eval_legendre(ell[None,:],x[:,None]);dp=np.zeros_like(p);ddp=np.zeros_like(p)
    if degree>=1:dp[:,1:]=(ell[None,1:]+1)/2*eval_jacobi(ell[None,1:]-1,1,1,x[:,None])
    if degree>=2:ddp[:,2:]=(ell[None,2:]+1)*(ell[None,2:]+2)/4*eval_jacobi(ell[None,2:]-2,2,2,x[:,None])
    factor=np.sqrt((2*ell+1)/(4*np.pi))[None,:]/wavelength[:,None]**2
    values=[(1-x[:,None]**2)*p]
    if order>=1:values.append(((1-x[:,None]**2)*dp-2*x[:,None]*p)/n[:,None])
    if order>=2:values.append(((1-x[:,None]**2)*ddp-4*x[:,None]*dp-2*p)/(2*n[:,None]**2))
    return np.stack(values)*factor[None,:,:]


def primary_tail_mask(pieces,*,beta_tolerance=1e-5,clock_tolerance_ns=1e-3):
    """Conservative source-wide split; independent of which OM sees a piece."""
    clock=(pieces.end_ns-pieces.start_ns)/pieces.length_m
    return ((abs(pieces.beta-1)>beta_tolerance)
            | (abs(clock-1/.299792458)*pieces.length_m/2>clock_tolerance_ns))


def subdivide_tail_elements(source,max_length_m):
    """Bound the two-node source compiler quadrature on fitted long chords."""
    from ..sources import SpectralLightElements
    if not np.isfinite(max_length_m) or max_length_m<=0:raise ValueError('invalid substep length')
    counts=np.maximum(1,np.ceil(source.length_m/max_length_m).astype(int))
    if np.all(counts==1):return source
    indices=np.repeat(np.arange(len(source)),counts)
    local=np.arange(len(indices))-np.repeat(np.cumsum(counts)-counts,counts)
    fraction=local/counts[indices];share=1/counts[indices]
    lengths=source.length_m[indices];duration=source.end_ns[indices]-source.start_ns[indices]
    return SpectralLightElements(source.start_m[indices]+(fraction*lengths)[:,None]*source.direction[indices],
        source.direction[indices],lengths*share,source.coefficient0[indices]*share,source.coefficient2[indices]*share,
        source.beta[indices],source.start_ns[indices]+fraction*duration,
        source.start_ns[indices]+(fraction+share)*duration,np.arange(len(indices)),source.uid[indices],cone_model=source.cone_model)


def transport_ge2_tail_batch(kernel,source,store,*,full=False,cell_m=.25,
                             azimuthal_degree=4,time_origin_ns=None):
    """Compile actual-beta source once, then apply to all supplied OMs.

    Pass only the globally assigned tail source. Do not add per-pair direct
    results for that same source again. For full=True, supply a kernel subset
    containing only OMs selected by the preceding all-OM charge pass.
    """
    from ..shower_basis import ShowerBasis,Ge2KernelStore,transport_hybrid_basis
    from ..readout import inverse_bins
    from scipy.interpolate import CubicSpline
    if not len(source):raise ValueError('tail source must be nonempty')
    original_count=len(source)
    source=subdivide_tail_elements(source,cell_m/2)
    if azimuthal_degree<0 or azimuthal_degree>kernel.config.source_degree:raise ValueError('invalid angular band')
    local=copy(kernel)
    local.config=replace(kernel.config,threshold_pe=0. if full else 1e100,
                         azimuthal_degree=int(azimuthal_degree))
    if isinstance(store,(str,bytes)) or hasattr(store,'__fspath__'):store=Ge2KernelStore.load(store)
    local._ge2_store=store
    count=len(local.detector.positions_m);edges=np.asarray(local.config.relative_time_edges_ns)
    valid=source.beta*np.max(local.medium.sample(local.wavelength.wavelength_nm)['phase_index'])>1
    filtered=source if valid.all() else source.subset(valid)
    if not len(filtered):raise ValueError('tail has no above-threshold source')
    def no_prompt(*unused,**kwargs):
        """Stand-in prompt response: zero orders 0 and 1 for the >=2-only batch."""
        origins=kwargs.get('time_origin_ns')
        if origins is None:
            first=int(np.argmin(filtered.start_ns));fastest=float(np.min(local.medium.sample(local.wavelength.wavelength_nm)['group_index']))
            origins=(filtered.start_ns[first]+np.linalg.norm(local.detector.positions_m-filtered.start_m[first],axis=1)*fastest/.299792458)
        return SimpleNamespace(relative_time_edges_ns=edges,time_origin_ns=np.asarray(origins),
            charge_orders_pe=np.zeros((count,2)),charge_pe=np.zeros(count),computed=np.ones(count,bool),
            bins_orders_pe=np.zeros((count,len(edges)-1,2)),
            metadata=dict(prompt_algorithm='not calculated: isolated >=2 tail',order1_level=[0]*count))
    local.transport_prompt=no_prompt
    basis=ShowerBasis(cell_m=cell_m,track_uid=None,spectral_fusion=True,transverse='stencil',
        time_step_ns=.02,radial_high_m=local.config.radial_range_m[1],radial_tail_tolerance_pe=1e-4)
    wall=perf_counter();cpu=process_time()
    # Points closer than the radial minimum are added by lighthit.near_field.
    response=transport_hybrid_basis(local,filtered,basis,frequency_nodes=len(local.config.omega_per_ns),store=store,
        near_points='exclude')
    cpu=process_time()-cpu;wall=perf_counter()-wall
    origins=response.time_origin_ns if time_origin_ns is None else np.asarray(time_origin_ns,float)
    if origins.shape!=(count,) or not np.isfinite(origins).all():raise ValueError('invalid receiver time origins')
    spectrum=None;bins=None
    if full:
        spectrum=response.spectrum_ge2_pe*np.exp(-1j*response.omega_per_ns[:,None]*origins[None,:])
        outomega=local.config.readout_omega_per_ns
        if outomega is None:outomega=response.omega_per_ns
        relative=spectrum if np.array_equal(outomega,response.omega_per_ns) else CubicSpline(response.omega_per_ns,spectrum,axis=0)(outomega)
        bins=inverse_bins(outomega,relative,edges)
    metadata=dict(source_elements=original_count,subdivided_elements=len(source),max_substep_m=float(np.max(source.length_m)),
        above_threshold_elements=len(filtered),modules=count,
        cpu_seconds=cpu,wall_seconds=wall,full=bool(full),cell_m=float(cell_m),azimuthal_degree=int(azimuthal_degree),
        compiler_metadata=response.metadata,source_beta_range=[float(source.beta.min()),float(source.beta.max())],
        approximation='existing cubic axial slices, transverse stencil, angular band and0.02ns time lattice',
        source_policy='actual beta and wavelength-dependent cone; no beta1 approximation',
        prompt_computed=False)
    return Ge2TailBatch(response.charge_components_pe[:,2].copy(),spectrum,bins,response.omega_per_ns.copy(),origins.copy(),metadata)
