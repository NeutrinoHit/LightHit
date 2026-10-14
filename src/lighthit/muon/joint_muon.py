"""The muon route: all scattering orders from G4 steps and secondary families.

Direct light is computed exactly on the original steps. The primary uses the
finite-track atlases with direct fallbacks. Secondaries are split into packets
by their ancestry: Q1 by the grouped prompt solver, >=2 by point RTE sources.
OMs are chosen from an independent preview and reopened where needed.
:class:`lighthit.muon.MuonTransport` is the user-facing entry.
"""
from dataclasses import dataclass, replace
from time import perf_counter, process_time
import numpy as np


@dataclass(frozen=True)
class JointMuonConfig:
    """Settings of the muon route; the defaults are the validated fast choice.

    Secondary Q1: grouped physical-time solver with the nine bands fused and weak
    packet-OM pairs omitted under a certified bound (``first_*``). Secondary >=2:
    point RTE sources applied in float32 batches, with predicted weak pairs omitted
    (``ge2_*``). ``first_backend='atlas'``, ``ge2_backend='basis'`` and
    ``ge2_operator='scalar'`` select earlier research alternatives.
    """
    z_step_m: float = .5
    photon_mu_step: float = .03
    photon_phi_step_rad: float = .06
    tau_step_ns: float = .1
    transverse_step_m: float = .1
    ring_nodes: int = 96
    first_chunk_elements: int = 4096
    ge2_z_step_m: float = .25
    ge2_tau_step_ns: float = .5
    ge2_transverse_step_m: float = .1
    ge2_backend: str = 'joint'
    ge2_operator: str = 'blas'
    first_backend: str = 'grouped'
    prompt_radius_m: float = 400.
    first_bins_tolerance_pe: float = 1e-6
    primary_uid: int = 0
    first_skip_epsilon: float = 1e-3
    first_skip_seed_fraction: float = .5
    first_fuse_bands: bool = True
    ge2_batch_points: int = 2000
    ge2_blas_precision: str = 'float32'
    ge2_skip_epsilon: float = 1e-3
    ge2_skip_safety: float = 3.
    ge2_skip_near_factor: float = 3.

    def __post_init__(self):
        """Reject grids, budgets and combinations that the route does not support."""
        numeric=[self.z_step_m,self.photon_mu_step,self.photon_phi_step_rad,
            self.tau_step_ns,self.transverse_step_m,self.ge2_z_step_m,
            self.ge2_tau_step_ns,self.ge2_transverse_step_m,self.prompt_radius_m]
        if not np.isfinite(numeric).all() or min(numeric)<=0:
            raise ValueError('joint source grid lengths must be positive and finite')
        if not isinstance(self.ring_nodes,(int,np.integer)) or self.ring_nodes<4:
            raise ValueError('at least four ring nodes required')
        if not isinstance(self.first_chunk_elements,(int,np.integer)) or self.first_chunk_elements<1:
            raise ValueError('positive first_chunk_elements required')
        if self.ge2_backend not in ('joint','basis'):
            raise ValueError('ge2_backend must be joint or basis')
        if self.ge2_operator not in ('scalar','blas'):
            raise ValueError('ge2_operator must be scalar or blas')
        if self.first_backend not in ('atlas','grouped'):
            raise ValueError('first_backend must be atlas or grouped')
        if not np.isfinite(self.first_bins_tolerance_pe) or self.first_bins_tolerance_pe<0:
            raise ValueError('first_bins_tolerance_pe must be finite and nonnegative')
        if not (np.isfinite(self.first_skip_epsilon) and 0<=self.first_skip_epsilon<1):
            raise ValueError('first_skip_epsilon must be finite and in [0, 1)')
        if self.first_skip_epsilon and self.first_backend!='grouped':
            raise ValueError('first_skip_epsilon requires first_backend=grouped')
        if not (np.isfinite(self.first_skip_seed_fraction) and 0<self.first_skip_seed_fraction<=1):
            raise ValueError('first_skip_seed_fraction must be in (0, 1]')
        if not (np.isfinite(self.ge2_skip_epsilon) and 0<=self.ge2_skip_epsilon<1):
            raise ValueError('ge2_skip_epsilon must be finite and in [0, 1)')
        if self.first_fuse_bands and self.first_backend!='grouped':
            raise ValueError('first_fuse_bands requires first_backend=grouped')
        if (not isinstance(self.ge2_batch_points,(int,np.integer)) or isinstance(self.ge2_batch_points,bool)
                or self.ge2_batch_points<0):
            raise ValueError('ge2_batch_points must be a nonnegative integer')
        if self.ge2_blas_precision not in ('float64','float32'):
            raise ValueError("ge2_blas_precision must be 'float64' or 'float32'")
        if (self.ge2_batch_points or self.ge2_blas_precision!='float64') and (
                self.ge2_backend!='joint' or self.ge2_operator!='blas'):
            raise ValueError('ge2 batching/precision require ge2_backend=joint and ge2_operator=blas')
        if self.ge2_skip_epsilon and self.ge2_backend!='joint':
            raise ValueError('ge2_skip_epsilon requires ge2_backend=joint')
        if not (np.isfinite(self.ge2_skip_safety) and self.ge2_skip_safety>=1):
            raise ValueError('ge2_skip_safety must be finite and >= 1')
        if not (np.isfinite(self.ge2_skip_near_factor) and self.ge2_skip_near_factor>=1):
            raise ValueError('ge2_skip_near_factor must be finite and >= 1')


def secondary_joint_transport(kernel, source, ancestry, receiver_indices, *,
                              time_origin_ns, ge2_store, config=None,
                              first_operator=None, progress=None):
    """Stream every secondary packet, without a per-family charge threshold.

    ``first_operator`` has the direct_photon_histogram signature. Source
    geometry supports each family; receivers beyond the explicit prompt radius
    are omitted consistently with the previous packet solver. >=2 retains the
    supplied finite L/M/frequency RTE pack and its radial window. The caller must
    select receivers independently and label uncomputed OMs.
    """
    from .muon_packets import partition_muon
    from .joint_source import compile_joint_source,compile_photon_histogram
    from .point_ge2 import apply_point_spectral
    from .._shower_basis import compile_axial_slices,apply_multi
    from ..axial_source import AxisFrame
    from ..readout import inverse_bins
    from scipy.interpolate import CubicSpline
    config=config or JointMuonConfig()
    if config.first_backend=='atlas':
        # Research route: secondary Q1 from photon histograms (experimental).
        from ..experimental.joint_first import direct_photon_histogram
        first_operator=first_operator or direct_photon_histogram
    ids=np.asarray(receiver_indices,np.int64);origins=np.asarray(time_origin_ns,float)
    if ids.ndim!=1 or origins.shape!=ids.shape or not np.isfinite(origins).all():
        raise ValueError('one finite local time origin per selected receiver required')
    positions=np.ascontiguousarray(kernel.detector.positions_m[ids])
    looks=np.ascontiguousarray(-kernel.detector.orientations[ids])
    area=np.asarray(kernel.detector.effective_area_m2)[ids]
    alpha=np.ascontiguousarray(kernel.acceptance()['alpha'],float)
    wave=kernel.wavelength;sample=kernel.medium.sample(wave.wavelength_nm)
    omega=kernel.config.omega_per_ns
    bundle=ge2_store.bundle(kernel,np.arange(len(omega)))
    scale=wave.weight_nm[:,None]*np.asarray(kernel.detector.spectral_weight(wave.wavelength_nm))[:,None]*area[None,:]
    edges=np.asarray(kernel.config.relative_time_edges_ns)
    q1=np.zeros(len(ids));b1=np.zeros((len(ids),len(edges)-1));s2=np.zeros((len(omega),len(ids)),complex)
    packets=partition_muon(source,ancestry,primary_uid=config.primary_uid)
    per_packet_budget=config.first_bins_tolerance_pe/max(1,len(packets)-1)
    cpu,wall=process_time(),perf_counter();details=[]
    first_metadata=None;first_rows={}
    if config.first_backend=='grouped':
        from .secondary_first import evaluate_secondary_first
        q1,b1,first_metadata=evaluate_secondary_first(kernel,source,packets,ids,
            time_origin_ns=origins,radius_m=config.prompt_radius_m,progress=progress,
            skip_epsilon=config.first_skip_epsilon,skip_seed_fraction=config.first_skip_seed_fraction,
            fuse_bands=config.first_fuse_bands)
        first_rows={row['packet']:row for row in first_metadata['packets']}
    ge2_keep,ge2_skip=None,None
    if config.ge2_skip_epsilon>0:
        ge2_keep,ge2_skip=_ge2_skip_plan(kernel,source,packets,positions,looks,alpha,scale,
            ge2_store,bundle,sample,wave,config)
    ge2_batch,ge2_batch_count,ge2_batch_cpu=[],0,0.
    def flush_ge2():
        """Apply the collected packets to all OMs in one batched call."""
        nonlocal ge2_batch,ge2_batch_count,ge2_batch_cpu
        if ge2_batch:
            from .point_ge2_blas import apply_point_spectral_blas_multi
            t=process_time()
            s2[:]+=apply_point_spectral_blas_multi(bundle,ge2_batch,positions,looks,alpha,scale,
                azimuthal_degree=kernel.config.azimuthal_degree,
                radial_window_m=kernel.config.radial_range_m,precision=config.ge2_blas_precision)
            ge2_batch_cpu+=process_time()-t
        ge2_batch,ge2_batch_count=[],0
    for number,packet in enumerate(packets):
        if packet.kind=='primary':continue
        part=packet.source(source)
        ends=part.start_m+part.length_m[:,None]*part.direction
        centre=.5*(np.minimum(part.start_m.min(0),ends.min(0))+np.maximum(part.start_m.max(0),ends.max(0)))
        reach=max(np.linalg.norm(part.start_m-centre,axis=1).max(),np.linalg.norm(ends-centre,axis=1).max())
        distance=np.linalg.norm(positions-centre,axis=1)-reach
        work=np.flatnonzero(distance<=config.prompt_radius_m)
        row=dict(packet=number,kind=packet.kind,family_uid=packet.family_uid,elements=len(part),
                 prompt_receivers=len(work),first_compile_cpu_s=0.,first_apply_cpu_s=0.,
                 ge2_compile_cpu_s=0.,ge2_apply_cpu_s=0.,first_clock_rows=0,ge2_rows=0)
        if number in first_rows:
            for key in ('first_compile_cpu_s','first_apply_cpu_s'):
                row[key]=first_rows[number][key]
        if len(work) and config.first_backend=='atlas':
            first_frame=AxisFrame.of(part.field(0,phase_index=float(sample['phase_index'].max())))
            for begin in range(0,len(part),config.first_chunk_elements):
                chunk=part.subset((np.arange(len(part))>=begin)&(np.arange(len(part))<begin+config.first_chunk_elements))
                for lam,(wavelength,phase) in enumerate(zip(wave.wavelength_nm,sample['phase_index'])):
                    if not np.any(chunk.beta*phase>1):continue
                    t=process_time()
                    hist=compile_photon_histogram(chunk,float(wavelength),float(phase),
                        z_step_m=config.z_step_m,mu_step=config.photon_mu_step,
                        phi_step_rad=config.photon_phi_step_rad,tau_step_ns=config.tau_step_ns,
                        transverse_step_m=config.transverse_step_m,ring_nodes=config.ring_nodes,
                        frame=first_frame,
                        retain_samples=first_operator is not direct_photon_histogram)
                    row['first_compile_cpu_s']+=process_time()-t;row['first_clock_rows']+=len(hist.geometry_id)
                    t=process_time()
                    extra=({} if first_operator is direct_photon_histogram else
                           {'bins_tolerance_pe':per_packet_budget/(len(wave.wavelength_nm)*
                               max(1,int(np.ceil(len(part)/config.first_chunk_elements))))})
                    result=first_operator(kernel,hist,ids[work],edges,origins[work],
                        wavelength_weight_nm=float(wave.weight_nm[lam]),**extra)
                    q,b=result[:2]
                    q1[work]+=q;b1[work]+=b
                    if len(result)>2:
                        row['first_operator_metadata']=result[2]
                    row['first_apply_cpu_s']+=process_time()-t
        target=np.flatnonzero(distance<=kernel.config.radial_range_m[1])
        if ge2_keep is not None and number in ge2_keep:
            row['ge2_skipped_receivers']=int(np.count_nonzero(~ge2_keep[number][target]))
            target=target[ge2_keep[number][target]]
        if len(target) and np.any(part.beta*sample['phase_index'].max()>1):
            t=process_time()
            frame=AxisFrame.of(part.field(0,phase_index=float(sample['phase_index'].max())))
            try:
                joint=(None if config.ge2_backend=='basis' else compile_joint_source(part,wave.wavelength_nm,sample['phase_index'],
                    z_step_m=config.ge2_z_step_m,mu_step=2.01,tau_step_ns=config.ge2_tau_step_ns,
                    transverse_step_m=config.ge2_transverse_step_m,degree=int(bundle['degree']),
                    azimuthal_degree=kernel.config.azimuthal_degree,frame=frame))
            except ValueError as error:
                if 'max_payload_bytes' not in str(error):raise
                joint=None;row['ge2_fallback']='axial basis after histogram payload guard'
            row['ge2_compile_cpu_s']+=process_time()-t
            if joint is not None and config.ge2_batch_points:
                row['ge2_rows']=len(joint.times_ns)
                ge2_batch.append((frame,joint.points_m,joint.angular,joint.times_ns,target))
                ge2_batch_count+=len(joint.times_ns)
                if ge2_batch_count>=config.ge2_batch_points:flush_ge2()
            elif joint is not None:
                row['ge2_rows']=len(joint.times_ns);t=process_time()
                s2[:,target]+=apply_point_spectral(bundle,frame,joint.points_m,joint.angular,
                    joint.times_ns,positions[target],looks[target],alpha,scale[:,target],
                    azimuthal_degree=kernel.config.azimuthal_degree,
                    radial_window_m=kernel.config.radial_range_m,backend=config.ge2_operator,
                    blas_precision=config.ge2_blas_precision)
                row['ge2_apply_cpu_s']+=process_time()-t
            else:
                geometry=part.field(0,phase_index=float(sample['phase_index'].max()))
                for lam,(wavelength,phase) in enumerate(zip(wave.wavelength_nm,sample['phase_index'])):
                    valid=part.beta*phase>1
                    if not valid.any():continue
                    subset=part if valid.all() else part.subset(valid)
                    a,b=subset.field(0,phase_index=float(phase)),subset.field(2,phase_index=float(phase))
                    field=replace(a,photons=np.ascontiguousarray((a.photons-b.photons/phase**2)/wavelength**2))
                    t=process_time()
                    compiled=compile_axial_slices(field,int(bundle['degree']),omega,
                        azimuthal_degree=kernel.config.azimuthal_degree,cell_m=config.ge2_z_step_m,
                        time_step_ns=.1,time_deposit='cubic',frame=frame,geometry_reference=geometry,
                        stencil_center='centroid',axial_clock_ns_per_m=1/.299792458)
                    row['ge2_compile_cpu_s']+=process_time()-t;t=process_time()
                    band=dict(bundle,coefficients=bundle['coefficients'][lam:lam+1],
                              absorption=bundle['absorption'][lam:lam+1],speed=bundle['speed'][lam:lam+1])
                    s2[:,target]+=apply_multi(band,compiled,compiled,positions[target],looks[target],
                        alpha,np.ones(1),np.zeros(1),scale[lam:lam+1,target],
                        radial_window_m=kernel.config.radial_range_m)
                    row['ge2_apply_cpu_s']+=process_time()-t
        details.append(row)
        if progress:progress('secondary packet',row)
    flush_ge2()
    outomega=kernel.config.readout_omega_per_ns
    outomega=omega if outomega is None else np.asarray(outomega)
    relative=s2*np.exp(-1j*omega[:,None]*origins[None,:])
    if not np.array_equal(outomega,omega):relative=CubicSpline(omega,relative,axis=0)(outomega)
    b2=inverse_bins(outomega,relative,edges)
    return q1,b1,s2[0].real,b2,dict(cpu_s=process_time()-cpu,wall_s=perf_counter()-wall,
        packets=details,orders=[1,'>=2'],scope='all secondary families on supplied receivers',
        waveform_scope='>=2 retains sampled RTE frequency/readout; first order physical time',
        prompt_radius_m=config.prompt_radius_m,first_backend=config.first_backend,
        first_metadata=first_metadata,ge2_batched_apply_cpu_s=ge2_batch_cpu,
        ge2_batch_points=config.ge2_batch_points,ge2_blas_precision=config.ge2_blas_precision,
        **({'ge2_skip':ge2_skip} if ge2_skip is not None else {})),s2


def _ge2_skip_plan(kernel,source,packets,positions,looks,alpha,scale,ge2_store,bundle,
                   sample,wave,config):
    """Choose secondary packet-receiver pairs whose >=2 apply may be omitted.

    Each packet's joint source is collapsed to one point at its |l=0| weighted
    centroid with the summed angular coefficients; the omega=0 operator then
    predicts its all-time >=2 charge at every receiver. This is a prediction,
    not a bound. Only receivers at least ``ge2_skip_near_factor`` times the
    packet's largest point offset away, with every point on the same side of
    the radial window as the centroid, are candidates. Per receiver, candidates
    are omitted by largest points/|prediction| while safety*sum|prediction|
    stays <= epsilon*|sum of all predictions|. Packets that cannot be
    predicted (payload guard) are always computed.
    """
    from .joint_source import compile_joint_source
    from .point_ge2 import apply_point_spectral
    from ..axial_source import AxisFrame
    started=process_time()
    zero=dict(ge2_store.bundle(kernel,np.array([0])))
    low,high=map(float,kernel.config.radial_range_m)
    nrec=len(positions);phase_max=float(sample['phase_index'].max())
    predicted,candidate,cost={},{},{}
    for number,packet in enumerate(packets):
        if packet.kind=='primary':continue
        part=packet.source(source)
        if not np.any(part.beta*phase_max>1):continue
        ends=part.start_m+part.length_m[:,None]*part.direction
        centre=.5*(np.minimum(part.start_m.min(0),ends.min(0))+np.maximum(part.start_m.max(0),ends.max(0)))
        reach=max(np.linalg.norm(part.start_m-centre,axis=1).max(),np.linalg.norm(ends-centre,axis=1).max())
        target=np.flatnonzero(np.linalg.norm(positions-centre,axis=1)-reach<=high)
        if not len(target):continue
        frame=AxisFrame.of(part.field(0,phase_index=phase_max))
        try:
            joint=compile_joint_source(part,wave.wavelength_nm,sample['phase_index'],
                z_step_m=config.ge2_z_step_m,mu_step=2.01,tau_step_ns=config.ge2_tau_step_ns,
                transverse_step_m=config.ge2_transverse_step_m,degree=int(bundle['degree']),
                azimuthal_degree=kernel.config.azimuthal_degree,frame=frame)
        except ValueError as error:
            if 'max_payload_bytes' not in str(error):raise
            continue
        weight=np.abs(joint.angular[:,:,0]).sum(0)
        if not len(weight) or not weight.sum()>0:continue
        point=(weight[:,None]*joint.points_m).sum(0)/weight.sum()
        offset=float(np.linalg.norm(joint.points_m-point,axis=1).max())
        value=apply_point_spectral(zero,frame,point[None,:],joint.angular.sum(axis=1,keepdims=True),
            np.zeros(1),positions[target],looks[target],alpha,scale[:,target],
            azimuthal_degree=kernel.config.azimuthal_degree,radial_window_m=(low,high),
            backend=config.ge2_operator)[0].real
        distance=np.linalg.norm(positions[target]-point,axis=1)
        inside=(distance-offset>=low)&(distance+offset<=high)
        outside=distance-offset>high
        ok=(inside|outside)&(distance>=config.ge2_skip_near_factor*offset)
        row=np.zeros(nrec);row[target]=value;predicted[number]=row
        mask=np.zeros(nrec,bool);mask[target[ok]]=True;candidate[number]=mask
        cost[number]=float(len(joint.times_ns))
    numbers=np.array(sorted(predicted),np.int64)
    keep={int(n):np.ones(nrec,bool) for n in numbers}
    total=np.zeros(nrec);omitted=np.zeros(nrec);skipped=0
    for n in numbers:total+=predicted[n]
    for j in range(nrec):
        budget=config.ge2_skip_epsilon*abs(total[j])
        pool=[n for n in numbers if candidate[n][j]]
        if not pool or not budget>0:continue
        ratio=np.array([cost[n]/max(config.ge2_skip_safety*abs(predicted[n][j]),1e-300) for n in pool])
        used=0.
        for index in np.argsort(-ratio,kind='stable'):
            n=pool[index];charge=config.ge2_skip_safety*abs(predicted[n][j])
            if used+charge<=budget:
                used+=charge;keep[int(n)][j]=False;skipped+=1
        omitted[j]=used/config.ge2_skip_safety
    return keep,dict(epsilon=config.ge2_skip_epsilon,safety=config.ge2_skip_safety,
        near_factor=config.ge2_skip_near_factor,predict_cpu_s=process_time()-started,
        predicted_packets=len(numbers),skipped_pairs=skipped,
        predicted_omitted_charge_pe=omitted.tolist(),predicted_total_charge_pe=total.tolist(),
        statement=('prediction-based omission, not a certificate: one centroid point per packet '
                   'with summed angular coefficients at omega=0'))


def transport_joint_muon(kernel,source,ancestry,*,preview,primary_first_store,
                         primary_ge2_store,ge2_store,joint_first_atlas=None,
                         config=None,primary_options=None,screen_config=None,progress=None,
                         time_origin_ns=None,selection_extra_charge=None):
    """Compose all orders and all source families on independently selected OMs.

    The empirical preview/sentinels/reopening controller is shared with the
    previous prototype. Unselected scattered charges remain labelled estimates.
    Original charged steps supply converged spectral ballistic light; source
    histograms are used only by scattered-light calculations.

    ``time_origin_ns`` replaces the per-OM origin (earliest local arrival of
    this source) by a common event clock; ``selection_extra_charge`` adds light
    of other sources of the event to every threshold decision (see
    :func:`lighthit.muon.adaptive_screen.adaptive_candidates`). The returned
    ``active`` then refers to the event sum; bins exist for every evaluated OM.
    """
    from copy import copy
    from dataclasses import asdict
    from .universal_primary import transport_universal_primary,subset_kernel
    from .adaptive_screen import adaptive_candidates
    from .._shower_basis import earliest_element_arrival
    from ..ballistic_interval import ballistic_interval
    from ..transport import HybridTransportResponse
    from functools import partial
    config=config or JointMuonConfig()
    elements=kernel._source_elements(source)
    n=len(kernel.detector.positions_m);omega=kernel.config.omega_per_ns
    if not np.array_equal(np.asarray(primary_ge2_store.omega_per_ns),omega):
        raise ValueError('primary atlas and joint RTE frequency grids differ')
    sample=kernel.medium.sample(kernel.wavelength.wavelength_nm)
    if time_origin_ns is None:
        origins=earliest_element_arrival(np.ascontiguousarray(kernel.detector.positions_m),
            elements.start_m,elements.direction,elements.length_m,elements.start_ns,
            float(sample['group_index'].min()))
    else:
        origins=np.asarray(time_origin_ns,float)
        if origins.shape!=(n,) or not np.isfinite(origins).all():
            raise ValueError('time_origin_ns must hold one finite origin per OM')
    extra=np.zeros(n) if selection_extra_charge is None else np.asarray(selection_extra_charge,float)
    if extra.shape!=(n,) or not np.isfinite(extra).all():
        raise ValueError('selection_extra_charge must hold one finite value per OM')
    began,cpu=perf_counter(),process_time();stage=perf_counter()
    ballistic=ballistic_interval(kernel,elements,time_origin_ns=origins,
        relative_edges_ns=kernel.config.relative_time_edges_ns)
    q0_seconds=perf_counter()-stage
    # The exact direct charge participates in global selection.
    coarse=copy(preview);coarse.charge_components_pe=np.asarray(preview.charge_components_pe).copy()
    coarse.charge_components_pe[:,0]=ballistic.charge_pe
    coarse.charge_pe=coarse.charge_components_pe.sum(1)
    components=np.zeros((n,len(kernel.config.relative_time_edges_ns)-1,3))
    components[:,:,0]=ballistic.bins_pe
    spectrum=np.zeros((len(omega),n),complex);batches=[]
    options={} if primary_options is None else dict(primary_options)
    first_operator=None
    if config.first_backend=='atlas':
        from ..experimental.joint_first_atlas import evaluate_photon_histogram_atlas
        first_operator=partial(evaluate_photon_histogram_atlas,joint_first_atlas)
    def fine(ids):
        """Primary and secondary transport on one batch of selected OMs."""
        if progress:progress('fine batch started',dict(receivers=len(ids)))
        local=subset_kernel(kernel,ids);local.config=replace(kernel.config,threshold_pe=0.)
        primary=transport_universal_primary(local,elements,primary_first_store,primary_ge2_store,
            primary_uid=config.primary_uid,time_origin_ns=origins[ids],skip_ballistic=True,**options)
        if progress:progress('primary complete',dict(receivers=len(ids),
            cpu_seconds=primary.metadata['cpu_seconds'],
            elapsed_seconds=primary.metadata['elapsed_seconds']))
        q1,b1,q2,b2,meta,secondary_spectrum=secondary_joint_transport(kernel,elements,ancestry,ids,
            time_origin_ns=origins[ids],ge2_store=ge2_store,config=config,
            first_operator=first_operator,progress=progress)
        charge=np.column_stack((ballistic.charge_pe[ids],primary.charge_orders_pe[:,1]+q1,
                               primary.charge_orders_pe[:,2]+q2))
        components[ids,:,1]=primary.bins_orders_pe[:,:,1]+b1
        components[ids,:,2]=primary.bins_orders_pe[:,:,2]+b2
        spectrum[:,ids]=primary.spectrum_ge2_pe+secondary_spectrum
        batches.append(dict(receivers=len(ids),primary=primary.metadata,secondary=meta))
        return charge
    charge,evaluated,audit=adaptive_candidates(coarse,kernel.detector,kernel.config.threshold_pe,
                                             fine,screen_config,
                                             extra_charge=None if selection_extra_charge is None else extra)
    active=evaluated&(charge.sum(1)+extra>=kernel.config.threshold_pe)
    spectrum[0,~evaluated]=charge[~evaluated,2]
    metadata=dict(elapsed_seconds=perf_counter()-began,cpu_seconds=process_time()-cpu,
        ballistic_seconds=q0_seconds,config=asdict(config),batches=batches,smart_screen=audit,
        fine_charge_mask=evaluated.tolist(),active_modules=int(active.sum()),
        inactive_order1_charge_may_be_coarse=bool((~evaluated).any()),
        inactive_ge2_charge_may_be_coarse=bool((~evaluated).any()),
        prompt_order1_level=np.where(evaluated,2,1).tolist(),
        computed_orders=[0,1,'>=2'],first_backend=config.first_backend,ge2_backend=config.ge2_backend,
        source_representation=('original grouped secondary Q1; joint harmonic bins for >=2'
            if config.first_backend=='grouped' and config.ge2_backend=='joint' else
            'explicit local source representations, recorded separately by collision order'),
        primary_uid=config.primary_uid,
        limitations=['empirical OM selection, not an inactivity certificate',
                     'finite source bins need response convergence',
                     '>=2 shares existing finite RTE frequency/radial/angular approximations'])
    return HybridTransportResponse(kernel.detector,omega,kernel.config.relative_time_edges_ns,
        spectrum,components,charge,origins,active,'hybrid-joint-muon-experimental',metadata)
