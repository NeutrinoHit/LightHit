"""Independent coarse charge preview for experimental source-aware screening.

Original ballistic charge is retained. First and higher scattering orders use
two omega-zero operator lanes on the same coarse spectral source. These lanes
are NOT time frequencies and must never be Fourier-inverted. This estimator
is not a rigorous physical bound and is not enabled by production transport.
"""
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
from time import perf_counter, process_time

import numpy as np
from numba import njit, prange


@dataclass(frozen=True)
class ChargePreviewConfig:
    degree: int = 8
    cell_m: float = 2.
    track_uid: int | None = None
    stencil_m: float = .2
    stencil_threshold_m: float = .03
    tail_tolerance_pe: float = 1e-4
    use_detector_tree: bool = False
    source_block_size: int = 256
    azimuthal_degree: int = 0

    def __post_init__(self):
        if isinstance(self.degree, bool) or not isinstance(self.degree, int) or self.degree < 0:
            raise ValueError("degree must be a nonnegative integer")
        if (isinstance(self.azimuthal_degree,bool) or not isinstance(self.azimuthal_degree,int)
                or not 0<=self.azimuthal_degree<=self.degree):
            raise ValueError("azimuthal_degree must lie between zero and degree")
        values = [self.cell_m, self.stencil_m, self.stencil_threshold_m, self.tail_tolerance_pe]
        if not np.isfinite(values).all() or min(values[:2]) <= 0 or min(values[2:]) < 0:
            raise ValueError("invalid preview geometry or tail tolerance")
        if (isinstance(self.source_block_size,bool) or not isinstance(self.source_block_size,int)
                or self.source_block_size<1):
            raise ValueError("source_block_size must be a positive integer")


@dataclass
class ChargePreview:
    charge_orders_pe: np.ndarray
    flags: dict
    metadata: dict

    @property
    def charge_components_pe(self):
        """Ordered columns (0, 1, >=2), integrated expected photoelectrons."""
        return self.charge_orders_pe

    @property
    def charge_pe(self):
        return self.charge_orders_pe.sum(axis=1)

    @property
    def unsafe(self):
        """Receivers requiring explicit treatment rather than trust in preview."""
        return (self.flags["near_source"] | self.flags["near_compiled_point"]
                | self.flags["outside_all_compiled_points"]
                | self.flags["far_tail_estimate_above_tolerance"]
                | self.flags["negative_scattered_charge"] | self.flags["nonfinite"])

    @property
    def unsafe_mask(self):
        return self.unsafe


def _operator_lanes(first, ge2, degree):
    """Combine compatible q1/ge2 omega-zero operators; preserve their labels."""
    for name in ("degree", "acceptance_degree", "phase"):
        if first[name] != ge2[name]:
            raise ValueError(f"first/ge2 packs differ in {name}")
    for name in ("pair_lambda", "pair_mu", "omega", "log_r", "absorption", "speed"):
        if not np.array_equal(first[name], ge2[name]):
            raise ValueError(f"first/ge2 packs differ in {name}")
    if (first["coefficients"].shape != ge2["coefficients"].shape
            or degree > ge2["degree"] or ge2["omega"][0] != 0.):
        raise ValueError("incompatible operator dimensions or missing omega zero")
    coeff = np.stack((first["coefficients"][:, :, :degree+1, :, :, 0],
                      ge2["coefficients"][:, :, :degree+1, :, :, 0]), axis=-1)
    return dict(ge2, degree=degree, coefficients=np.ascontiguousarray(coeff),
                omega=np.zeros(2))


def load_charge_operators(kernel, ge2_store, first_pack_directory, degree):
    """Read existing compatible packs only; never prepare or write a pack."""
    key = ge2_store._key(kernel, np.arange(len(kernel.config.omega_per_ns)))
    if key not in ge2_store._bundles:
        raise ValueError("prepared >=2 store does not contain the kernel grid")
    directory = Path(first_pack_directory)
    header = json.loads((directory / "first-collision.json").read_text())
    if header.get("kind") != "experimental-first-collision-only-v1":
        raise ValueError("expected a separately labelled first-collision pack")
    if header.get("key") != [list(key[0]), key[1], list(key[2])]:
        raise ValueError("first-collision pack belongs to a different numerical kernel")
    first = dict(header["meta"], coefficients=np.load(directory / "first-collision.npy", mmap_mode="r"))
    for name in ("pair_lambda", "pair_mu", "omega", "log_r", "absorption", "speed"):
        first[name] = np.asarray(first[name])
    bundle = _operator_lanes(first, ge2_store._bundles[key], degree)
    return bundle, {"cache_keys": list(key[0]), "source_cache_degree": key[1],
                    "first_pack_directory": str(directory),
                    "operator_lane_labels": ["q1", "q_ge2"],
                    "operator_lane_omega": [0., 0.],
                    "operator_lanes_are_not_a_frequency_grid": True,
                    "prepared_cache_readonly": True}


@njit(cache=True, parallel=True)
def _radial_diagnostics(points, receivers, photon_amplitudes, absorption,
                        out_scale, low, high, peak_acceptance):
    """A loose existing-style tail estimate; deliberately not an error bound."""
    n = len(receivers)
    minimum = np.full(n, np.inf)
    tail = np.zeros(n)
    for d in prange(n):
        for j in range(len(points)):
            dx = receivers[d,0]-points[j,0]
            dy = receivers[d,1]-points[j,1]
            dz = receivers[d,2]-points[j,2]
            radius = np.sqrt(dx*dx+dy*dy+dz*dz)
            minimum[d] = min(minimum[d],radius)
            if radius > high:
                for lam in range(len(absorption)):
                    tail[d] += (10.*peak_acceptance*out_scale[lam,d]*photon_amplitudes[lam,j]
                                *np.exp(-absorption[lam]*radius)/(4.*np.pi*radius*radius))
    return minimum,tail


def charge_preview(kernel, source, *, ge2_store, first_pack_directory,
                   config=None, progress=None):
    """Return all original q0 and independent coarse q1/ge2 charges per OM.

    No fine response is accepted by this function, so it cannot leak a fine
    reference into screening. Only the source, model and prepared operators
    determine the returned charge/geometry flags. The caller owns candidate
    selection and the subsequent fine/fallback calculation.
    An optional ``progress(info, partial_charge)`` callback receives copies of
    incomplete charge arrays for checkpoints; these must not be used to select
    OMs before this function returns successfully.
    """
    from .. import prompt
    from .._shower_basis import compile_axial_slices, apply_multi_spectral
    from .._prompt_numba import min_distance_to_segments
    from ..axial_source import AxisFrame
    config = config or ChargePreviewConfig()
    start,cpu_start = perf_counter(),process_time()
    timings={}
    stage=perf_counter()
    bundle,provenance=load_charge_operators(kernel,ge2_store,first_pack_directory,config.degree)
    tables=prompt._spectral_tables(kernel)
    elements=prompt._source_elements(kernel,source)
    if elements.cone_model != "spectral":
        raise ValueError("charge preview currently requires wavelength-dependent cones")
    valid=elements.beta*float(np.max(tables["phase_index"]))>1
    if not np.any(valid):
        raise ValueError("source is below Cherenkov threshold at all wavelengths")
    if not np.all(valid):
        elements=elements.subset(valid)
    positions=np.ascontiguousarray(kernel.detector.positions_m,float)
    looks=np.ascontiguousarray(-kernel.detector.orientations,float)
    areas=np.asarray(kernel.detector.effective_area_m2,float)
    alpha=np.ascontiguousarray(kernel.acceptance()["alpha"],float)
    low,high=kernel.config.radial_range_m
    n=len(positions)
    detector_tree=None
    if config.use_detector_tree:
        from .detector_tree import DetectorTree,make_source_blocks
        detector_tree=DetectorTree.from_detector(kernel.detector)
    charge=np.zeros((n,3))
    timings["prepare_seconds"]=perf_counter()-stage

    stage=perf_counter()
    physical_minimum=min_distance_to_segments(positions,
             np.ascontiguousarray(elements.start_m),np.ascontiguousarray(elements.direction),
             np.ascontiguousarray(elements.length_m))
    timings["physical_geometry_seconds"]=perf_counter()-stage
    stage=perf_counter()
    ballistic=prompt._ballistic_backend()
    fields=prompt._ballistic_fields(source,elements,tables)
    for lam,wavelength in enumerate(tables["wavelength_nm"]):
        item=fields(lam)
        if item is None:
            continue
        field,cone=item
        charge[:,0]+=(tables["weight_nm"][lam]*tables["efficiency"][lam]*areas
                      *ballistic(field,positions,looks,kernel.medium.band(float(wavelength)),cone,alpha)[0])
    timings["original_ballistic_seconds"]=perf_counter()-stage
    if progress is not None:
        progress(dict(stage="original_ballistic",complete=False,
                      elapsed_seconds=perf_counter()-start),charge.copy())

    pieces=[("source",elements)]
    if config.track_uid is not None:
        mask=elements.uid==config.track_uid
        if not np.any(mask):
            raise ValueError("selected primary UID is absent from source")
        pieces=[("primary",elements.subset(mask))]
        if np.any(~mask):
            pieces.append(("secondary",elements.subset(~mask)))
    peak_acceptance=float(np.max(np.abs(kernel.detector.angular_acceptance(np.linspace(-1.,1.,257)))))
    compiled_minimum=np.full(n,np.inf);far_tail=np.zeros(n)
    compile_seconds=apply_seconds=diagnostic_seconds=0.
    details=[]
    for label,piece in pieces:
        frame=AxisFrame.of(piece.field(0,phase_index=float(np.max(tables["phase_index"]))))
        groups=[]
        for lam,phase in enumerate(tables["phase_index"]):
            mask=piece.beta*phase>1
            if not np.any(mask):
                continue
            subset=piece if np.all(mask) else piece.subset(mask)
            first,second=subset.field(0,phase_index=float(phase)),subset.field(2,phase_index=float(phase))
            wavelength=float(tables["wavelength_nm"][lam])
            field=replace(first,photons=np.ascontiguousarray(
                   first.photons/wavelength**2-second.photons/(wavelength**2*phase**2)))
            stage=perf_counter()
            # Compile ONLY [0.], preserving the charge-only/no-time-lattice path.
            compiled=compile_axial_slices(field,config.degree,np.array([0.]),
                  azimuthal_degree=config.azimuthal_degree,cell_m=config.cell_m,axial_order=3,transverse="stencil",
                  stencil_m=config.stencil_m,stencil_threshold_m=config.stencil_threshold_m,
                  time_step_ns=.02,frame=frame,channel_dtype=np.complex64)
            compile_seconds+=perf_counter()-stage
            if (not groups or lam!=groups[-1][-1][0]+1
                    or not np.array_equal(compiled.points_m(),groups[-1][0][1].points_m())):
                groups.append([])
            groups[-1].append((lam,compiled))
        for group in groups:
            lo,hi=group[0][0],group[-1][0]+1
            band=dict(bundle,coefficients=bundle["coefficients"][lo:hi],
                      absorption=bundle["absorption"][lo:hi],speed=bundle["speed"][lo:hi])
            out_scale=np.stack([tables["weight_nm"][lam]*tables["efficiency"][lam]*areas for lam,_ in group])
            points=np.ascontiguousarray(group[0][1].points_m())
            stage=perf_counter()
            amplitudes=np.ascontiguousarray(np.stack([
                np.sqrt(4*np.pi)*np.max(np.abs(compiled.channels[:,:,0]),axis=1) for _,compiled in group]))
            minimum,tail=_radial_diagnostics(points,positions,amplitudes,
                               band["absorption"],out_scale,low,high,peak_acceptance)
            compiled_minimum=np.minimum(compiled_minimum,minimum);far_tail+=tail
            diagnostic_seconds+=perf_counter()-stage
            # Two independent operator lanes reuse the compiled omega-zero source.
            repeated=[replace(compiled,channels=np.repeat(compiled.channels,2,axis=2)) for _,compiled in group]
            stage=perf_counter()
            tree_stats=None
            if detector_tree is None:
                value=apply_multi_spectral(band,repeated,positions,looks,alpha,out_scale,
                                          radial_window_m=(low,high))
                charge[:,1:]+=value.real.T
            else:
                selection=detector_tree.range_candidates(
                    make_source_blocks(points,block_size=config.source_block_size),
                    maximum_distance_m=high,minimum_cache_distance_m=low)
                tree_stats=asdict(selection.stats)
                for block in selection.blocks:
                    receivers=block.detector_indices
                    if not len(receivers):
                        continue
                    keep=block.source.indices
                    local_sources=[replace(item,z_m=item.z_m[keep],
                                   transverse_m=item.transverse_m[keep],channels=item.channels[keep])
                                   for item in repeated]
                    value=apply_multi_spectral(band,local_sources,positions[receivers],looks[receivers],
                               alpha,out_scale[:,receivers],radial_window_m=(low,high))
                    charge[receivers,1:]+=value.real.T
            elapsed=perf_counter()-stage;apply_seconds+=elapsed
            details.append(dict(source=label,wavelength_bands=len(group),points=len(points),
                                apply_seconds=elapsed,tree_stats=tree_stats))
            if progress is not None:
                progress(dict(stage="operator_group",complete=False,source=label,
                              first_wavelength_index=lo,last_wavelength_index=hi-1,
                              completed_groups=len(details),elapsed_seconds=perf_counter()-start),
                         charge.copy())
    timings.update(compile_seconds=compile_seconds,apply_seconds=apply_seconds,
                   radial_diagnostics_seconds=diagnostic_seconds)
    flags=dict(near_source=physical_minimum<low,near_compiled_point=compiled_minimum<low,
               outside_all_compiled_points=compiled_minimum>high,
               far_tail_estimate_above_tolerance=far_tail>config.tail_tolerance_pe,
               negative_scattered_charge=np.any(charge[:,1:]<0,axis=1),
               nonfinite=~np.isfinite(charge).all(axis=1))
    metadata=dict(**provenance,config=asdict(config),elements=len(elements),modules=n,
        wavelength_nm=tables["wavelength_nm"].tolist(),radial_window_m=[low,high],
        elapsed_seconds=perf_counter()-start,cpu_seconds=process_time()-cpu_start,timings=timings,
        groups=details,flag_counts={key:int(value.sum()) for key,value in flags.items()},
        maximum_far_tail_estimate_pe=float(far_tail.max()),
        tail_estimate_scope="loose existing point-source screening estimate, not a rigorous error bound",
        minimum_physical_distance_m=float(physical_minimum.min()),
        minimum_compiled_distance_m=float(compiled_minimum.min()),
        computed_observable="integrated expected photoelectrons only; no bins or spectrum",
        source_compilation_frequency_grid=[0.],preview_is_a_bound=False)
    return ChargePreview(charge,flags,metadata)
