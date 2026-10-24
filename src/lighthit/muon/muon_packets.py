"""An ancestry-aware partition of a muon event, without replacing its light.

Large electromagnetic descendant families are compact quasi-showers. Soft
delta-electron families are grouped locally along the primary. Hadronic and
late sources retain their own records. Every source element occurs exactly
once; wavelength, beta, angular, spatial and emission-time information remain
in the original elements. A packet is not an isotropic flash or a mean shower.
"""
from dataclasses import dataclass
from dataclasses import replace
from copy import copy
from time import perf_counter

import numpy as np

C_M_PER_NS = .299792458


@dataclass(frozen=True)
class SourcePacket:
    """Indices of source elements forming one packet, with its family and kind."""
    kind: str
    indices: np.ndarray
    family_uid: int | None = None
    root_energy_GeV: float | None = None

    def source(self, elements):
        """The packet as ``SpectralLightElements`` (elements in source order)."""
        indices = np.sort(self.indices)
        names = ("start_m", "direction", "length_m", "coefficient0", "coefficient2",
                 "beta", "start_ns", "end_ns", "row_index", "uid")
        return replace(elements, **{name: getattr(elements, name)[indices] for name in names})


def partition_muon(elements, ancestry, *, primary_uid=0, shower_energy_GeV=.1,
                   soft_cell_m=5., prompt_delay_ns=10., late_cell_ns=20.):
    """Partition using the full neutral-inclusive ancestry table.

    ``ancestry`` has particle columns uid, family_uid, pdgid, energy_GeV,
    start_m, start_ns, and an element_family array aligned with ``elements``.
    Delayed emission is kept in separate time packets, never discarded.
    Spatial/time grouping here changes no source values; transport accuracy
    depends on the solver subsequently chosen for each packet.
    """
    scales = (shower_energy_GeV, soft_cell_m, prompt_delay_ns, late_cell_ns)
    if not np.isfinite(scales).all() or min(scales) <= 0:
        raise ValueError("partition scales must be finite and positive")
    family = np.asarray(ancestry["element_family"], np.int64)
    if family.shape != (len(elements),):
        raise ValueError("element_family must align with the complete source")
    primary = np.asarray(elements.uid) == primary_uid
    if not np.any(primary):
        raise ValueError("the selected primary does not emit in this source")
    lookup = {int(uid): i for i, uid in enumerate(ancestry["uid"])}
    packets = [SourcePacket("primary", np.flatnonzero(primary), int(primary_uid))]
    first = int(np.flatnonzero(primary)[0])
    if elements.coefficient0[primary].sum() <= 0:
        raise ValueError("the selected primary must have positive light yield")
    axis = np.average(elements.direction[primary], axis=0, weights=elements.coefficient0[primary])
    if np.linalg.norm(axis) < 1e-12:
        raise ValueError("the selected primary has no well-defined mean direction")
    axis /= np.linalg.norm(axis)
    midpoints = elements.midpoints_m
    z = (midpoints - elements.start_m[first]) @ axis
    mid_time = .5 * (elements.start_ns + elements.end_ns)
    creators = np.asarray(ancestry.get("creator", np.full(len(lookup), "")))
    soft = []
    secondary = np.flatnonzero(~primary)
    order = secondary[np.argsort(family[secondary], kind="stable")]
    families, starts = np.unique(family[order], return_index=True)
    ends = np.r_[starts[1:], len(order)]
    for fid, lo, hi in zip(families, starts, ends):
        indices = order[lo:hi]
        if fid < 0 or int(fid) not in lookup:
            packets.append(SourcePacket("unknown", indices, int(fid)))
            continue
        row = lookup[int(fid)]
        energy = float(ancestry["energy_GeV"][row])
        pdg = int(ancestry["pdgid"][row])
        creator = str(creators[row])
        distance = np.linalg.norm(midpoints[indices] - ancestry["start_m"][row], axis=1)
        delay = mid_time[indices] - float(ancestry["start_ns"][row]) - distance / C_M_PER_NS
        late = delay > prompt_delay_ns
        for tick in np.unique(np.floor(delay[late] / late_cell_ns).astype(np.int64)):
            take = indices[late][np.floor(delay[late] / late_cell_ns).astype(np.int64) == tick]
            packets.append(SourcePacket("late", take, int(fid), energy))
        indices = indices[~late]
        if not len(indices):
            continue
        is_em = abs(pdg) == 11 or pdg == 22
        if "nucl" in creator.lower():
            is_em = False
        if is_em and energy >= shower_energy_GeV:
            packets.append(SourcePacket("shower", indices, int(fid), energy))
        elif is_em:
            soft.extend(indices.tolist())
        else:
            packets.append(SourcePacket("general", indices, int(fid), energy))
    soft = np.asarray(soft, np.int64)
    if len(soft):
        # Keep time-position correlation: delayed primary decays must not
        # merge into the front of the moving ionisation halo.
        front_time = elements.start_ns[first] + z[soft] / C_M_PER_NS
        keys = np.stack((np.floor(z[soft] / soft_cell_m),
                         np.floor((mid_time[soft] - front_time) / late_cell_ns)), axis=1).astype(np.int64)
        _, inverse = np.unique(keys, axis=0, return_inverse=True)
        for group in range(int(inverse.max()) + 1):
            packets.append(SourcePacket("soft", soft[inverse == group]))
    flattened = np.concatenate([p.indices for p in packets])
    if not np.array_equal(np.sort(flattened), np.arange(len(elements))):
        raise ValueError("packet partition lost or duplicated source elements")
    return packets


def order1_packets(kernel, elements, packets, *, time_origin_ns, radius_m=400., config=None):
    """Sum local detailed packet responses on a common clock, without OM cuts.

    The distance cut is experimental and must be checked by refinement. It is
    applied to the complete support sphere, never just to a root or centroid.
    Nuclear/late records retain their original elements. This reference
    implementation does not substitute a fitted mean-shower angular template.
    """
    from ..prompt import PromptConfig
    from .muon_integral import order1_integrals
    config = config or PromptConfig(element_method="grouped", screen=False,
                                    batch_charge_tolerance_pe=1e-6)
    if not np.isfinite(radius_m) or radius_m <= 0:
        raise ValueError("radius_m must be finite and positive")
    origins = np.asarray(time_origin_ns, float)
    detector = kernel.detector
    positions = detector.positions_m
    q = np.zeros(len(positions))
    bins = np.zeros((len(positions), len(kernel.config.relative_time_edges_ns)-1))
    start = perf_counter()
    details = []
    for packet in packets:
        if packet.kind == "primary":
            continue
        part = packet.source(elements)
        ends = part.start_m + part.length_m[:, None] * part.direction
        centre = .5 * (np.minimum(part.start_m.min(0), ends.min(0))
                        + np.maximum(part.start_m.max(0), ends.max(0)))
        reach = max(np.linalg.norm(part.start_m-centre, axis=1).max(),
                    np.linalg.norm(ends-centre, axis=1).max())
        work = np.flatnonzero(np.linalg.norm(positions-centre, axis=1) - reach <= radius_m)
        if not len(work):
            details.append(dict(kind=packet.kind, family_uid=packet.family_uid,
                                elements=len(part), receivers=0, seconds=0.))
            continue
        local = copy(kernel)
        local.detector = detector.subset(work)
        begin = perf_counter()
        # A single output sum has one omission budget, not one per packet.
        settings = replace(config, batch_charge_tolerance_pe=config.batch_charge_tolerance_pe/max(len(packets),1))
        qp, bp, _ = order1_integrals(local, part, time_origin_ns=origins[work], config=settings)
        q[work] += qp
        bins[work] += bp
        details.append(dict(kind=packet.kind, family_uid=packet.family_uid,
                            elements=len(part), receivers=len(work), seconds=perf_counter()-begin))
    return q, bins, dict(elapsed_seconds=perf_counter()-start, packets=details,
                         radius_m=radius_m, common_clock=True, global_threshold_applied=False)
