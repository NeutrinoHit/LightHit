"""Experimental primary-track integrals plus local secondary quasi-showers.

Ballistic light always uses the original charged steps. The existing global
coarse screen selects fine work from the SUM of all collision components.
The fine first-order evaluator uses fitted primary segments and detailed local
secondary packets, retaining the actual spectrum, angles and emission times.
An optional >=2 family partition uses the existing shower basis locally.
"""
from copy import copy
from dataclasses import asdict, replace
from time import perf_counter

import numpy as np

from ..muon.muon_integral import fit_track_integrals
from ..muon.muon_packets import SourcePacket, partition_muon


class MuonFineStrategy:
    name = "primary-integrals-and-local-quasi-showers"

    def __init__(self, source, packets, *, phase_indices, primary_uid=0, track_max_length_m=1., radius_m=400., azimuthal_degree=4, shower_azimuthal_degree=None, primary_polyline=False):
        if source.cone_model != "spectral":
            raise ValueError("the muon component prototype requires spectral cones")
        self.primary_uid = int(primary_uid)
        self.radius_m = float(radius_m)
        if not np.isfinite(self.radius_m) or self.radius_m <= 0:
            raise ValueError("prompt packet radius must be finite and positive")
        self.azimuthal_degree = int(azimuthal_degree)
        if (shower_azimuthal_degree is not None and
                (not isinstance(shower_azimuthal_degree, int)
                 or not 0 <= shower_azimuthal_degree <= self.azimuthal_degree)):
            raise ValueError("shower_azimuthal_degree must lie in the configured angular band")
        self.shower_azimuthal_degree = shower_azimuthal_degree
        self.source = source
        self.packets = packets
        self.labels = np.empty(len(source), np.int64)
        for index, packet in enumerate(packets):
            self.labels[packet.indices] = index
        self.sort = np.argsort(source.row_index)
        self.rows = source.row_index[self.sort]
        if np.any(np.diff(self.rows) <= 0):
            raise ValueError("source row indices must be unique")
        start = perf_counter()
        if primary_polyline:
            from ..muon.primary_polyline import fit_primary_polyline
            from ..muon.muon_integral import TrackIntegralFit
            fit = fit_primary_polyline(source.subset(source.uid == primary_uid), phase_indices=phase_indices)
            errors = fit.diagnostics['max_errors']
            self.fit = TrackIntegralFit(fit.elements, fit.diagnostics['original_elements'], fit.ranges,
                errors['position_m'], errors['time_ns'], errors['direction_rms_rad'])
        else:
            self.fit = fit_track_integrals(source.subset(source.uid == primary_uid),
                                          max_length_m=track_max_length_m, phase_indices=phase_indices)
        self.primary_polyline = bool(primary_polyline)
        self.fit_seconds = perf_counter() - start
        self.calls = []

    def _labels(self, elements):
        indices = np.searchsorted(self.rows, elements.row_index)
        if np.any(indices >= len(self.rows)) or not np.array_equal(self.rows[indices], elements.row_index):
            raise ValueError("fine evaluator received a different source")
        return self.labels[self.sort[indices]]

    def __call__(self, mode, kernel, source, elements, segments, positions, looks, areas,
                 origins, work, tables, coef, ipow, jpow, degree, g, edges, config,
                 order0_charge, *, coarse_only=False, selection_extra_charge=None):
        from .. import prompt
        if coarse_only or mode != "grouped":
            return prompt._order1(mode, kernel, source, elements, segments, positions, looks, areas,
                                  origins, work, tables, coef, ipow, jpow, degree, g, edges,
                                  config, order0_charge, coarse_only=coarse_only,
                                  selection_extra_charge=selection_extra_charge)
        phase = float(tables["phase_index"][0])
        label = self._labels(elements)
        q, bins = np.zeros(len(positions)), np.zeros((len(positions), len(edges)-1))
        timings = {"primary_integral_seconds": 0., "secondary_packets_seconds": 0.}
        omissions = np.zeros((len(positions), 2))
        element_total = elements.coefficient0.sum()
        # All components must share these origins and work set. No per-family
        # PE threshold is used: individually weak components may sum to a hit.
        ordered = np.argsort(label, kind="stable")
        unique, starts = np.unique(label[ordered], return_index=True)
        for number, lo, hi in zip(unique, starts, np.r_[starts[1:], len(ordered)]):
            selected = ordered[lo:hi]
            packet = self.packets[int(number)]
            primary = packet.kind == "primary"
            if primary:
                part = self.fit.elements
                valid = part.beta * phase > 1.
                if not np.any(valid):
                    continue
                if not np.all(valid):
                    part = part.subset(valid)
                kind = "segment"
            else:
                part = SourcePacket(packet.kind, selected).source(elements)
                kind = "grouped"
            part = replace(part, cone_model="frozen", reference_phase_index=phase)
            target = work
            if not primary:
                ends = part.start_m + part.length_m[:, None] * part.direction
                centre = .5 * (np.minimum(part.start_m.min(0), ends.min(0))
                                + np.maximum(part.start_m.max(0), ends.max(0)))
                reach = max(np.linalg.norm(part.start_m-centre, axis=1).max(),
                            np.linalg.norm(ends-centre, axis=1).max())
                target = work[np.linalg.norm(positions[work]-centre, axis=1)-reach <= self.radius_m]
            if not len(target):
                continue
            settings = replace(config, screen=False,
                               batch_charge_tolerance_pe=config.batch_charge_tolerance_pe
                               * float(elements.coefficient0[selected].sum()/element_total))
            start = perf_counter()
            qp, bp, meta = prompt._order1(
                kind, kernel, part, part, prompt._element_segments(part), positions, looks,
                areas, origins, target, tables, coef, ipow, jpow, degree, g, edges, settings,
                order0_charge)
            if meta.get("order1_segments_too_close", 0):
                raise ValueError("fitted primary approaches an OM closer than min_distance_m")
            q += qp
            bins += bp
            omissions += meta.get("order1_batch_omission", 0.)
            timings["primary_integral_seconds" if primary else "secondary_packets_seconds"] += perf_counter()-start
        level = np.zeros(len(positions), np.int8)
        level[work] = 2
        self.calls.append({**timings, "receivers": len(work), "wavelength_nm": float(tables["wavelength_nm"][0])})
        return q, bins, {"timings": timings, "order1_level": level,
                         "order1_batch_omission": omissions,
                         "order1_components": self.name}

    def ge2_partition(self, elements, basis):
        label = self._labels(elements)
        order = np.argsort(label, kind="stable")
        unique, start = np.unique(label[order], return_index=True)
        for number, lo, hi in zip(unique, start, np.r_[start[1:], len(order)]):
            packet = self.packets[int(number)]
            part = SourcePacket(packet.kind, order[lo:hi]).source(elements)
            primary = packet.kind == "primary"
            angular = (0 if primary else self.shower_azimuthal_degree
                       if packet.kind == "shower" and self.shower_azimuthal_degree is not None
                       else self.azimuthal_degree)
            yield (f"{packet.kind}-{number}", part, angular,
                   basis.track_cell_m if primary else basis.secondary_cell_m,
                   basis.track_transverse if primary else basis.transverse)


def transport_muon_components(kernel, source, ancestry, *, primary_uid=0,
                              frequency_nodes=33, prompt_config=None, basis=None,
                              shower_energy_GeV=.1, soft_cell_m=5.,
                              track_max_length_m=1., prompt_radius_m=400.,
                              partition_ge2=False, track_stencil_center="axis",
                              shower_azimuthal_degree=None, primary_polyline=False):
    """Explicit experimental route; the default transport is unchanged.

    A radial-cutoff convergence test is required for `prompt_radius_m`.
    The source partition retains every true G4 element, including late and
    non-EM descendants. No averaged angular template or isotropic halo is used.
    """
    from .. import prompt
    from ..shower_basis import ShowerBasis, transport_hybrid_basis
    elements = kernel._source_elements(source)
    packets = partition_muon(elements, ancestry, primary_uid=primary_uid,
                             shower_energy_GeV=shower_energy_GeV, soft_cell_m=soft_cell_m)
    strategy = MuonFineStrategy(elements, packets, primary_uid=primary_uid,
                                track_max_length_m=track_max_length_m, radius_m=prompt_radius_m,
                                phase_indices=kernel.medium.sample(kernel.wavelength.wavelength_nm)["phase_index"],
                                shower_azimuthal_degree=shower_azimuthal_degree,
                                primary_polyline=primary_polyline,
                                azimuthal_degree=kernel.config.azimuthal_degree)
    local = copy(kernel)
    def evaluate_prompt(source, config=None, **kw):
        return prompt.transport_prompt(local, source, config, _fine_order1=strategy, **kw)
    local.transport_prompt = evaluate_prompt
    settings = prompt_config or prompt.PromptConfig(screen_cell_m=5., screen_pixel_face=4,
                  screen_ring_nodes=12, screen_margin=6., batch_charge_tolerance_pe=1e-6)
    if settings.element_method not in ("auto", "grouped"):
        raise ValueError("muon components require prompt element_method='auto' or 'grouped'")
    # Auto normally sends tiny sources to segment mode. The explicit component
    # route must still use its own primary/secondary policy for those sources.
    settings = replace(settings, element_method="grouped")
    basis = basis or ShowerBasis(track_uid=primary_uid, spectral_fusion=True, apply_tolerance_pe=1e-6)
    response = transport_hybrid_basis(local, source, basis, frequency_nodes=frequency_nodes,
                   prompt_config=settings, source_partition=strategy.ge2_partition if partition_ge2 else None,
                   track_stencil_center=track_stencil_center)
    response.metadata["muon_components"] = dict(
        experimental=True, primary_uid=primary_uid, original_primary_elements=strategy.fit.original_elements,
        primary_polyline=strategy.primary_polyline,
        primary_integral_pieces=len(strategy.fit.elements), fit_seconds=strategy.fit_seconds,
        fit_max_position_error_m=strategy.fit.maximum_position_error_m,
        fit_max_time_error_ns=strategy.fit.maximum_time_error_ns,
        packet_counts={kind: sum(p.kind == kind for p in packets) for kind in {p.kind for p in packets}},
        prompt_radius_m=prompt_radius_m, partition_ge2=partition_ge2,
        shower_azimuthal_degree=shower_azimuthal_degree,
        source_approximation="fitted primary only for order1; detailed secondary elements in local packets",
        first_order_calls=strategy.calls)
    return response
