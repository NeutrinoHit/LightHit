"""Events with several light sources: one detector response, one OM selection.

Light is additive, so an event made of several sources (a muon and the hadronic
shower at its vertex, two showers, a muon bundle) is the sum of their
responses. Two things must not be done per source:

* **Time origin.** Bins of different sources can be added only on one clock.
  Every OM gets one origin, a lower bound on the earliest possible arrival
  from any charged step of the event (fastest group speed); each route
  computes its bins relative to it.
* **OM acceptance.** An OM is active when the *sum* of all sources reaches the
  threshold, not when one source does. Each route selects the OMs it computes
  in detail with the light of the other sources added to its threshold test
  (``selection_extra_charge``), and every OM whose summed charge can still
  reach the threshold is completed in every source before the decision.

:class:`EventTransport` runs this for a list of :class:`SourceComponent`
(``kind="shower"`` through :mod:`lighthit.shower`, ``kind="muon"`` through
:mod:`lighthit.muon`). Showers go first (cheap; their charges are then known
to the muon selection). Afterwards, per source, the OMs with
``q0 + safety * (|q1| + |q2|) >= threshold`` summed over sources (exact or
finished charges enter without the factor) that this source did not compute in
detail are recomputed by it on just those OMs, until no such OM remains. The
final ``active`` is the summed detailed charge above threshold, and every
active OM carries the bins of every source. A single component is passed
through unchanged. A calibration flash (:class:`lighthit.IsotropicFlash`) is
transported alone with its own monochromatic kernel.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter, process_time

import numpy as np

__all__ = ["SourceComponent", "EventTransport", "common_time_origins", "laser_config", "laser_kernel"]


@dataclass(frozen=True)
class SourceComponent:
    """One light source of an event and the route that transports it.

    ``source`` are its charged steps (``SpectralLightElements`` with spectral
    cones). A muon needs its family table ``ancestry`` (``element_family``
    aligned with ``source``) and the muon's ``primary_uid``. ``label`` is free
    text recorded in the metadata.
    """
    kind: str
    source: object
    ancestry: dict | None = None
    primary_uid: int | None = None
    label: str = ""

    def __post_init__(self):
        """Check the kind and the muon inputs."""
        if self.kind not in ("shower", "muon"):
            raise ValueError("component kind must be 'shower' or 'muon'")
        if self.kind == "muon" and (self.ancestry is None or self.primary_uid is None):
            raise ValueError("a muon component needs ancestry and primary_uid")
        if getattr(self.source, "cone_model", None) != "spectral":
            raise ValueError("components require spectral cones (cone_model='spectral')")


def common_time_origins(kernel, sources):
    """Per-OM lower bound on the earliest arrival from any step of ``sources``.

    The emission time of a step is at least its start time and the path at
    least the distance to the step, travelled at the fastest group speed of the
    kernel's wavelength quadrature (as each route does for one source).
    """
    from ._shower_basis import earliest_element_arrival
    group = float(np.min(kernel.medium.sample(kernel.wavelength.wavelength_nm)["group_index"]))
    positions = np.ascontiguousarray(kernel.detector.positions_m, float)
    origins = np.full(len(positions), np.inf)
    for source in sources:
        origins = np.minimum(origins, earliest_element_arrival(
            positions, np.ascontiguousarray(source.start_m), np.ascontiguousarray(source.direction),
            np.ascontiguousarray(source.length_m), np.ascontiguousarray(source.start_ns), group))
    return origins


@dataclass
class _Part:
    """Accumulated result of one component."""
    component: SourceComponent
    charge: np.ndarray
    bins: np.ndarray
    spectrum: np.ndarray
    detailed: np.ndarray
    runs: list = field(default_factory=list)


class EventTransport:
    """All scattering orders for an event given as a list of sources.

    ``kernel`` is the event kernel (spectral cache with the 65/321-frequency
    grids). The muon and shower routes are created on first use from the
    prepared tables next to its cache (``muon_data``/``shower_data`` override).
    ``laser_kernel`` is the monochromatic kernel for calibration flashes.
    ``safety`` inflates scattered charges that a route only estimated when
    deciding which OMs must be completed in every source.
    """

    def __init__(self, kernel, *, muon_data=None, shower_data=None, laser_kernel=None,
                 threads=None, safety=3.0, max_rounds=4):
        """Remember the kernels; the routes open their tables lazily."""
        if not np.isfinite(safety) or safety < 1 or int(max_rounds) < 1:
            raise ValueError("safety must be >= 1 and max_rounds positive")
        self.kernel = kernel
        self.laser_kernel = laser_kernel
        self.muon_data, self.shower_data = muon_data, shower_data
        self.threads, self.safety, self.max_rounds = threads, float(safety), int(max_rounds)
        self._muon = self._shower = None
        self.route_fallbacks = {}

    def muon_route(self):
        """The :class:`lighthit.muon.MuonTransport` of this detector."""
        if self._muon is None:
            from .muon import MuonTransport
            self._muon = MuonTransport(self.kernel, self.muon_data, threads=self.threads)
        return self._muon

    def shower_route(self):
        """The :class:`lighthit.shower.ShowerTransport` of this detector."""
        if self._shower is None:
            from .shower import ShowerTransport
            self._shower = ShowerTransport(self.kernel, self.shower_data, threads=self.threads)
        return self._shower

    def _run(self, component, **options):
        """One route call for one component (full detector or ``receivers``).

        A muon whose track the muon route refuses (a primary-track piece whose
        line passes within its validated 3 m inner radius of an OM, typical of
        the scattered end of a stopping muon) is transported by the shower
        route instead; the substitution is recorded in ``route_fallbacks``.
        """
        key = id(component)
        if component.kind == "shower" or key in self.route_fallbacks:
            return self.shower_route()(component.source, **options)
        try:
            return self.muon_route()(component.source, component.ancestry,
                                     primary_uid=int(component.primary_uid), **options)
        except ValueError as error:
            if "unvalidated inner" not in str(error):
                raise
            self.route_fallbacks[key] = dict(label=component.label, reason=str(error),
                                             route="lighthit.shower")
            return self.shower_route()(component.source, **options)

    def __call__(self, components, *, progress=None):
        """Response (``HybridTransportResponse``) of the event's sources.

        ``components`` is a list of :class:`SourceComponent`, or a single
        :class:`lighthit.IsotropicFlash` (transported by ``laser_kernel``).
        """
        from .sources import IsotropicFlash
        if isinstance(components, IsotropicFlash):
            return self.flash(components)
        components = list(components)
        if not components:
            raise ValueError("an event needs at least one source component")
        if any(isinstance(item, IsotropicFlash) for item in components):
            raise ValueError("a calibration flash is transported alone (different kernel)")
        self.route_fallbacks = {}
        if len(components) == 1:
            response = self._run(components[0])
            response.metadata["event_components"] = [self._describe(components[0])]
            response.metadata["route_fallbacks"] = list(self.route_fallbacks.values())
            return response
        return self._compose(components, progress)

    def flash(self, flash):
        """Isotropic calibration flash through the monochromatic laser kernel."""
        if self.laser_kernel is None:
            raise ValueError("a calibration flash needs EventTransport(laser_kernel=...)")
        response = self.laser_kernel.transport(flash)
        response.metadata["route"] = "lighthit.transport (isotropic flash)"
        return response

    @staticmethod
    def _describe(component):
        """Metadata record of a component."""
        return dict(kind=component.kind, label=component.label, elements=len(component.source),
                    primary_uid=component.primary_uid)

    def _compose(self, components, progress):
        """Several sources on one clock with one OM selection (module docstring)."""
        from .transport import HybridTransportResponse
        began, cpu = perf_counter(), process_time()
        kernel = self.shower_route().kernel if any(c.kind == "shower" for c in components) \
            else self.muon_route().kernel
        n = len(kernel.detector.positions_m)
        edges = np.asarray(kernel.config.relative_time_edges_ns, float)
        threshold = float(kernel.config.threshold_pe)
        origins = common_time_origins(kernel, [c.source for c in components])
        order = sorted(range(len(components)), key=lambda i: components[i].kind != "shower")
        parts = [None] * len(components)
        known = np.zeros(n)
        omega = None
        for index in order:
            component = components[index]
            started = process_time()
            response = self._run(component, time_origin_ns=origins, selection_extra_charge=known)
            if omega is None:
                omega = np.asarray(response.omega_per_ns)
            elif not np.allclose(response.omega_per_ns, omega, rtol=0, atol=1e-12):
                raise ValueError("source routes use different RTE frequency grids")
            if not np.allclose(response.time_origin_ns, origins, rtol=0, atol=1e-9):
                raise ValueError(f"{component.kind} route did not use the event clock")
            detailed = np.asarray(response.active, bool).copy()
            mask = response.metadata.get("fine_charge_mask")
            if mask is not None:
                detailed |= np.asarray(mask, bool)
            # A source that does not reach an OM (outside its routes' radial
            # range) contributes exactly nothing there: nothing to complete.
            detailed |= np.all(np.asarray(response.charge_components_pe) == 0.0, axis=1)
            part = _Part(component, np.asarray(response.charge_components_pe, float).copy(),
                         np.where(detailed[:, None, None], response.components_pe, 0.0),
                         np.asarray(response.spectrum_ge2_pe).copy(), detailed)
            part.runs.append(dict(receivers="all", cpu_seconds=process_time() - started,
                                  route_active=int(np.count_nonzero(response.active)),
                                  route_metadata_keys=sorted(response.metadata)))
            parts[index] = part
            known += part.charge.sum(axis=1)
            if progress:
                progress("component", self._describe(component))
        rounds = []
        for round_number in range(self.max_rounds):
            upper = np.zeros(n)
            for part in parts:
                q = part.charge
                inflated = np.maximum(q[:, 0], 0.0) + self.safety * np.abs(q[:, 1:]).sum(axis=1)
                upper += np.where(part.detailed, q.sum(axis=1), inflated)
            wanted = upper >= threshold
            done = True
            record = dict(round=round_number, candidates=int(wanted.sum()), recomputed={})
            for part in parts:
                need = np.flatnonzero(wanted & ~part.detailed & (np.abs(part.charge).sum(axis=1) > 0))
                if not len(need):
                    continue
                done = False
                started = process_time()
                local = self._run(part.component, time_origin_ns=origins[need],
                                  selection_extra_charge=None, receivers=need)
                part.charge[need] = local.charge_components_pe
                part.bins[need] = local.components_pe
                part.spectrum[:, need] = local.spectrum_ge2_pe
                part.detailed[need] = True
                part.runs.append(dict(receivers=need.tolist(), cpu_seconds=process_time() - started))
                record["recomputed"][part.component.label or part.component.kind] = len(need)
            rounds.append(record)
            if done:
                break
        else:
            raise RuntimeError("source composition did not converge; raise max_rounds")
        charge = sum(part.charge for part in parts)
        detailed = np.logical_and.reduce([part.detailed for part in parts])
        active = (charge.sum(axis=1) >= threshold) & detailed
        if threshold == 0:
            active = detailed & (np.abs(charge).sum(axis=1) > 0)
        missing = (charge.sum(axis=1) >= threshold) & ~detailed
        if np.any(missing):
            raise RuntimeError("an OM above threshold lacks the detailed charge of a source")
        components_pe = np.zeros((n, len(edges) - 1, 3))
        components_pe[active] = sum(part.bins for part in parts)[active]
        spectrum = sum(part.spectrum for part in parts)
        metadata = dict(
            route="lighthit.event", elapsed_seconds=perf_counter() - began,
            cpu_seconds=process_time() - cpu, threshold_pe=threshold,
            time_origin="earliest arrival from any step of the event, fastest group speed",
            selection="summed charge of all sources; OMs completed in every source while "
                      "q0 + safety*(|q1|+|q2|) of estimated sources can reach the threshold",
            safety=self.safety, completion_rounds=rounds, active_modules=int(active.sum()),
            event_components=[dict(self._describe(part.component), runs=part.runs,
                                   charge_pe=float(part.charge.sum()),
                                   detailed_modules=int(part.detailed.sum()))
                              for part in parts],
            inactive_charge_may_be_estimated=bool(np.any(~detailed)),
            route_fallbacks=list(self.route_fallbacks.values()),
            computed_orders=[0, 1, ">=2"])
        return HybridTransportResponse(kernel.detector, omega, edges, spectrum, components_pe,
                                       charge, origins, active, "event", metadata)


def laser_config(detector, *, cache_directory, bin_ns=5.0, cache_policy="require",
                 omega_per_ns=None, max_radius_m=None):
    """Kernel settings of a point calibration flash on the whole detector.

    One position-independent radial range is derived from the detector
    centroid (``point_source_radial_range``); ``max_radius_m`` extends it.
    The cache is built once per wavelength (``cache_policy="build"``) and then
    reused with ``"require"``.
    """
    from . import KernelConfig, describe_geometry, point_source_radial_range
    radial_range = point_source_radial_range(detector, describe_geometry(detector).centroid_m)
    if max_radius_m is not None:
        maximum = float(max_radius_m)
        if not np.isfinite(maximum) or maximum <= radial_range[0]:
            raise ValueError("max_radius_m must exceed the radial minimum")
        radial_range = (radial_range[0], maximum)
    return KernelConfig(
        **({} if omega_per_ns is None else {"omega_per_ns": omega_per_ns}),
        relative_time_edges_ns=np.arange(-60., 740. + bin_ns, bin_ns),
        radial_range_m=radial_range, k_panel_per_m=min(0.04, 2 * np.pi / radial_range[1]),
        cache_directory=cache_directory, cache_policy=cache_policy)


def laser_kernel(medium, detector, *, cache_directory, **options):
    """``TransportKernel`` for :meth:`EventTransport.flash` (see :func:`laser_config`)."""
    from .transport import TransportKernel
    return TransportKernel(medium, detector,
                           laser_config(detector, cache_directory=cache_directory, **options))
