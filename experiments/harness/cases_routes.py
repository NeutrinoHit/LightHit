"""Default routes (the baseline of every comparison) and their references."""
from __future__ import annotations

from .cases import case
from . import references as R

SHOWERS = ("electron-100GeV", "pion-100GeV", "nue-cc")
MUONS = ("muon-1TeV", "muon-100GeV", "muon-300GeV", "muon-3000GeV", "muon-1TeV-seed43")
EVENTS = ("numu-like", "numu-cc", "nue-cc", "muon-stopping", "muon-near-om", "muon-100GeV")


# ---------------------------------------------------------------- baselines
def _shower(name):
    def run(ctx):
        from lighthit.shower import ShowerTransport
        route = ctx.cached("shower-route", lambda: ShowerTransport(ctx.kernel()))
        return route(ctx.shower_source(name))
    return run


def _muon(name):
    def run(ctx):
        import lighthit as lh
        route = ctx.cached("muon-route", lambda: lh.MuonTransport(ctx.kernel()))
        steps, ancestry, uid = ctx.muon_source(name)
        return route(steps, ancestry, primary_uid=uid)
    return run


def _event(name):
    def run(ctx):
        from lighthit.event import EventTransport
        route = ctx.cached("event-route", lambda: EventTransport(ctx.kernel()))
        return route(ctx.components(name))
    return run


# The references are computed on the default route's OMs and clock, so the default route is
# compared with them after they exist (report.py recomputes the metrics from the saved responses).
for _name in SHOWERS:
    case("default-shower-route", _name, reference=f"reference-shower/{_name}",
         notes="ShowerTransport with defaults")(_shower(_name))
for _name in MUONS:
    case("default-muon-route", _name,
         reference=f"reference-muon/{_name}" if _name in ("muon-1TeV", "muon-300GeV") else None,
         notes="MuonTransport with defaults")(_muon(_name))
for _name in EVENTS:
    case("default-event-route", _name, notes="EventTransport with defaults")(_event(_name))


@case("default-laser-route", "laser-532nm", notes="EventTransport(laser_kernel=...) flash")
def _(ctx):
    from lighthit.event import EventTransport
    return EventTransport(ctx.kernel(), laser_kernel=ctx.laser_kernel())(_flash())


@case("kernel-transport-prompt", "electron-100GeV", reference="reference-shower/electron-100GeV",
      baseline="default-shower-route/electron-100GeV",
      notes="TransportKernel.transport_prompt with the default PromptConfig (orders 0 and 1 only)")
def _(ctx):
    return ctx.kernel().transport_prompt(ctx.electron())


def _flash():
    from .inputs import laser_flash
    return laser_flash()


# ---------------------------------------------------------------- references
def _prompt_reference(default_case, source_of, **prompt):
    """Orders 0 and 1 on the original steps with the segment method (no grouping,
    no screen, per band, continuous-spectrum order 0) on the default route's OMs."""
    def run(ctx):
        import lighthit as lh
        kernel = ctx.kernel()
        indices, origins, _ = R.om_subset(default_case)
        sub = R.subset_kernel(kernel, indices, threshold_pe=0.0)
        config = lh.PromptConfig(**{**dict(element_method="segment", ballistic_method="interval",
                                           screen=False, fuse_bands=False), **prompt})
        response = sub.transport_prompt(source_of(ctx), config, compute_bins_for_all=True,
                                        time_origin_ns=origins[indices])
        return R.embed([(response, (0, 1))], indices, len(origins), origins,
                       metadata=dict(reference="prompt segment", prompt=prompt))
    return run


def _ge2_per_band_reference(default_case, source_of):
    """Order >=2 of transport_hybrid's per-band route (elements compiled per band on the
    directional tables, no shower basis) on the default route's OMs; orders 0 and 1 from
    the segment prompt reference of the same input."""
    def run(ctx):
        from .record import RESULTS, load_response
        kernel = ctx.kernel()
        indices, origins, _ = R.om_subset(default_case)
        sub = R.subset_kernel(kernel, indices, threshold_pe=0.0)
        response = sub.transport_hybrid(source_of(ctx))
        return R.embed([(response, (2,))], indices, len(origins), response_origin(response, indices, origins),
                       metadata=dict(reference="transport_hybrid per-band >=2"))
    return run


def response_origin(response, indices, origins):
    """Full-length origins with the subset response's own clock on ``indices``."""
    import numpy as np
    out = np.array(origins, float)
    out[indices] = np.asarray(response.time_origin_ns)
    return out


for _name in SHOWERS:
    case("reference-original-steps", _name, heavy=True,
         notes="orders 0 and 1: PromptConfig(element_method='segment', ballistic_method='interval', "
               "screen=False, fuse_bands=False) on the OMs of the default shower route",
         warmup=False)(_prompt_reference(f"default-shower-route/{_name}",
                                         lambda ctx, n=_name: ctx.shower_source(n)))
    case("reference-per-band-ge2", _name, heavy=True, warmup=False,
         notes="order >=2: TransportKernel.transport_hybrid (per-band, elements on directional "
               "tables, 65 frequencies) on the OMs of the default shower route")(
        _ge2_per_band_reference(f"default-shower-route/{_name}",
                                lambda ctx, n=_name: ctx.shower_source(n)))


def _finer_basis_reference(default_case, source_of):
    """Order >=2 of the shower route with a finer basis on all nine bands (0.125 m slices,
    0.01 ns phase lattice, band_rank=9), on the default route's OMs and clock."""
    def run(ctx):
        from dataclasses import replace
        from lighthit.shower import ShowerConfig, ShowerTransport
        indices, origins, _ = R.om_subset(default_case)
        base = ShowerConfig()
        config = replace(base, basis=replace(base.basis, cell_m=0.125, time_step_ns=0.01), band_rank=9)
        route = ShowerTransport(ctx.kernel(), config=config)
        response = route(source_of(ctx), receivers=indices, time_origin_ns=origins[indices])
        return R.embed([(response, (2,))], indices, len(origins), origins,
                       metadata=dict(reference="shower route, finer basis, nine bands"))
    return run


for _name in SHOWERS:
    case("reference-finer-basis-ge2", _name, warmup=False,
         notes="order >=2: ShowerTransport with ShowerBasis(cell_m=0.125, time_step_ns=0.01) and "
               "band_rank=9, receivers = the default route's OMs, its clock")(
        _finer_basis_reference(f"default-shower-route/{_name}", lambda ctx, n=_name: ctx.shower_source(n)))

#: order >=2 reference per input: per-band cloud-in-cell where transport_hybrid accepts the
#: source; the pion has delayed steps and transport_hybrid refuses it (prompt and RTE origins differ)
GE2_REFERENCE = {"electron-100GeV": "reference-per-band-ge2", "pion-100GeV": "reference-finer-basis-ge2",
                 "nue-cc": "reference-finer-basis-ge2"}
# Inputs whose order >=2 reference does not fit the measuring machine: the combined
# reference then holds orders 0 and 1 only and order >=2 is not compared.
GE2_REFERENCE_UNAVAILABLE = {
    "muon-1TeV": "the per-band cloud-in-cell order >=2 of the 1 TeV muon (all original steps) was killed "
                 "for memory on the 24 GB machine, both on all 41 OMs and in chunks of 8 OMs: its memory "
                 "is set by the source lattice, not by the OMs (the 300 GeV muon peaks at 13.7 GB)"}


def _combined_reference(name):
    """reference-original-steps (orders 0, 1) + the order >=2 reference of GE2_REFERENCE."""
    def run(ctx):
        import numpy as np
        from .record import RESULTS, load_response
        low = load_response(RESULTS / "reference-original-steps" / f"{name}.npz")
        if name in GE2_REFERENCE_UNAVAILABLE:
            charge = low["charge_components_pe"].copy()
            bins = low["components_pe"].astype(float).copy()
            charge[:, 2] = 0.0
            bins[:, :, 2] = 0.0
            covered = low["covered"]
            return R.Embedded(charge, bins, covered & (charge.sum(1) >= 0.01), low["time_origin_ns"],
                              dict(reference="original-step orders 0+1 only; order >=2: "
                                   + GE2_REFERENCE_UNAVAILABLE[name]), covered, computed_orders=(0, 1))
        high = load_response(RESULTS / GE2_REFERENCE.get(name, "reference-per-band-ge2") / f"{name}.npz")
        if not np.allclose(low["time_origin_ns"], high["time_origin_ns"], atol=1e-6):
            raise ValueError("the two references use different per-OM clocks")
        charge = low["charge_components_pe"].copy()
        bins = low["components_pe"].astype(float).copy()
        charge[:, 2] = high["charge_components_pe"][:, 2]
        bins[:, :, 2] = high["components_pe"][:, :, 2]
        covered = low.get("covered")
        if covered is None:
            covered = (low["charge_components_pe"].sum(1) > 0) | (high["charge_components_pe"].sum(1) > 0)
        return R.Embedded(charge, bins, covered & (charge.sum(1) >= 0.01), low["time_origin_ns"],
                          dict(reference="original-step orders 0+1, order >=2 from "
                               + GE2_REFERENCE.get(name, "reference-per-band-ge2")), covered)
    return run


for _name in SHOWERS:
    case("reference-shower", _name, warmup=False,
         notes="combined reference: reference-original-steps + the order >=2 reference of GE2_REFERENCE")(
        _combined_reference(_name))


# ---------------------------------------------------------------- kernel.transport (full tables)
def _linked_cache(name):
    """A cache directory of symbolic links to the bundle's RTE tables, so that a
    folded cache built by build_folded_directional() lands here, not in the bundle."""
    from .inputs import bundle_path
    from .record import RESULTS
    target = RESULTS / "work" / name
    target.mkdir(parents=True, exist_ok=True)
    for path in bundle_path("tables", "rte-cache").glob("directional-*.npz"):
        link = target / path.name
        if not link.exists():
            link.symlink_to(path)
    return target


def _full(cone, folded):
    def run(ctx):
        from dataclasses import replace
        import lighthit as lh
        directory = _linked_cache("rte-cache-linked-folded" if folded else "rte-cache-linked")
        kernel = ctx.cached("kernel-full", lambda: ctx.kernel(cache_directory=directory,
                                                               spectral_folded_cache=bool(folded)))
        if folded:
            ctx.cached("fold", lambda: kernel.build_folded_directional(progress=False))
        source = ctx.electron()
        if cone == "frozen":
            source = replace(source, cone_model="frozen")
        return kernel.transport(source)
    return run


for _cone, _folded in (("frozen", False), ("frozen", True), ("spectral", False)):
    case("kernel-transport-full", f"electron-100GeV-{_cone}" + ("-folded" if _folded else ""),
         reference="reference-shower/electron-100GeV", baseline="default-shower-route/electron-100GeV",
         notes=f"TransportKernel.transport on the bundle's 65-frequency directional tables, {_cone} cones"
               + (", wavelength-folded kernels (build_folded_directional)" if _folded else ""))(_full(_cone, _folded))
