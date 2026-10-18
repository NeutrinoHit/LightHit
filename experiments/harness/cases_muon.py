"""Muon-route alternatives and references.

Each alternative changes one group of settings of MuonTransport (JointMuonConfig,
the adaptive screen, or the primary options that MuonTransport fixes) and is
compared with the muon reference of the same input: orders 0 and 1 of all
original steps by the prompt solver without grouping shortcuts, order >=2 by
the per-band cloud-in-cell route, on the OMs of the default muon route.
"""
from __future__ import annotations

from dataclasses import replace

from .cases import case
from . import references as R

INPUTS = ("muon-1TeV", "muon-300GeV")
# references also for the 100 GeV muon (it reaches the muon route since the near-field fix)
REFERENCE_INPUTS = INPUTS + ("muon-100GeV",)


class _Muon:
    """MuonTransport with changed JointMuonConfig / screen / primary options."""

    def __init__(self, kernel, config=None, screen=None, primary=None):
        import lighthit as lh
        from lighthit.muon.adaptive_screen import AdaptiveScreenConfig
        from lighthit.muon.joint_muon import JointMuonConfig
        self.route = lh.MuonTransport(
            kernel, config=replace(JointMuonConfig(), **(config or {})),
            screen=replace(AdaptiveScreenConfig(scatter_margin=2.0), **(screen or {})))
        self.primary = primary or {}

    def __call__(self, steps, ancestry, primary_uid):
        if not self.primary:
            return self.route(steps, ancestry, primary_uid=primary_uid)
        from lighthit.muon import joint_muon
        original = joint_muon.transport_joint_muon
        changes = self.primary

        def patched(*args, primary_options=None, **kwargs):
            return original(*args, primary_options={**(primary_options or {}), **changes}, **kwargs)

        import lighthit.muon.api as api
        api.transport_joint_muon = patched
        try:
            return self.route(steps, ancestry, primary_uid=primary_uid)
        finally:
            api.transport_joint_muon = original


def _route(config=None, screen=None, primary=None):
    def make(name):
        def run(ctx):
            route = ctx.cached("route", lambda: _Muon(ctx.kernel(), config, screen, primary))
            steps, ancestry, uid = ctx.muon_source(name)
            return route(steps, ancestry, uid)
        return run
    return make


def _receivers(name):
    """Default MuonTransport with receivers= the OMs of the default run: every one
    computed in detail (no preview, no OM selection)."""
    def run(ctx):
        import lighthit as lh
        indices, origins, _ = R.om_subset(f"default-muon-route/{name}")
        route = ctx.cached("route", lambda: lh.MuonTransport(ctx.kernel()))
        steps, ancestry, uid = ctx.muon_source(name)
        response = route(steps, ancestry, primary_uid=uid, receivers=indices,
                         time_origin_ns=origins[indices])
        return R.embed([(response, (0, 1, 2))], indices, len(origins), origins)
    return run


def _hybrid_fallback(name):
    """NTSim's fallback for a muon event: ShowerBasis(track_uid=primary, spectral_fusion=True),
    33 frequencies, default PromptConfig."""
    def run(ctx):
        import lighthit as lh
        steps, _, uid = ctx.muon_source(name)
        return ctx.kernel().transport_hybrid(
            steps, frequency_nodes=33, ge2_basis=lh.ShowerBasis(track_uid=uid, spectral_fusion=True))
    return run


def _ntsim_fast(components="off", stencil="axis", polyline=False):
    """NTSim's expert hybrid muon run (ntsim/run/check_muon_fast.sh): muon-fast prompt
    screen, prompt batch and >=2 apply tolerances 1e-6, 33 frequencies, the axial
    track channel with spectral fusion, the prepared >=2 pack; MUON_COMPONENTS
    selects lighthit.experimental.muon_components ("prompt" or "all")."""
    def make(name):
        def run(ctx):
            import lighthit as lh
            kernel = ctx.cached("kernel-with-pack", lambda: _with_pack(ctx))
            steps, ancestry, uid = ctx.muon_source(name)
            prompt = lh.PromptConfig(screen=True, screen_cell_m=5.0, screen_pixel_face=4,
                                     screen_ring_nodes=12, screen_margin=6.0,
                                     batch_charge_tolerance_pe=1e-6)
            basis = lh.ShowerBasis(track_uid=uid, spectral_fusion=True, apply_tolerance_pe=1e-6)
            if components == "off":
                return kernel.transport_hybrid(steps, frequency_nodes=33, prompt_config=prompt,
                                               ge2_basis=basis)
            from lighthit.experimental.muon_components import transport_muon_components
            return transport_muon_components(
                kernel, steps, ancestry, primary_uid=uid, frequency_nodes=33, basis=basis,
                prompt_config=prompt, partition_ge2=components == "all",
                track_stencil_center=stencil, **({"primary_polyline": True} if polyline else {}))
        return run
    return make


def _with_pack(ctx):
    from .inputs import bundle_path
    kernel = ctx.kernel()
    kernel.ge2_kernel_store(str(bundle_path("tables", "ge2-pack")))
    return kernel


def _smart(name):
    def run(ctx):
        from lighthit.muon.adaptive_screen import transport_smart_muon
        from .record import RESULTS
        pack = RESULTS / "work" / "muon-segments-20261009" / "first-cache" / "pack65"
        if not (pack / "pack.json").is_file() and not any(pack.glob("*")):
            raise FileNotFoundError(f"{pack} missing: run scripts/shower_basis/probe_shower_first_cache.py")
        steps, ancestry, uid = ctx.muon_source(name)
        kernel = ctx.cached("kernel-with-pack", lambda: _with_pack(ctx))
        return transport_smart_muon(kernel, steps, ancestry, primary_uid=uid,
                                    first_pack_directory=str(pack))
    return run


VARIANTS = {
    "muon-first-band-fusion": {"per-band": _route(config=dict(first_fuse_bands=False))},
    "muon-first-packet-skip": {"no-skip": _route(config=dict(first_skip_epsilon=0.0))},
    "muon-ge2-pair-skip": {"no-skip": _route(config=dict(ge2_skip_epsilon=0.0))},
    "muon-ge2-operator": {"float64": _route(config=dict(ge2_blas_precision="float64")),
                          "scalar": _route(config=dict(ge2_operator="scalar", ge2_batch_points=0,
                                                       ge2_blas_precision="float64"))},
    "muon-ge2-axial-basis": {"basis": _route(config=dict(ge2_backend="basis", ge2_operator="scalar",
                                                         ge2_batch_points=0, ge2_blas_precision="float64",
                                                         ge2_skip_epsilon=0.0))},
    "muon-adaptive-screen": {"margin-4": _route(screen=dict(scatter_margin=4.0))},
    "muon-primary-direct-ge2-quadrature": {"fixed12": _route(primary=dict(direct_ge2_quadrature="fixed12")),
                                           "adaptive": _route(primary=dict(direct_ge2_quadrature="adaptive"))},
    "muon-primary-q1-charge-gauss": {"order-0": _route(primary=dict(q1_charge_gauss_order=0))},
    "muon-primary-beta-batch": {"off": _route(primary=dict(batch_beta_charge=False))},
    "muon-primary-beta-atlases": {"off": _route(primary=dict(first_beta_atlas=None, ge2_beta_atlas=None))},
}

for _idea, _variants in VARIANTS.items():
    for _variant, _make in _variants.items():
        for _name in INPUTS:
            case(_idea, f"{_name}-{_variant}", reference=f"reference-muon/{_name}",
                 baseline=f"default-muon-route/{_name}")(_make(_name))

for _name in INPUTS:
    case("muon-receivers-all-detailed", _name, reference=f"reference-muon/{_name}",
         baseline=f"default-muon-route/{_name}",
         notes="MuonTransport(receivers=OMs of the default run): no preview or OM selection")(
        _receivers(_name))
    case("hybrid-muon-fallback", _name, reference=f"reference-muon/{_name}",
         baseline=f"default-muon-route/{_name}")(_hybrid_fallback(_name))
    for _variant, _options in {"off": {}, "prompt": dict(components="prompt"),
                               "all": dict(components="all"),
                               "all-centroid": dict(components="all", stencil="centroid"),
                               "all-polyline": dict(components="all", polyline=True)}.items():
        case("ntsim-hybrid-muon-fast", f"{_name}-{_variant}", reference=f"reference-muon/{_name}",
             baseline=f"default-muon-route/{_name}",
             notes=f"check_muon_fast.sh MUON_COMPONENTS/stencil/polyline = {_variant}")(
            _ntsim_fast(**_options)(_name))
    case("smart-muon", _name, reference=f"reference-muon/{_name}",
         baseline=f"default-muon-route/{_name}",
         notes="lighthit.muon.adaptive_screen.transport_smart_muon (experimental.smart_muon)")(
        _smart(_name))


# ---------------------------------------------------------------- reference
def _muon_prompt_reference(name, **prompt):
    def run(ctx):
        import lighthit as lh
        indices, origins, _ = R.om_subset(f"default-muon-route/{name}")
        sub = R.subset_kernel(ctx.kernel(), indices, threshold_pe=0.0)
        steps, _, _ = ctx.muon_source(name)
        config = lh.PromptConfig(**{**dict(element_method="grouped", ballistic_method="interval",
                                           screen=False, fuse_bands=False), **prompt})
        response = sub.transport_prompt(steps, config, compute_bins_for_all=True,
                                        time_origin_ns=origins[indices])
        return R.embed([(response, (0, 1))], indices, len(origins), origins,
                       metadata=dict(reference="prompt, all original steps", prompt=prompt))
    return run


def _muon_ge2_reference(name):
    """transport_hybrid rejects a long muon at its radial check (centroid distance +- extent);
    the cloud-in-cell lattice of the per-band route is run through the basis driver instead
    (transport_hybrid_basis(representation='cells'), whose radial check is per point)."""
    def run(ctx):
        import lighthit as lh
        from lighthit.shower_basis import transport_hybrid_basis
        from .inputs import bundle_path
        from types import SimpleNamespace
        import numpy as np
        from .cases_routes import GE2_REFERENCE_UNAVAILABLE
        if name in GE2_REFERENCE_UNAVAILABLE:
            raise RuntimeError("not computed: " + GE2_REFERENCE_UNAVAILABLE[name])
        indices, origins, _ = R.om_subset(f"default-muon-route/{name}")
        steps, _, _ = ctx.muon_source(name)
        store = None
        pieces = []
        size = GE2_REFERENCE_CHUNK.get(name, len(indices))
        for first in range(0, len(indices), size):
            sub = R.subset_kernel(ctx.kernel(), indices[first:first + size], threshold_pe=0.0)
            if store is None:
                store = sub.ge2_kernel_store(str(bundle_path("tables", "ge2-pack")))
            pieces.append(transport_hybrid_basis(sub, steps, lh.ShowerBasis(), store=store,
                                                 representation="cells"))
        response = SimpleNamespace(
            charge_components_pe=np.concatenate([np.asarray(p.charge_components_pe) for p in pieces]),
            components_pe=np.concatenate([np.asarray(p.components_pe) for p in pieces]),
            time_origin_ns=np.concatenate([np.asarray(p.time_origin_ns) for p in pieces]))
        from .cases_routes import response_origin
        return R.embed([(response, (2,))], indices, len(origins),
                       response_origin(response, indices, origins),
                       metadata=dict(reference="cloud-in-cell >=2 per band (transport_hybrid_basis cells), all original steps",
                                     om_chunk=size))
    return run


# OM chunk size per input (a long source has a per-OM time origin, so chunks give the same
# per-OM response); none is needed by the inputs measured so far.
GE2_REFERENCE_CHUNK: dict = {}


for _name in REFERENCE_INPUTS:
    case("reference-original-steps", _name, heavy=True, warmup=False,
         notes="orders 0 and 1: transport_prompt on all original steps with "
               "PromptConfig(element_method='grouped', ballistic_method='interval', screen=False, "
               "fuse_bands=False) on the OMs of the default muon route")(_muon_prompt_reference(_name))
    case("reference-per-band-ge2", _name, heavy=True, warmup=False,
         notes="order >=2: the per-band cloud-in-cell lattice (KernelConfig.cell_m) through "
               "transport_hybrid_basis(representation='cells'), all original steps, OMs of the default "
               "muon route")(_muon_ge2_reference(_name))
    from .cases_routes import _combined_reference
    case("reference-muon", _name, warmup=False,
         notes="combined: reference-original-steps + reference-per-band-ge2")(_combined_reference(_name))
