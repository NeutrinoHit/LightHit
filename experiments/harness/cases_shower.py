"""Shower-route alternatives: one setting changed from ShowerConfig() at a time.

Each case runs ShowerTransport on a standard shower input with the changed
setting and is compared with the combined shower reference of that input
(original-step orders 0 and 1, per-band order >=2). The default route on the
same input is the speed baseline.
"""
from __future__ import annotations

from dataclasses import replace

from .cases import case

INPUTS = ("electron-100GeV", "pion-100GeV")


def _config(**changes):
    from lighthit.shower import ShowerConfig
    base = ShowerConfig()
    prompt = changes.pop("prompt", None)
    basis = changes.pop("basis", None)
    if prompt:
        changes["prompt"] = replace(base.prompt, **prompt)
    if basis:
        changes["basis"] = replace(base.basis, **basis)
    return replace(base, **changes)


def _route(**changes):
    def make(name):
        def run(ctx):
            from lighthit.shower import ShowerTransport
            route = ctx.cached("route", lambda: ShowerTransport(ctx.kernel(), config=_config(**changes)))
            return route(ctx.shower_source(name))
        return run
    return make


def _hybrid(**options):
    """TransportKernel.transport_hybrid (the per-band routes) on a shower input."""
    def make(name):
        def run(ctx):
            import lighthit as lh
            kwargs = dict(options)
            if kwargs.pop("basis", False):
                kwargs["ge2_basis"] = lh.ShowerBasis(**kwargs.pop("basis_options", {}))
            if "prompt" in kwargs:
                kwargs["prompt_config"] = lh.PromptConfig(**kwargs.pop("prompt"))
            return ctx.kernel().transport_hybrid(ctx.shower_source(name), **kwargs)
        return run
    return make


#: idea -> {variant: setting}; one ShowerConfig change per variant
VARIANTS = {
    # orders 0 and 1 (PromptConfig inside ShowerConfig)
    "shower-order1-band-fusion": {"per-band": _route(prompt=dict(fuse_bands=False)),
                                  "fused-64": _route(prompt=dict(fused_pixel_face=64))},
    "shower-order0-continuous-spectrum": {"quadrature": _route(prompt=dict(ballistic_method="quadrature"))},
    "shower-order1-batch-tolerance": {"1e-3": _route(prompt=dict(batch_charge_tolerance_pe=1e-3))},
    "shower-order1-element-method": {"ring": _route(prompt=dict(element_method="ring", fuse_bands=False))},
    # order >=2 source basis
    "shower-basis-axial-order": {"linear": _route(basis=dict(axial_order=1))},
    "shower-basis-transverse": {"collapse": _route(basis=dict(transverse="collapse"))},
    "shower-basis-time-deposit": {"nearest": _route(basis=dict(time_deposit="nearest")),
                                  "cubic": _route(basis=dict(time_deposit="cubic"))},
    "shower-basis-cell": {"0.125m": _route(basis=dict(cell_m=0.125)),
                          "0.5m": _route(basis=dict(cell_m=0.5))},
    "shower-basis-time-step": {"0.01ns": _route(basis=dict(time_step_ns=0.01)),
                               "0.1ns": _route(basis=dict(time_step_ns=0.1))},
    # band fold and frequencies
    "shower-band-fold-rank": {"rank-2": _route(band_rank=2), "rank-4": _route(band_rank=4),
                              "rank-9": _route(band_rank=9)},
    "shower-band-fold-subdivide": {"subdivide-2": _route(fold_subdivide=2)},
    "shower-frequency-nodes": {"33": _route(frequency_nodes=33)},
    # predictor of the zero-frequency order >=2 charge
    "shower-ge2-predictor": {"safety-1": _route(predictor_safety=1.0),
                             "cell-8m": _route(predictor_cell_m=8.0)},
    # the per-band routes of TransportKernel.transport_hybrid
    "hybrid-basis-per-band": {"default": _hybrid(basis=True)},
    "hybrid-cloud-in-cell": {"default": _hybrid()},
}

NOTES = {
    "hybrid-basis-per-band": "the previous per-band route; also what NTSim's LightHitTransport calls for a shower when the event "
                             "route is unavailable or an expert option is set: "
                             "transport_hybrid(source, frequency_nodes=65, ge2_basis=ShowerBasis("
                             "track_uid=None, spectral_fusion=False)) with the default PromptConfig",
    "hybrid-cloud-in-cell": "transport_hybrid(source) with ge2_basis=None: elements compiled per "
                            "band on the directional tables (the per-band >=2 reference)",
}

for _idea, _variants in VARIANTS.items():
    for _variant, _make in _variants.items():
        for _name in INPUTS:
            case(_idea, f"{_name}-{_variant}" if _variant != "default" else _name,
                 reference=f"reference-shower/{_name}", baseline=f"default-shower-route/{_name}",
                 notes=NOTES.get(_idea, ""))(_make(_name))
