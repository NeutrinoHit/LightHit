#!/usr/bin/env python3
"""Compose a recalculated prompt with an unchanged saved RTE response.

An accuracy-study utility, not a full-event timing benchmark. Newly active OMs
are evaluated through the RTE cache rather than receiving missing/zero spectra.
"""
import paths  # inputs and outputs: scripts/shower_basis/paths.py
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from time import perf_counter

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--prompt-directory", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cache", default=str(paths.RTE_CACHE.path()))
    p.add_argument("--pack", default=str(paths.GE2_PACK.path()))
    args = p.parse_args()
    import lighthit as lh
    import ntsim
    from ntsim.LightTransport import bgvd_model_from_ntsim
    ref_meta = json.loads(args.reference.with_suffix(".json").read_text())
    prompt_meta = json.loads((args.prompt_directory / "prompt-metadata.json").read_text())
    prompt_config = json.loads((args.prompt_directory / "configuration.json").read_text())
    source_hash = hashlib.sha256(args.source.read_bytes()).hexdigest()
    if source_hash != prompt_config["source_sha256"] or source_hash != ref_meta["source_sha256"]:
        raise ValueError("source identity differs between reference and prompt")
    with np.load(args.source, allow_pickle=False) as z:
        source_meta = json.loads(str(z["metadata"]))
        elements = lh.SpectralLightElements(**{k: z[k] for k in z.files if k != "metadata"},
                                            cone_model="spectral", provenance=source_meta["provenance"])
    with np.load(args.reference, allow_pickle=False) as z:
        ref = {k: z[k] for k in z.files}
    with np.load(args.prompt_directory / "response-prompt.npz", allow_pickle=False) as z:
        prompt = {k: z[k] for k in z.files}
    for key in ("positions", "edges", "origin"):
        np.testing.assert_array_equal(ref[key], prompt[key])
    np.testing.assert_array_equal(ref["charge"][:, 2], prompt["selection_extra_charge_ge2"])
    model = bgvd_model_from_ntsim(ntsim.bgvd.Telescope(n_clusters=tuple(ref_meta["clusters"])),
                                 ntsim.bgvd.Water())
    np.testing.assert_array_equal(model.detector.positions_m, ref["positions"])
    charge, bins = ref["charge"].copy(), ref["bins"].copy()
    spectrum = ref["spectrum_ge2"].copy()
    charge[:, :2] = prompt["charge"]
    bins[:, :, :2] = prompt["bins"]
    active = prompt["active"].copy()
    added = np.flatnonzero(active & ~ref["active"])
    extra_seconds = 0.
    if len(added):
        d = model.detector
        small = replace(d, positions_m=d.positions_m[added], orientations=d.orientations[added],
                        effective_area_m2=d.effective_area_m2[added],
                        identifiers={k: np.asarray(v)[added] for k, v in d.identifiers.items()})
        cfg = lh.KernelConfig(omega_per_ns=np.linspace(0., 1.2, 65),
                              readout_omega_per_ns=np.linspace(0., 1.2, 321),
                              relative_time_edges_ns=ref["edges"], cache_directory=args.cache,
                              cache_policy="require", threshold_pe=ref_meta["metadata"]["threshold_pe"])
        kernel = replace(model, detector=small).kernel(cfg)
        kernel.ge2_kernel_store(args.pack)
        def supplied_prompt(source, settings=None, **kw):
            np.testing.assert_allclose(kw["time_origin_ns"], ref["origin"][added], rtol=0, atol=1e-10)
            return SimpleNamespace(
                relative_time_edges_ns=ref["edges"], time_origin_ns=ref["origin"][added],
                charge_orders_pe=prompt["charge"][added], bins_orders_pe=prompt["bins"][added],
                charge_pe=prompt["charge"][added].sum(1), computed=np.ones(len(added), bool),
                metadata={**prompt_meta, "order1_level": np.asarray(prompt_meta["order1_level"])[added].tolist()})
        kernel.transport_prompt = supplied_prompt
        start = perf_counter()
        extra = kernel.transport_hybrid(elements, frequency_nodes=len(ref["omega"]),
                     ge2_basis=lh.ShowerBasis(track_uid=0, spectral_fusion=True, apply_tolerance_pe=1e-6))
        extra_seconds = perf_counter() - start
        charge[added, 2] = extra.charge_components_pe[:, 2]
        bins[added, :, 2] = extra.components_pe[:, :, 2]
        spectrum[:, added] = extra.spectrum_ge2_pe
        active[added] = extra.active
    bins[~active] = 0.
    spectrum[1:, ~active] = 0.
    spectrum[0] = charge[:, 2]
    base_metadata = dict(ref_meta["metadata"])
    reference_timings = {k: base_metadata.pop(k) for k in list(base_metadata) if k.endswith("seconds")}
    metadata = {**base_metadata, "rte_reused_from": str(args.reference),
                "reference_timings": reference_timings,
                "prompt_elapsed_seconds": prompt_meta["timings"]["elapsed_seconds"],
                "component_refresh_seconds": prompt_meta["timings"]["elapsed_seconds"] + extra_seconds,
                "reused_rte": True,
                "rte_reference_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
                "new_rte_om_indices": added.tolist(), "extra_rte_seconds": extra_seconds,
                "prompt_timings": prompt_meta["timings"],
                "prompt_order1_level": prompt_meta["order1_level"],
                "prompt_order1_backward_profile": prompt_meta.get("order1_backward_profile"),
                "timing_scope": "component reuse accuracy study, not a fresh full-event benchmark",
                "active_modules": int(active.sum()), "source_sha256": source_hash}
    response = lh.HybridTransportResponse(model.detector, ref["omega"], ref["edges"], spectrum,
                     bins, charge, ref["origin"], active, "hybrid-basis-prompt-refresh", metadata)
    args.output.mkdir(parents=True, exist_ok=True)
    response.save(args.output / "response.npz")
    (args.output / "response.json").write_text(json.dumps(metadata, indent=2))
    payload = lh.viewer_payload(response, source=lh.G4Shower(elements, str(args.source), 0),
                 event_id="muon-corrected", label=f"Muon {source_meta['energy_GeV']:g} GeV, five clusters, corrected order 1")
    lh.write_event_viewer(payload, args.output / "viewer.html")
    print(json.dumps({"active": int(active.sum()), "new_rte_oms": added.tolist(),
                      "extra_rte_seconds": extra_seconds, "viewer": str(args.output / "viewer.html")}))


if __name__ == "__main__":
    main()
