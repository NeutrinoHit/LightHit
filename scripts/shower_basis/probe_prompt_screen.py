#!/usr/bin/env python3
"""Empirical coarse-screen probe for one saved five-cluster muon event.

Only prompt orders 0 and 1 are computed. The saved >=2 charge is used for
identical hybrid OM selection; no RTE cache is applied. This tests one event,
not a certified screening bound. Inactive coarse charges can change freely.
Run from the checkout with NUMBA_NUM_THREADS=OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=1.
"""
import paths  # inputs and outputs: scripts/shower_basis/paths.py
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import resource
import sys
from time import perf_counter, process_time

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    base = paths.work("muon-five-cluster-20261008")
    parser.add_argument("--source", type=Path, default=paths.muon_source(300))
    parser.add_argument("--reference", type=Path, default=base / "reference300/response-0.npz")
    parser.add_argument("--output", type=Path, default=base / "probe-screen300")
    parser.add_argument("--require-bitwise-q0", action="store_true",
                        help="Require all q0 charges and bins to match bitwise, e.g. regenerated G4 source")
    parser.add_argument("--expect-physics-change", action="store_true",
                        help="record differences from an old physics implementation without treating q1 changes as a failed speed regression")
    args = parser.parse_args()
    if (args.output / "response-prompt.npz").exists():
        parser.error("output already contains a response; preserve it and choose another output")

    import lighthit as lh
    import ntsim
    import numba
    from ntsim.LightTransport import bgvd_model_from_ntsim

    if numba.get_num_threads() != 1:
        parser.error("this experiment requires NUMBA_NUM_THREADS=1")
    ref_meta = json.loads(args.reference.with_suffix(".json").read_text())
    source_hash = hashlib.sha256(args.source.read_bytes()).hexdigest()
    if source_hash != ref_meta["source_sha256"]:
        parser.error("reference response is for a different source file")
    clusters = ref_meta["clusters"]
    if len(clusters) != 5:
        parser.error("reference must contain exactly five clusters")
    with np.load(args.source, allow_pickle=False) as archive:
        source = lh.SpectralLightElements(
            **{key: archive[key] for key in archive.files if key != "metadata"},
            cone_model="spectral")
        source_meta = json.loads(str(archive["metadata"]))
    with np.load(args.reference, allow_pickle=False) as archive:
        reference = {key: archive[key] for key in archive.files}
    model = bgvd_model_from_ntsim(
        ntsim.bgvd.Telescope(n_clusters=tuple(clusters)), ntsim.bgvd.Water())
    if not np.array_equal(reference["positions"], model.detector.positions_m):
        parser.error("reference geometry differs from the current five-cluster model")
    if model.fingerprint != source_meta["model_fingerprint"]:
        parser.error("current medium/detector model differs from source-generation provenance")
    kernel = model.kernel(lh.KernelConfig(
        omega_per_ns=np.linspace(0., 1.2, 65),
        readout_omega_per_ns=np.linspace(0., 1.2, 321),
        relative_time_edges_ns=reference["edges"],
        threshold_pe=ref_meta["metadata"]["threshold_pe"],
        cache_directory=str(paths.RTE_CACHE.path()),
        cache_policy="require"))
    settings = lh.PromptConfig(screen_cell_m=5., screen_pixel_face=4,
                               screen_ring_nodes=12, screen_margin=6.,
                               batch_charge_tolerance_pe=1e-6)
    args.output.mkdir(parents=True, exist_ok=True)
    provenance = dict(
        source=str(args.source), source_sha256=source_hash,
        reference=str(args.reference),
        reference_sha256=hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        source_association=ref_meta.get("source_association", "identical saved source file"),
        require_bitwise_q0=args.require_bitwise_q0,
        expect_physics_change=args.expect_physics_change,
        clusters=clusters, model_fingerprint=model.fingerprint,
        version=lh.__version__, threads=numba.get_num_threads(),
        elements=len(source), modules=len(model.detector.positions_m),
        prompt_config=asdict(settings),
        scope="Empirical coarse-screen control on this event; no certified bound. "
              "Fine settings unchanged; inactive coarse q1 charges may differ.")
    (args.output / "configuration.json").write_text(json.dumps(provenance, indent=2))
    print(json.dumps(provenance), flush=True)
    begin, cpu_begin = perf_counter(), process_time()
    response = kernel.transport_prompt(
        source, settings, time_origin_ns=reference["origin"],
        selection_extra_charge=reference["charge"][:, 2])
    elapsed, cpu_elapsed = perf_counter() - begin, process_time() - cpu_begin
    np.savez_compressed(
        args.output / "response-prompt.npz", charge=response.charge_orders_pe,
        bins=response.bins_orders_pe, active=response.active,
        origin=response.time_origin_ns, edges=response.relative_time_edges_ns,
        positions=model.detector.positions_m,
        selection_extra_charge_ge2=reference["charge"][:, 2])
    (args.output / "prompt-metadata.json").write_text(
        json.dumps(response.metadata, indent=2))

    old_active, new_active = reference["active"], response.active
    union = old_active | new_active
    old_level = np.asarray(ref_meta["metadata"]["prompt_order1_level"])
    new_level = np.asarray(response.metadata["order1_level"])
    charge_error = response.charge_orders_pe - reference["charge"][:, :2]
    bin_error = response.bins_orders_pe - reference["bins"][:, :, :2]
    max_charge = np.max(np.abs(charge_error[union]), axis=0, initial=0.)
    bin_l1 = np.abs(bin_error).sum(axis=1)
    max_bin_l1 = np.max(bin_l1[union], axis=0, initial=0.)
    lost = np.flatnonzero(old_active & ~new_active)
    gained = np.flatnonzero(new_active & ~old_active)
    not_fine = np.flatnonzero(union & ((old_level != 2) | (new_level != 2)))
    tolerance = settings.batch_charge_tolerance_pe
    q0_charge_bitwise = np.array_equal(response.charge_orders_pe[:, 0], reference["charge"][:, 0])
    q0_bins_bitwise = np.array_equal(response.bins_orders_pe[:, :, 0], reference["bins"][:, :, 0])
    accepted = (not len(lost) and not len(gained) and not len(not_fine)
                and max_charge[0] <= 1e-12 and max_bin_l1[0] <= 1e-12
                and max_charge[1] <= tolerance and max_bin_l1[1] <= tolerance
                and (not args.require_bitwise_q0 or (q0_charge_bitwise and q0_bins_bitwise)))
    reference_window = np.abs(reference["bins"][:, :, 1]).sum(axis=1)
    comparison = dict(
        accepted_for_this_event=bool(accepted),
        criterion="Identical hybrid active set, fine q1 on its union, q0 unchanged "
                  "within 1e-12 PE, q1 charge and per-OM bin-L1 differences <=1e-6 PE.",
        wall_seconds=elapsed, process_cpu_seconds=cpu_elapsed,
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                     / (2**20 if sys.platform == "darwin" else 2**10),
        reference_prompt_seconds=ref_meta["metadata"]["prompt_elapsed_seconds"],
        reference_over_trial_wall_ratio=ref_meta["metadata"]["prompt_elapsed_seconds"] / elapsed,
        timing_scope="Same thread count; wall times may include cache loading/JIT and concurrent-system load.",
        old_active=int(old_active.sum()), new_active=int(new_active.sum()),
        lost_active=lost.tolist(), gained_active=gained.tolist(),
        union_oms_without_both_fine=not_fine.tolist(),
        all_om_q0_charges_bitwise_identical=bool(q0_charge_bitwise),
        all_om_q0_bins_bitwise_identical=bool(q0_bins_bitwise),
        maximum_absolute_charge_error_pe_by_order=max_charge.tolist(),
        maximum_bin_l1_error_pe_by_order=max_bin_l1.tolist(),
        maximum_relative_q1_charge_error=float(np.max(
            np.abs(charge_error[union, 1]) / np.maximum(
                np.abs(reference["charge"][union, 1]), 1e-300), initial=0.)),
        maximum_relative_q1_bin_l1=float(np.max(
            bin_l1[union, 1] / np.maximum(reference_window[union], 1e-300), initial=0.)),
        old_fine_modules=int((old_level == 2).sum()), new_fine_modules=int((new_level == 2).sum()),
        inactive_charge_comparison="Excluded: coarse settings differ intentionally.",
        timings=response.metadata["timings"],
        global_screen=response.metadata.get("order1_global_screen"),
        batch_screen=response.metadata.get("order1_batch_screen"), **provenance)
    (args.output / "comparison.json").write_text(json.dumps(comparison, indent=2))
    print(json.dumps({key: value for key, value in comparison.items()
                      if key not in ("batch_screen", "prompt_config")}, indent=2), flush=True)
    if not accepted and not args.expect_physics_change:
        raise SystemExit("Candidate rejected for this event; inspect comparison.json")


if __name__ == "__main__":
    main()
