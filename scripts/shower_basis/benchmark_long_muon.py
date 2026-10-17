#!/usr/bin/env python3
"""Reproducible five-cluster muon benchmark; private inputs/results stay local.

Generate each G4 event once, then reuse its saved charged-step source for every
transport variant. Run with NUMBA_NUM_THREADS=OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=1.
Times include the first-call JIT unless --repeat 2 is used; both calls are saved.
"""
import paths  # inputs and outputs: scripts/shower_basis/paths.py
import argparse
from dataclasses import fields, replace
import hashlib
import json
from pathlib import Path
import resource
import sys
from time import perf_counter

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("generate", "run"))
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path)
    p.add_argument("--energy", type=float, default=300.)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--clusters", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    p.add_argument("--position", type=float, nargs=3, default=[-100., -950., 350.])
    p.add_argument("--direction", type=float, nargs=3, default=[0., 1., 0.])
    p.add_argument("--cache", type=Path, default=paths.RTE_CACHE.path())
    p.add_argument("--pack", type=Path, default=paths.GE2_PACK.path())
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--frequencies", type=int, default=33)
    p.add_argument("--track-uid", type=int, default=0)
    p.add_argument("--all-harmonics", action="store_true")
    p.add_argument("--component", choices=("all", "primary", "secondary"), default="all")
    p.add_argument("--receiver-indices", type=int, nargs="+")
    p.add_argument("--receiver-index-file", type=Path)
    p.add_argument("--profile", action="store_true")
    p.add_argument("--progress", action="store_true")
    p.add_argument("--reuse-prompt", type=Path,
                   help="accuracy studies ONLY: reuse orders 0+1 from this saved response, report RTE-only timing")
    p.add_argument("--apply-tolerance-pe", type=float, default=0.)
    p.add_argument("--prompt-batch-tolerance-pe", type=float, default=0.)
    p.add_argument("--prompt-profile", choices=("default", "muon-fast"), default="default")
    p.add_argument("--degree", type=int, help="Experimental source harmonic truncation using the same full-rank RTE cache")
    p.add_argument("--cell-m", type=float, default=0.5)
    args = p.parse_args()
    if args.receiver_index_file:
        args.receiver_indices = json.loads(args.receiver_index_file.read_text())
    import ntsim
    import lighthit as lh
    from ntsim.LightTransport import bgvd_model_from_ntsim, source_from_event
    model = bgvd_model_from_ntsim(ntsim.bgvd.Telescope(n_clusters=tuple(args.clusters)), ntsim.bgvd.Water())
    if args.action == "generate":
        if args.source.exists():
            p.error("source already exists; choose a new path to preserve the event")
        from ntsim.event_chain import Chain, EmptyEventSource
        direction = np.asarray(args.direction, float)
        direction /= np.linalg.norm(direction)
        chain = Chain(
            ntsim.primary.ToyGen(energy=args.energy, pdg=13, position=tuple(args.position),
                                 direction=tuple(direction), time=0.) << "primary_gen",
            ntsim.modules.ParticlePropagator(random_seed=args.seed, enable_cherenkov=False,
                                             cascade_bounds_e=(0., 0.)) << "g4prop")
        start = perf_counter()
        event = next(chain(EmptyEventSource()))
        source = source_from_event(event, cone_model="spectral",
                                   max_phase_index=float(np.max(model.medium.phase_index))).elements
        arrays = {f.name: getattr(source, f.name) for f in fields(source)
                  if isinstance(getattr(source, f.name), np.ndarray)}
        primary = source.uid == args.track_uid
        meta = dict(energy_GeV=args.energy, seed=args.seed, clusters=args.clusters,
                    position_m=args.position, direction=direction.tolist(),
                    generated_seconds=perf_counter()-start, elements=len(source),
                    primary_elements=int(primary.sum()),
                    primary_chord_length_m=float(source.length_m[primary].sum()),
                    source_bounds_m=[source.start_m.min(0).tolist(), source.start_m.max(0).tolist()],
                    provenance=source.provenance, model_fingerprint=model.fingerprint)
        args.source.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.source, **arrays, metadata=json.dumps(meta))
        args.source.with_suffix(".json").write_text(json.dumps(meta, indent=2))
        print(json.dumps(meta, indent=2), flush=True)
        return
    if args.output is None:
        p.error("run requires --output")
    with np.load(args.source, allow_pickle=False) as z:
        source = lh.SpectralLightElements(**{k: z[k] for k in z.files if k != "metadata"}, cone_model="spectral")
        source_meta = json.loads(str(z["metadata"]))
    if args.component != "all":
        source = source.subset((source.uid == args.track_uid) if args.component == "primary"
                               else (source.uid != args.track_uid))
    if args.receiver_indices:
        ids = np.array(args.receiver_indices)
        d = model.detector
        model = replace(model, detector=replace(d, positions_m=d.positions_m[ids],
                        orientations=d.orientations[ids], effective_area_m2=d.effective_area_m2[ids],
                        identifiers={k: np.asarray(v)[ids] for k, v in d.identifiers.items()}))
    kernel = model.kernel(lh.KernelConfig(omega_per_ns=np.linspace(0., 1.2, 65),
                        readout_omega_per_ns=np.linspace(0., 1.2, 321),
                        relative_time_edges_ns=np.arange(-60., 745., 5.),
                        cache_directory=args.cache, cache_policy="require"))
    if args.reuse_prompt:
        from types import SimpleNamespace
        saved_meta = json.loads(args.reuse_prompt.with_suffix(".json").read_text())
        if saved_meta["source_sha256"] != hashlib.sha256(args.source.read_bytes()).hexdigest():
            p.error("reused prompt must belong to the identical source file")
        with np.load(args.reuse_prompt, allow_pickle=False) as z:
            rows = np.asarray(args.receiver_indices) if args.receiver_indices else slice(None)
            if not np.array_equal(z["positions"][rows], model.detector.positions_m):
                p.error("reused prompt detector geometry differs")
            old = saved_meta["metadata"]
            prompt = SimpleNamespace(
                relative_time_edges_ns=z["edges"].copy(), time_origin_ns=z["origin"][rows].copy(),
                charge_orders_pe=z["charge"][rows, :2].copy(),
                bins_orders_pe=z["bins"][rows, :, :2].copy(),
                charge_pe=z["charge"][rows, :2].sum(axis=1),
                computed=np.asarray(old["prompt_order1_level"])[rows] == 2,
                metadata={"prompt_algorithm": old["prompt_algorithm"],
                          "order1_level": np.asarray(old["prompt_order1_level"])[rows].tolist()})
        def reused_prompt(source, config=None, **kw):
            if kw.get("time_origin_ns") is not None:
                np.testing.assert_allclose(kw["time_origin_ns"], prompt.time_origin_ns, rtol=0, atol=1e-10)
            return prompt
        kernel.transport_prompt = reused_prompt
    store = kernel.ge2_kernel_store(args.pack)
    if args.progress:
        import lighthit._shower_basis as sb
        def traced(name, function):
            def call(*a, **kw):
                start = perf_counter()
                print(name, "begin", flush=True)
                result = function(*a, **kw)
                print(name, "seconds", perf_counter()-start, flush=True)
                return result
            return call
        for name in ("compile_axial_slices", "apply_multi", "apply_multi_spectral"):
            setattr(sb, name, traced(name, getattr(sb, name)))
    if args.degree is not None:
        # Prototype only: keep medium scattering rank and all cache settings;
        # truncate the SOURCE expansion, not the physical transport operator.
        import lighthit._shower_basis as sb
        original_compile = sb.compile_axial_slices
        def compile_reduced(field, degree, frequencies, **kw):
            return original_compile(field, args.degree, frequencies, **kw)
        sb.compile_axial_slices = compile_reduced
        original_bundle = store.bundle
        def bundle_reduced(kernel, indices):
            b = original_bundle(kernel, indices)
            return dict(b, degree=args.degree, coefficients=b["coefficients"][:, :, :args.degree + 1])
        store.bundle = bundle_reduced
    basis = lh.ShowerBasis(track_uid=None if args.all_harmonics or args.component == "secondary"
                          else args.track_uid, spectral_fusion=True,
                          track_cell_m=args.cell_m, secondary_cell_m=args.cell_m,
                          **({"apply_tolerance_pe": args.apply_tolerance_pe}
                             if args.apply_tolerance_pe else {}))
    args.output.mkdir(parents=True, exist_ok=True)
    import numba
    print("source", len(source), "OMs", len(model.detector.positions_m),
          "threads", numba.get_num_threads(), "basis", basis, flush=True)
    for trial in range(args.repeat):
        profiler = None
        if args.profile:
            import cProfile
            profiler = cProfile.Profile()
            profiler.enable()
        start = perf_counter()
        prompt_options = {"batch_charge_tolerance_pe": args.prompt_batch_tolerance_pe}
        if args.prompt_profile == "muon-fast":
            prompt_options.update(screen_cell_m=5., screen_pixel_face=4,
                                  screen_ring_nodes=12, screen_margin=6.)
        response = kernel.transport_hybrid(source, frequency_nodes=args.frequencies, ge2_basis=basis,
                    **({"prompt_config": lh.PromptConfig(**prompt_options)}
                       if args.prompt_batch_tolerance_pe or args.prompt_profile != "default" else {}))
        wall = perf_counter()-start
        if profiler:
            profiler.disable()
            profiler.dump_stats(str(args.output / f"profile-{trial}.pstats"))
        meta = dict(wall_seconds=wall, peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss /
                    (2**20 if sys.platform == "darwin" else 2**10), version=lh.__version__,
                    threads=numba.get_num_threads(), source=source_meta,
                    reused_prompt=bool(args.reuse_prompt),
                    source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
                    clusters=args.clusters, active=int(response.active.sum()), metadata=response.metadata)
        np.savez_compressed(args.output / f"response-{trial}.npz",
                            charge=response.charge_components_pe, bins=response.components_pe,
                            active=response.active, origin=response.time_origin_ns,
                            spectrum_ge2=response.spectrum_ge2_pe, omega=response.omega_per_ns,
                            edges=response.relative_time_edges_ns, positions=model.detector.positions_m)
        (args.output / f"response-{trial}.json").write_text(json.dumps(meta, indent=2, default=str))
        print(json.dumps({k: v for k, v in meta.items() if k not in ("source", "metadata")}), flush=True)
        print(json.dumps({k: v for k, v in response.metadata.items() if k.endswith("seconds")}), flush=True)


if __name__ == "__main__":
    main()
