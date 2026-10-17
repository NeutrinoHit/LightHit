#!/usr/bin/env python3
"""One-core primary-only direct-integral experiment on saved five-cluster muons.

Keeps the original primary source for the grouped/segment reference. Nothing
here changes prompt order zero or >=2. Choose a few receivers before doing a
whole detector sweep; raw source and output files intentionally remain local.
"""
import paths  # inputs and outputs: scripts/shower_basis/paths.py
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from time import perf_counter, process_time

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    base = paths.work("muon-five-cluster-20261008")
    p.add_argument("--energy", type=int, default=1000)
    p.add_argument("--source", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--receivers", nargs="+", type=int, default=[197])
    p.add_argument("--active", action="store_true", help="All active OMs in the saved corrected full response")
    p.add_argument("--max-length", type=float, default=10.)
    p.add_argument("--position-tolerance", type=float, default=.01)
    p.add_argument("--time-tolerance", type=float, default=.03)
    p.add_argument("--direction-tolerance", type=float, default=.001)
    p.add_argument("--reference-method", choices=["grouped", "segment", "ring"], default="grouped")
    p.add_argument("--fit-method", choices=["segment", "grouped", "ring"], default="segment")
    p.add_argument("--reuse-reference", type=Path)
    p.add_argument("--reference-only", action="store_true")
    p.add_argument("--refine", action="store_true", help="Double path/azimuth and longitudinal quadrature resolution")
    args = p.parse_args()
    import lighthit as lh
    import numba
    import ntsim
    from ntsim.LightTransport import bgvd_model_from_ntsim
    from lighthit.experimental.muon_integral import fit_track_integrals, order1_integrals
    if numba.get_num_threads() != 1:
        p.error("requires NUMBA_NUM_THREADS=1")
    path = args.source or paths.muon_source(args.energy)
    with np.load(path) as z:
        source = lh.SpectralLightElements(**{k: z[k] for k in z.files if k != "metadata"},
                                          cone_model="spectral")
        source_metadata = json.loads(str(z["metadata"]))
    source = source.subset(source.uid == 0)
    with np.load(paths.muon_response(args.energy)) as z:
        if args.active:
            args.receivers = np.flatnonzero(z["active"]).tolist()
        origins = z["time_origin_ns"][args.receivers]
        edges = z["relative_time_edges_ns"]
    model = bgvd_model_from_ntsim(ntsim.bgvd.Telescope(n_clusters=(1,2,3,4,5)), ntsim.bgvd.Water())
    if model.fingerprint != source_metadata["model_fingerprint"]:
        p.error("saved source model provenance differs from the current five-cluster model")
    full_model_fingerprint = model.fingerprint
    det = model.detector
    ids = np.asarray(args.receivers)
    det = replace(det, positions_m=det.positions_m[ids], orientations=det.orientations[ids],
                  effective_area_m2=det.effective_area_m2[ids],
                  identifiers={k: np.asarray(v)[ids] for k,v in det.identifiers.items()})
    model = replace(model, detector=det)
    kernel = model.kernel(lh.KernelConfig(relative_time_edges_ns=edges, threshold_pe=0.,
                          cache_directory=str(paths.RTE_CACHE.path()),
                          cache_policy="require"))
    args.output.mkdir(parents=True, exist_ok=True)
    config = lh.PromptConfig(element_method=args.reference_method, screen=False)
    if args.refine:
        config = replace(config, path_step=.05, path_step_tail=.125,
                         a_gauss=5, phi_gauss=8, phi_uniform=32, front_width_ns=1.25)
    # Warm the same numerical entry points on a single real source element.
    warm = source.subset(np.arange(len(source)) == len(source)//2)
    order1_integrals(kernel, warm, time_origin_ns=origins, config=config)
    order1_integrals(kernel, warm, time_origin_ns=origins,
                     config=replace(config, element_method=args.fit_method))
    provenance = dict(source=str(path), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                      energy_GeV=args.energy, receiver_indices=args.receivers,
                      full_detector_modules=1440, clusters=[1,2,3,4,5],
                      primary_elements=len(source), threads=numba.get_num_threads(),
                      model_fingerprint=full_model_fingerprint, refined_quadrature=args.refine,
                      scope="primary only, order one; original order zero and >=2 unchanged")
    if args.reuse_reference:
        z = np.load(args.reuse_reference)
        ref_q, ref_b = z["charge"], z["bins"]
        ref_meta = json.loads(args.reuse_reference.with_suffix(".json").read_text())
        if ref_meta["source_sha256"] != provenance["source_sha256"] or ref_meta["receiver_indices"] != args.receivers:
            p.error("reference source/receiver mismatch")
    else:
        wall, cpu = perf_counter(), process_time()
        ref_q, ref_b, detail = order1_integrals(kernel, source, time_origin_ns=origins, config=config)
        ref_meta = dict(**provenance, method=args.reference_method,
                        wall_seconds=perf_counter()-wall, cpu_seconds=process_time()-cpu, diagnostics=detail)
        np.savez_compressed(args.output / "reference.npz", charge=ref_q, bins=ref_b, origins=origins, edges=edges)
        (args.output / "reference.json").write_text(json.dumps(ref_meta, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x)))
        print("REFERENCE", json.dumps({k:v for k,v in ref_meta.items() if k != "diagnostics"}), flush=True)
    if args.reference_only:
        return
    began = perf_counter()
    fit = fit_track_integrals(source, max_length_m=args.max_length,
                    position_tolerance_m=args.position_tolerance,
                    time_tolerance_ns=args.time_tolerance,
                    direction_tolerance_rad=args.direction_tolerance)
    fit_seconds = perf_counter()-began
    print("FIT", len(fit.elements), "pieces", fit_seconds, "seconds", flush=True)
    wall, cpu = perf_counter(), process_time()
    q, b, detail = order1_integrals(kernel, fit.elements, time_origin_ns=origins,
                                   config=replace(config, element_method=args.fit_method))
    result = dict(**provenance, method=args.fit_method, reference_method=ref_meta["method"],
                  wall_seconds=perf_counter()-wall, cpu_seconds=process_time()-cpu,
                  reference_wall_seconds=ref_meta["wall_seconds"], fit_seconds=fit_seconds,
                  pieces=len(fit.elements), max_length_m=args.max_length,
                  position_tolerance_m=args.position_tolerance, time_tolerance_ns=args.time_tolerance,
                  direction_tolerance_rad=args.direction_tolerance,
                  maximum_position_error_m=fit.maximum_position_error_m,
                  maximum_time_error_ns=fit.maximum_time_error_ns,
                  maximum_direction_rms_rad=fit.maximum_direction_rms_rad,
                  charge_reference_pe=ref_q.tolist(), charge_pe=q.tolist(),
                  charge_relative_error=(np.abs(q-ref_q)/np.maximum(np.abs(ref_q),1e-300)).tolist(),
                  bin_l1_relative_error=(np.abs(b-ref_b).sum(1)/np.maximum(np.abs(ref_b).sum(1),1e-300)).tolist(),
                  diagnostics=detail)
    np.savez_compressed(args.output / "candidate.npz", charge=q, bins=b, origins=origins, edges=edges)
    (args.output / "candidate.json").write_text(json.dumps(result, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x)))
    print("CANDIDATE", json.dumps({k:v for k,v in result.items() if k != "diagnostics"}), flush=True)


if __name__ == "__main__":
    main()
