#!/usr/bin/env python3
"""Reject-or-validate cached first scattering for a compact real EM source.

This experiment extracts order ONE (coefficient axis 0) from existing RTE
directional tables, stores it separately, and applies the existing spectral
shower source operator. The output is explicitly q1, never an >=2 production
pack. Direct corrected prompt q1 is the control. No clipping/renormalisation.
"""
import paths  # inputs and outputs: scripts/shower_basis/paths.py
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from time import perf_counter, process_time

import numpy as np


def first_store(kernel, directory):
    from lighthit.shower_basis import Ge2KernelStore
    from lighthit._directional_numba import PreparedDirectionalKernel

    class FirstOnly(Ge2KernelStore):
        def bundle(self, kernel, indices):
            key = self._key(kernel, indices)
            if not any(k[:2] == key[:2] and set(key[2]).issubset(k[2]) for k in self._bundles):
                raise ValueError("first-collision pack does not match requested kernel")
            # Only the existing dense/subset lookup paths can be reached.
            return super().bundle(kernel, indices)

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "first-collision.json"
    indices = np.arange(len(kernel.config.omega_per_ns))
    key = Ge2KernelStore._key(kernel, indices)
    if path.exists():
        saved = json.loads(path.read_text())
        if saved["kind"] != "experimental-first-collision-only-v1":
            raise ValueError("refuse a non-order1 pack")
        if saved["key"] != [list(key[0]), key[1], list(key[2])]:
            raise ValueError("existing first-collision pack belongs to another numerical kernel")
        meta = saved["meta"]
    else:
        began = perf_counter()
        coefficients, absorption, speed = None, [], []
        for lam, wavelength in enumerate(kernel.wavelength.wavelength_nm):
            cache = kernel._directional_cache_for(float(wavelength))
            full = PreparedDirectionalKernel.from_cache(cache, degree=kernel.config.source_degree,
                                                        frequency_indices=indices)
            kernel._directional_caches.pop(kernel._directional_cache_key(float(wavelength)), None)
            one = full.coefficients[:, :, :, :, 0, :]
            if coefficients is None:
                coefficients = np.lib.format.open_memmap(directory / "first-collision.npy",
                    mode="w+", dtype=np.complex64, shape=(len(kernel.wavelength.wavelength_nm),)+one.shape)
                meta = dict(degree=full.degree, acceptance_degree=full.acceptance_degree,
                            pair_lambda=full.pair_lambda, pair_mu=full.pair_mu,
                            omega=full.omega_per_ns, log_r=full.log_r, phase=full.radial_phase)
            coefficients[lam] = one
            absorption.append(full.absorption_per_m)
            speed.append(full.speed_m_per_ns)
            full = cache = one = None
            print("FIRST PACK wavelength", lam, "seconds", perf_counter()-began, flush=True)
        coefficients.flush()
        del coefficients
        meta.update(absorption=np.asarray(absorption), speed=np.asarray(speed))
        meta = {k: v.tolist() if isinstance(v, np.ndarray) else v for k,v in meta.items()}
        path.write_text(json.dumps(dict(kind="experimental-first-collision-only-v1",
                         key=[list(key[0]), key[1], list(key[2])], meta=meta,
                         preparation_seconds=perf_counter()-began), indent=2))
    meta = dict(meta)
    for name in ("pair_lambda", "pair_mu"):
        meta[name] = np.asarray(meta[name], np.int64)
    for name in ("omega", "log_r", "absorption", "speed"):
        meta[name] = np.asarray(meta[name], float)
    store = FirstOnly()
    store._bundles[key] = dict(meta, coefficients=np.load(directory / "first-collision.npy", mmap_mode="r"))
    return store


def main():
    p = argparse.ArgumentParser(description=__doc__)
    base = paths.work("muon-segments-20261009")
    p.add_argument("--source", type=Path, default=base / "structure/electron-current-100GeV.npz")
    p.add_argument("--output", type=Path, default=base / "first-cache/electron100")
    p.add_argument("--pack", type=Path, default=base / "first-cache/pack65")
    p.add_argument("--receivers", type=int, nargs="+")
    p.add_argument("--nearest", type=int, default=4)
    p.add_argument("--frequencies", type=int, nargs="+", default=[65, 33])
    p.add_argument("--family-ancestry", type=Path)
    p.add_argument("--family-root", type=int)
    p.add_argument("--position", type=float, nargs=3,
                   help="Translate the complete source so its first start is at this pose; recorded explicitly")
    args = p.parse_args()
    import lighthit as lh
    import ntsim
    import numba
    from ntsim.LightTransport import bgvd_model_from_ntsim
    from lighthit.shower_basis import transport_hybrid_basis
    if numba.get_num_threads() != 1:
        p.error("requires NUMBA_NUM_THREADS=1")
    with np.load(args.source) as z:
        source = lh.SpectralLightElements(**{k: z[k] for k in z.files if k != "metadata"}, cone_model="spectral")
    if args.family_ancestry:
        with np.load(args.family_ancestry) as z:
            mask = z["element_family"] == args.family_root
        source = source.subset(mask)
    if args.position:
        source = source.moved(translation=np.asarray(args.position)-source.start_m[0])
    model = bgvd_model_from_ntsim(ntsim.bgvd.Telescope(n_clusters=(1,2,3,4,5)), ntsim.bgvd.Water())
    det = model.detector
    ids = np.asarray(args.receivers if args.receivers else
                     np.argsort(np.linalg.norm(det.positions_m-source.centroid_m, axis=1))[:args.nearest])
    det = replace(det, positions_m=det.positions_m[ids], orientations=det.orientations[ids],
                  effective_area_m2=det.effective_area_m2[ids],
                  identifiers={k: np.asarray(v)[ids] for k,v in det.identifiers.items()})
    model = replace(model, detector=det)
    kernel = model.kernel(lh.KernelConfig(omega_per_ns=np.linspace(0.,1.2,65),
                     readout_omega_per_ns=np.linspace(0.,1.2,321), threshold_pe=0.,
                     relative_time_edges_ns=np.arange(-60.,745.,5.),
                     cache_directory=str(paths.RTE_CACHE.path()),
                     cache_policy="require"))
    args.output.mkdir(parents=True, exist_ok=True)
    provenance = dict(source=str(args.source), source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
                      source_elements=len(source), receivers=ids.tolist(), threads=numba.get_num_threads(),
                      family_root=args.family_root, model_fingerprint=model.fingerprint,
                      translated_first_position_m=args.position,
                      full_detector_modules=1440, prepared_operator="FIRST collision, not >=2",
                      source_representation="spectral shower basis, L32 M4 cell0.25m",
                      distances_from_source_centroid_m=np.linalg.norm(det.positions_m-source.centroid_m,axis=1).tolist())
    print(json.dumps(provenance), flush=True)
    reference_path = args.output / "direct.npz"
    if reference_path.exists():
        p.error("output already exists; preserve old results and choose another directory")
    start, cpu = perf_counter(), process_time()
    reference = kernel.transport_prompt(source, lh.PromptConfig(element_method="grouped", screen=False),
                                         compute_bins_for_all=True)
    direct_seconds, direct_cpu = perf_counter()-start, process_time()-cpu
    np.savez_compressed(reference_path, charge=reference.charge_orders_pe[:,1],
                        bins=reference.bins_orders_pe[:,:,1], origins=reference.time_origin_ns,
                        edges=reference.relative_time_edges_ns, q0=reference.charge_orders_pe[:,0])
    print("DIRECT", direct_seconds, "q1", reference.charge_orders_pe[:,1], flush=True)
    store = first_store(kernel, args.pack)
    # Reuse correct prompt only to satisfy the driver's interface/origin check.
    # Read ONLY column2 below: in this experimental invocation it carries q1.
    # No summed three-column result is saved or interpreted as a physical event.
    kernel.transport_prompt = lambda *a, **kw: reference
    for frequencies in args.frequencies:
        start, cpu = perf_counter(), process_time()
        response = transport_hybrid_basis(kernel, source,
                    lh.ShowerBasis(spectral_fusion=True), frequency_nodes=frequencies, store=store)
        elapsed, cpu_elapsed = perf_counter()-start, process_time()-cpu
        q, bins = response.charge_components_pe[:,2], response.components_pe[:,:,2]
        qref, bref = reference.charge_orders_pe[:,1], reference.bins_orders_pe[:,:,1]
        meta = dict(**provenance, frequencies=frequencies, application_seconds=elapsed,
                    application_cpu_seconds=cpu_elapsed, direct_seconds=direct_seconds,
                    direct_cpu_seconds=direct_cpu, q1_direct_pe=qref.tolist(), q1_cached_pe=q.tolist(),
                    charge_relative_error=(np.abs(q-qref)/np.maximum(np.abs(qref),1e-300)).tolist(),
                    bins_l1_relative_error=(np.abs(bins-bref).sum(1)/np.maximum(np.abs(bref).sum(1),1e-300)).tolist(),
                    negative_bin_mass_pe=np.maximum(-bins,0).sum(1).tolist(),
                    active=response.active.tolist(), timing_scope="first-call JIT may be included; first-pack preparation excluded",
                    metadata=response.metadata)
        np.savez_compressed(args.output/f"cached{frequencies}.npz", charge=q, bins=bins,
                            origins=response.time_origin_ns, spectrum=response.spectrum_ge2_pe,
                            omega=response.omega_per_ns, edges=response.relative_time_edges_ns)
        (args.output/f"cached{frequencies}.json").write_text(json.dumps(meta,indent=2))
        print("CACHED", json.dumps({k:v for k,v in meta.items() if k != "metadata"}),flush=True)


if __name__ == "__main__":
    main()
