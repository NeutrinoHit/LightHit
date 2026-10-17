#!/usr/bin/env python3
"""Diagnostic coarse >=2 charge for screening; never a physical error bound.

Uses the unchanged 65-frequency prepared operator at omega=0, truncating only
the compiled source degree and spatial/azimuthal representation. Prompt and
full-frequency transport are not evaluated. Signed estimates remain signed.
"""
import paths  # inputs and outputs: scripts/shower_basis/paths.py
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import signal
from time import perf_counter, process_time

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--energy", type=int, default=1000)
    p.add_argument("--degree", type=int, default=8)
    p.add_argument("--cell-m", type=float, default=2.)
    p.add_argument("--cpu-limit", type=float, default=60.)
    p.add_argument("--fused", action="store_true", help="Share geometry across bands with identical source points")
    p.add_argument("--output", type=Path, default=paths.work("muon-segments-20261009/charge-screen/1000-l8"))
    args = p.parse_args()
    import lighthit as lh
    import numba
    import ntsim
    from ntsim.LightTransport import bgvd_model_from_ntsim
    from lighthit.experimental.axial_source import AxisFrame
    from lighthit._shower_basis import compile_axial_slices, apply_multi, apply_multi_spectral
    from lighthit.shower_basis import Ge2KernelStore
    if numba.get_num_threads() != 1:
        p.error("requires NUMBA_NUM_THREADS=1")
    if not 0 <= args.degree <= 32:
        p.error("source degree must be between 0 and 32")
    base = paths.work("muon-five-cluster-20261008")
    sourcefile = paths.muon_source(args.energy)
    with np.load(sourcefile) as z:
        source = lh.SpectralLightElements(**{k:z[k] for k in z.files if k != "metadata"}, cone_model="spectral")
        source_meta = json.loads(str(z["metadata"]))
    referencefile = paths.muon_response(args.energy)
    with np.load(referencefile) as z:
        reference_charge = z["charge_components_pe"].copy()
        reference_active = z["active"].copy()
        reference_meta = json.loads(str(z["metadata"]))
    model = bgvd_model_from_ntsim(ntsim.bgvd.Telescope(n_clusters=(1,2,3,4,5)), ntsim.bgvd.Water())
    if model.fingerprint != source_meta["model_fingerprint"]:
        p.error("model differs from saved source provenance")
    kernel = model.kernel(lh.KernelConfig(omega_per_ns=np.linspace(0.,1.2,65),
                          cache_directory=str(paths.RTE_CACHE.path()),
                          cache_policy="require"))
    store = Ge2KernelStore.load(str(paths.GE2_PACK.path()))
    full = store.bundle(kernel, np.arange(65))
    bundle = dict(full, degree=args.degree,
                  coefficients=full["coefficients"][:, :, :args.degree+1, :, :, :1],
                  omega=full["omega"][:1])
    assert np.shares_memory(bundle["coefficients"], full["coefficients"])
    assert bundle["coefficients"].shape[2] == args.degree+1
    assert np.array_equal(bundle["omega"], [0.])
    wavelengths = kernel.wavelength.wavelength_nm
    weights = kernel.wavelength.weight_nm
    sample = kernel.medium.sample(wavelengths)
    max_phase = float(np.max(sample["phase_index"]))
    valid = source.beta*max_phase > 1
    if not np.all(valid):
        source = source.subset(valid)
    pieces = [("primary", source.subset(source.uid == 0)),
              ("secondary", source.subset(source.uid != 0))]
    frames = {name: AxisFrame.of(piece.field(0, phase_index=max_phase)) for name,piece in pieces}
    positions = model.detector.positions_m
    looks = np.ascontiguousarray(-model.detector.orientations)
    alpha = np.ascontiguousarray(kernel.acceptance()["alpha"], float)
    areas = np.asarray(model.detector.effective_area_m2)
    efficiency = np.asarray(model.detector.spectral_weight(wavelengths))
    low, high = kernel.config.radial_range_m
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/"charge.npz").exists():
        p.error("preserve previous output; choose another directory")

    def one_band(lam):
        return dict(bundle, coefficients=bundle["coefficients"][lam:lam+1],
                    absorption=bundle["absorption"][lam:lam+1], speed=bundle["speed"][lam:lam+1])

    def field_for(piece, lam):
        phase = float(sample["phase_index"][lam]); wavelength = float(wavelengths[lam])
        mask = piece.beta*phase > 1
        if not np.any(mask):
            return None
        subset = piece if np.all(mask) else piece.subset(mask)
        first, second = subset.field(0, phase_index=phase), subset.field(2, phase_index=phase)
        return replace(first, photons=np.ascontiguousarray(
            first.photons/wavelength**2-second.photons/(wavelength**2*phase**2)))

    def compile_field(field, frame):
        return compile_axial_slices(field, args.degree, np.array([0.]),
                azimuthal_degree=0, cell_m=args.cell_m, axial_order=3,
                transverse="stencil", stencil_m=.2, stencil_threshold_m=.03,
                time_step_ns=.02, frame=frame, channel_dtype=np.complex64)

    warm_begin = perf_counter()
    tiny = pieces[0][1].subset(np.arange(len(pieces[0][1])) == 0)
    warm = compile_field(field_for(tiny,0), frames["primary"])
    apply_multi(one_band(0), warm, warm, positions[:1], looks[:1], alpha,
                 np.ones(1),np.zeros(1), (weights[0]*efficiency[0]*areas[:1])[None,:],
                 radial_window_m=(low,high))
    if args.fused:
        apply_multi_spectral(one_band(0), [warm], positions[:1], looks[:1], alpha,
                   (weights[0]*efficiency[0]*areas[:1])[None,:], radial_window_m=(low,high))
    warm_seconds = perf_counter()-warm_begin
    rows=[];charge=np.zeros((len(positions),2));t0,c0=perf_counter(),process_time()

    def stop(signum, frame):
        raise TimeoutError("coarse charge CPU budget exceeded")
    signal.signal(signal.SIGPROF, stop)
    signal.setitimer(signal.ITIMER_PROF, args.cpu_limit)
    complete=False
    pending = {0: [], 1: []}
    try:
        for lam in range(len(wavelengths)):
            for component,(name,piece) in enumerate(pieces):
                step=perf_counter();field=field_for(piece,lam)
                if field is None:
                    continue
                compiled=compile_field(field,frames[name]);compiled_at=perf_counter()
                if args.fused:
                    pending[component].append((lam,compiled))
                    rows.append(dict(wavelength_nm=float(wavelengths[lam]),component=name,
                           elements=len(field),source_points=len(compiled.points_m()),
                           compile_seconds=compiled_at-step,apply_seconds=0.))
                    continue
                result=apply_multi(one_band(lam), compiled,compiled,positions,looks,alpha,
                       np.ones(1),np.zeros(1),(weights[lam]*efficiency[lam]*areas)[None,:],
                       radial_window_m=(low,high))
                charge[:,component]+=result[0].real
                item=dict(wavelength_nm=float(wavelengths[lam]),component=name,
                          elements=len(field),source_points=len(compiled.points_m()),
                          compile_seconds=compiled_at-step,apply_seconds=perf_counter()-compiled_at)
                rows.append(item);print(json.dumps(item),flush=True)
        if args.fused:
            for component, entries in pending.items():
                groups=[]
                for lam, compiled in entries:
                    if (not groups or lam != groups[-1][-1][0]+1
                            or not np.array_equal(compiled.points_m(),groups[-1][0][1].points_m())):
                        groups.append([])
                    groups[-1].append((lam,compiled))
                for group in groups:
                    lo,hi=group[0][0],group[-1][0]+1
                    band=dict(bundle,coefficients=bundle["coefficients"][lo:hi],
                              absorption=bundle["absorption"][lo:hi],speed=bundle["speed"][lo:hi])
                    scales=np.stack([weights[lam]*efficiency[lam]*areas for lam,_ in group])
                    start=perf_counter()
                    result=apply_multi_spectral(band,[source for _,source in group],
                          positions,looks,alpha,scales,radial_window_m=(low,high))
                    charge[:,component]+=result[0].real
                    item=dict(component=pieces[component][0],fused_bands=len(group),
                              source_points=len(group[0][1].points_m()),
                              compile_seconds=0.,apply_seconds=perf_counter()-start)
                    rows.append(item);print(json.dumps(item),flush=True)
        complete=True
    except TimeoutError:
        print("CPU limit reached; incomplete estimate is not used for screening",flush=True)
    finally:
        signal.setitimer(signal.ITIMER_PROF,0)
    wall,cpu=perf_counter()-t0,process_time()-c0
    report=dict(complete=complete,wall_seconds=wall,cpu_seconds=cpu,warm_seconds=warm_seconds,
          source_sha256=hashlib.sha256(sourcefile.read_bytes()).hexdigest(),
          reference_sha256=hashlib.sha256(referencefile.read_bytes()).hexdigest(),
          energy_GeV=args.energy,modules=len(positions),source_degree=args.degree,
          azimuthal_degree=0,cell_m=args.cell_m,radial_window_m=[low,high],
          fused_spectral_geometry=args.fused,
          model_fingerprint=model.fingerprint,threads=numba.get_num_threads(),
          coefficients_share_existing_65_pack=True,omega_per_ns=[0.],wavelengths_nm=wavelengths.tolist(),
          scope="empirical coarse source approximation; no rigorous screening/error bound",stages=rows)
    if complete:
        coarse=charge.sum(1);fine=reference_charge[:,2];delta=fine-coarse
        report.update(negative_coarse_modules=int((coarse<0).sum()),minimum_coarse_pe=float(coarse.min()),
            maximum_underestimate_pe=float(delta.max()),maximum_overestimate_pe=float((-delta).max()),
            maximum_relative_underestimate_where_fine_ge_1e_4=float(np.max(
                delta[fine>=1e-4]/fine[fine>=1e-4],initial=0.)),
            reference_active_count=int(reference_active.sum()),
            inactive_reference_q1="may be the old coarse prompt estimate; active reference q1 is fine",
            screens={})
        for margin in (2.,4.,6.):
            candidates=reference_charge[:,0]+margin*(coarse+reference_charge[:,1])>=.01
            report["screens"][str(margin)]=dict(candidates=int(candidates.sum()),
                    missed_old_active=np.flatnonzero(reference_active&~candidates).tolist(),
                    additional_vs_old_active=np.flatnonzero(candidates&~reference_active).tolist())
        np.savez_compressed(args.output/"charge.npz",coarse_components=charge,coarse_ge2=coarse,
                            reference_charge=reference_charge,reference_active=reference_active,
                            positions=positions)
    (args.output/"report.json").write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='stages'},indent=2),flush=True)


if __name__ == "__main__":
    main()
