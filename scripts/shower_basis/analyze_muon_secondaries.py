#!/usr/bin/env python3
"""Describe local secondary sources, without transport or an isotropic replacement.

Inputs and outputs: scripts/shower_basis/paths.py. Photon weights use all nine actual
BGVD quadrature wavelengths, physical beta thresholds and both source fields.
UID summaries describe particles, not cascade ancestry. Spatial bins are only a
diagnostic partition; their transport error is not established by this script.
"""
import paths  # inputs and outputs: scripts/shower_basis/paths.py
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from numpy.polynomial.legendre import legvander

C = 0.299792458


def capture_ancestry(path, output):
    """Regenerate one saved one-primary event and prove source identity."""
    from dataclasses import fields
    import ntsim
    from ntsim.LightTransport import bgvd_model_from_ntsim, source_from_event
    from ntsim.event_chain import Chain, EmptyEventSource
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["metadata"]))
        expected = {k: z[k] for k in z.files if k != "metadata"}
    model = bgvd_model_from_ntsim(ntsim.bgvd.Telescope(n_clusters=tuple(meta["clusters"])), ntsim.bgvd.Water())
    if model.fingerprint != meta["model_fingerprint"]:
        raise ValueError("current model differs from the recorded source-generation model")
    chain = Chain(
        ntsim.primary.ToyGen(energy=meta["energy_GeV"], pdg=13,
                            position=tuple(meta["position_m"]), direction=tuple(meta["direction"]), time=0.) << "primary_gen",
        ntsim.modules.ParticlePropagator(random_seed=meta["seed"], enable_cherenkov=False,
                                         cascade_bounds_e=(0., 0.), save_process_name=True) << "g4prop")
    event = next(chain(EmptyEventSource()))
    source = source_from_event(event, cone_model="spectral", max_phase_index=float(model.medium.phase_index.max())).elements
    difference = {k: float(np.max(np.abs(getattr(source, k)-v))) for k, v in expected.items()}
    for k, v in expected.items():
        np.testing.assert_array_equal(getattr(source, k), v, err_msg=f"regenerated source differs: {k}")
    assert len(event.tracks) == 1, "mapping below is deliberately only for one primary"
    tracks = next(iter(event.tracks.values()))
    uid = np.asarray(tracks.uid, np.int64)
    if tracks.metadata.get("parent_uid_space") == "ntsim":
        parent_uid = np.asarray(tracks.metadata["parent_uid"], np.int64)
    else:
        # Legacy NTSim used continuous_number=-1 for this one-primary event but
        # left parent IDs raw. Never subtract twice after the metadata fix.
        raw_parent = tracks.metadata.get("g4_parent_uid", tracks.metadata["parent_uid"])
        parent_uid = np.asarray(raw_parent, np.int64)-1
    unique, first = np.unique(uid, return_index=True)
    parents = dict(zip(unique.tolist(), parent_uid[first].tolist()))
    family = {0: 0}
    def find_family(key):
        path = []
        while key not in family:
            path.append(key)
            parent = parents.get(key)
            if parent is None:
                family[key] = -2
                break
            if parent == 0:
                family[key] = key
                break
            if parent < 0 or parent in path:
                raise ValueError(f"bad ancestry {path}, parent={parent}")
            key = parent
        answer = family[key]
        for child in path:
            family[child] = answer
        return answer
    family_uid = np.array([find_family(int(k)) for k in unique], np.int64)
    element_family = family_uid[np.searchsorted(unique, source.uid)]
    if tracks.metadata.get("parent_uid_space") == "ntsim":
        from ntsim.LightTransport.SourceFamilies import families_from_event
        partition = families_from_event(event, source, primary_uid=0)
        assert partition.complete, partition
        np.testing.assert_array_equal(partition.element_family_uid, element_family)
    process = np.asarray(tracks.metadata["process_name"]).astype(str)
    target = output / f"{path.stem}-ancestry.npz"
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, uid=unique, parent_uid=parent_uid[first],
                        family_uid=family_uid, pdgid=np.asarray(tracks.pdgid)[first],
                        start_m=np.asarray(tracks.pos_m)[first], start_ns=np.asarray(tracks.t_ns)[first],
                        energy_GeV=np.asarray(tracks.Etot_GeV)[first], creator=process[first],
                        element_family=element_family, element_pdgid=np.asarray(tracks.pdgid)[source.row_index],
                        metadata=json.dumps(dict(source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                                 source_difference=difference, unknown_families=int(np.sum(family_uid < 0)))))
    print(json.dumps(dict(output=str(target), source_difference=difference,
                         particles=len(unique), family_count=len(np.unique(element_family)),
                         unknown_families=int(np.sum(family_uid < 0)))), flush=True)


def ancestry_summary(source, path, wavelength, phase, quadrature):
    """Exact cascade descendants of each first secondary of the primary muon."""
    with np.load(path, allow_pickle=False) as z:
        ancestry = {k: z[k] for k in z.files if k != "metadata"}
        provenance = json.loads(str(z["metadata"]))
    density = source.photon_density_per_nm(wavelength, phase)
    density[source.beta[:, None]*phase <= 1.] = 0.
    weight = (density*quadrature).sum(1)
    mask = (source.uid != 0) & (weight > 0)
    families, inverse = np.unique(ancestry["element_family"][mask], return_inverse=True)
    q = np.bincount(inverse, weights=weight[mask])
    count = np.bincount(inverse)
    roots = np.searchsorted(ancestry["uid"], families)
    if np.any(families < 0) or not np.array_equal(ancestry["uid"][roots], families):
        raise ValueError("incomplete primary-secondary ancestry")
    roots_energy = ancestry["energy_GeV"][roots]
    roots_pdg = ancestry["pdgid"][roots]
    roots_process = ancestry["creator"][roots]
    delta = source.midpoints_m[mask]-ancestry["start_m"][roots][inverse]
    radius = np.linalg.norm(delta, axis=1)
    delay = ((source.start_ns[mask]+source.end_ns[mask])/2.-ancestry["start_ns"][roots][inverse]-radius/C)
    family_radius_rms = np.sqrt(np.bincount(inverse, weights=weight[mask]*radius**2)/q)
    family_delay_mean = np.bincount(inverse, weights=weight[mask]*delay)/q
    family_delay_rms = np.sqrt(np.bincount(inverse, weights=weight[mask]*(delay-family_delay_mean[inverse])**2)/q)
    family_direction = np.column_stack([np.bincount(inverse, weights=weight[mask]*source.direction[mask, k])/q
                                       for k in range(3)])
    family_direction /= np.maximum(np.linalg.norm(family_direction, axis=1)[:, None], 1e-30)
    axial = np.einsum("ij,ij->i", delta, family_direction[inverse])
    transverse = np.sqrt(np.maximum(radius**2-axial**2, 0.))
    family_transverse_rms = np.sqrt(np.bincount(inverse, weights=weight[mask]*transverse**2)/q)
    angle = np.einsum("ij,ij->i", source.direction[mask], family_direction[inverse])
    direction_polynomial = legvander(np.clip(angle, -1., 1.), 8)
    photon_moments = np.zeros((len(q), 9))
    for j, phase_value in enumerate(phase):
        cone_polynomial = legvander(np.minimum(1., 1./(source.beta[mask]*phase_value)), 8)
        for l in range(9):
            photon_moments[:, l] += np.bincount(inverse, weights=density[mask, j]*quadrature[j]*
                                                direction_polynomial[:, l]*cone_polynomial[:, l], minlength=len(q))
    photon_moments /= q[:, None]
    rank = np.argsort(q)[::-1]
    process_summary = {}
    for process in np.unique(roots_process):
        choose = roots_process == process
        process_summary[str(process)] = dict(families=int(choose.sum()), light_fraction=float(q[choose].sum()/q.sum()),
                                             root_energy_GeV_photon_quantiles=quantiles(roots_energy[choose], q[choose]),
                                             elements=int(count[choose].sum()))
    return dict(provenance=provenance, families=len(families),
                charged_secondary_fraction_pdg={str(p): float(weight[mask][ancestry["element_pdgid"][mask]==p].sum()/q.sum())
                                               for p in np.unique(ancestry["element_pdgid"][mask])},
                process_summary=process_summary,
                root_energy_GeV_photon_quantiles=quantiles(roots_energy, q),
                light_fraction_root_energy_above_GeV={str(e): float(q[roots_energy>e].sum()/q.sum()) for e in (.01,.1,1.,10.,100.)},
                families_root_energy_above_GeV={str(e): int(np.sum(roots_energy>e)) for e in (.01,.1,1.,10.,100.)},
                elements_root_energy_above_GeV={str(e): int(count[roots_energy>e].sum()) for e in (.01,.1,1.,10.,100.)},
                family_light_fraction_top={str(k): float(q[rank[:k]].sum()/q.sum()) for k in (1,5,10,50,100,1000)},
                family_radius_rms_m_photon_quantiles=quantiles(family_radius_rms,q),
                family_transverse_rms_m_photon_quantiles=quantiles(family_transverse_rms,q),
                family_delay_rms_ns_photon_quantiles=quantiles(family_delay_rms,q),
                element_delay_ns_photon_quantiles=quantiles(delay,weight[mask],(.5,.9,.99,.999,.9999)),
                photon_fraction_delay_above_ns={str(t): float(weight[mask][delay>t].sum()/q.sum())
                                                for t in (.1,1.,10.,100.,1000.)},
                root_energy_groups={f"{lo}-{hi}": dict(
                    families=int(np.sum((roots_energy>=lo)&(roots_energy<hi))),
                    light_fraction=float(q[(roots_energy>=lo)&(roots_energy<hi)].sum()/q.sum()),
                    angular_moments_about_family_axis=np.average(photon_moments[(roots_energy>=lo)&(roots_energy<hi)],
                                                                  axis=0, weights=q[(roots_energy>=lo)&(roots_energy<hi)]).tolist(),
                    radius_rms_m_photon_quantiles=quantiles(family_radius_rms[(roots_energy>=lo)&(roots_energy<hi)],
                                                          q[(roots_energy>=lo)&(roots_energy<hi)]))
                    for lo, hi in ((0.,.01),(.01,.1),(.1,1.),(1.,10.),(10.,100.),(100.,10000.))
                    if np.any((roots_energy>=lo)&(roots_energy<hi))},
                top50=[dict(family_uid=int(families[k]), pdgid=int(roots_pdg[k]), creator=str(roots_process[k]),
                            energy_GeV=float(roots_energy[k]), photons=float(q[k]), light_fraction=float(q[k]/q.sum()),
                            elements=int(count[k]), radius_rms_m=float(family_radius_rms[k]),
                            delay_mean_ns=float(family_delay_mean[k]), delay_rms_ns=float(family_delay_rms[k]),
                            transverse_rms_m=float(family_transverse_rms[k]),
                            photon_angular_moments=photon_moments[k].tolist(),
                            axis=family_direction[k].tolist(),
                            start_m=ancestry["start_m"][roots[k]].tolist()) for k in rank[:50]])


def quantiles(x, weights, probabilities=(0.5, 0.9, 0.99)):
    order = np.argsort(x)
    cumulative = np.cumsum(weights[order])
    return np.asarray(x)[order[np.minimum(np.searchsorted(cumulative, np.asarray(probabilities)*cumulative[-1]),
                                         len(order)-1)]].tolist()


def mean(x, weights):
    return np.sum(np.asarray(x) * weights) / np.sum(weights)


def angular_moments(source, mask, axis, weights, phases, degree=8):
    cosine = np.clip(source.direction[mask] @ axis, -1., 1.)
    direction = legvander(cosine, degree)
    result = []
    for j, phase in enumerate(phases):
        ring = legvander(np.minimum(1., 1. / (source.beta[mask] * phase)), degree)
        result.append(np.sum(direction * ring * weights[mask, j, None], axis=0)
                      / weights[mask, j].sum())
    return np.asarray(result)


def source_summary(source, meta, wavelength, phase, quadrature, *, is_muon):
    mid = source.midpoints_m
    time = (source.start_ns + source.end_ns) / 2.
    density = source.photon_density_per_nm(wavelength, phase)
    density[source.beta[:, None] * phase[None, :] <= 1.] = 0.
    if density.min() < -1e-10:
        raise ValueError("negative physical spectral density")
    spectral = density * quadrature[None, :]
    weight = spectral.sum(axis=1)
    primary = source.uid == 0
    if is_muon:
        origin = source.start_m[primary][0]
        end = source.start_m[primary][-1] + source.length_m[primary][-1] * source.direction[primary][-1]
        axis = (end - origin) / np.linalg.norm(end - origin)
        primary_projection = (mid[primary] - origin) @ axis
        order = np.argsort(primary_projection)
        s = (mid - origin) @ axis
        local_center = np.column_stack([np.interp(s, primary_projection[order], mid[primary][order, k])
                                        for k in range(3)])
        local_time = np.interp(s, primary_projection[order], time[primary][order])
        delta = mid - local_center
        rho = np.linalg.norm(delta - (delta @ axis)[:, None] * axis, axis=1)
        delay = time - local_time
        selected = ~primary
    else:
        axis = source.principal_axis
        if axis @ np.average(source.direction, axis=0, weights=weight) < 0:
            axis = -axis
        origin = np.average(mid, axis=0, weights=weight)
        s = (mid - origin) @ axis
        rho = np.linalg.norm(mid - origin - s[:, None] * axis, axis=1)
        delay = time - s / C
        delay -= mean(delay, weight)
        selected = np.ones(len(source), bool)
    uid, inv = np.unique(source.uid[selected], return_inverse=True)
    uid_weight = np.bincount(inv, weights=weight[selected])
    uid_length = np.bincount(inv, weights=source.length_m[selected])
    uid_steps = np.bincount(inv)
    ranked = np.argsort(uid_weight)[::-1]
    subset_weight = weight[selected]
    total = subset_weight.sum()
    moments = angular_moments(source, selected, axis, spectral, phase)
    result = dict(elements=len(source), selected_elements=int(selected.sum()),
                  selected_uids=len(uid), selected_photon_fraction=float(total / weight.sum()),
                  total_emitted_photons=float(weight.sum()),
                  selected_emitted_photons=float(total), axis=axis.tolist(),
                  axial_span_m=float(np.ptp(s[selected])),
                  secondary_rho_m_quantiles=quantiles(rho[selected], subset_weight),
                  delay_ns_quantiles=quantiles(delay[selected], subset_weight, (0.01, 0.5, 0.9, 0.99)),
                  uid_photon_weighted_length_m_quantiles=quantiles(uid_length, uid_weight),
                  uid_photon_weighted_steps_quantiles=quantiles(uid_steps, uid_weight),
                  uid_light_fraction_top={str(k): float(uid_weight[ranked[:k]].sum()/total)
                                          for k in (1, 10, 100, 1000)},
                  uid_light_fraction_length_below={str(k): float(uid_weight[uid_length < k].sum()/total)
                                                   for k in (0.01, 0.1, 1., 5.)},
                  photon_legendre_moments_by_wavelength=moments.tolist(),
                  photon_legendre_moments_spectral_average=(moments * spectral[selected].sum(0)[:, None]
                                                            / total).sum(0).tolist())
    # Physical energy is not recoverable without PDG labels. The beta-only
    # electron-equivalent diagnostic is named explicitly, never called energy.
    gamma = 1. / np.sqrt(np.maximum(1. - source.beta[selected] ** 2, 1e-30))
    electron_equivalent = 0.00051099895 * (gamma - 1.)
    result["electron_equivalent_step_kinetic_GeV_quantiles"] = quantiles(electron_equivalent, subset_weight)
    result["electron_equivalent_light_fraction_step_energy_above_GeV"] = {
        str(k): float(subset_weight[electron_equivalent > k].sum()/total) for k in (.01, .1, 1., 10.)}
    result["axial_partitions"] = {}
    for width in (1., 2., 5., 10., 20.):
        labels = np.floor((s[selected] - s.min()) / width).astype(int)
        total_bins = labels.max() + 1
        bweight = np.bincount(labels, weights=subset_weight, minlength=total_bins)
        populated = bweight > 0
        rank = np.argsort(bweight)[::-1]
        n = int(populated.sum())
        cell_mean_delay = np.divide(np.bincount(labels, weights=subset_weight*delay[selected], minlength=total_bins),
                                    bweight, out=np.zeros(total_bins), where=populated)
        cell_time_rms = np.sqrt(np.divide(np.bincount(labels, weights=subset_weight*(delay[selected]-cell_mean_delay[labels])**2,
                                                       minlength=total_bins), bweight, out=np.zeros(total_bins), where=populated))
        cell_rho_rms = np.sqrt(np.divide(np.bincount(labels, weights=subset_weight*rho[selected]**2, minlength=total_bins),
                                        bweight, out=np.zeros(total_bins), where=populated))
        cosine = source.direction[selected] @ axis
        p = legvander(cosine, 8)
        pm = np.zeros((total_bins, 9))
        for j, phase_value in enumerate(phase):
            ring = legvander(np.minimum(1., 1./(source.beta[selected]*phase_value)), 8)
            for l in range(9):
                pm[:, l] += np.bincount(labels, weights=p[:, l]*ring[:, l]*spectral[selected, j], minlength=total_bins)
        pm[populated] /= bweight[populated, None]
        result["axial_partitions"][str(width)] = dict(
            populated_bins=n,
            photon_fraction_brightest_bins={str(k): float(bweight[rank[:k]].sum()/total) for k in (1, 5, 10, 50)},
            photon_fraction_brightest_fraction_bins={str(f): float(bweight[rank[:max(1, int(n*f))]].sum()/total)
                                                    for f in (.01, .05, .1, .2)},
            cell_time_rms_ns_photon_quantiles=quantiles(cell_time_rms[populated], bweight[populated]),
            cell_rho_rms_m_photon_quantiles=quantiles(cell_rho_rms[populated], bweight[populated]),
            bright10_moments=np.average(pm[rank[:10]], axis=0, weights=bweight[rank[:10]]).tolist(),
            other_moments=np.average(pm[rank[10:]], axis=0, weights=bweight[rank[10:]]).tolist()
                if bweight[rank[10:]].sum() else None,
            top20=[dict(s_m=float(s.min()+(k+.5)*width), photons=float(bweight[k]),
                        photon_fraction=float(bweight[k]/total), time_rms_ns=float(cell_time_rms[k]),
                        rho_rms_m=float(cell_rho_rms[k]), moments=pm[k].tolist()) for k in rank[:20]])
    result["definitions"] = {
        "weights": "Emitted photons integrated over actual BGVD 9-node wavelength quadrature; no QE or optical attenuation.",
        "angular": "Photon Legendre moments P_l(track direction dot event axis)*P_l(1/(beta*n(lambda))), exact ring-azimuth average.",
        "delay": "Muon: midpoint emission time minus local primary midpoint interpolated time. Shower: t-s/c, centered by photon mean.",
        "rho": "Muon: distance from interpolated actual primary centerline. Shower: distance from photon PCA axis.",
        "uid": "A charged particle ID is not a cascade family. No ancestry is reconstructed from spatial bins.",
        "energy": "electron-equivalent energy from beta is only diagnostic because the NPZ has no PDG labels."}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, nargs="+", default=[paths.muon_source(e) for e in (300, 1000, 3000)])
    parser.add_argument("--output", type=Path, default=paths.work("muon-segments-20261009/structure"))
    parser.add_argument("--shower-events", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--capture-source", type=Path,
                        help="Regenerate one source, prove identity, save particle ancestry, then exit; one G4 manager per process.")
    parser.add_argument("--skip-showers", action="store_true")
    parser.add_argument("--capture-electron-energy", type=float,
                        help="Generate a same-physics-list electron reference in this separate process and summarize.")
    args = parser.parse_args()
    if args.capture_source:
        capture_ancestry(args.capture_source, args.output)
        return
    import lighthit as lh
    import ntsim
    from ntsim.LightTransport import bgvd_model_from_ntsim
    model = bgvd_model_from_ntsim(ntsim.bgvd.Telescope(n_clusters=(1,2,3,4,5)), ntsim.bgvd.Water())
    quadrature = lh.WavelengthQuadrature.gauss_legendre(*model.medium.wavelength_range_nm, nodes=9)
    wavelength = quadrature.wavelength_nm
    phase = model.medium.sample(wavelength)["phase_index"]
    args.output.mkdir(parents=True, exist_ok=True)
    if args.capture_electron_energy:
        from dataclasses import fields
        from ntsim.LightTransport import source_from_event
        from ntsim.event_chain import Chain, EmptyEventSource
        chain = Chain(ntsim.primary.ToyGen(energy=args.capture_electron_energy, pdg=11,
                                           position=(-100., -100., 350.), direction=(0.,1.,0.), time=0.) << "primary_gen",
                      ntsim.modules.ParticlePropagator(random_seed=42, enable_cherenkov=False,
                                                       cascade_bounds_e=(0.,0.)) << "g4prop")
        event = next(chain(EmptyEventSource()))
        source = source_from_event(event, cone_model="spectral", max_phase_index=float(model.medium.phase_index.max())).elements
        stem = f"electron-current-{args.capture_electron_energy:g}GeV"
        np.savez_compressed(args.output/f"{stem}.npz", **{f.name:getattr(source,f.name) for f in fields(source)
                                                       if isinstance(getattr(source,f.name),np.ndarray)})
        result = source_summary(source, {}, wavelength, phase, quadrature.weight_nm, is_muon=False)
        result["generation"] = dict(energy_GeV=args.capture_electron_energy, seed=42, mode_custom_physlist="fast",
                                     cascade_bounds_e=[0.,0.], position_m=[-100.,-100.,350.],direction=[0.,1.,0.])
        (args.output/f"{stem}.json").write_text(json.dumps(result,indent=2))
        print(stem, "elements",len(source),"moments",result["photon_legendre_moments_spectral_average"],flush=True)
        return
    summary = dict(model_fingerprint=model.fingerprint, wavelength_nm=wavelength.tolist(),
                   phase_index=phase.tolist(), quadrature_weight_nm=quadrature.weight_nm.tolist(), inputs={})
    for path in args.sources:
        begin = perf_counter()
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["metadata"]))
            source = lh.SpectralLightElements(**{k: z[k] for k in z.files if k != "metadata"}, cone_model="spectral")
        result = source_summary(source, meta, wavelength, phase, quadrature.weight_nm, is_muon=True)
        ancestry_path = args.output / f"{path.stem}-ancestry.npz"
        if ancestry_path.exists():
            result["true_secondary_families"] = ancestry_summary(source, ancestry_path, wavelength, phase, quadrature.weight_nm)
        result.update(source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), metadata=meta)
        name = path.stem
        (args.output / f"{name}.json").write_text(json.dumps(result, indent=2))
        summary["inputs"][name] = str(path)
        print(name, "seconds", perf_counter()-begin, "secondary fraction", result["selected_photon_fraction"], flush=True)
    # Showers: the bundle 100 GeV electron (Geant4 HDF5, events --shower-events) and the
    # regenerated 1 TeV electron (one NTSim event, experiments/regenerate_inputs.py).
    for energy in ("100GeV", "1TeV"):
        if args.skip_showers:
            break
        path = (paths.ELECTRON_100GEV.path() if energy == "100GeV"
                else paths.inputs() / "electron-1TeV-steps.npz")
        for event in (args.shower_events if energy == "100GeV" else [0]):
            begin = perf_counter()
            if energy == "100GeV":
                source = lh.G4Shower.from_hdf5(path, event=event,
                                               reference_phase_index=float(phase.max())).elements
            else:
                with np.load(path, allow_pickle=False) as z:
                    source = lh.SpectralLightElements(**{k: z[k] for k in z.files if k != "metadata"})
            result = source_summary(source, {}, wavelength, phase, quadrature.weight_nm, is_muon=False)
            result.update(input_path=str(path), event=event, source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            name = f"shower{energy}-event{event}"
            (args.output / f"{name}.json").write_text(json.dumps(result, indent=2))
            summary["inputs"][name] = str(path)
            print(name, "seconds", perf_counter()-begin, flush=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
