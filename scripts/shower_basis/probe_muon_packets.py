#!/usr/bin/env python3
"""Compare detailed quasi-shower packets with the same secondary source sum."""
import paths  # inputs and outputs: scripts/shower_basis/paths.py
import argparse
from dataclasses import replace
import json
from pathlib import Path
from time import perf_counter

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--energy", type=int, default=1000)
    p.add_argument("--om-indices", type=int, nargs="+", default=[196, 197, 198, 1241])
    p.add_argument("--radius", type=float, default=400.)
    p.add_argument("--soft-cell", type=float, default=5.)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    import lighthit as lh
    import ntsim
    from ntsim.LightTransport import bgvd_model_from_ntsim
    from lighthit.experimental.muon_packets import partition_muon, order1_packets
    from lighthit.experimental.muon_integral import order1_integrals
    base = paths.work("muon-five-cluster-20261008")
    with np.load(paths.muon_source(args.energy), allow_pickle=False) as z:
        source = lh.SpectralLightElements(**{k: z[k] for k in z.files if k != "metadata"}, cone_model="spectral")
    with np.load(paths.muon_ancestry(args.energy)) as z:
        ancestry = {k: z[k] for k in z.files if k != "metadata"}
    packets = partition_muon(source, ancestry, soft_cell_m=args.soft_cell)
    reference = paths.muon_response(args.energy)
    with np.load(reference) as z:
        ids = np.asarray(args.om_indices)
        origins = z["time_origin_ns"][ids]
    model = bgvd_model_from_ntsim(ntsim.bgvd.Telescope(n_clusters=(1,2,3,4,5)), ntsim.bgvd.Water())
    d = model.detector
    model = replace(model, detector=replace(d, positions_m=d.positions_m[ids], orientations=d.orientations[ids],
                     effective_area_m2=d.effective_area_m2[ids], identifiers={k: np.asarray(v)[ids] for k,v in d.identifiers.items()}))
    kernel = model.kernel(lh.KernelConfig(omega_per_ns=np.linspace(0.,1.2,65), cache_policy="require"))
    settings = lh.PromptConfig(element_method="grouped", screen=False, batch_charge_tolerance_pe=1e-6)
    args.output.mkdir(parents=True, exist_ok=True)
    print("packet_counts", {kind:sum(x.kind==kind for x in packets) for kind in set(x.kind for x in packets)}, flush=True)
    start = perf_counter()
    qa, ba, ma = order1_integrals(kernel, source.subset(source.uid != 0), time_origin_ns=origins, config=settings)
    reference_seconds = perf_counter()-start
    print("reference_seconds", reference_seconds, flush=True)
    qb, bb, mb = order1_packets(kernel, source, packets, time_origin_ns=origins, radius_m=args.radius, config=settings)
    result = dict(energy=args.energy, om_indices=ids.tolist(), reference_seconds=reference_seconds,
                   packet_seconds=mb["elapsed_seconds"], reference_charge_pe=qa.tolist(), packet_charge_pe=qb.tolist(),
                   charge_relative_error=(abs(qb-qa)/np.maximum(abs(qa),1e-30)).tolist(),
                   bins_relative_l1=(abs(bb-ba).sum(1)/np.maximum(abs(ba).sum(1),1e-30)).tolist(), metadata=mb)
    np.savez_compressed(args.output/"responses.npz", reference_charge=qa, reference_bins=ba, packet_charge=qb, packet_bins=bb)
    (args.output/"comparison.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='metadata'},indent=2),flush=True)


if __name__ == "__main__":
    main()
