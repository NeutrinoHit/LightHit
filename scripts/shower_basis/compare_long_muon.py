#!/usr/bin/env python3
"""Compare saved long-muon responses on identical OMs, origins and bins."""
import argparse
import json
from pathlib import Path

import numpy as np


def compare(reference, candidate):
    def read(path):
        with np.load(path, allow_pickle=False) as z:
            if "charge" in z:
                return {k: z[k] for k in z.files}
            viewer = json.loads((path.parent / "viewer.json").read_text())
            return dict(charge=z["charge_components_pe"], bins=z["components_pe"],
                        active=z["active"], origin=z["time_origin_ns"],
                        edges=z["relative_time_edges_ns"],
                        positions=np.asarray(viewer["detector"]["positions_m"]),
                        native_metadata=json.loads(str(z["metadata"])))
    a, b = read(reference), read(candidate)
    for key in ("positions", "edges"):
        np.testing.assert_array_equal(a[key], b[key])
    np.testing.assert_allclose(a["origin"], b["origin"], rtol=0, atol=1e-10)
    union = a["active"] | b["active"]
    bright = np.flatnonzero(union)
    bright = bright[np.argsort(-a["charge"][bright].sum(1))[:10]]
    result = dict(reference=str(reference), candidate=str(candidate),
                  active_reference=int(a["active"].sum()), active_candidate=int(b["active"].sum()),
                  lost=int(np.sum(a["active"] & ~b["active"])),
                  gained=int(np.sum(b["active"] & ~a["active"])), bright_indices=bright.tolist())
    for name, columns in (("0", [0]), ("1", [1]), (">=2", [2]), ("total", [0, 1, 2])):
        qa, qb = a["charge"][:, columns].sum(1), b["charge"][:, columns].sum(1)
        ba, bb = a["bins"][:, :, columns].sum(2), b["bins"][:, :, columns].sum(2)
        dq, db = abs(qb-qa), abs(bb-ba)
        result[name] = dict(
            charge_reference_pe=float(qa.sum()), charge_candidate_pe=float(qb.sum()),
            active_charge_reference_pe=float(qa[a["active"]].sum()),
            active_charge_candidate_pe=float(qb[b["active"]].sum()),
            charge_max_abs_pe=float(dq.max()), bins_max_abs_pe=float(db.max()),
            charge_max_abs_active_pe=float(np.max(dq[union], initial=0.)),
            bins_l1_max_abs_active_pe=float(np.max(db[union].sum(1), initial=0.)),
            bins_l1_max_abs_pe=float(db.sum(1).max()),
            charge_max_rel_bright=float(np.max(dq[bright]/np.maximum(abs(qa[bright]), 1e-30), initial=0.)),
            bins_l1_max_rel_bright=float(np.max(db[bright].sum(1)/np.maximum(abs(ba[bright]).sum(1), 1e-30), initial=0.)),
            charge_max_rel_active=float(np.max(dq[union]/np.maximum(abs(qa[union]), 1e-30), initial=0.)),
            bins_l1_max_rel_active=float(np.max(db[union].sum(1)/np.maximum(abs(ba[union]).sum(1), 1e-30), initial=0.)),
            bins_l1_global=float(db.sum()/max(abs(ba).sum(), 1e-30)),
            reference_negative_charge_count=int(np.sum(qa < -1e-12)),
            reference_negative_bin_mass_pe=float(np.maximum(-ba, 0).sum()),
            candidate_negative_bin_mass_pe=float(np.maximum(-bb, 0).sum()),
            reference_window_charge_pe=float(ba.sum()), candidate_window_charge_pe=float(bb.sum()))
    for label, path, arrays in (("reference", reference, a), ("candidate", candidate, b)):
        meta = json.loads(path.with_suffix(".json").read_text())
        stages = meta.get("metadata", arrays.get("native_metadata", {}))
        result[label+"_timing"] = {"wall_seconds": meta.get("wall_seconds", meta.get("elapsed_seconds")),
                                   "peak_rss_mib": meta.get("peak_rss_mib"),
                                   "threads": meta.get("threads"),
                                   "reused_prompt": meta.get("reused_prompt", False),
                                   "stages": {k: v for k, v in stages.items() if k.endswith("seconds")}}
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("reference", type=Path)
    p.add_argument("candidate", type=Path)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    result = compare(args.reference, args.candidate)
    text = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(text)
    print(text)
