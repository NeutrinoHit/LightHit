"""Muons with showers: basis versus the hybrid's own cell source.

The default ``transport_hybrid`` rejects a 100 m muon at its conservative
radial check (centroid distance +- extent).  The reference here is the same
cloud-in-cell source (``KernelConfig.cell_m``) run through the basis driver
(``representation="cells"``), whose radial check is exact per point; for
showers this driver reproduces ``transport_hybrid`` (see the benchmark).

Args: kind events(comma) [basis json list]
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from common import lh, model, config, shower
from lighthit.shower_basis import transport_hybrid_basis

kind = sys.argv[1]
events = [int(e) for e in sys.argv[2].split(",")]
variants = json.loads(sys.argv[3]) if len(sys.argv) > 3 else [{}]
k = model().kernel(config())
store = k.ge2_kernel_store()
for event in events:
    source = shower(kind, event)
    runs = [("cells", {})] + [("basis", v) for v in variants]
    for representation, options in runs:
        label = representation if representation == "cells" else "basis_" + "_".join(
            f"{a}{b}" for a, b in sorted(options.items())) if options else "basis"
        path = f"muon_{label}_{kind}_{event}.npz"
        if os.path.exists(path):
            continue
        started = time.time()
        response = transport_hybrid_basis(k, source, lh.ShowerBasis(**options), store=store,
                                          representation=representation)
        seconds = time.time() - started
        np.savez(path, charge=response.charge_components_pe, bins=response.components_pe,
                 active=response.active, origin=response.time_origin_ns)
        meta = response.metadata
        print(json.dumps({"kind": kind, "event": event, "variant": label, "seconds": seconds,
                          **{key: round(value, 2) for key, value in meta.items()
                             if key.endswith("seconds")},
                          "points": meta["ge2_basis_summary"]["cells"],
                          "active": meta["active_modules"]}), flush=True)
