# ruff: noqa: F403, F405  (prototype scripts share the helpers of common/proto)
"""Run transport_hybrid(ge2_basis=...) on several events in one process.

Args: kind events(comma) out_prefix [basis json] [store dir]
Prints per-event timings; the first event includes preparing the order >=2
kernel store unless a saved pack is given.
"""
import os
import sys
import time
import json
import resource
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from common import *
kind, events, prefix = sys.argv[1], [int(e) for e in sys.argv[2].split(",")], sys.argv[3]
basis = lh.ShowerBasis(**(json.loads(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] else {}))
k = model().kernel(config())
if len(sys.argv) > 5:
    t = time.time(); k.ge2_kernel_store(sys.argv[5]); print("store load", time.time() - t)
for ev in events:
    src = shower(kind, ev)
    t = time.time(); r = k.transport_hybrid(src, frequency_nodes=65, ge2_basis=basis); dt = time.time() - t
    np.savez(f"{prefix}_{kind}_{ev}.npz", charge=r.charge_components_pe, bins=r.components_pe,
             active=r.active, origin=r.time_origin_ns)
    m = r.metadata
    print(json.dumps({"event": ev, "seconds": dt, "peak_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                      **{key: round(v, 3) for key, v in m.items() if key.endswith("seconds")},
                      "points": m["ge2_basis_summary"]["cells"], "slices": m["ge2_basis_summary"].get("slices"),
                      "active": m["active_modules"], "store_mib": m["ge2_kernel_store_bytes"] / 2**20}), flush=True)
