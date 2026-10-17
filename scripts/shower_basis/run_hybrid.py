# ruff: noqa: F403, F405  (prototype scripts share the helpers of common/proto)
"""Run transport_hybrid; save orders/bins; print timings. Args: kind event cell_m out [extra json]"""
import os
import sys, time, json, resource
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from common import *
kind, event, cell, out = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
extra = json.loads(sys.argv[5]) if len(sys.argv) > 5 else {}
repeat = extra.pop("repeat", 2)
k = model().kernel(config(cell_m=cell, **extra))
src = shower(kind, event)
times = []
for _ in range(repeat):
    t = time.time(); r = k.transport_hybrid(src, frequency_nodes=65); times.append(time.time() - t)
meta = {kk: v for kk, v in r.metadata.items() if isinstance(v, (int, float, str, bool))}
np.savez(out, charge=r.charge_components_pe, bins=r.components_pe, active=r.active,
         origin=r.time_origin_ns, spectrum_ge2=getattr(r, "spectrum_ge2_pe", np.zeros(0)),
         omega=r.omega_per_ns if hasattr(r, "omega_per_ns") else np.zeros(0))
print(json.dumps({"times": times, "peak_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                  "active": int(r.active.sum()), "meta": meta}, indent=0, default=str))
