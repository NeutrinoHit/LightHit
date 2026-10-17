"""Warm timing of the basis route on muons, with a memory-mapped kernel pack.

Args: pack_dir kind:event [kind:event ...]
"""
import json
import os
import resource
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import lh, model, config, shower

k = model().kernel(config())
k.ge2_kernel_store(sys.argv[1])
for spec in sys.argv[2:]:
    kind, event = spec.split(":")
    source = shower(kind, int(event))
    for repeat in range(2):
        started = time.time()
        response = k.transport_hybrid(source, frequency_nodes=65, ge2_basis=lh.ShowerBasis())
        meta = response.metadata
        print(json.dumps({"event": spec, "call": repeat, "seconds": round(time.time() - started, 2),
                          **{key: round(value, 2) for key, value in meta.items()
                             if key.endswith("seconds")},
                          "points": meta["ge2_basis_summary"]["cells"],
                          "peak_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}),
              flush=True)
