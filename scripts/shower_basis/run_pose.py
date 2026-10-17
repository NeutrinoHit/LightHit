"""A second, unrelated pose: hybrid and basis on the same event, compared.

Args: kind event px py pz dx dy dz
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from common import lh, model, config, shower

kind, event = sys.argv[1], int(sys.argv[2])
position = [float(v) for v in sys.argv[3:6]]
direction = np.asarray([float(v) for v in sys.argv[6:9]])
direction = (direction / np.linalg.norm(direction)).tolist()
k = model().kernel(config())
source = shower(kind, event, position=position, direction=direction)
out = {}
for name, basis in (("basis", lh.ShowerBasis()), ("hybrid", None)):
    path = f"pose_{name}_{kind}_{event}.npz"
    started = time.time()
    response = k.transport_hybrid(source, frequency_nodes=65, ge2_basis=basis)
    np.savez(path, charge=response.charge_components_pe, bins=response.components_pe,
             active=response.active, origin=response.time_origin_ns)
    out[name] = dict(np.load(path))
    print(json.dumps({"method": name, "seconds": time.time() - started,
                      "active": int(response.active.sum())}), flush=True)
ref, test = out["hybrid"], out["basis"]
idx = np.flatnonzero(ref["active"] | test["active"])
total = ref["charge"][idx].sum(1)
bright = np.argsort(-total)[:10]
q, qt = ref["charge"][idx, 2], test["charge"][idx, 2]
l1t = (np.abs(test["bins"][idx].sum(2) - ref["bins"][idx].sum(2)).sum(1)
       / np.abs(ref["bins"][idx].sum(2)).sum(1))
l1 = np.abs(test["bins"][idx, :, 2] - ref["bins"][idx, :, 2]).sum(1) / np.abs(ref["bins"][idx, :, 2]).sum(1)
print(json.dumps({"pose": {"position": position, "direction": direction},
                  "active": [int(ref["active"].sum()), int(test["active"].sum())],
                  "lost": int((ref["active"] & ~test["active"]).sum()),
                  "total_charge_rel_bright": float((np.abs(test["charge"][idx].sum(1) - total)
                                                    / total)[bright].max()),
                  "ge2_charge_rel_max": float((np.abs(qt - q) / np.abs(q)).max()),
                  "ge2_L1_max": float(l1.max()), "total_L1_bright": float(l1t[bright].max()),
                  "total_L1_max": float(l1t.max())}))
