# ruff: noqa: F403, F405  (prototype scripts share the helpers of common/proto)
import os
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from proto import *
ref = dict(np.load("ref_e100_5.npz"))
k = model().kernel(config()); src = shower("e100", 5); act = ref["active"]
runs = [("collapse", 0.25, 0.2), ("collapse", 0.25, 0.2), ("stencil", 0.25, 0.1), ("stencil", 0.25, 0.2), ("stencil", 0.25, 0.4), ("stencil", 0.12, 0.2)]
res = {}
for mode, cz, h in runs:
    tm = {}
    q, b = ge2(k, src, act, cell_z=cz, slices=mode, stencil_m=h, timings=tm)
    res[(mode, cz, h)] = (q, b)
    errors(ref, q, b, act, f"{mode} cz={cz} h={h} cells={tm['cells']} " + " ".join(f"{k}={v:.1f}" for k, v in tm.items() if k not in ('cells','prepare')))
q0, b0 = res[("stencil", 0.12, 0.2)]
for key, (q, b) in res.items():
    idx = np.flatnonzero(act)
    l1 = np.abs(b[idx] - b0[idx]).sum(1) / np.abs(b0[idx]).sum(1)
    print("vs stencil0.12:", key, "charge rel", np.max(np.abs(q[idx]-q0[idx])/np.abs(q0[idx])), "L1 ge2", l1.max())
np.savez("exp4.npz", **{f"{m}_{c}_{h}_q": v[0] for (m,c,h), v in res.items()}, **{f"{m}_{c}_{h}_b": v[1] for (m,c,h), v in res.items()})
