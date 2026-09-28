# ruff: noqa: F403, F405  (prototype scripts share the helpers of common/proto)
import os
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from proto import *
ref = dict(np.load("ref_e100_5.npz"))
k = model().kernel(config()); src = shower("e100", 5); act = ref["active"]
res = {}
for mode, cz in [("stencil", 0.06), ("stencil", 0.5), ("collapse", 0.5), ("collapse", 0.12), ("collapse", 0.06)]:
    tm = {}
    q, b = ge2(k, src, act, cell_z=cz, slices=mode, stencil_m=0.2, timings=tm)
    res[(mode, cz)] = (q, b)
    errors(ref, q, b, act, f"{mode} cz={cz} cells={tm['cells']} " + " ".join(f"{k}={v:.1f}" for k, v in tm.items() if k not in ('cells','prepare')))
e4 = np.load("exp4.npz")
res[("stencil", 0.25)] = (e4["stencil_0.25_0.2_q"], e4["stencil_0.25_0.2_b"])
res[("stencil", 0.12)] = (e4["stencil_0.12_0.2_q"], e4["stencil_0.12_0.2_b"])
res[("collapse", 0.25)] = (e4["collapse_0.25_0.2_q"], e4["collapse_0.25_0.2_b"])
q0, b0 = res[("stencil", 0.06)]
idx = np.flatnonzero(act)
res[("hybrid3D", 0.12)] = (ref["charge"][:, 2], ref["bins"][:, :, 2])
for key in sorted(res):
    q, b = res[key]
    l1 = np.abs(b[idx] - b0[idx]).sum(1) / np.abs(b0[idx]).sum(1)
    print("vs stencil0.06:", key, "charge rel %.2e" % np.max(np.abs(q[idx]-q0[idx])/np.abs(q0[idx])), "L1 ge2 %.2e" % l1.max())
np.savez("exp5.npz", **{f"{m}_{c}_q": v[0] for (m,c), v in res.items()}, **{f"{m}_{c}_b": v[1] for (m,c), v in res.items()})
