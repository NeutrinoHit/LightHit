# ruff: noqa: F403, F405  (prototype scripts share the helpers of common/proto)
import os
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from proto import *
ref = dict(np.load("ref_e100_5.npz")); e5 = np.load("exp5.npz")
q0, b0 = e5["stencil_0.06_q"], e5["stencil_0.06_b"]
k = model().kernel(config()); src = shower("e100", 5); act = ref["active"]; idx = np.flatnonzero(act)
out = {}
for mode, cz, o in [("collapse", 0.25, 3), ("stencil", 0.5, 3), ("stencil", 0.25, 3), ("stencil", 0.33, 3), ("collapse", 0.5, 3)]:
    tm = {}
    q, b = ge2(k, src, act, cell_z=cz, slices=mode, axial_order=o, timings=tm)
    l1 = np.abs(b[idx] - b0[idx]).sum(1) / np.abs(b0[idx]).sum(1)
    print(f"{mode} cz={cz} order={o} cells={tm['cells']} apply={tm['zero_apply']+tm['full_apply']:.1f} compile={tm['compile']:.1f} | vs converged: charge {np.max(np.abs(q[idx]-q0[idx])/np.abs(q0[idx])):.2e} L1ge2 {l1.max():.2e}", flush=True)
    out[f"{mode}_{cz}_{o}_q"] = q; out[f"{mode}_{cz}_{o}_b"] = b
np.savez("exp6.npz", **out)
