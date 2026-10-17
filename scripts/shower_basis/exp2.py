# ruff: noqa: F403, F405  (prototype scripts share the helpers of common/proto)
import os
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import proto
from proto import *
ref = dict(np.load("ref_e100_5.npz"))
k = model().kernel(config()); src = shower("e100", 5); act = ref["active"]
for deg in (32, 24, 16, 12):
    proto._PREPARED.clear()
    for cz, M in [(0.12, 4), (0.25, 4), (0.25, 2), (0.25, 0)]:
        if deg != 32 and (cz, M) != (0.25, 4): continue
        tm = {}
        q, b = ge2(k, src, act, cell_z=cz, axial_only=True, degree=deg, M=M, timings=tm)
        errors(ref, q, b, act, f"L={deg} cz={cz} M={M} cells={tm['cells']} " + " ".join(f"{k}={v:.1f}" for k, v in tm.items() if k not in ('cells','prepare')))
