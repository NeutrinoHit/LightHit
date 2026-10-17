# ruff: noqa: F403, F405  (prototype scripts share the helpers of common/proto)
import os
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import proto
from proto import *
ref = dict(np.load("ref_e100_5.npz"))
k = model().kernel(config()); src = shower("e100", 5); act = ref["active"]
for step in (1, 2, 4):
    proto._PREPARED.clear()
    idx = np.arange(0, 65, step)
    tm = {}
    q, b = ge2(k, src, act, cell_z=0.25, axial_only=True, indices=idx, timings=tm)
    errors(ref, q, b, act, f"freq={len(idx)} " + " ".join(f"{k}={v:.1f}" for k, v in tm.items() if k not in ('cells','prepare')))
