# ruff: noqa: F403, F405  (prototype scripts share the helpers of common/proto)
import os
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from proto import *
ref = dict(np.load("ref_e100_5.npz")); c05 = dict(np.load("cell_0.5.npz"))
k = model().kernel(config()); src = shower("e100", 5); act = ref["active"]
for cz, ct, ax in [(0.5, None, False), (0.5, None, False), (0.25, None, False), (0.5, None, True), (0.25, None, True), (0.5, 0.25, False)]:
    tm = {}
    q, b = ge2(k, src, act, cell_z=cz, cell_t=ct, axial_only=ax, timings=tm)
    errors(ref, q, b, act, f"cz={cz} ct={ct} ax={ax} cells={tm['cells']} " + " ".join(f"{k}={v:.1f}" for k, v in tm.items() if k != 'cells'))
    if (cz, ct, ax) == (0.5, None, False):
        print("  vs hybrid cell0.5: charge", np.abs(q - c05["charge"][:, 2]).max(), "bins", np.abs(b - c05["bins"][:, :, 2]).max())
