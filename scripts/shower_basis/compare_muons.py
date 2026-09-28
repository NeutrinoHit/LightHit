"""Compare muon basis results with the cell reference (per-OM metrics)."""
import sys

import numpy as np

for kind, event in (item.split(":") for item in sys.argv[1:]):
    ref = dict(np.load(f"muon_cells_{kind}_{event}.npz"))
    test = dict(np.load(f"muon_basis_{kind}_{event}.npz"))
    idx = np.flatnonzero(ref["active"] | test["active"])
    total = ref["charge"][idx].sum(1)
    bright = np.argsort(-total)[:10]
    q, qt = ref["charge"][idx, 2], test["charge"][idx, 2]
    l1 = np.abs(test["bins"][idx, :, 2] - ref["bins"][idx, :, 2]).sum(1) / np.abs(ref["bins"][idx, :, 2]).sum(1)
    tb, tt = ref["bins"][idx].sum(2), test["bins"][idx].sum(2)
    l1t = np.abs(tt - tb).sum(1) / np.abs(tb).sum(1)
    early = (np.arange(-60., 740., 5.) >= 0) & (np.arange(-55., 745., 5.) <= 20)
    print(f"{kind}:{event} active {int(ref['active'].sum())}/{int(test['active'].sum())} "
          f"lost {int((ref['active'] & ~test['active']).sum())} | total q bright "
          f"{(np.abs(test['charge'][idx].sum(1) - total) / total)[bright].max():.1e} | q>=2 max "
          f"{(np.abs(qt - q) / np.abs(q)).max():.1e} abs {np.abs(qt - q).max():.1e} | "
          f"L1>=2 bright {l1[bright].max():.1e} max {l1.max():.1e} | L1 total bright "
          f"{l1t[bright].max():.1e} max {l1t.max():.1e} | early abs "
          f"{np.abs(tt[:, early].sum(1) - tb[:, early].sum(1)).max():.1e} | ge2 share "
          f"{q.sum() / total.sum():.3f}")
