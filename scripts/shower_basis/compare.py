"""Compare two saved hybrid results (order >=2, totals, bins). Args: ref test"""
import sys, numpy as np
a = dict(np.load(sys.argv[1])); b = dict(np.load(sys.argv[2]))
act = a["active"] | b["active"]
print("active ref/test/lost:", int(a["active"].sum()), int(b["active"].sum()), int((a["active"] & ~b["active"]).sum()))
for name, cols in (("ge2", [2]), ("total", [0, 1, 2])):
    qa = a["charge"][:, cols].sum(1); qb = b["charge"][:, cols].sum(1)
    ba = a["bins"][:, :, cols].sum(2); bb = b["bins"][:, :, cols].sum(2)
    idx = np.flatnonzero(act)
    rel = (qb[idx] - qa[idx]) / np.maximum(np.abs(qa[idx]), 1e-12)
    l1 = np.abs(bb[idx] - ba[idx]).sum(1) / np.maximum(np.abs(ba[idx]).sum(1), 1e-12)
    bright = qa[idx].argsort()[::-1][:10]
    print(f"{name}: sum ref {qa[idx].sum():.4f} test {qb[idx].sum():.4f}; max|rel| {np.abs(rel).max():.2e} (bright10 {np.abs(rel[bright]).max():.2e});"
          f" max abs {np.abs(qb[idx]-qa[idx]).max():.2e} pe; L1 max {l1.max():.2e} bright10 {l1[bright].max():.2e}")
qa = a["charge"][:, 2]; qall = a["charge"].sum(1)
print("ge2 share of total on active:", float(qa[act].sum() / qall[act].sum()))
