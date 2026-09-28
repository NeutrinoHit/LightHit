"""Frequency-interpolation error of order >=2 (65 computed -> 321 readout).

The same basis source is applied on a dense frequency grid (every
``step``-th node of a 321-frequency cache; step 1 = no interpolation) and on
the 65-node grid of the production hybrid; both are read out on the 321-node
grid with the hybrid's cubic interpolation after time-origin demodulation.
The RTE and the source are identical in both, so the difference is the
interpolation error alone -- the same for the default hybrid and for the
basis route.

Args: kind event [step]
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.interpolate import CubicSpline
from common import WORK_DIR, config, model, shower
from lighthit._directional_numba import PreparedDirectionalKernel
from lighthit.readout import inverse_bins
from proto import basis_sources, origins_of

kind, event = sys.argv[1], int(sys.argv[2])
step = int(sys.argv[3]) if len(sys.argv) > 3 else 2
dense = np.linspace(0., 1.2, 321)
grids = {"dense": np.unique(np.append(np.arange(0, 321, step), 320)),
         "sparse": np.unique(np.rint(np.linspace(0, 320, 65)).astype(int))}
kernel = model().kernel(config(omega_per_ns=dense, readout_omega_per_ns=dense,
                               cache_directory=WORK_DIR + "/cache321"))
source = shower(kind, event)
reference = dict(np.load(f"{WORK_DIR}/bench-{kind}/basis_{kind}_{event}.npz"))
targets = np.flatnonzero(reference["active"])
positions = kernel.detector.positions_m[targets]
looks = np.ascontiguousarray(-kernel.detector.orientations[targets], float)
area = np.asarray(kernel.detector.effective_area_m2, float)[targets]
alpha = np.ascontiguousarray(kernel.acceptance()["alpha"], float)
edges = kernel.config.relative_time_edges_ns
bins = {}
for name, index in grids.items():
    omega = dense[index]
    s0, s2, elements, sample = basis_sources(kernel, source, omega=omega)
    origins = origins_of(kernel, elements, sample)[targets]
    spectrum = np.zeros((len(omega), len(targets)), complex)
    for w, wt, ph in zip(kernel.wavelength.wavelength_nm, kernel.wavelength.weight_nm,
                         sample["phase_index"]):
        # one wavelength at a time: a 321-node spline for all nine at once
        # does not fit in 15 GB
        cache = kernel._directional_cache_for(float(w))
        prepared = PreparedDirectionalKernel.from_cache(
            cache, degree=kernel.config.source_degree, frequency_indices=index)
        kernel._directional_caches.clear()
        cache = None
        r = prepared.apply(s0, positions, looks, alpha, source_omega_per_ns=omega,
                           receiver_block=1, second_source=s2, first_scale=1 / w ** 2,
                           second_scale=1 / (w ** 2 * ph ** 2), ge2_only=True)
        spectrum += (wt * area[None, :] * float(kernel.detector.spectral_weight(float(w)))
                     * r[:, :, 1])
        prepared = None
    relative = spectrum * np.exp(-1j * omega[:, None] * origins[None, :])
    if len(omega) != len(dense):
        relative = (CubicSpline(omega, relative.real, axis=0)(dense)
                    + 1j * CubicSpline(omega, relative.imag, axis=0)(dense))
    bins[name] = inverse_bins(dense, relative, edges)
a, b = bins["dense"], bins["sparse"]
l1 = np.abs(b - a).sum(1) / np.abs(a).sum(1)
total = reference["bins"][targets].sum(2)
l1_total = np.abs(b - a).sum(1) / np.abs(total).sum(1)
bright = np.argsort(-reference["charge"][targets].sum(1))[:10]
print(json.dumps({"kind": kind, "event": event, "dense_nodes": len(grids["dense"]),
                  "ge2_L1_max": float(l1.max()), "ge2_L1_bright": float(l1[bright].max()),
                  "total_L1_max": float(l1_total.max()),
                  "total_L1_bright": float(l1_total[bright].max()),
                  "ge2_window_rel_max": float(np.max(np.abs(b.sum(1) - a.sum(1))
                                                     / np.abs(a.sum(1))))}))
