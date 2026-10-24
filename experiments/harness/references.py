"""Reference calculations on a subset of OMs, embedded in full-detector arrays.

A reference is computed only for the OMs that matter for a comparison: those
active in the default route's saved response for the same input, plus those
whose default charge is at least ``margin`` times the threshold (so that OMs
near the threshold are seen from both sides). The reference runs with
threshold 0 on that subset, on the default route's per-OM clock, and is
embedded back into arrays over all OMs (zero charge, inactive elsewhere); its
``active`` is its own charge >= the kernel threshold. Metrics against such a
reference are therefore restricted to that subset (see ``subset_note``).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .record import RESULTS, load_response


@dataclass
class Embedded:
    """Minimal response object (what metrics.compare and save_response read)."""
    charge_components_pe: np.ndarray
    components_pe: np.ndarray | None
    active: np.ndarray
    time_origin_ns: np.ndarray
    metadata: dict = field(default_factory=dict)
    covered: np.ndarray | None = None      # OMs the reference computed
    computed_orders: tuple = (0, 1, 2)     # orders the reference holds (others are zero)


def om_subset(default_case: str, *, margin: float = 0.5, threshold: float = 0.01):
    """(indices, time origins) from the saved response of ``default_case``."""
    path = RESULTS / f"{default_case}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"run the default case {default_case} first")
    base = load_response(path)
    total = base["charge_components_pe"].sum(1)
    chosen = np.flatnonzero(base["active"] | (total >= margin * threshold))
    return chosen, base["time_origin_ns"], base


def subset_kernel(kernel, indices, **config):
    import lighthit as lh
    d = kernel.detector
    sub = d.subset(indices)
    return lh.TransportKernel(kernel.medium, sub, replace(kernel.config, **config))


def embed(parts, indices, n_om, origins, threshold=0.01, metadata=None):
    """Sum per-order pieces computed on ``indices`` into full arrays.

    ``parts`` is a list of (response, orders) where ``orders`` lists which of
    0, 1, 2 to take from that response.
    """
    charge = np.zeros((n_om, 3))
    bins = None
    for response, orders in parts:
        from .metrics import three_orders
        q, b = three_orders(response)
        if bins is None:
            bins = np.zeros((n_om, b.shape[1], 3))
        for k in orders:
            charge[indices, k] += q[:, k]
            bins[indices, :, k] += b[:, :, k]
    active = np.zeros(n_om, bool)
    active[indices] = charge[indices].sum(1) >= threshold
    covered = np.zeros(n_om, bool)
    covered[indices] = True
    return Embedded(charge, bins, active, np.asarray(origins, float),
                    dict(metadata or {}, reference_subset=len(indices)), covered)
