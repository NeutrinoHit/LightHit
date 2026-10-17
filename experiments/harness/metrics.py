"""Accuracy metrics between two responses on the same detector and clock.

A response here is anything with ``charge_components_pe`` (OM, 3),
``components_pe`` (OM, bin, 3) where available, ``active`` and
``time_origin_ns``, or a saved ``.npz`` with the same arrays
(``experiments.harness.record.save_response``).

Definitions (per scattering order k = 0, 1, 2 meaning >=2, and "total"):

* ``charge_pe``: summed charge over all OMs;
* ``charge_bias``: (sum Q - sum Q_ref) / sum Q_ref over all OMs;
* ``bin_l1``: sum over OMs active in both, and bins, of |b - b_ref| divided by
  the same sum of |b_ref| (only when both have bins and the same time origins);
* ``worst_om``: largest |Q - Q_ref| / Q_ref over OMs with Q_ref >= ``floor_pe``
  (default: the threshold, 0.01 pe), with its OM index and Q_ref;
* ``active``: counts of OMs active in both, only in the response, only in the
  reference, and the summed reference charge of OMs it misses.
"""
from __future__ import annotations

import numpy as np

ORDERS = ("0", "1", ">=2")


def three_orders(response):
    """(charge (OM, 3), bins (OM, bin, 3) or None) of any response.

    A ``PromptTransportResponse`` holds orders 0 and 1 only
    (``charge_orders_pe``, ``bins_orders_pe``); its order >=2 is returned as
    zeros and must not be compared.
    """
    if hasattr(response, "charge_orders_pe"):
        q = np.asarray(response.charge_orders_pe, float)
        b = np.asarray(response.bins_orders_pe, float)
        return (np.concatenate([q, np.zeros((len(q), 1))], 1),
                np.concatenate([b, np.zeros(b.shape[:2] + (1,))], 2))
    charge = np.asarray(response.charge_components_pe, float)
    try:
        bins = np.asarray(response.components_pe, float)
    except Exception:
        bins = None
    return charge, bins


def _arrays(response):
    if isinstance(response, dict):
        get = response.__getitem__
        charge = np.asarray(get("charge_components_pe"), float)
        bins = np.asarray(get("components_pe"), float) if "components_pe" in response else None
    else:
        get = lambda k: getattr(response, k)  # noqa: E731
        charge, bins = three_orders(response)
    active = np.asarray(get("active"), bool)
    origin = np.asarray(get("time_origin_ns"), float)
    return charge, bins, active, origin


def compare(response, reference, *, orders=(0, 1, 2), floor_pe: float = 0.01) -> dict:
    """Metrics of ``response`` against ``reference`` (see module docstring)."""
    q, b, a, t = _arrays(response)
    qr, br, ar, tr = _arrays(reference)
    if q.shape != qr.shape:
        raise ValueError(f"different OM sets: {q.shape} vs {qr.shape}")
    covered = (reference.get("covered") if isinstance(reference, dict)
               else getattr(reference, "covered", None))
    if covered is not None:      # a reference computed on a subset of OMs: compare there only
        keep = np.asarray(covered, bool)
        q, a, t, qr, ar, tr = q[keep], a[keep], t[keep], qr[keep], ar[keep], tr[keep]
        b = b[keep] if b is not None else None
        br = br[keep] if br is not None else None
    both = a & ar
    same_clock = bool(np.allclose(t[both], tr[both], rtol=0, atol=1e-6)) if both.any() else True
    out = dict(compared_oms=int(len(q)), active=dict(both=int(both.sum()), only_response=int((a & ~ar).sum()),
                           only_reference=int((~a & ar).sum()),
                           missed_reference_charge_pe=float(qr.sum(1)[~a & ar].sum())),
               same_clock=same_clock, orders={})
    selections = [(ORDERS[k], [k]) for k in orders] + [("total", list(orders))]
    for name, ks in selections:
        qk, qrk = q[:, ks].sum(1), qr[:, ks].sum(1)
        total_ref = float(qrk.sum())
        entry = dict(charge_pe=float(qk.sum()), reference_charge_pe=total_ref,
                     charge_bias=(float(qk.sum()) - total_ref) / total_ref if total_ref else None)
        mask = qrk >= floor_pe
        if mask.any():
            rel = np.abs(qk[mask] - qrk[mask]) / qrk[mask]
            j = int(np.argmax(rel))
            index = int(np.flatnonzero(mask)[j])
            entry["worst_om"] = dict(relative_error=float(rel[j]), om=index,
                                     reference_pe=float(qrk[index]), pe=float(qk[index]))
            entry["om_charge_relative_l1"] = float(np.abs(qk[mask] - qrk[mask]).sum() / qrk[mask].sum())
        if b is not None and br is not None and same_clock and both.any() and b.shape == br.shape:
            bk, brk = b[both][:, :, ks].sum(2), br[both][:, :, ks].sum(2)
            norm = float(np.abs(brk).sum())
            entry["bin_l1"] = float(np.abs(bk - brk).sum()) / norm if norm else None
        out["orders"][name] = entry
    return out


def headline(metrics: dict) -> dict:
    """The few numbers shown in REPORT.md."""
    out = {}
    for name, entry in metrics["orders"].items():
        out[name] = {k: entry.get(k) for k in ("charge_bias", "bin_l1")}
        if "worst_om" in entry:
            out[name]["worst_om"] = entry["worst_om"]["relative_error"]
    out["active"] = metrics["active"]
    return out
