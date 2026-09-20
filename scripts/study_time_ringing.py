#!/usr/bin/env python3
"""Separate frequency-spacing, cutoff and bin-width effects on time ringing.

The exact full-HG first-scattering spectrum is used because its physical-time
bins are independently available from ``single_bins``.  This gives an absolute
reference that a compact-event-vs-element-sum comparison cannot provide when
both sides share the same finite frequency window.
"""
from pathlib import Path
import argparse
import json

import numpy as np

from lighthit import synthetic_medium
from lighthit.readout import inverse_bins
from lighthit.single import single_bins, single_spectrum


CASES = (
    ("baseline", 0.15, 41, 20.0),
    ("more_nodes_same_cutoff", 0.15, 161, 20.0),
    ("higher_cutoff_same_spacing", 0.60, 161, 20.0),
    ("higher_cutoff_more_nodes", 0.60, 321, 20.0),
    ("higher_cutoff_narrow_bins", 0.60, 321, 5.0),
    ("higher_cutoff_wide_bins", 0.60, 321, 40.0),
    ("cutoff_1p2_same_spacing", 1.20, 321, 20.0),
    ("cutoff_2p4_same_spacing", 2.40, 641, 20.0),
    ("cutoff_2p4_wide_bins", 2.40, 641, 40.0),
)


def evaluate(name, omega_max, nodes, width, radius, stop, medium):
    omega = np.linspace(0.0, omega_max, nodes)
    front = radius / medium.speed_m_per_ns
    relative_edges = np.arange(0.0, stop + 0.5 * width, width)
    edges = front + relative_edges
    spectrum, quadrature_error = single_spectrum(
        omega, radius, None, medium, backend="numba")
    got = inverse_bins(omega, spectrum[:, None], edges)[0]
    exact = single_bins(edges, radius, None, medium, backend="numba")
    scale = max(float(exact.sum()), 1e-300)
    negative = np.abs(got[got < 0]).sum()
    return {
        "name": name, "omega_max_per_ns": omega_max, "nodes": nodes,
        "delta_omega_per_ns": float(omega[1] - omega[0]),
        "period_ns": float(2 * np.pi / (omega[1] - omega[0])),
        "bin_width_ns": width, "quadrature_error": quadrature_error,
        "relative_l1_error": float(np.abs(got - exact).sum() / scale),
        "maximum_bin_error_over_exact_peak": float(
            np.max(np.abs(got - exact)) / max(np.max(exact), 1e-300)),
        "negative_mass_over_exact_window": float(negative / scale),
        "reconstructed_window_over_exact": float(got.sum() / scale),
        "centres_relative_ns": ((relative_edges[:-1] + relative_edges[1:]) / 2).tolist(),
        "reconstructed": got.tolist(), "exact": exact.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=".build/time-ringing")
    parser.add_argument("--radius", type=float, default=50.0)
    parser.add_argument("--stop", type=float, default=400.0)
    args = parser.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    medium = synthetic_medium()
    rows = [evaluate(*case, args.radius, args.stop, medium) for case in CASES]
    report = {"medium": {"absorption_per_m": medium.absorption_per_m,
                          "scattering_per_m": medium.scattering_per_m,
                          "g": medium.g, "group_index": medium.group_index},
              "radius_m": args.radius, "window_relative_ns": [0.0, args.stop],
              "component": "exact full-HG first scattering",
              "interpretation": {
                  "nodes": "At fixed cutoff, more nodes reduce delta-omega and move periodic copies farther away; they do not improve time resolution.",
                  "cutoff": "At fixed delta-omega, a larger omega_max narrows the band-limit ringing and improves time resolution.",
                  "bins": "A wider bin applies a stronger sinc low-pass filter and reduces visible oscillations, at the cost of time resolution."},
              "cases": rows}
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(json.dumps([{k: row[k] for k in (
            "name", "relative_l1_error", "negative_mass_over_exact_window")}
                          for row in rows], indent=2))
        return
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    selected = [rows[0], rows[2], rows[6], rows[8]]
    for ax, row in zip(axes.flat, selected):
        x = row["centres_relative_ns"]
        ax.step(x, row["exact"], where="mid", label="exact physical-time bins")
        ax.step(x, row["reconstructed"], where="mid", label="finite-frequency inversion")
        ax.axhline(0, color="grey", lw=.6)
        ax.set_title(row["name"] + f"\nL1={row['relative_l1_error']:.3g}, "
                     f"negative={row['negative_mass_over_exact_window']:.3g}")
        ax.set(xlabel="time after front, ns", ylabel="first-order signal / bin")
        ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "ringing.png", dpi=150); plt.close(fig)
    print(json.dumps([{k: row[k] for k in (
        "name", "relative_l1_error", "maximum_bin_error_over_exact_peak",
        "negative_mass_over_exact_window", "reconstructed_window_over_exact")}
                      for row in rows], indent=2))


if __name__ == "__main__":
    main()
