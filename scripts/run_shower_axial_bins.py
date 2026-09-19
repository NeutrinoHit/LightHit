"""Time bins of a stored shower, on a window that actually contains the light.

The spectrum of a module at 317 m carries arrival times out past a microsecond,
while a frequency grid of spacing ``dw`` can only reconstruct times up to about
``pi / dw``. Binning such a module from ``t = 0`` on a coarse grid therefore
reports ringing in an empty window and calls it an error. Two changes fix it
and are the point of this script:

* each module's bins are measured from its own geometric arrival
  ``r / v``, so the signal sits at the start of the window instead of a
  microsecond into it. The shift is exact -- a multiplication of the spectrum
  by ``exp(i omega r / v)`` -- and nothing is resampled;
* the frequency grid is dense enough for the window that remains.

Everything else matches ``run_shower_axial.py``: same contract, same cache
settings, same compact source, and the reference is the same kernel summed
element by element. Only the control modules are evaluated, because the
element sum is what costs and it is only needed where the comparison is made.

Negative bins are kept exactly as computed. They are the diagnostic that says
the frequency window is too narrow; clipping them would hide the one thing the
reader needs to judge.

Run:
  python scripts/run_shower_axial_bins.py --input /path/sim_e_100GeV_10.h5 \
      --event 5 --output .build/review/shower-axial
"""
from pathlib import Path
import argparse
import json
import sys
from time import perf_counter

import numpy as np

from lighthit import SolverSettings, synthetic_medium
from lighthit.cache import CacheGrid, ResponseCache
from lighthit.readout import inverse_bins
from lighthit.experimental.axial_source import AxisFrame, AxialSource, axial_response
from lighthit.experimental.event_moments import KernelChannels, direct_response
from lighthit.experimental.g4_source import SourceContract, load_event

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_shower_moments import array_positions, serial  # noqa: E402

ORDERS = {"first": 0, "two_or_more": 1}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--event", type=int, default=5)
    parser.add_argument("--output", default=".build/review/shower-axial")
    parser.add_argument("--angular-degree", type=int, default=32)
    parser.add_argument("--azimuthal-degree", type=int, default=4)
    parser.add_argument("--cell-m", type=float, default=0.12)
    parser.add_argument("--frequencies", type=int, default=81)
    parser.add_argument("--omega-max", type=float, default=0.5)
    parser.add_argument("--k-max", type=float, default=8.0)
    parser.add_argument("--radii", type=int, default=160)
    parser.add_argument("--controls", type=int, default=6)
    parser.add_argument("--bin-ns", type=float, default=10.0)
    parser.add_argument("--bins", type=int, default=48)
    parser.add_argument("--early-ns", type=float, default=50.0)
    parser.add_argument("--cache", default=None)
    arguments = parser.parse_args()
    out = Path(arguments.output)
    out.mkdir(parents=True, exist_ok=True)

    medium = synthetic_medium()
    speed = medium.speed_m_per_ns
    omega = np.linspace(0.0, arguments.omega_max, arguments.frequencies)
    step = float(omega[1] - omega[0])
    horizon = float(np.pi / step)
    span = arguments.bin_ns * arguments.bins
    report = {"settings": vars(arguments), "omega_step_per_ns": step,
              "reconstruction_horizon_ns": horizon, "window_ns": span,
              "window_fits": bool(span <= horizon),
              "note": ("Bins are measured from each module's own geometric "
                       "arrival r/v. Negative bins are kept as computed.")}
    print(f"frequency step {step:.5f} rad/ns, horizon {horizon:.0f} ns, "
          f"window {span:.0f} ns -> {'fits' if span <= horizon else 'DOES NOT FIT'}",
          flush=True)

    elements = load_event(arguments.input, arguments.event,
                          SourceContract()).moved(translation=np.array([20.0, 15.0, -20.0]))
    receivers = array_positions()
    distance = np.linalg.norm(receivers - elements.centroid_m, axis=1)
    order = np.argsort(distance)
    picks = order[np.linspace(0, len(order) - 1, arguments.controls).astype(int)]
    control, control_distance = receivers[picks], distance[picks]
    arrival = control_distance / speed
    report["control"] = {"distance_m": control_distance.tolist(),
                         "geometric_arrival_ns": arrival.tolist()}
    print("control at", np.round(control_distance, 1), "m, arriving at",
          np.round(arrival), "ns", flush=True)

    reach = 1.5 * elements.extent_m
    low = max(3.0, (distance.min() - reach) * 0.95)
    high = (distance.max() + reach) * 1.05
    if arguments.cache and Path(arguments.cache).exists():
        cache = ResponseCache.load(arguments.cache)
    else:
        began = perf_counter()
        cache = ResponseCache.build(
            medium, SolverSettings(24, arguments.angular_degree, arguments.k_max, 0.04, 10),
            CacheGrid.geometric(low, high, arguments.radii, omega))
        report["cache_build_seconds"] = perf_counter() - began
        print(f"cache built in {report['cache_build_seconds']:.0f} s", flush=True)
        if arguments.cache:
            cache.save(arguments.cache)
    omega = cache.grid.omega_per_ns
    kernels = {name: KernelChannels.of(cache, index, arguments.angular_degree)
               for name, index in ORDERS.items()}

    began = perf_counter()
    source = AxialSource.of(elements, arguments.angular_degree, omega,
                            azimuthal_degree=arguments.azimuthal_degree,
                            cell_m=arguments.cell_m, deposit="linear",
                            element_order=2, frame=AxisFrame.of(elements))
    report["compact_source"] = {**source.summary, "seconds": perf_counter() - began}
    print(f"source: {source.summary['cells']} cells in "
          f"{report['compact_source']['seconds']:.0f} s", flush=True)

    edges = np.arange(arguments.bins + 1) * arguments.bin_ns
    shift = np.exp(1j * omega[:, None] * arrival[None, :])
    early = edges[:-1] < arguments.early_ns
    report["bins"] = {"edges_ns": edges.tolist(), "per_order": {}}
    for name, kernel in kernels.items():
        began = perf_counter()
        compact = axial_response(kernel, source, control)
        apply_seconds = perf_counter() - began
        began = perf_counter()
        exact = direct_response(kernel, elements, control)
        element_seconds = perf_counter() - began
        got = inverse_bins(omega, compact * shift, edges)
        want = inverse_bins(omega, exact * shift, edges)
        peak = np.abs(want).max(axis=1)
        total = want.sum(axis=1)
        report["bins"]["per_order"][name] = {
            "apply_seconds": apply_seconds,
            "element_sum_seconds": element_seconds,
            "peak_per_module": peak.tolist(),
            "integral_per_module": total.tolist(),
            "max_bin_error_over_peak": (np.abs(got - want).max(axis=1) / peak).tolist(),
            "integral_error": (np.abs(got.sum(axis=1) - total)
                               / np.abs(total)).tolist(),
            "early_window_error": [
                float(abs(got[i, early].sum() - want[i, early].sum())
                      / max(abs(want[i, early].sum()), 1e-300))
                for i in range(len(picks))],
            "early_window_share_of_signal": [
                float(want[i, early].sum() / total[i]) for i in range(len(picks))],
            "negative_mass_fraction_reference": float(
                np.abs(want[want < 0]).sum() / np.abs(want).sum()),
            "negative_mass_fraction_compact": float(
                np.abs(got[got < 0]).sum() / np.abs(got).sum())}
        np.save(out / f"control-bins-{name}.npy", np.stack((got, want)))
        row = report["bins"]["per_order"][name]
        print(f"{name}: max bin/peak "
              + " ".join(f"{x:.1e}" for x in row["max_bin_error_over_peak"])
              + " | integral " + " ".join(f"{x:.1e}" for x in row["integral_error"]),
              flush=True)
    np.save(out / "control-edges-ns.npy", edges)
    (out / "shower-axial-bins.json").write_text(
        json.dumps(report, default=serial, indent=2), encoding="utf-8")
    print(f"written: {out / 'shower-axial-bins.json'}")


if __name__ == "__main__":
    main()
