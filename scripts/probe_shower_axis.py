"""Is a stored shower one dimensional, and in which of its variables?

Before choosing a basis it is worth asking what the source actually looks like
in the coordinates the basis would use. This script takes the same source
contract every other route uses and reports, per event and with no transport
at all:

* the axis, from the photon-weighted second moment of the emission points;
* the transverse spread about that axis, and what it is worth as an angle at
  the nearest module — the number that decides whether the transverse
  coordinates can be dropped;
* the residual emission time ``tau = t - t0 - z/c0``, which is what is thrown
  away by the assumption that emission time is just the arrival of the front;
* the emission direction in the axis frame: the polar cosine, and whether the
  azimuth really is uniform;
* how strongly ``z``, the polar cosine and ``tau`` are correlated, because a
  basis built from marginal profiles can only be right if they are not.

Photon weights are used throughout: an element that radiates nothing should
not move the axis.

Run:
  python scripts/probe_shower_axis.py --input /path/sim_e_100GeV_10.h5 \
      --events 0 1 2 3 4 5 --output .build/review/shower-basis
"""
from pathlib import Path
import argparse
import json

import numpy as np

from lighthit.experimental.g4_source import SourceContract, load_event

VACUUM_M_PER_NS = 0.299792458


def axis_frame(points, weights):
    """Photon-weighted centre and principal axis of a cloud of emission points."""
    share = weights / weights.sum()
    centre = (share[:, None] * points).sum(axis=0)
    offset = points - centre
    covariance = (share[:, None, None] * offset[:, :, None] * offset[:, None, :]).sum(axis=0)
    values, vectors = np.linalg.eigh(covariance)
    axis = vectors[:, int(np.argmax(values))]
    # The sign is free; point it the way the light-weighted directions go.
    if (share * (points @ axis)).sum() < (share * (points @ -axis)).sum():
        axis = -axis
    return centre, axis, np.sqrt(np.maximum(values[::-1], 0.0))


def weighted_quantiles(values, weights, quantiles):
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    share = np.cumsum(weights) / weights.sum()
    return np.interp(quantiles, share, values)


def describe(values, weights, label):
    mean = float(np.average(values, weights=weights))
    variance = float(np.average((values - mean) ** 2, weights=weights))
    low, median, high, tail = weighted_quantiles(
        values, weights, [0.01, 0.5, 0.99, 0.999])
    return {"name": label, "mean": mean, "rms": float(np.sqrt(variance)),
            "median": float(median), "q01": float(low), "q99": float(high),
            "q999": float(tail), "min": float(values.min()), "max": float(values.max())}


def weighted_correlation(first, second, weights):
    a = first - np.average(first, weights=weights)
    b = second - np.average(second, weights=weights)
    denominator = np.sqrt(np.average(a * a, weights=weights)
                          * np.average(b * b, weights=weights))
    return float(np.average(a * b, weights=weights) / denominator) if denominator else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--events", type=int, nargs="*", default=list(range(10)))
    parser.add_argument("--output", default=".build/review/shower-basis")
    parser.add_argument("--nearest-module-m", type=float, default=25.7,
                        help="distance used to turn a transverse size into an angle")
    parser.add_argument("--label", default=None)
    arguments = parser.parse_args()
    out = Path(arguments.output)
    out.mkdir(parents=True, exist_ok=True)
    label = arguments.label or Path(arguments.input).stem

    report = {"input": str(arguments.input), "label": label,
              "vacuum_speed_m_per_ns": VACUUM_M_PER_NS,
              "nearest_module_m": arguments.nearest_module_m, "events": []}
    for number in arguments.events:
        elements = load_event(arguments.input, number, SourceContract())
        points = elements.midpoints_m
        weights = np.asarray(elements.photons, float)
        centre, axis, spread = axis_frame(points, weights)

        offset = points - centre
        z = offset @ axis
        transverse = offset - z[:, None] * axis
        radius = np.linalg.norm(transverse, axis=1)

        # tau is measured from the earliest emission, so that it is a delay and
        # not an offset: t0 is the time of the first light, z from the centre.
        start = elements.start_ns - elements.start_ns.min()
        tau = start - (z - z.min()) / VACUUM_M_PER_NS

        cosine = elements.direction @ axis
        first = np.cross(axis, [0.0, 0.0, 1.0])
        if np.linalg.norm(first) < 1e-8:
            first = np.cross(axis, [0.0, 1.0, 0.0])
        first /= np.linalg.norm(first)
        second = np.cross(axis, first)
        azimuth = np.arctan2(elements.direction @ second, elements.direction @ first)

        # Uniformity of the azimuth as the length of its mean unit vector: 0 is
        # perfectly uniform, 1 is a single direction.
        share = weights / weights.sum()
        anisotropy = float(np.hypot((share * np.cos(azimuth)).sum(),
                                    (share * np.sin(azimuth)).sum()))

        row = {"event": number, "elements": len(elements),
               "photons": float(weights.sum()),
               "axis": axis.tolist(), "centre_m": centre.tolist(),
               "principal_rms_m": spread.tolist(),
               "longitudinal": describe(z, weights, "z"),
               "transverse_radius": describe(radius, weights, "rho"),
               "residual_time_ns": describe(tau, weights, "tau"),
               "cone_cosine": describe(elements.cone_cosine, weights, "mu_C"),
               "direction_cosine": describe(cosine, weights, "u.axis"),
               "azimuth_anisotropy": anisotropy,
               "correlations": {
                   "z_vs_cosine": weighted_correlation(z, cosine, weights),
                   "z_vs_tau": weighted_correlation(z, tau, weights),
                   "cosine_vs_tau": weighted_correlation(cosine, tau, weights)}}
        radius99 = row["transverse_radius"]["q99"]
        row["transverse_angle_at_nearest_mrad"] = (
            1e3 * radius99 / arguments.nearest_module_m)
        row["residual_time_over_span_ns"] = {
            "tau_q99": row["residual_time_ns"]["q99"],
            "emission_span": float(start.max())}
        report["events"].append(row)
        print(f"event {number}: {len(elements)} elements, axis rms "
              f"{spread[0]:.2f}/{spread[1]:.3f}/{spread[2]:.3f} m, "
              f"rho99 {radius99*100:.1f} cm ({row['transverse_angle_at_nearest_mrad']:.1f} mrad), "
              f"tau99 {row['residual_time_ns']['q99']:.3f} ns, "
              f"azimuth anisotropy {anisotropy:.3f}", flush=True)

    (out / f"{label}-axis.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"written: {out / (label + '-axis.json')}")


if __name__ == "__main__":
    main()
