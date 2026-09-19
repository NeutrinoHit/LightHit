"""A small multi-segment event, computed in three poses from one cache.

Run: python scripts/run_segment_event.py --output .build/review/event

The event is eight straight Cherenkov segments with their own positions,
directions, start times, lengths, velocities and light yields; nothing is
replaced by an average axis or a product of averaged distributions. The same
cache serves the original event, a translated copy and a rotated copy, because
moving the event changes the arguments of the cached moments, not the transport
solve behind them.

What the run produces, per pose:

* the complex spectrum at every receiver and frequency, saved as arrays;
* the charge at ``omega = 0``;
* time bins, with the unscattered light placed at its exact arrival time and
  the scattered orders inverted from the spectrum; negative bins from the
  finite frequency window are kept, not clipped;
* invariance checks: a joint rotation of event and receivers, a time shift,
  linearity in the photon yield, and additivity when a segment is split.

The medium is the open synthetic one; the yields are illustrative numbers, not
a calibrated Cherenkov spectrum.
"""
from pathlib import Path
import argparse
import json
import platform
import sys
from time import perf_counter

import numpy as np
from scipy.spatial.transform import Rotation

from lighthit import Medium, SolverSettings
from lighthit.cache import BandedResponseCache, CacheGrid, ResponseCache
from lighthit.readout import inverse_bins
from lighthit.experimental.cone_segment import ConeSegment, segment_spectrum
from lighthit.experimental.segment_reference import (SegmentGeometry,
                                                     segment_ballistic_field)

MEDIUM = Medium(0.04, 0.05, 0.7, 1.35, 450.0, "course-synthetic")
PHASE_INDEX = 1.34


def build_event():
    """Eight segments: a kinked primary and several delayed branches."""
    def unit(vector):
        vector = np.asarray(vector, float)
        return tuple(vector / np.linalg.norm(vector))

    speed = 0.299792458
    pieces = []
    # A primary that changes direction twice; each piece starts where the
    # previous one ended, at the time the particle gets there.
    start, direction, time = np.zeros(3), unit([0.1, 0.05, 1.0]), 0.0
    for length, beta, turn in ((4.0, 0.999, [0.05, 0.02, 0.0]),
                               (3.5, 0.995, [-0.08, 0.06, 0.0]),
                               (2.5, 0.990, [0.03, -0.10, 0.0])):
        pieces.append(ConeSegment(tuple(start), direction, length, beta,
                                  PHASE_INDEX, 9000.0, time))
        start = start + length * np.asarray(direction)
        time += length / (beta * speed)
        direction = unit(np.asarray(direction) + np.asarray(turn))
    # Branches leaving the primary at different points, times and speeds.
    branches = [((0.9, [0.6, -0.3, 0.7], 2.0, 0.96, 3000.0, 0.35)),
                ((2.2, [-0.5, 0.5, 0.7], 1.6, 0.93, 2200.0, 0.80)),
                ((4.6, [0.2, 0.8, 0.55], 2.4, 0.97, 2600.0, 1.60)),
                ((6.1, [-0.7, -0.2, 0.68], 1.2, 0.88, 1500.0, 2.10)),
                ((8.3, [0.35, -0.6, 0.72], 1.8, 0.91, 1800.0, 2.90))]
    primary = pieces[0]
    axis = np.asarray(primary.direction, float)
    for offset, direction, length, beta, yield_per_m, delay in branches:
        origin = np.asarray(primary.start_m, float) + offset * axis
        pieces.append(ConeSegment(tuple(origin), unit(direction), length, beta,
                                  PHASE_INDEX, yield_per_m, delay))
    return pieces


def receiver_array():
    """Two vertical strings of seven receivers each, fixed in the world frame."""
    heights = np.arange(-9.0, 34.0, 7.0)
    strings = [np.array([24.0, 3.0]), np.array([-17.0, 14.0])]
    return np.array([[x, y, z] for x, y in strings for z in heights])


def pose(segments, rotation=None, shift=None, pivot=None, delay=0.0):
    """Move a whole event rigidly, keeping every segment's own parameters."""
    matrix = np.eye(3) if rotation is None else rotation
    centre = np.zeros(3) if pivot is None else np.asarray(pivot, float)
    offset = np.zeros(3) if shift is None else np.asarray(shift, float)
    moved = []
    for piece in segments:
        start = matrix @ (np.asarray(piece.start_m, float) - centre) + centre + offset
        direction = matrix @ np.asarray(piece.direction, float)
        moved.append(ConeSegment(tuple(start), tuple(direction), piece.length_m,
                                 piece.beta, piece.phase_index, piece.photons_per_m,
                                 piece.start_time_ns + delay))
    return moved


def event_spectrum(cache, segments, receivers, *, longitudinal_order=48):
    """Sum the per-segment spectra. Shape: (frequency, receiver, order)."""
    total = None
    for piece in segments:
        part = segment_spectrum(cache, piece, receivers,
                                longitudinal_order=longitudinal_order)
        total = part if total is None else total + part
    return total


def ballistic_arrivals(segments, receivers):
    """Charge and exact arrival time of the unscattered light, per segment."""
    rows = []
    for piece in segments:
        geometry = SegmentGeometry.of(piece)
        fluence, _, time = segment_ballistic_field(receivers, geometry, MEDIUM)
        rows.append((fluence, time))
    return rows


def time_profile(cache, spectrum, segments, receivers, edges_ns):
    """Time bins: exact arrival for order 0, inversion for the scattered ones."""
    omega = cache.bands[0].grid.omega_per_ns if isinstance(cache, BandedResponseCache) \
        else cache.grid.omega_per_ns
    bins = np.zeros((len(receivers), len(edges_ns) - 1, 3))
    for order in (1, 2):
        bins[:, :, order] = inverse_bins(omega, spectrum[:, :, order], edges_ns)
    for fluence, time in ballistic_arrivals(segments, receivers):
        index = np.searchsorted(edges_ns, time) - 1
        inside = (fluence > 0) & (index >= 0) & (index < len(edges_ns) - 1)
        np.add.at(bins[:, :, 0], (np.flatnonzero(inside), index[inside]),
                  fluence[inside])
    return bins


def radial_span(events, receivers):
    """Every emission-node distance any pose will ask the cache for."""
    low, high = np.inf, 0.0
    for segments in events:
        for piece in segments:
            start = np.asarray(piece.start_m, float)
            axis = np.asarray(piece.direction, float)
            nodes = start[None, :] + np.linspace(0, piece.length_m, 128)[:, None] * axis
            distance = np.linalg.norm(receivers[:, None, :] - nodes[None, :, :], axis=2)
            low, high = min(low, distance.min()), max(high, distance.max())
    return low, high


def invariance_checks(cache, segments, receivers, spectrum):
    """Rotation, translation, time shift, yield linearity and splitting.

    Two measures are reported for each check. ``per_receiver`` divides by each
    receiver's own largest component, so a receiver a thousand times fainter
    than the brightest one is held to its own scale; ``per_event`` divides by
    the largest component anywhere. The faint receivers set ``per_receiver``,
    which is why the two differ by orders of magnitude.
    """
    omega = cache.grid.omega_per_ns
    scale = np.max(np.abs(spectrum), axis=(0, 2))
    results = {}

    def compare(difference, reference=None):
        reference = scale if reference is None else reference
        return {"per_receiver": float(np.max(np.abs(difference)
                                             / reference[None, :, None])),
                "per_event": float(np.max(np.abs(difference)) / np.max(reference))}

    matrix = Rotation.from_rotvec([0.31, -0.42, 0.65]).as_matrix()
    shift = np.array([13.0, -6.0, 4.0])
    moved = pose(segments, rotation=matrix, shift=shift)
    together = event_spectrum(cache, moved, receivers @ matrix.T + shift)
    results["joint_rotation_and_translation"] = compare(together - spectrum)

    delayed = event_spectrum(cache, pose(segments, delay=17.0), receivers)
    expected = spectrum * np.exp(1j * omega * 17.0)[:, None, None]
    results["time_shift"] = compare(delayed - expected)

    brighter = [ConeSegment(p.start_m, p.direction, p.length_m, p.beta,
                            p.phase_index, 3.0 * p.photons_per_m, p.start_time_ns)
                for p in segments]
    results["yield_linearity"] = compare(
        event_spectrum(cache, brighter, receivers) - 3 * spectrum, 3 * scale)

    piece = segments[1]
    half = piece.length_m / 2
    first = ConeSegment(piece.start_m, piece.direction, half, piece.beta,
                        piece.phase_index, piece.photons_per_m, piece.start_time_ns)
    second = ConeSegment(tuple(np.asarray(piece.start_m, float)
                               + half * np.asarray(piece.direction, float)),
                         piece.direction, half, piece.beta, piece.phase_index,
                         piece.photons_per_m,
                         piece.start_time_ns + half / (piece.beta * 0.299792458))
    whole = segment_spectrum(cache, piece, receivers, longitudinal_order=64)
    split = (segment_spectrum(cache, first, receivers, longitudinal_order=48)
             + segment_spectrum(cache, second, receivers, longitudinal_order=48))
    results["segment_splitting"] = compare(
        whole - split, np.max(np.abs(whole), axis=(0, 2)))

    far = receivers * 40.0
    try:
        event_spectrum(cache, segments, far)
        results["outside_cache_raises"] = False
    except ValueError as error:
        results["outside_cache_raises"] = True
        results["outside_cache_message"] = str(error)
    return results


def figures(out, events, labels, receivers, charges, bins, edges):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), sharex=True, sharey=True)
    for axis, segments, label in zip(axes, events, labels):
        for piece in segments:
            start = np.asarray(piece.start_m, float)
            end = start + piece.length_m * np.asarray(piece.direction, float)
            axis.plot([start[0], end[0]], [start[2], end[2]], "-", lw=2)
        axis.plot(receivers[:, 0], receivers[:, 2], "kv", ms=5, label="receivers")
        axis.set_title(label)
        axis.set_xlabel("x [m]")
        axis.grid(alpha=0.3)
    axes[0].set_ylabel("z [m]")
    axes[0].legend(loc="upper left", fontsize=8)
    figure.suptitle("Event geometry in three poses (x-z projection)")
    figure.tight_layout()
    figure.savefig(out / "event-geometry.png", dpi=130)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8.5, 4.4))
    index = np.arange(len(receivers))
    for label, value in zip(labels, charges):
        axis.semilogy(index, value.sum(axis=1), "o-", label=label)
    axis.set_xlabel("receiver index")
    axis.set_ylabel("charge [photons / m$^2$]")
    axis.set_title("Total charge per receiver, same cache, three poses")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(out / "event-charge.png", dpi=130)
    plt.close(figure)

    centres = (edges[:-1] + edges[1:]) / 2
    brightest = int(np.argmax(charges[0].sum(axis=1)))
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.4))
    for axis, receiver in zip(axes, (brightest, (brightest + 6) % len(receivers))):
        for order, name in enumerate(("unscattered", "once scattered", ">= 2")):
            axis.plot(centres, bins[0][receiver, :, order], label=name)
        axis.plot(centres, bins[0][receiver].sum(axis=1), "k--", lw=1, label="total")
        axis.set_title(f"receiver {receiver}")
        axis.set_xlabel("time [ns]")
        axis.set_ylabel("charge per bin [photons / m$^2$]")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    figure.suptitle("Time bins of the original pose; negative bins are kept as computed")
    figure.tight_layout()
    figure.savefig(out / "event-time-bins.png", dpi=130)
    plt.close(figure)


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=".build/review/event")
    parser.add_argument("--frequencies", type=int, default=101)
    parser.add_argument("--omega-max", type=float, default=0.5)
    parser.add_argument("--spatial-degree", type=int, default=340)
    parser.add_argument("--k-max", type=float, default=6.0)
    parser.add_argument("--longitudinal-order", type=int, default=48)
    arguments = parser.parse_args()
    out = Path(arguments.output)
    out.mkdir(parents=True, exist_ok=True)

    segments = build_event()
    receivers = receiver_array()
    # Rotate about the y axis so the change is visible in the x-z view.
    matrix = Rotation.from_rotvec([0.0, 0.9, 0.0]).as_matrix()
    centroid = np.mean([piece.start_m for piece in segments], axis=0)
    events = [segments,
              pose(segments, shift=np.array([9.0, -7.0, 5.0])),
              pose(segments, rotation=matrix, pivot=centroid)]
    labels = ["original", "translated", "rotated"]

    low, high = radial_span(events, receivers)
    omega = np.linspace(0.0, arguments.omega_max, arguments.frequencies)
    settings = SolverSettings(32, arguments.spatial_degree, arguments.k_max, 0.04, 10)
    grid = CacheGrid.geometric(low * 0.97, high * 1.03, 44, omega)
    start = perf_counter()
    cache = BandedResponseCache([ResponseCache.build(MEDIUM, settings, grid)])
    build_seconds = perf_counter() - start

    # cache_fast wraps the single band with a compiled Horner spline instead
    # of scipy's; event_spectrum -> segment_spectrum -> _scatter takes a bare
    # band exactly like a one-band BandedResponseCache (same code path), so
    # this changes nothing about what gets computed, only how fast the
    # radial moments are evaluated. time_profile keeps the original `cache`
    # since it only reads bookkeeping (grid.omega_per_ns), not moments.
    try:
        from lighthit.experimental.spline_fast import PreparedMultipoles
        cache_fast = PreparedMultipoles.of(cache.bands[0])
        spline_backend = "numba (spline_fast.PreparedMultipoles)"
    except ImportError:
        cache_fast = cache.bands[0]
        spline_backend = "scipy (ResponseCache.moments_at)"

    edges = np.arange(0.0, 620.0, 5.0)
    spectra, charges, profiles, timings = [], [], [], []
    for segments_here in events:
        began = perf_counter()
        spectrum = event_spectrum(cache_fast, segments_here, receivers,
                                  longitudinal_order=arguments.longitudinal_order)
        seconds = perf_counter() - began
        spectra.append(spectrum)
        charges.append(spectrum[0].real)
        profiles.append(time_profile(cache, spectrum, segments_here, receivers, edges))
        timings.append(seconds)

    checks = invariance_checks(cache_fast, segments, receivers, spectra[0])
    figures(out, events, labels, receivers, charges, profiles, edges)

    np.savez_compressed(
        out / "event-arrays.npz", omega_per_ns=omega, receivers_m=receivers,
        time_edges_ns=edges,
        **{f"spectrum_{label}": value for label, value in zip(labels, spectra)},
        **{f"charge_{label}": value for label, value in zip(labels, charges)},
        **{f"bins_{label}": value for label, value in zip(labels, profiles)})

    report = {
        "medium": {"absorption_per_m": MEDIUM.absorption_per_m,
                   "scattering_per_m": MEDIUM.scattering_per_m, "g": MEDIUM.g,
                   "group_index": MEDIUM.group_index,
                   "provenance": MEDIUM.provenance},
        "segments": [{"start_m": p.start_m, "direction": p.direction,
                      "length_m": p.length_m, "beta": p.beta,
                      "cone_cosine": p.cone_cosine,
                      "photons_per_m": p.photons_per_m,
                      "start_time_ns": p.start_time_ns} for p in segments],
        "receivers_m": receivers,
        "cache": {"radius_range_m": [low * 0.97, high * 1.03], "radii": 44,
                  "frequencies": arguments.frequencies,
                  "omega_max_per_ns": arguments.omega_max,
                  "spline_backend": spline_backend,
                  "settings": {"scattering_degree": settings.scattering_degree,
                               "spatial_degree": settings.spatial_degree,
                               "k_max_per_m": settings.k_max_per_m,
                               "k_panel_per_m": settings.k_panel_per_m},
                  "build_seconds": build_seconds,
                  "moment_megabytes": cache.bands[0].moments.nbytes / 2 ** 20},
        "poses": [{"label": label, "seconds": seconds,
                   "charge_by_receiver": charge.sum(axis=1),
                   "charge_by_order": charge.sum(axis=0),
                   "negative_bin_fraction":
                       float(np.sum(np.minimum(bins.sum(axis=2), 0.0))
                             / np.sum(np.abs(bins.sum(axis=2))))}
                  for label, seconds, charge, bins in
                  zip(labels, timings, charges, profiles)],
        "invariance": checks,
        "longitudinal_order": arguments.longitudinal_order,
        "environment": {"python": sys.version, "platform": platform.platform(),
                        "numpy": np.__version__},
        "notes": ["Photon yields are illustrative inputs, not a calibrated "
                  "Cherenkov spectrum.",
                  "Time bins keep negative values produced by the finite "
                  "frequency window; nothing is clipped or renormalised.",
                  "Order 0 is placed at its exact arrival time, so it is not "
                  "subject to the inversion's ringing."],
    }
    (out / "event.json").write_text(json.dumps(report, default=serial, indent=2),
                                    encoding="utf-8")
    print(json.dumps({"build_seconds": build_seconds,
                      "pose_seconds": timings, "invariance": checks},
                     default=serial, indent=2))


if __name__ == "__main__":
    main()
