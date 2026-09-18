"""Independent controls for the finite Cherenkov segment.

Run: python scripts/verify_segment.py --output .build/review/segment

Nothing here is a self-test of the cached route against itself. Every reference
number comes from :mod:`lighthit.experimental.segment_reference`, which builds
the once-scattered signal in coordinate space, and from a ray-counting estimate
of the unscattered fluence. The script reports four separate things:

1. a directed point flash, where the cached first order is compared with a
   single coordinate quadrature while the spatial cutoff is refined, so that
   angular truncation and spatial discretisation are not confused;
2. the unscattered segment fluence against ray counting;
3. the finite segment's first order over several geometries, against the
   coordinate reference using the same truncated kernel the solver applies,
   with the prototype's longitudinal quadrature refined;
4. the same comparison at nonzero frequency, and one Monte Carlo cross-check.

The ``>=2`` component has no independent control here. Its stability under
refinement is measured and reported as stability, not as verification.
"""
from pathlib import Path
import argparse
import json
import platform
import sys
from time import perf_counter

import numpy as np
from scipy.special import eval_legendre

from lighthit import Medium, SolverSettings
from lighthit.cache import BandedResponseCache, CacheGrid, ResponseCache
from lighthit.experimental.cone_segment import (ConeSegment, ballistic_spectrum,
                                                segment_spectrum)
from lighthit.experimental.segment_reference import (SegmentGeometry,
                                                     ballistic_by_ray_counting,
                                                     directed_first_order,
                                                     segment_first_order,
                                                     segment_first_order_mc)

MEDIUM = Medium(0.04, 0.05, 0.7, 1.35, 450.0, "course-synthetic")


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    raise TypeError(type(value).__name__)


def directed_point_check(quick):
    """Cached finite-L first order against one coordinate quadrature."""
    radius, cosine, degree = 12.0, 0.3, 32
    detector = np.array([0.0, 0.0, radius])
    emission = np.array([np.sqrt(1 - cosine ** 2), 0.0, cosine])
    radii = np.array([10.0, radius, 15.0])
    truncated = directed_first_order([0.0], np.zeros(3), emission, detector,
                                     MEDIUM, kind="truncated", degree=degree)[0].real
    full = directed_first_order([0.0], np.zeros(3), emission, detector,
                                MEDIUM, kind="hg")[0].real
    rows = []
    ladder = [(6.0, 160, 0.03), (10.0, 260, 0.02), (16.0, 400, 0.02)]
    if not quick:
        ladder.append((24.0, 560, 0.015))
    for k_max, spatial, panel in ladder:
        settings = SolverSettings(degree, spatial, k_max, panel, 10)
        start = perf_counter()
        cache = ResponseCache.build(MEDIUM, settings, CacheGrid(radii, np.array([0.0])))
        weights = eval_legendre(np.arange(spatial + 1), cosine)
        value = float((cache.moments[0, 1, :, 0] * weights).sum().real)
        rows.append({"k_max_per_m": k_max, "spatial_degree": spatial,
                     "k_panel_per_m": panel, "cached_first_order": value,
                     "relative_to_truncated_reference": abs(value - truncated) / truncated,
                     "build_seconds": perf_counter() - start})
    truncation = []
    for level in (4, 8, 16, 32, 64):
        value = directed_first_order([0.0], np.zeros(3), emission, detector, MEDIUM,
                                     kind="truncated", degree=level)[0].real
        truncation.append({"L": level, "coordinate_first_order": value,
                           "relative_to_full_hg": abs(value - full) / full})
    return {"geometry": {"radius_m": radius, "emission_cosine": cosine,
                         "scattering_degree": degree},
            "reference_truncated": truncated, "reference_full_hg": full,
            "spatial_refinement": rows, "angular_truncation": truncation,
            "note": "The residual of the cached route shrinks with the spatial "
                    "cutoff at fixed L, so it is not an L effect."}


def ballistic_check(segments, detectors, quick):
    """Analytic cone Jacobian against counting rays through a small sphere."""
    rows = []
    samples = 400_000 if quick else 2_000_000
    for name, segment in segments.items():
        for index, detector in enumerate(detectors):
            analytic = float(ballistic_spectrum([0.0], segment, detector[None, :]
                                                - np.asarray(segment.start_m),
                                                MEDIUM)[0, 0].real)
            if analytic == 0.0:
                rows.append({"segment": name, "detector": index, "analytic": 0.0,
                             "counted": None, "note": "cone does not reach it"})
                continue
            for radius in (0.2, 0.05, 0.01):
                counted, error, edge = ballistic_by_ray_counting(
                    segment, detector, MEDIUM, acceptance_radius_m=radius,
                    samples=samples)
                rows.append({"segment": name, "detector": index,
                             "sphere_radius_m": radius, "analytic": analytic,
                             "counted": counted, "counting_error": error,
                             "window_edge_fraction": edge,
                             "relative_difference": (counted - analytic) / analytic,
                             "difference_in_sigma": (counted - analytic) / error
                             if error > 0 else None})
    return rows


def _band_for(segment, detectors, degree, k_max, spatial, panel, radii_count,
              frequencies):
    """A cache band that covers every emission-node distance of this geometry."""
    geometry = SegmentGeometry.of(segment)
    start = np.asarray(geometry.start_m, float)
    axis = np.asarray(geometry.direction, float)
    nodes = start[None, :] + np.linspace(0, geometry.length_m, 64)[:, None] * axis[None, :]
    distances = np.linalg.norm(detectors[:, None, :] - nodes[None, :, :], axis=2)
    low, high = distances.min() * 0.97, distances.max() * 1.03
    grid = CacheGrid.geometric(low, high, radii_count, frequencies)
    settings = SolverSettings(degree, spatial, k_max, panel, 10)
    return ResponseCache.build(MEDIUM, settings, grid)


def segment_check(segments, detectors, quick, frequencies):
    """Prototype segment orders against the coordinate reference.

    The charge comparison runs over every geometry; the frequency comparison
    and the full-Henyey-Greenstein comparison are done on one geometry each,
    because the coordinate reference costs tens of seconds per call and
    repeating it everywhere would buy nothing new.
    """
    degree = 32
    spatial, k_max, panel = (260, 10.0, 0.02) if quick else (400, 16.0, 0.02)
    radii_count = 24 if quick else 40
    fine = dict(radial_order=20, polar_order=20) if quick \
        else dict(radial_order=28, polar_order=28)
    coarse_orders = {key: value - 8 for key, value in fine.items()}
    tolerance = dict(epsrel=1e-8)
    rows = []
    for name, segment in segments.items():
        start = perf_counter()
        cache = _band_for(segment, detectors, degree, k_max, spatial, panel,
                          radii_count, frequencies)
        build_seconds = perf_counter() - start
        banded = BandedResponseCache([cache])
        prototype = {}
        for order in (16, 32, 64) if quick else (16, 32, 64, 128):
            prototype[order] = segment_spectrum(banded, segment, detectors,
                                                longitudinal_order=order)
        for index, detector in enumerate(detectors):
            geometry = SegmentGeometry.of(segment)
            root, _ = geometry.ballistic_root(detector)
            lit = 0.0 < root < geometry.length_m
            began = perf_counter()
            reference = segment_first_order([0.0], segment, detector, MEDIUM,
                                            kind="truncated", degree=degree,
                                            **fine, **tolerance)[0]
            reference_seconds = perf_counter() - began
            coarse = segment_first_order([0.0], segment, detector, MEDIUM,
                                         kind="truncated", degree=degree,
                                         **coarse_orders, **tolerance)[0]
            entry = {"segment": name, "detector": index, "ballistically_lit": lit,
                     "ballistic_root_m": root,
                     "reference_first_order": reference.real,
                     "reference_self_convergence":
                         float(abs(reference - coarse) / abs(reference)),
                     "reference_seconds": reference_seconds,
                     "longitudinal_scan": {}}
            for order, values in prototype.items():
                first = values[0, index, 1]
                entry["longitudinal_scan"][order] = {
                    "first_order": first.real,
                    "relative_to_reference": float(abs(first - reference)
                                                   / abs(reference)),
                    "two_or_more": values[0, index, 2].real,
                    "ballistic": values[0, index, 0].real}
            if name == "long_fast" and index == 1:
                # Far from any ballistic root the tensor rule converges too,
                # so the two discretisations can be compared directly.
                product = segment_first_order([0.0], segment, detector, MEDIUM,
                                              kind="truncated", degree=degree,
                                              product_rule=True,
                                              longitudinal_order=24,
                                              azimuth_order=48, **tolerance)[0]
                entry["product_rule_reference"] = product.real
                entry["polar_vs_product_rule"] = \
                    float(abs(product - reference) / abs(reference))
            if name == "long_fast" and index == 0:
                full = segment_first_order([0.0], segment, detector, MEDIUM,
                                           kind="hg", **fine, **tolerance)[0]
                entry["reference_full_hg"] = full.real
                entry["truncation_share_of_reference"] = \
                    float(abs(full - reference) / abs(full))
                spectrum = segment_first_order(frequencies, segment, detector,
                                               MEDIUM, kind="truncated",
                                               degree=degree, **fine, **tolerance)
                top = max(prototype)
                entry["frequency_scan"] = [
                    {"omega_per_ns": float(w),
                     "prototype_real": prototype[top][j, index, 1].real,
                     "prototype_imag": prototype[top][j, index, 1].imag,
                     "reference_real": spectrum[j].real,
                     "reference_imag": spectrum[j].imag,
                     "relative_difference": float(abs(prototype[top][j, index, 1]
                                                      - spectrum[j]) / abs(spectrum[j]))}
                    for j, w in enumerate(frequencies)]
            rows.append(entry)
        rows[-1]["cache_build_seconds"] = build_seconds
        rows[-1]["cache_settings"] = {"scattering_degree": degree,
                                      "spatial_degree": spatial,
                                      "k_max_per_m": k_max, "k_panel_per_m": panel,
                                      "radii": radii_count}
    return rows


def monte_carlo_cross_check(segments, detectors, quick):
    """One sampled control of the same first order, with its own caveats."""
    samples = 400_000 if quick else 2_000_000
    rows = []
    for name in list(segments)[:2]:
        segment = segments[name]
        for index in (0, 1):
            detector = detectors[index]
            quadrature = segment_first_order([0.0], segment, detector, MEDIUM,
                                             kind="hg", radial_order=20,
                                             polar_order=20, epsrel=1e-8)[0].real
            began = perf_counter()
            mean, error = segment_first_order_mc([0.0], segment, detector, MEDIUM,
                                                 kind="hg", samples=samples,
                                                 batches=16, seed=20260917)
            rows.append({"segment": name, "detector": index,
                         "quadrature": quadrature, "monte_carlo": mean[0].real,
                         "batch_standard_error": error[0].real,
                         "relative_difference": (mean[0].real - quadrature) / quadrature,
                         "seconds": perf_counter() - began,
                         "samples": samples})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=".build/review/segment")
    parser.add_argument("--quick", action="store_true",
                        help="Smaller caches, fewer samples, coarser references")
    arguments = parser.parse_args()
    out = Path(arguments.output)
    out.mkdir(parents=True, exist_ok=True)

    segments = {
        "short_fast": ConeSegment((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 2.0, 0.99, 1.34, 1.0),
        "long_fast": ConeSegment((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 8.0, 0.99, 1.34, 1.0),
        "slower_tilted": ConeSegment((1.0, -2.0, 0.0),
                                     tuple(np.array([0.3, 0.2, 0.93])
                                           / np.linalg.norm([0.3, 0.2, 0.93])),
                                     5.0, 0.80, 1.34, 1.0),
    }
    detectors = np.array([[10.0, 0.0, 14.0],    # lit by the fast cones
                          [12.0, 0.0, -6.0],    # behind the segments
                          [-24.0, 9.0, 21.0]])  # farther away
    frequencies = np.array([0.0, 0.05, 0.2])

    report = {"medium": {"absorption_per_m": MEDIUM.absorption_per_m,
                         "scattering_per_m": MEDIUM.scattering_per_m,
                         "g": MEDIUM.g, "group_index": MEDIUM.group_index,
                         "provenance": MEDIUM.provenance},
              "detectors_m": detectors,
              "segments": {name: {"start_m": s.start_m, "direction": s.direction,
                                  "length_m": s.length_m, "beta": s.beta,
                                  "phase_index": s.phase_index,
                                  "cone_cosine": s.cone_cosine,
                                  "photons_per_m": s.photons_per_m}
                           for name, s in segments.items()},
              "frequencies_per_ns": frequencies,
              "environment": {"python": sys.version, "platform": platform.platform(),
                              "numpy": np.__version__}}
    began = perf_counter()
    report["directed_point_flash"] = directed_point_check(arguments.quick)
    print("directed point flash done", flush=True)
    report["ballistic_ray_counting"] = ballistic_check(segments, detectors, arguments.quick)
    print("ballistic counting done", flush=True)
    report["segment_first_order"] = segment_check(segments, detectors,
                                                  arguments.quick, frequencies)
    print("segment comparison done", flush=True)
    report["monte_carlo"] = monte_carlo_cross_check(segments, detectors, arguments.quick)
    report["total_seconds"] = perf_counter() - began
    report["caveats"] = [
        "The >=2 order has no independent control in this script.",
        "The sampled control has a logarithmically divergent variance: the "
        "unscattered cone field carries a 1/impact ridge along the segment, so "
        "its batch standard error understates the true uncertainty.",
        "Every number uses the open synthetic medium; no private optics.",
    ]
    (out / "segment-verification.json").write_text(
        json.dumps(report, default=serial, indent=2), encoding="utf-8")
    print(f"written: {out / 'segment-verification.json'}")


if __name__ == "__main__":
    main()
