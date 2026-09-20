#!/usr/bin/env python3
"""Measure the directional-OM route stage by stage, against the m=0 route.

Every stage is timed separately, because they scale differently and a single
total hides which one is the bottleneck:

    angular      the m-block solve, per frequency
    cache        building and storing the radial coefficients
    compile      projecting the event onto the source channels
    apply        contracting source, kernel and module response
    ballistic    the unscattered term with the exact arrival direction

The reference for each is the same stage of the existing m = 0 route, run on
the same geometry with the same settings, so the reported factor is a ratio of
measurements and not of estimates.

No private data is used: the geometry is a synthetic array of the stated shape
and the module response is a stand-in cubic. ``--bgvd <path>`` substitutes the
private model at runtime and is what ``scripts/smoke_bgvd_directional.py``
does for the final check; nothing derived from it is written here.

Examples
--------
python scripts/benchmark_directional.py --output .build/review/directional
python scripts/benchmark_directional.py --modules 2304 --wavelengths 9 \\
    --frequencies 41 --source-degree 32 --stages angular cache
"""
from __future__ import annotations

import argparse
import json
import resource
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import lighthit as lh                                             # noqa: E402
from lighthit.angular import angular_components, resolvent_rows   # noqa: E402
from lighthit.cache import CacheGrid, ResponseCache               # noqa: E402
from lighthit.directional import (DirectionalCache,               # noqa: E402
                                  acceptance_bandwidth,
                                  directional_response)
from lighthit.green import SolverSettings                         # noqa: E402
from lighthit.medium import Medium                                # noqa: E402


def peak_rss_mib():
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports kibibytes, macOS bytes.
    return usage / 1024 if sys.platform != "darwin" else usage / 1024 ** 2


def stand_in_acceptance(x):
    """A nonnegative cubic of roughly the right shape. Not a measured curve."""
    x = np.asarray(x, float)
    return 0.40 + 0.46 * x + 0.10 * x ** 2 + 0.04 * x ** 3


def synthetic_array(modules):
    """A BGVD-shaped array: clusters of strings of downward-looking modules."""
    per_string = 36
    per_cluster = 8 * per_string
    clusters = max(1, modules // per_cluster)
    positions, orientations = [], []
    for cluster in range(clusters):
        centre = np.array([300.0 * (cluster % 3), 300.0 * (cluster // 3), 0.0])
        for string in range(8):
            angle = 2 * np.pi * string / 8
            base = centre + 60.0 * np.array([np.cos(angle), np.sin(angle), 0.0])
            for module in range(per_string):
                positions.append(base + np.array([0.0, 0.0, -15.0 * module]))
                orientations.append([0.0, 0.0, 1.0])
    positions = np.asarray(positions)[:modules]
    orientations = np.asarray(orientations, float)[:modules]
    return positions, orientations


def spectral_medium():
    wavelength = np.linspace(370.0, 600.0, 24)
    absorption = 0.012 + 0.00004 * (wavelength - 400.0) ** 2 / 100.0
    scattering = 0.045 * (450.0 / wavelength) ** 0.8
    phase = 1.34 + 6.0e3 / wavelength ** 2
    group = phase + 0.02
    return lh.SpectralMedium(wavelength, absorption, scattering, phase, group,
                             g=0.9, provenance="synthetic benchmark water")


def timed(function, *args, **kwargs):
    before = perf_counter()
    value = function(*args, **kwargs)
    return value, perf_counter() - before


def stage_angular(medium, settings, frequencies, acceptance_degree, repeats=1):
    k, _ = settings.quadrature()
    omega = np.linspace(0.0, 0.15, frequencies)
    report = {}
    old = new = 0.0
    for _ in range(repeats):
        for frequency in omega:
            _, seconds = timed(angular_components, k, frequency, medium,
                               settings.scattering_degree, settings.spatial_degree,
                               backend="numpy")
            old += seconds
            _, seconds = timed(resolvent_rows, k, frequency, medium,
                               settings.scattering_degree, settings.spatial_degree,
                               acceptance_degree, backend="numpy", report=report)
            new += seconds
    return {"m0_seconds": old / repeats, "m_blocks_seconds": new / repeats,
            "factor": new / old if old else None, "k_nodes": len(k),
            "frequencies": frequencies, **report}


def stage_cache(medium, settings, grid, acceptance_degree):
    old, old_seconds = timed(ResponseCache.build, medium, settings, grid,
                             radial_phase="flight")
    new, new_seconds = timed(DirectionalCache.build, medium, settings, grid,
                             acceptance_degree, radial_phase="flight")
    return new, {"m0_seconds": old_seconds, "m_blocks_seconds": new_seconds,
                 "factor": new_seconds / old_seconds,
                 "m0_bytes": int(old.moments.nbytes),
                 "m_blocks_bytes": int(new.coefficients.nbytes),
                 "bytes_factor": new.coefficients.nbytes / old.moments.nbytes,
                 "pairs": new.table.pairs, "triples": new.table.triples,
                 "m0_timings": old.timings_s, "m_blocks_timings": new.timings_s}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", default=".build/review/directional")
    parser.add_argument("--modules", type=int, default=2304)
    parser.add_argument("--wavelengths", type=int, default=9)
    parser.add_argument("--frequencies", type=int, default=41)
    parser.add_argument("--source-degree", type=int, default=32)
    parser.add_argument("--azimuthal-degree", type=int, default=4)
    parser.add_argument("--scattering-degree", type=int, default=24)
    parser.add_argument("--acceptance-degree", type=int, default=3)
    parser.add_argument("--radial-nodes", type=int, default=220)
    parser.add_argument("--track-length-m", type=float, default=8.0)
    parser.add_argument("--cell-m", type=float, default=0.12)
    parser.add_argument("--apply-modules", type=int, default=16,
                        help="modules used for the apply stage; the numpy "
                             "reference is far too slow for the whole array")
    parser.add_argument("--stages", nargs="+",
                        default=["angular", "cache", "compile", "apply"])
    parser.add_argument("--bgvd", default=None,
                        help="path to a private bgvd-model checkout, used at "
                             "runtime only")
    arguments = parser.parse_args()
    output = Path(arguments.output)
    output.mkdir(parents=True, exist_ok=True)

    if arguments.bgvd:
        from lighthit.bgvd import load_bgvd_model
        model = load_bgvd_model(arguments.bgvd)
        medium_spectral, detector = model.medium, model.detector
        provenance = "private bgvd-model (runtime only)"
    else:
        medium_spectral = spectral_medium()
        positions, orientations = synthetic_array(arguments.modules)
        detector = lh.DetectorArray(
            positions, orientations, np.pi * 0.2159 ** 2, stand_in_acceptance,
            lambda w: 0.25 * np.exp(-((np.asarray(w, float) - 420.0) / 90.0) ** 2))
        provenance = "synthetic array and stand-in cubic acceptance"

    bandwidth, alpha, residual = acceptance_bandwidth(detector.angular_acceptance)
    settings = SolverSettings(arguments.scattering_degree, arguments.source_degree,
                              8.0, 0.04, 10, True)
    band = medium_spectral.band(float(np.median(medium_spectral.wavelength_nm)))
    omega = np.linspace(0.0, 0.15, arguments.frequencies)
    grid = CacheGrid.geometric(3.0, 300.0, arguments.radial_nodes, omega)

    results = {"provenance": provenance, "modules": len(detector),
               "wavelengths": arguments.wavelengths,
               "frequencies": arguments.frequencies,
               "source_degree": arguments.source_degree,
               "azimuthal_degree": arguments.azimuthal_degree,
               "scattering_degree": arguments.scattering_degree,
               "acceptance_degree_requested": arguments.acceptance_degree,
               "acceptance_bandwidth_measured": bandwidth,
               "acceptance_residual_above_bandwidth": residual,
               "radial_nodes": arguments.radial_nodes,
               "k_nodes": len(settings.quadrature()[0])}

    if "angular" in arguments.stages:
        print("angular ...", flush=True)
        results["angular"] = stage_angular(band, settings, arguments.frequencies,
                                           arguments.acceptance_degree)
        results["angular"]["peak_rss_mib"] = peak_rss_mib()
        print(json.dumps(results["angular"], indent=2))

    cache = None
    if "cache" in arguments.stages:
        print("cache ...", flush=True)
        cache, summary = stage_cache(band, settings, grid,
                                     arguments.acceptance_degree)
        summary["peak_rss_mib"] = peak_rss_mib()
        results["cache"] = summary
        print(json.dumps({k: v for k, v in summary.items()
                          if not k.endswith("timings")}, indent=2))

    if "compile" in arguments.stages or "apply" in arguments.stages:
        from lighthit.experimental.axial_source import AxisFrame, AxialSource
        try:
            from lighthit.experimental.axial_fast import compile_axial_source_fast
            compiler, compiler_name = compile_axial_source_fast, "numba"
        except ImportError:
            compiler, compiler_name = AxialSource.of, "numpy"
        centre = detector.positions_m.mean(axis=0) + np.array([20.0, 15.0, 40.0])
        track = lh.CherenkovTrack(centre, [0.3, 0.4, np.sqrt(1 - 0.25)],
                                  arguments.track_length_m, beta=0.999)
        elements = track.to_elements(step_m=max(arguments.cell_m, 0.25))
        field0 = elements.field(0)
        frame = AxisFrame.of(field0)
        compiled, compile_seconds = timed(
            compiler, field0, arguments.source_degree, omega,
            azimuthal_degree=arguments.azimuthal_degree, cell_m=arguments.cell_m,
            element_order=2, frame=frame)
        results["compile"] = {
            "compiler": compiler_name, "seconds": compile_seconds,
            "cells": int(compiled.channels.shape[0]),
            "channels": int(compiled.channels.shape[1]),
            "source_bytes": int(compiled.channels.nbytes),
            "elements": len(field0), "peak_rss_mib": peak_rss_mib()}
        print(json.dumps(results["compile"], indent=2))

    if "apply" in arguments.stages:
        if cache is None:
            cache = DirectionalCache.build(band, settings, grid,
                                           arguments.acceptance_degree,
                                           radial_phase="flight")
        legacy = ResponseCache.build(band, settings, grid, radial_phase="flight")
        count = min(arguments.apply_modules, len(detector))
        distance = np.linalg.norm(detector.positions_m - centre, axis=1)
        order = np.argsort(distance)
        inside = order[(distance[order] > 20.0) & (distance[order] < 250.0)][:count]
        receivers = detector.positions_m[inside]
        looks = -detector.orientations[inside]
        # Compare like with like. Timing the fused directional kernel against
        # the *numpy* m = 0 route once made the new method look faster than the
        # old one, which it is not: it measured numba against numpy and called
        # the difference physics. Each pair below names both of its backends.
        from lighthit.experimental.axial_source import axial_response

        class _Adapter:
            degree = settings.spatial_degree
            omega_per_ns = omega

            @staticmethod
            def multipoles(radii, frequency_slice=slice(None)):
                block = legacy.moments_at(radii)[frequency_slice]
                return np.transpose(block[:, :, :, 1], (1, 2, 0))

        apply_alpha = alpha[:bandwidth + 1]
        summary = {"modules": int(count),
                   "cells": int(compiled.channels.shape[0]),
                   "frequencies": len(omega)}
        _, numpy_m0 = timed(axial_response, _Adapter(), compiled, receivers)
        summary["m0_numpy_seconds"] = numpy_m0
        _, numpy_directional = timed(
            directional_response, cache, compiled, receivers, looks,
            apply_alpha, source_omega_per_ns=omega)
        summary["directional_numpy_seconds"] = numpy_directional
        summary["directional_over_m0_numpy"] = (
            numpy_directional / numpy_m0 if numpy_m0 else None)
        try:
            from lighthit._directional_numba import PreparedDirectionalKernel
            from lighthit.experimental.axial_fast import PreparedAxialKernel
        except ImportError:
            summary["numba"] = "unavailable: only the numpy pair was measured"
        else:
            prepared = PreparedDirectionalKernel.from_cache(cache)
            _, summary["directional_numba_first_call_seconds"] = timed(
                prepared.apply, compiled, receivers, looks, apply_alpha,
                source_omega_per_ns=omega)
            _, numba_directional = timed(
                prepared.apply, compiled, receivers, looks, apply_alpha,
                source_omega_per_ns=omega)
            legacy_prepared = PreparedAxialKernel.from_cache(
                legacy, degree=settings.spatial_degree)
            _, summary["m0_numba_first_call_seconds"] = timed(
                legacy_prepared.apply, compiled, receivers,
                source_omega_per_ns=omega)
            _, numba_m0 = timed(legacy_prepared.apply, compiled, receivers,
                                source_omega_per_ns=omega)
            summary["directional_numba_seconds"] = numba_directional
            summary["m0_numba_seconds"] = numba_m0
            # the only two ratios worth quoting
            summary["directional_over_m0_numba"] = (
                numba_directional / numba_m0 if numba_m0 else None)
            summary["directional_numba_speedup_over_numpy"] = (
                numpy_directional / numba_directional if numba_directional else None)
            summary["seconds_per_module_numba"] = numba_directional / count
        summary["peak_rss_mib"] = peak_rss_mib()
        results["apply"] = summary
        print(json.dumps(results["apply"], indent=2))

    results["peak_rss_mib"] = peak_rss_mib()
    path = output / "directional-benchmark.json"
    path.write_text(json.dumps(results, indent=2, default=str))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
