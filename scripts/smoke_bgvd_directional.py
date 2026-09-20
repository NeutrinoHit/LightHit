#!/usr/bin/env python3
"""Private BGVD check of the directional-OM route. Run locally, keep the output local.

Nothing from the private ``bgvd-model`` checkout is copied, printed in full or
written anywhere inside the repository: the default output directory is
``.build/`` which ``.gitignore`` excludes, and what is written there is a small
JSON of ratios, residuals and timings.

What it checks, in this order:

1. the measured Legendre bandwidth of the real module response, and whether the
   private polynomial is clipped at zero anywhere in ``[-1, 1]`` -- if it is,
   ``L_A = 3`` is a truncation and the residual says how big a one;
2. that an isotropic acceptance reproduces the existing m = 0 answer on the
   real geometry, to the last bit;
3. that a joint rotation of source, modules and geometry leaves the answer
   alone, and that turning the modules alone does not;
4. a full directional transport on the real array, with per-stage timings.

Usage
-----
    python scripts/smoke_bgvd_directional.py --bgvd ~/path/to/bgvd-model-master
    python scripts/smoke_bgvd_directional.py --bgvd ... --modules 2304 \\
        --frequencies 41 --output .build/review/bgvd-directional
"""
from __future__ import annotations

import argparse
import json
import resource
import sys
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import lighthit as lh                                              # noqa: E402
from lighthit.bgvd import load_bgvd_model                          # noqa: E402
from lighthit.cache import CacheGrid, ResponseCache                # noqa: E402
from lighthit.directional import (DirectionalCache,                # noqa: E402
                                  acceptance_bandwidth,
                                  directional_response)
from lighthit.green import SolverSettings                          # noqa: E402
from lighthit.experimental.axial_source import AxisFrame           # noqa: E402
from lighthit.experimental.event_moments import real_spherical_harmonics  # noqa: E402


def peak_rss_mib():
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return usage / 1024 if sys.platform != "darwin" else usage / 1024 ** 2


def acceptance_report(detector, probe=2048):
    bandwidth, alpha, residual = acceptance_bandwidth(
        detector.angular_acceptance, max_degree=64, quadrature_order=probe)
    x = np.linspace(-1.0, 1.0, 4001)
    values = np.asarray(detector.angular_acceptance(x), float)
    # The private response is max(polynomial, 0); if the polynomial itself dips
    # below zero inside the interval the clip is active and the response is not
    # a cubic any more. A flat run of exact zeros is the signature.
    zeros = np.isclose(values, 0.0, atol=1e-15)
    clipped = bool(zeros.sum() > 2)
    return {"measured_bandwidth": int(bandwidth),
            "residual_above_bandwidth": float(residual),
            "alpha_ratio_to_peak": [float(v) for v in
                                    np.abs(alpha[:8]) / np.max(np.abs(alpha))],
            "clip_active_in_interval": clipped,
            "zero_fraction_of_interval": float(zeros.mean()),
            "is_cubic_within_1e-10": bool(bandwidth <= 3 and residual < 1e-10)}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bgvd", required=True,
                        help="path to the private bgvd-model checkout")
    parser.add_argument("--output", default=".build/review/bgvd-directional")
    parser.add_argument("--dataset", default="2021")
    parser.add_argument("--modules", type=int, default=0,
                        help="0 keeps the whole array")
    parser.add_argument("--frequencies", type=int, default=41)
    parser.add_argument("--wavelength-nodes", type=int, default=9)
    parser.add_argument("--source-degree", type=int, default=32)
    parser.add_argument("--azimuthal-degree", type=int, default=4)
    parser.add_argument("--scattering-degree", type=int, default=24)
    parser.add_argument("--acceptance-degree", type=int, default=None)
    parser.add_argument("--radial-nodes", type=int, default=220)
    parser.add_argument("--track-length-m", type=float, default=20.0)
    parser.add_argument("--cache-directory", default=None)
    parser.add_argument("--angular-backend", default="numba",
                        choices=("numba", "numpy"))
    arguments = parser.parse_args()
    output = Path(arguments.output)
    output.mkdir(parents=True, exist_ok=True)

    model = load_bgvd_model(arguments.bgvd, dataset=arguments.dataset)
    detector = model.detector
    results = {"dataset": arguments.dataset, "modules_in_model": len(detector),
               "acceptance": acceptance_report(detector)}
    print(json.dumps(results["acceptance"], indent=2))

    if arguments.modules and arguments.modules < len(detector):
        keep = np.linspace(0, len(detector) - 1, arguments.modules).astype(int)
        detector = lh.DetectorArray(
            detector.positions_m[keep], detector.orientations[keep],
            np.asarray(detector.effective_area_m2)[keep],
            detector.angular_acceptance, detector.spectral_efficiency,
            identifiers={name: np.asarray(value)[keep]
                         for name, value in detector.identifiers.items()},
            provenance=detector.provenance)

    config = lh.KernelConfig(
        omega_per_ns=np.linspace(0.0, 0.15, arguments.frequencies),
        wavelength_nodes=arguments.wavelength_nodes,
        scattering_degree=arguments.scattering_degree,
        source_degree=arguments.source_degree,
        azimuthal_degree=arguments.azimuthal_degree,
        acceptance_degree=arguments.acceptance_degree,
        radial_nodes=arguments.radial_nodes,
        angular_backend=arguments.angular_backend,
        cache_directory=arguments.cache_directory)
    kernel = lh.TransportKernel(model.medium, detector, config)
    measured = kernel.acceptance()
    results["acceptance_used"] = {
        "degree": measured["degree"],
        "residual_above_degree": measured["residual_above_degree"]}

    # ---- 2. isotropic acceptance reproduces the existing m = 0 answer
    band = model.medium.band(float(np.median(model.medium.wavelength_nm)))
    settings = SolverSettings(arguments.scattering_degree,
                              arguments.source_degree, 8.0, 0.04, 10, True)
    omega = np.array([0.0])
    radii = np.geomspace(3.0, 300.0, 40)
    grid = CacheGrid(radii, omega)
    legacy = ResponseCache.build(band, settings, grid, radial_phase="flight",
                                 angular_backend=arguments.angular_backend)
    isotropic = DirectionalCache.build(band, settings, grid, 0,
                                       radial_phase="flight",
                                       angular_backend=arguments.angular_backend)
    ell = np.arange(arguments.source_degree + 1)
    expected = legacy.moments / np.sqrt(2 * ell + 1)[None, None, :, None]
    results["isotropic_reduction_relative_error"] = float(
        np.max(np.abs(isotropic.coefficients[:, :, :, 0, :] - expected))
        / np.max(np.abs(expected)))

    # ---- 3 and 4. a track through the array
    centre = detector.positions_m.mean(axis=0)
    axis = np.array([0.3, 0.4, np.sqrt(1 - 0.25)])
    track = lh.CherenkovTrack(centre - axis * arguments.track_length_m / 2,
                              axis, arguments.track_length_m, beta=0.9999)
    began = perf_counter()
    response = kernel.transport(track, method="directional")
    results["transport"] = {
        "seconds": perf_counter() - began,
        "active_modules": int(response.active.sum()),
        "total_charge_pe": float(response.charge_pe.sum()),
        "detector_angular_model": response.metadata["detector_angular_model"],
        "metadata": {key: value for key, value in response.metadata.items()
                     if isinstance(value, (int, float, str, bool))},
    }

    rotation_angle = 0.7
    unit = np.array([1.0, 2.0, -0.5])
    unit /= np.linalg.norm(unit)
    cross = np.array([[0, -unit[2], unit[1]], [unit[2], 0, -unit[0]],
                      [-unit[1], unit[0], 0]])
    rotation = (np.eye(3) + np.sin(rotation_angle) * cross
                + (1 - np.cos(rotation_angle)) * cross @ cross)
    turned_detector = lh.DetectorArray(
        detector.positions_m @ rotation.T, detector.orientations @ rotation.T,
        detector.effective_area_m2, detector.angular_acceptance,
        detector.spectral_efficiency, provenance=detector.provenance)
    turned_kernel = lh.TransportKernel(model.medium, turned_detector, config)
    turned_track = lh.CherenkovTrack(
        rotation @ (centre - axis * arguments.track_length_m / 2),
        rotation @ axis, arguments.track_length_m, beta=0.9999)
    turned = turned_kernel.transport(turned_track, method="directional")
    reference = response.charge_pe
    results["joint_rotation_relative_error"] = float(
        np.max(np.abs(turned.charge_pe - reference))
        / max(np.max(np.abs(reference)), 1e-300))

    flipped_detector = lh.DetectorArray(
        detector.positions_m, -detector.orientations,
        detector.effective_area_m2, detector.angular_acceptance,
        detector.spectral_efficiency, provenance=detector.provenance)
    flipped = lh.TransportKernel(model.medium, flipped_detector, config)
    flipped_response = flipped.transport(track, method="directional")
    results["module_flip_relative_change"] = float(
        np.max(np.abs(flipped_response.charge_pe - reference))
        / max(np.max(np.abs(reference)), 1e-300))
    results["module_flip_active_modules"] = int(flipped_response.active.sum())
    results["peak_rss_mib"] = peak_rss_mib()

    path = output / "bgvd-directional-smoke.json"
    path.write_text(json.dumps(results, indent=2, default=str))
    print(json.dumps({key: value for key, value in results.items()
                      if key != "acceptance"}, indent=2, default=str))
    print(f"wrote {path} (outside git)")


if __name__ == "__main__":
    main()
