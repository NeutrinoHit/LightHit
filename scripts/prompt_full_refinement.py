"""Order 1: prompt (direct) versus the full directional path under refinement.

The full path obtains order 1 from a finite-rank angular representation
(scattering degree L, source degree L_s, azimuthal degree, radial table,
frequency grid).  This script fixes the source, medium, detector, wavelength
quadrature and time bins, refines the full path's angular degrees, and
compares its order-1 charge and bins with a refined prompt reference (the
refined segment rule for the track, uncompressed ring cones for the shower).
Synthetic medium and geometry only; one cache per degree pair is built in a
temporary directory and shared by both sources.

    python scripts/prompt_full_refinement.py --output OUT.json
"""
import argparse
import json
import tempfile
import time
from dataclasses import replace

import numpy as np

import lighthit as lh


def cubic(x):
    x = np.asarray(x, float)
    return 0.3082 + 0.54192 * x + 0.19831 * x ** 2 - 0.04912 * x ** 3


def setup():
    medium = lh.SpectralMedium(
        [350., 450., 550., 610.], [0.20, 0.07, 0.055, 0.16], [0.035, 0.028, 0.015, 0.016],
        [1.350, 1.341, 1.336, 1.334], [1.401, 1.378, 1.363, 1.351], g=0.9,
        provenance="synthetic near-Baikal spectral medium (not a calibration)")
    # three strings 25-45 m from the source axis, OMs every 10 m, looking down
    strings = [(25., 0.), (-15., 30.), (10., -45.)]
    positions = np.array([[x, y, z] for x, y in strings for z in np.arange(-60., 101., 10.)])
    orientations = np.tile([0., 0., -1.], (len(positions), 1))
    detector = lh.DetectorArray(positions, orientations, np.pi * 0.216 ** 2, cubic,
                                lambda w: 0.25 * np.ones_like(np.asarray(w, float)))
    return medium, detector


def sources():
    pose = lh.SourcePose([0., 0., 0.], [0.25, 0.1, 0.963], 0.)
    return {
        "track": lh.CherenkovTrack.centered([0., 0., 20.], [0.25, 0.1, 0.963], 60.),
        "shower": lh.SyntheticShower.gaussian(pose, charged_track_length_m=300.,
                                              elements=2000, seed=4),
    }


def config(cache, degree, source_degree, azimuthal):
    return lh.KernelConfig(
        omega_per_ns=np.linspace(0., 1.2, 321),
        relative_time_edges_ns=np.arange(-60., 740. + 5., 5.),
        wavelength_nodes=3, scattering_degree=degree, source_degree=source_degree,
        azimuthal_degree=azimuthal, radial_range_m=(3., 200.), radial_nodes=120,
        threshold_pe=0.01, cache_directory=cache, spectral_folded_cache=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--degrees", default="12:16,24:32,36:48")
    args = parser.parse_args()
    medium, detector = setup()
    result = {"medium": medium.provenance, "modules": len(detector), "cases": {}}
    references = {}
    for name, source in sources().items():
        kernel = lh.TransportKernel(medium, detector, config(None, 24, 32, 4))
        started = time.perf_counter()
        prompt = kernel.transport_prompt(source)
        prompt_seconds = time.perf_counter() - started
        fine = kernel.transport_prompt(source, replace(
            lh.PromptConfig(), a_gauss=6, front_width_ns=0.6, phi_uniform=48,
            path_step=0.05, path_step_tail=0.1,
            element_method="segment" if name == "track" else "ring",
            ring_a_step_m=0.02))
        q = fine.charge_orders_pe[:, 1]
        references[name] = fine
        result["cases"][name] = {
            "prompt_seconds": prompt_seconds,
            "reference": "refined segment rule" if name == "track" else
                         "ring (uncompressed point cones, 2 cm)",
            "prompt_total_order1_pe": float(prompt.charge_orders_pe[:, 1].sum()),
            "reference_total_order1_pe": float(q.sum()),
            "prompt_vs_reference_max": float(np.max(
                np.abs(prompt.charge_orders_pe[:, 1] - q) / np.maximum(q, 0.01))),
            "full": []}
        print(name, {k: v for k, v in result["cases"][name].items() if k != "full"},
              flush=True)
    # the full path's caches depend on the medium, detector and degrees only:
    # build once per degree pair and transport both sources
    for pair in args.degrees.split(","):
        degree, source_degree = (int(v) for v in pair.split(":"))
        with tempfile.TemporaryDirectory() as cache:
            kernel_full = lh.TransportKernel(medium, detector,
                                             config(cache, degree, source_degree, 4))
            started = time.perf_counter()
            kernel_full.build(method="axial", progress=False)
            build = time.perf_counter() - started
            for name, source in sources().items():
                fine = references[name]
                q = fine.charge_orders_pe[:, 1]
                scale = np.maximum(q, 0.01)
                started = time.perf_counter()
                full = kernel_full.transport(source)
                seconds = time.perf_counter() - started
                f1 = full.charge_components_pe[:, 1]
                computed = full.active | fine.active
                difference = (f1 - q) / scale
                bins_diff = np.abs(full.components_pe[:, :, 1]
                                   - fine.bins_orders_pe[:, :, 1]).sum(axis=1) / scale
                both = full.active & fine.active
                entry = {
                    "scattering_degree": degree, "source_degree": source_degree,
                    "build_seconds": build, "transport_seconds": seconds,
                    "total_order1_full_pe": float(f1.sum()),
                    "total_order1_reference_pe": float(q.sum()),
                    "max_abs_charge_diff_over_max(q,0.01)":
                        float(np.abs(difference[computed]).max()),
                    "median_abs_charge_diff_over_max(q,0.01)":
                        float(np.median(np.abs(difference[computed]))),
                    "max_bins_L1_over_max(q,0.01)_both_active":
                        float(bins_diff[both].max()) if both.any() else None,
                    "median_bins_L1_over_max(q,0.01)_both_active":
                        float(np.median(bins_diff[both])) if both.any() else None,
                    "negative_order1_bins_pe_full":
                        float(np.minimum(full.components_pe[:, :, 1], 0).sum()),
                    "negative_order1_charge_modules_full": int((f1 < 0).sum())}
                result["cases"][name]["full"].append(entry)
                print(name, entry, flush=True)
        with open(args.output, "w") as handle:
            json.dump(result, handle, indent=1)


if __name__ == "__main__":
    main()
