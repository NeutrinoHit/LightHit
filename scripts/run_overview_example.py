"""The end-to-end example of the overview chapter, on public parameters only.

Every input is written out explicitly here: no private optical table and no
stored event file is read. The run is deterministic, so repeating it must
reproduce the printed numbers bit for bit; ``--repeats`` checks that.

    python scripts/run_overview_example.py --output .build/overview
"""
import argparse
import importlib.metadata
import json
import os
import platform
from pathlib import Path

import numpy as np

import lighthit as lh

# A synthetic spectral medium: forward-peaked scattering and absorption of the
# order of deep lake water, but not equal to any measured table. Not a
# calibration (see PROVENANCE.md).
WAVELENGTH_NM = [400.0, 450.0, 500.0, 550.0]
ABSORPTION_PER_M = [0.030, 0.021, 0.038, 0.090]
SCATTERING_PER_M = [0.030, 0.022, 0.017, 0.013]
PHASE_INDEX = [1.3435, 1.3390, 1.3360, 1.3340]
GROUP_INDEX = [1.3860, 1.3740, 1.3670, 1.3630]
ASYMMETRY = 0.9

# Three modules on one vertical string looking down, one looking up at the
# same depth as the track centre, and one far module that the radial range of
# the cache does not cover.
OM_POSITIONS_M = [[25.0, 0.0, -20.0], [25.0, 0.0, 0.0],
                  [25.0, 0.0, 20.0], [60.0, 0.0, 0.0], [0.0, 0.0, 400.0]]
OM_ORIENTATIONS = [[0.0, 0.0, -1.0], [0.0, 0.0, -1.0],
                   [0.0, 0.0, -1.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]
OM_RADIUS_M = 0.2159          # a nominal 17-inch photocathode sphere
QUANTUM_EFFICIENCY = 0.25     # flat in wavelength, on purpose


def angular_acceptance(cosine):
    """Head-on response 1 at x=+1, 0.04 at x=-1; a quadratic, not a measurement.

    It is positive on the whole interval, so no clipping is needed and the
    acceptance is exactly band-limited at Legendre degree 2.
    """
    x = np.clip(np.asarray(cosine, float), -1.0, 1.0)
    return 0.40 + 0.48 * x + 0.12 * x * x


def spectral_efficiency(wavelength_nm):
    return np.full_like(np.asarray(wavelength_nm, float), QUANTUM_EFFICIENCY)


def build_kernel(cache_directory):
    medium = lh.SpectralMedium(
        WAVELENGTH_NM, ABSORPTION_PER_M, SCATTERING_PER_M,
        PHASE_INDEX, GROUP_INDEX, g=ASYMMETRY,
        provenance="synthetic spectral medium, overview example, not a calibration")
    detector = lh.DetectorArray(
        OM_POSITIONS_M, OM_ORIENTATIONS,
        float(np.pi * OM_RADIUS_M ** 2),
        angular_acceptance, spectral_efficiency,
        identifiers={"om": np.arange(1, len(OM_POSITIONS_M) + 1)},
        provenance="synthetic five-module array, overview example")
    config = lh.KernelConfig(
        omega_per_ns=np.linspace(0.0, 0.15, 41),
        relative_time_edges_ns=np.arange(-60.0, 750.0, 20.0),
        wavelength_nodes=9, wavelength_range_nm=(400.0, 550.0),
        scattering_degree=24, source_degree=32, azimuthal_degree=4,
        cell_m=0.12, k_max_per_m=8.0, k_panel_per_m=0.04, k_order=10,
        radial_range_m=(3.0, 300.0), radial_nodes=220,
        angular_backend="numpy", threshold_pe=0.01,
        cache_directory=cache_directory)
    return lh.TransportKernel(medium, detector, config)


def source():
    """A 20 m muon-like track rising past the string, beta = 1."""
    return lh.CherenkovTrack(start_m=[0.0, 0.0, -10.0], direction=[0.0, 0.0, 1.0],
                             length_m=20.0, beta=1.0)


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def summarise(response, kernel, track):
    charge = response.charge_components_pe
    bins = response.bins_pe
    acceptance = kernel.acceptance()
    return {
        "schema": "lighthit-overview-v1",
        "environment": {
            "python": platform.python_version(),
            "lighthit": lh.__version__,
            "numpy": np.__version__,
            "scipy": package_version("scipy"),
            "numba": package_version("numba"),
            "threads": {name: os.environ.get(name) for name in
                        ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                         "MKL_NUM_THREADS", "NUMBA_NUM_THREADS")},
        },
        "configuration": {
            "omega_per_ns": kernel.config.omega_per_ns.tolist(),
            "relative_time_edges_ns": kernel.config.relative_time_edges_ns.tolist(),
            "wavelength_nodes": kernel.config.wavelength_nodes,
            "wavelength_range_nm": list(kernel.config.wavelength_range_nm),
            "scattering_degree": kernel.config.scattering_degree,
            "source_degree": kernel.config.source_degree,
            "azimuthal_degree": kernel.config.azimuthal_degree,
            "cell_m": kernel.config.cell_m,
            "k_max_per_m": kernel.config.k_max_per_m,
            "k_panel_per_m": kernel.config.k_panel_per_m,
            "k_order": kernel.config.k_order,
            "radial_range_m": list(kernel.config.radial_range_m),
            "radial_nodes": kernel.config.radial_nodes,
            "angular_backend": kernel.config.angular_backend,
            "threshold_pe": kernel.config.threshold_pe,
        },
        "source": {
            "kind": "CherenkovTrack",
            "start_m": np.asarray(track.start_m, float).tolist(),
            "direction": np.asarray(track.direction, float).tolist(),
            "length_m": track.length_m,
            "beta": track.beta,
            "time_ns": track.time_ns,
            "reference_phase_index": track.reference_phase_index,
        },
        "acceptance": {
            "degree": acceptance["degree"],
            "alpha": acceptance["alpha"].tolist(),
            "residual_above_degree": acceptance["residual_above_degree"],
        },
        "om": np.asarray(response.detector.identifiers["om"]).tolist(),
        "distance_m": np.linalg.norm(response.detector.positions_m, axis=1).tolist(),
        "charge_pe": response.charge_pe.tolist(),
        "charge_order0_pe": charge[:, 0].tolist(),
        "charge_order1_pe": charge[:, 1].tolist(),
        "charge_order2plus_pe": charge[:, 2].tolist(),
        "window_charge_pe": bins.sum(axis=1).tolist(),
        "negative_bin_mass_pe": np.maximum(-bins, 0.0).sum(axis=1).tolist(),
        "active": response.active.tolist(),
        "method": response.method,
        "detector_angular_model": response.metadata.get("detector_angular_model"),
        "transport_backend": response.metadata.get("backend"),
        "transport_metadata": {
            key: value for key, value in response.metadata.items()
            if not key.endswith("_seconds")
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=".build/overview")
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    kernel = build_kernel(out / "cache")
    track = source()
    kernel.build(source=track)

    first = None
    for _ in range(max(1, args.repeats)):
        summary = summarise(kernel.transport(track, method="track"), kernel, track)
        if first is None:
            first = summary
        elif summary != first:
            raise SystemExit("run is not reproducible: repeated transport differs")

    first["repeats"] = max(1, args.repeats)
    first["reproducible"] = True
    (out / "overview.json").write_text(json.dumps(first, indent=2), encoding="utf-8")

    head = f"{'OM':>3} {'r, m':>7} {'q, p.e.':>10} {'order 0':>10} {'order 1':>10} {'order >=2':>10}"
    print(head)
    print("-" * len(head))
    for i, om in enumerate(first["om"]):
        print(f"{om:>3} {first['distance_m'][i]:>7.2f} {first['charge_pe'][i]:>10.4f} "
              f"{first['charge_order0_pe'][i]:>10.4f} {first['charge_order1_pe'][i]:>10.4f} "
              f"{first['charge_order2plus_pe'][i]:>10.4f}")
    print("module model:", first["detector_angular_model"])
    print("written:", out / "overview.json")


if __name__ == "__main__":
    main()
