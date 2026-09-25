"""Numerical example for the optical-module chapter.

Public synthetic parameters only. No transport solve, no cache, no Numba, no
measured detector table. Everything printed is recomputed here, so the stored
JSON is an output of this script and never a hand-written table.

    python scripts/run_optical_module_example.py --output .build/optical-module-example
"""
import argparse
import importlib.metadata
import json
import os
import platform
from pathlib import Path

import numpy as np
from scipy.special import eval_legendre

import lighthit as lh
from lighthit.cache import acceptance_coefficients, isotropic_acceptance
from lighthit.directional import acceptance_bandwidth

# One point source at the origin and three identical modules 30 m away,
# differing only in where they look.
SOURCE_M = [0.0, 0.0, 0.0]
OM_POSITIONS_M = [[0.0, 0.0, -30.0], [30.0, 0.0, 0.0], [0.0, 0.0, 30.0]]
OM_ORIENTATIONS = [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
OM_LABELS = ["head-on", "side-on", "back-on"]

REFERENCE_AREA_M2 = 0.05
SPECTRAL_EFFICIENCY = 0.2
FLUENCE_PER_M2 = 1000.0
WAVELENGTH_NM = 450.0

QUADRATURE_ORDER = 2048
PROBE_DEGREE = 64
BANDWIDTH_TOLERANCE = 1e-10
RECONSTRUCTION_POINTS = 401


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def quadratic_acceptance(cosine):
    """A(+1) = 1, A(0) = 0.4, A(-1) = 0.04: positive on the whole interval."""
    c = np.clip(np.asarray(cosine, float), -1.0, 1.0)
    return 0.40 + 0.48 * c + 0.12 * c * c


def clipped_acceptance(cosine):
    """Clipped inside the interval, so its coefficients do not terminate."""
    c = np.clip(np.asarray(cosine, float), -1.0, 1.0)
    return np.maximum(0.0, 0.2 + 0.9 * c)


def flat_efficiency(wavelength_nm):
    return np.full_like(np.asarray(wavelength_nm, float), SPECTRAL_EFFICIENCY)


def negative_acceptance(cosine):
    return np.asarray(cosine, float) - 0.5          # negative below c = 0.5


def negative_efficiency(wavelength_nm):
    return np.full_like(np.asarray(wavelength_nm, float), -0.1)


def detector(area_m2=REFERENCE_AREA_M2, efficiency=flat_efficiency,
             acceptance=quadratic_acceptance):
    return lh.DetectorArray(
        OM_POSITIONS_M, OM_ORIENTATIONS, area_m2, acceptance, efficiency,
        identifiers={"label": np.array(OM_LABELS)},
        provenance="synthetic three-module array, optical-module example")


def rejects(build):
    """True when the contract refuses the argument, with the message kept."""
    try:
        build()
    except ValueError as error:
        return {"rejected": True, "message": str(error)}
    return {"rejected": False, "message": None}


def geometry(array):
    cosine = array.head_on_cosine(SOURCE_M)
    norms = np.linalg.norm(array.orientations, axis=1)
    return {
        "source_m": SOURCE_M,
        "labels": OM_LABELS,
        "positions_m": array.positions_m.tolist(),
        "orientations": array.orientations.tolist(),
        "orientation_norms": norms.tolist(),
        "orientation_norm_error": float(np.max(np.abs(norms - 1.0))),
        "distance_m": np.linalg.norm(
            np.asarray(SOURCE_M) - array.positions_m, axis=1).tolist(),
        "head_on_cosine": cosine.tolist(),
        "head_on_cosine_error": float(np.max(np.abs(cosine - [1.0, 0.0, -1.0]))),
    }


def response(array):
    cosine = array.head_on_cosine(SOURCE_M)
    angular = array.angular_weight(SOURCE_M)
    spectral = float(array.spectral_weight(np.array([WAVELENGTH_NM]))[0])
    area = np.asarray(array.effective_area_m2, float)
    expected = area * spectral * angular * FLUENCE_PER_M2
    return {
        "fluence_per_m2": FLUENCE_PER_M2,
        "wavelength_nm": WAVELENGTH_NM,
        "reference_area_m2": area.tolist(),
        "spectral_efficiency": spectral,
        "cosine": cosine.tolist(),
        "angular_acceptance": angular.tolist(),
        "angular_expected": [1.0, 0.4, 0.04],
        "angular_error": float(np.max(np.abs(angular - [1.0, 0.4, 0.04]))),
        "expected_photoelectrons": expected.tolist(),
    }


def legendre_block(array):
    alpha = acceptance_coefficients(array.angular_acceptance, PROBE_DEGREE,
                                    quadrature_order=QUADRATURE_ORDER)
    exact = 2 * np.pi * np.array([0.88, 0.32, 0.032])
    bandwidth, measured, residual = acceptance_bandwidth(
        array.angular_acceptance, max_degree=PROBE_DEGREE,
        tolerance=BANDWIDTH_TOLERANCE, quadrature_order=QUADRATURE_ORDER)

    # A(c) = sum_l (2l+1)/(4 pi) alpha_l P_l(c), truncated at the bandwidth.
    grid = np.linspace(-1.0, 1.0, RECONSTRUCTION_POINTS)
    degrees = np.arange(bandwidth + 1)
    rebuilt = ((2 * degrees + 1) / (4 * np.pi) * alpha[:bandwidth + 1]
               ) @ eval_legendre(degrees[:, None], grid[None, :])
    reconstruction_error = float(np.max(np.abs(
        rebuilt - quadratic_acceptance(grid))))

    isotropic = acceptance_coefficients(isotropic_acceptance, 4,
                                        quadrature_order=QUADRATURE_ORDER)
    # The residuals below are quadrature arithmetic, not properties of the
    # acceptance, so they are reported relative to the peak coefficient and
    # bounded rather than quoted as values.
    peak = float(np.max(np.abs(alpha)))
    isotropic_peak = float(np.max(np.abs(isotropic)))
    return {
        "quadrature_order": QUADRATURE_ORDER,
        "probe_degree": PROBE_DEGREE,
        "tolerance": BANDWIDTH_TOLERANCE,
        "alpha": alpha[:5].tolist(),
        "alpha_exact_to_degree_2": exact.tolist(),
        "peak_alpha": peak,
        "alpha_error_to_degree_2": float(np.max(np.abs(alpha[:3] - exact))),
        "relative_alpha_error_to_degree_2": float(
            np.max(np.abs(alpha[:3] - exact)) / peak),
        "max_abs_alpha_above_degree_2": float(np.max(np.abs(alpha[3:]))),
        "max_relative_alpha_above_degree_2": float(np.max(np.abs(alpha[3:])) / peak),
        "bandwidth": bandwidth,
        "residual_above_bandwidth": residual,
        "reconstruction_points": RECONSTRUCTION_POINTS,
        "reconstruction_max_error": reconstruction_error,
        "isotropic_alpha": isotropic.tolist(),
        "isotropic_alpha0_error": float(abs(isotropic[0] - 4 * np.pi)),
        "isotropic_max_abs_alpha_above_0": float(np.max(np.abs(isotropic[1:]))),
        "isotropic_max_relative_above_0": float(
            np.max(np.abs(isotropic[1:])) / isotropic_peak),
        "measured_equals_alpha": float(np.max(np.abs(measured - alpha))),
    }


def clipped_block():
    alpha = acceptance_coefficients(clipped_acceptance, PROBE_DEGREE,
                                    quadrature_order=QUADRATURE_ORDER)
    peak = float(np.max(np.abs(alpha)))
    bandwidth, _, residual = acceptance_bandwidth(
        clipped_acceptance, max_degree=PROBE_DEGREE,
        tolerance=BANDWIDTH_TOLERANCE, quadrature_order=QUADRATURE_ORDER)
    relative = np.abs(alpha) / peak
    # A local slope between two degrees: the coefficients oscillate, so this
    # indicates the envelope rather than proving a power law.
    slope = float(-np.log(relative[64] / relative[32]) / np.log(64 / 32))
    return {
        "definition": "max(0, 0.2 + 0.9 c)",
        "kink_at_cosine": -0.2 / 0.9,
        "probe_degree": PROBE_DEGREE,
        "alpha": alpha[:5].tolist(),
        "relative_alpha_at_degree": {str(d): float(relative[d])
                                     for d in (3, 8, 16, 32, 64)},
        "max_relative_above_degree_2": float(np.max(relative[3:])),
        "local_decay_exponent_32_to_64": slope,
        "bandwidth_in_probed_range": bandwidth,
        "reached_probe_degree": bandwidth >= PROBE_DEGREE,
        "residual_above_bandwidth": residual,
        "note": "the probe shows only that the response is not band-limited "
                "inside the probed range; it measures no infinite bandwidth",
    }


def scaling(array):
    base = np.asarray(response(array)["expected_photoelectrons"], float)
    doubled_area = np.asarray(response(
        detector(area_m2=2 * REFERENCE_AREA_M2))["expected_photoelectrons"], float)
    half_efficiency = np.asarray(response(detector(
        efficiency=lambda w: np.full_like(np.asarray(w, float),
                                          SPECTRAL_EFFICIENCY / 2)
    ))["expected_photoelectrons"], float)
    return {
        "base_pe": base.tolist(),
        "double_area_pe": doubled_area.tolist(),
        "half_efficiency_pe": half_efficiency.tolist(),
        "area_linearity_error": float(np.max(np.abs(doubled_area - 2 * base))),
        "efficiency_linearity_error": float(np.max(np.abs(
            half_efficiency - base / 2))),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=".build/optical-module-example")
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    array = detector()
    summary = {
        "schema": "lighthit-optical-module-v1",
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
            "quadrature_order": QUADRATURE_ORDER,
            "probe_degree": PROBE_DEGREE,
            "bandwidth_tolerance": BANDWIDTH_TOLERANCE,
            "reconstruction_points": RECONSTRUCTION_POINTS,
            "angular_acceptance": "0.40 + 0.48 c + 0.12 c^2",
            "spectral_efficiency": SPECTRAL_EFFICIENCY,
            "reference_area_m2": REFERENCE_AREA_M2,
            "fluence_per_m2": FLUENCE_PER_M2,
        },
        "geometry": geometry(array),
        "response": response(array),
        "legendre": legendre_block(array),
        "scaling": scaling(array),
        "clipped_acceptance": clipped_block(),
        "contract": {
            "negative_angular_response": rejects(
                lambda: detector(acceptance=negative_acceptance)),
            "negative_spectral_response": rejects(
                lambda: detector(efficiency=negative_efficiency)),
            "negative_coefficients": rejects(
                lambda: acceptance_coefficients(negative_acceptance, 2)),
        },
    }

    g, r, l, s, c, k = (summary["geometry"], summary["response"],
                        summary["legendre"], summary["scaling"],
                        summary["contract"], summary["clipped_acceptance"])
    summary["checks"] = {
        "orientations_are_unit": g["orientation_norm_error"] < 1e-15,
        "head_side_back_cosines": g["head_on_cosine_error"] < 1e-15,
        "angular_values": r["angular_error"] < 1e-15,
        "quadratic_coefficients_exact":
            l["relative_alpha_error_to_degree_2"] < 1e-10,
        "coefficients_terminate_at_two":
            l["max_relative_alpha_above_degree_2"] < 1e-10,
        "measured_bandwidth_is_two": l["bandwidth"] == 2,
        "reconstruction_matches": l["reconstruction_max_error"] < 1e-10,
        "isotropic_limit_is_four_pi": l["isotropic_alpha0_error"] < 1e-10,
        "isotropic_has_no_higher_terms": l["isotropic_max_relative_above_0"] < 1e-10,
        "linear_in_area": s["area_linearity_error"] < 1e-12,
        "linear_in_efficiency": s["efficiency_linearity_error"] < 1e-12,
        "negative_angular_rejected": c["negative_angular_response"]["rejected"],
        "negative_spectral_rejected": c["negative_spectral_response"]["rejected"],
        "negative_coefficients_rejected": c["negative_coefficients"]["rejected"],
        "clipped_has_substantial_tail": k["max_relative_above_degree_2"] > 1e-4,
        "clipped_not_band_limited_in_range": k["reached_probe_degree"],
    }
    failed = [name for name, ok in summary["checks"].items() if not ok]
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"{'module':>9} {'cosine':>8} {'A(c)':>8} {'N_d, p.e.':>10}")
    print("-" * 39)
    for i, label in enumerate(OM_LABELS):
        print(f"{label:>9} {r['cosine'][i]:>8.3f} {r['angular_acceptance'][i]:>8.3f} "
              f"{r['expected_photoelectrons'][i]:>10.3f}")
    print(f"alpha = {l['alpha'][:3]}, bandwidth {l['bandwidth']}, "
          f"residual {l['residual_above_bandwidth']:.2e}")
    print(f"reconstruction error {l['reconstruction_max_error']:.2e}, "
          f"isotropic alpha_0 - 4pi = {l['isotropic_alpha0_error']:.2e}")
    print(f"clipped response: bandwidth in probed range {k['bandwidth_in_probed_range']}"
          f" of {PROBE_DEGREE}, largest relative coefficient above degree 2 "
          f"{k['max_relative_above_degree_2']:.2e}")
    print("checks failed:", failed or "none")
    print("written:", out / "summary.json")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
