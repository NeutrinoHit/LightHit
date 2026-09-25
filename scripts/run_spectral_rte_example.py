"""Numerical example for the spectral-transfer-equation chapter.

Public synthetic parameters only: no measured optical table is read. The run
touches no transport cache, needs no Numba, and finishes in well under a
second. Everything it prints is recomputed here, so the stored JSON is an
output of this script and never a hand-written table.

    python scripts/run_spectral_rte_example.py --output .build/spectral-rte-example
"""
import argparse
import importlib.metadata
import json
import os
import platform
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import eval_legendre

import lighthit as lh
from lighthit.medium import C_VACUUM_M_PER_NS
from lighthit.single import hg_phase

# The same public synthetic table the overview chapter uses, so that the two
# chapters describe one medium. Near deep lake water in magnitude, equal to no
# measured value: this is not a calibration.
WAVELENGTH_NM = [400.0, 450.0, 500.0, 550.0]
ABSORPTION_PER_M = [0.030, 0.021, 0.038, 0.090]
SCATTERING_PER_M = [0.030, 0.022, 0.017, 0.013]
PHASE_INDEX = [1.3435, 1.3390, 1.3360, 1.3340]
GROUP_INDEX = [1.3860, 1.3740, 1.3670, 1.3630]
ASYMMETRY = 0.9

PROBE_NM = 462.5          # strictly between two table nodes
FLIGHT_DISTANCE_M = 100.0
LEGENDRE_DEGREES = 9
QUADRATURE_ORDER = 512


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def spectral_medium():
    return lh.SpectralMedium(
        WAVELENGTH_NM, ABSORPTION_PER_M, SCATTERING_PER_M,
        PHASE_INDEX, GROUP_INDEX, g=ASYMMETRY,
        provenance="synthetic spectral medium, transfer-equation example, "
                   "not a calibration")


def hand_interpolation(wavelength_nm):
    """Linear interpolation written out independently of numpy.interp."""
    nodes = np.asarray(WAVELENGTH_NM, float)
    upper = int(np.searchsorted(nodes, wavelength_nm))
    lower = upper - 1
    weight = (wavelength_nm - nodes[lower]) / (nodes[upper] - nodes[lower])
    def between(values):
        values = np.asarray(values, float)
        return float(values[lower] + weight * (values[upper] - values[lower]))
    return {
        "bracket_nm": [float(nodes[lower]), float(nodes[upper])],
        "weight": float(weight),
        "absorption_per_m": between(ABSORPTION_PER_M),
        "scattering_per_m": between(SCATTERING_PER_M),
        "phase_index": between(PHASE_INDEX),
        "group_index": between(GROUP_INDEX),
    }


def probe(medium):
    sampled = medium.sample(PROBE_NM)
    band = medium.band(PROBE_NM)
    hand = hand_interpolation(PROBE_NM)
    absorption = float(sampled["absorption_per_m"])
    scattering = float(sampled["scattering_per_m"])
    extinction = absorption + scattering
    interpolation_error = max(
        abs(float(sampled[key]) - hand[key])
        for key in ("absorption_per_m", "scattering_per_m",
                    "phase_index", "group_index"))
    return {
        "wavelength_nm": PROBE_NM,
        "bracket_nm": hand["bracket_nm"],
        "interpolation_weight": hand["weight"],
        "absorption_per_m": absorption,
        "scattering_per_m": scattering,
        "extinction_per_m": extinction,
        "phase_index": float(sampled["phase_index"]),
        "group_index": float(sampled["group_index"]),
        "group_speed_m_per_ns": band.speed_m_per_ns,
        "absorption_length_m": 1.0 / absorption,
        "scattering_length_m": 1.0 / scattering,
        "extinction_length_m": 1.0 / extinction,
        "single_scattering_albedo": scattering / extinction,
        "band_extinction_per_m": band.extinction_per_m,
        "band_extinction_error": abs(band.extinction_per_m - extinction),
        "linear_interpolation_max_error": interpolation_error,
    }, band


def which_index_is_stored(band):
    """The monochromatic medium carries the group index and not the phase one."""
    fields = sorted(band.__dataclass_fields__)
    return {
        "dataclass_fields": fields,
        "has_group_index": "group_index" in fields,
        "has_phase_index": "phase_index" in fields,
        "group_index": band.group_index,
        "speed_m_per_ns": band.speed_m_per_ns,
        "speed_error": abs(band.speed_m_per_ns
                           - C_VACUUM_M_PER_NS / band.group_index),
    }


def flight_time(sampled):
    group = sampled["group_index"]
    phase = sampled["phase_index"]
    with_group = FLIGHT_DISTANCE_M * group / C_VACUUM_M_PER_NS
    with_phase = FLIGHT_DISTANCE_M * phase / C_VACUUM_M_PER_NS
    return {
        "distance_m": FLIGHT_DISTANCE_M,
        "group_index": group,
        "phase_index": phase,
        "time_with_group_index_ns": with_group,
        "time_with_phase_index_ns": with_phase,
        "difference_ns": with_group - with_phase,
        "relative_difference": (with_group - with_phase) / with_group,
    }


def henyey_greenstein(g):
    cosine, weight = leggauss(QUADRATURE_ORDER)
    phase = hg_phase(cosine, g)
    normalisation = 2 * np.pi * float(np.dot(weight, phase))
    mean_cosine = 2 * np.pi * float(np.dot(weight, cosine * phase))
    legendre = []
    for degree in range(LEGENDRE_DEGREES):
        measured = 2 * np.pi * float(np.dot(
            weight, eval_legendre(degree, cosine) * phase))
        expected = g ** degree
        legendre.append({"degree": degree, "measured": measured,
                         "expected": expected,
                         "absolute_error": abs(measured - expected)})
    return {
        "g": g,
        "quadrature": {"rule": "Gauss-Legendre in cos(theta)",
                       "order": QUADRATURE_ORDER},
        "normalisation": normalisation,
        "normalisation_error": abs(normalisation - 1.0),
        "mean_cosine": mean_cosine,
        "mean_cosine_error": abs(mean_cosine - g),
        "legendre": legendre,
        "legendre_max_error": max(item["absolute_error"] for item in legendre),
    }


def range_guard(medium):
    low, high = medium.wavelength_range_nm
    result = {"range_nm": [low, high]}
    for name, value in (("below", low - 1.0), ("above", high + 1.0)):
        try:
            medium.sample(value)
        except ValueError as error:
            result[name] = {"wavelength_nm": value, "raised": True,
                            "message": str(error)}
        else:
            result[name] = {"wavelength_nm": value, "raised": False}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=".build/spectral-rte-example")
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    medium = spectral_medium()
    sampled, band = probe(medium)
    summary = {
        "schema": "lighthit-spectral-rte-v1",
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
        "medium_table": {
            "wavelength_nm": WAVELENGTH_NM,
            "absorption_per_m": ABSORPTION_PER_M,
            "scattering_per_m": SCATTERING_PER_M,
            "phase_index": PHASE_INDEX,
            "group_index": GROUP_INDEX,
            "g": ASYMMETRY,
            "provenance": medium.provenance,
        },
        "probe": sampled,
        "monochromatic_band": which_index_is_stored(band),
        "flight_time": flight_time(sampled),
        "henyey_greenstein": henyey_greenstein(ASYMMETRY),
        "wavelength_range_guard": range_guard(medium),
    }
    summary["checks"] = {
        "interpolation_is_linear": summary["probe"]["linear_interpolation_max_error"] < 1e-12,
        "band_matches_sample": summary["probe"]["band_extinction_error"] < 1e-12,
        "band_has_group_index": summary["monochromatic_band"]["has_group_index"],
        "band_omits_phase_index": not summary["monochromatic_band"]["has_phase_index"],
        "group_speed_matches_index": summary["monochromatic_band"]["speed_error"] < 1e-12,
        "hg_normalised": summary["henyey_greenstein"]["normalisation_error"] < 1e-10,
        "hg_mean_cosine": summary["henyey_greenstein"]["mean_cosine_error"] < 1e-10,
        "hg_legendre_eigenvalues": summary["henyey_greenstein"]["legendre_max_error"] < 1e-9,
        "range_guard_rejects_below": summary["wavelength_range_guard"]["below"]["raised"],
        "range_guard_rejects_above": summary["wavelength_range_guard"]["above"]["raised"],
    }
    failed = [name for name, ok in summary["checks"].items() if not ok]
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    p = summary["probe"]
    print(f"probe at {p['wavelength_nm']} nm, between {p['bracket_nm']} nm")
    print(f"  mu_a = {p['absorption_per_m']:.6f} 1/m   "
          f"mu_s = {p['scattering_per_m']:.6f} 1/m   "
          f"mu_t = {p['extinction_per_m']:.6f} 1/m")
    print(f"  n_ph = {p['phase_index']:.6f}   n_g = {p['group_index']:.6f}   "
          f"v_g = {p['group_speed_m_per_ns']:.6f} m/ns")
    print(f"  lengths: absorption {p['absorption_length_m']:.3f} m, "
          f"scattering {p['scattering_length_m']:.3f} m, "
          f"extinction {p['extinction_length_m']:.3f} m")
    f = summary["flight_time"]
    print(f"flight over {f['distance_m']:.0f} m: "
          f"{f['time_with_group_index_ns']:.3f} ns with n_g, "
          f"{f['time_with_phase_index_ns']:.3f} ns with n_ph, "
          f"difference {f['difference_ns']:.3f} ns "
          f"({100*f['relative_difference']:.2f}%)")
    h = summary["henyey_greenstein"]
    print(f"HG g={h['g']}: normalisation error {h['normalisation_error']:.2e}, "
          f"mean-cosine error {h['mean_cosine_error']:.2e}, "
          f"max |a_l - g^l| {h['legendre_max_error']:.2e}")
    print("checks failed:", failed or "none")
    print("written:", out / "summary.json")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
