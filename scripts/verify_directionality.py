"""Independent controls for the arrival-direction claims.

Run: python scripts/verify_directionality.py --output .build/claude-review/directionality

Four things are separated on purpose, because they are different statements:

1. the stationary P_N eigenmode sum, refined in N and compared with the
   original real-k solver route, which uses the exact free tail instead;
2. a surface-crossing Monte Carlo written for this check
   (:mod:`lighthit.experimental.crossing_mc`), which shares no estimator with
   the track-length sampler in :mod:`lighthit.experimental.shell_mc`: crossings
   of an exact sphere carry no shell-width bias;
3. the identity ``d<f>/da = -Cov(f, S)``, evaluated on one set of paths with
   several absorption coefficients, which is the precise form of "absorption
   selects paths" and says nothing on its own about the sign;
4. the factorisation ``I_a = exp(-a v t) I_0`` at fixed place and time, checked
   on the exact coordinate first order, next to the time-integrated mean
   cosine, which does move with absorption.

No claim is made here that directionality must grow with distance; the
measured numbers are printed as they come out.
"""
from pathlib import Path
import argparse
import json
import platform
import sys
from time import perf_counter

import numpy as np

from lighthit import Medium, SolverSettings, synthetic_medium
from lighthit.cache import CacheGrid, ResponseCache
from lighthit.single import single_scattering_rate
from lighthit.experimental.crossing_mc import crossing_estimate, mean_cosine
from lighthit.experimental.stationary_modes import (StationaryModes,
                                                    collision_components,
                                                    single_scalar_current)

RADII = np.array([11.0, 20.0, 38.0, 69.0, 128.0, 225.0, 400.0, 800.0])


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def stationary_table(medium):
    """The eigenmode charges and mean cosines, refined in N."""
    models = {n: StationaryModes.build(medium, n) for n in (64, 128, 256, 512)}
    charges, currents = collision_components(RADII, models[256])
    reference = charges.sum(axis=1)
    convergence = {}
    for degree, model in models.items():
        q, _ = collision_components(RADII, model)
        convergence[degree] = float(np.max(np.abs(q.sum(axis=1) - reference) / reference))
    leading_q, _ = models[256].scalar_current(RADII, leading_only=True)
    return {"radii_m": RADII, "charge_by_order": charges,
            "current_by_order": currents,
            "mean_cosine_by_order": currents / charges,
            "mean_cosine_total": currents.sum(axis=1) / charges.sum(axis=1),
            "relative_charge_change_vs_N256": convergence,
            "leading_attenuation_per_m": models[256].leading_attenuation_per_m,
            "leading_limit_mean_cosine": models[256].leading_mean_cosine,
            "leading_mode_share_of_charge": leading_q / charges.sum(axis=1),
            "monotonic_in_distance": {
                "one": bool(np.all(np.diff((currents / charges)[:, 1]) > 0)),
                "two_or_more": bool(np.all(np.diff((currents / charges)[:, 2]) > 0)),
                "total": bool(np.all(np.diff(currents.sum(1) / charges.sum(1)) > 0))}}


def real_k_comparison(medium, quick):
    """The original Fourier route against the eigenmode sum, refined in k_max.

    The two discretisations disagree in kind: the solver keeps the exact free
    tail above its scattering degree, the eigenmode sum truncates the whole
    angular space. Agreement is therefore evidence about the physics, not a
    statement that either is exact.
    """
    radii = np.array([11.0, 20.0, 40.0, 80.0, 128.0])
    modes = StationaryModes.build(medium, 256)
    exact_q, exact_f = modes.scalar_current(radii)
    rows = []
    for k_max in ((4.0, 6.0) if quick else (4.0, 6.0, 8.0)):
        settings = SolverSettings(128, 1, k_max, 0.025, 10)
        start = perf_counter()
        cache = ResponseCache.build(medium, settings, CacheGrid(radii, np.array([0.0])))
        ballistic = np.exp(-medium.extinction_per_m * radii) / (4 * np.pi * radii ** 2)
        charge = ballistic + cache.moments[0, :, 0, :].real.sum(axis=1)
        # The degree-1 moment carries the current; the 1/3 is the Legendre
        # normalisation of the expansion the cache stores.
        current = ballistic + cache.moments[0, :, 1, :].real.sum(axis=1) / 3
        rows.append({"k_max_per_m": k_max, "radii_m": radii,
                     "relative_charge_difference": (charge - exact_q) / exact_q,
                     "mean_cosine_difference": current / charge - exact_f / exact_q,
                     "build_seconds": perf_counter() - start})
    return {"radii_m": radii, "eigenmode_charge": exact_q,
            "eigenmode_mean_cosine": exact_f / exact_q, "scan": rows}


def crossing_check(medium, quick):
    """The independent sampler against the eigenmode mean cosines."""
    radii = np.array([20.0, 80.0, 128.0])
    absorptions = np.array([0.035, 0.07, 0.14])
    photons = 20_000 if quick else 80_000
    batches = 8 if quick else 16
    start = perf_counter()
    data = crossing_estimate(scattering_per_m=medium.scattering_per_m, g=medium.g,
                             absorptions=absorptions, radii_m=radii,
                             photons_per_batch=photons, batches=batches,
                             max_path_m=1200.0, seed=20260917)
    seconds = perf_counter() - start
    index = int(np.argmin(np.abs(absorptions - medium.absorption_per_m)))
    charges, currents = collision_components(radii, StationaryModes.build(medium, 256))
    theory = currents / charges
    rows = []
    for order in (1, 2):
        sampled, error = mean_cosine(data, order)
        rows.append({"order": "one" if order == 1 else "two_or_more",
                     "radii_m": radii,
                     "monte_carlo_mean_cosine": sampled[index],
                     "standard_error": error[index],
                     "eigenmode_mean_cosine": theory[:, order],
                     "difference_in_sigma": (sampled[index] - theory[:, order]) / error[index]})
    sampled_charge = data[:, index, :, :, 0].mean(axis=0).sum(axis=1)
    return {"photons": photons * batches, "batches": batches, "seconds": seconds,
            "absorption_used_per_m": float(absorptions[index]),
            "relative_charge_difference": sampled_charge / charges.sum(axis=1) - 1,
            "orders": rows,
            "caveat": "The 1/|mu| crossing weight has a heavy tail at grazing "
                      "crossings; batch spreads are empirical, not guarantees."}


def covariance_identity(medium, quick):
    """d<u>/da against -Cov(u, S) on one set of paths."""
    radii = np.array([20.0, 80.0])
    step = 2e-3
    centre = medium.absorption_per_m
    absorptions = np.array([centre - step, centre, centre + step])
    photons = 20_000 if quick else 60_000
    data = crossing_estimate(scattering_per_m=medium.scattering_per_m, g=medium.g,
                             absorptions=absorptions, radii_m=radii,
                             photons_per_batch=photons, batches=8,
                             max_path_m=1200.0, seed=4242)
    average = data.mean(axis=0)
    rows = []
    for order, label in ((1, "one"), (2, "two_or_more")):
        weight = average[:, :, order, 0]
        cosine = average[:, :, order, 1] / weight
        path = average[:, :, order, 2] / weight
        cosine_path = average[:, :, order, 3] / weight
        covariance = cosine_path[1] - cosine[1] * path[1]
        derivative = (cosine[2] - cosine[0]) / (2 * step)
        rows.append({"order": label, "radii_m": radii,
                     "d_mean_cosine_d_absorption": derivative,
                     "negative_covariance": -covariance,
                     "relative_difference": (derivative + covariance)
                     / np.maximum(np.abs(covariance), 1e-300),
                     "mean_path_m": path[1]})
    return {"absorptions_per_m": absorptions, "photons": photons * 8,
            "finite_difference_step": step, "orders": rows,
            "meaning": "A negative covariance of arrival cosine and path length "
                       "is what makes absorption sharpen the arrival direction; "
                       "its sign is measured here, not assumed."}


def factorisation_check(medium):
    """exp(-a v t) at fixed place and time, on the exact coordinate first order."""
    radius = 40.0
    times = np.linspace(radius / medium.speed_m_per_ns * 1.01, 900.0, 25)
    other = Medium(medium.absorption_per_m + 0.03, medium.scattering_per_m,
                   medium.g, medium.group_index, medium.wavelength_nm,
                   medium.provenance)
    base = single_scattering_rate(times, radius, None, medium)
    shifted = single_scattering_rate(times, radius, None, other)
    predicted = np.exp(-0.03 * medium.speed_m_per_ns * times)
    ratio = np.where(base > 0, shifted / np.maximum(base, 1e-300), np.nan)
    charge_base = single_scalar_current(radius, medium)
    charge_other = single_scalar_current(radius, other)
    return {"radius_m": radius, "times_ns": times,
            "ratio": ratio, "predicted": predicted,
            "max_relative_deviation": float(np.nanmax(np.abs(ratio / predicted - 1))),
            "time_integrated_mean_cosine": {
                "absorption_%.3f" % medium.absorption_per_m:
                    float(charge_base[1] / charge_base[0]),
                "absorption_%.3f" % other.absorption_per_m:
                    float(charge_other[1] / charge_other[0])},
            "reading": "At fixed (r, t) the angular shape is untouched by "
                       "uniform absorption; after integrating over time the "
                       "mean cosine does move, because path weights change."}


def track_length_cross_check(medium, quick):
    """The shipped track-length sampler, when its optional dependency is there."""
    try:
        from lighthit.experimental.shell_mc import (shell_estimate,
                                                    ratio_with_standard_error)
    except ImportError as error:
        return {"status": "skipped", "reason": str(error)}
    radii = np.array([20.0, 80.0])
    photons = 400 if quick else 3000
    start = perf_counter()
    batch = shell_estimate(absorptions=[medium.absorption_per_m], radii_m=radii,
                           widths_m=0.5, photons_per_batch=photons, batches=8,
                           max_path_m=1000.0, seed=20260917)
    mean, error = ratio_with_standard_error(batch)
    charges, currents = collision_components(radii, StationaryModes.build(medium, 256))
    theory = (currents / charges)[:, 2]
    return {"status": "ran", "photons": photons * 8, "seconds": perf_counter() - start,
            "shell_width_m": 0.5, "radii_m": radii,
            "mean_cosine_two_or_more": mean[0, :, 2],
            "standard_error": error[0, :, 2],
            "eigenmode_point_value": theory,
            "difference_in_sigma": (mean[0, :, 2] - theory) / error[0, :, 2],
            "note": "Shell averages are compared with point values here; the "
                    "shell width itself biases the comparison."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=".build/claude-review/directionality")
    parser.add_argument("--quick", action="store_true")
    arguments = parser.parse_args()
    out = Path(arguments.output)
    out.mkdir(parents=True, exist_ok=True)
    medium = synthetic_medium()
    report = {"medium": {"absorption_per_m": medium.absorption_per_m,
                         "scattering_per_m": medium.scattering_per_m,
                         "g": medium.g, "group_index": medium.group_index,
                         "provenance": medium.provenance},
              "environment": {"python": sys.version, "platform": platform.platform(),
                              "numpy": np.__version__}}
    began = perf_counter()
    report["stationary_modes"] = stationary_table(medium)
    print("stationary table done", flush=True)
    report["real_k_comparison"] = real_k_comparison(medium, arguments.quick)
    print("real-k comparison done", flush=True)
    report["crossing_monte_carlo"] = crossing_check(medium, arguments.quick)
    print("crossing MC done", flush=True)
    report["covariance_identity"] = covariance_identity(medium, arguments.quick)
    print("covariance identity done", flush=True)
    report["absorption_factorisation"] = factorisation_check(medium)
    report["track_length_cross_check"] = track_length_cross_check(medium, arguments.quick)
    report["total_seconds"] = perf_counter() - began
    (out / "directionality.json").write_text(
        json.dumps(report, default=serial, indent=2), encoding="utf-8")
    print(f"written: {out / 'directionality.json'}")


if __name__ == "__main__":
    main()
