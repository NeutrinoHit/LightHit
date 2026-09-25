"""Numerical example for the scattering-orders and spatial-inversion chapter.

The multipole inversion is rebuilt here from the formula derived in the
chapter and compared against the package; the first order is checked against a
direct integration in physical time; and the radial cache is checked on its own
grid nodes, between them, and outside its range.

Public synthetic medium only. No private data, no Numba requirement.

    python scripts/run_scattering_orders_example.py --output .build/scattering-orders-example
"""
import argparse
import importlib.metadata
import json
import os
import platform
from importlib.util import find_spec
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.special import eval_legendre, spherical_jn

from lighthit.angular import angular_components, dense_finite_rank_reference
from lighthit.cache import (MOMENT_ORDERS, CacheGrid, ResponseCache,
                            first_order_consistency)
from lighthit.green import GreenResult, PointGreenSolver, SolverSettings
from lighthit.medium import Medium
from lighthit.single import single_scattering_rate, single_spectrum

ABSORPTION_PER_M = 0.04
SCATTERING_PER_M = 0.05
ASYMMETRY = 0.7
GROUP_INDEX = 1.35
WAVELENGTH_NM = 450.0

SCATTERING_DEGREE = 8
SPATIAL_DEGREE = 12
K_MAX_PER_M = 3.0
K_PANEL_PER_M = 0.05
K_ORDER = 8

OMEGA_PER_NS = np.array([0.0, 0.05])
RADIUS_MIN_M, RADIUS_MAX_M, RADIUS_NODES = 5.0, 60.0, 16
INTERPOLATION_NODES = [16, 32, 64]
PROBE_RADIUS_M = 20.0
OFF_GRID_RADIUS_M = 17.3
SOURCE_COSINE = 0.35
SINGLE_COSINE = 0.0
ANGULAR_PROBE_K = 0.2


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def medium():
    return Medium(ABSORPTION_PER_M, SCATTERING_PER_M, ASYMMETRY, GROUP_INDEX,
                  WAVELENGTH_NM, "synthetic medium, scattering-orders example")


def settings():
    return SolverSettings(SCATTERING_DEGREE, SPATIAL_DEGREE, K_MAX_PER_M,
                          K_PANEL_PER_M, K_ORDER, True)


def multipoles(m, s, omega, radius, *, component):
    """M_l(r, omega) rebuilt from the chapter's formula, not from the package.

    M_l = i^l sqrt((2l+1)/2) int k^2 dk / (2 pi^2) j_l(k r) h_l(k, omega)
    """
    k, kw = s.quadrature()
    degrees = np.arange(s.spatial_degree + 1)
    free, first, tail = angular_components(k, omega, m, s.scattering_degree,
                                           s.spatial_degree)
    block = {"free": free, "one_finite_L": first, "two_or_more": tail}[component]
    phase = (1j) ** degrees * np.sqrt((2 * degrees + 1) / 2)
    radial = spherical_jn(degrees[None, :], k[:, None] * radius)
    weight = (kw * k * k) / (2 * np.pi ** 2)
    return phase * np.einsum("k,kj,kj->j", weight, radial,
                             block[:, :s.spatial_degree + 1], optimize=True)


def orders_block(m, s):
    """The three angular pieces, and their sum against the dense control."""
    k = np.array([ANGULAR_PROBE_K])
    free, first, tail = angular_components(k, 0.0, m, s.scattering_degree,
                                           s.scattering_degree)
    total = (free + first + tail)[0]
    control = dense_finite_rank_reference(ANGULAR_PROBE_K, 0.0, m,
                                          s.scattering_degree,
                                          s.scattering_degree,
                                          quadrature_order=1024)
    scale = float(np.max(np.abs(control)))
    return {
        "k_per_m": ANGULAR_PROBE_K,
        "omega_per_ns": 0.0,
        "free_h0": [free[0, 0].real, free[0, 0].imag],
        "one_h0": [first[0, 0].real, first[0, 0].imag],
        "two_or_more_h0": [tail[0, 0].real, tail[0, 0].imag],
        "sum_vs_dense_relative": float(np.max(np.abs(total - control)) / scale),
        "one_over_free_h0": float(abs(first[0, 0] / free[0, 0])),
        "two_or_more_over_one_h0": float(abs(tail[0, 0] / first[0, 0])),
    }


def ballistic_block(m, s):
    """The isotropic-flash ballistic spectrum against its closed form."""
    solver = PointGreenSolver(m, s)
    result = solver.solve(OMEGA_PER_NS, [[0.0, 0.0, PROBE_RADIUS_M]],
                          direction=None)
    packaged = result.components[:, 0, 0]
    q0 = (np.exp(-m.extinction_per_m * PROBE_RADIUS_M)
          / (4 * np.pi * PROBE_RADIUS_M ** 2))
    analytic = q0 * np.exp(1j * OMEGA_PER_NS * PROBE_RADIUS_M / m.speed_m_per_ns)
    return {
        "radius_m": PROBE_RADIUS_M,
        "charge_per_m2": q0,
        "front_time_ns": PROBE_RADIUS_M / m.speed_m_per_ns,
        "packaged": [[z.real, z.imag] for z in packaged],
        "analytic": [[z.real, z.imag] for z in analytic],
        "relative_error": float(np.max(np.abs(packaged - analytic)) / abs(q0)),
    }, result


def inversion_block(m, s):
    """A hand-built multipole inversion against the solver's own component."""
    direction = np.array([0.0, 0.0, 1.0])
    displacement = PROBE_RADIUS_M * np.array(
        [np.sqrt(1 - SOURCE_COSINE ** 2), 0.0, SOURCE_COSINE])
    solver = PointGreenSolver(m, s)
    result = solver.solve(OMEGA_PER_NS, [displacement], direction=direction)
    degrees = np.arange(s.spatial_degree + 1)
    legendre = eval_legendre(degrees, SOURCE_COSINE)
    rebuilt = np.array([
        float_sum for float_sum in (
            np.dot(multipoles(m, s, float(w), PROBE_RADIUS_M,
                              component="two_or_more"), legendre)
            for w in OMEGA_PER_NS)])
    packaged = result.components[:, 0, 2]
    return {
        "radius_m": PROBE_RADIUS_M,
        "cosine": SOURCE_COSINE,
        "spatial_degree": s.spatial_degree,
        "k_nodes": len(s.quadrature()[0]),
        "packaged": [[z.real, z.imag] for z in packaged],
        "rebuilt": [[z.real, z.imag] for z in rebuilt],
        "relative_error": float(np.max(np.abs(rebuilt - packaged))
                                / np.max(np.abs(packaged))),
    }


def first_order_time_block(m):
    """single_spectrum at omega = 0 against a direct integral in time."""
    front = PROBE_RADIUS_M / m.speed_m_per_ns
    spectrum, estimate = single_spectrum(np.array([0.0]), PROBE_RADIUS_M,
                                         SINGLE_COSINE, m)
    def integrand(excess):
        return float(single_scattering_rate(np.array([front + excess]),
                                            PROBE_RADIUS_M, SINGLE_COSINE, m)[0])
    horizon = 36 / (m.extinction_per_m * m.speed_m_per_ns)
    value, error = quad(integrand, 0.0, horizon, limit=400)
    return {
        "radius_m": PROBE_RADIUS_M,
        "cosine": SINGLE_COSINE,
        "front_time_ns": front,
        "horizon_ns": horizon,
        "spectrum_at_zero": float(spectrum[0].real),
        "spectrum_imaginary_part": float(abs(spectrum[0].imag)),
        "time_integral": float(value),
        "time_integral_estimate": float(error),
        "quadrature_estimate": float(estimate),
        "relative_difference": float(abs(value - spectrum[0].real)
                                     / abs(spectrum[0].real)),
    }


FIRST_ORDER_DEGREES = [4, 8, 16, 32]
FIRST_ORDER_RADII = [10.0, 20.0, 40.0]


def first_order_convergence_block(m):
    """Does the finite-rank first order approach the full-HG one as L grows?

    Isotropic flash and isotropic receiver, where only the l = 0 multipole
    survives, so the comparison isolates the angular rank of the medium from
    the angular resolution of the output.
    """
    rows = []
    for radius in FIRST_ORDER_RADII:
        exact, _ = single_spectrum(np.array([0.0]), radius, None, m)
        reference = float(exact[0].real)
        ratios = {}
        for degree in FIRST_ORDER_DEGREES:
            s = SolverSettings(degree, 0, 6.0, 0.04, 10, True)
            k, kw = s.quadrature()
            _, first, _ = angular_components(k, 0.0, m, degree, 0)
            weight = (kw * k * k) / (2 * np.pi ** 2)
            moment = np.sqrt(0.5) * np.sum(
                weight * spherical_jn(0, k * radius) * first[:, 0])
            ratios[str(degree)] = float(moment.real / reference)
        rows.append({"radius_m": radius, "full_hg": reference, "ratio": ratios})
    worst_at_16 = max(abs(row["ratio"]["16"] - 1.0) for row in rows)
    return {
        "geometry": "isotropic flash, isotropic receiver, omega = 0",
        "degrees": FIRST_ORDER_DEGREES,
        "rows": rows,
        "worst_deviation_at_degree_16": worst_at_16,
        "note": "the directed point-source/point-receiver first order is not "
                "tested this way: it carries an integrable 1/rho singularity "
                "at the ballistic root, and the multipole route does not "
                "converge for it at practical settings",
    }


def cache_block(m, s):
    grid = CacheGrid.geometric(RADIUS_MIN_M, RADIUS_MAX_M, RADIUS_NODES,
                               OMEGA_PER_NS)
    cache = ResponseCache.build(m, s, grid)
    node_index = RADIUS_NODES // 2
    node_radius = float(grid.radii_m[node_index])

    on_grid = {}
    for axis, name in enumerate(MOMENT_ORDERS):
        rebuilt = np.array([multipoles(m, s, float(w), node_radius, component=name)
                            for w in OMEGA_PER_NS])
        stored = cache.moments[:, node_index, :, axis]
        on_grid[name] = float(np.max(np.abs(rebuilt - stored))
                              / np.max(np.abs(stored)))

    off_grid = {}
    interpolated = cache.moments_at(np.array([OFF_GRID_RADIUS_M]))
    for axis, name in enumerate(MOMENT_ORDERS):
        rebuilt = np.array([multipoles(m, s, float(w), OFF_GRID_RADIUS_M,
                                       component=name) for w in OMEGA_PER_NS])
        off_grid[name] = float(np.max(np.abs(interpolated[:, 0, :, axis] - rebuilt))
                               / np.max(np.abs(rebuilt)))

    refusals = {}
    for name, radius in (("below", RADIUS_MIN_M * 0.5),
                         ("above", RADIUS_MAX_M * 2.0)):
        try:
            cache.moments_at(np.array([radius]))
        except ValueError as error:
            refusals[name] = {"radius_m": radius, "raised": True,
                              "message": str(error)}
        else:
            refusals[name] = {"radius_m": radius, "raised": False}

    # How the interpolation error behaves as the radial grid is refined. The
    # moments themselves are unchanged; only the spacing between stored nodes
    # is, so this isolates interpolation from every other error in the route.
    exact = {name: np.array([multipoles(m, s, float(w), OFF_GRID_RADIUS_M,
                                        component=name) for w in OMEGA_PER_NS])
             for name in MOMENT_ORDERS}
    refinement = []
    for nodes in INTERPOLATION_NODES:
        finer = ResponseCache.build(
            m, s, CacheGrid.geometric(RADIUS_MIN_M, RADIUS_MAX_M, nodes,
                                      OMEGA_PER_NS))
        values = finer.moments_at(np.array([OFF_GRID_RADIUS_M]))
        worst = max(
            float(np.max(np.abs(values[:, 0, :, axis] - exact[name]))
                  / np.max(np.abs(exact[name])))
            for axis, name in enumerate(MOMENT_ORDERS))
        refinement.append({"radial_nodes": nodes,
                           "log_spacing": float(np.log(RADIUS_MAX_M / RADIUS_MIN_M)
                                                / (nodes - 1)),
                           "relative_error": worst})

    consistency = first_order_consistency(
        cache, np.geomspace(RADIUS_MIN_M * 1.1, RADIUS_MAX_M * 0.9, 12))
    return {
        "moment_orders": list(MOMENT_ORDERS),
        "stores_ballistic": False,
        "grid": {"radii": RADIUS_NODES, "range_m": [RADIUS_MIN_M, RADIUS_MAX_M],
                 "spacing": "geometric",
                 "frequencies": OMEGA_PER_NS.tolist(),
                 "shape": list(cache.moments.shape)},
        "scale_removed": "exp(-mu_a r) / (4 pi r^2)",
        "on_grid_relative": on_grid,
        "off_grid_radius_m": OFF_GRID_RADIUS_M,
        "off_grid_relative": off_grid,
        "extrapolation": refusals,
        "interpolation_refinement": refinement,
        "first_order_consistency": consistency,
        "all_finite": bool(np.isfinite(cache.moments).all()),
    }, cache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=".build/scattering-orders-example")
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    m, s = medium(), settings()
    orders = orders_block(m, s)
    ballistic, green = ballistic_block(m, s)
    inversion = inversion_block(m, s)
    first_order = first_order_time_block(m)
    first_order_convergence = first_order_convergence_block(m)
    cache_summary, cache = cache_block(m, s)

    # The stored component order is part of the contract.
    # GreenResult.save writes the file and returns nothing, so keep the path.
    path = out / "green-probe.npz"
    green.save(path)
    with np.load(path, allow_pickle=False) as data:
        names = json.loads(str(data["metadata"]))["component_names"]
    path.unlink()

    numba_available = find_spec("numba") is not None
    backends = {"numba_available": numba_available}
    if numba_available:
        k, _ = s.quadrature()
        d0 = m.extinction_per_m
        from lighthit.angular import _free_moments_and_ratios
        nb_b, nb_r = _free_moments_and_ratios(k, d0, s.spatial_degree, backend="numba")
        np_b, np_r = _free_moments_and_ratios(k, d0, s.spatial_degree, backend="numpy")
        backends["moment_difference"] = float(np.max(np.abs(nb_b - np_b)))
        backends["ratio_difference"] = float(np.max(np.abs(nb_r - np_r)))
    else:
        backends["note"] = "optional accelerate extra absent; NumPy backend only"

    summary = {
        "schema": "lighthit-scattering-orders-v1",
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": package_version("scipy"),
            "numba": package_version("numba"),
            "lighthit": package_version("lighthit"),
            "threads": {name: os.environ.get(name) for name in
                        ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                         "MKL_NUM_THREADS", "NUMBA_NUM_THREADS")},
        },
        "configuration": {
            "absorption_per_m": ABSORPTION_PER_M,
            "scattering_per_m": SCATTERING_PER_M,
            "g": ASYMMETRY, "group_index": GROUP_INDEX,
            "speed_m_per_ns": m.speed_m_per_ns,
            "scattering_degree": SCATTERING_DEGREE,
            "spatial_degree": SPATIAL_DEGREE,
            "k_max_per_m": K_MAX_PER_M, "k_panel_per_m": K_PANEL_PER_M,
            "k_order": K_ORDER, "taper": True,
            "k_nodes": int(len(s.quadrature()[0])),
            "omega_per_ns": OMEGA_PER_NS.tolist(),
        },
        "orders": orders,
        "ballistic": ballistic,
        "spatial_inversion": inversion,
        "first_order_in_time": first_order,
        "first_order_convergence": first_order_convergence,
        "cache": cache_summary,
        "component_names": names,
        "backends": backends,
    }
    summary["checks"] = {
        "orders_sum_to_dense_reference": orders["sum_vs_dense_relative"] < 1e-10,
        "ballistic_matches_closed_form": ballistic["relative_error"] < 1e-12,
        "hand_built_inversion_matches_solver":
            inversion["relative_error"] < 1e-12,
        "first_order_matches_time_integral":
            first_order["relative_difference"] < 1e-6,
        "first_order_real_at_zero_frequency":
            first_order["spectrum_imaginary_part"] < 1e-18,
        "finite_rank_first_order_converges_to_full_hg":
            first_order_convergence["worst_deviation_at_degree_16"] < 1e-3,
        "cache_reproduces_moments_on_grid":
            max(cache_summary["on_grid_relative"].values()) < 1e-12,
        "cache_interpolation_improves_with_nodes": all(
            a["relative_error"] > b["relative_error"] for a, b in zip(
                cache_summary["interpolation_refinement"],
                cache_summary["interpolation_refinement"][1:])),
        "cache_interpolates_on_finest_grid":
            cache_summary["interpolation_refinement"][-1]["relative_error"] < 1e-4,
        "cache_refuses_extrapolation_below":
            cache_summary["extrapolation"]["below"]["raised"],
        "cache_refuses_extrapolation_above":
            cache_summary["extrapolation"]["above"]["raised"],
        "cache_stores_two_orders": list(MOMENT_ORDERS) == ["one_finite_L",
                                                           "two_or_more"],
        "component_order_is_stable":
            names == ["ballistic", "one_exact_HG", "two_or_more_HG_L"],
        "everything_finite": bool(
            cache_summary["all_finite"]
            and np.isfinite(green.components).all()),
        "backends_agree": (backends.get("ratio_difference", 0.0) < 1e-12
                           if numba_available else True),
    }
    failed = [name for name, ok in summary["checks"].items() if not ok]
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"angular orders at k = {ANGULAR_PROBE_K} 1/m: "
          f"sum vs dense {orders['sum_vs_dense_relative']:.2e}; "
          f"|one|/|free| = {orders['one_over_free_h0']:.4f}, "
          f"|>=2|/|one| = {orders['two_or_more_over_one_h0']:.4f}")
    print(f"ballistic at {PROBE_RADIUS_M} m: charge {ballistic['charge_per_m2']:.6e} "
          f"m^-2, front {ballistic['front_time_ns']:.3f} ns, "
          f"relative error {ballistic['relative_error']:.2e}")
    print(f"hand-built inversion vs solver: {inversion['relative_error']:.2e} "
          f"(J = {SPATIAL_DEGREE}, {summary['configuration']['k_nodes']} k nodes)")
    print(f"first order at omega = 0: spectrum {first_order['spectrum_at_zero']:.9e}, "
          f"time integral {first_order['time_integral']:.9e}, "
          f"relative {first_order['relative_difference']:.2e}")
    print("cache moments on grid nodes:",
          {k_: f"{v:.1e}" for k_, v in cache_summary["on_grid_relative"].items()})
    print(f"cache interpolation at {OFF_GRID_RADIUS_M} m:",
          {k_: f"{v:.1e}" for k_, v in cache_summary["off_grid_relative"].items()})
    print("interpolation against radial nodes:")
    for row in cache_summary["interpolation_refinement"]:
        print(f"  {row['radial_nodes']:>3} nodes, log spacing "
              f"{row['log_spacing']:.4f}: {row['relative_error']:.2e}")
    print("finite-rank first order against full HG (isotropic geometry):")
    for row in first_order_convergence["rows"]:
        ratios = " ".join(f"L={d}: {row['ratio'][str(d)]:.4f}"
                          for d in FIRST_ORDER_DEGREES)
        print(f"  r = {row['radius_m']:>5.1f} m  {ratios}")
    fc = cache_summary["first_order_consistency"]
    print(f"first-order routes differ by {fc['max_relative_first_order']:.2e} "
          f"(median {fc['median_relative_first_order']:.2e}), "
          f"{fc['max_first_order_error_over_total']:.2e} of the total")
    print("component names:", names)
    print("numba available:", numba_available)
    print("checks failed:", failed or "none")
    print("written:", out / "summary.json")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
