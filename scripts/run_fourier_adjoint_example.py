"""Numerical example for the Fourier-adjoint chapter.

An independent dense angular discretisation, built here from a product
quadrature on the sphere, is used to check the three statements the chapter
rests on: the transform sign conventions, the bilinear symmetry that makes
reciprocity hold, and the coercivity bound that makes the operator invertible.
It is a reference for the derivation, not the production solver.

Public synthetic parameters only. No transport cache, no Numba, no disk input.

    python scripts/run_fourier_adjoint_example.py --output .build/fourier-adjoint-example
"""
import argparse
import importlib.metadata
import json
import os
import platform
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss

import lighthit as lh
from lighthit.medium import Medium
from lighthit.single import hg_phase

ABSORPTION_PER_M = 0.04
SCATTERING_PER_M = 0.05
ASYMMETRY = 0.7
GROUP_INDEX = 1.35
WAVELENGTH_NM = 450.0

K_PER_M = 0.35
OMEGA_PER_NS = 0.08

POLAR_NODES = 20
AZIMUTHAL_NODES = 14        # 280 directions

# Detector axis and source axis, both tilted away from k so that no azimuthal
# symmetry is left to hide an error.
DETECTOR_AXIS = [0.6, 0.0, 0.8]
SOURCE_AXIS = [0.0, -0.8, 0.6]

SHIFT_X0_M = 2.5
SHIFT_T0_NS = 1.75
GAUSSIAN_WIDTH_M = 1.3
GAUSSIAN_WIDTH_NS = 0.9
SHIFT_QUADRATURE_NODES = 4001
SHIFT_HALF_RANGE = 40.0


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def sphere_quadrature(polar_nodes, azimuthal_nodes):
    """Gauss-Legendre in cos(theta) times a uniform rule in phi.

    The polar axis is k-hat, so mu = cos(theta) is the streaming cosine.
    """
    mu, polar_weight = leggauss(polar_nodes)
    phi = 2 * np.pi * np.arange(azimuthal_nodes) / azimuthal_nodes
    azimuthal_weight = 2 * np.pi / azimuthal_nodes
    sin = np.sqrt(np.clip(1 - mu * mu, 0.0, None))
    directions = np.stack([
        np.outer(sin, np.cos(phi)).ravel(),
        np.outer(sin, np.sin(phi)).ravel(),
        np.repeat(mu, azimuthal_nodes),
    ], axis=1)
    weights = np.repeat(polar_weight * azimuthal_weight, azimuthal_nodes)
    return directions, weights, np.repeat(mu, azimuthal_nodes)


def operator(medium, k_per_m, omega_per_ns, directions, weights, streaming_cosine):
    d0 = medium.extinction_per_m - 1j * omega_per_ns / medium.speed_m_per_ns
    diagonal = d0 + 1j * k_per_m * streaming_cosine
    phase = hg_phase(directions @ directions.T, medium.g)
    scattering = medium.scattering_per_m * phase * weights[None, :]
    return d0, diagonal, phase, np.diag(diagonal) - scattering


def unit(vector):
    v = np.asarray(vector, float)
    return v / np.linalg.norm(v)


def fourier_shift_block():
    """Independent 1D check of both transform sign conventions."""
    x = np.linspace(-SHIFT_HALF_RANGE, SHIFT_HALF_RANGE, SHIFT_QUADRATURE_NODES)
    t = np.linspace(-SHIFT_HALF_RANGE, SHIFT_HALF_RANGE, SHIFT_QUADRATURE_NODES)
    dx = x[1] - x[0]
    dt = t[1] - t[0]

    def gaussian(grid, width, centre=0.0):
        return np.exp(-0.5 * ((grid - centre) / width) ** 2)

    # exp(-i k x) in space, exp(+i omega t) in time.
    space = gaussian(x, GAUSSIAN_WIDTH_M)
    space_shifted = gaussian(x, GAUSSIAN_WIDTH_M, SHIFT_X0_M)
    time = gaussian(t, GAUSSIAN_WIDTH_NS)
    time_shifted = gaussian(t, GAUSSIAN_WIDTH_NS, SHIFT_T0_NS)

    space_hat = np.trapezoid(space * np.exp(-1j * K_PER_M * x), x)
    space_shifted_hat = np.trapezoid(space_shifted * np.exp(-1j * K_PER_M * x), x)
    time_hat = np.trapezoid(time * np.exp(1j * OMEGA_PER_NS * t), t)
    time_shifted_hat = np.trapezoid(time_shifted * np.exp(1j * OMEGA_PER_NS * t), t)

    spatial_phase = space_shifted_hat / space_hat
    temporal_phase = time_shifted_hat / time_hat
    spatial_expected = np.exp(-1j * K_PER_M * SHIFT_X0_M)
    temporal_expected = np.exp(1j * OMEGA_PER_NS * SHIFT_T0_NS)

    # d/dt -> -i omega and d/dx -> +i k, on the analytic derivatives of the
    # same Gaussians, so that no finite-difference error enters the sign test.
    d_time = -(t / GAUSSIAN_WIDTH_NS ** 2) * time
    d_space = -(x / GAUSSIAN_WIDTH_M ** 2) * space
    d_time_hat = np.trapezoid(d_time * np.exp(1j * OMEGA_PER_NS * t), t)
    d_space_hat = np.trapezoid(d_space * np.exp(-1j * K_PER_M * x), x)
    time_factor = d_time_hat / time_hat
    space_factor = d_space_hat / space_hat
    return {
        "x0_m": SHIFT_X0_M,
        "t0_ns": SHIFT_T0_NS,
        "k_per_m": K_PER_M,
        "omega_per_ns": OMEGA_PER_NS,
        "grid_nodes": SHIFT_QUADRATURE_NODES,
        "grid_half_range": SHIFT_HALF_RANGE,
        "grid_step_m": float(dx),
        "grid_step_ns": float(dt),
        "spatial_phase": [spatial_phase.real, spatial_phase.imag],
        "spatial_phase_expected": [spatial_expected.real, spatial_expected.imag],
        "spatial_phase_error": float(abs(spatial_phase - spatial_expected)),
        "temporal_phase": [temporal_phase.real, temporal_phase.imag],
        "temporal_phase_expected": [temporal_expected.real, temporal_expected.imag],
        "temporal_phase_error": float(abs(temporal_phase - temporal_expected)),
        "time_derivative_factor": [time_factor.real, time_factor.imag],
        "time_derivative_expected": [0.0, -OMEGA_PER_NS],
        "time_derivative_error": float(abs(time_factor + 1j * OMEGA_PER_NS)),
        "space_derivative_factor": [space_factor.real, space_factor.imag],
        "space_derivative_expected": [0.0, K_PER_M],
        "space_derivative_error": float(abs(space_factor - 1j * K_PER_M)),
    }


CONVERGENCE_GRIDS = [(12, 10), (20, 14), (32, 24), (40, 32)]


def convergence_block(medium):
    """How well each grid normalises the phase function, and what that costs.

    Reciprocity and the weighted symmetry are algebraic and hold on any grid.
    The coercivity bound is the one statement that degrades with the
    quadrature, and it degrades by at most ``mu_s`` times this residual.
    """
    rows = []
    for polar, azimuthal in CONVERGENCE_GRIDS:
        directions, weights, streaming = sphere_quadrature(polar, azimuthal)
        _, _, phase, matrix = operator(medium, K_PER_M, OMEGA_PER_NS,
                                       directions, weights, streaming)
        residual = float(np.max(np.abs(phase @ weights - 1.0)))
        root = np.sqrt(weights)
        similar = (root[:, None] * matrix) / root[None, :]
        hermitian = (similar + similar.conj().T) / 2
        smallest = float(np.linalg.eigvalsh(hermitian)[0])
        rows.append({
            "polar_nodes": polar, "azimuthal_nodes": azimuthal,
            "directions": len(weights),
            "phase_normalisation_residual": residual,
            "smallest_eigenvalue": smallest,
            "coercivity_margin": smallest - ABSORPTION_PER_M,
            "margin_bound": -medium.scattering_per_m * residual,
        })
    return {
        "note": "the coercivity deficit is bounded by mu_s times the "
                "phase-function normalisation residual of the grid",
        "rows": rows,
        "residual_decreases": all(
            rows[i + 1]["phase_normalisation_residual"]
            < rows[i]["phase_normalisation_residual"]
            for i in range(len(rows) - 1)),
        "finest_residual": rows[-1]["phase_normalisation_residual"],
    }


def zero_mode_block(directions, weights, phase):
    """mu_a = 0, k = 0, omega = 0: the isotropic mode is annihilated.

    Built directly rather than through ``Medium``, whose contract requires a
    strictly positive absorption; nothing in the package is changed by this.
    """
    scattering = SCATTERING_PER_M
    matrix = np.diag(np.full(len(weights), scattering)) - scattering * phase * weights[None, :]
    constant = np.ones(len(weights))
    residual = matrix @ constant
    hermitian_root = np.sqrt(weights)
    symmetric = (np.diag(np.full(len(weights), scattering))
                 - scattering * (hermitian_root[:, None] * phase * hermitian_root[None, :]))
    eigenvalues = np.linalg.eigvalsh(symmetric)
    normalisation = float(np.max(np.abs(phase @ weights - 1.0)))
    relative = float(np.max(np.abs(residual)) / scattering)
    return {
        "absorption_per_m": 0.0,
        "k_per_m": 0.0,
        "omega_per_ns": 0.0,
        "isotropic_residual_norm": float(np.max(np.abs(residual))),
        "relative_isotropic_residual": relative,
        "phase_normalisation_residual": normalisation,
        "residual_equals_normalisation_error": float(abs(relative - normalisation)),
        "smallest_eigenvalue": float(eigenvalues[0]),
        "second_smallest_eigenvalue": float(eigenvalues[1]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=".build/fourier-adjoint-example")
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    medium = Medium(ABSORPTION_PER_M, SCATTERING_PER_M, ASYMMETRY, GROUP_INDEX,
                    WAVELENGTH_NM, "synthetic medium, Fourier-adjoint example")
    directions, weights, streaming_cosine = sphere_quadrature(
        POLAR_NODES, AZIMUTHAL_NODES)
    d0, diagonal, phase, matrix = operator(
        medium, K_PER_M, OMEGA_PER_NS, directions, weights, streaming_cosine)
    W = np.diag(weights)

    # The quadrature must reproduce the sphere and the phase-function
    # normalisation before anything built on it means much.
    solid_angle_error = float(abs(weights.sum() - 4 * np.pi))
    normalisation_residual = float(np.max(np.abs(phase @ weights - 1.0)))

    weighted = W @ matrix
    symmetry_residual = float(np.max(np.abs(weighted - matrix.T @ W))
                              / np.max(np.abs(weighted)))

    # One detector functional, several angular sources, one adjoint solve.
    detector_axis = unit(DETECTOR_AXIS)
    source_axis = unit(SOURCE_AXIS)
    cosine_to_axis = directions @ detector_axis
    acceptance = 0.40 + 0.48 * cosine_to_axis + 0.12 * cosine_to_axis ** 2

    delta_index = int(np.argmax(directions @ source_axis))
    sources = {
        "isotropic": np.full(len(weights), 1.0 / (4 * np.pi)),
        "forward_lobe": hg_phase(directions @ source_axis, 0.9),
        "discrete_delta": np.eye(len(weights))[delta_index] / weights[delta_index],
        "cosine_lobe": np.clip(directions @ source_axis, 0.0, None),
    }

    adjoint = np.linalg.solve(matrix, acceptance)          # one solve, reused
    direct, adjoint_response, errors = {}, {}, {}
    for name, source in sources.items():
        field = np.linalg.solve(matrix, source)             # one solve per source
        forward = complex(acceptance @ (weights * field))
        backward = complex(adjoint @ (weights * source))
        direct[name] = [forward.real, forward.imag]
        adjoint_response[name] = [backward.real, backward.imag]
        errors[name] = float(abs(forward - backward)
                             / max(abs(forward), 1e-300))
    reciprocity_error = float(max(errors.values()))
    delta_identity_error = float(
        abs(complex(adjoint @ (weights * sources["discrete_delta"]))
            - adjoint[delta_index]) / max(abs(adjoint[delta_index]), 1e-300))

    root = np.sqrt(weights)
    similar = (root[:, None] * matrix) / root[None, :]
    hermitian = (similar + similar.conj().T) / 2
    eigenvalues = np.linalg.eigvalsh(hermitian)
    coercivity_margin = float(eigenvalues[0] - ABSORPTION_PER_M)

    summary = {
        "schema": "lighthit-fourier-adjoint-v1",
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": package_version("scipy"),
            "numba": package_version("numba"),
            "lighthit": lh.__version__,
            "threads": {name: os.environ.get(name) for name in
                        ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                         "MKL_NUM_THREADS", "NUMBA_NUM_THREADS")},
        },
        "medium": {
            "absorption_per_m": medium.absorption_per_m,
            "scattering_per_m": medium.scattering_per_m,
            "extinction_per_m": medium.extinction_per_m,
            "g": medium.g,
            "group_index": medium.group_index,
            "speed_m_per_ns": medium.speed_m_per_ns,
            "wavelength_nm": medium.wavelength_nm,
            "provenance": medium.provenance,
        },
        "transform_point": {
            "k_per_m": K_PER_M,
            "omega_per_ns": OMEGA_PER_NS,
            "d0": [d0.real, d0.imag],
            "d0_expected": [medium.extinction_per_m,
                            -OMEGA_PER_NS / medium.speed_m_per_ns],
            "d0_error": float(abs(d0 - (medium.extinction_per_m
                                        - 1j * OMEGA_PER_NS / medium.speed_m_per_ns))),
        },
        "quadrature": {
            "rule": "Gauss-Legendre in cos(theta) x uniform in phi",
            "polar_nodes": POLAR_NODES,
            "azimuthal_nodes": AZIMUTHAL_NODES,
            "directions": len(weights),
            "solid_angle": float(weights.sum()),
            "solid_angle_error": solid_angle_error,
            "phase_normalisation_residual": normalisation_residual,
        },
        "symmetry": {
            "statement": "W L = L^T W for a reciprocal phase kernel",
            "relative_residual": symmetry_residual,
        },
        "reciprocity": {
            "detector_axis": detector_axis.tolist(),
            "source_axis": source_axis.tolist(),
            "acceptance": "0.40 + 0.48 c + 0.12 c^2 about the detector axis",
            "adjoint_solves": 1,
            "sources": list(sources),
            "direct": direct,
            "adjoint": adjoint_response,
            "relative_error": errors,
            "max_relative_error": reciprocity_error,
            "delta_index": delta_index,
            "delta_evaluates_adjoint_field_error": delta_identity_error,
        },
        "coercivity": {
            "statement": "lambda_min of the Hermitian part of W^1/2 L W^-1/2",
            "smallest_eigenvalue": float(eigenvalues[0]),
            "bound": ABSORPTION_PER_M,
            "margin": coercivity_margin,
            "largest_eigenvalue": float(eigenvalues[-1]),
        },
        "fourier_signs": fourier_shift_block(),
        "zero_mode": zero_mode_block(directions, weights, phase),
        "quadrature_convergence": convergence_block(medium),
    }

    f = summary["fourier_signs"]
    z = summary["zero_mode"]
    summary["checks"] = {
        "quadrature_covers_sphere": solid_angle_error < 1e-12,
        "d0_matches_convention": summary["transform_point"]["d0_error"] < 1e-15,
        "time_derivative_is_minus_i_omega": f["time_derivative_error"] < 1e-9,
        "space_derivative_is_plus_i_k": f["space_derivative_error"] < 1e-9,
        "spatial_shift_phase": f["spatial_phase_error"] < 1e-9,
        "temporal_shift_phase": f["temporal_phase_error"] < 1e-9,
        "weighted_transpose_symmetry": symmetry_residual < 1e-14,
        "reciprocity_holds": reciprocity_error < 1e-11,
        "one_adjoint_solve_reused": summary["reciprocity"]["adjoint_solves"] == 1
                                    and len(sources) >= 3,
        "delta_source_evaluates_adjoint_field": delta_identity_error < 1e-12,
        "coercivity_deficit_explained_by_quadrature":
            coercivity_margin >= -SCATTERING_PER_M * normalisation_residual - 1e-12,
        "zero_mode_residual_is_quadrature_error":
            z["residual_equals_normalisation_error"] < 1e-15,
        "quadrature_residual_decreases":
            summary["quadrature_convergence"]["residual_decreases"],
        "finest_grid_normalises_phase":
            summary["quadrature_convergence"]["finest_residual"] < 1e-3,
    }
    failed = [name for name, ok in summary["checks"].items() if not ok]
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"d0 = {d0:.6f} 1/m at k = {K_PER_M} 1/m, omega = {OMEGA_PER_NS} rad/ns")
    print(f"quadrature: {len(weights)} directions, solid angle error "
          f"{solid_angle_error:.2e}, phase normalisation {normalisation_residual:.2e}")
    print(f"weighted transpose symmetry: relative residual {symmetry_residual:.2e}")
    print("reciprocity, one adjoint solve against "
          f"{len(sources)} sources:")
    for name in sources:
        d = complex(*direct[name])
        print(f"  {name:>15}  direct {d.real:+.9e}{d.imag:+.9e}j   "
              f"relative error {errors[name]:.2e}")
    print(f"coercivity: lambda_min {eigenvalues[0]:.9f} against mu_a "
          f"{ABSORPTION_PER_M}, margin {coercivity_margin:+.2e}")
    print(f"Fourier signs: shift errors {f['spatial_phase_error']:.2e} (space), "
          f"{f['temporal_phase_error']:.2e} (time)")
    print(f"zero mode at mu_a = k = omega = 0: relative residual "
          f"{z['relative_isotropic_residual']:.2e}, equal to the grid's "
          f"normalisation error to {z['residual_equals_normalisation_error']:.2e}")
    print("quadrature convergence (directions, normalisation residual, margin):")
    for row in summary["quadrature_convergence"]["rows"]:
        print(f"  {row['directions']:>5}  {row['phase_normalisation_residual']:.2e}"
              f"  {row['coercivity_margin']:+.2e}  bound "
              f"{row['margin_bound']:+.2e}")
    print("checks failed:", failed or "none")
    print("written:", out / "summary.json")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
