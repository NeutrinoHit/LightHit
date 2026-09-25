"""Numerical example for the angular-system chapter.

Everything here is rebuilt from the definitions -- the orthonormal basis, the
streaming recurrence, the tridiagonal matrix and its tail correction -- and
then compared against the package. Public synthetic medium only; no spatial
cache, no private data, no Numba requirement.

    python scripts/run_angular_system_example.py --output .build/angular-system-example
"""
import argparse
import importlib.metadata
import json
import os
import platform
from importlib.util import find_spec
from pathlib import Path

import numpy as np
from scipy.special import eval_legendre, roots_legendre

import lighthit as lh
from lighthit.angular import (angular_components, dense_finite_rank_reference,
                              free_moments_and_tail, solve_tail_system)
from lighthit.medium import Medium

ABSORPTION_PER_M = 0.04
SCATTERING_PER_M = 0.05
ASYMMETRY = 0.7
GROUP_INDEX = 1.35
WAVELENGTH_NM = 450.0

K_PER_M = 0.2
OMEGA_PER_NS = 0.0
SCATTERING_DEGREE = 8

TRUNCATION_DEGREES = [3, 8, 16, 32]
BRANCH_K = np.geomspace(1e-3, 40.0, 61)
DEEP_EXTRA = 400
QUADRATURE_ORDER = 1024


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def medium():
    return Medium(ABSORPTION_PER_M, SCATTERING_PER_M, ASYMMETRY, GROUP_INDEX,
                  WAVELENGTH_NM, "synthetic medium, angular-system example")


def orthonormal(degrees, mu):
    """p_l = sqrt((2l+1)/2) P_l, orthonormal on [-1, 1]."""
    return eval_legendre(degrees[:, None], mu) * np.sqrt((2 * degrees + 1) / 2)[:, None]


def coupling(degrees):
    """a_l = l / sqrt(4 l^2 - 1), with a_0 = 0."""
    a = np.zeros(len(degrees), float)
    positive = degrees > 0
    l = degrees[positive].astype(float)
    a[positive] = l / np.sqrt(4 * l * l - 1)
    return a


def basis_block():
    mu, w = roots_legendre(QUADRATURE_ORDER)
    degrees = np.arange(SCATTERING_DEGREE + 2)
    p = orthonormal(degrees, mu)
    gram = (p * w) @ p.T
    orthonormality = float(np.max(np.abs(gram - np.eye(len(degrees)))))

    # Streaming: mu p_l = a_l p_(l-1) + a_(l+1) p_(l+1), checked pointwise and
    # by projection.
    a = coupling(degrees)
    pointwise = 0.0
    for l in range(1, len(degrees) - 1):
        rebuilt = a[l] * p[l - 1] + a[l + 1] * p[l + 1]
        pointwise = max(pointwise, float(np.max(np.abs(mu * p[l] - rebuilt))))
    projection = (p * (w * mu)) @ p.T
    expected = np.zeros_like(projection)
    for l in range(len(degrees)):
        if l > 0:
            expected[l, l - 1] = a[l]
        if l + 1 < len(degrees):
            expected[l, l + 1] = a[l + 1]
    projection_error = float(np.max(np.abs(projection - expected)))

    constant = p @ w                      # integral of p_l over [-1, 1]
    constant_expected = np.zeros(len(degrees))
    constant_expected[0] = np.sqrt(2)
    return {
        "quadrature_order": QUADRATURE_ORDER,
        "degrees": len(degrees),
        "orthonormality_error": orthonormality,
        "streaming_pointwise_error": pointwise,
        "streaming_projection_error": projection_error,
        "coupling_a": a.tolist(),
        "constant_projection": constant.tolist(),
        "constant_projection_error": float(np.max(np.abs(constant - constant_expected))),
    }


def assemble(d0, k, gamma, tail):
    """The tridiagonal matrix of the chapter, built here from its definition."""
    size = len(gamma)
    a = coupling(np.arange(size + 1))
    matrix = np.diag(d0 - gamma).astype(complex)
    for l in range(1, size):
        matrix[l, l - 1] = matrix[l - 1, l] = 1j * k * a[l]
    matrix[size - 1, size - 1] += 1j * k * a[size] * tail
    return matrix


def solve_at(medium_, k, omega, degree, *, tail=True):
    d0 = medium_.extinction_per_m - 1j * omega / medium_.speed_m_per_ns
    _, terminal = free_moments_and_tail(np.array([k]), d0, degree)
    gamma = medium_.scattering_per_m * medium_.g ** np.arange(degree + 1)
    ratio = terminal[0] if tail else 0.0 + 0.0j
    matrix = assemble(d0, k, gamma, ratio)
    rhs = np.zeros(degree + 1, complex)
    rhs[0] = np.sqrt(2)
    return d0, ratio, matrix, np.linalg.solve(matrix, rhs)


def deep_continued_fraction(d0, k, degree, extra=DEEP_EXTRA):
    """An independent evaluation of the same ratio, from a much deeper start."""
    z = 1j * d0 / k
    ratio = 0.0 + 0.0j                     # a deliberately different seed
    for l in range(degree + 1 + extra, 0, -1):
        ratio = l / ((2 * l + 1) * z - (l + 1) * ratio)
        if l - 1 == degree:
            unnormalised = ratio
    return np.sqrt((2 * degree + 3) / (2 * degree + 1)) * unnormalised


def branch_block(d0, degrees):
    """Reproduce the documented branch rule independently of the package."""
    rows = []
    for degree in degrees:
        z = 1j * d0 / BRANCH_K
        root = np.sqrt(z - 1) * np.sqrt(z + 1)
        decay = 1 / (z + root)
        decay = np.where(abs(decay) > 1, 1 / decay, decay)
        eta = -np.log(np.abs(decay))
        forward = eta * (degree + 3) < 3
        depth = degree + 1 + np.maximum(32, np.ceil(28 / eta[~forward])).astype(int) \
            if np.any(~forward) else np.array([], int)
        rows.append({
            "degree": int(degree),
            "k_nodes": len(BRANCH_K),
            "forward_nodes": int(forward.sum()),
            "miller_nodes": int((~forward).sum()),
            "max_miller_depth": int(depth.max()) if len(depth) else 0,
            "min_decay_rate": float(eta.min()),
            "max_decay_rate": float(eta.max()),
        })
    return {
        "rule": "forward when eta * (N + 3) < 3, else Miller from "
                "N + 1 + max(32, ceil(28 / eta))",
        "constants": {"forward_threshold": 3, "floor_depth": 32, "depth_scale": 28},
        "note": "implementation safety choices validated for the current "
                "double-precision code, not physical parameters",
        "k_range_per_m": [float(BRANCH_K[0]), float(BRANCH_K[-1])],
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=".build/angular-system-example")
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    m = medium()
    L = SCATTERING_DEGREE
    d0, ratio, matrix, field = solve_at(m, K_PER_M, OMEGA_PER_NS, L)

    # Structure of the assembled matrix.
    structure = {
        "d0": [d0.real, d0.imag],
        "d0_expected": [m.extinction_per_m, -OMEGA_PER_NS / m.speed_m_per_ns],
        "row0_diagonal": [(d0 - m.scattering_per_m).real, (d0 - m.scattering_per_m).imag],
        "row0_diagonal_expected": [m.absorption_per_m,
                                   -OMEGA_PER_NS / m.speed_m_per_ns],
        "tail_ratio": [ratio.real, ratio.imag],
        "last_diagonal": [matrix[L, L].real, matrix[L, L].imag],
        "diagonal": [[matrix[i, i].real, matrix[i, i].imag] for i in range(L + 1)],
        "first_offdiagonal": [[matrix[i, i + 1].real, matrix[i, i + 1].imag]
                              for i in range(L)],
        "symmetry_error": float(np.max(np.abs(matrix - matrix.T))),
        "bandwidth_violation": float(np.max(np.abs(
            matrix - np.triu(np.tril(matrix, 1), -1)))),
        "hermitian_gap": float(np.max(np.abs(matrix - matrix.conj().T))),
    }

    # The package's tridiagonal solve against the dense solve of the same matrix.
    gamma = m.scattering_per_m * m.g ** np.arange(L + 1)
    rhs = np.zeros((1, L + 1), complex)
    rhs[0, 0] = np.sqrt(2)
    packaged = solve_tail_system(np.array([K_PER_M]), d0, np.array([ratio]),
                                 gamma, rhs)[0]
    tridiagonal_error = float(np.max(np.abs(packaged - field))
                              / np.max(np.abs(field)))

    # The assembled components against an independent dense reference.
    parts = angular_components(np.array([K_PER_M]), OMEGA_PER_NS, m, L, L)
    assembled = sum(part[0] for part in parts)
    reference = dense_finite_rank_reference(K_PER_M, OMEGA_PER_NS, m, L, L,
                                            quadrature_order=QUADRATURE_ORDER)
    reference_error = float(np.max(np.abs(assembled - reference))
                            / np.max(np.abs(reference)))
    assembled_vs_direct = float(np.max(np.abs(assembled - field))
                                / np.max(np.abs(field)))

    # Independent deep continued fraction for the same terminal ratio.
    deep = deep_continued_fraction(d0, K_PER_M, L)
    ratio_error = float(abs(deep - ratio) / abs(ratio))

    # k = 0 is its own branch, not a limit.
    zero_b, zero_tail = free_moments_and_tail(np.array([0.0]), d0, L)
    zero_block = {
        "b0": [zero_b[0, 0].real, zero_b[0, 0].imag],
        "b0_expected": [(np.sqrt(2) / d0).real, (np.sqrt(2) / d0).imag],
        "b0_error": float(abs(zero_b[0, 0] - np.sqrt(2) / d0)),
        "max_abs_b_above_zero": float(np.max(np.abs(zero_b[0, 1:]))),
        "tail_ratio": [zero_tail[0].real, zero_tail[0].imag],
        "max_abs_tail": float(abs(zero_tail[0])),
    }

    # The solve must not grow with the output degree.
    wide = angular_components(np.array([K_PER_M]), OMEGA_PER_NS, m, L, 3 * L)
    wide_assembled = sum(part[0] for part in wide)
    j_block = {
        "output_degrees": [L, 3 * L],
        "low_degree_difference": float(np.max(np.abs(
            wide_assembled[:L + 1] - assembled))),
        "relative_low_degree_difference": float(np.max(np.abs(
            wide_assembled[:L + 1] - assembled)) / np.max(np.abs(assembled))),
        "extended_length": int(len(wide_assembled)),
    }

    # Exact tail against the hard truncation h_(L+1) = 0, both against the
    # independent dense reference at the same scattering degree.
    truncation_rows = []
    for degree in TRUNCATION_DEGREES:
        _, _, _, exact = solve_at(m, K_PER_M, OMEGA_PER_NS, degree)
        _, _, _, hard = solve_at(m, K_PER_M, OMEGA_PER_NS, degree, tail=False)
        control = dense_finite_rank_reference(K_PER_M, OMEGA_PER_NS, m, degree,
                                              degree,
                                              quadrature_order=QUADRATURE_ORDER)
        scale = np.max(np.abs(control))
        truncation_rows.append({
            "scattering_degree": degree,
            "exact_tail_error": float(np.max(np.abs(exact - control)) / scale),
            "hard_truncation_error": float(np.max(np.abs(hard - control)) / scale),
            "h0_exact": [exact[0].real, exact[0].imag],
            "h0_hard": [hard[0].real, hard[0].imag],
            "h0_reference": [control[0].real, control[0].imag],
        })

    numba_available = find_spec("numba") is not None
    backend_block = {"numba_available": numba_available}
    if numba_available:
        from lighthit.angular import _free_moments_and_ratios
        numpy_b, numpy_r = _free_moments_and_ratios(BRANCH_K, d0, L, backend="numpy")
        numba_b, numba_r = _free_moments_and_ratios(BRANCH_K, d0, L, backend="numba")
        backend_block["moment_difference"] = float(np.max(np.abs(numpy_b - numba_b)))
        backend_block["ratio_difference"] = float(np.max(np.abs(numpy_r - numba_r)))
    else:
        backend_block["note"] = ("the optional accelerate extra is absent; the "
                                 "NumPy backend is the only one exercised here")

    summary = {
        "schema": "lighthit-angular-system-v1",
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
        "configuration": {
            "absorption_per_m": ABSORPTION_PER_M,
            "scattering_per_m": SCATTERING_PER_M,
            "g": ASYMMETRY,
            "group_index": GROUP_INDEX,
            "speed_m_per_ns": m.speed_m_per_ns,
            "k_per_m": K_PER_M,
            "omega_per_ns": OMEGA_PER_NS,
            "scattering_degree": L,
            "truncation_degrees": TRUNCATION_DEGREES,
            "quadrature_order": QUADRATURE_ORDER,
            "deep_extra_degrees": DEEP_EXTRA,
        },
        "basis": basis_block(),
        "matrix": structure,
        "solves": {
            "tridiagonal_vs_dense_relative": tridiagonal_error,
            "assembled_vs_direct_relative": assembled_vs_direct,
            "assembled_vs_reference_relative": reference_error,
            "reference": "dense_finite_rank_reference, Gauss angular quadrature",
        },
        "tail_ratio": {
            "packaged": [ratio.real, ratio.imag],
            "deep_continued_fraction": [deep.real, deep.imag],
            "relative_difference": ratio_error,
            "seed": "zero, from degree L + %d" % DEEP_EXTRA,
        },
        "zero_wavenumber": zero_block,
        "output_degree": j_block,
        "hard_truncation": truncation_rows,
        "branches": branch_block(d0, TRUNCATION_DEGREES),
        "backends": backend_block,
    }

    b = summary["basis"]
    worst_exact = max(row["exact_tail_error"] for row in truncation_rows)
    summary["checks"] = {
        "basis_orthonormal": b["orthonormality_error"] < 1e-12,
        "streaming_recurrence_pointwise": b["streaming_pointwise_error"] < 1e-12,
        "streaming_recurrence_projection": b["streaming_projection_error"] < 1e-12,
        "constant_projects_to_sqrt_two": b["constant_projection_error"] < 1e-12,
        "matrix_is_symmetric": structure["symmetry_error"] < 1e-15,
        "matrix_is_tridiagonal": structure["bandwidth_violation"] == 0.0,
        "matrix_is_not_hermitian": structure["hermitian_gap"] > 0.0,
        "row0_is_absorption": abs(complex(*structure["row0_diagonal"])
                                  - complex(*structure["row0_diagonal_expected"])) < 1e-15,
        "tridiagonal_matches_dense": tridiagonal_error < 1e-12,
        "assembled_matches_direct": assembled_vs_direct < 1e-12,
        "assembled_matches_reference": reference_error < 1e-10,
        "tail_ratio_matches_deep_fraction": ratio_error < 1e-12,
        "zero_wavenumber_moment": zero_block["b0_error"] < 1e-15,
        "zero_wavenumber_has_no_higher_moments":
            zero_block["max_abs_b_above_zero"] == 0.0,
        "zero_wavenumber_has_no_tail": zero_block["max_abs_tail"] == 0.0,
        "solve_independent_of_output_degree":
            j_block["relative_low_degree_difference"] < 1e-12,
        "exact_tail_accurate": worst_exact < 1e-10,
        "hard_truncation_visibly_worse":
            truncation_rows[0]["hard_truncation_error"]
            > 1e3 * max(truncation_rows[0]["exact_tail_error"], 1e-16),
        "all_finite": bool(np.isfinite(field).all() and np.isfinite(assembled).all()),
        "backends_agree": (
            max(backend_block.get("moment_difference", 0.0),
                backend_block.get("ratio_difference", 0.0)) < 1e-12
            if numba_available else True),
    }
    failed = [name for name, ok in summary["checks"].items() if not ok]
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"d0 = {d0:.6f} 1/m, k = {K_PER_M} 1/m, L = {L}")
    print(f"row 0 diagonal = {complex(*structure['row0_diagonal']):.6f} "
          f"(mu_a = {m.absorption_per_m})")
    print(f"tail ratio r_L = {ratio:.9f}; deep continued fraction differs by "
          f"{ratio_error:.2e}")
    print(f"tridiagonal vs dense {tridiagonal_error:.2e}; assembled vs dense "
          f"reference {reference_error:.2e}")
    print(f"{'L':>4} {'exact tail':>14} {'hard truncation':>18}")
    for row in truncation_rows:
        print(f"{row['scattering_degree']:>4} {row['exact_tail_error']:>14.3e} "
              f"{row['hard_truncation_error']:>18.3e}")
    print("branch counts over "
          f"{len(BRANCH_K)} k nodes from {BRANCH_K[0]:g} to {BRANCH_K[-1]:g} 1/m:")
    for row in summary["branches"]["rows"]:
        print(f"  L = {row['degree']:>3}: forward {row['forward_nodes']:>3}, "
              f"Miller {row['miller_nodes']:>3}, deepest start "
              f"{row['max_miller_depth']}")
    print("numba available:", numba_available)
    print("checks failed:", failed or "none")
    print("written:", out / "summary.json")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
