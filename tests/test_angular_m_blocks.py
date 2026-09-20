"""The azimuthal blocks of the angular solve, and their free tail closure."""
import numpy as np
import pytest

from lighthit.angular import (angular_components, coupling_coefficients,
                              dense_block_reference, free_tail_ratios_m,
                              resolvent_rows, solve_tail_system_m,
                              tail_ratios_continued_fraction,
                              _free_moments_and_ratios)
from lighthit.medium import Medium


def medium():
    return Medium(0.02, 0.05, 0.9, 1.35, 450.0, "test medium")


def wavenumbers():
    return np.concatenate([np.array([0.0]), np.geomspace(0.02, 8.0, 30)])


def test_coupling_coefficients_depend_on_m_squared_and_vanish_at_the_floor():
    for m in (1, 2, 3):
        np.testing.assert_array_equal(coupling_coefficients(8, m),
                                      coupling_coefficients(8, -m))
        a = coupling_coefficients(8, m)
        assert a[m] == 0.0                      # the block starts at l = |m|
        assert np.all(a[:m] == 0.0)
        assert a[m + 1] > 0
    ell = np.arange(1, 9)
    np.testing.assert_allclose(coupling_coefficients(8, 0)[1:],
                               ell / np.sqrt(4 * ell ** 2 - 1))


@pytest.mark.parametrize("omega", [0.0, 0.15])
def test_tail_ratio_m_zero_reproduces_the_existing_scheme(omega):
    k = wavenumbers()
    d0 = medium().extinction_per_m - 1j * omega / medium().speed_m_per_ns
    ratios = free_tail_ratios_m(k, d0, 34, 3)
    _, reference = _free_moments_and_ratios(k, d0, 34)
    assert np.max(np.abs(ratios[:, 0] - reference)) < 1e-13


@pytest.mark.parametrize("omega", [0.0, 0.15])
def test_tail_ratio_matches_a_deeper_continued_fraction(omega):
    """The raising relation is checked against the closure it accelerates."""
    k = wavenumbers()
    d0 = medium().extinction_per_m - 1j * omega / medium().speed_m_per_ns
    degree = 30
    report = {}
    ratios = free_tail_ratios_m(k, d0, degree, 3, report=report)
    for m in (1, 2, 3):
        deep = tail_ratios_continued_fraction(k, d0, degree, m, depth=40000)
        window = slice(m + 1, degree)
        relative = (np.abs(ratios[:, m, window] - deep[:, window])
                    / np.maximum(np.abs(deep[:, window]), 1e-300))
        assert np.nanmax(relative) < 1e-11
    assert report["tail_raise_fallback_nodes"] == 0


def test_free_tail_ratio_is_zero_at_zero_wavenumber():
    d0 = medium().extinction_per_m + 0j
    ratios = free_tail_ratios_m(np.array([0.0, 1.0]), d0, 8, 2)
    assert np.all(ratios[0] == 0)
    assert np.any(ratios[1] != 0)


def test_continued_fraction_solution_satisfies_its_own_recurrence():
    """The branch is the decaying one: the ratios close the recurrence exactly."""
    k = np.array([0.7])
    d0 = 0.06 - 0.02j
    degree, m = 24, 2
    ratios = tail_ratios_continued_fraction(k, d0, degree, m, depth=6000)[0]
    z = 1j * d0 / k[0]
    a = coupling_coefficients(degree + 1, m)
    y = np.zeros(degree + 2, complex)
    y[m] = 1.0
    for l in range(m, degree + 1):
        y[l + 1] = y[l] * ratios[l]
    residual = [abs(z * y[l] - a[l] * y[l - 1] - a[l + 1] * y[l + 1])
                / abs(z * y[l]) for l in range(m + 1, degree)]
    assert max(residual) < 1e-12
    assert abs(y[degree]) < abs(y[m])           # it decays


@pytest.mark.parametrize("omega", [0.0, 0.1])
def test_resolvent_rows_reduce_to_the_existing_solver_at_m_zero(omega):
    """With an isotropic acceptance the new path is the old path, exactly.

    The existing ``angular_components`` solves with the isotropic source
    ``sqrt(2) e_0``; the row form solves with ``e_0``, so the two differ by that
    normalisation and by nothing else.
    """
    k = wavenumbers()
    _, first, multiple = angular_components(k, omega, medium(), 16, 20)
    rows_first, rows_multiple = resolvent_rows(k, omega, medium(), 16, 20, 0)
    assert (np.max(np.abs(np.sqrt(2) * rows_first[(0, 0)] - first))
            / np.max(np.abs(first))) < 1e-14
    assert (np.max(np.abs(np.sqrt(2) * rows_multiple[(0, 0)] - multiple))
            / np.max(np.abs(multiple))) < 1e-14


def test_resolvent_rows_match_a_dense_inverse_where_truncation_is_harmless():
    """An independent dense inverse, in a regime where the free tail decays fast."""
    strong = Medium(0.30, 0.05, 0.9, 1.35, 450.0, "strongly attenuating")
    k = np.geomspace(0.02, 0.5, 8)
    omega, L, J, top = 0.05, 10, 14, 90
    d0 = strong.extinction_per_m - 1j * omega / strong.speed_m_per_ns
    first, multiple = resolvent_rows(k, omega, strong, L, J, 3)

    def dense(m, with_gamma):
        ell = np.arange(m, top + 1)
        a = coupling_coefficients(top + 1, m)
        scatter = np.where(ell <= L, strong.scattering_per_m * strong.g ** ell, 0.0)
        gamma = scatter if with_gamma else np.zeros(len(ell))
        out = np.empty((len(k), len(ell), len(ell)), complex)
        for index, value in enumerate(k):
            matrix = np.diag(d0 - gamma).astype(complex)
            for j in range(1, len(ell)):
                matrix[j, j - 1] = matrix[j - 1, j] = 1j * value * a[ell[j]]
            out[index] = np.linalg.inv(matrix)
        return out, scatter

    for m in range(4):
        free, scatter = dense(m, False)
        full, _ = dense(m, True)
        reference_first = np.einsum("kab,b,kbc->kac", free, scatter, free)
        reference_rest = np.einsum("kab,b,kbc->kac", full, scatter, reference_first)
        for lam in range(m, 4):
            for got, want in ((first[(m, lam)], reference_first),
                              (multiple[(m, lam)], reference_rest)):
                block = want[:, lam - m, :J + 1 - m]
                assert (np.max(np.abs(got[:, m:J + 1] - block))
                        / np.max(np.abs(block))) < 1e-12


def test_resolvent_rows_are_symmetric_in_the_two_indices():
    """Reciprocity: the operator is complex symmetric, so a row is a column."""
    k = np.geomspace(0.05, 3.0, 6)
    first, multiple = resolvent_rows(k, 0.07, medium(), 12, 12, 3)
    for rows in (first, multiple):
        for (m, lam), block in rows.items():
            other = rows[(m, m)] if lam != m else rows[(m, lam)]
            assert block[:, lam].shape == (len(k),)
            np.testing.assert_allclose(block[:, m], other[:, lam], rtol=1e-11,
                                       atol=0)


def test_solve_tail_system_m_rejects_inconsistent_shapes():
    k = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="Incompatible"):
        solve_tail_system_m(k, 1.0 + 0j, np.zeros(2), np.zeros(4),
                            np.zeros((2, 3), complex), 1)


def test_dense_block_reference_is_symmetric():
    k = np.array([0.4])
    block = dense_block_reference(k, 0.0, medium(), 6, 10, 2)[0]
    np.testing.assert_allclose(block, block.T, rtol=1e-12, atol=0)


def test_the_block_depends_on_m_only_through_its_square():
    """G^(m) = G^(-m) is derived from a_lm, and here it is measured."""
    k = np.geomspace(0.05, 4.0, 7)
    d0 = medium().extinction_per_m - 1j * 0.09 / medium().speed_m_per_ns
    degree, m = 18, 3
    tail = free_tail_ratios_m(k, d0, degree, m)[:, m, degree]
    gamma = medium().scattering_per_m * medium().g ** np.arange(m, degree + 1)
    rhs = np.zeros((len(k), degree + 1 - m), complex)
    rhs[:, 0] = 1.0
    positive = solve_tail_system_m(k, d0, tail, gamma, rhs, m)
    negative = solve_tail_system_m(k, d0, tail, gamma, rhs, -m)
    np.testing.assert_array_equal(positive, negative)
