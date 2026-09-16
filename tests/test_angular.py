import numpy as np
import pytest
from dataclasses import replace
from scipy.special import roots_legendre, eval_legendre
from lighthit import synthetic_medium
from lighthit.angular import free_moments_and_tail, angular_components, dense_finite_rank_reference


@pytest.mark.parametrize("omega", [0.0, 0.07, 0.7])
def test_exact_tail_against_dense_resolvent(omega):
    medium = synthetic_medium()
    k = np.array([0.0, 1e-7, 0.03, 0.2, 1.0])
    parts = angular_components(k, omega, medium, 8, 18)
    for index, ki in enumerate(k):
        reference = dense_finite_rank_reference(ki, omega, medium, 8, 18, 1500)
        got = sum(p[index] for p in parts)
        np.testing.assert_allclose(got, reference, atol=4e-10, rtol=3e-10)


@pytest.mark.parametrize("g", [0.0, 0.7, -0.4, 0.9])
def test_zero_k_total_number(g):
    m = replace(synthetic_medium(), g=g)
    for omega in [0.0, 0.1]:
        parts = angular_components([0.0], omega, m, 16, 24)
        full = sum(p[0] for p in parts)
        expected = np.sqrt(2) / (m.absorption_per_m - 1j * omega / m.speed_m_per_ns)
        np.testing.assert_allclose(full[0], expected, rtol=5e-14)
        np.testing.assert_allclose(full[1:], 0, atol=1e-14)


@pytest.mark.parametrize("omega", [0.0, 0.13, 1.0])
def test_isotropic_scattering_all_orders_closed_form(omega):
    m = replace(synthetic_medium(), g=0.0)
    k = np.r_[0.0, np.geomspace(1e-6, 8, 45)]
    b, one, multi = angular_components(k, omega, m, 0, 40)
    d0 = m.extinction_per_m - 1j * omega / m.speed_m_per_ns
    A = np.empty_like(k, complex)
    A[0] = 1 / d0
    A[1:] = np.arctan(k[1:] / d0) / k[1:]
    np.testing.assert_allclose(one, m.scattering_per_m * A[:, None] * b, atol=1e-12, rtol=1e-10)
    fac = (m.scattering_per_m * A)**2 / (1 - m.scattering_per_m * A)
    np.testing.assert_allclose(multi, fac[:, None] * b, atol=1e-12, rtol=1e-10)


def test_retained_moments_do_not_depend_on_where_free_tail_is_eliminated():
    m = synthetic_medium()
    k = np.geomspace(1e-5, 6, 60)
    p20 = angular_components(k, 0.12, m, 12, 20)
    p90 = angular_components(k, 0.12, m, 12, 90)
    for a, b in zip(p20, p90):
        np.testing.assert_allclose(a, b[:, :21], rtol=3e-10, atol=2e-12)


def test_free_negative_frequency_conjugation():
    m = synthetic_medium()
    k = np.array([0, 0.03, 0.2, 1.0])
    pos = angular_components(k, 0.2, m, 10, 24)
    neg = angular_components(k, -0.2, m, 10, 24)
    parity = (-1.0)**np.arange(25)
    for a, b in zip(pos, neg):
        np.testing.assert_allclose(b, a.conj() * parity, rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize("k,d0,degree", [([-1], 1, 2), ([1], 0, 2), ([1], 1, -1), ([1], 1, True)])
def test_invalid_free_input(k, d0, degree):
    with pytest.raises(ValueError):
        free_moments_and_tail(k, d0, degree)
