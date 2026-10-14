"""Roundoff-level controls for the grouped Q1 polynomial shortcuts."""
import math

import numpy as np
import pytest

numba = pytest.importorskip("numba")
from lighthit import _prompt_numba as kernels
from lighthit.prompt import acceptance_monomials


@numba.njit
def _old_add_moments(n, s_buf, r, sin_t, cos_t, jac_sin, mom0, mom2,
                     coef, ipow, jpow, max_power, g, g0, g2, cpow, spow,
                     backward=False):
    """Independent copy of the original monomial path evaluation."""
    rs = r * sin_t
    y_theta = 0.0 if backward else sin_t / (1.0 + cos_t)
    jac = 0.0 if backward else 2.0 / (r * jac_sin)
    for k in range(n):
        if backward:
            shift = s_buf[k] - r * cos_t
            denominator = rs * rs + shift * shift
            c = (rs * rs - shift * shift) / denominator
            s = 2.0 * rs * shift / denominator
            path_jacobian = 2.0 * r / denominator
        else:
            y = (s_buf[k] - r) / rs + y_theta
            inv = 1.0 / (1.0 + y * y)
            c = (1.0 - y * y) * inv
            s = 2.0 * y * inv
            path_jacobian = jac * inv
        cpow[0] = 1.0
        spow[0] = 1.0
        for p in range(1, max_power + 1):
            cpow[p] = cpow[p - 1] * c
            spow[p] = spow[p - 1] * s
        a0 = a2 = 0.0
        for m in range(len(coef)):
            basis = coef[m] * cpow[ipow[m]] * spow[jpow[m]]
            a0 += basis * mom0[m]
            a2 += basis * mom2[m]
        den = 1.0 + g * g - 2.0 * g * c
        value = path_jacobian * (1.0 - g * g) / (4.0 * math.pi) / (den * math.sqrt(den))
        g0[k] += a0 * value
        g2[k] += a2 * value


def _profiles(degree, theta, g):
    rng = np.random.default_rng(371 + degree)
    alpha = rng.normal(size=degree + 1)
    coef, ipow, jpow, power = acceptance_monomials(alpha)
    mom0 = rng.normal(size=len(coef)) * 1e4
    mom2 = rng.normal(size=len(coef)) * 1e4
    r = 38.0
    nodes = r + np.r_[0.0, np.geomspace(1e-11, 3e4, 120)]
    result = []
    for implementation in (_old_add_moments, kernels._batch_add_moments):
        g0 = np.zeros(len(nodes))
        g2 = np.zeros(len(nodes))
        implementation(len(nodes), nodes, r, math.sin(theta), math.cos(theta),
                       math.sin(theta), mom0, mom2, coef, ipow, jpow, power,
                       g, g0, g2, np.empty(power + 1), np.empty(power + 1),
                       theta > math.pi / 2)
        result.append((g0, g2))
    return nodes, result


@pytest.mark.parametrize("degree", [0, 1, 2, 3, 4, 6])
@pytest.mark.parametrize("theta", [1e-10, .02, .8, 1.5707963268, 2.4, math.pi - 1e-12, math.pi])
@pytest.mark.parametrize("g", [0.0, .9, -.45])
def test_group_moments_match_original_monomials(degree, theta, g):
    _, (original, accelerated) = _profiles(degree, theta, g)
    for old, new in zip(original, accelerated):
        # Signed random moments deliberately include cancellation. The
        # absolute tolerance scales with the largest profile value.
        np.testing.assert_allclose(new, old, rtol=2e-13,
                                   atol=2e-14 * np.max(np.abs(old)))


def _old_ifuncs(z):
    if abs(z) >= 1:
        ez = math.exp(z)
        return ((ez - 1) / z, (ez * (z - 1) + 1) / z**2,
                (ez * (z*z - 2*z + 2) - 2) / z**3)
    term = 1.0
    out = np.zeros(3)
    for n in range(24):
        for j in range(3):
            out[j] += term / (n + j + 1)
        term *= z / (n + 1)
    return out


@pytest.mark.parametrize("z", np.r_[np.linspace(-.999999, .999999, 41),
                                  -100., -40., -2., -1., 0., 1., 2., 40.])
def test_panel_integrals_match_original_and_independent_gauss(z):
    gx, gw = np.polynomial.legendre.leggauss(96)
    x, weights = (gx + 1) / 2, gw / 2
    expected = np.array([np.dot(weights, x**j * np.exp(z*x)) for j in range(3)])
    actual = kernels._ifuncs(float(z), math.exp(z))
    np.testing.assert_allclose(actual, _old_ifuncs(z), rtol=2e-14, atol=2e-16)
    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-16)


@pytest.mark.parametrize("width,kappa", [(1e-5, -.003), (4., -.1), (40., -.03), (70., -1.)])
@pytest.mark.parametrize("coefficients", [(1., 0., 0.), (2., -3., .7), (-10., 8., -2.)])
def test_signed_quadratic_panels_against_independent_gauss(width, kappa, coefficients):
    gx, gw = np.polynomial.legendre.leggauss(96)
    x, weights = width * (gx + 1) / 2, width * gw / 2
    a, b, c = coefficients
    expected = np.dot(weights, np.exp(kappa * x) * (a + x * (b + x * c)))
    actual = kernels._quad_integral(1., a, b, c, width, kappa, math.exp(kappa*width))
    np.testing.assert_allclose(actual, expected, rtol=3e-13, atol=1e-13)


def test_moment_shortcut_preserves_multiple_wavelength_deposition():
    nodes, (original, accelerated) = _profiles(3, .8, .9)
    edges = np.arange(-50., 1500., 4.)
    mus = np.array([.02, .03, .05])
    mut = np.array([.04, .08, .12])
    speed = np.array([.20, .22, .23])
    s0w = np.array([1., .5, .3])
    s2w = np.array([-.54, -.3, -.2])
    kappa = -mut * speed
    eedge = np.exp(np.outer(kappa, edges))
    itab = kernels.interval_table(edges, kappa)
    result = []
    for g0, g2 in (original, accelerated):
        pre = np.zeros((3, 3))
        acc = np.zeros((3, len(edges)-1, 4))
        charge = np.zeros(3)
        bins = np.zeros(len(edges)-1)
        kernels._batch_deposit(len(nodes), nodes, g0, g2, 38., -40., 24.,
                               mus, mut, speed, s0w, s2w, edges, eedge,
                               pre, acc, charge, 4, np.empty(len(nodes)),
                               np.empty(len(nodes)), 5., 0., 0., 0.)
        kernels._finalize(edges, eedge, itab, pre, acc, charge, .05, bins)
        result.append((charge, bins))
    for old, new in zip(*result):
        np.testing.assert_allclose(new, old, rtol=5e-12,
                                   atol=1e-13 * np.max(np.abs(old)))
