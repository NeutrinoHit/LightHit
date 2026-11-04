import numpy as np
import pytest
from scipy.special import roots_legendre
from lighthit import synthetic_medium, Medium
from lighthit.experimental.stationary_modes import StationaryModes
from lighthit.experimental.plane_parallel import (PlaneParallelModes, single_scattered_radiance,
                                                  henyey_greenstein)

Z = np.array([0.5, 3., 10., 40., 128., 400., 1000.])


@pytest.fixture(scope="module")
def modes():
    return PlaneParallelModes.build(synthetic_medium(), 255)


def test_leading_mode_is_the_point_source_mode(modes):
    s = StationaryModes.build(modes.medium, 255)
    assert modes.leading_attenuation_per_m == s.leading_attenuation_per_m
    assert modes.asymptotic_mean_cosine == pytest.approx(s.leading_mean_cosine, rel=1e-13)
    assert modes.leading_attenuation_per_m == pytest.approx(.0780239173344, rel=2e-11)


def test_degree_convergence(modes):
    a = PlaneParallelModes.build(modes.medium, 127).irradiance(Z)
    b = modes.irradiance(Z)
    for x, y in ((a.downward, b.downward), (a.upward, b.upward), (a.scalar, b.scalar)):
        np.testing.assert_allclose(x[1:], y[1:], rtol=2e-7)
        # within a metre of the entry plane the upward light converges slowest
        np.testing.assert_allclose(x[0], y[0], rtol=1e-5)


def test_energy_balance_analytic_and_quadrature(modes):
    m = modes.medium
    total, above = modes.integrated_scalar()
    assert m.absorption_per_m * total == pytest.approx(1, abs=1e-13)
    assert 0 < above < 1e-2 * total
    # independent: Gauss panels of the evaluated E_0 over z in [-400, 4000]
    x, w = roots_legendre(24)
    edges = np.concatenate([-np.geomspace(400, 1e-3, 40), [0], np.geomspace(1e-3, 4000, 80)])
    a, b = edges[:-1, None], edges[1:, None]
    zz = ((b - a) / 2 * x + (b + a) / 2).ravel()
    ww = ((b - a) / 2 * w).ravel()
    e0 = modes.irradiance(zz).scalar
    assert m.absorption_per_m * (ww @ e0) == pytest.approx(1, rel=1e-10)
    assert ww[zz < 0] @ e0[zz < 0] == pytest.approx(above, rel=1e-9)


def test_local_balance(modes):
    m, h = modes.medium, 1e-3
    z = np.array([2., 30., 150., 700.])
    net = lambda x: modes.irradiance(x).net
    div = (net(z + h) - net(z - h)) / (2 * h)
    np.testing.assert_allclose(div, -m.absorption_per_m * modes.irradiance(z).scalar, rtol=1e-7)


def test_diffuse_attenuation_matches_finite_difference(modes):
    z, h = np.array([1., 20., 200., 900.]), 1e-3
    ed = lambda x: np.log(modes.irradiance(x).downward)
    np.testing.assert_allclose(modes.diffuse_attenuation(z), -(ed(z + h) - ed(z - h)) / (2 * h), rtol=1e-7)
    assert modes.diffuse_attenuation([1000.])[0] == pytest.approx(modes.leading_attenuation_per_m, rel=1e-4)


def test_deep_ratio_and_asymptotic_radiance(modes):
    e = modes.irradiance([900., 1000.]).downward
    # the continuum near 1/mu_t still adds ~1e-4 at 900-1000 m
    assert e[1] / e[0] == pytest.approx(np.exp(-100 * modes.leading_attenuation_per_m), rel=1e-3)
    mu = np.linspace(-1, 1, 41)
    dev = []
    for z in (200., 500., 1000.):
        shape = modes.radiance([z], mu)[0] / modes.irradiance([z]).scalar_scattered[0]
        dev.append(np.max(np.abs(shape / modes.asymptotic_radiance(mu) - 1)))
    assert dev[2] < 3e-3 and dev[2] < dev[1] < dev[0]


def test_slowest_mode_solves_the_transport_equation(modes):
    m, k = modes.medium, modes.leading_index
    psi = modes.vectors[:, k]
    ell = np.arange(modes.degree + 1)
    mu = np.linspace(-1, 1, 21)
    from lighthit.experimental.plane_parallel import _legendre
    p = _legendre(modes.degree, mu)
    source = m.scattering_per_m * ((2 * ell + 1) / (4 * np.pi) * m.g ** ell * psi) @ p
    exact = source / (m.extinction_per_m - mu / modes.lengths_m[k])
    np.testing.assert_allclose(modes.angular_basis(mu)[:, k], exact, rtol=1e-9)


def test_weak_scattering_is_single_scattering():
    m = Medium(.04, 1e-7, .9, 1.35)
    modes = PlaneParallelModes.build(m, 511)
    mu = np.concatenate([np.linspace(-1, -.1, 19), np.linspace(.1, .99, 20)])
    z = np.array([2., 20., 60.])
    # second order is ~1e-5 here; near the plane P_N converges slowest in angle
    np.testing.assert_allclose(modes.radiance(z, mu), single_scattered_radiance(z, mu, m), rtol=1e-4)
    mu = np.linspace(-1, -.3, 8)
    np.testing.assert_allclose(modes.radiance([-5.], mu), single_scattered_radiance([-5.], mu, m), rtol=2e-5)


def test_no_scattering_has_no_scattered_light():
    m = Medium(.04, 0., .9, 1.35)
    e = PlaneParallelModes.build(m, 63).irradiance(Z)
    assert np.all(e.scalar_scattered == 0) and np.all(e.downward_scattered == 0)
    np.testing.assert_allclose(e.downward, np.exp(-.04 * Z), rtol=1e-15)


def test_irradiance_is_angular_integral(modes):
    x, w = roots_legendre(400)
    z = np.array([5., 80.])
    rad = modes.radiance(z, x)
    e = modes.irradiance(z)
    np.testing.assert_allclose(2 * np.pi * rad @ w, e.scalar_scattered, rtol=1e-11)
    h, v = (x + 1) / 2, w / 2   # half-range Gauss rule
    np.testing.assert_allclose(2 * np.pi * modes.radiance(z, h) @ (v * h), e.downward_scattered, rtol=1e-11)
    np.testing.assert_allclose(2 * np.pi * modes.radiance(z, -h) @ (v * h), e.upward_scattered, rtol=1e-11)
    assert henyey_greenstein(x, .9) @ w * 2 * np.pi == pytest.approx(1, rel=1e-10)
