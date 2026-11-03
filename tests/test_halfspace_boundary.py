"""Planar-interface conservation, refraction, aperture and reciprocal weights."""
import numpy as np
import pytest
from scipy.integrate import quad

from lighthit.experimental.halfspace_boundary import HalfSpaceBoundary, ExternalAperture
from lighthit.medium import Medium


@pytest.mark.parametrize("incidence", ["inside", "outside"])
def test_matched_interface_is_exact_at_all_angles(incidence):
    mu = np.linspace(0., 1., 501)
    c = HalfSpaceBoundary(n=1.).coefficients(mu, incidence=incidence)
    np.testing.assert_array_equal(c.reflection, np.zeros_like(mu))
    np.testing.assert_array_equal(c.transmission, np.ones_like(mu))
    np.testing.assert_array_equal(c.transmitted_cosine, mu)
    np.testing.assert_array_equal(c.radiance_factor, np.ones_like(mu))
    assert not c.total_internal_reflection.any()


@pytest.mark.parametrize("n", [.7, 1., 1.4, 2.4])
@pytest.mark.parametrize("incidence", ["inside", "outside"])
def test_normal_incidence_and_flux_conservation(n, incidence):
    mu = np.linspace(0., 1., 1001)
    c = HalfSpaceBoundary(n).coefficients(mu, incidence=incidence)
    np.testing.assert_allclose(c.reflection + c.transmission, 1., atol=5e-16, rtol=0)
    assert np.all((c.reflection >= 0) & (c.reflection <= 1))
    assert np.all((c.transmission >= 0) & (c.transmission <= 1))
    assert c.reflection[-1] == pytest.approx(((n - 1) / (n + 1))**2)
    assert c.transmission[-1] == pytest.approx(4 * n / (n + 1)**2)


@pytest.mark.parametrize("n", [.7, 1.4, 2.4])
def test_fresnel_matches_independent_angle_formulas(n):
    # Angle formulas use sine and tangent ratios, independently of the
    # cosine/impedance implementation. Points avoid the removable zeros.
    theta_i = np.linspace(.02, min(1.3, np.arcsin(min(1., 1 / n)) - .02), 75)
    theta_t = np.arcsin(n * np.sin(theta_i))
    rs = (np.sin(theta_i - theta_t) / np.sin(theta_i + theta_t))**2
    rp = (np.tan(theta_i - theta_t) / np.tan(theta_i + theta_t))**2
    c = HalfSpaceBoundary(n).coefficients(np.cos(theta_i))
    np.testing.assert_allclose(c.reflection, (rs + rp) / 2, rtol=2e-12, atol=2e-15)
    np.testing.assert_allclose(c.transmitted_cosine, np.cos(theta_t), atol=2e-15)


def test_tissue_air_critical_angle_and_normal_entry_fraction():
    n = 1.4
    critical_cosine = np.sqrt(1 - 1 / n**2)
    mu = np.array([0., critical_cosine - 1e-8, critical_cosine + 1e-8, 1.])
    c = HalfSpaceBoundary(n).coefficients(mu)
    np.testing.assert_array_equal(c.total_internal_reflection, [True, True, False, False])
    np.testing.assert_array_equal(c.reflection[:2], [1., 1.])
    np.testing.assert_array_equal(c.transmission[:2], [0., 0.])
    assert c.reflection[-1] == pytest.approx(1 / 36)
    assert c.transmission[-1] == pytest.approx(35 / 36)


@pytest.mark.parametrize("n", [.7, 1.4, 2.4])
def test_refraction_and_radiance_are_reciprocal(n):
    mu = np.linspace(.02, 1., 333)
    boundary = HalfSpaceBoundary(n)
    forward = boundary.coefficients(mu)
    valid = ~forward.total_internal_reflection
    reverse = boundary.coefficients(forward.transmitted_cosine[valid], incidence="outside")
    np.testing.assert_allclose(reverse.transmitted_cosine, mu[valid], atol=2e-14)
    np.testing.assert_allclose(reverse.reflection, forward.reflection[valid], atol=2e-14)
    np.testing.assert_allclose(forward.radiance_factor[valid] * reverse.radiance_factor,
                               forward.transmission[valid]**2, atol=2e-14)


@pytest.mark.parametrize("n", [.7, 1., 1.4])
def test_physical_and_reduced_radiance_boundary_agree(n):
    boundary = HalfSpaceBoundary(n)
    mu = np.linspace(.01, 1., 80)
    inside_index, outside_index = 1.7, 1.7 / n
    outward = np.exp(2j * mu)
    external = np.exp(-3j * mu)
    physical = boundary.incoming_radiance(mu, inside_index**2 * outward,
                                          external_radiance=outside_index**2 * external)
    reduced = boundary.incoming_reduced_radiance(mu, outward, external=external)
    np.testing.assert_allclose(physical, inside_index**2 * reduced, atol=3e-15)


def test_equilibrium_radiance_and_group_index_are_independent():
    medium = Medium(.2, 2., .8, group_index=1.47)
    speed = medium.speed_m_per_ns
    mu = np.linspace(0., 1., 101)
    for n in (1., 1.4):
        boundary = HalfSpaceBoundary(n)
        np.testing.assert_allclose(boundary.incoming_radiance(mu, n**2, external_radiance=1.),
                                   n**2, atol=1e-15)
    assert medium.speed_m_per_ns == speed


@pytest.mark.parametrize("n", [1., 1.4])
def test_aperture_uses_external_angle_and_reciprocal_detector_weight(n):
    angle = .3
    aperture = ExternalAperture(angle)
    boundary = HalfSpaceBoundary(n)
    # Construct angles on either side of the exterior aperture edge.
    external_angles = np.array([angle / 2, angle * 1.5])
    mu = np.sqrt(1 - (np.sin(external_angles) / n)**2)
    c = boundary.coefficients(mu)
    weight = aperture.internal_weight(boundary, mu)
    assert weight[0] == pytest.approx(c.transmission[0])
    assert weight[1] == 0
    np.testing.assert_allclose(aperture.escaped_flux_density(boundary, mu, 2 + 3j),
                               mu * weight * (2 + 3j))
    np.testing.assert_allclose(boundary.incoming_reduced_radiance(mu, 0., external=[1., 0.]),
                               weight)


def test_hemispherical_escape_obeys_etendue_reciprocity():
    n = 1.4
    boundary = HalfSpaceBoundary(n)
    aperture = ExternalAperture()
    mu_critical = np.sqrt(1 - 1 / n**2)
    internal = quad(lambda mu: float(aperture.escaped_flux_density(boundary, mu, 1.)),
                    mu_critical, 1., epsabs=1e-12, epsrel=1e-11)[0]
    external = quad(lambda mu: mu * float(boundary.coefficients(mu, incidence="outside").transmission),
                    0., 1., epsabs=1e-12, epsrel=1e-11)[0]
    assert n**2 * internal == pytest.approx(external, rel=2e-11)


@pytest.mark.parametrize("n", [1., 1.4])
@pytest.mark.parametrize("angle", [.4, np.pi / 2])
def test_external_radiance_integral_equals_collected_internal_flux(n, angle):
    boundary = HalfSpaceBoundary(n)
    aperture = ExternalAperture(angle)
    mu_min = np.sqrt(1 - (np.sin(angle) / n)**2)

    def internal_radiance(mu):
        return 1 + .3 * mu**2

    internal = quad(lambda mu: float(aperture.escaped_flux_density(
        boundary, mu, internal_radiance(mu))), mu_min, 1., epsabs=1e-12)[0]

    def external_flux(mu_out):
        # Integrate in the *other* hemisphere, with its own solid angle.
        mu_in = np.sqrt(1 - (1 - mu_out**2) / n**2)
        c = boundary.coefficients(mu_in)
        return mu_out * float(c.radiance_factor) * internal_radiance(mu_in)

    external = quad(external_flux, np.cos(angle), 1., epsabs=1e-12)[0]
    assert external == pytest.approx(internal, rel=2e-10)


@pytest.mark.parametrize("n", [0., -1., np.nan, np.inf])
def test_invalid_index_is_rejected(n):
    with pytest.raises(ValueError):
        HalfSpaceBoundary(n)


@pytest.mark.parametrize("mu", [-.1, 1.01, np.nan, np.inf])
def test_invalid_cosine_is_rejected(mu):
    with pytest.raises(ValueError):
        HalfSpaceBoundary(1.4).coefficients(mu)


@pytest.mark.parametrize("angle", [-.1, np.pi, np.nan, np.inf])
def test_invalid_aperture_is_rejected(angle):
    with pytest.raises(ValueError):
        ExternalAperture(angle)
