"""Half-space conservation, exact first collision, reciprocity and adjoints."""
from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import quad_vec
from scipy.special import roots_legendre
from scipy.special import ndtr

from lighthit.medium import Medium
from lighthit.experimental.halfspace import HalfSpaceSolver, HalfSpaceSettings, SurfacePort
from lighthit.experimental.halfspace_boundary import HalfSpaceBoundary


SETTINGS = HalfSpaceSettings(4, 8, 6)


def solver(n=1., *, medium=None, settings=SETTINGS):
    return HalfSpaceSolver(Medium(.2, 1., .3, 1.4) if medium is None else medium,
                           HalfSpaceBoundary(n), settings)


@pytest.mark.parametrize("n", [1., 1.4])
@pytest.mark.parametrize("g", [0., .3])
def test_photon_balance_and_physical_boundary(n, g):
    k = solver(n, medium=Medium(.2, 1., g, 1.4))
    a = k.solve()
    balance = a.photon_balance()
    assert abs(balance["residual"]) < 2e-12
    assert balance["specular"] + balance["absorbed"] + balance["escaped"] == pytest.approx(1., abs=2e-12)
    assert 0 < balance["escaped"] < balance["entered"] < 1.00000001
    f = a.field([0.])[0]
    np.testing.assert_allclose(f[:k.half], k.reflection*f[k.half:], atol=2e-13)


@pytest.mark.parametrize("n", [1., 1.4])
def test_no_scattering_has_only_the_directed_interior_beam(n):
    k = solver(n, medium=Medium(.2, 0., .9, 1.47))
    a = k.solve()
    z = np.array([0., .3, 2.])
    np.testing.assert_allclose(a.beam(z)[:, 0], a.entered_fraction*np.exp(-.2*z))
    np.testing.assert_array_equal(a.field(z), np.zeros((3, len(k.mu))))
    np.testing.assert_array_equal(a.response(SurfacePort.aperture()), np.zeros(3))
    assert a.photon_balance()["absorbed"] == pytest.approx(a.entered_fraction)


def test_exact_isotropic_first_collision_of_a_normal_beam():
    medium = Medium(.2, 1., 0., 1.4)
    k = solver(medium=medium)
    d = SurfacePort.aperture(polar_nodes=24, azimuth_nodes=4)
    first = k.response(SurfacePort.pencil(), d)[1]
    expected = medium.scattering_per_m/(2*medium.extinction_per_m)*(1-np.log(2))
    assert first == pytest.approx(expected, abs=3e-15)


@pytest.mark.parametrize("n", [1., 1.4])
def test_complex_first_collision_matches_continuous_depth_integral(n):
    k = solver(n, medium=Medium(.2, 1., 0., 1.4))
    s = SurfacePort.pencil([np.sin(.4), 0., np.cos(.4)])
    d = SurfacePort.pencil([0., np.sin(.6), np.cos(.6)])
    q, omega = np.array([.8, .3]), .1
    us, _, ts = s.internal_rays(k.boundary)
    ud, _, td = d.internal_rays(k.boundary)
    incident, outgoing = us[0], -ud[0]
    base = k.medium.extinction_per_m-1j*omega/k.medium.speed_m_per_ns
    rate_s = (base+1j*q@incident[:2])/incident[2]
    rate_d = (base+1j*q@outgoing[:2])/(-outgoing[2])
    integral = quad_vec(lambda z: np.exp(-(rate_s+rate_d)*z), 0., np.inf,
                        epsabs=1e-13, epsrel=1e-12)[0]
    expected = (k.medium.scattering_per_m/(4*np.pi)*ts[0]*td[0]
                / (n*n*incident[2]*(-outgoing[2]))*integral)
    actual = k.response(s, d, q_per_m=q, omega_per_ns=omega)[1]
    assert actual == pytest.approx(expected, abs=2e-14)


@pytest.mark.parametrize("n", [1., 1.4])
@pytest.mark.parametrize("q,omega", [((0., 0.), 0.), ((.8, .3), .1)])
def test_weighted_source_detector_reciprocity_at_each_order(n, q, omega):
    k = solver(n)
    s = SurfacePort.pencil([np.sin(.2), 0., np.cos(.2)])
    d = SurfacePort.aperture(.7, polar_nodes=3, azimuth_nodes=8)
    forward = k.response(s, d, q_per_m=q, omega_per_ns=omega)/d.collection_factor
    reverse = k.response(d, s, q_per_m=-np.array(q), omega_per_ns=omega)/s.collection_factor
    np.testing.assert_allclose(forward, reverse, rtol=2e-11, atol=2e-13)


def test_real_medium_transform_conjugation_and_resolved_modes():
    # More than 128 ordinates exercises the generalized transport pencil.
    k = solver(settings=HalfSpaceSettings(6, 12, 8))
    s, d = SurfacePort.pencil(), SurfacePort.aperture(.7, polar_nodes=3, azimuth_nodes=8)
    a = k.solve(s, q_per_m=(.8, .3), omega_per_ns=.1)
    b = k.solve(s, q_per_m=(-.8, -.3), omega_per_ns=-.1)
    assert a.eigen_residual < 1e-11
    np.testing.assert_allclose(a.response(d).conj(), b.response(d), atol=2e-13)


def test_arbitrary_direction_readout_and_separated_first_field():
    k = solver(1.4)
    a = k.solve(q_per_m=(.8, .3), omega_per_ns=.1)
    z = np.array([0., .2, 1.])
    directions = k.direction[k.half:]
    np.testing.assert_allclose(a.upward_field(directions, z), a.field(z)[:, k.half:], atol=3e-13)
    np.testing.assert_allclose(a.upward_field(directions, z, first_only=True),
                               a.first_field(z)[:, k.half:], atol=2e-14)
    assert np.max(abs(a.field([300.]))) < 1e-30


def test_source_port_linearity():
    k = solver(1.4)
    a = SurfacePort.pencil()
    b = SurfacePort.pencil([np.sin(.4), 0., np.cos(.4)])
    mix = SurfacePort(np.concatenate([a.directions_external, b.directions_external]), np.array([.3, .7]))
    d = SurfacePort.aperture(.7, polar_nodes=3, azimuth_nodes=8)
    expected = .3*k.response(a, d)+.7*k.response(b, d)
    np.testing.assert_allclose(k.response(mix, d), expected, atol=1e-13)


@pytest.mark.parametrize("n", [1., 1.4])
@pytest.mark.parametrize("q,omega", [((0., 0.), 0.), ((.8, .3), .1)])
def test_angular_adjoint_absorption_pairing_matches_finite_difference(n, q, omega):
    k = solver(n)
    s, d = SurfacePort.pencil(), SurfacePort.aperture(.7, polar_nodes=3, azimuth_nodes=8)
    x, w = roots_legendre(120)
    z, weight = (x+1)*30, w*30
    derivative = weight @ k.depth_sensitivity(s, d, z, q_per_m=q, omega_per_ns=omega)
    epsilon = 1e-5
    plus = solver(n, medium=replace(k.medium, absorption_per_m=.2+epsilon))
    minus = solver(n, medium=replace(k.medium, absorption_per_m=.2-epsilon))
    finite = (plus.response(s, d, q_per_m=q, omega_per_ns=omega).sum()
              - minus.response(s, d, q_per_m=q, omega_per_ns=omega).sum())/(2*epsilon)
    assert derivative == pytest.approx(finite, rel=2e-7, abs=1e-9)


def test_banana_reciprocity_preserves_spatial_and_angular_ports():
    k = solver(1.4)
    s, d = SurfacePort.pencil(), SurfacePort.aperture(.7, polar_nodes=2, azimuth_nodes=4)
    point = np.array([[.2, 0., .3], [.5, .1, .7], [.8, 0., .3]])
    a = k.banana(s, d, point, detector_xy_m=(1., 0.), source_sigma_m=.4,
                 detector_sigma_m=.5, q_max_per_m=6., q_nodes=8)
    b = k.banana(d, s, point, source_xy_m=(1., 0.), detector_xy_m=(0., 0.),
                 source_sigma_m=.5, detector_sigma_m=.4, q_max_per_m=6., q_nodes=8)
    np.testing.assert_allclose(a/d.collection_factor, b/s.collection_factor, atol=2e-12)


def test_hankel_inversion_rejects_tilted_ports():
    k = solver()
    s = SurfacePort.pencil([np.sin(.2), 0., np.cos(.2)])
    with pytest.raises(ValueError, match="axisymmetric"):
        k.radial_response(s, SurfacePort.pencil(), [1.], q_max_per_m=3.)


def test_underresolved_negative_phase_is_explicitly_rejected():
    with pytest.raises(ValueError, match="negative"):
        solver(medium=Medium(.2, 1., .9, 1.4))


@pytest.mark.parametrize("n", [1., 1.4])
def test_first_collision_time_bins_match_exact_smoothed_exponential(n):
    k = solver(n, medium=Medium(.2, 1., 0., 1.4), settings=HalfSpaceSettings(2, 4, 0))
    s, d = SurfacePort.pencil(), SurfacePort.aperture(.7, polar_nodes=6, azimuth_nodes=4)
    sigma = .5
    edges = np.linspace(-3., 35., 20)
    actual = k.time_bins(s, d, np.linspace(0., 12., 241), edges, sigma_ns=sigma)[:, 1]
    rate = k.medium.extinction_per_m*k.medium.speed_m_per_ns
    cdf = ndtr(edges/sigma)-np.exp(-rate*edges+.5*(rate*sigma)**2)*ndtr(edges/sigma-rate*sigma)
    expected = k.response(s, d)[1].real*np.diff(cdf)
    np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=2e-9)


@pytest.mark.parametrize("n", [1., 1.4])
@pytest.mark.parametrize("omega", [0., .1])
def test_analytic_azimuthal_mean_matches_full_q_zero_problem(n, omega):
    medium = Medium(.2, 1., .3, 1.4)
    full = solver(n, medium=medium)
    axial = solver(n, medium=medium, settings=HalfSpaceSettings(4, 1, 6, plane_parallel=True))
    s, d = SurfacePort.pencil(), SurfacePort.aperture(.7, polar_nodes=4, azimuth_nodes=8)
    np.testing.assert_allclose(axial.response(s, d, omega_per_ns=omega),
                               full.response(s, d, omega_per_ns=omega), atol=3e-13)
    z = np.array([0., .3, 2.])
    full_field = full.solve(s, omega_per_ns=omega).field(z).reshape(3, -1, 8).mean(axis=2)
    np.testing.assert_allclose(axial.solve(s, omega_per_ns=omega).field(z), full_field, atol=3e-13)


def test_plane_parallel_high_anisotropy_convergence_and_photon_balance():
    medium = Medium(.01, 25., .9, 1.35)
    s, d = SurfacePort.pencil(), SurfacePort.aperture(polar_nodes=32, azimuth_nodes=4)
    values = []
    for polar, degree in ((64, 96), (80, 128)):
        k = solver(1.35, medium=medium, settings=HalfSpaceSettings(polar, 1, degree, plane_parallel=True))
        a = k.solve(s)
        assert abs(a.photon_balance()["residual"]) < 2e-11
        assert k.phase_minimum > 0
        assert len(k.mu) <= 320
        values.append(a.response(d))
        reverse = k.response(d, s)
        np.testing.assert_allclose(a.response(d)/d.collection_factor, reverse, atol=2e-12)
    assert abs(values[1].sum()-values[0].sum())/abs(values[1].sum()) < 3e-5


def test_plane_parallel_rejects_spatial_or_nonaxial_queries():
    k = solver(settings=HalfSpaceSettings(4, 1, 6, plane_parallel=True))
    s = SurfacePort.pencil()
    tilted = SurfacePort.pencil([np.sin(.2), 0., np.cos(.2)])
    with pytest.raises(ValueError, match="q=0"):
        k.solve(s, q_per_m=(1., 0.))
    with pytest.raises(ValueError, match="axisymmetric"):
        k.solve(tilted)
    with pytest.raises(ValueError, match="axisymmetric"):
        k.response(s, tilted)
    with pytest.raises(ValueError, match="spatial inversion"):
        k.radial_response(s, s, [1.], q_max_per_m=3.)


def harmonic_pair(n, polar=4, azimuth=12, degree=6, medium=None):
    """Dense ordinates and the cosine-moment basis with no harmonic truncation."""
    medium = Medium(.2, 1., .3, 1.4) if medium is None else medium
    dense = solver(n, medium=medium, settings=HalfSpaceSettings(polar, azimuth, degree))
    moments = solver(n, medium=medium, settings=HalfSpaceSettings(
        polar, azimuth, degree, azimuth_harmonics=azimuth//2-1))
    return dense, moments


def test_normalized_associated_legendre_matches_reference_functions():
    from math import factorial
    from scipy.special import lpmv
    from lighthit.experimental.halfspace import _normalized_legendre
    x = np.linspace(-1., 1., 9)
    table = _normalized_legendre(x, 9, 4)
    for m in range(5):
        for ell in range(10):
            if ell < m:
                assert not table[m, ell].any()
                continue
            scale = np.sqrt((2*ell+1)/(4*np.pi)*factorial(ell-m)/factorial(ell+m))
            np.testing.assert_allclose(abs(table[m, ell]), abs(scale*lpmv(m, ell, x)), atol=2e-15)


@pytest.mark.parametrize("n", [1., 1.4])
@pytest.mark.parametrize("q,omega", [((0., 0.), 0.), ((.8, 0.), 0.), ((-2., 0.), .1)])
def test_cosine_moments_equal_even_discrete_ordinates(n, q, omega):
    # phase_degree <= azimuth_nodes/2: the uniform azimuth grid has no aliasing,
    # and its even subspace is exactly spanned by the retained moments.
    dense, moments = harmonic_pair(n, 6, 16, 8)
    assert n == 1. or len(moments.mu) > 128
    s = SurfacePort.pencil([np.sin(.3), 0., np.cos(.3)])
    d = SurfacePort.aperture(.7, polar_nodes=3, azimuth_nodes=8)
    for source in (s, d):
        a = moments.solve(source, q_per_m=q, omega_per_ns=omega)
        assert a.eigen_residual < 1e-11
        np.testing.assert_allclose(a.response(d), dense.response(source, d, q_per_m=q, omega_per_ns=omega),
                                   rtol=2e-12, atol=3e-15)


def test_cosine_moment_banana_equals_discrete_ordinates_with_orders():
    dense, moments = harmonic_pair(1.4)
    s, d = SurfacePort.pencil(), SurfacePort.aperture(.7, polar_nodes=2, azimuth_nodes=4)
    point = np.array([[.2, 0., .3], [.5, .1, .7], [.8, -.2, .3], [0., 0., .5]])
    keywords = dict(detector_xy_m=(1., 0.), source_sigma_m=.4, detector_sigma_m=.5, q_max_per_m=6., q_nodes=8)
    expected = dense.banana(s, d, point, **keywords)
    np.testing.assert_allclose(moments.banana(s, d, point, **keywords), expected, rtol=1e-11, atol=1e-15)
    for k in (dense, moments):
        orders = k.banana(s, d, point, by_order=True, **keywords)
        assert orders.shape == (len(point), 3) and not orders[:, 0].any()
        np.testing.assert_allclose(orders.sum(axis=1), expected, rtol=1e-11, atol=1e-15)
    reverse = moments.banana(d, s, point, source_xy_m=(1., 0.), detector_xy_m=(0., 0.),
                             source_sigma_m=.5, detector_sigma_m=.4, q_max_per_m=6., q_nodes=8, by_order=True)
    np.testing.assert_allclose(moments.banana(s, d, point, by_order=True, **keywords)/d.collection_factor,
                               reverse/s.collection_factor, atol=2e-12)


def test_first_order_banana_matches_once_scattered_absorption_derivative():
    # Only order-1 paths survive mu_s -> 0 at first order in mu_s.
    medium = Medium(.2, 1e-8, .3, 1.4)
    k = solver(1., medium=medium, settings=HalfSpaceSettings(4, 12, 6, azimuth_harmonics=5))
    s, d = SurfacePort.pencil(), SurfacePort.aperture(.7, polar_nodes=3, azimuth_nodes=8)
    x, w = roots_legendre(40)
    z, weight = (x+1)*20, w*20
    derivative = weight @ k.depth_sensitivity(s, d, z)
    epsilon = 1e-5
    plus = solver(1., medium=replace(medium, absorption_per_m=.2+epsilon), settings=k.settings)
    minus = solver(1., medium=replace(medium, absorption_per_m=.2-epsilon), settings=k.settings)
    finite = (plus.response(s, d)[1]-minus.response(s, d)[1])/(2*epsilon)
    assert derivative == pytest.approx(finite, rel=1e-6)
    point = np.array([[.3, 0., .4], [.9, .2, .8]])
    orders = k.banana(s, d, point, detector_xy_m=(1., 0.), source_sigma_m=.4, detector_sigma_m=.4,
                      q_max_per_m=6., q_nodes=8, by_order=True)
    assert np.all(abs(orders[:, 2]) < 1e-6*abs(orders[:, 1]))


@pytest.mark.parametrize("n", [1., 1.4])
def test_cosine_moment_balance_and_azimuthal_mean_limit(n):
    k = solver(n, settings=HalfSpaceSettings(4, 4, 6, azimuth_harmonics=0))
    axial = solver(n, settings=HalfSpaceSettings(4, 1, 6, plane_parallel=True))
    s, d = SurfacePort.pencil(), SurfacePort.aperture(.7, polar_nodes=4, azimuth_nodes=8)
    a = k.solve(s)
    assert abs(a.photon_balance()["residual"]) < 2e-12
    np.testing.assert_allclose(a.response(d), axial.response(s, d), atol=3e-14)
    # Moment coefficients use c_0 = 1/sqrt(2 pi); the axial field is the azimuthal mean.
    np.testing.assert_allclose(a.first_field([0., .4]), np.sqrt(2*np.pi)*axial.solve(s).first_field([0., .4]),
                               atol=3e-14)
    np.testing.assert_allclose(a.field([0., .4]), np.sqrt(2*np.pi)*axial.solve(s).field([0., .4]), atol=3e-14)


@pytest.mark.parametrize("q,omega", [((.8, .3), .1), ((0., .5), 0.)])
def test_cosine_moment_adjoint_pairing_and_rotated_wavevector(q, omega):
    k = solver(1.4, settings=HalfSpaceSettings(4, 12, 6, azimuth_harmonics=5))
    s = SurfacePort.pencil()
    x, w = roots_legendre(120)
    z, weight = (x+1)*30, w*30
    derivative = weight @ k.depth_sensitivity(s, s, z, q_per_m=q, omega_per_ns=omega)
    epsilon = 1e-5
    plus = solver(1.4, medium=replace(k.medium, absorption_per_m=.2+epsilon), settings=k.settings)
    minus = solver(1.4, medium=replace(k.medium, absorption_per_m=.2-epsilon), settings=k.settings)
    finite = (plus.response(s, s, q_per_m=q, omega_per_ns=omega).sum()
              - minus.response(s, s, q_per_m=q, omega_per_ns=omega).sum())/(2*epsilon)
    assert derivative == pytest.approx(finite, rel=2e-7, abs=1e-9)
    magnitude = float(np.hypot(*q))
    np.testing.assert_allclose(k.response(s, s, q_per_m=q, omega_per_ns=omega),
                               k.response(s, s, q_per_m=(magnitude, 0.), omega_per_ns=omega), atol=3e-14)


def test_cosine_moments_reject_ports_without_mirror_symmetry_and_bad_settings():
    k = solver(settings=HalfSpaceSettings(4, 8, 6, azimuth_harmonics=3))
    skew = SurfacePort.pencil([0., np.sin(.3), np.cos(.3)])
    k.solve(skew, q_per_m=(0., 1.))
    for q in ((1., 0.), (0., 0.)):  # the moment axis is x at q=0
        with pytest.raises(ValueError, match="mirror-symmetric"):
            k.solve(skew, q_per_m=q)
    with pytest.raises(ValueError, match="q=0"):
        k.solve(q_per_m=(1., 0.)).first_field([0.])
    for bad in (dict(azimuth_harmonics=4), dict(azimuth_harmonics=-1), dict(azimuth_harmonics=3, plane_parallel=True)):
        with pytest.raises(ValueError):
            HalfSpaceSettings(4, 8, 6, **bad)
    with pytest.raises(ValueError, match="2\\*polar_nodes"):
        HalfSpaceSettings(4, 32, 8, azimuth_harmonics=3)
    # High anisotropy no longer needs an azimuth grid finer than the phase degree.
    assert len(solver(1.35, medium=Medium(10., 25000., .9, 1.35),
                      settings=HalfSpaceSettings(40, 8, 78, azimuth_harmonics=3)).mu) == 2*80*4


def test_complex_eigensolver_is_accurate_above_128_unknowns():
    from scipy.linalg import eig
    from lighthit.experimental.halfspace import _complex_eig
    rng = np.random.default_rng(5)
    matrix = rng.standard_normal((300, 300)) + 1j*rng.standard_normal((300, 300))
    lam, vectors = _complex_eig(matrix)
    residual = np.linalg.norm(matrix @ vectors - vectors*lam)/np.linalg.norm(matrix)
    assert residual < 1e-13
    reference = eig(matrix, np.eye(300), right=False)
    np.testing.assert_allclose(np.sort_complex(lam), np.sort_complex(reference), rtol=0, atol=1e-11)


@pytest.mark.parametrize("q,omega,banana", [((3., 0.), .3, True), ((0., -1.), 2., False), ((.5, .5), -.7, False)])
def test_cosine_moments_at_frequency_match_generalized_pencil(monkeypatch, q, omega, banana):
    # Forward scattering, Fresnel boundary and more than 128 unknowns: the
    # standard complex problem M^-1 A' must reproduce the generalized QZ solution.
    from scipy.linalg import eig
    import lighthit.experimental.halfspace as module
    medium = Medium(.5, 40., .8, 1.35)
    settings = HalfSpaceSettings(21, 8, 40, azimuth_harmonics=3)
    s, d = SurfacePort.pencil(), SurfacePort.aperture(.7, polar_nodes=4, azimuth_nodes=8)
    point = np.array([[.02, 0., .03], [.05, .01, .06]])

    def evaluate(k):
        solution = k.solve(s, q_per_m=q, omega_per_ns=omega)
        values = [solution.response(d)]
        if banana:
            values.append(k.banana(s, d, point, detector_xy_m=(.05, 0.), source_sigma_m=.02,
                                   detector_sigma_m=.02, q_max_per_m=150., q_nodes=6,
                                   omega_per_ns=omega, by_order=True))
        return solution.eigen_residual, values

    k = solver(1.35, medium=medium, settings=settings)
    assert len(k.mu) > 128
    residual, expected = evaluate(k)
    assert residual < 1e-12
    monkeypatch.setattr(module, "_complex_eig", lambda matrix: eig(matrix, np.eye(len(matrix))))
    _, actual = evaluate(solver(1.35, medium=medium, settings=settings))
    for a, b in zip(expected, actual):
        np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-12*np.abs(b).max())


def test_first_order_banana_at_frequency_matches_once_scattered_derivative():
    medium = Medium(.2, 1e-8, .3, 1.4)
    k = solver(1., medium=medium, settings=HalfSpaceSettings(4, 12, 6, azimuth_harmonics=5))
    s, d = SurfacePort.pencil(), SurfacePort.aperture(.7, polar_nodes=3, azimuth_nodes=8)
    q, omega = (.6, 0.), .4
    x, w = roots_legendre(40)
    z, weight = (x+1)*20, w*20
    derivative = weight @ k.depth_sensitivity(s, d, z, q_per_m=q, omega_per_ns=omega)
    epsilon = 1e-5
    plus = solver(1., medium=replace(medium, absorption_per_m=.2+epsilon), settings=k.settings)
    minus = solver(1., medium=replace(medium, absorption_per_m=.2-epsilon), settings=k.settings)
    finite = (plus.response(s, d, q_per_m=q, omega_per_ns=omega)[1]
              - minus.response(s, d, q_per_m=q, omega_per_ns=omega)[1])/(2*epsilon)
    assert derivative == pytest.approx(finite, rel=1e-6)
    point = np.array([[.3, 0., .4], [.9, .2, .8]])
    orders = k.banana(s, d, point, detector_xy_m=(1., 0.), source_sigma_m=.4, detector_sigma_m=.4,
                      q_max_per_m=6., q_nodes=8, omega_per_ns=omega, by_order=True)
    assert np.all(abs(orders[:, 2]) < 1e-6*abs(orders[:, 1]))
    assert np.all(abs(orders[:, 1].imag) > 1e-3*abs(orders[:, 1].real))
