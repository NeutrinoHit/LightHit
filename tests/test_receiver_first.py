"""Finite-target tables checked against independent first-flight cap rays."""
import numpy as np
import pytest
from scipy.integrate import quad_vec
from scipy.interpolate import PchipInterpolator

pytest.importorskip('numba')

from lighthit.receiver import SphereSurface
from lighthit.receiver_first import first_entry_profile, MOMENT_POWERS


@pytest.mark.parametrize('b', [1.2, -1.2])
def test_narrow_hg_peak_retains_its_positive_minimum_denominator(b):
    from lighthit._receiver_first_numba import _hg_azimuth
    gap = 1e-20
    actual = _hg_azimuth(gap, b)
    # As gap -> 0, the peak near phi=0 (or pi) has this elementary integral.
    expected = np.full(4, np.sqrt(2)/(gap*np.sqrt(abs(b))))
    if b < 0:
        expected[1::2] *= -1
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, rtol=2e-13)


def _cap_moments(r, theta, g, mu, order=48):
    """Integrate the scattering site and visible cap, not a path profile."""
    rho, z = r*np.sin(theta), r*np.cos(theta)
    hit = z > 0 and rho < 1
    upper = z-np.sqrt(1-rho*rho) if hit else np.inf
    rx, rw = np.polynomial.legendre.leggauss(order)
    q, qw = (rx+1)/2, rw/2
    phi, pw = np.pi*(rx+1), np.pi*rw

    def integrand(ell):
        axial = z-ell
        distance = np.hypot(rho, axial)
        root = np.sqrt(distance**2-1+q*q)
        c = root/distance
        s = np.sqrt(np.maximum(0., 1-c*c))
        dx = c[:, None]*rho/distance-s[:, None]*axial/distance*np.cos(phi)
        dz = c[:, None]*axial/distance+s[:, None]*rho/distance*np.cos(phi)
        phase = (1-g*g)/(4*np.pi*(1+g*g-2*g*dz)**1.5)
        entry = (distance**2-1)/(root+q)
        jac = q/(distance*root)
        weights = (np.exp(-mu*(ell+entry))*qw*jac)[:, None]*pw*phase/np.pi
        return np.array([np.sum(weights*dx**i*dz**j) for i, j in MOMENT_POWERS])

    return quad_vec(integrand, 0., upper, epsabs=1e-10, epsrel=2e-8)[0]


def _fold(profile, mu):
    path = profile.path_m
    interpolant = PchipInterpolator(path, profile.cumulative_m_inv, axis=0)
    x, w = np.polynomial.legendre.leggauss(8)
    half = np.diff(path)/2
    nodes = (path[1:]+path[:-1])[:, None]/2+half[:, None]*x
    weight = half[:, None]*w*np.exp(-mu*nodes)
    return np.sum(weight[:, :, None]*interpolant.derivative()(nodes), axis=(0, 1))


@pytest.mark.parametrize('theta,g', [(0., .9), (.7, .9), (2.8, .9), (2.8, -.4)])
def test_first_entry_direction_moments_match_independent_cap_rays(theta, g):
    r, mu = 6., .12
    profile = first_entry_profile(SphereSurface(1.), distance_m=r,
                                  direction_cosine=np.cos(theta), g=g,
                                  max_path_m=r+200)
    actual = _fold(profile, mu)
    expected = _cap_moments(r, theta, g, mu)
    np.testing.assert_allclose(actual, expected, rtol=4e-6, atol=2e-7*expected[0])
    increments = np.diff(profile.cumulative_m_inv[:, 0])
    assert np.all(increments >= 0)
    assert profile.path_m[0] == r-1


def test_direct_pencil_stops_before_any_return_from_behind_the_om():
    r, impact = 100., .9
    theta = np.arcsin(impact/r)
    profile = first_entry_profile(SphereSurface(1.), distance_m=r,
                                  direction_cosine=np.cos(theta), g=.9,
                                  max_path_m=120.)
    cosine = np.cos(theta)
    rho = r*np.sqrt((1-cosine)*(1+cosine))
    entry = r*cosine-np.sqrt(1-rho**2)
    tangent = np.sqrt(r*r-1)
    assert np.min(abs(profile.path_m-entry)) < 2e-13
    after = profile.path_m >= tangent
    np.testing.assert_array_equal(profile.cumulative_m_inv[after],
                                  np.broadcast_to(profile.cumulative_m_inv[-1],
                                                  profile.cumulative_m_inv[after].shape))
    assert np.isfinite(profile.cumulative_m_inv).all()


@pytest.mark.parametrize('r,cosine', [(1.001, -.8), (10000., np.cos(.9/10000.))])
def test_near_glass_and_far_forward_profiles_are_finite(r, cosine):
    profile = first_entry_profile(SphereSurface(1.), distance_m=r,
                                  direction_cosine=cosine, g=.9,
                                  max_path_m=r+10, path_nodes=32)
    assert np.isfinite(profile.cumulative_m_inv).all()


@pytest.mark.parametrize('impact', [1-1e-7, 1+1e-7])
def test_both_sides_of_grazing_pencil_use_consistent_geometry(impact):
    r = 100.
    cosine = np.sqrt(1-(impact/r)**2)
    profile = first_entry_profile(SphereSurface(1.), distance_m=r,
                                  direction_cosine=cosine, g=.9,
                                  max_path_m=120., path_nodes=32)
    assert np.isfinite(profile.cumulative_m_inv).all()
    tangent = np.sqrt((r-1)*(r+1))
    if impact < 1:
        after = profile.cumulative_m_inv[profile.path_m >= tangent]
        np.testing.assert_array_equal(after, np.broadcast_to(after[-1], after.shape))
    else:
        assert profile.cumulative_m_inv[-1, 0] > profile.cumulative_m_inv[-2, 0]


def test_refined_grazing_grid_avoids_singular_endpoint_rounding():
    r, impact = 1000., 1-1e-8
    profile = first_entry_profile(SphereSurface(1.), distance_m=r,
                                  direction_cosine=np.sqrt(1-(impact/r)**2), g=.9,
                                  max_path_m=1001., path_nodes=4096)
    assert np.isfinite(profile.cumulative_m_inv).all()
    assert np.all(np.diff(profile.cumulative_m_inv[:, 0]) >= 0)


def test_geometry_radius_rescales_path_and_cumulative_units():
    profiles = [first_entry_profile(SphereSurface([radius, 3*radius]), receiver=0,
                                   distance_m=6*radius, direction_cosine=.6, g=.9,
                                   max_path_m=24*radius, path_nodes=32)
                for radius in (.2, .4)]
    a, b = profiles
    np.testing.assert_allclose(b.path_m, 2*a.path_m, rtol=2e-15)
    np.testing.assert_allclose(b.cumulative_m_inv, a.cumulative_m_inv/2, rtol=2e-14)


@pytest.mark.parametrize('changes', [dict(distance_m=1.), dict(direction_cosine=1.1),
                                    dict(g=1.), dict(max_path_m=4.),
                                    dict(path_nodes=2.5), dict(path_order=True)])
def test_invalid_profile_geometry_or_quadrature_is_rejected(changes):
    options = dict(distance_m=5., direction_cosine=.5, g=.9, max_path_m=10.)
    options.update(changes)
    with pytest.raises(ValueError):
        first_entry_profile(SphereSurface(1.), **options)
