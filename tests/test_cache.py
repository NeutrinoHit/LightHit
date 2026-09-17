import numpy as np
import pytest

from lighthit.cache import (BandedResponseCache, CacheGrid, ResponseCache,
                            acceptance_coefficients, cosine_acceptance,
                            first_order_consistency, hemispherical_acceptance,
                            isotropic_acceptance, validation_report)
from lighthit.green import PointGreenSolver, SolverSettings
from lighthit.medium import Medium

MEDIUM = Medium(0.04, 0.05, 0.7, 1.35, 450.0, "synthetic-example-not-Baikal")
SETTINGS = SolverSettings(16, 60, 3.0, 0.05, 8)


@pytest.fixture(scope="module")
def cache():
    grid = CacheGrid.geometric(8.0, 24.0, 10, [0.0])
    return ResponseCache.build(MEDIUM, SETTINGS, grid)


def test_grid_rejects_unsorted_and_short_axes():
    with pytest.raises(ValueError):
        CacheGrid(np.array([3.0, 1.0]), np.array([0.0]))
    with pytest.raises(ValueError):
        CacheGrid(np.array([1.0]), np.array([-1.0]))
    with pytest.raises(ValueError):
        CacheGrid.geometric(8.0, 24.0, 3, [0.0])
    with pytest.raises(ValueError):
        CacheGrid.geometric(24.0, 8.0, 10, [0.0])


def test_isotropic_acceptance_reproduces_the_isotropic_detector():
    alpha = acceptance_coefficients(isotropic_acceptance, 6)
    assert alpha[0] == pytest.approx(4 * np.pi, rel=1e-12)
    assert np.allclose(alpha[1:], 0.0, atol=1e-12)


def test_hemispherical_acceptance_has_two_coefficients():
    alpha = acceptance_coefficients(hemispherical_acceptance, 8)
    assert alpha[0] == pytest.approx(2 * np.pi, rel=1e-12)
    assert alpha[1] == pytest.approx(2 * np.pi / 3, rel=1e-12)
    assert np.allclose(alpha[2:], 0.0, atol=1e-12)


def test_acceptance_rejects_negative_response():
    with pytest.raises(ValueError):
        acceptance_coefficients(lambda x: x, 4)


def test_on_grid_directed_charge_matches_the_solver(cache):
    radii = cache.grid.radii_m
    cosines = np.full_like(radii, 0.3)
    cached = cache.directed_charge(radii, cosines)
    sin = np.sqrt(1 - cosines ** 2)
    displacement = np.stack([radii * sin, np.zeros_like(radii), radii * cosines], axis=1)
    exact = PointGreenSolver(MEDIUM, SETTINGS).solve(
        [0.0], displacement, direction=(0.0, 0.0, 1.0)).components[0].real
    assert np.allclose(cached[:, 1], exact[:, 1], rtol=1e-10)
    assert np.allclose(cached[:, 2], exact[:, 2], rtol=1e-9)


def test_isotropic_detector_via_acceptance_matches_the_isotropic_branch(cache):
    radii = cache.grid.radii_m[2:-2]
    alpha = acceptance_coefficients(isotropic_acceptance, 4)
    cached = cache.acceptance_charge(radii, np.zeros_like(radii), alpha,
                                     exact_first_order=True,
                                     acceptance=isotropic_acceptance)
    displacement = np.stack([np.zeros_like(radii), np.zeros_like(radii), radii], axis=1)
    exact = PointGreenSolver(MEDIUM, SETTINGS).solve(
        [0.0], displacement, direction=None).components[0].real
    assert np.allclose(cached[:, 0], exact[:, 0], rtol=1e-10)
    assert np.allclose(cached[:, 1], exact[:, 1], rtol=1e-10)
    assert np.allclose(cached[:, 2], exact[:, 2], rtol=1e-6)


def test_interpolation_error_is_small_off_grid(cache):
    rng = np.random.default_rng(4)
    radii = np.exp(rng.uniform(np.log(8.5), np.log(23.0), 12))
    cosines = rng.uniform(-0.9, 0.9, 12)
    report = validation_report(cache, radii, cosines)
    assert report["median_relative_error_total"] < 1e-3


def test_photon_scaling_is_linear(cache):
    radii = np.array([12.0, 18.0])
    alpha = acceptance_coefficients(hemispherical_acceptance, 4)
    one = cache.acceptance_charge(radii, np.zeros_like(radii), alpha)
    many = cache.acceptance_charge(radii, np.zeros_like(radii), alpha, photons=1e6)
    assert np.allclose(many, 1e6 * one, rtol=1e-12)


def test_queries_outside_the_cached_range_are_refused(cache):
    with pytest.raises(ValueError):
        cache.directed_charge(np.array([4.0]), np.array([0.0]))
    with pytest.raises(ValueError):
        cache.directed_charge(np.array([40.0]), np.array([0.0]))


def test_shape_mismatch_is_refused(cache):
    with pytest.raises(ValueError):
        cache.directed_charge(np.array([12.0, 14.0]), np.array([0.0]))


def test_charge_requires_a_cached_zero_frequency():
    grid = CacheGrid(np.geomspace(8.0, 24.0, 6), np.array([0.3]))
    partial = ResponseCache.build(MEDIUM, SETTINGS, grid)
    with pytest.raises(ValueError):
        partial.directed_charge(np.array([12.0]), np.array([0.0]))


def test_round_trip_through_disk(cache, tmp_path):
    path = tmp_path / "cache.npz"
    cache.save(path)
    loaded = ResponseCache.load(path)
    assert np.allclose(loaded.moments, cache.moments)
    assert loaded.medium == cache.medium
    assert loaded.settings == cache.settings
    radii = np.array([11.0, 19.0])
    cosines = np.array([0.2, -0.4])
    assert np.allclose(loaded.directed_charge(radii, cosines),
                       cache.directed_charge(radii, cosines))


def test_banded_cache_dispatches_and_matches_its_bands():
    low = ResponseCache.build(MEDIUM, SolverSettings(16, 60, 3.0, 0.05, 8),
                              CacheGrid.geometric(8.0, 16.0, 8, [0.0]))
    high = ResponseCache.build(MEDIUM, SolverSettings(16, 110, 3.0, 0.05, 8),
                               CacheGrid.geometric(16.0, 32.0, 8, [0.0]))
    banded = BandedResponseCache([low, high])
    assert banded.radius_range_m == (8.0, 32.0)
    radii = np.array([10.0, 24.0])
    cosines = np.array([0.1, 0.1])
    joint = banded.directed_charge(radii, cosines)
    assert np.allclose(joint[0], low.directed_charge(radii[:1], cosines[:1])[0])
    assert np.allclose(joint[1], high.directed_charge(radii[1:], cosines[1:])[0])


def test_banded_cache_refuses_gaps():
    low = ResponseCache.build(MEDIUM, SETTINGS, CacheGrid.geometric(8.0, 16.0, 8, [0.0]))
    high = ResponseCache.build(MEDIUM, SETTINGS, CacheGrid.geometric(20.0, 32.0, 8, [0.0]))
    with pytest.raises(ValueError):
        BandedResponseCache([low, high])
    with pytest.raises(ValueError):
        BandedResponseCache([])


def test_module_axis_sign_convention(cache):
    """A module facing the flash collects more than one facing away."""
    alpha = acceptance_coefficients(cosine_acceptance, 32)
    banded = BandedResponseCache([cache])
    above = np.array([[0.0, 0.0, 12.0]])     # module sits above the flash
    facing_down = banded.charge_for_modules(above, (0.0, 0.0, -1.0), alpha,
                                            acceptance=cosine_acceptance)
    facing_up = banded.charge_for_modules(above, (0.0, 0.0, 1.0), alpha,
                                          acceptance=cosine_acceptance)
    # The photon travels upwards, so a downward-facing module sees it head-on.
    assert facing_down[0, 0] > 0
    assert facing_up[0, 0] == pytest.approx(0.0, abs=1e-30)
    assert facing_down.sum() > facing_up.sum()


def test_finite_L_first_order_tracks_the_exact_quadrature(cache):
    radii = np.array([9.0, 14.0, 22.0])
    report = first_order_consistency(cache, radii)
    assert report["max_first_order_error_over_total"] < 1e-2


def test_smooth_acceptance_needs_few_degrees(cache):
    """The multipole sum stops at the acceptance bandwidth, not the grid degree."""
    radii = np.array([12.0])
    smooth = acceptance_coefficients(hemispherical_acceptance, cache.degree)
    padded = np.concatenate([smooth, np.zeros(5)])
    assert np.allclose(cache.acceptance_charge(radii, np.array([0.4]), smooth),
                       cache.acceptance_charge(radii, np.array([0.4]), padded))
