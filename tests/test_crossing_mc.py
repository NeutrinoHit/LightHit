"""The crossing sampler must reproduce what can be checked without sampling."""
import numpy as np
import pytest

from lighthit import synthetic_medium
from lighthit.experimental.crossing_mc import crossing_estimate, mean_cosine
from lighthit.experimental.stationary_modes import (StationaryModes,
                                                    collision_components)

MEDIUM = synthetic_medium()
RADII = np.array([20.0, 80.0])


@pytest.fixture(scope="module")
def sample():
    return crossing_estimate(scattering_per_m=MEDIUM.scattering_per_m, g=MEDIUM.g,
                             absorptions=[MEDIUM.absorption_per_m], radii_m=RADII,
                             photons_per_batch=8000, batches=4, max_path_m=1200.0,
                             seed=20260917)


def test_unscattered_order_crosses_head_on(sample):
    """Order 0 arrives along the radius exactly; only its survival is sampled."""
    cosine, error = mean_cosine(sample, order=0)
    np.testing.assert_allclose(cosine[0], 1.0, rtol=1e-12)
    np.testing.assert_allclose(error[0], 0.0, atol=1e-12)
    exact = np.exp(-MEDIUM.extinction_per_m * RADII) / (4 * np.pi * RADII ** 2)
    charge = sample[:, 0, :, 0, 0].mean(axis=0)
    np.testing.assert_allclose(charge, exact, rtol=0.05)


def test_total_charge_matches_the_eigenmode_sum(sample):
    charges, _ = collision_components(RADII, StationaryModes.build(MEDIUM, 256))
    sampled = sample[:, 0, :, :, 0].mean(axis=0).sum(axis=1)
    np.testing.assert_allclose(sampled, charges.sum(axis=1), rtol=3e-3)


@pytest.mark.parametrize("order", [1, 2])
def test_mean_cosine_matches_the_eigenmode_sum(sample, order):
    charges, currents = collision_components(RADII, StationaryModes.build(MEDIUM, 256))
    sampled, error = mean_cosine(sample, order)
    deviation = (sampled[0] - (currents / charges)[:, order]) / error[0]
    assert np.all(np.abs(deviation) < 4)


def test_absorption_derivative_equals_minus_covariance():
    """The path-selection identity, on one set of trajectories."""
    step = 2e-3
    centre = MEDIUM.absorption_per_m
    data = crossing_estimate(scattering_per_m=MEDIUM.scattering_per_m, g=MEDIUM.g,
                             absorptions=[centre - step, centre, centre + step],
                             radii_m=[20.0], photons_per_batch=6000, batches=4,
                             max_path_m=1200.0, seed=4242).mean(axis=0)
    weight = data[:, :, 2, 0]
    cosine = data[:, :, 2, 1] / weight
    path = data[:, :, 2, 2] / weight
    cosine_path = data[:, :, 2, 3] / weight
    derivative = (cosine[2] - cosine[0]) / (2 * step)
    covariance = cosine_path[1] - cosine[1] * path[1]
    np.testing.assert_allclose(derivative, -covariance, rtol=1e-3)
    assert np.all(covariance < 0)


@pytest.mark.parametrize("kwargs", [{"absorptions": [-0.1]}, {"radii_m": [0.0]},
                                    {"g": 1.0}, {"max_path_m": 1.0},
                                    {"batches": 0}])
def test_invalid_inputs_are_refused(kwargs):
    arguments = {"radii_m": [20.0], "photons_per_batch": 10, "batches": 1}
    arguments.update(kwargs)
    with pytest.raises(ValueError):
        crossing_estimate(**arguments)
