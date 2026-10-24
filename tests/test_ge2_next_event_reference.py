"""Independent normalization controls for the collision-sampling reference."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad

pytest.importorskip('numba')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from harness.ge2_reference import cap_score, collision_batch


@pytest.mark.parametrize('g', [0., .9])
@pytest.mark.parametrize('relative_distance', [1.01, 1.1, 2., 5., 20., 200.])
def test_cap_mass_matches_analytic_hg_probability(g, relative_distance):
    radius = .216
    low = math.sqrt(1-1/relative_distance**2)
    if g == 0:
        expected = (1-low)/2
    else:
        expected = ((1-g*g)/(2*g)
                    *(1/(1-g)-1/math.sqrt((1-g)**2+2*g*(1-low))))
    empty = np.empty(0)
    actual = cap_score(np.array([0., 0., relative_distance*radius]),
        np.array([0., 0., 1.]), np.array([0., 0., 1.]), radius, 0., g,
        np.array([1., 0., 0., 0.]), empty, empty, empty, empty)
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-13)


@pytest.mark.parametrize('importance', [0, 4])
def test_first_collision_and_all_order_normalization(importance):
    n, phase_index = 160_000, 1.35
    cosine = 1/phase_index
    mu_a, mu_s, g, r = .05, .02, .9, 30.
    mu_t, albedo = mu_a+mu_s, mu_s/(mu_a+mu_s)
    max_sin2 = np.array([1-cosine*cosine])
    start = np.zeros((1, 3))
    axis = np.array([[0., 0., 1.]])
    receivers = np.array([[0., 0., r]])
    empty = np.empty(0)
    total, square, _, normal, normal2 = collision_batch(n, 92571,
        start, axis, np.zeros(1), np.ones(1), np.ones(1),
        np.array([350., 600.]), np.full(2, phase_index), np.full(2, mu_t),
        np.full(2, mu_s), np.array([[1., 0., 0., 0., 0., 0.]]), 1., max_sin2, g,
        receivers, axis, np.array([.1]), np.ones(1), np.array([1., 0., 0., 0.]),
        empty, empty, empty, empty, np.ones(1), importance)
    # Integral of every collision density over all space. This checks the
    # forced collisions, each importance ratio and roulette compensation.
    expected = np.array([albedo, albedo**2/(1-albedo)])
    mean = normal/n
    error = np.sqrt(np.maximum(0., normal2-normal*normal/n)/(n-1)/n)
    rounding = n*np.finfo(float).eps*abs(expected)
    assert np.all(abs(mean-expected) <= 5*error+rounding)

    # For a point on the axis every cone azimuth has the same first-order
    # response. Its independent one-dimensional flight integral is nonsingular.
    direction = np.array([math.sqrt(1-cosine*cosine), 0., cosine])
    def integrand(flight):
        towards = receivers[0]-flight*direction
        distance = np.linalg.norm(towards)
        angle = np.dot(direction, towards)/distance
        hg = (1-g*g)/(4*np.pi*(1+g*g-2*g*angle)**1.5)
        return mu_s*np.exp(-mu_t*(flight+distance))*hg/distance**2
    reference = quad(integrand, 0., np.inf, epsabs=1e-13, epsrel=1e-10)[0]
    value = total[0, 0]/n
    sigma = math.sqrt(max(0., square[0, 0]-total[0, 0]**2/n)/(n-1)/n)
    assert abs(value-reference) <= 5*sigma
