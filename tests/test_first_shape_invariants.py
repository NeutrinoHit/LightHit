"""Controls for receiver-independent factors in grouped secondary Q1."""
import math

import numpy as np
import pytest

pytest.importorskip("numba")
from lighthit import _prompt_numba as kernels
from lighthit.prompt import acceptance_monomials


@pytest.mark.parametrize("degree", [0, 1, 3, 6])
@pytest.mark.parametrize("averaged", [False, True])
def test_fused_moment_fill_is_bitwise_identical(degree, averaged):
    rng = np.random.default_rng(321)
    _, ipow, jpow, power = acceptance_monomials(np.ones(degree + 1))
    for c1 in np.r_[rng.uniform(-1., 1., 20), -1., 0., 1.]:
        c3 = rng.uniform(-1., 1.)
        weight0, weight2 = rng.normal(size=2) * 1e6
        old0, old2 = np.empty(len(ipow)), np.empty(len(ipow))
        new0, new2 = np.empty(len(ipow)), np.empty(len(ipow))
        cp, sp = np.empty(power + 1), np.empty(power + 1)
        kernels._fill_moments_averaged(weight0, c1, c3, averaged,
                                       ipow, jpow, power, old0, cp, sp)
        kernels._fill_moments_averaged(weight2, c1, c3, averaged,
                                       ipow, jpow, power, old2, cp, sp)
        kernels._fill_moments_averaged_pair(weight0, weight2, c1, c3, averaged,
                                            ipow, jpow, power, new0, new2, cp, sp)
        np.testing.assert_array_equal(new0, old0)
        np.testing.assert_array_equal(new2, old2)


@pytest.mark.parametrize("custom_rule", [False, True])
def test_prepared_gaussian_rule_preserves_original_arithmetic(custom_rule):
    rng = np.random.default_rng(183)
    if custom_rule:
        tau = rng.uniform(.001, math.pi / 2 - .001, 32)
        weights = rng.uniform(.001, .1, len(tau))
    else:
        tau, weights = kernels._GAUSS_TAU, kernels._GAUSS_W
    tangent, weighted_sec2 = kernels._prepare_gauss_inverse(tau, weights)
    cases = [(0., 0., 0., 0.), (0., 0., .01, 1e-30), (1e-10, 0., 1e-30, 1e-30)]
    cases += [(rng.normal(), rng.normal(), 10**rng.uniform(-16, 3),
               10**rng.uniform(-16, 3)) for _ in range(50)]
    for u, v, var_u, var_v in cases:
        old = kernels.gauss_inverse_mean(u, v, var_u, var_v, tau, weights)
        new = kernels.gauss_inverse_mean_prepared(u, v, var_u, var_v,
                                                   tangent, weighted_sec2)
        assert old == new


def _pencils(rank):
    rng = np.random.default_rng(822)
    directions = rng.normal(size=(16, 3))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    major = rng.normal(size=(16, 3))
    major -= np.sum(major * directions, axis=1)[:, None] * directions
    major /= np.linalg.norm(major, axis=1)[:, None]
    positions = rng.uniform([-105., -900., 330.], [-95., 900., 370.], (16, 3))
    matrix = rng.normal(size=(16, 3, rank))
    covariance = matrix @ matrix.transpose(0, 2, 1)
    packed = covariance[:, (0, 1, 2, 0, 0, 1), (0, 1, 2, 1, 2, 2)]
    return positions, directions, major, packed


@pytest.mark.parametrize("rank", [0, 1, 3])
@pytest.mark.parametrize("custom_rule", [False, True])
def test_prepared_shape_matches_original_for_random_covariances(rank, custom_rule):
    pos, directions, major, cov = _pencils(rank)
    rng = np.random.default_rng(183)
    var_major = 10**rng.uniform(-9, -1, len(pos))
    var_minor = var_major * 10**rng.uniform(-16, 0, len(pos))
    prepared = kernels._prepare_pencil_shape(directions, major, cov)
    tau = kernels._GAUSS_TAU if not custom_rule else np.linspace(.01, 1.56, 32)
    weights = kernels._GAUSS_W if not custom_rule else np.full(32, .048)
    tangent, weighted_sec2 = kernels._prepare_gauss_inverse(tau, weights)
    for p in range(len(pos)):
        for distance in (1e-3, 2., 40., 400.):
            for theta in (0., 1e-10, .002, .03, .2, 1., 2.9, math.pi):
                receiver = pos[p] + distance * (math.cos(theta) * directions[p]
                                                + math.sin(theta) * major[p])
                arguments = (p, distance, theta, math.sin(theta), *receiver, pos)
                old = kernels._pencil_shape_factor(*arguments, directions, major,
                    var_major, var_minor, cov, 3., tau, weights)
                new = kernels._pencil_shape_factor_prepared(*arguments, major,
                    var_major, var_minor, prepared, 3., tangent, weighted_sec2)
                assert old[2:] == new[2:]
                np.testing.assert_array_equal(new[:2], old[:2])


def test_prepared_shape_on_receiver_array_matches_original():
    pos, directions, major, cov = _pencils(3)
    prepared = kernels._prepare_pencil_shape(directions, major, cov)
    var_major, var_minor = np.full(16, .003), np.full(16, .0002)
    tau, weights = kernels._GAUSS_TAU, kernels._GAUSS_W
    tangent, weighted_sec2 = kernels._prepare_gauss_inverse(tau, weights)
    # Receiver positions exercise multiple strings and depths independently
    # of the angle/distance constructions above.
    receivers = np.array([[0., 0., 300.], [60., 0., 350.], [-30., 51.96, 450.],
                          [-30., -51.96, 250.], [300., 240., 500.], [-250., 500., 370.]])
    for p in range(len(pos)):
        for receiver in receivers:
            r, theta, sine, *_ = kernels._pencil_frame(*(receiver - pos[p]),
                                                       *directions[p], 0., 0., 1.)
            arguments = (p, r, theta, sine, *receiver, pos)
            old = kernels._pencil_shape_factor(*arguments, directions, major,
                var_major, var_minor, cov, 3., tau, weights)
            new = kernels._pencil_shape_factor_prepared(*arguments, major,
                var_major, var_minor, prepared, 3., tangent, weighted_sec2)
            np.testing.assert_array_equal(new, old)


@pytest.mark.parametrize("smoothing", [.001, .3, 1., 3., 100., 1e-200, 1e200])
@pytest.mark.parametrize("matrix", [(.001, .001, 0.), (.01, .002, .003),
                                    (.01, -.02, .003), (-.01, -.02, .003),
                                    (0., 0., 0.), (1e-300, 1e-300, 0.),
                                    (1e151, 1e151, 0.), (1e200, -1e200, 1e200)])
def test_far_gate_preserves_shape_switch_and_extreme_arithmetic(smoothing, matrix):
    m11, m22, m12 = matrix
    directions, major = np.array([[0., 0., 1.]]), np.array([[1., 0., 0.]])
    pos = np.zeros((1, 3))
    cov = np.array([[m11, m22, 0., m12, 0., 0.]])
    variance = np.zeros(1)
    prepared = kernels._prepare_pencil_shape(directions, major, cov)
    tau, weights = kernels._GAUSS_TAU, kernels._GAUSS_W
    tangent, weighted_sec2 = kernels._prepare_gauss_inverse(tau, weights)
    angles = [0., 1e-10, .02, .7, math.pi]
    # Exercise both sides and the floating-point neighbours of the exact
    # smoothing switch when the original eigensystem has finite arithmetic.
    if max(abs(m11), abs(m22), abs(m12)) < 1e150:
        disc = math.sqrt(max(.25 * (m11-m22)**2 + m12*m12, 0.))
        mu1 = .5*(m11+m22) + disc
        boundary = smoothing * math.sqrt(max(mu1, 0.))
        if 0. < boundary < math.pi:
            angles += [np.nextafter(boundary, 0.), boundary,
                       np.nextafter(boundary, math.inf),
                       boundary*(1.-2e-12), boundary*(1.+2e-12)]
    for theta in angles:
        receiver = np.array([math.sin(theta), 0., math.cos(theta)])
        arguments = (0, 1., theta, math.sin(theta), *receiver, pos)
        old = kernels._pencil_shape_factor(*arguments, directions, major,
            variance, variance, cov, smoothing, tau, weights)
        new = kernels._pencil_shape_factor_prepared(*arguments, major,
            variance, variance, prepared, smoothing, tangent, weighted_sec2)
        assert old[2:] == new[2:]
        np.testing.assert_array_equal(new[:2], old[:2])
