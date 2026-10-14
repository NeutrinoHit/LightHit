"""Fused cubic grouping preserves original arithmetic and layout selection."""
from pathlib import Path
import importlib.util
import os

import numpy as np
import pytest


def moment_module():
    """Allow an isolated candidate to be checked before installing it."""
    candidate = os.environ.get("LIGHTHIT_FIRST_MOMENT_MODULE")
    if candidate:
        spec = importlib.util.spec_from_file_location("first_fused_candidate", Path(candidate))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    from lighthit import _prompt_numba
    return _prompt_numba


MODULE = moment_module()
IPOW = np.array([i for k in range(4) for i in range(k + 1)], dtype=np.int64)
JPOW = np.array([k - i for k in range(4) for i in range(k + 1)], dtype=np.int64)


def original_add(weight0, weight2, c1, c3, averaged, row):
    mom0 = np.empty(10)
    mom2 = np.empty(10)
    MODULE._fill_moments_averaged_pair(
        weight0, weight2, c1, c3, averaged, IPOW, JPOW, 3,
        mom0, mom2, np.empty(4), np.empty(4))
    for m in range(10):
        row[m] += mom0[m]
        row[10 + m] += mom2[m]
    return abs(mom0[0])


@pytest.mark.parametrize("averaged", [False, True])
def test_fused_cubic_preserves_bits_including_signed_zero(averaged):
    rng = np.random.default_rng(417)
    special = np.array([-1., np.nextafter(-1., -np.inf), -0., 0., 1.,
                        np.nextafter(1., np.inf), 1e-300, -1e-300])
    inputs = [(a, b, c1, c3)
              for a, b in [(0., -0.), (-0., 0.), (1., -2.), (-3., 4.),
                           (1e-300, -1e300)]
              for c1 in special for c3 in special]
    inputs += [tuple(values) for values in rng.uniform(-1., 1., (300, 4))]
    for weight0, weight2, c1, c3 in inputs:
        expected = np.zeros(26)
        actual = expected.copy()
        omega = original_add(weight0, weight2, c1, c3, averaged, expected)
        other = MODULE._accumulate_cubic_moments_pair(
            weight0, weight2, c1, c3, averaged, actual)
        np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
        assert np.float64(other).view(np.uint64) == np.float64(omega).view(np.uint64)


@pytest.mark.parametrize("averaged", [False, True])
def test_fused_cubic_preserves_repeated_accumulation(averaged):
    rng = np.random.default_rng(991)
    expected = rng.normal(size=26)
    actual = expected.copy()
    for weight0, weight2, c1, c3 in rng.uniform(-1., 1., (700, 4)):
        original_add(weight0, weight2, c1, c3, averaged, expected)
        MODULE._accumulate_cubic_moments_pair(
            weight0, weight2, c1, c3, averaged, actual)
    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))


def test_cubic_layout_gate_rejects_nonstandard_moment_tables():
    assert MODULE._is_cubic_moment_layout(IPOW, JPOW, 3)
    assert not MODULE._is_cubic_moment_layout(IPOW, JPOW, 4)
    assert not MODULE._is_cubic_moment_layout(IPOW[:-1], JPOW[:-1], 3)
    assert not MODULE._is_cubic_moment_layout(IPOW, JPOW[:-1], 3)
    permutation = np.arange(10)
    permutation[[3, 6]] = permutation[[6, 3]]
    assert not MODULE._is_cubic_moment_layout(IPOW[permutation], JPOW[permutation], 3)
    assert not MODULE._is_cubic_moment_layout(JPOW, IPOW, 3)


def test_precomputed_geometry_key_preserves_original_integer_key():
    rng = np.random.default_rng(609)
    for i_tau, i_r, backward, theta_bins, i_theta in rng.integers(
            [0, 0, 0, 1, 0], [2**21, 2**21, 2, 2048, 2048], (1000, 5)):
        original = (((i_tau * (1 << 21) + i_r) * 2 + backward) * theta_bins) + i_theta
        base = (i_tau * (1 << 21) + i_r) * 2 * theta_bins
        assert base + backward * theta_bins + i_theta == original


def test_cached_look_powers_and_frame_preserve_original_bits():
    rng = np.random.default_rng(112)
    directions = rng.normal(size=(200, 3))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    special = np.array([[0., 0., 1.], [-0., 0., -1.], [0., -0., 1.],
                        [1., -0., 0.], [-1., 0., -0.]])
    directions = np.vstack([directions, special])
    rays = rng.normal(size=(len(directions), 3))
    rays /= np.linalg.norm(rays, axis=1)[:, None]
    looks = np.vstack([special, rng.normal(size=(8, 3))])
    for nx, ny, nz in looks:
        powers = np.empty((len(directions), 3))
        MODULE._prepare_pencil_look_powers(directions, nx, ny, nz, powers)
        for p, (sx, sy, sz) in enumerate(directions):
            original = MODULE._pencil_frame_from_ray(
                13., *rays[p], sx, sy, sz, nx, ny, nz)
            cached = MODULE._pencil_frame_from_ray_cached_look(
                13., *rays[p], sx, sy, sz, nx, ny, nz, powers[p, 0])
            np.testing.assert_array_equal(np.array(cached).view(np.uint64),
                                          np.array(original).view(np.uint64))
            c1, c3 = original[4:]
            cp1 = 1.0 * c1
            cp2 = cp1 * c1
            cp3 = cp2 * c1
            np.testing.assert_array_equal(powers[p].view(np.uint64),
                                          np.array([cp1, cp2, cp3]).view(np.uint64))
            for averaged in [False, True]:
                expected = np.zeros(26)
                actual = expected.copy()
                original_add(2.5, -0.7, c1, c3, averaged, expected)
                MODULE._accumulate_cubic_moments_pair_prepared(
                    2.5, -0.7, *powers[p], c3, averaged, actual)
                np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))


def _grouped_arguments(degree=3, strided_looks=False):
    from lighthit.prompt import acceptance_monomials
    coef, ipow, jpow, degree = acceptance_monomials(np.array([1., .2, .1, .05, .01])[:degree + 1])
    edges = np.arange(-20., 300., 5.)
    mut = np.array([.08, .05])
    speed = np.array([.22, .23])
    kappa = -mut * speed
    receivers = np.array([[10., 3., 7.], [20., 5., 2.], [0., 0., 12.],
                          [10., 3., -7.], [20., 5., -2.], [0., 0., -12.]])
    looks = np.array([[0., 0., 1.], [0., 0., 1.], [-0., 0., 1.],
                      [.6, 0., .8], [0., 0., -1.], [0., 0., -1.]])
    if strided_looks:
        container = np.zeros((len(looks), 6))
        container[:, ::2] = looks
        looks = container[:, ::2]
        assert not looks.flags.c_contiguous
    n = len(receivers)
    return [receivers, looks, np.full(n, .05), np.zeros(n), np.arange(n, dtype=np.int64),
            np.array([[0., 0., 0.], [0., 0., 1.], [0., 0., 0.], [0., 0., 1.]]),
            np.array([[0., 0., 1.], [0., 0., 1.], [0., 0., -1.], [0., 0., -1.]]),
            np.array([1., 2., 3., 4.]), np.array([-.5, -1., -1.5, -2.]),
            np.array([0., 5., 0., 5.]), np.tile([1., 0., 0.], (4, 1)),
            np.full(4, .001), np.full(4, .001), np.zeros((4, 6)),
            MODULE._GAUSS_TAU, MODULE._GAUSS_W, coef, ipow, jpow, degree,
            .9, .1 / 1.9, .1, .25, 25., np.array([.03, .02]), mut, speed,
            np.ones(2), np.full(2, .5), edges, np.exp(np.outer(kappa, edges)),
            MODULE.interval_table(edges, kappa), 4, 2048, 6., 2e-3, .08, .5, .02,
            float(speed.max()), 1]


@pytest.mark.parametrize("degree", [3, 4])
@pytest.mark.parametrize("strided_looks", [False, True])
def test_complete_grouped_function_preserves_frozen_reference(degree, strided_looks):
    reference_path = os.environ.get("LIGHTHIT_FIRST_MOMENT_REFERENCE")
    arguments = _grouped_arguments(degree, strided_looks)
    actual = MODULE.grouped_pencils_order1(*arguments)
    assert np.all(actual[0] > 0)
    assert np.all(np.isfinite(actual[1]))
    if reference_path:
        spec = importlib.util.spec_from_file_location("first_fused_reference", reference_path)
        reference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reference)
        expected = reference.grouped_pencils_order1(*arguments)
        for current, original in zip(actual, expected):
            if current.dtype == np.float64:
                np.testing.assert_array_equal(current.view(np.uint64), original.view(np.uint64))
            else:
                np.testing.assert_array_equal(current, original)
    else:
        # Permuting the monomial table selects the generic accumulation path
        # while representing exactly the same polynomial acceptance.
        permutation = np.arange(len(arguments[16]))
        permutation[1:] = permutation[1:][::-1]
        alternate = list(arguments)
        for index in [16, 17, 18]:
            alternate[index] = arguments[index][permutation].copy()
        expected = MODULE.grouped_pencils_order1(*alternate)
        for current, original in zip(actual, expected):
            np.testing.assert_allclose(current, original, rtol=1e-13, atol=1e-16)
