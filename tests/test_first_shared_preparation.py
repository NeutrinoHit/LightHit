"""Shared Q1 source preparation preserves receiver-private numerical work."""
import importlib.util
import inspect
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


candidate_path = os.environ.get('LIGHTHIT_FIRST_SHARED_MODULE')
if candidate_path:
    kernels = load_module(candidate_path, 'lighthit._first_shared_candidate')
else:
    from lighthit import _prompt_numba as kernels

reference_path = os.environ.get('LIGHTHIT_FIRST_SHARED_REFERENCE')
reference = (load_module(reference_path, 'lighthit._first_shared_reference')
             if reference_path else None)


def arguments(degree=3, common=True, reordered=False):
    from lighthit.prompt import acceptance_monomials
    alpha = np.array([1., .2, .1, .05, .01])[:degree + 1]
    coef, ipow, jpow, degree = acceptance_monomials(alpha)
    if reordered:
        order = np.arange(len(coef))
        order[1:] = order[1:][::-1]
        coef, ipow, jpow = coef[order], ipow[order], jpow[order]
    receivers = np.array([[10., 3., 7.], [20., 5., 2.], [0., 0., 12.],
                          [10., 3., -7.], [20., 5., -2.], [0., 0., -12.]])
    looks = np.tile([0., 0., 1.], (6, 1))
    if not common:
        looks[2] = [-0., 0., 1.]
        looks[3] = [.6, 0., .8]
        looks[4:] = [0., 0., -1.]
    edges = np.arange(-20., 300., 5.)
    mut = np.array([.08, .05])
    speed = np.array([.22, .23])
    kappa = -mut * speed
    # Input preparation stays lightweight for structural tests: the same
    # analytic table formula can run without compiling a staged kernel.
    intervals = np.empty((len(kappa), len(edges) - 1, 3))
    for lam, value in enumerate(kappa):
        for k, width in enumerate(np.diff(edges)):
            z = value * width
            intervals[lam, k] = kernels._ifuncs.py_func(z, math.exp(z))
    return [receivers, looks, np.full(6, .05), np.zeros(6), np.arange(6, dtype=np.int64),
            np.array([[0., 0., 0.], [0., 0., 1.], [0., 0., 0.], [0., 0., 1.]]),
            np.array([[0., 0., 1.], [.01, 0., np.sqrt(1 - .01**2)],
                      [0., 0., -1.], [0., 0., -1.]]),
            np.array([1., 2., 3., 4.]), np.array([-.5, -1., -1.5, -2.]),
            np.array([0., 5., 0., 5.]), np.tile([1., 0., 0.], (4, 1)),
            np.full(4, .001), np.full(4, .001), np.zeros((4, 6)),
            kernels._GAUSS_TAU, kernels._GAUSS_W, coef, ipow, jpow, degree,
            .9, .1 / 1.9, .1, .25, 25., np.array([.03, .02]), mut, speed,
            np.ones(2), np.full(2, .5), edges, np.exp(np.outer(kappa, edges)),
            intervals, 4, 2048, 6., 2e-3, .08, .5, .02,
            float(speed.max()), 1]


def exact_arrays(actual, expected):
    for left, right in zip(actual, expected):
        if left.dtype == np.float64:
            np.testing.assert_array_equal(left.view(np.uint64), right.view(np.uint64))
        else:
            np.testing.assert_array_equal(left, right)


@pytest.mark.parametrize('degree', [3, 4])
@pytest.mark.parametrize('common', [False, True])
@pytest.mark.parametrize('reordered', [False, True])
@pytest.mark.parametrize('compute_bins', [False, True])
def test_shared_preparation_preserves_full_grouped_reference_and_block_schedule(
        degree, common, reordered, compute_bins):
    args = arguments(degree, common, reordered)
    results = []
    for blocks in [1, 4]:
        args[-1] = blocks
        omission = np.full((6, 2), -17.)
        result = kernels.grouped_pencils_order1(*args, compute_bins, 1e-8, omission)
        assert np.all(result[0] > 0)
        assert result[1].shape == (6, len(args[30]) - 1 if compute_bins else 0)
        assert np.all(omission[:, 0] >= 0)
        results.append((*result, omission))
        if reference is not None:
            expected_omission = np.full((6, 2), -17.)
            expected = reference.grouped_pencils_order1(
                *args, compute_bins, 1e-8, expected_omission)
            exact_arrays((*result, omission), (*expected, expected_omission))
    exact_arrays(results[0], results[1])


def test_common_look_requires_bitwise_equality_only_on_requested_receivers():
    args = arguments()
    directions, looks = args[6], args[1].copy()
    looks[3, 0] = -0.
    work = np.array([0, 1, 2, 4, 5], dtype=np.int64)
    common, powers = kernels._prepare_common_pencil_look(directions, looks, work)
    assert common
    expected = np.empty_like(powers)
    kernels._prepare_pencil_look_powers(directions, *looks[0], expected)
    exact_arrays([powers], [expected])
    common, powers = kernels._prepare_common_pencil_look(directions, looks, np.arange(6, dtype=np.int64))
    assert not common
    assert powers.shape == (0, 3)
    common, powers = kernels._prepare_common_pencil_look(directions, looks, np.empty(0, dtype=np.int64))
    assert not common
    assert powers.shape == (0, 3)


@pytest.mark.parametrize('empty_work,blocks', [(True, 4), (False, 0), (False, -1)])
def test_no_work_returns_zero_without_source_preparation(monkeypatch, empty_work, blocks):
    args = arguments()
    args[-1] = blocks
    if empty_work:
        args[4] = np.empty(0, dtype=np.int64)

    def unexpected(*unused):
        raise AssertionError('source preparation should be skipped')

    for name in ['_exact_pencil_geometry_ids', '_prepare_pencil_shape',
                 '_prepare_gauss_inverse', '_is_cubic_moment_layout',
                 'build_theta_thresholds', 'build_theta_bit_lut', '_prepare_common_pencil_look']:
        monkeypatch.setattr(kernels, name, unexpected)
    omission = np.full((6, 2), -17.)
    result = kernels.grouped_pencils_order1.py_func(*args, True, 0., omission)
    for array in result:
        np.testing.assert_array_equal(array, np.zeros_like(array))
    np.testing.assert_array_equal(omission, np.full((6, 2), -17.))


def test_parallel_blocks_read_shared_source_arrays_safely():
    import numba
    if numba.config.NUMBA_NUM_THREADS < 2:
        pytest.skip('parallel runtime limited to one thread')
    previous = numba.get_num_threads()
    args = arguments(common=True)
    args[-1] = 4
    try:
        numba.set_num_threads(1)
        omission_one = np.zeros((6, 2))
        one = kernels.grouped_pencils_order1(*args, True, 1e-8, omission_one)
        numba.set_num_threads(2)
        omission_two = np.zeros((6, 2))
        two = kernels.grouped_pencils_order1(*args, True, 1e-8, omission_two)
        exact_arrays((*one, omission_one), (*two, omission_two))
    finally:
        numba.set_num_threads(previous)


def test_source_preparation_runs_once_and_blocks_share_the_same_arrays(monkeypatch):
    args = arguments()
    args[-1] = 4
    counts = {}
    ids = np.array([0, 1, 0, 1], dtype=np.int64)
    representatives = np.array([0, 1], dtype=np.int64)
    shape = np.zeros((4, 7))
    tangent, weighted = np.ones(2), np.ones(2)
    thresholds = np.array([.002, .003])
    lut = np.zeros(4, dtype=np.int64)
    powers = np.zeros((4, 3))
    preparations = {
        '_exact_pencil_geometry_ids': (ids, representatives),
        '_prepare_pencil_shape': shape,
        '_prepare_gauss_inverse': (tangent, weighted),
        '_is_cubic_moment_layout': True,
        'build_theta_thresholds': thresholds,
        'build_theta_bit_lut': (lut, np.int64(0)),
        '_prepare_common_pencil_look': (True, powers),
    }

    def counted(name, result):
        def prepare(*unused):
            counts[name] = counts.get(name, 0) + 1
            return result
        return prepare

    for name, result in preparations.items():
        monkeypatch.setattr(kernels, name, counted(name, result))
    block_signature = inspect.signature(kernels._grouped_pencils_order1_block.py_func)
    seen = []

    def block_spy(*values):
        seen.append(block_signature.bind(*values).arguments)

    monkeypatch.setattr(kernels, '_grouped_pencils_order1_block', block_spy)
    kernels.grouped_pencils_order1.py_func(*args, True, 0., np.zeros((6, 2)))
    assert counts == {name: 1 for name in preparations}
    assert [block['block'] for block in seen] == [0, 1, 2, 3]
    shared_arrays = dict(geometry_ids=ids, representatives=representatives,
                         prepared_shape=shape, gauss_tangent=tangent,
                         gauss_weighted_sec2=weighted, theta_thresholds=thresholds,
                         theta_lut=lut, shared_look_powers=powers)
    for block in seen:
        assert block['n_blocks'] == 4
        assert block['common_look'] is True
        for name, value in shared_arrays.items():
            assert block[name] is value


def test_grown_hash_rows_are_reset_when_reused_by_next_receiver():
    # More than 2048 distinct clocks grow the initial 4096-slot table.
    # Huge screening tolerance avoids expensive profile integration while
    # retaining grouped moments and their independent omission bounds.
    args = arguments(common=True)
    n = 3000
    args[5] = np.zeros((n, 3))
    args[6] = np.tile([0., 0., 1.], (n, 1))
    args[7] = 1. + .01 * (np.arange(n) % 17)
    args[8] = -.5 * args[7]
    args[9] = np.arange(n, dtype=float) * 2. + .25
    args[10] = np.tile([1., 0., 0.], (n, 1))
    args[11] = np.full(n, 1e-6)
    args[12] = np.full(n, 1e-6)
    args[13] = np.zeros((n, 6))
    args[4] = np.array([0, 1], dtype=np.int64)
    args[-1] = 1
    omission = np.zeros((6, 2))
    combined = kernels.grouped_pencils_order1(*args, False, 1e30, omission)
    np.testing.assert_array_equal(combined[2][:2], [n, n])
    np.testing.assert_array_equal(omission[:2, 0], [n, n])
    assert np.all(omission[:2, 1] > 0)
    for d in [0, 1]:
        one = list(args)
        one[4] = np.array([d], dtype=np.int64)
        isolated_omission = np.zeros((6, 2))
        isolated = kernels.grouped_pencils_order1(*one, False, 1e30, isolated_omission)
        exact_arrays([x[d:d + 1] for x in (*combined, omission)],
                     [x[d:d + 1] for x in (*isolated, isolated_omission)])
    if reference is not None:
        expected_omission = np.zeros((6, 2))
        expected = reference.grouped_pencils_order1(*args, False, 1e30, expected_omission)
        exact_arrays((*combined, omission), (*expected, expected_omission))
