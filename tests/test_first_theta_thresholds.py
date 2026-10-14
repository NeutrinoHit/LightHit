"""Exact angular-bin transitions and scalar fallback behavior."""
import importlib.util
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from numba import njit

candidate_path = os.environ.get('LIGHTHIT_THETA_CLASSIFIER')
if candidate_path:
    spec = importlib.util.spec_from_file_location('q1_theta_classifier', Path(candidate_path))
    classifier = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = classifier
    spec.loader.exec_module(classifier)
else:
    from lighthit import _prompt_numba as classifier


@njit
def classify_scalar(angles, theta_lo, log_theta_ratio):
    result = np.empty(angles.shape[0], dtype=np.int64)
    for i in range(angles.shape[0]):
        result[i] = classifier.scalar_theta_index(angles[i], theta_lo, log_theta_ratio)
    return result


@njit
def classify_threshold(angles, theta_lo, log_theta_ratio, thresholds):
    result = np.empty(angles.shape[0], dtype=np.int64)
    for i in range(angles.shape[0]):
        result[i] = classifier.threshold_theta_index(angles[i], theta_lo, log_theta_ratio, thresholds)
    return result


@njit
def classify_bit_lut(angles, theta_lo, log_theta_ratio, thresholds, lut, first):
    result = np.empty(angles.shape[0], dtype=np.int64)
    for i in range(angles.shape[0]):
        result[i] = classifier.bit_lut_theta_index(angles[i], theta_lo, log_theta_ratio, thresholds, lut, first)
    return result


def configuration(theta_lo, ratio):
    log_ratio = math.log(1.0 + ratio)
    bins = 2 + int(math.log(math.pi / theta_lo) / log_ratio)
    return theta_lo, log_ratio, bins


@pytest.mark.parametrize('theta_lo,ratio', [
    (2e-3, .08), (1e-9, .1), (.1, .005), (.5, .5),
    (1e-12, .5), (.001, .002),
    (math.pi, .1), (np.nextafter(math.pi, 0.), .1)])
def test_every_transition_and_sixty_four_neighbors(theta_lo, ratio):
    lo, lr, bins = configuration(theta_lo, ratio)
    thresholds = classifier.build_theta_thresholds(lo, lr, bins)
    assert len(thresholds) == bins - 1
    values = []
    for threshold in thresholds:
        below = above = threshold
        values.append(threshold)
        for _ in range(64):
            below = np.nextafter(below, -np.inf)
            above = np.nextafter(above, np.inf)
            values.extend([below, above])
    angles = np.asarray(values)
    expected = classify_scalar(angles, lo, lr)
    np.testing.assert_array_equal(classify_threshold(angles, lo, lr, thresholds), expected)
    lut, first = classifier.build_theta_bit_lut(lo, thresholds)
    np.testing.assert_array_equal(classify_bit_lut(angles, lo, lr, thresholds, lut, first), expected)


def test_random_angles_and_configurations_match_original():
    rng = np.random.default_rng(620)
    angles = np.concatenate([rng.uniform(0., math.pi, 20000),
                             np.exp(rng.uniform(-200., math.log(math.pi), 20000)),
                             [-0., 0., math.pi, np.inf, -np.inf, np.nan]])
    for _ in range(35):
        lo = float(np.exp(rng.uniform(-150., .4)))
        lr = float(rng.uniform(.01, 2.))
        bins = 2 + int(math.log(math.pi / lo) / lr)
        thresholds = classifier.build_theta_thresholds(lo, lr, bins)
        expected = classify_scalar(angles, lo, lr)
        np.testing.assert_array_equal(classify_threshold(angles, lo, lr, thresholds), expected)
        lut, first = classifier.build_theta_bit_lut(lo, thresholds)
        np.testing.assert_array_equal(classify_bit_lut(angles, lo, lr, thresholds, lut, first), expected)


@pytest.mark.parametrize('lo,lr,bins,adjustments', [
    (0., .1, 30, 128), (-.1, .1, 30, 128),
    (np.nan, .1, 30, 128), (np.inf, .1, 30, 128),
    (.1, 0., 30, 128), (.1, -.1, 30, 128), (.1, np.nan, 30, 128),
    (.1, .1, 5000, 128), (.1, .1, 3, 128),
    (.1, .1, 30, -1)])
def test_unsupported_configuration_requests_original_fallback(lo, lr, bins, adjustments):
    assert classifier.build_theta_thresholds(lo, lr, bins, adjustments).size == 0


def test_empty_table_and_out_of_range_angles_use_scalar_behavior():
    lo, lr, bins = configuration(.002, .08)
    thresholds = classifier.build_theta_thresholds(lo, lr, bins)
    angles = np.array([-2., -0., 0., math.pi, 2 * math.pi, np.nan, -np.inf, np.inf])
    expected = classify_scalar(angles, lo, lr)
    for table in [thresholds, np.empty(0)]:
        np.testing.assert_array_equal(classify_threshold(angles, lo, lr, table), expected)


def test_exhausted_adjustment_budget_requests_original_fallback():
    lo, lr, bins = configuration(.002, .08)
    assert classifier.build_theta_thresholds(lo, lr, bins, 0).size == 0


def test_extreme_transition_adjustments_fall_back_without_changing_indices():
    lo, lr, bins = configuration(1e-100, 1.)
    thresholds = classifier.build_theta_thresholds(lo, lr, bins)
    assert thresholds.size == 0
    angles = np.array([lo, 2 * lo, 1., math.pi])
    lut, first = classifier.build_theta_bit_lut(lo, thresholds)
    np.testing.assert_array_equal(classify_bit_lut(angles, lo, lr, thresholds, lut, first),
                                  classify_scalar(angles, lo, lr))


def test_empty_bit_lut_uses_binary_fallback():
    lo, lr, bins = configuration(.002, .08)
    thresholds = classifier.build_theta_thresholds(lo, lr, bins)
    angles = np.linspace(0., math.pi, 10001)
    np.testing.assert_array_equal(classify_bit_lut(angles, lo, lr, thresholds, np.empty(0, dtype=np.int64), 0),
                                  classify_scalar(angles, lo, lr))


def test_register_bitcast_matches_numpy_bits_including_signed_zero():
    @njit
    def bits(values):
        result = np.empty(len(values), dtype=np.uint64)
        for i in range(len(values)):
            result[i] = classifier._float64_bits(values[i])
        return result
    rng = np.random.default_rng(903)
    values = np.concatenate([rng.normal(size=1000), [-0., 0., np.nan, np.inf, -np.inf,
                             np.nextafter(0., 1.), np.finfo(np.float64).max]])
    np.testing.assert_array_equal(bits(values), values.view(np.uint64))
