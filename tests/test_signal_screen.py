import numpy as np
import pytest

from lighthit import synthetic_medium
from lighthit.experimental.g4_source import LightElements, SourceContract
from lighthit.experimental.signal_screen import (isotropic_axial_proxy,
                                                  screening_candidates)


def elements():
    start = np.array([[0., 0., -1.], [0.1, 0., 0.], [-0.1, 0., 1.]])
    direction = np.tile([0., 0., 1.], (3, 1))
    length = np.full(3, 0.1)
    photons = np.array([10., 20., 30.])
    times = np.arange(3.)
    return LightElements(start, direction, length, photons, np.full(3, .75),
                         times, times + .1, np.arange(3), np.arange(3),
                         SourceContract(), {"synthetic": True})


def test_proxy_is_positive_and_falls_with_distance():
    value = isotropic_axial_proxy(
        elements(), [[3., 0., 0.], [30., 0., 0.]], synthetic_medium(), bins=2)
    assert value[0] > value[1] > 0


def test_proxy_is_rigid_motion_invariant():
    source = elements()
    receivers = np.array([[3., 2., 4.], [-5., 1., 7.]])
    angle = .7
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0.],
                         [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
    shift = np.array([8., -3., 2.])
    first = isotropic_axial_proxy(source, receivers, synthetic_medium())
    moved = source.moved(rotation=rotation, translation=shift)
    second = isotropic_axial_proxy(
        moved, receivers @ rotation.T + shift, synthetic_medium())
    np.testing.assert_allclose(second, first, rtol=2e-14)


def test_candidates_include_ballistic_and_safety_factor():
    mask, estimate = screening_candidates(
        [1., 0.], [.01, .2], threshold_pe=.01,
        effective_area_m2=.05, efficiency=.2, safety_factor=10)
    np.testing.assert_array_equal(mask, [True, True])
    np.testing.assert_allclose(estimate, [.011, .02])


@pytest.mark.parametrize("kwargs", [
    {"threshold_pe": -1}, {"effective_area_m2": 0},
    {"efficiency": 2}, {"safety_factor": .5},
])
def test_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        screening_candidates([0.], [0.], **kwargs)
