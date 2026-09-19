"""ballistic_fast matches the closed-form order-0 sum it replaces."""
import numpy as np
import pytest

pytest.importorskip("numba")

from lighthit.medium import Medium
from lighthit.experimental.g4_source import LightElements
from lighthit.experimental.ballistic_fast import (vectorised_ballistic_fast,
                                                   vectorised_ballistic_bins_fast)


def _reference_ballistic(elements, receivers, medium, cone_cosine):
    """The numpy formula this module replaces, kept local so the test does
    not depend on importing a script (@eq-ballistic-fluence, as written in
    ``scripts/run_shower_moments.py::vectorised_ballistic``)."""
    total = np.zeros(len(receivers))
    sine = np.sqrt(np.maximum(1 - cone_cosine ** 2, 0.0))
    for index, receiver in enumerate(receivers):
        relative = receiver[None, :] - elements.start_m
        along = np.einsum("ij,ij->i", relative, elements.direction)
        impact = np.sqrt(np.maximum(np.einsum("ij,ij->i", relative, relative)
                                    - along ** 2, 1e-300))
        root = along - impact * cone_cosine / sine
        distance = impact / sine
        lit = (root >= 0) & (root < elements.length_m)
        weight = np.where(lit, elements.photons / np.maximum(elements.length_m, 1e-300)
                          * np.exp(-medium.extinction_per_m * distance)
                          / (2 * np.pi * impact * sine), 0.0)
        total[index] = weight.sum()
    return total


def _elements(rng, n):
    start = rng.uniform(-3, 3, size=(n, 3))
    direction = rng.normal(size=(n, 3))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    length = rng.uniform(1e-4, 0.02, size=n)
    photons = rng.uniform(0.1, 50.0, size=n)
    cone_cosine = 0.74 + rng.uniform(-0.01, 0.01, size=n)
    zeros, ones, idx = np.zeros(n), np.ones(n), np.arange(n)
    return LightElements(start, direction, length, photons, cone_cosine,
                          zeros, ones, idx, idx, None, {})


@pytest.fixture
def medium():
    return Medium(absorption_per_m=0.07, scattering_per_m=0.022, g=0.9,
                  group_index=1.36)


def test_matches_the_numpy_formula_on_random_geometry(medium):
    rng = np.random.default_rng(20260919)
    elements = _elements(rng, 4000)
    receivers = rng.uniform(-50, 50, size=(37, 3))
    receivers[:, 2] += 20.0
    reference = _reference_ballistic(elements, receivers, medium, elements.cone_cosine)
    fast = vectorised_ballistic_fast(elements, receivers, medium, elements.cone_cosine)
    scale = max(np.abs(reference).max(), 1e-300)
    assert np.max(np.abs(fast - reference)) / scale < 1e-12


def test_lit_and_dark_edge_cases_agree_exactly(medium):
    start = np.array([[0.0, 0.0, 0.0]])
    direction = np.array([[0.0, 0.0, 1.0]])
    length = np.array([2.0])
    photons = np.array([1000.0])
    cone_cosine = np.array([0.75])
    elements = LightElements(start, direction, length, photons, cone_cosine,
                              np.zeros(1), np.ones(1), np.arange(1), np.arange(1),
                              None, {})
    receivers = np.array([[5.0, 0.0, 1.0], [0.0, 0.0, -5.0], [0.02, 0.0, 1.0]])
    reference = _reference_ballistic(elements, receivers, medium, elements.cone_cosine)
    fast = vectorised_ballistic_fast(elements, receivers, medium, elements.cone_cosine)
    assert np.allclose(reference, fast, rtol=0, atol=1e-14)
    assert (reference[:2] == 0.0).all(), "the two off-cone receivers must carry no ballistic light"
    assert reference[2] > 0.0, "the receiver on the cone must carry some"


def test_receiver_shape_is_validated(medium):
    elements = _elements(np.random.default_rng(1), 5)
    with pytest.raises(ValueError):
        vectorised_ballistic_fast(elements, np.zeros((3, 2)), medium, elements.cone_cosine)


def test_time_bins_keep_exact_arrivals_and_total(medium):
    rng = np.random.default_rng(41)
    elements = _elements(rng, 600)
    # Give every step a nontrivial, linearly varying emission time.
    start_ns = rng.uniform(0.0, 8.0, len(elements))
    end_ns = start_ns + rng.uniform(0.01, 0.2, len(elements))
    from dataclasses import replace
    elements = replace(elements, start_ns=start_ns, end_ns=end_ns)
    receivers = rng.uniform(-20.0, 20.0, size=(7, 3))
    origins = rng.uniform(0.0, 20.0, len(receivers))
    edges = np.linspace(-100.0, 500.0, 121)
    total, bins = vectorised_ballistic_bins_fast(
        elements, receivers, medium, elements.cone_cosine, origins, edges)
    np.testing.assert_allclose(total, vectorised_ballistic_fast(
        elements, receivers, medium, elements.cone_cosine), rtol=2e-14, atol=0)
    # The deliberately wide window contains every lit synthetic arrival.
    np.testing.assert_allclose(bins.sum(axis=1), total, rtol=2e-14, atol=0)


def test_time_bin_inputs_are_validated(medium):
    elements = _elements(np.random.default_rng(2), 5)
    receivers = np.zeros((2, 3))
    with pytest.raises(ValueError):
        vectorised_ballistic_bins_fast(elements, receivers, medium,
                                       elements.cone_cosine, [0.0], [0.0, 1.0])
    with pytest.raises(ValueError):
        vectorised_ballistic_bins_fast(elements, receivers, medium,
                                       elements.cone_cosine, [0.0, 0.0], [1.0, 0.0])
