"""The source contract, and the reader that feeds it.

The stored showers are private inputs, so the tests that need a file run only
when one is pointed at through ``LIGHTHIT_G4_FILE``. Everything that can be
checked without one is checked always.
"""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from lighthit.experimental.g4_source import (LightElements, SourceContract,
                                             cherenkov_yield_per_m, load_event)

def make_elements(count=50, seed=3):
    rng = np.random.default_rng(seed)
    start = rng.normal(scale=1.0, size=(count, 3))
    direction = rng.normal(size=(count, 3))
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    length = rng.uniform(0.01, 0.1, count)
    times = rng.uniform(0, 5, count)
    return LightElements(start, direction, length, rng.uniform(1, 10, count),
                         np.full(count, 1 / 1.35), times, times + 0.1,
                         np.arange(count), np.zeros(count, np.int64),
                         SourceContract(), {"synthetic": True})


def test_yield_is_zero_below_threshold_and_grows_with_the_band():
    assert cherenkov_yield_per_m(1 / 1.35 - 1e-6, 1.35, 400, 500) == 0.0
    narrow = cherenkov_yield_per_m(1.0, 1.35, 440.0, 460.0)
    wide = cherenkov_yield_per_m(1.0, 1.35, 400.0, 500.0)
    assert 0 < narrow < wide


def test_yield_matches_frank_tamm():
    beta, index = 0.95, 1.33
    value = cherenkov_yield_per_m(beta, index, 350.0, 600.0)
    expected = (2 * np.pi * 7.2973525693e-3 * (1 - 1 / (beta * index) ** 2)
                * (1 / 350e-9 - 1 / 600e-9))
    assert value == pytest.approx(expected, rel=1e-12)


def test_a_reversed_band_is_refused():
    with pytest.raises(ValueError):
        cherenkov_yield_per_m(1.0, 1.35, 500.0, 400.0)


def test_moving_an_event_is_rigid():
    elements = make_elements()
    matrix = Rotation.from_rotvec([0.2, 0.5, -0.3]).as_matrix()
    shift = np.array([4.0, -2.0, 7.0])
    moved = elements.moved(rotation=matrix, translation=shift, delay_ns=11.0)
    np.testing.assert_allclose(moved.start_m, elements.start_m @ matrix.T + shift, atol=1e-12)
    np.testing.assert_allclose(moved.length_m, elements.length_m)
    np.testing.assert_allclose(moved.photons, elements.photons)
    np.testing.assert_allclose(moved.start_ns, elements.start_ns + 11.0)
    np.testing.assert_allclose(np.linalg.norm(moved.direction, axis=1), 1.0, atol=1e-12)
    assert moved.extent_m == pytest.approx(elements.extent_m, rel=1e-12)


def test_an_improper_rotation_is_refused():
    with pytest.raises(ValueError):
        make_elements().moved(rotation=np.diag([1.0, 1.0, 2.0]))


def test_centroid_is_photon_weighted():
    elements = make_elements()
    weight = elements.photons / elements.photons.sum()
    expected = (weight[:, None] * elements.midpoints_m).sum(axis=0)
    np.testing.assert_allclose(elements.centroid_m, expected, atol=1e-12)


def test_reading_a_stored_event_keeps_the_provenance(stored_g4_file):
    elements = load_event(stored_g4_file, 5)
    summary = elements.summary()
    assert summary["elements"] > 0
    assert summary["photons"] > 0
    assert summary["contract"]["emission_line"].startswith("chord")
    assert summary["true_path_over_chord_median"] >= 1.0 - 1e-9
    assert elements.row_index.max() < summary["rows"]
    assert np.all(elements.cone_cosine > 0) and np.all(elements.cone_cosine <= 1)


def test_the_band_scales_the_photon_count_but_not_the_geometry(stored_g4_file):
    narrow = load_event(stored_g4_file, 5, SourceContract(wavelength_low_nm=440,
                                                          wavelength_high_nm=460))
    wide = load_event(stored_g4_file, 5, SourceContract(wavelength_low_nm=400,
                                                        wavelength_high_nm=500))
    assert len(narrow) == len(wide)
    np.testing.assert_allclose(narrow.start_m, wide.start_m)
    ratio = wide.photons.sum() / narrow.photons.sum()
    expected = ((1 / 400e-9 - 1 / 500e-9) / (1 / 440e-9 - 1 / 460e-9))
    assert ratio == pytest.approx(expected, rel=1e-9)
