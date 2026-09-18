"""Invariances a sum of segments must obey, on a small cache."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from lighthit import Medium, SolverSettings
from lighthit.cache import BandedResponseCache, CacheGrid, ResponseCache
from lighthit.experimental.cone_segment import ConeSegment, segment_spectrum

MEDIUM = Medium(0.04, 0.05, 0.7, 1.35)
SPEED = 0.299792458
RECEIVERS = np.array([[11.0, 0.0, 13.0], [-9.0, 6.0, 10.0], [4.0, -12.0, 16.0]])


def make_event():
    """Three segments with their own places, directions, times and speeds."""
    return [ConeSegment((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 3.0, 0.999, 1.34, 900.0, 0.0),
            ConeSegment((0.2, 0.1, 3.0), tuple(np.array([0.3, 0.1, 0.95])
                                               / np.linalg.norm([0.3, 0.1, 0.95])),
                        2.0, 0.97, 1.34, 400.0, 1.1),
            ConeSegment((-0.4, 0.5, 1.5), tuple(np.array([-0.5, 0.6, 0.62])
                                                / np.linalg.norm([-0.5, 0.6, 0.62])),
                        1.5, 0.90, 1.34, 250.0, 0.6)]


@pytest.fixture(scope="module")
def cache():
    return BandedResponseCache([ResponseCache.build(
        MEDIUM, SolverSettings(24, 90, 4.0, 0.05, 8),
        CacheGrid.geometric(7.0, 26.0, 18, [0.0, 0.05, 0.2]))])


def event_spectrum(cache, segments, receivers, order=32):
    return sum(segment_spectrum(cache, piece, receivers, longitudinal_order=order)
               for piece in segments)


def moved(segments, matrix=None, shift=None, delay=0.0, scale=1.0):
    matrix = np.eye(3) if matrix is None else matrix
    shift = np.zeros(3) if shift is None else shift
    return [ConeSegment(tuple(matrix @ np.asarray(p.start_m, float) + shift),
                        tuple(matrix @ np.asarray(p.direction, float)),
                        p.length_m, p.beta, p.phase_index,
                        scale * p.photons_per_m, p.start_time_ns + delay)
            for p in segments]


def test_joint_rotation_and_translation_leaves_the_event_unchanged(cache):
    segments = make_event()
    base = event_spectrum(cache, segments, RECEIVERS)
    matrix = Rotation.from_rotvec([0.31, -0.42, 0.65]).as_matrix()
    shift = np.array([12.0, -5.0, 3.0])
    same = event_spectrum(cache, moved(segments, matrix, shift),
                          RECEIVERS @ matrix.T + shift)
    np.testing.assert_allclose(same, base, rtol=2e-11, atol=1e-20)


def test_time_shift_is_a_pure_phase(cache):
    segments = make_event()
    base = event_spectrum(cache, segments, RECEIVERS)
    omega = cache.bands[0].grid.omega_per_ns
    delayed = event_spectrum(cache, moved(segments, delay=23.0), RECEIVERS)
    np.testing.assert_allclose(delayed, base * np.exp(1j * omega * 23.0)[:, None, None],
                               rtol=2e-11, atol=1e-20)


def test_yield_scaling_is_linear(cache):
    segments = make_event()
    base = event_spectrum(cache, segments, RECEIVERS)
    np.testing.assert_allclose(event_spectrum(cache, moved(segments, scale=2.5), RECEIVERS),
                               2.5 * base, rtol=1e-12, atol=1e-20)


def test_splitting_a_segment_adds_up(cache):
    piece = make_event()[0]
    half = piece.length_m / 2
    first = ConeSegment(piece.start_m, piece.direction, half, piece.beta,
                        piece.phase_index, piece.photons_per_m, piece.start_time_ns)
    second = ConeSegment(tuple(np.asarray(piece.start_m, float)
                               + half * np.asarray(piece.direction, float)),
                         piece.direction, half, piece.beta, piece.phase_index,
                         piece.photons_per_m,
                         piece.start_time_ns + half / (piece.beta * SPEED))
    whole = segment_spectrum(cache, piece, RECEIVERS, longitudinal_order=64)
    split = (segment_spectrum(cache, first, RECEIVERS, longitudinal_order=48)
             + segment_spectrum(cache, second, RECEIVERS, longitudinal_order=48))
    scale = np.max(np.abs(whole), axis=(0, 2))
    assert np.max(np.abs(whole - split) / scale[None, :, None]) < 1e-6


def test_receivers_outside_the_cache_raise_instead_of_returning_zero(cache):
    segments = make_event()
    with pytest.raises(ValueError, match="outside the cached radial range"):
        event_spectrum(cache, segments, RECEIVERS * 20.0)


def test_orders_are_separated_and_the_ballistic_one_is_sharp(cache):
    """Order 0 is analytic, so it is either exactly zero or exactly positive."""
    spectrum = event_spectrum(cache, make_event(), RECEIVERS)
    charge = spectrum[0].real
    assert charge.shape == (len(RECEIVERS), 3)
    assert np.all(charge[:, 1] > 0) and np.all(charge[:, 2] > 0)
    assert np.all(charge[:, 0] >= 0)
