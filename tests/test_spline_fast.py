"""PreparedMultipoles is a numerically exact, duck-typed drop-in for the
radial-spline half of ResponseCache -- checked at every call site that uses
it: moments_at itself, KernelChannels, direct_response and
cone_segment.segment_spectrum.
"""
import numpy as np
import pytest

pytest.importorskip("numba")

from lighthit import SolverSettings, synthetic_medium
from lighthit.cache import CacheGrid, ResponseCache
from lighthit.experimental.spline_fast import PreparedMultipoles
from lighthit.experimental.event_moments import KernelChannels, direct_response
from lighthit.experimental.g4_source import LightElements
from lighthit.experimental import cone_segment as cs


def _small_cache():
    medium = synthetic_medium()
    omega = np.linspace(0.0, 0.3, 5)
    settings = SolverSettings(12, 6, 6.0, 0.04, 10)
    grid = CacheGrid.geometric(3.0, 60.0, 20, omega)
    return ResponseCache.build(medium, settings, grid)


CACHE = _small_cache()
PREPARED = PreparedMultipoles.of(CACHE)


@pytest.fixture
def cache():
    return CACHE


@pytest.fixture
def prepared():
    return PREPARED


@pytest.fixture
def radii():
    return np.random.default_rng(7).uniform(5.0, 50.0, size=13)


def test_moments_at_full_degree_and_frequency(cache, prepared, radii):
    reference = cache.moments_at(radii)
    fast = prepared.moments_at(radii)
    assert fast.shape == reference.shape
    scale = max(np.abs(reference).max(), 1e-300)
    assert np.max(np.abs(fast - reference)) / scale < 1e-10


def test_moments_at_narrowed_degree(cache, prepared, radii):
    reference = cache.moments_at(radii, degrees=4)
    fast = prepared.moments_at(radii, degrees=4)
    scale = max(np.abs(reference).max(), 1e-300)
    assert np.max(np.abs(fast - reference)) / scale < 1e-10


def test_moments_at_every_single_frequency_index(cache, prepared, radii):
    for index in range(len(cache.grid.omega_per_ns)):
        reference = cache.moments_at(radii, frequency_index=index)
        fast = prepared.moments_at(radii, frequency_index=index)
        scale = max(np.abs(reference).max(), 1e-300)
        assert np.max(np.abs(fast - reference)) / scale < 1e-10


def test_kernel_channels_accept_prepared_multipoles_unchanged(cache, prepared, radii):
    reference = KernelChannels.of(cache, 1, 6)
    fast = KernelChannels.of(prepared, 1, 6)
    scale_val = reference.multipoles(radii)
    fast_val = fast.multipoles(radii)
    scale = max(np.abs(scale_val).max(), 1e-300)
    assert np.max(np.abs(fast_val - scale_val)) / scale < 1e-10


def test_direct_response_accepts_a_prepared_kernel(cache, prepared):
    rng = np.random.default_rng(11)
    n = 200
    start = rng.uniform(-1, 1, size=(n, 3))
    direction = rng.normal(size=(n, 3))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    length = rng.uniform(1e-4, 0.02, size=n)
    photons = rng.uniform(0.1, 10.0, size=n)
    cone_cosine = np.full(n, 0.74)
    start_ns = np.zeros(n)
    end_ns = start_ns + 0.01
    elements = LightElements(start, direction, length, photons, cone_cosine,
                              start_ns, end_ns, np.arange(n), np.arange(n), None, {})
    receivers = rng.uniform(10.0, 40.0, size=(4, 3))
    k_ref = KernelChannels.of(cache, 1, 6)
    k_fast = KernelChannels.of(prepared, 1, 6)
    reference = direct_response(k_ref, elements, receivers)
    fast = direct_response(k_fast, elements, receivers)
    scale = max(np.abs(reference).max(), 1e-300)
    assert np.max(np.abs(fast - reference)) / scale < 1e-10


def test_cone_segment_spectrum_accepts_a_prepared_cache(cache, prepared):
    rng = np.random.default_rng(13)
    segment = cs.ConeSegment(start_m=(0., 0., 0.), direction=(0., 0., 1.),
                             length_m=2.0, beta=0.99, phase_index=1.35,
                             photons_per_m=500.0)
    positions = rng.uniform(10.0, 30.0, size=(5, 3))
    reference = cs.segment_spectrum(cache, segment, positions, longitudinal_order=8)
    fast = cs.segment_spectrum(prepared, segment, positions, longitudinal_order=8)
    scale = max(np.abs(reference).max(), 1e-300)
    assert np.max(np.abs(fast - reference)) / scale < 1e-10


def test_banded_cache_is_refused_with_a_clear_message(cache):
    class FakeBanded:
        bands = [cache]
    with pytest.raises(ValueError, match="BandedResponseCache"):
        PreparedMultipoles.of(FakeBanded())
