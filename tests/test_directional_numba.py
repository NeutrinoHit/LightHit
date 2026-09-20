"""The fused directional kernels against their numpy twins.

Every routine in ``lighthit._directional_numba`` and the directional part of
``lighthit.experimental.ballistic_fast`` has a numpy twin that the rest of the
suite already checks against references. What is left to check is that the
fused version computes the same numbers, and that is what this module does.
It is skipped where numba is absent.
"""
import numpy as np
import pytest
from scipy.special import eval_legendre

pytest.importorskip("numba")

from lighthit.ballistic import ballistic_directional                # noqa: E402
from lighthit.cache import CacheGrid                                # noqa: E402
from lighthit.directional import DirectionalCache, directional_response  # noqa: E402
from lighthit.green import SolverSettings                           # noqa: E402
from lighthit.medium import Medium                                  # noqa: E402
from lighthit._directional_numba import PreparedDirectionalKernel   # noqa: E402
from lighthit.experimental.ballistic_fast import (                  # noqa: E402
    ballistic_directional_fast, vectorised_ballistic_fast)
from lighthit.experimental.axial_source import (AxialSource, AxisFrame,  # noqa: E402
                                                channel_index)
from lighthit.experimental.event_moments import real_spherical_harmonics  # noqa: E402
import lighthit as lh                                               # noqa: E402

DEGREE = 6
ALPHA = np.array([4 * np.pi, 1.3, -0.5, 0.2])


def medium():
    return Medium(0.02, 0.05, 0.9, 1.35, 450.0, "test medium")


def settings():
    return SolverSettings(6, DEGREE, 1.5, 0.1, 6, True)


def build_source(omega, seed=23, cells=3, azimuthal_degree=2):
    rng = np.random.default_rng(seed)
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    helper = np.array([0.0, 0.0, 1.0])
    if abs(float(axis @ helper)) > 0.9:
        helper = np.array([1.0, 0.0, 0.0])
    first = np.cross(helper, axis)
    first /= np.linalg.norm(first)
    frame = AxisFrame(np.array([1.0, -2.0, 0.5]), axis, first,
                      np.cross(axis, first))
    kept, channel_degree = channel_index(DEGREE, azimuthal_degree)
    table = (rng.normal(size=(cells, len(kept), len(omega)))
             + 1j * rng.normal(size=(cells, len(kept), len(omega))))
    return AxialSource(frame, rng.normal(size=cells) * 0.3,
                       rng.normal(size=(cells, 2)) * 0.2, table, kept,
                       channel_degree, DEGREE, azimuthal_degree, 3.5, {})


@pytest.mark.parametrize("radial_phase", ["none", "flight"])
def test_fused_apply_matches_the_numpy_definition(radial_phase):
    omega = np.array([0.0, 0.07])
    grid = CacheGrid(np.geomspace(4.0, 30.0, 14), omega)
    cache = DirectionalCache.build(medium(), settings(), grid, 3,
                                   radial_phase=radial_phase)
    source = build_source(omega)
    rng = np.random.default_rng(5)
    centre = source.points_m().mean(axis=0)
    receivers = centre[None, :] + np.array([[9.0, 4.0, -7.0], [-6.0, 11.0, 3.0]])
    looks = rng.normal(size=(2, 3))
    looks /= np.linalg.norm(looks, axis=1)[:, None]
    want = directional_response(cache, source, receivers, looks, ALPHA,
                                source_omega_per_ns=omega)
    prepared = PreparedDirectionalKernel.from_cache(cache)
    for block in (1, 2, 4):
        got = prepared.apply(source, receivers, looks, ALPHA,
                             source_omega_per_ns=omega, receiver_block=block)
        assert np.max(np.abs(got - want)) / np.max(np.abs(want)) < 1e-12


def test_fused_apply_on_a_frequency_subset():
    omega = np.array([0.0, 0.05, 0.11])
    grid = CacheGrid(np.geomspace(4.0, 30.0, 12), omega)
    cache = DirectionalCache.build(medium(), settings(), grid, 2,
                                   radial_phase="flight")
    source = build_source(omega[:1], seed=9)
    centre = source.points_m().mean(axis=0)
    receivers = centre[None, :] + np.array([[8.0, 3.0, -5.0]])
    looks = np.array([[0.0, 0.0, 1.0]])
    alpha = ALPHA[:3]
    want = directional_response(cache, source, receivers, looks, alpha,
                                source_omega_per_ns=omega[:1],
                                frequency_indices=np.array([0]))
    prepared = PreparedDirectionalKernel.from_cache(cache,
                                                    frequency_indices=[0])
    got = prepared.apply(source, receivers, looks, alpha,
                         source_omega_per_ns=omega[:1])
    assert np.max(np.abs(got - want)) / np.max(np.abs(want)) < 1e-12


def test_fused_ballistic_matches_the_numpy_twin():
    track = lh.CherenkovTrack([0, 0, 0], [0, 0, 1], 3.0, beta=0.999)
    elements = track.to_elements(step_m=0.25).field(0)
    cone = elements.cone_cosine
    receivers = np.array([[1.0, 0.0, 2.0], [2.0, 1.0, 4.0], [-3.0, 1.0, 5.0]])
    rng = np.random.default_rng(31)
    looks = rng.normal(size=(3, 3))
    looks /= np.linalg.norm(looks, axis=1)[:, None]
    edges = np.arange(-10.0, 90.0, 5.0)
    origins = np.zeros(3)
    slow = ballistic_directional(elements, receivers, looks, medium(), cone,
                                 ALPHA, time_origin_ns=origins,
                                 relative_edges_ns=edges)
    fast = ballistic_directional_fast(elements, receivers, looks, medium(),
                                      cone, ALPHA, time_origin_ns=origins,
                                      relative_edges_ns=edges)
    assert np.max(np.abs(slow[0] - fast[0])) / np.max(np.abs(fast[0])) < 1e-13
    assert np.max(np.abs(slow[1] - fast[1])) / np.max(np.abs(fast[1])) < 1e-13


def test_fused_ballistic_reduces_to_the_existing_kernel_for_a_constant_response():
    track = lh.CherenkovTrack([0, 0, 0], [0, 0, 1], 3.0, beta=0.999)
    elements = track.to_elements(step_m=0.25).field(0)
    receivers = np.array([[1.0, 0.0, 2.0], [0.5, -0.5, 1.5]])
    looks = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    plain = vectorised_ballistic_fast(elements, receivers, medium(),
                                      elements.cone_cosine)
    constant = 0.37
    scaled, _ = ballistic_directional_fast(elements, receivers, looks, medium(),
                                           elements.cone_cosine,
                                           np.array([4 * np.pi * constant]))
    np.testing.assert_allclose(scaled, constant * plain, rtol=1e-13, atol=0)
