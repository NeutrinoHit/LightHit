"""Properties the joint-moment route must have, on a small synthetic event."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from scipy.special import eval_legendre, roots_legendre

from lighthit import Medium, SolverSettings
from lighthit.cache import CacheGrid, ResponseCache
from lighthit.experimental.event_moments import (BlockPartition, KernelChannels,
                                                 compile_joint_moments,
                                                 direct_response, evaluate_moments,
                                                 monomial_powers,
                                                 real_spherical_harmonics)
from lighthit.experimental.g4_source import (LightElements, SourceContract,
                                             cherenkov_yield_per_m)

MEDIUM = Medium(0.04, 0.05, 0.7, 1.35)
RECEIVERS = np.array([[26.0, 0.0, 6.0], [-18.0, 14.0, -9.0], [5.0, -31.0, 12.0]])


def make_event(count=400, spread=1.5, seed=5, photons=1.0):
    """A cloud of short radiating elements with individual directions and times."""
    rng = np.random.default_rng(seed)
    start = rng.normal(scale=spread, size=(count, 3))
    direction = rng.normal(size=(count, 3))
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    length = rng.uniform(0.01, 0.05, count)
    times = np.abs(start[:, 2]) / 0.25 + rng.uniform(0, 2.0, count)
    return LightElements(start, direction, length,
                         np.full(count, photons), np.full(count, 1 / 1.35),
                         times, times + length / 0.3,
                         np.arange(count), np.zeros(count, np.int64),
                         SourceContract(), {"synthetic": True})


@pytest.fixture(scope="module")
def cache():
    # The radial grid has to be fine compared with the block: the moment route
    # fits the kernel *across* the block, so a coarse spline shows up as fit
    # noise rather than as a small offset.
    return ResponseCache.build(MEDIUM, SolverSettings(16, 60, 2.0, 0.05, 8),
                               CacheGrid.geometric(8.0, 60.0, 160, [0.0, 0.2]))


def test_real_harmonics_are_orthonormal():
    x, weights = roots_legendre(48)
    phi = 2 * np.pi * np.arange(96) / 96
    sine = np.sqrt(1 - x ** 2)
    directions = np.array([[s * np.cos(p), s * np.sin(p), c]
                           for c, s in zip(x, sine) for p in phi])
    measure = np.repeat(weights, len(phi)) * (2 * np.pi / len(phi))
    harmonics = real_spherical_harmonics(5, directions)
    gram = (harmonics * measure[:, None]).T @ harmonics
    np.testing.assert_allclose(gram, np.eye(36), atol=1e-12)


@pytest.mark.parametrize("degree", [0, 3, 7])
def test_real_harmonics_satisfy_the_addition_theorem(degree):
    rng = np.random.default_rng(2)
    first, second = rng.normal(size=(2, 3))
    first, second = first / np.linalg.norm(first), second / np.linalg.norm(second)
    values = [real_spherical_harmonics(degree, vector[None, :])[0]
              for vector in (first, second)]
    start = degree ** 2
    total = 4 * np.pi / (2 * degree + 1) * float(
        values[0][start:start + 2 * degree + 1] @ values[1][start:start + 2 * degree + 1])
    assert total == pytest.approx(eval_legendre(degree, first @ second), abs=1e-12)


def test_monomial_count():
    for degree, count in ((0, 1), (1, 4), (2, 10), (3, 20)):
        assert len(monomial_powers(degree)) == count


def test_a_point_event_needs_no_expansion(cache):
    """With every element at the block centre the degree-0 route is exact."""
    event = make_event(count=60, spread=0.0)
    event = LightElements(event.start_m, event.direction, np.full(len(event), 1e-9),
                          event.photons, event.cone_cosine, event.start_ns,
                          event.start_ns, event.row_index, event.uid,
                          event.contract, event.provenance)
    partition = BlockPartition.single(event)
    kernel = KernelChannels.of(cache, 1, 24, frequencies=[0.0, 0.2])
    moments, powers = compile_joint_moments(event, partition, 24, 0, [0.0, 0.2])
    approximate = evaluate_moments(kernel, moments, powers, partition, RECEIVERS)
    exact = direct_response(kernel, event, RECEIVERS)
    np.testing.assert_allclose(approximate, exact, rtol=2e-8)


def test_higher_degree_helps_a_spread_event(cache):
    event = make_event(count=300, spread=0.3)
    partition = BlockPartition.single(event)
    kernel = KernelChannels.of(cache, 1, 24, frequencies=[0.0])
    exact = direct_response(kernel, event, RECEIVERS)[0]
    errors = []
    for degree in (0, 1, 2):
        moments, powers = compile_joint_moments(event, partition, 24, degree, [0.0])
        value = evaluate_moments(kernel, moments, powers, partition, RECEIVERS)[0]
        errors.append(float(np.max(np.abs(value - exact) / np.abs(exact))))
    assert errors[0] > errors[1] > errors[2]
    assert errors[2] < 1e-4


def test_splitting_by_extent_shrinks_the_largest_block():
    """A bright core inside a sparse halo is what a shower looks like, and it
    is exactly the shape that defeats a photon-weighted median cut."""
    rng = np.random.default_rng(11)
    core = rng.normal(scale=0.05, size=(4000, 3))
    halo = rng.normal(scale=2.0, size=(200, 3))
    points = np.vstack((core, halo))
    photons = np.concatenate((np.full(len(core), 100.0), np.full(len(halo), 1.0)))
    event = LightElements(points, np.tile([0.0, 0.0, 1.0], (len(points), 1)),
                          np.full(len(points), 0.01), photons,
                          np.full(len(points), 1 / 1.35), np.zeros(len(points)),
                          np.full(len(points), 0.03), np.arange(len(points)),
                          np.zeros(len(points), np.int64), SourceContract(), {})
    largest = {}
    for strategy in ("photons", "extent"):
        largest[strategy] = [
            float(np.max(np.linalg.norm(
                BlockPartition.split(event, blocks, strategy=strategy).half_sizes_m,
                axis=1)))
            for blocks in (1, 4, 16, 64)]
    assert largest["extent"] == sorted(largest["extent"], reverse=True)
    assert largest["extent"][-1] < 0.5 * largest["extent"][0]
    assert largest["extent"][-1] < largest["photons"][-1]


def test_an_unknown_split_strategy_is_refused():
    with pytest.raises(ValueError):
        BlockPartition.split(make_event(count=20), 2, strategy="median")


def test_blocks_help_at_fixed_degree(cache):
    event = make_event(count=300, spread=1.2)
    kernel = KernelChannels.of(cache, 1, 24, frequencies=[0.0])
    exact = direct_response(kernel, event, RECEIVERS)[0]
    errors = {}
    for blocks in (1, 4):
        partition = BlockPartition.split(event, blocks)
        moments, powers = compile_joint_moments(event, partition, 24, 1, [0.0])
        value = evaluate_moments(kernel, moments, powers, partition, RECEIVERS)[0]
        errors[blocks] = float(np.max(np.abs(value - exact) / np.abs(exact)))
    assert errors[4] < errors[1]


def test_moments_are_linear_in_the_photon_count(cache):
    event = make_event(count=120, photons=1.0)
    brighter = make_event(count=120, photons=3.0)
    partition = BlockPartition.single(event)
    one, powers = compile_joint_moments(event, partition, 12, 1, [0.0])
    three, _ = compile_joint_moments(brighter, partition, 12, 1, [0.0])
    # Channels that cancel to zero carry no information; compare against the
    # size of the largest coefficient rather than against each entry.
    np.testing.assert_allclose(three, 3 * one, rtol=1e-9,
                               atol=1e-10 * np.abs(one).max())


def test_rotating_event_and_receivers_together_agrees_to_the_expansion_error(cache):
    """Not an identity: the fitting box is axis aligned, so rotating the event
    changes the box and with it the truncation error. The two answers may
    therefore differ at the level of that error, and not more."""
    event = make_event(count=200, spread=1.0)
    matrix = Rotation.from_rotvec([0.4, -0.2, 0.7]).as_matrix()
    moved = event.moved(rotation=matrix)
    kernel = KernelChannels.of(cache, 1, 20, frequencies=[0.0])
    base = BlockPartition.single(event)
    turned = BlockPartition.single(moved)
    first, powers = compile_joint_moments(event, base, 20, 2, [0.0])
    second, _ = compile_joint_moments(moved, turned, 20, 2, [0.0])
    here = evaluate_moments(kernel, first, powers, base, RECEIVERS)
    there = evaluate_moments(kernel, second, powers, turned, RECEIVERS @ matrix.T)
    np.testing.assert_allclose(there, here, rtol=5e-3,
                               atol=1e-9 * np.abs(here).max())


def test_the_receiver_chunk_changes_nothing_but_memory(cache):
    """The contraction batches receivers; the batch size must not be visible."""
    event = make_event(count=200, spread=0.8)
    partition = BlockPartition.split(event, 4)
    kernel = KernelChannels.of(cache, 1, 12, frequencies=[0.0, 0.2])
    moments, powers = compile_joint_moments(event, partition, 12, 2, [0.0, 0.2])
    whole = evaluate_moments(kernel, moments, powers, partition, RECEIVERS)
    for chunk in (1, 2):
        piece = evaluate_moments(kernel, moments, powers, partition, RECEIVERS,
                                 receiver_chunk=chunk)
        np.testing.assert_allclose(piece, whole, rtol=1e-12,
                                   atol=1e-14 * np.abs(whole).max())


def test_time_shift_is_a_pure_phase(cache):
    event = make_event(count=150)
    delayed = event.moved(delay_ns=13.0)
    partition = BlockPartition.single(event)
    omega = [0.0, 0.2]
    base, powers = compile_joint_moments(event, partition, 12, 1, omega)
    shifted, _ = compile_joint_moments(delayed, partition, 12, 1, omega)
    expected = base * np.exp(1j * np.asarray(omega) * 13.0)[None, None, None, :]
    np.testing.assert_allclose(shifted, expected, rtol=1e-9,
                               atol=1e-10 * np.abs(base).max())


def test_cherenkov_yield_threshold_and_scale():
    assert cherenkov_yield_per_m(0.5, 1.35, 400, 500) == 0.0
    value = cherenkov_yield_per_m(1.0, 1.35, 400.0, 500.0)
    expected = (2 * np.pi * 7.2973525693e-3 * (1 - 1 / 1.35 ** 2)
                * (1 / 400e-9 - 1 / 500e-9))
    assert value == pytest.approx(expected, rel=1e-12)
    wider = cherenkov_yield_per_m(1.0, 1.35, 300.0, 600.0)
    assert wider > value
