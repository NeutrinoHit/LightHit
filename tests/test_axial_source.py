"""Properties the compact axial source must have.

The identities here are the ones that broke, or nearly broke, while the module
was written: the emission phase must survive the move onto the grid, the
deposit must preserve the photon count and its first moment, and cutting the
azimuthal channels must be exactly an azimuthal average rather than something
that merely resembles one.
"""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from scipy.special import eval_legendre, roots_legendre

from lighthit import Medium, SolverSettings
from lighthit.cache import CacheGrid, ResponseCache
from lighthit.experimental.axial_source import (VACUUM_M_PER_NS, AxialSource,
                                                AxisFrame, axial_response,
                                                channel_index)
from lighthit.experimental.event_moments import (KernelChannels, direct_response,
                                                 real_spherical_harmonics)
from lighthit.experimental.g4_source import LightElements, SourceContract

MEDIUM = Medium(0.04, 0.05, 0.7, 1.35)
RECEIVERS = np.array([[26.0, 3.0, 6.0], [-18.0, 14.0, -9.0], [5.0, -31.0, 12.0]])


def make_event(count=400, spread=1.2, transverse=0.08, seed=5, omega_span=2.0):
    """A needle: long along z, thin across, with a spread of directions."""
    rng = np.random.default_rng(seed)
    start = np.column_stack((rng.normal(scale=transverse, size=count),
                             rng.normal(scale=transverse, size=count),
                             rng.normal(scale=spread, size=count)))
    direction = rng.normal(scale=0.4, size=(count, 3))
    direction[:, 2] += 1.0
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    length = rng.uniform(0.002, 0.01, count)
    times = (start[:, 2] - start[:, 2].min()) / VACUUM_M_PER_NS + rng.uniform(0, omega_span, count)
    return LightElements(start, direction, length, rng.uniform(1.0, 4.0, count),
                         np.full(count, 1 / 1.35), times, times + length / 0.22,
                         np.arange(count), np.zeros(count, np.int64),
                         SourceContract(), {"synthetic": True})


@pytest.fixture(scope="module")
def cache():
    return ResponseCache.build(MEDIUM, SolverSettings(16, 60, 2.0, 0.05, 8),
                               CacheGrid.geometric(8.0, 60.0, 120, [0.0, 0.2]))


def test_channel_index_counts_the_kept_orders():
    for degree, azimuthal, count in ((8, 0, 9), (8, 1, 25), (8, 8, 81), (32, 4, 277)):
        kept, degrees = channel_index(degree, azimuthal)
        assert len(kept) == count and len(degrees) == count
        assert np.all(np.diff(kept) > 0)


def test_axis_frame_coordinate_order_is_transverse_transverse_longitudinal():
    frame = AxisFrame.of(make_event(count=80))
    vectors = np.array([frame.first, frame.second, frame.axis,
                        2 * frame.first - 3 * frame.second + 5 * frame.axis])
    np.testing.assert_allclose(
        frame.rotate(vectors),
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [2, -3, 5]], atol=2e-15)


def test_the_emission_phase_survives_the_move_onto_the_grid():
    """The identity that a cell-centred front phase would silently break.

    Moving an element to a cell may change its propagation; it must not change
    when it radiated. Reconstructing the residual time from the cell rather
    than the element shifts the emission by (z_cell - z_i)/c0, which is two
    radians per metre at 0.6 rad/ns.
    """
    event = make_event(count=60, spread=2.0)
    omega = [0.0, 0.6]
    for deposit in ("nearest", "linear"):
        source = AxialSource.of(event, 0, omega, cell_m=1.5, deposit=deposit,
                                element_order=1)
        carried = source.channels[:, 0, :].sum(axis=0) * np.exp(
            1j * np.asarray(omega) * source.reference_ns)
        exact = np.array([(event.photons * np.exp(1j * w * event.midpoint_time_ns())).sum()
                          / np.sqrt(4 * np.pi) for w in omega])
        np.testing.assert_allclose(carried, exact, rtol=1e-12)


def test_the_linear_deposit_keeps_the_count_and_the_first_moment():
    event = make_event(count=500)
    frame = AxisFrame.of(event)
    z, transverse, _, _ = frame.coordinates(event)
    local = np.stack((transverse @ frame.first, transverse @ frame.second,
                      z), axis=-1)
    mass = event.photons.sum()
    first = (event.photons[:, None] * local).sum(axis=0)
    source = AxialSource.of(event, 0, [0.0], cell_m=0.5, deposit="linear",
                            element_order=1, frame=frame)
    weights = source.channels[:, 0, 0].real * np.sqrt(4 * np.pi)
    got = np.column_stack((source.transverse_m, source.z_m))
    assert weights.sum() == pytest.approx(mass, rel=1e-12)
    np.testing.assert_allclose((weights[:, None] * got).sum(axis=0), first,
                               atol=1e-9 * mass)


def test_the_nearest_deposit_keeps_the_count_but_not_the_moment():
    event = make_event(count=500)
    frame = AxisFrame.of(event)
    z, _, _, _ = frame.coordinates(event)
    source = AxialSource.of(event, 0, [0.0], cell_m=0.5, deposit="nearest",
                            element_order=1, frame=frame)
    weights = source.channels[:, 0, 0].real * np.sqrt(4 * np.pi)
    assert weights.sum() == pytest.approx(event.photons.sum(), rel=1e-12)
    moved = abs((weights * source.z_m).sum() - (event.photons * z).sum())
    assert moved > 1e-3 * event.photons.sum()


def test_keeping_m_zero_is_the_azimuthal_average():
    """Not a resemblance: the m=0 channel is the average over the azimuth of the
    emission direction about the axis, and a direct quadrature says so."""
    event = make_event(count=40)
    frame = AxisFrame.of(event)
    source = AxialSource.of(event, 6, [0.0], azimuthal_degree=0, cell_m=10.0,
                            deposit="nearest", element_order=1, frame=frame)
    local = frame.rotate(event.direction)
    angles = 2 * np.pi * np.arange(64) / 64
    cosine, sine = np.cos(angles), np.sin(angles)
    for degree in (0, 2, 5):
        spun = np.concatenate([
            np.stack((v[0] * cosine - v[1] * sine, v[0] * sine + v[1] * cosine,
                      np.full(64, v[2])), axis=-1) for v in local])
        harmonics = real_spherical_harmonics(degree, spun)[:, degree ** 2 + degree]
        averaged = harmonics.reshape(len(local), 64).mean(axis=1)
        expected = float((event.photons
                          * eval_legendre(degree, event.cone_cosine)
                          * averaged).sum())
        index = int(np.flatnonzero(source.kept == degree ** 2 + degree)[0])
        assert source.channels[:, index, 0].real.sum() == pytest.approx(expected, rel=1e-10)


def test_truncating_afterwards_matches_projecting_with_fewer_orders():
    event = make_event(count=120)
    frame = AxisFrame.of(event)
    whole = AxialSource.of(event, 6, [0.0, 0.2], azimuthal_degree=6, cell_m=0.5,
                           frame=frame)
    for azimuthal in (0, 2):
        apart = AxialSource.of(event, 6, [0.0, 0.2], azimuthal_degree=azimuthal,
                               cell_m=0.5, frame=frame)
        np.testing.assert_allclose(whole.truncated(azimuthal).channels,
                                   apart.channels, rtol=1e-12,
                                   atol=1e-12 * np.abs(apart.channels).max())


def test_adding_orders_after_the_projection_is_refused():
    source = AxialSource.of(make_event(count=20), 4, [0.0], azimuthal_degree=1,
                            cell_m=1.0)
    with pytest.raises(ValueError):
        source.truncated(3)


def test_an_unknown_deposit_is_refused():
    with pytest.raises(ValueError):
        AxialSource.of(make_event(count=20), 2, [0.0], deposit="spline")


def test_a_kernel_above_the_projected_degree_is_refused(cache):
    source = AxialSource.of(make_event(count=20), 4, [0.0], cell_m=1.0)
    kernel = KernelChannels.of(cache, 1, 8, frequencies=[0.0])
    with pytest.raises(ValueError):
        axial_response(kernel, source, RECEIVERS)


def test_the_response_converges_to_the_element_sum(cache):
    """Refining the cells must walk the compact answer onto the element sum."""
    event = make_event(count=600)
    frame = AxisFrame.of(event)
    kernel = KernelChannels.of(cache, 1, 12, frequencies=[0.0, 0.2])
    exact = direct_response(kernel, event, RECEIVERS)
    errors = []
    for cell in (0.4, 0.2, 0.1):
        source = AxialSource.of(event, 12, [0.0, 0.2], azimuthal_degree=12,
                                cell_m=cell, frame=frame, deposit="linear")
        value = axial_response(kernel, source, RECEIVERS)
        errors.append(float(np.max(np.abs(value - exact) / np.abs(exact))))
    assert errors[0] > errors[1] > errors[2]
    assert errors[2] < 1e-3


def test_the_linear_deposit_beats_the_nearest_one(cache):
    event = make_event(count=600)
    frame = AxisFrame.of(event)
    kernel = KernelChannels.of(cache, 1, 12, frequencies=[0.0, 0.2])
    exact = direct_response(kernel, event, RECEIVERS)
    got = {}
    for deposit in ("nearest", "linear"):
        source = AxialSource.of(event, 12, [0.0, 0.2], azimuthal_degree=12,
                                cell_m=0.3, frame=frame, deposit=deposit)
        value = axial_response(kernel, source, RECEIVERS)
        got[deposit] = float(np.max(np.abs(value - exact) / np.abs(exact)))
    assert got["linear"] < got["nearest"] / 3


def test_the_receiver_block_changes_nothing(cache):
    event = make_event(count=200)
    source = AxialSource.of(event, 10, [0.0, 0.2], azimuthal_degree=3, cell_m=0.3)
    kernel = KernelChannels.of(cache, 1, 10, frequencies=[0.0, 0.2])
    whole = axial_response(kernel, source, RECEIVERS)
    for budget in (2 ** 12, 2 ** 20):
        piece = axial_response(kernel, source, RECEIVERS, budget_bytes=budget)
        np.testing.assert_allclose(piece, whole, rtol=1e-12,
                                   atol=1e-14 * np.abs(whole).max())


def test_moving_event_and_receivers_together_agrees_to_the_cell_error(cache):
    """Not an identity, and worth saying why.

    The axis follows the event, so the longitudinal direction is covariant. The
    two transverse directions are not: they are completed from a fixed lab
    vector, and a shower's transverse spread is nearly isotropic, so there is
    no stable pair to take from the event itself. A rigid move therefore lands
    the cells differently and the two answers differ at the level of the cell
    error -- which is what this checks, by refining the cells and watching the
    difference fall.
    """
    event = make_event(count=300)
    matrix = Rotation.from_rotvec([0.4, -0.2, 0.7]).as_matrix()
    shift = np.array([3.0, -2.0, 5.0])
    moved = event.moved(rotation=matrix, translation=shift)
    kernel = KernelChannels.of(cache, 1, 10, frequencies=[0.0, 0.2])
    gaps = []
    for cell in (0.4, 0.2, 0.1):
        here = axial_response(kernel, AxialSource.of(event, 10, [0.0, 0.2],
                                                     azimuthal_degree=10, cell_m=cell),
                              RECEIVERS)
        there = axial_response(kernel, AxialSource.of(moved, 10, [0.0, 0.2],
                                                      azimuthal_degree=10, cell_m=cell),
                               RECEIVERS @ matrix.T + shift)
        gaps.append(float(np.max(np.abs(there - here) / np.abs(here))))
    assert gaps[0] > gaps[1] > gaps[2]
    assert gaps[2] < 3e-4


def test_a_time_shift_is_a_pure_phase(cache):
    event = make_event(count=200)
    omega = np.array([0.0, 0.2])
    source = AxialSource.of(event, 8, omega, azimuthal_degree=2, cell_m=0.3)
    delayed = AxialSource.of(event.moved(delay_ns=11.0), 8, omega,
                             azimuthal_degree=2, cell_m=0.3)
    kernel = KernelChannels.of(cache, 1, 8, frequencies=omega)
    here = axial_response(kernel, source, RECEIVERS)
    there = axial_response(kernel, delayed, RECEIVERS)
    np.testing.assert_allclose(there, here * np.exp(1j * omega * 11.0)[:, None],
                               rtol=1e-10, atol=1e-12 * np.abs(here).max())
