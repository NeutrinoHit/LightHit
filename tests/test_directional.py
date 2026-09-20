"""The exact directional-OM kernel: algebra, cache, apply and references."""
from dataclasses import replace

import numpy as np
import pytest
from scipy.special import eval_legendre

import lighthit as lh
from lighthit.ballistic import acceptance_from_coefficients, ballistic_directional
from lighthit.cache import CacheGrid, ResponseCache, acceptance_coefficients
from lighthit.directional import (CouplingTable, DirectionalCache,
                                  acceptance_bandwidth, clebsch_gordan,
                                  directional_response, real_rotation_rows,
                                  rotation_rows_explicit)
from lighthit.green import SolverSettings
from lighthit.medium import Medium
from lighthit.experimental import directional_reference as ref
from lighthit.experimental.axial_source import (AxialSource, AxisFrame,
                                                channel_index)
from lighthit.experimental.event_moments import real_spherical_harmonics

DEGREE = 6
ACCEPTANCE_DEGREE = 3
ALPHA = np.array([4 * np.pi, 1.3, -0.5, 0.2])


def medium():
    return Medium(0.02, 0.05, 0.9, 1.35, 450.0, "test medium")


def settings():
    return SolverSettings(DEGREE, DEGREE, 2.0, 0.1, 6, True)


@pytest.fixture(scope="module")
def grid():
    return CacheGrid(np.geomspace(4.0, 40.0, 12), np.array([0.0, 0.07]))


@pytest.fixture(scope="module")
def cache(grid):
    return DirectionalCache.build(medium(), settings(), grid, ACCEPTANCE_DEGREE,
                                  radial_phase="flight")


@pytest.fixture(scope="module")
def isotropic_cache(grid):
    return DirectionalCache.build(medium(), settings(), grid, 0,
                                  radial_phase="flight")


@pytest.fixture(scope="module")
def legacy_cache(grid):
    return ResponseCache.build(medium(), settings(), grid, radial_phase="flight")


def axis_frame(axis, centre=(0.0, 0.0, 0.0)):
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    helper = np.array([0.0, 0.0, 1.0])
    if abs(float(axis @ helper)) > 0.9:
        helper = np.array([1.0, 0.0, 0.0])
    first = np.cross(helper, axis)
    first /= np.linalg.norm(first)
    return AxisFrame(np.asarray(centre, float), axis, first, np.cross(axis, first))


def cone_source(frame, omega, cells=1, seed=0, azimuthal_degree=DEGREE):
    """A compiled source whose channels are those of Cherenkov cone elements."""
    rng = np.random.default_rng(seed)
    kept, channel_degree = channel_index(DEGREE, azimuthal_degree)
    table = np.zeros((cells, len(kept), len(omega)), complex)
    directions = rng.normal(size=(cells, 3))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    local = frame.rotate(directions)
    harmonics = real_spherical_harmonics(DEGREE, local, azimuthal_degree)
    cosines = rng.uniform(0.5, 0.9, cells)
    weights = rng.uniform(0.5, 2.0, cells)
    for cell in range(cells):
        legendre = eval_legendre(channel_degree, cosines[cell])
        for w in range(len(omega)):
            table[cell, :, w] = (weights[cell] * legendre * harmonics[cell]
                                 * np.exp(1j * omega[w] * 0.3 * cell))
    z = rng.normal(size=cells) * 0.4
    transverse = rng.normal(size=(cells, 2)) * 0.2
    return AxialSource(frame, z, transverse, table, kept, channel_degree,
                       DEGREE, azimuthal_degree, 0.0, {})


def lab_moments(source, cell=0, frequency=0):
    """The source channels of one cell, rotated back into the lab frame."""
    frame = source.frame
    rotation = np.stack((frame.first, frame.second, frame.axis), axis=1)
    flat = np.zeros((DEGREE + 1) ** 2)
    values = source.channels[cell, :, frequency].real
    for index, position in enumerate(source.kept):
        flat[position] = values[index]
    # rebuild the distribution on a quadrature grid and reproject in the lab
    nodes, weights = ref.spherical_quadrature(2 * DEGREE + 2)
    local = nodes @ rotation                      # lab -> frame components
    frame_harmonics = real_spherical_harmonics(DEGREE, local)
    field = frame_harmonics @ flat
    lab_harmonics = real_spherical_harmonics(DEGREE, nodes)
    return lab_harmonics.T @ (weights * field)


# ---------------------------------------------------------------- algebra

def test_clebsch_gordan_known_values():
    assert clebsch_gordan(1, 0, 1, 0, 0, 0) == pytest.approx(-1 / np.sqrt(3))
    assert clebsch_gordan(1, 1, 1, -1, 2, 0) == pytest.approx(1 / np.sqrt(6))
    assert clebsch_gordan(2, 0, 2, 0, 1, 0) == 0.0          # forbidden by parity
    assert clebsch_gordan(0, 0, 3, 1, 3, 1) == pytest.approx(1.0)


def test_coupling_table_obeys_the_parity_rule():
    table = CouplingTable.build(8, 3)
    assert np.all((table.triple_l + table.triple_lambda + table.triple_J) % 2 == 0)
    assert np.all(table.triple_J >= np.abs(table.triple_l - table.triple_lambda))
    assert np.all(table.triple_J <= table.triple_l + table.triple_lambda)
    assert table.pairs == 10                       # (lambda, |mu|), lambda <= 3
    # the stored count is the ten-per-degree of the design note, not sixteen
    assert table.triples == sum(min(lam, l) + 1 if l >= lam else
                                (min(lam, l) + 1)
                                for lam in range(4) for l in range(9))


def test_parity_rule_is_a_property_of_the_coefficients_not_a_filter():
    """A triple of odd parity really does have vanishing coupling."""
    for lam, l, J in ((1, 1, 1), (2, 3, 2), (3, 2, 2)):
        assert (l + lam + J) % 2 == 1
        folded = sum((-1.0) ** nu * clebsch_gordan(lam, nu, l, -nu, J, 0)
                     for nu in range(-min(lam, l), min(lam, l) + 1))
        assert abs(folded) < 1e-14


@pytest.mark.parametrize("degree,max_out,max_in", [(8, 3, 4), (24, 3, 4), (10, 2, 6)])
def test_explicit_real_rotation_matches_the_complex_construction(degree, max_out, max_in):
    rng = np.random.default_rng(3)
    theta = rng.uniform(0.01, np.pi - 0.01, 6)
    phi = rng.uniform(-np.pi, np.pi, 6)
    complex_route = real_rotation_rows(degree, max_out, max_in, np.cos(theta), phi)
    explicit = rotation_rows_explicit(degree, max_out, max_in, np.cos(theta), phi)
    assert np.max(np.abs(complex_route - explicit)) < 1e-13


def test_rotation_rows_carry_harmonics_between_frames():
    degree = 10
    rng = np.random.default_rng(11)
    theta, phi = 0.9, -2.1
    rows = real_rotation_rows(degree, 3, degree, np.array([np.cos(theta)]),
                              np.array([phi]))[0]
    ct, st, cf, sf = np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi)
    rotation = np.array([[ct * cf, -sf, st * cf], [ct * sf, cf, st * sf],
                         [-st, 0.0, ct]])
    vector = rng.normal(size=3)
    vector /= np.linalg.norm(vector)
    turned = real_spherical_harmonics(degree, (rotation.T @ vector)[None, :])[0]
    straight = real_spherical_harmonics(degree, vector[None, :])[0]
    for l in range(degree + 1):
        for nu in range(-min(l, 3), min(l, 3) + 1):
            value = sum(rows[l, nu + 3, mu + degree] * straight[l * l + l + mu]
                        for mu in range(-l, l + 1))
            assert value == pytest.approx(turned[l * l + l + nu], abs=1e-12)


# ------------------------------------------------------------------ cache

def test_lambda_zero_cache_is_the_existing_multipole_cache(isotropic_cache,
                                                           legacy_cache):
    ell = np.arange(DEGREE + 1)
    expected = legacy_cache.moments / np.sqrt(2 * ell + 1)[None, None, :, None]
    got = isotropic_cache.coefficients[:, :, :, 0, :]
    assert np.max(np.abs(got - expected)) / np.max(np.abs(expected)) < 1e-12


def test_directional_cache_contains_the_isotropic_one_unchanged(cache,
                                                                isotropic_cache):
    """No new cost is paid by the degrees the old kernel already had."""
    got = cache.coefficients[:, :, :, 0, :]
    want = isotropic_cache.coefficients[:, :, :, 0, :]
    assert np.max(np.abs(got - want)) / np.max(np.abs(want)) < 1e-13


def test_cache_round_trip(tmp_path, cache):
    path = tmp_path / "directional.npz"
    cache.save(path)
    loaded = DirectionalCache.load(path)
    np.testing.assert_array_equal(loaded.coefficients, cache.coefficients)
    assert loaded.acceptance_degree == cache.acceptance_degree
    assert loaded._metadata()["detector_angular_model"] == "exact_m_blocks"


def test_cache_refuses_out_of_range_radii(cache):
    with pytest.raises(ValueError, match="outside the cached range"):
        cache.coefficients_at(np.array([1.0]))


# ------------------------------------------------------------------ apply

def test_constant_acceptance_reduces_to_the_m_zero_contraction(isotropic_cache,
                                                               legacy_cache):
    """Isotropic acceptance gives strictly the old result, to the last bit."""
    omega = isotropic_cache.grid.omega_per_ns
    frame = axis_frame([0.3, -0.4, 0.86])
    source = cone_source(frame, omega, cells=3, seed=5, azimuthal_degree=2)
    receivers = np.array([[6.0, 4.0, 14.0], [-9.0, 2.0, 5.0]])
    looks = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    got = directional_response(isotropic_cache, source, receivers, looks,
                               np.array([4 * np.pi]), source_omega_per_ns=omega)
    points = source.points_m()
    expected = np.zeros_like(got)
    for index, receiver in enumerate(receivers):
        vectors = receiver[None, :] - points
        radii = np.linalg.norm(vectors, axis=1)
        harmonics = real_spherical_harmonics(DEGREE, frame.rotate(vectors))
        moments = legacy_cache.moments_at(radii)
        for column, (flat, l) in enumerate(zip(source.kept, source.channel_degree)):
            weight = harmonics[:, flat] * (4 * np.pi / (2 * l + 1))
            expected[:, index] += np.einsum(
                "p,wpo,pw->wo", weight, moments[:, :, l, :],
                source.channels[:, column, :])
    assert np.max(np.abs(got - expected)) / np.max(np.abs(expected)) < 1e-12


def test_production_matches_the_khat_quadrature_reference(cache):
    omega = cache.grid.omega_per_ns
    frame = axis_frame([0.0, 0.0, 1.0])
    source = cone_source(frame, omega, cells=1, seed=7)
    moments = lab_moments(source)
    direction = np.array([0.4, -0.3, 0.866])
    displacement = direction / np.linalg.norm(direction) * cache.grid.radii_m[5]
    look = np.array([0.2, 0.9, -0.39])
    look /= np.linalg.norm(look)
    got = directional_response(cache, source, displacement[None, :] + source.points_m()[0],
                               look[None, :], ALPHA, source_omega_per_ns=omega)
    for index, frequency in enumerate(omega):
        want = ref.khat_quadrature_response(
            medium(), settings(), frequency, moments, displacement, look, ALPHA,
            acceptance_degree=ACCEPTANCE_DEGREE, polar_order=30)
        assert (np.max(np.abs(got[index, 0] - want)) / np.max(np.abs(want))) < 1e-11


def test_khat_reference_agrees_with_a_dense_lab_frame_solve():
    """Neither route below knows that the problem separates in m."""
    strong = Medium(0.30, 0.06, 0.9, 1.35, 450.0, "strongly attenuating")
    degree = 10
    small = SolverSettings(4, degree, 0.4, 0.05, 6, True)
    rng = np.random.default_rng(17)
    moments = np.zeros((degree + 1) ** 2)
    moments[:25] = rng.normal(size=25) * 0.4
    moments[0] = 1.0
    displacement = np.array([1.7, -0.9, 2.2])
    look = rng.normal(size=3)
    look /= np.linalg.norm(look)
    for omega in (0.0, 0.08):
        blocks = ref.khat_quadrature_response(
            strong, small, omega, moments, displacement, look, ALPHA,
            acceptance_degree=3, polar_order=24)
        dense = ref.dense_angular_reference(
            strong, small, omega, moments, displacement, look, ALPHA,
            acceptance_degree=3, polar_order=24, quadrature_order=32)
        assert np.max(np.abs(blocks - dense)) / np.max(np.abs(dense)) < 1e-8


def test_coincident_axes_match_the_axisymmetric_formula(cache):
    """Source axis, module axis and displacement on one line."""
    omega = cache.grid.omega_per_ns
    axis = np.array([0.0, 0.0, 1.0])
    frame = axis_frame(axis)
    kept, channel_degree = channel_index(DEGREE, DEGREE)
    values = np.zeros(len(kept))
    m0 = np.array([1.0, 0.7, 0.45, 0.2, 0.08, 0.03, 0.01])
    for index, (flat, l) in enumerate(zip(kept, channel_degree)):
        if flat == l * l + l:
            values[index] = m0[l]
    table = np.repeat(values[None, :, None], len(omega), axis=2).astype(complex)
    source = AxialSource(frame, np.zeros(1), np.zeros((1, 2)), table, kept,
                         channel_degree, DEGREE, DEGREE, 0.0, {})
    radius = cache.grid.radii_m[4]
    got = directional_response(cache, source, (radius * axis)[None, :],
                               axis[None, :], ALPHA, source_omega_per_ns=omega)
    for index, frequency in enumerate(omega):
        want = ref.axisymmetric_reference(medium(), settings(), frequency, m0,
                                          radius, ALPHA,
                                          acceptance_degree=ACCEPTANCE_DEGREE)
        assert np.max(np.abs(got[index, 0] - want)) / np.max(np.abs(want)) < 1e-9


def test_joint_rotation_of_source_detector_and_geometry_preserves_the_answer(cache):
    omega = cache.grid.omega_per_ns
    frame = axis_frame([0.2, 0.5, 0.84])
    source = cone_source(frame, omega, cells=2, seed=13)
    receivers = np.array([[7.0, -3.0, 11.0], [-4.0, 8.0, 6.0]])
    looks = np.array([[0.0, 0.0, 1.0], [0.6, -0.8, 0.0]])
    base = directional_response(cache, source, receivers, looks, ALPHA,
                                source_omega_per_ns=omega)
    angle = 0.7
    axis = np.array([1.0, 2.0, -0.5])
    axis /= np.linalg.norm(axis)
    cross = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
    rotation = (np.eye(3) + np.sin(angle) * cross
                + (1 - np.cos(angle)) * cross @ cross)
    turned_frame = AxisFrame(rotation @ frame.centre_m, rotation @ frame.axis,
                             rotation @ frame.first, rotation @ frame.second)
    turned = replace(source, frame=turned_frame)
    moved = directional_response(cache, turned, receivers @ rotation.T,
                                 looks @ rotation.T, ALPHA,
                                 source_omega_per_ns=omega)
    assert np.max(np.abs(moved - base)) / np.max(np.abs(base)) < 1e-11


def test_turning_the_module_alone_changes_the_answer_as_the_quadrature_says(cache):
    omega = cache.grid.omega_per_ns
    frame = axis_frame([0.0, 0.0, 1.0])
    source = cone_source(frame, omega, cells=1, seed=21)
    moments = lab_moments(source)
    displacement = np.array([3.0, 4.0, 10.0])
    displacement = displacement / np.linalg.norm(displacement) * cache.grid.radii_m[6]
    position = displacement[None, :] + source.points_m()[0]
    answers = []
    for look in (np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, -1.0]),
                 np.array([1.0, 0.0, 0.0])):
        got = directional_response(cache, source, position, look[None, :], ALPHA,
                                   source_omega_per_ns=omega)
        want = ref.khat_quadrature_response(
            medium(), settings(), omega[0], moments, displacement, look, ALPHA,
            acceptance_degree=ACCEPTANCE_DEGREE, polar_order=30)
        assert np.max(np.abs(got[0, 0] - want)) / np.max(np.abs(want)) < 1e-11
        answers.append(got[0, 0, 1])
    assert abs(answers[0] - answers[1]) > 1e-3 * abs(answers[0])
    assert abs(answers[0] - answers[2]) > 1e-3 * abs(answers[0])


# ------------------------------------------------------------- acceptance

def test_acceptance_bandwidth_finds_a_cubic_and_flags_a_clipped_one():
    coefficients = np.array([0.6, 0.4, 0.2, 0.1])

    def cubic(x):
        x = np.asarray(x, float)
        return sum(coefficients[n] * x ** n for n in range(4))

    degree, alpha, residual = acceptance_bandwidth(cubic)
    assert degree == 3 and residual < 1e-11
    assert np.all(np.abs(alpha[4:]) < 1e-11 * np.max(np.abs(alpha)))

    def clipped(x):
        return np.maximum(cubic(x) - 0.7, 0.0)

    # A response clipped inside [-1, 1] is not band-limited: the coefficients
    # never fall below the tolerance, so the measured bandwidth runs into the
    # probe range instead of settling at three.
    degree, alpha, _ = acceptance_bandwidth(clipped, max_degree=40)
    assert degree == 40
    assert np.abs(alpha[10]) > 1e-6 * np.max(np.abs(alpha))


def test_acceptance_rebuild_matches_the_callable_for_a_band_limited_response():
    def cubic(x):
        x = np.asarray(x, float)
        return 0.6 + 0.4 * x + 0.2 * x ** 2 + 0.1 * x ** 3

    alpha = acceptance_coefficients(cubic, 3)
    probe = np.linspace(-1, 1, 41)
    np.testing.assert_allclose(acceptance_from_coefficients(alpha, probe),
                               cubic(probe), rtol=0, atol=1e-12)


# -------------------------------------------------------------- ballistic

def test_ballistic_uses_the_exact_arrival_direction_of_each_element():
    track = lh.CherenkovTrack([0, 0, 0], [0, 0, 1], 3.0, beta=0.999)
    elements = track.to_elements(step_m=0.25).field(0)
    cone = elements.cone_cosine
    receivers = np.array([[1.0, 0.0, 2.0], [2.0, 1.0, 4.0]])
    looks = np.array([[0.3, -0.4, 0.866], [0.0, 1.0, 0.0]])
    looks /= np.linalg.norm(looks, axis=1)[:, None]
    charge, _ = ballistic_directional(elements, receivers, looks, medium(),
                                      cone, ALPHA)
    # brute force, one element at a time, with the root written out here
    expected = np.zeros(len(receivers))
    for index, receiver in enumerate(receivers):
        for element in range(len(cone)):
            offset = receiver - elements.start_m[element]
            unit = elements.direction[element]
            along = offset @ unit
            impact = np.sqrt(max(offset @ offset - along ** 2, 1e-300))
            sine = np.sqrt(1 - cone[element] ** 2)
            root = along - impact * cone[element] / sine
            if not 0.0 <= root < elements.length_m[element]:
                continue
            distance = impact / sine
            arrival = (offset - root * unit) / distance
            weight = (elements.photons[element] / elements.length_m[element]
                      * np.exp(-medium().extinction_per_m * distance)
                      / (2 * np.pi * impact * sine))
            expected[index] += weight * float(acceptance_from_coefficients(
                ALPHA, np.array([arrival @ looks[index]]))[0])
    np.testing.assert_allclose(charge, expected, rtol=1e-12, atol=0)


def test_constant_acceptance_ballistic_is_the_old_kernel_times_the_constant():
    track = lh.CherenkovTrack([0, 0, 0], [0, 0, 1], 3.0, beta=0.999)
    elements = track.to_elements(step_m=0.25).field(0)
    receivers = np.array([[1.0, 0.0, 2.0], [0.5, -0.5, 1.5]])
    looks = np.array([[0.3, -0.4, 0.866], [0.0, 1.0, 0.0]])
    looks /= np.linalg.norm(looks, axis=1)[:, None]
    constant = 0.37
    scaled, _ = ballistic_directional(elements, receivers, looks, medium(),
                                      elements.cone_cosine,
                                      np.array([4 * np.pi * constant]))
    plain, _ = ballistic_directional(elements, receivers, looks, medium(),
                                     elements.cone_cosine, np.array([4 * np.pi]))
    np.testing.assert_allclose(scaled, constant * plain, rtol=1e-14, atol=0)


def test_ballistic_bins_keep_out_of_window_light_out_of_the_edges():
    track = lh.CherenkovTrack([0, 0, 0], [0, 0, 1], 3.0, beta=0.999)
    elements = track.to_elements(step_m=0.25).field(0)
    receivers = np.array([[1.0, 0.0, 2.0]])
    looks = np.array([[0.0, 0.0, 1.0]])
    edges = np.array([-100.0, 100.0])
    charge, bins = ballistic_directional(elements, receivers, looks, medium(),
                                         elements.cone_cosine, ALPHA,
                                         time_origin_ns=np.zeros(1),
                                         relative_edges_ns=edges)
    assert bins.shape == (1, 1)
    assert bins[0, 0] == pytest.approx(charge[0], rel=1e-12)
    narrow = np.array([-100.0, -99.0])
    _, empty = ballistic_directional(elements, receivers, looks, medium(),
                                     elements.cone_cosine, ALPHA,
                                     time_origin_ns=np.zeros(1),
                                     relative_edges_ns=narrow)
    assert empty.sum() == 0.0


# --------------------------------------------------------------- transport

def spectral_medium():
    return lh.SpectralMedium([400., 500.], [.04, .05], [.02, .018],
                             [1.34, 1.36], [1.37, 1.39], g=.9,
                             provenance="test spectrum")


def detector(acceptance):
    return lh.DetectorArray(
        [[10., 0., 0.], [0., 14., 0.]], [[-1., 0., 0.], [0., -1., 0.]], .05,
        acceptance, lambda w: np.ones_like(np.asarray(w, float)) * .2,
        identifiers={"om": [1, 2]})


def small_config(tmp_path, **kwargs):
    base = dict(omega_per_ns=np.array([0., 0.05]),
                relative_time_edges_ns=np.arange(-10., 111., 20.),
                wavelength_nodes=2, scattering_degree=4, source_degree=4,
                azimuthal_degree=2, cell_m=1., k_max_per_m=2.,
                k_panel_per_m=.2, k_order=4, radial_range_m=(2., 60.),
                radial_nodes=8, angular_backend="numpy", threshold_pe=0.,
                cache_directory=tmp_path)
    base.update(kwargs)
    return lh.KernelConfig(**base)


def test_a_clipped_module_response_is_refused_rather_than_truncated(tmp_path):
    def clipped(x):
        x = np.asarray(x, float)
        return np.maximum(0.2 + 0.9 * x, 0.0)

    kernel = lh.TransportKernel(spectral_medium(), detector(clipped),
                                small_config(tmp_path))
    with pytest.raises(ValueError, match="not band-limited"):
        kernel.acceptance()
    allowed = lh.TransportKernel(spectral_medium(), detector(clipped),
                                 small_config(tmp_path, acceptance_degree=3))
    measured = allowed.acceptance()
    assert measured["degree"] == 3
    assert measured["residual_above_degree"] > 1e-4


def directional_kernel(tmp_path, **kwargs):
    return lh.TransportKernel(
        spectral_medium(),
        detector(lambda x: np.clip((1 + np.asarray(x)) / 2, 0, 1)),
        small_config(tmp_path, **kwargs))


def isotropic_kernel(tmp_path, **kwargs):
    return lh.TransportKernel(
        spectral_medium(),
        detector(lambda x: np.ones_like(np.asarray(x, float))),
        small_config(tmp_path, **kwargs))


def test_the_method_names_the_source_and_the_module_decides_its_own_model(tmp_path):
    """A method is a source engine; the OM model follows the OM."""
    directional = directional_kernel(tmp_path)
    isotropic = isotropic_kernel(tmp_path)
    assert directional.acceptance()["degree"] == 1
    assert isotropic.acceptance()["degree"] == 0
    track = lh.CherenkovTrack([0, 0, -1], [0, 0, 1], 2., beta=.99)
    exact = directional.transport(track)
    assert exact.method == "track"
    assert exact.metadata["detector_angular_model"] == "exact_m_blocks"
    # An isotropic module keeps the existing route and pays nothing new: for a
    # constant acceptance the centroid cosine is not an approximation at all.
    assert isotropic.cache_kinds("auto") == ("multipole",)
    assert isotropic._directional_caches == {}


def test_an_isotropic_module_still_reaches_the_m_zero_engine(tmp_path):
    pytest.importorskip("numba")        # the m = 0 axial engine is numba-only
    kernel = isotropic_kernel(tmp_path)
    track = lh.CherenkovTrack([0, 0, -1], [0, 0, 1], 2., beta=.99)
    plain = kernel.transport(track)
    assert plain.method == "track"
    assert plain.metadata["detector_angular_model"] != "exact_m_blocks"
    assert kernel._directional_caches == {}


def test_the_superseded_centroid_engine_is_experimental_and_still_reachable(tmp_path):
    kernel = directional_kernel(tmp_path)
    track = lh.CherenkovTrack([0, 0, -1], [0, 0, 1], 2., beta=.99)
    assert kernel.available_methods()["axial_centroid"]["experimental"]
    assert kernel.available_methods()["track_centroid"]["experimental"]
    assert not kernel.available_methods()["axial"]["experimental"]
    with pytest.raises(ValueError, match="experimental"):
        kernel.transport(track, method="axial_centroid")


def test_build_makes_only_the_tables_the_method_will_read(tmp_path):
    """The old build() made the m = 0 tables and auto then ignored them."""
    directional = directional_kernel(tmp_path)
    assert directional.cache_kinds("auto") == ("directional",)
    assert directional.cache_kinds("isotropic") == ("multipole",)
    assert directional.cache_kinds("all") == ("multipole", "directional")
    directional.build()
    assert len(directional._directional_caches) == 2
    assert directional._caches == {}
    assert not list(tmp_path.glob("transport-*.npz"))
    assert len(list(tmp_path.glob("directional-*.npz"))) == 2

    isotropic = isotropic_kernel(tmp_path)
    assert isotropic.cache_kinds("auto") == ("multipole",)
    assert isotropic.cache_kinds("directional") == ("directional",)
    isotropic.build()
    assert isotropic._directional_caches == {}
    assert len(isotropic._caches) == 2

    both = directional_kernel(tmp_path)
    both.build(method="all")
    assert len(both._caches) == 2 and len(both._directional_caches) == 2

    # a source resolves the choice exactly rather than by guessing
    flash = directional_kernel(tmp_path)
    flash.build(source=lh.IsotropicFlash.broadband(
        [0, 0, 0], 1e5, lambda w: np.ones_like(np.asarray(w, float))))
    assert flash._directional_caches == {} and len(flash._caches) == 2

    with pytest.raises(ValueError, match="unknown method"):
        both.cache_kinds("nonsense")


def test_building_for_a_monochromatic_flash_builds_its_one_line(tmp_path):
    """The quadrature is not what a monochromatic source will ask for."""
    kernel = directional_kernel(tmp_path)
    laser = lh.IsotropicFlash.monochromatic([0, 0, 0], 1e5, 450.)
    assert kernel.source_wavelengths(laser) == pytest.approx([450.])
    kernel.build(source=laser)
    built = sorted(path.name for path in tmp_path.glob("*.npz"))
    assert len(built) == 1
    kernel.transport(laser)
    # the transport must find the table already there, not build a third one
    assert sorted(path.name for path in tmp_path.glob("*.npz")) == built
    assert len(kernel._caches) == 1

    # an explicit wavelength list still wins over what the source implies
    explicit = directional_kernel(tmp_path)
    explicit.build([420.0, 470.0], source=laser)
    assert len(explicit._caches) == 2


def test_directional_transport_reports_the_exact_model(tmp_path):
    kernel = directional_kernel(tmp_path)
    track = lh.CherenkovTrack([0, 0, -1], [0, 0, 1], 2., beta=.99)
    response = kernel.transport(track, method="directional")
    assert response.metadata["detector_angular_model"] == "exact_m_blocks"
    assert response.metadata["detector_acceptance_degree"] == 1
    assert response.metadata["detector_acceptance_residual_above_degree"] < 1e-9
    assert response.metadata["ballistic_angular_model"] == (
        "exact per-element arrival direction")
    # the spectral model is untouched
    assert response.metadata["spectral_source"] == (
        "S0=lambda^-2; S2=lambda^-2*n_phase^-2")
    assert response.metadata["source_fields"] == 2
    assert response.charge_components_pe.shape == (2, 3)
    assert response.components_pe.shape == (2, 6, 3)


def test_detector_efficiency_enters_linearly_and_unchanged(tmp_path):
    """efficiency(lambda) * transmission stays a multiplicative factor."""
    def make(scale):
        return lh.DetectorArray(
            [[10., 0., 0.]], [[-1., 0., 0.]], .05,
            lambda x: np.clip((1 + np.asarray(x)) / 2, 0, 1),
            lambda w: np.ones_like(np.asarray(w, float)) * scale)

    track = lh.CherenkovTrack([0, 0, -1], [0, 0, 1], 2., beta=.99)
    one = lh.TransportKernel(spectral_medium(), make(.2), small_config(tmp_path))
    two = lh.TransportKernel(spectral_medium(), make(.5), small_config(tmp_path))
    first = one.transport(track, method="directional").charge_components_pe
    second = two.transport(track, method="directional").charge_components_pe
    np.testing.assert_allclose(second, 2.5 * first, rtol=1e-12, atol=0)


def test_nothing_clips_a_negative_frequency_bin(tmp_path):
    """The readout is left exactly as it was: no clipping, no renormalisation."""
    kernel = lh.TransportKernel(
        spectral_medium(),
        detector(lambda x: np.clip((1 + np.asarray(x)) / 2, 0, 1)),
        small_config(tmp_path))
    track = lh.CherenkovTrack([0, 0, -1], [0, 0, 1], 2., beta=.99)
    response = kernel.transport(track, method="directional")
    total = response.components_pe[:, :, 1:].sum(axis=1)
    charge = response.charge_components_pe[:, 1:]
    # the inverse transform is the same one the other engines use: the bins add
    # up to the charge, whatever sign the coarse settings give it
    assert np.max(np.abs(total - charge)) < 0.2 * np.max(np.abs(charge)) + 1e-12
    # At these deliberately coarse settings the truncated multipole sum rings
    # negative -- the m = 0 route does the same, bit for bit. What matters is
    # that the sign survives to the output: nothing clips it and nothing
    # renormalises it away.
    assert np.any(charge < 0)
    assert np.any(response.components_pe[:, :, 1:] < 0)


def test_clebsch_gordan_survives_a_large_source_degree():
    """The naive product of six factorials overflows a double near j ~ 60."""
    value = clebsch_gordan(3, 1, 64, -1, 67, 0)
    assert np.isfinite(value) and abs(value) <= 1.0
    table = CouplingTable.build(64, 3)
    assert table.triples == 640 and table.pairs == 10
    assert np.isfinite(table.nu_weight).all()
    assert np.isfinite(table.pair_weight).all()
    # orthogonality of one coupled row, as an independent check at large j
    total = sum(clebsch_gordan(3, mu, 64, -mu, 67, 0) ** 2
                for mu in range(-3, 4))
    assert total == pytest.approx(1.0, rel=1e-12)


def test_wigner_start_value_is_finite_at_a_large_degree():
    from lighthit.directional import wigner_small_d
    d = wigner_small_d(70, 3, 4, np.array([0.3, -0.8]))
    assert np.isfinite(d).all()
    assert np.max(np.abs(d)) <= 1.0 + 1e-12
