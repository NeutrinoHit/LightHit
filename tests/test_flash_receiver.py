"""Isotropic flash on the glass receiver: contraction, direct flux and route."""
import json
from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import quad

import lighthit as lh
from lighthit._flash_receiver import direct_nodes, first_entry_contraction
from lighthit.cache import CacheGrid, ResponseCache
from lighthit.directional import DirectionalCache, directional_response
from lighthit.event import LASER_DATA, laser_config, laser_settings
from lighthit.experimental.axial_source import AxialSource, AxisFrame
from lighthit.green import SolverSettings
from lighthit.medium import Medium
from lighthit.receiver import SphereSurface
from lighthit.receiver_vertex import ReceiverVertexConfig

ALPHA = np.array([4 * np.pi, 1.3, -0.5, 0.2])
SMALL_VERTEX = ReceiverVertexConfig(angular_pairs=8, radial_nodes=32,
                                    return_angle_nodes=8, return_path_nodes=16)


def medium():
    return Medium(0.02, 0.05, 0.9, 1.35, 450.0, "test medium")


@pytest.fixture(scope="module")
def grid():
    return CacheGrid(np.geomspace(4.0, 40.0, 12), np.array([0.0, 0.07]))


@pytest.fixture(scope="module")
def glass_table(grid):
    return DirectionalCache.build(medium(), SolverSettings(6, 0, 2.0, 0.1, 6, True), grid, 3,
                                  radial_phase="flight", receiver_surface=SphereSurface(0.216),
                                  receiver_config=SMALL_VERTEX)


def test_centre_tables_give_the_multipole_flash_exactly(grid):
    """Source degree 0 of the directional table is the multipole flash table."""
    multipole = ResponseCache.build(medium(), SolverSettings(6, 6, 2.0, 0.1, 6, True), grid,
                                    radial_phase="flight")
    table = DirectionalCache.build(medium(), SolverSettings(6, 0, 2.0, 0.1, 6, True), grid, 3,
                                   radial_phase="flight")
    radii = np.array([4.5, 9.0, 21.0, 37.0])
    cosines = np.array([1.0, 0.3, -0.6, -1.0])
    want = multipole.acceptance_spectrum(radii, cosines, ALPHA)[:, :, 1:]
    got = first_entry_contraction(table, radii, cosines, ALPHA)
    assert np.max(np.abs(got - want)) / np.max(np.abs(want)) < 1e-12


def test_contraction_is_the_generic_directional_response_of_an_l0_source(glass_table):
    """The l = 0 shortcut agrees with equation (*) on a first-entry table."""
    omega = glass_table.grid.omega_per_ns
    frame = AxisFrame(np.zeros(3), np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]),
                      np.array([0.0, 1.0, 0.0]))
    # a unit isotropic emitter: f = S_00 Y_00 = 1 / (4 pi)
    channels = np.full((1, 1, len(omega)), 1 / np.sqrt(4 * np.pi), complex)
    source = AxialSource(frame, np.zeros(1), np.zeros((1, 2)), channels, np.array([0]),
                         np.array([0]), 0, 0, 0.0, {})
    receivers = np.array([[6.0, 4.0, 14.0], [-9.0, 2.0, 5.0], [3.0, -20.0, -8.0]])
    rng = np.random.default_rng(3)
    looks = rng.normal(size=(3, 3))
    looks /= np.linalg.norm(looks, axis=1)[:, None]
    want = directional_response(glass_table, source, receivers, looks, ALPHA,
                                source_omega_per_ns=omega)
    radii = np.linalg.norm(receivers, axis=1)
    # a photon arriving head-on travels along +look
    cosines = np.sum(looks * receivers, axis=1) / radii
    got = first_entry_contraction(glass_table, radii, cosines, ALPHA)
    assert np.max(np.abs(got - want)) / np.max(np.abs(want)) < 1e-12


def test_glass_table_is_not_the_centre_table(glass_table, grid):
    centre = DirectionalCache.build(medium(), SolverSettings(6, 0, 2.0, 0.1, 6, True), grid, 3,
                                    radial_phase="flight")
    radii, cosines = np.array([5.0, 20.0]), np.array([1.0, 0.0])
    glass = first_entry_contraction(glass_table, radii, cosines, ALPHA)[0].real
    point = first_entry_contraction(centre, radii, cosines, ALPHA)[0].real
    assert np.all(glass > 0) and np.all(point > 0)
    assert not np.allclose(glass, point, rtol=1e-3)
    # the glass shortens every entry path, far from the sphere by a small amount
    assert np.all(np.abs(glass / point - 1) < 0.05)


def acceptance(x):
    x = np.asarray(x, float)
    return 0.35 + 0.4 * x + 0.2 * x * x + 0.1 * x ** 3


@pytest.mark.parametrize("distance,cosine", [(0.5, 0.7), (3.0, -0.2), (40.0, 1.0)])
def test_direct_flux_matches_an_independent_polar_integral(distance, cosine):
    R, mu = 0.216, 0.08
    paths, weights = direct_nodes(np.array([distance]), np.array([cosine]), R, acceptance)
    got = float(np.sum(weights * np.exp(-mu * paths)))
    cap = np.arcsin(R / distance)
    s = np.sqrt(1 - cosine * cosine)

    def integrand(theta):
        entry = distance * np.cos(theta) - np.sqrt(max(R * R - (distance * np.sin(theta)) ** 2, 0.0))
        phi = np.linspace(0, 2 * np.pi, 64, endpoint=False)
        ring = acceptance(cosine * np.cos(theta) + s * np.sin(theta) * np.cos(phi)).mean() * 2 * np.pi
        return np.sin(theta) * np.exp(-mu * entry) * ring / (4 * np.pi)

    want = quad(integrand, 0.0, cap, epsabs=0, epsrel=1e-12, limit=200)[0]
    assert got == pytest.approx(want, rel=1e-9)


def test_direct_flux_tends_to_the_centre_formula_for_a_small_sphere():
    d, c, mu = np.array([30.0, 80.0]), np.array([0.4, -0.9]), 0.07
    paths, weights = direct_nodes(d, c, 1e-4, acceptance)
    got = np.sum(weights * np.exp(-mu * paths), axis=1) / (np.pi * 1e-8)
    want = acceptance(c) * np.exp(-mu * d) / (4 * np.pi * d * d)
    np.testing.assert_allclose(got, want, rtol=1e-5)
    # the leading finite-sphere term: the mean entry path is shorter by 2R/3
    np.testing.assert_allclose(got, want * np.exp(mu * 2e-4 / 3), rtol=1e-8)


# ---------------------------------------------------------------- the route

def spectral_medium():
    return lh.SpectralMedium([350., 610.], [0.08, 0.05], [0.03, 0.02],
                             [1.35, 1.335], [1.40, 1.35], g=0.9, provenance="flash test medium")


def small_detector():
    positions = np.array([[-5.0, 0.0, 0.0], [0.0, 9.0, 1.0], [-6.0, -6.0, 3.0],
                          [2.0, 3.0, -14.0], [11.0, -7.0, 4.0]])
    orientations = np.tile([0.0, 0.0, -1.0], (len(positions), 1))
    detector = lh.DetectorArray(positions, orientations, 0.05, acceptance,
                                lambda w: 0.25 * np.ones_like(np.asarray(w, float)))
    return replace(detector, receiver_surface=SphereSurface(0.216))


GLASS = dict(scattered_receiver="first-entry", scattering_degree=6, source_degree=0,
             receiver_vertex=SMALL_VERTEX.__dict__)


def small_config(directory, **settings):
    base = laser_config(small_detector(), cache_directory=directory, cache_policy="build",
                        omega_per_ns=np.linspace(0.0, 0.3, 6), kernel_settings=settings)
    return replace(base, radial_range_m=(3.0, 30.0), radial_nodes=12, k_max_per_m=2.0,
                   k_panel_per_m=0.1, k_order=6, threshold_pe=0.0,
                   relative_time_edges_ns=np.arange(-20.0, 80.0, 5.0))


@pytest.fixture(scope="module")
def flash():
    return lh.IsotropicFlash.monochromatic([0.0, 0.0, 0.0], 1e9, 450.0)


@pytest.fixture(scope="module")
def glass_route(tmp_path_factory, flash):
    directory = tmp_path_factory.mktemp("laser-glass")
    kernel = lh.TransportKernel(spectral_medium(), small_detector(), small_config(directory, **GLASS))
    kernel.build(source=flash)
    return directory, kernel, kernel.transport(flash)


def test_the_glass_flash_builds_and_reads_only_its_directional_table(glass_route):
    directory, kernel, response = glass_route
    assert kernel.cache_kinds("isotropic") == ("directional",)
    assert [p.name.split("-")[0] for p in directory.glob("*.npz")] == ["directional"]
    table = kernel._directional_cache_for(450.0)
    assert table.degree == 0 and table.receiver_contract["radius_m"] == 0.216
    assert response.metadata["scattered_receiver"] == "first-entry"
    assert response.charge_components_pe.shape == (5, 3)
    assert np.all(response.charge_components_pe > 0)


def test_glass_flash_differs_from_the_centre_flash_by_the_receiver_only(glass_route, flash, tmp_path):
    _, kernel, glass = glass_route
    centre = lh.TransportKernel(spectral_medium(), small_detector(), small_config(tmp_path))
    assert centre.config.scattered_receiver == "centre"
    assert centre.cache_kinds("isotropic") == ("multipole",)
    point = centre.transport(flash)
    ratio = glass.charge_components_pe / point.charge_components_pe
    # order 0: entry paths are shorter than the centre distance, by 2R/3 on average
    radii = np.linalg.norm(small_detector().positions_m, axis=1)
    mu = centre.medium.band(450.0).extinction_per_m
    np.testing.assert_allclose(ratio[:, 0], np.exp(mu * 2 * 0.216 / 3), rtol=2e-3 * (5 / radii).max())
    assert np.all(np.abs(ratio[:, 1:] - 1) < 0.1)
    np.testing.assert_array_equal(glass.time_origin_ns, point.time_origin_ns)


def test_glass_flash_bins_hold_the_charge_of_every_order(glass_route):
    _, kernel, response = glass_route
    bins = response.components_pe.sum(axis=1)
    np.testing.assert_allclose(bins[:, 0], response.charge_components_pe[:, 0], rtol=1e-12)


def test_laser_config_reads_the_receiver_from_laser_data(tmp_path):
    assert laser_settings(tmp_path) == {}
    assert laser_config(small_detector(), cache_directory=tmp_path).scattered_receiver == "centre"
    (tmp_path / LASER_DATA).write_text(json.dumps(dict(schema="lighthit/laser-data/1",
                                                       kernel_settings=GLASS)))
    config = laser_config(small_detector(), cache_directory=tmp_path)
    assert config.scattered_receiver == "first-entry" and config.source_degree == 0
    assert config.receiver_vertex == SMALL_VERTEX
    explicit = laser_config(small_detector(), cache_directory=tmp_path, kernel_settings={})
    assert explicit.scattered_receiver == "centre"
    (tmp_path / LASER_DATA).write_text(json.dumps(dict(schema="other")))
    with pytest.raises(ValueError, match="laser data schema"):
        laser_config(small_detector(), cache_directory=tmp_path)


def test_a_centre_table_never_serves_the_glass_flash(glass_route, flash, tmp_path):
    centre = lh.TransportKernel(spectral_medium(), small_detector(),
                                replace(small_config(tmp_path), source_degree=0,
                                        scattering_degree=6))
    centre._directional_cache_for(450.0, allow_build=True)
    glass = lh.TransportKernel(spectral_medium(), small_detector(),
                               replace(small_config(tmp_path, **GLASS), cache_policy="require"))
    with pytest.raises(FileNotFoundError, match="directional cache is missing"):
        glass.transport(flash)
