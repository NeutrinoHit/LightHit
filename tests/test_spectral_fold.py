"""The optional wavelength fold is checked against the ordinary solver."""
from dataclasses import replace

import numpy as np
import pytest

import lighthit as lh
from lighthit._spectral_fold import folded_cache_path

pytest.importorskip("numba")


def medium():
    return lh.SpectralMedium(
        [400., 500.], [.030, .042], [.020, .027],
        [1.34, 1.36], [1.37, 1.38], g=.9)


def detector():
    return lh.DetectorArray(
        [[10., 0., 0.], [0., 20., 0.]],
        [[-1., 0., 0.], [0., -1., 0.]], .05,
        lambda x: .6 + .4 * np.asarray(x, float),
        lambda w: .2 + .0001 * (np.asarray(w, float) - 400.))


def config(path, *, folded=False):
    return lh.KernelConfig(
        omega_per_ns=np.array([0., .04]),
        relative_time_edges_ns=np.arange(-20., 121., 20.),
        wavelength_nodes=2, scattering_degree=4, source_degree=4,
        azimuthal_degree=2, cell_m=1., k_max_per_m=2.,
        k_panel_per_m=.2, k_order=4, radial_range_m=(2., 60.),
        radial_nodes=12, angular_backend="numba", threshold_pe=0.,
        cache_directory=path, cache_policy="require",
        spectral_folded_cache=folded)


@pytest.fixture
def kernels(tmp_path):
    exact = lh.TransportKernel(medium(), detector(),
                               replace(config(tmp_path), cache_policy="build"))
    exact.build(method="track")
    path = exact.build_folded_directional(progress=False)
    assert path == folded_cache_path(exact)
    assert all((path / f"{name}.npy").is_file()
               for name in ("field0", "field2", "track_beta1"))
    fast = lh.TransportKernel(medium(), detector(), config(tmp_path, folded=True))
    return exact, fast


def test_folded_track_matches_wavelength_loop(kernels):
    exact, fast = kernels
    automatic = lh.TransportKernel(
        medium(), detector(), replace(config(fast.config.cache_directory),
                                      spectral_folded_cache="auto"))
    for beta in (1., .98):
        track = lh.CherenkovTrack.centered([0., 0., 0.], [0., 0., 1.],
                                          4., beta=beta)
        expected = exact.transport(track, method="track")
        actual = fast.transport(track, method="track")
        if beta == 1.:
            assert automatic.transport(track, method="track").metadata[
                "spectral_folded_cache"] is True
        assert actual.metadata["spectral_folded_cache"] is True
        np.testing.assert_allclose(actual.charge_components_pe,
                                   expected.charge_components_pe,
                                   rtol=3e-3, atol=1e-10)
        bin_error = np.abs(actual.components_pe - expected.components_pe)
        charge_scale = np.maximum(np.abs(expected.charge_pe), 1e-12)
        assert np.max(bin_error / charge_scale[:, None, None]) < 5e-4
        assert np.max(bin_error.sum(axis=(1, 2)) / charge_scale) < 2e-3


def test_folded_two_field_shower_matches_wavelength_loop(kernels):
    exact, fast = kernels
    track = lh.CherenkovTrack.centered([0., 0., 0.], [0., 0., 1.], 4.)
    elements = track.to_elements(step_m=1.)
    beta = np.array([1., .99, .98, .97])
    elements = replace(elements, beta=beta,
                       coefficient2=-elements.coefficient0 / beta ** 2)
    shower = lh.G4Shower(elements, "test", 0)
    expected = exact.transport(shower, method="axial")
    actual = fast.transport(shower, method="axial")
    assert actual.metadata["spectral_folded_cache"] is True
    np.testing.assert_allclose(actual.charge_components_pe,
                               expected.charge_components_pe,
                               rtol=3e-3, atol=1e-10)
    bin_error = np.abs(actual.components_pe - expected.components_pe)
    charge_scale = np.maximum(np.abs(expected.charge_pe), 1e-12)
    assert np.max(bin_error / charge_scale[:, None, None]) < 5e-4
    assert np.max(bin_error.sum(axis=(1, 2)) / charge_scale) < 2e-3


def test_folded_cache_is_explicit_and_bound_to_spectral_efficiency(tmp_path):
    kernel = lh.TransportKernel(medium(), detector(), config(tmp_path, folded=True))
    track = lh.CherenkovTrack.centered([0., 0., 0.], [0., 0., 1.], 4.)
    with pytest.raises(FileNotFoundError, match="build_folded_directional"):
        kernel.transport(track, method="track")
    ordinary = lh.TransportKernel(
        medium(), detector(), replace(config(tmp_path), cache_policy="build"))
    ordinary.build(method="track")
    automatic = lh.TransportKernel(
        medium(), detector(), replace(config(tmp_path),
                                      spectral_folded_cache="auto"))
    assert automatic.transport(track, method="track").metadata[
        "spectral_folded_cache"] is False

    altered_detector = lh.DetectorArray(
        detector().positions_m, detector().orientations, .05,
        detector().angular_acceptance,
        lambda w: np.full_like(np.asarray(w, float), .3))
    altered = lh.TransportKernel(medium(), altered_detector,
                                 config(tmp_path, folded=True))
    assert folded_cache_path(altered) != folded_cache_path(kernel)
