from pathlib import Path
from dataclasses import replace
import json

import numpy as np
import pytest

import lighthit as lh
from lighthit.experimental.g4_source import FINE_STRUCTURE, cherenkov_yield_per_m


def spectral_medium():
    return lh.SpectralMedium(
        [400., 500.], [.04, .05], [.02, .018], [1.34, 1.36], [1.37, 1.39],
        g=.9, provenance="test spectrum")


def detector():
    return lh.DetectorArray(
        [[10., 0., 0.], [0., 20., 0.]], [[-1., 0., 0.], [0., -1., 0.]],
        .05, lambda x: np.clip((1 + np.asarray(x)) / 2, 0, 1),
        lambda wavelength: np.ones_like(np.asarray(wavelength, float)) * .2,
        identifiers={"om": [1, 2]})


def isotropic_detector():
    return lh.DetectorArray(
        [[10., 0., 0.], [0., 20., 0.]], [[-1., 0., 0.], [0., -1., 0.]],
        .05, lambda x: np.ones_like(np.asarray(x, float)),
        lambda wavelength: np.ones_like(np.asarray(wavelength, float)) * .2)


def config(tmp_path, *, omega=(0.,), threshold=0):
    return lh.KernelConfig(
        omega_per_ns=np.asarray(omega), relative_time_edges_ns=np.arange(-10., 111., 20.),
        wavelength_nodes=2, scattering_degree=4, source_degree=4,
        azimuthal_degree=2, cell_m=1., k_max_per_m=2., k_panel_per_m=.2,
        k_order=4, radial_range_m=(2., 60.), radial_nodes=8,
        angular_backend="numpy", threshold_pe=threshold,
        cache_directory=tmp_path)


def test_default_frequency_grid_widens_cutoff_without_shortening_image_period():
    cfg = lh.KernelConfig()
    assert len(cfg.omega_per_ns) == 321
    assert cfg.omega_per_ns[-1] == pytest.approx(1.2)
    assert np.diff(cfg.omega_per_ns) == pytest.approx(0.00375)
    assert 2 * np.pi / np.diff(cfg.omega_per_ns).max() > np.ptp(
        cfg.relative_time_edges_ns)
    assert np.diff(cfg.relative_time_edges_ns) == pytest.approx(5.0)


def test_cache_policy_is_validated():
    with pytest.raises(ValueError, match="cache_policy"):
        lh.KernelConfig(cache_policy="typo")


def test_two_field_identity_matches_frank_tamm_differential():
    track = lh.CherenkovTrack([0, 0, 0], [0, 0, 1], 2., beta=.98)
    elements = track.to_elements(step_m=2.)
    wavelength = np.array([410., 470., 530.])
    phase = np.array([1.34, 1.35, 1.36])
    got = elements.photon_density_per_nm(wavelength, phase)[0]
    expected = (2 * np.pi * FINE_STRUCTURE * 1e9 * 2 / wavelength**2
                * (1 - 1 / (.98 * phase) ** 2))
    np.testing.assert_allclose(got, expected, rtol=2e-15)
    # Its integral for a constant phase reproduces the established band formula.
    low, high, n = 400., 500., 1.35
    analytic = cherenkov_yield_per_m(.98, n, low, high) * 2
    integral = (2 * np.pi * FINE_STRUCTURE * 1e9 * 2
                * (1 - 1 / (.98*n)**2) * (1/low - 1/high))
    assert integral == pytest.approx(analytic, rel=2e-15)


def test_isotropic_public_api_and_cache_reuse(tmp_path):
    # build() makes the tables the transport will read and no others; this
    # detector is directional, so building for a flash must not produce the
    # directional tables the flash never touches.
    source = lh.IsotropicFlash.monochromatic([0, 0, 0], 1e5, 450.)
    kernel = lh.TransportKernel(spectral_medium(), detector(),
                                config(tmp_path)).build(source=source)
    built = sorted(path.name for path in tmp_path.glob("*.npz"))
    first = kernel.transport(source)
    second = kernel.transport(source, method="isotropic")
    moved = lh.IsotropicFlash.monochromatic([2., 0., 0.], 1e5, 450.)
    independent_kernel = lh.TransportKernel(
        spectral_medium(), detector(),
        replace(config(tmp_path), cache_policy="require"))
    moved_response = independent_kernel.transport(moved)
    assert sorted(path.name for path in tmp_path.glob("*.npz")) == built
    assert first.charge_components_pe.shape == (2, 3)
    assert first.components_pe.shape == (2, 6, 3)
    assert np.all(first.charge_pe > 0)
    np.testing.assert_allclose(second.charge_components_pe, first.charge_components_pe)
    np.testing.assert_allclose(first.select("0+1").charge_pe,
                               first.charge_components_pe[:, :2].sum(axis=1))
    np.testing.assert_allclose(first.select("0+1").bins_pe,
                               first.components_pe[:, :, :2].sum(axis=2))
    np.testing.assert_allclose(first.select(">=2").bins_pe,
                               first.components_pe[:, :, 2])
    np.testing.assert_allclose(first.select("all").charge_pe, first.charge_pe)
    with pytest.raises(ValueError, match="component"):
        first.select("unknown")
    assert not np.allclose(moved_response.charge_pe, first.charge_pe)
    # A monochromatic flash reads exactly one table, so exactly one is built
    # and the transport that follows must not add another.
    assert len(list(tmp_path.glob("transport-*.npz"))) == 1
    assert not list(tmp_path.glob("directional-*.npz"))


def test_isotropic_flash_uses_exact_physical_time_first_order(tmp_path):
    from lighthit.single import single_bins

    source = lh.IsotropicFlash.monochromatic([0, 0, 0], 1e5, 450., time_ns=7.)
    cfg = config(tmp_path, omega=(0., .04), threshold=0)
    kernel = lh.TransportKernel(spectral_medium(), isotropic_detector(), cfg)
    response = kernel.transport(source)
    band = spectral_medium().band(450.)
    radii = np.linalg.norm(isotropic_detector().positions_m - source.position_m, axis=1)
    scale = source.photons * .05 * .2
    for index, radius in enumerate(radii):
        absolute_edges_from_emission = (
            response.relative_time_edges_ns + response.time_origin_ns[index] - source.time_ns)
        expected = scale * single_bins(
            absolute_edges_from_emission, radius, None, band,
            backend=kernel.resolved_angular_backend())
        np.testing.assert_allclose(response.components_pe[index, :, 1], expected,
                                   rtol=2e-10, atol=1e-14)
        assert np.all(response.components_pe[index, :, 1] >= 0)
    assert response.metadata["fourier_inverted_orders"] == [2]
    assert response.metadata["first_order"] == (
        "exact full-HG spectrum and physical-time bins")


def test_build_convenience_returns_an_explicit_reusable_kernel(tmp_path):
    source = lh.IsotropicFlash.monochromatic([0, 0, 0], 1e5, 450.)
    kernel = lh.build(spectral_medium(), detector(), source,
                      config=config(tmp_path))
    assert isinstance(kernel, lh.TransportKernel)
    assert len(kernel._caches) == 1
    assert kernel._directional_caches == {}
    response = kernel.transport(source)
    assert response.method == "isotropic"
    assert len(kernel._caches) == 1


def test_explicit_cache_build_reports_build_load_reuse_and_force(tmp_path, capsys):
    source = lh.IsotropicFlash.monochromatic([0, 0, 0], 1e5, 450.)
    cfg = config(tmp_path)
    events = []
    first = lh.TransportKernel(spectral_medium(), detector(), cfg)
    first.build(source=source, progress=events.append)
    assert first.last_build_report.built == 1
    assert first.last_build_report.loaded == 0
    assert events[-1].action == "built"
    assert not list(tmp_path.glob("*.tmp.npz"))
    manifest = json.loads((tmp_path / "cache-manifest.json").read_text())
    assert manifest["schema"] == "lighthit/cache-manifest/1"
    assert manifest["package_version"] == lh.__version__
    assert manifest["method"] == "isotropic"
    assert manifest["last_build"]["built"] == 1
    assert manifest["available_files"] == [
        path.name for path in sorted(tmp_path.glob("transport-*.npz"))]

    second = lh.TransportKernel(spectral_medium(), detector(), cfg)
    second.build(source=source, progress="console")
    assert second.last_build_report.built == 0
    assert second.last_build_report.loaded == 1
    assert "Nothing to build" in capsys.readouterr().out

    second.build(source=source, force=True)
    assert second.last_build_report.built == 1
    assert not list(tmp_path.glob("*.tmp.npz"))


def test_require_cache_policy_never_builds_during_transport(tmp_path):
    source = lh.IsotropicFlash.monochromatic([0, 0, 0], 1e5, 450.)
    cfg = replace(config(tmp_path), cache_policy="require")
    kernel = lh.TransportKernel(spectral_medium(), detector(), cfg)
    with pytest.raises(FileNotFoundError, match="kernel.build"):
        kernel.transport(source)
    assert not list(tmp_path.glob("*.npz"))

    kernel.build(source=source)
    response = kernel.transport(source)
    assert np.all(response.charge_pe > 0)


def test_radial_range_is_never_silently_extrapolated(tmp_path):
    far = lh.DetectorArray(
        [[100., 0., 0.]], [[-1., 0., 0.]], .05,
        lambda x: np.ones_like(np.asarray(x, float)),
        lambda wavelength: np.ones_like(np.asarray(wavelength, float)) * .2)
    kernel = lh.TransportKernel(spectral_medium(), far, config(tmp_path, threshold=.01))
    weak = kernel.transport(lh.IsotropicFlash.monochromatic([0, 0, 0], 1., 450.))
    assert weak.metadata["omitted_outside_range"] == 1
    assert not weak.active[0]
    with pytest.raises(ValueError, match="outside radial_range_m"):
        kernel.transport(lh.IsotropicFlash.monochromatic([0, 0, 0], 1e20, 450.))


def test_bright_laser_uses_geometry_covered_cache(tmp_path):
    far = lh.DetectorArray(
        [[10., 0., 0.], [100., 0., 0.]],
        [[-1., 0., 0.], [-1., 0., 0.]], .05,
        lambda x: np.ones_like(np.asarray(x, float)),
        lambda wavelength: np.ones_like(np.asarray(wavelength, float)) * .2)
    radial_range = lh.point_source_radial_range(
        far, [0., 0., 0.], minimum_range_m=(2., 60.))
    cfg = replace(config(tmp_path, omega=(0., .04), threshold=.01),
                  radial_range_m=radial_range)
    source = lh.IsotropicFlash.monochromatic([0., 0., 0.], 1e15, 450.)
    response = lh.TransportKernel(spectral_medium(), far, cfg).transport(source)
    assert response.active.tolist() == [True, True]
    assert np.isfinite(response.charge_components_pe).all()
    assert response.metadata["omitted_outside_range"] == 0


def test_experimental_method_requires_opt_in(tmp_path):
    kernel = lh.TransportKernel(spectral_medium(), detector(), config(tmp_path))
    source = lh.CherenkovTrack([0, 0, 0], [0, 0, 1], 1.)
    with pytest.raises(ValueError, match="experimental"):
        kernel.transport(source, method="axial_full")

    kernel.register_method("probe", lambda value, method: (method, value),
                           experimental=True, description="test hook")
    with pytest.raises(ValueError, match="experimental"):
        kernel.transport(source, method="probe")
    assert kernel.transport(source, method="probe", allow_experimental=True) == ("probe", source)


def test_bgvd_adapter_without_importing_heavy_private_package(tmp_path):
    package = tmp_path / "bgvd_model"; data = package / "data"; data.mkdir(parents=True)
    (package / "BaikalWater.py").write_text('''
import numpy as np
class BaikalWater:
 def __init__(self):
  self.wavelength=np.array([400.,500.]);self.absorption_inv_length=np.array([.04,.05])
  self.scattering_inv_length=np.array([.02,.01]);self.phase_refraction_index=np.array([1.34,1.36])
  self.group_refraction_index=np.array([1.37,1.39]);self.cs_scattering=np.array([1.]);self.cosines=np.array([1.])
''')
    (package / "OpticalModule.py").write_text('''
import numpy as np
radius=.2
angular_parameters=np.array([.3,-.5,.2,0.])
def efficiency(wavelength): return np.ones_like(np.asarray(wavelength,float))*.25
def transmission_gel_glass(wavelength): return np.ones_like(np.asarray(wavelength,float))*.8
''')
    (data / "median_om_coordinates_2021.csv").write_text(
        "cluster,subcluster,channel,mx_m,my_m,mz_m,dir_x,dir_y,dir_z\n"
        "1,1,1,0,0,0,0,0,-1\n1,1,2,0,0,15,0,0,-1\n")
    value = lh.load_bgvd_model(tmp_path, dataset="2021")
    assert len(value.detector) == 2
    np.testing.assert_allclose(value.detector.spectral_weight([420., 480.]), .2)
    assert value.medium.g == .9
    assert isinstance(value.kernel(config(tmp_path / "cache")), lh.TransportKernel)
    # Head-on +1 maps to the private polynomial's -1 convention.
    assert value.detector.angular_acceptance(np.array([1.]))[0] == pytest.approx(1.0)


@pytest.mark.filterwarnings("ignore:.*experimental.*")
def test_small_spectral_track_axial(tmp_path):
    pytest.importorskip("numba")
    cfg = config(tmp_path, omega=(0., .04), threshold=0)
    kernel = lh.TransportKernel(spectral_medium(), detector(), cfg)
    response = kernel.transport(
        lh.CherenkovTrack([0, 0, -1], [0, 0, 1], 1., beta=.99), method="track")
    assert response.spectrum_pe.shape == (2, 2, 3)
    assert response.components_pe.shape == (2, 6, 3)
    assert np.isfinite(response.charge_components_pe).all()
    assert response.metadata["source_fields"] == 1
    assert response.metadata["track_fast_path"] is True
