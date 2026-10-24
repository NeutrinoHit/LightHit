"""Physics-only table keys, legacy-key fallback and the portable model file."""
from dataclasses import replace
from pathlib import Path
import json
import os

import numpy as np
import pytest

import lighthit as lh

BUNDLE_MODEL = (Path(os.environ.get("LIGHTHIT_DATA", "")
                     or Path(__file__).resolve().parents[1] / "data" / "lighthit-bgvd-data")
                / "model" / "bgvd-model.npz")


def medium(provenance="first label"):
    return lh.SpectralMedium(
        [400., 500.], [.04, .05], [.02, .018], [1.34, 1.36], [1.37, 1.39],
        g=.9, provenance=provenance)


def response():
    return lh.OMResponse([.3082, -.54192, .19831, .04912],
                         np.linspace(300., 650., 8),
                         [.28, .35, .35, .3, .22, .12, .05, .02],
                         np.linspace(350., 650., 14),
                         [1.38e-3, 5e-3, .544, .804, .83, .866] + [.9] * 8)


def model(provenance="first label"):
    om = response()
    angular, spectral = om.functions()
    detector = lh.DetectorArray(
        [[10., 0., 0.], [0., 20., 0.], [0., 0., 15.]], [[1., 0., 0.], [0., 1., 0.], [0., 0., -1.]],
        np.pi * .216 ** 2, angular, spectral,
        receiver_surface=lh.SphereSurface([.216, .20, .25]),
        identifiers={"uid": [11101, 11102, 11103], "cluster_id": [0, 0, 1]},
        provenance="test detector")
    return lh.BGVDModel(medium(provenance), detector, "test", "0" * 64, om)


def config(directory, **changes):
    values = dict(
        omega_per_ns=np.array([0., .04]), relative_time_edges_ns=np.arange(-10., 111., 20.),
        wavelength_nodes=2, scattering_degree=4, source_degree=4,
        azimuthal_degree=2, cell_m=1., k_max_per_m=2., k_panel_per_m=.2,
        k_order=4, radial_range_m=(2., 60.), radial_nodes=8,
        angular_backend="numpy", threshold_pe=0, acceptance_degree=3,
        cache_directory=directory)
    values.update(changes)
    return lh.KernelConfig(**values)


FLASH = lh.IsotropicFlash.monochromatic([0., 0., 0.], 1e5, 450.)
TRACK = lh.CherenkovTrack([0, 0, -1], [0, 0, 1], 1., beta=.99)


def as_legacy(kernel, kind):
    """Rename this kernel's tables to the names LightHit 0.3.0a1 gave them."""
    prefix = kernel._TABLE_PREFIX[kind]
    for wavelength in kernel.wavelength.wavelength_nm:
        new = kernel._table_path(kind, kernel._table_key(kind, wavelength))
        old = kernel._table_path(kind, kernel._table_key(kind, wavelength, legacy=True))
        if new.is_file():
            new.replace(old)
    return sorted(Path(kernel.config.cache_directory).glob(f"{prefix}-*.npz"))


def test_table_keys_ignore_labels_but_not_physics(tmp_path):
    first = model("private-bgvd-model:sha256:abc").kernel(config(tmp_path))
    second = model("ntsim.bgvd.Water + installed bgvd_model").kernel(config(tmp_path))
    for wavelength in first.wavelength.wavelength_nm:
        assert first._cache_key(wavelength) == second._cache_key(wavelength)
        assert (first._directional_cache_key(wavelength)
                == second._directional_cache_key(wavelength))
        assert first._cache_key(wavelength, legacy=True) != second._cache_key(wavelength, legacy=True)
        assert first._cache_key(wavelength) != first._cache_key(wavelength, legacy=True)
    murkier = replace(medium(), absorption_per_m=np.array([.041, .05]))
    other = lh.TransportKernel(murkier, first.detector, first.config)
    assert other._cache_key(450.) != first._cache_key(450.)
    assert other._directional_cache_key(450.) != first._directional_cache_key(450.)


def test_load_bgvd_model_and_relabelled_copy_share_keys(tmp_path):
    package = write_private_package(tmp_path / "private")
    loaded = lh.load_bgvd_model(package, dataset="2021")
    relabelled = replace(loaded, medium=replace(loaded.medium, provenance="ntsim water"))
    a, b = loaded.kernel(config(tmp_path)), relabelled.kernel(config(tmp_path))
    for wavelength in a.wavelength.wavelength_nm:
        assert a._cache_key(wavelength) == b._cache_key(wavelength)
        assert a._directional_cache_key(wavelength) == b._directional_cache_key(wavelength)


def test_tables_of_another_platform_are_found_by_their_band_coefficients(tmp_path):
    builder = model().kernel(config(tmp_path)).build(source=FLASH)
    expected = builder.transport(FLASH)
    # the same water reproduced only to the last bits (another CPU or maths library)
    close = replace(medium(), absorption_per_m=np.array([.04, .05]) * (1 + 1e-13))
    other = lh.TransportKernel(close, builder.detector, config(tmp_path, cache_policy="require"))
    assert other._cache_key(450.) != builder._cache_key(450.)
    assert other._stored_table("multipole", 450.) is not None
    np.testing.assert_allclose(other.transport(FLASH).spectrum_pe, expected.spectrum_pe, rtol=1e-9)
    # another water is not taken for it
    murkier = replace(medium(), absorption_per_m=np.array([.04, .05]) * (1 + 1e-6))
    far = lh.TransportKernel(murkier, builder.detector, config(tmp_path, cache_policy="require"))
    assert far._stored_table("multipole", 450.) is None


def test_legacy_multipole_tables_are_read_and_new_ones_use_physics_keys(tmp_path):
    builder = model().kernel(config(tmp_path)).build(source=FLASH)
    expected = builder.transport(FLASH)
    legacy = as_legacy(builder, "multipole")
    assert len(legacy) == 1
    # The same model finds its 0.3.0a1 table by the legacy key ...
    same = model().kernel(config(tmp_path, cache_policy="require"))
    assert same._stored_table("multipole", 450.) == legacy[0]
    np.testing.assert_array_equal(same.transport(FLASH).spectrum_pe, expected.spectrum_pe)
    # ... and a relabelled model by the stored band coefficients.
    relabelled = model("another label").kernel(config(tmp_path, cache_policy="require"))
    assert relabelled._stored_table("multipole", 450.) == legacy[0]
    np.testing.assert_array_equal(relabelled.transport(FLASH).spectrum_pe, expected.spectrum_pe)
    # Reading never writes; a forced rebuild writes the physics-only name.
    assert sorted(tmp_path.glob("transport-*.npz")) == legacy
    model("another label").kernel(config(tmp_path)).build(source=FLASH, force=True)
    names = {path.name for path in tmp_path.glob("transport-*.npz")}
    assert names == {legacy[0].name, f"transport-{builder._cache_key(450.)}.npz"}


def test_other_settings_never_match_a_stored_table(tmp_path):
    model().kernel(config(tmp_path)).build(source=FLASH)
    for changes in (dict(radial_nodes=9), dict(scattering_degree=5),
                    dict(omega_per_ns=np.array([0., .05]))):
        kernel = model("another label").kernel(config(tmp_path, cache_policy="require", **changes))
        assert kernel._stored_table("multipole", 450.) is None
        with pytest.raises(FileNotFoundError, match="required multipole cache is missing"):
            kernel.transport(FLASH)
    water = replace(medium("another label"), scattering_per_m=np.array([.02, .019]))
    kernel = lh.TransportKernel(water, model().detector, config(tmp_path, cache_policy="require"))
    assert kernel._stored_table("multipole", 450.) is None


def test_legacy_directional_tables_and_ge2_pack_are_adopted(tmp_path):
    pytest.importorskip("numba")
    builder = model().kernel(config(tmp_path, angular_backend="auto"))
    expected = builder.transport(TRACK, method="track")
    store = builder.ge2_kernel_store()
    store.bundle(builder, [0, 1])
    store.save(tmp_path / "pack")
    legacy = as_legacy(builder, "directional")
    assert len(legacy) == 2
    header = json.loads((tmp_path / "pack" / "pack.json").read_text())
    for item in header["bundles"]:
        item["key"][0] = [builder._directional_cache_key(w, legacy=True)
                          for w in builder.wavelength.wavelength_nm]
    (tmp_path / "pack" / "pack.json").write_text(json.dumps(header))

    reader = model("another label").kernel(
        config(tmp_path, angular_backend="auto", cache_policy="require"))
    response = reader.transport(TRACK, method="track")
    np.testing.assert_array_equal(response.spectrum_pe, expected.spectrum_pe)
    loaded = reader.ge2_kernel_store(tmp_path / "pack")
    bundle = loaded.bundle(reader, [0, 1])
    assert isinstance(bundle["coefficients"], np.memmap)
    assert list(loaded._bundles) == [ge2_key(reader, [0, 1])]
    assert sorted(tmp_path.glob("directional-*.npz")) == legacy


def ge2_key(kernel, indices):
    return lh.Ge2KernelStore._key(kernel, indices)


def test_model_file_round_trip_is_bit_identical(tmp_path):
    pytest.importorskip("numba")
    original = model("private-bgvd-model:sha256:abc")
    path = original.save(tmp_path / "model" / "bgvd-model")
    assert path == tmp_path / "model" / "bgvd-model.npz"
    loaded = lh.load_model(path)
    for name in ("wavelength_nm", "absorption_per_m", "scattering_per_m",
                 "phase_index", "group_index"):
        np.testing.assert_array_equal(getattr(loaded.medium, name), getattr(original.medium, name))
    assert loaded.medium.g == original.medium.g
    assert loaded.medium.provenance == original.medium.provenance
    assert loaded.detector.provenance == original.detector.provenance
    assert (loaded.dataset, loaded.fingerprint) == (original.dataset, original.fingerprint)
    for name in ("positions_m", "orientations", "effective_area_m2"):
        np.testing.assert_array_equal(getattr(loaded.detector, name),
                                      getattr(original.detector, name))
    assert loaded.detector.identifiers.keys() == original.detector.identifiers.keys()
    for name, value in original.detector.identifiers.items():
        np.testing.assert_array_equal(loaded.detector.identifiers[name], value)
    np.testing.assert_array_equal(loaded.detector.spectral_efficiency.wavelength_knots_nm,
                                  original.detector.spectral_efficiency.wavelength_knots_nm)
    for source, method in ((FLASH, "isotropic"), (TRACK, "track")):
        a = original.kernel(config(tmp_path / "a", angular_backend="auto")).transport(source, method=method)
        b = loaded.kernel(config(tmp_path / "b", angular_backend="auto")).transport(source, method=method)
        np.testing.assert_array_equal(a.spectrum_pe, b.spectrum_pe)
        np.testing.assert_array_equal(a.components_pe, b.components_pe)
    # The second model reads the first model's tables.
    names = sorted(p.name for p in (tmp_path / "a").glob("*.npz"))
    assert names == sorted(p.name for p in (tmp_path / "b").glob("*.npz"))


def test_model_save_requires_a_matching_tabulated_response(tmp_path):
    original = model()
    without = replace(original, om_response=None)
    with pytest.raises(ValueError, match="om_response"):
        without.save(tmp_path / "x.npz")
    other = replace(original, om_response=replace(response(), efficiency=np.full(8, .3)))
    with pytest.raises(ValueError, match="differs"):
        other.save(tmp_path / "x.npz")
    assert not list(tmp_path.iterdir())


def test_model_files_without_schema_still_load(tmp_path):
    original = model()
    om, m, d = original.om_response, original.medium, original.detector
    path = tmp_path / "old.npz"
    np.savez(path, wavelength_nm=m.wavelength_nm, absorption_per_m=m.absorption_per_m,
             scattering_per_m=m.scattering_per_m, phase_index=m.phase_index,
             group_index=m.group_index, g=np.float64(m.g), medium_provenance=np.str_(m.provenance),
             positions_m=d.positions_m, orientations=d.orientations,
             effective_area_m2=d.effective_area_m2, uid=np.array([11101, 11102, 11103]),
             cluster_id=np.array([0, 0, 1]), detector_provenance=np.str_(d.provenance),
             angular_parameters=om.angular_parameters,
             efficiency_wavelength_nm=om.efficiency_wavelength_nm, efficiency=om.efficiency,
             transmission_wavelength_nm=om.transmission_wavelength_nm,
             transmission=om.transmission, dataset=np.str_("ntsim"), fingerprint=np.str_("f"))
    loaded = lh.load_model(path)
    assert sorted(loaded.detector.identifiers) == ["cluster_id", "uid"]
    np.testing.assert_array_equal(loaded.om_response.wavelength_knots_nm,
                                  np.unique(np.r_[om.efficiency_wavelength_nm,
                                                  om.transmission_wavelength_nm]))
    cosine = np.linspace(-1, 1, 41)
    np.testing.assert_array_equal(loaded.detector.angular_acceptance(cosine),
                                  d.angular_acceptance(cosine))


@pytest.mark.skipif(not BUNDLE_MODEL.is_file(), reason="internal data bundle not present")
def test_the_bundle_model_file_loads_with_the_bgvd_model_response():
    loaded = lh.load_model(BUNDLE_MODEL)
    assert len(loaded.detector) == 1440
    assert sorted(loaded.detector.identifiers) == ["cluster_id", "module_id", "string_id", "uid"]
    with np.load(BUNDLE_MODEL) as f:
        w = np.linspace(300., 700., 401)
        expected = (np.interp(w, f["efficiency_wavelength_nm"], f["efficiency"])
                    * np.exp(np.interp(w, f["transmission_wavelength_nm"], np.log(f["transmission"]))))
    np.testing.assert_array_equal(loaded.detector.spectral_efficiency(w), expected)


def write_private_package(root):
    package = root / "bgvd_model"
    (package / "data").mkdir(parents=True)
    (package / "BaikalWater.py").write_text('''
import numpy as np
class BaikalWater:
 def __init__(self):
  self.wavelength=np.array([400.,500.]);self.absorption_inv_length=np.array([.04,.05])
  self.scattering_inv_length=np.array([.02,.018]);self.phase_refraction_index=np.array([1.34,1.36])
  self.group_refraction_index=np.array([1.37,1.39])
''')
    (package / "OpticalModule.py").write_text('''
import numpy as np
radius=.216
angular_parameters=np.array([0.3082,-0.54192,0.19831,0.04912])
eff_values=np.array([0.28,0.35,0.35,0.3,0.22,0.12,0.05,0.02])
eff_wavelengths=np.linspace(300,650,8)
transmission_values=np.array([1.38e-3,5.e-3,0.544,0.804,0.83,0.866,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9])
transmission_wavelengths=np.linspace(350,650,14)
def efficiency(wavelength):
    return np.interp(wavelength, eff_wavelengths, eff_values)
def transmission_gel_glass(wavelength):
    return np.exp(np.interp(wavelength, transmission_wavelengths, np.log(transmission_values)))
''')
    (package / "data" / "median_om_coordinates_2021.csv").write_text(
        "cluster,subcluster,channel,mx_m,my_m,mz_m,dir_x,dir_y,dir_z\n"
        "1,1,1,0,0,0,0,0,-1\n1,1,2,0,0,15,0,0,-1\n")
    return root


def test_load_bgvd_model_is_portable(tmp_path):
    loaded = lh.load_bgvd_model(write_private_package(tmp_path / "private"), dataset="2021")
    assert loaded.om_response is not None
    np.testing.assert_array_equal(loaded.detector.spectral_efficiency.wavelength_knots_nm,
                                  loaded.om_response.wavelength_knots_nm)
    again = lh.load_model(loaded.save(tmp_path / "m.npz"))
    assert sorted(again.detector.identifiers) == ["channel", "cluster", "string"]
    w = np.linspace(380., 520., 57)
    np.testing.assert_array_equal(again.detector.spectral_efficiency(w),
                                  loaded.detector.spectral_efficiency(w))
