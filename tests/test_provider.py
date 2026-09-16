import numpy as np
import pytest
from lighthit.providers import load_bgvd_water


def test_private_interface_using_only_fake_test_data(tmp_path):
    package=tmp_path/'bgvd_model'; package.mkdir()
    (package/'BaikalWater.py').write_text('''class BaikalWater:
    wavelength = [400., 500.]
    absorption_inv_length = [0.02, 0.04]
    scattering_inv_length = [0.03, 0.05]
    group_refraction_index = [1.3, 1.4]
''')
    # Failing __init__/OM proves these modules are not imported by the adapter.
    (package/'__init__.py').write_text("raise RuntimeError('must not import package initializer')")
    (package/'OpticalModule.py').write_text("raise RuntimeError('must not import OM')")
    medium=load_bgvd_water(tmp_path,wavelength_nm=450,g=.8)
    assert medium.absorption_per_m == pytest.approx(.03)
    assert medium.group_index == pytest.approx(1.35)
    assert medium.g == .8
    with pytest.raises(ValueError):load_bgvd_water(tmp_path,wavelength_nm=600)


def test_missing_private_provider(tmp_path):
    with pytest.raises(FileNotFoundError):load_bgvd_water(tmp_path)
