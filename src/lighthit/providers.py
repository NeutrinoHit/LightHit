"""Optional private-water adapter: code only, no calibration arrays.

The API matches BaikalWater fields used by the supplied baikal-rte adapter.
This stage does NOT import OpticalModule: detector efficiency is identically 1.
"""
from hashlib import sha256
from importlib.util import find_spec, module_from_spec, spec_from_file_location
from pathlib import Path
import numpy as np
from .medium import Medium


def load_bgvd_water(package_path=None, *, wavelength_nm=450.0, g=0.9):
    """Read one wavelength from a separately supplied bgvd_model checkout.

    g is an explicitly chosen HG parameter; it is not provided by the water
    attenuation arrays. No data is downloaded, cached or logged by this function.
    """
    if package_path is None:
        spec = find_spec("bgvd_model")
        if spec is None or not spec.submodule_search_locations:
            raise ImportError("Supply the private bgvd_model checkout path or install it locally")
        directory = Path(next(iter(spec.submodule_search_locations)))
    else:
        directory = Path(package_path).expanduser().resolve()
        if (directory / "bgvd_model").is_dir():
            directory /= "bgvd_model"
    path = directory / "BaikalWater.py"
    if not path.is_file():
        raise FileNotFoundError(f"Expected the private module at {path}")
    spec = spec_from_file_location("_lighthit_private_water", path)
    if spec is None or spec.loader is None:
        raise ImportError("Unable to load private BaikalWater.py")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    water = module.BaikalWater()
    nodes = np.asarray(water.wavelength, float)
    if nodes.ndim != 1 or len(nodes) < 2 or not np.isfinite(nodes).all() or np.any(np.diff(nodes) <= 0):
        raise ValueError("Provider wavelength array must be finite and increasing")
    if not np.isfinite(wavelength_nm) or not nodes[0] <= wavelength_nm <= nodes[-1]:
        raise ValueError("Requested wavelength lies outside the private water table")

    def sample(name):
        values = np.asarray(getattr(water, name), float)
        if values.shape != nodes.shape or not np.isfinite(values).all():
            raise ValueError(f"Invalid provider field: {name}")
        return float(np.interp(wavelength_nm, nodes, values))

    return Medium(sample("absorption_inv_length"), sample("scattering_inv_length"),
                  g, sample("group_refraction_index"), wavelength_nm,
                  "private-bgvd-water:sha256:" + sha256(path.read_bytes()).hexdigest())
