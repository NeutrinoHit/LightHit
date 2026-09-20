"""Runtime adapter for a separately supplied private ``bgvd_model`` checkout.

No private table, geometry row or package source is copied into LightHit.  The
adapter reads the checkout at runtime, records content hashes, and deliberately
ignores its scattering indicatrix in favour of Henyey--Greenstein with g=0.9.
"""
from dataclasses import dataclass
from hashlib import sha256
from importlib.util import find_spec, module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np

from .model import DetectorArray, SpectralMedium


def _package_directory(package_path=None):
    if package_path is None:
        spec = find_spec("bgvd_model")
        if spec is None or not spec.submodule_search_locations:
            raise ImportError("Install bgvd-model locally or pass package_path")
        return Path(next(iter(spec.submodule_search_locations))).resolve()
    directory = Path(package_path).expanduser().resolve()
    if (directory / "bgvd_model").is_dir():
        directory /= "bgvd_model"
    if not directory.is_dir():
        raise FileNotFoundError(f"No bgvd_model directory at {directory}")
    return directory


def _load_file(path, name):
    spec = spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load private module {path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _hash(paths):
    digest = sha256()
    for path in paths:
        digest.update(Path(path).read_bytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class BGVDModel:
    medium: SpectralMedium
    detector: DetectorArray
    dataset: str
    fingerprint: str

    def kernel(self, config=None):
        """Create a :class:`TransportKernel` without introducing global state."""
        from .transport import TransportKernel
        return TransportKernel(self.medium, self.detector, config)


def load_bgvd_model(package_path=None, *, dataset="2021", clusters="all", g=0.9):
    """Load water, OM response and measured geometry from private bgvd-model.

    The OM spectral response is exactly
    ``efficiency(lambda) * transmission_gel_glass(lambda)``.  Geometry
    directions in the private CSV point along photon propagation for a head-on
    hit, whereas :class:`DetectorArray` uses +1 towards the source, so the
    private angular polynomial is evaluated at the negative head-on cosine.
    """
    directory = _package_directory(package_path)
    water_path = directory / "BaikalWater.py"
    om_path = directory / "OpticalModule.py"
    geometry_path = directory / "data" / f"median_om_coordinates_{dataset}.csv"
    for path in (water_path, om_path, geometry_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required private BGVD input is missing: {path}")
    water = _load_file(water_path, "_lighthit_bgvd_water").BaikalWater()
    om = _load_file(om_path, "_lighthit_bgvd_om")
    wavelength = np.asarray(water.wavelength, float)
    medium = SpectralMedium(
        wavelength, np.asarray(water.absorption_inv_length, float),
        np.asarray(water.scattering_inv_length, float),
        np.asarray(water.phase_refraction_index, float),
        np.asarray(water.group_refraction_index, float), g=float(g),
        provenance="private-bgvd-model:sha256:" + _hash((water_path,)))
    table = np.genfromtxt(geometry_path, delimiter=",", names=True)
    if not len(table):
        raise ValueError("private BGVD geometry is empty")
    if not isinstance(clusters, str) or clusters != "all":
        selected = np.atleast_1d(np.asarray(clusters, int))
        table = table[np.isin(table["cluster"].astype(int), selected)]
    positions = np.column_stack((table["mx_m"], table["my_m"], table["mz_m"]))
    directions = np.column_stack((table["dir_x"], table["dir_y"], table["dir_z"]))
    parameters = np.asarray(om.angular_parameters, float).copy()

    def angular_acceptance(head_on_cosine):
        x = -np.asarray(head_on_cosine, float)  # private convention: head-on is -1
        return np.maximum(sum(parameters[n] * x ** n for n in range(len(parameters))), 0.0)

    def spectral_efficiency(wavelength_nm):
        wavelength_nm = np.asarray(wavelength_nm, float)
        return (np.asarray(om.efficiency(wavelength_nm), float)
                * np.asarray(om.transmission_gel_glass(wavelength_nm), float))

    fingerprint = _hash((water_path, om_path, geometry_path))
    detector = DetectorArray(
        positions, directions, np.pi * float(om.radius) ** 2,
        angular_acceptance, spectral_efficiency,
        identifiers={"cluster": table["cluster"].astype(int),
                     "string": table["subcluster"].astype(int),
                     "channel": table["channel"].astype(int)},
        provenance=f"private-bgvd-model:{dataset}:sha256:{fingerprint}")
    return BGVDModel(medium, detector, str(dataset), fingerprint)


__all__ = ["BGVDModel", "load_bgvd_model"]
