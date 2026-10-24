"""Runtime adapter for a separately supplied private ``bgvd_model`` checkout.

No private table, geometry row or package source is copied into LightHit.  The
adapter reads the checkout at runtime, records content hashes, and deliberately
ignores its scattering indicatrix in favour of Henyey--Greenstein with g=0.9.
:meth:`BGVDModel.save` and :func:`load_model` keep a loaded model in one
portable ``.npz`` file, so the private checkout is not needed to reuse it.
"""
from dataclasses import dataclass
from hashlib import sha256
from importlib.util import find_spec, module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np

from ._version import VERSION
from .model import DetectorArray, SpectralMedium
from .receiver import SphereSurface

MODEL_SCHEMA = "lighthit/bgvd-model/2"


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


def _angular_acceptance(parameters):
    parameters = np.asarray(parameters, float).copy()

    def angular_acceptance(head_on_cosine):
        x = -np.asarray(head_on_cosine, float)  # private convention: head-on is -1
        return np.maximum(sum(parameters[n] * x ** n for n in range(len(parameters))), 0.0)
    return angular_acceptance


def _vector(values, name, *, increasing=False, positive=False, signed=False):
    array = np.asarray(values, float).copy()
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite nonempty vector")
    if increasing and (len(array) < 2 or np.any(np.diff(array) <= 0)):
        raise ValueError(f"{name} must increase")
    if positive and np.any(array <= 0):
        raise ValueError(f"{name} must be positive")
    if not signed and np.any(array < 0):
        raise ValueError(f"{name} must be nonnegative")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class OMResponse:
    """Tabulated BGVD optical-module response, evaluated as bgvd_model does.

    ``angular_parameters`` are polynomial coefficients in ``x = -cos``, with
    ``cos`` the :class:`DetectorArray` head-on cosine (bgvd_model counts
    head-on as -1); the polynomial is clipped at zero. The spectral response is
    ``efficiency * transmission``: efficiency interpolated linearly
    (``np.interp``), gel/glass transmission linearly in its logarithm.
    ``wavelength_knots_nm`` are the breakpoints advertised to the spectral
    quadratures (both tables' wavelengths by default).
    """
    angular_parameters: np.ndarray
    efficiency_wavelength_nm: np.ndarray
    efficiency: np.ndarray
    transmission_wavelength_nm: np.ndarray
    transmission: np.ndarray
    wavelength_knots_nm: np.ndarray | None = None

    def __post_init__(self):
        values = {
            "angular_parameters": _vector(self.angular_parameters, "angular_parameters",
                                          signed=True),
            "efficiency_wavelength_nm": _vector(self.efficiency_wavelength_nm,
                                                "efficiency_wavelength_nm",
                                                increasing=True, positive=True),
            "efficiency": _vector(self.efficiency, "efficiency"),
            "transmission_wavelength_nm": _vector(self.transmission_wavelength_nm,
                                                  "transmission_wavelength_nm",
                                                  increasing=True, positive=True),
            "transmission": _vector(self.transmission, "transmission", positive=True),
        }
        knots = self.wavelength_knots_nm
        if knots is None:
            knots = np.unique(np.r_[values["efficiency_wavelength_nm"],
                                    values["transmission_wavelength_nm"]])
        values["wavelength_knots_nm"] = _vector(knots, "wavelength_knots_nm", positive=True)
        if values["efficiency"].shape != values["efficiency_wavelength_nm"].shape:
            raise ValueError("efficiency must match efficiency_wavelength_nm")
        if values["transmission"].shape != values["transmission_wavelength_nm"].shape:
            raise ValueError("transmission must match transmission_wavelength_nm")
        for name, value in values.items():
            object.__setattr__(self, name, value)

    @classmethod
    def from_module(cls, om):
        """Tables of a loaded ``bgvd_model.OpticalModule``."""
        return cls(om.angular_parameters, om.eff_wavelengths, om.eff_values,
                   om.transmission_wavelengths, om.transmission_values)

    def functions(self):
        """``(angular_acceptance, spectral_efficiency)`` for :class:`DetectorArray`."""
        angular_acceptance = _angular_acceptance(self.angular_parameters)
        ew, ev = self.efficiency_wavelength_nm, self.efficiency
        tw, log_t = self.transmission_wavelength_nm, np.log(self.transmission)

        def spectral_efficiency(wavelength_nm):
            wavelength_nm = np.asarray(wavelength_nm, float)
            return np.interp(wavelength_nm, ew, ev) * np.exp(np.interp(wavelength_nm, tw, log_t))

        spectral_efficiency.wavelength_knots_nm = self.wavelength_knots_nm
        spectral_efficiency.linear_exponential_tables = (ew, ev, tw, log_t)
        return angular_acceptance, spectral_efficiency

    def check(self, detector):
        """Fail unless ``detector`` evaluates exactly this response."""
        angular, spectral = self.functions()
        cosine = np.linspace(-1.0, 1.0, 401)
        wavelength = np.unique(np.r_[np.linspace(250.0, 750.0, 501), self.wavelength_knots_nm])
        if not (np.array_equal(detector.angular_acceptance(cosine), angular(cosine))
                and np.array_equal(detector.spectral_efficiency(wavelength), spectral(wavelength))):
            raise ValueError("the detector response differs from its tabulated om_response")
        knots = getattr(detector.spectral_efficiency, "wavelength_knots_nm", None)
        if knots is not None and not np.array_equal(np.asarray(knots, float),
                                                    self.wavelength_knots_nm):
            raise ValueError("detector wavelength knots differ from om_response")


_IDENTIFIER_PREFIX = "identifier_"


@dataclass(frozen=True)
class BGVDModel:
    """BGVD water, detector and their identity; ``om_response`` makes it portable."""
    medium: SpectralMedium
    detector: DetectorArray
    dataset: str
    fingerprint: str
    om_response: OMResponse | None = None

    def kernel(self, config=None):
        """Create a :class:`TransportKernel` without introducing global state."""
        from .transport import TransportKernel
        return TransportKernel(self.medium, self.detector, config)

    def save(self, path):
        """Write the model to one ``.npz`` file; read it with :func:`load_model`.

        The file holds the medium arrays, ``g``, positions, orientations,
        effective areas, receiver surfaces, identifiers, provenance strings and the OM response as
        tables, and no code. A model without ``om_response`` cannot be saved,
        because its response functions are opaque callables.
        """
        response = self.om_response
        if response is None:
            raise ValueError("BGVDModel.save needs om_response (the tabulated OM "
                             "response); load_bgvd_model and load_model provide it")
        response.check(self.detector)
        medium, detector = self.medium, self.detector
        identifiers = {}
        for name, value in detector.identifiers.items():
            value = np.asarray(value)
            if value.dtype == object or value.dtype.kind not in "iufbU":
                raise ValueError(f"identifier {name!r} must be numeric or text")
            identifiers[_IDENTIFIER_PREFIX + str(name)] = value
        path = Path(path).expanduser()
        if path.suffix != ".npz":
            path = path.with_name(path.name + ".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.stem}.tmp.npz")
        try:
            np.savez(
                temporary, schema=np.str_(MODEL_SCHEMA), package_version=np.str_(VERSION),
                wavelength_nm=medium.wavelength_nm, absorption_per_m=medium.absorption_per_m,
                scattering_per_m=medium.scattering_per_m, phase_index=medium.phase_index,
                group_index=medium.group_index, g=np.float64(medium.g),
                medium_provenance=np.str_(medium.provenance),
                positions_m=detector.positions_m, orientations=detector.orientations,
                effective_area_m2=detector.effective_area_m2,
                receiver_surface=np.str_('point' if detector.receiver_surface is None else 'sphere'),
                receiver_radius_m=(np.array([], float) if detector.receiver_surface is None else
                                   detector.receiver_surface.radii(len(detector))),
                identifier_names=np.array([str(name) for name in detector.identifiers], str),
                detector_provenance=np.str_(detector.provenance),
                angular_parameters=response.angular_parameters,
                efficiency_wavelength_nm=response.efficiency_wavelength_nm,
                efficiency=response.efficiency,
                transmission_wavelength_nm=response.transmission_wavelength_nm,
                transmission=response.transmission,
                wavelength_knots_nm=response.wavelength_knots_nm,
                dataset=np.str_(self.dataset), fingerprint=np.str_(self.fingerprint),
                **identifiers)
            temporary.replace(path)
        finally:
            if temporary.exists():
                temporary.unlink()
        return path


_MODEL_ARRAYS = {
    "schema", "package_version", "wavelength_nm", "absorption_per_m", "scattering_per_m",
    "phase_index", "group_index", "g", "medium_provenance", "positions_m", "orientations",
    "effective_area_m2", "identifier_names", "detector_provenance", "angular_parameters",
    "efficiency_wavelength_nm", "efficiency", "transmission_wavelength_nm", "transmission",
    "wavelength_knots_nm", "dataset", "fingerprint", "receiver_surface", "receiver_radius_m"}


def load_model(path):
    """Read a :class:`BGVDModel` written by :meth:`BGVDModel.save`.

    Files written before the schema existed (the first ``lighthit-bgvd-data``
    bundle) are accepted: there every other per-OM array is an identifier and
    the wavelength knots are those of both response tables.
    """
    with np.load(Path(path).expanduser(), allow_pickle=False) as f:
        schema = str(f["schema"]) if "schema" in f.files else None
        if schema not in (None, 'lighthit/bgvd-model/1', MODEL_SCHEMA):
            raise ValueError(f"unsupported LightHit model file schema {schema!r}")
        medium = SpectralMedium(
            f["wavelength_nm"], f["absorption_per_m"], f["scattering_per_m"],
            f["phase_index"], f["group_index"], g=float(f["g"]),
            provenance=str(f["medium_provenance"]))
        response = OMResponse(
            f["angular_parameters"], f["efficiency_wavelength_nm"], f["efficiency"],
            f["transmission_wavelength_nm"], f["transmission"],
            f["wavelength_knots_nm"] if "wavelength_knots_nm" in f.files else None)
        modules = len(f["positions_m"])
        if schema is None:
            identifiers = {name: f[name] for name in f.files
                           if name not in _MODEL_ARRAYS and f[name].shape == (modules,)}
        else:
            identifiers = {str(name): f[_IDENTIFIER_PREFIX + str(name)]
                           for name in f["identifier_names"]}
        angular, spectral = response.functions()
        surface = None
        if schema == MODEL_SCHEMA and ('receiver_surface' not in f.files
                                       or 'receiver_radius_m' not in f.files):
            raise ValueError('model schema 2 needs explicit receiver-surface fields')
        if 'receiver_surface' in f.files:
            kind = str(f['receiver_surface'])
            if kind == 'sphere':
                surface = SphereSurface(f['receiver_radius_m'])
            elif kind != 'point':
                raise ValueError(f'unsupported receiver surface {kind!r}')
        detector = DetectorArray(
            f["positions_m"], f["orientations"], f["effective_area_m2"],
            angular, spectral, identifiers=identifiers,
            provenance=str(f["detector_provenance"]), receiver_surface=surface)
        return BGVDModel(medium, detector, str(f["dataset"]), str(f["fingerprint"]),
                         response)


def load_bgvd_model(package_path=None, *, dataset="2021", clusters="all", g=0.9):
    """Load water, OM response and measured geometry from private bgvd-model.

    The OM spectral response is exactly
    ``efficiency(lambda) * transmission_gel_glass(lambda)``, kept as the tables
    of :class:`OMResponse` (checked against the private functions).  Geometry
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
    tabulated = all(hasattr(om, name) for name in (
        "eff_wavelengths", "eff_values", "transmission_wavelengths", "transmission_values"))
    if tabulated:
        response = OMResponse.from_module(om)
        angular_acceptance, spectral_efficiency = response.functions()
    else:
        # A module exposing only functions keeps working, but cannot be saved.
        response = None
        angular_acceptance = _angular_acceptance(om.angular_parameters)

        def spectral_efficiency(wavelength_nm):
            wavelength_nm = np.asarray(wavelength_nm, float)
            return (np.asarray(om.efficiency(wavelength_nm), float)
                    * np.asarray(om.transmission_gel_glass(wavelength_nm), float))

    fingerprint = _hash((water_path, om_path, geometry_path))
    detector = DetectorArray(
        positions, directions, np.pi * float(om.radius) ** 2,
        angular_acceptance, spectral_efficiency,
        receiver_surface=SphereSurface(float(om.radius)),
        identifiers={"cluster": table["cluster"].astype(int),
                     "string": table["subcluster"].astype(int),
                     "channel": table["channel"].astype(int)},
        provenance=f"private-bgvd-model:{dataset}:sha256:{fingerprint}")
    if response is not None:
        # The tables must reproduce the private functions exactly.
        wavelength = np.unique(np.r_[np.linspace(250.0, 750.0, 501),
                                     response.wavelength_knots_nm])
        private = (np.asarray(om.efficiency(wavelength), float)
                   * np.asarray(om.transmission_gel_glass(wavelength), float))
        if not np.array_equal(private, spectral_efficiency(wavelength)):
            raise ValueError("private OM response is not the tabulated bgvd_model interpolation")
    return BGVDModel(medium, detector, str(dataset), fingerprint, response)


__all__ = ["OMResponse", "BGVDModel", "load_bgvd_model", "load_model"]
