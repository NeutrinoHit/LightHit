"""Public spectral-medium and detector-array contracts."""
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from numpy.polynomial.legendre import leggauss

from .medium import Medium


def _axis(values, name, *, positive=False):
    array = np.asarray(values, float)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite nonempty vector")
    if positive and np.any(array <= 0):
        raise ValueError(f"{name} must be positive")
    return array


@dataclass(frozen=True)
class WavelengthQuadrature:
    """Nodes in nm and integration weights in nm."""
    wavelength_nm: np.ndarray
    weight_nm: np.ndarray

    def __post_init__(self):
        wavelength = _axis(self.wavelength_nm, "wavelength_nm", positive=True)
        weight = _axis(self.weight_nm, "weight_nm", positive=True)
        if wavelength.shape != weight.shape or np.any(np.diff(wavelength) <= 0):
            raise ValueError("quadrature wavelengths must increase and match weights")
        object.__setattr__(self, "wavelength_nm", wavelength)
        object.__setattr__(self, "weight_nm", weight)

    @classmethod
    def gauss_legendre(cls, low_nm, high_nm, nodes=9):
        if not 0 < low_nm < high_nm:
            raise ValueError("require 0 < low_nm < high_nm")
        if isinstance(nodes, bool) or not isinstance(nodes, (int, np.integer)) or nodes < 1:
            raise ValueError("nodes must be a positive integer")
        x, w = leggauss(int(nodes))
        half = (high_nm - low_nm) / 2
        return cls(low_nm + half * (x + 1), half * w)


@dataclass(frozen=True)
class SpectralMedium:
    """Homogeneous optical coefficients sampled as functions of wavelength."""
    wavelength_nm: np.ndarray
    absorption_per_m: np.ndarray
    scattering_per_m: np.ndarray
    phase_index: np.ndarray
    group_index: np.ndarray
    g: float = 0.9
    provenance: str = "user-specified spectral medium"

    def __post_init__(self):
        wavelength = _axis(self.wavelength_nm, "wavelength_nm", positive=True)
        if len(wavelength) < 2 or np.any(np.diff(wavelength) <= 0):
            raise ValueError("wavelength_nm must contain at least two increasing nodes")
        object.__setattr__(self, "wavelength_nm", wavelength)
        for name in ("absorption_per_m", "scattering_per_m", "phase_index", "group_index"):
            value = _axis(getattr(self, name), name, positive=name != "scattering_per_m")
            if value.shape != wavelength.shape or (name == "scattering_per_m" and np.any(value < 0)):
                raise ValueError(f"{name} must match wavelength_nm")
            object.__setattr__(self, name, value)
        if not np.isfinite(self.g) or not -1 < self.g < 1:
            raise ValueError("HG requires finite -1 < g < 1")

    @property
    def wavelength_range_nm(self):
        return float(self.wavelength_nm[0]), float(self.wavelength_nm[-1])

    def sample(self, wavelength_nm):
        wavelength = np.asarray(wavelength_nm, float)
        low, high = self.wavelength_range_nm
        if not np.isfinite(wavelength).all() or np.any((wavelength < low) | (wavelength > high)):
            raise ValueError(f"wavelength outside medium range [{low:g}, {high:g}] nm")
        def interp(values):
            return np.interp(wavelength, self.wavelength_nm, values)
        return {"absorption_per_m": interp(self.absorption_per_m),
                "scattering_per_m": interp(self.scattering_per_m),
                "phase_index": interp(self.phase_index),
                "group_index": interp(self.group_index)}

    def band(self, wavelength_nm):
        sampled = self.sample(float(wavelength_nm))
        return Medium(float(sampled["absorption_per_m"]),
                      float(sampled["scattering_per_m"]), self.g,
                      float(sampled["group_index"]), float(wavelength_nm),
                      self.provenance)


@dataclass(frozen=True)
class DetectorArray:
    """Point OMs with a common angular shape and spectral response.

    ``orientation`` points towards the accepted/head-on hemisphere.
    ``angular_acceptance(+1)`` is therefore the head-on response.
    ``spectral_efficiency`` contains wavelength-dependent quantum efficiency
    and transmission; geometrical area remains separate.
    """
    positions_m: np.ndarray
    orientations: np.ndarray
    effective_area_m2: np.ndarray | float
    angular_acceptance: Callable[[np.ndarray], np.ndarray]
    spectral_efficiency: Callable[[np.ndarray], np.ndarray]
    identifiers: dict = field(default_factory=dict)
    provenance: str = "user-specified detector"

    def __post_init__(self):
        positions = np.asarray(self.positions_m, float)
        orientations = np.asarray(self.orientations, float)
        if (positions.ndim != 2 or positions.shape[1] != 3 or not len(positions)
                or not np.isfinite(positions).all() or orientations.shape != positions.shape
                or not np.isfinite(orientations).all()):
            raise ValueError("positions_m and orientations must be finite (N,3) arrays")
        norm = np.linalg.norm(orientations, axis=1)
        if np.any(norm <= 0):
            raise ValueError("detector orientations must be nonzero")
        orientations = orientations / norm[:, None]
        area = np.broadcast_to(np.asarray(self.effective_area_m2, float), (len(positions),)).copy()
        if not np.isfinite(area).all() or np.any(area <= 0):
            raise ValueError("effective_area_m2 must be finite and positive")
        for name, value in self.identifiers.items():
            if np.asarray(value).shape != (len(positions),):
                raise ValueError(f"identifier {name!r} must contain one value per OM")
        probe = np.linspace(-1, 1, 17)
        angular = np.asarray(self.angular_acceptance(probe), float)
        spectral = np.asarray(self.spectral_efficiency(np.array([350., 450., 550.])), float)
        if (angular.shape != probe.shape or not np.isfinite(angular).all()
                or np.any(angular < 0) or spectral.shape != (3,)
                or not np.isfinite(spectral).all() or np.any(spectral < 0)):
            raise ValueError("detector response functions must return finite nonnegative arrays")
        object.__setattr__(self, "positions_m", positions)
        object.__setattr__(self, "orientations", orientations)
        object.__setattr__(self, "effective_area_m2", area)

    def __len__(self):
        return len(self.positions_m)

    def head_on_cosine(self, source_position_m):
        """Cosine +1 when an OM orientation points from OM towards source."""
        toward_source = np.asarray(source_position_m, float)[None, :] - self.positions_m
        toward_source /= np.linalg.norm(toward_source, axis=1)[:, None]
        return np.sum(self.orientations * toward_source, axis=1)

    def angular_weight(self, source_position_m):
        return np.asarray(self.angular_acceptance(self.head_on_cosine(source_position_m)), float)

    def spectral_weight(self, wavelength_nm):
        value = np.asarray(self.spectral_efficiency(np.asarray(wavelength_nm, float)), float)
        if not np.isfinite(value).all() or np.any(value < 0):
            raise ValueError("nonfinite or negative detector spectral response")
        return value


__all__ = ["WavelengthQuadrature", "SpectralMedium", "DetectorArray"]
