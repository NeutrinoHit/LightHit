"""Public source objects, including the two-field Cherenkov spectrum."""
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

import numpy as np

from .experimental.g4_source import FINE_STRUCTURE, LightElements, SourceContract
from .experimental.hdf5_minimal import read_dataset


@dataclass(frozen=True)
class IsotropicFlash:
    position_m: np.ndarray
    photons: float
    wavelength_nm: float | None = None
    spectral_shape: Callable[[np.ndarray], np.ndarray] | None = None
    time_ns: float = 0.0

    def __post_init__(self):
        position = np.asarray(self.position_m, float)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError("position_m must be a finite 3-vector")
        if not np.isfinite(self.photons) or self.photons < 0 or not np.isfinite(self.time_ns):
            raise ValueError("photons must be nonnegative and time_ns finite")
        if (self.wavelength_nm is None) == (self.spectral_shape is None):
            raise ValueError("provide exactly one of wavelength_nm or spectral_shape")
        if self.wavelength_nm is not None and (not np.isfinite(self.wavelength_nm)
                                                or self.wavelength_nm <= 0):
            raise ValueError("wavelength_nm must be positive")
        object.__setattr__(self, "position_m", position)

    @classmethod
    def monochromatic(cls, position_m, photons, wavelength_nm, *, time_ns=0.0):
        return cls(position_m, photons, wavelength_nm=float(wavelength_nm), time_ns=time_ns)

    @classmethod
    def broadband(cls, position_m, photons, spectral_shape, *, time_ns=0.0):
        return cls(position_m, photons, spectral_shape=spectral_shape, time_ns=time_ns)


@dataclass(frozen=True)
class SpectralLightElements:
    """One geometry with the two wavelength-independent Cherenkov fields.

    Per nanometre,

    ``dN/dlambda = lambda^-2 * coefficient0
                   + lambda^-2*n_phase(lambda)^-2 * coefficient2``.

    ``coefficient2`` includes its physical negative sign.  The axial engine
    internally compiles ``-coefficient2`` as a positive field and subtracts it.
    The angular cone is frozen at ``reference_phase_index``; this is the stated
    two-field approximation, not an unrecorded per-wavelength cone update.
    """
    start_m: np.ndarray
    direction: np.ndarray
    length_m: np.ndarray
    coefficient0: np.ndarray
    coefficient2: np.ndarray
    beta: np.ndarray
    start_ns: np.ndarray
    end_ns: np.ndarray
    row_index: np.ndarray
    uid: np.ndarray
    reference_phase_index: float = 1.35
    provenance: dict | None = None

    def __post_init__(self):
        start = np.asarray(self.start_m, float)
        direction = np.asarray(self.direction, float)
        count = len(start)
        if start.shape != (count, 3) or direction.shape != start.shape or not count:
            raise ValueError("start_m and direction must be nonempty (N,3) arrays")
        arrays = {}
        for name in ("length_m", "coefficient0", "coefficient2", "beta",
                     "start_ns", "end_ns", "row_index", "uid"):
            value = np.asarray(getattr(self, name))
            if value.shape != (count,) or not np.isfinite(value.astype(float)).all():
                raise ValueError(f"{name} must contain one finite value per element")
            arrays[name] = value
        norm = np.linalg.norm(direction, axis=1)
        if (not np.isfinite(start).all() or np.any(norm <= 0)
                or np.any(arrays["length_m"] <= 0) or np.any(arrays["coefficient0"] < 0)
                or np.any(arrays["coefficient2"] > 0) or np.any(arrays["beta"] <= 0)):
            raise ValueError("invalid spectral element geometry or coefficients")
        object.__setattr__(self, "start_m", start)
        object.__setattr__(self, "direction", direction / norm[:, None])
        for name, value in arrays.items():
            object.__setattr__(self, name, value)

    def __len__(self):
        return len(self.length_m)

    @property
    def midpoints_m(self):
        return self.start_m + 0.5 * self.length_m[:, None] * self.direction

    @property
    def centroid_m(self):
        weight = self.coefficient0 / self.coefficient0.sum()
        return np.sum(weight[:, None] * self.midpoints_m, axis=0)

    @property
    def extent_m(self):
        ends = np.concatenate((self.start_m,
                               self.start_m + self.length_m[:, None] * self.direction))
        return float(np.max(np.linalg.norm(ends - self.centroid_m, axis=1)))

    def photon_density_per_nm(self, wavelength_nm, phase_index):
        wavelength = np.asarray(wavelength_nm, float)
        phase = np.asarray(phase_index, float)
        return (self.coefficient0[:, None] / wavelength[None, :] ** 2
                + self.coefficient2[:, None]
                / (wavelength[None, :] ** 2 * phase[None, :] ** 2))

    def field(self, index):
        if index not in (0, 2):
            raise ValueError("field index must be 0 or 2")
        photons = self.coefficient0 if index == 0 else -self.coefficient2
        contract = SourceContract(phase_index=self.reference_phase_index,
                                  wavelength_low_nm=350.0, wavelength_high_nm=610.0)
        return LightElements(
            self.start_m, self.direction, self.length_m, photons,
            1.0 / (self.beta * self.reference_phase_index),
            self.start_ns, self.end_ns, self.row_index.astype(np.int64),
            self.uid.astype(np.int64), contract,
            {**(self.provenance or {}), "spectral_field": index})

    def subset(self, mask):
        mask = np.asarray(mask, bool)
        if mask.shape != (len(self),) or not np.any(mask):
            raise ValueError("subset mask must retain at least one element")
        return SpectralLightElements(
            self.start_m[mask], self.direction[mask], self.length_m[mask],
            self.coefficient0[mask], self.coefficient2[mask], self.beta[mask],
            self.start_ns[mask], self.end_ns[mask], self.row_index[mask],
            self.uid[mask], self.reference_phase_index,
            {**(self.provenance or {}), "subset_elements": int(mask.sum())})

    def moved(self, rotation=None, translation=None, delay_ns=0.0):
        matrix = np.eye(3) if rotation is None else np.asarray(rotation, float)
        shift = np.zeros(3) if translation is None else np.asarray(translation, float)
        if (matrix.shape != (3, 3) or not np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-12)
                or not np.isclose(np.linalg.det(matrix), 1, atol=1e-12)):
            raise ValueError("rotation must be a proper orthogonal matrix")
        return replace(self, start_m=self.start_m @ matrix.T + shift,
                       direction=self.direction @ matrix.T,
                       start_ns=self.start_ns + delay_ns,
                       end_ns=self.end_ns + delay_ns,
                       provenance={**(self.provenance or {}), "pose": "moved"})


@dataclass(frozen=True)
class CherenkovTrack:
    start_m: np.ndarray
    direction: np.ndarray
    length_m: float
    beta: float = 1.0
    time_ns: float = 0.0
    reference_phase_index: float = 1.35

    def to_elements(self, *, step_m=0.5):
        start = np.asarray(self.start_m, float)
        direction = np.asarray(self.direction, float)
        direction /= np.linalg.norm(direction)
        count = max(1, int(np.ceil(self.length_m / step_m)))
        length = self.length_m / count
        offset = np.arange(count) * length
        points = start[None, :] + offset[:, None] * direction[None, :]
        beta = np.full(count, self.beta)
        common = 2 * np.pi * FINE_STRUCTURE * 1e9 * length
        times = self.time_ns + offset / (self.beta * 0.299792458)
        return SpectralLightElements(
            points, np.broadcast_to(direction, (count, 3)), np.full(count, length),
            np.full(count, common), -np.full(count, common) / beta ** 2, beta,
            times, times + length / (self.beta * 0.299792458),
            np.arange(count), np.zeros(count, np.int64), self.reference_phase_index,
            {"source": "CherenkovTrack", "step_m": length})


@dataclass(frozen=True)
class G4Shower:
    elements: SpectralLightElements
    input_path: str
    event: int

    @classmethod
    def from_hdf5(cls, path, event=0, *, reference_phase_index=1.35,
                  max_elements=None):
        path = Path(path)
        rows = read_dataset(path, f"event_{int(event)}/tracks")
        names = set(rows.dtype.names or ())
        required = {"uid", "x_m", "y_m", "z_m", "t_ns", "beta", "step_length_m"}
        if not required.issubset(names):
            raise ValueError(f"G4 tracks must carry {sorted(required)}")
        uid = np.asarray(rows["uid"], np.int64)
        index = np.flatnonzero(np.r_[False, uid[1:] == uid[:-1]])
        if max_elements is not None:
            index = index[:int(max_elements)]
        before, after = index - 1, index
        position = np.column_stack((rows["x_m"], rows["y_m"], rows["z_m"])).astype(float)
        start = position[before]
        chord = position[after] - start
        length = np.linalg.norm(chord, axis=1)
        beta = 0.5 * (np.asarray(rows["beta"], float)[before]
                      + np.asarray(rows["beta"], float)[after])
        true_path = np.asarray(rows["step_length_m"], float)[after]
        keep = (length > 0) & (true_path > 0) & (beta * reference_phase_index > 1)
        kept = np.flatnonzero(keep)
        common = 2 * np.pi * FINE_STRUCTURE * 1e9 * true_path[kept]
        elements = SpectralLightElements(
            start[kept], chord[kept] / length[kept, None], length[kept],
            common, -common / beta[kept] ** 2, beta[kept],
            np.asarray(rows["t_ns"], float)[before][kept],
            np.asarray(rows["t_ns"], float)[after][kept],
            index[kept].astype(np.int64), uid[after][kept], reference_phase_index,
            {"file": str(path), "event": int(event), "rows": int(len(rows)),
             "spectral_model": "two fields S0=lambda^-2, S2=lambda^-2*n_phase^-2"})
        return cls(elements, str(path), int(event))


__all__ = ["IsotropicFlash", "SpectralLightElements", "CherenkovTrack", "G4Shower"]
