"""Public source objects, including the two-field Cherenkov spectrum."""
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

import numpy as np

from .experimental.g4_source import FINE_STRUCTURE, LightElements, SourceContract
from .experimental.hdf5_minimal import read_dataset


def _unit_vector(value, name):
    vector = np.asarray(value, float)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError(f"{name} must be a finite 3-vector")
    norm = np.linalg.norm(vector)
    if norm <= 0:
        raise ValueError(f"{name} must be nonzero")
    return vector / norm


def _principal_axis(elements):
    points = elements.midpoints_m
    weight = np.asarray(elements.coefficient0, float)
    share = weight / weight.sum()
    centre = np.sum(share[:, None] * points, axis=0)
    offset = points - centre
    covariance = np.einsum("i,ij,ik->jk", share, offset, offset)
    values, vectors = np.linalg.eigh(covariance)
    if float(np.max(np.abs(values))) <= 1e-24:
        axis = np.sum(share[:, None] * elements.direction, axis=0)
    else:
        axis = vectors[:, int(np.argmax(values))]
    if float(np.sum(share * (elements.direction @ axis))) < 0:
        axis = -axis
    return centre, axis / np.linalg.norm(axis)


def _rotation_between(source, target):
    source = _unit_vector(source, "source direction")
    target = _unit_vector(target, "target direction")
    cosine = float(np.clip(source @ target, -1.0, 1.0))
    if np.isclose(cosine, 1.0, atol=1e-14):
        return np.eye(3)
    if np.isclose(cosine, -1.0, atol=1e-14):
        helper = np.array([1.0, 0.0, 0.0])
        if abs(float(source @ helper)) > 0.9:
            helper = np.array([0.0, 1.0, 0.0])
        axis = np.cross(source, helper)
        axis /= np.linalg.norm(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)
    cross = np.cross(source, target)
    skew = np.array([[0.0, -cross[2], cross[1]],
                     [cross[2], 0.0, -cross[0]],
                     [-cross[1], cross[0], 0.0]])
    sine2 = float(cross @ cross)
    return np.eye(3) + skew + skew @ skew * ((1.0 - cosine) / sine2)


@dataclass(frozen=True)
class SourcePose:
    """Explicit target centroid, axis and earliest emission time for a source."""
    position_m: np.ndarray
    direction: np.ndarray
    time_ns: float = 0.0

    def __post_init__(self):
        position = np.asarray(self.position_m, float).copy()
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError("position_m must be a finite 3-vector")
        if not np.isfinite(self.time_ns):
            raise ValueError("time_ns must be finite")
        direction = _unit_vector(self.direction, "direction").copy()
        position.setflags(write=False)
        direction.setflags(write=False)
        object.__setattr__(self, "position_m", position)
        object.__setattr__(self, "direction", direction)


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

    @property
    def principal_axis(self):
        return _principal_axis(self)[1]

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

    def placed(self, pose: SourcePose):
        """Move the photon-weighted centroid and principal axis to ``pose``.

        ``pose.time_ns`` is the new earliest element start time. The full
        proper rotation and translation are recorded in source provenance.
        """
        if not isinstance(pose, SourcePose):
            raise TypeError("pose must be a SourcePose")
        source_centroid, source_axis = _principal_axis(self)
        rotation = _rotation_between(source_axis, pose.direction)
        translation = pose.position_m - rotation @ source_centroid
        delay = float(pose.time_ns - np.min(self.start_ns))
        return replace(
            self,
            start_m=self.start_m @ rotation.T + translation,
            direction=self.direction @ rotation.T,
            start_ns=self.start_ns + delay,
            end_ns=self.end_ns + delay,
            provenance={
                **(self.provenance or {}),
                "pose": {
                    "source_centroid_m": source_centroid.tolist(),
                    "source_direction": source_axis.tolist(),
                    "target_centroid_m": pose.position_m.tolist(),
                    "target_direction": pose.direction.tolist(),
                    "target_earliest_time_ns": float(pose.time_ns),
                    "rotation": rotation.tolist(),
                    "translation_m": translation.tolist(),
                    "delay_ns": delay,
                },
            },
        )


@dataclass(frozen=True)
class CherenkovTrack:
    start_m: np.ndarray
    direction: np.ndarray
    length_m: float
    beta: float = 1.0
    time_ns: float = 0.0
    reference_phase_index: float = 1.35

    @classmethod
    def centered(cls, position_m, direction, length_m, *, beta=1.0,
                 time_ns=0.0, reference_phase_index=1.35):
        direction = _unit_vector(direction, "direction")
        position = np.asarray(position_m, float)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError("position_m must be a finite 3-vector")
        return cls(position - 0.5 * float(length_m) * direction, direction,
                   length_m, beta, time_ns, reference_phase_index)

    def placed(self, pose: SourcePose):
        return CherenkovTrack.centered(
            pose.position_m, pose.direction, self.length_m, beta=self.beta,
            time_ns=pose.time_ns, reference_phase_index=self.reference_phase_index)

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

    @property
    def centroid_m(self):
        return self.elements.centroid_m

    @property
    def principal_axis(self):
        return self.elements.principal_axis

    @property
    def extent_m(self):
        return self.elements.extent_m

    def placed(self, pose: SourcePose):
        return replace(self, elements=self.elements.placed(pose))

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


@dataclass(frozen=True)
class SyntheticShower:
    """Deterministic synthetic axial cascade for examples and smoke tests."""
    elements: SpectralLightElements
    seed: int = 0

    @classmethod
    def gaussian(cls, pose: SourcePose, *, charged_track_length_m=100.0,
                 longitudinal_sigma_m=1.2, transverse_sigma_m=0.08,
                 elements=64, beta=0.999, reference_phase_index=1.35,
                 seed=0):
        if elements < 3:
            raise ValueError("synthetic shower needs at least three elements")
        if not np.isfinite(beta) or not 0 < beta <= 1:
            raise ValueError("beta must be finite and in (0, 1]")
        if (not np.isfinite(reference_phase_index)
                or reference_phase_index <= 0):
            raise ValueError("reference_phase_index must be finite and positive")
        for name, value in (("charged_track_length_m", charged_track_length_m),
                            ("longitudinal_sigma_m", longitudinal_sigma_m),
                            ("transverse_sigma_m", transverse_sigma_m)):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        rng = np.random.default_rng(seed)
        z = np.linspace(-3 * longitudinal_sigma_m, 3 * longitudinal_sigma_m,
                        int(elements))
        share = np.exp(-0.5 * (z / longitudinal_sigma_m) ** 2)
        share /= share.sum()
        transverse = rng.normal(scale=transverse_sigma_m, size=(len(z), 2))
        angular = rng.normal(scale=0.025, size=(len(z), 2))
        start = np.column_stack((transverse, z))
        direction = np.column_stack((angular, np.ones(len(z))))
        direction /= np.linalg.norm(direction, axis=1)[:, None]
        segment_length = max(0.01, 6 * longitudinal_sigma_m / len(z))
        length = np.full(len(z), segment_length)
        common = 2 * np.pi * FINE_STRUCTURE * 1e9 * charged_track_length_m * share
        time = (z - z.min()) / (beta * 0.299792458)
        raw = SpectralLightElements(
            start, direction, length, common, -common / beta ** 2,
            np.full(len(z), beta), time, time + length / (beta * 0.299792458),
            np.arange(len(z)), np.zeros(len(z), np.int64), reference_phase_index,
            {"source": "SyntheticShower", "seed": int(seed),
             "charged_track_length_m": float(charged_track_length_m),
             "longitudinal_sigma_m": float(longitudinal_sigma_m),
             "transverse_sigma_m": float(transverse_sigma_m)})
        return cls(raw.placed(pose), int(seed))

    @property
    def centroid_m(self):
        return self.elements.centroid_m

    @property
    def principal_axis(self):
        return self.elements.principal_axis

    @property
    def extent_m(self):
        return self.elements.extent_m

    def placed(self, pose: SourcePose):
        return replace(self, elements=self.elements.placed(pose))


__all__ = ["SourcePose", "IsotropicFlash", "SpectralLightElements",
           "CherenkovTrack", "G4Shower", "SyntheticShower"]
