"""One source contract for a stored shower: steps in, light elements out.

A Geant4 file stores charged-particle vertices. Every consecutive pair of
vertices inside one particle is a step, and each step radiates along its own
chord, at its own time, with its own speed. This module turns those steps into
the only representation the rest of the calculation is allowed to use, so that
the unscattered order, the first order, the moment route and any detailed
control all start from exactly the same light.

What the contract fixes, once:

* the emission line is the **chord** between the two stored vertices, and the
  direction is the chord direction;
* the yield uses the **true path length** stored with the step, not the chord,
  so that a step whose true path is longer than its chord is not dimmed;
* the Cherenkov angle uses the **physical** beta of the particle, never the
  chord divided by the elapsed time;
* time runs linearly along the chord between the two stored times;
* a step below the Cherenkov threshold radiates nothing;
* the row index and the particle uid of every step are kept, so any element
  can be traced back to the stored data.

The yield is the Frank-Tamm number of photons per unit path in one explicit
wavelength band. The band is an input of the calculation, not a hidden
default, and the number of photons scales linearly with it.

Reading uses :mod:`lighthit.experimental.hdf5_minimal` when ``h5py`` is not
installed. Nothing here writes to the input file.
"""
from dataclasses import dataclass
import numpy as np

from .hdf5_minimal import read_dataset

__all__ = ["LightElements", "SourceContract", "load_event", "cherenkov_yield_per_m"]

FINE_STRUCTURE = 7.2973525693e-3


def cherenkov_yield_per_m(beta, phase_index, wavelength_low_nm, wavelength_high_nm,
                          charge_squared=1.0):
    """Frank-Tamm photons per metre of true path in one wavelength band."""
    beta = np.asarray(beta, float)
    inverse = 1.0 / (wavelength_low_nm * 1e-9) - 1.0 / (wavelength_high_nm * 1e-9)
    if inverse <= 0:
        raise ValueError("The wavelength band must be given as (low, high) in nm")
    sine_squared = 1.0 - 1.0 / np.maximum(beta * phase_index, 1e-12) ** 2
    return np.where(beta * phase_index > 1.0,
                    2 * np.pi * FINE_STRUCTURE * charge_squared * sine_squared * inverse,
                    0.0)


@dataclass(frozen=True)
class SourceContract:
    """The choices that turn stored steps into light, kept in one place."""
    phase_index: float = 1.35
    wavelength_low_nm: float = 400.0
    wavelength_high_nm: float = 500.0
    speed_of_light_m_per_ns: float = 0.299792458

    def as_dict(self):
        return {"phase_index": self.phase_index,
                "wavelength_low_nm": self.wavelength_low_nm,
                "wavelength_high_nm": self.wavelength_high_nm,
                "yield_definition": "Frank-Tamm over the stated band, per metre "
                                    "of the stored true path length",
                "emission_line": "chord between the two stored vertices",
                "beta": "physical beta of the step, averaged over its two ends",
                "time": "linear along the chord between the two stored times"}


@dataclass(frozen=True)
class LightElements:
    """Straight radiating elements, one per step above threshold.

    ``photons`` is the number of photons the element emits in the band, spread
    uniformly along its chord. ``cone_cosine`` is ``1/(beta n)``. Times are the
    emission times at the two ends of the chord.
    """
    start_m: np.ndarray
    direction: np.ndarray
    length_m: np.ndarray
    photons: np.ndarray
    cone_cosine: np.ndarray
    start_ns: np.ndarray
    end_ns: np.ndarray
    row_index: np.ndarray
    uid: np.ndarray
    contract: SourceContract
    provenance: dict

    def __len__(self):
        return len(self.length_m)

    @property
    def midpoints_m(self):
        return self.start_m + 0.5 * self.length_m[:, None] * self.direction

    @property
    def centroid_m(self):
        weight = self.photons / self.photons.sum()
        return (weight[:, None] * self.midpoints_m).sum(axis=0)

    @property
    def extent_m(self):
        """Largest distance from the photon-weighted centroid to an endpoint."""
        ends = np.concatenate([self.start_m,
                               self.start_m + self.length_m[:, None] * self.direction])
        return float(np.max(np.linalg.norm(ends - self.centroid_m, axis=1)))

    def summary(self):
        return {"elements": len(self), "photons": float(self.photons.sum()),
                "centroid_m": self.centroid_m.tolist(),
                "extent_m": self.extent_m,
                "time_span_ns": [float(self.start_ns.min()), float(self.end_ns.max())],
                "chord_length_m": {"total": float(self.length_m.sum()),
                                   "median": float(np.median(self.length_m)),
                                   "max": float(self.length_m.max())},
                "cone_cosine": {"min": float(self.cone_cosine.min()),
                                "max": float(self.cone_cosine.max())},
                **self.provenance}

    def moved(self, rotation=None, translation=None, delay_ns=0.0):
        """The same event in a new pose, with every element moved exactly."""
        matrix = np.eye(3) if rotation is None else np.asarray(rotation, float)
        shift = np.zeros(3) if translation is None else np.asarray(translation, float)
        if matrix.shape != (3, 3) or not np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-12):
            raise ValueError("rotation must be a proper 3x3 rotation matrix")
        return LightElements(self.start_m @ matrix.T + shift, self.direction @ matrix.T,
                             self.length_m, self.photons, self.cone_cosine,
                             self.start_ns + delay_ns, self.end_ns + delay_ns,
                             self.row_index, self.uid, self.contract,
                             {**self.provenance, "pose": "moved"})


def load_event(path, event=5, contract=SourceContract(), *, max_elements=None):
    """Read one stored event and apply the source contract to every step.

    Steps are taken within each particle uid: a row whose stored step length is
    zero starts a particle and does not itself radiate. Steps below the
    Cherenkov threshold are dropped, and how many were dropped is reported.
    """
    rows = read_dataset(path, f"event_{int(event)}/tracks")
    names = set(rows.dtype.names or ())
    required = {"uid", "pdgid", "x_m", "y_m", "z_m", "t_ns", "beta", "step_length_m"}
    if not required.issubset(names):
        raise ValueError(f"The G4 dataset must carry {sorted(required)}")
    uid = np.asarray(rows["uid"], np.int64)
    same = np.r_[False, uid[1:] == uid[:-1]]
    index = np.flatnonzero(same)                     # rows that end a step
    if max_elements is not None:
        index = index[:int(max_elements)]
    before, after = index - 1, index
    position = np.column_stack([rows["x_m"], rows["y_m"], rows["z_m"]]).astype(float)
    start = position[before]
    chord = position[after] - start
    length = np.linalg.norm(chord, axis=1)
    keep = length > 0
    beta = 0.5 * (np.asarray(rows["beta"], float)[before]
                  + np.asarray(rows["beta"], float)[after])
    true_path = np.asarray(rows["step_length_m"], float)[after]
    yield_per_m = cherenkov_yield_per_m(beta, contract.phase_index,
                                        contract.wavelength_low_nm,
                                        contract.wavelength_high_nm)
    photons = yield_per_m * true_path
    keep &= photons > 0
    kept = np.flatnonzero(keep)
    direction = chord[kept] / length[kept][:, None]
    provenance = {"file": str(path), "event": int(event),
                  "rows": int(len(rows)), "steps": int(len(index)),
                  "steps_below_threshold_or_zero_chord": int(len(index) - len(kept)),
                  "photons_per_m_median": float(np.median(yield_per_m[kept])),
                  "true_path_over_chord_median":
                      float(np.median(true_path[kept] / length[kept])),
                  "contract": contract.as_dict()}
    return LightElements(start[kept], direction, length[kept], photons[kept],
                         1.0 / (beta[kept] * contract.phase_index),
                         np.asarray(rows["t_ns"], float)[before][kept],
                         np.asarray(rows["t_ns"], float)[after][kept],
                         index[kept].astype(np.int64), uid[after][kept],
                         contract, provenance)
