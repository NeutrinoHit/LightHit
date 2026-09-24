"""Provider-neutral detector geometry summaries for placement and diagnostics."""
from dataclasses import dataclass

import numpy as np

from .model import DetectorArray


def _readonly_vector(value):
    array = np.asarray(value, float).copy()
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class GeometryRegion:
    """Axis-aligned extent and geometric centre of a detector selection."""
    modules: int
    centroid_m: np.ndarray
    bounds_min_m: np.ndarray
    bounds_max_m: np.ndarray
    size_m: np.ndarray
    bounding_radius_m: float

    @classmethod
    def from_positions(cls, positions_m):
        positions = np.asarray(positions_m, float)
        if (positions.ndim != 2 or positions.shape[1:] != (3,) or not len(positions)
                or not np.isfinite(positions).all()):
            raise ValueError("positions_m must be a finite nonempty (N,3) array")
        centroid = positions.mean(axis=0)
        low = positions.min(axis=0)
        high = positions.max(axis=0)
        return cls(
            len(positions), _readonly_vector(centroid), _readonly_vector(low),
            _readonly_vector(high), _readonly_vector(high - low),
            float(np.max(np.linalg.norm(positions - centroid[None, :], axis=1))))

    def as_dict(self):
        return {
            "modules": self.modules,
            "centroid_m": self.centroid_m.tolist(),
            "bounds_min_m": self.bounds_min_m.tolist(),
            "bounds_max_m": self.bounds_max_m.tolist(),
            "size_m": self.size_m.tolist(),
            "bounding_radius_m": self.bounding_radius_m,
        }


@dataclass(frozen=True)
class DetectorGeometrySummary:
    """Whole-array geometry plus optional regions grouped by cluster id."""
    array: GeometryRegion
    clusters: dict[int, GeometryRegion]

    @property
    def centroid_m(self):
        return self.array.centroid_m

    @property
    def bounds_min_m(self):
        return self.array.bounds_min_m

    @property
    def bounds_max_m(self):
        return self.array.bounds_max_m

    @property
    def size_m(self):
        return self.array.size_m

    @property
    def bounding_radius_m(self):
        return self.array.bounding_radius_m

    def as_dict(self):
        return {
            "array": self.array.as_dict(),
            "clusters": {str(key): value.as_dict()
                         for key, value in self.clusters.items()},
        }


def describe_geometry(detector: DetectorArray):
    """Summarise explicit module coordinates without assuming a detector type."""
    cluster = None
    for name in ("cluster_id", "cluster"):
        if name in detector.identifiers:
            cluster = np.asarray(detector.identifiers[name])
            break
    clusters = {}
    if cluster is not None:
        numeric = cluster.astype(float)
        if (not np.isfinite(numeric).all()
                or np.any(numeric != np.floor(numeric))):
            raise ValueError("cluster identifiers must be finite integers")
        for value in np.unique(cluster):
            clusters[int(value)] = GeometryRegion.from_positions(
                detector.positions_m[cluster == value])
    return DetectorGeometrySummary(
        GeometryRegion.from_positions(detector.positions_m), clusters)


def point_source_radial_range(detector: DetectorArray, position_m, *,
                              minimum_range_m=(3.0, 300.0), margin=1.05):
    """Return a cache range covering every OM for a point source.

    This is a geometric coverage rule, not a numerical convergence test. It
    avoids changing which OMs are evaluated when source brightness changes.
    """
    position = np.asarray(position_m, float)
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError("position_m must be a finite 3-vector")
    low, high = map(float, minimum_range_m)
    if not 0 < low < high or not np.isfinite([low, high, margin]).all() or margin <= 1:
        raise ValueError("require a finite positive radial range and margin > 1")
    radii = np.linalg.norm(detector.positions_m - position[None, :], axis=1)
    if np.any(radii < low):
        raise ValueError(
            f"point source approaches {int(np.sum(radii < low))} OMs closer "
            f"than radial minimum {low:g} m")
    return low, max(high, float(radii.max()) * margin)


__all__ = ["GeometryRegion", "DetectorGeometrySummary", "describe_geometry",
           "point_source_radial_range"]
