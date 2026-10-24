"""Receiver surfaces supplied by the detector geometry, separate from area."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SphereSurface:
    """Spherical glass boundary, with one radius or a radius per OM.

    Radii describe geometry. Effective collection areas and efficiencies
    remain independent detector-response quantities.
    """
    radius_m: np.ndarray | float

    def __post_init__(self):
        radius = np.asarray(self.radius_m, float)
        if radius.ndim > 1 or not np.isfinite(radius).all() or np.any(radius <= 0):
            raise ValueError('surface radius_m must be finite and positive')
        object.__setattr__(self, 'radius_m', radius.copy())

    def radii(self, count):
        """One physical radius for each receiver."""
        return np.broadcast_to(self.radius_m, (count,))

    def subset(self, indices):
        return self if self.radius_m.ndim == 0 else SphereSurface(self.radius_m[indices])

    def entry_distance(self, origin_m, direction, *, receiver=0):
        """First intersection of a unit ray with the local glass boundary.

        Origins are relative to the OM centre; misses return infinity.
        An origin inside the boundary is already at the receiver.
        """
        origin = np.asarray(origin_m, float)
        direction = np.asarray(direction, float)
        if origin.shape[-1:] != (3,) or direction.shape[-1:] != (3,):
            raise ValueError('ray origins and directions need three coordinates')
        if not np.isfinite(origin).all() or not np.isfinite(direction).all():
            raise ValueError('ray origins and directions must be finite')
        if not np.allclose(np.linalg.norm(direction, axis=-1), 1., rtol=1e-12, atol=1e-12):
            raise ValueError('ray directions must be unit vectors')
        radius = self.radius_m if self.radius_m.ndim == 0 else self.radius_m[receiver]
        projection = -np.sum(origin * direction, axis=-1)
        outside = np.sum(origin * origin, axis=-1) - radius * radius
        discriminant = projection * projection - outside
        root = np.sqrt(np.maximum(discriminant, 0.))
        denominator = projection + root
        distance = np.divide(outside, denominator, out=np.full_like(outside, np.inf),
                             where=denominator > 0)
        return np.where(outside <= 0, 0.,
                        np.where((discriminant >= 0) & (projection > 0), distance, np.inf))
