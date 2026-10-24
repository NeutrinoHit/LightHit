"""Finite scalar-fluence probes around a point source and its reference axis."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class AxisymmetricCell:
    """The volume r_min < r < r_max, mu_min < cos(theta) < mu_max.

    Azimuth covers 2*pi. A directed source and scalar receiver in an isotropic
    medium are axially symmetric, so the azimuth average loses no information.
    Cells are nonabsorbing probes: all visits and recrossings contribute.
    They are neither glass surfaces nor first-entry photon counters.
    """
    r_min_m: float
    r_max_m: float
    mu_min: float
    mu_max: float
    label: str = ""

    def __post_init__(self):
        values = np.array([self.r_min_m, self.r_max_m, self.mu_min, self.mu_max], float)
        if not np.isfinite(values).all():
            raise ValueError("cell bounds must be finite")
        if not 0 < self.r_min_m < self.r_max_m or not -1 <= self.mu_min < self.mu_max <= 1:
            raise ValueError("require 0 < r_min < r_max and -1 <= mu_min < mu_max <= 1")

    @property
    def volume_m3(self):
        return 2*np.pi/3*(self.r_max_m-self.r_min_m)*(
            self.r_max_m**2+self.r_max_m*self.r_min_m+self.r_min_m**2)*(self.mu_max-self.mu_min)

    @property
    def centre_m(self):
        """Representative meridional point; not a replacement for averaging."""
        r = (self.r_min_m+self.r_max_m)/2
        mu = (self.mu_min+self.mu_max)/2
        return np.array([r*np.sqrt((1-mu)*(1+mu)), 0., r*mu])

    def refined(self, factor=0.5):
        """Shrink radial and polar-cosine widths about their centres.

        A forward cap stays attached to mu=1, retaining its ballistic ray.
        A backward cap similarly stays attached to mu=-1.
        """
        if not np.isfinite(factor) or not 0 < factor <= 1:
            raise ValueError("refinement factor must be in (0,1]")
        r = (self.r_min_m+self.r_max_m)/2
        dr = (self.r_max_m-self.r_min_m)*factor/2
        width = (self.mu_max-self.mu_min)*factor
        mu = (self.mu_min+self.mu_max)/2
        lo, hi = mu-width/2, mu+width/2
        if self.mu_max == 1.: lo, hi = 1-width, 1.
        if self.mu_min == -1.: lo, hi = -1., -1+width
        return AxisymmetricCell(r-dr, r+dr, lo, hi, self.label)


def cell_arrays(cells):
    cells = tuple(cells)
    if not cells or any(not isinstance(c, AxisymmetricCell) for c in cells):
        raise ValueError("provide at least one AxisymmetricCell")
    return cells, np.array([[c.r_min_m, c.r_max_m, c.mu_min, c.mu_max] for c in cells], float)
