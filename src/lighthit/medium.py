"""One monochromatic band of a homogeneous, infinite medium; metres and ns."""
from dataclasses import dataclass
import math

C_VACUUM_M_PER_NS = 0.299792458


@dataclass(frozen=True)
class Medium:
    """HG scattering; no detector response or spectral averaging is included.

    ``g`` is a model input, not inferred from either attenuation coefficient.
    Absorption must be positive in this first implementation.
    """
    absorption_per_m: float
    scattering_per_m: float
    g: float
    group_index: float
    wavelength_nm: float = 450.0
    provenance: str = "user-specified"

    def __post_init__(self):
        for name in ("absorption_per_m", "scattering_per_m", "g", "group_index", "wavelength_nm"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if self.absorption_per_m <= 0 or self.scattering_per_m < 0:
            raise ValueError("Require absorption_per_m > 0 and scattering_per_m >= 0")
        if not -1 < self.g < 1:
            raise ValueError("HG requires -1 < g < 1")
        if self.group_index <= 0 or self.wavelength_nm <= 0:
            raise ValueError("Positive group index and wavelength required")

    @property
    def extinction_per_m(self):
        return self.absorption_per_m + self.scattering_per_m

    @property
    def speed_m_per_ns(self):
        return C_VACUUM_M_PER_NS / self.group_index


def synthetic_medium():
    """Explicit test parameters, chosen near typical deep Lake Baikal water at
    450 nm (absorption exceeding scattering, strongly forward-peaked g) but not
    equal to any measured value: this is not a calibration. See PROVENANCE.md.
    """
    return Medium(0.07, 0.022, 0.9, 1.36, 450.0, "synthetic-near-baikal-not-calibration")
