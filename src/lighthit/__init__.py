"""LightHit: time-dependent light transport and detector response."""
from .medium import Medium, synthetic_medium
from .model import DetectorArray, SpectralMedium, WavelengthQuadrature
from .green import PointGreenSolver, SolverSettings, GreenResult
from .single import single_scattering_rate
from .sources import IsotropicFlash, CherenkovTrack, G4Shower, SpectralLightElements
from .transport import KernelConfig, TransportKernel, TransportResponse
from .bgvd import BGVDModel, load_bgvd_model
from .api import build
from .viewer import viewer_payload, write_event_viewer

__all__ = [
    "Medium", "synthetic_medium", "DetectorArray", "SpectralMedium",
    "WavelengthQuadrature", "PointGreenSolver", "SolverSettings", "GreenResult",
    "single_scattering_rate", "IsotropicFlash", "CherenkovTrack", "G4Shower",
    "SpectralLightElements", "KernelConfig", "TransportKernel", "TransportResponse",
    "BGVDModel", "load_bgvd_model", "build", "viewer_payload",
    "write_event_viewer",
]
__version__ = "0.2.0a6"
