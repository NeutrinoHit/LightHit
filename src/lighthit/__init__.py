"""LightHit: time-dependent light transport and detector response."""
from ._version import VERSION as __version__
from .medium import Medium, synthetic_medium
from .model import DetectorArray, SpectralMedium, WavelengthQuadrature
from .geometry import (GeometryRegion, DetectorGeometrySummary,
                       describe_geometry, point_source_radial_range)
from .green import PointGreenSolver, SolverSettings, GreenResult
from .single import single_scattering_rate
from .sources import (SourcePose, IsotropicFlash, CherenkovTrack, G4Shower,
                      SpectralLightElements, SyntheticShower)
from .transport import (KernelConfig, CacheProgress, CacheBuildEntry,
                        CacheBuildReport, TransportKernel, TransportResponse,
                        TransportComponent)
from .prompt import PromptConfig, PromptTransportResponse, OrderNotComputedError
from .bgvd import BGVDModel, load_bgvd_model
from .api import build
from .viewer import viewer_payload, merge_event_viewers, write_event_viewer

__all__ = [
    "Medium", "synthetic_medium", "DetectorArray", "SpectralMedium",
    "GeometryRegion", "DetectorGeometrySummary", "describe_geometry",
    "point_source_radial_range",
    "WavelengthQuadrature", "PointGreenSolver", "SolverSettings", "GreenResult",
    "single_scattering_rate", "SourcePose", "IsotropicFlash", "CherenkovTrack",
    "G4Shower", "SyntheticShower", "SpectralLightElements", "KernelConfig",
    "CacheProgress", "CacheBuildEntry", "CacheBuildReport", "TransportKernel",
    "TransportResponse", "TransportComponent",
    "PromptConfig", "PromptTransportResponse", "OrderNotComputedError",
    "BGVDModel", "load_bgvd_model", "build", "viewer_payload",
    "merge_event_viewers", "write_event_viewer",
]
