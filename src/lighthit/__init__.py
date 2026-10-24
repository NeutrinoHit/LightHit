"""LightHit: time-dependent light transport and detector response."""
from ._version import VERSION as __version__
from .medium import Medium, synthetic_medium
from .model import DetectorArray, SpectralMedium, WavelengthQuadrature
from .receiver import SphereSurface
from .geometry import (GeometryRegion, DetectorGeometrySummary,
                       describe_geometry, point_source_radial_range)
from .green import PointGreenSolver, SolverSettings, GreenResult
from .single import single_scattering_rate
from .sources import (SourcePose, IsotropicFlash, CherenkovTrack, G4Shower,
                      SpectralLightElements, SyntheticShower)
from .transport import (KernelConfig, CacheProgress, CacheBuildEntry,
                        CacheBuildReport, TransportKernel, TransportResponse,
                        HybridTransportResponse,
                        TransportComponent)
from .prompt import PromptConfig, PromptTransportResponse, OrderNotComputedError
from .shower_basis import ShowerBasis, Ge2KernelStore
from .bgvd import BGVDModel, OMResponse, load_bgvd_model, load_model
from .api import build
from .viewer import viewer_payload, merge_event_viewers, write_event_viewer

__all__ = [
    "Medium", "synthetic_medium", "DetectorArray", "SpectralMedium", "SphereSurface",
    "GeometryRegion", "DetectorGeometrySummary", "describe_geometry",
    "point_source_radial_range",
    "WavelengthQuadrature", "PointGreenSolver", "SolverSettings", "GreenResult",
    "single_scattering_rate", "SourcePose", "IsotropicFlash", "CherenkovTrack",
    "G4Shower", "SyntheticShower", "SpectralLightElements", "KernelConfig",
    "ShowerBasis", "Ge2KernelStore",
    "CacheProgress", "CacheBuildEntry", "CacheBuildReport", "TransportKernel",
    "TransportResponse", "HybridTransportResponse", "TransportComponent",
    "PromptConfig", "PromptTransportResponse", "OrderNotComputedError",
    "BGVDModel", "OMResponse", "load_bgvd_model", "load_model", "build", "viewer_payload",
    "merge_event_viewers", "write_event_viewer", "MuonTransport", "MuonData",
]


def __getattr__(name):
    # The muon route needs Numba; import it only when asked for.
    if name in ("MuonTransport", "MuonData"):
        from . import muon
        return getattr(muon, name)
    raise AttributeError(f"module 'lighthit' has no attribute {name!r}")

