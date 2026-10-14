"""Shower light transport.

``ShowerTransport(kernel)(source)`` returns the expected photoelectrons of
every OM for a G4 shower (no dominant muon): exact continuous-spectrum direct
light, band-fused grouped first scattering, and the axial shower basis on
band-folded RTE kernels for multiple scattering. Prepared tables are found
through ``shower-data.json`` (or the muon manifest) next to the RTE cache;
see :mod:`lighthit.shower.data`. The previous per-band route remains
``TransportKernel.transport_hybrid(..., ge2_basis=ShowerBasis())``.
"""
from .api import (FREQUENCIES_PER_NS, READOUT_PER_NS, ShowerConfig, ShowerTransport,
                  shower_kernel, transport_shower)
from .data import ShowerData

__all__ = ["ShowerTransport", "ShowerConfig", "ShowerData", "transport_shower",
           "shower_kernel", "FREQUENCIES_PER_NS", "READOUT_PER_NS"]
