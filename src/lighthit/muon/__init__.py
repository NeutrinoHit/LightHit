"""Muon light transport.

``MuonTransport(kernel)(source, ancestry, primary_uid=uid)`` returns the
expected photoelectrons of every OM for a G4 muon event. The prepared tables
are described by ``muon-data.json`` (see :mod:`lighthit.muon.data`).
The modules of this package are the parts of that route; alternatives that
were tried and not adopted live in :mod:`lighthit.experimental`.
"""
from .api import FREQUENCIES_PER_NS, READOUT_PER_NS, MuonTransport, muon_kernel, transport_muon
from .data import MuonData, write_manifest
from .joint_muon import JointMuonConfig
from .preview import independent_preview

MuonConfig = JointMuonConfig

__all__ = ["MuonTransport", "MuonData", "MuonConfig", "JointMuonConfig", "transport_muon",
           "muon_kernel", "independent_preview", "write_manifest",
           "FREQUENCIES_PER_NS", "READOUT_PER_NS"]
