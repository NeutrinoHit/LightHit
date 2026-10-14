"""Moved to ``lighthit.muon.muon_packets``; this name keeps older scripts working."""
import sys

from lighthit.muon import muon_packets as _module

sys.modules[__name__] = _module
