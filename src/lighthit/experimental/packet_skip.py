"""Moved to ``lighthit.muon.packet_skip``; this name keeps older scripts working."""
import sys

from lighthit.muon import packet_skip as _module

sys.modules[__name__] = _module
