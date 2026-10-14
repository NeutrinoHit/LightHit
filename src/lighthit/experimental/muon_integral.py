"""Moved to ``lighthit.muon.muon_integral``; this name keeps older scripts working."""
import sys

from lighthit.muon import muon_integral as _module

sys.modules[__name__] = _module
