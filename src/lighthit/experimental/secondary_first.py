"""Moved to ``lighthit.muon.secondary_first``; this name keeps older scripts working."""
import sys

from lighthit.muon import secondary_first as _module

sys.modules[__name__] = _module
