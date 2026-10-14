"""Moved to ``lighthit.muon.positive_fluence``; this name keeps older scripts working."""
import sys

from lighthit.muon import positive_fluence as _module

sys.modules[__name__] = _module
