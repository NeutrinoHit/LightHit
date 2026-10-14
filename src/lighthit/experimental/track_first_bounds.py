"""Moved to ``lighthit.muon.track_first_bounds``; this name keeps older scripts working."""
import sys

from lighthit.muon import track_first_bounds as _module

sys.modules[__name__] = _module
