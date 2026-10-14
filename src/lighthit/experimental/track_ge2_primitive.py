"""Moved to ``lighthit.muon.track_ge2_primitive``; this name keeps older scripts working."""
import sys

from lighthit.muon import track_ge2_primitive as _module

sys.modules[__name__] = _module
