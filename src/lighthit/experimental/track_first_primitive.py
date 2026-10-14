"""Moved to ``lighthit.muon.track_first_primitive``; this name keeps older scripts working."""
import sys

from lighthit.muon import track_first_primitive as _module

sys.modules[__name__] = _module
