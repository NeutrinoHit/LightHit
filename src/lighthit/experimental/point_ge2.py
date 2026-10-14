"""Moved to ``lighthit.muon.point_ge2``; this name keeps older scripts working."""
import sys

from lighthit.muon import point_ge2 as _module

sys.modules[__name__] = _module
