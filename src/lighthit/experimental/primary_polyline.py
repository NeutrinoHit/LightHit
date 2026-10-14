"""Moved to ``lighthit.muon.primary_polyline``; this name keeps older scripts working."""
import sys

from lighthit.muon import primary_polyline as _module

sys.modules[__name__] = _module
