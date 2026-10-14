"""Moved to ``lighthit.axial_fast``; this name keeps older scripts working."""
import sys

from lighthit import axial_fast as _module

sys.modules[__name__] = _module
