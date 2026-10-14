"""Moved to ``lighthit.axial_source``; this name keeps older scripts working."""
import sys

from lighthit import axial_source as _module

sys.modules[__name__] = _module
