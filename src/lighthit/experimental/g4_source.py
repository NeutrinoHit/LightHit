"""Moved to ``lighthit.g4_source``; this name keeps older scripts working."""
import sys

from lighthit import g4_source as _module

sys.modules[__name__] = _module
