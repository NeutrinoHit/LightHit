"""Moved to ``lighthit.muon.adaptive_screen``; this name keeps older scripts working."""
import sys

from lighthit.muon import adaptive_screen as _module

sys.modules[__name__] = _module
