"""Moved to ``lighthit.muon.universal_primary``; this name keeps older scripts working."""
import sys

from lighthit.muon import universal_primary as _module

sys.modules[__name__] = _module
