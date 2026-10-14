"""Moved to ``lighthit.muon.first_charge_only``; this name keeps older scripts working."""
import sys

from lighthit.muon import first_charge_only as _module

sys.modules[__name__] = _module
