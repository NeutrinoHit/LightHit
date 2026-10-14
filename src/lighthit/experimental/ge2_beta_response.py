"""Moved to ``lighthit.muon.ge2_beta_response``; this name keeps older scripts working."""
import sys

from lighthit.muon import ge2_beta_response as _module

sys.modules[__name__] = _module
