"""Moved to ``lighthit.muon.joint_muon``; this name keeps older scripts working."""
import sys

from lighthit.muon import joint_muon as _module

sys.modules[__name__] = _module
