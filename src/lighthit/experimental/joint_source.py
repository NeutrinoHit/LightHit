"""Moved to ``lighthit.muon.joint_source``; this name keeps older scripts working."""
import sys

from lighthit.muon import joint_source as _module

sys.modules[__name__] = _module
