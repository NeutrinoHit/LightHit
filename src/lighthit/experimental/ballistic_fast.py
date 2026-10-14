"""Moved to ``lighthit.ballistic_fast``; this name keeps older scripts working."""
import sys

from lighthit import ballistic_fast as _module

sys.modules[__name__] = _module
