"""Moved to ``lighthit.event_moments``; this name keeps older scripts working."""
import sys

from lighthit import event_moments as _module

sys.modules[__name__] = _module
