"""Moved to ``lighthit.hdf5_minimal``; this name keeps older scripts working."""
import sys

from lighthit import hdf5_minimal as _module

sys.modules[__name__] = _module
