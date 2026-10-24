"""Independent transport checks and small reproducible teaching experiments.

Event routes do not import this package. Optional Monte Carlo acceleration is
loaded only when explicitly requested.
"""
from .cells import AxisymmetricCell

__all__ = ["AxisymmetricCell"]
