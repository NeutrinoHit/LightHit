"""Comparisons of physics parameters that do not depend on the platform.

Values computed by quadrature and interpolation (band coefficients, spectral
tables) agree between platforms (CPU, maths library) only up to the last bits,
so they are compared within a relative tolerance instead of bit for bit.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

RTOL = 1e-9


def fingerprint(values, *, allow_nan=True):
    """SHA256 of ``values`` serialised as the tables do (sorted keys, no spaces)."""
    text = json.dumps(values, sort_keys=True, separators=(",", ":"), allow_nan=allow_nan)
    return hashlib.sha256(text.encode()).hexdigest()


def _leaves(value, prefix=""):
    if isinstance(value, dict):
        for key in sorted(value):
            yield from _leaves(value[key], f"{prefix}{key}.")
    else:
        yield prefix, value


def same_values(actual, expected, *, rtol=RTOL):
    """Same keys, equal strings and booleans, numbers within ``rtol`` of each leaf's scale."""
    a, e = dict(_leaves(actual)), dict(_leaves(expected))
    if a.keys() != e.keys():
        return False
    for key, x in a.items():
        y = e[key]
        if isinstance(x, (str, bool)) or isinstance(y, (str, bool)) or x is None or y is None:
            if x != y:
                return False
            continue
        x, y = np.asarray(x, float), np.asarray(y, float)
        if x.shape != y.shape:
            return False
        scale = float(np.max(np.abs(y), initial=0.0))
        if not np.allclose(x, y, rtol=rtol, atol=rtol * scale, equal_nan=True):
            return False
    return True
