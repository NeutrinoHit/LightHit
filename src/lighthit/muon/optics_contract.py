"""Platform-independent check of the optical contract of the muon tables.

A table stores the SHA256 fingerprint of its optical contract (spectral tables,
acceptance, area, medium samples) together with the contract values. The
values are computed from the medium by quadrature and interpolation, so a
kernel on another platform (CPU, maths library) reproduces them only to the
last bits and its fingerprint differs. ``contract_matches`` accepts a kernel
when its fingerprint equals the stored one or, failing that, when the values
stored in the table hash to the stored fingerprint (they are the contract the
table was built with) and the kernel's values agree with them within ``rtol``.
"""
from __future__ import annotations

from .._portable import RTOL, fingerprint, same_values


def contract_matches(actual, stored, stored_fingerprint, *, allow_nan=True, rtol=RTOL):
    """True when the kernel contract ``actual`` is the table's contract.

    ``stored`` is the mapping that holds the table's contract values (any extra
    keys are ignored); ``stored_fingerprint`` the fingerprint recorded with them.
    """
    if fingerprint(actual, allow_nan=allow_nan) == stored_fingerprint:
        return True
    if stored is None or not all(key in stored for key in actual):
        return False
    recorded = {key: stored[key] for key in actual}
    if fingerprint(recorded, allow_nan=allow_nan) != stored_fingerprint:
        return False
    return same_values(actual, recorded, rtol=rtol)
