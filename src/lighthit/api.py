"""Small, stateless convenience entry points for the public API."""

from .transport import TransportKernel


def build(medium, detector, source=None, *, config=None, method="auto",
          wavelengths_nm=None):
    """Create a reusable transport kernel and prebuild the tables it needs.

    Parameters are explicit objects rather than process-global configuration,
    so several media and detector arrays can safely coexist in one program.
    When ``source`` is supplied, its source type and wavelength support select
    exactly the cache tables that a later ``kernel.transport(source)`` reads.
    """
    kernel = TransportKernel(medium, detector, config)
    return kernel.build(wavelengths_nm, method=method, source=source)


__all__ = ["build"]
