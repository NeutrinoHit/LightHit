# LightHit — current state

LightHit provides a spectral, time-dependent RTE response for isotropic
flashes, Cherenkov tracks and stored G4 showers. The production API combines a
homogeneous spectral medium, wavelength-dependent OM efficiency, directional
acceptance and detector geometry into integrated and time-binned expected
photoelectrons.

The Green-function solver separates ballistic, once-scattered and
multiply-scattered light. Free angular transport has an exact infinite tail;
the HG scattering operator, spatial inversion, frequency range, wavelength
quadrature and source representation retain explicit numerical truncations.
Directional sources read by directional modules are handled by fixed-$m$
blocks and a fused Numba contraction.

The private BGVD model and G4 event files are runtime inputs and are excluded
from distributions. The public package uses HG with explicit `g`; it does not
copy the private scattering indicatrix. Exact provenance and model boundaries
are recorded in `PROVENANCE.md` and the Quarto book under `docs/`.

The package version is defined in `pyproject.toml` and `lighthit.__version__`.
Release checks are documented in `PUBLISHING.md`. LightHit is licensed under
BSD-3-Clause; see `LICENSE`.
