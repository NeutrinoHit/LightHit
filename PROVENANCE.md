# Provenance and attribution

LightHit starts a new history with its own package, `src/lighthit`. The
older `baikal-rte` checkout is not imported as a dependency.

Preparation used archives supplied by Dmitry — `src.tgz`, `tests.tgz`,
`scripts.tgz` — and a written mathematical derivation. Fourier conventions,
normalizations, the split by scattering order, and the water interface were
checked against them.

`angular.py` reuses the forward/Miller-recursion construction for free
moments from the earlier `baikal_rte/spectral.py`, including the free-tail
ratio calculation. The tridiagonal elimination and its specialization to a
point isotropic detector are a separate implementation, checked against an
independent dense $B$-matrix solver.

`providers.py` retains the original monochromatic water-only adapter.
`bgvd.py` adds the production runtime adapter requested by the project owner:
it reads `BaikalWater.py`, `OpticalModule.py`, and one geometry CSV from a
separately supplied checkout. It uses absorption, scattering, phase/group
indices, OM radius, the two documented spectral-response functions, the angular
polynomial and geometry. It explicitly does **not** use the private scattering
indicatrix: the transport operator remains HG with user-visible `g=0.9`.
No private source or table is included; tests use a synthetic interface double,
and local real-package runs are kept outside version control.

Demonstration medium: absorption length 25 m, scattering length 20 m,
$g=0.7$, $n_g=1.35$, 450 nm. This is an arbitrarily chosen test case, not a
reading of Baikal optics. `synthetic_medium()` in `medium.py` is a second,
separate set of parameters — absorption length 14.3 m, scattering length
45.5 m, $g=0.9$, $n_g=1.36$, 450 nm — chosen close to a measured Baikal-water
table at 450 nm without being equal to it; neither set is a calibration. The
private table itself is not distributed.

## Documentation sources

The chapters under `docs/chapters/` rewrite, in English and in LightHit's own
notation, mathematical material developed for the author's Baikal-GVD lecture
notes and the earlier `baikal-rte` research checkout. The photon-balance,
detector-functional and window-kernel derivations are rewrites rather than
reproductions and are scoped to the algorithms implemented here. Internal
lecture notes are not presented as already published sources.

`docs/bgvd-450nm/` was produced by running this package's solver against one
wavelength of a separately supplied private optical-water table through
`load_bgvd_water`. The private package is not included in this repository, is
not a dependency, and its path is not recorded in committed artifacts.

## Licensing

LightHit is distributed under the BSD 3-Clause License. Copyright and the full
license text are recorded in `LICENSE`; package metadata carries the matching
license declaration.

## Separation of sources and results

The source notebook is supplied without outputs. Numerical reports, the
executed notebook, and the HTML preview are separate artifacts. For a run
against a private medium, all outputs must stay under `.build/` or outside
Git. `.gitignore` does not protect the content of an already-tracked
notebook's outputs.
