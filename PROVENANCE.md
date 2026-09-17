# Provenance of the first delivery

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

`providers.py` mirrors only the interface `BaikalWater.py` already exposes
publicly: `wavelength`, `absorption_inv_length`, `scattering_inv_length`,
`group_refraction_index`. No private table and no `OpticalModule.py` are
included or executed by this package; the provider double used in tests
contains only synthetic numbers.

Demonstration medium: absorption length 25 m, scattering length 20 m,
$g=0.7$, $n_g=1.35$, 450 nm. This is an arbitrarily chosen test case, not a
reading of Baikal optics. `synthetic_medium()` in `medium.py` is a second,
separate set of parameters — absorption length 14.3 m, scattering length
45.5 m, $g=0.9$, $n_g=1.36$, 450 nm — chosen close to the *measured* Baikal
water table at 450 nm (`docs/chapters/04-bgvd-water.qmd`) without being
equal to it; neither set is a calibration.

## Documentation sources

`docs/chapters/01-rte.qmd` rewrites, in English and for this package's own
notation, the photon-balance derivation of the radiative transfer equation
from the author's `baikal-rte/rte-book` lecture notes
(`chapters/01-balance.qmd` there); it is a rewrite, not a reproduction, and
is scoped to what this package needs. `docs/chapters/03-histogram-binning.qmd`
draws on the detector-functional and window-kernel construction of
`baikal-rte/docs/01-spectral-detector.qmd` and the closed-form isotropic
decomposition of `baikal-rte/docs/02-isotropic-derivation.qmd`, again
rewritten for LightHit's own truncated-HG scheme rather than copied — the
$g=0$ closed form there is a special case this package's finite-$L$
construction reduces to, not the construction itself. All three source
documents are internal lecture material by the same author; they are not
published, and this book does not represent them as already public.

`docs/chapters/04-bgvd-water.qmd` and `docs/bgvd-450nm/` (a report, a
spectrum array, and three figures) were produced by running this package's
own solver, unmodified, against one wavelength of a private optical-water
package's measured table, through `load_bgvd_water`. The private package
itself was read only to extract that one table; it is not included in this
repository, is not a dependency, and its path is not recorded in any
committed file.

## Licensing

No license and no copyright holders are assigned in this delivery. Before
any public commit, the project owner needs to agree the applicable license
and the attribution of the carried-over mathematical and software
construction. A public empty repository existing already does not stand in
for that decision.

## Separation of sources and results

The source notebook is supplied without outputs. Numerical reports, the
executed notebook, and the HTML preview are separate artifacts. For a run
against a private medium, all outputs must stay under `.build/` or outside
Git. `.gitignore` does not protect the content of an already-tracked
notebook's outputs.

No commit, push, pull request, or other change to a remote repository was
made in preparing this delivery.
