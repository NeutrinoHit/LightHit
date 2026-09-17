# LightHit — state of the first part

Implemented: an independent package, a point isotropic detector, directed
and isotropic monochromatic flashes, every scattering order, charge, time
bins, a CLI, a synthetic example, a notebook, and the private-water adapter
code. Normalization: response per unit effective area, efficiency 1.

Numerical scheme: exact orders 0 and full-HG 1; orders $\ge2$ use finite-HG
scattering with an exact free angular tail. The matrix has size $L+1$;
output angular degrees up to $J$ are continued by recursion. $k$ and
$\omega$ are integrated numerically.

Math: `docs/` (a short Quarto book — `chapters/01-rte.qmd` through
`chapters/04-bgvd-water.qmd`, `appendices/notation.qmd`).
Exact scope of the checks performed: `docs/VALIDATION.md`.
`synthetic_medium()` in `src/lighthit/medium.py` now uses parameters close
to, but not equal to, measured Baikal water at 450 nm (absorption exceeding
scattering, a strongly forward-peaked $g=0.9$); it is still not a
calibration (`PROVENANCE.md`). `docs/chapters/04-bgvd-water.qmd` and
`docs/bgvd-450nm/` record one run against the actual private water table at
450 nm, including the $(L,k_{\max})$ convergence study that run needed.

No commit and no publication had happened before this pass; a license and
copyright holders are for the author to decide.

The next small, checkable step is discussed after a local run of this
example and a look at its time front. The earlier exploratory `baikal-rte`
project remains separate and is not imported automatically.
