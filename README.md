# LightHit

Time-dependent light transport and detector response.

First delivery: the Green's function of an instantaneous, monochromatic
point flash and a point isotropic receiver. A **directed single photon**
and an **isotropic single photon** are both supported. The medium is
homogeneous and unbounded; the phase function is Henyey–Greenstein (HG).

## What the calculation returns

`PointGreenSolver.solve` computes the complex response spectrum per unit
effective area, m⁻². The zero-frequency component gives the integrated
signal ("charge"). `result.readout` gives bin integrals, m⁻²; dividing by
the bin width gives the mean registration rate, m⁻²·ns⁻¹.

Detection efficiency is 1, and sensitivity is the same in every direction.
Area is never silently replaced by a bare number: multiplying by a small
effective area gives the expected photon count. There is no OM surface, no
first-entry condition, and no shadowing in this model.

A directed delta flash observed exactly on its own forward ray gives a
singular response; that request is rejected. The API treats
$\cos\theta\ge1-10^{-12}$ as singular. Geometries that close would need a
finite aperture or a finite angular source distribution instead. In the
shipped demonstration the angle is 60°, so the direct light is zero and the
whole signal is scattered. For an isotropic flash the direct light is a
delta function in time with a finite integral; bin integrals across it are
exact.

## Install into a clean checkout

Python 3.11 or newer is required. Commands below run from
`~/Projects/LightHit`:

```bash
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[notebook,dev]'
python -m pytest -q
```

None of these commands creates a Git commit or publishes anything.

## A first calculation without Jupyter

```bash
python -m lighthit --config examples/point-green.toml \
  --output .build/point-green --plots
```

Output:

- `.build/point-green/spectrum.npz`: frequencies and the 0, 1, ≥2 components;
- `.build/point-green/profiles.npz`: unsmeared bins and bins with the given readout;
- `.build/point-green/report.json`: parameters, charge, timings, diagnostics;
- `.build/point-green/figures/`: separate profile and spectrum plots.

Charge without building a time spectrum:

```bash
python -m lighthit --charge-only --repeat 3 --output .build/charge
```

A quick pass and a stricter numerical setting:

```bash
python -m lighthit --preset quick --output .build/quick --plots
python -m lighthit --preset refined --output .build/refined --plots
python scripts/compare_runs.py .build/point-green .build/refined
```

Numerical library threads can be pinned for comparable timings:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python -m lighthit --repeat 3 --output .build/benchmark
```

`--repeat` repeats the calculation itself, not loading a saved result. All
timings are kept in the report. This first release has no on-disk transport
cache. Within one `solve` call, the angular solution is shared across every
observation point.

## Notebook

```bash
python -m jupyter lab notebooks/01_point_green.ipynb
```

Jupyter menu: **Run → Run All Cells**. The notebook calls the same package
as the CLI. It shows normalization, geometry, the phase function, the
spectral equations, charge, time profiles, error diagnostics, and timings.

Non-interactive execution and HTML:

```bash
mkdir -p .build/notebook
python -m jupyter nbconvert --to notebook --execute \
  notebooks/01_point_green.ipynb --output 01_point_green.executed \
  --output-dir .build/notebook --ExecutePreprocessor.timeout=600
python -m jupyter nbconvert --to html \
  .build/notebook/01_point_green.executed.ipynb
```

The source notebook is kept in Git without outputs. After running it,
especially against private water, strip outputs before `git add`:

```bash
python scripts/strip_notebook_outputs.py notebooks/01_point_green.ipynb
```

`.gitignore` does not remove outputs already committed to a tracked
notebook.

## Python API

```python
import numpy as np
from lighthit import PointGreenSolver, SolverSettings, synthetic_medium

medium = synthetic_medium()  # test values close to Baikal water, not a calibration
solver = PointGreenSolver(medium, SolverSettings())
omega = np.linspace(0.0, 1.2, 241)  # rad/ns; include 0 for the charge
result = solver.solve(
    omega,
    displacement_m=[17.32050807568877, 0.0, 10.0],  # detector minus source, m
    direction=[0.0, 0.0, 1.0],
)
print(result.charge_per_m2)
print(result.timings_s)
front = result.front_time_ns[0]
edges = np.arange(front - 30, front + 651, 2.0)
raw = result.readout(edges, sigma_ns=0.0)
measured = result.readout(edges, sigma_ns=3.0)
```

For an isotropic flash, pass `direction=None`. For several detectors, pass
`displacement_m` of shape `(D,3)`. Emission time is set by
`emission_time_ns`; the number of emitted photons by `photons`. A point
isotropic detector needs no normal vector.

## Private water

The distribution contains **no** private optical tables, OM coefficients,
event files, private output spectra, and no dependency on the older
`baikal-rte` checkout. The adapter reads the interface known from that
earlier project, `BaikalWater.py`; it does not import `OpticalModule.py`,
so efficiency here stays 1.

```python
from lighthit.providers import load_bgvd_water

medium = load_bgvd_water(
    "/full/local/path/to/bgvd-model",
    wavelength_nm=450.0,
    g=0.9,  # an explicit HG choice, not read from the water coefficients
)
```

The path may point at a checkout or at its `bgvd_model` directory. If the
private package is installed, the path argument can be omitted.

For the CLI, copy a configuration to `.local-inputs/private-green.toml` and
replace only the medium section:

```toml
[medium]
kind = "bgvd_water"
path = "/full/local/path/to/bgvd-model"
wavelength_nm = 450.0
g = 0.9
```

Run it with `--config .local-inputs/private-green.toml`. Keep all results
under `.build/`. `report.json` and the NPZ files record the coefficients
actually used; those output files for a private medium must not be
published. The adapter has been checked against a synthetic double of the
interface; it has not been run against the real private package in this
delivery.

A run at $g=0.9$ needs its own scan of settings: numbers validated for the
synthetic $g=0.7$ examples are not a validation of $g=0.9$ real water, and
the (scattering degree, spatial degree, $k_{\max}$) triple that converges at
one radius does not automatically converge at another.
`docs/chapters/04-bgvd-water.qmd` works through this in detail: the
constraint that actually governs convergence is the spatial multipole
degree tracking $k_{\max}\cdot r$, not $g$ as such.

## What exactly is solved, and where the formulas are

The math is written up as a short Quarto book in `docs/`:

| Chapter | Content |
|---|---|
| `docs/chapters/01-rte.qmd` | The radiative transfer equation, from a photon balance |
| `docs/chapters/02-point-source.qmd` | Point source, point detector: the adjoint system, the exact free tail, spatial inversion |
| `docs/chapters/03-histogram-binning.qmd` | Why orders 0 and 1 are binned in physical time instead of Fourier-inverted |
| `docs/chapters/04-bgvd-water.qmd` | The solver run against measured Baikal water at 450 nm |
| `docs/appendices/notation.qmd` | Symbol table |
| `docs/VALIDATION.md` | Numbers from the checks that were actually run |

Render it with Quarto (`cd docs && quarto preview`, or `quarto render` for
a static copy in `docs/_book/`); nothing in the book needs private data or
Geant4.

| File | Contents |
|---|---|
| `src/lighthit/medium.py` | Parameters of one spectral node, m and ns |
| `src/lighthit/angular.py` | Tridiagonal adjoint problem, exact free tail |
| `src/lighthit/single.py` | Coordinate-space first order with the full HG function |
| `src/lighthit/green.py` | Radial inversion and the combined 0+1+≥2 spectrum |
| `src/lighthit/readout.py` | Bins and instrument smearing, independent of transport |
| `src/lighthit/providers.py` | Local private-water provider |
| `tests/` | Independent matrix, analytic, and geometric checks |

Every collision order is included. **Orders $\ge2$ use HG coefficients only
up to $L$**, while the exact first order uses the full HG function — an
explicit, stated composite approximation of the full HG response, whose
difference from the full HG answer vanishes as $L\to\infty$. Free angular
transport is not truncated at $L$: its infinite tail is removed
analytically. The spatial angular inversion has its own, independent degree
$J$.

The $k$ range and quadrature, the frequency band, and the frequency step
are all finite as well. `quick`, `balanced`, `refined` change several
spatial/angular settings at once; they do not fix a mathematical error
bound. Unsmeared bins can oscillate right at the light front. The code
keeps negative values and mass before the front; nothing is clipped or
renormalized. Orders 0 and 1 are integrated directly in time, so their
fronts do not depend on the chosen frequency cutoff; the inverse Fourier
transform is applied only to the $\ge2$ part (`docs/chapters/03-histogram-binning.qmd`
explains why).

## What is not in the first module

A finite OM sphere, real acceptance and a PDE for it, cones and track
segments, showers, spectral convolution, a general spatial cache,
SVD/NUFFT, GPU execution, boundary surfaces, and an inhomogeneous medium.
Measurements on the test medium do not confirm accuracy on private optics.

Sources for the mathematical construction and the license status are in
[PROVENANCE.md](PROVENANCE.md). Copyright holders and a license must be
agreed before any publication; this delivery assigns neither.
