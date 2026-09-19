# NumPy and Numba backends

The angular backend remains explicit and defaults to `numpy`.  The exact
single-scattering quadrature has a separate `auto` backend: it compiles the
scalar full-HG rate with Numba when the optional dependency is installed and
falls back to the same NumPy/SciPy path when it is not.  The adaptive SciPy
quadrature, its tolerances, and the 48-node isotropic ellipsoid rule are
unchanged.

Install from the repository root:

```bash
python -m pip install -e '.[accelerate,dev]'
python -m pytest -q
```

Use the optional backend:

```bash
python -m lighthit --config examples/point-green.toml \
  --angular-backend numba --single-backend auto \
  --output .build/point-green-numba
```

Or in a notebook:

```python
solver = PointGreenSolver(
    medium, settings, angular_backend="numba", single_backend="auto"
)
result = solver.solve(omega, displacement_m, direction=direction)
```

Selecting `numba` without the dependency raises an installation hint. The default
angular backend never imports Numba.  For strict implementation comparisons,
pass `single_backend="numpy"` or `"numba"`; `auto` records the implementation it
resolved to. Both selected backends are recorded in `report.json` and the NPZ
metadata. No API positional argument has changed.

## What changes internally

`_free_moments_and_ratios_numpy` preserves the original implementation.
`_angular_numba._nodewise` executes the recurrence separately for each k node.
It uses the same forward criterion `eta*(N+3) < 3` and the same Miller start rule
`N+1+max(32, ceil(28/eta))`, with the node's own eta rather than the smallest eta
of the entire backward batch. The branch selection, analytic F0, normalizations,
and k=0 limit are unchanged. The compiled loop uses complex128, no fastmath,
and no parallel Numba thread pool. The tridiagonal solve still gets normalized
ratios, including when the corresponding high-degree moments underflow.

## Reproduce timings and numerical comparison

```bash
python scripts/benchmark_backends.py \
  --config examples/point-green.toml --repeat 3 \
  --output .build/benchmark-backends
```

The script starts one fresh process per backend, sequentially. Numerical-library
threads are set to one before imports. Numba receives a fresh temporary compilation
cache. Outputs include each first-use solve (including any JIT compilation),
repeated warmed solves, separate charge-only solves, readout, peak process RSS,
versions, every numerical setting, and differences of spectra and time bins.

First-use and warm timings must not be mixed. The JIT cache contains machine code,
not transport results; every timed solve still recomputes transport. RSS measures
the whole process, including Numba's compiler/runtime. Do not add timing assertions
to unit tests. Compare equal numerical settings and report observed timings.

The report measures implementation equivalence, not convergence in L, J, k or
frequency. Those convergence studies remain separate.

The shipped `examples/point-green.toml` uses explicit parameters with g=0.7.
The function `synthetic_medium()` in the reviewed commit uses a different test
medium with g=0.9. This benchmark uses the config verbatim for BOTH backends;
it does not substitute the function's different defaults.

## Mathematical definitions

`chapters/05-numerical-scheme.qmd` derives the actual finite matrices and both
right-hand sides, spells out the first rows, explains the normalization of b and
the tail ratio, and maps each formula to the implementing function.

## Exact first scattering

`single.py` retains the public vectorized density and SciPy's adaptive
integrators.  At the thousands of scalar nodes requested by those integrators,
`single_fast.py` fuses the HG evaluation and the isotropic 48-node ellipsoidal
average into one compiled loop.  For a spectrum it also writes the complex
frequency phases in that same call.  This avoids constructing several tiny
NumPy arrays per adaptive node; it does not truncate the HG kernel or replace
the quadrature.  The isotropic density remains `+inf` exactly on the light
front, while the integrator assigns that single endpoint a finite neutral value
so the integrable logarithmic singularity cannot contaminate a spectrum with
`NaN`.
