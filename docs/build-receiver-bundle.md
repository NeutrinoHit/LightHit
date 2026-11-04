# Obtain the optical cache before calculating events

Before event calculations, obtain and verify a checked runtime bundle, or
build its tables once from the serialized model. Ordinary users of a supplied
bundle do not need to repeat the expensive preparation. To build tables from
the LightHit checkout on Linux or macOS, with accelerated dependencies, run:

```sh
python -m pip install -e '.[accelerate]'
PYTHONPATH=src python tools/tables/build_receiver_bundle.py \
  --model data/lighthit-bgvd-data/model/bgvd-model.npz \
  --output data/lighthit-glass-data \
  --jobs 8
```

The input is one serialized model containing the water, OM response and actual
spherical glass geometry. This build needs neither Geant4 events nor an
installed NTSim or `bgvd_model` provider. The example uses the model from the
existing data bundle; it leaves that bundle unchanged. Use a new output
folder. The recipe requires cubic OM acceptance and one common glass radius.

This is expensive, one-time preparation for a water/OM model, including the
separate primary-muon tables. `--jobs` controls independent preparation
workers, not the event calculation. Primary rows use independent radius
processes with one source-cell thread each; each process owns a separate
checkpoint and only the parent joins the final manifest. Start with fewer workers on a machine
with limited RAM. Temporary Bessel files are written on the output volume,
under `.prepare/tmp`; allow disk space for both products and checkpoints.
The previously measured event times are not build-time estimates.

## A supplied runtime bundle

Colleagues who receive a checked runtime bundle do not need to build tables
or install Geant4. Use its pinned LightHit version and run the included
`check.py` on the two fixed charged-event fixtures:

```sh
python lighthit-glass-data/check.py \
  --bundle lighthit-glass-data --report cache-check.json
```

The published 0.3.0a5 wheel omits `spectral_response_basis`, although the
muon route needs it. Use a software artifact containing the packaging fix,
pinned by its checksum, rather than assuming that any installation labelled
0.3.0a5 is sufficient. The working source includes the required module.

Legacy cubature files may contain only an exact optical fingerprint. Across
platforms, last-bit quadrature/interpolation differences can change that
fingerprint. The portable check accepts an explicitly stored full contract,
or the primary atlas's recorded contract, only when its original fingerprint
matches the rule and all physical values agree under the existing `1e-9`
comparison tolerance. It does not accept a different water or OM model.
Newly generated rules store their complete optical contract.

The fixtures carry input/reference hashes. Their numerical reproduction
check is separate from the original photon-MC significance and distance
criteria; failed weak-tail MC groups remain in the report. Array bytes and
primary-atlas manifests are preserved when exporting; only directory
references are relocated. To prepare such a delivery from existing checked
first-entry tables, inspect the actual dependency closure first:

```sh
PYTHONPATH=src python tools/tables/export_receiver_bundle.py \
  --source-root "$PWD" \
  --model data/lighthit-bgvd-data/model/bgvd-model.npz \
  --shower-data PATH_TO_GLASS/shower-data.json \
  --muon-data PATH_TO_GLASS/muon-data.json \
  --output data/lighthit-glass-runtime --plan
```

Remove `--plan` to copy and verify that closure. The exporter rewrites the
full/stationary pack and beta-time directory references, copies the exact
runtime arrays, and checks component loading before publishing readiness.
Source dependencies outside `--source-root` are rejected. Construction
blocks are excluded. The two-event fixtures and their provenance must be
supplied separately; they are validation inputs, not table training data.

The checked runtime closure of the recorded glass data is 25,980,752,808
bytes in 255 files before fixtures. This is a different product from the
older approximately 10 GB centre-reference bundle. Preparation needs
additional space for temporary kernels and active jobs; neither runtime
size is a peak-build disk estimate.

## Inspect, resume and verify

Inspect the exact commands before allocating resources:

```sh
PYTHONPATH=src python tools/tables/build_receiver_bundle.py \
  --model data/lighthit-bgvd-data/model/bgvd-model.npz \
  --output data/lighthit-glass-data --jobs 8 --plan
```

`--plan` reads the model and prints the dependency order without creating the
output or building tables. To continue after interruption, repeat the build
command. Completed stages are verified by content hashes and reused. Radial jobs
cover two or three radii independently of worker count; joined-array slices
are flushed and hashed before their temporary job files are discarded.
An interruption retains committed joined slices, including when the number
of workers changes. Frequency-bank blocks are discarded only after their
completed directional tables pass hash verification. Worker count can
change on resume. A changed model, recipe, code or numerical environment is
rejected rather than silently mixing tables: choose a new output directory.
Each stage prints its log path under `.prepare/logs`. A process lock prevents
two builders from writing the same output.

The finished directory has `bundle.json`, `model/bgvd-model.npz` and
`tables/shower-data.json`, `tables/muon-data.json`. The final route manifests
are published after all component loaders and required synthetic checks
succeed; a failed build has no completed `bundle.json`. The supplied model is
copied into the bundle. Runtime paths are relative, so the completed bundle
can be moved to a colleague's machine.

Verify files and hashes after building or transferring:

```sh
PYTHONPATH=src python tools/tables/build_receiver_bundle.py \
  --output data/lighthit-glass-data --check
```

A shorter installation check avoids reading every large array:

```sh
PYTHONPATH=src python tools/tables/build_receiver_bundle.py \
  --output data/lighthit-glass-data --check --quick
```

The quick check verifies file sizes and route manifests; it does not replace
the SHA256 check. Keep `.prepare` while a build is incomplete. It contains
construction checkpoints and logs, not runtime inputs. Some component builders
also retain row checkpoints beside their tables. A distribution of a
completed cache needs `model/`, `tables/` and `bundle.json` only; the preparation
workspace is not required by event calculations.

## Use the completed cache

To use glass-referenced Q1 and >=2 while retaining the existing bundle's
saved demonstration events:

```sh
export LIGHTHIT_DATA="$PWD/data/lighthit-bgvd-data"
export LIGHTHIT_SHOWER_DATA="$PWD/data/lighthit-glass-data/tables/shower-data.json"
export LIGHTHIT_MUON_DATA="$PWD/data/lighthit-glass-data/tables/muon-data.json"
```

`ShowerTransport`, `MuonTransport` and `EventTransport` then find these route
manifests. Explicit `data=`, `shower_data=` and `muon_data=` arguments remain
available. For calculations using your own charged steps, `LIGHTHIT_DATA`
may instead name `data/lighthit-glass-data`; the builder deliberately does
not generate or copy demonstration events. The Python environment still
needs the matching LightHit code.

Without the new route manifests, the existing 32-band bundle continues to
use its centre-reference scattered kernels and first-entry direct light.
Setting `scattered_receiver="first-entry"` alone cannot convert old tables;
receiver conventions and fingerprints must agree.

## What the recipe builds and checks

| Products | Recipe |
|---|---|
| Q1 reference and adaptive CDFs | Glass-radius-dependent near grid, logarithmic far grid to 1000 m; 33 hit / 97 miss nodes, 257 reference time nodes, 65 adaptive nodes |
| Shared geometric profiles | 513 path-grid nodes, 256 integration nodes; extended primary geometry covers the existing source domain |
| Physical spectral Q1 | All 32 bands, 65 coarse CDF nodes and geometric refinement/error data |
| Event RTE bank and packs | Source degree 32, scattering degree 96, cubic response, 65 frequencies; exact stationary charge slice and existing three-node fold |
| Point-cone preview reference | Separate stationary degree-128 bank; event degree remains 32 |
| Selection proposal and glass preview | Fixed model-only sampling and existing preview axes; positive first-entry moments with all-orientation checks |
| Primary Q1 / >=2 and beta charge modes | Existing track parameterizations and synthetic validation domains |
| Physical beta-time anchors | Beta 0.999 and 1, 30 radii and 192 longitudinal cells; actual cones/clocks, aligned glass fronts and unchanged 1% interpolation target |
| Positive wavelength rule | Model-only training; rank-24 candidate, checked independently for charge and waveform before use |

The recipe records source/model hashes and numerical-library versions.
Failed validation stops preparation. It does not weaken a tolerance, fit an
event normalization or switch to a slower event algorithm to obtain a pass.
Existing atlas flags and exact fallbacks remain in force.

The orchestration is checked with interrupted/resumed subprocesses,
publication failures and relocated manifests. A complete numerical build
from an empty directory on Linux, with 48 preparation workers and a cold
compiler cache, finished all 22 stages in about 5 h 8 min. No event input,
manual compiler warm-up or table editing was used in that calculation.
The completed SHA256 check passed, and repeating the same command left
all runtime file modification times unchanged.

The primary Q1 grid includes 4 m to resolve its near-radius curvature.
Its full beta-charge validation has a maximum scaled error of 0.712%
against the unchanged 1% target. The >=2 correction retains two modes
in delta = `1/beta - 1`, interpolating exact cone harmonics at zero,
half the supported delta range, and its endpoint. Its maximum cone error
is 0.00947% against 0.2%; the total atlas error is 0.563% against 1%.
Independent beta values between the anchors are included. Legacy Taylor
payloads remain readable; row resume checks the actual cone-mode hash.

Physical Q1 time rows use 192 longitudinal cells and 30 radii, including
13.5, 27.5, 52.5, 55 and 57.5 m. Sixteen radius checks, including points
between those rows, give a maximum scaled charge/bin error of 0.726%
against 1%. The original physical beta anchors, source clocks and
alignment to the glass front are retained. The primary charge grid
remains at 96 cells, so its rows cannot replace the finer time rows.

Time rows use `--row-storage stored`: standard ZIP containers with
uncompressed NPY members. This avoids repeatedly inflating larger time
rows during event calculations. The NPY member bytes are identical to
the compressed representation; no physical value, grid, tolerance or
query formula changes. The numerical recipe above was checked first with
compressed rows; lossless storage conversion was then checked against
all row members and the full muon response. The current builder writes
stored time rows directly. The standalone row builder retains compressed
storage by default and rejects resume/reuse with a different storage layout.

On the same Linux VM and installed software, complete warm calls on the
fixed charged-step fixtures measured:

| Event | Earlier checked data | Refined runtime data |
|---|---:|---:|
| Electron, 100 GeV | 38.62 s | 38.34 s |
| Muon, 100 GeV | 115.76 s | 116.73 s |

These calls include all three orders and source compilation. Table opening
precedes the timer; the first event call warms the route. With stored time
rows, the muon charge and bins are bitwise identical to the refined
compressed data. Relative to the earlier data, its Q1 charge changes by
0.058% and bin L1 by 0.063% of the earlier Q1 charge. Electron differences
are at numerical precision. The full event signals agree with the original
photon-MC criterion. The earlier electron >=2 component flag (about 2.2
standard deviations), the muon near-distance Q1 flag on one OM, and weak
distance flags remain visible; these checks do not certify every individual
OM or arbitrary time binning.

The recorded runtime delivery occupies about 27.03 GB, including fixed
validation inputs. Stored time rows account for 636 MB more than their
compressed equivalents. The observed construction workspace reached
46.81 GB. These disk sizes differ from RAM requirements and from archive
sizes. The reference build used a 64-CPU, 471-GiB machine; 48 concurrent
solvers need substantially more RAM than event transport. Start with fewer
preparation workers on smaller machines. Resource samples are observations,
not a portable upper bound. Ordinary use of a supplied bundle requires no
rebuilding of these tables.

For reproducible one-thread numerical tests from the checkout:

```sh
NUMBA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m pytest -q tests src/lighthit/tests
```

Keep component checks and complete-event benchmarks separate when changing
a model or numerical environment. See the [receiver study](receiver-surface.md)
for the scientific scope and remaining limitations.
