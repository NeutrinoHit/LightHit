# Build the optical cache before calculating events

The first preparation step is to build the cache. From the LightHit checkout on Linux or macOS,
with the accelerated dependencies installed, run:

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
workers, not the event calculation. Start with fewer workers on a machine
with limited RAM. Temporary Bessel files are written on the output volume,
under `.prepare/tmp`; allow disk space for both products and checkpoints.
The previously measured event times are not build-time estimates.

## Inspect, resume and verify

Inspect the exact commands before allocating resources:

```sh
PYTHONPATH=src python tools/tables/build_receiver_bundle.py \
  --model data/lighthit-bgvd-data/model/bgvd-model.npz \
  --output data/lighthit-glass-data --jobs 8 --plan
```

`--plan` reads the model and prints the dependency order without creating the
output or building tables. To continue after interruption, repeat the build
command. Completed stages are verified by content hashes and reused; child
builders retain their available row/frequency checkpoints. Worker count can
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
| Physical beta-time anchors | Beta 0.999 and 1, 25 radii; actual cones/clocks, aligned glass fronts and unchanged 1% interpolation target |
| Positive wavelength rule | Model-only training; rank-24 candidate, checked independently for charge and waveform before use |

The recipe records source/model hashes and numerical-library versions.
Failed validation stops preparation. It does not weaken a tolerance, fit an
event normalization or switch to a slower event algorithm to obtain a pass.
Existing atlas flags and exact fallbacks remain in force.

The orchestration is checked with interrupted/resumed subprocesses,
publication failures and relocated manifests. Short real Q1/proposal builds
exercise the standalone model path. The complete large recipe has not yet
been rerun from an empty directory as one command. The underlying component
builders and the separately assembled glass data were checked in the
[receiver study](receiver-surface.md); a newly built bundle still needs its
own addressed event accuracy and complete-time check before a new timing or
accuracy claim. Component validation is recorded separately from that check.
