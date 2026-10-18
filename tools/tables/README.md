# Table builders

The prepared tables of the data bundle (`$LIGHTHIT_DATA/tables`) are built once
per water and OM model. The RTE tables and the order >=2 pack come from the
library (`TransportKernel.build`); the muon tables come from the scripts in this
folder. Every script takes its inputs and output as arguments; none reads a
fixed local path. Run them from the repository root with `PYTHONPATH=src`; the
scripts that read the BGVD water and OM tables directly need the `bgvd_model`
package installed (`pip install -e $LIGHTHIT_DATA/bgvd-model`).

## Build order

| Step | Command | Produces (bundle file) | Needs |
|---|---|---|---|
| 1 | `kernel.build(source=...)` and `kernel.ge2_kernel_store().save(...)` (see `docs/transport-kernel.md`) with the 65-frequency `KernelConfig` of the routes (default 32 wavelength nodes) | `tables/rte-cache/directional-*.npz`, the full order >=2 tensor of `tables/ge2-pack` | the BGVD model |
| 2 | `build_ge2_charge_pack.py --source PACK --output OUT` | `tables/ge2-pack/charge-0.npy` (exact zero-frequency slice) and its `pack.json` entry | step 1 |
| 3 | `build_track_first_atlas.py --output OUT ...` | `tables/muon/q1-atlas` (order 1 of beta = 1 track pieces) | `bgvd_model` |
| 4 | `build_track_ge2_atlas.py --pack PACK --cache RTE --output OUT` | `tables/muon/ge2-atlas` (order >=2 of beta = 1 track pieces) | steps 1, `bgvd_model` |
| 5 | `build_response_cubature.py --output OUT` | `basis-rank16.npz` (not in the bundle; input of step 6) | `bgvd_model` |
| 6 | `build_first_beta_charge.py --base Q1_ATLAS --cubature OUT/basis-rank16.npz --output OUT` | `tables/muon/first-beta-charge.npz` | steps 3, 5 |
| 7 | `build_ge2_beta_charge.py --base GE2_ATLAS --pack PACK --cache RTE --output OUT` | `OUT/beta-charge.npz`, copied to `tables/muon/ge2-beta-charge.npz` (OUT is a directory with rows and `report.json`) | steps 1, 4 |
| 8 | `build_positive_fluence.py --output FILE ...` | `tables/muon/preview-atlas.npz` | the BGVD model (bundle `model/bgvd-model.npz`) |
| 9 | `lighthit.muon.write_manifest(...)` | `tables/muon-data.json` | all of the above |

The laser table (`tables/laser-cache`) is built by
`lighthit.event.laser_kernel(..., cache_policy="build")` followed by
`kernel.build(source=flash)` (`docs/event-and-laser.md`).

## The arguments of the bundle tables

What each bundle file records about its own build (read from its metadata):

* **ge2-pack**: `pack.json` holds the keys (the directional table key of each of the 32 wavelength nodes,
  source degree 32, 65 frequencies). The bundle names the full tensor
  `full-0.npy` inside the pack, while step 2 writes a relative path to its
  source pack: the bundle was repacked after step 2. Only `--source` is a
  parameter.
* **q1-atlas**: `manifest.json` holds the whole contract: 96 cells, 0.1 ns
  near-front step, 900 ns at the smallest radius, reference radius 30 m and the
  19 radii 3 5 8 12 15 18 21.5 25 30 35 42.5 50 60 70 85 100 150 220 300 m.
  Thirteen rows record `reused_from` an earlier 13-radius store; 15, 21.5, 30,
  42.5, 60 and 85 m were built later (`--reuse-rows-from`). The held-out radii
  of `--validate-midpoints` are in `midpoint-checks.json`. Recorded build CPU
  per row: 10-48 s.
* **ge2-atlas**: `manifest.json` and `report.json` hold the optics (cache keys,
  source degree 32, acceptance degree 3, radius window 3-300 m), order 10 per
  row and the validation settings (tolerance 0.01, bright charge 1e-5 pe,
  backward tolerance 1e-6 pe); 44 rows (42 with the nine-node tables of
  LightHit <= 0.3.0a4). `--passes`, `--max-nodes` and the initial `--nodes`
  are not recorded; the 32-node bundle was built with the script defaults (3,
  65 and the 13 nodes 3.01 ... 299 m) and `NUMBA_NUM_THREADS=1`.
* **first-beta-charge.npz**: anchor beta 0.999, `base_manifest_sha256` of the
  q1-atlas, `cubature_sha256` 49cefe0b... of `basis-rank16.npz`, 17 spectral
  nodes. A rebuilt cubature can be checked against that hash before step 6.
* **ge2-beta-charge.npz**: `beta_min` 0.99, Taylor order 2,
  `base_manifest_sha256` of the ge2-atlas.
* **preview-atlas.npz**: 6,000,000 samples per order, seed 81, HG g 0.9; the
  bins follow from its edges: `--count 6000000 --seed 81 --rho-bins 96
  --z-bins 384 --extent-m 500` (the script defaults differ).

## Checks

`experiments/run.py` rebuilds two of them and compares with the bundle:
`ge2-pack-tables/charge-slice-rebuild` (step 2, sha256 of `charge-0.npy`) and
`positive-fluence-preview/atlas-rebuild` (step 8, every array). The results
are in `experiments/REPORT.md`.
