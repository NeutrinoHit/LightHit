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
| 10 | `ShowerTransport(kernel).tables()["folded"]` saved by `lighthit.band_fold.save_folded(folded, OUT)`; the manifest gains `"ge2_band_fold": "ge2-band-fold"` | `tables/ge2-band-fold` (the shower route's default fold: rank 3, no subdivision, 65 frequencies) | steps 1, 2, 9 |

The model file `model/bgvd-model.npz` is NTSim's `bgvd_model_from_ntsim`
(clusters 1-5) with the OM response tables attached (`BGVDModel.save`). Every
table depends on the water: after a change of `bgvd_model` the whole bundle is
rebuilt (the bundle of 2026-10-19 follows the water grid 350-600 nm in 10 nm
steps; the earlier bundles used 350-610 nm in 10.4 nm steps).

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
  The bundle of 2026-10-19 built all 19 rows in one store with `--check-rows`
  except 300 m: its third check interval reaches no time bin of the 806 ns
  window, so the relative bin error is 0/0 and the row was finished without
  the check. The held-out radii of `--validate-midpoints 13.5 19.75 27.5 32.5
  38.75 55 77.5 92.5` are in `midpoint-checks.json`. Build CPU per row: 10-35 s.
* **ge2-atlas**: `manifest.json` and `report.json` hold the optics (cache keys,
  source degree 32, acceptance degree 3, radius window 3-300 m), order 10 per
  row and the validation settings (tolerance 0.01, bright charge 1e-5 pe,
  backward tolerance 1e-6 pe); 44 rows (42 with the nine-node tables of
  LightHit <= 0.3.0a4). `--passes`, `--max-nodes` and the initial `--nodes`
  are not recorded; the 32-node bundle was built with the script defaults (3,
  65 and the 13 nodes 3.01 ... 299 m) and `NUMBA_NUM_THREADS=1`.
* **first-beta-charge.npz**: anchor beta 0.999, `base_manifest_sha256` of the
  q1-atlas, `cubature_sha256` 441c53f7... of `basis-rank16.npz` (49cefe0b... before the water grid fix), 17 spectral
  nodes. A rebuilt cubature can be checked against that hash before step 6.
* **ge2-beta-charge.npz**: `beta_min` 0.99, Taylor order 2,
  `base_manifest_sha256` of the ge2-atlas.
* **preview-atlas.npz**: 6,000,000 samples per order, seed 81, HG g 0.9; the
  bins follow from its edges: `--count 6000000 --seed 81 --rho-bins 96
  --z-bins 384 --extent-m 500` (the script defaults differ). It also stores
  ten arrival-direction moments for the cubic OM acceptance; the scalar
  density and sampling spread retain the same paths and random stream.
  To extend an existing scalar atlas without changing its optical tables or
  axes, use `--template OLD.npz --count 6000000 --seed 81 --output NEW.npz`.
  The old file is never overwritten. Primary track atlases need no rebuild.

## Checks

`experiments/run.py` rebuilds two of them and compares with the bundle:
`ge2-pack-tables/charge-slice-rebuild` (step 2, sha256 of `charge-0.npy`) and
`positive-fluence-preview/atlas-rebuild` (step 8, every array). The results
are in `experiments/REPORT.md`.
