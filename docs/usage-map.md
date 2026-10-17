# Usage map: what the default routes execute

Measured, not read from the code: each scenario below ran in its own process on the
standard inputs (`experiments/harness/inputs.py`, data bundle `$LIGHTHIT_DATA` and the
regenerated NTSim events) under `experiments/usage/usage_trace.py`, which records every
Python function of `lighthit` that starts and every Numba function that holds a compiled
signature at the end of the run. Regenerate with `experiments/usage/scenarios.py`,
`experiments/usage/run_ntsim.sh` and `experiments/usage/build_map.py`; the full table
(every function with its scenarios) is `experiments/usage/usage-map.json`.

A function listed as not executed was not called in any of these runs; it may still be
reachable with other inputs or options (for example a near-field error branch).

## Scenarios

| Scenario | What ran | Wall s | Peak RSS MiB | Route (metadata) |
|---|---|---:|---:|---|
| shower | ShowerTransport, bundle Geant4 100 GeV electron (event 5) | 267.7 | 5363 | lighthit.shower |
| muon | MuonTransport, bundle 1 TeV muon with its secondaries | 138.4 | 2433 | lighthit.muon |
| event | EventTransport, 1 TeV muon + 100 GeV electron shower | 88.1 | 7713 | lighthit.event |
| event-stopping-muon | EventTransport, regenerated 6 GeV muon aimed 1.5 m past an OM, stopping 15 m before it (route fallback to the shower route) | 5.7 | 3035 | lighthit.shower; fallbacks: 1 |
| event-muon-near-om | EventTransport, regenerated 6 GeV muon passing 1.76 m from an OM (fallback entered, then the shower route raises) | 5.1 | 2780 | ; raised ValueError: 1 OMs have source points closer than the validated radial minimum 3 m |
| laser | EventTransport with laser_kernel, 532 nm flash (bundle laser table) | 0.8 | 571 | lighthit.transport (isotropic flash) |
| prompt | TransportKernel.transport_prompt, 100 GeV electron (orders 0 and 1) | 21.4 | 2873 |  |
| hybrid | TransportKernel.transport_hybrid, 100 GeV electron (per-band order >=2) | 174.7 | 5592 |  |
| ntsim-electron | NTSim run_lighthit.py, e- 100 GeV | 11.0 | 5722 |  |
| ntsim-muon | NTSim run_lighthit.py, mu- 1 TeV | 48.3 | 2711 |  |
| ntsim-nue-cc | NTSim run_lighthit.py, e- 70 + pi+ 25 + p 5 GeV | 11.5 | 5604 |  |
| ntsim-numu-cc | NTSim run_lighthit.py, mu- 800 + pi+ 150 + p 50 GeV | 74.2 | 9974 |  |
| ntsim-laser | NTSim run_lighthit.py --laser, 532 nm flash | 4.4 | 943 |  |

Wall time includes imports, table loading and any Numba compilation; it is not a
benchmark (see `experiments/REPORT.md`).

## Modules

Functions executed / defined, and the scenarios that imported the module.

| Module | Executed | Imported by |
|---|---:|---|
| `lighthit` | 1 / 1 | all |
| `lighthit.__main__` | 0 / 0 | not imported |
| `lighthit._angular_numba` | 0 / 2 | not imported |
| `lighthit._ballistic_interval_numba` | 9 / 10 | shower, muon, event, event-stopping-muon, event-muon-near-om, ntsim-electron, ntsim-muon, ntsim-nue-cc, ntsim-numu-cc |
| `lighthit._directional_numba` | 8 / 9 | shower, muon, event, event-stopping-muon, event-muon-near-om, hybrid, ntsim-electron, ntsim-muon, ntsim-nue-cc, ntsim-numu-cc |
| `lighthit._prompt_numba` | 47 / 60 | shower, muon, event, event-stopping-muon, prompt, hybrid, ntsim-electron, ntsim-muon, ntsim-nue-cc, ntsim-numu-cc |
| `lighthit._shower_basis` | 8 / 13 | shower, muon, event, event-stopping-muon, event-muon-near-om, ntsim-electron, ntsim-muon, ntsim-nue-cc, ntsim-numu-cc |
| `lighthit._spectral_fold` | 0 / 10 | not imported |
| `lighthit._track_transport` | 0 / 6 | not imported |
| `lighthit._version` | 0 / 0 | all |
| `lighthit.angular` | 1 / 15 | all |
| `lighthit.api` | 0 / 1 | all |
| `lighthit.axial_fast` | 5 / 9 | shower, muon, event, event-stopping-muon, event-muon-near-om, hybrid, ntsim-electron, ntsim-muon, ntsim-nue-cc, ntsim-numu-cc |
| `lighthit.axial_source` | 4 / 9 | shower, muon, event, event-stopping-muon, event-muon-near-om, hybrid, ntsim-electron, ntsim-muon, ntsim-nue-cc, ntsim-numu-cc |
| `lighthit.ballistic` | 0 / 2 | all |
| `lighthit.ballistic_fast` | 3 / 7 | shower, event, event-stopping-muon, prompt, hybrid, ntsim-electron, ntsim-nue-cc, ntsim-numu-cc |
| `lighthit.ballistic_interval` | 4 / 4 | shower, muon, event, event-stopping-muon, event-muon-near-om, ntsim-electron, ntsim-muon, ntsim-nue-cc, ntsim-numu-cc |
| `lighthit.band_fold` | 7 / 9 | shower, event, event-stopping-muon, event-muon-near-om, ntsim-electron, ntsim-nue-cc, ntsim-numu-cc |
| `lighthit.bgvd` | 8 / 16 | all |
| `lighthit.cache` | 12 / 44 | all |
| `lighthit.cli` | 0 / 4 | not imported |
| `lighthit.directional` | 7 / 27 | all |
| `lighthit.event` | 12 / 12 | event, event-stopping-muon, event-muon-near-om, laser, ntsim-electron, ntsim-muon, ntsim-nue-cc, ntsim-numu-cc, ntsim-laser |
| `lighthit.event_moments` | 0 / 16 | shower, muon, event, event-stopping-muon, event-muon-near-om, hybrid, ntsim-electron, ntsim-muon, ntsim-nue-cc, ntsim-numu-cc |
| `lighthit.examples` | 0 / 0 | not imported |
| `lighthit.examples.synthetic` | 0 / 14 | not imported |
| `lighthit.experimental` | 0 / 0 | not imported |
| `lighthit.experimental.adaptive_screen` | 0 / 0 | not imported |
| `lighthit.experimental.axial_blocked` | 0 / 2 | not imported |
| `lighthit.experimental.axial_fast` | 0 / 0 | not imported |
| `lighthit.experimental.axial_source` | 0 / 0 | not imported |
| `lighthit.experimental.ballistic_fast` | 0 / 0 | not imported |
| `lighthit.experimental.ballistic_spectral` | 0 / 3 | not imported |
| `lighthit.experimental.charge_preview` | 0 / 9 | not imported |
| `lighthit.experimental.cone_segment` | 0 / 6 | not imported |
| `lighthit.experimental.crossing_mc` | 0 / 4 | not imported |
| `lighthit.experimental.detector_tree` | 0 / 16 | not imported |
| `lighthit.experimental.directional_reference` | 0 / 8 | not imported |
| `lighthit.experimental.effective_segments` | 0 / 6 | not imported |
| `lighthit.experimental.event_moments` | 0 / 0 | not imported |
| `lighthit.experimental.filon_ge2` | 0 / 0 | not imported |
| `lighthit.experimental.first_charge_beta_atlas` | 0 / 0 | not imported |
| `lighthit.experimental.first_charge_only` | 0 / 0 | not imported |
| `lighthit.experimental.g4_source` | 0 / 0 | not imported |
| `lighthit.experimental.ge2_beta_response` | 0 / 0 | not imported |
| `lighthit.experimental.ge2_charge_beta_atlas` | 0 / 0 | not imported |
| `lighthit.experimental.hdf5_minimal` | 0 / 0 | not imported |
| `lighthit.experimental.joint_first` | 0 / 7 | not imported |
| `lighthit.experimental.joint_first_atlas` | 0 / 17 | not imported |
| `lighthit.experimental.joint_muon` | 0 / 0 | not imported |
| `lighthit.experimental.joint_source` | 0 / 0 | not imported |
| `lighthit.experimental.local_source_rank` | 0 / 1 | not imported |
| `lighthit.experimental.muon_components` | 0 / 6 | not imported |
| `lighthit.experimental.muon_integral` | 0 / 0 | not imported |
| `lighthit.experimental.muon_packets` | 0 / 0 | not imported |
| `lighthit.experimental.packet_skip` | 0 / 0 | not imported |
| `lighthit.experimental.point_ge2` | 0 / 0 | not imported |
| `lighthit.experimental.point_ge2_blas` | 0 / 0 | not imported |
| `lighthit.experimental.positive_fluence` | 0 / 0 | not imported |
| `lighthit.experimental.primary_polyline` | 0 / 0 | not imported |
| `lighthit.experimental.primary_prompt` | 0 / 1 | not imported |
| `lighthit.experimental.prompt_backward` | 0 / 10 | not imported |
| `lighthit.experimental.secondary_first` | 0 / 0 | not imported |
| `lighthit.experimental.segment_reference` | 0 / 16 | not imported |
| `lighthit.experimental.shell_mc` | 0 / 3 | not imported |
| `lighthit.experimental.signal_screen` | 0 / 2 | not imported |
| `lighthit.experimental.smart_muon` | 0 / 2 | not imported |
| `lighthit.experimental.spectral_response_basis` | 0 / 4 | not imported |
| `lighthit.experimental.spline_fast` | 0 / 8 | not imported |
| `lighthit.experimental.stationary_modes` | 0 / 8 | not imported |
| `lighthit.experimental.stationary_pencils` | 0 / 1 | not imported |
| `lighthit.experimental.track_first_bounds` | 0 / 0 | not imported |
| `lighthit.experimental.track_first_primitive` | 0 / 0 | not imported |
| `lighthit.experimental.track_ge2_primitive` | 0 / 0 | not imported |
| `lighthit.experimental.universal_primary` | 0 / 0 | not imported |
| `lighthit.g4_source` | 2 / 10 | all |
| `lighthit.geometry` | 5 / 11 | all |
| `lighthit.green` | 1 / 13 | all |
| `lighthit.hdf5_minimal` | 1 / 22 | all |
| `lighthit.medium` | 3 / 4 | all |
| `lighthit.model` | 12 / 13 | all |
| `lighthit.muon` | 0 / 0 | shower, muon, event, event-stopping-muon, event-muon-near-om, ntsim-electron, ntsim-muon, ntsim-nue-cc, ntsim-numu-cc |
| `lighthit.muon.adaptive_screen` | 5 / 5 | shower, muon, event, event-stopping-muon, event-muon-near-om, ntsim-electron, ntsim-muon, ntsim-nue-cc, ntsim-numu-cc |
| `lighthit.muon.api` | 5 / 6 | shower, muon, event, event-stopping-muon, event-muon-near-om, ntsim-electron, ntsim-muon, ntsim-nue-cc, ntsim-numu-cc |
| `lighthit.muon.data` | 3 / 5 | shower, muon, event, event-stopping-muon, event-muon-near-om, ntsim-electron, ntsim-muon, ntsim-nue-cc, ntsim-numu-cc |
| `lighthit.muon.filon_ge2` | 5 / 6 | muon, event, ntsim-muon, ntsim-numu-cc |
| `lighthit.muon.first_charge_beta_atlas` | 3 / 7 | muon, event, event-stopping-muon, event-muon-near-om, ntsim-muon, ntsim-numu-cc |
| `lighthit.muon.first_charge_only` | 1 / 1 | muon, event, event-stopping-muon, event-muon-near-om, ntsim-muon, ntsim-numu-cc |
| `lighthit.muon.ge2_beta_response` | 0 / 5 | not imported |
| `lighthit.muon.ge2_charge_beta_atlas` | 3 / 5 | muon, event, event-stopping-muon, event-muon-near-om, ntsim-muon, ntsim-numu-cc |
| `lighthit.muon.joint_muon` | 6 / 6 | shower, muon, event, event-stopping-muon, event-muon-near-om, ntsim-electron, ntsim-muon, ntsim-nue-cc, ntsim-numu-cc |
| `lighthit.muon.joint_source` | 4 / 10 | muon, event, ntsim-muon, ntsim-numu-cc |
| `lighthit.muon.muon_integral` | 1 / 3 | muon, event, event-stopping-muon, event-muon-near-om, ntsim-muon, ntsim-numu-cc |
| `lighthit.muon.muon_packets` | 2 / 3 | muon, event, ntsim-muon, ntsim-numu-cc |
| `lighthit.muon.packet_skip` | 10 / 10 | muon, event, ntsim-muon, ntsim-numu-cc |
| `lighthit.muon.point_ge2` | 1 / 2 | muon, event, ntsim-muon, ntsim-numu-cc |
| `lighthit.muon.point_ge2_blas` | 7 / 9 | muon, event, ntsim-muon, ntsim-numu-cc |
| `lighthit.muon.positive_fluence` | 4 / 12 | shower, muon, event, event-stopping-muon, event-muon-near-om, ntsim-electron, ntsim-muon, ntsim-nue-cc, ntsim-numu-cc |
| `lighthit.muon.preview` | 7 / 7 | shower, muon, event, event-stopping-muon, event-muon-near-om, ntsim-electron, ntsim-muon, ntsim-nue-cc, ntsim-numu-cc |
| `lighthit.muon.primary_polyline` | 4 / 5 | muon, event, event-stopping-muon, event-muon-near-om, ntsim-muon, ntsim-numu-cc |
| `lighthit.muon.secondary_first` | 7 / 7 | shower, muon, event, event-stopping-muon, ntsim-electron, ntsim-muon, ntsim-nue-cc, ntsim-numu-cc |
| `lighthit.muon.track_first_bounds` | 4 / 5 | muon, event, event-stopping-muon, event-muon-near-om, ntsim-muon, ntsim-numu-cc |
| `lighthit.muon.track_first_primitive` | 16 / 33 | muon, event, event-stopping-muon, event-muon-near-om, ntsim-muon, ntsim-numu-cc |
| `lighthit.muon.track_ge2_primitive` | 24 / 33 | muon, event, event-stopping-muon, event-muon-near-om, ntsim-muon, ntsim-numu-cc |
| `lighthit.muon.universal_primary` | 8 / 9 | muon, event, event-stopping-muon, event-muon-near-om, ntsim-muon, ntsim-numu-cc |
| `lighthit.prompt` | 19 / 31 | all |
| `lighthit.prompt_reference` | 0 / 7 | not imported |
| `lighthit.providers` | 0 / 2 | not imported |
| `lighthit.readout` | 2 / 6 | all |
| `lighthit.selftest` | 0 / 2 | not imported |
| `lighthit.shower` | 0 / 0 | shower, event, event-stopping-muon, event-muon-near-om, ntsim-electron, ntsim-nue-cc, ntsim-numu-cc |
| `lighthit.shower.api` | 10 / 10 | shower, event, event-stopping-muon, event-muon-near-om, ntsim-electron, ntsim-nue-cc, ntsim-numu-cc |
| `lighthit.shower.data` | 2 / 3 | shower, event, event-stopping-muon, event-muon-near-om, ntsim-electron, ntsim-nue-cc, ntsim-numu-cc |
| `lighthit.shower.ge2` | 5 / 5 | shower, event, event-stopping-muon, event-muon-near-om, ntsim-electron, ntsim-nue-cc, ntsim-numu-cc |
| `lighthit.shower_basis` | 8 / 22 | all |
| `lighthit.single` | 0 / 14 | all |
| `lighthit.single_fast` | 0 / 3 | not imported |
| `lighthit.sources` | 15 / 32 | all |
| `lighthit.transport` | 29 / 61 | all |
| `lighthit.viewer` | 17 / 18 | all |

## Functions per module

Grouped by the set of scenarios that executed them.

### `lighthit`

* **shower, muon, event, event-stopping-muon, prompt, hybrid, ntsim-electron, ntsim-muon, ntsim-nue-cc, ntsim-numu-cc**: `__getattr__`

### `lighthit._angular_numba`

* **not executed**: `_nodewise` (numba), `free_moments_and_ratios`

### `lighthit._ballistic_interval_numba`

* **shower**: `_acceptance` (numba), `_arrival` (numba), `_stationary_cuts` (numba), `_time_root` (numba), `_integral` (numba), `_precheck` (numba), `_pair_contribution` (numba)
* **shower, muon, event, event-stopping-muon, event-muon-near-om, ntsim-electron, ntsim-muon, ntsim-nue-cc, ntsim-numu-cc**: `interval_candidates` (numba), `interval_kernel_pairs` (numba)
* **not executed**: `interval_kernel` (numba)

### `lighthit._directional_numba`

* **hybrid**: `_apply` (numba), `PreparedDirectionalKernel.from_cache`, `PreparedDirectionalKernel.apply`
* **shower, muon**: `_start_value` (numba)
* **shower, muon, event, event-stopping-muon, hybrid, ntsim-electron, ntsim-muon, ntsim-nue-cc, ntsim-numu-cc**: `_start_norm_table`
* **shower, muon, hybrid**: `_parity` (numba, inlined), `_wigner_into` (numba), `_harmonics_into` (numba)
* **not executed**: `_apply_m0` (numba)

### `lighthit._prompt_numba`

* **muon**: `_stationary_path_gauss` (numba), `_batch_add_pencil` (numba), `_pencil_frame` (numba, inlined), `_gauss_legendre` (numba), `_phi_rule` (numba), `_segment_a_breaks` (numba), `_segment_order1_block` (numba)
* **muon, event, ntsim-muon, ntsim-numu-cc**: `segment_order1` (numba)
* **shower**: `_hg` (numba, inlined), `_prepare_gauss_inverse` (numba), `_find_bin` (numba), `_deposit_panel` (numba), `_cubic_moment_coefficients` (numba, inlined), `_batch_add_moments` (numba), `_prepare_pencil_look_powers` (numba), `_pencil_frame_from_ray_cached_look` (numba, inlined), `_grow` (numba), `_fill_moments_averaged_pair` (numba, inlined), `_is_cubic_moment_layout` (numba), `_accumulate_cubic_moments_pair_prepared` (numba, inlined), `_prepare_pencil_shape` (numba), `_pencil_shape_factor_from_ray` (numba, inlined), `_grouped_batch_charge_estimate` (numba), `_geometry_bit_hash` (numba, inlined), `_exact_pencil_geometry_ids` (numba), `_prepare_receiver_geometry` (numba), `_float64_bits`, `_float64_bits.<locals>.codegen`, `scalar_theta_index` (numba, inlined), `build_theta_thresholds` (numba), `build_theta_bit_lut` (numba), `_prepare_common_pencil_look` (numba), `_grouped_pencils_order1_block` (numba)
* **shower, event, event-stopping-muon, prompt, hybrid, ntsim-electron, ntsim-nue-cc, ntsim-numu-cc**: `min_distance_to_segments` (numba)
* **shower, muon**: `_quad_integral` (numba, inlined), `_batch_grid` (numba), `_batch_sigma` (numba, inlined), `_batch_deposit` (numba), `_finalize` (numba)
* **shower, muon, event, event-stopping-muon, prompt, hybrid, ntsim-electron, ntsim-muon, ntsim-nue-cc, ntsim-numu-cc**: `_ifuncs` (numba), `interval_table`, `_pixel` (numba, inlined), `_hash_slot` (numba, inlined), `compress_elements` (numba), `_grouped_pencils_order1_impl` (numba), `grouped_pencils_order1`, `finish_pencils` (numba)
* **not executed**: `gauss_inverse_mean` (numba), `gauss_inverse_mean_prepared` (numba, inlined), `_pencil_frame_from_ray` (numba, inlined), `_scratch_note`, `pencils_order1` (numba), `_fill_moments_averaged` (numba, inlined), `_accumulate_cubic_moments_pair` (numba, inlined), `_pencil_shape_factor_prepared` (numba), `_pencil_shape_factor` (numba), `threshold_theta_index` (numba, inlined), `bit_lut_theta_index` (numba, inlined), `_rings_order1_block` (numba), `rings_order1` (numba)

### `lighthit._shower_basis`

* **muon, event, event-stopping-muon, event-muon-near-om, ntsim-muon, ntsim-nue-cc, ntsim-numu-cc**: `earliest_element_arrival` (numba)
* **shower, event, event-stopping-muon, event-muon-near-om, ntsim-electron, ntsim-nue-cc, ntsim-numu-cc**: `_cone_harmonics` (numba), `_deposit_elements` (numba), `_axial_weights`, `compile_axial_slices`
* **shower, event, event-stopping-muon, ntsim-electron, ntsim-nue-cc, ntsim-numu-cc**: `_active_pair_bands` (numba, inlined), `_apply_multi_spectral` (numba), `apply_multi_spectral`
* **not executed**: `_align_geometry_reference`, `_source_norm_bound`, `_radial_norm_bound`, `_apply_multi` (numba), `apply_multi`

### `lighthit._spectral_fold`

* **not executed**: `_profile`, `_stored_profile`, `folded_cache_path`, `_original_files`, `_same_grid`, `_prepared_shape`, `build_folded_cache`, `FoldedKernels.zero_frequency`, `load_folded_cache`, `optional_folded_cache`

### `lighthit._track_transport`

* **not executed**: `_radial_windows`, `_track_frame`, `transport_track_directional`, `transport_track_directional.<locals>.apply_zero`, `transport_track_directional.<locals>.prepare_full`, `transport_track_directional.<locals>.apply_full`

### `lighthit.angular`

* **all**: `_degree`
* **not executed**: `_free_moments_and_ratios_numpy`, `_check_backend`, `_free_moments_and_ratios`, `free_moments_and_tail`, `solve_tail_system`, `angular_components`, `dense_finite_rank_reference`, `coupling_coefficients`, `_decay_rate`, `tail_ratios_continued_fraction`, `free_tail_ratios_m`, `solve_tail_system_m`, `resolvent_rows`, `dense_block_reference`

### `lighthit.api`

* **not executed**: `build`

### `lighthit.axial_fast`

* **hybrid**: `_element_angular` (numba), `accumulate_cell_gemm`, `compile_axial_source_fast`
* **shower, muon, event, event-stopping-muon, event-muon-near-om, hybrid, ntsim-electron, ntsim-muon, ntsim-nue-cc, ntsim-numu-cc**: `_layout`
* **shower, muon, hybrid**: `_harmonics_into` (numba)
* **not executed**: `_apply` (numba), `PreparedAxialKernel.from_cache`, `PreparedAxialKernel.apply`, `rigidly_moved_source`

### `lighthit.axial_source`

* **shower, event, event-stopping-muon, event-muon-near-om, hybrid, ntsim-electron, ntsim-nue-cc, ntsim-numu-cc**: `AxisFrame.coordinates`, `AxialSource.points_m`
* **shower, muon, event, event-stopping-muon, event-muon-near-om, hybrid, ntsim-electron, ntsim-muon, ntsim-nue-cc, ntsim-numu-cc**: `AxisFrame.of`, `AxisFrame.rotate`
* **not executed**: `channel_index`, `AxialSource.of`, `AxialSource.truncated`, `_weighted_quantile`, `axial_response`

### `lighthit.ballistic`

* **not executed**: `acceptance_from_coefficients`, `ballistic_directional`

### `lighthit.ballistic_fast`

* **prompt, hybrid**: `_acceptance` (numba, inlined), `_ballistic_directional_kernel` (numba), `ballistic_directional_fast`
* **not executed**: `_ballistic_kernel` (numba), `_ballistic_bins_kernel` (numba), `vectorised_ballistic_fast`, `vectorised_ballistic_bins_fast`

### `lighthit.ballistic_interval`

* **shower, muon, event, event-stopping-muon, event-muon-near-om, ntsim-electron, ntsim-muon, ntsim-nue-cc, ntsim-numu-cc**: `BallisticIntervalConfig.__post_init__`, `prepare_optics`, `ballistic_interval`, `ballistic_interval.<locals>.evaluate`

### `lighthit.band_fold`

* **shower, event, event-stopping-muon, event-muon-near-om, ntsim-electron, ntsim-nue-cc, ntsim-numu-cc**: `_legendre_table`, `BandFold.of`, `BandFold.of.<locals>.residuals`, `BandFold.metadata`, `fold_bundle`, `zero_frequency`
* **shower, event, event-stopping-muon, ntsim-electron, ntsim-nue-cc, ntsim-numu-cc**: `BandFold.source_residual`
* **not executed**: `save_folded`, `load_folded`

### `lighthit.bgvd`

* **shower, muon, event, event-stopping-muon, event-muon-near-om, laser, prompt, hybrid**: `_angular_acceptance`, `_angular_acceptance.<locals>.angular_acceptance`, `_vector`, `OMResponse.__post_init__`, `OMResponse.functions`, `OMResponse.functions.<locals>.spectral_efficiency`, `load_model`
* **shower, muon, event, event-stopping-muon, event-muon-near-om, laser, prompt, hybrid, ntsim-electron, ntsim-muon, ntsim-nue-cc, ntsim-numu-cc**: `BGVDModel.kernel`
* **not executed**: `_package_directory`, `_load_file`, `_hash`, `OMResponse.from_module`, `OMResponse.check`, `BGVDModel.save`, `load_bgvd_model`, `load_bgvd_model.<locals>.spectral_efficiency`

### `lighthit.cache`

* **all**: `acceptance_coefficients`
* **laser, hybrid, ntsim-laser**: `_validate_axis`, `CacheGrid.__post_init__`
* **laser, ntsim-laser**: `ResponseCache.__post_init__`, `ResponseCache.radius_range_m`, `ResponseCache.degree`, `ResponseCache._scale`, `ResponseCache.moments_at`, `ResponseCache._contract`, `ResponseCache.acceptance_spectrum`, `ResponseCache._from_parts`, `ResponseCache.load`
* **not executed**: `_validate_photons`, `isotropic_acceptance`, `cosine_acceptance`, `hemispherical_acceptance`, `CacheGrid.geometric`, `ResponseCache.clear_interpolation_cache`, `ResponseCache.build`, `ResponseCache.directed_spectrum`, `ResponseCache._zero_view`, `ResponseCache.directed_charge`, `ResponseCache.acceptance_charge`, `ResponseCache._charge`, `ResponseCache._arrays`, `ResponseCache._metadata`, `ResponseCache.save`, `_acceptance_from_coefficients`, `BandedResponseCache.__post_init__`, `BandedResponseCache.medium`, `BandedResponseCache.radius_range_m`, `BandedResponseCache.build_time_s`, `BandedResponseCache._dispatch`, `BandedResponseCache.directed_spectrum`, `BandedResponseCache.acceptance_spectrum`, `BandedResponseCache._charge`, `BandedResponseCache._zero_view`, `BandedResponseCache.directed_charge`, `BandedResponseCache.acceptance_charge`, `BandedResponseCache.charge_for_modules`, `BandedResponseCache.save`, `BandedResponseCache.load`, `validation_report`, `first_order_consistency`

### `lighthit.cli`

* **not executed**: `peak_rss_mib`, `configure`, `save_figures`, `main`

### `lighthit.directional`

* **hybrid**: `clebsch_gordan`, `CouplingTable.pairs`, `CouplingTable.build`, `DirectionalCache.__post_init__`, `DirectionalCache.degree`, `DirectionalCache.load`
* **shower, muon, event, event-stopping-muon, event-muon-near-om, prompt, hybrid, ntsim-electron, ntsim-muon, ntsim-nue-cc, ntsim-numu-cc**: `acceptance_bandwidth`
* **not executed**: `CouplingTable.triples`, `_start_value`, `wigner_small_d`, `_real_basis_matrix`, `real_rotation_rows`, `DirectionalCache.radius_range_m`, `DirectionalCache.acceptance_degree`, `DirectionalCache.clear_interpolation_cache`, `DirectionalCache.build`, `DirectionalCache._scale`, `DirectionalCache.coefficients_at`, `DirectionalCache._metadata`, `DirectionalCache.save`, `_frame_angles`, `directional_response`, `dense_source_channels`, `_rotate_into_frame`, `_real_harmonics`, `rotation_rows_explicit`, `rotation_rows_explicit.<locals>.dval`

### `lighthit.event`

* **event, event-stopping-muon, event-muon-near-om, laser, ntsim-electron, ntsim-muon, ntsim-nue-cc, ntsim-numu-cc, ntsim-laser**: `EventTransport.__init__`, `EventTransport.__call__`
* **event, event-stopping-muon, event-muon-near-om, ntsim-electron, ntsim-muon, ntsim-nue-cc, ntsim-numu-cc**: `SourceComponent.__post_init__`, `EventTransport._run`
* **event, event-stopping-muon, event-muon-near-om, ntsim-electron, ntsim-nue-cc, ntsim-numu-cc**: `EventTransport.shower_route`
* **event, event-stopping-muon, event-muon-near-om, ntsim-muon, ntsim-numu-cc**: `EventTransport.muon_route`
* **event, event-stopping-muon, ntsim-electron, ntsim-muon, ntsim-nue-cc, ntsim-numu-cc**: `EventTransport._describe`
* **event, ntsim-numu-cc**: `common_time_origins`, `EventTransport._compose`
* **laser, ntsim-laser**: `EventTransport.flash`, `laser_config`, `laser_kernel`

### `lighthit.event_moments`

* **not executed**: `real_spherical_harmonics`, `monomial_powers`, `_monomials`, `BlockPartition.count`, `BlockPartition._reach`, `BlockPartition.single`, `BlockPartition.split`, `BlockPartition.split.<locals>.half_diagonal`, `BlockPartition.summary`, `KernelChannels.of`, `KernelChannels.omega_per_ns`, `KernelChannels.multipoles`, `KernelChannels.channels`, `compile_joint_moments`, `evaluate_moments`, `direct_response`

### `lighthit.examples.synthetic`

* **not executed**: `synthetic_medium_model`, `baikal_like_detector`, `baikal_like_detector.<locals>.angular_acceptance`, `baikal_like_detector.<locals>.spectral_efficiency`, `quick_config`, `standard_config`, `example_sources`, `_config`, `_json_default`, `geometry_command`, `cache_command`, `run_command`, `parser`, `main`

### `lighthit.experimental.axial_blocked`

* **not executed**: `BlockedAxialKernel.of`, `BlockedAxialKernel.apply`

### `lighthit.experimental.ballistic_spectral`

* **not executed**: `_cone_bounds` (numba), `_spectral_kernel` (numba), `spectral_ballistic_charge`

### `lighthit.experimental.charge_preview`

* **not executed**: `ChargePreviewConfig.__post_init__`, `ChargePreview.charge_components_pe`, `ChargePreview.charge_pe`, `ChargePreview.unsafe`, `ChargePreview.unsafe_mask`, `_operator_lanes`, `load_charge_operators`, `_radial_diagnostics` (numba), `charge_preview`

### `lighthit.experimental.cone_segment`

* **not executed**: `ConeSegment.__post_init__`, `ConeSegment.speed_m_per_ns`, `ConeSegment.cone_cosine`, `ballistic_spectrum`, `_scatter`, `segment_spectrum`

### `lighthit.experimental.crossing_mc`

* **not executed**: `_henyey_greenstein_cosines`, `_turn`, `crossing_estimate`, `mean_cosine`

### `lighthit.experimental.detector_tree`

* **not executed**: `_points`, `_indices`, `AABB.__post_init__`, `AABB.from_points`, `aabb_distances`, `SourceBlock.__post_init__`, `make_source_blocks`, `AbsoluteChargeBound.__post_init__`, `DetectorTree.__init__`, `DetectorTree.__init__.<locals>.labels`, `DetectorTree.__init__.<locals>.groups`, `DetectorTree.__init__.<locals>.node`, `DetectorTree.__init__.<locals>.string_nodes`, `DetectorTree.from_detector`, `DetectorTree.range_candidates`, `DetectorTree.range_candidates.<locals>.priority`

### `lighthit.experimental.directional_reference`

* **not executed**: `spherical_quadrature`, `_frame_from_axis`, `khat_quadrature_response`, `_node_angles`, `_batched_solve`, `_direction_operator`, `dense_angular_reference`, `axisymmetric_reference`

### `lighthit.experimental.effective_segments`

* **not executed**: `EffectiveSegments.__len__`, `EffectiveSegments.summary`, `fit_effective_segments`, `effective_segment_response`, `full_response`, `vectorised_ballistic`

### `lighthit.experimental.joint_first`

* **not executed**: `_evaluate_path` (numba), `_temporal_contract` (numba), `PencilFirstCDF.build`, `PencilFirstCDF.cdf_path`, `PencilFirstCDF.contract`, `contract_joint_pencils`, `direct_photon_histogram`

### `lighthit.experimental.joint_first_atlas`

* **not executed**: `_coordinate` (numba), `_lagrange4` (numba), `_prepare` (numba), `_pchip_slopes` (numba), `_sample_curve` (numba), `_contract` (numba), `_stationary_direct` (numba), `_acceptance_extrema`, `_positive_charge_bound`, `PreparedFirstCDF.contract`, `UniversalFirstCDFAtlas.build`, `UniversalFirstCDFAtlas.save`, `UniversalFirstCDFAtlas.load`, `UniversalFirstCDFAtlas.prepare`, `UniversalFirstCDFAtlas.charge_coefficients`, `photon_source_guard`, `evaluate_photon_histogram_atlas`

### `lighthit.experimental.local_source_rank`

* **not executed**: `charge_preserving_ranks`

### `lighthit.experimental.muon_components`

* **not executed**: `MuonFineStrategy.__init__`, `MuonFineStrategy._labels`, `MuonFineStrategy.__call__`, `MuonFineStrategy.ge2_partition`, `transport_muon_components`, `transport_muon_components.<locals>.evaluate_prompt`

### `lighthit.experimental.primary_prompt`

* **not executed**: `transport_primary_prompt`

### `lighthit.experimental.prompt_backward`

* **not executed**: `_bounds`, `build_tree`, `_next` (numba), `_sample_cosine` (numba), `_passes` (numba), `_task` (numba), `_run` (numba), `_frame`, `direct_lobes`, `backward_order1`

### `lighthit.experimental.segment_reference`

* **not executed**: `phase_function`, `directed_first_order`, `directed_first_order.<locals>.integrand`, `SegmentGeometry.of`, `SegmentGeometry.cone_cosine`, `SegmentGeometry.speed_m_per_ns`, `SegmentGeometry.frame`, `SegmentGeometry.cone_directions`, `SegmentGeometry.ballistic_root`, `segment_ballistic_field`, `_polar_sectors`, `_polar_extent`, `segment_first_order`, `segment_first_order.<locals>.contribution`, `segment_first_order_mc`, `ballistic_by_ray_counting`

### `lighthit.experimental.shell_mc`

* **not executed**: `_walk` (numba), `shell_estimate`, `ratio_with_standard_error`

### `lighthit.experimental.signal_screen`

* **not executed**: `isotropic_axial_proxy`, `screening_candidates`

### `lighthit.experimental.smart_muon`

* **not executed**: `transport_smart_muon`, `transport_smart_muon.<locals>.fine`

### `lighthit.experimental.spectral_response_basis`

* **not executed**: `ResponseCubature.load`, `ResponseCubature.validate_optics`, `fit_response_cubature`, `evaluate_cubature`

### `lighthit.experimental.spline_fast`

* **not executed**: `_horner_eval` (numba), `PreparedMultipoles.__init__`, `PreparedMultipoles.of`, `PreparedMultipoles.grid`, `PreparedMultipoles.medium`, `PreparedMultipoles.radius_range_m`, `PreparedMultipoles.moments`, `PreparedMultipoles.moments_at`

### `lighthit.experimental.stationary_modes`

* **not executed**: `StationaryModes.build`, `StationaryModes.scalar_current`, `StationaryModes.leading_attenuation_per_m`, `StationaryModes.leading_mean_cosine`, `StationaryModes.isolated_leading_mode`, `single_scalar_current`, `single_scalar_current.<locals>.integrand`, `collision_components`

### `lighthit.experimental.stationary_pencils`

* **not executed**: `stationary_pencils_charge`

### `lighthit.g4_source`

* **shower, event, event-stopping-muon, event-muon-near-om, hybrid, ntsim-electron, ntsim-nue-cc, ntsim-numu-cc**: `LightElements.__len__`
* **shower, muon, event, event-stopping-muon, event-muon-near-om, hybrid, ntsim-electron, ntsim-muon, ntsim-nue-cc, ntsim-numu-cc**: `LightElements.midpoints_m`
* **not executed**: `cherenkov_yield_per_m`, `SourceContract.as_dict`, `LightElements.midpoint_time_ns`, `LightElements.centroid_m`, `LightElements.extent_m`, `LightElements.summary`, `LightElements.moved`, `load_event`

### `lighthit.geometry`

* **laser, ntsim-laser**: `_readonly_vector`, `GeometryRegion.from_positions`, `DetectorGeometrySummary.centroid_m`, `describe_geometry`, `point_source_radial_range`
* **not executed**: `GeometryRegion.as_dict`, `DetectorGeometrySummary.bounds_min_m`, `DetectorGeometrySummary.bounds_max_m`, `DetectorGeometrySummary.size_m`, `DetectorGeometrySummary.bounding_radius_m`, `DetectorGeometrySummary.as_dict`

### `lighthit.green`

* **shower, muon, event, event-stopping-muon, event-muon-near-om, laser, hybrid, ntsim-electron, ntsim-muon, ntsim-nue-cc, ntsim-numu-cc, ntsim-laser**: `SolverSettings.__post_init__`
* **not executed**: `SolverSettings.quick`, `SolverSettings.refined`, `SolverSettings.quadrature`, `GreenResult.spectrum`, `GreenResult.radii_m`, `GreenResult.cosines`, `GreenResult.front_time_ns`, `GreenResult.charge_per_m2`, `GreenResult.readout`, `GreenResult.save`, `PointGreenSolver.__init__`, `PointGreenSolver.solve`

### `lighthit.hdf5_minimal`

* **shower, event, prompt, hybrid**: `read_dataset`
* **not executed**: `_unpack`, `_Header.one`, `File.__init__`, `File._offset`, `File._length`, `File._symbol_entry`, `File._heap_name`, `File._symbol_nodes`, `File._group_links`, `File._object_header`, `File._header_span`, `File._datatype`, `File._datatype_span`, `File._filters`, `File._undo_filters`, `File._dataspace`, `File._layout`, `File._chunk_addresses`, `File.dataset`, `File.groups`, `list_datasets`

### `lighthit.medium`

* **all**: `Medium.__post_init__`
* **laser, prompt, hybrid, ntsim-laser**: `Medium.extinction_per_m`, `Medium.speed_m_per_ns`
* **not executed**: `synthetic_medium`

### `lighthit.model`

* **all**: `_axis`, `WavelengthQuadrature.__post_init__`, `WavelengthQuadrature.gauss_legendre`, `SpectralMedium.__post_init__`, `SpectralMedium.wavelength_range_nm`, `SpectralMedium.sample`, `SpectralMedium.sample.<locals>.interp`, `SpectralMedium.band`, `DetectorArray.__post_init__`, `DetectorArray.spectral_weight`
* **laser, ntsim-laser**: `DetectorArray.head_on_cosine`
* **shower, event, event-stopping-muon, prompt, hybrid, ntsim-electron, ntsim-nue-cc, ntsim-numu-cc**: `DetectorArray.__len__`
* **not executed**: `DetectorArray.angular_weight`

### `lighthit.muon.adaptive_screen`

* **muon, event, event-stopping-muon, event-muon-near-om, ntsim-muon, ntsim-numu-cc**: `AdaptiveScreenConfig.__post_init__`, `detector_groups`, `detector_groups.<locals>.labels`, `adaptive_candidates`
* **shower, muon, event, event-stopping-muon, event-muon-near-om, ntsim-electron, ntsim-muon, ntsim-nue-cc, ntsim-numu-cc**: `__getattr__`

### `lighthit.muon.api`

* **muon, event, event-stopping-muon, event-muon-near-om, ntsim-muon, ntsim-numu-cc**: `muon_kernel`, `MuonTransport.__init__`, `MuonTransport.tables`, `MuonTransport.__call__`, `MuonTransport._event`
* **not executed**: `transport_muon`

### `lighthit.muon.data`

* **muon, event, event-stopping-muon, event-muon-near-om, ntsim-muon, ntsim-numu-cc**: `MuonData.load`, `MuonData.find`, `MuonData.check`
* **not executed**: `MuonData.paths`, `write_manifest`

### `lighthit.muon.filon_ge2`

* **muon, event, ntsim-muon, ntsim-numu-cc**: `_projection`, `backward_panels`, `retarded_inverse`, `wavelength_plane`, `backward_ge2_filon_many`
* **not executed**: `warm_direct_rte`

### `lighthit.muon.first_charge_beta_atlas`

* **muon, event, event-stopping-muon, event-muon-near-om, ntsim-muon, ntsim-numu-cc**: `FirstChargeBetaAtlas.__init__`, `FirstChargeBetaAtlas.resident_bytes`, `FirstChargeBetaAtlas.validate_kernel`
* **not executed**: `FirstChargeBetaAtlas.supports`, `FirstChargeBetaAtlas.query_charge_many`, `FirstChargeBetaAtlas.query_charge_many.<locals>.values`, `FirstChargeBetaAtlas.query_charge_many.<locals>.values.<locals>.at`

### `lighthit.muon.first_charge_only`

* **muon, event, ntsim-muon, ntsim-numu-cc**: `first_scattering_charge_only`

### `lighthit.muon.ge2_beta_response`

* **not executed**: `cone_beta_taylor`, `primary_tail_mask`, `subdivide_tail_elements`, `transport_ge2_tail_batch`, `transport_ge2_tail_batch.<locals>.no_prompt`

### `lighthit.muon.ge2_charge_beta_atlas`

* **muon, event, event-stopping-muon, event-muon-near-om, ntsim-muon, ntsim-numu-cc**: `Ge2ChargeBetaAtlas.__init__`, `Ge2ChargeBetaAtlas.resident_bytes`, `Ge2ChargeBetaAtlas.validate_kernel`
* **not executed**: `Ge2ChargeBetaAtlas.supports`, `Ge2ChargeBetaAtlas.segment_charge_many`

### `lighthit.muon.joint_muon`

* **muon, event, event-stopping-muon, event-muon-near-om, ntsim-muon, ntsim-numu-cc**: `JointMuonConfig.__post_init__`, `transport_joint_muon`, `transport_joint_muon.<locals>.fine`
* **muon, event, ntsim-muon, ntsim-numu-cc**: `secondary_joint_transport`, `secondary_joint_transport.<locals>.flush_ge2`, `_ge2_skip_plan`

### `lighthit.muon.joint_source`

* **muon, event, ntsim-muon, ntsim-numu-cc**: `_charged_harmonics` (numba), `_accumulate_spectral_cones` (numba), `_positive`, `compile_joint_source`
* **not executed**: `JointSource.spectrum`, `_compile_photon_histogram_streaming`, `_compile_photon_histogram_streaming.<locals>.position_anchor`, `_compile_photon_histogram_streaming.<locals>.clock_anchor`, `compile_photon_histogram`, `_unique_integer_rows`

### `lighthit.muon.muon_integral`

* **muon, event, ntsim-muon, ntsim-numu-cc**: `order1_integrals`
* **not executed**: `fit_track_integrals`, `fit_track_integrals.<locals>.fit`

### `lighthit.muon.muon_packets`

* **muon, event, ntsim-muon, ntsim-numu-cc**: `SourcePacket.source`, `partition_muon`
* **not executed**: `order1_packets`

### `lighthit.muon.packet_skip`

* **muon**: `_hg_angle_upper` (numba), `_yukawa_upper` (numba), `_ray_bound` (numba), `_azimuth_ray_bound` (numba)
* **muon, event, ntsim-muon, ntsim-numu-cc**: `_packet_bounds` (numba), `bound_constants`, `azimuth_partition`, `packet_first_bounds`, `seed_packets`, `select_skips`

### `lighthit.muon.point_ge2`

* **muon, event, ntsim-muon, ntsim-numu-cc**: `apply_point_spectral`
* **not executed**: `_apply_points` (numba)

### `lighthit.muon.point_ge2_blas`

* **muon, event, ntsim-muon, ntsim-numu-cc**: `_immutable`, `_coefficient_matrix`, `_supported_pairs` (numba), `_angular_weights` (numba), `_accumulate` (numba), `apply_point_spectral_blas`, `apply_point_spectral_blas_multi`
* **not executed**: `clear_matrix_cache`, `matrix_cache_info`

### `lighthit.muon.positive_fluence`

* **muon**: `_bracket` (numba)
* **muon, event, event-stopping-muon, event-muon-near-om, ntsim-muon, ntsim-numu-cc**: `prefixes`, `PositiveFluenceAtlas.__post_init__`, `PositiveFluenceAtlas.load`
* **not executed**: `universal_endpoints` (numba), `endpoint_histogram` (numba), `_prefix` (numba), `segment_queries` (numba), `point_queries` (numba), `PositiveFluenceAtlas.save`, `PositiveFluenceAtlas.segments`, `PositiveFluenceAtlas.points`

### `lighthit.muon.preview`

* **muon, event, event-stopping-muon, event-muon-near-om, ntsim-muon, ntsim-numu-cc**: `group_elements`, `group_elements.<locals>.reduce`, `_screen_points` (numba), `query_groups`, `acceptance_maximum`, `check_atlas_optics`, `independent_preview`

### `lighthit.muon.primary_polyline`

* **muon, event, event-stopping-muon, event-muon-near-om, ntsim-muon, ntsim-numu-cc**: `PrimaryPolylineConfig.__post_init__`, `fit_primary_polyline`, `fit_primary_polyline.<locals>.recurse`, `primary_spectral_quadrature`
* **not executed**: `order1_primary_polyline`

### `lighthit.muon.secondary_first`

* **muon, event, ntsim-muon, ntsim-numu-cc**: `evaluate_secondary_first`, `evaluate_secondary_first.<locals>.prepare`, `evaluate_secondary_first.<locals>.account`, `evaluate_secondary_first.<locals>.apply`, `_skipping_units`, `_skipping_units.<locals>.run`
* **shower, muon, event, event-stopping-muon, ntsim-electron, ntsim-muon, ntsim-nue-cc, ntsim-numu-cc**: `_evaluation_units`

### `lighthit.muon.track_first_bounds`

* **muon, event, event-stopping-muon, event-muon-near-om, ntsim-muon, ntsim-numu-cc**: `hg_angle_integral_over_sine_upper`, `yukawa_convolution_elementary_upper`, `infinite_track_first_scattering_bounds`, `combined_first_scattering_segment_bounds`
* **not executed**: `yukawa_convolution_upper`

### `lighthit.muon.track_first_primitive`

* **muon, event, event-stopping-muon, event-muon-near-om, ntsim-muon, ntsim-numu-cc**: `first_scattering_segment_bounds`, `TrackFirstAtlasStore.__init__`, `TrackFirstAtlasStore.__init__.<locals>.Access.__init__`, `TrackFirstAtlasStore.validate_kernel`, `track_first_optical_contract`
* **muon, event, ntsim-muon, ntsim-numu-cc**: `_checked_source_bounds`, `TrackFirstAtlas.query`, `TrackFirstAtlas.query.<locals>.evaluate`, `TrackFirstAtlas.query.<locals>.evaluate.<locals>.interval`, `TrackFirstAtlas.query.<locals>.evaluate.<locals>.interval.<locals>.endpoint`, `TrackFirstAtlas.query_charge_many`, `TrackFirstAtlas.query_charge_many.<locals>.evaluate`, `TrackFirstAtlas.query_charge_many.<locals>.evaluate.<locals>.endpoint`, `TrackFirstAtlasStore.__init__.<locals>.Access.__getitem__`, `TrackFirstAtlasStore._load_row`, `TrackFirstAtlasStore.resident_bytes`
* **not executed**: `TrackFirstPrimitive.__post_init__`, `TrackFirstPrimitive._spatial_interval`, `TrackFirstPrimitive._spatial_interval.<locals>.at`, `TrackFirstPrimitive.query`, `TrackFirstPrimitive.segment`, `build_track_first_primitive`, `build_track_first_rows`, `orientation_nodes`, `TrackFirstAtlas.__post_init__`, `TrackFirstAtlas.query_charge`, `TrackFirstAtlas.query_charge.<locals>.evaluate`, `TrackFirstAtlas.query_charge.<locals>.evaluate.<locals>.endpoint`, `TrackFirstAtlas.segment_charge`, `TrackFirstAtlas.segment`, `TrackFirstAtlas.save`, `TrackFirstAtlas.load`, `build_track_first_atlas`

### `lighthit.muon.track_ge2_primitive`

* **muon**: `_charge_endpoint`
* **muon, event, event-stopping-muon, event-muon-near-om, ntsim-muon, ntsim-numu-cc**: `track_ge2_optics_metadata`, `DiskTrackGe2Atlas.__init__`, `DiskTrackGe2Atlas._verify_file`, `DiskTrackGe2Atlas.validate_kernel`, `DiskTrackGe2Atlas.segment_charge_many`, `_disk_charge_many`, `_charge_pairs`
* **muon, event, ntsim-muon, ntsim-numu-cc**: `SegmentPrimitive.primitive`, `cone_point_evaluator`, `cone_point_evaluator.<locals>.evaluate`, `look_polynomials`, `_ReadMeter.__init__`, `_ReadMeter.read`, `_ReadMeter.__getattr__`, `DiskTrackGe2Atlas.is_validated`, `DiskTrackGe2Atlas.query_supported`, `DiskTrackGe2Atlas._row`, `DiskTrackGe2Atlas.resident_bytes`, `DiskTrackGe2Atlas._segment`, `DiskTrackGe2Atlas._segment.<locals>.front_at`, `DiskTrackGe2Atlas.segment`, `DiskTrackGe2Atlas.segment_charge`, `_cone_many_looks`
* **not executed**: `SegmentPrimitive.build`, `SegmentPrimitive.segment`, `orientation_basis_looks`, `TrackGe2MiniAtlas.save`, `TrackGe2MiniAtlas.load`, `TrackGe2MiniAtlas.build`, `TrackGe2MiniAtlas.segment`, `track_ge2_axial_edges`, `build_track_ge2_orientation_row`

### `lighthit.muon.universal_primary`

* **muon, event, event-stopping-muon, event-muon-near-om, ntsim-muon, ntsim-numu-cc**: `subset_kernel`, `split_source`, `DirectGe2.__init__`, `transport_universal_primary`
* **muon, event, ntsim-muon, ntsim-numu-cc**: `_legendre_rule`, `_direct_ge2_order`, `DirectGe2._load_bundle`, `DirectGe2.many`
* **not executed**: `DirectGe2.__call__`

### `lighthit.prompt`

* **prompt, hybrid**: `_ballistic_fields.<locals>.spectral`
* **shower, event, event-stopping-muon, hybrid, ntsim-electron, ntsim-nue-cc, ntsim-numu-cc**: `PromptTransportResponse.charge_pe`
* **shower, event, event-stopping-muon, prompt, hybrid, ntsim-electron, ntsim-nue-cc, ntsim-numu-cc**: `_source_elements`, `transport_prompt`, `_jit_state`, `_ballistic_backend`, `_ballistic_fields`, `_min_distance_to_segments`
* **shower, muon, event, event-stopping-muon, event-muon-near-om, prompt, hybrid, ntsim-electron, ntsim-muon, ntsim-nue-cc, ntsim-numu-cc**: `PromptConfig.__post_init__`, `_spectral_tables`
* **shower, muon, event, event-stopping-muon, prompt, hybrid, ntsim-electron, ntsim-muon, ntsim-nue-cc, ntsim-numu-cc**: `acceptance_monomials`, `acceptance_polynomial`, `_thread_count`, `_element_segments`, `_order1`, `_order1.<locals>.grouped`, `_blocks`, `compress`, `compress.<locals>.run`
* **not executed**: `PromptComponent.rate_pe_per_ns`, `PromptTransportResponse.bins_pe`, `PromptTransportResponse.window_charge_orders_pe`, `PromptTransportResponse.outside_window_pe`, `PromptTransportResponse.select`, `PromptTransportResponse.save`, `PromptTransportResponse.load`, `PromptTransportResponse.viewer_components`, `_json_default`, `_track_segments`, `_ballistic_fields.<locals>.track`, `_ballistic_fields.<locals>.shower`

### `lighthit.prompt_reference`

* **not executed**: `pencil_geometry`, `_integrand`, `_chi_of_path`, `pencil_order1_bins`, `pencil_order1_monte_carlo`, `ring_order1_bins`, `ring_order1_bins.<locals>.pencil`

### `lighthit.providers`

* **not executed**: `load_bgvd_water`, `load_bgvd_water.<locals>.sample`

### `lighthit.readout`

* **shower, muon, event, event-stopping-muon, laser, hybrid, ntsim-electron, ntsim-muon, ntsim-nue-cc, ntsim-numu-cc, ntsim-laser**: `frequency_weights`, `inverse_bins`
* **not executed**: `TimeProfile.values_per_m2`, `TimeProfile.rate_per_m2_ns`, `TimeProfile.centers_ns`, `time_bins`

### `lighthit.selftest`

* **not executed**: `run`, `main`

### `lighthit.shower.api`

* **shower, event, event-stopping-muon, event-muon-near-om, ntsim-electron, ntsim-nue-cc, ntsim-numu-cc**: `_default_prompt`, `ShowerConfig.__post_init__`, `shower_kernel`, `ShowerTransport.__init__`, `ShowerTransport.tables`, `ShowerTransport.__call__`, `ShowerTransport._event`, `transport_shower`, `transport_shower.<locals>.compiled_part`
* **shower, event, event-stopping-muon, ntsim-electron, ntsim-nue-cc, ntsim-numu-cc**: `transport_shower.<locals>.exact`

### `lighthit.shower.data`

* **shower, event, event-stopping-muon, event-muon-near-om, ntsim-electron, ntsim-nue-cc, ntsim-numu-cc**: `ShowerData.load`, `ShowerData.find`
* **not executed**: `ShowerData.of`

### `lighthit.shower.ge2`

* **shower, event, event-stopping-muon, event-muon-near-om, ntsim-electron, ntsim-nue-cc, ntsim-numu-cc**: `_band_field`, `node_sources`, `delay_groups`
* **shower, event, event-stopping-muon, ntsim-electron, ntsim-nue-cc, ntsim-numu-cc**: `collapse`, `apply_folded`

### `lighthit.shower_basis`

* **shower, event, event-stopping-muon, event-muon-near-om, ntsim-electron, ntsim-nue-cc, ntsim-numu-cc**: `ShowerBasis.__post_init__`, `_frequency_indices`, `_window_report_from_photons`
* **shower, muon, event, event-stopping-muon, event-muon-near-om, ntsim-electron, ntsim-muon, ntsim-nue-cc, ntsim-numu-cc**: `Ge2KernelStore.__init__`, `Ge2KernelStore._key`, `Ge2KernelStore._adopt`, `Ge2KernelStore.bundle`, `Ge2KernelStore.load`
* **not executed**: `Ge2KernelStore.__len__`, `Ge2KernelStore.nbytes`, `Ge2KernelStore.kernel`, `Ge2KernelStore.save`, `_radial_window_report`, `transport_hybrid_basis`, `_transport_hybrid_spectral_basis`, `_transport_hybrid_spectral_basis.<locals>.band_pieces`, `_transport_hybrid_spectral_basis.<locals>.one_band`, `_transport_hybrid_spectral_basis.<locals>.source_field`, `_transport_hybrid_spectral_basis.<locals>.compile_field`, `_transport_hybrid_spectral_basis.<locals>.apply_spectral_group`, `_transport_hybrid_spectral_basis.<locals>.apply_spectral_group.<locals>.subset`, `_transport_hybrid_spectral_basis.<locals>.spectral_groups`

### `lighthit.single`

* **not executed**: `hg_phase`, `_unit_rule`, `_single_scattering_path_rate_numpy`, `_check_single_backend`, `_path_rate_backend`, `single_quadrature_backend`, `_quadrature_geometry`, `single_scattering_rate`, `single_spectrum`, `single_spectrum.<locals>.integrand`, `single_spectrum.<locals>.integrand`, `single_bins`, `single_bins.<locals>.rate`, `single_bins.<locals>.integrand`

### `lighthit.single_fast`

* **not executed**: `_hg_phase_scalar` (numba, inlined), `single_scattering_path_rate` (numba, inlined), `single_spectrum_integrand` (numba)

### `lighthit.sources`

* **hybrid**: `SpectralLightElements.centroid_m`
* **laser, ntsim-laser**: `IsotropicFlash.__post_init__`, `IsotropicFlash.monochromatic`
* **shower, event, event-stopping-muon, prompt, hybrid, ntsim-electron, ntsim-nue-cc, ntsim-numu-cc**: `SpectralLightElements.photon_density_per_nm`
* **shower, event, prompt, hybrid**: `_unit_vector`, `_rotation_between`, `SourcePose.__post_init__`, `SpectralLightElements.placed`, `G4Shower.from_hdf5`
* **shower, event, prompt, hybrid, ntsim-electron, ntsim-nue-cc, ntsim-numu-cc**: `_principal_axis`
* **shower, muon, event, event-stopping-muon, event-muon-near-om, prompt, hybrid, ntsim-electron, ntsim-muon, ntsim-nue-cc, ntsim-numu-cc**: `SpectralLightElements.__post_init__`, `SpectralLightElements.__len__`, `SpectralLightElements.midpoints_m`, `SpectralLightElements.field`, `SpectralLightElements.subset`
* **not executed**: `IsotropicFlash.broadband`, `SpectralLightElements.extent_m`, `SpectralLightElements.principal_axis`, `SpectralLightElements.moved`, `CherenkovTrack.centered`, `CherenkovTrack.placed`, `CherenkovTrack.earliest_arrival_ns`, `CherenkovTrack.to_elements`, `G4Shower.centroid_m`, `G4Shower.principal_axis`, `G4Shower.extent_m`, `G4Shower.placed`, `SyntheticShower.gaussian`, `SyntheticShower.centroid_m`, `SyntheticShower.principal_axis`, `SyntheticShower.extent_m`, `SyntheticShower.placed`

### `lighthit.transport`

* **all**: `KernelConfig.__post_init__`, `TransportKernel.__init__`
* **hybrid**: `TransportKernel._directional_cache_for`, `TransportKernel.transport_hybrid`, `TransportKernel._transport_directional`, `TransportKernel._transport_directional.<locals>.compile_spectral`
* **laser, hybrid, ntsim-laser**: `TransportKernel._notify_cache`
* **laser, ntsim-laser**: `TransportKernel._cache_key`, `TransportKernel._cache_for`, `TransportKernel._auto_method`, `TransportKernel.transport`, `TransportKernel._response`, `TransportKernel._flash_nodes`, `TransportKernel._transport_isotropic`
* **ntsim-electron, ntsim-muon, ntsim-nue-cc, ntsim-numu-cc**: `HybridTransportResponse.charge_pe`, `HybridTransportResponse.save`
* **ntsim-laser**: `TransportResponse.charge_pe`, `TransportResponse.save`
* **shower, event, event-stopping-muon, prompt, hybrid, ntsim-electron, ntsim-nue-cc, ntsim-numu-cc**: `TransportKernel.transport_prompt`
* **shower, muon, event, event-stopping-muon, event-muon-near-om, hybrid, ntsim-electron, ntsim-muon, ntsim-nue-cc, ntsim-numu-cc**: `TransportKernel._directional_cache_key`, `TransportKernel._source_elements`
* **shower, muon, event, event-stopping-muon, event-muon-near-om, laser, hybrid, ntsim-electron, ntsim-muon, ntsim-nue-cc, ntsim-numu-cc, ntsim-laser**: `TransportKernel._solver_settings`, `TransportKernel._band_key`, `TransportKernel._table_key`, `TransportKernel._table_path`, `TransportKernel._stored_table`
* **shower, muon, event, event-stopping-muon, event-muon-near-om, ntsim-electron, ntsim-muon, ntsim-nue-cc, ntsim-numu-cc**: `TransportKernel._directional_cache_aliases`, `TransportKernel.ge2_kernel_store`
* **shower, muon, event, event-stopping-muon, event-muon-near-om, prompt, hybrid, ntsim-electron, ntsim-muon, ntsim-nue-cc, ntsim-numu-cc**: `TransportKernel.acceptance`
* **not executed**: `CacheBuildEntry.as_dict`, `CacheBuildReport.built`, `CacheBuildReport.loaded`, `CacheBuildReport.as_dict`, `_ConsoleCacheProgress.__init__`, `_ConsoleCacheProgress.plan`, `_ConsoleCacheProgress.__call__`, `_ConsoleCacheProgress.complete`, `TransportComponent.rate_pe_per_ns`, `TransportResponse.bins_pe`, `TransportResponse.select`, `TransportResponse.components_at_frequency_cutoff`, `HybridTransportResponse.bins_pe`, `HybridTransportResponse.select`, `TransportKernel.register_method`, `TransportKernel.available_methods`, `TransportKernel.resolved_angular_backend`, `TransportKernel._stored_tables`, `TransportKernel._table_matches`, `TransportKernel._atomic_cache_save`, `TransportKernel._json_value`, `TransportKernel._write_cache_manifest`, `TransportKernel._cache_for.<locals>.update`, `TransportKernel._directional_cache_for.<locals>.update`, `TransportKernel.cache_kinds`, `TransportKernel.source_wavelengths`, `TransportKernel.build`, `TransportKernel.build_folded_directional`, `TransportKernel._transport_elements`, `TransportKernel._transport_axial_full`, `TransportKernel._transport_axial`, `TransportKernel._directional_apply`

### `lighthit.viewer`

* **ntsim-electron, ntsim-muon, ntsim-nue-cc, ntsim-numu-cc**: `source_timing`, `_validate_timing`, `merge_event_viewers`
* **ntsim-electron, ntsim-muon, ntsim-nue-cc, ntsim-numu-cc, ntsim-laser**: `_identifier`, `_display_item`, `_display_sources`, `viewer_payload`, `validate_event_viewer`, `save_event_result`, `_embedded_payload`, `_fill`, `write_event_viewer`
* **ntsim-electron, ntsim-nue-cc, ntsim-numu-cc**: `_weighted_quantiles`, `_display_spindle`
* **ntsim-laser**: `_display_source`, `_display_track_uid`
* **ntsim-muon, ntsim-numu-cc**: `_display_primary_track`
* **not executed**: `load_event_result`
