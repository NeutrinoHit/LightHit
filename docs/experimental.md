# `lighthit.experimental`

No default route imports a module of this package with default settings
(checked by importing `lighthit`, `lighthit.shower`, `lighthit.muon`,
`lighthit.event` and the core modules and inspecting `sys.modules`). Every
module imports and every test of it passes.

**Shipped in the wheel** (`packaging_filter.REQUIRED_EXPERIMENTAL`, same list in
`MANIFEST.in`): `__init__`, the aliases `axial_fast`, `axial_source`,
`ballistic_fast`, `event_moments`, `g4_source`, `hdf5_minimal`, and
`muon_components` (used by NTSim). Everything else exists only in the
repository.

## Aliases (26 files)

Six-line modules that set `sys.modules[__name__]` to a module that moved:

* `axial_fast`, `axial_source`, `ballistic_fast`, `event_moments`,
  `g4_source`, `hdf5_minimal` -> `lighthit.<name>` (shipped);
* `adaptive_screen`, `filon_ge2`, `first_charge_beta_atlas`,
  `first_charge_only`, `ge2_beta_response`, `ge2_charge_beta_atlas`,
  `joint_muon`, `joint_source`, `muon_integral`, `muon_packets`, `packet_skip`,
  `point_ge2`, `point_ge2_blas`, `positive_fluence`, `primary_polyline`,
  `secondary_first`, `track_first_bounds`, `track_first_primitive`,
  `track_ge2_primitive`, `universal_primary` -> `lighthit.muon.<name>` (not
  shipped).

Many tests still import through these names.

## Modules with their own code

| Module | What it does | Status | Tests |
|---|---|---|---|
| `muon_components` | explicit muon route (fitted primary integrals + local secondary packets, order >=2 by the axial basis); NTSim option `--hybrid-muon-components` | used by NTSim (opt-in); superseded by `lighthit.muon` | test_muon_components, test_secondary_first |
| `cone_segment` | reference model of one finite straight Cherenkov segment: analytic ballistic, Gauss quadrature of orders 1 and >=2 against multipole tables | optional, tested | test_cone_segment, test_segment_event |
| `effective_segments` | a shower fitted by N straight segments; also the numpy reference `vectorised_ballistic` | optional, tested | test_effective_segments |
| `segment_reference` | coordinate-space single-scattering reference for flashes and segments (quadrature and MC) | reference, tested | test_segment_reference |
| `crossing_mc` | surface-crossing Monte Carlo of fluence and current per order for a flash | reference, tested | test_crossing_mc |
| `shell_mc` | track-length Monte Carlo of shell-averaged fluence (Numba) | reference, tested | test_shell_mc |
| `stationary_modes` | stationary P_N eigenmode check of charge and current for a flash | reference, tested | test_stationary_modes |
| `directional_reference` | slow independent references of the directional kernel | reference, tested | test_directional |
| `detector_tree` | conservative cluster/string/OM bounding-box queries | optional, tested; used only by `charge_preview` | test_detector_tree |
| `signal_screen` | isotropic absorption-only receiver pre-screen | optional, tested; no caller | test_signal_screen |
| `spline_fast` | Numba Horner replacement of `ResponseCache.moments_at` | optional, tested; no caller | test_spline_fast |
| `ballistic_spectral` | integrated ballistic charge over the spectrum | superseded by `ballistic_interval` | test_ballistic_spectral |
| `charge_preview` | coarse per-OM preview from zero-frequency operator lanes | superseded by `lighthit.muon.preview` | test_charge_preview |
| `smart_muon` | earlier muon route: preview + selection + `muon_components` | superseded by `lighthit.muon`; untested | none |
| `primary_prompt` | orders 0 and 1 of the primary only | superseded by `muon.universal_primary` | test_ballistic_interval |
| `joint_first`, `joint_first_atlas` | one-scatter CDF and its universal atlas (`JointMuonConfig(first_backend="atlas")`) | superseded (default `first_backend="grouped"`) | test_joint_first, test_joint_first_atlas |
| `prompt_backward` | backward next-event Monte Carlo for one scattering | reference; no caller | test_prompt |
| `local_source_rank` | low-rank diagnosis of source coefficients | unused | test_local_source_rank |
| `spectral_response_basis` | positive empirical wavelength cubature | used to build the beta < 1 order-1 table of the muon route (the learned cubature behind `primary_first_beta`); no runtime caller | test_spectral_response_basis |
| `stationary_pencils` | all-time first-scattering charge of grouped pencils | unused | test_stationary_pencils |
| `axial_blocked` | numpy reordered axial apply | unused, untested | none |

## Registry and measurements

Each module above belongs to an idea of the registry `experiments/ideas/`
(for example `smart-muon`, `charge-preview`, `joint-first-cdf`,
`flash-references`, `spline-fast-multipoles`, `axial-source-route` for
`axial_blocked`), which records its status (default, used by default, tool,
superseded, rejected, open, not reproducible), how to run it and its measured
result; `experiments/REPORT.md` is the table. Decisions about keeping or
retiring a module start from that record. Which functions the default routes
execute is measured in `usage-map.md`.

Opt-in branches of library code that import modules not in the wheel:
`lighthit.muon.adaptive_screen.transport_smart_muon` (needs `smart_muon`,
`charge_preview`) and `JointMuonConfig(first_backend="atlas")` (needs
`joint_first`, `joint_first_atlas`).
