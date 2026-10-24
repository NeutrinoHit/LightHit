# What a response holds and how each number is computed

This page is a reference for reading a LightHit response: for every field of
the default routes it says what the number is, which code computes it, from
which estimate, and with which tables. Every statement is taken from the code
(cited as `file.py: function`, paths below `src/lighthit/`); measured
accuracy is in `accuracy.md`, the route walk-throughs in `shower-route.md`,
`muon-route.md` and `event-and-laser.md`.

Default routes: `ShowerTransport` (`shower/api.py`), `MuonTransport`
(`muon/api.py`, `muon/joint_muon.py`, `muon/universal_primary.py`),
`EventTransport` (`event.py`) and its calibration flash (the laser,
`TransportKernel._transport_isotropic` in `transport.py`).

Units: pe (expected photoelectrons), ns, m, nm, rad/ns. OM index `i` is the
row of `kernel.detector`.

## 1. Fields of a response

The shower, muon and event routes return a `HybridTransportResponse`, the laser
a `TransportResponse` (`transport.py`).

| Field | Shape | Meaning |
|---|---|---|
| `charge_components_pe` | (OM, 3) | all-time expected pe of order 0, 1, >=2 (zero frequency, no time window) |
| `charge_pe` | (OM,) | `charge_components_pe.sum(1)` |
| `components_pe` | (OM, bin, 3) | pe of each order inside each bin `[edge_k, edge_k+1)` of `relative_time_edges_ns`, counted from `time_origin_ns[i]` |
| `bins_pe` | (OM, bin) | `components_pe.sum(2)` |
| `relative_time_edges_ns` | (bin+1,) | `KernelConfig.relative_time_edges_ns`, default `arange(-60, 745, 5)`: 160 bins of 5 ns, -60 to 740 ns |
| `time_origin_ns` | (OM,) | absolute time of bin zero of OM `i` (below) |
| `omega_per_ns` | (65,) | frequencies of the stored spectrum (`linspace(0, 1.2, 65)`; laser: the laser kernel's grid, 321 nodes) |
| `spectrum_ge2_pe` (hybrid) | (65, OM) complex | order >=2 spectrum on the absolute clock; laser: `spectrum_pe` (omega, OM, 3) |
| `active` | (OM,) bool | the OM's summed charge reached `KernelConfig.threshold_pe` and it was computed in full (section 2); with `detail_threshold_pe` more OMs can be computed in full than are active |
| `method`, `metadata` | | route label (`"shower"`, `"joint-muon"`, `"event"`, laser `"isotropic"`); qualifiers of the numbers (section 2) |

Rules that hold for every route:

* A charge is the all-time integral; a bin holds only light inside the window.
  `bins_pe.sum(1)` is smaller than `charge_pe` by the light outside
  [-60, 740) ns, and the order >=2 bins come from a finite frequency band
  (`readout.py: inverse_bins`: trapezoid weights in omega, exact
  integration of `exp(-i omega t)` over each bin, never clipped), so they can
  be slightly negative.
* Order 0 is unscattered light, order 1 light scattered exactly once (full
  Henyey-Greenstein phase function, extinction `mu_a + mu_s` on both flights,
  group velocity, OM acceptance at the true arrival direction;
  `prompt.py`, module docstring), order >=2 everything else, from the RTE
  tables (section 4). Orders 0 and 1 are integrated in physical time; order
  >=2 is computed on 65 frequencies, demodulated by `time_origin_ns`,
  interpolated by a cubic spline onto 321 readout frequencies
  (`KernelConfig.readout_omega_per_ns`) and inverted to bins.
* `time_origin_ns[i]`: shower route (`shower/api.py: transport_shower`)
  the caller's `time_origin_ns`; otherwise, if the source extent exceeds the
  table maximum (300 m) or its time span half the window (400 ns),
  `_shower_basis.py: earliest_element_arrival` (per OM, the minimum over steps
  of start time plus distance to the step at the fastest group speed
  `c / min n_g`); otherwise the start of the earliest step plus its distance
  to the OM at that speed. Muon route (`muon/joint_muon.py:
  transport_joint_muon`): `earliest_element_arrival` over all steps, or the
  caller's. Event: `event.py: common_time_origins` (the same bound over all
  steps of all sources). Laser: flash time plus distance at the fastest group
  speed of its wavelength(s).

## 2. Active, evaluated and estimated OMs

> **Subject to change.** This section and section 6 describe the selection
> code of 2026-10-22 (thresholds, charge screens, predictors), including the
> detail threshold and adaptive selection of muon OMs
> added that day. The older measurements in experiments/REPORT.md
> (photon-mc-matrix) predate these changes and must be refreshed. Check the cited functions before
> relying on a detail.

`KernelConfig.threshold_pe` (default 0.01 pe) decides which OMs are active.
Which OMs are computed in full is decided at `KernelConfig.detail_threshold_pe`
when it is set below `threshold_pe` (default None: at `threshold_pe`): the
shower and muon routes run inside `transport.py: detail_threshold`, which
lowers their kernel's threshold to the detail threshold for the call, so
every screen, predictor and adaptive candidate of the tables below and of
section 6 reads "detail threshold" where it says threshold; then `active` is
narrowed to `charge + selection_extra_charge >= threshold_pe` and the
metadata record `threshold_pe` and `detail_threshold_pe`. The event route
uses the detail threshold in its completion rounds (section 6). Every active
OM has every order computed in full (no estimate; the muon route's bounded or
predicted pair omissions of section 6 apply to all OMs) and all its bins;
inactive OMs can carry estimates, or, between the two thresholds, full values
and bins. What an OM gets (with a detail threshold, "active" in the column
heads means computed at the detail threshold):

**Shower route** (`shower/api.py: transport_shower`, orders 0+1 from
`prompt.py: transport_prompt`):

| Order | OM farther than 300 m from every step | inactive OM within 300 m | active OM |
|---|---|---|---|
| 0, charge | 0 (route raises if a conservative bound reaches the threshold) | exact (`ballistic_interval`) | exact |
| 1, charge | 0 | coarse charge (level 1) or detailed charge and bins (level 2), `prompt_order1_level` | full |
| >=2, charge | 0 if no basis point lies in [3, 300] m (`ge2_charge_level` 0) | predicted from collapsed points (level 1) or exact basis charge (level 2); exact delayed light added | exact |
| bins, all orders | 0 | 0 | full |
| `spectrum_ge2_pe` | 0 | omega = 0 entry only (the stored >=2 charge without delayed light) | all 65 frequencies |

The bound for the far OMs (`prompt.py: transport_prompt`) is
`10 * emitted photons * area * max acceptance * max efficiency *
exp(-min mu_a d) / (4 pi d^2)` at the distance `d` to the nearest step.
Near-field order >=2 (section 5) is added to the charge of every OM that
has steps within 3 m, and to the bins only where `active`
(`near_field.py: add_near_field`; `active` is not re-evaluated afterwards).

**Muon route** (`muon/joint_muon.py: transport_joint_muon`):

Every OM receives direct light. The adaptive controller selects OMs for
scattered charge and bins with the actual angular response; `fine_charge_mask`
identifies them. Other OMs use the grouped-source preview's arrival-direction
moments contracted with the actual cubic OM acceptance. The contraction
retains the sampled angular distribution, including OM tilt; it still has
the atlas's cell, beta, grouping and sampling errors. The scalar preview
at maximum acceptance continues to control selection.

For
evaluated OMs the >=2 charge and the >=2 spectrum can come from different
methods (atlas charge, direct time profile), so `spectrum_ge2_pe[0]` need not
equal `charge_components_pe[:, 2]` (`muon/universal_primary.py`). The
near-field charge is added only to evaluated OMs (`fine_charge_mask`).

**Event route, several sources** (`event.py: EventTransport._compose`): the
charge of each order is the sum over sources; a source contributes its
detailed charge where it computed the OM in detail (route `active`, which for
the shower route is the only mark; `fine_charge_mask` of the muon route; or
exactly zero charge) and its route's estimate elsewhere. Route `active` is
taken at `threshold_pe`, so shower OMs computed in detail only because of the
detail threshold count as estimated. `active = summed charge >= threshold and
detailed in every source`, with the detail threshold when it is set;
`components_pe` is filled for these OMs, then `active` is narrowed to
`threshold_pe` (`detail_threshold.report`). One source: the route's response
is returned unchanged (plus `event_components`, `route_fallbacks`).

**Laser** (`transport.py: TransportKernel._transport_isotropic`): no
estimates. OMs whose distance to the flash lies in the table range get all
orders; outside it, only the analytic order 0 (charge and bin), orders 1 and
>=2 zero (the call raises if `10 *` an absorption-only estimate reaches the
threshold). Bins are filled for every OM; `active = inside range and charge >=
threshold`.

There is no field named `covered` in the current code; what was computed is
read from `active` and the metadata below.

**Metadata keys that qualify numbers**

| Key | Route | Meaning |
|---|---|---|
| `ge2_charge_level` (+ `_legend`) | shower | per OM: 0 outside the radial window, 1 predicted (inactive), 2 exact basis charge |
| `ge2_predictor` | shower | collapsed points, `cell_m`, `safety`, `candidates`, `reachable`, `fallback_to_exact`, `exact_over_prediction_max/min` |
| `prompt_order1_level` | shower | per OM, from the prompt: 0 not computed (beyond 300 m), 1 coarse charge only, 2 detailed charge and bins |
| `prompt_order1_level` | muon | 1 preview estimate, 2 detailed calculation |
| `inactive_order1_charge_may_be_coarse` | shower, muon | some OM retains a coarse estimate |
| `inactive_ge2_charge_may_be_coarse` | muon | true when some OM retains a preview estimate |
| `inactive_scattered_estimate` | muon | grouped spectral cone fluence with arrival-direction moments; finite-cell and sampling estimate |
| `threshold_pe`, `detail_threshold_pe` | shower, event (muon: only with a detail threshold) | the activity threshold; the detail threshold (both written by `detail_threshold.report` when it is below `threshold_pe`) |
| `fine_charge_mask` | muon | per OM: evaluated in full |
| `smart_screen` | muon | the selection audit (rounds, margins, sentinels, fallback; `inactivity_certified: False`) |
| `batches[*].primary` | muon | per fine batch: `ge2_near_field_oms`, `q1_omitted_bound_pe` (per OM omitted Q1 bound), fallback counters |
| `batches[*].secondary.ge2_skip` | muon | omitted packet-OM >=2 pairs and their predicted charge per OM (`predicted_omitted_charge_pe`) |
| `near_field` | shower, muon | list of records: OMs with steps within 3 m, clamped >=2 charge, `charge_added_oms`, `bins_added_oms` |
| `radial_tail` | shower | `policy`, `radial_high_m`, `tolerance_pe`, `max_estimate_pe`, and the OMs whose estimated light beyond 300 m exceeds 1e-4 pe with their estimates (`oms_above_tolerance`, `estimate_pe`; not computed) |
| `radial_window_m`, `radial_windowed_modules`, `radial_tail_estimate_max_pe` | shower | the window and how many OMs lost far points |
| `delayed_light` | shower | steps emitted more than 20 ns behind the light front (charge only or shifted clocks) |
| `band_fold`, `band_fold_source_residual` | shower | node bands and the a-priori relative band-interpolation residual of this source |
| `source_coefficient_fraction_dropped` | shower | light of steps below Cherenkov threshold at the largest phase index |
| `route_fallbacks` | event | muons transported by the shower route instead |
| `inactive_charge_may_be_estimated`, `completion_rounds` | event (several sources) | some OM carries an estimate of some source; the completion rounds |
| `fourier_inverted_orders`, `omitted_outside_range` | laser | orders read out from the spectrum; OMs outside the table range |

A composed event (several sources) builds new metadata (`event-and-laser.md`)
and does not carry the route keys (`near_field`, `radial_tail`,
`ge2_charge_level`, `fine_charge_mask`, ...): the first run of each source in
`event_components[*].runs` lists only their names (`route_metadata_keys`).
The `RuntimeWarning`s of the routes are still emitted.

## 3. Wavelength treatment

| Quantity | Wavelength treatment | Code |
|---|---|---|
| Order 0, shower and muon event routes | continuous spectrum: panels at the water-table knots, the OM efficiency knots and a 20 nm grid; efficiency as checked cubics; Gauss order 8, doubled up to 64 per OM until charge and bins change by <= `1e-10 + 2e-6 |q|` pe; panels split at the support limits of each step-OM pair and at arrival-time stationary points and bin edges | `ballistic_interval.py: prepare_optics, first_entry_ballistic`; `_surface_ballistic.py` |
| Order 1, shower and muon secondaries | `KernelConfig.wavelength_nodes` (32) Gauss-Legendre nodes on the water range (350-600 nm); weights `s0w = w eta / lambda^2`, `s2w = w eta / (lambda^2 n^2)`; steps above threshold in every band form one fused unit with a frozen cone at the `s0w`-weighted mean phase index, each band keeping its own optics, yield and clock; the other steps per band | `prompt.py: _spectral_tables`; `muon/secondary_first.py: _evaluation_units` |
| Order >=2, shower | source compiled in 3 node bands (nodes 10, 16, 21 of 32: 411.6, 481.0, 538.4 nm in the bundle), each with its own threshold, cone and yield, scaled by `w eta`; the other 29 nodes enter through folded kernels `sum_lambda L_l[lambda, j] K_lambda` | `band_fold.py: BandFold.of, fold_bundle`; `shower/ge2.py: node_sources` |
| Order >=2, muon (primary atlas, beta < 1 tail, secondaries, near field) | all 32 nodes | `muon/track_ge2_primitive.py`, `muon/point_ge2_blas.py`, `near_field.py` |
| Order >=2, muon primary, direct Filon batch | with the prepared fold (`ge2_band_fold`): the fold's 3 node bands and folded kernels for pieces whose fold residual at their beta is <= 1e-3, all 32 nodes for the others and for single-piece charges; without it all 32 nodes | `muon/universal_primary.py: DirectGe2.many, foldable`; `muon/filon_ge2.py: backward_ge2_filon_many(spectral=)` |
| Order 1, muon primary | its own composite rule: 2-point Gauss-Legendre on every interval between water-table and efficiency knots (70 nodes on the bundle model), independent of `wavelength_nodes`; used by the atlas build, its optical contract and the direct segment fallback | `muon/primary_polyline.py: primary_spectral_quadrature`; `muon/track_first_primitive.py: track_first_optical_contract` |
| Order 1, muon primary, beta < 1 pieces | `first-beta-charge.npz`: a correction mode fitted at build time with a 17-node learned cubature (`cubature_sha256` in its metadata) and normalised by the 70-node yield; at run time a table lookup | `muon/first_charge_beta_atlas.py`; `tools/tables/build_first_beta_charge.py` |
| Preview (muon) | atlas built on the kernel's 32-node tables | `muon/preview.py: check_atlas_optics` |
| Laser | the flash line only (532 nm); a flash with a spectral shape uses the kernel's 32 nodes | `transport.py: _flash_nodes` |

The runtime learned cubature of the primary's order 1
(`transport_universal_primary(first_cubature=...)`) is not used by
`MuonTransport` (it passes none).

**The supplied OM surface in order 0.** `DetectorArray.receiver_surface`
contains explicit geometry, independent of `effective_area_m2`. The BGVD
adapter supplies `SphereSurface` from the detector radius; NTSim supplies the
radii in `telescope.detector_array`. Model schema 2 stores these radii. Legacy
model files lack surface geometry and must be refreshed or supplied with it
before explicit surface-mode direct light is calculated. Without geometry,
`receiver_mode="auto"` is a point model; collection area supplies no radius.

`BallisticIntervalConfig(receiver_mode="surface")` uses that geometry at every
distance and source size; `receiver_mode="point"` is a convergence control.
The standalone `ballistic_interval` retains a leading-order cone window
with centre-ray attenuation, acceptance and time. The event routes use the
first-entry integral described below. Moving scattered orders and the flash
path to the actual boundary remains open.
The default `receiver_mode="auto"` uses supplied geometry at every distance
and source size. Conservative limiting-cone tests reject impossible pairs
before expensive support calculations; tabulated OM efficiency and
transmission use their exact linear-times-exponential law rather than many
verified cubic subdivisions. Geometry support alone does not complete the
glass-reference change.

`ballistic_interval.first_entry_ballistic` provides the explicit direct-light
entry integral. `PromptConfig(ballistic_method="first-entry")` computes its
charge for all working OMs and defers its bins until receiver selection.
It retains the existing scattered-light grouping and tables. The complete
entry calculation is more expensive and is now the default order 0 of the
shower and muon event routes. The muon route also defers bins until its
selection batches. `direct-first-entry.md` describes the controls and timing.

## 4. Tables

### Files of the data bundle (`$LIGHTHIT_DATA/tables`)

| File | Content | Used by |
|---|---|---|
| `rte-cache/directional-<key>.npz` | one per wavelength node (32): RTE response `t_(l lambda |mu|)(r, omega)` of both scattered orders, cubic splines in log r on 220 radii from 3 to 300 m, 65 frequencies, flight phase removed (`directional.py: DirectionalCache`) | source of the >=2 pack; muon kernel (`cache_policy="require"`) |
| `rte-cache/cache-manifest.json` | what `build()` wrote (method, kinds, wavelengths, config) | not read by the routes |
| `ge2-pack/pack.json`, `full-0.npy`, `charge-0.npy` | `Ge2KernelStore`: only the order >=2 spline coefficients, stacked (wavelength 32, radial interval 219, degree 33, pair 10, cubic power 4, frequency), `complex64`; `charge-0` is the exact omega = 0 slice | shower (folded), muon secondaries, muon direct fallback, near field |
| `ge2-band-fold/folded.json`, `folded.npy` | the pack folded onto 3 node bands (rank 3, no subdivision, 65 frequencies) | shower; direct >=2 Filon batch of the muon primary (optional) |
| `muon/q1-atlas/` | `TrackFirstAtlasStore`: order 1 of a beta = 1 track per unit `C0`, 19 impact radii 3-300 m, axial range `|u| <= 8.5 rho`, charge prefixes and time CDFs (SHA-256 per file) | muon primary order 1 |
| `muon/ge2-atlas/` | `DiskTrackGe2Atlas`: order >=2 of beta = 1 track pieces, 44 impact rows from 3.01 to 299 m, 65 frequencies, validated intervals in `manifest.json` (other impacts go to the direct fallback) | muon primary order >=2 |
| `muon/first-beta-charge.npz` | correction of the q1 atlas charge for `beta >= 0.999` | muon primary order 1 |
| `muon/ge2-beta-charge.npz` | Taylor modes (order 2 in `1/beta - 1`) of the >=2 charge for `beta >= 0.99` | muon primary order >=2 |
| `muon/preview-atlas.npz` | Monte-Carlo positive fluence and ten arrival-direction moments of orders 1 and >=2 (6e6 samples per order, seed 81), `rho` up to 479 m, `z` in +-500 m, 7 beta nodes | scalar OM selection and angular charge estimates for unselected OMs |
| `muon-data.json` | manifest of relative paths (`rte_cache`, `ge2_pack`, five muon tables, `ge2_band_fold`) | `MuonData`, `ShowerData` |
| `laser-cache/transport-<key>.npz` | multipole table (`cache.py: ResponseCache`) at 532 nm, 321 frequencies, radial range 3-488.4 m (from the bundle detector) | laser |

### Names and keys

* **RTE tables** (`transport.py: _cache_key, _directional_cache_key,
  _band_key`): `<kind>-<first 20 hex digits of sha256>` of the JSON of
  `medium` = the band at the node wavelength without its provenance label
  (`wavelength_nm`, `absorption_per_m`, `scattering_per_m`, `g`,
  `group_index`; `model.py: SpectralMedium.band`), `settings` =
  `SolverSettings` (`scattering_degree` 24, `spatial_degree` =
  `source_degree` 32, `k_max_per_m` 8, `k_panel_per_m` 0.04, `k_order` 10,
  `taper`), the omega list, `radial_range_m`, `radial_nodes`, `"flight"`; the
  directional key adds `acceptance_degree` (3 for BGVD) and
  `"directional_m_blocks"`. The phase index, the OM efficiency, area,
  acceptance coefficients and positions are in no RTE key.
* **>=2 pack** (`shower_basis.py: Ge2KernelStore._key`): the tuple of the
  directional keys of all kernel wavelengths, `source_degree`, the frequency
  indices. `_adopt` re-keys a stored bundle when each of its wavelength keys is
  an alias of this kernel's table (the legacy key or the file name found by
  `_stored_table`).
* **Band fold**: no key; the shower route matches it by rank, subdivision
  and frequencies, the muon primary by frequencies and degree (below).
* **Muon atlases**: no key; directory names come from `muon-data.json`; each
  table carries its optical contract.

### How a kernel finds a table

`transport.py: _stored_table`: in memory; the file of the physics key; the
file of the legacy key (LightHit <= 0.3.0a1, provenance included); then every
`<kind>-*.npz` of the cache directory whose stored metadata pass
`_table_matches`. `Ge2KernelStore.bundle` returns the bundle of its key, or a
slice of a denser bundle with the same wavelength keys and degree, or builds it
from the directional tables. The routes take manifests from
`$LIGHTHIT_SHOWER_DATA` / `$LIGHTHIT_MUON_DATA`, else `shower-data.json` or
`muon-data.json` in the cache directory or its parent (`shower/data.py:
ShowerData.find`, `muon/data.py: MuonData.find`), and force the kernel to
65 table and 321 readout frequencies on that cache with `cache_policy="require"`
(`shower/api.py: shower_kernel`, `muon/api.py: muon_kernel`).

### Checks on load

| Table | Check | Code |
|---|---|---|
| RTE table (scan step) | stored band equal within relative 1e-9 per leaf (`_portable.py: same_values`, `RTOL = 1e-9`); settings exactly equal; `radial_phase == "flight"`; directional: `detector_angular_model == "exact_m_blocks"` and equal acceptance degree; omega exactly equal; radii within 1e-12 | `transport.py: _table_matches` |
| >=2 pack | schema `lighthit/ge2-kernel-pack/1`; with `cache_policy="require"` a pack of another node count raises `ValueError`; a missing key falls back to the directional tables (`FileNotFoundError` under "require") | `shower_basis.py: Ge2KernelStore.load, bundle` |
| band fold | used only if rank, subdivision and frequencies (atol 1e-12) equal the route's, otherwise folded in memory; then its bands must exist and the worst design residual recomputed for this kernel must equal the stored one within relative 1e-9 | `shower/api.py: ShowerTransport.tables` |
| q1 atlas, first-beta | manifest `status == "complete"`, SHA-256 of every file; optical contract (acceptance coefficients, area of the first OM, `g`, spectral order and node count, the 70-node spectral tables) by `optics_contract.py: contract_matches`; every OM's area exactly the atlas area; first-beta: SHA-256 of the base manifest, `synthetic_validation_passed` | `muon/track_first_primitive.py: TrackFirstAtlasStore.validate_kernel`; `muon/first_charge_beta_atlas.py` |
| ge2 atlas, ge2-beta | row SHA-256; optics (32 node wavelengths and weights, absorption, scattering, phase and group index per node, `g`, acceptance coefficients, efficiency per node, one shared OM area, `beta = 1`, clock `1/c`) by `contract_matches`; ge2-beta: base validation plus identical fingerprint strings | `muon/track_ge2_primitive.py: track_ge2_optics_match`; `muon/ge2_charge_beta_atlas.py` |
| preview atlas | same node count as the kernel, else a `ValueError` naming `wavelength_nodes`; wavelength, `mus`, `mut`, phase index, `s0w`, `s2w`, HG `g` within relative 1e-9 | `muon/preview.py: check_atlas_optics` |
| direct >=2 fallback of the primary | the first 65-frequency bundle of the pack, its absorption equal to the kernel's within relative 1e-9 (`_portable.RTOL`) | `muon/universal_primary.py: DirectGe2`; `muon/filon_ge2.py` |
| band fold in the muon primary | frequencies (atol 1e-12) and degree equal to the pack bundle's, no subdivision, bands within the kernel's nodes, the design residual recomputed for this kernel equal to the stored one within relative 1e-9; otherwise the fold is not used (every node is applied) and the reason is recorded in `timings['fold_unused']` | `muon/universal_primary.py: DirectGe2._load_fold` |

`contract_matches`: accepted when the kernel's contract hashes to the stored
fingerprint, or when the values stored in the table hash to that fingerprint
and the kernel's values agree with them within relative 1e-9. The tolerance
exists because band coefficients and spectral tables come from quadrature and
interpolation, which another CPU or maths library reproduces only to the last
bits (`_portable.py`).

### Settings that make tables incompatible

* **Water** (absorption, scattering, `g`, group index; for the muon tables
  and the fold also the phase index): every table. A changed water table
  gives other RTE keys; the scan accepts only relative differences up to
  1e-9.
* **Wavelength quadrature** (`wavelength_nodes`, `wavelength_range_nm`, the
  medium's range): every RTE key (the node wavelength is in the band), the
  pack key, the fold, the ge2 atlases and the preview. The q1 atlas depends
  on the range and on the water and efficiency knots, not on
  `wavelength_nodes`.
* **Solver and grid** (`scattering_degree`, `source_degree`, `k_*`,
  `radial_range_m`, `radial_nodes`, the 65 frequencies): RTE tables, pack,
  fold, ge2 atlases.
* **OM response**: the acceptance degree (directional tables); the acceptance
  coefficients, the efficiency and the area (muon atlases; efficiency also the
  fold and the preview). The detector geometry enters no route table; the
  laser table's radial range follows the farthest OM from the detector
  centroid (`event.py: laser_config`).
* Not part of any check: `azimuthal_degree`, `cell_m`, the time edges, the
  readout grid, the threshold. The prepared fold is checked only through its
  band residual, which depends on wavelengths, weights, efficiency and phase
  index, not on absorption or scattering.

## 5. Radial range: below 3 m and beyond 300 m

The order >=2 tables cover distances `radial_range_m = (3, 300)` m. Orders 0
and 1 use no radial table.

**Below 3 m (near-field clamp, `near_field.py`).** The routes leave out of
their >=2 sums every source point closer than 3 m to an OM: shower basis
points (`shower_basis.py: _window_report_from_photons`, counted as `near`),
secondary point sources (radial window of `muon/point_ge2_blas.py`), the part
of a primary piece within 3 m of the OM (the interval `|u| < sqrt(9 - rho^2)`
is cut out, `muon/universal_primary.py`), and the beta < 1 tail
(`near_points="exclude"`). `near_field_ge2` then integrates the original
steps within 3 m (6-point Gauss on 5 cm panels) with the >=2 Green function
taken at 3 m along the same directions and the flight phase of the actual
distance, and `add_near_field` adds it (section 2), lists the OMs in
`metadata["near_field"]` and names them in a `RuntimeWarning`.

Muon rules near an OM:

* a primary piece whose line passes through the OM centre (`rho == 0`) within
  300 m along the piece raises `ValueError("primary enters unvalidated inner
  >=2 radius ...")`; `EventTransport._run` then transports that source with the
  shower route (`route_fallbacks`);
* primary order 1 uses the atlas only for impact `3 <= rho <= 300` m, beta = 1
  and `|u| <= 8.5 rho`; other pieces are integrated directly
  (`first_charge_only`, `muon_integral`), which raise if a piece passes within
  `PromptConfig.min_distance_m` (0.1 m) of the OM centre (not caught);
* the preview forces an OM into the full calculation when a source group is
  within 3 m (`muon/preview.py: independent_preview`).

**Beyond 300 m.**

| Route, order | What happens |
|---|---|
| shower, orders 0 and 1 | OMs farther than 300 m (`PromptConfig.max_distance_m` = `radial_range_m[1]`) from every step get 0; the route raises if the conservative bound of section 2 reaches the threshold |
| shower, order >=2 | basis points beyond 300 m are left out. Their light is estimated per OM as `10 * area * max acceptance * efficiency * sum photons * A(d) / (4 pi d^2)`, scaled from the 3 node bands to the full quadrature weight; `A(d) = sum_k s_k exp(-mu_a,k d) / sum_k s_k`, `s_k = w_k eta_k / lambda_k^2` (`shower_basis.py: spectral_attenuation`). OMs above `ShowerBasis.radial_tail_tolerance_pe` (1e-4 pe) get a `RuntimeWarning` and are listed in `metadata["radial_tail"]`; `radial_tail_policy="raise"` stops the route (`shower_basis.py: screen_radial_tail`). That light is not added |
| muon, order 0 | no limit (all OMs, all steps) |
| muon primary, order 1 | no limit: pieces outside the atlas are integrated directly; pieces are dropped per OM while their summed analytic bound stays <= 1e-6 pe (`muon/track_first_bounds.py`) |
| muon secondaries, order 1 | packets whose support sphere is farther than 400 m (`JointMuonConfig.prompt_radius_m`) are skipped for that OM |
| muon, order >=2 | only the track within 300 m of the OM (`reach = sqrt(300^2 - rho^2)`), only secondary points within 300 m; no estimate of the rest, no warning |
| muon, unselected OM | arrival-direction moments contracted with OM acceptance; grouping, histogram and sampling approximations |
| laser | orders 1 and >=2 only inside the laser table range; outside only order 0 (raise if `10 *` an absorption-only estimate reaches the threshold) |

## 6. Selection of OMs

> **Subject to change.** As section 2.

In this section "threshold" is the detail threshold when
`KernelConfig.detail_threshold_pe` is set below `threshold_pe` (section 2);
`active` is narrowed to `threshold_pe` at the end of each route.

**Shower** (`shower/api.py: transport_shower`; `ShowerConfig`):

1. Predicted >=2 charge of every reachable OM: the zero-frequency node-band
   sources collapsed to one point per 2 m cube (`predictor_cell_m`; sum of
   channels at the centroid of the `|l=0|` weights, `shower/ge2.py:
   collapse`), applied with the folded kernels. Delayed light is computed
   exactly for every reachable OM.
2. Orders 0 and 1 (`prompt.py: transport_prompt`) with the extra selection
   charge `E = 2 |predicted| + |delayed| + other sources`
   (`predictor_safety` 2). For a grouped source (more than 256 steps) and a
   nonzero threshold a separate coarse order-1 pass evaluates every OM
   within 300 m. Detailed charge and bins are computed where
   `q0 + E + screen_margin * q1_coarse >= threshold`. The coarse pass uses
   2 m cells, 8 pixels per cube face and 24 ring nodes. A parent within six
   apparent angular extents of a receiver direction is replaced by its 4x4
   angular children (`screen_refine_factor`); parent and children come from
   the same ring moments, and each emission is counted once. Two radial Gauss
   nodes integrate the inverse-angle-weighted Gaussian measure. Charge batches
   use distance and angle, with `merge_log_r` (0.02), independently of emission
   time; `path_step` applies over the full charge integral. No charge is scaled
   from another OM. With at most 256 steps every OM within 300 m gets the full
   order 1. The coarse/fine ratios remain diagnostics (`accuracy.md`).

3. Exact >=2 charge for the candidates `q0 + q1 + 2 |predicted| + delayed +
   other >= threshold` (all reachable OMs at threshold 0). If any exact charge
   exceeds twice its prediction, every reachable OM is computed exactly
   (`ge2_predictor.fallback_to_exact`).
4. `active = (q0 + q1 + q2 + other >= threshold)` among OMs with a nonzero
   order 0+1 or a reachable >=2 point; a predicted >=2 never activates an OM
   (`RuntimeError`). Full frequencies and bins only for active OMs.

**Muon** (`muon/adaptive_screen.py: adaptive_candidates`; `MuonTransport`
passes `AdaptiveScreenConfig(scatter_margin=2)`):

* scores: `q0` exact plus other sources, `s = max(q1_preview, 0) +
  max(q2_preview, 0)` with the preview at the maximum of the acceptance (the
  preview supplies approximate charges on unselected OMs); wanted when `q0 + margin * s >= threshold`, or
  forced, or a sentinel;
* forced by the preview: a source group within 3 m, or spread
  `sigma > 0.25 s` with `s + 4 sigma >= threshold / 8`; sentinels: per
  (cluster, string) the OM with the largest `q0 + s` if it reaches
  `threshold / 64`;
* after each round of full calculations: an evaluated OM below the score but
  above the threshold forces its whole cluster; a preview blind to scattered
  light (`s <= 1e-8 threshold` while the full order 1+>=2 exceeds
  `threshold / 8`) forces all OMs; the margin becomes `max(margin, 1.5 *
  max(full / preview scattered))` over informative OMs, and all OMs are
  forced when that exceeds 128;
* rounds 0-2 evaluate the new wanted OMs; in round 3 (`max_rounds`), or at
  threshold 0, every OM is evaluated if any wanted OM is left;
* `audit_all_rejected=True` is an optional full control, disabled by default;
* unselected OMs retain angular preview estimates from arrival-direction moments;
* `active = evaluated and q0 + q1 + q2 + other >= threshold`.

Inside the full calculation (threshold 0 on the selected OMs) two omissions
remain: secondary order 1 skips packet-OM pairs while their certified bounds
stay below 1e-3 of the OM's seed charge (`muon/packet_skip.py`); secondary
>=2 skips, per OM, pairs at least 3 packet sizes away while 3 times their
summed predicted charge (one centroid point per packet at omega = 0) stays
<= 1e-3 of the OM's predicted total, a prediction, not a bound
(`muon/joint_muon.py: _ge2_skip_plan`).

**Event** (`event.py: EventTransport._compose`, `safety` 3, `max_rounds` 4;
threshold `min(threshold_pe, detail_threshold_pe)`): showers first, each route called with the summed charge of the sources done so
far as `selection_extra_charge` and the common clock; then rounds: per OM the
upper sum `sum over sources of (detailed charge, or max(q0, 0) + 3 (|q1| +
|q2|))`; every OM where it reaches the threshold and some source with nonzero
charge has not computed it in detail is recomputed by that source on those OMs
(`receivers`, threshold 0, all orders in full); the loop ends when a round
recomputes nothing (`RuntimeError` after 4 rounds).

**Laser**: no selection; every OM in range is computed in full.

**Parallel blocks of the grouped order 1** (`prompt.py: _grouped_blocks`): the
OMs of a grouped source are split into blocks computed in parallel; each block
holds a scratch table per pencil, so the number of blocks running at once is
limited by `PromptConfig.grouped_scratch_bytes`. The default (None) is a
minimum of the required array storage and currently available memory
(`_scratch_budget`; Linux cgroup limits are included). An explicit value caps
the scratch instead. The block estimate includes geometry, look and shape
arrays, moment-hash growth, and wavelength/time work arrays. If even one
block does not fit, the calculation raises `MemoryError` before allocation.
There is no fraction of installed RAM or fixed GiB range. The budget changes only the time and the peak
memory, not the charges: every OM is still computed by one block. Earlier measurements with fixed budgets: the 1 TeV
pion (AMD EPYC 7542, 471 GB) takes 628 s on one thread, and on 16 threads 413 s
with a 1 GiB budget and 81 s with 16 GiB, with equal charges.
The muon route parallelises less (Python stages between the compiled ones):
the 10 TeV muon takes 5814 s on one thread and 2885 s on 16; several events
in separate processes use the cores better (`MuonTransport(threads=...)`).

## 7. Comparison with the photon Monte Carlo

**Monte Carlo** (`experiments/photon_mc.py`): one process regenerates the
Geant4 event of `experiments/regenerate_inputs.py` (same generator and Geant4
seed) and runs NTSim's chain `CherenkovGenerator` (phase index of the BGVD
water, emission volume the telescope world with a 100 m margin) ->
`MCPhotonPropagator` (Henyey-Greenstein, `n_scatterings=30`) ->
`SmartRayTracer` on clusters 1-5. The photon stages are seeded with
`--seed`, default `20261021 + run`; Geant4 keeps the event's own seed, so runs
differ only in photon statistics. `--suppression K` sets NTSim's
`photon_suppression`: a Poisson number of photons with mean `1/K` of the
expected number, each with weight `K` (unbiased). Output per run: the steps
(`-steps.npz`, `-parts.npz`) and the hits (`-hits.npz`: OM uid, time, weight
`phe`, metadata with the seed).

**Scattering order of a hit** (`--orders`, `_order_tracer`): the index of the
photon path segment that reaches the OM, `searchsorted(vertex times, hit
time, "right") - 1`, i.e. the number of scatterings before the hit. A path
holds 30 vertices (emission and 29 scattering points), so the recorded orders
are 0-28; light needing more scatterings is not in the Monte Carlo.

**Matrix** (`experiments/harness/cases_mcmatrix.py`, cases
`photon-mc-matrix/<event>`):

* LightHit: `EventTransport(kernel)` with defaults (65/321 frequencies,
  threshold 0.01 pe) once, on the steps of the first run;
  `charge_components_pe` of every OM, inactive OMs included (their estimates
  enter the sums). The step files of all runs are hashed and the result
  records `same_steps_in_all_runs`.
* Monte Carlo: per run and OM the `phe` sums of orders 0, 1, >=2 (orders above
  2 merged), the mean over runs.
* Distance of an OM: from its centre to the nearest of the start, middle and
  end points of all steps of the step file. Bins: 0-20, 20-50, 50-100,
  100-150, 150-200, 200-300, > 300 m.
* Error of a bin: the standard deviation (ddof 1) over runs of the bin's
  charge divided by `sqrt(runs)`; with one run `sqrt(sum phe^2)`. The ratio
  error is that error relative to the MC charge times the ratio (LightHit
  counts as exact).
* Pass: a bin with more than 0.1 % of the event's MC charge passes when
  `|LightHit/MC - 1| <= max(2 sigma, 3 %)` for the total of all orders; the
  per-order ratios are reported, not tested. `MATRIX` lists the planned runs
  and suppression per event; the case uses every run found in `inputs/mc`.

**Reference point of an OM**: the Monte Carlo records a hit where the photon
enters the OM sphere (time and attenuation up to the glass); LightHit
computes event order 0 to that same first entry with the supplied geometry.
The scattered orders still use centre-reference tables. The retained
leading interval comparison has direct light about
(2/3) R n_g / c = 0.66 ns later (R = 0.216 m) and about 1 % weaker
(attenuation over 2/3 R); those historical differences are measured in
`accuracy.md`, Photon Monte Carlo over energies.

**Single runs** (`experiments/harness/cases_mc.py`, cases
`photon-mc-comparison/<event>-run<N>`): totals with the MC error
`sqrt(sum phe^2)`; pulls `(LightHit - MC) / sigma` over OMs with >= 0.1 pe in
either (sigma per OM `sqrt(sum phe^2)`, or `sqrt(LightHit)` without MC
hits); the L1 of normalised 5 ns time histograms on LightHit's clock for
active OMs with >= 1 pe in both. No pass criterion.
