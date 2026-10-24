# Shower route (`lighthit.shower`)

`ShowerTransport(kernel, data=None, *, config=None, threads=None)` returns the
response of one shower (any set of charged steps without a dominant muon: e,
gamma, hadrons, the vertex of a nu_e CC or NC event) in all three orders.
`transport_shower(kernel, source, ...)` is the same calculation as a function
with explicit tables. Modules: `shower/api.py`, `shower/ge2.py`,
`shower/data.py`, plus `prompt.py`, `ballistic_interval.py`, `_shower_basis.py`,
`shower_basis.py`, `band_fold.py`.

## Tables

* `data`: a `ShowerData`, a manifest path, or None. With None,
  `ShowerData.find(kernel.config.cache_directory)` tries, in order,
  `$LIGHTHIT_SHOWER_DATA`, `<cache>/shower-data.json`,
  `<cache>/../shower-data.json`, `<cache>/muon-data.json`,
  `<cache>/../muon-data.json`. A manifest needs `rte_cache` and `ge2_pack`
  (relative to the manifest); `ge2_band_fold` is optional.
  `ShowerData.of(rte_cache, ge2_pack)` takes the paths directly.
* `shower_kernel` reuses the given kernel when its cache directory is
  `data.rte_cache` and its grids are the route's (65 table frequencies,
  321 readout frequencies, both to 1.2 rad/ns); otherwise it builds a sibling
  kernel with those grids and `cache_policy="require"`.
* On the first event the >=2 pack is memory-mapped, the band fold is computed
  (`BandFold.of(kernel, rank=3)`) and the pack is folded onto the node bands.
  A prepared fold (`ge2_band_fold`) of the same rank, subdivision and
  frequencies is loaded instead and its bands are reused; its stored residual
  must match the kernel's (otherwise `ValueError`). A prepared fold of other
  settings is ignored and the route folds in memory.

## Requirements on the source

`cone_model="spectral"` (otherwise `ValueError`); the source defaults are
frozen, so use `replace(elements, cone_model="spectral")`. Steps with
`beta * n_max <= 1` are dropped.

## Steps of one event

1. **Delayed emission (order >=2 only).** Relative to the earliest step, a
   step's delay is `start_ns - t0 - |x - x0|/c`. Steps delayed by at most
   `prompt_delay_ns` (20 ns) are prompt. Later steps that cannot reach the
   readout window count in the charge only (zero frequency); the others are
   grouped in `delayed_cell_ns` (20 ns) cells, each read on its own shifted
   clock. Orders 0 and 1 are computed in physical time on all steps.
2. **Order >=2 source.** Each node band compiles its own threshold, phase index
   and photon weights `C0/lambda^2 + C2/(lambda^2 n^2)` into an axial-slice
   basis (`_shower_basis.compile_axial_slices` with `ShowerBasis`: slices every
   0.25 m on the principal axis, cubic axial weights, a five-point transverse
   stencil reproducing moments up to second order, emission phases on a
   0.02 ns lattice), scaled by the band's `w * eta`.
3. **Radial window.** Basis points closer than the table's `r_min` are left
   out of order >=2 and the near-field clamp adds the light of the steps closer
   than `r_min` (`near_field.py`, `transport-kernel.md`; reported in
   `metadata["near_field"]` and by a `RuntimeWarning`); the light beyond
   `r_max` is not computed: it is estimated per OM, and OMs above 1e-4 pe
   (`ShowerBasis.radial_tail_tolerance_pe`) get a `RuntimeWarning` and are
   listed with their estimates in `metadata["radial_tail"]`
   (`radial_tail_policy="raise"` stops the route instead).
4. **Predictor.** Each band's points are collapsed into 2 m cells
   (`predictor_cell_m`) and the zero-frequency order >=2 charge is predicted for
   every reachable OM.
5. **Orders 0 and 1.** `kernel.transport_prompt` with the route's
   `PromptConfig(ballistic_method="first-entry", fuse_bands=True,
   fused_pixel_face=96, fused_ring_nodes=192)`:
   * order 0: `first_entry_ballistic` on the original steps over the continuous
     spectrum (wavelength support per step-OM pair from geometry, split at
     arrival-time stationary points and bin edges, Gauss order doubled from 8
     to at most 64 until the charge changes by <= 1e-10 + 2e-6 |q|); the OM
     uses its supplied sphere at every distance and source size
     (`transport-kernel.md`, Direct light on the OM sphere);
   * order 1: the exact single-scattering integral (full HG, extinction on both
     legs, group velocity, acceptance at the true arrival direction) evaluated
     by grouped pencils; steps above threshold in every band form one fused
     unit with a frozen cone at the light-weighted mean phase index, the rest
     stay per band. A separate coarse charge pass selects OMs for the detailed
     calculation, with predicted order >=2 charge added to the selection.
     Coarse angular pixels are refined locally near a receiver direction;
     two positive radial nodes retain the curvature of the HG profile.
     Charge-only batches merge by distance, independently of emission time.
     Unselected charges are kept without event-dependent rescaling. The
     detailed grid and bins are unchanged (`accuracy.md`, Coarse order 1).
6. **Exact zero-frequency order >=2** for OMs whose threshold decision depends
   on it. If any exact value exceeds twice the prediction, all reachable OMs
   are computed exactly (`ge2_predictor.fallback_to_exact`). A predicted value
   never activates an OM.
7. **All frequencies** for active OMs (and each delayed group), demodulation by
   the OM origin, cubic interpolation from 65 to 321 frequencies, inversion to
   bins (`readout.inverse_bins`).

Time origin per OM: given by the caller (`time_origin_ns`), otherwise the
earliest possible arrival (first step plus flight at the fastest group
speed, or `earliest_element_arrival` per OM for large or long sources).

## Detail threshold

`ShowerTransport.__call__` runs inside `transport.py: detail_threshold`. When
`KernelConfig.detail_threshold_pe` is set below `threshold_pe`, every
threshold decision of steps 5-7 (screen, exact >=2 candidates, active set,
full frequencies and bins) uses the detail threshold; afterwards `active` is
narrowed to the OMs whose charge plus `selection_extra_charge` reaches
`threshold_pe`, and the metadata record `threshold_pe` and
`detail_threshold_pe`. OMs between the two thresholds keep their full charges
and bins with `active` false. None (default) or a value at or above
`threshold_pe` changes nothing.

## What is exact and what is approximated

* Order 0 uses the original steps and continuous spectrum. The receiver is
  the supplied OM sphere at every distance. Extinction, acceptance and
  timing are integrated on actual first-entry rays, with charge and bin-L1
  convergence checks (`direct-first-entry.md`).
* Approximated: order 1 grouping and band fusion; the order >=2 source basis
  (axial interpolation, transverse stencil, time lattice); node-band
  interpolation of the kernels, cubic refit of the folded kernels, `complex64`
  storage; predicted order >=2 charge of inactive OMs; the 65 -> 321 frequency
  interpolation. Measured sizes: `accuracy.md`.

## ShowerConfig defaults

`basis=ShowerBasis()`, `prompt` as above, `frequency_nodes=65`,
`band_rank=3`, `fold_subdivide=1`, `predictor_cell_m=2.0`,
`predictor_safety=2.0`, `prompt_delay_ns=20`, `delayed_cell_ns=20`. A
`ShowerBasis` with `track_uid` or `apply_tolerance_pe` is refused.

## Band fold (`band_fold.py`)

For each degree l, the detector-weighted Cherenkov spectral columns
`F_l(lambda; beta)` are interpolated from three node bands
(`L_l = F_l F_l[J]^+`, the node set J chosen to minimise the worst residual
for beta > 0.8: by exhaustive search up to 10,000 subsets, three of 32 nodes
take 4960 subsets and about 18 s; above that greedily, one band at a time). Folded kernels
`sum_lambda L_l[lambda, j] K_lambda` are refitted as cubics in log r relative to
a common attenuation and flight speed; a source is then compiled in three bands
instead of all wavelength nodes (32).

## Tests

`tests/test_shower_route.py`: band-node exactness, full-rank fold convergence,
agreement with the per-band basis route (orders 0 and 1 equal, order >=2
within 2% charge and 5% bin L1), predictor never activating an OM, fused bands,
delayed-copy identity, the saved pack after `build()`. No test constructs
`ShowerTransport` with a manifest.
