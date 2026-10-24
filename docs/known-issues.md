# Known issues

Found by reading the code; none changes a result of the default routes unless
stated. Fixed items are removed from this list.

## Behaviour

* Numba is effectively required for charged sources: the default element path
  imports `_spectral_fold`, which needs Numba, so the numpy fallbacks in
  `transport.py` (`ballistic.py`, `AxialSource.of`, `directional_response`) are
  unreachable with the default `spectral_folded_cache="auto"`.
* `kernel.build(source=spectral_elements)` with an isotropic OM builds the
  multipole table, but `_transport_elements` takes the directional path for
  spectral cones; with `cache_policy="require"` the transport then fails.
* `build(source=...)` silently overrides an explicit `method`.
* The flash path contracts the acceptance up to `source_degree` without the
  bandwidth check that element routes apply.
* With `cache_directory=None`, hybrid runs and `Ge2KernelStore.bundle` drop
  directional tables from memory, so later events rebuild them.
* `EventTransport` completion: with `max_rounds=4` at most three recomputing
  rounds succeed (a recomputation in the last round raises "did not converge").
* Source points closer than 3 m to an OM (muon and shower routes): order >=2
  of these points uses the near-field clamp (`transport-kernel.md`, Near
  passes), checked against the photon Monte Carlo on one event only (6 GeV
  muon at 1.76 m: 0.167 pe vs 0.145 +- 0.024 pe). `TransportKernel.transport`
  and `transport_hybrid_basis` (default `near_points="raise"`) still refuse them.
* The shower and muon event routes integrate order 0 to first entry on the
  supplied receiver surface at every distance and source size. The complete
  event cost is higher than the retained interval comparison
  (`direct-first-entry.md`). A detector without surface geometry is a point
  model. Orders 1 and >=2, the standalone leading `ballistic_interval`, the
  frozen-cone order 0 and the flash path still use the OM centre. The first-entry row builder in
  `receiver-first-entry.md` is tested separately and is not yet connected to
  these routes or their prepared tables.

* Muon route: order >=2 charge and time profile of the same pair can come from
  different methods (atlas or batch charge, direct Filon profile), so a bin sum
  need not equal the charge column; near-field errors other than "unvalidated
  inner" are not turned into a shower fallback.
* `G4Shower.from_hdf5` drops steps below threshold at `reference_phase_index`
  (default 1.35); pass the medium's largest phase index.
* `ShowerData.load` does not check that `ge2_band_fold` exists; a missing fold
  is silently recomputed in memory. `MuonData.load` does check it.
* With `detail_threshold_pe` below `threshold_pe`, `EventTransport` counts a
  shower OM as detailed only when the route marks it `active` (at
  `threshold_pe`); shower OMs already computed in detail between the two
  thresholds are recomputed in the completion rounds when they qualify
  there (extra time).

## Comparison with NTSim's photon Monte Carlo

* The saved 100 GeV electron comparison gives order >=2 at 20–50 m
  1.098 ± 0.029 of sixteen photon-MC runs. An independent continuous-spectrum
  first-entry calculation gives 0.6486 ± 0.0051 pe, consistent with the fast
  route's 0.6391 pe, versus the saved MC 0.5823 pe. A systematic +10% route
  bias is therefore unconfirmed; a low fluctuation of the saved sample is
  the leading explanation. Suppression is 1 in that sample. On the same
  fifteen OMs the 1 and 10 TeV electrons give fast/MC ratios 1.013 ± 0.015
  and 0.997 ± 0.020 (squared-weight errors); the latter uses suppression 10
  and has fewer effective hits despite its higher energy. A smaller angular truncation effect, about
  1.4–1.6% on the brightest OM in three node bands, is measured separately.
  First-entry order 0 at 150–200 m is 0.854 ± 0.071; this weak group needs
  an addressed check before attributing its deficit to a systematic error.
  The distance breakdown and statistical scope are in `accuracy.md`.

Remaining model differences and historical centre-reference effects (`accuracy.md`,
Order 0 near emitters and Photon Monte Carlo over energies):

* In the retained leading interval comparison, the Monte Carlo attenuates
  to the glass while LightHit attenuates to the OM centre plane: about 1-1.5 % more light in the
  Monte Carlo at every distance, below the statistical error of most
  comparisons.
* With the former event order 0, the Monte Carlo timed hits at the glass
  and LightHit at the centre: the latter was about 0.6-0.7 ns later (measured 0.56 and
  0.69 ns on 1 TeV showers, expected (2/3) R n_g / c = 0.66 ns).
* NTSim's Monte Carlo emits the Frank-Tamm light of charge 1 for every charged
  track; LightHit uses the charge squared (differs only for nuclear fragments).

Open (`accuracy.md`, Photon Monte Carlo over energies):

* The older four-run 10 TeV muon comparison gave order 0 within 50 m at
  0.93-0.95 of MC. The addressed follow-up with independent full-statistics
  samples for its three largest families gives 0.974 +- 0.011 at 0-20 m and
  1.003 +- 0.013 at 20-50 m; both bins pass the unchanged criterion
  (`accuracy.md`). That deficit is not reproduced. The production calculation
  still needs actual glass-entry attenuation and timing. The older 10 TeV
  electron ratio, 0.975, has not been remeasured on the glass reference.
* Muon route far from the track: at 100-150 m, an addressed reconstruction
  with the current angular preview and first-entry Q0 gives 1.094 +- 0.021
  of MC, versus the older isotropic estimate 1.118 +- 0.021 (0.4% of event
  charge). It retains four saved detailed scattered responses and is not a
  new full-event selection pass. The group still fails. Ready-RTE and
  independent collision controls do not yet separate kernel truncation from
  the limited MC statistics; squared weights give a larger ratio error,
  0.042. Details and the unchanged criterion are in `accuracy.md`.
* The 10 GeV electron at 20-50 m passes for total charge (1.035 +- 0.036),
  while Q0 remains 1.130 +- 0.062 of saved MC. Direct positive-ray controls
  on all fourteen receivers confirm its numerical first-entry integral;
  the component discrepancy is not declared resolved.

Data bundles of LightHit <= 0.3.0a4 hold tables for nine wavelength nodes,
which lost 6 % of the scattered light at 50 m and 15-30 % beyond 80 m
(`accuracy.md`, Wavelength quadrature). The routes refuse such a bundle with
the default `KernelConfig` (32 nodes); `KernelConfig(wavelength_nodes=9)`
reads it and reproduces the old results.

The water table of the `bgvd_model` package (used by NTSim and by the bundle
model alike) placed its 26 absorption and scattering values on
`linspace(350, 610, 26)`, a 10.4 nm step, while they belong to 350-600 nm in
10 nm steps. The bundle of 2026-10-19 is built on the corrected grid, and
NTSim's Cherenkov band is 350-600 nm. A `bgvd_model` without the correction
gives another water: `bgvd_model_from_ntsim` then differs from the bundle
model, and the routes do not find their tables (cache keys follow the water).

## Interface

* Sources default to `cone_model="frozen"` while every event route needs
  `"spectral"`; `G4Shower.from_hdf5` and `SyntheticShower.gaussian` cannot set
  it.
* No test constructs `ShowerTransport`, `ShowerData`, `MuonTransport`,
  `MuonData` or `transport_muon` (their building blocks are tested).
* `MuonTransport`/`ShowerTransport` replace the order >=2 store of the caller's
  kernel when they reuse it.
* `transport_muon` opens all tables on every call.
* Dead parameters: `ShowerTransport.__call__(progress=)`,
  `node_sources(geometry=)`, the `first_backend="atlas"` parameters of
  `JointMuonConfig`, `selection_margin` of `transport_universal_primary` on the
  route; unused functions: `MuonData.paths`, `muon_packets.order1_packets`,
  `ge2_beta_response.primary_tail_mask`, `filon_ge2.warm_direct_rte`.
* The `lighthit` CLI runs only the single-band point solver, has no tests, and
  defaults to the numpy angular backend.

## Docstrings that disagree with the code

* `hdf5_minimal`: says compression is refused; deflate, shuffle and fletcher32
  are decoded.
* `g4_source.load_event`: says zero-length rows start a particle; the code
  splits on uid changes.
* `build_folded_directional`: says opting in never changes the default path;
  with `"auto"` a built folded cache is used.
* `event.laser_config` / `laser_kernel`: called "monochromatic"; the wavelength
  comes from the flash.
* `ShowerBasis.time_deposit`: also accepts `"nearest"`.
* `PromptConfig.screen`: for spectral cones the screen runs only when
  `selection_extra_charge` is passed.
* `_angular_numba`: says it is imported only on explicit request; `"auto"`
  selects it.
* `docs/transport-kernel.md` says `TransportResponse`, `HybridTransportResponse`
  and `PromptTransportResponse` share `charge_components_pe` and
  `components_pe`; the prompt response has `charge_orders_pe` and
  `bins_orders_pe` (two orders) instead.
