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
* Order 0 sees the OM sphere only for steps within
  `BallisticIntervalConfig.finite_receiver_m` (20 m); beyond, the point
  receiver at the OM centre samples the conical shells of short steps: 3-11 %
  median, up to 98 % on single dim OMs at 20-100 m (`accuracy.md`). The sphere
  is the leading order in `R/rho` (2 % off at the window edge at 2 m) and does
  not describe a step whose line crosses the sphere. Orders 1 and >=2 and the
  frozen-cone order 0 (`ballistic.py`, `ballistic_fast.py`, the quadrature
  method and the flash path) use the OM centre at every distance.
* Muon route: order >=2 charge and time profile of the same pair can come from
  different methods (atlas or batch charge, direct Filon profile), so a bin sum
  need not equal the charge column; near-field errors other than "unvalidated
  inner" are not turned into a shower fallback.
* `G4Shower.from_hdf5` drops steps below threshold at `reference_phase_index`
  (default 1.35); pass the medium's largest phase index.
* `ShowerData.load` does not check that `ge2_band_fold` exists; a missing fold
  is silently recomputed in memory.
* `lighthit.muon.preview.check_atlas_optics` does not compare the HG `g`.

## Comparison with NTSim's photon Monte Carlo

Difference by design (`accuracy.md`, Order 0 near emitters):

* The Monte Carlo attenuates each photon to its entry point on the glass
  sphere, LightHit to the OM centre plane: about 1-1.5 % more light in the
  Monte Carlo at every distance, below the statistical error of the
  comparisons made so far.

Open (`accuracy.md`, Wavelength quadrature):

* Muon route far from the track: on the 100 GeV muon, OMs 60-100 m from the
  track piece get 1.9 times the Monte-Carlo order 1 and 2.6 times its order
  >=2, OMs beyond 100 m 3.5 times its order >=2 (0.05 pe of 1.58 pe in all).
  The isotropic point source agrees at these distances, so the excess comes
  from the muon route (primary >=2 atlas or direct batch, secondaries), not
  from the RTE tables. One event only.
* With 32 wavelength nodes the direct order >=2 batch of the muon primary
  runs over every node: 100 GeV muon 36 -> 84 s (48.6 s in that batch). The
  shower route folds the nodes onto three bands; the muon batch does not.

Data bundles of LightHit <= 0.3.0a4 hold tables for nine wavelength nodes,
which lost 6 % of the scattered light at 50 m and 15-30 % beyond 80 m
(`accuracy.md`, Wavelength quadrature). The routes refuse such a bundle with
the default `KernelConfig` (32 nodes); `KernelConfig(wavelength_nodes=9)`
reads it and reproduces the old results.

The water table of the `bgvd_model` package (used by NTSim and by the bundle
model alike) places its 26 absorption and scattering values on
`linspace(350, 610, 26)`, a 10.4 nm step. If the values belong to 350-600 nm
in 10 nm steps, the water is more transparent at a given wavelength than both
programs assume: about 2 % more detected light at 20 m and 3 % at 50-100 m.

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
