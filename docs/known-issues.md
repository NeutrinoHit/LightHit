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
* Muon route, found by measurement (registry ideas `default-muon-route`,
  `default-event-route`): `transport_universal_primary` raises "primary enters
  unvalidated inner >=2 radius" when the *infinite line* of a primary polyline
  piece passes within 3.01 m of an OM inside the order >=2 reach (`rho` is the
  distance to the line, not to the piece). The tail of a stopping muon is cut
  into pieces of a few centimetres with scattered directions, so ordinary
  stopping muons trigger it: the regenerated 100 GeV muon, which stays at least
  93 m from every OM, raises because the lines of three tail pieces of 0-6 cm
  pass 0.96-1.44 m from OMs 146-341 m away. `MuonTransport` fails;
  `EventTransport` computes the muon with the shower route instead.
* `TransportKernel.transport` reads out on the table frequencies
  (`readout_omega_per_ns` is used only by the hybrid and shower routes). With
  the 65 route frequencies up to 1.2 rad/ns the waveform repeats every 335 ns,
  so the 805 ns window of the route configuration holds three copies: the bin
  sums are three times the charges, which are correct (registry idea
  `kernel-transport-full`, 100 GeV electron). Nothing warns about a window
  longer than `2 pi / d_omega`.
* The muon->shower fallback cannot help a muon that really passes within 3 m of
  an OM: the shower route then raises "source points closer than the validated
  radial minimum 3 m" (regenerated 6 GeV muon passing an OM at 1.76 m), and the
  event has no result.
* Muon route: order >=2 charge and time profile of the same pair can come from
  different methods (atlas or batch charge, direct Filon profile), so a bin sum
  need not equal the charge column; near-field errors other than "unvalidated
  inner" are not turned into a shower fallback.
* `G4Shower.from_hdf5` drops steps below threshold at `reference_phase_index`
  (default 1.35); pass the medium's largest phase index.
* `ShowerData.load` does not check that `ge2_band_fold` exists; a missing fold
  is silently recomputed in memory.
* `lighthit.muon.preview.check_atlas_optics` does not compare the HG `g`.

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
