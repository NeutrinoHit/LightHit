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
