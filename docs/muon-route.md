# Muon route (`lighthit.muon`)

`MuonTransport(kernel, data=None, *, config=None, screen=None, threads=None)`
returns the response of one muon with all its descendants:
`muon(steps, ancestry, primary_uid=0)`. Exported from `lighthit`: `MuonTransport`,
`MuonData` (loaded lazily, they need Numba). `lighthit.muon` also exports
`MuonConfig` (= `JointMuonConfig`), `transport_muon`, `muon_kernel`,
`independent_preview`, `write_manifest`, `FREQUENCIES_PER_NS`, `READOUT_PER_NS`.

## Inputs

* `steps`: `SpectralLightElements` with `cone_model="spectral"`, all charged
  steps of the muon and its secondaries, `coefficient2 = -coefficient0/beta^2`.
* `ancestry`: a dict with `element_family` (one entry per step: the index of
  the first-generation secondary it descends from, or the primary) and the
  family roots: `uid`, `pdgid`, `energy_GeV`, `start_m`, `start_ns`, optional
  `creator`. NTSim builds it with `ntsim.LightTransport.LightHitTransport.muon_ancestry`.
* `primary_uid`: the UID of the muon's own steps.

## Tables: `muon-data.json`

Found like the shower manifest: `$LIGHTHIT_MUON_DATA`, `<cache>/muon-data.json`,
`<cache>/../muon-data.json`. The first seven keys of the table are required,
`ge2_band_fold` is optional (`MuonData.ge2_band_fold`); paths are
relative to the manifest. `MuonData.load` checks that every named table
exists.

| Key | Content | Used for |
|---|---|---|
| `rte_cache` | RTE tables, 65 frequencies | `muon_kernel` (policy "require") |
| `ge2_pack` | `Ge2KernelStore` pack | secondaries order >=2; direct fallback of the primary |
| `primary_first_atlas` | finite-track order-1 atlas (beta = 1), `TrackFirstAtlasStore` | primary order 1 |
| `primary_ge2_atlas` | finite-track order >=2 atlas (beta = 1), `DiskTrackGe2Atlas` | primary order >=2 |
| `primary_first_beta` | beta < 1 correction of the order-1 atlas | primary order 1, slow pieces |
| `primary_ge2_beta` | beta < 1 Taylor modes of the order >=2 charge | primary order >=2, slow pieces |
| `preview_atlas` | Monte-Carlo positive fluence atlas | OM preselection |
| `ge2_band_fold` (optional) | the >=2 pack folded onto 3 node bands (the shower route's fold) | direct >=2 fallback of the primary, nearly relativistic pieces |

The atlases are built once per medium and OM response by build functions in
`muon/track_first_primitive.py`, `muon/track_ge2_primitive.py`,
`muon/first_charge_beta_atlas.py`, `muon/ge2_charge_beta_atlas.py` and
`muon/positive_fluence.py`; the driver scripts are in `tools/tables/` (build
order, arguments and what each bundle file records: `tools/tables/README.md`). The prepared atlases are shipped in the data bundle
(`sources-and-models.md`). The order-1 and order >=2 atlases store their
optical contract (spectral tables, acceptance, area, medium samples) and its
SHA-256 fingerprint; a kernel is accepted when its contract has that
fingerprint or agrees within relative 1e-9 with the stored values that have
it (`muon/optics_contract.py: contract_matches`), because another platform
reproduces the interpolated values only to the last bits. The beta tables
check their base atlas, the preview atlas its optical tables within relative
1e-9 (`muon/preview.py: check_atlas_optics`).

## One event (`muon/joint_muon.py: transport_joint_muon`)

1. **Time origin** per OM: `earliest_element_arrival` over all steps at the
   fastest group speed, or the caller's `time_origin_ns`.
2. **Order 0 for all OMs**: prepared first-entry charge on all original
   steps with the supplied OM sphere at every distance. Extinction and the
   optical clock end at the actual entry. Bins are deferred until selection
   and evaluated once per requested batch. A detector without surface
   geometry retains the mathematical point model.
3. **Preview** (`muon/preview.py`): steps are grouped (4 m cells, direction and
   beta classes) and the positive fluence atlas gives a cheap order 1 and >=2
   estimate per OM. It is an estimate, not a bound.
4. **OM selection** (`muon/adaptive_screen.py: adaptive_candidates`): score
   `q0 + margin * (q1 + q2)` plus light of other sources
   (`selection_extra_charge`), with the preview's scattered charges at the
   maximum of the acceptance; one sentinel OM per string; after each fine
   round clusters with surprises are reopened and the margin recalibrated.
   Rounds 0-2 evaluate the newly wanted OMs; in round 3 (`max_rounds`), or at
   threshold 0, every OM of the detector is evaluated if any wanted OM is
   still left (`wanted[:] = True`). `MuonTransport` uses `scatter_margin=2`.
   An optional final complete audit (`audit_all_rejected=True`) evaluates
   the remaining OMs; it is disabled by default. Unselected OMs keep the
   preview-based scattered-charge estimate described below.
   The threshold of these decisions is `KernelConfig.detail_threshold_pe`
   when it is set below `threshold_pe` (Detail threshold, below).
5. **Fine evaluation** of every OM in the scheduled batches (threshold 0 on a subset kernel):
   * **Primary** (`muon/universal_primary.py`): fitted by a connected polyline
     (`muon/primary_polyline.py`, chords up to 250 m with tolerances on
     position, time, direction and beta).
     * Order 1: the finite-track atlas where the piece is inside its radii and
       has beta = 1; remaining pieces are ranked by analytic upper bounds
       (`muon/track_first_bounds.py`) and dropped while the cumulative bound
       stays <= 1e-6 pe; the rest from the beta atlas or the direct segment
       integral. Time bins from the atlas, or from `order1_integrals`.
     * Order >=2: the disk atlas for beta = 1 pieces in its validated range;
       otherwise the beta atlas, a batched axial-basis evaluation of slow pieces
       (`muon/ge2_beta_response.py`) or the direct pack evaluation with Filon
       panels (`muon/filon_ge2.py: backward_ge2_filon_many`). With a prepared
       fold (`ge2_band_fold`) the Filon batch (`universal_primary.py:
       DirectGe2.many`) applies the folded kernels in the fold's 3 node bands
       (`spectral=`: their wavelengths, weights, phase indices and
       efficiencies) to every piece whose fold residual at its beta
       (`BandFold.residual`, worst over degrees) is at most `fold_tolerance`
       = 1e-3; the other pieces, and the single-piece charge evaluations, use
       every wavelength node. The fold is refused (`ValueError`) when its
       frequencies, degree or bands do not fit the pack and kernel or its
       design residual recomputed for this kernel differs from the stored one
       by more than relative 1e-9. Without a fold every piece uses every
       node. Accuracy and speed: measured in experiments/REPORT.md
       (photon-mc-matrix).
   * **Secondaries** (`muon/muon_packets.py`): steps are partitioned by family:
     delayed emission (> 10 ns after the root's light-cone time) into 20 ns
     packets; e+-/gamma above 0.1 GeV (not nuclear) into showers; softer EM into
     5 m x 20 ns packets; other particles separately.
     * Order 1 (`muon/secondary_first.py`): the grouped physical-time solver
       with fused bands; packets are skipped while a certified upper bound
       (`muon/packet_skip.py`) stays below 1e-3 of the seed packets' charge.
       Rings are compressed into pencils by ranges of whole position cells,
       so a large shower packet needs about the memory of its final pencil
       table; the per-OM evaluation runs fewer blocks at once when the
       pencils are many (`prompt._grouped_blocks`; scratch budget
       `PromptConfig.grouped_scratch_bytes`, by default the needed storage
       limited by available memory, `prompt._scratch_budget`).
     * Order >=2: each packet becomes sparse joint source cells
       (`muon/joint_source.py`) applied to the pack with BLAS in float32
       batches (`muon/point_ge2_blas.py`); weak packet-OM pairs are predicted
       and omitted (prediction, not a bound).
6. Readout from 65 to 321 frequencies and inversion to bins.

The response has `metadata["route"] = "lighthit.muon"` and
`method = "joint-muon"`. Order 0 (charge and bins) is computed for every OM
on its original steps with the finite-window approximation above.
The adaptive preview selects OMs for the detailed order 1 and >=2 calculation,
including bins. `metadata["fine_charge_mask"]` identifies those OMs. Other OMs
retain a charge estimate from the same grouped sources and the preview atlas's
arrival-direction moments (`muon/angular_preview.py`). Ten moments in the
local radial/axial frame contract with the cubic OM acceptance, including
tilted OMs. The polynomial contraction is exact for the mirror-symmetric
sampled ensemble; finite cells, beta interpolation, source grouping and
sampling remain approximations. No isotropic acceptance factor is applied.
The maximum-acceptance scalar preview still controls selection. Existing
scalar-only preview atlases need rebuilding with
`tools/tables/build_positive_fluence.py`; detailed primary atlases are reused.

## Detail threshold

`MuonTransport.__call__` runs inside `transport.py: detail_threshold`: when
`KernelConfig.detail_threshold_pe` is set below `threshold_pe`, the route
kernel's threshold is that value for the whole call, so the preview forcing,
the sentinels and the adaptive candidates evaluate every OM whose estimate
reaches the detail threshold. Afterwards `active` is narrowed to evaluated OMs
whose charge plus `selection_extra_charge` reaches `threshold_pe`, and the
metadata record `threshold_pe` and `detail_threshold_pe`. Evaluated OMs below
`threshold_pe` keep their full charges and bins with `active` false. A value
at or above `threshold_pe`, or None (default), changes nothing.

## Near passes

Only the part of a primary piece that comes closer to an OM than the 3 m radial
minimum of the order >=2 tables leaves the tables' sums: the distance is taken
to the piece itself, not to its line, so the scattered centimetre pieces at the
end of a stopping muon are computed as usual (the direct fallback accepts an
impact below 3 m as long as every point is outside). The near part of the
primary, the near secondary points (skipped by the point kernels) and the near
points of the beta < 1 tail (`transport_hybrid_basis(near_points="exclude")`)
are then added by the near-field clamp of `near_field.py`
(`transport-kernel.md`); the OMs are listed in `metadata["near_field"]` and a
`RuntimeWarning` names them. A primary line through the very centre of an OM
still raises `ValueError("primary enters unvalidated inner >=2 radius ...")`;
`EventTransport` then re-runs that source with the shower route
(`route_fallbacks`). Other near-field errors (`first_charge_only`,
`muon_integral`) are not caught.

## Tests

The building blocks are tested (`test_joint_muon`, `test_universal_primary`,
`test_adaptive_screen`, `test_packet_skip`, `test_secondary_first`,
`test_point_ge2*`, `test_track_*`, `test_filon_ge2`, `test_*_beta_atlas`,
`test_positive_fluence`, `test_primary_polyline`, `test_joint_source`,
`test_muon_integral`, `test_first_charge_only`). The public entry
(`MuonTransport`, `transport_muon`, `MuonData`) has no test.

Old import names `lighthit.experimental.<module>` of every `lighthit.muon`
module still work in the repository (aliases), not in the wheel.
