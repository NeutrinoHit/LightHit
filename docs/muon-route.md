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
`<cache>/../muon-data.json`. All seven keys are required; paths are relative to
the manifest. `MuonData.load` checks that they exist.

| Key | Content | Used for |
|---|---|---|
| `rte_cache` | RTE tables, 65 frequencies | `muon_kernel` (policy "require") |
| `ge2_pack` | `Ge2KernelStore` pack | secondaries order >=2; direct fallback of the primary |
| `primary_first_atlas` | finite-track order-1 atlas (beta = 1), `TrackFirstAtlasStore` | primary order 1 |
| `primary_ge2_atlas` | finite-track order >=2 atlas (beta = 1), `DiskTrackGe2Atlas` | primary order >=2 |
| `primary_first_beta` | beta < 1 correction of the order-1 atlas | primary order 1, slow pieces |
| `primary_ge2_beta` | beta < 1 Taylor modes of the order >=2 charge | primary order >=2, slow pieces |
| `preview_atlas` | Monte-Carlo positive fluence atlas | OM preselection |

The atlases are built once per medium and OM response by build functions in
`muon/track_first_primitive.py`, `muon/track_ge2_primitive.py`,
`muon/first_charge_beta_atlas.py`, `muon/ge2_charge_beta_atlas.py` and
`muon/positive_fluence.py`; the driver scripts that ran them are not part of
the repository. The prepared atlases are shipped in the data bundle
(`sources-and-models.md`).

## One event (`muon/joint_muon.py: transport_joint_muon`)

1. **Time origin** per OM: `earliest_element_arrival` over all steps at the
   fastest group speed, or the caller's `time_origin_ns`.
2. **Order 0 for all OMs**: `ballistic_interval` on all original steps.
3. **Preview** (`muon/preview.py`): steps are grouped (4 m cells, direction and
   beta classes) and the positive fluence atlas gives a cheap order 1 and >=2
   estimate per OM. It is an estimate, not a bound.
4. **OM selection** (`muon/adaptive_screen.py`): score
   `q0 + margin * (q1 + q2)` plus light of other sources
   (`selection_extra_charge`); one sentinel OM per string; after each fine round
   clusters with surprises are reopened and the margin recalibrated; at most
   three rounds, then all wanted OMs. `MuonTransport` uses `scatter_margin=2`.
5. **Fine evaluation** of the selected OMs (threshold 0 on a subset kernel):
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
       panels (`muon/filon_ge2.py`).
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
       pencils are many (`prompt._grouped_blocks`, 1 GiB of scratch).
     * Order >=2: each packet becomes sparse joint source cells
       (`muon/joint_source.py`) applied to the pack with BLAS in float32
       batches (`muon/point_ge2_blas.py`); weak packet-OM pairs are predicted
       and omitted (prediction, not a bound).
6. Readout from 65 to 321 frequencies and inversion to bins.

The response has `metadata["route"] = "lighthit.muon"` and
`method = "joint-muon"`. OMs that were not evaluated in detail keep the preview
order 1 and >=2 charges and have no bins.

## Unsupported muons

`transport_universal_primary` raises `ValueError("primary enters unvalidated
inner >=2 radius ...")` when a primary piece passes through an OM or within
3.01 m of it (typical for the end of a stopping muon). `MuonTransport` does not
catch it; `EventTransport` does and re-runs that source with the shower route
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
