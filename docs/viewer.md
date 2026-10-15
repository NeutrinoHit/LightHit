# Event viewer (`viewer.py`, `assets/`)

`lh.viewer_payload(response, *, source=None, event_id="event", label=None,
detector_label="detector")` turns a response into a portable JSON payload
(schema `lighthit/event-viewer/1`): OM positions and identifiers, bin edges,
per-OM time origins, active flags, binned components by order and charges, the
response metadata, and the sources. `lh.write_event_viewer(payload, path,
*, json_path=None, legacy=False)` writes one offline HTML file and a JSON
companion; `lh.merge_event_viewers(*payloads)` combines compatible events into
one viewer.

## Sources shown

`source=` takes one object or a list:

| Source | Shown as |
|---|---|
| `IsotropicFlash` | a point at its position, light sphere from its time |
| `CherenkovTrack` | a straight track with a moving head and timing |
| muon `SourceComponent` | a straight track fitted to the steps of its `primary_uid`, with timing |
| shower `SourceComponent`, `SpectralLightElements`, `G4Shower`, `SyntheticShower` | a spindle along the light-weighted principal axis, spanning the central 99% of the light, with timing |

A response with a muon `track_uid` in its metadata and no explicit muon source
shows that muon as a track. Timing is a weighted straight-line fit of the
steps' emission times (`viewer.source_timing`).

## The page

`write_event_viewer` inlines `assets/studio.html`, `studio.css`,
`studio_app.js` (interface, scene, panels), `studio_gl.js` (three.js renderer:
HDR scene, bloom, tone mapping), `studio_model.js` (display model on the
detector clock), `studio_movie.js` (WebCodecs H.264 encoder and MP4 muxer),
`event_viewer_smoothing.js` (Gaussian readout display) and
`vendor/three.min.js` (three.js r160 with OrbitControls and post-processing,
MIT notice in the file and in `vendor/three.LICENSE`; rebuilt by
`scripts/build_three_vendor.sh`). The page needs no network access.

* OMs glow with the expected photoelectrons accumulated on the detector clock
  (uniform within each stored bin); colour by first-light time, order mixture
  or charge.
* Tracks show a Cherenkov front at `n = 1.34` travelling at `c/n_g`; showers
  and flashes an expanding sphere; showers also their axis and an arrow. The
  front is illustrative.
* Module tab: the signed waveform of one OM by order; Movie tab: frame-exact
  MP4 rendering; PNG stills.

`legacy=True` writes the previous Plotly viewer (`assets/legacy/`, needs
`lighthit[viewer]`).

## Tests

`tests/test_viewer.py`, `test_viewer_studio.py`, `test_viewer_movie.py`,
`test_viewer_video_export.py` (Node.js and ffmpeg parts skip when the tools are
missing).
