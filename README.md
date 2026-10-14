# LightHit

LightHit computes the expected number of photoelectrons that Cherenkov light
of charged particles (or a calibration flash) produces in every optical module
(OM) of a neutrino telescope, per time bin. It solves spectral radiative
transfer in a homogeneous scattering medium and keeps three collision orders
separate: direct light (order 0), light scattered once (order 1) and light
scattered two or more times (order >=2). It replaces photon-by-photon
Monte-Carlo propagation; the charged particles themselves come from Geant4
(e.g. through NTSim).

| Source | Route | Orders 0 and 1 | Order >=2 |
|---|---|---|---|
| shower (e, gamma, hadrons; nu_e CC, NC) | `lighthit.shower.ShowerTransport` | exact direct light on the G4 steps over the continuous spectrum; once-scattered light from grouped "pencils", all wavelength bands fused | axial shower basis on band-folded RTE kernels |
| muon with its secondaries | `lighthit.muon.MuonTransport` | exact direct light; primary from finite-track atlases, secondaries grouped | atlases (primary) and point RTE sources (secondaries) |
| several sources (nu_mu CC, bundles, lists) | `lighthit.event.EventTransport` | each source by its route | each source by its route |
| isotropic calibration flash | `kernel.transport(IsotropicFlash)` | full directional RTE | full directional RTE |

All routes return the same response: `charge_pe` per OM (all arrival times),
`bins_pe` per OM and bin, `components_pe` split by order, `active` (OMs above
threshold) and `time_origin_ns` (the per-OM clock the bins refer to).

## Install

Python 3.11 or newer.

```bash
python -m pip install 'lighthit[accelerate]'   # numerical routes need Numba
lighthit-selftest                               # quick installed-package check
```

From a checkout (development):

```bash
python -m pip install -e '.[accelerate,viewer,test]'
python -m pytest -q tests
```

The first run of each route compiles its Numba kernels (a few minutes, once
per installation; the compiled code is cached on disk).

## 1. A complete example with public inputs

Everything below is artificial (a Baikal-like water and an 8-string, 48-OM
array shipped with the package), runs in seconds and needs no private data.
It prepares the tables, then transports one shower, two showers in one event
and a calibration flash.

```python
from dataclasses import replace
import numpy as np
import lighthit as lh
from lighthit.examples.synthetic import synthetic_medium_model, baikal_like_detector
from lighthit.shower import ShowerData, ShowerTransport
from lighthit.event import EventTransport, SourceComponent

medium = synthetic_medium_model()            # artificial Baikal-like water
detector = baikal_like_detector(clusters=1)  # 8 strings x 6 OMs, artificial
config = lh.KernelConfig(
    omega_per_ns=np.linspace(0.0, 1.2, 65),               # RTE frequencies
    readout_omega_per_ns=np.linspace(0.0, 1.2, 321),      # readout grid
    relative_time_edges_ns=np.arange(-60.0, 745.0, 5.0),  # 5 ns bins
    wavelength_nodes=3, scattering_degree=8, source_degree=8, azimuthal_degree=2,
    radial_range_m=(2.0, 150.0), radial_nodes=48,
    k_max_per_m=4.0, k_panel_per_m=0.1, k_order=6,
    threshold_pe=0.01, cache_directory="demo-cache", cache_policy="build")
kernel = lh.TransportKernel(medium, detector, config)

centre = lh.describe_geometry(detector).centroid_m
def shower_at(offset, direction, seed):
    pose = lh.SourcePose(centre + np.asarray(offset, float), direction)
    shower = lh.SyntheticShower.gaussian(pose, charged_track_length_m=300.0,
                                         elements=2000, seed=seed)
    return replace(shower.elements, cone_model="spectral")   # per-wavelength cone
first = shower_at([-12.0, 8.0, 0.0], [0.35, -0.15, 0.925], seed=7)
second = shower_at([20.0, -15.0, 30.0], [-0.30, 0.20, 0.93], seed=8)

# Once: the RTE cache and the >=2 kernel pack (reused by every later event).
kernel.build(source=first)
kernel.ge2_kernel_store().save("demo-cache/ge2-pack")
data = ShowerData.of("demo-cache", "demo-cache/ge2-pack")

# 1. One shower.
response = ShowerTransport(kernel, data)(first)
print("shower", int(response.active.sum()), "active OMs,",
      response.charge_components_pe.sum(axis=0).round(2), "pe in orders 0, 1, >=2")

# 2. Two showers in one event: one clock, OMs selected on the summed charge.
event = EventTransport(kernel, shower_data=data)
both = event([SourceComponent("shower", first, label="A"),
              SourceComponent("shower", second, label="B")])
print("event", int(both.active.sum()), "active OMs; completed per source:",
      both.metadata["completion_rounds"][0]["recomputed"])

# 3. An isotropic calibration flash (its own monochromatic table, built once).
flash = lh.IsotropicFlash.monochromatic(centre + [0.0, 0.0, 30.0], 1e10, 450.0)
kernel.build(source=flash)
laser = EventTransport(kernel, laser_kernel=kernel)(flash)
print("laser", int(laser.active.sum()), "active OMs,", round(float(laser.charge_pe.sum()), 1), "pe")

# 4. Orders 0 and 1 only, without any RTE table. The default PromptConfig sums
#    direct light over the wavelength nodes and groups each band separately;
#    the shower route uses the continuous spectrum and fused bands instead.
prompt = kernel.transport_prompt(first)
print("prompt", prompt.charge_orders_pe.sum(axis=0).round(2))

# Keep a response and look at it in the browser.
response.save("shower-response.npz")
lh.write_event_viewer(lh.viewer_payload(response, source=first, event_id="shower"),
                      "viewer.html")
```

Expected output (the last digits may differ between platforms):

```text
shower 34 active OMs, [66.49 50.68 22.5 ] pe in orders 0, 1, >=2
event 33 active OMs; completed per source: {'A': 14, 'B': 15}
laser 48 active OMs, 153762.2 pe
prompt [67.96 50.31]
```

`lighthit-demo` (geometry, cache, run) is the same synthetic array as a
command-line tool.

## 2. Baikal-GVD, stand-alone

### Inputs

* **Medium and detector.** `lh.load_bgvd_model(path, dataset="2021")` reads
  the water, OM response and geometry from the separately supplied
  `bgvd-model` package (not distributed). Inside NTSim the same objects come
  from `ntsim.LightTransport.bgvd_model_from_ntsim` (section 3).
  `model.save("bgvd-model.npz")` stores a loaded model in one file (arrays and
  response tables, no code) and `lh.load_model("bgvd-model.npz")` reads it back
  without the package; the internal data bundle `lighthit-bgvd-data` ships its
  model this way (`model/bgvd-model.npz`).
* **Charged steps.** `lh.SpectralLightElements` (start, direction, chord and
  true length, beta, times, UID per step) or `lh.G4Shower.from_hdf5`.
* **Prepared tables**, built once per medium and OM response and reused: the
  RTE cache (65 frequencies to 1.2 rad/ns, 9 wavelengths), the >=2 kernel
  pack and, for muons, the primary-track atlases. A manifest next to the cache
  lists them: `shower-data.json` (`rte_cache`, `ge2_pack`) or
  `muon-data.json` (those plus the muon atlases), so the routes find them from
  the kernel alone. The muon atlases are prepared by the LightHit research
  pipeline and supplied together with the tables.
* **Tables follow the physics, not the labels.** A table is found by the
  water coefficients at its wavelength, `g`, the solver settings and the grid;
  provenance strings play no part. The bundle's tables therefore serve
  `load_model`, `load_bgvd_model` and NTSim's model alike. Tables written by
  0.3.0a1 (whose names included the provenance) are still read: by their old
  name, or by the coefficients stored inside them. The laser table also
  depends on the detector extent (its radial range), so it is shared only
  between models with the same geometry.

```python
import json
import numpy as np
import lighthit as lh

model = lh.load_bgvd_model("bgvd-model/bgvd_model", dataset="2021")
config = lh.KernelConfig(
    omega_per_ns=np.linspace(0.0, 1.2, 65),
    readout_omega_per_ns=np.linspace(0.0, 1.2, 321),
    relative_time_edges_ns=np.arange(-60.0, 745.0, 5.0),
    cache_directory="tables/lighthit-cache", cache_policy="require")
kernel = model.kernel(config)

# Once (tens of minutes): build with cache_policy="build", then save the >=2
# pack and the manifest. Every event job afterwards uses cache_policy="require".
#   kernel.build(source=some_shower)
#   kernel.ge2_kernel_store().save("tables/ge2-pack")
#   json.dump({"rte_cache": "lighthit-cache", "ge2_pack": "ge2-pack"},
#             open("tables/shower-data.json", "w"))
```

### A shower from Geant4

```python
from dataclasses import replace
from lighthit.shower import ShowerTransport

phase_max = float(np.max(kernel.medium.sample(kernel.wavelength.wavelength_nm)["phase_index"]))
g4 = lh.G4Shower.from_hdf5("sim_e_100GeV_10.h5", event=5, reference_phase_index=phase_max)
electron = replace(g4.elements, cone_model="spectral").placed(
    lh.SourcePose(position_m=[5.0, -269.5, 330.0], direction=[0.35, -0.20, 0.915]))

shower = ShowerTransport(kernel)   # finds shower-data.json / muon-data.json next to the cache
response = shower(electron)
print(response.metadata["route"], response.active.sum(), response.charge_components_pe.sum(axis=0))
```

`reference_phase_index=phase_max` keeps every step that emits at any of the
wavelengths; the spectral cone then applies each band's own threshold.
`placed` moves the shower's light centroid and axis to the given pose.

### A muon with its secondaries

Besides the steps, the muon route needs the family table of the secondaries:
which step descends from which first-generation secondary of the muon
(`element_family`, aligned with the steps) and the family roots (`uid`,
`pdgid`, `energy_GeV`, `start_m`, `start_ns`, `creator`). NTSim builds it with
`ntsim.LightTransport.LightHitTransport.muon_ancestry`.

```python
with np.load("muon1000.npz") as f:
    muon_steps = lh.SpectralLightElements(**{k: f[k] for k in f.files if k != "metadata"},
                                          cone_model="spectral")
with np.load("muon1000-ancestry.npz") as f:
    ancestry = {k: f[k] for k in f.files}

muon = lh.MuonTransport(kernel)    # tables from muon-data.json next to the cache
response = muon(muon_steps, ancestry, primary_uid=0)
```

### Several sources in one event

```python
from lighthit.event import EventTransport, SourceComponent

event = EventTransport(kernel)
response = event([
    SourceComponent("muon", muon_steps, ancestry, primary_uid=0, label="muon"),
    SourceComponent("shower", electron, label="vertex shower"),
])
print(response.active.sum(), response.metadata["completion_rounds"])
for part in response.metadata["event_components"]:  # per-source summary
    print(part["label"], part["kind"], round(part["charge_pe"], 2), "pe")
```

### Calibration laser

An isotropic monochromatic point flash, all orders through the directional
RTE. It needs its own point-source cache for that wavelength (built once;
slow for the full array):

```python
from lighthit.event import laser_kernel

laser = laser_kernel(model.medium, model.detector, cache_directory="tables/laser-cache",
                     omega_per_ns=np.linspace(0.0, 1.2, 321), cache_policy="build")
flash = lh.IsotropicFlash.monochromatic([0.0, -270.0, 360.0], 1e12, 532.0)
response = EventTransport(kernel, laser_kernel=laser)(flash)
```

## 3. Inside NTSim

NTSim keeps primary generation and Geant4 propagation and calls LightHit
instead of photon Monte Carlo (`enable_cherenkov=False`). With the prepared
tables next to `--cache`, every event is split into its light sources and
routed automatically; no option is needed.

| Event | LightHit sources | Route |
|---|---|---|
| muon | the muon with all its descendants | `lighthit.muon` |
| electron, photon, hadron | one shower | `lighthit.shower` |
| nu_e CC, NC (particles at one vertex) | one shower | `lighthit.shower` |
| nu_mu CC | the muon (with descendants) + the hadronic vertex shower | `lighthit.event` |
| list of primaries, muon bundle | each muon of at least 20 m; showers grouped by vertex | `lighthit.event` |
| calibration laser | isotropic flash, no Geant4 | `kernel.transport(IsotropicFlash)` |

Command line (`run/run_lighthit.py` in NTSim; `CACHE` is the prepared RTE
cache with `muon-data.json` beside it):

```bash
# electron (shower route)
python run/run_lighthit.py 1 --cache "$CACHE" --clusters 1 2 3 4 5 \
  --pdg 11 --energy 100 --position 5 -269.5 330 --direction 0.35 -0.2 0.915 --output out/electron

# muon (muon route)
python run/run_lighthit.py 1 --cache "$CACHE" --clusters 1 2 3 4 5 \
  --pdg 13 --energy 1000 --position 5 -269.5 330 --direction 0.2 0.9 0.3 --output out/muon

# nu_e CC final state at one vertex: one shower
python run/run_lighthit.py 1 --cache "$CACHE" --clusters 1 2 3 4 5 --position 5 -269.5 330 \
  --particle 11 70 0.35 -0.2 0.915 --particle 211 25 0.45 -0.1 0.88 --particle 2212 5 0.2 -0.3 0.93 \
  --output out/nue

# nu_mu CC final state: muon + hadronic vertex shower, one event
python run/run_lighthit.py 1 --cache "$CACHE" --clusters 1 2 3 4 5 --position 5 -269.5 330 \
  --particle 13 800 0.2 0.9 0.3 --particle 211 150 0.3 0.85 0.4 --particle 2212 50 0.1 0.95 0.2 \
  --output out/numu

# calibration flash (the laser cache is built once with --build-laser-cache)
python run/run_lighthit.py 1 --cache "$CACHE" --clusters 1 2 3 4 5 --position 0 -270 360 \
  --laser 1e12 532 --laser-cache tables/laser-cache --output out/laser
```

Each run writes `event-<id>-hybrid-<route>.npz` (the response),
`event-<id>-hybrid-<route>.json` (the metadata below) and `viewer.html`.
`--muon-route off`, `--shower-route off` or any `--hybrid-*` option keep the
older hybrid route.

In Python, the same processor in an NTSim chain; `ParticleList` places any set
of primaries (e.g. a neutrino final state) at their vertices:

```python
import numpy as np
import lighthit as lh
import ntsim
from ntsim.event_chain import Chain, EmptyEventSource
from ntsim.LightTransport import LightHitTransport, bgvd_model_from_ntsim

model = bgvd_model_from_ntsim(ntsim.bgvd.Telescope(n_clusters=(1, 2, 3, 4, 5)), ntsim.bgvd.Water())
kernel = model.kernel(lh.KernelConfig(
    omega_per_ns=np.linspace(0.0, 1.2, 65), readout_omega_per_ns=np.linspace(0.0, 1.2, 321),
    relative_time_edges_ns=np.arange(-60.0, 745.0, 5.0),
    cache_directory=CACHE, cache_policy="require"))

primaries = ntsim.primary.ParticleList(                 # nu_e CC: electron + hadrons
    pdg=(11, 211, 2212), energy=(70.0, 25.0, 5.0),
    direction=((0.35, -0.20, 0.915), (0.45, -0.10, 0.88), (0.20, -0.30, 0.93)),
    position=((5.0, -269.5, 330.0),))                   # one vertex, or one per particle
geant = ntsim.modules.ParticlePropagator(random_seed=42, enable_cherenkov=False,
                                         cascade_bounds_e=(0.0, 0.0), save_process_name=True)
light = LightHitTransport(kernel=kernel, output_directory="ntsim-out")

for event in Chain(primaries, geant, light)(EmptyEventSource()):
    info = event.header.metadata["lighthit"]      # route, components, active OMs, ...
    response = light.responses[event.id]          # charge_pe, bins_pe, active, ...
    print(info["route"], info["active_modules"], info["total_expected_pe"])
    break
```

`event.header.metadata["lighthit"]` records the route (`lighthit.shower`,
`lighthit.muon`, `lighthit.event`), the sources found in the event
(`components`: kind, label, primaries, number of steps, light fraction), the
active OMs, the total expected photoelectrons and any route substitution
(`route_fallbacks`). `LightHitTransport(kernel, laser_kernel=...).laser(position,
photons, wavelength)` returns a flash response without Geant4.

## Reading a response

```python
response.charge_pe             # (OM,)          all-time expected photoelectrons
response.bins_pe               # (OM, bin)      photoelectrons in each bin
response.components_pe         # (OM, bin, 3)   the same split into orders 0, 1, >=2
response.charge_components_pe  # (OM, 3)
response.active                # (OM,)          total charge >= threshold
response.time_origin_ns        # (OM,)          absolute time of each OM's bin zero
response.relative_time_edges_ns
response.metadata              # route, timings, selection details
response.save("event.npz")
```

Bins are defined relative to each OM's origin, a lower bound on the earliest
possible light arrival. Bin sums can differ from the charge (light outside the
window). Order >=2 comes from a finite frequency band and may show small
signed ringing; it is never clipped. OMs below threshold carry estimated
scattered charges and no bins.

## Interactive viewer

A response becomes one offline HTML file (WebGL2, no network access, no Python
plotting dependency):

```python
payload = lh.viewer_payload(response, source=first, event_id="shower",
                            label="synthetic shower")
lh.write_event_viewer(payload, "lighthit-viewer.html")

# Compatible payloads share one viewer.
combined = lh.merge_event_viewers(laser_payload, track_payload, shower_payload)
lh.write_event_viewer(combined, "multi-event-viewer.html")
```

Open `lighthit-viewer.html` in a current Chrome, Edge or Safari. A portable
`lighthit-viewer.json` companion is written beside it.

- **Scene.** Every OM glows with its expected photoelectron response,
  accumulated on the detector clock (`time_origin + relative time`, uniform
  within each stored bin). Colour shows the first-light time, the mixture of
  scattering orders 0 / 1 / ≥2, or the charge; size grows with charge and a
  flash marks the arrival rate. The glow is a mean response, not sampled hits.
- **Source and light front.** A timed track shows a moving head and its
  Cherenkov front: light leaves at the Cherenkov angle for `n = 1.34` and travels
  at `c/n_g` (`medium.group_index` when present, else 1.36), so a β = 1 track
  carries a cone behind its head. Showers and point sources show an expanding
  sphere; without stored timing their emission time is estimated from
  direct-light arrivals. The front and its fade length are illustrative.
- **Analysis.** The Module tab shows the selected OM's signed waveform by order
  and a time map of all lit OMs; the 3 ns Gaussian display redistributes binned
  integrals under a uniform-within-bin assumption and never changes stored data.
- **Movies and stills.** The Movie tab renders an H.264 MP4 frame by frame
  (WebCodecs; HD to 4K, square and vertical, 24–60 fps) with an opening hold,
  an event clock that can slow down where light arrives (the on-screen clock
  always shows true detector time), a closing orbit and camera keyframes. Still
  images are saved as PNG up to 4K. Keys: Space, ← →, R, 0–4, P, F, S.

The source timing is an optional `timing` block of an axis source
(`lighthit.viewer.source_timing` builds it from light elements); payloads
without it stay valid. The previous Plotly viewer remains available as
`lh.write_event_viewer(payload, path, legacy=True)` and needs
`python -m pip install 'lighthit[viewer]'`.

## Several sources: one clock, one OM acceptance

Light adds up, so an event is the sum of its sources, but two things are
decided for the whole event:

* every OM has one time origin (the earliest possible arrival from any step of
  the event), and the bins of all sources refer to it;
* an OM is active when the **sum** of the sources reaches the threshold. Each
  source is computed with the light of the already computed sources added to
  its own threshold test (showers first). Then every OM whose summed charge
  can still reach the threshold (estimated scattered charges inflated by a
  factor 3) is completed in every source that did not compute it in detail.

An OM seen weakly by the muon and weakly by the vertex shower is therefore
active with both waveforms. On test events the composed response equals,
bit for bit, the sum of independent threshold-free calculations of each
source.

## Accuracy and speed

Measured on NTSim/Geant4 events in five Baikal-GVD clusters (1440 OMs), one
CPU thread, after compilation. Errors are relative to calculations on the
original G4 steps or to a finer source basis, not to photon Monte Carlo.

| Event | Time | Previous route |
|---|---:|---:|
| electron 100 GeV | 5.5–6.4 s | 25–26 s |
| electron 1 TeV | 42 s | 105 s |
| pi+ 100 GeV | 21 s | — |
| nu_e CC (e 70 + pi 25 + p 5 GeV) | 14 s | — |
| muon 100 GeV / 1 TeV | 11 / 10 s | — |
| nu_mu CC (mu 800 + pi 150 + p 50 GeV) | 73 s | — |

* Direct light is exact on the G4 steps over the continuous spectrum. The
  earlier nine-wavelength quadrature underestimated it by about 9% for showers
  (up to 100% on single OMs), because a millimetre step is seen only on its
  Cherenkov cone.
* Once-scattered light (grouped) against original-step integration: bin L1
  at most 3.3% (100 GeV) and 3.1% (1 TeV), charge bias below 0.6%. This is the
  largest numerical error of the shower calculation.
* Order >=2 against a finer basis: at most 0.15% bin L1 on 100 GeV electrons,
  0.4% at 1 TeV and 1% on a hadronic shower.
* Hadronic showers emit about 3% of their light 50 ns to milliseconds late
  (neutron captures, decays). That light is read on its own shifted clock, or
  counted only in the charge when it cannot reach the readout window.

## Limitations

* Point OMs in a homogeneous medium with an HG phase function: no
  boundaries, shadowing, electronics, noise or trigger.
* The muon route refuses a primary-track piece whose line passes within 3 m
  of an OM, which happens at the scattered end of stopping muons; such a muon
  is then transported by the shower route (`route_fallbacks`), correctly but
  more slowly.
* The laser is an isotropic flash; a collimated beam or a measured diffuser
  pattern is not represented.
* Tables (RTE cache, >=2 pack, muon atlases) belong to one medium and OM
  response; a change of either requires new tables.

## Release

The version is defined only in `src/lighthit/_version.py`.

```bash
python -m pytest -q tests
LH_VERSION="$(python -c 'from lighthit._version import VERSION; print(VERSION)')"
rm -rf dist build
python -m build --sdist --wheel --outdir dist
python -m twine check "dist/lighthit-${LH_VERSION}-py3-none-any.whl" "dist/lighthit-${LH_VERSION}.tar.gz"
git tag -a "v${LH_VERSION}" -m "LightHit ${LH_VERSION}"
python -m twine upload -u __token__ "dist/lighthit-${LH_VERSION}-py3-none-any.whl" "dist/lighthit-${LH_VERSION}.tar.gz"
```

The distribution contains the package, its public synthetic example and the
self-test only: no G4 data, detector model, prepared tables, research scripts
or documentation sources (`MANIFEST.in` and `packaging_filter.py`).

## Licence and data

BSD 3-Clause, see `LICENSE`. The free-moment recursion and free-tail ratios in
`angular.py` derive from the author's earlier `baikal-rte` code. Optical
coefficients, the OM response and geometry are read at run time from a
separately supplied model; no private code, calibration table, G4 event or
prepared table is distributed. Synthetic media and detectors in this package
are artificial test inputs, not a calibration.
