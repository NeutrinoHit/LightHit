# LightHit: how the code works

These pages describe the code of this repository as it is, module by module:
what each part computes, how, with which defaults, who calls it and what tests
it. Nothing here describes earlier designs. Where a statement is a measurement
rather than a property of the code, it says so (see `accuracy.md`).

The user guide with runnable examples is the top-level `README.md`. The
documentation is not part of the PyPI package.

| Page | Content |
|---|---|
| `transport-kernel.md` | `TransportKernel` and `KernelConfig`: tables, cache keys, `build()`, the order >=2 pack, `transport` / `transport_prompt` / `transport_hybrid`, responses, readout |
| `shower-route.md` | `lighthit.shower.ShowerTransport`, end to end |
| `muon-route.md` | `lighthit.muon.MuonTransport`, end to end, and `muon-data.json` |
| `event-and-laser.md` | `lighthit.event.EventTransport`: several sources on one clock, completion, fallbacks, the laser kernel |
| `sources-and-models.md` | source objects, Geant4 input, the BGVD model loaders and model file, manifests and the data bundle |
| `viewer.md` | the HTML event viewer and its payload |
| `experimental.md` | every module of `lighthit.experimental` with its status |
| `accuracy.md` | measured accuracy and speed, and what the tests check |
| `known-issues.md` | defects and inconsistencies found in the code, to fix |

## Layers

```text
sources.py, g4_source.py, hdf5_minimal.py      charged steps -> radiating elements
model.py, medium.py, bgvd.py, geometry.py      medium, detector, BGVD adapter
angular.py, cache.py, directional.py, green.py RTE in k-space -> cached tables
transport.py (+ _directional_numba, axial_*,  TransportKernel: tables, transport,
  _track_transport, _spectral_fold, readout)    prompt, hybrid
prompt.py, _prompt_numba.py, ballistic*.py     orders 0 and 1 in physical time
shower_basis.py, _shower_basis.py, band_fold   order >=2 source basis and packs
shower/                                        shower route
muon/                                          muon route
event.py                                       several sources, laser kernel
viewer.py, assets/                             HTML viewer
experimental/                                  research variants and references
```

The default routes are `ShowerTransport`, `MuonTransport`, `EventTransport`
(and its flash path through `TransportKernel.transport`). None of them imports
a module of `lighthit.experimental` with default settings.

## Requirements in practice

* Python >= 3.11, numpy, scipy.
* Numba (`lighthit[accelerate]`) is required by every route for charged
  sources: the default element path imports `_spectral_fold`, which imports
  `_directional_numba`, which needs Numba; `_transport_axial` raises without
  it. Without Numba only an isotropic flash and the single-band point solver
  (`PointGreenSolver`, the `lighthit` CLI) run.
* `h5py` is optional; without it `hdf5_minimal` reads the Geant4 HDF5 layout.
