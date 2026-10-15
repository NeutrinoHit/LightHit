# Sources, models and prepared data

## Source objects (`sources.py`, all exported from `lighthit`)

| Object | Content |
|---|---|
| `SourcePose(position_m, direction, time_ns=0)` | where to place a source |
| `IsotropicFlash.monochromatic(position_m, photons, wavelength_nm, *, time_ns=0)`, `.broadband(position_m, photons, spectral_shape, ...)` | isotropic point flash (laser) |
| `SpectralLightElements` | charged steps: start, direction, chord and true length, beta, start/end times, `coefficient0`, `coefficient2 <= 0`, `uid`, `row_index`, `cone_model` |
| `CherenkovTrack.centered(position_m, direction, length_m, *, beta=1.0, time_ns=0, reference_phase_index=1.35, cone_model="frozen")` | one straight track; `to_elements(step_m=0.5)` |
| `G4Shower.from_hdf5(path, event=0, *, reference_phase_index=1.35, max_elements=None)` | Geant4 steps of a stored event |
| `SyntheticShower.gaussian(pose, *, charged_track_length_m=100.0, longitudinal_sigma_m=1.2, transverse_sigma_m=0.08, elements=64, beta=0.999, reference_phase_index=1.35, seed=0)` | deterministic artificial cascade |

**Spectrum of a step.** `dN/dlambda = C0/lambda^2 + C2/(lambda^2 n^2)` with
`C2 = -C0/beta^2` for a charge-one particle; each band applies its own
Cherenkov threshold and cone `cos theta = 1/(beta n)`.

**Cone model.** `SpectralLightElements`, `CherenkovTrack` and `SyntheticShower`
default to `cone_model="frozen"` (one cone at `reference_phase_index`);
`G4Shower.from_hdf5` and `SyntheticShower.gaussian` have no `cone_model` argument. The shower, muon and event
routes require `cone_model="spectral"` and refuse other sources, so use
`dataclasses.replace(source.elements, cone_model="spectral")`. NTSim's
`source_from_event` produces spectral cones by default.

**Placement.** `placed(pose)` rotates the light-weighted principal axis onto
`pose.direction`, moves the light centroid to `pose.position_m` and shifts the
earliest start to `pose.time_ns`; the transform is recorded in `provenance`.

**Reading Geant4 steps.** `G4Shower.from_hdf5` reads `event_<n>/tracks`: a
step joins consecutive rows with the same uid; the chord is the emission line,
beta the mean of its ends, `C0 = 2 pi alpha 1e9 x true path length`, and a step
is kept only if `beta * reference_phase_index > 1`. Pass the largest phase
index of the medium as `reference_phase_index` so that no step that radiates in
some band is dropped. Without `h5py` the built-in `hdf5_minimal` reader is used
(superblock v0, version-1 headers, 1-D datasets; deflate, shuffle and fletcher32
decoded, the checksum not verified).

## Medium and detector (`model.py`, `medium.py`)

`SpectralMedium(wavelength_nm, absorption_per_m, scattering_per_m, phase_index,
group_index, g=0.9, provenance=...)` interpolates its tables linearly;
`band(lambda)` returns a single-band `Medium`. `DetectorArray(positions_m,
orientations, effective_area_m2, angular_acceptance, spectral_efficiency,
identifiers={}, provenance=...)`: point OMs with one angular shape (a callable
of the head-on cosine) and one spectral efficiency (a callable of wavelength;
an optional `wavelength_knots_nm` attribute is used by the continuous-spectrum
quadratures).

## Baikal-GVD (`bgvd.py`)

* `lh.load_bgvd_model(package_path, *, dataset="2021", clusters="all", g=0.9)`
  reads a private `bgvd_model` checkout: `BaikalWater` (HG with `g`), the
  geometry CSV `data/median_om_coordinates_<dataset>.csv`, effective area
  `pi r^2`, angular acceptance `max(sum p_n (-cos)^n, 0)`, spectral efficiency
  `efficiency (linear) x gel/glass transmission (log-linear)`, checked bit for
  bit against the package's own functions.
* `model.save(path)` writes one `.npz` (arrays and the OM response tables, no
  code); `lh.load_model(path)` reads it without the private package. Models
  from NTSim (`bgvd_model_from_ntsim`) cannot be saved yet (no tabulated
  response attached).
* `lh.OMResponse` holds the tabulated OM response; `model.kernel(config)` returns
  a `TransportKernel`.

## Prepared tables and manifests

A route finds its tables from the kernel's `cache_directory` through a manifest
next to it (or in its parent directory):

* `shower-data.json`: `rte_cache`, `ge2_pack`, optional `ge2_band_fold`;
* `muon-data.json`: the seven keys of `muon-route.md` (a superset; the shower
  route also accepts it).

Tables follow the physics, not labels (`transport-kernel.md`, cache keys), so
one table set serves `load_model`, `load_bgvd_model` and NTSim's model alike.
The laser table also depends on the detector extent (its radial range).

## The internal data bundle `lighthit-bgvd-data`

Not distributed publicly. Layout:

```text
model/bgvd-model.npz        clusters 1-5 (1440 OMs), the model NTSim builds
tables/rte-cache/           RTE tables, 65 frequencies, 9 wavelengths
tables/ge2-pack/            order >=2 pack
tables/muon/                the five muon atlases
tables/laser-cache/         532 nm laser table (5 ns bins)
tables/muon-data.json       manifest (relative paths)
events/                     Geant4 100 GeV electron (HDF5); 1 TeV muon steps and family table (npz)
bgvd-model/                 the bgvd_model package needed by NTSim
```

Point `LIGHTHIT_DATA` to it; the notebooks and the NTSim command line use
`$LIGHTHIT_DATA/tables/rte-cache` as the cache.
