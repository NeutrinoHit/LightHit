# LightHit

LightHit computes the expected optical-module response to light propagating in
a homogeneous scattering medium. It solves the time-dependent radiative
transfer equation (RTE), including unscattered, once-scattered and
multiply-scattered light, and returns expected photoelectrons per module and
per time bin.

The medium, detector and source are ordinary Python objects. There is no global
initialisation state, so several configurations can safely coexist in one
program.

LightHit is currently an alpha release. Validate numerical settings for your
medium and geometry before using its results in an analysis.

## Installation

```bash
python -m pip install lighthit
```

Numba acceleration is strongly recommended for tracks, showers and large
detector arrays:

```bash
python -m pip install 'lighthit[accelerate]'
```

For the installed synthetic demonstration, viewer and wheel self-tests:

```bash
python -m pip install 'lighthit[demo,test]'
lighthit-selftest
```

Python 3.11 or newer is required.

## Quick start

The example below defines a spectral medium, two optical modules and a
monochromatic isotropic flash.

```python
import numpy as np
import lighthit as lh

# Optical properties tabulated versus wavelength.
medium = lh.SpectralMedium(
    wavelength_nm=[400.0, 450.0, 500.0],
    absorption_per_m=[0.030, 0.020, 0.040],
    scattering_per_m=[0.030, 0.022, 0.017],
    phase_index=[1.344, 1.339, 1.336],
    group_index=[1.386, 1.374, 1.367],
    g=0.9,
)

# Response functions receive NumPy arrays and return arrays of the same shape.
def angular_acceptance(head_on_cosine):
    return np.ones_like(np.asarray(head_on_cosine, dtype=float))

def spectral_efficiency(wavelength_nm):
    return np.full_like(np.asarray(wavelength_nm, dtype=float), 0.20)

detector = lh.DetectorArray(
    positions_m=[[20.0, 0.0, 0.0], [35.0, 0.0, 0.0]],
    orientations=[[-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
    effective_area_m2=0.05,
    angular_acceptance=angular_acceptance,
    spectral_efficiency=spectral_efficiency,
)

source = lh.IsotropicFlash.monochromatic(
    position_m=[0.0, 0.0, 0.0],
    photons=1.0e8,
    wavelength_nm=450.0,
)

# build() returns a reusable kernel and prepares only the tables this source
# needs. The same kernel can then transport more compatible events.
kernel = lh.build(medium, detector, source)
response = kernel.transport(source)

print(response.charge_pe)   # integrated expected photoelectrons, one per OM
print(response.bins_pe)     # expected photoelectrons, shape (OM, time bin)
```

The first build may take noticeably longer than later event calculations.
Set `KernelConfig(cache_directory=...)` to reuse transport tables across
processes.

## Sources

The high-level dispatcher accepts:

```python
flash = lh.IsotropicFlash.monochromatic(
    position_m=[0, 0, 0], photons=1e8, wavelength_nm=450
)

track = lh.CherenkovTrack(
    start_m=[0, 0, -20],
    direction=[0.2, 0.1, 0.97],
    length_m=40,
    beta=1.0,
)

shower = lh.G4Shower.from_hdf5("event.h5", event=0)

synthetic_shower = lh.SyntheticShower.gaussian(
    lh.SourcePose(
        position_m=[25.0, -10.0, -720.0],
        direction=[0.4, 0.2, -0.89442719],
        time_ns=0.0,
    ),
    charged_track_length_m=300.0,
)
```

For tracks and showers, `kernel.transport(source)` automatically selects the
source engine and uses the detector's angular response. Research methods are
never selected unless explicitly enabled.

## Configuring the calculation

Numerical settings are collected in `KernelConfig`:

```python
config = lh.KernelConfig(
    wavelength_nodes=9,
    threshold_pe=0.01,
    cache_directory="lighthit-cache",
)

kernel = lh.build(medium, detector, track, config=config)
response = kernel.transport(track)
```

For production jobs, prepare the cache explicitly and forbid event processing
from starting an expensive build unexpectedly:

```python
build_config = lh.KernelConfig(cache_directory="lighthit-cache")
kernel = lh.TransportKernel(medium, detector, build_config)
kernel.build(source=track, progress="console")
print(kernel.last_build_report.as_dict())

run_config = lh.KernelConfig(
    cache_directory="lighthit-cache",
    cache_policy="require",
)
kernel = lh.TransportKernel(medium, detector, run_config)
response = kernel.transport(track)
```

Repeated `build` calls reuse compatible tables and report that there is nothing
to do. Pass `force=True` only for an intentional rebuild. Detector coordinates
can be inspected without detector-specific assumptions:

```python
geometry = lh.describe_geometry(detector)
print(geometry.centroid_m)
print(geometry.bounds_min_m, geometry.bounds_max_m)
```

Important inputs are explicit:

- `SpectralMedium` contains absorption and scattering coefficients in m⁻¹,
  phase and group refractive indices, and the Henyey–Greenstein parameter `g`.
- `DetectorArray` contains OM positions, orientations, effective areas,
  angular acceptance and wavelength-dependent detection efficiency.
- A detector orientation points from the OM toward a head-on source;
  `angular_acceptance(+1)` is the head-on response.
- `spectral_efficiency` should include all wavelength-dependent detection
  factors required by the application, such as quantum efficiency and optical
  transmission.

## Reading the response

`kernel.transport(...)` returns a `TransportResponse`.

| Attribute | Meaning |
|---|---|
| `charge_pe` | Integrated expected photoelectrons, shape `(OM,)` |
| `bins_pe` | Expected photoelectrons per relative time bin, shape `(OM, bin)` |
| `charge_components_pe` | Integrated contributions `[ballistic, one, two-or-more]` |
| `components_pe` | The same three components per time bin |
| `active` | OMs for which the full time spectrum was evaluated |
| `relative_time_edges_ns` | Bin edges relative to each OM's time origin |
| `time_origin_ns` | One absolute time origin per OM |
| `metadata` | Method, backend and numerical diagnostics |

Absolute bin edges for every module are

```python
absolute_edges_ns = (
    response.time_origin_ns[:, None]
    + response.relative_time_edges_ns[None, :]
)
```

The integrated charge is evaluated independently at zero frequency. A finite
time window and a finite frequency grid mean that `bins_pe.sum(axis=1)` need
not equal `charge_pe` exactly. Signed ringing is reported rather than silently
clipped or renormalised.

Responses can be stored as compressed NumPy files:

```python
response.save("response.npz")
```

## Interactive viewer

Install the optional viewer dependency:

```bash
python -m pip install 'lighthit[viewer]'
```

Turn a response into a standalone interactive HTML file:

```python
payload = lh.viewer_payload(
    response,
    source=source,
    event_id="flash-450nm",
    label="450 nm calibration flash",
)

viewer_path = lh.write_event_viewer(payload, "lighthit-viewer.html")
print(viewer_path)
```

Compatible event payloads can share one viewer:

```python
combined = lh.merge_event_viewers(laser_payload, track_payload, shower_payload)
lh.write_event_viewer(combined, "multi-event-viewer.html")
```

Open `lighthit-viewer.html` in a web browser. It contains the detector geometry,
integrated charge by scattering order, a selectable per-OM time histogram and a
time animation. A portable `lighthit-viewer.json` companion is written beside
the HTML file.

## Installed synthetic demonstration

The wheel includes one public-data-only Baikal-like example. Its geometry and
OM response are illustrative and are not a Baikal calibration:

```bash
lighthit-demo geometry
lighthit-demo cache --profile quick --cache ./demo-cache --sources all
lighthit-demo run --profile quick --cache ./demo-cache \
  --output ./demo-output --sources all
```

The repository, but not the wheel, contains detector-specific laser, track and
G4-shower examples with explicit source positions and directions.

## Scope

The current model assumes a homogeneous, unbounded medium with elastic
Henyey–Greenstein scattering. It does not include boundaries, layered media,
structural shadowing, polarisation, detector electronics, trigger or noise.

## License

LightHit is distributed under the BSD 3-Clause License.
