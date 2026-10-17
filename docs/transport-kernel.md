# TransportKernel, tables and responses

`lighthit.TransportKernel(medium, detector, config)` (`transport.py`) holds a
`SpectralMedium`, a `DetectorArray` and a `KernelConfig`, owns the radiative
transfer tables, and transports sources. The shower and muon routes build
sibling kernels from it (same medium and detector, their own frequency grids).

## KernelConfig (`transport.py`, frozen dataclass)

| Field | Default | Meaning |
|---|---|---|
| `omega_per_ns` | `linspace(0, 1.2, 321)` | table frequency grid; must start at 0 and increase |
| `relative_time_edges_ns` | `arange(-60, 745, 5)` | 5 ns bins relative to each OM's time origin |
| `wavelength_nodes` | 9 | Gauss-Legendre wavelength nodes |
| `wavelength_range_nm` | None | None: the medium's full range |
| `scattering_degree` | 24 | Henyey-Greenstein (HG) truncation L |
| `source_degree` | 32 | source/output multipole degree (`SolverSettings.spatial_degree`) |
| `azimuthal_degree` | 4 | largest \|m\| kept when compiling an axial source |
| `cell_m` | 0.12 | axial compile cell; tracks are split at `max(cell_m, 0.25)` |
| `k_max_per_m`, `k_panel_per_m`, `k_order` | 8.0, 0.04, 10 | k quadrature: panels of width `k_panel` with `k_order` Gauss nodes up to `k_max` |
| `radial_range_m` | (3, 300) | radial range of the tables; no extrapolation outside |
| `radial_nodes` | 220 | geometric radial grid |
| `angular_backend` | "auto" | numba if installed, else numpy |
| `acceptance_degree` | None | None: measured bandwidth of the OM acceptance; an integer forces a truncation |
| `acceptance_tolerance` | 1e-10 | tolerance of that measurement |
| `receiver_block` | 8 | Numba task grouping |
| `threshold_pe` | 0.01 | an OM is active when its charge reaches this |
| `cache_directory` | None | None: tables only in memory |
| `cache_policy` | "build" | "build" builds missing tables during transport; "require" raises `FileNotFoundError` |
| `spectral_folded_cache` | "auto" | use a folded cache found on disk; True requires one; False disables |
| `allow_experimental` | False | needed for methods flagged experimental |
| `readout_omega_per_ns` | None | denser readout grid of every route (`transport`, hybrid, shower, muon); must end at the table cutoff |

`kernel.acceptance()` measures the Legendre coefficients of the OM acceptance up
to `max(2*scattering_degree, 64)`. It raises if the response is not band-limited
there (unless `acceptance_degree` is set) or if the bandwidth exceeds
`scattering_degree`.

## The tables

The RTE is solved in k-space for each wavelength band (`angular.py`: free
Legendre moments with a Miller backward recurrence, tridiagonal solve with an
exact free-streaming tail; HG eigenvalues `mu_s g^l` up to L) and inverted to
radius. Two table kinds exist, both stored per wavelength as cubic splines in
log r of coefficients divided by `exp(-mu_a r)/(4 pi r^2)`, with the flight
phase `exp(-i omega r / v)` removed:

* **multipole** (`cache.ResponseCache`, file `transport-<key>.npz`): moments
  `M_l(r, omega)` for isotropic OMs and the flash path;
* **directional** (`directional.DirectionalCache`, file
  `directional-<key>.npz`): `t_(l lambda |mu|)(r, omega)` for OMs with a
  measured angular acceptance (rows `lambda <= L_A` of the m-block resolvent),
  both scattered orders (exactly 1 at finite L, and >=2).

**Cache keys.** `sha256` (first 20 hex digits) of: the band's medium constants
without the provenance label (wavelength, absorption, scattering, g, group
index), the solver settings, the omega list, radial range and node count,
`phase="flight"`; directional keys add the acceptance degree. Lookup order:
in-memory, physics key, the LightHit <= 0.3.0a1 key (which included the
provenance), then a scan of the directory comparing the stored metadata. Not in
any key: the angular backend, the detector geometry (except the directional
acceptance degree), `cell_m`, `azimuthal_degree`, the wavelength quadrature.

## build()

`kernel.build(wavelengths_nm=None, *, method="auto", source=None, force=False,
progress=None, ge2=True)` builds the tables a transport will read, regardless of
`cache_policy`, writes them atomically and writes `cache-manifest.json`.

* `source=` decides the method and wavelengths exactly (a monochromatic flash
  needs its own line and the multipole table; element sources the directional
  table when the OM acceptance has degree >= 1).
* When the directional tables of every kernel wavelength were built, the kernel
  has a `cache_directory` and `ge2` is true, `build()` also prepares the order
  >=2 pack over all frequencies (below). The directional tables then leave
  memory (they stay on disk).

Prepare a reusable table set:

```python
kernel.build(source=some_shower)                 # RTE tables and the >=2 pack
kernel.ge2_kernel_store().save("tables/ge2-pack")
```

## The order >=2 pack (`shower_basis.Ge2KernelStore`)

A bundle keeps only the order >=2 cubic radial-spline coefficients of the
directional tables, stacked over wavelength: shape (wavelength, radial interval,
degree, pair, cubic power, frequency), `complex64` by default. Its key is (the
directional keys of every kernel wavelength, `source_degree`, frequency
indices). `bundle(kernel, indices)` returns a stored bundle, slices a denser
stored bundle (memory-mapped view for regular strides), or builds it from the
directional tables. `save(dir)` writes `<n>.npy` per bundle plus `pack.json`;
`load(dir)` memory-maps them. `kernel.ge2_kernel_store(path)` loads a pack onto
the kernel; without a path it returns the kernel's store.

## Transport paths

* `kernel.transport(source)`:
  * `IsotropicFlash` -> `_transport_isotropic`: multipole table at the flash
    wavelength(s). Order 0 exact ballistic with the callable acceptance; order 1
    the exact full-HG physical-time integral (`single.single_bins`) when the
    acceptance is constant, otherwise the finite-L multipole first order;
    order >=2 from the table. The acceptance is contracted up to
    `source_degree` without the bandwidth check.
  * a straight `CherenkovTrack` with a frozen cone (Numba) -> `_track_transport`.
  * element sources -> `_transport_directional` when the cone is spectral or the
    acceptance degree is >= 1 (zero-frequency pass for every OM, full frequencies
    for active OMs), else `_transport_axial` on the multipole table. A folded
    cache (`build_folded_directional`) found on disk is used automatically for
    frozen cones (`spectral_folded_cache="auto"`).
* `kernel.transport_prompt(source, config=PromptConfig())` -> `prompt.py`:
  orders 0 and 1 in physical time, no tables (see `shower-route.md`).
* `kernel.transport_hybrid(source, frequency_nodes=65, ...)`: orders 0 and 1 from
  the prompt solver, order >=2 per band from the directional tables, or with
  `ge2_basis=ShowerBasis()` from `shower_basis.transport_hybrid_basis` (the
  per-band route that the shower route replaces).

## Responses and readout

`TransportResponse`, `HybridTransportResponse` and `PromptTransportResponse`
share the reading interface (`README.md`, Reading a response):
`charge_pe`, `charge_components_pe` (OM, 3), `bins_pe`, `components_pe`
(OM, bin, 3), `active`, `time_origin_ns`, `relative_time_edges_ns`,
`metadata`, `save()`. The prompt response refuses to return order >=2
(`OrderNotComputedError`).

`readout.inverse_bins` turns a demodulated spectrum into bin integrals: a
half-axis Fourier integral with trapezoid frequency weights and exact
integration of `exp(-i omega t)` over each bin, optionally with a Gaussian
factor. It never clips or renormalises, so finite-band ringing stays visible.

A sum over frequencies spaced by `d_omega` repeats in time every
`2 pi / d_omega`: 335 ns for the 65 route frequencies up to 1.2 rad/ns, shorter
than the default 805 ns window. `readout.readout_bins`, used by
`kernel.transport` for the Fourier-derived orders, therefore interpolates the
demodulated spectrum onto `readout_omega_per_ns` (cubic spline of the real and
imaginary parts, as the hybrid routes do for order >=2) and warns
(`RuntimeWarning`) when the window is longer than the period of the grid it
reads out on. The spline resolves waveforms much shorter than the table period
(a 20 ns exponential keeps its charge to 0.5 %); a tail comparable to half the
period is attenuated (an 80 ns exponential loses 10 %).

## Direct light on the OM sphere (`ballistic_interval`)

Order 0 of spectral elements (`ballistic_interval`, used by the shower, muon
and event routes) treats an OM as the sphere whose cross-section is its
`effective_area_m2` (radius 0.216 m for Baikal-GVD, the OM of NTSim's photon
Monte Carlo) for elements that come closer than
`BallisticIntervalConfig.finite_receiver_m` (default 20 m) to the OM centre,
and as its centre for farther elements.

A short step lights a thin conical shell. At one wavelength the rays through
the OM centre leave the step from one point, so a point receiver counts the
whole shell when it passes the centre and nothing when it misses it. The rays
crossing the sphere leave the step from a window of half-width `R/sin(theta)`
around that point, weighted by the semicircle law of the sphere's chords
across the cone surface; the charge of an element is the part of this window
inside the element (the hard window `[0, L)` of the point receiver). This is
the leading order in `R/rho`: attenuation, acceptance and arrival time stay
those of the centre ray, attenuation is taken to the centre (the reference
plane of the effective area), and a step whose line passes through the sphere
(`rho < R`) is not described. Collinear pieces of a straight track tile their
windows, so splitting a track does not change its charge. The support in
phase index widens by the angular radius of the sphere, and the two inner
indices where the window starts to leave the element at either end split the
spectral panels.

Measured against an independent integration of the rays from every emission
point and wavelength that cross the sphere, on the 6 GeV muon passing OM 267
at 1.76 m: OM 267 59.70 pe against 59.65 pe, OMs 266 and 268 within 1.2 %;
farther OMs of showers agree within 0.1 % (`finite-receiver-order0` in the
registry; numbers and the photon Monte-Carlo comparison in `accuracy.md`).
Beyond `finite_receiver_m` the point receiver stays; its order-0 error per OM
is `accuracy.md`'s table. `finite_receiver_m=0` is the point receiver
everywhere (bitwise the earlier results), `inf` the sphere everywhere (10-14
times the order-0 time: a far short step then reaches the sphere over most of
the spectrum instead of a narrow band).

## Near passes (`near_field.py`)

The order >=2 tables start at `radial_range_m[0]` (3 m). The muon and shower
routes leave source points closer than that out of their order >=2 sums and
`near_field.add_near_field` adds them back with an explicit approximation, the
near-field clamp: the order >=2 Green function of a point at distance
`r < r_min` is taken at `r_min` along the same directions (source cone and OM
orientation exact), with the flight phase of the actual distance. It rests on
the bounded behaviour of multiply scattered light near its source: two
scatterings within `r` are `O(r^2)` likely while the dilution is `r^-2`, so
`G>=2` tends to a finite limit instead of the `r^-2` growth of direct light.
The near portions are integrated on the original steps (Gauss, 5 cm panels).
Every OM that uses it is listed in `metadata["near_field"]` (OMs, clamped
charge) and named in a `RuntimeWarning`. Orders 0 and 1 do not use the radial
tables and are unaffected.

Measured on a 6 GeV muon passing an OM at 1.76 m (`muon-near-om` of the
registry): the clamp gives 0.051 pe of the OM's 0.167 pe order >=2 (orders 0
and 1: 59.7 and 3.6 pe; order 0 was 63.6 pe with the point receiver). The NTSim photon Monte Carlo of the same event (31
runs, hits split by their number of scatterings) gives 0.145 +- 0.024 pe for
order >=2 at that OM; without the near part LightHit would give 0.116 pe. Moving
the clamp radius from 3 m to 3.5 m or 4 m lowers the OM's order >=2 by 6 % or
17 % of the near part, so `G>=2` still grows towards the source and the clamp
at 3 m more likely underestimates the near part than overestimates it. The
clamp is an approximation, not a validated near-field solution: its error is
not bounded beyond these measurements. The boundary between the tables and the
clamp follows the original steps exactly for the primary of the muon route and
to within one basis cell (0.12-0.25 m) for shower basis points.

## Other modules of this layer

| Module | Role | Status |
|---|---|---|
| `angular.py`, `_angular_numba.py` | k-space angular solve (table building) | internal |
| `cache.py` | multipole tables, acceptance coefficients | core |
| `directional.py`, `_directional_numba.py` | directional tables and their fused Numba apply | core |
| `green.py` | `SolverSettings` (hashed into keys), `PointGreenSolver` single-band point Green spectra | settings core; solver optional |
| `single.py`, `single_fast.py` | exact full-HG single scattering for point flashes | internal (flash path) |
| `readout.py` | spectra to bins | core |
| `near_field.py` | order >=2 of source points closer than the radial minimum (clamp) | core |
| `medium.py`, `model.py` | `Medium`, `SpectralMedium`, `WavelengthQuadrature`, `DetectorArray` | core |
| `geometry.py` | `describe_geometry`, `point_source_radial_range` | public; used by the laser kernel |
| `api.py` | `lh.build(medium, detector, config, ...)`: kernel + `build()` | public |
| `cli.py`, `__main__.py` | `lighthit` CLI: single-band point Green calculation from a TOML file (not `TransportKernel`) | standalone, untested |
| `providers.py` | one wavelength of a private `BaikalWater.py` (CLI only) | standalone |
| `selftest.py` | `lighthit-selftest`: two-OM flash on a two-band medium | standalone |
| `examples/synthetic.py` | `lighthit-demo`: artificial 48-OM array, laser, track, shower | standalone |
