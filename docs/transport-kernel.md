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
| `readout_omega_per_ns` | None | denser readout grid (hybrid and shower routes); must end at the table cutoff |

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

## Other modules of this layer

| Module | Role | Status |
|---|---|---|
| `angular.py`, `_angular_numba.py` | k-space angular solve (table building) | internal |
| `cache.py` | multipole tables, acceptance coefficients | core |
| `directional.py`, `_directional_numba.py` | directional tables and their fused Numba apply | core |
| `green.py` | `SolverSettings` (hashed into keys), `PointGreenSolver` single-band point Green spectra | settings core; solver optional |
| `single.py`, `single_fast.py` | exact full-HG single scattering for point flashes | internal (flash path) |
| `readout.py` | spectra to bins | core |
| `medium.py`, `model.py` | `Medium`, `SpectralMedium`, `WavelengthQuadrature`, `DetectorArray` | core |
| `geometry.py` | `describe_geometry`, `point_source_radial_range` | public; used by the laser kernel |
| `api.py` | `lh.build(medium, detector, config, ...)`: kernel + `build()` | public |
| `cli.py`, `__main__.py` | `lighthit` CLI: single-band point Green calculation from a TOML file (not `TransportKernel`) | standalone, untested |
| `providers.py` | one wavelength of a private `BaikalWater.py` (CLI only) | standalone |
| `selftest.py` | `lighthit-selftest`: two-OM flash on a two-band medium | standalone |
| `examples/synthetic.py` | `lighthit-demo`: artificial 48-OM array, laser, track, shower | standalone |
