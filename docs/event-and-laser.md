# Several sources and the laser (`lighthit.event`)

`EventTransport(kernel, *, muon_data=None, shower_data=None, laser_kernel=None,
threads=None, safety=3.0, max_rounds=4)` composes the routes for one event. Public names: `SourceComponent`,
`EventTransport`, `common_time_origins`, `laser_config`, `laser_kernel` (not
re-exported at the top level).

## Components

`SourceComponent(kind, source, ancestry=None, primary_uid=None, label="")`:
`kind` is `"shower"` or `"muon"` (a muon needs `ancestry` and `primary_uid`);
the source must be `SpectralLightElements` with `cone_model="spectral"`.

* **One component**: its route is called directly with its own clock; only
  `event_components` and `route_fallbacks` are added to the metadata.
* **An `IsotropicFlash`**: `laser_kernel.transport(flash)`; refused without a
  laser kernel or when mixed with components.

## Several components

1. **One clock.** `common_time_origins`: per OM the earliest possible arrival
   over all steps of all components (fastest group speed). Every route is
   called with these origins and must reproduce them.
2. **First pass**, showers first, then muons: each route gets the summed
   charge of the components already computed as `selection_extra_charge`, so
   its threshold test sees the other sources' light. An OM counts as detailed
   for a source when the route marks it `active` (shower route: only then),
   when it is in the muon route's `fine_charge_mask`, or when the source gives
   it exactly zero charge. The route's `active` is taken at `threshold_pe`, so
   a shower OM computed in detail only because of the detail threshold (below)
   does not count as detailed and is recomputed in the completion rounds if
   it qualifies there.
3. **Completion rounds** (`safety`, `max_rounds`): an upper bound of the
   summed charge is formed (detailed charges as they are, estimated scattered
   charges times 3); every OM that may reach the threshold and is not detailed
   in some source is recomputed in that source (`receivers=`, full detail,
   threshold 0). The loop ends when a round recomputes nothing. The threshold
   of these rounds is `min(threshold_pe, detail_threshold_pe)`.
4. **Result**: `active = (summed charge >= threshold) & detailed in every
   source`; bins are summed for these OMs. It raises if an OM above threshold
   lacks a detailed charge. With a detail threshold, `active` is then narrowed
   to `threshold_pe` (`transport.py: detail_threshold.report`); the bins of the
   OMs between the two thresholds stay.

The composed response gets new metadata: `route = "lighthit.event"`,
`threshold_pe`, `safety`, `selection`, `time_origin`, `event_components` (per
source: kind, label, elements, primary UID, `runs`, `charge_pe`,
`detailed_modules`), `completion_rounds`, `active_modules`,
`inactive_charge_may_be_estimated`, `route_fallbacks`, `computed_orders`,
timings, and with a detail threshold `detail_threshold_pe`. The route keys
(`near_field`, `radial_tail`, `ge2_charge_level`, `fine_charge_mask`,
`smart_screen`, ...) are not kept; the first run of each source lists their
names (`event_components[*].runs[0].route_metadata_keys`). The routes'
`RuntimeWarning`s are still emitted.

## Detail threshold

`KernelConfig.detail_threshold_pe` (default None) below `threshold_pe`
separates the two decisions: every source route lowers its own kernel's
threshold to it for the call (`transport.py: detail_threshold`), so each
route computes in detail every OM whose estimate reaches it, and the
completion rounds use it too; `active` still means a summed charge of at
least `threshold_pe`. The event response records both thresholds. One
component: the route's response is returned with `active` at
`threshold_pe`.

**Fallback**: a muon whose route raises the "unvalidated inner" error (a
primary line through an OM centre, see `muon-route.md`) is transported by the
shower route instead, also in later completion rounds, and listed in
`route_fallbacks`. A muon or shower passing an OM closer than 3 m needs no
fallback: both routes add the near-field clamp (`metadata["near_field"]`).

## Laser kernel

`laser_config(detector, cache_directory, bin_ns=5, cache_policy="require",
omega_per_ns=None, max_radius_m=None, kernel_settings=None)` returns a
`KernelConfig` for a flash:

* radial range `(3, max(300, 1.05 * farthest OM from the detector centroid))`
  from `point_source_radial_range` (it depends on the detector, not on where
  the flash is); `max_radius_m` replaces the upper bound;
* `k_panel_per_m = min(0.04, 2 pi / r_max)`, bins from -60 to 740 ns, other
  fields at their defaults (321 frequencies unless `omega_per_ns` is given);
* `cache_policy="require"` by default: build the table first,
  `laser.build(source=flash)` (the table at the flash line only);
* `kernel_settings` replace configuration fields, as the `kernel_settings` of
  `shower-data.json` and `muon-data.json` do. By default they are read from
  `laser-data.json` in `cache_directory` (`laser_settings(cache_directory)`,
  schema `lighthit/laser-data/1`); a table directory without that file is the
  centre-referenced multipole table of earlier releases and keeps the
  defaults.

### Receiver of the flash

Without settings the flash is read at the OM centre: multipole table, order 0
the analytic `N A eta A(c) exp(-mu_t d) / (4 pi d^2)`.

With `scattered_receiver="first-entry"` the flash is counted at the first
entry into the glass sphere, the receiver of the shower and muon routes
(`_flash_receiver.py`):

* orders 1 and >=2: the first-entry directional table
  (`DirectionalCache` with the glass boundary vertex). The flash is a pure
  `l = 0` source, so the table is built with `source_degree=0` (9.3 MB at 532
  nm) and only `t_(0 lambda 0)` is read:
  `R_o = sum_lambda alpha_lambda sqrt(2 lambda + 1) t_(0 lambda 0, o) P_lambda(c) / (4 pi)`.
  On a centre table `t_(0 lambda 0) = M_lambda / sqrt(2 lambda + 1)` and this
  is the multipole flash to rounding (`tests/test_flash_receiver.py`); on a
  first-entry table it equals the generic `directional_response` of an `l = 0`
  source. Order 1 is the finite-L first order of the table (no exact
  physical-time order 1 on this path); both scattered orders are read out
  from the spectrum;
* order 0: the direct flux into the sphere, integrated over the cap the
  sphere subtends (Gauss in the impact parameter, trapezoid in azimuth),
  with extinction and clock to the entry point, the acceptance of each ray
  direction and the weight `area / (pi R^2)` per entering photon. Far from
  the sphere it exceeds the centre formula by `exp(mu_t 2R/3)`, 1.06 % at
  532 nm for R = 0.216 m;
* time origins, the active rule and the out-of-range check are those of the
  centre path.

`tools/tables/build_glass_laser.py` builds such a table from a model and the
`shower-data.json` of a glass bundle (its scattering degree and vertex
numerics, so the flash sees the same glass) and writes `laser-data.json`
beside it. The boundary vertex dominates the cost: about 31 s per frequency
at one thread; the 321-frequency 532 nm table took 70 min with three worker
processes (frequency blocks are checkpointed; repeat the command to resume).
A flash on it takes 0.31 s at one thread (0.10 s on the centre table), most
of it the order-0 quadrature.

Against the photon MC of the same 532 nm flash (1e8 photons, hits counted at
the sphere; `photon-mc-matrix/laser-532nm@glass-laser`) the glass table gives
0.975 +- 0.018 of the total charge (0.967 on the centre table), orders 0/1/>=2
at 0.962/0.957/1.009. The scattering degree is not the cause: a centre table
at degree 96 reproduces the degree-24 one to 4e-10. The glass also moves the
direct light and the early scattered light about 1 ns earlier, onto the MC's
clock at the sphere: the share of order 1 within the 5 ns before the centre
arrival at 50-150 m rises from 0.21-0.24 to 0.37-0.42 (MC 0.46-0.52), of order
>=2 from 0.06-0.09 to 0.13-0.19 (MC 0.12-0.18). One MC run (about 3.5 % on
orders 0 and 1) cannot separate the two tables in charge.

`laser_kernel(medium, detector, *, cache_directory, **options)` wraps it in a
`TransportKernel`. The flash itself is `IsotropicFlash.monochromatic(position,
photons, wavelength_nm)`; the transport is the flash path of
`transport-kernel.md`. Nothing in the kernel is monochromatic; the wavelength
comes from the flash.

## Tests

`tests/test_event.py`: composition with fake routes (one clock, summed-charge
selection, completion, the out-of-reach case) and a real laser flash;
`tests/test_geometry_and_pose.py` for `laser_config`;
`tests/test_flash_receiver.py` for the glass flash (contraction against the
multipole and generic directional kernels, direct flux against an independent
polar integral and the small-sphere limit, the route, `laser-data.json`). The
route fallback has no test.
