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
omega_per_ns=None, max_radius_m=None)` returns a `KernelConfig` for a flash:

* radial range `(3, max(300, 1.05 * farthest OM from the detector centroid))`
  from `point_source_radial_range` (it depends on the detector, not on where
  the flash is); `max_radius_m` replaces the upper bound;
* `k_panel_per_m = min(0.04, 2 pi / r_max)`, bins from -60 to 740 ns, other
  fields at their defaults (321 frequencies unless `omega_per_ns` is given);
* `cache_policy="require"` by default: build the table first,
  `laser.build(source=flash)` (multipole table at the flash line only).

`laser_kernel(medium, detector, *, cache_directory, **options)` wraps it in a
`TransportKernel`. The flash itself is `IsotropicFlash.monochromatic(position,
photons, wavelength_nm)`; the transport is the flash path of
`transport-kernel.md`. Nothing in the kernel is monochromatic; the wavelength
comes from the flash.

## Tests

`tests/test_event.py`: composition with fake routes (one clock, summed-charge
selection, completion, the out-of-reach case) and a real laser flash;
`tests/test_geometry_and_pose.py` for `laser_config`. The route fallback has no
test.
