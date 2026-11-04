# Scattered light at the glass surface

`KernelConfig(scattered_receiver="first-entry")` selects a finite spherical
receiver for orders 1 and >=2. The radius comes from the detector's
`SphereSurface`, independently of effective collection area. The present
prepared kernels require one common radius. They stop a photon at its first
entry into the sphere; subsequent exits and returns are not additional hits.

This option requires separately prepared data. The default centre-reference
tables remain available. A surface kernel rejects centre-reference caches,
packs and folded kernels rather than silently mixing receiver conventions.
The route manifests record the boundary and solver settings in
`kernel_settings`; response metadata records `receiver_reference_by_order`.

For one-command preparation from a serialized model, start with
[build the optical cache](build-receiver-bundle.md).

## Orders and preparation

| Order | Surface calculation | Fast event calculation |
|---|---|---|
| 0 | Existing first-entry cone/sphere integration | Original steps, charge before OM selection; bins on selected OMs |
| 1 | Geometric pencil profiles with full HG scattering and ten arrival-direction moments for cubic acceptance | Existing source compression and parent/child selection, finite-kernel interpolation, shared spectra and single-band remainder units |
| >=2 | A receiver boundary vertex with an exact free-space tail | The existing source degree, harmonic layout, kernel packs and band-folded application |

For >=2 the boundary jump separates direct incidence, the first return, and
returns after at least two scatterings. The first return is evaluated by the
geometric one-scattering integral. The remaining boundary response uses the
local exterior transport solve. Propagation to distant sources uses the
free-space tail; a distant radial PN solution is not substituted for that
tail. Collision-series stopping criteria control numerical remainder, without
imposing a maximum physical scattering order.

`experiments/receiver_surface.py` prepares geometric Q1 data, spectral CDFs
and the directional bank. `experiments/prepare_receiver_route.py` checks the
completed bank and forms a shower pack with the existing band nodes, including
the exact compact zero-frequency companion used for charge-only screening. Primary
muon track rows and beta corrections must also be rebuilt for the surface
contract. `tools/tables/build_surface_preview.py` prepares finite-kernel
arrival moments on the existing coarse preview grid. The old MC table may
remain a separate `preview_proposal_atlas` for selection; output charges use
the new moments and the actual OM orientation.

Primary Q1 time tables include the physical cone and the emission speed
`beta*c`. The optional `primary_first_time_beta` manifest interpolates the
charge and time measure between validated beta anchors in `beta**-2 - 1`.
Radial time interpolation aligns the known glass-front advance in
`(time + R/v_fastest)/radius`; responses at stored radius nodes are unchanged.
The original residual-clock guard of 0.001 ns remains in force. Unsupported
pieces use direct segment integration with their actual beta and clock.

An optional `primary_first_cubature` supplies a positive wavelength rule for
these direct pieces. Charge validation does not authorize time compression:
the rule must separately pass waveform L1 checks. Below its validated beta
range the native wavelength quadrature is retained. In selected muon batches
the time integration also supplies the all-time charge, avoiding a duplicate
charge-only integration. Neither optimization changes receiver selection or
normalizes the result to an event or MC response.

The reference kernels used to prepare a point-cone preview need more angular
resolution than a compressed shower source. An under-resolved point kernel
can produce a negative density even when event-level integrated sources are
accurate. Check angular convergence during preparation; increasing that
reference degree does not increase the event source degree or the cost of a
preview lookup. A stationary reference pack can be built with
`experiments/prepare_receiver_route.py --stationary-only`; it is separate
from the full-frequency event pack.

The preview builder checks the cubic response over every OM orientation by
Bernstein subdivision on the radial/axial look disk. Unresolved or negative
spectral cells are recomputed by an independent positive Neumann-series
reference with an analytic first-scattering tail. Its collision prefix has
unbounded order with importance weights; photons stop at an earlier glass
entry. The per-cell sampling error is retained. This replaces the local
calculation, without clipping a predicted charge or fitting event responses.

The table integrator also resolves a floating-point entry-cut case at a
large grazing geometry. A mesh created in Python and the compiled density
can put the direct-entry singularity on adjacent floats. The compiled
quadrature splits at its own entry value before choosing interior nodes;
it never substitutes an iteration cap or a physical path cutoff. Invalid
HG denominators fail explicitly instead of entering a nonterminating
elliptic iteration. A regression checks all ten moments against a refined
path mesh at the critical geometry.

## Numerical checks

The time interpolator has a shared absolute L1 budget across common-spectrum
and remainder units. Its bound is relative to the stored fine CDF. It does
not certify source grouping, spatial interpolation or the transport solver;
those need separate controls. Charge-only OM selection evaluates the same
groups and charge kernels in compiled receiver batches and does not read
time-bin error tables.

Primary and secondary omission bounds also cover the supplied sphere.
Their conservative factor four bounds surface flux divided by projected
area; it is used only in omission certificates, never as a signal multiplier.
Secondary packets also use the visible-cap inequality
`2*pi*(1-sqrt(1-R**2/D**2)) <= 2*pi*R**2/D**2`, with extinction advanced
by at most `R`, to tighten the infinite-line bound. The minimum of the
valid bounds is used. Bounds for a spectral remainder include only its
emitting elements; the fused unit uses its actual fixed cone while retaining
each band's own threshold and yield. The per-unit 0.1% budget is unchanged.

The secondary route reuses bitwise equal emission positions and clocks,
without reordering pencil accumulation. The charge/error pass supplies the
same spatial brackets and spectral scales to the time pass when the grids
and optics match. Immutable CDF rows are copied lazily to contiguous time
lanes, preserving the scalar reduction order. The cache uses the required
payload and available RAM; writable tables or insufficient headroom retain
the canonical interpolator (`contiguous_time=False` is an explicit control).
Groups wholly outside the existing readout window retain their all-time
charge and need no time-CDF contraction.

`experiments/receiver_event_check.py` compares full shower or muon responses on the
same saved source steps, optics, MC sample and distance groups. Timing includes
source compilation, selection, every collision order and readout. Table setup
and first-call compilation are reported separately from the second call.
`--kind muon --parts ...` also verifies the saved event ancestry. After the
installation has been warmed, `--repeats 1` allows one addressed full-event
check without repeating table preparation or generating new MC events.
`--profile-order1` records the spectral-unit and grouping costs. A comparison
must retain failed per-order and distance-group flags even when total charge
agrees with MC.

Geometry/profile, first-entry MC, cubic acceptance, cache compatibility,
source grouping, spectral fusion and CDF error controls are covered by
`tests/test_receiver_*.py`. These controls complement the photon laboratory;
they do not make the finite angular, radial and frequency approximations exact.
The existing event-kernel radial window and the >=2 amplitude clamp below
its minimum radius remain separate limitations.

## Addressed event measurements

The following full-event checks use the exact saved charged steps and the
existing unsuppressed photon MC, on one VM thread. Times include source
compilation and all orders. The e100 and mu100 comparisons have an existing
warm baseline; e10 and pion use one paired call per route on the warmed
installation. These measurements do not include one-time table construction.
The mu100 row includes the later secondary-Q1 optimization. Other full-event
times precede that optimization; mu1TeV was not repeated in full.

| Event | Centre scattered tables, s | Glass scattered tables, s | Q0/MC | Q1/MC | >=2/MC | Total/MC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Electron 100 GeV | 36.57 | 38.46 | 0.9757 | 1.0031 | 1.0407 | 0.9968 |
| Electron 10 GeV | 16.16 | 10.85 | 1.0594 | 0.9829 | 0.9969 | 1.0206 |
| Pion 100 GeV | 76.76 | 66.55 | 0.9925 | 1.0058 | 0.9627 | 0.9919 |
| Muon 100 GeV | 117.52 | 116.80 | 1.0133 | 1.0329 | 1.0211 | 1.0204 |
| Muon 1 TeV | — | 551.58 | 1.0028 | 1.0408 | 0.9929 | 1.0136 |

All total-charge criteria pass with the existing `max(3%, 2 ratio SE)` rule.
This does not make every order or distance group pass. Electron 100 GeV
has a failed >=2 aggregate criterion and its existing 20–50 m excess
(1.1025); electron 10 GeV retains the direct-light flag at 20–50 m
(1.1302). The single near OM of muon 100 GeV retains its Q1 flag below
20 m (1.2802, historically 1.2725). No event normalization or tolerance
change is applied. Pion 100 GeV now passes all tested distance groups;
its total at 50–100 m is 0.9726 rather than 0.9660.
Muon 1 TeV retains the tail flags at 100–150 m: total 1.1012 and >=2
1.1211. Receiver transfer alone has not removed that residual. This group
carries 0.3973% of the MC event charge; its total excess corresponds to
0.0402% of the full event charge. The complete 1 TeV muon total and all
integrated collision orders pass. The tail remains a separate accuracy
limitation; no fitted correction is applied. The addressed monochromatic
and spectral-fold controls are recorded in [accuracy.md](accuracy.md).

The 1 TeV check also exposed `infinity * false` in the packet-certificate
sum for a pair that must be computed. The omission sum now selects only
skipped pairs. A separate recomputation from saved computed-pair masks and
the original steps verifies every spectral unit against the unchanged
0.1% seed-charge budget, without repeating light transport.

The optimized mu100 complete call takes 118.47 s initially and 116.80 s
after warming, with source compilation repeated; table setup is 0.93 s.
The preceding glass route took 141.08 s. Secondary Q1 falls from 53.98 to
29.70 s, close to the previous centre value of 28.31 s. The complete warm
route therefore regains its previous wall time; Q1 alone retains about 5%
additional cost on this event.

Against the preceding glass response, Q0 and >=2 charges and bins, origins,
active mask and detailed receiver set are bitwise equal. The maximum Q1
charge change per OM is 9.35e-7 relative, and its bin-L1 change divided by
all-time Q1 charge is at most 7.72e-7. All omission certificates retain the
same per-unit 0.001 limit. In addressed controls against glass Q1 without
packet omission, the largest relative charge errors are 5.19e-6 for four
mu100 OMs and 1.35e-7 for four OMs and 16 mu1TeV packets. The latter warm
Q1 check takes 2.00 s versus 2.27 s for the previous centre route on the same
packets. This is not a new full-event mu1TeV timing claim.
