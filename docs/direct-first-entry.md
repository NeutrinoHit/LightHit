# Direct light at the first glass entry

`lighthit.ballistic_interval.first_entry_ballistic` integrates the original
spectral Cherenkov elements to their first intersection with the supplied
`SphereSurface`. Both extinction and the optical clock end at that entry.
Effective collection area remains independent of the physical radius.
Detector transit time is a separate response.

The shower and muon event routes use this evaluator for order 0 by default.
The shower prepares direct charge before selection and requests bins only
for selected receivers; the muon route does the same for each evaluated
batch. Their grouped order 1 and prepared order >=2 tables still refer to
the centre. This does not complete the surface-reference change for all
orders. The standalone `ballistic_interval` retains its leading cone model;
`PromptConfig(ballistic_method="interval")` and
`JointMuonConfig(ballistic_method="interval")` select that comparison.

## Integration

The analytic path-moment charge and finite-disk arrival CDF are control
variates. The numerical integral adds the difference between actual entry
rays and that analytic response. It applies no event normalization or fixed
path/time shift.

For the charge, put c=1/(beta n), s=sqrt(1-c²), and let h be emission position
relative to the centre ray, in units R/s. On the actual clipped emission
window [ha,hb], the analytic mass is

```
I0 + (mu R c/s + c R/(2 rho)) I1 + (pi/4) mu R I2,
I0 = integral sqrt(1-h²) dh,
I1 = integral h sqrt(1-h²) dh,
I2 = integral (1-h²) dh.
```

Here mu is total extinction and rho the distance to the source axis. The
three elementary moments retain both endpoints, beta and receiver geometry.
The pi/4 is the uniform-arc mean of sqrt(1-u²). The remaining geometric and
extinction terms are recovered by the corrective integral. Thus a short or
asymmetric window retains its own path distribution. Geometry and spectral
branch cuts are prepared once and reused during quadrature refinement.

For a point of emission, wavelength is transformed into the normalized
meridional miss h of the cone on the sphere. The known semicircle measure
sqrt(1-h²) is integrated with positive quadrature. Source intervals split at
cone-window and time-edge crossings. At fixed source point and wavelength,
entry distance is monotone on the visible azimuth arc; the time-edge root
therefore gives a truncated arc integral rather than another sampled bin
indicator. Receiver orientation is retained in the angular acceptance.

Geometry bounds permit omission of small pair residuals. The charge bound
includes the quadratic remainder after the analytic path and curvature
moments, plus angular-response variation. Near the glass, the previous
cone bound plus the known analytic increment supplies another bound; the
tighter one is used. For arrival bins the original disk bound is retained,
including transport of the disk clock to
the exact entry clock. With e=R/rho<1, the uniform-arc time-distribution
total-variation bound is

```
V(e) = 2 min(1, sqrt((1+sqrt(1-e²))/(2(1-e)))-1).
```

It tends to zero with receiver angular size. Positive integration over
emission position and wavelength preserves the bound, so it bounds the sum
of absolute omitted errors over disjoint bins, including a partial window.
One quarter of the numerical tolerance is allocated to the analytic
quadrature, one quarter to omission, and half to residual convergence.
These fractions allocate error; they never multiply the physical charge.

Before quadrature, a separate positive bound can omit complete pair
contributions within one eighth of the absolute tolerance. Every hitting
cone has azimuthal half-width at most asin(R/rho), its emission window has
length at most 2R/s, and its entry path exceeds the closest segment distance
minus R. Spectral extrema are taken on each supported optical interval.
The omitted full charge also bounds its mass in any disjoint time bins and
is deducted from the existing omission budget.

The charge pass retains contributions and quadrature changes per optical
panel when the required matrices fit available memory. A panel whose entire
arrival interval lies in one bin can reuse that charge. The sum of absolute
panel error estimates and omitted-residual bounds is checked per OM and
deducted from the same omission budget. Panels crossing an edge still use
the entry-ray integral. Insufficient memory disables reuse without changing
the physical response or tolerance. The reuse check is a numerical
convergence estimate; its time-interval enclosure is a geometric bound.

Quadrature orders are compared per receiver for charge and bin L1. Only
receivers failing the comparison are refined. Axial pairs and constant-phase
optics use direct positive ray integration, without the wavelength/impact
inverse. Failure of residual convergence can also select this independent
integration for the affected receivers alone. Failure at the configured
maximum order is explicit. Order comparisons are numerical convergence
checks, not a formal error proof for arbitrary response functions.

Emitting segments must lie outside the glass. A segment intersecting the OM
is rejected: propagating its in-glass emission as homogeneous water would
require a different material/source model. Without supplied geometry, auto
mode remains the mathematical point receiver.

## Cost and validation

On one 100 GeV electron with 42,194 original steps and 1,440 OMs, the explicit
event calculation kept the same 27 active receivers and bitwise unchanged
scattered components. Direct charge increased by about 1.01%. The complete
warm event initially took 19.30 s with path moments versus 19.77 s with the
previous cone control. With bounded full-pair omission, panel reuse and
equivalent arithmetic simplifications it takes 14.93 s. A new paired
measurement of the default route gives 10.05 s on the same Mac with one CPU
thread, including selection and convergence passes. The entry calculation
spends 3.19 s on charge and 3.40 s on bins. It remains about 49% slower than
the preceding fast route. This is the default event order 0; the scattered-light acceleration is
retained. These are warm
times; compilation is excluded.

The numerical tolerance remains 2e-6 relative plus 1e-10 pe per OM. Against
the preceding entry response on the exact original MC steps, the maximum
charge difference uses 0.047 of that tolerance and the maximum bin-L1
difference uses 0.038. The retained interval comparison remains bitwise equal to its
saved charge, bins, active set, time origins and >=2 spectrum. In a separate
charge pass over all 1,440 receivers, full-pair bounds retain 89,550 of
311,850 candidates; this count includes no fixed distance cutoff.

Charge controls on five receivers use an independent integration order; the
largest discrepancy of the path-moment result is 0.021 of the allowed error.
Those five receivers lie at 25–60 m. This convergence result does not certify
physical accuracy at all distances; `accuracy.md` gives the separate
distance comparison with 16 saved photon-MC runs and its open discrepancies.
`tests/test_ballistic_first_entry.py` additionally checks analytic axial
charge, the entry clock, misses, invalid in-glass emission, and deferred bins
on unselected receivers. Its dispersive tilted-receiver controls compare
both charge and bins with independent rays. `test_ballistic_linear_control.py`
checks partial-window moments, source partition, receiver-specific refinement
and the residual bound at three distances. The direct evaluator performs no interpolation in
the first-scattering profiles described in `receiver-first-entry.md`.
`test_ballistic_full_bound.py` checks the positive bound, spectral support,
partial-panel reuse, receiver reordering, bin crossings and the memory
fallback. These changes do not move the scattered orders to the glass.

## Addressed arc-CDF alternative

Registry idea `entry-cdf-moments` replaces the inner angular integral by
eight universal depth moments. It keeps the first-entry time geometry and
uses a bounded polynomial approximation of the positive arc density;
geometries outside that error certificate retain angular quadrature. The
CDF follows from elementary moment recurrences, so the universal full-CDF
constants occupy 64 bytes. One sixty-fourth of the existing relative error
budget is reserved for this approximation and removed from residual
quadrature; the requested charge and bin-L1 tolerances are unchanged.

The prototype passed one OM and then two other geometries at 25.02, 39.82
and 61.99 m, reducing their bin time by about 3–5%. A subsequent complete
warm event gave 15.31 s with this CDF versus 15.04–15.06 s for the current
entry method and 10.21 s for the fast method, on the same Mac with one
thread. Entry bin times were 3.26 s versus 3.41–3.52 s; unchanged order-1
time varied by about 0.45 s. These measurements support a small local gain,
not a demonstrated total-event speedup or closure of the gap to the fast
route. The maximum full-event bin-L1 difference is 4.24e-8 of the unchanged
numerical tolerance, with the same active set and scattered response.

The prototype remains in the experiment harness and is not connected to
production. It is an arc CDF at fixed emission point and wavelength;
integration over wavelength, source position and their time crossings
remains. A table of the entire spectral/source response of an OM was not
established by this test.
