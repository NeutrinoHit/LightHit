# First-entry receiver profiles

`lighthit.receiver_first.first_entry_profile` builds one row of the
first-scattering Green function for the glass boundary supplied by
`SphereSurface`. This is a table-building primitive. The default transport
routes still use their existing centre-reference tables and prompt kernels;
this primitive alone does not change their optical path or arrival times.

The input is an outside point source emitting a pencil along the local z
axis. The receiver centre is in the xz plane at distance r and angle theta.
The radius R comes from the detector geometry, independently of its effective
collection area. For total optical path S, the kernel integrates the first
scattering site and outgoing directions that first enter the receiver at S.
If the initial pencil intersects the receiver, its first flight stops at that
intersection. Integrating unrestricted inward flux on the surface would also
count photons that had already entered and returned, which is a different
observable.

## Stored quantities

The output is ten cumulative moments of the arrival direction:

```
1, x, z, xx, xz, zz, xxx, xxz, xzz, zzz
```

Reflection in the xz plane removes odd powers of y. The remaining y moments
follow from x² + y² + z² = 1, so these ten fields suffice for an arbitrary
cubic angular acceptance and receiver orientation. The fields carry no
wavelength-dependent attenuation, scattering coefficient, photon yield or
detector efficiency. For one wavelength, fold their path increments with
mu_s exp(-mu_t S), the source yield, angular acceptance and effective area.
The arrival clock is t_emit + n_group S/c. Detector transit time is a separate
response.

The geometry is evaluated on a unit sphere. At physical radius R, paths scale
by R, the path density by R^-2 and the cumulative moments by R^-1. A shared
geometry table can therefore serve different radii without making 32 copies
for the wavelength nodes. Its shape still depends on the scattering
asymmetry g.

## Integration and finite table extent

At a fixed total path, the outgoing azimuth is integrated analytically using
complete elliptic integrals of the Henyey–Greenstein phase function. Positive
quadrature remains over the first-flight length and the total path. The path
grid splits at the glass front r-R, the initial pencil's direct entry, and
the tangent length sqrt(r²-R²). The direct entry is an internal logarithmic
singularity when the pencil hits the receiver. A grid with only the front and
tangent endpoints can miss its contribution.

For a pencil that hits the receiver, the first-order profile vanishes after
the tangent length. Other pencils have an infinite tail. `max_path_m` is an
explicit table extent: the builder does not choose an attenuation cutoff or
claim that the omitted spectral tail is zero. The consumer must supply its
tail bound. `path_nodes`, `path_order` and `scatter_order` control numerical
resolution and must be checked by convergence when building interpolated
tables.

`tests/test_receiver_first.py` compares all ten attenuated moments with an
independent integral over the scattering site and visible surface cap. It
also checks first-entry termination, near-glass and far-forward geometry,
and the physical-radius scaling. These row checks do not establish the
accuracy or speed of a future interpolated table in an event route.
