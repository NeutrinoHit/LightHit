# Half-space transport: boundary and mathematical contract

The medium occupies `z > 0`; a smooth, lossless dielectric interface is at
`z = 0`. The first calculation is a pencil beam incident from the exterior
along the positive z axis. It remains a directed beam, including its
unscattered part. The first experimental observable is the flux that leaves
the medium through a specified exterior angular aperture and detector area.

The checkout contains an experimental volume reference
`lighthit.experimental.halfspace`, the dielectric boundary in
`halfspace_boundary`, and an independent continuous-history reference in
`halfspace_mc`. They implement directed boundary sources, separated volume
collision orders, frequency spectra, surface flux, reciprocal ports and
local absorption sensitivities. These modules are not shipped in the wheel
and are not part of the production OM routes. Experimental tissue data have
not been compared.

## Reproduce from a checkout

Use Python 3.11 or newer and install the development dependencies in a chosen
environment:

```sh
python -m pip install -e '.[validation,test]'
python -m pytest -q tests/test_halfspace*.py
python experiments/reproduce_halfspace.py --output-directory RESULTS
```

Run these commands at the repository root. `RESULTS` denotes a new, empty
local results directory selected by the caller. The last command runs the
76 tests, synthetic spatial/adjoint checks, seven material cases, polar
quadrature audit, coarse/fine time spectra, independent time-domain MC and
all report plots. It takes several minutes on a desktop machine. It uses
only this checkout and its installed dependencies; no private optical
bundle, Geant4 installation, previous cache or network connection is needed.

The inputs for the material cases are committed in
`experiments/fixtures/halfspace-materials.json`. The fixture records which
parameters were supplied by DN, the literature sources, unit conventions,
selected midpoints and unmeasured absorption scans. It is an illustrative
model specification rather than a measured calibration of those samples.

The output contains `manifest.json` with runtime versions, source SHA-256,
step return codes and timings. Sources are checked for changes during the
run. Individual JSON reports preserve fixed seeds, statistical errors,
convergence changes and pass/fail criteria. A failed step returns nonzero
and leaves its log and preceding outputs available. `--no-figures` runs the
same numerical checks and readout without rendering images.

The generated layout is:

```text
manifest.json
synthetic/validation.json         full-HG spectral, spatial and adjoint checks
synthetic/figures/banana.npz      spatial sensitivity array
materials/validation.json        sourced profiles and absorption scans
quadrature/validation.json       bulk/receiver convergence audit
time/coarse/spectrum.npz         192 directions, 301 frequencies
time/fine/spectrum.npz           256 directions, 601 frequencies
time/validation.json             time-grid/band checks and full-HG MC
time/profile/                    rebinned arrays and time-profile PNG/SVG
report/                          four scientific report figures, PNG/SVG
```

Individual entry points are `validate_halfspace.py`,
`validate_halfspace_materials.py`, `audit_halfspace_quadrature.py`,
`halfspace_time_profile.py`, `validate_halfspace_time.py` and
`plot_halfspace_report.py`, all under `experiments/`. Each accepts `--help`.
The figure generator reads the numerical reports and arrays, so figures can
be regenerated without rerunning transport. Timing measurements depend on
the machine and library versions; scientific checks use the recorded
tolerances, not exact runtime or bitwise comparison of MC results.

## Indices and stages

The boundary parameter is the **phase-index ratio**

$$
n(\lambda)=n_{\rm in}(\lambda)/n_{\rm out}(\lambda).
$$

The two stages use the same boundary operator:

1. `n = 1`: matched interface, exact zero reflection and unchanged direction;
   an outgoing ray leaves the transport domain.
2. Tissue to air, initially `n = 1.4`: angle-dependent Fresnel reflection and
   transmission, refraction and total internal reflection.

The approximate value 1.4 is a model setting, not a measured tissue spectrum.
The group index remains a separate medium input: `v = c / n_group`.
Setting the boundary ratio to one does not change that velocity. The current
`SpectralMedium` stores phase and group indices separately; its `band()`
returns only the group index in `Medium`, so the boundary ratio must be
constructed from `SpectralMedium.sample()` or supplied explicitly.

Both stages use the same source, detector, units and normalizations.
Stage 1 validates the method; stage 2 is required for the tissue-air
experimental comparison.

## Volume equation and directed illumination

For photon radiance I, in metres and nanoseconds,

$$
\left[v^{-1}\partial_t+\Omega\cdot\nabla+\mu_a+\mu_s\right]I
=\mu_s\int_{S^2}p(\Omega,\Omega')I(\Omega')\,d\Omega'+S.
$$

The scattering function is normalized to one over outgoing solid angle.
Optical coefficients supplied in inverse millimetres must be converted to
inverse metres at the input. A reduced scattering coefficient is distinct
from `mu_s`: for an HG medium, `mu_s_prime = mu_s * (1-g)`.

An exterior normal pencil beam has an angular delta at `Omega = +ez` and a
spatial delta at its entry point. Its entry weight is

$$
T_0=1-\left(\frac{n-1}{n+1}\right)^2.
$$

The zero-order interior beam retains this direction and attenuates as
`T0 * exp(-(mu_a+mu_s)*z)`, with arrival time `z/v` from surface entry.
For n=1.4, `R0 = 1/36` and `T0 = 35/36`. Normalization is per exterior
incident photon. Direct exterior specular reflection is reported separately
from photons that enter the volume and later leave it.

The collision orders are 0, 1 and >=2 volume scatterings. Reflection at the
plane does not increment the volume scattering count. A frequency-domain
calculation uses the existing inverse-time convention `exp(-i omega t)`;
the Fourier streaming operator contains `-i omega/v`.

## Fresnel boundary and measured flux

Let mu be the positive magnitude of the interior normal cosine, and let
`Omega_ref = (Omega_x, Omega_y, -Omega_z)` be its specular partner. Snell's
law gives `sin(theta_out) = n * sin(theta_in)`. Above the critical angle the
transmission is zero. For the scalar model, R is the mean of the s and p
Fresnel power reflectances and `T = 1-R`.

For interior incoming directions the physical-radiance boundary is

$$
I_{\rm in}^{+}(\Omega)=R(\mu)I_{\rm in}^{-}(\Omega_{\rm ref})
   +n^2T(\mu)I_{\rm out}^{\rm incident}(\Omega_{\rm out}).
$$

Along a transmitted ray, outgoing physical radiance instead satisfies
`I_out = T * I_in / n**2`. These index factors arise from the refractive
solid-angle Jacobian. They do not multiply the transmitted photon-packet
weight, which is T.

For an exterior acceptance A, the measured escape flux can be integrated
entirely in interior directions:

$$
D[I]=\int_{\mathcal A_{\rm detector}}dA
      \int_{\Omega_z<0}d\Omega\,
      |\Omega_z|\,T(|\Omega_z|)A(\Omega_{\rm out})I(\rho,0,\Omega).
$$

There is no additional n-squared multiplier in this formula. The
`ExternalAperture` half-angle is measured **in the exterior**. Its
`internal_weight()` returns `T*A`; `escaped_flux_density()` also includes
the interior normal cosine. Detector area, solid-angle quadrature weights,
spectral response and time integration are separate factors. A finite
exterior propagation distance also needs its own ray mapping and flight
time; an angular cone alone describes collection at the surface.

## Reciprocal source and receiver ports

Source and receiver share a boundary phase-space port: surface points,
directions, angular response, spatial area and quadrature weights. The
normal pencil beam is a delta port, while a finite detector aperture is an
integrated port. Swapping them preserves their respective angular shapes;
the reversed aperture source is not an isotropic replacement for the beam.

Use reduced radiance `u = I / n_phase**2` and the invariant surface measure

$$
d\Gamma=n_{\rm phase}^2|\Omega\cdot\nu|\,dA\,d\Omega.
$$

In these variables transmission multiplies u by T in both directions.
The receiver weight `T*A` is also its reversed incoming reduced-radiance
source weight. Port emission and collection must use the same measure and
normalization. A detector defined as a functional of physical interior
radiance has incoming adjoint weight `T*A` in the internal flux pairing;
physical exterior radiance injection additionally carries n-squared.
Keeping these conventions explicit avoids mixing physical and reduced
radiance when checking reciprocity.

For a source port s and detector port d, verify the weighted relation
`H(d <- s) = H(reverse(s) <- reverse(d))` at each frequency and collision
order. The phase function and interface are reciprocal under direction
reversal. Delta directions should be treated analytically or with dedicated
source terms, rather than smeared across the angular quadrature.

The absorption sensitivity (the banana) requires the forward and adjoint
angular fields in the volume. With a physical-radiance adjoint normalized
by the detector functional,

$$
\delta D=-\int_H d^3r\,\delta\mu_a(r)
                 \int_{S^2}d\Omega\,I_s(r,\Omega)\psi_d(r,\Omega).
$$

The adjoint is driven by the reversed detector port. If its implementation
stores a forward-propagating reversed-direction field J, then
`psi_d(r,Omega) = J_d(r,-Omega)` with the corresponding port normalization.
For time-resolved measurements the time-domain product becomes a
convolution; at fixed frequency the above expression is the bilinear
transfer-function variation. Here the adjoint is the transpose adjoint:
it has reversed streaming, the same `-i omega/v` term, and the pairing has
no complex conjugation. A Hermitian-adjoint implementation would instead
conjugate its adjoint field in the pairing. Replacing this angular pairing by a product of
scalar fluences would introduce an additional approximation. The volume
solver must therefore retain access to angular fields.

## Numerical reference

Fourier transformation in the two surface coordinates and in time reduces
the volume equation to angular boundary-value problems in depth. The
reference uses discrete ordinates with paired positive/negative normal
cosines and a uniform azimuth rule. For n>1 the cosine quadrature is split
at the total-internal-reflection threshold. Both boundary cases use this
same solver.

Write `U_i = sqrt(w_i) * I_i` for the scattered angular field. Its equation is

$$
M\partial_z U+A U=\sum_s f_s e^{-d_s z},\qquad
M_{ii}=\Omega_{i,z},\qquad
A_{ij}=\delta_{ij}(\mu_t-i\omega/v+i\mathbf q\cdot\Omega_{i,\parallel})
       -\mu_s\sqrt{w_iw_j}\,p(\Omega_i\cdot\Omega_j).
$$

Each source ray remains an analytic angular delta, with internal amplitude
`b_s = source_flux_weight * T_entry / c_s`, where `c_s` is the ray's
positive normal cosine. Its depth rate is
`d_s = (mu_t-i*omega/v+i*q.Omega_parallel)/c_s`, and its collision forcing is
`f_i,s = scattering_per_m * sqrt(w_i) * p(Omega_i.Omega_s) * b_s`.

Solve the generalized eigenproblem `A V = M V Lambda`. Half the modes decay
into z>0. Let `g = inverse(V) inverse(M) f`. For a decaying eigenvalue lambda
the driven coordinate contains

$$
c e^{-\lambda z}
+g\frac{e^{-d_s z}-e^{-\lambda z}}{\lambda-d_s};
$$

the removable equal-rate limit is `g*z*exp(-lambda*z)`. The growing-mode
coordinate is `g*exp(-d_s*z)/(lambda-d_s)`, selected by the condition at
infinity. Only the decaying homogeneous coefficients c are free, and the
Fresnel boundary determines them. Thus no finite depth wall or depth mesh
is introduced. An explicit generalized eigen-equation residual is checked
before any response is returned.

The upward field in an arbitrary receiver direction is evaluated by the
collision integral through these modes, without angular interpolation.
The once-scattered field has its own analytic flight integral. The >=2
reference is the total scattered response minus that first order; a tiny
residual requires a conditioning/convergence check before a relative-error
claim. Direct exterior specular reflection is separate; escaped volume
order 0 is zero for these inward boundary sources.

The phase function is the configured finite HG Legendre expansion. Its
quadrature normalization and minimum on [-1,1] are checked. A negative
truncation is rejected. Full HG is sampled independently by the Monte Carlo
reference. Increasing g requires higher phase and angular resolution; the
dense spatial reference has cubic eigenproblem cost. A smaller exact
azimuthal projection is available for the laterally integrated q=0 problem.

### Laterally integrated high-anisotropy calculation

With q=0 and axial source/receiver ports, only the azimuthal mean contributes.
The Legendre addition theorem gives

$$
\bar p(\mu,\mu')=\frac{1}{4\pi}\sum_{l=0}^{L}(2l+1)g^l
                        P_l(\mu)P_l(\mu').
$$

The angular weights become `2*pi*dmu`. This removes the azimuth grid from
the generalized eigenproblem while retaining the full configured polar
phase expansion and the analytic normal pencil. It computes the spatial
integral of the pencil-beam response, or equivalently the plane-parallel
response per incident area. It does not replace that beam by an isotropic
source and does not replace `mu_s` with `mu_s_prime`.

For example,
`HalfSpaceSettings(polar_nodes=80, azimuth_nodes=1, phase_degree=128, plane_parallel=True)`
resolves positive finite-HG approximations at g=0.9 and g=0.91 with much
smaller matrices. The azimuth reduction is exact for the stated observable;
polar and phase truncation still need convergence checks. The source and
receiver must be axial, and q must be zero. Spatial inversion and `banana()`
are explicitly rejected in this mode. `field()` then returns the azimuthal
mean rather than a full azimuthal field.

`experiments/validate_halfspace_materials.py` accepts a JSON profile file
with coefficients in cm⁻¹, converts them to m⁻¹ and compares this calculation
with full-HG histories. Missing absorption requires an explicitly labelled
positive scan. The report preserves the input metadata and source hashes.
The same histories traced at the smallest absorption can be reweighted to
larger absorption by `exp(-delta_mu_a*path)`. This changes only the escape
estimator; the stored base-history photon balance remains unchanged.
Group-index placeholders in a stationary profile do not establish a time
model. Literature indices and optical coefficients must keep their sample,
wavelength and model qualifications.

## Ports, spatial inversion and time

`SurfacePort.pencil()` is the normal directed source and its differential
directional receiver. `SurfacePort.aperture(alpha)` uses exterior
incident-flux quadrature weights proportional to `mu_out*dOmega_out`,
normalized to sum to one. The same object acts as a source or receiver.
For an aperture its `collection_factor = pi*sin(alpha)**2` converts mean
exterior radiance to collected flux per area. Reciprocity is therefore
checked after division by the receiving port's collection factor.

`response()` returns the laterally integrated complex escaped-flux spectrum
in orders `(0,1,>=2)`, per exterior incident photon. At q=omega=0 this is an
escape fraction into the aperture. The normal specular fraction is available
separately through `solve(source).specular_fraction`.

For axial angular ports, `radial_response()` performs a Hankel inversion.
Spatial source and receiver footprints can be normalized circular Gaussians;
they preserve the specified source directions. Output is escaped photons/m²
per incident photon. The receiver footprint averages the flux density;
its physical area is a separate factor. Offset, footprint, angular
quadrature, transverse cutoff and integration nodes each need appropriate
resolution checks.

`field()` retains the scattered angular radiance at depth; `beam()` holds
the analytic angular deltas separately. `depth_sensitivity()` evaluates the
laterally uniform absorption perturbation using the angular adjoint
pairing. `banana()` reconstructs local fields by azimuthal cosine moments
and Fourier-Bessel inversion, including both beam/diffuse cross terms.
It returns `dD/dmu_a(r)` in m⁻⁴ for finite Gaussian spatial ports. Source and
detector can be interchanged, preserving their footprints and angular
shapes. Integrating this kernel times a local absorption change over volume
gives the first-order change of measured flux density.

`time_bins()` applies the existing inverse-time convention to the
laterally integrated spectrum, with an optional Gaussian timing response.
Finite frequency bandwidth and spacing can produce ringing and periodic
copies; results are neither clipped nor renormalized. Full time-profile
accuracy needs a frequency convergence study for the chosen medium,
aperture, window and timing response.

The runnable example `experiments/halfspace_time_profile.py` separates the
transport solve from readout:

```sh
python experiments/halfspace_time_profile.py compute --output-directory result
python experiments/halfspace_time_profile.py plot --spectrum result/spectrum.npz --output-directory rebinned --timing-sigma-ns 0.2 --bin-width-ns 0.02 --time-start-ns -1
```

Choose a local output directory for each result. The example inserts the
checkout's `src` into its import path; the experimental modules need not be
installed in a wheel. Its plotting dependencies are in the `validation`
extra. In a prepared development environment no installation step is needed.

`compute` saves `spectrum.npz`, `profile.npz`, `run.json`, and PNG/SVG plots.
`plot` uses the saved spectrum without solving transport. `profile.npz`
contains `edges_ns`, `components_fraction` with columns 0/1/>=2, separate
`specular_bins_fraction`, and their sum `total_bins_fraction`. Entries are
photon fractions integrated over each bin, per one exterior incident
photon. Divide by bin width for a rate, or multiply by incident photon count
for expected counts. No conditional normalization to reflected photons is
applied. The prompt specular fraction is added analytically at t=0 with the
same Gaussian timing response.

Defaults illustrate a normal milk-like beam: mu_s=250 cm^-1, g=0.9,
mu_a=0.01 cm^-1, phase-index ratio 1.35 and group index 1.35. The absorption
and group index are specified assumptions, not an experimental calibration.
The CLI accepts coefficients in cm^-1 and converts them to the library's
m^-1. Its `--timing-sigma-ns` is a standard deviation, including any Gaussian
source/readout width; FWHM is approximately 2.355 sigma. Negative displayed
times can arise from this symmetric timing response.

The default solve has 192 ordinates, 301 frequencies from 0 to 60 rad/ns,
and frequency spacing 0.2 rad/ns. A measured local run took 27.50 s for the
spectrum and 2.7 ms for inversion into 210 bins. A 256-ordinate,
601-frequency refinement took 144.19 s. Reusing a spectrum for 550 new bins
took 4.0 ms for inversion and about 0.7 s including the figure. These are
local measurements rather than performance guarantees.

The uniform frequency grid repeats in time with period 2*pi/delta_omega;
the CLI rejects an output window at least that long. Finer bins do not
replace a sufficient frequency bandwidth. For a nonnegative physical
impulse, |H(omega)| <= H(0), and a Gaussian timing width sigma gives the
following omitted-band bound for one bin:

$$
|\delta B_k|\leq
\frac{H(0)\Delta t_k}{\sigma\sqrt{2\pi}}
\operatorname{erfc}\!\left(\frac{\omega_{\max}\sigma}{\sqrt2}\right).
$$

`run.json` records the corresponding maximum-bin and summed-bin bounds.
They bound the omitted Gaussian-filtered band only. Angular error and
frequency quadrature/aliasing need separate checks; sigma=0 has no such
Gaussian bound. Narrowing the timing response can require recomputing the
spectrum with a larger cutoff.

For the stated illustrative milk case at sigma=0.1 ns, the default and
refined profiles differ by 2.53e-6 in L1 normalized by the all-time scattered
charge over [-0.5,10] ns. The observed frequency-step change is 1.33e-6;
the omitted Gaussian-band L1 bound is 7.36e-8 absolute. A 300000-history
continuous full-HG comparison gives a maximum resolved-bin discrepancy of
2.24 standard errors. Near-zero order-1 tail bins are checked against the
absolute band bound, not a relative error against an unresolved MC zero.
This validates that particular broadened q=0 example. It does not establish
a measured group index or a finite-offset time profile at high g.

## Run and validation scope

```python
from lighthit.medium import Medium
from lighthit.experimental.halfspace import (
    HalfSpaceSolver, HalfSpaceSettings, SurfacePort,
)
from lighthit.experimental.halfspace_boundary import HalfSpaceBoundary

# Explicit synthetic inputs, not measured tissue coefficients.
medium = Medium(absorption_per_m=0.2, scattering_per_m=1.0,
                g=0.3, group_index=1.4)
kernel = HalfSpaceSolver(medium, HalfSpaceBoundary(n=1.4),
                        HalfSpaceSettings(6, 12, 8))
source = SurfacePort.pencil()
receiver = SurfacePort.aperture(0.7)
orders = kernel.response(source, receiver)
balance = kernel.solve(source).photon_balance()
radial = kernel.radial_response(
    source, receiver, [0.0, 0.5, 1.0],
    source_sigma_m=0.4, detector_sigma_m=0.4,
    q_max_per_m=12.0, q_nodes=48,
)
```

Tests: `python -m pytest -q tests/test_halfspace*.py`.
Independent validation:
`python experiments/validate_halfspace.py --output validation.json`.
The optional `--figures-directory` writes the sensitivity image and arrays
to the supplied directory. The report records optical inputs, quadrature
settings, photon counts, seeds, source hashes, runtime versions, statistical
errors and all failures.

For the synthetic medium above, the validation uses n=1 and n=1.4, normal
incidence, exterior half-angles 0.7 and pi/2, q=0/0.8/2 m⁻¹ and
omega=0/0.1 ns⁻¹. It compares one million independent point-beam histories
per boundary with three angular/phase resolutions. Separate two-million
history Gaussian-beam samples check spatial flux and three local absorption
cells. Path lengths in those cells provide an independent derivative
estimator, including reflected flight segments. Selected source/detector
swaps, exact first-collision limits, photon balance and absorption finite
differences are also checked. Unit tests compare once-scattered time bins
with the exact Gaussian-smoothed exponential for isotropic scattering.

This establishes a synthetic reference. It does not certify arbitrary g,
all source-detector separations, all time windows or experimental tissue
parameters. The validation sequence for an experimental model remains:

1. Verify the matched boundary with the true normal beam, separated orders,
   independent Monte Carlo and absorption/escape balance.
2. Apply n=1.4 with the same beam, Fresnel input and output, and detector
   aperture. Verify both the full incident-photon balance including direct
   specular reflection and the balance conditional on entry.
3. Swap the source and detector ports and verify weighted reciprocity.
4. Validate selected local absorption sensitivities with an independent
   path-length reference or a solver with localized material perturbations.
5. Add the experimental beam footprint, aperture, optical coefficients,
   wavelength, timing response and calibration before comparing numbers.

Photon balance at roundoff level is an algebraic consistency check, not a
bound on angular discretization error. For the synthetic g=0.3, n=1.4 case,
8 polar nodes per interval and L=10 give an escape fraction 0.1296974582
from the internal balance, versus 0.1295900510 from the continuous receiver
integral (128 receiver nodes). Increasing the bulk polar resolution brings
these two integrals together. At 128 polar nodes per interval and L=18 the
values are 0.1296069775 and 0.1296069442, respectively. Refining the receiver
alone does not resolve the bulk error. The Fresnel threshold needs its own
polar convergence study; a tiny balance residual must not be quoted as a
physical accuracy claim. The observed refinement changes likewise are not
rigorous error bounds.

The scalar boundary assumes unpolarized radiance. A comparison that resolves
polarization requires a vector transport model. No bulk scattering
coefficient, anisotropy, experimental aperture or measurement is inferred
from n alone.

## References

* M. Machida, [three-dimensional half-space transport with anisotropic
  scattering](https://arxiv.org/abs/2203.12327).
* Pharr, Jakob and Humphreys, [specular reflection and
  transmission](https://www.pbr-book.org/4ed/Reflection_Models/Specular_Reflection_and_Transmission)
  and [dielectric radiance transformation](https://www.pbr-book.org/4ed/Reflection_Models/Dielectric_BSDF).
