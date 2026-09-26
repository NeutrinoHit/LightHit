# Prompt transport: scattering orders 0 and exactly 1 without the RTE cache

`TransportKernel.transport_prompt(source)` returns the prompt signal of a
Cherenkov track or of a shower / light-element source at every optical module:
unscattered light (order 0) and light scattered **exactly once** (order 1), in
the ordinary LightHit time bins. Orders `>= 2` are not computed and are
recorded as *not computed* (`computed_orders == (0, 1)`), never as zero.

Nothing in this path builds, loads or applies the multipole, directional or
spectral-folded RTE caches, and nothing calls `TransportKernel.transport`
(`tests/test_prompt.py::test_prompt_never_touches_the_rte_cache` replaces all
of them by functions that raise). `transport` and `TransportResponse.select`
are unchanged.

The decomposition is that of chapter 6: with the free propagator `T` and one
collision `V`, order 0 is `T` and order 1 is `T V T`. Chapter 6 computes the
full-HG first order for a *point* source and a *scalar* receiver
(`lighthit.single`); a directional module needs the distribution of arrival
directions, and an extended Cherenkov source needs the integral over emission
points and cone azimuths. Those two steps, and the numerical algorithm, are
what this note adds.

## 1. One directed photon, one directional module

A photon leaves `x` in direction `s0` at time `t_e`. A point module at `R`
records a photon travelling in direction `s1` with efficiency `A(s1 . n)`,
`n = -orientation` (the ballistic convention). Put `r = |R - x|`,
`r_hat = (R - x)/r`, `cos(theta) = s0 . r_hat`,
`e_perp = (r_hat - cos(theta) s0)/sin(theta)`.

The photon survives to distance `l` along `s0` with `exp(-mu_t l)`, scatters
there with probability `mu_s dl` into `p_HG(s0 . s1) dOmega_1`, survives the
distance `rho` to the module with `exp(-mu_t rho)`, and the module collects the
solid angle `1/rho^2` per unit area. Hence

    N1 = int_0^inf dl mu_s exp(-mu_t (l + rho)) p(s0 . s1) A(s1 . n) / rho^2,
    t  = t_e + (l + rho)/v_g.

`mu_t = mu_a + mu_s` enters both free flights: a photon that scatters a second
time has left order 1, whether or not it is absorbed later.

**Scattering-angle variable.** The scatter point, `s0`, `r_hat` and `s1` are
coplanar. Let `chi` be the scattering angle (`s0 . s1 = cos chi`). Elementary
geometry of the triangle `x`, scatter point, `R` gives

    l    = r cos(theta) - r sin(theta) cot(chi),
    rho  = r sin(theta) / sin(chi),
    S    = l + rho = r cos(theta) + r sin(theta) tan(chi/2),
    s1   = cos(chi) s0 + sin(chi) e_perp,
    dl / rho^2 = dchi / (r sin theta),

so that exactly

    N1 = mu_s / (r sin theta) * int_theta^pi dchi exp(-mu_t S(chi)) p(cos chi)
             * A(cos(chi) c1 + sin(chi) c3),        c1 = s0.n,  c3 = e_perp.n.

The `1/rho^2` of the second flight is absorbed by the change of variables; the
phase function depends on the integration variable alone; the scatter at the
source (`chi = theta`) is the front `S = r`, and `chi -> pi` is the tail. With
`y = tan(chi/2)` the path is linear, `S = r cos(theta) + r sin(theta) y`, and
`cos chi = (1-y^2)/(1+y^2)`, `sin chi = 2y/(1+y^2)` need no trigonometry. The
HG factor becomes `(1+y^2)^{1/2} / ((1-g)^2 + (1+g)^2 y^2)^{3/2}` up to
constants: a peak at `y_g = (1-g)/(1+g)` and a `1/y^2` tail.

With `A == 1` this is exactly the directed density of
`lighthit.single.single_scattering_rate` after `dS = (S - r nu)/rho dl`; the
reference implementation reproduces `single_spectrum` and `single_bins` to
`1e-14` (`prompt_reference.py`), and an independent next-event Monte Carlo of
the `l` integral agrees with the directional acceptance
(`test_reference_directional_acceptance_against_monte_carlo`).

The only singular factor left is `1/sin(theta)`: a photon aimed at the module
gives `N1 ~ 1/theta`. It is integrable over emission directions, and it is the
reason the source integrals below need care.

**Acceptance.** `A` is the module response rebuilt from the kernel's measured
Legendre coefficients (`kernel.acceptance()`), exactly as the full path and
the ballistic kernel use it; for BGVD it is a cubic. A polynomial of degree
`L_A` in `x = cos(chi) c1 + sin(chi) c3` expands into monomials
`c1^i c3^j cos^i(chi) sin^j(chi)`, `i + j <= L_A`, so a group of pencils is
carried exactly by the moments `sum w c1^i c3^j` (10 for a cubic).

**Spectrum.** For wavelength node `k` the photon count of an element is
`c0 lambda^-2 + c2 lambda^-2 n_ph^-2` (the two Cherenkov fields of
`SpectralLightElements`), times the quadrature weight, the detector spectral
efficiency and the effective area -- the same factors as the full path. The
Cherenkov cone is frozen at `reference_phase_index`, as in the full path and
its ballistic term. `mu_a`, `mu_s`, `v_g = c/n_g` are the medium's values at
the node.

## 2. The Cherenkov source

An element is a chord `x(a) = x0 + a u`, `0 <= a < h`, emitting per unit
length along a cone `s0 . u = cos(theta_C)`, azimuth `phi` uniform, at time
`t(a)` linear along the chord. Order 1 at a module is

    N1 = int_0^h da q(a) int dphi/(2 pi) N1_pencil(x(a), s0(phi), t(a)).

`theta(a, phi)` vanishes only at the Cherenkov root `a*` (where the cone of
`x(a*)` passes through the module, the ballistic root) and `phi = 0`. Near it
`theta^2 ~ (b dA / r*^2)^2 + sin(gamma) sin(theta_C) phi^2`, so the double
integral has an integrable `1/|distance|` point singularity; after the azimuth
integral the `a` integrand is `~ log|a - a*|`.

**Straight segments (CherenkovTrack; small element sources).** Tensor rule:

* azimuth: `phi = (theta_min/k) sinh(w)` about the ring's closest approach to
  the module direction, `k^2 = sin(gamma) sin(theta_C)`, Gauss-Legendre panels
  in `w` (uniform nodes when the ring stays 0.5 rad away). The substitution
  turns `1/theta` into a bounded integrand;
* along the segment: Gauss-Legendre panels graded geometrically towards `a*`
  (log singularity) and subdivided so that each panel spans at most
  `front_width_ns` of *front time* `t_f(a) = t(a) + r(a)/v_g` for every
  wavelength (see section 3).

**Showers.** A 100 GeV electron has ~2e5 steps with directions spread over
tens of degrees; per-element evaluation at every module is far too expensive,
and grouping steps by position and direction barely compresses them. What
compresses is the emitted angular distribution per spatial cell:

1. *Compression (once per event).* Every element's ring is sampled at
   `ring_nodes` azimuths (sub-segments of at most `cell_m/2`) and deposited
   into bins (position cell `cell_m`, equi-angular cube-sphere pixel). Each
   bin ("pencil") keeps the weights of both fields, the first moments of
   position, time and direction, and the second moments of direction
   (including the exact variance `(2 pi sin(theta_C)/N)^2/12` of the arc each
   node stands for) and of position (including the sub-segment's own spread).
   The geometry and moments use `coefficient0` weights for both fields; the
   second field retains its separate total spectral weight.
2. *Apparent shape.* Seen from a module, a pencil's directions form a 2D
   distribution about its mean with covariance
   `C = C_dir + P C_pos P / r^2` (`P` projects perpendicular to the line of
   sight), omitting position-direction cross covariance. Where
   `theta < smoothing_factor * sqrt(max eig C)` the singular
   factor is replaced by the mean of `1/theta` over a 2D Gaussian with
   that covariance,

       E[1/rho] = (2/sqrt(pi)) int_0^inf ds prod_i (1 + 2 s^2 mu_i)^(-1/2)
                  exp(-s^2 sum_i p_i^2/(1 + 2 s^2 mu_i)),

   (from `1/rho = (2/sqrt(pi)) int exp(-s^2 rho^2) ds`), and the regular part
   `I(theta) = int_theta^pi ...` is evaluated at the harmonic-mean angle
   `1/<1/theta>` -- exact when `I` is linear in `theta`, which it is to first
   order because `dI/dtheta = -p(cos theta) A(...) exp(...)`. Outside, the
   second-order correction `(1/2) C_ij (3 p_i p_j - d^2 delta_ij)/d^5` is
   applied. Without these two steps the bright modules on the Cherenkov cone
   of a real electron are underestimated by 12 %; with them by 1.5 %,
   converging with the pixel size (section 6).

3. *Per-module grouping.* Pencils are merged into groups of equal
   (arrival proxy `tau = t + r/v_max`, `log r`, angle `theta`) with relative
   widths `merge_*`; the singular weight `w <1/sin theta>/r` and the
   acceptance moments are summed exactly; `theta`, `r`, `t` are weighted means.
   Groups with equal (`tau`, `r`) form one batch (section 3).
4. *Screening.* A coarse compression (2 m cells, 11 deg pixels) is evaluated
   at every module; the full-resolution pass runs where
   `q0 + 3 q1_coarse >= threshold`. The coarse/fine ratio on the fine modules
   is reported (and the fine set widened if it ever violates the margin);
   screened modules keep the coarse charge and no bins (they are inactive).

These are grouped compression approximations, expected to be benign for the
validated ultra-relativistic electron and muon cases. For materially varying
beta or difficult near-cone geometry, use `element_method="ring"` or
`"segment"` as a convergence/control calculation.

`element_method="ring"` evaluates every element as point cones (no
compression) and is the reference for the grouped algorithm;
`element_method="segment"` applies the direct segment quadrature to every element.

## 3. Time bins

The required output is photoelectrons in the actual bins of
`KernelConfig.relative_time_edges_ns`. Densities are never sampled at bin
centres.

**Batches.** All pencils of one emission point (all azimuth nodes of one `a`;
all groups of one (`tau`, `r`)) share `r`, the front `S = r` and
`t = t_e + S/v_g`. Their `lambda`-independent profiles
`G(S) = (1/(r sin theta)) p(cos chi) A(x) * 2/((1 + y^2) r sin theta)` (per
unit path) are summed on one grid `S = r + sigma sinh(u)`, uniform in `u`
(linear at the front, geometric in the tail), `sigma` half the forward scale
`r sin(theta_min) max(y_g, tan(theta_min/2)/2)`. The pencil values are closed
forms at the grid nodes.

**Exact exponential, quadratic remainder.** For wavelength `k` the time
density is `exp(kappa t) H(t)` with `kappa = -mu_t v_g`, because
`S = v_g (t - t_e)`: the attenuation is exact and only
`H = mu_s v_g (s0w G0 + s2w G2) exp(-kappa t_e)` is interpolated, by
quadratics in `t` over pairs of grid intervals. Every panel integral is
`E_a width (alpha I0(z) + width (beta I1(z) + width gamma I2(z)))`,
`I_k(z) = int_0^1 x^k exp(z x) dx`, `z = kappa width`, with Taylor series for
`|z| < 1`, so short panels are free of cancellation.

**Prefix events.** `kappa` depends on the wavelength only, so all batches of a
module share it. A panel covering at most `direct_bins` bins is integrated bin
by bin over the actual edges; a longer panel adds its global quadratic
`A + B t + C t^2` to the bin where it starts and removes it where it ends,
with exact partial-bin corrections. One prefix pass per module and wavelength
integrates `exp(kappa t)(A + B t + C t^2)` over every bin. No batch loops over
all bins. The integrated charge is the sum of the same panel integrals over all
times; the tail beyond `mu_t,min (S - r) = tail_exponent` (25) is dropped
(relative bound `e^-25`).

**Fronts.** The density jumps at a batch's front `t_f`. A quadrature cell of
emission points spreads its fronts over `W = |dt_f/da| * da` (tracks, rings)
or `sqrt(12 var(tau))` (groups); the step `H(t_f) exp(kappa(t - t_f))` is
spread over `W` as a linear ramp normalised to the exact step mass, and the
continuous remainder is deposited as is. `W -> 0` under refinement; without it
a finite set of fronts imprints a staircase on the bins.

## 4. Order 0

`ballistic_directional_fast` (or its NumPy twin) with exactly the field, cone,
medium band, acceptance coefficients, time origin and edges of the full path:
charges for every computed module, bins for the active ones. The prompt and
full ballistic components agree to rounding (`2.5e-16` relative on a 120 m
track; identical totals on the G4 events).

## 5. Work, memory and what is (not) tabulated

| Stage | Work | Depends on |
|:--|:--|:--|
| compression | `N_el * ring_nodes` deposits | source only |
| module selection, order 0 | `N_OM * N_el * N_lambda` | geometry |
| per-module grouping | `N_OM_fine * N_pencil` | module, pencil |
| batch profiles | `groups * N_S` closed forms | module, group |
| deposits | `batches * N_S * N_lambda` | module, wavelength |
| prefix pass | `N_OM * N_lambda * N_bins` | module, wavelength |

`N_S` is ~60-150 path nodes. The only reusable tables are `exp(kappa_k e_j)`
at the bin edges and `I_k(kappa_k width_j)`, `N_lambda * N_bins * 4` numbers
(a few kB); they depend on the medium and the bin edges only and contain no
transport. Per thread: accumulators `N_lambda * N_bins * 7` doubles, path
buffers, and a per-module hash of groups (a few MB). Pencils: 23 doubles each
before reduction, ~20 after (~70 MB for 4e5 pencils).

## 6. Validation

`docs/research/prompt-transport-validation.md` has the numbers from
`scripts/prompt_benchmark.py`, `scripts/prompt_full_refinement.py` and the
reference runs. The lightweight tests in `tests/test_prompt.py` cover:

* the `single.py` directed and isotropic limits;
* forward and backward Monte Carlo;
* zero scattering;
* ballistic identity;
* quadrature convergence;
* element-algorithm agreement;
* semantics, save/load and the viewer;
* cache independence.

In short:

| Case | Order 0 | Order 1, prompt | Order 1, full path | Warm time, prompt vs full |
|:--|:--|:--|:--|:--|
| 120 m track | identical to the full path | within 1e-4 of its refined value, confirmed by the backward Monte Carlo | within 0.1 to 0.6 % in charge; order-1 bins differ by 7 to 12 % (L1) | 5.9 s vs 11.8 s |
| G4 electron, vs uncompressed ring reference | identical | -3.5 % to +1 % | -8 % to +3.4 % | 6.6 s / 1 GB vs 402 s / 13.6 GB |
| G4 muon, vs ring | identical | -2.3 % to -0.1 % | -2.7 % to +6.7 % | 14 s vs no whole-event run in 15 GB; 451 s on 10 slabs |
| Thin synthetic showers | identical | within 3 % | errs by up to 54 % | |

The full path's errors come from its finite angular representation.

## 7. Limitations

* The pencil compression of large element sources is an approximation with a
  measured, mostly negative, bias on modules on the Cherenkov cone (-1.5 % at
  default resolution for the 100 GeV electron, -0.5 % at 4x finer pixels);
  `element_method="ring"` is the uncompressed reference.
* The Cherenkov cone is frozen at `reference_phase_index`, as in the full
  path.
* Point modules: the order-1 kernel has an integrable `1/theta` singularity
  for photons aimed at the module; modules must not sit on a source segment
  (`min_distance_m`).
* Time bins inherit the smoothing of discrete fronts over a quadrature cell
  (`front_width_ns`); the measured bin errors are in the validation note.
* Screening evaluates modules whose coarse order-0+1 estimate is far below
  threshold with the coarse pass only (`order1_level == 1`). Their order 1 is
  accurate to roughly 30 %, not to the full-resolution level.
* The numba kernels are compiled on first use: about 170 s (segment path),
  250 s (grouped) and 80 s (ring) with an empty cache, once per
  installation (`cache=True`).
* Only orders 0 and 1 are computed. There is no estimate of order >= 2, which
  is significant for the bright modules at 5 ns resolution. The threshold and
  `active` therefore count 0+1 only, and the viewer disables the
  "Total"/"Two or more" views.

## 8. Backward Monte Carlo (experimental)

`lighthit.experimental.prompt_backward` is an adjoint next-event estimator of
the same order-1 quantity, centred on the module. It samples the scatter
point as `y = R - b s2`, and each segment whose Cherenkov shell contains `y`
contributes its analytic direct field. It is used as an independent,
unbiased check, and its results and cost are in section 8 of the validation
note.

* **Tracks:** a few 1e6 samples per module give 0.2 to 0.3 % in about 0.5 s.
* **The G4 electron:** each sample has to find the ~12 of 211,576 millimetre
  steps whose shell contains `y`. That makes it two to three orders of
  magnitude slower than the deterministic path at 1 % accuracy, although its
  bin shapes are the most accurate of any method.
