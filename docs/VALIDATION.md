# Checks performed for the first LightHit delivery

Snapshot: 2026-09-16. Every physics run listed here used only the public
synthetic medium. Parameters and numerical bounds are set in
`examples/point-green.toml`.

## Main run

One photon, direction +z. Detector at 20 m, 60° from the initial ray.
$\mu_a=0.04\ \mathrm m^{-1}$, $\mu_s=0.05\ \mathrm m^{-1}$, $g=0.7$,
$n_g=1.35$, 450 nm.

Spectrum: 241 frequencies from 0 to 1.2 rad/ns. Multiply-scattered part:
$L=32$, $J=160$, $k_{\max}=6\ \mathrm m^{-1}$, panel width
$0.04\ \mathrm m^{-1}$, order 8 (1200 $k$ nodes), taper on. The exact first
order uses the full HG function. Charge units are m⁻² per photon.

| Quantity | Result |
|---|---:|
| Front time | 90.0623057 ns |
| $Q_0$ | 0 |
| $Q_1$ | $8.6745098588\times10^{-6}\ \mathrm m^{-2}$ |
| $Q_{\ge2}$ | $2.3222056021\times10^{-5}\ \mathrm m^{-2}$ |
| Total $Q$ | $3.1896565880\times10^{-5}\ \mathrm m^{-2}$ |
| Full spectrum, median of 3 independent runs | 7.0193 s |
| Full spectrum, individual timings | 7.0107, 7.0193, 7.3961 s |
| Charge only, median of 3 runs | 0.2369 s |
| Readout without smearing | 0.1424 s |
| Readout, sigma = 3 ns | 0.0847 s |
| Peak RSS, whole CLI process, with plots | 159.45 MiB |

Linux x86_64, Python 3.13.5, NumPy 2.3.5, SciPy 1.17.0.
`OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=1`. These are sandbox timings
from this specific run, not a speed estimate for a Mac. There is no hidden
transport cache between calls. RSS is the whole-process high-water mark,
including dependencies and plotting.

## Numerical differences

Refined setting: $L=48$, $J=208$, $k_{\max}=8\ \mathrm m^{-1}$, panel width
$0.025\ \mathrm m^{-1}$, order 10. Same frequency grid. One refined
spectrum: 28.354 s.

| Comparison, main vs. refined | Difference |
|---|---:|
| Relative charge difference | $3.1953\times10^{-6}=0.00031953\%$ |
| L1 difference, unsmeared bins / refined $Q$ | $0.00866446=0.866446\%$ |
| L1 difference, sigma = 3 ns bins / refined $Q$ | $6.4345\times10^{-5}=0.0064345\%$ |

Several parameters changed at once. These numbers do not isolate the error
of any single truncation and do not check the frequency step or band on
their own.

Main run: negative mass of unsmeared bins / $Q$ = 0.00177381 (0.177381%);
absolute unsmeared signal before the front / $Q$ = 0.00440734 (0.440734%).
At sigma = 3 ns, negative mass / $Q$ = $3.0453\times10^{-6}$. No value was
ever removed and the profile was never renormalized. A Gaussian readout
physically produces a tail before the front; its mass is not a test of
strict causality.

## Code checks

48 tests passed, including:

- the exact free tail against an independent dense $B$-matrix solver;
- the isotropic all-orders spectrum against an independent infinite
  sine-transform inversion;
- the $k=0$ monopole and photon-number conservation under scattering;
- an independent coordinate-space first order and HG normalization;
- general rotation, time shift, and linearity in photon number;
- batched observations, pure ballistic transport, singular geometry, and
  invalid input;
- the private adapter against a synthetic double, without importing
  `OpticalModule`.

Notebook: 26 cells, 13 executable. All ran without error. Full execution of
this notebook version took 19.64 s, including extra examples, checks, and
plots.

The wheel was built without downloading dependencies, installed into a
separate directory, and imported from there outside the source tree. The
charge computed from the wheel matched. This is not a check of installing
every dependency into a completely empty environment; no network install
from the README commands was needed in the sandbox.

Full JSON, logs, NPZ files, and the executed notebook ship as a separate
results archive. The source notebook in this delivery carries no outputs.

## Not checked

Real private water and its dependencies and tables (a separate,
preliminary pass against one wavelength of the real table is in
`docs/chapters/04-bgvd-water.qmd`), a finite OM, the narrow neighborhood of
the direct ray, large distances without a grid refinement, every possible
$g$, an independent full Monte Carlo of the directed response, and an
absolute error bound on the unsmeared time front. A transport cache, cones
and tracks, shower processing, and an inhomogeneous medium are not
implemented in this delivery.
