# Prompt transport: validation and timing

Numbers for `TransportKernel.transport_prompt` (orders 0 and exactly 1)
against independent references and against the full LightHit transport.
The method is in `prompt-transport.md`. Every number here can be regenerated
with the commands in section 9. Per the separation rule in `PROVENANCE.md`,
the raw outputs of runs against the private BGVD model (responses,
per-module JSON) stay outside Git. Only the aggregate figures below are
recorded.

These measurements used the frozen reference Cherenkov cone. The optional
per-wavelength `cone_model="spectral"` is outside this validation set.

## 1. Set-up

* **Machine:** 4 CPUs, 15 GB RAM, Linux x86-64.
* **Software:** Python 3.11.15, NumPy 2.4.6, SciPy 1.17.1, numba 0.67.0 (4
  threads), LightHit 0.2.0a8.
* **Detector and medium:** the BGVD model, dataset 2021 (`load_bgvd_model`,
  g = 0.9), with the production `KernelConfig`: 9 wavelength nodes; 5 ns bins
  from -60 to 740 ns relative to each module's earliest direct arrival;
  threshold 0.01 p.e.
* **Pose:** every source is placed at `[5, -269.5, 330]` m with direction
  `[0.35, -0.2, 0.915]` and t = 0, identically for both methods.
* **Full path:** reads a production directional cache built once with
  `cache_policy="require"`. Building it took 2381 s for the directional cache
  plus 504 s for the folded cache, with a 13.4 GB peak and 17 GB on disk. That
  cost is not included in any timing below.
* **External inputs used (SHA-256):**

| Asset (private validation bundle) | SHA-256 |
|:--|:--|
| `bgvd-model-master.zip` | `0f9016168ab18023cabe07e4c4e7824eff753b4caeba5de4546ed76b33039fc0` |
| `sim_e_100GeV_10.h5` | `f496788e96093951c1c068980048d687f8b942363ba4089ca93f7c50bb8f1c05` |
| `sim_mu_100GeV_10.h5` | `da3ff7d40552ffaf7f8202850354e028a148ad7a47c3bd84839c44d53d5801b2` |
| unpacked model tree (`tree_sha256` in the benchmark script) | `238728a9a6a972378e7c833ec82ffd46abdc2f86ed09422f6188d857f225e274` |

Charge differences are quoted relative to the reference. Where many modules
are compared, the error scale is `max(q_ref, 0.01 p.e.)`. "Bins L1" is
`sum_k |b_k - b_ref,k| / sum_k b_ref,k` over the 160 window bins of a module.

### References

The comparisons below use three validation references:

* **Ring.** `transport_prompt(..., element_method="ring", ring_a_step_m=0.02)`
  treats every element as a chain of point Cherenkov cones 2 cm apart. There
  is no grouping or pencil merging, and each cone uses the per-pencil
  quadrature that `tests/test_prompt.py` checks against `single.py`,
  adaptive quadrature and Monte Carlo.
* **Refined prompt.** The segment rule at `a_gauss=6, front_width_ns=0.6,
  phi_uniform=48, path_step=0.05, path_step_tail=0.1`. It is used for tracks.
* **Backward Monte Carlo.** The adjoint next-event estimator in
  `lighthit.experimental.prompt_backward` (section 8). It uses an independent
  integration and sampling formulation while reusing common source, spectral,
  HG and binning utilities; its estimator is unbiased.

## 2. Lightweight tests (`tests/test_prompt.py`, 19 tests, about 10 s warm)

| Requirement | Test |
|:--|:--|
| A: zero scattering | `mu_s = 0` gives order 1 exactly 0 and order 0 unchanged |
| B: ballistic identical | prompt order 0 (charges, bins, time origins) equals the full path's to `rtol 1e-12` |
| C: point and scalar limit | reference pencil = `single.py` directed point (`1e-9`); fast pencil vs `single.py` scalar limit (`2e-4`, bins L1 `5e-4`); isotropic point vs `single.py` (ring and segment rules) |
| D: quadrature convergence | pencil and track convergence under path-step / front-width / azimuth refinement |
| E: element rules agree | grouped vs ring vs segment on a synthetic shower |
| Monte Carlo | reference pencil vs next-event Monte Carlo; backward estimator vs prompt on a track and a shower |
| Semantics | order >= 2 is "not computed" (`OrderNotComputedError`), never zero; save/load round trip; viewer payload |
| Independence | cache classes, `transport`, `build` and the folded cache are monkeypatched to raise; the prompt path still runs |
| Full path unchanged | `transport()` and `select("0+1")` give the same results as before |

The expensive real-data comparisons are not in pytest. They run through
`scripts/prompt_benchmark.py`.

## 3. Track, 120 m (`CherenkovTrack`, beta = 1)

| | first call | warm call | peak RSS |
|:--|--:|--:|--:|
| prompt | 6.21 s | 5.88 s | 221 MiB |
| full (cache loaded) | 16.6 s | 11.8 s | 3.0 GB |

* **Order 0:** identical.
* **Default vs refined prompt:** order 1 differs by -1.2e-4 relative on the 12
  brightest modules (0.011 to 0.31 p.e.).
* **Backward Monte Carlo** (4e6 samples per module, standard error 0.2 to
  0.4 %): 11 of the 12 modules agree with the refined prompt to within 2
  sigma. The exception is module 88 in one run: -0.75 % at 0.23 % s.e.
  (3.2 sigma). Three other runs on the same module gave -0.17 %, -0.17 % and
  +0.32 %; the standard-error estimate is itself noisy for this heavy-tailed
  estimator. Bins L1 between the Monte Carlo and the prompt path is 0.1 to
  0.9 %.
* **Full path vs prompt:**
  * Bright-module order 1 agrees to 0.1 to 0.6 %; the total differs by -0.8 %.
  * Order-1 bins L1 is 7 to 12 % of the full window. This difference comes
    from the full path's smoothing: the Monte Carlo agrees with the prompt
    bins to below 1 %.
  * Modules 93 and 94, past the end of the track, get 0.002 p.e. from the
    prompt path and 0.0085 p.e. from the full path. The prompt value matches
    independent adaptive integration to 4e-6, so the difference is a
    cone-truncation artefact of the full path.
* **Active modules:** 21 for prompt, 29 for full. The full path's threshold
  also counts order >= 2.

## 4. Synthetic showers (`SyntheticShower.gaussian`, 500 m charged length)

| elements | prompt first / warm | prompt RSS | full first / warm | full RSS |
|--:|--:|--:|--:|--:|
| 1,000 | 0.83 / 0.27 s | 227 MiB | 81 / 56 s | 7.7 GB |
| 10,000 | 0.75 / 0.48 s | 282 MiB | 105 / 95 s | 9.5 GB |
| 100,000 | 2.17 / 1.72 s | 305 MiB | 178 / 171 s | 11.1 GB |

These showers have a very small angular spread, so order 0 is zero on most
modules and order 1 is dominated by sharp, forward-peaked structure. That is
the worst case for any finite angular representation.

The table below compares order 1 against the ring reference on the union of
the 6 brightest modules by prompt and by full: 7 modules, 0.064 to 0.34 p.e.

| elements | prompt (grouped) | backward MC | full |
|--:|:--|:--|:--|
| 1,000 | -2.3 % to -0.8 %; bins L1 0.8 to 2.3 % | within 1 sigma (s.e. 0.5 to 1 %); L1 0.3 to 0.9 % | -54 % to +32 %; L1 6 to 57 % |
| 10,000 | -2.9 % to +0.2 %; bins L1 0.4 to 3.0 % | within 1 sigma (s.e. 0.7 to 1.6 %); L1 0.2 to 1.1 % | -54 % to +32 %; L1 9 to 58 % |

* **Full path:** also produces negative order-1 charges (for example
  -0.014 p.e.) on modules the cone misses. The prompt order-1 total is about
  10 % above the full path's for all three sizes (1e3 to 1e5 elements).
* **Prompt path:** the residual is the known small negative bias of the pencil
  compression (section 5).

## 5. Real 100 GeV electron (event 0)

The event has 223,080 G4 steps, of which 211,576 are above the Cherenkov
threshold at every wavelength (median step 0.5 mm, 519 m charged length).

| method | first call | warm call | peak RSS |
|:--|--:|--:|--:|
| prompt | 7.24 s | 6.62 s | 962 MiB |
| full (whole event) | 399.5 s | 402.1 s | 13.56 GB |
| full, 2 m slabs sharing the origin | 351.6 s | 367.4 s | 10.1 GB |

**Prompt warm-call breakdown (6.61 s):**

| stage | time |
|:--|--:|
| order 0 (9 wavelengths, all modules) | 2.46 s |
| selection | 0.75 s |
| order-1 screen | 0.78 s |
| order-1 compression | 1.49 s |
| order-1 evaluation | 0.97 s |
| order-0 bins | 0.12 s |

**Order 0 and totals:**

* Order 0 is identical (2.8992 p.e. total).
* Order-1 total: 1.8845 p.e. (prompt) vs 1.8955 p.e. (full), -0.6 %. The 0+1
  total differs by -0.23 %.
* Active modules: prompt 24, full 26, both 24.
* The slab version of the full path reproduces the whole-event full path to
  1e-4 in charge and 0.7 to 1.1 % bins L1, which validates the slab method
  used for the muon.

**Order 1 against the ring reference, 8 brightest modules:**

| module | ring (p.e.) | prompt | full | backward MC (1.28e5 samples) |
|--:|--:|--:|--:|--:|
| 89 | 0.3077 | -1.5 % | +1.2 % | +0.5 % +- 1.0 % |
| 269 | 0.2295 | -1.7 % | -6.3 % | -0.6 % +- 0.9 % |
| 270 | 0.1847 | -2.0 % | -8.0 % | -0.1 % +- 1.2 % |
| 268 | 0.1724 | -2.1 % | -0.2 % | +0.6 % +- 0.8 % |
| 88 | 0.1572 | -2.3 % | +1.9 % | -1.4 % +- 0.9 % |
| 271 | 0.1357 | -3.5 % | +3.4 % | +0.3 % +- 1.3 % |
| 54 | 0.1038 | -0.7 % | -1.0 % | -0.6 % +- 1.1 % |
| 55 | 0.1016 | +1.0 % | -5.1 % | -0.5 % +- 1.2 % |
| **bins L1** | | **1.0 to 3.5 %** | **7.0 to 15.6 %** | **0.3 to 1.7 %** |

* **Full path:** also has negative order-1 bins on these modules.
* **Prompt grouped bias:** -1.5 % on module 89 at the default resolution.
  Finer pixels reduce it to -0.6 % (`pixel_face=64`) and -0.4 %
  (`pixel_face=128`).
* **Screening:** 27 modules are evaluated at full resolution and 617 by the
  coarse pass only. On the modules that received both, the coarse/fine ratio
  is 0.71 to 1.05. The largest charge a module left at the coarse level
  carries is 0.0072 p.e., below the 0.01 p.e. threshold.

## 6. Real 100 GeV muon (event 0)

The event has 77,194 G4 steps along about 100 m of the muon track plus its
secondaries.

**The full path cannot transport this event as a whole on 15 GB:**

* The whole-event process is killed for running out of memory (exit 137).
* The 10 m slabs that share the whole event's time origin (the method that
  worked for the electron) fail the full path's own check: "2 OMs outside the
  validated directional radial range have a conservative estimate above
  threshold". Each slab has to carry the event's earliest element, so its
  apparent extent becomes the whole event.

**Both methods on the same 10 plain slabs.** Each slab has its own time
origin, so charges are compared summed over slabs and bins slab by slab
(`prompt-slabs` / `full-slabs` in the benchmark):

| method | call | peak RSS |
|:--|--:|--:|
| prompt, whole event | 14.9 s first / 14.0 s warm | 1.26 GB |
| prompt, 10 slabs | 9.4 s | 0.64 GB |
| full, 10 slabs | 450.7 s | 10.6 GB |

**Order 0 and totals (slab sums):**

* Order 0 is identical (2.9769 p.e. total, relative difference 1e-9).
* Order-1 total: prompt -2.5 % against full.
* Active modules: prompt 29 (whole event), full 29 (slab union).
* The full path has 12 modules with a negative order-1 charge, and -0.018 p.e.
  of negative bins in total.
* The prompt whole event and the sum of prompt slabs agree to within 1.6 % on
  the bright modules. That spread is the grouping resolution.

**Order 1 against the ring reference (whole event), 10 brightest modules:**

| module | ring (p.e.) | prompt | full (slab sum) |
|--:|--:|--:|--:|
| 265 | 0.3739 | -0.1 % | +2.1 % |
| 266 | 0.2674 | -0.3 % | -0.3 % |
| 90 | 0.1463 | -0.9 % | +3.2 % |
| 267 | 0.1461 | -0.7 % | +0.1 % |
| 89 | 0.1464 | -0.5 % | -0.2 % |
| 268 | 0.0828 | -0.9 % | +0.9 % |
| 269 | 0.0590 | -2.3 % | -2.7 % |
| 270 | 0.0423 | -0.6 % | +2.1 % |
| 91 | 0.0305 | -1.3 % | +6.7 % |
| 88 | 0.0319 | -1.6 % | -0.1 % |

The prompt bins against the ring reference have L1 0.1 to 2.5 %.

## 7. Full path under refinement (synthetic medium)

`scripts/prompt_full_refinement.py` fixes a synthetic near-Baikal medium, a
3-string detector, a 60 m track and a 2000-element shower. It then raises the
full path's angular degrees `(L, L_s)` and compares order 1 with a refined
prompt result.

**Setup:**

* 3 wavelengths, 321 frequencies, 120 radial nodes, azimuthal degree 4.
* 51 modules, g = 0.9.
* Each full-path cache is built once per degree pair, with the axial method,
  and shared by both sources.

**Reference quality:**

* Track: the refined segment rule agrees with the default prompt to 1.3e-4.
* Shower: the ring reference gives 3.242 p.e. of order 1 in total. The
  default grouped prompt gives 3.181 p.e., max error 2.8 % of `max(q, 0.01)`.

The table lists, for the full path, the error on modules either method
marks active. Charge columns are `max(q, 0.01)`-scaled and give max /
median. Bins L1 is scaled the same way over modules both mark active and
gives median / max.

| `(L, L_s)` | build | source | full order-1 total vs reference | charge error, max / median | bins L1, median / max | modules with negative order 1 | transport |
|:--|--:|:--|--:|--:|--:|--:|--:|
| (12, 16) | 2263 s | track | -3.3 % | 98 % / 2.9 % | 28 % / 179 % | 12 (-0.039 p.e. negative bins) | 33 s |
| (12, 16) | 2263 s | shower | +21.9 % | 414 % / 59.5 % | 92 % / 394 % | 20 (-0.136 p.e. negative bins) | 148 s |
| (24, 32) | 2117 s | track | -0.7 % | 10 % / 1.1 % | 11 % / 25 % | 7 (-0.018 p.e. negative bins) | 67 s |
| (24, 32) | 2117 s | shower | +12.4 % | 229 % / 29.5 % | 50 % / 257 % | 13 (-0.083 p.e. negative bins) | 179 s |
| (36, 48) | 545 s | track | +0.3 % | 21 % / 0.5 % | 9 % / 17 % | 8 (-0.013 p.e. negative bins) | 94 s |
| (36, 48) | 545 s | shower | killed: out of memory (13.9 GB RSS) in `transport` | | | | |

The build times depend on CPU contention with other jobs on the same 4
cores. The first `(12, 16)` build took 798 s; the `(36, 48)` build ran
alone and took 545 s.

As the angular degrees rise, the full path moves towards the prompt and ring
references:

* **Track:** the median error falls from 2.9 % to 1.1 % to 0.5 %, and the
  worst from 98 % to 10 % (21 % at `(36, 48)`). The
  order-1 total goes from -3.3 % to -0.7 % to +0.3 %.
* **Shower:** the total error falls from +22 % to +12 %. At `(36, 48)` the
  full path cannot transport the 2000-element shower within 15 GB.
* `(24, 32)` is the production default of `KernelConfig`, the one used for
  all the BGVD comparisons.

On the track, the number of modules with negative order-1 charge falls from
12 to 7 or 8. The references stay fixed, so the differences at production degrees come from
the full path's truncation, not from the prompt path.

## 8. Backward (adjoint) Monte Carlo prototype

`lighthit.experimental.prompt_backward.backward_order1` implements the
estimator centred on the module. The scatter point is `y = R - b s2`, the
`1/b^2` cancels, and each sample scores

    A_eff * A(s2.n) / q(s2, b) * mu_s e^{-mu_t b} * sum_i B_i(y) p_HG(s1_i . s2)

with the analytic direct field `B_i` of each segment whose cone contains
`y`:

    xi* = z - rho cot(theta_C),  a* = rho / sin(theta_C),  0 <= xi* < h,
    B   = Y e^{-mu_t a*} / (2 pi rho sin theta_C)

The arrival time is `t0 + xi* dt/dxi + (a* + b) / v_g`, binned directly.
Wavelengths reuse each sample through the bounded weight
`e^{-(mu_t - mu*) b}`.

**Findings:**

* **Sampling `s2` from the acceptance alone is not enough for g = 0.9.** On
  the track the relative standard error was 2.0 % at 1e6 samples. A defensive
  mixture fixes this: 30 % acceptance sampling and 70 % HG lobes about the
  directions in which direct light reaches the module. The standard error
  drops to 0.4 % at 1e6 samples, a 25x variance reduction.
* **Near-source component for showers.** A third component samples `b`
  log-uniformly in the distance to the lobe's source point. Lobes are also
  smoothed over tree nodes instead of the few exact cone hits at the module.
  Together these two changes halve the electron's per-sample relative
  standard deviation, from 7 to 3.6.
* **Tracks (one segment):** 4e6 samples per module in about 0.5 s (4
  threads) give 0.2 to 0.3 % standard error, unbiased (section 3). This is a
  very good independent check. The deterministic segment rule is still more
  accurate at equal cost: 1e-4 in about 0.3 s per module.
* **Real electron (211,576 segments):**
  * The estimator is unbiased and has the best bin shapes of any method here
    (L1 0.3 to 1.7 % at 1 % charge error).
  * Each sample must find the few (about 12) segments whose Cherenkov shell
    contains `y`. For 0.5 mm G4 steps with nearly isotropic low-energy
    directions, neither a (cell x direction-pixel) grid nor a cone-ball BVH
    prunes well: it takes 25 to 40 k segment tests or about 50 k node tests
    per sample.
  * Cost: 1.28e5 samples per module (1 % s.e.) take 75 s per module on 4
    threads, about 10 minutes for the 8 bright modules. The deterministic
    prompt path takes 6.6 s for the whole event at -3.5 to +1 %.
* **Synthetic showers (1e3 to 1e4 elements):** 1e6 or 3e5 samples per module,
  11.6 s and 28.6 s for 7 modules, within 1 sigma of the ring reference.

**Verdict.** At this stage the backward estimator is a validation tool, not a
replacement for the deterministic production path. It confirms the ring
reference on the electron to about 1 %, and it shows the full path's shower
errors are real. It becomes competitive only when the per-sample segment
search costs O(10) segments. That needs one of two changes: G4 steps merged
into longer straight segments, which is an approximation, or a query
structure built for thin cone shells. Neither was attempted here.

## 9. Reproduction

Data and model live outside the repository. The full-path cache is built
once with `scripts/build_cache.py` or `kernel.build`.

    python scripts/prompt_benchmark.py --bgvd-model MODEL_DIR \
        --g4-electron sim_e_100GeV_10.h5 --g4-muon sim_mu_100GeV_10.h5 \
        --cache CACHE_DIR --output OUT --cases track --methods prompt,full
    python scripts/prompt_benchmark.py ... --cases shower-1000,shower-10000,shower-100000
    python scripts/prompt_benchmark.py ... --cases electron \
        --methods prompt,full,full-chunked --chunk-m 2
    python scripts/prompt_benchmark.py ... --cases muon \
        --methods prompt,prompt-slabs,full-slabs --chunk-m 10 --repeat 1
    python scripts/prompt_full_refinement.py --output refinement.json

Each (case, method) pair runs in a fresh process. The first call includes
loading compiled kernels from numba's on-disk cache.

**Cold compilation** (empty `NUMBA_CACHE_DIR`, once per installation):

| kernel family | compile time |
|:--|--:|
| segment path (tracks) | about 170 s |
| grouped path | about 250 s |
| ring path | about 80 s |

Afterwards the first call costs the same as a warm call.
