# Accuracy and speed

## What the test suite checks

The tests (`pytest tests src/lighthit/tests`) use synthetic media and
detectors. They check exact identities and tolerances of the numerical
methods, for example:

* table building, cache keys, readout, acceptance contraction (`test_cache*`,
  `test_directional*`, `test_readout`, `test_model_file`);
* the shower route against the per-band basis route: orders 0 and 1 equal,
  order >=2 within 2% charge and 5% bin L1 (`test_shower_route`);
* order 1 against independent quadrature and Monte-Carlo references
  (`test_prompt`, `test_segment_reference`);
* event composition and the laser flash (`test_event`);
* muon building blocks (`test_joint_muon`, `test_universal_primary`, ...).

They do not check the Geant4/Baikal-GVD numbers below.

## Measured during development (LightHit 0.3.0a1)

NTSim/Geant4 events in five Baikal-GVD clusters (1440 OMs), one CPU thread,
after Numba compilation. Errors are relative to calculations on the original
Geant4 steps or to a finer source basis, not to photon Monte Carlo. The
scripts that produced these numbers are not in the repository.

| Event | Time |
|---|---:|
| electron 100 GeV | 5.5-6.4 s |
| electron 1 TeV | 42 s |
| pi+ 100 GeV | 21 s |
| nu_e CC (e 70 + pi 25 + p 5 GeV) | 14 s |
| muon 100 GeV / 1 TeV | 11 / 10 s |
| nu_mu CC (mu 800 + pi 150 + p 50 GeV) | 73 s |

* Order 0 is exact on the Geant4 steps over the continuous spectrum. A
  nine-wavelength quadrature underestimated it by about 9% for showers (up to
  100% on single OMs).
* Order 1 (grouped, fused bands) against original-step integration: bin L1 at
  most 3.3% (100 GeV) and 3.1% (1 TeV), charge bias below 0.6%. This is the
  largest numerical error of the shower calculation.
* Order >=2 against a finer basis: at most 0.15% bin L1 on 100 GeV electrons,
  0.4% at 1 TeV, 1% on a hadronic shower.
* Hadronic showers emit about 3% of their light 50 ns to milliseconds late;
  that light is read on its own shifted clock or counted in the charge only.

## Reproducible comparison with photon Monte Carlo

`examples/LightHit/lighthit-in-ntsim.ipynb` in NTSim (section 2) runs one
Geant4 event (100 GeV electron) through `LightHitTransport` and through
NTSim's `CherenkovGenerator` -> `MCPhotonPropagator` (up to 30 scatterings,
full photon statistics) -> `SmartRayTracer`. LightHit gives 5.89 pe in about
6 s; three Monte-Carlo runs gave 5.82, 5.86 and 5.48 pe (each +- 0.23
statistical) in about 320 s. Per-OM charges and arrival times agree within
about two Monte-Carlo standard deviations.
