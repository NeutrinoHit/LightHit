import sys, time, json; sys.path.insert(0,"src")
import numpy as np, warnings
warnings.filterwarnings("ignore")
from lighthit.medium import synthetic_medium
from lighthit.green import SolverSettings
from lighthit.cache import (CacheGrid, ResponseCache, BandedResponseCache,
                            validation_report, first_order_consistency)
m = synthetic_medium(); L = 64
BANDS = [(10.0,  20.0, 12.0,  364, 20),
         (20.0,  60.0,  6.0,  526, 32),
         (60.0, 400.0,  3.0, 1620, 50)]
log = open("docs/cache-baikal/build2.log","w",buffering=1)
def say(*a): print(*a, file=log); print(*a, flush=True)
bands=[]; t_all=time.perf_counter()
for (lo,hi,kmax,J,nr) in BANDS:
    s = SolverSettings(L, J, kmax, kmax/240.0, 8, True)
    t0=time.perf_counter()
    band = ResponseCache.build(m, s, CacheGrid.geometric(lo,hi,nr,[0.0]),
                               angular_backend="numba")
    say(f"band [{lo},{hi}] kmax={kmax} J={J} R={nr}: {time.perf_counter()-t0:.1f}s "
        f"{json.dumps({k:(round(v,2) if isinstance(v,float) else v) for k,v in band.timings_s.items()})}")
    bands.append(band)
cache = BandedResponseCache(bands)
cache.save("docs/cache-baikal/cache.npz")
say(f"TOTAL build {time.perf_counter()-t_all:.1f}s; file "
    f"{__import__('os').path.getsize('docs/cache-baikal/cache.npz')/1e6:.2f} MB")
rng = np.random.default_rng(20260917)
say("--- interpolation validation (directed, off grid) ---")
for (lo,hi) in [(10.5,19.5),(21.0,58.0),(62.0,390.0)]:
    rq = np.exp(rng.uniform(np.log(lo),np.log(hi),24)); cq = rng.uniform(-0.95,0.92,24)
    rep = validation_report(cache, rq, cq)
    say(f"  r in [{lo},{hi}]: "+json.dumps({k:(f"{v:.3e}" if isinstance(v,float) else v) for k,v in rep.items()}))
say("--- finite-L first order vs exact HG quadrature (isotropic detector) ---")
for (lo,hi) in [(10.5,19.5),(21.0,58.0),(62.0,390.0)]:
    rq = np.exp(rng.uniform(np.log(lo),np.log(hi),16))
    rep = first_order_consistency(cache, rq)
    say(f"  r in [{lo},{hi}]: "+json.dumps({k:(f"{v:.3e}" if isinstance(v,float) else v) for k,v in rep.items()}))
say("DONE")
