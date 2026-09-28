# ruff: noqa: F403, F405  (prototype scripts share the helpers of common/proto)
import os
import time, resource, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
k = model().kernel(config(cache_policy="build"))
t = time.time(); k.build(method="axial", progress="console"); print("build seconds", time.time() - t,
      "peak MiB", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
