"""Provenance, timing and storage of one measurement."""
from __future__ import annotations

import json
import os
import platform
import resource
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS = Path(os.environ.get("LIGHTHIT_EXPERIMENT_RESULTS", ROOT / "experiments" / "results"))
THREAD_VARIABLES = ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS")


def commit() -> dict:
    def git(*args):
        try:
            return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True,
                                  text=True, check=True).stdout.strip()
        except Exception:
            return None
    status = git("status", "--porcelain", "--", "src")
    return dict(commit=git("rev-parse", "--short=12", "HEAD"), src_modified=bool(status))


def machine() -> dict:
    info = dict(platform=platform.platform(), python=platform.python_version(),
                processor=platform.processor(), cpu_count=os.cpu_count())
    try:
        info["cpu"] = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                     capture_output=True, text=True).stdout.strip() or None
    except Exception:
        pass
    try:
        info["load_average"] = list(os.getloadavg())
    except OSError:
        pass
    for name in ("numpy", "scipy", "numba", "lighthit"):
        try:
            module = __import__(name)
            info[name] = getattr(module, "__version__", "unknown")
        except Exception:
            info[name] = None
    info["threads"] = {k: os.environ.get(k) for k in THREAD_VARIABLES}
    try:
        import numba
        info["numba_threads"] = numba.get_num_threads()
    except Exception:
        pass
    return info


def peak_rss_mib() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (2 ** 20 if sys.platform == "darwin" else 2 ** 10)


@contextmanager
def timed(out: dict, key: str = "timing"):
    """Wall and CPU (all threads of this process) time of the block."""
    usage0 = resource.getrusage(resource.RUSAGE_SELF)
    wall0 = time.perf_counter()
    yield
    usage1 = resource.getrusage(resource.RUSAGE_SELF)
    out[key] = dict(wall_s=time.perf_counter() - wall0,
                    cpu_s=(usage1.ru_utime - usage0.ru_utime) + (usage1.ru_stime - usage0.ru_stime),
                    peak_rss_mib=peak_rss_mib())


def save_response(response, path: Path) -> Path:
    """The comparable arrays of a response as one .npz."""
    path.parent.mkdir(parents=True, exist_ok=True)
    from .metrics import three_orders
    charge, bins = three_orders(response)
    arrays = dict(charge_components_pe=charge, active=np.asarray(response.active),
                  time_origin_ns=np.asarray(response.time_origin_ns),
                  orders_computed=np.asarray(getattr(response, "computed_orders", (0, 1, 2)), str))
    if bins is not None:
        arrays["components_pe"] = bins.astype(np.float32)
    for name in ("relative_time_edges_ns", "omega_per_ns", "covered"):
        value = getattr(response, name, None)
        if value is not None:
            arrays[name] = np.asarray(value)
    meta = getattr(response, "metadata", {}) or {}
    arrays["metadata"] = json.dumps(meta, default=_jsonable)
    np.savez_compressed(path, **arrays)
    return path


def load_response(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        out = {k: data[k] for k in data.files if k != "metadata"}
        out["metadata"] = json.loads(str(data["metadata"])) if "metadata" in data.files else {}
    return out


def _jsonable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist() if value.size <= 64 else f"<array {value.shape} {value.dtype}>"
    if isinstance(value, Path):
        return str(value)
    return repr(value)


def write_result(idea: str, case: str, payload: dict) -> Path:
    """experiments/results/<idea>/<case>.json (the latest run of that case)."""
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    record = dict(idea=idea, case=case, date=stamp, machine=machine(), source=commit(), **payload)
    path = RESULTS / idea / f"{case}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1, default=_jsonable) + "\n")
    return path
