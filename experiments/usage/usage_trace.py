"""Record which functions of the ``lighthit`` package a run executes.

Python functions are seen through ``sys.monitoring`` (PY_START, disabled after
the first call of each code object, so the overhead is one callback per
function).  Numba-compiled functions never run their Python code; a dispatcher
counts as executed when it holds at least one compiled (or cache-loaded)
signature at the end of the run, which happens only when it was called
directly or from another compiled function.

Use as a wrapper::

    python experiments/usage/usage_trace.py OUT.json -- script.py args...
    python experiments/usage/usage_trace.py OUT.json -m module args...

or in-process with ``start(package_dir)`` / ``finish(out_path, scenario)``.
"""
from __future__ import annotations

import json
import os
import resource
import runpy
import sys
import time
from pathlib import Path

TOOL = sys.monitoring.PROFILER_ID if hasattr(sys, "monitoring") else None
_seen: dict[tuple[str, str], int] = {}
_root = ""
_started = 0.0


def _on_start(code, offset):
    filename = code.co_filename
    if filename.startswith(_root):
        _seen.setdefault((filename, code.co_qualname), code.co_firstlineno)
    return sys.monitoring.DISABLE


def start(package_dir: str | os.PathLike) -> None:
    """Begin recording calls of code defined below ``package_dir``."""
    global _root, _started
    if TOOL is None:
        raise RuntimeError("sys.monitoring (Python >= 3.12) is required")
    _root = str(Path(package_dir).resolve()) + os.sep
    _started = time.perf_counter()
    sys.monitoring.use_tool_id(TOOL, "lighthit-usage")
    sys.monitoring.register_callback(TOOL, sys.monitoring.events.PY_START, _on_start)
    sys.monitoring.set_events(TOOL, sys.monitoring.events.PY_START)


def _numba_executed():
    """(module, qualname, inline) of lighthit dispatchers with a compiled signature."""
    try:
        from numba.core.registry import CPUDispatcher
    except Exception:  # numba missing
        return [], []
    executed, idle = [], []
    for name, module in list(sys.modules.items()):
        if module is None or not (name == "lighthit" or name.startswith("lighthit.")):
            continue
        file = getattr(module, "__file__", None) or ""
        if not str(Path(file).resolve()).startswith(_root):
            continue
        for value in list(vars(module).values()):
            if not isinstance(value, CPUDispatcher):
                continue
            func = value.py_func
            if func.__module__ != name:
                continue
            record = dict(file=str(Path(func.__code__.co_filename).resolve()),
                          qualname=func.__code__.co_qualname,
                          line=func.__code__.co_firstlineno,
                          inline=value.targetoptions.get("inline", "never"))
            (executed if value.signatures else idle).append(record)
    return executed, idle


def finish(out_path: str | os.PathLike, scenario: str, extra: dict | None = None) -> dict:
    """Stop recording and write the JSON record."""
    sys.monitoring.set_events(TOOL, 0)
    sys.monitoring.free_tool_id(TOOL)
    executed, idle = _numba_executed()
    usage = resource.getrusage(resource.RUSAGE_SELF)
    record = dict(
        scenario=scenario,
        package_root=_root,
        python=[dict(file=f, qualname=q, line=line) for (f, q), line in sorted(_seen.items())],
        numba_executed=executed,
        numba_not_executed=idle,
        modules=sorted(n for n, m in sys.modules.items()
                       if m is not None and (n == "lighthit" or n.startswith("lighthit."))),
        wall_seconds=time.perf_counter() - _started,
        cpu_seconds=usage.ru_utime + usage.ru_stime,
        peak_rss_mib=usage.ru_maxrss / (2 ** 20 if sys.platform == "darwin" else 2 ** 10),
        **(extra or {}))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(record, indent=1))
    return record


def _main(argv):
    out, rest = argv[0], argv[1:]
    if rest and rest[0] == "--":
        rest = rest[1:]
    import importlib.util
    spec = importlib.util.find_spec("lighthit")
    package = Path(spec.origin).parent
    start(package)
    status = 0
    try:
        if rest[0] == "-m":
            sys.argv = rest[1:]
            runpy.run_module(rest[1], run_name="__main__", alter_sys=True)
        else:
            sys.argv = rest
            sys.path.insert(0, str(Path(rest[0]).resolve().parent))
            runpy.run_path(rest[0], run_name="__main__")
    except SystemExit as exc:
        status = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
    finally:
        finish(out, os.environ.get("LIGHTHIT_USAGE_SCENARIO", Path(out).stem),
               dict(command=rest, exit_status=status))
    return status


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
