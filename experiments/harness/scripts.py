"""Run a research script as a measured case.

The script runs in a subprocess with this process's environment (thread
variables fixed by experiments/run.py, LIGHTHIT_DATA, PYTHONPATH=src), its
stdout and stderr go to experiments/results/<idea>/<case>.log, and its JSON
report (a file it writes, or the last JSON object it prints) becomes the
case's metrics. Wall time, CPU time and peak RSS are the child's.
"""
from __future__ import annotations

import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

from .record import RESULTS, ROOT

SCRIPTS = ROOT / "scripts"


def _last_json(text: str):
    """The last complete JSON object or array printed (multi-line allowed)."""
    decoder = json.JSONDecoder()
    found = None
    for start in [i for i, c in enumerate(text) if c in "{["][::-1]:
        try:
            value, end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, (dict, list)) and len(text[start:start + end]) > 2:
            found = value
            break
    return found


def _note_inputs(ctx, command):
    """Hashes of the standard inputs (the scripts read them through paths.py) and
    of any input file named on the command line (files under 2 GB)."""
    if ctx is None:
        return
    from . import inputs as I
    try:
        standard = [I.bundle_path("model", "bgvd-model.npz"), I.bundle_path("tables", "muon-data.json"),
                    I.bundle_path("events", "geant4-electron-100GeV.h5"),
                    I.bundle_path("events", "muon-1TeV-steps.npz"),
                    I.bundle_path("events", "muon-1TeV-ancestry.npz")]
    except SystemExit:
        standard = []
    named = [Path(c) for c in command[2:] if isinstance(c, str) and c.startswith("/")]
    for path in standard + [p for p in named if p.is_file() and p.stat().st_size < 2 ** 31
                            and not str(p).startswith(str(RESULTS))]:
        if Path(path).is_file():
            ctx._note(path)
    regenerated = I.inputs_dir()
    for path in sorted(regenerated.glob("*-steps.npz")):
        ctx._note(path)


def run_script(ctx, idea, name, argv, *, cwd=None, report=None, headline=None, env=None,
               timeout=4 * 3600, clean=()):
    """Run ``python scripts/<argv[0]> argv[1:]``; return the metrics dict.

    ``clean``: output directories removed before the run (scripts that refuse
    to overwrite)."""
    import shutil
    for path in clean:
        if Path(path).exists():
            shutil.rmtree(path)
    log = RESULTS / idea / f"{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    script = (ROOT / argv[0]) if argv[0].startswith("tools/") else SCRIPTS / argv[0]
    command = ([sys.executable, str(script)] if script.suffix == ".py" else ["bash", str(script)])
    command += [str(a() if callable(a) else a) for a in argv[1:]]
    work = str(RESULTS / "work")
    for arg in command[2:]:
        # an output file below results/work: its folder must exist (most scripts make it themselves)
        if arg.startswith(work) and Path(arg).suffix in (".json", ".npz", ".npy"):
            Path(arg).parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, **(env or {}))
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "src"), str(SCRIPTS / "shower_basis"), str(ROOT / "tools" / "tables")]
        + [p for p in environment.get("PYTHONPATH", "").split(os.pathsep) if p])
    environment.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    _note_inputs(ctx, command)
    workdir = Path(cwd) if cwd else ROOT
    workdir.mkdir(parents=True, exist_ok=True)
    usage0 = resource.getrusage(resource.RUSAGE_CHILDREN)
    began = time.perf_counter()
    done = subprocess.run(command, cwd=workdir, env=environment, capture_output=True, text=True,
                          timeout=timeout)
    wall = time.perf_counter() - began
    usage1 = resource.getrusage(resource.RUSAGE_CHILDREN)
    log.write_text(f"$ {' '.join(command)}\n(cwd {workdir})\n\n{done.stdout}\n--- stderr ---\n{done.stderr}")
    rss = usage1.ru_maxrss / (2 ** 20 if sys.platform == "darwin" else 2 ** 10)
    metrics = dict(command=[str(Path(c).relative_to(ROOT)) if str(c).startswith(str(ROOT)) else c
                            for c in command],
                   exit_status=done.returncode, log=str(log.relative_to(RESULTS)),
                   timing=dict(wall_s=wall, cpu_s=(usage1.ru_utime - usage0.ru_utime)
                               + (usage1.ru_stime - usage0.ru_stime), peak_rss_mib=rss,
                               note="child process: includes imports and Numba compilation"),
                   stdout_tail=done.stdout[-3000:], stderr_tail=done.stderr[-2000:])
    data = None
    if report is not None:
        path = Path(report)
        path = path if path.is_absolute() else workdir / path
        if path.is_file():
            data = json.loads(path.read_text())
            metrics["report_file"] = str(path)
    if data is None:
        data = _last_json(done.stdout)
    if data is not None:
        metrics["report"] = data
        if headline is not None:
            try:
                metrics["headline"] = headline(data)
            except Exception as exc:     # keep the run; say why there is no headline
                metrics["headline_error"] = f"{type(exc).__name__}: {exc}"
    return metrics


def script_case(idea, name, argv, **options):
    """Register a script as case idea/name (kind metrics, no warm-up)."""
    from .cases import case
    case_options = {k: options.pop(k) for k in ("notes", "heavy", "reference", "baseline", "tags")
                    if k in options}
    case(idea, name, kind="metrics", warmup=False, **case_options)(
        lambda ctx: run_script(ctx, idea, name, argv, **options))
