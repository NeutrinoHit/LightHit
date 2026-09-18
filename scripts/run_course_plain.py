#!/usr/bin/env python3
"""Execute the course without Jupyter, one fresh interpreter per notebook.

``scripts/run_course.py`` is the reference runner and needs nbformat, nbclient,
nbconvert and jupyter_client. This fallback needs none of them: it reads the
notebook JSON directly, runs the code cells of one notebook in a single fresh
subprocess in their original order, and captures stdout, stderr, tracebacks and
figures. A fresh subprocess per notebook is the same isolation a fresh kernel
gives; it is not a kernel, so ``display()``, widgets, magics and rich reprs of
bare final expressions are not reproduced. Cells using them are reported as
skipped rather than silently ignored.

Outputs go to ``--output`` (default ``.build/course-plain``): an executed HTML
page per notebook, the figures, and ``execution.json`` with per-cell status and
timings. Source notebooks are never modified.
"""
from __future__ import annotations
import argparse
import html
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter

RUNNER = r'''
import json, sys, traceback, io, os
from contextlib import redirect_stdout, redirect_stderr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

cells = json.loads(sys.argv[1])
figure_dir = sys.argv[2]
stem = sys.argv[3]
namespace = {"__name__": "__main__"}
report = []
for index, source in cells:
    out, err = io.StringIO(), io.StringIO()
    status, message = "ok", None
    plt.close("all")
    try:
        with redirect_stdout(out), redirect_stderr(err):
            exec(compile(source, f"<cell {index}>", "exec"), namespace)
    except BaseException:
        status, message = "failed", traceback.format_exc()
    figures = []
    for number in plt.get_fignums():
        name = f"{stem}-cell{index}-fig{number}.png"
        plt.figure(number).savefig(os.path.join(figure_dir, name), dpi=110,
                                   bbox_inches="tight")
        figures.append(name)
    plt.close("all")
    report.append({"index": index, "status": status, "stdout": out.getvalue(),
                   "stderr": err.getvalue(), "traceback": message,
                   "figures": figures})
    if status == "failed":
        break
sys.stdout.write("\n@@RESULT@@" + json.dumps(report))
'''

MAGIC_PREFIXES = ("%", "!", "?")

# The notebooks call display()/Markdown() for narrative output. Outside a kernel
# those names must still exist; printing the text keeps the evidence visible.
SHIM = '''"""Minimal stand-in used only by run_course_plain.py."""


class Markdown:
    def __init__(self, data):
        self.data = data

    def __str__(self):
        return str(self.data)


class HTML(Markdown):
    pass


def display(*objects, **_ignored):
    for item in objects:
        print(item if isinstance(item, str) else str(item))
'''


def write_shim(out):
    """A tiny IPython.display substitute, kept out of the repository tree."""
    package = out / "_shim" / "IPython"
    package.mkdir(parents=True, exist_ok=True)
    # matplotlib probes IPython.get_ipython(); returning None keeps it in the
    # plain-interpreter path instead of installing a REPL display hook.
    (package / "__init__.py").write_text(
        "from . import display  # noqa: F401\n\nversion_info = (8, 24, 0, '')\n"
        "__version__ = '8.24.0'\n\n\ndef get_ipython():\n    return None\n",
        encoding="utf-8")
    (package / "display.py").write_text(SHIM, encoding="utf-8")
    return package.parent


def code_cells(notebook):
    """Code cells as (index, source), flagging ones a plain interpreter cannot run."""
    runnable, skipped = [], []
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if any(line.lstrip().startswith(MAGIC_PREFIXES)
               for line in source.splitlines()):
            skipped.append({"index": index, "reason": "IPython magic or shell escape"})
            continue
        runnable.append((index, source))
    return runnable, skipped


def markdown_html(cell):
    text = html.escape("".join(cell["source"]))
    return f"<pre class='md'>{text}</pre>"


def cell_html(index, source, result):
    body = [f"<pre class='code'>{html.escape(source)}</pre>"]
    if result is None:
        body.append("<p class='note'>not executed</p>")
        return "\n".join(body)
    for stream in ("stdout", "stderr"):
        if result[stream]:
            body.append(f"<pre class='{stream}'>{html.escape(result[stream])}</pre>")
    if result["traceback"]:
        body.append(f"<pre class='stderr'>{html.escape(result['traceback'])}</pre>")
    for name in result["figures"]:
        body.append(f"<img src='figures/{html.escape(name)}' alt='figure'>")
    return "\n".join(body)


STYLE = """body{max-width:1000px;margin:2rem auto;font:16px/1.6 system-ui;padding:0 1rem}
pre{padding:.7rem;border-radius:6px;overflow-x:auto;white-space:pre-wrap}
pre.md{background:#f6f6f6}pre.code{background:#eef3fb}
pre.stdout{background:#f2faf2}pre.stderr{background:#fdf0f0}
img{max-width:100%;margin:.6rem 0}.note{color:#a33}"""


def render(path, notebook, results, skipped, out):
    lookup = {r["index"]: r for r in results}
    pieces = [f"<!doctype html><html lang='ru'><meta charset='utf-8'>",
              f"<title>{html.escape(path.name)}</title><style>{STYLE}</style>",
              f"<h1>{html.escape(path.name)}</h1>",
              "<p class='note'>Executed without Jupyter: one fresh interpreter for "
              "the whole notebook, cells in order.</p>"]
    skipped_index = {s["index"] for s in skipped}
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "markdown":
            pieces.append(markdown_html(cell))
        elif cell["cell_type"] == "code":
            source = "".join(cell["source"])
            if index in skipped_index:
                pieces.append(f"<pre class='code'>{html.escape(source)}</pre>"
                              "<p class='note'>skipped: IPython magic or shell escape</p>")
            else:
                pieces.append(cell_html(index, source, lookup.get(index)))
    (out / (path.stem + ".html")).write_text("\n".join(pieces), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=".build/course-plain")
    parser.add_argument("--timeout", type=int, default=1800,
                        help="Per-notebook timeout in seconds")
    parser.add_argument("--select", nargs="*")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = Path(args.output).expanduser().resolve()
    (out / "figures").mkdir(parents=True, exist_ok=True)
    files = sorted((root / "notebooks/course").glob("[0-9][0-9]_*.ipynb"))
    if args.select:
        wanted = {Path(x).stem for x in args.select}
        unknown = wanted - {p.stem for p in files}
        if unknown:
            raise SystemExit(f"Unknown notebooks: {sorted(unknown)}")
        files = [p for p in files if p.stem in wanted]
    if not files:
        raise SystemExit("No notebooks found")
    environment = dict(os.environ)
    shim = write_shim(out)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(root / "src"), str(shim)]
        + ([environment["PYTHONPATH"]] if "PYTHONPATH" in environment else []))
    environment["MPLBACKEND"] = "Agg"
    report = {"python": sys.version, "executable": sys.executable,
              "platform": platform.platform(), "runner": "run_course_plain.py",
              "isolation": "one fresh subprocess per notebook",
              "shims": ["IPython.display (display/Markdown/HTML print as text)"],
              "notebooks": [], "source_notebooks_modified": False}
    failures = 0
    for path in files:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        cells, skipped = code_cells(notebook)
        print(f"Execute {path.name} ({len(cells)} code cells)", flush=True)
        start = perf_counter()
        process = subprocess.run(
            [sys.executable, "-c", RUNNER, json.dumps(cells),
             str(out / "figures"), path.stem],
            cwd=str(root), env=environment, capture_output=True, text=True,
            timeout=args.timeout)
        elapsed = perf_counter() - start
        marker = process.stdout.rfind("@@RESULT@@")
        results = json.loads(process.stdout[marker + len("@@RESULT@@"):]) if marker >= 0 else []
        status = "ok" if results and all(r["status"] == "ok" for r in results) \
            and len(results) == len(cells) else "failed"
        failures += status == "failed"
        render(path, notebook, results, skipped, out)
        report["notebooks"].append({
            "name": path.name, "status": status, "seconds": elapsed,
            "code_cells": len(cells), "skipped_cells": skipped,
            "figures": [name for r in results for name in r["figures"]],
            "cells": [{k: r[k] for k in ("index", "status", "stdout", "stderr",
                                         "traceback", "figures")} for r in results],
            "process_stderr": process.stderr[-4000:] if status == "failed" else ""})
        (out / "execution.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  {status}: {elapsed:.2f} s", flush=True)
    report["total_seconds"] = sum(r["seconds"] for r in report["notebooks"])
    (out / "execution.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    links = "\n".join(f"<li><a href='{html.escape(Path(r['name']).stem)}.html'>"
                      f"{html.escape(r['name'])}</a> — {r['seconds']:.2f} с, {r['status']}</li>"
                      for r in report["notebooks"])
    (out / "index.html").write_text(
        "<!doctype html><html lang='ru'><meta charset='utf-8'>"
        f"<title>LightHit course (plain runner)</title><style>{STYLE}</style>"
        "<h1>LightHit: учебный курс</h1><p>Исполнено без Jupyter: "
        "по одному свежему интерпретатору на ноутбук.</p><ol>" + links +
        "</ol><p><a href='execution.json'>Протокол исполнения</a></p></html>",
        encoding="utf-8")
    print(f"HTML index: {out / 'index.html'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
