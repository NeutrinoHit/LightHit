#!/usr/bin/env python
"""Merge usage records into experiments/usage/usage-map.json and docs/usage-map.md.

    python experiments/usage/build_map.py [RECORD.json ...]

Default records: experiments/results/usage/*.json (written by scenarios.py and
by usage_trace.py around NTSim's run_lighthit.py). Every function, method and
nested function defined in src/lighthit (tests excluded) is listed with the
scenarios that executed it. Numba functions declared ``inline="always"`` are
compiled into their callers and never hold a signature of their own; they are
marked executed ("inlined") when an executed compiled function of the same
module refers to them by name.
"""
from __future__ import annotations

import ast
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "src" / "lighthit"
OUT_JSON = Path(__file__).resolve().parent / "usage-map.json"
OUT_MD = ROOT / "docs" / "usage-map.md"

ORDER = ["shower", "muon", "event", "event-stopping-muon", "event-muon-near-om", "laser", "prompt", "hybrid",
         "ntsim-electron", "ntsim-muon", "ntsim-nue-cc", "ntsim-numu-cc", "ntsim-laser"]
DESCRIPTION = {
    "shower": "ShowerTransport, bundle Geant4 100 GeV electron (event 5)",
    "muon": "MuonTransport, bundle 1 TeV muon with its secondaries",
    "event": "EventTransport, 1 TeV muon + 100 GeV electron shower",
    "event-stopping-muon": "EventTransport, regenerated 6 GeV muon aimed 1.5 m past an OM, "
                           "stopping 15 m before it (muon route)",
    "event-muon-near-om": "EventTransport, regenerated 6 GeV muon passing 1.76 m from an OM "
                          "(muon route with the near-field clamp)",
    "laser": "EventTransport with laser_kernel, 532 nm flash (bundle laser table)",
    "prompt": "TransportKernel.transport_prompt, 100 GeV electron (orders 0 and 1)",
    "hybrid": "TransportKernel.transport_hybrid, 100 GeV electron (per-band order >=2)",
    "ntsim-electron": "NTSim run_lighthit.py, e- 100 GeV",
    "ntsim-muon": "NTSim run_lighthit.py, mu- 1 TeV",
    "ntsim-nue-cc": "NTSim run_lighthit.py, e- 70 + pi+ 25 + p 5 GeV",
    "ntsim-numu-cc": "NTSim run_lighthit.py, mu- 800 + pi+ 150 + p 50 GeV",
    "ntsim-laser": "NTSim run_lighthit.py --laser, 532 nm flash",
}


class Definitions(ast.NodeVisitor):
    """(qualname, line, numba decorator options) of every def in a module."""

    def __init__(self):
        self.stack, self.found = [], []

    def _numba(self, node):
        for deco in node.decorator_list:
            target = deco.func if isinstance(deco, ast.Call) else deco
            name = ast.unparse(target)
            if name.split(".")[-1] in ("njit", "jit", "vectorize", "guvectorize", "cfunc"):
                inline = "never"
                if isinstance(deco, ast.Call):
                    for kw in deco.keywords:
                        if kw.arg == "inline" and isinstance(kw.value, ast.Constant):
                            inline = kw.value.value
                return dict(decorator=name, inline=inline)
        return None

    def visit_ClassDef(self, node):
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def _function(self, node):
        qualname = ".".join(self.stack + [node.name])
        names = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
        self.found.append(dict(qualname=qualname, line=node.lineno, numba=self._numba(node),
                               names=names))
        self.stack += [node.name, "<locals>"]
        self.generic_visit(node)
        del self.stack[-2:]

    visit_FunctionDef = visit_AsyncFunctionDef = _function


def definitions():
    out = {}
    for path in sorted(PACKAGE.rglob("*.py")):
        rel = path.relative_to(PACKAGE.parent)
        if "tests" in rel.parts:
            continue
        visitor = Definitions()
        visitor.visit(ast.parse(path.read_text(), str(path)))
        module = ".".join(rel.with_suffix("").parts).removesuffix(".__init__")
        out[module] = dict(file=str(rel), functions=visitor.found)
    return out


def _portable(value):
    """Command lines without private absolute paths ($NTSIM, $LIGHTHIT_DATA, repository-relative)."""
    import os
    if isinstance(value, list):
        return [_portable(v) for v in value]
    if not isinstance(value, str):
        return value
    text = value
    for variable in ("LIGHTHIT_DATA", "NTSIM"):
        root = os.environ.get(variable)
        if root:
            text = text.replace(str(Path(root).resolve()), f"${variable}").replace(root.rstrip("/"), f"${variable}")
    text = text.replace(str(ROOT) + "/", "")
    if text.startswith("/"):
        text = "<local>/" + Path(text).name
    return text


def main(paths):
    paths = [Path(p) for p in paths] or sorted((ROOT / "experiments" / "results" / "usage").glob("*.json"))
    records = [json.loads(p.read_text()) for p in paths]
    defs = definitions()
    by_file = {str(PACKAGE.parent / d["file"]): m for m, d in defs.items()}
    executed = defaultdict(set)          # (module, qualname) -> scenarios
    imported = defaultdict(set)          # module -> scenarios
    runs = {}
    for record in records:
        name = record["scenario"]
        root = Path(record["package_root"])
        runs[name] = {k: record.get(k) for k in ("wall_seconds", "cpu_seconds", "peak_rss_mib",
                                                  "scenario_seconds", "numba_threads", "result",
                                                  "command", "exit_status")}
        runs[name]["command"] = _portable(runs[name]["command"])
        for module in record["modules"]:
            imported[module].add(name)
        for item in record["python"] + record["numba_executed"]:
            rel = Path(item["file"]).resolve().relative_to(root)
            module = by_file.get(str(PACKAGE / rel))
            if module is not None:
                executed[(module, item["qualname"])].add(name)
    table, summary = {}, {}
    for module, info in defs.items():
        rows = []
        for fn in info["functions"]:
            scenarios = set(executed.get((module, fn["qualname"]), ()))
            how = "numba" if fn["numba"] else "python"
            if fn["numba"] and fn["numba"]["inline"] == "always" and not scenarios:
                for other in info["functions"]:
                    if other["numba"] and fn["qualname"].split(".")[-1] in other["names"]:
                        scenarios |= executed.get((module, other["qualname"]), set())
                how = "numba-inlined"
            rows.append(dict(qualname=fn["qualname"], line=fn["line"], kind=how,
                             executed_by=[s for s in ORDER if s in scenarios]
                             + sorted(scenarios - set(ORDER))))
        table[module] = dict(file=info["file"], imported_by=[s for s in ORDER if s in imported[module]]
                             + sorted(imported[module] - set(ORDER)), functions=rows)
        summary[module] = dict(functions=len(rows), executed=sum(bool(r["executed_by"]) for r in rows))
    scenarios = [s for s in ORDER if s in runs] + sorted(set(runs) - set(ORDER))
    data = dict(scenarios={s: dict(description=DESCRIPTION.get(s, ""), **runs[s]) for s in scenarios},
                modules=table)
    OUT_JSON.write_text(json.dumps(data, indent=1) + "\n")
    OUT_MD.write_text(markdown(data, scenarios))
    print(f"wrote {OUT_JSON.relative_to(ROOT)} and {OUT_MD.relative_to(ROOT)}: "
          f"{sum(v['executed'] for v in summary.values())} of "
          f"{sum(v['functions'] for v in summary.values())} functions executed")


def _short(names, scenarios):
    return "all" if set(names) == set(scenarios) and len(scenarios) > 1 else ", ".join(names)


def markdown(data, scenarios):
    modules = data["modules"]
    lines = ["# Usage map: what the default routes execute", "",
             "Measured, not read from the code: each scenario below ran in its own process on the",
             "standard inputs (`experiments/harness/inputs.py`, data bundle `$LIGHTHIT_DATA` and the",
             "regenerated NTSim events) under `experiments/usage/usage_trace.py`, which records every",
             "Python function of `lighthit` that starts and every Numba function that holds a compiled",
             "signature at the end of the run. Regenerate with `experiments/usage/scenarios.py`,",
             "`experiments/usage/run_ntsim.sh` and `experiments/usage/build_map.py`; the full table",
             "(every function with its scenarios) is `experiments/usage/usage-map.json`.", "",
             "A function listed as not executed was not called in any of these runs; it may still be",
             "reachable with other inputs or options (for example a near-field error branch).", "",
             "## Scenarios", "", "| Scenario | What ran | Wall s | Peak RSS MiB | Route (metadata) |",
             "|---|---|---:|---:|---|"]
    for s in scenarios:
        run = data["scenarios"][s]
        route = (run.get("result") or {}).get("route") or ""
        fb = (run.get("result") or {}).get("route_fallbacks")
        if fb:
            route += f"; fallbacks: {len(fb)}"
        if (run.get("result") or {}).get("error"):
            route += f"; raised {run['result']['error']}"
        lines.append(f"| {s} | {run['description']} | {run['wall_seconds']:.1f} | "
                     f"{run['peak_rss_mib']:.0f} | {route} |")
    lines += ["", "Wall time includes imports, table loading and any Numba compilation; it is not a",
              "benchmark (see `experiments/REPORT.md`).", "",
              "## Modules", "",
              "Functions executed / defined, and the scenarios that imported the module.", "",
              "| Module | Executed | Imported by |", "|---|---:|---|"]
    for module, info in modules.items():
        rows = info["functions"]
        done = sum(bool(r["executed_by"]) for r in rows)
        lines.append(f"| `{module}` | {done} / {len(rows)} | "
                     f"{_short(info['imported_by'], scenarios) or 'not imported'} |")
    lines += ["", "## Functions per module", "",
              "Grouped by the set of scenarios that executed them.", ""]
    for module, info in modules.items():
        rows = info["functions"]
        if not rows:
            continue
        groups = defaultdict(list)
        for r in rows:
            key = _short(r["executed_by"], scenarios) or "not executed"
            label = f"`{r['qualname']}`" + (" (numba)" if r["kind"] == "numba" else
                                             " (numba, inlined)" if r["kind"] == "numba-inlined" else "")
            groups[key].append(label)
        lines += [f"### `{module}`", ""]
        for key in sorted(groups, key=lambda k: (k == "not executed", k != "all", k)):
            lines.append(f"* **{key}**: {', '.join(groups[key])}")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main(sys.argv[1:])
