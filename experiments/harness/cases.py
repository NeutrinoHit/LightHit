"""Registry of measurable cases.

A case belongs to an idea (``experiments/ideas/<idea>.toml``) and computes one
response, or one dict of metrics, on a standard input. Cases are registered by
the modules ``experiments/harness/cases_*.py`` with the ``case`` decorator:

    @case("shower-route", "electron-100GeV", reference="shower-reference/electron-100GeV")
    def _(ctx):
        return ShowerTransport(ctx.kernel())(ctx.electron())

``reference`` names another case (``idea/case``) whose saved response this one
is compared with; ``baseline`` names the case whose timing it is compared with
(default: the default route on the same input). ``kind`` is "response" (the
function returns a response, saved and compared) or "metrics" (it returns a
dict that is stored as it is, e.g. a script's own comparison).
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

HERE = Path(__file__).resolve().parent
CASES: dict[str, "Case"] = {}


@dataclass
class Case:
    idea: str
    name: str
    function: Callable
    reference: str | None = None
    baseline: str | None = None
    kind: str = "response"
    repeat: int = 1
    warmup: bool = True
    heavy: bool = False
    inputs: tuple = ()
    notes: str = ""
    tags: tuple = field(default_factory=tuple)

    @property
    def key(self) -> str:
        return f"{self.idea}/{self.name}"


def case(idea, name, *, reference=None, baseline=None, kind="response", repeat=1,
         warmup=True, heavy=False, inputs=(), notes="", tags=()):
    def register(function):
        item = Case(idea, name, function, reference, baseline, kind, repeat, warmup, heavy,
                    tuple(inputs), notes, tuple(tags))
        if item.key in CASES:
            raise ValueError(f"duplicate case {item.key}")
        CASES[item.key] = item
        return function
    return register


def load_all():
    for path in sorted(HERE.glob("cases_*.py")):
        importlib.import_module(f"harness.{path.stem}")
    return CASES
