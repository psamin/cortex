"""Hand-built sim tasks (see base.py). ``make(name)`` returns (env, oracle) for a task."""

from __future__ import annotations

import importlib

TASKS = {"jenga": "jenga", "suture": "suture", "assembly": "assembly", "unscrew": "unscrew", "screwdriver": "screwdriver",
         "conveyor": "conveyor", "unscrew_black": "unscrew_black", "pick_place": "pick_place"}


def make(name: str, height: int | None = None, width: int | None = None):
    mod = importlib.import_module(f"{__name__}.{TASKS[name]}")
    kw = {k: v for k, v in (("height", height), ("width", width)) if v}
    return mod.Env(**kw), mod.Oracle()
