"""Primitive defaults for everyday objects, so an agent can say "water bottle" instead of numbers.

Every value taken from here is marked ``assumed`` on the resulting ObjectSpec; the
agent is asked to confirm or correct it. Sizes follow ObjectSpec.size conventions.
"""

from __future__ import annotations

import re

from .spec import ObjectSpec

# name -> (shape, size, mass_kg, rgba). Aliases map onto these keys.
CATALOGUE: dict[str, tuple[str, list[float], float, tuple[float, float, float, float]]] = {
    "cube": ("box", [0.03, 0.03, 0.03], 0.01, (0.95, 0.15, 0.12, 1.0)),
    "block": ("box", [0.04, 0.04, 0.04], 0.03, (0.95, 0.55, 0.12, 1.0)),
    "water bottle": ("cylinder", [0.065, 0.20], 0.52, (0.55, 0.75, 0.95, 1.0)),  # 500 ml, full
    "small water bottle": ("cylinder", [0.045, 0.14], 0.24, (0.55, 0.75, 0.95, 1.0)),  # ~220 ml, full
    "empty water bottle": ("cylinder", [0.065, 0.20], 0.02, (0.55, 0.75, 0.95, 1.0)),
    "can": ("cylinder", [0.066, 0.122], 0.37, (0.80, 0.80, 0.82, 1.0)),
    "cup": ("cylinder", [0.08, 0.095], 0.25, (0.95, 0.95, 0.92, 1.0)),
    "marker": ("cylinder", [0.018, 0.13], 0.015, (0.10, 0.10, 0.10, 1.0)),
    "ball": ("sphere", [0.04], 0.03, (0.20, 0.40, 0.95, 1.0)),
    "tennis ball": ("sphere", [0.067], 0.058, (0.85, 0.95, 0.20, 1.0)),
}
ALIASES = {
    "bottle": "water bottle",
    "water": "water bottle",
    "soda can": "can",
    "tin can": "can",
    "mug": "cup",
    "pen": "marker",
    "box": "block",
}
# Words that are usually not the object in a prompt ("can you...", "put it in the box"). lookup()
# still resolves them when an agent names an object explicitly; prompt seeding skips them.
NOT_SEEDED = {"can", "box", "block"}
# Families where the default is a guess between very different objects: ask which one.
VARIANTS = {"water bottle": ["water bottle", "empty water bottle", "small water bottle"]}


def lookup(name: str) -> str | None:
    """Catalogue key for a free-text object name, or None."""
    n = re.sub(r"[\s_]+", " ", name.lower().strip())
    if n in CATALOGUE:
        return n
    if n in ALIASES:
        return ALIASES[n]
    for key in sorted(CATALOGUE, key=len, reverse=True):  # "small water bottle" before "water bottle"
        if key in n:
            return key
    return next((v for a, v in ALIASES.items() if re.search(rf"\b{a}\b", n)), None)


def from_catalogue(name: str, key: str | None = None) -> ObjectSpec | None:
    key = key or lookup(name)
    if key is None:
        return None
    shape, size, mass, rgba = CATALOGUE[key]
    return ObjectSpec(name=slug(name), shape=shape, size=list(size), mass_kg=mass, rgba=rgba,  # type: ignore[arg-type]
                      assumed={"shape", "size", "mass_kg"})


def mentioned(prompt: str) -> list[str]:
    """Catalogue keys mentioned in a prompt, longest match first, without overlaps."""
    p = prompt.lower()
    found: list[str] = []
    for key in sorted(CATALOGUE, key=len, reverse=True):
        if key in NOT_SEEDED:
            continue
        if re.search(rf"\b{re.escape(key)}s?\b", p) and not any(key in f for f in found):
            found.append(key)
    for alias, key in ALIASES.items():
        if alias in NOT_SEEDED:
            continue
        if re.search(rf"\b{alias}s?\b", p) and key not in found and not any(alias in f for f in found):
            found.append(key)
    return found


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "object"
