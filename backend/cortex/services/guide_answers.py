"""Map a person's free-text reply onto the open intake question, as an ``update_intake`` patch.

Deliberately small and predictable (no model): option names or their distinctive words,
an option number, yes/confirm for the default, numbers with units for object sizes and
masses, and catalogue object names. Anything else returns None and the guide asks again,
listing the options.
"""

from __future__ import annotations

import re
from typing import Any

from ..intake import IntakeSpec, Question, catalogue

YES = re.compile(r"^\s*(y|yes|yeah|yep|yup|ok|okay|sure|confirm(ed)?|correct|right|fine|accept|default|defaults|"
                 r"looks (good|right|fine)|sounds good|that'?s (right|correct|it)|go( ahead)?|use (the )?defaults?|"
                 r"do it|continue|next)\b", re.I)
_LEN = re.compile(r"(\d+(?:\.\d+)?)\s*(mm|cm|m)\b", re.I)
_MASS = re.compile(r"(\d+(?:\.\d+)?)\s*(kg|g)\b", re.I)
_NUM = re.compile(r"-?\d+(?:\.\d+)?")
_ORDINAL = {"first": 1, "second": 2, "third": 3, "fourth": 4, "last": -1}


def norm(s: str) -> str:
    return re.sub(r"[\s_\-]+", " ", s.lower()).strip()


def is_yes(text: str) -> bool:
    return bool(YES.match(text))


def match_option(text: str, options: list[str]) -> str | None:
    """The option the text names: the whole option, its index, or a word only that option has."""
    if not options:
        return None
    t = norm(text)
    whole = [o for o in options if re.search(rf"\b{re.escape(norm(o))}\b", t)]
    if whole:
        return max(whole, key=len)  # "small water bottle" beats "water bottle"
    m = re.fullmatch(r"(?:option\s*)?#?(\d+)", t)
    if m and 1 <= int(m.group(1)) <= len(options):
        return options[int(m.group(1)) - 1]
    for word, i in _ORDINAL.items():
        if re.search(rf"\b{word}\b", t):
            return options[i - 1 if i > 0 else -1] if abs(i) <= len(options) else None
    words = {o: set(norm(o).split()) for o in options}
    common = set.intersection(*words.values()) if len(options) > 1 else set()
    hits = [o for o, ws in words.items() if any(re.search(rf"\b{re.escape(w)}\b", t) for w in ws - common)]
    return hits[0] if len(hits) == 1 else None


def catalogue_values(key: str) -> dict[str, Any]:
    shape, size, mass, _ = catalogue.CATALOGUE[key]
    return {"shape": shape, "size": list(size), "mass_kg": mass}


def option_patch(spec: IntakeSpec, q: Question, option: str) -> dict[str, Any] | None:
    """The patch that answers ``q`` with ``option``."""
    if q.path == "objects":
        return {"objects": [{"name": option}]}
    if q.path.startswith("objects.") and option in catalogue.CATALOGUE:
        return {"objects": [{"name": q.path.split(".", 1)[1], **catalogue_values(option)}]}
    if q.path == "task.pick":
        return {"task": {"pick": option}}
    if q.path == "task.place":
        return {"task": {"place": [option]}}
    if q.path == "task.reset_policy":
        return {"task": {"reset_policy": option}}
    return None


def _object_dims(spec: IntakeSpec, name: str, text: str) -> dict[str, Any] | None:
    o = next((x for x in spec.objects if x.name == name), None)
    if o is None:
        return None
    scale = {"mm": 0.001, "cm": 0.01, "m": 1.0}
    lens = [float(v) * scale[u.lower()] for v, u in _LEN.findall(text)]
    mass = [float(v) * (1.0 if u.lower() == "kg" else 0.001) for v, u in _MASS.findall(text)]
    out: dict[str, Any] = {}
    need = {"box": 3, "cylinder": 2, "sphere": 1}[o.shape]
    if lens:
        if o.shape == "box" and len(lens) == 1:
            lens = lens * 3  # "a 4 cm cube"
        if len(lens) >= need:
            out.update(shape=o.shape, size=[round(v, 4) for v in lens[:need]])
    if mass:
        out["mass_kg"] = round(mass[0], 4)
    return {"objects": [{"name": name, **out}]} if out else None


def _set_path(path: str, value: Any) -> dict[str, Any]:
    out: Any = value
    for part in reversed(path.split(".")):
        out = {part: out}
    return out


def answer(spec: IntakeSpec, q: Question, text: str) -> dict[str, Any] | None:
    """A patch for ``q`` from free text, or None if the text does not answer it."""
    if q.path == "objects":
        keys = catalogue.mentioned(text) or ([k] if (k := catalogue.lookup(text)) else [])
        return {"objects": [{"name": k} for k in keys]} if keys else None
    if q.path.startswith("objects.") and q.type == "confirm":
        dims = _object_dims(spec, q.path.split(".", 1)[1], text)
        if dims:
            return dims
    opt = match_option(text, q.options)
    if opt is not None:
        p = option_patch(spec, q, opt)
        if p is not None:
            return p
    if q.type == "number" and (m := _NUM.search(text)):
        return _set_path(q.path, float(m.group()))
    if is_yes(text) and q.patch:
        return q.patch
    return None


def option_buttons(spec: IntakeSpec, q: Question, limit: int = 8) -> list[dict[str, Any]]:
    """One-click answers for a question card: each option's patch, or the ready-made default."""
    out = []
    for o in q.options[:limit]:
        p = option_patch(spec, q, o)
        if p is not None:
            out.append({"label": o.replace("_", " "), "patch": p})
    if not out and q.patch:
        label = {"confirm": "Confirm", "list": "Use the suggested defaults"}.get(q.type, "Accept the default")
        out.append({"label": label, "patch": q.patch})
    return out
