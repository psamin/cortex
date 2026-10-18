"""Where an intake scene comes from: a photo of the user's setup, their detailed description, or the default.

  photo        the user's picture. Cortex keeps it with the project and asks the user's Claude Code
               (headless `claude -p`, Read tool only, no MCP servers, no shell) to estimate the scene:
               objects (catalogue items where they match), where they start, place zones, the table,
               lighting and obstacles.
  description  the user's own words, read the same way (no image).
  default      the built-in tabletop: a 3 cm red cube and a target zone. No model call.
  prompt       (no scene source) the task prompt only; catalogue objects it names are pre-filled.

A model's reading of a photo is an estimate, not a measurement. The estimate goes through
intake.apply_patch like any agent's patch (same validation), positions are pulled inside the arm's
workspace, and every estimated object stays `assumed` so the human is asked to confirm it. The result
says what was estimated and what Cortex moved or left out (`adjusted`, with reasons), so the agent can show the human before building.
"""

from __future__ import annotations

import base64
import binascii
import io
import json
import re
import uuid
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from ..context import Ctx
from ..errors import UserError
from ..intake import IntakeSpec, apply_patch, catalogue, start
from ..intake.service import IntakeError
from ..intake.spec import MAX_OBJECTS, MAX_OBSTACLES, MAX_ZONES, TEXT

SceneSourceKind = Literal["prompt", "photo", "description", "default"]
MAX_IMAGE_BYTES = 15 * 1024 * 1024
MAX_SIDE_PX = 1568  # longest side Claude reads at full detail
TIMEOUT_S = 240
DEFAULT_PROMPT = "Pick up the red cube and place it in the target zone."
DEFAULT_PATCHES: list[dict[str, Any]] = [
    {"objects": [{"name": "cube", "start": [0.22, 0.12]}], "zones": [{"name": "target", "center": [0.22, -0.12]}],
     "task": {"pick": "cube", "place": ["target"]}},
    {"objects": [{"name": "cube", "confirm": True}]},  # the default cube is known, not a guess
]


class SceneSource(BaseModel):
    """How the human wants the scene set up. `kind` defaults from what is given: an image means photo,
    a description means description, neither means the prompt alone."""

    kind: SceneSourceKind | None = None
    image_b64: str | None = Field(None, max_length=MAX_IMAGE_BYTES * 4 // 3 + 64)
    description: str | None = Field(None, max_length=4000)

    @model_validator(mode="after")
    def _resolve(self) -> SceneSource:
        if self.kind is None:
            self.kind = "photo" if self.image_b64 else "description" if self.description else "prompt"
        if self.kind == "photo" and not self.image_b64:
            raise ValueError("scene_source 'photo' needs the photo (image_b64, or image_path through the MCP server)")
        if self.kind == "description" and not (self.description or "").strip():
            raise ValueError("scene_source 'description' needs a description of the setup")
        return self


# ------------------------------------------------------------------ the photo

def decode_photo(image_b64: str) -> bytes:
    """A JPEG of the photo (any common format in, longest side at most MAX_SIDE_PX), or a UserError."""
    from PIL import Image, UnidentifiedImageError

    raw = image_b64.split(",", 1)[1] if image_b64.startswith("data:") else image_b64
    try:
        data = base64.b64decode(raw, validate=False)
    except (binascii.Error, ValueError):
        raise UserError("the photo is not valid base64") from None
    if len(data) > MAX_IMAGE_BYTES:
        raise UserError(f"the photo is larger than {MAX_IMAGE_BYTES // 2**20} MB; send a smaller one")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError):
        raise UserError("the photo could not be read as an image (JPEG, PNG or WebP)") from None
    img = img.convert("RGB")
    img.thumbnail((MAX_SIDE_PX, MAX_SIDE_PX))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


# ------------------------------------------------------------------ asking Claude Code

def _catalogue_text() -> str:
    return "; ".join(f"{k} ({shape} {size} m, {mass} kg)" for k, (shape, size, mass, _) in catalogue.CATALOGUE.items())


PROMPT = """You are reconstructing a tabletop robot workcell so it can be rebuilt in a MuJoCo simulation.
{source}

The robot is an SO-101 arm on the table. Frame: metres from the arm base, +x forward (away from the arm),
+y to the arm's left, z up from the table. The table is centred 0.20 m in front of the base. The arm can
reach x {x0:.2f} to {x1:.2f} and y {y0:.2f} to {y1:.2f}; everything to pick or place must be in that box.
The user's task: "{task}"
Catalogue objects (when an object matches one, give its name in "catalogue"): {catalogue}

Answer with ONLY this JSON object (no prose, no code fence):
{{"objects": [{{"name": "snake_case", "catalogue": "catalogue name or null", "shape": "box|cylinder|sphere",
   "size": [...], "mass_kg": 0.1, "start": [x, y], "rgba": [r, g, b, 1]}}],
 "zones": [{{"name": "snake_case", "center": [x, y], "half_size": [hx, hy]}}],
 "pick": "object name or null", "place": "zone name or null",
 "environment": {{"lighting": "dim|normal|bright", "table_size": [width_x, depth_y], "table_rgba": [r, g, b, 1],
   "description": "one sentence about the surroundings",
   "obstacles": [{{"name": "snake_case", "shape": "box|cylinder|sphere", "size": [...], "center": [x, y]}}]}},
 "notes": ["anything you were unsure about"]}}
Rules: sizes in metres: box = [x, y, z] full extents, cylinder = [diameter, height], sphere = [diameter].
Objects are things the arm could pick up (under 5 kg, under 0.5 m); a bowl, tray, coaster or marked spot
the object goes onto or into is a zone (its footprint). Fixed things the arm must avoid are obstacles.
Estimate positions from what you can see or read (the arm base, table edges, known object sizes). If you
cannot tell, keep the value inside the reachable box and say so in notes. At most {n_obj} objects,
{n_zone} zones and {n_obs} obstacles. Colours are 0-1."""

PHOTO_SOURCE = "The user's photo of their setup is the image file ./{image} in the current directory. Open it with the Read tool first."
TEXT_SOURCE = 'The user described their setup:\n"""\n{description}\n"""'


async def ask_claude_code(ctx: Ctx, prompt: str, workdir: Path) -> dict[str, Any]:
    from . import claude_code_agent as cc
    from .claude_code_diagnosis import run_claude

    if why := cc.available():
        raise UserError(f"Reading the scene needs Claude Code on this machine ({why}). Use scene_source='default', "
                        "or describe the objects, zones and positions yourself with update_intake.")
    reply = await run_claude(prompt, workdir, ctx.settings.claude_code_model, timeout_s=TIMEOUT_S)
    m = re.search(r"\{.*\}", reply, re.DOTALL)
    try:
        data = json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        data = None
    if not isinstance(data, dict):
        raise UserError("Claude Code did not return a scene Cortex could read; try again, describe the setup in "
                        "words, or use scene_source='default'.")
    return data


# ------------------------------------------------------------------ the estimate -> intake patches

def _num(v: Any) -> float | None:
    return float(v) if isinstance(v, int | float) and not isinstance(v, bool) else None


def _vec(v: Any, n: int) -> list[float] | None:
    if isinstance(v, list | tuple) and len(v) == n and all(_num(x) is not None for x in v):
        return [float(x) for x in v]
    return None


def _rgba(v: Any) -> list[float] | None:
    c = _vec(v, 4) or (_vec(v, 3) and [*_vec(v, 3), 1.0])  # type: ignore[misc]
    return [min(max(x, 0.0), 1.0) for x in c] if c else None


def _clamp_xy(p: list[float], ws, notes: list[str], what: str) -> list[float]:
    (x0, x1), (y0, y1) = ws.x, ws.y
    q = [round(min(max(p[0], x0 + 0.01), x1 - 0.01), 3), round(min(max(p[1], y0 + 0.01), y1 - 0.01), 3)]
    if q != [round(p[0], 3), round(p[1], 3)]:
        notes.append(f"{what} at {p} was outside the arm's reach; moved to {q}")
    return q


def _shape_size(o: dict[str, Any]) -> tuple[str, list[float]] | None:
    shape = o.get("shape")
    need = {"box": 3, "cylinder": 2, "sphere": 1}.get(shape) if isinstance(shape, str) else None
    size = _vec(o.get("size"), need) if need else None
    if not size or any(s <= 0 or s > 0.5 for s in size):
        return None
    return shape, [round(s, 4) for s in size]  # type: ignore[return-value]


def environment_patch(est: dict[str, Any]) -> dict[str, Any]:
    env = est.get("environment") if isinstance(est.get("environment"), dict) else {}
    out: dict[str, Any] = {}
    if env.get("lighting") in ("dim", "normal", "bright"):
        out["lighting"] = env["lighting"]
    if (ts := _vec(env.get("table_size"), 2)) and all(0.3 <= t <= 2.0 for t in ts):
        out["table_size"] = [round(t, 3) for t in ts]
    if c := _rgba(env.get("table_rgba")):
        out["table_rgba"] = c
    if isinstance(env.get("description"), str) and env["description"].strip():
        out["description"] = env["description"].strip()[:TEXT]
    return {"environment": out} if out else {}


def scene_patch(est: dict[str, Any], ws, notes: list[str]) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    """(objects + zones + task patch, obstacle items, estimated object names) from Claude's reading."""
    objects, names = [], []
    for o in (est.get("objects") or [])[:MAX_OBJECTS]:
        if not isinstance(o, dict) or not isinstance(o.get("name"), str):
            continue
        name = catalogue.slug(o["name"])
        key = catalogue.lookup(o.get("catalogue") or "") if isinstance(o.get("catalogue"), str) else None
        key = key or catalogue.lookup(o["name"])
        if key:
            shape, size, mass, rgba = catalogue.CATALOGUE[key]
            item: dict[str, Any] = {"name": name, "shape": shape, "size": list(size), "mass_kg": mass, "rgba": list(rgba)}
        elif (ss := _shape_size(o)) and (m := _num(o.get("mass_kg"))) and 0 < m <= 5:
            item = {"name": name, "shape": ss[0], "size": ss[1], "mass_kg": round(m, 3)}
        else:
            notes.append(f"dropped object '{o['name']}': no catalogue match and no usable shape, size and mass")
            continue
        if c := _rgba(o.get("rgba")):
            item["rgba"] = c
        if p := _vec(o.get("start"), 2):
            item["start"] = _clamp_xy(p, ws, notes, f"object '{name}'")
        if name in names:
            continue
        objects.append(item)
        names.append(name)
    zones = []
    for z in (est.get("zones") or [])[:MAX_ZONES]:
        if not (isinstance(z, dict) and isinstance(z.get("name"), str) and (c := _vec(z.get("center"), 2))):
            continue
        hs = _vec(z.get("half_size"), 2) or [0.04, 0.04]
        zones.append({"name": catalogue.slug(z["name"]), "center": _clamp_xy(c, ws, notes, f"zone '{z['name']}'"),
                      "half_size": [round(min(max(h, 0.02), 0.10), 3) for h in hs]})
    patch: dict[str, Any] = {}
    if objects:
        patch["objects"] = objects
    if zones:
        patch["zones"] = zones
    task: dict[str, Any] = {}
    pick = catalogue.slug(est["pick"]) if isinstance(est.get("pick"), str) else None
    if pick in names:
        task["pick"] = pick
    elif len(names) == 1:
        task["pick"] = names[0]
    place = catalogue.slug(est["place"]) if isinstance(est.get("place"), str) else None
    if place in {z["name"] for z in zones}:
        task["place"] = [place]
    if task:
        patch["task"] = task
    env = est.get("environment") if isinstance(est.get("environment"), dict) else {}
    obstacles = []
    for ob in (env.get("obstacles") or [])[:MAX_OBSTACLES]:
        if isinstance(ob, dict) and isinstance(ob.get("name"), str) and (ss := _shape_size(ob)) \
                and (c := _vec(ob.get("center"), 2)):
            obstacles.append({"name": catalogue.slug(ob["name"]), "shape": ss[0], "size": ss[1], "center": c})
    return patch, obstacles, names


def _try(spec: IntakeSpec, patch: dict[str, Any], notes: list[str], what: str) -> IntakeSpec:
    if not patch:
        return spec
    try:
        return apply_patch(spec, patch)
    except IntakeError as e:
        notes.append(f"left out {what}: {e}")
        return spec


def apply_estimate(base: IntakeSpec, est: dict[str, Any]) -> tuple[IntakeSpec, dict[str, Any]]:
    """Apply Claude's reading in parts (table first, so positions are checked against the workspace it
    gives), dropping only the part that fails validation. Estimated objects stay `assumed`."""
    notes: list[str] = []
    spec = _try(base, environment_patch(est), notes, "the table and lighting")
    patch, obstacles, names = scene_patch(est, spec.workspace, notes)
    with_task = _try(spec, patch, notes, "the objects and zones")
    if with_task is spec and patch.get("task"):  # a bad task reference should not cost the objects
        with_task = _try(spec, {k: v for k, v in patch.items() if k != "task"}, notes, "the objects and zones")
    spec = with_task
    for ob in obstacles:  # one by one: an obstacle over a zone or the base is dropped, the rest stay
        spec = _try(spec, {"environment": {"obstacles": [ob]}}, notes, f"obstacle '{ob['name']}'")
    for o in spec.objects:
        if o.name in names:
            o.assumed = o.assumed | {"shape", "size", "mass_kg"} | ({"start"} if o.start else set())
    model_notes = [str(n)[:200] for n in (est.get("notes") or []) if isinstance(n, str | int | float)][:8]
    return spec, {"estimated_objects": names, "zones": [z.name for z in spec.zones],
                  "obstacles": [o.name for o in spec.environment.obstacles],
                  "adjusted": notes, "model_notes": model_notes}


# ------------------------------------------------------------------ entry point

def _n(items: list[Any], word: str) -> str:
    return f"{len(items)} {word}{'' if len(items) == 1 else 's'}"


async def initial_spec(ctx: Ctx, prompt: str, mode: str, src: SceneSource) -> tuple[IntakeSpec, dict[str, Any], bytes | None]:
    """(intake spec, scene_setup report, photo JPEG or None) for a new or restarted intake."""
    if src.kind == "default":
        spec = start("", mode)
        spec.prompt = (prompt.strip() or DEFAULT_PROMPT)[:TEXT]
        for p in DEFAULT_PATCHES:
            spec = apply_patch(spec, p)
        return spec, {"source": "default", "summary": "the default scene: a 3 cm red cube at (0.22, 0.12) and a "
                                                      "target zone at (0.22, -0.12)"}, None
    if src.kind == "prompt":
        return start(prompt, mode), {"source": "prompt", "summary": "objects named in the task prompt, from the "
                                                                     "catalogue"}, None
    # The photo or description is the one source of the objects (the model also reads the task), so the prompt
    # seeds none: a prompt's "mug" would otherwise absorb the photo's "red_mug" under another name.
    base = start("", mode)
    base.prompt = prompt.strip()[:TEXT]

    photo = decode_photo(src.image_b64) if src.kind == "photo" else None  # type: ignore[arg-type]
    workdir = ctx.data_dir / "scene_setup" / uuid.uuid4().hex[:12]
    workdir.mkdir(parents=True, exist_ok=True)
    if photo:
        (workdir / "photo.jpg").write_bytes(photo)
        source = PHOTO_SOURCE.format(image="photo.jpg")
        if src.description:
            source += "\nThe user also said: " + TEXT_SOURCE.format(description=src.description.strip())
    else:
        source = TEXT_SOURCE.format(description=(src.description or "").strip())
    (x0, x1), (y0, y1) = base.workspace.x, base.workspace.y
    prompt_text = PROMPT.format(source=source, x0=x0, x1=x1, y0=y0, y1=y1, task=prompt.strip() or "pick and place",
                                catalogue=_catalogue_text(), n_obj=MAX_OBJECTS, n_zone=MAX_ZONES, n_obs=MAX_OBSTACLES)
    est = await ask_claude_code(ctx, prompt_text, workdir)
    spec, report = apply_estimate(base, est)
    if src.description and not spec.environment.description:
        spec.environment.description = src.description.strip()[:TEXT]
    model = ctx.settings.claude_code_model or "claude-code"
    what = "the photo" if photo else "the description"
    report = {"source": src.kind, "model": f"Claude Code ({model})", **report,
              "summary": f"{_n(report['estimated_objects'], 'object')}, {_n(report['zones'], 'zone')} and "
                         f"{_n(report['obstacles'], 'obstacle')} estimated from {what} by Claude Code; confirm "
                         "them with the human, then build and compare the render with the real setup"}
    return spec, report, photo
