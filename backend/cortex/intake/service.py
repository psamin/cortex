"""Intake: build an IntakeSpec from a prompt and agent patches, and say what is still missing.

Pure functions (no I/O). The tool layer persists the spec on the project.

Patch semantics (``apply_patch``):
  * dicts merge recursively; scalars and plain lists replace;
  * ``objects``, ``zones``, ``cameras`` and ``environment.obstacles`` are lists keyed by
    ``name``: an item merges into the existing item with that name (object names also
    match through the catalogue, so "bottle" updates ``water_bottle``),
    ``{"name": n, "_delete": true}`` removes it, and an unknown name is added (objects
    given only a name are looked up in the catalogue);
  * a field the agent sets is removed from its owner's ``assumed`` set, and
    ``{"name": n, "confirm": true}`` confirms all of an object's assumed values;
  * ``assumed`` itself cannot be patched, and unknown keys are errors.
"""

from __future__ import annotations

import copy
import json
from typing import Any

from pydantic import ValidationError

from . import catalogue
from .spec import SO101_JOINTS, TEXT, IntakeSpec, IntakeStatus, Question

KEYED_LISTS = {"objects", "zones", "cameras", "environment.obstacles"}
MAX_PATCH_CHARS = 20_000
BASE_CLEARANCE = 0.09  # m around the arm base that obstacles must leave free
# so101 actuator ctrlrange from the Menagerie model, in so101_follower units.
SO101_LIMITS = [(-110.0, 110.0), (-100.0, 100.0), (-96.8, 96.8), (-95.0, 95.0), (-157.2, 162.8), (0.0, 100.0)]
DEFAULT_ZONES = [{"name": "spot_a", "center": [0.22, 0.12]}, {"name": "spot_b", "center": [0.22, -0.12]}]


class IntakeError(ValueError):
    """A patch Cortex refuses; the message says why and what would be accepted."""


def start(prompt: str = "", mode: str = "sim") -> IntakeSpec:
    # the spec keeps the first TEXT characters; objects are seeded from the whole prompt (a long one failed the intake)
    spec = IntakeSpec(prompt=prompt.strip()[:TEXT], mode=mode)  # type: ignore[arg-type]
    for key in catalogue.mentioned(prompt):
        obj = catalogue.from_catalogue(key, key)
        if obj is not None:
            spec.objects.append(obj)
    if len(spec.objects) == 1:
        spec.task.pick = spec.objects[0].name
    return spec


def apply_patch(spec: IntakeSpec, patch: dict[str, Any]) -> IntakeSpec:
    if not isinstance(patch, dict):
        raise IntakeError("patch must be an object")
    if len(json.dumps(patch, default=str)) > MAX_PATCH_CHARS:
        raise IntakeError(f"patch is too large (over {MAX_PATCH_CHARS} characters); send it in parts")
    _reject_assumed(patch, "")
    merged = _merge(spec.model_dump(mode="json"), patch, path="")
    try:
        out = IntakeSpec.model_validate(merged)
    except ValidationError as e:
        raise IntakeError(_short(e)) from e
    _fit_workspace_to_table(out)
    _check_robot(out)
    _check_refs(out)
    return out


TABLE_CENTER_X = 0.20  # the upstream table is centred 0.2 m in front of the arm base
TABLE_EDGE = 0.02  # keep objects this far from the edge


def _fit_workspace_to_table(spec: IntakeSpec) -> None:
    """A default (assumed) workspace is clipped to the table; one the agent set must fit on it."""
    hx, hy = spec.environment.table_size[0] / 2 - TABLE_EDGE, spec.environment.table_size[1] / 2 - TABLE_EDGE
    tx, ty = (TABLE_CENTER_X - hx, TABLE_CENTER_X + hx), (-hy, hy)
    ws = spec.workspace
    for axis, (lo, hi) in (("x", tx), ("y", ty)):
        cur = getattr(ws, axis)
        if axis in ws.assumed:
            setattr(ws, axis, (max(cur[0], lo), min(cur[1], hi)))
        elif cur[0] < lo - 1e-9 or cur[1] > hi + 1e-9:
            raise IntakeError(f"workspace.{axis} {cur} runs off the table ({lo:.2f}, {hi:.2f}); "
                              "make the table bigger or the workspace smaller")
        if getattr(ws, axis)[0] >= getattr(ws, axis)[1]:
            raise IntakeError(f"the table is too small for the arm's workspace along {axis}")


def status(spec: IntakeSpec, previous: IntakeSpec | None = None) -> IntakeStatus:
    qs = questions(spec)
    return IntakeStatus(spec=spec, questions=qs, complete=not any(q.required for q in qs), notes=_notes(spec),
                        changed=diff(previous, spec) if previous is not None else [],
                        still_assumed=assumed_paths(spec))


def questions(spec: IntakeSpec) -> list[Question]:
    q: list[Question] = []
    names = [o.name for o in spec.objects]
    if not spec.objects:
        q.append(Question(path="objects", type="list", prompt="Which objects are on the table?",
                          why="Cortex builds the sim scene and the grasp plan from them.",
                          options=sorted(catalogue.CATALOGUE), patch={"objects": [{"name": "cube"}]}))
    for o in spec.objects:
        key = catalogue.lookup(o.name)
        family = next((f for f, members in catalogue.VARIANTS.items() if key in members), None)
        if o.assumed and family:
            q.append(Question(
                path=f"objects.{o.name}", type="enum",
                prompt=f"Which {family} is it? Options differ in size and weight, which decides if the arm can grasp it.",
                why="Grasp width, payload and sim fidelity depend on the exact object.",
                options=catalogue.VARIANTS[family],
                patch={"objects": [{"name": o.name, **_catalogue_values(catalogue.VARIANTS[family][-1])}]},
                option_patches={v: {"objects": [{"name": o.name, **_catalogue_values(v)}]}
                                for v in catalogue.VARIANTS[family]}))
        elif o.assumed:
            q.append(Question(path=f"objects.{o.name}", type="confirm",
                              prompt=f"Confirm or correct {o.name}: {o.shape}, size {o.size} m, {o.mass_kg} kg.",
                              why="These are catalogue defaults; grasp, payload and sim fidelity depend on them.",
                              patch={"objects": [{"name": o.name, "confirm": True}]}))
    if spec.task.pick is None:
        q.append(Question(path="task.pick", type="enum", prompt="Which object should the policy pick up?",
                          why="The task's pick target.", options=names,
                          patch={"task": {"pick": names[0]}} if names else None))
    if not spec.zones:
        q.append(Question(path="zones", type="list", prompt="Where on the table can objects be placed?",
                          why="Place targets and reset spots; table frame, metres, x away from the arm base, "
                              f"y to its left; workspace x {spec.workspace.x}, y {spec.workspace.y}.",
                          patch={"zones": DEFAULT_ZONES}))
    zone_names = [z.name for z in spec.zones]
    if not spec.task.place:
        q.append(Question(path="task.place", type="list", prompt="Which zone is the place target?",
                          why="The first zone listed is the target; the others are used by cycle_zones resets.",
                          options=zone_names, patch={"task": {"place": zone_names[:1]}} if zone_names else None))
    elif spec.task.reset_policy != "return_to_origin" and len(spec.task.place) < 2:
        q.append(Question(path="task.place", type="list",
                          prompt=f"{spec.task.reset_policy} needs at least two zones in task.place.",
                          why="Resets move the object to another zone, so there must be one to move it to.",
                          options=zone_names, patch={"task": {"place": zone_names[:2]}} if len(zone_names) > 1 else None))
    q.append(Question(path="task.reset_policy", type="enum", required=False,
                      prompt="After each demo, put the object back where it started or somewhere new?",
                      why="random_zone and cycle_zones give positional diversity for free.",
                      options=["return_to_origin", "random_zone", "cycle_zones"],
                      patch={"task": {"reset_policy": spec.task.reset_policy}}))
    env = spec.environment
    if env.assumed:
        q.append(Question(path="environment", type="object", required=False,
                          prompt="Describe the surroundings so the sim matches them: table size and colour, lighting, "
                                 "and any obstacles (shelves, walls) near the arm.",
                          why="Scenes that look and collide like the real one transfer better.",
                          patch={"environment": {"lighting": env.lighting, "table_size": list(env.table_size)}}))
    if spec.mode == "real":
        q += _real_questions(spec)
    return q


def _real_questions(spec: IntakeSpec) -> list[Question]:
    q: list[Question] = []
    r = spec.robot
    if not r.port:
        q.append(Question(path="robot.port", type="string", prompt="Which serial port is the follower arm on?",
                          why="Cortex connects through LeRobot's so101_follower driver.",
                          patch={"robot": {"port": "COM5"}}))
    if not r.calibration_file:
        q.append(Question(path="robot.calibration_file", type="string",
                          prompt="Path to the arm's LeRobot calibration file?",
                          why="Real motion is refused without calibration; joint offsets also tune the sim arm."))
    if "home_pose" in r.assumed:
        q.append(Question(path="robot.home_pose", type="list",
                          prompt=f"Confirm the home pose {r.home_pose} (degrees, gripper 0-100).",
                          why="Every demo starts and ends here; it must be collision-free on your table.",
                          patch={"robot": {"home_pose": r.home_pose}}))
    overhead = [c for c in spec.cameras if c.role == "overhead"]
    if not overhead or not overhead[0].device:
        q.append(Question(path="cameras", type="list", prompt="Which camera looks down at the table?",
                          why="Astra points at objects in it and the verifier checks success with it.",
                          patch={"cameras": [{"name": "top", "role": "overhead", "device": "0"}]}))
    elif overhead[0].pos is None:
        q.append(Question(path="cameras", type="object",
                          prompt="The overhead camera is not calibrated. Run calibrate_cameras.",
                          why="Pixel-to-table mapping for Astra and the verifier needs the camera pose."))
    return q


def _notes(spec: IntakeSpec) -> list[str]:
    notes = []
    if "home_pose" in spec.robot.assumed and spec.mode == "sim":
        notes.append(f"robot.home_pose is the SO-101 default {spec.robot.home_pose}.")
    if spec.workspace.assumed:
        notes.append(f"workspace is the SO-101 default reach box x {spec.workspace.x}, y {spec.workspace.y}.")
    return notes


def assumed_paths(spec: IntakeSpec) -> list[str]:
    out = [f"robot.{a}" for a in sorted(spec.robot.assumed)]
    out += [f"workspace.{a}" for a in sorted(spec.workspace.assumed)]
    out += [f"environment.{a}" for a in sorted(spec.environment.assumed)]
    out += [f"objects.{o.name}.{a}" for o in spec.objects for a in sorted(o.assumed)]
    return out


def diff(a: IntakeSpec | None, b: IntakeSpec) -> list[str]:
    out: list[str] = []
    _diff(a.model_dump(mode="json") if a else {}, b.model_dump(mode="json"), "", out)
    return out


def _diff(a: Any, b: Any, path: str, out: list[str]) -> None:
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k != "assumed":
                _diff(a.get(k), b.get(k), f"{path}.{k}" if path else k, out)
    elif isinstance(a, list) and isinstance(b, list) and path in KEYED_LISTS:
        ka = {x["name"]: x for x in a}
        kb = {x["name"]: x for x in b}
        for n in sorted(set(ka) | set(kb)):
            _diff(ka.get(n), kb.get(n), f"{path}.{n}", out)
    elif a != b:
        out.append(path)


# ------------------------------------------------------------------ merge


def _reject_assumed(patch: Any, path: str) -> None:
    if isinstance(patch, dict):
        if "assumed" in patch:
            raise IntakeError(f"{path or 'patch'}: 'assumed' is managed by Cortex; set the value or send confirm")
        for k, v in patch.items():
            _reject_assumed(v, f"{path}.{k}" if path else k)
    elif isinstance(patch, list):
        for v in patch:
            _reject_assumed(v, path)


def _merge(base: Any, patch: Any, path: str) -> Any:
    if isinstance(base, dict) and isinstance(patch, dict):
        out = dict(base)
        for k, v in patch.items():
            sub = f"{path}.{k}" if path else k
            if sub in KEYED_LISTS:
                out[k] = _merge_keyed(base.get(k) or [], v, sub)
            elif k in out:
                out[k] = _merge(out[k], v, sub)
            else:
                out[k] = copy.deepcopy(v)
            if isinstance(out.get("assumed"), list) and k in out["assumed"]:
                out["assumed"] = [a for a in out["assumed"] if a != k]
        return out
    return copy.deepcopy(patch)


def _resolve_object_name(name: str, items: dict[str, dict]) -> str:
    """Existing item for a free-text object name: exact, slugged, or the same catalogue key."""
    if name in items:
        return name
    s = catalogue.slug(name)
    if s in items:
        return s
    key = catalogue.lookup(name)
    if key is not None:
        for existing in items:
            if catalogue.lookup(existing) == key or catalogue.lookup(existing.replace("_", " ")) == key:
                return existing
    return s


def _merge_keyed(base: list[dict], patch: Any, field: str) -> list[dict]:
    if not isinstance(patch, list):
        raise IntakeError(f"{field} must be a list of objects with a 'name'")
    items = {b["name"]: b for b in base}
    order = [b["name"] for b in base]
    for p in patch:
        if not isinstance(p, dict) or not p.get("name"):
            raise IntakeError(f"every {field} item needs a 'name'")
        p = dict(p)
        name = (_resolve_object_name(p.pop("name"), items) if field == "objects"
                else catalogue.slug(str(p.pop("name"))))
        if p.pop("_delete", False):
            items.pop(name, None)
            order = [n for n in order if n != name]
            continue
        if field == "objects" and p.pop("confirm", False):
            if name not in items:
                raise IntakeError(f"cannot confirm unknown object '{name}'")
            items[name] = {**items[name], "assumed": []}
        if name in items:
            items[name] = _merge(items[name], p, f"{field}.{name}")
            continue
        if field == "objects" and not {"shape", "size", "mass_kg"} <= p.keys():
            known = catalogue.from_catalogue(name)
            if known is None:
                raise IntakeError(f"'{name}' is not in the catalogue ({', '.join(sorted(catalogue.CATALOGUE))}); "
                                  "give shape, size and mass_kg")
            p = {**known.model_dump(mode="json"), **p, "assumed": sorted(set(known.assumed) - set(p))}
        items[name] = {**p, "name": name}
        order.append(name)
    return [items[n] for n in order]


def _catalogue_values(key: str) -> dict[str, Any]:
    shape, size, mass, _ = catalogue.CATALOGUE[key]
    return {"shape": shape, "size": size, "mass_kg": mass}


def _check_robot(spec: IntakeSpec) -> None:
    limits = spec.robot.joint_limits
    if limits is not None:
        if len(limits) != 6:
            raise IntakeError("joint_limits needs 6 (low, high) pairs, one per joint")
        for joint, (lo, hi), (mlo, mhi) in zip(SO101_JOINTS, limits, SO101_LIMITS):
            if lo >= hi or lo < mlo or hi > mhi:
                raise IntakeError(f"{joint} limits ({lo}, {hi}) must narrow the model's ({mlo}, {mhi})")
    for joint, v, (lo, hi) in zip(SO101_JOINTS, spec.robot.home_pose, limits or SO101_LIMITS):
        if not lo <= v <= hi:
            raise IntakeError(f"home pose {joint}={v} is outside the joint limits ({lo}, {hi})")


def _check_refs(spec: IntakeSpec) -> None:
    names = {o.name for o in spec.objects}
    zones = {z.name for z in spec.zones}
    if len(names) != len(spec.objects):
        raise IntakeError("object names must be unique")
    if spec.task.pick is not None and spec.task.pick not in names:
        raise IntakeError(f"task.pick '{spec.task.pick}' is not an object ({', '.join(sorted(names)) or 'none yet'})")
    missing = [z for z in spec.task.place if z not in zones]
    if missing:
        raise IntakeError(f"task.place names unknown zones {missing} (known: {', '.join(sorted(zones)) or 'none'})")
    (x0, x1), (y0, y1) = spec.workspace.x, spec.workspace.y
    for z in spec.zones:
        x, y = z.center
        if not (x0 <= x <= x1 and y0 <= y <= y1):
            raise IntakeError(f"zone '{z.name}' at ({x}, {y}) is outside the workspace x {spec.workspace.x}, "
                              f"y {spec.workspace.y} (metres, table frame)")
    for o in spec.objects:
        if o.start is not None and not (x0 <= o.start[0] <= x1 and y0 <= o.start[1] <= y1):
            raise IntakeError(f"object '{o.name}' start {o.start} is outside the workspace")
    _check_obstacles(spec)


def _footprint(shape: str, size: list[float]) -> tuple[float, float]:
    """Half extents of an obstacle on the table (axis-aligned)."""
    if shape == "box":
        return size[0] / 2, size[1] / 2
    return size[0] / 2, size[0] / 2


def _check_obstacles(spec: IntakeSpec) -> None:
    """Obstacles may not cover the arm base, a place zone or an object's start."""
    points = [("the arm base", (0.0, 0.0), BASE_CLEARANCE)]
    points += [(f"zone '{z.name}'", tuple(z.center), max(z.half_size)) for z in spec.zones]
    points += [(f"object '{o.name}' start", tuple(o.start), o.grasp_width / 2) for o in spec.objects if o.start]
    for ob in spec.environment.obstacles:
        hx, hy = _footprint(ob.shape, ob.size)
        for what, (px, py), r in points:
            if abs(px - ob.center[0]) < hx + r and abs(py - ob.center[1]) < hy + r:
                raise IntakeError(f"obstacle '{ob.name}' overlaps {what}; move it or make it smaller")


def _short(e: ValidationError) -> str:
    return "; ".join(f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in e.errors()[:5])
