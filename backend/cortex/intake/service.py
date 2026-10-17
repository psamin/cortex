"""Intake: build an IntakeSpec from a prompt and agent patches, and say what is still missing.

Pure functions (no I/O). The tool layer persists the spec on the project.

Patch semantics (``apply_patch``):
  * dicts merge recursively; scalars and plain lists replace;
  * ``objects``, ``zones`` and ``cameras`` are lists keyed by ``name``: an item merges into
    the existing item with that name, ``{"name": n, "_delete": true}`` removes it, and an
    unknown name is added (objects with only a name are looked up in the catalogue);
  * a field the agent sets is removed from that item's ``assumed`` set, and
    ``{"name": n, "confirm": true}`` confirms all of an object's assumed values.
"""

from __future__ import annotations

import copy
from typing import Any

from pydantic import ValidationError

from . import catalogue
from .spec import SO101_JOINTS, IntakeSpec, IntakeStatus, Question

KEYED_LISTS = ("objects", "zones", "cameras")
# so101 actuator ctrlrange from the Menagerie model, in so101_follower units.
SO101_LIMITS = [(-110.0, 110.0), (-100.0, 100.0), (-96.8, 96.8), (-95.0, 95.0), (-157.2, 162.8), (0.0, 100.0)]
DEFAULT_ZONES = [{"name": "spot_a", "center": [0.22, 0.12]}, {"name": "spot_b", "center": [0.22, -0.12]}]


class IntakeError(ValueError):
    """A patch Cortex refuses; the message says why and what would be accepted."""


def start(prompt: str = "", mode: str = "sim") -> IntakeSpec:
    spec = IntakeSpec(prompt=prompt.strip(), mode=mode)  # type: ignore[arg-type]
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
    base = spec.model_dump(mode="json")
    merged = _merge(base, patch, path="")
    try:
        out = IntakeSpec.model_validate(merged)
    except ValidationError as e:
        raise IntakeError(_short(e)) from e
    _check_limits(out)
    _check_refs(out)
    return out


def status(spec: IntakeSpec) -> IntakeStatus:
    qs = questions(spec)
    return IntakeStatus(spec=spec, questions=qs, complete=not any(q.required for q in qs), notes=_notes(spec))


def questions(spec: IntakeSpec) -> list[Question]:
    q: list[Question] = []
    names = [o.name for o in spec.objects]
    if not spec.objects:
        q.append(Question(path="objects", type="list", prompt="Which objects are on the table?",
                          why="Cortex builds the sim scene and the grasp plan from them.",
                          default=[{"name": "water bottle"}],
                          options=sorted(catalogue.CATALOGUE)))
    for o in spec.objects:
        if o.assumed:
            q.append(Question(path=f"objects[{o.name}]", type="confirm",
                              prompt=f"Confirm or correct {o.name}: {o.shape}, size {o.size} m, {o.mass_kg} kg.",
                              why="These are catalogue defaults; grasp, payload and sim fidelity depend on them.",
                              default={"name": o.name, "confirm": True}))
    if spec.task.pick is None:
        q.append(Question(path="task.pick", type="enum", prompt="Which object should the policy pick up?",
                          why="The task's pick target.", options=names))
    if not spec.zones:
        q.append(Question(path="zones", type="list", prompt="Where on the table can objects be placed?",
                          why="Place targets and reset spots; table frame, metres, x away from the base.",
                          default=DEFAULT_ZONES))
    if not spec.task.place:
        q.append(Question(path="task.place", type="list", prompt="Which zone is the place target?",
                          why="The first zone listed is the target; the others are used by cycle_zones resets.",
                          options=[z.name for z in spec.zones]))
    q.append(Question(path="task.reset_policy", type="enum", required=False,
                      prompt="After each demo, put the object back where it started or somewhere new?",
                      why="random_zone and cycle_zones give positional diversity for free.",
                      options=["return_to_origin", "random_zone", "cycle_zones"], default=spec.task.reset_policy))
    if spec.mode == "real":
        q += _real_questions(spec)
    return q


def _real_questions(spec: IntakeSpec) -> list[Question]:
    q: list[Question] = []
    r = spec.robot
    if not r.port:
        q.append(Question(path="robot.port", type="string", prompt="Which serial port is the follower arm on?",
                          why="Cortex connects through LeRobot's so101_follower driver.",
                          default="COM5"))
    if not r.calibration_file:
        q.append(Question(path="robot.calibration_file", type="string",
                          prompt="Path to the arm's LeRobot calibration file?",
                          why="Real motion is refused without calibration; joint offsets also tune the sim arm."))
    if "home_pose" in r.assumed:
        q.append(Question(path="robot.home_pose", type="list",
                          prompt=f"Confirm the home pose {r.home_pose} (degrees, gripper 0-100).",
                          why="Every demo starts and ends here; it must be collision-free on your table.",
                          default=r.home_pose))
    overhead = [c for c in spec.cameras if c.role == "overhead"]
    if not overhead or not overhead[0].device:
        q.append(Question(path="cameras", type="list", prompt="Which camera looks down at the table?",
                          why="Astra points at objects in it and the verifier checks success with it.",
                          default=[{"name": "top", "role": "overhead", "device": "0"}]))
    elif overhead[0].pos is None:
        q.append(Question(path="cameras", type="object",
                          prompt="The overhead camera is not calibrated. Run calibrate_cameras.",
                          why="Pixel-to-table mapping for Astra and the verifier needs the camera pose."))
    return q


def _notes(spec: IntakeSpec) -> list[str]:
    notes = []
    for o in spec.objects:
        if catalogue.lookup(o.name) is None and o.assumed:
            notes.append(f"{o.name}: not in the catalogue; give shape, size and mass.")
    return notes


# ------------------------------------------------------------------ merge


def _merge(base: Any, patch: Any, path: str) -> Any:
    if isinstance(base, dict) and isinstance(patch, dict):
        out = dict(base)
        for k, v in patch.items():
            sub = f"{path}.{k}" if path else k
            if k in KEYED_LISTS and path == "":
                out[k] = _merge_keyed(base.get(k) or [], v, k)
            elif k in out:
                out[k] = _merge(out[k], v, sub)
                if isinstance(out.get("assumed"), list) and k in out["assumed"]:
                    out["assumed"] = [a for a in out["assumed"] if a != k]
            else:
                out[k] = copy.deepcopy(v)
        return out
    return copy.deepcopy(patch)


def _merge_keyed(base: list[dict], patch: Any, field: str) -> list[dict]:
    if not isinstance(patch, list):
        raise IntakeError(f"{field} must be a list of objects with a 'name'")
    items = {b["name"]: b for b in base}
    order = [b["name"] for b in base]
    for p in patch:
        if not isinstance(p, dict) or not p.get("name"):
            raise IntakeError(f"every {field} item needs a 'name'")
        name = p["name"]
        if field == "objects" and name not in items:
            known = catalogue.from_catalogue(name)
            if known is not None and known.name in items:
                name = known.name
        if p.get("_delete"):
            items.pop(name, None)
            order = [n for n in order if n != name]
            continue
        p = {k: v for k, v in p.items() if k not in ("_delete",)}
        if field == "objects" and p.pop("confirm", False) and name in items:
            items[name] = {**items[name], "assumed": []}
        if name in items:
            items[name] = _merge(items[name], {k: v for k, v in p.items() if k != "name"}, f"{field}[{name}]")
        else:
            if field == "objects" and not {"shape", "size", "mass_kg"} <= p.keys():
                known = catalogue.from_catalogue(name)
                if known is None:
                    raise IntakeError(f"'{name}' is not in the catalogue ({', '.join(sorted(catalogue.CATALOGUE))}); "
                                      "give shape, size and mass_kg")
                given = {k for k in p if k not in ("name",)}
                p = {**known.model_dump(mode="json"), **{k: v for k, v in p.items() if k != "name"}}
                p["assumed"] = sorted(set(known.assumed) - given)
                name = known.name
            items[name] = {**p, "name": name}
            order.append(name)
    return [items[n] for n in order]


def _check_limits(spec: IntakeSpec) -> None:
    lims = spec.robot.joint_limits
    if lims is None:
        return
    if len(lims) != 6:
        raise IntakeError("joint_limits needs 6 (low, high) pairs, one per joint")
    for joint, (lo, hi), (mlo, mhi) in zip(SO101_JOINTS, lims, SO101_LIMITS):
        if lo >= hi or lo < mlo or hi > mhi:
            raise IntakeError(f"{joint} limits ({lo}, {hi}) must narrow the model's ({mlo}, {mhi})")
    for joint, v, (lo, hi) in zip(SO101_JOINTS, spec.robot.home_pose, lims):
        if not lo <= v <= hi:
            raise IntakeError(f"home pose {joint}={v} is outside the narrowed limits ({lo}, {hi})")


def _check_refs(spec: IntakeSpec) -> None:
    names = {o.name for o in spec.objects}
    zones = {z.name for z in spec.zones}
    if spec.task.pick is not None and spec.task.pick not in names:
        raise IntakeError(f"task.pick '{spec.task.pick}' is not an object ({', '.join(sorted(names)) or 'none yet'})")
    missing = [z for z in spec.task.place if z not in zones]
    if missing:
        raise IntakeError(f"task.place names unknown zones {missing} (known: {', '.join(sorted(zones)) or 'none'})")
    if len({o.name for o in spec.objects}) != len(spec.objects):
        raise IntakeError("object names must be unique")


def _short(e: ValidationError) -> str:
    return "; ".join(f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in e.errors()[:5])
