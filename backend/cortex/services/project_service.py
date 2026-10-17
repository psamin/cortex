"""Turn a natural-language task into a task spec, success criteria and an
initial collection plan.

The MVP supports one task family (tabletop pick-and-place on an SO-101), so the
spec is built deterministically from the prompt; unsupported requests are
mapped onto it and the mismatch is reported back rather than hidden.
"""

from __future__ import annotations

import re

from ..models import (
    CollectionPlan,
    Condition,
    CreateProjectRequest,
    Mode,
    PlanCell,
    Project,
    SuccessCriterion,
    TaskSpec,
)

REGIONS = ["left", "center", "right"]
LIGHTS = ["normal", "dim", "bright"]


def parse_objects(prompt: str) -> tuple[str, str]:
    p = prompt.lower()
    obj = next((w for w in ("cube", "block", "box", "ball") if w in p), "cube")
    target = next((w for w in ("bowl", "plate", "bin", "box", "tray") if w in p and w != obj), "bowl")
    return obj, target


def build_spec(req: CreateProjectRequest) -> TaskSpec:
    obj, target = parse_objects(req.task_prompt)
    return TaskSpec(
        task=req.task_prompt.strip(),
        robot="so101",
        objects=[obj, target],
        goal=f"pick up the {obj} and place it in the {target}",
        success_criteria=[
            SuccessCriterion(id="lifted", description=f"{obj} leaves the table",
                             measurable=f"{obj} height > 3 cm above table at some step"),
            SuccessCriterion(id="placed", description=f"{obj} ends inside the {target}",
                             measurable=f"{obj} centre within {target} radius and below its rim at episode end"),
            SuccessCriterion(id="released", description="gripper releases the object",
                             measurable="gripper open and not in contact at episode end"),
        ],
        condition_axes={"cube_region": REGIONS, "lighting": LIGHTS},
    )


def initial_plan(budget: int) -> CollectionPlan:
    """Most of the budget goes to normal lighting across all three positions
    (where the policy must work first). A smaller share probes lighting so
    coverage gaps there are visible from day one."""
    core = [Condition(cube_region=r, lighting="normal") for r in REGIONS]
    probes = [Condition(cube_region="center", lighting=lt) for lt in ("dim", "bright")]
    core_each = max(1, round(budget * 0.75 / len(core)))
    probe_each = max(1, (budget - core_each * len(core)) // len(probes))
    cells = [PlanCell(condition=c, target=core_each,
                      reason="core coverage: policy must handle every start position") for c in core]
    cells += [PlanCell(condition=c, target=probe_each,
                       reason="lighting probe: detect visual brittleness early") for c in probes]
    return CollectionPlan(cells=cells, budget_total=sum(c.target for c in cells))


def create_project(req: CreateProjectRequest) -> Project:
    spec = build_spec(req)
    name = req.name or _title(req.task_prompt)
    p = Project(name=name, task_prompt=req.task_prompt, mode=req.mode, spec=spec,
                plan=initial_plan(req.demo_budget))
    if req.autopilot:
        p.autopilot = req.autopilot
    return p


def _title(prompt: str) -> str:
    words = re.sub(r"[^a-zA-Z0-9 ]", "", prompt).split()
    words = [w for w in words if w.lower() not in {"i", "want", "my", "to", "a", "the", "teach", "robot"}]
    return " ".join(words[:5]).capitalize() or "Untitled task"


def unsupported_notes(req: CreateProjectRequest) -> list[str]:
    notes = []
    if req.mode == Mode.real:
        notes.append("Real-robot mode records through the local LeRobot bridge and needs explicit approval for every motion.")
    if not re.search(r"pick|place|put|move", req.task_prompt.lower()):
        notes.append("Only pick-and-place tasks are supported in this version; the task was mapped onto pick-and-place.")
    return notes
