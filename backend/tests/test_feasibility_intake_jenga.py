"""#310 bug 3: an agent's Jenga intake spec made build_simulation raise inside feasibility.static_torque.

The drop zone at (0.26, 0.17) has no carry pose with the gripper pointing down, so IK returned no solution
and the payload check was handed None. Feasibility must report that as not feasible, with the reason."""

import json

from cortex import intake
from cortex.sim.feasibility import check
from cortex.sim.scene import build_scene
from cortex.sim.so101 import configure_gl

configure_gl()
PROMPT = ("Teach the robot to play Jenga: pull a single wooden Jenga block out of a stacked tower on the table "
          "without knocking the tower over, and place the extracted block in a drop zone to the side.")
# the agent's last intake patch before the 500 (project prj_55afe58378b9, from chat prj_6dbe063be1f2)
PATCH = {"objects": [{"name": "jenga_block", "shape": "box", "size": [0.025, 0.075, 0.015], "mass_kg": 0.02,
                      "rgba": [0.76, 0.6, 0.35, 1], "start": [0.175, 0], "start_jitter_m": 0.015}],
         "zones": [{"name": "drop_zone", "center": [0.26, 0.17]}],
         "task": {"pick": "jenga_block", "place": ["drop_zone"], "reset_policy": "return_to_origin"},
         "environment": {"lighting": "normal", "table_size": [0.8, 0.8], "description": "A Jenga tower behind the block.",
                         "obstacles": [{"name": "jenga_tower", "shape": "box", "size": [0.075, 0.075, 0.09],
                                        "center": [0.25, 0]}]}}


def test_unreachable_carry_pose_is_reported_not_raised(tmp_path):
    spec = intake.apply_patch(intake.start(PROMPT), PATCH)
    r = check(spec, build_scene(spec, tmp_path))
    assert not r.ok
    zone = next(p for p in r.places if p.name == "drop_zone")
    assert not zone.ok and zone.carry_err_mm is None and "carry pose" in (zone.detail or "")
    assert zone.suggestion is not None  # a reachable spot on the same bearing
    assert r.summary.startswith("not feasible: drop_zone (the arm has no carry pose")
    start = next(p for p in r.places if p.name == "jenga_block")
    assert start.ok and start.detail is None
    json.dumps(r.model_dump(mode="json"), allow_nan=False)  # the API can send it
