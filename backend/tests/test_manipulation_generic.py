"""The manipulation tools on an intake scene (real pinned model): object-colour perception, side grasps for tall
objects, placement into a named zone, and go_home. Privileged-state sequences are labelled as such."""

import numpy as np
import pytest

from cortex import intake
from cortex.agentic.astra import scene_prompt
from cortex.sim.manipulation import ManipulationSession, tool_schemas
from cortex.sim.scene import build_scene
from cortex.sim.scene_env import SceneEnv
from cortex.sim.so101 import Variation, configure_gl

configure_gl()


@pytest.fixture(scope="module")
def bottle_env(tmp_path_factory):
    spec = intake.start("pick up the small water bottle")
    name = spec.objects[0].name
    spec = intake.apply_patch(spec, {"objects": [{"name": name, "confirm": True}],
                                     "zones": [{"name": "spot_a", "center": [0.30, -0.13]}],
                                     "task": {"place": ["spot_a"]}})
    env = SceneEnv(build_scene(spec, tmp_path_factory.mktemp("scene")), episode_steps=10_000)
    yield env
    env.close()


def session(env, mode="privileged", seed=1):
    env.reset(seed)
    s = ManipulationSession(env, mode=mode)
    s.reset_environment(seed, Variation())
    return s


def ok(s, name, args=None):
    out, _ = s.call(name, args or {})
    assert out["ok"], (name, out)
    return out


def test_task_context_names_the_object_and_zones(bottle_env):
    s = session(bottle_env)
    ctx = s.task_context()
    assert ctx["object"] == "small_water_bottle" and ctx["grasp_mode"] == "side"
    assert ctx["zones"] == ["spot_a"] and ctx["target_zone"] == "spot_a"
    tools = {t["function"]["name"]: t["function"] for t in tool_schemas("pixels", ctx)}
    assert "go_home" in tools
    assert tools["propose_grasp"]["parameters"]["properties"]["object_id"]["enum"] == ["small_water_bottle"]
    assert tools["place_object"]["parameters"]["properties"]["zone"]["enum"] == ["spot_a"]
    assert "small water bottle" in tools["check_task_success"]["description"]
    scene, plan = scene_prompt(ctx)
    assert "small water bottle" in scene and "spot_a" in scene and "red cube" not in scene + plan
    assert "go_home" in plan and "slide" in plan
    assert s.call("propose_grasp", {"object_id": "blue_mug"})[0]["error"]["code"] == "unknown_object"


def test_camera_mode_segments_the_object_by_its_colour(bottle_env):
    s = session(bottle_env, mode="camera")
    ok(s, "observe_scene")
    true = bottle_env.cube_pos()
    # point at the visible top of the bottle, a few pixels off
    u, v = s.table_to_pixel(np.r_[true[:2], bottle_env.obj_h])
    g = ok(s, "propose_grasp", {"image_point": {"u": u + 4, "v": v + 3}})
    assert g["source"] == "camera_backprojection" and g["grasp_mode"] == "side"
    assert g["evidence"]["refined"] and g["evidence"]["segmentation"].startswith("object colour")
    assert np.linalg.norm(np.array(g["estimated_xyz"][:2]) - true[:2]) < 0.02, (g["estimated_xyz"], true)
    # the zone comes from the task spec in camera mode, and says so
    assert s.call("place_object", {"zone": "nowhere"})[0]["error"]["code"] == "not_holding"


def test_privileged_tool_sequence_reaches_success_on_the_small_bottle(bottle_env):
    """observe -> propose_grasp -> approach -> close -> verify -> place_object(zone) -> go_home (privileged state)."""
    s = session(bottle_env)
    ok(s, "observe_scene")
    g = ok(s, "propose_grasp")
    assert g["reachable"], g
    assert g["grasp_mode"] == "side" and len(g["approach"]) == 3 and "wrist_roll_deg" in g
    ok(s, "set_gripper", {"state": "open"})
    for k in ("transit", "pregrasp", "grasp", "slide"):
        ok(s, "move_end_effector", {"target": g[k]})
    ok(s, "set_gripper", {"state": "closed"})
    assert ok(s, "verify_grasp")["holding_object"]
    ok(s, "move_end_effector", {"target": g["lift"]})
    p = ok(s, "place_object", {"zone": "spot_a"})
    assert p["source"] == "privileged_ground_truth" and p["grasp_mode"] == "side"
    h = ok(s, "go_home")
    assert h["reached_home"], h
    for _ in range(30):  # let the bottle settle before the simulator judges
        s._step(s.cmd)
    assert ok(s, "check_task_success")["success"], bottle_env.cube_pos()
    assert len(s.rec.action) == len(s.rec.state) == len(s.rec.front) > 100
