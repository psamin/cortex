"""The generic scripted skill completes home -> pick -> place -> home on intake scenes (real pinned model)."""

import numpy as np
import pytest

from cortex import intake
from cortex.sim.pick_place_skill import PickPlaceSkill
from cortex.sim.scene import build_scene
from cortex.sim.scene_env import SceneEnv
from cortex.sim.so101 import configure_gl

configure_gl()


def run(prompt, zones, tmp_path, seed=1, max_steps=3000):
    spec = intake.start(prompt)
    name = spec.objects[0].name
    spec = intake.apply_patch(spec, {"objects": [{"name": name, "confirm": True}], "zones": zones,
                                     "task": {"place": [zones[0]["name"]]}})
    env = SceneEnv(build_scene(spec, tmp_path))
    obs = env.reset(seed)
    skill = PickPlaceSkill(env, height=spec.objects[0].height, grasp_width=spec.objects[0].grasp_width)
    skill.plan(obs["agent_pos"])
    phases = []
    for _ in range(max_steps):
        env.step(skill.act())
        if not phases or phases[-1] != skill.phase:
            phases.append(skill.phase)
        if skill.done:
            break
    for _ in range(30):  # let the object settle
        env.step(np.asarray(env.world.joint_state(), dtype=np.float32))
    home_err = float(np.abs(env.world.joint_state()[:5] - np.asarray(env.module.HOME)[:5]).max())
    out = dict(success=env.in_box(), failed=skill.failed, phases=phases, home_err=home_err)
    env.close()
    return out


def test_cube_top_grasp_home_to_home(tmp_path):
    r = run("pick up the cube", [{"name": "spot_a", "center": [0.22, 0.12]}], tmp_path)
    assert r["failed"] is None, r
    assert r["success"], r
    assert r["phases"][0] == "hover" and r["phases"][-1] == "home" and r["home_err"] < 2.0



def test_small_bottle_side_grasp_home_to_home(tmp_path):
    r = run("pick up the small water bottle", [{"name": "spot_a", "center": [0.30, -0.13]}], tmp_path)
    assert r["failed"] is None, r
    assert r["success"], r
    assert "slide" in r["phases"] and r["phases"][-1] == "home" and r["home_err"] < 2.0
