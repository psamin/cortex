"""Conveyors in intake scenes (issue #255): spec validation, the scene's bed, the belt drive in SceneEnv,
the skill's intercept of a moving object, and the reachable intercept window in feasibility."""

import xml.etree.ElementTree as ET  # nosec B405

import numpy as np
import pytest
from pydantic import ValidationError

from cortex import intake
from cortex.intake.service import IntakeError
from cortex.intake.spec import Conveyor, IntakeSpec
from cortex.sim import feasibility
from cortex.sim.pick_place_skill import PickPlaceSkill
from cortex.sim.scene import SceneInfo, build_scene
from cortex.sim.scene_env import SceneEnv
from cortex.sim.so101 import FPS, configure_gl

configure_gl()

LINE = {"name": "line", "start": [0.20, 0.30], "end": [0.20, -0.16], "width": 0.08, "height": 0.03,
        "speed_mps": 0.05}
CUBE = {"name": "part", "shape": "box", "size": [0.025, 0.025, 0.025], "mass_kg": 0.02, "on": "line"}
BIN = {"name": "bin", "center": [0.10, -0.19], "half_size": [0.035, 0.035]}


def _spec(**line) -> IntakeSpec:
    spec = intake.start("")
    return intake.apply_patch(spec, {"environment": {"conveyors": [{**LINE, **line}]}, "objects": [CUBE],
                                     "zones": [BIN], "task": {"pick": "part", "place": ["bin"]}})


# ------------------------------------------------------------------ spec


def test_conveyor_spec_validates_units_and_references():
    spec = _spec()
    c = spec.environment.conveyors[0]
    assert c.speed_mps == 0.05 and c.height == 0.03 and spec.objects[0].on == "line"
    with pytest.raises(ValidationError):
        Conveyor(**{**LINE, "speed_mps": -0.1})  # m/s, positive
    with pytest.raises(ValidationError):
        Conveyor(**{**LINE, "speed_mps": 5.0})  # 5 m/s: someone sent cm/s or mm/s
    with pytest.raises(ValidationError, match="long"):
        Conveyor(**{**LINE, "end": [0.20, 0.27]})  # 3 cm long
    with pytest.raises(ValidationError):
        Conveyor(**{**LINE, "width": 8.0})  # metres, not cm
    with pytest.raises(IntakeError, match="not a conveyor"):
        intake.apply_patch(spec, {"objects": [{"name": "part", "on": "belt_b"}]})
    with pytest.raises(IntakeError, match="not on conveyor"):
        intake.apply_patch(spec, {"objects": [{"name": "part", "start": [0.30, 0.0]}]})
    with pytest.raises(IntakeError, match="arm base"):
        intake.apply_patch(spec, {"environment": {"conveyors": [{"name": "line", "start": [0.0, 0.3],
                                                                 "end": [0.0, -0.3]}]}})
    # Keyed by name: a patch changes one field of the belt.
    faster = intake.apply_patch(spec, {"environment": {"conveyors": [{"name": "line", "speed_mps": 0.08}]}})
    assert faster.environment.conveyors[0].speed_mps == 0.08 and faster.environment.conveyors[0].width == 0.08


# ----------------------------------------------------------------- scene


def test_scene_has_the_bed_and_the_object_rests_on_it(tmp_path):
    info = build_scene(_spec(), tmp_path)
    wb = ET.parse(info.path).getroot().find("worldbody")  # nosec B314
    assert wb.find("body[@name='belt_line']/geom[@name='belt_line_bed']") is not None
    part = wb.find("body[@name='obj_part']")
    x, y, z = (float(v) for v in part.get("pos").split())
    assert info.on == {"part": "line"} and info.belts()["line"].over(np.array([x, y]))
    assert z == pytest.approx(0.03 + 0.0125, abs=1e-3)
    assert info.conveyors["line"]["speed_mps"] == 0.05
    again = SceneInfo.from_dict(info.to_dict(tmp_path), tmp_path)
    assert again.belts()["line"].speed == 0.05 and again.on == info.on


@pytest.fixture(scope="module")
def line_env(tmp_path_factory):
    spec = _spec()
    env = SceneEnv(build_scene(spec, tmp_path_factory.mktemp("line")))
    yield spec, env
    env.close()


def test_scene_env_drives_the_belt(line_env):
    _, env = line_env
    obs = env.reset(0, jitter=0.0)
    belt = env.belt_of()
    s0, lat0 = belt.coords(env.cube_pos())
    hold = np.asarray(obs["agent_pos"], dtype=np.float32)
    for _ in range(45):
        env.step(hold, render=False)
    s1, lat1 = belt.coords(env.cube_pos())
    assert (s1 - s0) == pytest.approx(0.05 * 45 / FPS, rel=0.03) and abs(lat1 - lat0) < 1e-3


def test_skill_intercepts_the_moving_object_and_places_it(line_env):
    spec, env = line_env
    o = spec.objects[0]
    for seed in (0, 1):
        obs = env.reset(seed)
        skill = PickPlaceSkill(env, height=o.height, grasp_width=o.grasp_width)
        skill.plan(obs["agent_pos"])
        phases = []
        for _ in range(1500):
            env.step(skill.act(), render=False)
            if not phases or phases[-1] != skill.phase:
                phases.append(skill.phase)
            if skill.done:
                break
        for _ in range(30):
            env.step(np.asarray(env.world.joint_state(), dtype=np.float32), render=False)
        assert skill.failed is None and env.in_box(), (seed, phases, skill.failed, env.cube_pos())
        assert phases[:4] == ["hover", "lower", "wait", "close"] and skill.intercept is not None


def test_skill_intercepts_on_a_fast_belt(tmp_path):
    spec = _spec(speed_mps=0.08)  # 8 cm/s: a late close grabs the trailing edge and the object slips out
    env = SceneEnv(build_scene(spec, tmp_path))
    try:
        obs = env.reset(0)
        skill = PickPlaceSkill(env, height=spec.objects[0].height, grasp_width=spec.objects[0].grasp_width)
        skill.plan(obs["agent_pos"])
        for _ in range(1500):
            env.step(skill.act(), render=False)
            if skill.done:
                break
        for _ in range(30):
            env.step(np.asarray(env.world.joint_state(), dtype=np.float32), render=False)
        assert skill.failed is None and env.in_box(), env.cube_pos()
    finally:
        env.close()


def test_feasibility_reports_the_intercept_window(line_env):
    spec, env = line_env
    rep = feasibility.check(spec, env.info, env)
    (c,) = rep.conveyors
    assert c.ok and c.object == "part" and c.window_m is not None, c
    lo, hi = c.window_m
    assert 0.0 < lo < hi < 0.46 - 0.05 and c.time_in_window_s == pytest.approx((hi - lo) / 0.05, abs=0.05)
    assert not any(p.kind == "start" and p.name == "part" for p in rep.places)  # the upstream start is fine
