"""Scene generation, SceneEnv success semantics and feasibility, on the real pinned EnvHub model."""

import numpy as np
import pytest

from cortex import intake
from cortex.sim.feasibility import check, jaw_opening
from cortex.sim.grasp import grasp_mode
from cortex.sim.scene import build_scene
from cortex.sim.scene_env import SceneEnv
from cortex.sim.so101 import configure_gl

configure_gl()
ZONES = [{"name": "spot_a", "center": [0.22, 0.12]}, {"name": "spot_b", "center": [0.30, -0.13]}]


def make(prompt, tmp_path, **patch):
    spec = intake.start(prompt)
    spec = intake.apply_patch(spec, {"zones": ZONES, "task": {"place": ["spot_a", "spot_b"]}, **patch})
    return spec, build_scene(spec, tmp_path)


@pytest.fixture
def cube_env(tmp_path):
    spec, info = make("pick up the cube", tmp_path)
    env = SceneEnv(info)
    env.reset(3)
    yield spec, info, env
    env.close()


def test_scene_has_objects_zones_obstacles_and_parks_upstream_bodies(tmp_path):
    spec, info = make("pick up the small water bottle", tmp_path, environment={
        "obstacles": [{"name": "shelf", "shape": "box", "size": [0.04, 0.1, 0.1], "center": [0.38, 0.05]}]})
    xml = info.path.read_text()
    for name in ("obj_small_water_bottle", "zone_spot_a", "zone_spot_b", "obs_shelf", "assetdir="):
        assert name in xml
    assert info.pick == "small_water_bottle" and grasp_mode(info.heights[info.pick]) == "side"
    assert np.hypot(*info.starts[info.pick]) >= 0.295  # side grasps need a far start
    env = SceneEnv(info)
    env.reset(0)
    w = env.world
    for b in w.cube_bodies + [w.box_body]:
        assert w.data.xpos[b][2] < -0.5  # upstream cubes and box are out of sight
    env.close()


def test_success_needs_zone_rest_and_upright(cube_env):
    _, _, env = cube_env
    assert not env.in_box()  # starts outside every zone
    env.place_object(env.pick, (0.22, 0.12))
    for _ in range(20):
        env.step(np.asarray(env.module.HOME, dtype=np.float32))
    assert env.in_zone() and env.resting() and env.upright() and env.in_box() and env.reward() == 1.0
    env.set_target("spot_b")
    assert not env.in_box()
    env.set_target(xy=(0.22, 0.12))
    assert env.in_box()


def test_reset_is_seeded_and_uses_start(cube_env):
    _, _, env = cube_env
    env.reset(7)
    a = env.cube_pos()
    env.reset(7)
    assert np.allclose(a, env.cube_pos())
    env.reset(7, start=(0.25, 0.0), jitter=0.0)
    assert np.allclose(env.cube_pos()[:2], (0.25, 0.0), atol=1e-6)


def test_jaw_opening_is_measured_from_the_model(cube_env):
    _, _, env = cube_env
    assert 0.05 < jaw_opening(env, 60) < 0.07
    assert jaw_opening(env, 0) < 0.01


def test_feasibility_accepts_cube_and_refuses_full_bottle(tmp_path):
    spec, info = make("pick up the cube", tmp_path / "a")
    r = check(spec, info)
    assert r.objects[0].ok, r.objects[0]
    spec, info = make("pick up the water bottle", tmp_path / "b")
    r = check(spec, info)
    o = r.objects[0]
    assert not r.ok and not o.grasp.ok and o.tall and o.grasp_mode == "side"
    assert o.suggestion == "try a small water bottle"


def test_side_grasp_closes_jaws_horizontally(cube_env):
    import mujoco

    from cortex.sim.expert import approach_error_deg, closing_direction, side_approach, solve_ik
    from cortex.sim.grasp import ik_seeds

    _, _, env = cube_env
    xyz = np.array([0.31, 0.0, 0.084])
    axis = side_approach(xyz[:2])
    best = min((solve_ik(env, xyz, s, approach=axis, iters=150) for s in ik_seeds(xyz, np.asarray(env.module.HOME))),
               key=lambda r: r[1])
    q, err = best
    assert err < 0.004 and approach_error_deg(env, q, axis) < 5
    w = env.world
    d = mujoco.MjData(w.model)
    d.qpos[:] = w.data.qpos
    d.qpos[w.qadr[:5]] = np.deg2rad(q[:5])
    mujoco.mj_kinematics(w.model, d)
    assert abs(closing_direction(w.model, d)[2]) < 0.1  # jaws close sideways around a standing bottle


def test_held_or_moving_object_is_not_success(cube_env):
    _, _, env = cube_env
    env.place_object(env.pick, (0.22, 0.12))
    env.world.data.qvel[env.world.model.jnt_dofadr[env.world.model.body_jntadr[env.cube]]] = 0.5
    assert not env.settled() and not env.in_box()


def test_box_offset_does_not_accumulate(cube_env):
    from cortex.sim.so101 import Variation

    _, _, env = cube_env
    for _ in range(3):
        env.reset(1, Variation(box_offset=(0.02, 0.0)))
    assert np.allclose(env.box_pos()[:2], (0.24, 0.12))


def test_small_table_clips_default_workspace():
    spec = intake.apply_patch(intake.start(), {"environment": {"table_size": [0.4, 0.4]}})
    assert np.allclose(spec.workspace.y, (-0.18, 0.18))
    with pytest.raises(intake.IntakeError, match="off the table"):
        intake.apply_patch(spec, {"workspace": {"y": [-0.3, 0.3]}})
