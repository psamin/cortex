"""#225: the arm grasps the cube where it is (real pinned EnvHub model, headless).

Three bugs made demo collection miss the cube:
  * the wrist roll that aligns the jaws with the cube was set after the IK solve and never re-solved with the
    other joints, so the gripper landed up to 15 mm off while the reported IK error said 0.3 mm;
  * that roll used the wrong sign of the pan joint, so the jaws closed 20-36 degrees off the cube's faces;
  * the descent to the grasp was one joint move with each joint clipped on its own, so the fingertips dipped
    to the table short of the cube and swept it away before the jaws closed.
"""

import mujoco
import numpy as np
import pytest

from cortex.sim.expert import closing_direction, solve_ik
from cortex.sim.manipulation import ManipulationSession
from cortex.sim.policies import ExpertPolicy
from cortex.sim.so101 import PickPlaceEnv, Variation
from cortex.workers.sim_worker import run_episode


@pytest.fixture(scope="module")
def env():
    e = PickPlaceEnv(episode_steps=10_000)
    yield e
    e.close()


def fk(env, q):
    w = env.world
    d = mujoco.MjData(w.model)
    d.qpos[:] = w.data.qpos
    d.qpos[w.qadr[:5]] = np.deg2rad(q[:5])
    mujoco.mj_kinematics(w.model, d)
    return d.site_xpos[env.site].copy(), closing_direction(w.model, d)


def face_error_deg(closing: np.ndarray, yaw: float) -> float:
    """Angle between the jaws' closing direction and the nearest cube face normal."""
    a = np.rad2deg(np.arctan2(closing[1], closing[0]) - yaw)
    return float(abs((a + 45) % 90 - 45))


@pytest.mark.parametrize("region", ["left", "center", "right"])
def test_top_grasp_ik_lands_on_the_cube_with_the_jaws_across_its_faces(env, region):
    home = np.asarray(env.module.HOME, dtype=np.float64)  # a far seed, like the agent tools' observe pose
    for seed in range(4):
        env.reset(30_000 + seed, Variation(cube_region=region))
        c, yaw = env.cube_pos(), env.cube_yaw()
        for z in (0.012, 0.055):  # grasp and pregrasp heights
            target = np.r_[c[:2], z]
            q, err = solve_ik(env, target, home, yaw)
            site, closing = fk(env, q)
            miss = float(np.linalg.norm(site - target))
            assert miss < 0.002, (region, seed, z, miss)
            assert abs(miss - err) < 5e-4, "the reported IK error must be the physical one, roll included"
            assert face_error_deg(closing, yaw) < 3.0, (region, seed, np.rad2deg(yaw), q)
            assert abs(closing[2]) < 0.1  # jaws close horizontally


def test_oracle_never_flips_the_wrist_between_waypoints(env):
    """Where the face-aligned roll is near 45 degrees (the right region), choosing the roll nearest zero flipped the
    wrist 90 degrees between neighbouring line waypoints and the demo ran out of steps."""
    for i in range(12):
        obs = env.reset(10_040 + i, Variation(cube_region="right" if i % 2 else "center", cube_offset=(0.0, -0.012)))
        ex = ExpertPolicy()
        ex.reset(env, obs, 10_040 + i)
        at_cube = ("hover", "approach", "grasp", "close", "lift")
        rolls = np.array([q[4] for (q, _), ph in zip(ex.ex.waypoints, ex.ex.phases) if ph in at_cube])
        assert np.abs(np.diff(rolls)).max() < 10.0, rolls.round(1)


def pushed_before_close(env, policy, seed, var, max_steps=240):
    """Run one oracle episode; return (result, how far the cube moved before the gripper started closing)."""
    env.reset(seed, var)  # run_episode resets again with the same seed: the same start
    start = env.cube_pos().copy()
    moved = {}
    act = policy.act

    def watch(o):
        a = act(o)
        if "mm" not in moved and getattr(policy.ex, "phase", "") == "close":
            moved["mm"] = float(np.linalg.norm(env.cube_pos()[:2] - start[:2]) * 1000)
        return a
    policy.act = watch
    r = run_episode(env, policy, seed, var, max_steps, None, False)
    return r, moved.get("mm", np.inf)


def test_oracle_descent_leaves_the_cube_in_place_on_the_seed_that_failed_in_the_studio(env):
    # seed 10026, centre: the user's record_demonstrations run lost this demo (grasp_failed); the descent shoved
    # the cube 34 mm before the jaws closed
    r, mm = pushed_before_close(env, ExpertPolicy(), 10026, Variation(cube_region="center", lighting="bright"))
    assert r["success"], r
    assert mm < 3.0, mm


@pytest.mark.parametrize("region", ["left", "center", "right"])
def test_oracle_records_every_region(env, region):
    for i, offset in enumerate([(0.0, 0.0), (0.012, -0.012), (-0.012, 0.01), (0.008, 0.0)]):
        r, mm = pushed_before_close(env, ExpertPolicy(), 10_020 + i, Variation(cube_region=region, cube_offset=offset))
        assert r["success"] and r["steps"] < 240, (region, offset, r)
        assert mm < 3.0, (region, offset, mm)


def agent_grasp(env, mode, seed, region):
    """The agent's tool sequence for a top grasp, with a perfect pointer in camera mode."""
    s = ManipulationSession(env, mode=mode)
    s.reset_environment(seed, Variation(cube_region=region))
    start = env.cube_pos().copy()
    assert s.call("observe_scene", {"camera": "overhead"})[0]["ok"]
    args = {}
    if mode == "camera":
        u, v = s.table_to_pixel(start)
        args = {"image_point": {"u": u, "v": v}}
    g, _ = s.call("propose_grasp", args)
    assert g["ok"] and g["reachable"], g
    assert s.call("set_gripper", {"state": "open"})[0]["ok"]
    for k in ("pregrasp", "grasp"):
        out, _ = s.call("move_end_effector", {"target": g[k]})
        assert out["ok"], (k, out)
        assert out["error_m"] < 0.004, (k, out)
    pushed = float(np.linalg.norm(env.cube_pos()[:2] - start[:2]))
    site = env.world.data.site_xpos[env.site]
    return s, pushed, float(np.linalg.norm(site[:2] - env.cube_pos()[:2]))


@pytest.mark.parametrize("mode,region", [("privileged", "left"), ("privileged", "center"), ("privileged", "right"),
                                         ("camera", "center")])
def test_agent_tools_grasp_the_cube_where_it_is(env, mode, region):
    # seed 20001: the centre cube was shoved 12 mm by the old descent, and the pregrasp landed 7 mm off
    s, pushed, site_err = agent_grasp(env, mode, 20_001, region)
    assert pushed < 0.003, pushed
    assert site_err < 0.005, site_err
    assert s.call("set_gripper", {"state": "closed"})[0]["ok"]
    assert s.call("verify_grasp", {})[0]["holding_object"]
    p = s.ee_pos()
    assert s.call("move_end_effector", {"target": {"x": float(p[0]), "y": float(p[1]), "z": 0.075}})[0]["ok"]
    place = {} if mode == "privileged" else {"image_point": dict(zip("uv", s.table_to_pixel(env.box_pos())))}
    if mode == "camera":
        s.call("observe_scene", {"camera": "overhead"})
    out, _ = s.call("place_object", place)
    assert out["ok"], out
    for _ in range(20):
        s._step(s.cmd)
    assert env.in_box()


@pytest.mark.parametrize("seed", range(4))
def test_camera_estimate_keeps_the_cube_apart_from_the_green_box(env, seed):
    """A cube beside the box: the box (and the turquoise arm) are saturated too, so a saturation mask merged them
    into one blob and the estimate landed 19-25 mm toward the box. Only saturated red pixels count now."""
    s = ManipulationSession(env, mode="camera")
    s.reset_environment(seed, Variation(cube_region="left", cube_offset=(-0.03, 0.025)))
    assert s.call("observe_scene", {"camera": "overhead"})[0]["ok"]
    true = env.cube_pos()
    u, v = s.table_to_pixel(true)
    g, _ = s.call("propose_grasp", {"image_point": {"u": u, "v": v}})
    assert g["ok"] and g["evidence"]["refined"], g
    assert np.linalg.norm(np.array(g["estimated_xyz"][:2]) - true[:2]) < 0.005, (g["estimated_xyz"], true)
