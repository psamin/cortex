"""Screwdriver task (issue #161): tool pick-up, the thread physics, strict success and the oracle."""

import numpy as np
import pytest

from cortex.sim.so101 import configure_gl
from cortex.sim.tasks import make
from cortex.sim.tasks.screwdriver import PITCH, TURNS
from cortex.sim.tasks.workcell import GRIP_OPEN

configure_gl()


@pytest.fixture(scope="module")
def task():
    env, oracle = make("screwdriver")
    yield env, oracle
    env.close()


def _step(env, action, n, torque=0.0):
    d = env.world.data
    for _ in range(n):
        d.xfrc_applied[env.screw, 5] = torque  # N m, counter-clockwise from above
        env.step(np.asarray(action, dtype=np.float32), render=False)
    d.xfrc_applied[env.screw] = 0


def test_oracle_unscrews_with_the_screwdriver(task):
    env, oracle = task
    for seed in (0, 1, 2):
        obs = env.reset(seed)
        oracle.reset(env, obs, seed)
        for _ in range(env.max_steps):
            env.step(oracle.act(obs), render=False)
            if oracle.done:
                break
        assert oracle.done and 2 <= oracle.engagements <= 4, (seed, oracle.engagements)
        action = oracle.act(obs)
        _step(env, action, 30)
        m = env.metrics()
        assert env.success() and m["turns"] >= TURNS and m["tip_above_head_mm"] >= 10, (seed, m)
        assert m["rise_mm"] < 1000 * PITCH * TURNS + 2  # the loose screw stayed in its hole
    # Dropping the screwdriver is not a success.
    action[5] = GRIP_OPEN
    _step(env, action, 40)
    assert env.disengaged() and not env.driver_held() and not env.success()


def test_reset_is_not_a_success_and_seeds_differ(task):
    env, _ = task
    env.reset(0)
    assert not env.success() and not env.disengaged() and env.turned() == 0
    a = (env.screw_xy(), env.recess_yaw(), env.head_half, env.world.data.xpos[env.driver].copy(), env.driver_yaw())
    env.reset(1)
    b = (env.screw_xy(), env.recess_yaw(), env.head_half, env.world.data.xpos[env.driver].copy(), env.driver_yaw())
    assert np.linalg.norm(a[0] - b[0]) > 1e-3 and np.linalg.norm(a[3] - b[3]) > 1e-3
    assert abs(a[1] - b[1]) > 1e-3 and abs(a[4] - b[4]) > 1e-3 and a[2] != b[2]


def test_untouched_scene_stays_put(task):
    env, _ = task
    obs = env.reset(3)
    d = env.world.data
    s0, d0 = d.xpos[env.screw].copy(), d.xpos[env.driver].copy()
    _step(env, obs["agent_pos"], 300)
    assert abs(env.turned()) < np.deg2rad(0.5) and np.linalg.norm(d.xpos[env.screw] - s0) < 1e-4
    assert np.linalg.norm(d.xpos[env.driver] - d0) < 5e-4 and d.xmat[env.driver][8] > np.cos(np.deg2rad(0.5))


def test_turning_raises_it_by_the_pitch_then_it_sits_loose(task):
    env, _ = task
    obs = env.reset(4)
    z0 = env.world.data.xpos[env.screw][2]
    _step(env, obs["agent_pos"], 60, torque=0.015)
    turned, risen = env.turned(), env.world.data.xpos[env.screw][2] - z0
    assert turned > np.pi / 2 and not env.disengaged()
    assert abs(risen - PITCH * turned / (2 * np.pi)) < 1e-4, (turned, risen)
    for _ in range(600):
        _step(env, obs["agent_pos"], 1, torque=0.015)
        if env.disengaged():
            break
    _step(env, obs["agent_pos"], 60)
    assert env.disengaged() and abs(env.rise() - PITCH * TURNS) < 5e-4  # off the thread, it stays up


def test_half_unscrewed_is_not_a_success(task):
    env, _ = task
    obs = env.reset(5)
    for _ in range(600):
        _step(env, obs["agent_pos"], 1, torque=0.015)
        if env.turned() >= np.pi * TURNS:
            break
    _step(env, obs["agent_pos"], 20)
    assert not env.disengaged() and abs(env.rise() - PITCH * TURNS / 2) < 5e-4
    assert not env.success()
