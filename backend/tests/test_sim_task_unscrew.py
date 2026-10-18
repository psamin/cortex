"""Unscrew task (issue #161): the thread physics, strict success, per-seed variation and the oracle."""

import numpy as np
import pytest

from cortex.sim.so101 import configure_gl
from cortex.sim.tasks import make
from cortex.sim.tasks.unscrew import PITCH, TURNS
from cortex.sim.tasks.workcell import GRIP_OPEN

configure_gl()


@pytest.fixture(scope="module")
def task():
    env, oracle = make("unscrew")
    yield env, oracle
    env.close()


def _step(env, action, n, torque=0.0, pull=0.0):
    d = env.world.data
    for _ in range(n):
        d.xfrc_applied[env.bolt, 2] = pull  # N, up
        d.xfrc_applied[env.bolt, 5] = torque  # N m, counter-clockwise from above
        env.step(np.asarray(action, dtype=np.float32), render=False)
    d.xfrc_applied[env.bolt] = 0


def test_oracle_unscrews_and_pulls_the_bolt_out(task):
    env, oracle = task
    for seed in (0, 1, 2):
        obs = env.reset(seed)
        oracle.reset(env, obs, seed)
        for _ in range(env.max_steps):
            env.step(oracle.act(obs), render=False)
            if oracle.done:
                break
        assert oracle.done and 2 <= oracle.grasps <= 4, (seed, oracle.grasps)
        action = oracle.act(obs)
        _step(env, action, 30)
        m = env.metrics()
        assert env.success() and m["turns"] >= TURNS and m["tip_above_fixture_mm"] >= 10, (seed, m)
    # Let go: off the thread the bolt drops back into its hole, which is not a success.
    action[5] = GRIP_OPEN
    _step(env, action, 40)
    assert env.disengaged() and env.tip_z() < env.fixture_top() and not env.success()


def test_reset_is_not_a_success_and_seeds_differ(task):
    env, _ = task
    env.reset(0)
    assert not env.success() and not env.disengaged() and env.turned() == 0
    a = (env.bolt_xy(), env.head_yaw(), env.half_af)
    env.reset(1)
    b = (env.bolt_xy(), env.head_yaw(), env.half_af)
    assert np.linalg.norm(a[0] - b[0]) > 1e-3 and abs(a[1] - b[1]) > 1e-3 and a[2] != b[2]


def test_untouched_bolt_stays_put(task):
    env, _ = task
    obs = env.reset(3)
    z0 = env.world.data.xpos[env.bolt][2]
    _step(env, obs["agent_pos"], 300)
    assert abs(env.turned()) < np.deg2rad(0.5) and abs(env.world.data.xpos[env.bolt][2] - z0) < 1e-4


def test_turning_raises_it_by_the_pitch(task):
    env, _ = task
    obs = env.reset(4)
    z0 = env.world.data.xpos[env.bolt][2]
    _step(env, obs["agent_pos"], 90, torque=0.15)
    turned, risen = env.turned(), env.world.data.xpos[env.bolt][2] - z0
    assert turned > np.pi / 2 and not env.disengaged()
    assert abs(risen - PITCH * turned / (2 * np.pi)) < 1e-4, (turned, risen)


def test_half_unscrewed_is_not_a_success_and_cannot_be_pulled_out(task):
    env, _ = task
    obs = env.reset(5)
    for _ in range(400):
        _step(env, obs["agent_pos"], 1, torque=0.15)
        if env.turned() >= np.pi * TURNS:  # half the engagement
            break
    _step(env, obs["agent_pos"], 20)
    assert not env.disengaged() and abs(env.rise() - PITCH * TURNS / 2) < 1e-3
    _step(env, obs["agent_pos"], 60, pull=25.0)  # a straight pull does not strip the thread
    assert env.rise() < PITCH * TURNS / 2 + 1e-3 and env.tip_z() < env.fixture_top()
    assert not env.success()
