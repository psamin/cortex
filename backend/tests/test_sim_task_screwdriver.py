"""Screwdriver task (issue #161): tool pick-up, the thread physics (with breakaway), strict success and
the oracle (a firm break-loose turn, then flicks)."""

import numpy as np
import pytest

from cortex.sim.so101 import configure_gl
from cortex.sim.tasks import make
from cortex.sim.tasks.screwdriver import BREAKAWAY, PITCH, TURNS
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


def test_oracle_breaks_it_loose_then_flicks_it_out(task):
    env, oracle = task
    for seed in (0, 1, 2):
        obs = env.reset(seed)
        oracle.reset(env, obs, seed)
        turned = {"break": [0.0, 0], "flick": [0.0, 0]}
        for _ in range(env.max_steps):
            action = oracle.act(obs)
            before = env.turned()
            env.step(action, render=False)
            key = "break" if oracle.phase == "break loose" else "flick" if oracle.phase.startswith("flick") else None
            if key:
                turned[key][0] += env.turned() - before
                turned[key][1] += 1
            if oracle.done:
                break
        # The recorder runs until done: the whole motion is over and success already holds.
        assert oracle.done and env.success(), (seed, env.metrics())
        _step(env, action, 30)
        m = env.metrics()
        assert env.success() and m["turns"] >= TURNS and m["tip_above_head_mm"] >= 10, (seed, m)
        assert m["rise_mm"] < 1000 * PITCH * TURNS + 2  # the loose screw stayed in its hole
        slow, fast = (turned[k][0] / turned[k][1] for k in ("break", "flick"))
        assert fast > 3 * slow > 0, (seed, slow, fast)  # a firm turn, then quick flicks
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
    _step(env, obs["agent_pos"], 60, torque=0.03)
    turned, risen = env.turned(), env.world.data.xpos[env.screw][2] - z0
    assert 0.1 < turned < BREAKAWAY and not env.disengaged()
    assert abs(risen - PITCH * turned / (2 * np.pi)) < 1e-4, (turned, risen)
    for _ in range(1200):
        _step(env, obs["agent_pos"], 1, torque=0.03)
        if env.disengaged():
            break
    _step(env, obs["agent_pos"], 60)
    assert env.disengaged() and abs(env.rise() - PITCH * TURNS) < 5e-4  # off the thread, it stays up


def test_it_turns_much_faster_once_broken_loose(task):
    env, _ = task
    obs = env.reset(6)
    _step(env, obs["agent_pos"], 40, torque=0.03)
    tight = env.turned() / 40  # rad per step on the seat
    for _ in range(1200):
        _step(env, obs["agent_pos"], 1, torque=0.03)
        if env.turned() > BREAKAWAY + 0.3:
            break
    t0 = env.turned()
    _step(env, obs["agent_pos"], 5, torque=0.03)
    loose = (env.turned() - t0) / 5
    assert env.metrics()["broke_loose"] and loose > 5 * tight > 0, (tight, loose)


def test_half_unscrewed_is_not_a_success(task):
    env, _ = task
    obs = env.reset(5)
    for _ in range(1200):
        _step(env, obs["agent_pos"], 1, torque=0.03)
        if env.turned() >= np.pi * TURNS:
            break
    _step(env, obs["agent_pos"], 30)
    assert not env.disengaged() and env.turned() < 2 * np.pi * TURNS
    assert abs(env.rise() - PITCH * env.turned() / (2 * np.pi)) < 2e-4
    assert not env.success()
