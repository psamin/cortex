"""Conveyor task (issue #255): the belt drives parts at its speed, a missed part falls off the end, and
the privileged-state oracle intercepts the moving part and bins it. Render-free: the oracle needs no pixels."""

import numpy as np
import pytest

from cortex.sim.so101 import FPS, configure_gl
from cortex.sim.tasks import make
from cortex.sim.tasks.conveyor import BELT, S0

configure_gl()


@pytest.fixture(scope="module")
def task():
    env, oracle = make("conveyor")
    yield env, oracle
    env.close()


def _hold(env, obs, n):
    home = np.asarray(obs["agent_pos"], dtype=np.float32)
    for _ in range(n):
        env.step(home, render=False)


def test_belt_moves_the_part_at_its_speed_after_the_delay(task):
    env, _ = task
    obs = env.reset(0)
    env.speed, env.start_step = 0.05, 15
    _hold(env, obs, 15)
    assert abs(env.along() - S0) < 1e-3  # waits for the belt to start
    s0, lat0 = BELT.coords(env.part_pos())
    _hold(env, obs, 60)  # 2 s
    s1, lat1 = BELT.coords(env.part_pos())
    assert (s1 - s0) == pytest.approx(0.05 * 60 / FPS, rel=0.02)
    assert abs(lat1 - lat0) < 1e-3 and env.on_belt() and not env.success()


def test_a_missed_part_rides_off_the_end_and_fails(task):
    env, _ = task
    obs = env.reset(1)
    env.speed, env.start_step = 0.08, 0
    _hold(env, obs, int((BELT.length - S0) / 0.08 * FPS) + 45)
    assert env.fell_off() and not env.on_belt() and not env.success()
    assert env.metrics()["fell_off"] is True


def test_seeds_vary_the_line(task):
    env, _ = task
    env.reset(2)
    a = (env.speed, env.start_step, env.lateral, tuple(env.half))
    env.reset(3)
    b = (env.speed, env.start_step, env.lateral, tuple(env.half))
    assert a != b and 0.03 <= a[0] <= 0.08 and 0.03 <= b[0] <= 0.08


@pytest.mark.parametrize("seed", range(4))
def test_oracle_intercepts_the_moving_part_and_bins_it(task, seed):
    env, oracle = task
    obs = env.reset(seed)
    oracle.reset(env, obs, seed)
    held = 0
    for _ in range(env.max_steps):
        env.step(oracle.act(obs), render=False)
        held = held + 1 if oracle.done else 0
        if held >= 30:
            break
    m = env.metrics()
    assert oracle.done and env.success() and not m["fell_off"], (seed, m)
    assert 0 < oracle.intercept < BELT.length  # caught on the belt, not after it
