"""Suture task (issue #161): a rigid needle passed through two eyelets; its oracle solves the seeds."""

import numpy as np
import pytest

from cortex.sim.so101 import configure_gl
from cortex.sim.tasks import make


@pytest.fixture(scope="module")
def task():
    configure_gl()
    env, oracle = make("suture")
    yield env, oracle
    env.close()


def run_oracle(env, oracle, seed: int) -> tuple[bool, int]:
    obs = env.reset(seed)
    oracle.reset(env, obs, seed)
    while not oracle.done and env.steps < env.max_steps:
        action = oracle.act(obs)
        env.step(action, render=False)
    assert oracle.done and oracle.ex.failed is None, oracle.ex.failed
    steps = env.steps
    for _ in range(10):  # the rollout runner also holds a few steps after the policy finishes
        _, _, success, _, _ = env.step(action, render=False)
    return success, steps


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_oracle_threads_both_eyelets(task, seed):
    env, oracle = task
    success, steps = run_oracle(env, oracle, seed)
    assert success, env.metrics()
    assert steps <= env.max_steps
    assert env.metrics()["eyelets_threaded"] == 2


def test_reset_is_not_success(task):
    env, _ = task
    for seed in range(5):
        env.reset(seed)
        assert not env.success() and not env.threaded()


def test_seeds_vary_the_scene(task):
    env, _ = task
    m = env.world.model

    def snapshot(seed):
        env.reset(seed)
        return np.r_[env.cube_pos(), env.eyelets()[0].ravel(), m.mat_rgba[env.mat["skin"]],
                     m.mat_rgba[env.mat["steel"]], m.cam_quat[env.front_cam]]

    a, b = snapshot(0), snapshot(1)
    assert np.abs(a[:3] - b[:3]).max() > 1e-3  # needle start
    assert np.abs(a[3:9] - b[3:9]).max() > 1e-3  # eyelets (pad pose, slide along the incision)
    assert not np.allclose(a[9:], b[9:])  # skin tone, needle tint, camera
    assert np.allclose(snapshot(0), a)  # and a seed is reproducible


def test_untouched_scene_stays_still(task):
    env, _ = task
    obs = env.reset(3)
    p0 = env.needle_points()
    for _ in range(300):
        env.step(obs["agent_pos"], render=False)
    assert np.abs(env.needle_points() - p0).max() < 1e-3
    assert env.still() and not env.success()
