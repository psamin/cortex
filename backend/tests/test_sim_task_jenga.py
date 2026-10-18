"""Jenga task (issue #161): the oracle pushes the red block out of the tower and the tower stands."""

import numpy as np
import pytest

from cortex.sim.so101 import configure_gl
from cortex.sim.tasks import make


@pytest.fixture(scope="module")
def jenga():
    configure_gl()
    env, oracle = make("jenga")
    yield env, oracle
    env.close()


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_oracle_solves_seed(jenga, seed):
    env, oracle = jenga
    obs = env.reset(seed)
    oracle.reset(env, obs, seed)
    for _ in range(env.max_steps):
        env.step(oracle.act(obs), render=False)
        if oracle.done:
            break
    assert oracle.done and oracle.failed is None
    m = env.metrics()
    assert env.success(), m
    assert m["target_displacement_m"] > 0.3 * 0.075 and m["max_other_displacement_m"] < 0.005, m


def test_reset_is_not_success_and_seeds_differ(jenga):
    env, _ = jenga
    env.reset(0)
    assert not env.success() and abs(env.displacement()) < 1e-3
    layout0, colour0 = env.start_pos.copy(), env.world.model.mat_rgba[env.mats[0][0]].copy()
    env.reset(1)
    assert not env.success()
    assert np.linalg.norm(env.start_pos - layout0, axis=1).min() > 0.002  # every block moved
    assert not np.allclose(env.world.model.mat_rgba[env.mats[0][0]], colour0)


def test_untouched_tower_stands(jenga):
    env, _ = jenga
    obs = env.reset(3)
    for _ in range(300):
        _, _, ok, _, _ = env.step(obs["agent_pos"], render=False)
        assert not ok
    moved, tilt = env.others()
    assert moved < 0.001 and np.rad2deg(tilt) < 0.5 and abs(env.displacement()) < 0.001
