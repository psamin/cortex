import numpy as np
import pytest

from dreamcache.environments.cloning import clone_env, reseed, state_fingerprint
from dreamcache.environments.minigrid_adapter import make_env

ENVS = ["MiniGrid-Dynamic-Obstacles-8x8-v0", "MiniGrid-DoorKey-8x8-v0",
        "MiniGrid-Dynamic-Obstacles-16x16-v0"]


def advance(env, n, seed=0):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        _, _, term, trunc, _ = env.step(int(rng.integers(0, 3)))
        if term or trunc:
            break


@pytest.mark.parametrize("env_id", ENVS)
def test_clone_preserves_state(env_id):
    env, _ = make_env(env_id, 3)
    advance(env, 4, seed=1)
    sim = clone_env(env)
    assert state_fingerprint(sim) == state_fingerprint(env)
    assert sim.step_count == env.unwrapped.step_count


@pytest.mark.parametrize("env_id", ENVS)
def test_identical_transitions_with_identical_rng(env_id):
    env, _ = make_env(env_id, 5)
    advance(env, 3, seed=2)
    a, b = clone_env(env), clone_env(env)
    for act in [2, 0, 2, 1, 2, 2]:
        ra = a.step(act)
        rb = b.step(act)
        assert np.array_equal(ra[0]["image"], rb[0]["image"]) and ra[1:4] == rb[1:4]
        assert state_fingerprint(a) == state_fingerprint(b)
        if ra[2] or ra[3]:
            break


@pytest.mark.parametrize("env_id", ENVS)
def test_sandbox_does_not_mutate_parent(env_id):
    env, _ = make_env(env_id, 7)
    before = state_fingerprint(env)
    sim = clone_env(env)
    reseed(sim, 123)
    for act in [2, 2, 1, 2, 0, 2]:
        _, _, term, trunc, _ = sim.step(act)
        if term or trunc:
            break
    assert state_fingerprint(env) == before


def test_moving_obstacles_are_independent_objects():
    env, _ = make_env(ENVS[0], 0)
    sim = clone_env(env)
    assert all(o1 is not o2 for o1, o2 in zip(env.unwrapped.obstacles, sim.obstacles))
    assert [tuple(o.cur_pos) for o in env.unwrapped.obstacles] == [tuple(o.cur_pos) for o in sim.obstacles]


def test_reseed_changes_obstacle_futures():
    env, _ = make_env(ENVS[0], 0)
    outs = set()
    for s in range(8):
        sim = clone_env(env)
        reseed(sim, s)
        sim.step(0)
        outs.add(tuple(tuple(o.cur_pos) for o in sim.obstacles))
    assert len(outs) > 1
