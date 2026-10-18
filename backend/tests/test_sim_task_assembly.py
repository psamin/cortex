"""Factory assembly task (issue #161): scene, per-seed variation, strict success and the scripted oracle."""

import numpy as np
import pytest

from cortex.sim.so101 import configure_gl
from cortex.sim.tasks import make
from cortex.sim.tasks.assembly import FIX_TOP, PARTS, PLATE_H, SEAT_DEPTH

configure_gl()


@pytest.fixture(scope="module")
def task():
    env, oracle = make("assembly")
    yield env, oracle
    env.close()


def _hold(env, obs, n):
    for _ in range(n):
        env.step(np.asarray(obs["agent_pos"]), render=False)


def test_oracle_inserts_both_pegs(task):
    env, oracle = task
    for seed in (0, 1, 2):
        obs = env.reset(seed)
        oracle.reset(env, obs, seed)
        held = None
        for _ in range(env.max_steps):
            env.step(oracle.act(obs), render=False)
            if held is None and oracle.phase == "release round":  # in the hole, still gripped
                held = (env.part_bottom("round")[2] < env.fixture_top() - SEAT_DEPTH,
                        env.touching_robot("round"), env.success())
            if oracle.done:
                break
        assert oracle.done, seed  # within max_steps
        assert held == (True, True, False), (seed, held)  # deep enough, but held: not a success
        _hold(env, {"agent_pos": oracle.act(obs)}, 20)
        assert env.success(), (seed, env.metrics())


def test_reset_is_not_a_success_and_seeds_differ(task):
    env, _ = task
    env.reset(0)
    assert not env.success() and not any(env.seated(p) for p in PARTS)
    a = (env.cube_pos(), env.hole_xy("square"), dict(env.half_w), dict(env.finish))
    env.reset(1)
    b = (env.cube_pos(), env.hole_xy("square"), dict(env.half_w), dict(env.finish))
    assert np.linalg.norm(a[0] - b[0]) > 1e-3 and np.linalg.norm(a[1] - b[1]) > 1e-3
    assert a[2] != b[2]


def test_untouched_scene_stays_still(task):
    env, _ = task
    obs = env.reset(3)
    p0 = {p: env.world.data.xpos[env.part_body[p]].copy() for p in PARTS}
    _hold(env, obs, 300)
    for p in PARTS:
        assert np.linalg.norm(env.world.data.xpos[env.part_body[p]] - p0[p]) < 1e-3, p
        assert env.part_axis(p)[2] > np.cos(np.deg2rad(1.0)), p
    assert not env.success()


def test_dropped_next_to_the_hole_or_in_the_wrong_hole_is_not_seated(task):
    env, _ = task
    obs = env.reset(4)
    # Square peg placed in its hole (the checker accepts it); round peg dropped on the fixture top
    # between the two holes.
    sq = env.hole_xy("square")
    env.set_free("part_square", (sq[0], sq[1], PLATE_H + env.half_h["square"] + 0.002), yaw=env.fixture_yaw())
    beside = env.world.data.xpos[env.fixture][:2]  # fixture centre, 3.5 cm from each hole
    env.set_free("part_round", (beside[0], beside[1], FIX_TOP + env.half_h["round"] + 0.005))
    _hold(env, obs, 60)
    assert env.seated("square")
    assert not env.seated("round") and not env.success()
    # The round peg fits the square hole too, but that is not its hole.
    env.set_free("part_round", (sq[0], sq[1], PLATE_H + env.half_h["round"] + 0.002))
    env.set_free("part_square", (0.30, 0.0, env.half_h["square"] + 0.002))  # on the mat, clear of all
    env.world.data.qvel[:] = 0
    _hold(env, obs, 60)
    assert env.part_bottom("round")[2] < env.fixture_top() - SEAT_DEPTH
    assert not env.seated("round") and not env.seated("square") and not env.success()
