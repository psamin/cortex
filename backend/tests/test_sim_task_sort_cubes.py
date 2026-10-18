"""Sort cubes by color (issue #334): the oracle sorts every cube; a cube in the wrong bin is not success."""

import numpy as np
import pytest

pytest.importorskip("mujoco")

from cortex.sim.so101 import configure_gl  # noqa: E402
from cortex.sim.tasks import make, run  # noqa: E402
from cortex.sim.tasks.sort_cubes import BIN_FLOOR, COLORS, CUBE_HALF  # noqa: E402

configure_gl()


@pytest.fixture(scope="module")
def task():
    env, oracle = make("sort_cubes")
    yield env, oracle
    env.close()


def _drop_into_bins(env, bins: dict[str, str]) -> None:
    """Put each cube (key) on the floor of a bin (value) and let it settle, the arm holding its pose."""
    obs = env.reset(0)
    for cube, b in bins.items():
        x, y, _ = env.bin_xyz(b)
        env.set_free(f"{cube}_cube", (x, y, BIN_FLOOR + CUBE_HALF + 0.002))
    hold = np.asarray(obs["agent_pos"], np.float32)
    for _ in range(20):
        env.step(hold, render=False)


def test_oracle_sorts_every_cube(task):
    env, oracle = task
    for seed in (0, 1):
        env.reset(seed)
        assert not env.success() and env.metrics()["sorted"] == 0
        ep = run.episode(env, oracle, seed, record=False)
        assert oracle.done and ep["success"] and ep["sorted"] == 3 and ep["steps"] < env.max_steps, ep


def test_a_cube_in_the_wrong_bin_is_not_success(task):
    env, _ = task
    _drop_into_bins(env, {c: c for c in COLORS})
    assert env.success() and env.metrics()["sorted"] == 3
    _drop_into_bins(env, {"red": "green", "green": "red", "blue": "blue"})
    m = env.metrics()
    assert not env.success() and m["sorted"] == 1 and m["in_wrong_bin"] == 2


def test_oracle_runs_under_dart_noise(task):
    env, oracle = task
    ep = run.episode(env, oracle, 0, record=False, noise_deg=1.0)
    assert oracle.done and ep["steps"] < env.max_steps, ep
