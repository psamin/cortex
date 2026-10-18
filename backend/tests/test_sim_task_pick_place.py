"""Pick-place task (issue #309): the cube tabletop on the task registry, solved by the scripted oracle."""

import pytest

pytest.importorskip("mujoco")

from cortex.sim.so101 import configure_gl  # noqa: E402
from cortex.sim.tasks import make, run  # noqa: E402

configure_gl()


@pytest.fixture(scope="module")
def task():
    env, oracle = make("pick_place")
    yield env, oracle
    env.close()


def test_oracle_places_the_cube_in_the_box(task):
    env, oracle = task
    for seed in (0, 1, 2):  # one per cube start region
        env.reset(seed)
        assert not env.success()
        ep = run.episode(env, oracle, seed, record=False)
        assert oracle.done and ep["success"] and ep["lifted"] and ep["steps"] < env.max_steps, ep


def test_oracle_runs_under_dart_noise(task):
    env, oracle = task
    ep = run.episode(env, oracle, 0, record=False, noise_deg=2.0)
    assert oracle.done and ep["steps"] < env.max_steps, ep
