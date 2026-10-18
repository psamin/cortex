"""#152 finding 20: on the built-in cube task a held cube is not success (real pinned EnvHub model)."""

import numpy as np
import pytest

from cortex.sim.expert import GRIP_OPEN
from cortex.sim.policies import ExpertPolicy
from cortex.sim.so101 import PickPlaceEnv, Variation
from cortex.workers.sim_worker import run_episode

SEED = 3


@pytest.fixture(scope="module")
def env():
    e = PickPlaceEnv(episode_steps=10_000)
    yield e
    e.close()


class NeverReleases:
    """The scripted expert's motion, but once the gripper closes it never opens again: it carries the cube
    over the box and keeps holding it. No ``ex`` attribute, so the rollout treats it like a learned policy."""

    name = "never_releases (test)"
    needs_pixels = False

    def reset(self, env, obs, seed):
        self.inner = ExpertPolicy()
        self.inner.reset(env, obs, seed)
        self.closed_at: float | None = None

    def act(self, obs):
        a = np.array(self.inner.act(obs), dtype=np.float32)
        if a[5] < GRIP_OPEN / 2 or self.closed_at is not None:
            self.closed_at = a[5] if self.closed_at is None else min(self.closed_at, a[5])
            a[5] = self.closed_at
        return a


def test_holding_the_cube_over_the_box_is_not_success(env):
    r = run_episode(env, NeverReleases(), SEED, Variation(), 300, None, False)
    assert r["max_cube_z"] > 0.045, "the cube should have been lifted (the test needs a real grasp)"
    assert not r["success"]
    assert env.held()


def test_scripted_expert_still_succeeds(env):
    r = run_episode(env, ExpertPolicy(), SEED, Variation(), 400, None, False)
    assert r["success"]
    assert not env.held() and env.settled()
