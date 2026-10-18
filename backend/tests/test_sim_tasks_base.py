"""TaskEnv base (issue #161): a task scene builds, resets per seed and steps the pinned SO-101."""

import xml.etree.ElementTree as ET  # nosec B405

import numpy as np

from cortex.sim.so101 import configure_gl
from cortex.sim.tasks.base import TaskEnv


class _Block(TaskEnv):
    name = "block"
    focus = "blk"

    def build(self, mj, wb):
        b = ET.SubElement(wb, "body", name="blk", pos="0.2 0 0.02")
        ET.SubElement(b, "freejoint")
        ET.SubElement(b, "geom", type="box", size="0.02 0.02 0.02")

    def reset_task(self, rng):
        self.set_free("blk", (0.2, rng.uniform(-0.05, 0.05), 0.02))

    def success(self):
        return self.cube_pos()[2] > 0.1


def test_task_env_resets_per_seed_and_steps():
    configure_gl()
    env = _Block()
    obs = env.reset(0)
    assert obs["pixels"]["front"].shape[2] == 3 and obs["agent_pos"].shape == (6,)
    y0 = env.cube_pos()[1]
    env.reset(1)
    assert env.cube_pos()[1] != y0
    for _ in range(10):
        _, r, ok, _, _ = env.step(np.asarray(obs["agent_pos"]), render=False)
    assert not ok and r == 0.0 and abs(env.cube_pos()[2] - 0.02) < 0.005  # rests on the table
    env.close()
