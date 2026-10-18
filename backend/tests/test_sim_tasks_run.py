"""Sim task runner (issue #161): oracle demos become a LeRobotDataset; failed episodes are dropped."""

import argparse
import json
import sys
import types
import xml.etree.ElementTree as ET  # nosec B405

import numpy as np

from cortex.sim import tasks
from cortex.sim.so101 import configure_gl
from cortex.sim.tasks import run
from cortex.sim.tasks.base import TaskEnv


class _Env(TaskEnv):
    name = "fake"
    focus = "blk"
    instruction = "Hold still."
    max_steps = 60

    def build(self, mj, wb):
        b = ET.SubElement(wb, "body", name="blk", pos="0.2 0 0.02")
        ET.SubElement(b, "freejoint")
        ET.SubElement(b, "geom", type="box", size="0.02 0.02 0.02")

    def reset_task(self, rng):
        self.set_free("blk", (0.2, 0.0, 0.02))

    def success(self):
        return self.steps >= 5 and not self.fail


class _Oracle:
    needs_pixels = False

    def reset(self, env, obs, seed):
        env.fail = seed == 1  # seed 1 is a failed demo
        self.q, self.n = np.asarray(obs["agent_pos"], np.float32), 0

    @property
    def done(self):
        return self.n >= 3

    def act(self, obs):
        self.n += 1
        return self.q


def test_demos_keep_only_successful_oracle_episodes(tmp_path, monkeypatch):
    configure_gl()
    monkeypatch.setitem(tasks.TASKS, "fake", "fake")
    monkeypatch.setitem(sys.modules, "cortex.sim.tasks.fake", types.SimpleNamespace(Env=_Env, Oracle=_Oracle))
    root = tmp_path / "ds"
    run.demos(argparse.Namespace(task="fake", root=str(root), episodes=2, size="96x128", examples=None, n_examples=0,
                                 shard=0, shards=1))
    meta = json.loads((root / "meta" / "cortex_sim.json").read_text())
    assert meta["kept_seeds"] == [0, 2] and meta["failed_seeds"] == [1]
    info = json.loads((root / "meta" / "info.json").read_text())
    assert info["total_episodes"] == 2 and info["features"]["observation.images.wrist"]["shape"] == [96, 128, 3]


def test_parallel_shards_merge_into_one_dataset(tmp_path, monkeypatch):
    configure_gl()
    monkeypatch.setitem(tasks.TASKS, "fake", "fake")
    monkeypatch.setitem(sys.modules, "cortex.sim.tasks.fake", types.SimpleNamespace(Env=_Env, Oracle=_Oracle))
    roots = []
    for i in range(2):  # seeds 0, 2 (shard 0) and 1, 3 (shard 1); seed 1 fails
        roots.append(tmp_path / f"shard{i}")
        run.demos(argparse.Namespace(task="fake", root=str(roots[-1]), episodes=4, size="96x128", examples=None,
                                     n_examples=0, shard=i, shards=2))
    out = tmp_path / "merged"
    run.merge(argparse.Namespace(task="fake", root=str(out), shards_roots=[str(r) for r in roots]))
    meta = json.loads((out / "meta" / "cortex_sim.json").read_text())
    assert meta["kept_seeds"] == [0, 2, 3, 5] and meta["failed_seeds"] == [1] and meta["shards"] == 2
    assert json.loads((out / "meta" / "info.json").read_text())["total_episodes"] == 4
