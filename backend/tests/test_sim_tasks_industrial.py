"""Industrial arm base (issue #161): the Menagerie UR10e with a suction cup or a parallel gripper."""

import argparse
import json
import sys
import types
import xml.etree.ElementTree as ET  # nosec B405

import numpy as np
import pytest

from cortex.sim import tasks
from cortex.sim.so101 import configure_gl
from cortex.sim.tasks import run
from cortex.sim.tasks.industrial import VAC_OFF, VAC_OPEN, VAC_SEALED, ArmOracle, IndustrialEnv, down, geom

configure_gl()
BOX = (0.55, 0.0, 0.03)  # a 6 cm cube on the floor in front of the arm


class _Env(IndustrialEnv):
    name = "industrial_fake"
    instruction = "Lift the box."
    max_steps = 400
    home = np.array([0.0, -100.0, 110.0, -100.0, -90.0, 0.0])  # cup / fingers pointing down

    def build(self, mj, wb):
        ET.SubElement(wb, "light", pos="0 0 3", dir="0 0 -1")
        ET.SubElement(wb, "geom", name="floor", type="plane", size="2 2 0.1")
        ET.SubElement(wb, "camera", name="front", pos="1.4 -1.0 0.8", xyaxes="0.58 0.81 0 -0.3 0.22 0.93")
        b = ET.SubElement(wb, "body", name="box0", pos=" ".join(map(str, BOX)))
        ET.SubElement(b, "freejoint")
        geom(b, "box0", "box", (0.03, 0.03, 0.03), mass="0.2", friction="1 0.01 0.001")
        self.boxes = ["box0"]

    def reset_task(self, rng):
        self.set_free("box0", BOX, (1, 0, 0, 0))

    def success(self):
        return bool(self.world.data.xpos[self.box_ids[0]][2] > 0.1)


class _Gripper(_Env):
    effector = "gripper"
    home_grip = 100.0


class _Oracle(ArmOracle):
    """Above the box, down onto it, grab (suction on or fingers closed), lift."""

    def _script(self):
        env = self.env
        grab, release = (100.0, 0.0) if env.effector == "suction" else (0.0, 100.0)
        top = np.array([BOX[0], BOX[1], 0.06 if env.effector == "suction" else 0.035])
        q = self._ik(top + [0, 0, 0.1], down(0.0), self.cmd, iters=300)
        yield self._joint(q, release)
        yield self._line(top - [0, 0, 0.004], down(0.0), release, n=30, settle=5)
        yield self._hold(grab, 15)
        yield self._line(top + [0, 0, 0.15], down(0.0), grab, n=40, settle=10)


@pytest.fixture(scope="module", params=[_Env, _Gripper], ids=["suction", "gripper"])
def env(request):
    e = request.param()
    yield e
    e.close()


def test_the_tool_picks_up_a_box(env):
    ep = run.episode(env, _Oracle(), 0, record=False)
    assert ep["success"] and env.world.data.xpos[env.box_ids[0]][2] > 0.15, ep
    assert env.joint_names == ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3", env.effector]
    assert env.robot_type == f"ur10e_{env.effector}"


def test_suction_seals_only_on_contact():
    env = _Env()
    obs = env.reset(0)
    oracle = _Oracle()
    oracle.reset(env, obs, 0)
    above = oracle._ik(np.array([BOX[0], BOX[1], 0.16]), down(0.0), oracle.cmd, iters=300)
    for _ in range(60):
        env.step(np.r_[above, 100.0].astype(np.float32), render=False)  # suction on in the air
    assert env.attached is None and env.state()[6] == VAC_OPEN
    for _ in range(5):
        env.step(np.r_[above, 0.0].astype(np.float32), render=False)
    assert env.state()[6] == VAC_OFF
    env.close()


def test_holding_and_noise_follow_the_tool():
    e = _Gripper.__new__(_Gripper)  # holding() reads only the effector
    assert e.holding(np.r_[np.zeros(6), 0.0]) and not e.holding(np.r_[np.zeros(6), 100.0])
    s = _Env.__new__(_Env)
    assert s.holding(np.r_[np.zeros(6), 100.0]) and not s.holding(np.r_[np.zeros(6), 0.0])
    assert VAC_SEALED > VAC_OPEN > VAC_OFF
    rng = np.random.default_rng(0)
    nudge = run.Nudge(rng, 3.0, arm=6, holding=e.holding)
    free = np.stack([nudge(np.r_[np.zeros(6), 100.0].astype(np.float32)) for _ in range(300)])
    held = np.stack([nudge(np.zeros(7, np.float32)) for _ in range(300)])
    assert np.all(free[:, 6] == 100) and np.all(held[:, 6] == 0)  # the tool command is never nudged
    assert np.abs(held[:, :6]).std() < 0.5 * np.abs(free[:, :6]).std()  # smaller while holding


def test_demos_record_the_arm_and_its_tool(tmp_path, monkeypatch):
    monkeypatch.setitem(tasks.TASKS, "industrial_fake", "industrial_fake")
    monkeypatch.setitem(sys.modules, "cortex.sim.tasks.industrial_fake",
                        types.SimpleNamespace(Env=_Gripper, Oracle=_Oracle))
    root = tmp_path / "ds"
    run.demos(argparse.Namespace(task="industrial_fake", root=str(root), episodes=1, size="96x128", examples=None,
                                 n_examples=0, shard=0, shards=1, noise_deg=0.0, layouts=0))
    meta = json.loads((root / "meta" / "cortex_sim.json").read_text())
    info = json.loads((root / "meta" / "info.json").read_text())
    names = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3", "gripper"]
    assert meta["kept_seeds"] == [0] and info["robot_type"] == "ur10e_gripper"
    for key in ("observation.state", "action"):
        assert info["features"][key]["shape"] == [7] and info["features"][key]["names"] == names
