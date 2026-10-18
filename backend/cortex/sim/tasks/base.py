"""Hand-built SO-101 task scenes for sim-only policies (issue #161).

A task subclasses ``TaskEnv``: ``build`` adds its bodies to the pinned EnvHub tabletop XML,
``reset_task`` randomizes them per seed, and ``success`` is the verifier. It keeps PickPlaceEnv's
interface (same robot, cameras, units: so101_follower degrees, gripper 0-100, 30 fps position
control), so recording, evaluation and the viewer run unchanged.

Its ``Oracle`` reads privileged sim state and must solve every seed; demonstrations are only its
successful episodes. Oracle protocol: ``reset(env, obs, seed)``, ``act(obs) -> (6,) float32``,
``done`` (True once the motion is finished), ``needs_pixels = False``.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import xml.etree.ElementTree as ET  # nosec B405 - our own pinned file
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from ..scene import FLOOR_Z, PARK_Z
from ..so101 import IMG_H, IMG_W, PickPlaceEnv, Variation, envhub_module

LIGHT_RANGE = (0.7, 1.3)  # per-seed lighting scale


def base_scene() -> tuple[ET.Element, ET.Element]:
    """The EnvHub tabletop with its cubes and box parked out of sight; returns (mujoco root, worldbody)."""
    m = envhub_module()
    root = Path(m.repo_root())
    mj = ET.parse(root / m.SCENE_FILE).getroot()  # nosec B314
    ET.SubElement(mj, "compiler", assetdir=root.as_posix())
    mj.insert(0, mj[-1])
    wb = mj.find("worldbody")
    assert wb is not None
    for b in wb.findall("body"):
        name = b.get("name", "")
        if name in m.CUBE_BODIES:
            b.set("pos", f"{0.6 + 0.08 * m.CUBE_BODIES.index(name):.3f} 1.0 {FLOOR_Z + 0.015:.3f}")
        elif name == m.BOX_BODY:
            b.set("pos", f"0.9 1.2 {PARK_Z}")
    return mj, wb


def rgba(c: tuple[float, ...]) -> str:
    return " ".join(f"{v:.3f}" for v in c)


class TaskEnv(PickPlaceEnv):
    name = "task"
    instruction = ""
    max_steps = 600
    focus = ""  # body whose position the reward/viewer tracks (``cube_pos``)

    def __init__(self, height: int = IMG_H, width: int = IMG_W):
        mj, wb = base_scene()
        self.build(mj, wb)
        xml = ET.tostring(mj, encoding="unicode")
        out = Path(tempfile.gettempdir()) / "cortex_tasks"
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{self.name}_{hashlib.sha256(xml.encode()).hexdigest()[:12]}.xml"
        if not path.exists():  # parallel recorders share this file: write it whole, then move it into place
            tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
            tmp.write_text(xml)
            os.replace(tmp, path)
        super().__init__(height=height, width=width, episode_steps=self.max_steps, scene=str(path),
                         instruction=self.instruction)

    # ------------------------------------------------------------ per task
    def build(self, mj: ET.Element, wb: ET.Element) -> None:
        raise NotImplementedError

    def bind_task(self) -> None:
        """Look up the task's body ids / joint addresses once the model exists."""

    def reset_task(self, rng: np.random.Generator) -> None:
        raise NotImplementedError

    def success(self) -> bool:
        raise NotImplementedError

    def metrics(self) -> dict[str, Any]:
        """Extra per-episode numbers for evaluation reports."""
        return {}

    # -------------------------------------------------------------- helpers
    def body(self, name: str) -> int:
        return mujoco.mj_name2id(self.world.model, mujoco.mjtObj.mjOBJ_BODY, name)

    def free_qadr(self, name: str) -> int:
        return self.world._free_qadr(self.body(name))

    def set_free(self, name: str, pos: tuple[float, float, float], yaw: float = 0.0) -> None:
        a = self.free_qadr(name)
        self.world.data.qpos[a:a + 3] = pos
        self.world.data.qpos[a + 3:a + 7] = (np.cos(yaw / 2), 0, 0, np.sin(yaw / 2))

    # ------------------------------------------------------ PickPlaceEnv API
    def _bind(self) -> None:
        super()._bind()
        self.parked = []  # base_scene parks the upstream cubes itself
        self.bind_task()

    def reset(self, seed: int, var: Variation | None = None) -> dict[str, Any]:
        self.env.reset(seed=seed)
        if self.world is None:
            self._bind()
        rng = np.random.default_rng(seed)
        self.reset_task(rng)
        d = self.world.data
        d.qvel[:] = 0
        scale = var.light_scale if var is not None else rng.uniform(*LIGHT_RANGE)
        self._apply_light(scale)
        mujoco.mj_forward(self.world.model, d)
        self.max_cube_z = 0.0
        self.steps = 0
        return self._obs()

    def cube_pos(self) -> np.ndarray:
        return self.world.data.xpos[self.body(self.focus)].copy()

    def in_box(self) -> bool:
        return self.success()

    def reward(self) -> float:
        return float(self.success())
