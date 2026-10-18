"""SO-101 single-cube pick-and-place built on the official LeRobot EnvHub env.

Source env: `lerobot/so101-tabletop-mujoco` (Apache-2.0), pinned to a commit and
loaded through LeRobot's own EnvHub loader. We reuse its MuJoCo scene, Menagerie
SO-101 arm, cameras, units (so101_follower degrees, gripper 0-100) and 30 fps
position control, and add on top:

  * one task cube (red); the other three cubes are parked on the floor, off camera
  * condition control: cube start region (left/center/right) and lighting scale
  * success = red cube resting inside the green box ("bowl" in the task prompt)
  * a dense shaped reward for evaluation reporting
"""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass
from functools import lru_cache
from types import ModuleType
from typing import Any

import mujoco
import numpy as np

ENVHUB_REPO = "lerobot/so101-tabletop-mujoco"
ENVHUB_REVISION = "12e504e860c369aad1a9a9eab3fdfbac658ed6dd"
ENVHUB_URI = f"{ENVHUB_REPO}@{ENVHUB_REVISION}:env.py"

FPS = 30
IMG_H, IMG_W = 96, 128
CAMERAS = {"front": "front", "wrist": "wrist_cam"}
TASK_TEXT = "Pick up the red cube and place it in the green box."

# Cube start regions in the robot base frame (x forward, y to the robot's left).
# The box sits at (0.22, 0.125); regions are spread across the reachable workspace.
REGION_XY = {
    "left": (0.25, 0.035),
    "center": (0.22, -0.04),
    "right": (0.17, -0.12),
}
REGION_JITTER = 0.015
LIGHT_SCALE = {"dim": 0.3, "normal": 1.0, "bright": 2.0}
PARKED = [(0.6, 1.0, -0.735), (0.7, 1.0, -0.735), (0.8, 1.0, -0.735)]


@lru_cache
def envhub_module() -> ModuleType:
    """Download (pinned) and import the EnvHub env module via LeRobot's loader.

    Sandboxes without LeRobot/torch installed use the same pinned download and
    import steps directly (huggingface_hub + importlib)."""
    from huggingface_hub import snapshot_download

    # The env only fetches its scene and arm assets when the scene file is missing next to env.py,
    # so a partially cached snapshot (scene present, arm missing) fails to compile. Fetch it whole.
    snapshot_download(ENVHUB_REPO, revision=ENVHUB_REVISION)
    try:
        from lerobot.envs.utils import _download_hub_file, _import_hub_module
    except ImportError:
        import importlib.util

        from huggingface_hub import hf_hub_download

        path = hf_hub_download(ENVHUB_REPO, "env.py", revision=ENVHUB_REVISION)
        spec = importlib.util.spec_from_file_location("hub_env_lerobot_so101_tabletop_mujoco", path)
        module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(module)  # type: ignore[union-attr]
        return module
    repo_id, _, local_file, _ = _download_hub_file(ENVHUB_URI, trust_remote_code=True, hub_cache_dir=None)
    return _import_hub_module(local_file, repo_id)


@dataclass
class Variation:
    cube_region: str = "center"
    lighting: str = "normal"
    cube_offset: tuple[float, float] = (0.0, 0.0)
    light_scale: float = 1.0  # multiplied with the lighting preset
    box_offset: tuple[float, float] = (0.0, 0.0)  # goal (box) shift; the env restores it on reset


class PickPlaceEnv:
    """Gymnasium-style wrapper (reset/step) around the EnvHub SO101Env."""

    def __init__(self, height: int = IMG_H, width: int = IMG_W, episode_steps: int = 300,
                 scene: str | None = None, instruction: str = TASK_TEXT):
        m = envhub_module()
        self.module = m
        cfg = {**m.DEFAULTS}
        # Our reset places the task cube itself; keep only joint noise and camera jitter.
        cfg["randomize"] = {"joints": 2.0, "cameras": cfg["randomize"].get("cameras", {})}
        cfg.update(cameras=CAMERAS, observation_height=height, observation_width=width,
                   instruction=instruction, episode_length=episode_steps)
        if scene is not None:  # a generated scene (sim/scene.py); None = the upstream tabletop
            cfg["scene"] = scene
        self.env = m.SO101Env(**cfg)
        self.episode_steps = episode_steps
        self.world = None
        self._light0: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self.max_cube_z = 0.0

    # ------------------------------------------------------------- internals
    def _bind(self) -> None:
        self.world = self.env._ensure_world()
        w = self.world
        self.cube = w.cube_bodies[0]  # red_cube
        self.cube_qadr = w._free_qadr(self.cube)
        self.parked = [w._free_qadr(b) for b in w.cube_bodies[1:]]
        self.site = mujoco.mj_name2id(w.model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
        self._light0 = (w.model.light_diffuse.copy(), w.model.vis.headlight.diffuse.copy(),
                        w.model.vis.headlight.ambient.copy())

    def _apply_light(self, scale: float) -> None:
        d, hd, ha = self._light0  # type: ignore[misc]
        mdl = self.world.model
        mdl.light_diffuse[:] = np.clip(d * scale, 0, 1)
        mdl.vis.headlight.diffuse[:] = np.clip(hd * scale, 0, 1)
        mdl.vis.headlight.ambient[:] = np.clip(ha * scale, 0, 1)

    def _obs(self) -> dict[str, Any]:
        return self.env._obs(self.world)

    # --------------------------------------------------------------- public
    def reset(self, seed: int, var: Variation) -> dict[str, Any]:
        self.env.reset(seed=seed)
        if self.world is None:
            self._bind()
        w, d = self.world, self.world.data
        rng = np.random.default_rng(seed)
        x, y = REGION_XY[var.cube_region]
        x += var.cube_offset[0] + rng.uniform(-REGION_JITTER, REGION_JITTER)
        y += var.cube_offset[1] + rng.uniform(-REGION_JITTER, REGION_JITTER)
        a = self.cube_qadr
        d.qpos[a:a + 3] = (x, y, 0.015)
        yaw = rng.uniform(-0.4, 0.4)
        d.qpos[a + 3:a + 7] = (np.cos(yaw / 2), 0, 0, np.sin(yaw / 2))
        for qa, pos in zip(self.parked, PARKED):
            d.qpos[qa:qa + 3] = pos
        if var.box_offset != (0.0, 0.0):
            w.model.body_pos[w.box_body, :2] += np.asarray(var.box_offset)
        d.qvel[:] = 0
        self._apply_light(LIGHT_SCALE[var.lighting] * var.light_scale)
        mujoco.mj_forward(w.model, d)
        self.max_cube_z = 0.0
        self.steps = 0
        return self._obs()

    def cube_pos(self) -> np.ndarray:
        return self.world.data.xpos[self.cube].copy()

    def box_pos(self) -> np.ndarray:
        return self.world.data.xpos[self.world.box_body].copy()

    def cube_yaw(self) -> float:
        q = self.world.data.xquat[self.cube]
        return float(np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2)))

    def in_box(self) -> bool:
        w = self.world
        center, rot = w.data.xpos[w.box_body], w.data.xmat[w.box_body].reshape(3, 3)
        local = (w.data.xpos[self.cube] - center) @ rot
        return bool(np.all(np.abs(local[:2]) < self.module.BOX_INNER_HALF)
                    and local[2] < self.module.IN_BOX_MAX_HEIGHT)

    def reward(self) -> float:
        """1.0 when the cube rests in the box, else shaped progress in [0, 0.75)."""
        if self.in_box():
            return 1.0
        c, b = self.cube_pos(), self.box_pos()
        lifted = 0.25 if self.max_cube_z > 0.045 else 0.0
        return lifted + 0.5 * max(0.0, 1 - np.linalg.norm(c[:2] - b[:2]) / 0.3) * (lifted > 0)

    def step(self, action: np.ndarray, render: bool = True) -> tuple[dict[str, Any] | None, float, bool, bool,
                                                                      dict[str, Any]]:
        """One control period. ``render=False`` runs the same physics without rendering the observation cameras
        (obs is None). Policies that read simulator state, such as the oracles, don't need the pixels, and the
        renders dominate the step's cost (~0.1 s each on this machine)."""
        if render:
            obs, _, _, _, _ = self.env.step(np.asarray(action, dtype=np.float32))
        else:
            obs = None
            w = self.world
            w.data.ctrl[w.actuators] = w.to_ctrl(np.asarray(action, dtype=np.float32))
            mujoco.mj_step(w.model, w.data, nstep=w.substeps)
            self.env._steps += 1  # keep the upstream env's own step counter in sync
        self.steps += 1
        self.max_cube_z = max(self.max_cube_z, float(self.cube_pos()[2]))
        success = self.in_box()
        truncated = self.steps >= self.episode_steps
        return obs, self.reward(), success, truncated, {"is_success": success}

    def render_front(self) -> np.ndarray:
        return self.env.render("front")

    def close(self) -> None:
        self.env.close()


def configure_gl() -> None:
    """Headless Linux workers need EGL/OSMesa; macOS (CGL) and Windows (WGL via glfw) work as is."""
    if platform.system() == "Linux":  # os.uname() does not exist on Windows
        os.environ.setdefault("MUJOCO_GL", "egl")
