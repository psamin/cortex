"""PickPlaceEnv over a generated scene: any primitive object, any place zone.

It keeps PickPlaceEnv's interface so the manipulation tools, the scripted expert
and the workers run unchanged. ``cube_*`` means the pick object, ``box_pos`` is the
current target zone's centre on the table, and ``in_box`` is "the object rests
upright inside the target zone and nothing is holding it".

The cycle protocol swaps the target with ``set_target`` (demo: the task zone;
reset: the origin or the next zone).
"""

from __future__ import annotations

import mujoco
import numpy as np

from .scene import SceneInfo
from .so101 import LIGHT_SCALE, SETTLED_SPEED, PickPlaceEnv, Variation  # noqa: F401  (SETTLED_SPEED re-exported)

REST_TOL_Z = 0.012  # object centre within this of its resting height
UPRIGHT_COS = 0.98  # cylinder axis within ~11 degrees of vertical (a 20 degree lean fails)
FACE_DOWN_COS = 0.9  # a box rests on some face
ZONE_MARGIN = 0.005


class SceneEnv(PickPlaceEnv):
    def __init__(self, info: SceneInfo, height: int = 96, width: int = 128, episode_steps: int = 300):
        super().__init__(height=height, width=width, episode_steps=episode_steps, scene=str(info.path),
                         instruction=f"Pick up the {(info.pick or 'object').replace('_', ' ')}.")
        self.info = info
        self.pick = info.pick or next(iter(info.objects))
        self.target: tuple[tuple[float, float], tuple[float, float]] | None = (
            info.zones[info.place[0]] if info.place else None)
        self.target_name = info.place[0] if info.place else None
        self.base_target = self.target  # reset offsets apply to this, so they never accumulate
        self.obj_h = info.heights[self.pick]
        self.shape = info.shapes.get(self.pick, "box")

    # -------------------------------------------------------------- binding
    def _bind(self) -> None:
        super()._bind()
        w = self.world
        self.cube = mujoco.mj_name2id(w.model, mujoco.mjtObj.mjOBJ_BODY, self.info.objects[self.pick])
        self.cube_qadr = w._free_qadr(self.cube)
        self.parked = []  # the upstream cubes are parked by the generated scene itself
        self.obj_geoms = {g for g in range(w.model.ngeom) if w.model.geom_bodyid[g] == self.cube}
        self.others = {n: w._free_qadr(mujoco.mj_name2id(w.model, mujoco.mjtObj.mjOBJ_BODY, b))
                       for n, b in self.info.objects.items() if n != self.pick}

    def set_target(self, zone: str | None = None, xy: tuple[float, float] | None = None,
                   half: tuple[float, float] = (0.04, 0.04)) -> None:
        """Target a named zone, or an arbitrary spot (a reset back to the origin)."""
        if zone is not None:
            self.target, self.target_name = self.info.zones[zone], zone
        elif xy is not None:
            self.target, self.target_name = (xy, half), None
        else:
            raise ValueError("set_target needs a zone or an xy")
        self.base_target = self.target

    # ---------------------------------------------------------------- reset
    def reset(self, seed: int, var: Variation | None = None, start: tuple[float, float] | None = None,
              jitter: float = 0.015) -> dict:
        """Place the pick object at ``start`` (default: its nominal start) plus seeded jitter.
        ``var`` contributes lighting and an extra offset; its cube_region is ignored here."""
        var = var or Variation(lighting=self.info.lighting)
        self.env.reset(seed=seed)
        if self.world is None:
            self._bind()
        w, d = self.world, self.world.data
        rng = np.random.default_rng(seed)
        x, y = start or self.info.starts[self.pick]
        x += var.cube_offset[0] + rng.uniform(-jitter, jitter)
        y += var.cube_offset[1] + rng.uniform(-jitter, jitter)
        self.place_object(self.pick, (x, y), yaw=rng.uniform(-0.4, 0.4))
        self.target = self.base_target
        if var.box_offset != (0.0, 0.0) and self.base_target is not None:
            (cx, cy), half = self.base_target
            self.target = ((cx + var.box_offset[0], cy + var.box_offset[1]), half)
        d.qvel[:] = 0
        self._apply_light(LIGHT_SCALE[var.lighting] * var.light_scale)
        mujoco.mj_forward(w.model, d)
        self.max_cube_z = 0.0
        self.steps = 0
        return self._obs()

    def place_object(self, name: str, xy: tuple[float, float], yaw: float = 0.0) -> None:
        qa = self.cube_qadr if name == self.pick else self.others[name]
        h = self.info.heights[name]
        d = self.world.data
        d.qpos[qa:qa + 3] = (xy[0], xy[1], h / 2 + 0.0005)
        d.qpos[qa + 3:qa + 7] = (np.cos(yaw / 2), 0, 0, np.sin(yaw / 2))

    # ------------------------------------------------------------- queries
    def box_pos(self) -> np.ndarray:
        if self.target is None:
            return np.array([0.22, 0.12, 0.0])
        (cx, cy), _ = self.target
        return np.array([cx, cy, 0.0])

    def upright(self) -> bool:
        """Cylinders stand on an end; boxes rest on any face; spheres have no orientation."""
        R = self.world.data.xmat[self.cube].reshape(3, 3)
        if self.shape == "sphere":
            return True
        if self.shape == "box":
            return bool(np.max(np.abs(R[2, :])) > FACE_DOWN_COS)
        return bool(abs(R[2, 2]) > UPRIGHT_COS)

    def _rest_height(self) -> float:
        if self.shape != "box":
            return self.obj_h / 2
        # A box that landed on another face rests at half of that face's extent.
        R = self.world.data.xmat[self.cube].reshape(3, 3)
        half = self.world.model.geom_size[min(self.obj_geoms)]
        return float(half[int(np.argmax(np.abs(R[2, :])))])

    def resting(self) -> bool:
        p = self.cube_pos()
        return bool(abs(p[2] - self._rest_height()) < REST_TOL_Z)

    def in_zone(self) -> bool:
        if self.target is None:
            return False
        (cx, cy), (hx, hy) = self.target
        p = self.cube_pos()
        return bool(abs(p[0] - cx) <= hx + ZONE_MARGIN and abs(p[1] - cy) <= hy + ZONE_MARGIN)

    def in_box(self) -> bool:
        """Task success: inside the target zone, resting on the table, upright, released and still."""
        return self.in_zone() and self.resting() and self.upright() and self.settled() and not self.held()

    def reward(self) -> float:
        if self.in_box():
            return 1.0
        c, b = self.cube_pos(), self.box_pos()
        lifted = 0.25 if self.max_cube_z > self.obj_h / 2 + 0.03 else 0.0
        return lifted + 0.5 * max(0.0, 1 - np.linalg.norm(c[:2] - b[:2]) / 0.3) * (lifted > 0)
