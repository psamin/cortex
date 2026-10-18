"""Jenga (issue #161): push the red block out through the back of a seven-layer tower without toppling it.

The tower is 21 wooden blocks of real Jenga proportions (7.5 x 2.5 x 1.5 cm), three per layer, layers
alternating direction, standing in front of the SO-101. The middle block of layer 2 or 4 (0 = bottom)
is painted red: the target. Its layer runs along the line from the arm, so the arm can push it along
its long axis with the nose of the fixed jaw (moving jaw swung wide open, out of the way).

What the gripper allows, measured on the Menagerie SO-101 collision model: the fixed jaw is a wedge,
0.8 cm thick at the nose and 1.5 cm thick 3 cm back. A 1.5 cm slot therefore takes about 3 cm of jaw,
so the push stops at 2.8 cm, a third of the block. Success is the target pushed out by at least 0.3 of
its length (it sticks out of the far face) with every other block within 1 cm of its start and none
tilted by 10 degrees. Pulling it clear from the far side is out of the arm's reach.

Physics: the target is a loose block, 0.3 mm thinner than its neighbours, as a player would pick. In
MuJoCo a block carrying its full share of the layers above drags them along by friction. Stiff
contacts (solref 5 ms) keep the tower still on its own; they need a 2 ms physics step, so this task
runs the arm with 17 substeps per 30 fps control step instead of 7, with the arm's own options.

Per-seed variation: which layer, the tower's distance from the arm (3 cm range) and yaw (+-10 deg; it
faces the arm to within 2 deg, so it also moves sideways by up to 5 cm), every block's length and
width (+-0.15 / 0.05 mm) and placement (+-0.5 mm, 0.25 deg), layer heights (+-0.1 mm), wood tone and
grain scale per block, the red paint, table tint, lighting and the front camera's pose.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET  # nosec B405
from collections.abc import Iterator
from typing import Any

import mujoco
import numpy as np

from ..so101 import FPS
from .base import TaskEnv, rgba

BLOCK = np.array([0.075, 0.025, 0.015])  # full length, width, height (m)
N_LAYERS = 7
SPACING = BLOCK[1] + 0.0012  # centre to centre within a layer: real towers have small gaps
TARGET_LAYERS = (2, 4)  # layers that run along the line from the arm; the target is their middle block
LOOSE = 0.0003  # the target is this much thinner than its layer
PAN_AXIS = np.array([0.0388, 0.0])  # shoulder_pan axis on the table (Menagerie SO-101)
NEAR_RANGE = (0.265, 0.295)  # pan axis -> the target's near end face (m)
YAW_RANGE = np.deg2rad(10.0)
REL_YAW = np.deg2rad(2.0)  # tower yaw relative to the line from the arm
SIZE_JITTER = np.array([0.0003, 0.0001])  # +- full length, width of each block
HEIGHT_JITTER = 0.0001  # +- per layer: blocks of a layer share a height, so none rocks
PLACE_JITTER = (0.0005, 0.0002, np.deg2rad(0.25))  # along the block, across it, yaw
SUBSTEPS = 17  # physics steps per control step (1.96 ms)
WOOD = [(0.86, 0.68, 0.46), (0.80, 0.60, 0.38), (0.90, 0.75, 0.55), (0.74, 0.55, 0.36)]  # beech to oak
PAINT = (0.80, 0.10, 0.08)
SUCCESS_DISP = 0.3 * BLOCK[0]
INTACT_TOL = 0.01
TILT_TOL = np.deg2rad(10.0)


def block_name(layer: int, i: int) -> str:
    return f"jenga_{layer}_{i}"


def look_at(eye: np.ndarray, target: np.ndarray) -> str:
    """MJCF ``xyaxes`` for a camera at `eye` looking at `target`, horizon level."""
    f = (target - eye) / np.linalg.norm(target - eye)
    r = np.cross(f, [0.0, 0.0, 1.0])
    r /= np.linalg.norm(r)
    return " ".join(f"{v:.4f}" for v in (*r, *np.cross(r, f)))


def small_rotation(rng: np.random.Generator, max_angle: float) -> np.ndarray:
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    a = rng.uniform(0, max_angle)
    return np.r_[np.cos(a / 2), np.sin(a / 2) * axis]


class Env(TaskEnv):
    name = "jenga"
    instruction = "Push the red block out of the Jenga tower without knocking the tower over."
    max_steps = 900
    view = {"lookat": (0.28, 0.03, 0.08), "distance": 0.75, "azimuth": 112, "elevation": -20}  # run.Presenter

    def build(self, mj: ET.Element, wb: ET.Element) -> None:
        asset, default = mj.find("asset"), mj.find("default")
        assert asset is not None and default is not None
        # The upstream wood grain as a 2D texture: grain lines run along the blocks, scaled per block.
        ET.SubElement(asset, "texture", name="jenga_grain", type="2d", file="textures/wood.png")
        for b in range(3 * N_LAYERS):
            ET.SubElement(asset, "material", name=f"jenga_wood_{b}", texture="jenga_grain", rgba=rgba((*WOOD[0], 1)),
                          specular="0.15", shininess="0.25", texuniform="true", texrepeat="9 9")
        cls = ET.SubElement(default, "default", {"class": "jenga"})
        ET.SubElement(cls, "geom", type="box", size=" ".join(f"{s / 2:.4f}" for s in BLOCK), density="550",
                      condim="4", friction="0.45 0.005 0.0001", solref="0.005 1")
        for k in range(N_LAYERS):
            for i in range(3):
                body = ET.SubElement(wb, "body", name=block_name(k, i), pos=f"{0.6 + 0.1 * i} {0.1 * k} 0.1")
                ET.SubElement(body, "freejoint")
                ET.SubElement(body, "geom", {"class": "jenga"}, name=f"{block_name(k, i)}_geom",
                              material=f"jenga_wood_{3 * k + i}")
        for cam in wb.findall("camera"):
            if cam.get("name") == "front":  # three-quarter view of the arm and the tower's far face
                eye = np.array([0.56, -0.29, 0.25])
                cam.set("pos", " ".join(f"{v:.4f}" for v in eye))
                cam.set("xyaxes", look_at(eye, np.array([0.23, 0.0, 0.07])))
                cam.set("fovy", "45")

    def bind_task(self) -> None:
        m = self.world.model
        self.world.substeps = SUBSTEPS
        m.opt.timestep = 1 / (FPS * SUBSTEPS)
        self.blocks = [[self.body(block_name(k, i)) for i in range(3)] for k in range(N_LAYERS)]
        self.geoms = [[m.body_geomadr[b] for b in row] for row in self.blocks]
        self.mats = [[m.geom_matid[g] for g in row] for row in self.geoms]
        self.front = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "front")
        self.front_quat0 = m.cam_quat[self.front].copy()
        self.table_mat = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, "table")
        self.table_rgba0 = m.mat_rgba[self.table_mat].copy()

    # ------------------------------------------------------------------ reset
    def reset_task(self, rng: np.random.Generator) -> None:
        m, d = self.world.model, self.world.data
        self.layer, self.index = int(rng.choice(TARGET_LAYERS)), 1
        yaw = rng.uniform(-YAW_RANGE, YAW_RANGE)
        bearing = yaw + rng.uniform(-REL_YAW, REL_YAW)
        u, n = np.array([np.cos(yaw), np.sin(yaw)]), np.array([-np.sin(yaw), np.cos(yaw)])
        sizes = np.tile(BLOCK, (N_LAYERS, 3, 1))
        sizes[..., :2] += rng.uniform(-1, 1, (N_LAYERS, 3, 2)) * SIZE_JITTER
        sizes[..., 2] += rng.uniform(-HEIGHT_JITTER, HEIGHT_JITTER, (N_LAYERS, 1))
        sizes[self.layer, self.index, 2] -= LOOSE
        # The target's near end lies on the line from the pan axis, so the jaw can push straight along it.
        near = PAN_AXIS + rng.uniform(*NEAR_RANGE) * np.array([np.cos(bearing), np.sin(bearing)])
        centre = near + sizes[self.layer, self.index, 0] / 2 * u
        z = 0.0
        for k in range(N_LAYERS):
            along, across = (u, n) if k % 2 == 0 else (n, -u)
            for i in range(3):
                g = self.geoms[k][i]
                m.geom_size[g] = sizes[k, i] / 2
                m.geom_rbound[g] = np.linalg.norm(sizes[k, i]) / 2
                da, dc, dyaw = (rng.uniform(-j, j) for j in PLACE_JITTER)
                if (k, i) == (self.layer, self.index):
                    da = 0.0
                xy = centre + ((i - 1) * SPACING + dc) * across + da * along
                self.set_free(block_name(k, i), (xy[0], xy[1], z + sizes[k, i, 2] / 2),
                              yaw + (0.0 if k % 2 == 0 else np.pi / 2) + dyaw)
            z += sizes[k, :, 2].max()
        self._paint(rng)
        mujoco.mju_mulQuat(m.cam_quat[self.front], self.front_quat0, small_rotation(rng, np.deg2rad(1.5)))
        tint = rng.uniform(0.88, 1.08) * rng.uniform(0.97, 1.03, 3)
        m.mat_rgba[self.table_mat, :3] = np.clip(self.table_rgba0[:3] * tint, 0, 1)
        # Let the fresh stack take its load (hundredths of a millimetre) before the episode starts.
        d.qvel[:] = 0
        mujoco.mj_step(m, d, nstep=120)
        d.qvel[:] = 0
        ids = np.ravel(self.blocks)
        self.start_pos = d.xpos[ids].copy()
        self.start_mat = d.xmat[ids].reshape(-1, 3, 3).copy()

    def _paint(self, rng: np.random.Generator) -> None:
        m = self.world.model
        base = np.asarray(WOOD[rng.integers(len(WOOD))])
        for k in range(N_LAYERS):
            for i in range(3):
                mat = self.mats[k][i]
                m.mat_texrepeat[mat] = rng.uniform(6.0, 12.0, 2)
                if (k, i) == (self.layer, self.index):
                    color = np.asarray(PAINT) * rng.uniform(0.9, 1.1, 3)
                else:
                    color = base * rng.uniform(0.9, 1.08) * rng.uniform(0.96, 1.04, 3)
                m.mat_rgba[mat] = (*np.clip(color, 0, 1), 1.0)

    # ---------------------------------------------------------------- verifier
    def target(self) -> int:
        return self.blocks[self.layer][self.index]

    def cube_pos(self) -> np.ndarray:
        return self.world.data.xpos[self.target()].copy()

    def target_axis(self) -> np.ndarray:
        """Unit long axis of the target at the start of the episode (pointing away from the arm)."""
        return self.start_mat[self.layer * 3 + self.index][:, 0].copy()

    def target_length(self) -> float:
        return float(2 * self.world.model.geom_size[self.geoms[self.layer][self.index], 0])

    def displacement(self) -> float:
        """How far the target has moved along its own long axis (m)."""
        k = self.layer * 3 + self.index
        return float((self.world.data.xpos[self.target()] - self.start_pos[k]) @ self.target_axis())

    def others(self) -> tuple[float, float]:
        """Largest displacement (m) and tilt (rad) of any block other than the target."""
        d = self.world.data
        ids = np.ravel(self.blocks)
        keep = np.arange(len(ids)) != self.layer * 3 + self.index
        moved = np.linalg.norm(d.xpos[ids] - self.start_pos, axis=1)[keep]
        up = d.xmat[ids].reshape(-1, 3, 3)[:, :, 2]
        tilt = np.arccos(np.clip(np.einsum("ij,ij->i", up, self.start_mat[:, :, 2]), -1, 1))[keep]
        return float(moved.max()), float(tilt.max())

    def intact(self) -> bool:
        moved, tilt = self.others()
        return moved < INTACT_TOL and tilt < TILT_TOL

    def success(self) -> bool:
        return self.displacement() > SUCCESS_DISP and self.intact()

    def metrics(self) -> dict[str, Any]:
        moved, tilt = self.others()
        return {"target_displacement_m": self.displacement(), "max_other_displacement_m": moved,
                "max_other_tilt_deg": float(np.rad2deg(tilt)), "target_layer": self.layer}


# ------------------------------------------------------------------------- oracle
PITCH = np.deg2rad(14.6)  # jaw tilted down so the fixed jaw's wedge is level in the slot
TIP = np.array([0.0066, 0.0, -0.0225])  # nose of the fixed jaw, gripperframe site frame (x approach, z jaws)
ELBOW_MAX = np.deg2rad(86.0)  # the folded elbow stalls near 91 degrees: keep the IK clear of it
GRIP_PUSH = 95.0  # moving jaw swung wide open, up and back out of the slot
PUSH_DEPTH = 0.028  # nose past the target's starting end face (m)
STANDOFF = 0.025
PUSH_SPEED = 0.012  # m/s


def min_jerk(n: int) -> np.ndarray:
    """n samples in (0, 1] of a minimum-jerk profile: smooth start and stop."""
    t = np.arange(1, n + 1) / n
    return 10 * t**3 - 15 * t**4 + 6 * t**5


class Oracle:
    """Scripted push reading privileged state: lines the fixed jaw's nose up with the red block's end
    face, pushes it 2.8 cm straight along its long axis at about 1.2 cm/s, backs out and returns home.
    Every move follows a minimum-jerk profile; straight-line moves are solved by IK at each control
    step, seeded from the previous one. Pace and pauses vary a little per seed."""

    needs_pixels = False

    def __init__(self) -> None:
        self.done = False
        self.phase = ""
        self.failed: str | None = None

    def reset(self, env: Env, obs: dict[str, Any], seed: int) -> None:
        self.env = env
        w = env.world
        self.m, self.d = w.model, mujoco.MjData(w.model)
        self.qadr = w.qadr[:5]
        self.dofs = np.array([self.m.joint(n).dofadr[0] for n in env.module.JOINTS[:5]])
        self.lo, self.hi = w.lo[:5].copy(), w.hi[:5].copy()
        self.lo[2], self.hi[2] = -ELBOW_MAX, ELBOW_MAX
        self.gripper_body = self.m.site_bodyid[env.site]
        self.home = np.asarray(env.module.HOME, dtype=np.float64)
        self.cmd = np.asarray(obs["agent_pos"], dtype=np.float64).copy()
        self.done, self.failed, self.phase = False, None, ""
        self.rng = np.random.default_rng(seed + 10_000)  # a little human variation between demonstrations
        self.steps = self._plan()

    # ------------------------------------------------------------------ IK
    def _axes(self, point: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Approach (site x) and jaw-closing (site z) axes at `point`: in the arm's vertical plane
        through the pan axis, pitched down by PITCH, jaws closing vertically."""
        b = np.arctan2(point[1] - PAN_AXIS[1], point[0] - PAN_AXIS[0])
        h = np.array([np.cos(b), np.sin(b), 0.0])
        return np.cos(PITCH) * h + [0, 0, -np.sin(PITCH)], np.sin(PITCH) * h + [0, 0, np.cos(PITCH)]

    def _ik(self, point: np.ndarray, seed: np.ndarray, iters: int = 150) -> tuple[np.ndarray, float]:
        """Damped least squares: joints (degrees) putting the jaw nose at `point` in the push
        orientation. Returns (q, nose position error in m)."""
        m, d, site = self.m, self.d, self.env.site
        a, c = self._axes(point)
        d.qpos[:] = self.env.world.data.qpos
        d.qpos[self.qadr] = np.clip(np.deg2rad(seed[:5]), self.lo, self.hi)
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        last = np.inf
        for _ in range(iters):
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            R = d.site_xmat[site].reshape(3, 3)
            p = d.site_xpos[site] + R @ TIP
            e = np.r_[point - p, 0.075 * (np.cross(R[:, 0], a) + np.cross(R[:, 2], c))]
            err = float(np.linalg.norm(e))
            if err < 5e-5 or (err < 1e-3 and last - err < 1e-7):  # converged, or stalled at the closest pose
                break
            last = err
            mujoco.mj_jac(m, d, jp, jr, p, self.gripper_body)
            J = np.vstack([jp[:, self.dofs], 0.15 * jr[:, self.dofs]])
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-5 * np.eye(6), e)
            d.qpos[self.qadr] = np.clip(d.qpos[self.qadr] + np.clip(dq, -0.2, 0.2), self.lo, self.hi)
        mujoco.mj_kinematics(m, d)
        R = d.site_xmat[site].reshape(3, 3)
        return np.rad2deg(d.qpos[self.qadr]), float(np.linalg.norm(point - d.site_xpos[site] - R @ TIP))

    def _solve(self, point: np.ndarray, seed: np.ndarray) -> np.ndarray:
        q, e = self._ik(point, seed)
        if e > 1e-3:
            pan = np.rad2deg(np.arctan2(point[1] - PAN_AXIS[1], point[0] - PAN_AXIS[0]))
            for s in ([pan, 0, 0, 0, 0], [pan, 30, 0, 0, 0], [pan, 10, 30, -40, 0]):
                q2, e2 = self._ik(point, np.asarray(s, dtype=np.float64))
                if e2 < e:
                    q, e = q2, e2
        if e > 1e-3:
            raise RuntimeError(f"no IK for the jaw at {np.round(point, 3).tolist()} ({e * 1000:.1f} mm off)")
        return q

    # ---------------------------------------------------------------- moves
    def _joint_move(self, q: np.ndarray, grip: float, speed: float = 50.0) -> Iterator[np.ndarray]:
        """Minimum-jerk joint-space move; `speed` is the mean rate of the joint moving most (deg/s)."""
        start, goal = self.cmd.copy(), np.r_[q[:5], grip]
        n = max(8, int(np.abs(goal[:5] - start[:5]).max() / speed * FPS))
        for s in min_jerk(n):
            yield start + (goal - start) * s

    def _line(self, a: np.ndarray, b: np.ndarray, speed: float) -> Iterator[np.ndarray]:
        """Straight line of the jaw nose from a to b at a mean `speed` (m/s), minimum-jerk in time."""
        q = self.cmd[:5]
        for s in min_jerk(max(6, int(np.linalg.norm(b - a) / speed * FPS))):
            q = self._solve(a + (b - a) * s, q)
            yield np.r_[q, GRIP_PUSH]

    def _hold(self, n: int) -> Iterator[np.ndarray]:
        for _ in range(n):
            yield self.cmd.copy()

    def _plan(self) -> Iterator[tuple[str, np.ndarray]]:
        env = self.env
        u = env.target_axis()
        u = np.array([u[0], u[1], 0.0]) / np.linalg.norm(u[:2])
        face = env.start_pos[env.layer * 3 + env.index] - u * env.target_length() / 2  # near end face centre
        pre = face - u * STANDOFF
        above = pre + [0, 0, self.rng.uniform(0.025, 0.04)]
        pace = self.rng.uniform(0.8, 1.2)
        plan = [
            ("reach", lambda: self._joint_move(self._solve(above, self.cmd), GRIP_PUSH, 50.0 * pace)),
            ("align", lambda: self._line(above, pre, 0.04 * pace)),
            ("settle", lambda: self._hold(int(self.rng.integers(4, 10)))),
            ("approach", lambda: self._line(pre, face - u * 0.003, 0.02 * pace)),
            ("push", lambda: self._line(face - u * 0.003, face + u * PUSH_DEPTH, PUSH_SPEED * pace)),
            ("hold", lambda: self._hold(int(self.rng.integers(6, 12)))),
            ("retract", lambda: self._line(face + u * PUSH_DEPTH, pre, 0.03 * pace)),
            ("home", lambda: self._joint_move(self.home, self.home[5], 50.0 * pace)),
        ]
        for phase, make in plan:
            for cmd in make():
                yield phase, cmd

    def act(self, obs: dict[str, Any]) -> np.ndarray:
        if not self.done:
            try:
                self.phase, self.cmd = next(self.steps)
            except StopIteration:
                self.done = True
            except RuntimeError as e:  # no IK: hold still and report it
                self.failed, self.done = str(e), True
        return self.cmd.astype(np.float32)
