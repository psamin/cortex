"""The factory workcell shared by the assembly and unscrew tasks (issue #161).

* ``add_workcell``: brushed-aluminium, steel, cap-screw and tray materials, a dark rubber mat over the
  workspace, and a front camera closer and steeper than upstream (it frames the cell and sees into
  holes at 240x320 and 96x128).
* ``WorkcellEnv``: per-seed workbench tint and a small front-camera rotation (the upstream reset already
  jitters its position), plus robot-contact checks.
* ``TopDownOracle``: scripted top-down motion for oracles that read privileged sim state: IK with the
  jaws' closing axis at a given heading, min-jerk joint moves and Cartesian paths, gripper moves.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET  # nosec B405 - we build this XML ourselves
from collections.abc import Iterator
from typing import Any

import mujoco
import numpy as np

from ..expert import solve_ik
from .base import TaskEnv

PLATE_H = 0.004  # steel base plate under a fixture
BLOCK_H = 0.022  # aluminium block on the plate
FIX_TOP = PLATE_H + BLOCK_H  # fixture top above the table
TABLE_RGB = np.array([0.55, 0.57, 0.60])  # grey steel workbench, tinted per seed
FACE = 0.0199  # the fixed jaw's inner face sits this far behind the gripper site along the closing axis
GRIP_OPEN, GRIP_SHUT = 28.0, 0.0  # jaws ~2 cm clear of a 2.2 cm part's far face; closed


def yaw_quat(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def rot(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]])


def wrap(a: float, period: float = 2 * np.pi) -> float:
    return (a + period / 2) % period - period / 2


def min_jerk(n: int) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n + 1)[1:]
    return 10 * t**3 - 15 * t**4 + 6 * t**5


def box(parent: ET.Element, name: str, center: tuple[float, float, float], half: tuple[float, float, float],
        material: str, yaw: float = 0.0, **kw: str) -> None:
    ET.SubElement(parent, "geom", name=name, type="box", material=material,
                  pos=" ".join(f"{v:.5f}" for v in center), size=" ".join(f"{v:.5f}" for v in half),
                  quat=" ".join(f"{v:.6f}" for v in yaw_quat(yaw)), **kw)


def add_workcell(mj: ET.Element, wb: ET.Element) -> None:
    asset = mj.find("asset")
    assert asset is not None
    ET.SubElement(asset, "texture", name="brushed", type="cube", builtin="flat", width="128", height="128",
                  rgb1="0.58 0.60 0.62", mark="random", markrgb="0.70 0.72 0.74", random="0.25")
    mats = {
        "fixture_al": dict(texture="brushed", rgba="1 1 1 1", specular="0.7", shininess="0.5"),
        "fixture_steel": dict(rgba="0.32 0.33 0.35 1", specular="0.5", shininess="0.4"),
        "bolt": dict(rgba="0.07 0.07 0.08 1", specular="0.8", shininess="0.8"),
        "rubber_mat": dict(rgba="0.11 0.12 0.12 1", specular="0.02", shininess="0.05"),
        "tray": dict(rgba="0.12 0.16 0.22 1", specular="0.15", shininess="0.2"),
    }
    for name, attrs in mats.items():
        ET.SubElement(asset, "material", name=name, **attrs)
    # Rubber mat: visual only, 1 mm proud of the table, so everything still rests on the table.
    ET.SubElement(wb, "geom", name="mat", type="box", pos="0.19 0 0.0005", size="0.16 0.26 0.0005",
                  material="rubber_mat", contype="0", conaffinity="0")
    for cam in wb.findall("camera"):
        if cam.get("name") == "front":
            e = np.deg2rad(46.0)
            cam.set("pos", "0.40 0 0.25")
            cam.set("xyaxes", f"0 1 0 {-np.sin(e):.4f} 0 {np.cos(e):.4f}")


def add_plate(body: ET.Element, hx: float, hy: float, top: float = FIX_TOP) -> None:
    """The steel base plate (10 mm wider than the block) and four visual cap screws on the block top."""
    box(body, "fix_plate", (0, 0, PLATE_H / 2), (hx + 0.010, hy + 0.010, PLATE_H / 2), "fixture_steel")
    for i, (sx, sy) in enumerate([(-1, -1), (-1, 1), (1, -1), (1, 1)]):
        ET.SubElement(body, "geom", name=f"fix_bolt{i}", type="cylinder", size="0.0035 0.0012",
                      pos=f"{sx * (hx - 0.008):.4f} {sy * (hy - 0.008):.4f} {top + 0.0010:.4f}",
                      material="bolt", contype="0", conaffinity="0")


def add_ring(body: ET.Element, name: str, center: tuple[float, float], radius: float, z: float, half_h: float,
             n: int = 16, t: float = 0.008) -> None:
    """A round hole wall: ``n`` boxes whose inner faces form an n-gon of inscribed radius ``radius``,
    closed out to ``radius + t`` (adjacent boxes overlap, so there are no gaps)."""
    half_len = (radius + t) * np.tan(np.pi / n)
    r = radius + t / 2
    for k in range(n):
        a = 2 * np.pi * k / n
        box(body, f"{name}{k}", (center[0] + r * np.cos(a), center[1] + r * np.sin(a), z), (t / 2, half_len, half_h),
            "fixture_al", yaw=a)


def add_holed_block(body: ET.Element, name: str, hx: float, hy: float, z0: float, z1: float, hole_r: float,
                    material: str = "fixture_al", t: float = 0.004, **kw: str) -> None:
    """A block from z0 to z1 with a round through-hole of radius ``hole_r`` at its centre: bars around a
    square cutout, and a ring of boxes (thickness ``t``) inside it. ``kw`` goes to every geom."""
    zc, zh = (z0 + z1) / 2, (z1 - z0) / 2
    c = hole_r + 0.001
    assert hole_r + t >= c * np.sqrt(2), "the ring must cover the cutout's corners"
    for s in (-1, 1):
        box(body, f"{name}_bar{s:+d}", (0, s * (hy + c) / 2, zc), (hx, (hy - c) / 2, zh), material, **kw)
        box(body, f"{name}_side{s:+d}", (s * (hx + c) / 2, 0, zc), ((hx - c) / 2, c, zh), material, **kw)
    half_len = (hole_r + t) * np.tan(np.pi / 16)
    for k in range(16):
        a = 2 * np.pi * k / 16
        r = hole_r + t / 2
        box(body, f"{name}_ring{k}", (r * np.cos(a), r * np.sin(a), zc), (t / 2, half_len, zh), material, yaw=a,
            **kw)


def add_tray(wb: ET.Element, xy: np.ndarray, yaw: float, inner: tuple[float, float] = (0.043, 0.095),
             floor: float = 0.003) -> None:
    t = ET.SubElement(wb, "body", name="tray", pos=f"{xy[0]} {xy[1]} 0",
                      quat=" ".join(f"{v:.6f}" for v in yaw_quat(yaw)))
    ix, iy = inner
    w, h = 0.004, 0.014
    box(t, "tray_floor", (0, 0, floor / 2), (ix + w, iy + w, floor / 2), "tray")
    for s in (-1, 1):
        box(t, f"tray_wall_x{s:+d}", (s * (ix + w / 2), 0, h / 2), (w / 2, iy + w, h / 2), "tray")
        box(t, f"tray_wall_y{s:+d}", (0, s * (iy + w / 2), h / 2), (ix, w / 2, h / 2), "tray")


class WorkcellEnv(TaskEnv):
    """A TaskEnv on the workcell: subclasses call ``super().bind_task()`` and end ``reset_task`` with
    ``self.jitter_look(rng)``."""

    def bind_task(self) -> None:
        m = self.world.model
        self.table_mat = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, "table")
        self.front_cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "front")
        self.cam_quat0 = m.cam_quat[self.front_cam].copy()
        root = m.body_rootid[self.body("gripper")]
        self.robot_bodies = {b for b in range(m.nbody) if m.body_rootid[b] == root}

    def jitter_look(self, rng: np.random.Generator) -> None:
        m = self.world.model
        tint = rng.uniform(0.8, 1.15) * rng.uniform(0.96, 1.04, 3)  # brightness, then a slight cast
        m.mat_rgba[self.table_mat] = (*np.clip(TABLE_RGB * tint, 0, 1), 1)
        tilt = rng.uniform(-0.02, 0.02, 3)  # rad, about the camera's own axes
        dq = np.zeros(4)
        mujoco.mju_axisAngle2Quat(dq, tilt / (np.linalg.norm(tilt) or 1.0), np.linalg.norm(tilt))
        q = np.zeros(4)
        mujoco.mju_mulQuat(q, self.cam_quat0, dq)
        m.cam_quat[self.front_cam] = q

    def robot_touches(self, body: int) -> bool:
        d, m = self.world.data, self.world.model
        for c in d.contact[:d.ncon]:
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            if (b1 == body and b2 in self.robot_bodies) or (b2 == body and b1 in self.robot_bodies):
                return True
        return False


class TopDownOracle:
    """Base for scripted oracles that read privileged sim state (so they are demonstrators, not learned
    policies). ``_script`` is a generator of command lists; each list plays out (one command per step)
    before the script resumes, so every segment is planned from the state it starts in. Motions are
    min-jerk in time; Cartesian paths solve IK every few mm (or degrees) and interpolate joints."""

    name = "oracle (privileged state)"
    needs_pixels = False

    def reset(self, env: WorkcellEnv, obs: dict[str, Any], seed: int) -> None:
        self.env = env
        self.d = mujoco.MjData(env.world.model)
        self.cmd = np.asarray(obs["agent_pos"], dtype=np.float64).copy()
        self.queue: list[np.ndarray] = []
        self.script = self._script()
        self.done = False
        self.phase = "start"

    def act(self, obs: dict[str, Any]) -> np.ndarray:
        while not self.queue and not self.done:
            try:
                self.queue = list(next(self.script))
            except StopIteration:
                self.done = True
        if self.queue:
            self.cmd = self.queue.pop(0)
        return self.cmd.astype(np.float32)

    def _script(self) -> Iterator[list[np.ndarray]]:
        raise NotImplementedError

    # -------------------------------------------------------------- kinematics
    def _fk(self, q: np.ndarray) -> tuple[np.ndarray, float]:
        """Gripper site position and jaw yaw (heading of the closing axis) at arm joints ``q`` (deg)."""
        w, d = self.env.world, self.d
        d.qpos[:] = w.data.qpos
        d.qpos[w.qadr[:5]] = np.deg2rad(q[:5])
        mujoco.mj_kinematics(w.model, d)
        R = d.site_xmat[self.env.site].reshape(3, 3)
        return d.site_xpos[self.env.site].copy(), float(np.arctan2(R[1, 2], R[0, 2]))

    def _site(self) -> tuple[np.ndarray, float]:
        """Measured gripper site position and jaw yaw."""
        d = self.env.world.data
        R = d.site_xmat[self.env.site].reshape(3, 3)
        return d.site_xpos[self.env.site].copy(), float(np.arctan2(R[1, 2], R[0, 2]))

    def _ik(self, xyz: np.ndarray, yaw: float, seed: np.ndarray) -> np.ndarray:
        """Top-down IK with the jaws' closing axis at heading ``yaw``. The wrist roll is set after each
        position solve; it moves the site, so the target is shifted by the miss and solved again."""
        goal = np.asarray(xyz, dtype=np.float64)
        tgt, q = goal.copy(), np.asarray(seed[:5], dtype=np.float64).copy()
        for _ in range(6):
            q, _ = solve_ik(self.env, tgt, q, iters=100)
            for _ in range(3):
                q[4] += np.rad2deg(wrap(yaw - self._fk(q)[1]))
            miss = goal - self._fk(q)[0]
            if np.linalg.norm(miss) < 3e-4:
                break
            tgt = tgt + miss
        return q

    # ------------------------------------------------------------- primitives
    def _joint(self, q: np.ndarray, grip: float, settle: int = 4, n: int | None = None) -> list[np.ndarray]:
        q1 = np.r_[q[:5], grip]
        n = n or int(np.clip(np.abs(q1[:5] - self.cmd[:5]).max() / 1.6, 20, 60))
        out = [self.cmd + (q1 - self.cmd) * s for s in min_jerk(n)]
        return out + [q1] * settle

    def _path(self, points: list[tuple[np.ndarray, float]], grip: float, n: int, settle: int = 4) -> list[np.ndarray]:
        """Through the (site xyz, jaw yaw) waypoints from the current command, min-jerk over ``n`` steps."""
        qs = [self.cmd[:5]]
        for xyz, yaw in points:
            qs.append(self._ik(xyz, yaw, qs[-1]))
        qs_arr, g0, k = np.array(qs), self.cmd[5], len(points)
        out = []
        for s in min_jerk(n):
            x = s * k
            i = min(int(x), k - 1)
            out.append(np.r_[qs_arr[i] + (qs_arr[i + 1] - qs_arr[i]) * (x - i), g0 + (grip - g0) * s])
        return out + [out[-1]] * settle

    def _line(self, xyz: np.ndarray, yaw: float, grip: float, n: int, settle: int = 4) -> list[np.ndarray]:
        """Straight Cartesian line of the site (IK every 5 mm), turning the jaws by the short way."""
        p0, psi0 = self._fk(self.cmd)
        xyz = np.asarray(xyz, dtype=np.float64)
        k = int(np.clip(np.linalg.norm(xyz - p0) / 0.005, 1, 12))
        dpsi = wrap(yaw - psi0)
        pts = [(p0 + (xyz - p0) * s, psi0 + dpsi * s) for s in np.linspace(0.0, 1.0, k + 1)[1:]]
        return self._path(pts, grip, n, settle)

    def _grip(self, grip: float, n: int, settle: int) -> list[np.ndarray]:
        q1 = self.cmd.copy()
        q1[5] = grip
        return [self.cmd + (q1 - self.cmd) * s for s in min_jerk(n)] + [q1] * settle
