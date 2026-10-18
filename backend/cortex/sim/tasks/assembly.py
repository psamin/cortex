"""Factory assembly (issue #161): insert two machined pegs from a parts tray into a fixture.

Scene (metres, table top at z = 0, robot frame: +x away from the base, +y to its left):

* a dark rubber mat over the workspace;
* a parts tray (right of the arm) holding a round peg and a square peg, standing upright;
* a machined fixture (left of the arm): a steel base plate under a brushed-aluminium block with a
  round hole and a square hole. MuJoCo has no CSG, so the block is built from boxes around two square
  cutouts; the round hole is a ring of 16 boxes inside its cutout.

Both the tray and the fixture lie along an arc about 0.21 m from the arm base, where the SO-101
reaches straight down at carry height. Holes are 25.5 mm across; pegs are 21.5-22.5 mm, so the
clearance is 3-4 mm.

Success: every peg is seated in the hole of its own shape and released (see ``Env.seated``).
The ``Oracle`` reads privileged sim state (part and hole poses) and is a demonstrator, not a policy.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET  # nosec B405 - we build this XML ourselves
from typing import Any

import mujoco
import numpy as np

from ..expert import solve_ik
from .base import TaskEnv

PARTS = ("round", "square")
HOLE_HALF = 0.01275  # holes are 25.5 mm across (round: the ring's inscribed circle)
PART_HALF_W = (0.01075, 0.01125)  # peg half widths: 3.0-4.0 mm clearance
PART_HALF_H = (0.025, 0.027)  # peg half heights: 5.0-5.4 cm tall

PLATE_H = 0.004  # steel base plate; the hole floors are its top
BLOCK_H = 0.022  # aluminium block on the plate
FIX_TOP = PLATE_H + BLOCK_H  # fixture top above the table
FIX_HALF = (0.035, 0.075)  # block half extents (local x radial, local y along the holes)
HOLE_Y = {"round": -0.035, "square": 0.035}  # hole centres along the block
FIX_XY, FIX_YAW = np.array([0.176, 0.114]), np.deg2rad(33.0)
FIX_JITTER, FIX_YAW_JITTER = 0.015, np.deg2rad(10.0)

TRAY_XY, TRAY_YAW = np.array([0.172, -0.120]), np.deg2rad(-35.0)
TRAY_FLOOR = 0.003
TRAY_IN = (0.043, 0.095)  # inner half extents
SLOT_Y = {"round": -0.05, "square": 0.05}  # nominal part spots along the tray
PART_JITTER = 0.02

SEAT_DEPTH = 0.015  # a seated peg's bottom is at least this far below the fixture top
MAX_TILT = np.deg2rad(12.0)
MAX_SPEED = 0.05

# name -> (rgb, specular, shininess)
FINISHES = {
    "anodized red": ((0.62, 0.07, 0.08), 0.6, 0.6),
    "anodized blue": ((0.10, 0.24, 0.62), 0.6, 0.6),
    "black oxide": ((0.09, 0.09, 0.10), 0.5, 0.7),
    "brass": ((0.74, 0.58, 0.26), 0.9, 0.8),
}
TABLE_RGB = np.array([0.55, 0.57, 0.60])  # grey steel workbench, tinted per seed


def _yaw_quat(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def _rot(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]])


def _box(parent: ET.Element, name: str, center: tuple[float, float, float], half: tuple[float, float, float],
         material: str, yaw: float = 0.0, **kw: str) -> None:
    ET.SubElement(parent, "geom", name=name, type="box", material=material,
                  pos=" ".join(f"{v:.5f}" for v in center), size=" ".join(f"{v:.5f}" for v in half),
                  quat=" ".join(f"{v:.6f}" for v in _yaw_quat(yaw)), **kw)


class Env(TaskEnv):
    name = "assembly"
    instruction = "Insert the round peg into the round hole and the square peg into the square hole."
    max_steps = 900
    focus = "part_round"

    # ----------------------------------------------------------------- scene
    def build(self, mj: ET.Element, wb: ET.Element) -> None:
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
            **{f"part_{p}": dict(rgba="0.6 0.6 0.6 1", specular="0.6", shininess="0.6") for p in PARTS},
        }
        for name, attrs in mats.items():
            ET.SubElement(asset, "material", name=name, **attrs)

        # Rubber mat: visual only, 1 mm proud of the table, so everything still rests on the table.
        ET.SubElement(wb, "geom", name="mat", type="box", pos="0.19 0 0.0005", size="0.16 0.26 0.0005",
                      material="rubber_mat", contype="0", conaffinity="0")

        for cam in wb.findall("camera"):  # closer and steeper than upstream: frames the cell, sees into holes
            if cam.get("name") == "front":
                e = np.deg2rad(46.0)
                cam.set("pos", "0.40 0 0.25")
                cam.set("xyaxes", f"0 1 0 {-np.sin(e):.4f} 0 {np.cos(e):.4f}")
        self._build_fixture(wb)
        self._build_tray(wb)
        for p in PARTS:
            b = ET.SubElement(wb, "body", name=f"part_{p}", pos=f"0.6 {1.2 + 0.1 * PARTS.index(p)} 0.1")
            ET.SubElement(b, "freejoint", name=f"part_{p}_free")
            kind, size = ("cylinder", f"{PART_HALF_W[1]} {PART_HALF_H[1]}") if p == "round" else \
                ("box", f"{PART_HALF_W[1]} {PART_HALF_W[1]} {PART_HALF_H[1]}")
            ET.SubElement(b, "geom", name=f"part_{p}_geom", type=kind, size=size, mass="0.035",
                          material=f"part_{p}", condim="4", friction="1 0.01 0.001", solref="0.01 1")

    def _build_fixture(self, wb: ET.Element) -> None:
        f = ET.SubElement(wb, "body", name="fixture", pos=f"{FIX_XY[0]} {FIX_XY[1]} 0",
                          quat=" ".join(f"{v:.6f}" for v in _yaw_quat(FIX_YAW)))
        hx, hy = FIX_HALF
        _box(f, "fix_plate", (0, 0, PLATE_H / 2), (hx + 0.010, hy + 0.010, PLATE_H / 2), "fixture_steel")
        zc, zh = PLATE_H + BLOCK_H / 2, BLOCK_H / 2
        cut = {"round": HOLE_HALF + 0.001, "square": HOLE_HALF}  # square cutouts; the ring fills the round one
        # Full-width bars between and beside the cutouts, then the two side bars at each cutout.
        edges = [-hy, HOLE_Y["round"] - cut["round"], HOLE_Y["round"] + cut["round"],
                 HOLE_Y["square"] - cut["square"], HOLE_Y["square"] + cut["square"], hy]
        for i, (a, b) in enumerate(zip(edges[::2], edges[1::2])):
            _box(f, f"fix_bar{i}", (0, (a + b) / 2, zc), (hx, (b - a) / 2, zh), "fixture_al")
        for p in PARTS:
            c = cut[p]
            for s in (-1, 1):
                _box(f, f"fix_{p}_side{s:+d}", (s * (hx + c) / 2, HOLE_Y[p], zc), ((hx - c) / 2, c, zh),
                     "fixture_al")
        n, t = 16, 0.008  # ring: inscribed radius HOLE_HALF, closed out to HOLE_HALF + t
        half_len = (HOLE_HALF + t) * np.tan(np.pi / n)
        for k in range(n):
            a = 2 * np.pi * k / n
            r = HOLE_HALF + t / 2
            _box(f, f"fix_ring{k}", (r * np.cos(a), HOLE_Y["round"] + r * np.sin(a), zc), (t / 2, half_len, zh),
                 "fixture_al", yaw=a)
        for i, (sx, sy) in enumerate([(-1, -1), (-1, 1), (1, -1), (1, 1)]):  # cap screws, visual only
            ET.SubElement(f, "geom", name=f"fix_bolt{i}", type="cylinder", size="0.0035 0.0012",
                          pos=f"{sx * (hx - 0.008):.4f} {sy * (hy - 0.008):.4f} {FIX_TOP + 0.0010:.4f}",
                          material="bolt", contype="0", conaffinity="0")

    def _build_tray(self, wb: ET.Element) -> None:
        t = ET.SubElement(wb, "body", name="tray", pos=f"{TRAY_XY[0]} {TRAY_XY[1]} 0",
                          quat=" ".join(f"{v:.6f}" for v in _yaw_quat(TRAY_YAW)))
        ix, iy = TRAY_IN
        w, h = 0.004, 0.014
        _box(t, "tray_floor", (0, 0, TRAY_FLOOR / 2), (ix + w, iy + w, TRAY_FLOOR / 2), "tray")
        for s in (-1, 1):
            _box(t, f"tray_wall_x{s:+d}", (s * (ix + w / 2), 0, h / 2), (w / 2, iy + w, h / 2), "tray")
            _box(t, f"tray_wall_y{s:+d}", (0, s * (iy + w / 2), h / 2), (ix, w / 2, h / 2), "tray")

    # ------------------------------------------------------------- per seed
    def bind_task(self) -> None:
        m = self.world.model
        self.part_body = {p: self.body(f"part_{p}") for p in PARTS}
        self.part_geom = {p: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"part_{p}_geom") for p in PARTS}
        self.part_mat = {p: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, f"part_{p}") for p in PARTS}
        self.table_mat = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, "table")
        self.fixture = self.body("fixture")
        self.front_cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "front")
        self.cam_quat0 = m.cam_quat[self.front_cam].copy()
        root = m.body_rootid[self.body("gripper")]
        self.robot_bodies = {b for b in range(m.nbody) if m.body_rootid[b] == root}
        self.half_w = dict.fromkeys(PARTS, PART_HALF_W[1])
        self.half_h = dict.fromkeys(PARTS, PART_HALF_H[1])
        self.finish: dict[str, str] = {}

    def reset_task(self, rng: np.random.Generator) -> None:
        m = self.world.model
        # Fixture pose. The upstream reset restores body poses from the XML first.
        m.body_pos[self.fixture, :2] = FIX_XY + rng.uniform(-FIX_JITTER, FIX_JITTER, 2)
        m.body_quat[self.fixture] = _yaw_quat(FIX_YAW + rng.uniform(-FIX_YAW_JITTER, FIX_YAW_JITTER))
        # Parts: size within the clearance budget, finish, and pose in the tray.
        names = list(FINISHES)
        picks = rng.choice(len(names), size=len(PARTS), replace=False)
        for p, k in zip(PARTS, picks):
            hw, hh = rng.uniform(*PART_HALF_W), rng.uniform(*PART_HALF_H)
            self.half_w[p], self.half_h[p] = hw, hh
            g = self.part_geom[p]
            half = np.array([hw, hw, hh])
            m.geom_size[g] = (hw, hh, 0) if p == "round" else half
            m.geom_aabb[g] = np.r_[0, 0, 0, half]
            m.geom_rbound[g] = np.linalg.norm(half)
            rgb, spec, shine = FINISHES[names[k]]
            self.finish[p] = names[k]
            m.mat_rgba[self.part_mat[p]] = (*rgb, 1)
            m.mat_specular[self.part_mat[p]] = spec
            m.mat_shininess[self.part_mat[p]] = shine
            local = np.array([0.0, SLOT_Y[p]]) + rng.uniform(-PART_JITTER, PART_JITTER, 2)
            xy = TRAY_XY + _rot(TRAY_YAW) @ local
            self.set_free(f"part_{p}", (xy[0], xy[1], TRAY_FLOOR + hh), yaw=rng.uniform(-np.pi / 4, np.pi / 4))
        # Workbench tint and a small front-camera rotation (the upstream reset jitters its position).
        tint = rng.uniform(0.8, 1.15) * rng.uniform(0.96, 1.04, 3)  # brightness, then a slight cast
        m.mat_rgba[self.table_mat] = (*np.clip(TABLE_RGB * tint, 0, 1), 1)
        tilt = rng.uniform(-0.02, 0.02, 3)  # rad, about the camera's own axes
        dq = np.zeros(4)
        mujoco.mju_axisAngle2Quat(dq, tilt / (np.linalg.norm(tilt) or 1.0), np.linalg.norm(tilt))
        q = np.zeros(4)
        mujoco.mju_mulQuat(q, self.cam_quat0, dq)
        m.cam_quat[self.front_cam] = q

    # ------------------------------------------------------------ geometry
    def fixture_top(self) -> float:
        return float(self.world.data.xpos[self.fixture][2] + FIX_TOP)

    def fixture_yaw(self) -> float:
        R = self.world.data.xmat[self.fixture].reshape(3, 3)
        return float(np.arctan2(R[1, 0], R[0, 0]))

    def hole_xy(self, p: str) -> np.ndarray:
        d = self.world.data
        R = d.xmat[self.fixture].reshape(3, 3)
        return d.xpos[self.fixture][:2] + R[:2, 1] * HOLE_Y[p]

    def part_axis(self, p: str) -> np.ndarray:
        return self.world.data.xmat[self.part_body[p]].reshape(3, 3)[:, 2].copy()

    def part_bottom(self, p: str) -> np.ndarray:
        return self.world.data.xpos[self.part_body[p]] - self.part_axis(p) * self.half_h[p]

    def part_yaw(self, p: str) -> float:
        R = self.world.data.xmat[self.part_body[p]].reshape(3, 3)
        return float(np.arctan2(R[1, 0], R[0, 0]))

    def touching_robot(self, p: str) -> bool:
        d, m = self.world.data, self.world.model
        b = self.part_body[p]
        for c in d.contact[:d.ncon]:
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            if (b1 == b and b2 in self.robot_bodies) or (b2 == b and b1 in self.robot_bodies):
                return True
        return False

    def seated(self, p: str) -> bool:
        """Peg ``p`` is seated: upright (tilt <= 12 deg), its bottom >= 1.5 cm below the fixture top,
        its bottom centre inside the hole of its own shape, at rest and not touching the robot."""
        d = self.world.data
        if self.part_axis(p)[2] < np.cos(MAX_TILT):
            return False
        bottom = self.part_bottom(p)
        if bottom[2] > self.fixture_top() - SEAT_DEPTH:
            return False
        R = d.xmat[self.fixture].reshape(3, 3)
        off = (bottom[:2] - self.hole_xy(p)) @ R[:2, :2]  # in the fixture frame
        inside = np.linalg.norm(off) < HOLE_HALF if p == "round" else np.all(np.abs(off) < HOLE_HALF)
        if not inside:
            return False
        a = self.world.model.jnt_dofadr[self.world.model.body_jntadr[self.part_body[p]]]
        if np.linalg.norm(d.qvel[a:a + 3]) > MAX_SPEED:
            return False
        return not self.touching_robot(p)

    def success(self) -> bool:
        return all(self.seated(p) for p in PARTS)

    def metrics(self) -> dict[str, Any]:
        return {"seated": [p for p in PARTS if self.seated(p)], "finish": dict(self.finish)}


# ---------------------------------------------------------------------------------------------- oracle
GRIP_OPEN, GRIP_SHUT = 28.0, 0.0  # approach/release opening (jaws ~2 cm clear of the far face), closed
FACE = 0.0199  # the fixed jaw's inner face sits this far behind the gripper site along the closing axis
GAP = 0.0015  # fixed-jaw clearance to the peg on the way down
GRASP_DEPTH = 0.016  # fingertips this far below the peg top
CARRY_CLEAR = 0.014  # peg bottom above the fixture top while carrying
INSERT_LOW = 0.004  # peg bottom above the hole floor when the gripper lets go
ALIGN_TOL = 0.0006  # peg bottom to hole centre (m) before the descent


def _min_jerk(n: int) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n + 1)[1:]
    return 10 * t**3 - 15 * t**4 + 6 * t**5


def _wrap(a: float, period: float = 2 * np.pi) -> float:
    return (a + period / 2) % period - period / 2


class Oracle:
    """Scripted assembler: reads privileged sim state (peg, hole and gripper poses), so it is a
    demonstrator, not a learned policy. For each peg: hover, descend, close, lift, carry above its
    hole with the jaws turned so a square peg lines up with its hole, centre the peg's bottom over the
    hole (closed loop, a few mm moves), descend slowly into it, release, back off and rise; then home.
    Motions are min-jerk in time; vertical moves are straight Cartesian lines (IK every 5 mm)."""

    name = "assembly oracle (privileged state)"
    needs_pixels = False

    def reset(self, env: Env, obs: dict[str, Any], seed: int) -> None:
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
                q[4] += np.rad2deg(_wrap(yaw - self._fk(q)[1]))
            miss = goal - self._fk(q)[0]
            if np.linalg.norm(miss) < 3e-4:
                break
            tgt = tgt + miss
        return q

    # ------------------------------------------------------------- primitives
    def _joint(self, q: np.ndarray, grip: float, settle: int = 4, n: int | None = None) -> list[np.ndarray]:
        q1 = np.r_[q[:5], grip]
        n = n or int(np.clip(np.abs(q1[:5] - self.cmd[:5]).max() / 1.6, 20, 60))
        out = [self.cmd + (q1 - self.cmd) * s for s in _min_jerk(n)]
        return out + [q1] * settle

    def _line(self, xyz: np.ndarray, yaw: float, grip: float, n: int, settle: int = 4) -> list[np.ndarray]:
        p0, psi0 = self._fk(self.cmd)
        xyz = np.asarray(xyz, dtype=np.float64)
        k = int(np.clip(np.linalg.norm(xyz - p0) / 0.005, 1, 12))
        dpsi = _wrap(yaw - psi0)
        qs = [self.cmd[:5]]
        for s in np.linspace(0.0, 1.0, k + 1)[1:]:
            qs.append(self._ik(p0 + (xyz - p0) * s, psi0 + dpsi * s, qs[-1]))
        qs_arr, g0 = np.array(qs), self.cmd[5]
        out = []
        for s in _min_jerk(n):
            x = s * k
            i = min(int(x), k - 1)
            out.append(np.r_[qs_arr[i] + (qs_arr[i + 1] - qs_arr[i]) * (x - i), g0 + (grip - g0) * s])
        return out + [out[-1]] * settle

    def _grip(self, grip: float, n: int, settle: int) -> list[np.ndarray]:
        q1 = self.cmd.copy()
        q1[5] = grip
        return [self.cmd + (q1 - self.cmd) * s for s in _min_jerk(n)] + [q1] * settle

    # ----------------------------------------------------------------- script
    def _script(self):
        # The square peg sits at the tray end nearer the fixture and goes into the far hole, so neither
        # carry passes over a peg that is still standing (a carried peg clears the fixture, not a peg).
        for p in ("square", "round"):
            yield from self._pick(p)
            yield from self._insert(p)
        self.phase = "home"
        home = np.asarray(self.env.module.HOME, dtype=np.float64)
        yield self._joint(home, home[5], settle=6, n=45)

    def _pick(self, p: str):
        env = self.env
        c = env.world.data.xpos[env.part_body[p]].copy()
        radial = float(np.arctan2(c[1], c[0]))
        yaw = radial
        if p == "square":  # close on a pair of faces, the pair nearest the radial heading
            yaw = env.part_yaw(p) + round(_wrap(radial - env.part_yaw(p)) / (np.pi / 2)) * np.pi / 2
        u = np.array([np.cos(yaw), np.sin(yaw)])
        site = c[:2] + (FACE - env.half_w[p] - GAP) * u
        z = c[2] + env.half_h[p] - GRASP_DEPTH
        self.phase = f"approach {p}"
        yield self._joint(self._ik(np.r_[site, z + 0.035], yaw, self.cmd), GRIP_OPEN, settle=4)
        yield self._line(np.r_[site, z], yaw, GRIP_OPEN, 28, settle=4)
        self.phase = f"grasp {p}"
        yield self._grip(GRIP_SHUT, 14, settle=8)
        self.phase = f"lift {p}"
        s, _ = self._site()
        lift = FIX_TOP + CARRY_CLEAR + (s[2] - env.part_bottom(p)[2])
        yield self._line(np.r_[site, lift], yaw, GRIP_SHUT, 28, settle=3)

    def _insert(self, p: str):
        env = self.env
        hole, fyaw = env.hole_xy(p), env.fixture_yaw()
        s, psi = self._site()
        b = env.part_bottom(p)
        held = _rot(-psi) @ (b[:2] - s[:2])  # peg bottom in the jaw frame
        dz = s[2] - b[2]
        radial = float(np.arctan2(hole[1], hole[0]))
        yaw = radial
        if p == "square":  # turn the jaws so the peg's faces line up with the hole's
            rel = _wrap(env.part_yaw(p) - psi, np.pi / 2)
            yaw = fyaw - rel
            yaw += round(_wrap(radial - yaw) / (np.pi / 2)) * np.pi / 2
        self.phase = f"carry {p}"
        target = np.r_[hole - _rot(yaw) @ held, FIX_TOP + CARRY_CLEAR + dz]
        yield self._joint(self._ik(target, yaw, self.cmd), GRIP_SHUT, settle=6)
        self.phase = f"align {p}"
        for _ in range(4):
            err = hole - env.part_bottom(p)[:2]
            turn = _wrap(fyaw - env.part_yaw(p), np.pi / 2) if p == "square" else 0.0
            if np.linalg.norm(err) < ALIGN_TOL and abs(turn) < np.deg2rad(1.5):
                break
            p0, psi0 = self._fk(self.cmd)
            yield self._line(np.r_[p0[:2] + err, p0[2]], psi0 + turn, GRIP_SHUT, 10, settle=5)
        self.phase = f"insert {p}"
        p0, psi0 = self._fk(self.cmd)
        low = PLATE_H + INSERT_LOW + dz
        yield self._line(np.r_[p0[:2], low], psi0, GRIP_SHUT, 45, settle=4)
        self.phase = f"release {p}"
        yield self._grip(GRIP_OPEN, 12, settle=8)
        self.phase = f"retreat {p}"
        back = p0[:2] - 0.002 * np.array([np.cos(psi0), np.sin(psi0)])  # fixed jaw off the peg
        yield self._line(np.r_[back, low + 0.003], psi0, GRIP_OPEN, 8, settle=0)
        yield self._line(np.r_[back, FIX_TOP + CARRY_CLEAR + dz + 0.01], psi0, GRIP_OPEN, 24, settle=2)
