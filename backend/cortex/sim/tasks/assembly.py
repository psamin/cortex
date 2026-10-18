"""Factory assembly (issue #161): insert two machined pegs from a parts tray into a fixture.

Scene (metres, table top at z = 0, robot frame: +x away from the base, +y to its left), on the shared
workcell (``workcell.py``: rubber mat, materials, front camera):

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

from .workcell import (
    BLOCK_H,
    FACE,
    FIX_TOP,
    GRIP_OPEN,
    GRIP_SHUT,
    PLATE_H,
    TopDownOracle,
    WorkcellEnv,
    add_plate,
    add_ring,
    add_tray,
    add_workcell,
    box,
    rot,
    wrap,
    yaw_quat,
)

PARTS = ("round", "square")
HOLE_HALF = 0.01275  # holes are 25.5 mm across (round: the ring's inscribed circle)
PART_HALF_W = (0.01075, 0.01125)  # peg half widths: 3.0-4.0 mm clearance
PART_HALF_H = (0.025, 0.027)  # peg half heights: 5.0-5.4 cm tall

FIX_HALF = (0.035, 0.075)  # block half extents (local x radial, local y along the holes)
HOLE_Y = {"round": -0.035, "square": 0.035}  # hole centres along the block
FIX_XY, FIX_YAW = np.array([0.176, 0.114]), np.deg2rad(33.0)
FIX_JITTER, FIX_YAW_JITTER = 0.015, np.deg2rad(10.0)

TRAY_XY, TRAY_YAW = np.array([0.172, -0.120]), np.deg2rad(-35.0)
TRAY_FLOOR = 0.003
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


class Env(WorkcellEnv):
    name = "assembly"
    instruction = "Insert the round peg into the round hole and the square peg into the square hole."
    max_steps = 900
    focus = "part_round"

    # ----------------------------------------------------------------- scene
    def build(self, mj: ET.Element, wb: ET.Element) -> None:
        add_workcell(mj, wb)
        asset = mj.find("asset")
        assert asset is not None
        for p in PARTS:
            ET.SubElement(asset, "material", name=f"part_{p}", rgba="0.6 0.6 0.6 1", specular="0.6",
                          shininess="0.6")
        self._build_fixture(wb)
        add_tray(wb, TRAY_XY, TRAY_YAW, floor=TRAY_FLOOR)
        for p in PARTS:
            b = ET.SubElement(wb, "body", name=f"part_{p}", pos=f"0.6 {1.2 + 0.1 * PARTS.index(p)} 0.1")
            ET.SubElement(b, "freejoint", name=f"part_{p}_free")
            kind, size = ("cylinder", f"{PART_HALF_W[1]} {PART_HALF_H[1]}") if p == "round" else \
                ("box", f"{PART_HALF_W[1]} {PART_HALF_W[1]} {PART_HALF_H[1]}")
            ET.SubElement(b, "geom", name=f"part_{p}_geom", type=kind, size=size, mass="0.035",
                          material=f"part_{p}", condim="4", friction="1 0.01 0.001", solref="0.01 1")

    def _build_fixture(self, wb: ET.Element) -> None:
        f = ET.SubElement(wb, "body", name="fixture", pos=f"{FIX_XY[0]} {FIX_XY[1]} 0",
                          quat=" ".join(f"{v:.6f}" for v in yaw_quat(FIX_YAW)))
        hx, hy = FIX_HALF
        add_plate(f, hx, hy)
        zc, zh = PLATE_H + BLOCK_H / 2, BLOCK_H / 2
        cut = {"round": HOLE_HALF + 0.001, "square": HOLE_HALF}  # square cutouts; the ring fills the round one
        # Full-width bars between and beside the cutouts, then the two side bars at each cutout.
        edges = [-hy, HOLE_Y["round"] - cut["round"], HOLE_Y["round"] + cut["round"],
                 HOLE_Y["square"] - cut["square"], HOLE_Y["square"] + cut["square"], hy]
        for i, (a, b) in enumerate(zip(edges[::2], edges[1::2])):
            box(f, f"fix_bar{i}", (0, (a + b) / 2, zc), (hx, (b - a) / 2, zh), "fixture_al")
        for p in PARTS:
            c = cut[p]
            for s in (-1, 1):
                box(f, f"fix_{p}_side{s:+d}", (s * (hx + c) / 2, HOLE_Y[p], zc), ((hx - c) / 2, c, zh),
                    "fixture_al")
        add_ring(f, "fix_ring", (0.0, HOLE_Y["round"]), HOLE_HALF, zc, zh)

    # ------------------------------------------------------------- per seed
    def bind_task(self) -> None:
        super().bind_task()
        m = self.world.model
        self.part_body = {p: self.body(f"part_{p}") for p in PARTS}
        self.part_geom = {p: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"part_{p}_geom") for p in PARTS}
        self.part_mat = {p: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, f"part_{p}") for p in PARTS}
        self.fixture = self.body("fixture")
        self.half_w = dict.fromkeys(PARTS, PART_HALF_W[1])
        self.half_h = dict.fromkeys(PARTS, PART_HALF_H[1])
        self.finish: dict[str, str] = {}

    def reset_task(self, rng: np.random.Generator) -> None:
        m = self.world.model
        # Fixture pose. The upstream reset restores body poses from the XML first.
        m.body_pos[self.fixture, :2] = FIX_XY + rng.uniform(-FIX_JITTER, FIX_JITTER, 2)
        m.body_quat[self.fixture] = yaw_quat(FIX_YAW + rng.uniform(-FIX_YAW_JITTER, FIX_YAW_JITTER))
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
            xy = TRAY_XY + rot(TRAY_YAW) @ local
            self.set_free(f"part_{p}", (xy[0], xy[1], TRAY_FLOOR + hh), yaw=rng.uniform(-np.pi / 4, np.pi / 4))
        self.jitter_look(rng)

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
        return self.robot_touches(self.part_body[p])

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
GAP = 0.0015  # fixed-jaw clearance to the peg on the way down
GRASP_DEPTH = 0.016  # fingertips this far below the peg top
CARRY_CLEAR = 0.014  # peg bottom above the fixture top while carrying
INSERT_LOW = 0.004  # peg bottom above the hole floor when the gripper lets go
ALIGN_TOL = 0.0006  # peg bottom to hole centre (m) before the descent


class Oracle(TopDownOracle):
    """Scripted assembler (privileged state: peg, hole and gripper poses). For each peg: hover, descend,
    close, lift, carry above its hole with the jaws turned so a square peg lines up with its hole,
    centre the peg's bottom over the hole (closed loop, a few mm moves), descend slowly into it,
    release, back off and rise; then home."""

    name = "assembly oracle (privileged state)"

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
            yaw = env.part_yaw(p) + round(wrap(radial - env.part_yaw(p)) / (np.pi / 2)) * np.pi / 2
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
        held = rot(-psi) @ (b[:2] - s[:2])  # peg bottom in the jaw frame
        dz = s[2] - b[2]
        radial = float(np.arctan2(hole[1], hole[0]))
        yaw = radial
        if p == "square":  # turn the jaws so the peg's faces line up with the hole's
            rel = wrap(env.part_yaw(p) - psi, np.pi / 2)
            yaw = fyaw - rel
            yaw += round(wrap(radial - yaw) / (np.pi / 2)) * np.pi / 2
        self.phase = f"carry {p}"
        target = np.r_[hole - rot(yaw) @ held, FIX_TOP + CARRY_CLEAR + dz]
        yield self._joint(self._ik(target, yaw, self.cmd), GRIP_SHUT, settle=6)
        self.phase = f"align {p}"
        for _ in range(4):
            err = hole - env.part_bottom(p)[:2]
            turn = wrap(fyaw - env.part_yaw(p), np.pi / 2) if p == "square" else 0.0
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
