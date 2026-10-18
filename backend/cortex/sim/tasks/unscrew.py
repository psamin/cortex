"""Unscrew (issue #161): unscrew a hex-head bolt from a machined fixture and pull it out of its hole.

Scene, on the shared workcell (``workcell.py``: rubber mat, materials, front camera): a steel base
plate under a brushed-aluminium block, with an M12-style hex bolt threaded into a hole at its centre.

The thread is modelled physically. The bolt hangs off the fixture on a hinge (z) and a slide (z),
and a joint equality makes ``rise = PITCH / (2 pi) * turn``. So turning it counter-clockwise (seen from
above) raises it, and frictionloss keeps it from turning on its own. The head is three overlapping
boxes forming a hexagon, so the jaws clamp two flats and the torque goes through normal forces, not
only friction. Once the bolt has turned through the thread engagement (``TURNS`` turns), the coupling
is switched off (``eq_active``) and the bolt slides out freely. It stays on its axis; if let go, it
drops back into the hole.

Success: unscrewed, pulled out, and held (see ``Env.success``). The ``Oracle`` reads privileged state
(bolt pose and turn) and ratchets with the wrist roll: grasp, turn as far as the roll joint allows,
release, lift, turn back, regrasp. Then it pulls the bolt out.
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
    add_workcell,
    box,
    rot,
    yaw_quat,
)

PITCH = 0.005  # bolt rise per turn (m); coarser than a real M12 (1.75), so 2 turns lift the head a visible 1 cm
TURNS = 2.0  # thread engagement: turns until the thread lets go
HEAD_H = 0.014  # hex head height
HEAD_HALF_AF = (0.0105, 0.0115)  # half the width across flats: 21-23 mm heads
SHANK_R, SHANK_L = 0.006, 0.021  # shank radius and length below the head
HOLE_R = 0.0075  # the fixture hole: 1.5 mm radial clearance around the shank
SEAT_GAP = 0.0005  # head above the fixture top when fully screwed in
# Thread stiffness. MuJoCo regularises a joint equality with both joints' unscaled inverse weights, so a
# light hinge (1/I ~ 2000) makes the coupling a ~100 N/m spring: the bolt sags 5 mm and lags the turn.
# A 0.005 kg m^2 hinge armature and a near-hard solimp keep the coupling error to ~0.01 mm while turning
# and ~0.4 mm under a 25 N pull. Frictionloss holds it still; a straight pull under ~60 N cannot
# back-drive the thread (pitch / 2 pi * F < frictionloss).
TURN_ARMATURE, TURN_DAMPING, TURN_FRICTION = 0.005, 0.05, 0.05
THREAD_SOLIMP = "0.999 0.9999 0.001"

FIX_HALF = (0.035, 0.045)
FIX_XY, FIX_YAW = np.array([0.19, 0.08]), np.deg2rad(20.0)
FIX_JITTER, FIX_YAW_JITTER = 0.015, np.deg2rad(10.0)

OUT_CLEAR = 0.01  # pulled out: the bolt's tip at least this far above the fixture top
MAX_SPEED = 0.05

# name -> (rgb, specular, shininess)
FINISHES = {
    "zinc": ((0.70, 0.72, 0.76), 0.8, 0.7),
    "black oxide": ((0.08, 0.08, 0.09), 0.5, 0.7),
    "brass": ((0.74, 0.58, 0.26), 0.9, 0.8),
}


class Env(WorkcellEnv):
    name = "unscrew"
    instruction = "Unscrew the bolt and pull it out of the fixture."
    max_steps = 1200
    focus = "bolt"

    # ----------------------------------------------------------------- scene
    def build(self, mj: ET.Element, wb: ET.Element) -> None:
        add_workcell(mj, wb)
        asset = mj.find("asset")
        assert asset is not None
        ET.SubElement(asset, "material", name="bolt_finish", rgba="0.7 0.72 0.76 1", specular="0.8", shininess="0.7")
        f = ET.SubElement(wb, "body", name="fixture", pos=f"{FIX_XY[0]} {FIX_XY[1]} 0",
                          quat=" ".join(f"{v:.6f}" for v in yaw_quat(FIX_YAW)))
        hx, hy = FIX_HALF
        add_plate(f, hx, hy)
        zc, zh = PLATE_H + BLOCK_H / 2, BLOCK_H / 2
        c = HOLE_R + 0.001  # square cutout around the hole; a ring of boxes makes it round
        box(f, "fix_bar0", (0, -(hy + c) / 2, zc), (hx, (hy - c) / 2, zh), "fixture_al")
        box(f, "fix_bar1", (0, (hy + c) / 2, zc), (hx, (hy - c) / 2, zh), "fixture_al")
        for s in (-1, 1):
            box(f, f"fix_side{s:+d}", (s * (hx + c) / 2, 0, zc), ((hx - c) / 2, c, zh), "fixture_al")
        add_ring(f, "fix_ring", (0.0, 0.0), HOLE_R, zc, zh)

        # The bolt: origin at the underside of the head, on the hole axis.
        b = ET.SubElement(f, "body", name="bolt", pos=f"0 0 {FIX_TOP + SEAT_GAP}")
        ET.SubElement(b, "joint", name="bolt_turn", type="hinge", axis="0 0 1", range="0 3600",
                      damping=f"{TURN_DAMPING}", frictionloss=f"{TURN_FRICTION}", armature=f"{TURN_ARMATURE}")
        # The scene compiles angles in degrees: the turn's range is 0-10 turns.
        ET.SubElement(b, "joint", name="bolt_rise", type="slide", axis="0 0 1", range="0 0.1", damping="0.2")
        hw = HEAD_HALF_AF[1]
        for k in range(3):  # a hexagon is the union of three rectangles (across-flats x side) 60 deg apart
            box(b, f"bolt_head{k}", (0, 0, HEAD_H / 2), (hw, hw / np.sqrt(3), HEAD_H / 2), "bolt_finish",
                yaw=k * np.pi / 3, mass="0.012", friction="1 0.01 0.001")
        ET.SubElement(b, "geom", name="bolt_shank", type="cylinder", size=f"{SHANK_R} {SHANK_L / 2}",
                      pos=f"0 0 {-SHANK_L / 2}", material="bolt_finish", mass="0.015")
        for i in range(4):  # thread crests, visual only
            ET.SubElement(b, "geom", name=f"bolt_crest{i}", type="cylinder", size=f"{SHANK_R + 0.0006} 0.0006",
                          pos=f"0 0 {-0.003 - i * PITCH:.4f}", material="bolt_finish", contype="0", conaffinity="0",
                          mass="0")
        eq = mj.find("equality")
        if eq is None:
            eq = ET.SubElement(mj, "equality")
        ET.SubElement(eq, "joint", name="thread", joint1="bolt_rise", joint2="bolt_turn",
                      polycoef=f"0 {PITCH / (2 * np.pi):.8f} 0 0 0", solref="0.01 1",
                      solimp=THREAD_SOLIMP)

    # ------------------------------------------------------------- per seed
    def bind_task(self) -> None:
        super().bind_task()
        m = self.world.model
        self.fixture = self.body("fixture")
        self.bolt = self.body("bolt")
        self.turn_q = m.joint("bolt_turn").qposadr[0]
        self.rise_q = m.joint("bolt_rise").qposadr[0]
        self.rise_v = m.joint("bolt_rise").dofadr[0]
        self.thread = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, "thread")
        self.head_geoms = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"bolt_head{k}") for k in range(3)]
        self.finish_mat = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, "bolt_finish")
        self.half_af = HEAD_HALF_AF[1]
        self.finish = ""

    def reset_task(self, rng: np.random.Generator) -> None:
        m = self.world.model
        # The upstream reset restores body poses and re-arms the thread (eq_active) first.
        m.body_pos[self.fixture, :2] = FIX_XY + rng.uniform(-FIX_JITTER, FIX_JITTER, 2)
        m.body_quat[self.fixture] = yaw_quat(FIX_YAW + rng.uniform(-FIX_YAW_JITTER, FIX_YAW_JITTER))
        m.body_quat[self.bolt] = yaw_quat(rng.uniform(0, np.pi / 3))  # head orientation (hex symmetry)
        hw = rng.uniform(*HEAD_HALF_AF)
        self.half_af = hw
        half = np.array([hw, hw / np.sqrt(3), HEAD_H / 2])
        for g in self.head_geoms:
            m.geom_size[g] = half
            m.geom_aabb[g] = np.r_[0, 0, 0, half]
            m.geom_rbound[g] = np.linalg.norm(half)
        names = list(FINISHES)
        self.finish = names[rng.integers(len(names))]
        rgb, spec, shine = FINISHES[self.finish]
        m.mat_rgba[self.finish_mat] = (*rgb, 1)
        m.mat_specular[self.finish_mat] = spec
        m.mat_shininess[self.finish_mat] = shine
        self.jitter_look(rng)

    def step(self, action: np.ndarray, render: bool = True):
        # The thread lets go once the bolt has turned through the engagement; it stays released.
        if self.world.data.eq_active[self.thread] and self.turned() >= 2 * np.pi * TURNS:
            self.world.data.eq_active[self.thread] = 0
        return super().step(action, render)

    # ------------------------------------------------------------ geometry
    def fixture_top(self) -> float:
        return float(self.world.data.xpos[self.fixture][2] + FIX_TOP)

    def bolt_xy(self) -> np.ndarray:
        return self.world.data.xpos[self.bolt][:2].copy()

    def head_top(self) -> float:
        return float(self.world.data.xpos[self.bolt][2] + HEAD_H)

    def tip_z(self) -> float:
        return float(self.world.data.xpos[self.bolt][2] - SHANK_L)

    def head_yaw(self) -> float:
        """Heading of one pair of flats' normal (the others are +-60 deg)."""
        R = self.world.data.xmat[self.bolt].reshape(3, 3)
        return float(np.arctan2(R[1, 0], R[0, 0]))

    def turned(self) -> float:
        """Counter-clockwise turn from fully screwed in (rad)."""
        return float(self.world.data.qpos[self.turn_q])

    def rise(self) -> float:
        return float(self.world.data.qpos[self.rise_q])

    def disengaged(self) -> bool:
        return not self.world.data.eq_active[self.thread]

    def held(self) -> bool:
        return self.robot_touches(self.bolt) and abs(self.world.data.qvel[self.rise_v]) < MAX_SPEED

    def success(self) -> bool:
        """Unscrewed (turned through all ``TURNS`` of engagement, so the thread let go), pulled out
        (the tip >= 1 cm above the fixture top) and held still by the gripper. Off the thread, a bolt
        that is let go drops back into the hole."""
        return self.disengaged() and self.tip_z() >= self.fixture_top() + OUT_CLEAR and self.held()

    def metrics(self) -> dict[str, Any]:
        return {"turns": round(self.turned() / (2 * np.pi), 3), "rise_mm": round(1000 * self.rise(), 1),
                "disengaged": self.disengaged(), "tip_above_fixture_mm": round(1000 * (self.tip_z() - self.fixture_top()), 1),
                "finish": self.finish}


# ---------------------------------------------------------------------------------------------- oracle
GAP = 0.0008  # fixed jaw clearance to a flat on the way down
GRASP_DEPTH = 0.0055  # fingertips this far below the head top (the fixed jaw reaches ~6 mm lower)
ROLL_MARGIN = np.deg2rad(10.0)  # stay this far inside the wrist roll's range
OVERTURN = np.deg2rad(15.0)  # turn this far past the end of the thread
REGRASP_LIFT = 0.02  # jaws clear the head before turning back
PULL_CLEAR = 0.016  # pull until the tip is this far above the fixture top
MAX_GRASPS = 6


class Oracle(TopDownOracle):
    """Ratcheting unscrewer (privileged state: bolt pose, turn and the arm's kinematics). Grasp the head
    across two flats with the wrist rolled to one end of its range; turn counter-clockwise to the other
    end, following the head up the thread; open, lift, turn back, regrasp; repeat until the thread lets
    go, then pull the bolt straight out and hold it."""

    name = "unscrew oracle (privileged state)"

    def _script(self):
        env = self.env
        w = env.world
        self.roll_lo, self.roll_hi = w.lo[4] + ROLL_MARGIN, w.hi[4] - ROLL_MARGIN
        self.grasps = 0
        while not env.disengaged() and self.grasps < MAX_GRASPS:
            yield from self._grasp()
            yield from self._turn()
            if not env.disengaged():
                yield from self._let_go()
        self.phase = "pull"
        s, psi = self._fk(self.cmd)
        up = env.fixture_top() + PULL_CLEAR - env.tip_z()
        yield self._line(np.r_[s[:2], s[2] + up], psi, GRIP_SHUT, 36, settle=4)
        self.phase = "hold"

    def _roll_offset(self, q: np.ndarray) -> float:
        """Jaw yaw minus wrist roll at joints q (rad): it depends on the pan, so it is nearly constant here."""
        return self._fk(q)[1] - np.deg2rad(q[4])

    def _start_yaw(self, q: np.ndarray) -> float:
        """The first jaw yaw at or above the low end of the roll range that closes across two flats."""
        lo = self._roll_offset(q) + self.roll_lo
        h = self.env.head_yaw()
        return h + np.ceil((lo - h) / (np.pi / 3)) * np.pi / 3

    def _clamp_site(self, yaw: float) -> np.ndarray:
        """Site xy with the fixed jaw GAP off a flat, the jaws closing along ``yaw``."""
        env = self.env
        return env.bolt_xy() + (FACE - env.half_af - GAP) * np.array([np.cos(yaw), np.sin(yaw)])

    def _grasp(self):
        env = self.env
        self.grasps += 1
        z = env.head_top() - GRASP_DEPTH
        if self.grasps == 1:
            self.phase = "approach"
            q = self._ik(np.r_[env.bolt_xy(), z + 0.025], env.head_yaw(), self.cmd)
            yaw = self._start_yaw(q)
            yield self._joint(self._ik(np.r_[self._clamp_site(yaw), z + 0.025], yaw, q), GRIP_OPEN, settle=4)
        self.phase = f"grasp {self.grasps}"
        _, yaw = self._fk(self.cmd)
        yield self._line(np.r_[self._clamp_site(yaw), z], yaw, GRIP_OPEN, 22, settle=4)
        yield self._grip(GRIP_SHUT, 14, settle=8)

    def _turn(self):
        """Orbit the commanded site about the bolt axis while the jaws turn, rising with the thread."""
        env = self.env
        self.phase = f"turn {self.grasps}"
        p0, yaw0 = self._fk(self.cmd)
        axis = env.bolt_xy()
        off = rot(-yaw0) @ (p0[:2] - axis)  # commanded site in the jaw frame, kept through the turn
        room = self._roll_offset(self.cmd) + self.roll_hi - yaw0
        left = 2 * np.pi * TURNS + OVERTURN - env.turned()
        dyaw = float(np.clip(min(room, left), np.deg2rad(20.0), None))
        k = max(2, int(np.ceil(np.rad2deg(dyaw) / 6.0)))
        pts = []
        for s in np.linspace(0.0, 1.0, k + 1)[1:]:
            yaw = yaw0 + dyaw * s
            pts.append((np.r_[axis + rot(yaw) @ off, p0[2] + PITCH * dyaw * s / (2 * np.pi)], yaw))
        yield self._path(pts, GRIP_SHUT, int(np.clip(np.rad2deg(dyaw) / 3.5, 20, 90)), settle=4)

    def _let_go(self):
        """Open, lift clear of the head, and turn the jaws back to the next grasp's start yaw."""
        self.phase = f"regrasp {self.grasps}"
        yield self._grip(GRIP_OPEN, 12, settle=4)
        p0, yaw0 = self._fk(self.cmd)
        yield self._line(np.r_[p0[:2], p0[2] + REGRASP_LIFT], yaw0, GRIP_OPEN, 16, settle=2)
        z = self._fk(self.cmd)[0][2]
        yaw1 = self._start_yaw(self.cmd)
        k = max(2, int(np.ceil(np.rad2deg(abs(yaw1 - yaw0)) / 6.0)))
        pts = [(np.r_[self._clamp_site(y), z], y) for y in np.linspace(yaw0, yaw1, k + 1)[1:]]
        yield self._path(pts, GRIP_OPEN, int(np.clip(np.rad2deg(abs(yaw1 - yaw0)) / 5.0, 20, 60)), settle=2)
