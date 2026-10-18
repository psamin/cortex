"""Screwdriver (issue #161): pick a screwdriver from its holder and unscrew a cross-head screw with it.

Scene, on the shared workcell (``workcell.py``): a 12 mm machined plate (steel base, brushed-aluminium
block) with a screw in a hole at its centre, and a tool holder (a block with a hole) where a stubby
screwdriver stands tip down. The SO-101 only reaches straight down to about 8 cm, so the plate is low,
the driver short (59 mm), and every grasp height stays under that.

* Screw: a square head made of boxes, four corner blocks on a base, leaving a cross recess. The driver's
  tip (two crossed blades) turns it through contact with the recess walls. The thread is the same
  physical model as ``unscrew``: hinge + slide coupled by a joint equality (rise = pitch / 2 pi * turn),
  with frictionloss so it holds still. After ``TURNS`` turns the coupling is switched off, and the
  slide's lower limit is raised to where the screw is, so the unthreaded screw sits loose on its
  last crest.
* Screwdriver: a free body with a square handle (the jaws clamp two faces, so it cannot spin in the
  grasp), a shaft and a cross tip. It stands in the holder's hole on the handle's bottom face.

Success: the screw is unthreaded, and the gripper holds the screwdriver still, its tip at least 1 cm
above the screw head (``Env.success``). The ``Oracle`` reads privileged state and keeps the driver at
the end: it picks the driver, carries it over the screw, and lines the tip up with the recess. Then it
ratchets: lower into the recess, turn with the wrist roll as far as the joint allows (following the
screw up), lift out, turn back, re-engage. Last, it lifts the driver clear.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET  # nosec B405 - we build this XML ourselves
from typing import Any

import mujoco
import numpy as np

from .unscrew import THREAD_SOLIMP
from .workcell import (
    FACE,
    GRIP_OPEN,
    GRIP_SHUT,
    PLATE_H,
    TopDownOracle,
    WorkcellEnv,
    add_holed_block,
    add_plate,
    add_workcell,
    box,
    min_jerk,
    rot,
    wrap,
    yaw_quat,
)

# Screw and plate
PITCH = 0.003  # rise per turn (m): 2 turns lift the head 6 mm, which keeps the final lift within reach
TURNS = 2.0  # thread engagement
# The cross tip has a short lever (~4 mm), so the thread turns easily: 0.01 N m of frictionloss holds it.
# A 0.001 kg m^2 armature with the near-hard solimp keeps the coupling tight (see unscrew.py).
TURN_ARMATURE, TURN_DAMPING, TURN_FRICTION = 0.001, 0.002, 0.01
PLATE_TOP = 0.012  # 4 mm steel plate + 8 mm aluminium block
SEAT_GAP = 0.0005
LOOSE_FRICTION = 2.0  # N, an unthreaded screw on its last crest (see Env.step)
HEAD_BASE, HEAD_H = 0.0025, 0.0075  # recess floor height, total head height (recess 5 mm deep)
HEAD_HALF = (0.0065, 0.0075)  # square head half width (13-15 mm heads)
SLOT_W = 0.003  # cross recess width
SHANK_R, SHANK_L = 0.003, 0.008
HOLE_R = 0.004
FIX_HALF = (0.03, 0.04)
FIX_XY, FIX_YAW = np.array([0.19, 0.08]), np.deg2rad(20.0)
FIX_JITTER, FIX_YAW_JITTER = 0.015, np.deg2rad(10.0)

# Screwdriver (body origin: centre of the handle's bottom face, z along the driver)
HANDLE_HALF_AF, HANDLE_L = 0.010, 0.032  # square handle, 20 mm across flats (one box: overlapping boxes
# for a hexagon make the jaw contacts chatter, and the driver creeps out of the grasp as the wrist turns)
SHAFT_R, SHAFT_L = 0.0025, 0.022
BLADE = (0.0055, 0.0011, 0.0025)  # each tip blade's half extents: 11 x 2.2 x 5 mm, nearly the recess span
TIP_Z = -SHAFT_L - 2 * BLADE[2]  # the tip's end, below the handle's bottom face

# Holder
HOLDER_HALF, HOLDER_TOP, HOLDER_HOLE = 0.015, 0.030, 0.0075  # the tip hangs 3 mm above the table
HOLDER_XY, HOLDER_JITTER = np.array([0.18, -0.09]), 0.01

# Contact bits. Everything else is contype = conaffinity = 1. The holder (bit 2) touches the robot, the
# driver's feet, shaft and tip, but not the handle (bit 4), which rests on it only through the feet.
HOLDER_BITS = {"contype": "2", "conaffinity": "3"}
# The handle is rubber-gripped: it outranks the gripper's contact parameters (priority 2) with more friction,
# spin friction and a stiffer contact. With the gripper's own soft contact the 30 g driver creeps down
# through the jaws (~8 mm over a demo) and falls out.
HANDLE_BITS = {"contype": "4", "conaffinity": "1", "priority": "2", "condim": "6", "friction": "1.5 0.02 0.002",
               "solref": "0.01 1", "solimp": "0.95 0.99 0.001"}
TIP_BITS = {"contype": "1", "conaffinity": "3"}

CLEAR = 0.01  # success: the driver's tip at least this far above the screw head
MAX_SPEED = 0.05

FINISHES = {  # screw: name -> (rgb, specular, shininess)
    "zinc": ((0.70, 0.72, 0.76), 0.8, 0.7),
    "black oxide": ((0.08, 0.08, 0.09), 0.5, 0.7),
    "brass": ((0.74, 0.58, 0.26), 0.9, 0.8),
}
HANDLES = {"red": (0.72, 0.10, 0.08), "yellow": (0.90, 0.70, 0.10), "black": (0.09, 0.09, 0.10),
           "blue": (0.10, 0.25, 0.62)}


def _quadrants(head_half: float) -> list[tuple[tuple[float, float, float], tuple[float, float, float]]]:
    """(centre, half size) of the four corner blocks that leave the cross recess."""
    q = (head_half - SLOT_W / 2) / 2
    c = (head_half + SLOT_W / 2) / 2
    z = HEAD_BASE + (HEAD_H - HEAD_BASE) / 2
    return [((sx * c, sy * c, z), (q, q, (HEAD_H - HEAD_BASE) / 2)) for sx in (-1, 1) for sy in (-1, 1)]


class Env(WorkcellEnv):
    name = "screwdriver"
    instruction = "Pick up the screwdriver and unscrew the screw with it."
    max_steps = 1200
    focus = "driver"

    # ----------------------------------------------------------------- scene
    def build(self, mj: ET.Element, wb: ET.Element) -> None:
        add_workcell(mj, wb)
        asset = mj.find("asset")
        assert asset is not None
        for name, rgba, spec in (("screw_finish", "0.7 0.72 0.76 1", "0.8"), ("handle", "0.72 0.1 0.08 1", "0.3"),
                                 ("chrome", "0.78 0.80 0.83 1", "0.9"), ("holder", "0.10 0.11 0.13 1", "0.3")):
            ET.SubElement(asset, "material", name=name, rgba=rgba, specular=spec, shininess="0.6")

        f = ET.SubElement(wb, "body", name="fixture", pos=f"{FIX_XY[0]} {FIX_XY[1]} 0",
                          quat=" ".join(f"{v:.6f}" for v in yaw_quat(FIX_YAW)))
        add_plate(f, *FIX_HALF, top=PLATE_TOP)
        add_holed_block(f, "fix", *FIX_HALF, PLATE_H, PLATE_TOP, HOLE_R)

        s = ET.SubElement(f, "body", name="screw", pos=f"0 0 {PLATE_TOP + SEAT_GAP}")
        # The scene compiles angles in degrees: the turn's range is 0-10 turns.
        ET.SubElement(s, "joint", name="screw_turn", type="hinge", axis="0 0 1", range="0 3600",
                      damping=f"{TURN_DAMPING}", frictionloss=f"{TURN_FRICTION}", armature=f"{TURN_ARMATURE}")
        # MuJoCo regularises a dof's friction and limit by its compile-time inverse weight; on the 10 g screw
        # alone 2 N of frictionloss is a weak damper (0.5 N lifts it 4 cm/s). 1 kg of armature makes it hold.
        ET.SubElement(s, "joint", name="screw_rise", type="slide", axis="0 0 1", range="0 0.1", damping="0.2",
                      armature="1.0")
        hh = HEAD_HALF[1]
        box(s, "screw_base", (0, 0, HEAD_BASE / 2), (hh, hh, HEAD_BASE / 2), "screw_finish", mass="0.004")
        for k, (c, half) in enumerate(_quadrants(hh)):
            box(s, f"screw_quad{k}", c, half, "screw_finish", mass="0.001", friction="0.5 0.005 0.0001")
        ET.SubElement(s, "geom", name="screw_shank", type="cylinder", size=f"{SHANK_R} {SHANK_L / 2}",
                      pos=f"0 0 {-SHANK_L / 2}", material="screw_finish", mass="0.002")
        eq = mj.find("equality")
        if eq is None:
            eq = ET.SubElement(mj, "equality")
        ET.SubElement(eq, "joint", name="thread", joint1="screw_rise", joint2="screw_turn",
                      polycoef=f"0 {PITCH / (2 * np.pi):.8f} 0 0 0", solref="0.01 1", solimp=THREAD_SOLIMP)

        h = ET.SubElement(wb, "body", name="holder", pos=f"{HOLDER_XY[0]} {HOLDER_XY[1]} 0")
        add_holed_block(h, "holder", HOLDER_HALF, HOLDER_HALF, 0.0, HOLDER_TOP, HOLDER_HOLE, material="holder",
                        t=0.005, **HOLDER_BITS)

        d = ET.SubElement(wb, "body", name="driver", pos=f"{HOLDER_XY[0]} {HOLDER_XY[1]} {HOLDER_TOP}")
        ET.SubElement(d, "freejoint", name="driver_free")
        hw = HANDLE_HALF_AF
        box(d, "driver_handle", (0, 0, HANDLE_L / 2), (hw, hw, HANDLE_L / 2), "handle", mass="0.021", **HANDLE_BITS)
        # Four small feet under the handle (hidden inside it) carry the driver on the holder's ring: a flat
        # face on the holder's 20 boxes makes dozens of contacts that chatter, and the driver hops off-centre.
        for k in range(4):
            a = np.pi / 4 + k * np.pi / 2
            ET.SubElement(d, "geom", name=f"driver_foot{k}", type="sphere", size="0.001", material="handle",
                          pos=f"{0.009 * np.cos(a):.5f} {0.009 * np.sin(a):.5f} 0.001", mass="0.0005", **TIP_BITS)
        ET.SubElement(d, "geom", name="driver_ferrule", type="cylinder", size=f"{SHAFT_R + 0.0015} 0.002",
                      pos="0 0 -0.002", material="chrome", contype="0", conaffinity="0", mass="0")
        ET.SubElement(d, "geom", name="driver_shaft", type="cylinder", size=f"{SHAFT_R} {SHAFT_L / 2}",
                      pos=f"0 0 {-SHAFT_L / 2}", material="chrome", mass="0.005", **TIP_BITS)
        for k in range(2):  # cross tip: two blades at right angles
            box(d, f"driver_blade{k}", (0, 0, -SHAFT_L - BLADE[2]), BLADE, "chrome", yaw=k * np.pi / 2,
                mass="0.0005", friction="0.5 0.005 0.0001", **TIP_BITS)
        ET.SubElement(d, "site", name="driver_tip", pos=f"0 0 {TIP_Z}")

    # ------------------------------------------------------------- per seed
    def bind_task(self) -> None:
        super().bind_task()
        m = self.world.model
        self.fixture, self.screw = self.body("fixture"), self.body("screw")
        self.holder, self.driver = self.body("holder"), self.body("driver")
        self.turn_q = m.joint("screw_turn").qposadr[0]
        self.rise_j = m.joint("screw_rise").id
        self.rise_q = m.jnt_qposadr[self.rise_j]
        self.rise_range0 = m.jnt_range[self.rise_j].copy()
        self.rise_dof = m.jnt_dofadr[self.rise_j]
        self.thread = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, "thread")
        self.tip_site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "driver_tip")
        self.base_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "screw_base")
        self.quad_geoms = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"screw_quad{k}") for k in range(4)]
        self.mats = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, n) for n in ("screw_finish", "handle")}
        self.head_half = HEAD_HALF[1]
        self.finish = self.handle_color = ""

    def _set_geom(self, g: int, pos: tuple[float, ...], half: tuple[float, ...]) -> None:
        m = self.world.model
        m.geom_pos[g], m.geom_size[g] = pos, half
        m.geom_aabb[g] = np.r_[0, 0, 0, half]
        m.geom_rbound[g] = np.linalg.norm(half)

    def reset_task(self, rng: np.random.Generator) -> None:
        m = self.world.model
        # The upstream reset restores body poses and re-arms the thread (eq_active); the slide range is ours.
        m.jnt_range[self.rise_j] = self.rise_range0
        m.dof_frictionloss[self.rise_dof] = 0.0
        m.body_pos[self.fixture, :2] = FIX_XY + rng.uniform(-FIX_JITTER, FIX_JITTER, 2)
        m.body_quat[self.fixture] = yaw_quat(FIX_YAW + rng.uniform(-FIX_YAW_JITTER, FIX_YAW_JITTER))
        m.body_quat[self.screw] = yaw_quat(rng.uniform(0, np.pi / 2))  # recess orientation (4-fold)
        hh = rng.uniform(*HEAD_HALF)
        self.head_half = hh
        self._set_geom(self.base_geom, (0, 0, HEAD_BASE / 2), (hh, hh, HEAD_BASE / 2))
        for g, (c, half) in zip(self.quad_geoms, _quadrants(hh)):
            self._set_geom(g, c, half)
        self.finish = list(FINISHES)[rng.integers(len(FINISHES))]
        rgb, spec, shine = FINISHES[self.finish]
        mat = self.mats["screw_finish"]
        m.mat_rgba[mat], m.mat_specular[mat], m.mat_shininess[mat] = (*rgb, 1), spec, shine
        self.handle_color = list(HANDLES)[rng.integers(len(HANDLES))]
        m.mat_rgba[self.mats["handle"]] = (*HANDLES[self.handle_color], 1)
        # The driver's start pose: the holder moves +-1 cm, the driver stands in it at any heading.
        xy = HOLDER_XY + rng.uniform(-HOLDER_JITTER, HOLDER_JITTER, 2)
        m.body_pos[self.holder, :2] = xy
        self.set_free("driver", (xy[0], xy[1], HOLDER_TOP + 0.0002), yaw=rng.uniform(-np.pi, np.pi))
        self.jitter_look(rng)

    def step(self, action: np.ndarray, render: bool = True):
        # The thread lets go after TURNS turns. The screw then sits loose on its last crest: it does not
        # drop, and it rubs there (LOOSE_FRICTION), so the driver's tip lifting off does not carry it along.
        d, m = self.world.data, self.world.model
        if d.eq_active[self.thread] and self.turned() >= 2 * np.pi * TURNS:
            d.eq_active[self.thread] = 0
            m.jnt_range[self.rise_j, 0] = max(0.0, self.rise() - 0.0002)
            m.dof_frictionloss[self.rise_dof] = LOOSE_FRICTION
        return super().step(action, render)

    # ------------------------------------------------------------ geometry
    def screw_xy(self) -> np.ndarray:
        return self.world.data.xpos[self.screw][:2].copy()

    def head_top(self) -> float:
        return float(self.world.data.xpos[self.screw][2] + HEAD_H)

    def recess_yaw(self) -> float:
        R = self.world.data.xmat[self.screw].reshape(3, 3)
        return float(np.arctan2(R[1, 0], R[0, 0]))

    def driver_yaw(self) -> float:
        """Heading of one tip blade (and of a pair of handle faces' normal)."""
        R = self.world.data.xmat[self.driver].reshape(3, 3)
        return float(np.arctan2(R[1, 0], R[0, 0]))

    def tip(self) -> np.ndarray:
        return self.world.data.site_xpos[self.tip_site].copy()

    def turned(self) -> float:
        return float(self.world.data.qpos[self.turn_q])

    def rise(self) -> float:
        return float(self.world.data.qpos[self.rise_q])

    def disengaged(self) -> bool:
        return not self.world.data.eq_active[self.thread]

    def driver_held(self) -> bool:
        d = self.world.data
        a = self.world.model.jnt_dofadr[self.world.model.body_jntadr[self.driver]]
        return self.robot_touches(self.driver) and float(np.linalg.norm(d.qvel[a:a + 3])) < MAX_SPEED

    def success(self) -> bool:
        """Unscrewed: the screw turned through all ``TURNS`` turns of thread, so it is free. And the
        gripper holds the screwdriver still with its tip >= 1 cm above the screw head (lifted clear)."""
        return self.disengaged() and self.driver_held() and self.tip()[2] >= self.head_top() + CLEAR

    def metrics(self) -> dict[str, Any]:
        return {"turns": round(self.turned() / (2 * np.pi), 3), "rise_mm": round(1000 * self.rise(), 1),
                "disengaged": self.disengaged(), "driver_held": self.driver_held(),
                "tip_above_head_mm": round(float(1000 * (self.tip()[2] - self.head_top())), 1),
                "finish": self.finish, "handle": self.handle_color}


# ---------------------------------------------------------------------------------------------- oracle
# Fixed-jaw clearance to a handle flat on the way down. Its tip spheres reach ~0.5 mm past FACE, so the real
# gap is ~1.5 mm; any less and the descending jaw clips the handle and turns it off its flats.
GAP = 0.002
GRIP_Z = 0.010  # fingertips this far above the handle's bottom face (the fixed jaw reaches ~6 mm lower)
CARRY_TIP = 0.035  # tip height while carrying (clears the holder top, plate and screw head)
INSERT = 0.0045  # tip depth in the 5 mm recess while turning
TIP_CLEAR = 0.005  # between turns the tip rises this far above the head before turning back
ROLL_MARGIN = np.deg2rad(10.0)
OVERTURN = np.deg2rad(15.0)
UNWIND = np.deg2rad(4.0)  # back the turn off by the blades' play (+-4 deg) before lifting the driver clear
ALIGN_TOL, ALIGN_YAW_TOL = 0.00025, np.deg2rad(1.0)  # the blades have 0.4 mm of play
MAX_ENGAGE = 6


class Oracle(TopDownOracle):
    """Screwdriver oracle (privileged state: driver, tip, screw and recess poses, turn). It keeps the
    driver at the end."""

    name = "screwdriver oracle (privileged state)"

    def _script(self):
        env = self.env
        w = env.world
        self.roll_lo, self.roll_hi = w.lo[4] + ROLL_MARGIN, w.hi[4] - ROLL_MARGIN
        self.engagements = 0
        yield from self._pick()
        yield from self._carry()
        while not env.disengaged() and self.engagements < MAX_ENGAGE:
            yield from self._engage()
            yield from self._turn()
            if not env.disengaged():
                yield from self._back_off()
        self.phase = "lift clear"
        p0, yaw0 = self._fk(self.cmd)
        axis = env.screw_xy()
        off = rot(-yaw0) @ (p0[:2] - axis)
        unwind = yaw0 - UNWIND  # ease the blades off the recess walls, so the loose screw stays put
        yield self._path([(np.r_[axis + rot(unwind) @ off, p0[2]], unwind)], GRIP_SHUT, 10, settle=3)
        p0, yaw0 = self._fk(self.cmd)
        up = env.head_top() + CLEAR + 0.006 - env.tip()[2]
        yield self._line(np.r_[p0[:2], p0[2] + up], yaw0, GRIP_SHUT, 24, settle=4)
        self.phase = "hold"

    def _roll_offset(self, q: np.ndarray) -> float:
        return self._fk(q)[1] - np.deg2rad(q[4])

    def _pick(self):
        env = self.env
        c = env.world.data.xpos[env.driver].copy()
        radial = float(np.arctan2(c[1], c[0]))
        h = env.driver_yaw()  # handle faces' normals: h + k * 90 deg; close on the pair nearest radial
        yaw = h + np.round(wrap(radial - h) / (np.pi / 2)) * np.pi / 2
        site = c[:2] + (FACE - HANDLE_HALF_AF - GAP) * np.array([np.cos(yaw), np.sin(yaw)])
        z = c[2] + GRIP_Z
        self.phase = "approach driver"
        yield self._joint(self._ik(np.r_[site, z + HANDLE_L], yaw, self.cmd), GRIP_OPEN, settle=4)
        yield self._line(np.r_[site, z], yaw, GRIP_OPEN, 26, settle=4)
        self.phase = "grasp driver"
        yield self._grip(GRIP_SHUT, 14, settle=8)
        self.phase = "lift driver"
        tip = env.tip()
        yield self._line(np.r_[site, z + (HOLDER_TOP + 0.005 - tip[2])], yaw, GRIP_SHUT, 30, settle=3)

    def _held(self) -> tuple[np.ndarray, float, float]:
        """The tip in the jaw frame (xy offset from the site), the site's height above the tip, and the
        tip blades' heading relative to the jaws."""
        env = self.env
        s, psi = self._site()
        tip = env.tip()
        return rot(-psi) @ (tip[:2] - s[:2]), s[2] - tip[2], wrap(env.driver_yaw() - psi, np.pi / 2)

    def _start_yaw(self, q: np.ndarray, rel: float) -> float:
        """The lowest jaw yaw within the roll range whose tip blades line up with the recess."""
        lo = self._roll_offset(q) + self.roll_lo
        want = self.env.recess_yaw() - rel  # jaw yaw that lines the blades up, modulo 90 deg
        return want + np.ceil((lo - want) / (np.pi / 2)) * np.pi / 2

    def _carry(self):
        env = self.env
        off, dz, rel = self._held()
        axis = env.screw_xy()
        q = self._ik(np.r_[axis, CARRY_TIP + dz], 0.0, self.cmd)
        yaw = self._start_yaw(q, rel)
        self.phase = "carry"
        yield self._joint(self._ik(np.r_[axis - rot(yaw) @ off, CARRY_TIP + dz], yaw, q), GRIP_SHUT, settle=6)

    def _engage(self):
        """Centre the tip over the recess and turn its blades into line (closed loop), then lower it in."""
        env = self.env
        self.engagements += 1
        self.phase = f"align {self.engagements}"
        for _ in range(4):
            err = env.screw_xy() - env.tip()[:2]
            turn = wrap(env.recess_yaw() - env.driver_yaw(), np.pi / 2)
            if np.linalg.norm(err) < ALIGN_TOL and abs(turn) < ALIGN_YAW_TOL:
                break
            p0, psi0 = self._fk(self.cmd)
            yield self._line(np.r_[p0[:2] + err, p0[2]], psi0 + turn, GRIP_SHUT, 8, settle=4)
        self.phase = f"insert {self.engagements}"
        p0, psi0 = self._fk(self.cmd)
        dz = p0[2] - env.tip()[2]  # commanded site above the measured tip
        yield self._line(np.r_[p0[:2], env.head_top() - INSERT + dz], psi0, GRIP_SHUT, 20, settle=4)

    def _turn(self):
        """Orbit the commanded site about the screw axis while the jaws turn. The height is re-planned every
        step from the measured head, so the tip stays at its depth in the recess whether the screw keeps up
        with the turn or not (an open-loop rise would lift the tip out when the screw lags)."""
        env = self.env
        self.phase = f"turn {self.engagements}"
        p0, yaw0 = self._fk(self.cmd)
        axis = env.screw_xy()
        off = rot(-yaw0) @ (p0[:2] - axis)
        depth = p0[2] - env.head_top()  # commanded site above the head top, kept through the turn
        room = self._roll_offset(self.cmd) + self.roll_hi - yaw0
        left = 2 * np.pi * TURNS + OVERTURN - env.turned()
        dyaw = float(np.clip(min(room, left), np.deg2rad(20.0), None))
        for s in min_jerk(int(np.clip(np.rad2deg(dyaw) / 2.5, 20, 120))):
            yaw = yaw0 + dyaw * s
            q = self._ik(np.r_[axis + rot(yaw) @ off, env.head_top() + depth], yaw, self.cmd)
            yield [np.r_[q, GRIP_SHUT]]
        yield [self.cmd.copy()] * 4

    def _back_off(self):
        """Lift the tip out of the recess and turn back to the next start, orbiting the screw axis."""
        env = self.env
        self.phase = f"back {self.engagements}"
        p0, yaw0 = self._fk(self.cmd)
        up = env.head_top() + TIP_CLEAR - env.tip()[2]  # measured: the tip clears the head before turning back
        yield self._line(np.r_[p0[:2], p0[2] + up], yaw0, GRIP_SHUT, 14, settle=3)
        p0, yaw0 = self._fk(self.cmd)
        axis = env.screw_xy()
        off = rot(-yaw0) @ (p0[:2] - axis)
        yaw1 = self._start_yaw(self.cmd, self._held()[2])
        k = max(2, int(np.ceil(np.rad2deg(abs(yaw1 - yaw0)) / 6.0)))
        pts = [(np.r_[axis + rot(y) @ off, p0[2]], y) for y in np.linspace(yaw0, yaw1, k + 1)[1:]]
        yield self._path(pts, GRIP_SHUT, int(np.clip(np.rad2deg(abs(yaw1 - yaw0)) / 5.0, 20, 60)), settle=2)
