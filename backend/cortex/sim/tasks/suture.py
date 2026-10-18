"""Suture, as a rigid proxy (issue #161): pick up the needle and pass it through both eyelets.

MuJoCo cannot simulate tissue being pierced or a thread being pulled through it reliably, so this
task is a rigid stand-in, and nothing here claims otherwise. The "skin" is a static, skin-toned
slab with a painted incision line. Next to the incision stand two rigid eyelets (rings built from
capsule segments; MuJoCo has no torus), one on each side, sharing one axis across the incision.
The needle is a rigid, steel-looking rod (4 mm, 7 cm, slightly curved point) parked on a foam needle
rest. Passing it through both eyelets in one push stands for one bite of a running stitch: in on the
near side of the incision, out on the far side. There is no thread.

Physics compromises, stated plainly:
  * the rod is 4 mm thick and weighs ~37 g, far more than a real needle. The SO-101's closed jaws
    stop at a ~2.9 mm gap, and MuJoCo's soft contacts push back in proportion to the lighter body's
    mass, so a needle-weight rod is only pinched by contact softness and creeps out of the jaws;
  * the needle's contacts are stiffer than the defaults and have torsional friction (like a
    needle holder's grip), and the eyelets have rolling friction so a released needle settles.

Success (``Env.success``) is strict and measured from the needle's pose:
  * the needle's centreline crosses each eyelet's plane inside the eyelet's opening
    (closer than ``RING_IN`` to the ring centre), with its tail on the near side of the near
    eyelet and its tip at least ``TIP_PAST`` beyond the far eyelet, in the pass direction;
  * and the needle is still (released, or held without moving).

Per-seed variation: needle start position (+-2 cm) and yaw, pad position and yaw, the eyelets'
position along the incision, skin tone, needle tint, table tint, lighting and a small front-camera
pose jitter.

``Oracle`` reads privileged sim state (needle, eyelet and gripper poses). It is a scripted
demonstrator, never a learned policy.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET  # nosec B405 - our own scene XML
from collections.abc import Iterator
from typing import Any

import mujoco
import numpy as np

from ..expert import GRIP_CLOSED, MAX_DEG_PER_STEP
from ..grasp import ik_seeds
from ..pick_place_skill import PickPlaceSkill, Target
from .base import TaskEnv

# Needle (its body frame: x along the shaft, tail to point; the point curves toward +y).
NEEDLE_R = 0.002  # 4 mm rod: the closed SO-101 jaws leave a ~2.9 mm gap, so thinner rods are barely pinched
NEEDLE_PTS = np.array([  # centreline polyline: tail, end of the straight shaft, bend, point
    [-0.035, 0.0, 0.0],
    [0.018, 0.0, 0.0],
    [0.018 + 0.013 * np.cos(np.deg2rad(10)), 0.013 * np.sin(np.deg2rad(10)), 0.0],
    [0.018 + 0.013 * np.cos(np.deg2rad(10)) + 0.007 * np.cos(np.deg2rad(16)),
     0.013 * np.sin(np.deg2rad(10)) + 0.007 * np.sin(np.deg2rad(16)), 0.0],
])
GRASP_X = -0.027  # the oracle holds the needle 8 mm from its tail
NEEDLE_START = (0.20, 0.10)  # nominal needle origin; it rests on the foam block
NEEDLE_JITTER = 0.02
NEEDLE_YAW = 0.4  # rad, around pointing away from the arm
# The needle's contact parameters win over the gripper pads' (priority 1): stiffer than the defaults, so
# the jaws' servo actually squeezes it, and torsional friction, so it does not pivot (pitch) in the jaws.
NEEDLE_CONTACT = dict(priority="2", condim="6", friction="1.5 0.05 0.001", solref="0.01 1",
                      solimp="0.95 0.99 0.001")
NEEDLE_DENSITY = 40000  # ~37 g: see the module docstring
REST_H = 0.015  # foam needle rest height
REST_OFFSET = (0.007, 0.001)  # rest centre in the needle frame (the tail overhangs, for the jaws)
REST_HALF = (0.024, 0.011)

# Practice pad: a slab with the incision across it (along its y axis).
PAD_POS = (0.275, -0.04)
PAD_JITTER = 0.015
PAD_YAW = 0.14  # rad
PAD_HALF = (0.05, 0.075, 0.005)
PAD_TOP = 2 * PAD_HALF[2]
# Eyelets: two rings on one axis (the pad's x), one each side of the incision.
RING_GAP = 0.02  # between the two ring planes
RING_H = 0.022  # ring centre above the pad top
RING_IN = 0.011  # opening radius (2.2 cm inner diameter, at the polygon's narrowest)
RING_TUBE = 0.0015
RING_SEGS = 16
RING_C = (RING_IN + RING_TUBE) / np.cos(np.pi / RING_SEGS)  # centreline radius of the 16-gon
EYELET_SLIDE = 0.02  # eyelet position along the incision, +-

TIP_PAST = 0.005  # the point must be this far beyond the far eyelet's plane
STILL = 0.01  # m/s: the needle's tail and point both slower than this

SKIN_TONES = np.array([[0.96, 0.80, 0.70], [0.88, 0.68, 0.55], [0.74, 0.52, 0.39], [0.55, 0.37, 0.27]])


def _yaw_quat(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def _rot_z(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _f(v) -> str:
    return " ".join(f"{x:.5f}" for x in v)


class Env(TaskEnv):
    name = "suture"
    instruction = "Pick up the surgical needle and pass it through both eyelets across the incision."
    max_steps = 900
    focus = "needle"

    # ------------------------------------------------------------------ scene
    def build(self, mj: ET.Element, wb: ET.Element) -> None:
        asset = mj.find("asset")
        assert asset is not None
        ET.SubElement(asset, "material", name="skin", rgba="0.88 0.68 0.55 1", specular="0.02", shininess="0.05")
        ET.SubElement(asset, "material", name="incision", rgba="0.42 0.05 0.07 1", specular="0.1")
        ET.SubElement(asset, "material", name="wound_edge", rgba="0.78 0.38 0.38 1", specular="0.02")
        ET.SubElement(asset, "material", name="eyelet", rgba="0.12 0.32 0.78 1", specular="0.5", shininess="0.6")
        ET.SubElement(asset, "material", name="steel", rgba="0.78 0.80 0.83 1", specular="1", shininess="0.95")
        ET.SubElement(asset, "material", name="foam", rgba="0.20 0.24 0.32 1", specular="0.05", shininess="0.1")

        pad = ET.SubElement(wb, "body", name="pad", pos=f"{PAD_POS[0]} {PAD_POS[1]} 0")
        ET.SubElement(pad, "geom", name="pad", type="box", size=_f(PAD_HALF), pos=f"0 0 {PAD_HALF[2]}",
                      material="skin")
        deco = dict(contype="0", conaffinity="0")
        ET.SubElement(pad, "geom", name="wound_edge", type="box", size="0.0028 0.062 0.0001",
                      pos=f"0 0 {PAD_TOP + 0.0001}", material="wound_edge", **deco)
        ET.SubElement(pad, "geom", name="incision", type="box", size="0.0009 0.06 0.0002",
                      pos=f"0 0 {PAD_TOP + 0.0002}", material="incision", **deco)
        eyelets = ET.SubElement(pad, "body", name="eyelets", pos=f"0 0 {PAD_TOP}")
        # Low sliding friction against the needle, rolling friction so a needle resting across the rings
        # settles, and as stiff as the needle's contact so it cannot tunnel through.
        ring = dict(material="eyelet", priority="3", condim="6", friction="0.3 0.005 0.002", solref="0.01 1",
                    solimp="0.95 0.99 0.001")
        for k, x in enumerate((-RING_GAP / 2, RING_GAP / 2)):
            ET.SubElement(eyelets, "geom", name=f"eyelet{k}_base", type="cylinder", size="0.004 0.0006",
                          pos=f"{x} 0 0.0006", **ring)
            ET.SubElement(eyelets, "geom", name=f"eyelet{k}_post", type="capsule", size=f"{RING_TUBE}",
                          fromto=_f((x, 0, 0.001, x, 0, RING_H - RING_C)), **ring)
            th = np.linspace(0, 2 * np.pi, RING_SEGS + 1)
            for j in range(RING_SEGS):
                a = (x, RING_C * np.cos(th[j]), RING_H + RING_C * np.sin(th[j]))
                b = (x, RING_C * np.cos(th[j + 1]), RING_H + RING_C * np.sin(th[j + 1]))
                ET.SubElement(eyelets, "geom", name=f"eyelet{k}_seg{j}", type="capsule", size=f"{RING_TUBE}",
                              fromto=_f((*a, *b)), **ring)

        rest = ET.SubElement(wb, "body", name="needle_rest", pos=f"{NEEDLE_START[0]} {NEEDLE_START[1]} 0")
        ET.SubElement(rest, "geom", name="needle_rest", type="box",
                      size=f"{REST_HALF[0]} {REST_HALF[1]} {REST_H / 2}", pos=f"0 0 {REST_H / 2}", material="foam")

        needle = ET.SubElement(wb, "body", name="needle",
                               pos=f"{NEEDLE_START[0]} {NEEDLE_START[1]} {REST_H + NEEDLE_R}")
        ET.SubElement(needle, "freejoint", name="needle")
        steel = dict(type="capsule", material="steel", density=f"{NEEDLE_DENSITY}", **NEEDLE_CONTACT)
        p = NEEDLE_PTS
        ET.SubElement(needle, "geom", name="needle_shaft", size=f"{NEEDLE_R}", fromto=_f((*p[0], *p[1])), **steel)
        ET.SubElement(needle, "geom", name="needle_bend", size=f"{NEEDLE_R}", fromto=_f((*p[1], *p[2])), **steel)
        ET.SubElement(needle, "geom", name="needle_point", size="0.0009", fromto=_f((*p[2], *p[3])), **steel)

        # Closer than the upstream view, so the eyelets and the needle are visible at 96x128.
        cam = wb.find("camera[@name='front']")
        assert cam is not None
        cam.attrib.update(pos="0.42 0.02 0.23", xyaxes="0 1 0 -0.656 0 0.755", fovy="55")

    def bind_task(self) -> None:
        m = self.world.model
        self.pad_id, self.eyelets_id = self.body("pad"), self.body("eyelets")
        self.rest_id, self.needle_id = self.body("needle_rest"), self.body("needle")
        self.needle_dof = int(m.jnt_dofadr[m.body_jntadr[self.needle_id]])
        mat = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, n) for n in ("skin", "steel", "table")}
        self.mat = mat
        self.mat_rgba0 = {k: m.mat_rgba[i].copy() for k, i in mat.items()}
        self.front_cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "front")
        self.front_quat0 = m.cam_quat[self.front_cam].copy()

    def reset_task(self, rng: np.random.Generator) -> None:
        m = self.world.model
        # Pad and eyelets (static bodies: move the model, the upstream reset restores it each episode).
        pxy = np.asarray(PAD_POS) + rng.uniform(-PAD_JITTER, PAD_JITTER, 2)
        m.body_pos[self.pad_id] = (*pxy, 0.0)
        m.body_quat[self.pad_id] = _yaw_quat(rng.uniform(-PAD_YAW, PAD_YAW))
        m.body_pos[self.eyelets_id] = (0.0, rng.uniform(-EYELET_SLIDE, EYELET_SLIDE), PAD_TOP)
        # Needle on its rest.
        nxy = np.asarray(NEEDLE_START) + rng.uniform(-NEEDLE_JITTER, NEEDLE_JITTER, 2)
        yaw = rng.uniform(-NEEDLE_YAW, NEEDLE_YAW)
        m.body_pos[self.rest_id] = (*(nxy + _rot_z(yaw)[:2, :2] @ REST_OFFSET), 0.0)
        m.body_quat[self.rest_id] = _yaw_quat(yaw)
        self.set_free("needle", (*nxy, REST_H + NEEDLE_R), yaw)
        # Appearance.
        tone = SKIN_TONES[rng.integers(len(SKIN_TONES))] * rng.uniform(0.95, 1.05, 3)
        m.mat_rgba[self.mat["skin"], :3] = np.clip(tone, 0, 1)
        m.mat_rgba[self.mat["steel"], :3] = np.clip(self.mat_rgba0["steel"][:3] * rng.uniform(0.85, 1.05)
                                                    + rng.uniform(-0.04, 0.04, 3), 0, 1)
        m.mat_rgba[self.mat["table"], :3] = np.clip(self.mat_rgba0["table"][:3] * rng.uniform(0.8, 1.05)
                                                    + rng.uniform(-0.03, 0.03, 3), 0, 1)
        # Front camera: the upstream reset jitters its position; add a small rotation (about 1.5 deg).
        axis = rng.normal(size=3)
        ang = np.deg2rad(rng.uniform(0, 1.5))
        dq = np.r_[np.cos(ang / 2), np.sin(ang / 2) * axis / np.linalg.norm(axis)]
        q = np.zeros(4)
        mujoco.mju_mulQuat(q, dq, self.front_quat0)
        m.cam_quat[self.front_cam] = q

    # ---------------------------------------------------------------- state
    def needle_points(self) -> np.ndarray:
        """The needle's centreline (tail, shaft end, bend, point) in world coordinates."""
        d = self.world.data
        return d.xpos[self.needle_id] + NEEDLE_PTS @ d.xmat[self.needle_id].reshape(3, 3).T

    def eyelets(self) -> tuple[np.ndarray, np.ndarray]:
        """Ring centres (near, far) and the shared pass axis (unit, from the near ring to the far one)."""
        d = self.world.data
        R = d.xmat[self.eyelets_id].reshape(3, 3)
        c = d.xpos[self.eyelets_id] + np.array([[-RING_GAP / 2, 0, RING_H], [RING_GAP / 2, 0, RING_H]]) @ R.T
        return c, R[:, 0].copy()

    def ring_crossings(self) -> list[float]:
        """Per eyelet: the radial distance (m) at which the needle's centreline crosses the ring plane
        going tail to point, or inf when it does not cross from the near side to the far side."""
        pts = self.needle_points()
        centres, axis = self.eyelets()
        out = []
        for c in centres:
            s = (pts - c) @ axis
            r = np.inf
            for i in range(len(pts) - 1):
                if s[i] <= 0 < s[i + 1]:
                    x = pts[i] + (pts[i + 1] - pts[i]) * (-s[i] / (s[i + 1] - s[i]))
                    r = float(np.linalg.norm(x - c))
                    break
            out.append(r)
        return out

    def tip_past(self) -> float:
        """Signed distance (m) of the needle point beyond the far eyelet's plane."""
        centres, axis = self.eyelets()
        return float((self.needle_points()[-1] - centres[1]) @ axis)

    def threaded(self) -> bool:
        """Through both eyelets' openings, tail on the near side, point at least TIP_PAST past the far one."""
        centres, axis = self.eyelets()
        tail_s = float((self.needle_points()[0] - centres[0]) @ axis)
        return bool(all(r < RING_IN for r in self.ring_crossings()) and tail_s < 0
                    and self.tip_past() >= TIP_PAST)

    def still(self) -> bool:
        """The needle's centreline is at rest: its tail and point move slower than STILL. (Spinning about
        its own axis does not count; a round rod lying across two rings can roll a little in place.)"""
        d = self.world.data
        v = d.qvel[self.needle_dof:self.needle_dof + 6]
        w = d.xmat[self.needle_id].reshape(3, 3) @ v[3:]  # free joint: angular velocity is in the body frame
        ends = self.needle_points()[[0, -1]] - d.xpos[self.needle_id]
        return bool(np.all(np.linalg.norm(v[:3] + np.cross(w, ends), axis=1) < STILL))

    def success(self) -> bool:
        return self.threaded() and self.still()

    def metrics(self) -> dict[str, Any]:
        return {"eyelets_threaded": int(sum(r < RING_IN for r in self.ring_crossings())),
                "tip_past_far_eyelet_m": self.tip_past(), "needle_max_z": self.max_cube_z}


# ---------------------------------------------------------------------- oracle
# Needle centre in the gripper-site frame when grasping: 1.5 mm off the fixed jaw's pad, at the site's
# depth (mid-pad; 1 mm lower already loses the rod out of the jaw tips).
GRASP_Z = -0.0193 + NEEDLE_R + 0.0015
GRIP_APPROACH = 35.0  # opening while descending onto the needle
HOVER, LIFT = 0.04, 0.045
PRE_INSERT = 0.018  # the point starts this far before the near eyelet
TIP_FINAL = 0.028  # and ends this far beyond the far eyelet (centre of mass between the eyelets)
AIM_Y = 0.5 * NEEDLE_PTS[-1, 1]  # aim the middle of the curved point's sideways offset at the axis
PASS_RATE = 1.5  # deg/step per joint during the push (a slower rate lets the needle creep in the jaws)
RELEASE_GRIP = 40.0
RELEASE_STEPS = (1.0, 2.0, 3.0, 4.0, 6.0, 9.0, 13.0, 20.0, 30.0)
RETREAT = 0.025  # straight up after releasing; higher is past the wrist-flex limit this far out


def pose_ik(env: Env, pos: np.ndarray, rot: np.ndarray, seed_deg: np.ndarray, iters: int = 200,
            w_rot: float = 0.3) -> tuple[np.ndarray, float, float]:
    """Joint degrees (5 arm joints) putting the gripper site at ``pos`` with orientation ``rot``.

    ``expert.solve_ik`` fixes only the approach axis and sets the wrist roll afterwards, which moves
    the site (it is 12 mm off the roll axis). A needle needs its yaw and position together, so this
    damped-least-squares solve uses the full orientation. Returns (q, position error m, angle error deg).
    """
    w = env.world
    m = w.model
    d = mujoco.MjData(m)
    d.qpos[:] = w.data.qpos
    qadr = w.qadr[:5]
    dofs = m.jnt_dofadr[[m.joint(n).id for n in env.module.JOINTS[:5]]]
    lo, hi = w.lo[:5], w.hi[:5]
    d.qpos[qadr] = np.clip(np.deg2rad(seed_deg[:5]), lo, hi)
    jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    for _ in range(iters):
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        R = d.site_xmat[env.site].reshape(3, 3)
        e_p = pos - d.site_xpos[env.site]
        e_o = 0.5 * sum(np.cross(R[:, i], rot[:, i]) for i in range(3))
        if np.linalg.norm(e_p) < 2e-4 and np.linalg.norm(e_o) < 2e-3:
            break
        mujoco.mj_jacSite(m, d, jp, jr, env.site)
        J = np.vstack([jp[:, dofs], w_rot * jr[:, dofs]])
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-5 * np.eye(6), np.r_[e_p, w_rot * e_o])
        d.qpos[qadr] = np.clip(d.qpos[qadr] + np.clip(dq, -0.2, 0.2), lo, hi)
    mujoco.mj_kinematics(m, d)
    R = d.site_xmat[env.site].reshape(3, 3)
    ang = np.rad2deg(np.arccos(np.clip((np.trace(R.T @ rot) - 1) / 2, -1, 1)))
    return np.rad2deg(d.qpos[qadr]).copy(), float(np.linalg.norm(pos - d.site_xpos[env.site])), float(ang)


def holding_rot(u: np.ndarray) -> np.ndarray:
    """Site orientation for holding a needle whose point is along horizontal ``u``: approach straight
    down, the needle along the site's -y, so the wrist camera (site +y) looks along it from behind."""
    u = np.array([u[0], u[1], 0.0]) / np.linalg.norm(u[:2])
    x, y = np.array([0.0, 0.0, -1.0]), -u
    return np.stack([x, y, np.cross(x, y)], axis=1)


class SutureSkill(PickPlaceSkill):
    """Grasp the needle near its tail, carry it to the eyelets' axis, push it through both, release,
    go home. Reuses PickPlaceSkill's rate-limited, closed-loop target follower (``act``)."""

    def plan(self, state_deg: np.ndarray) -> None:
        super().plan(state_deg)
        self.moves = self._suture()

    def _pose(self, pos: np.ndarray, rot: np.ndarray, phase: str, seed: np.ndarray | None = None
              ) -> np.ndarray | None:
        seeds = [self._measured() if seed is None else seed]
        best = None
        for s in seeds + [np.r_[q[:4], r, 0] for q in ik_seeds(pos, self.home)[:1] for r in (-120, -60, 0, 60, 120)]:
            q, e, a = pose_ik(self.env, pos, rot, s)
            if best is None or e + 1e-3 * a < best[1] + 1e-3 * best[2]:
                best = (q, e, a)
            if e < 1e-3 and a < 2.0:
                return q
        if best is not None and best[1] < 3e-3 and best[2] < 5.0:
            return best[0]
        self.failed = f"no IK for the {phase} pose ({best[1] * 1000:.1f} mm, {best[2]:.1f} deg)"
        return None

    def _pose_line(self, a: np.ndarray, b: np.ndarray, rot: np.ndarray, grip: float, n: int, phase: str,
                   rate: float = 2.0) -> Iterator[Target]:
        for t in np.linspace(0, 1, n + 1)[1:]:
            q = self._pose(a + (b - a) * t, rot, phase)
            if q is None:
                return
            yield Target(q, grip, rate=rate, settle=1, phase=phase)

    def _suture(self) -> Iterator[Target]:
        env: Env = self.env
        d = env.world.data
        Rn = d.xmat[env.needle_id].reshape(3, 3)
        grasp = d.xpos[env.needle_id] + Rn @ np.array([GRASP_X, 0.0, 0.0])
        rot = holding_rot(Rn[:, 0])
        site = grasp - rot @ np.array([0.0, 0.0, GRASP_Z])
        up = np.array([0.0, 0.0, HOVER])
        q = self._pose(site + up, rot, "hover")
        if q is None:
            return
        yield Target(q, GRIP_APPROACH, settle=6, phase="hover")
        yield from self._pose_line(site + up, site, rot, GRIP_APPROACH, 6, "approach")
        q = self._pose(site, rot, "close")  # hold the exact grasp pose, not the 0.7-degree arrival tolerance
        if q is None:
            return
        yield Target(q, GRIP_CLOSED, rate=1.0, settle=12, phase="close")
        yield from self._pose_line(site, site + [0, 0, LIFT], rot, GRIP_CLOSED, 6, "lift")

        # Carry the needle above the eyelets' axis, lower it onto the axis, and push it through. Each
        # phase re-measures how the needle is actually held when it starts.
        travel = TIP_FINAL + RING_GAP + PRE_INSERT
        yield from self._insert_moves([(-travel, LIFT)], "carry", rate=MAX_DEG_PER_STEP, settle=6)
        yield from self._insert_moves([(-travel, LIFT * (1 - t)) for t in np.linspace(0, 1, 7)[1:]], "lower")
        yield from self._insert_moves([(-travel * (1 - t), 0.0) for t in np.linspace(0, 1, 31)[1:]], "pass",
                                      rate=PASS_RATE)
        # Open gradually: letting go of the squeeze at once kicks the needle down through an eyelet.
        q = self.cur.q if self.cur is not None else self._measured()[:5]
        for g in RELEASE_STEPS:
            yield Target(q, g, settle=2, phase="release")
        yield Target(q, RELEASE_GRIP, settle=8, phase="release")
        here = d.site_xpos[env.site].copy()
        rot = holding_rot(-d.site_xmat[env.site].reshape(3, 3)[:, 1])
        yield from self._pose_line(here, here + [0, 0, RETREAT], rot, RELEASE_GRIP, 4, "retreat")
        yield from self._go_home()

    def _insert_moves(self, path: list[tuple[float, float]], phase: str, rate: float = 2.0,
                      settle: int = 1) -> Iterator[Target]:
        """Site targets that put the needle's aim point at (along the eyelets' axis from their midpoint,
        up above it), the shaft yawed onto the axis. The grasp is measured once, when the phase starts:
        re-measuring at every waypoint chases the needle when it brushes an eyelet."""
        env: Env = self.env
        d = env.world.data
        Rs = d.site_xmat[env.site].reshape(3, 3)
        R_ns = Rs.T @ d.xmat[env.needle_id].reshape(3, 3)
        p_ns = Rs.T @ (d.xpos[env.needle_id] - d.site_xpos[env.site])
        centres, axis = env.eyelets()
        base = holding_rot(np.array([1.0, 0.0]))
        shaft = base @ R_ns[:, 0]
        rot = _rot_z(np.arctan2(axis[1], axis[0]) - np.arctan2(shaft[1], shaft[0])) @ base
        # The aim point: the needle point that sits at the eyelets' midpoint at the end of the pass.
        aim = rot @ (R_ns @ np.array([NEEDLE_PTS[-1, 0] - (TIP_FINAL + RING_GAP / 2), AIM_Y, 0.0]) + p_ns)
        for along, up in path:
            q = self._pose(centres.mean(axis=0) + axis * along + np.array([0.0, 0.0, up]) - aim, rot, phase)
            if q is None:
                return
            yield Target(q, GRIP_CLOSED, rate=rate, settle=settle, phase=phase)


class Oracle:
    """Scripted suture demonstrator (privileged sim state). Protocol: see tasks/base.py."""

    name = "suture oracle (privileged state)"
    needs_pixels = False

    def reset(self, env: Env, obs: dict[str, Any], seed: int) -> None:
        self.ex = SutureSkill(env)
        self.ex.plan(np.asarray(obs["agent_pos"], dtype=np.float64))

    def act(self, obs: dict[str, Any]) -> np.ndarray:
        return self.ex.act()

    @property
    def done(self) -> bool:
        return self.ex.done
