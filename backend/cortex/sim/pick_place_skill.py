"""A scripted pick-and-place skill for any intake object: home -> pick -> place -> home.

Generalizes ``ScriptedExpert`` (cube, top grasp) to the objects and zones of an intake scene.
It reads privileged simulator state (object pose), so it is an *oracle* demonstrator and is
labelled as such wherever it is used.

* Short objects (<= 7 cm): top grasp. Hover, descend straight down, close, lift, carry at carry
  height, lower, open, retreat.
* Tall objects: side grasp, the sequence verified on the small bottle (#43/#50):
  1. a raised waypoint above and behind the object;
  2. the pre-grasp, 6 cm back along the approach;
  3. an approach with the object centred in the *open* jaws (clear of the fixed pad);
  4. a slide toward the fixed pad until it is 1 mm away;
  5. close, lift, carry, lower, open, back off.
  The jaw roll is deterministic (``expert._horizontal_jaw_roll``), so grasp offsets computed from one IK
  solution hold for the others.
* Objects riding a conveyor (issue #255): intercept. Predict the object's position from the belt
  speed and pick the earliest point inside reach that the arm can get to first. Hover there with the
  jaws across the belt, lower them so the object rides in between, close as its centre arrives, then
  lift and place as for a top grasp.
* Every episode ends by returning to the home pose ("go home").

Closed loop like the verified probe: each move is solved by IK when it starts (seeded from the
arm's measured joints) and advances only when the measured joints have arrived, with the same
rate limits as ``ScriptedExpert`` (4 deg/step arm, 8/step gripper; 2 deg/step on contact moves).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass

import mujoco
import numpy as np

from .expert import GRIP_CLOSED, GRIP_OPEN, MAX_DEG_PER_STEP, closing_direction, solve_ik
from .so101 import FPS
from .grasp import approaches, grasp_height, grasp_mode, ik_seeds

LIFT = 0.06
GRIP_HOME = 30.0
# Site frame for side grasps (x = approach, z = jaw closing), measured on the Menagerie SO-101.
OPEN_GAP_OFFSET = np.array([-0.020, 0.0, 0.017])  # object centre in the middle of the open jaws
SLIDE_DEPTH = -0.020
# A horizontal side grasp at the edge of the arc solves to just over 10 degrees; past this the
# jaws are no longer approaching the bottle.
MAX_APPROACH_ERR_DEG = 12.0
# Conveyor intercept (issue #255).
HOVER_DZ = 0.043  # hover this far above the grasp height
ENTRY_CLEAR = 0.04  # the object's centre is at least this far upstream when the jaws are down
END_MARGIN = 0.05  # intercept at least this far before the belt's end
INTERCEPT_STEP = 0.01  # candidate intercept points along the belt
CLOSE_LEAD_STEPS = 4  # control steps before the object's centre reaches the jaws that the close starts
MOVE_MARGIN = 30  # steps allowed for the rate-limited approach on top of the transit estimate


def pad_z(depth_x: float) -> float:
    """Height of the fixed jaw's inner face (site frame) at a depth along the approach; it slopes."""
    return float(np.interp(-depth_x, [0.0, 0.008, 0.025, 0.045], [-0.0193, -0.0218, -0.0235, -0.0253]))


@dataclass
class Target:
    q: np.ndarray  # 5 arm joints, degrees
    grip: float
    rate: float = MAX_DEG_PER_STEP
    settle: int = 4  # steps within tolerance before the next target
    phase: str = ""
    until: Callable[[], bool] | None = None  # also hold the target until this is true (a timed close)


class PickPlaceSkill:
    """Stateful per-episode skill. Call ``plan(state)`` after reset, then ``act()`` each step."""

    def __init__(self, env, params: dict[str, float] | None = None, height: float | None = None,
                 grasp_width: float | None = None, max_steps_per_target: int = 400):
        self.env = env
        self.p = params or {}
        self.h = height if height is not None else float(getattr(env, "obj_h", 0.03))
        self.r = (grasp_width if grasp_width is not None else 0.03) / 2
        self.mode = grasp_mode(self.h)
        self.intercept: float | None = None  # metres along the belt where a conveyor pick meets the object
        self.max_steps = max_steps_per_target
        self.home = np.asarray(env.module.HOME, dtype=np.float64)

    # ------------------------------------------------------------------ IK helpers
    def _ik(self, xyz: np.ndarray, axis: np.ndarray | None, seed: np.ndarray | None = None,
            yaw: float | None = None, only_seed: bool = False) -> tuple[float, np.ndarray | None]:
        best_e, best_q, best_wrong = np.inf, None, True
        seeds = ([seed] if seed is not None else []) + ([] if only_seed else ik_seeds(xyz, self.home))
        for s in seeds:
            q, e = solve_ik(self.env, xyz, s, yaw=yaw, approach=axis, iters=150)
            if axis is None:
                wrong = False
            else:
                ang, leftness = self._jaw_metrics(q, axis)
                if ang > MAX_APPROACH_ERR_DEG:
                    continue
                # Off-axis, a slightly nearer solution can close the jaws the other way and
                # push the bottle over. Keep the deterministic left-closing roll when one exists.
                wrong = leftness <= 0
            if (wrong, e) < (best_wrong, best_e):
                best_e, best_q, best_wrong = e, q, wrong
                if e < 0.002 and not wrong:
                    break
        return best_e, best_q

    def _jaw_metrics(self, q: np.ndarray, axis: np.ndarray) -> tuple[float, float]:
        """Approach error (degrees) and how far the open jaws close toward the arm's left of the approach."""
        w = self.env.world
        d = mujoco.MjData(w.model)
        d.qpos[:] = w.data.qpos
        d.qpos[w.qadr[:5]] = np.deg2rad(q[:5])
        lo, hi = w.lo[5], w.hi[5]
        d.qpos[w.qadr[5]] = lo + GRIP_OPEN / 100.0 * (hi - lo)
        mujoco.mj_kinematics(w.model, d)
        ax = d.site_xmat[self.env.site].reshape(3, 3)[:, 0]
        want = np.asarray(axis, dtype=np.float64)
        want = want / (np.linalg.norm(want) or 1.0)
        ang = float(np.rad2deg(np.arccos(np.clip(float(ax @ want), -1, 1))))
        left = np.cross(np.array([0.0, 0.0, 1.0]), ax)
        return ang, float(closing_direction(w.model, d) @ left)

    def _site_R(self, q: np.ndarray) -> np.ndarray:
        w = self.env.world
        d = mujoco.MjData(w.model)
        d.qpos[:] = w.data.qpos
        d.qpos[w.qadr[:5]] = np.deg2rad(q[:5])
        mujoco.mj_kinematics(w.model, d)
        return d.site_xmat[self.env.site].reshape(3, 3)

    def _ik_obj(self, centre: np.ndarray, axis: np.ndarray, offset: np.ndarray,
                seed: np.ndarray | None = None) -> tuple[float, np.ndarray | None, np.ndarray]:
        """IK so the object centre lands at a site-frame offset (fixed-point iteration on the site target)."""
        target = centre.copy()
        e, q = np.inf, None
        for _ in range(4):
            e, q = self._ik(target, axis, seed)
            if q is None:
                return e, None, target
            target = centre - self._site_R(q) @ offset
            seed = q
        e, q = self._ik(target, axis, seed)
        return e, q, target

    def _site_for_object(self, xy: np.ndarray, z: float, axis: np.ndarray,
                         seed: np.ndarray | None = None) -> tuple[float, np.ndarray | None, np.ndarray]:
        """IK so the held object's centre (a site-x offset of SLIDE_DEPTH behind the site) is at xy.

        The slide leaves the bottle about 2 cm short of the gripper site along the approach. Aiming
        the site at the zone centre therefore drops the bottle about 2 cm short.
        """
        target = np.array([float(xy[0]), float(xy[1]), float(z)])
        e, q = self._ik(target, axis, seed)
        if q is None:
            return e, None, target
        for _ in range(3):
            shift = (self._site_R(q) @ np.array([SLIDE_DEPTH, 0.0, 0.0]))[:2]
            target = np.array([float(xy[0]) - shift[0], float(xy[1]) - shift[1], float(z)])
            e2, q2 = self._ik(target, axis, q, only_seed=True)
            if q2 is None:
                break
            e, q = e2, q2
            if e < 0.002:
                break
        return e, q, target

    # ------------------------------------------------------------------ plan
    def plan(self, state_deg: np.ndarray) -> None:
        self.cmd = state_deg.astype(np.float64).copy()
        self.moves: Iterator[Target] = self._side() if self.mode == "side" else self._top()
        self.cur: Target | None = None
        self.steps_on, self.settled = 0, 0
        self.finished = False
        self.failed: str | None = None
        self.phase = "start"

    def _line(self, a: np.ndarray, b: np.ndarray, axis: np.ndarray | None, grip: float, n: int,
              phase: str, yaw: float | None = None, rate: float = 2.0) -> Iterator[Target]:
        for t in np.linspace(0, 1, n + 1)[1:]:
            e, q = self._ik(a + (b - a) * t, axis, self._measured(), yaw=yaw, only_seed=True)
            if q is None or e > 0.02:
                e, q = self._ik(a + (b - a) * t, axis, self._measured(), yaw=yaw)
            if q is None:
                self.failed = f"no IK on the {phase} path"
                return
            yield Target(q, grip, rate=rate, settle=1, phase=phase)

    def _measured(self) -> np.ndarray:
        return np.asarray(self.env.world.joint_state(), dtype=np.float64)

    def _go_home(self) -> Iterator[Target]:
        yield Target(self.home[:5], GRIP_HOME, phase="home", settle=6)

    def _top(self) -> Iterator[Target]:
        env, p = self.env, self.p
        belt = env.belt_of() if hasattr(env, "belt_of") else None
        if belt is not None and belt.speed > 0 and belt.over(env.cube_pos()):
            yield from self._intercept(belt)
            return
        obj, dst = env.cube_pos(), env.box_pos()
        yaw = env.cube_yaw() + p.get("yaw_offset", 0.0)
        gz = grasp_height(self.h) - 0.003 + p.get("grasp_dz", 0.0)
        g = np.r_[obj[0] + p.get("grasp_dx", 0.0), obj[1] + p.get("grasp_dy", 0.0), gz]
        carry = max(g[2] + LIFT, p.get("carry_z", 0.075))
        rz = max(self.h + 0.01, p.get("release_z", self.h + 0.025))
        r = np.r_[dst[0] + p.get("release_dx", 0.0), dst[1] + p.get("release_dy", 0.0)]
        e, q = self._ik(g + [0, 0, 0.043], None, self._measured(), yaw=yaw)
        if q is None:
            self.failed = "no IK above the object"
            return
        yield Target(q, GRIP_OPEN, settle=8, phase="hover")
        yield from self._line(g + [0, 0, 0.043], g, None, GRIP_OPEN, 6, "approach", yaw)
        _, q = self._ik(g, None, self._measured(), yaw=yaw, only_seed=True)
        yield Target(q, GRIP_CLOSED, rate=MAX_DEG_PER_STEP, settle=int(p.get("close_hold", 12)), phase="close")
        yield from self._place(g, carry, r, rz, yaw)

    def _place(self, g: np.ndarray, carry: float, r: np.ndarray, rz: float, yaw: float) -> Iterator[Target]:
        """From a closed top grasp at ``g``: lift to ``carry``, carry over ``r``, lower to ``rz``, release, home."""
        yield from self._line(g, np.r_[g[:2], carry], None, GRIP_CLOSED, 6, "lift", yaw)
        _, q = self._ik(np.r_[r, carry], None, self._measured())
        if q is None:
            self.failed = "no IK above the target"
            return
        yield Target(q, GRIP_CLOSED, settle=6, phase="carry")
        yield from self._line(np.r_[r, carry], np.r_[r, rz], None, GRIP_CLOSED, 6, "lower")
        _, q = self._ik(np.r_[r, rz], None, self._measured(), only_seed=True)
        yield Target(q, GRIP_OPEN, settle=10, phase="release")
        yield from self._line(np.r_[r, rz], np.r_[r, rz + 0.03], None, GRIP_OPEN, 4, "retreat")
        yield from self._go_home()

    # ------------------------------------------------------------- conveyor
    def _across(self, xyz: np.ndarray, yaw: float, seed: np.ndarray) -> tuple[float, np.ndarray | None]:
        """Top-down IK whose jaws close across the belt (heading ``yaw``), not along it: the IK matches a
        heading modulo 90 degrees, so the wrist roll is retried a quarter turn either way."""
        w = self.env.world
        d = mujoco.MjData(w.model)
        want = np.array([np.cos(yaw), np.sin(yaw)])
        best: tuple[float, np.ndarray | None] = (np.inf, None)
        for droll in (0.0, 90.0, -90.0):
            s = np.asarray(seed, dtype=np.float64).copy()
            s[4] += droll
            e, q = self._ik(xyz, None, s, yaw=yaw, only_seed=True)
            d.qpos[:] = w.data.qpos
            d.qpos[w.qadr[:5]] = np.deg2rad(q[:5])
            mujoco.mj_kinematics(w.model, d)
            if abs(float(closing_direction(w.model, d)[:2] @ want)) > 0.9 and e < best[0]:
                best = (e, q)
        return best

    def _intercept(self, belt) -> Iterator[Target]:
        env, p = self.env, self.p
        dt = 1.0 / FPS
        s0, lat = belt.coords(env.cube_pos())
        yaw = belt.yaw + np.pi / 2  # jaws close across the belt
        gz = belt.height + grasp_height(self.h) - 0.003 + p.get("grasp_dz", 0.0)
        seed = self._measured()
        plan = None
        for s in np.arange(s0 + ENTRY_CLEAR, belt.length - END_MARGIN, INTERCEPT_STEP):
            g = np.r_[belt.point(float(s), lat), gz]
            eh, qh = self._across(g + [0, 0, HOVER_DZ], yaw, seed)
            if qh is None or eh > 0.004:
                continue
            eg, qg = self._across(g, yaw, qh)
            if qg is None or eg > 0.004:
                continue
            transit = np.abs(qh - self._measured()[:5]).max() / MAX_DEG_PER_STEP + MOVE_MARGIN
            eta = (s - ENTRY_CLEAR - s0) / (belt.speed * dt)  # steps until the object is ENTRY_CLEAR short
            if eta >= transit:
                plan = (float(s), g, qh)
                break
        if plan is None:
            self.failed = "the object leaves the reach window before the arm can get there"
            return
        s, g, qh = plan
        self.intercept = s
        yield Target(qh, GRIP_OPEN, settle=4, phase="hover")
        yield from self._line(g + [0, 0, HOVER_DZ], g, None, GRIP_OPEN, 6, "lower", yaw)
        lead = belt.speed * dt * CLOSE_LEAD_STEPS

        def arrived() -> bool:
            return belt.coords(env.cube_pos())[0] >= s - lead

        _, q = self._across(g, yaw, self._measured())
        yield Target(q, GRIP_OPEN, settle=1, phase="wait", until=arrived)
        yield Target(q, GRIP_CLOSED, rate=MAX_DEG_PER_STEP, settle=int(p.get("close_hold", 12)), phase="close")
        dst = env.box_pos()
        carry = max(g[2] + LIFT, belt.height + p.get("carry_z", 0.075))
        rz = max(self.h + 0.01, p.get("release_z", self.h + 0.025))
        r = np.r_[dst[0] + p.get("release_dx", 0.0), dst[1] + p.get("release_dy", 0.0)]
        yield from self._place(g, carry, r, rz, yaw)

    def _side(self) -> Iterator[Target]:
        env, p = self.env, self.p
        belt = env.belt_of() if hasattr(env, "belt_of") else None
        if belt is not None and belt.speed > 0 and belt.over(env.cube_pos()):
            self.failed = "a tall object on a moving conveyor needs a side grasp, which has no intercept mode yet"
            return
        obj, dst = env.cube_pos(), env.box_pos()
        gz = grasp_height(self.h) + p.get("grasp_dz", 0.0)
        g = np.r_[obj[0] + p.get("grasp_dx", 0.0), obj[1] + p.get("grasp_dy", 0.0), gz]
        axis = qg = gt = pre = None
        for a in approaches(g[:2], "side"):
            e2, q2, t2 = self._ik_obj(g, a, OPEN_GAP_OFFSET)
            if q2 is None or e2 > 0.004:
                continue
            e1, q1 = self._ik(t2 - a * 0.06, a, q2)
            if q1 is not None and e1 < 0.008:
                axis, qg, gt, pre = a, q2, t2, t2 - a * 0.06
                break
        if axis is None:
            self.failed = "no reachable side grasp for this object position"
            return
        high = pre + np.array([0, 0, 0.10]) - axis * 0.04
        _, qh = self._ik(high, axis, qg)
        if qh is None:
            self.failed = "no IK for the transit waypoint"
            return
        yield Target(qh, GRIP_OPEN, settle=6, phase="transit")
        yield from self._line(high, pre, axis, GRIP_OPEN, 10, "pre-grasp")
        yield from self._line(pre, gt, axis, GRIP_OPEN, 8, "approach")
        slide = np.array([SLIDE_DEPTH, 0.0, pad_z(SLIDE_DEPTH) + self.r + 0.001])
        _, qs, gs = self._ik_obj(g, axis, slide, seed=qg)
        if qs is None:
            self.failed = "no IK for the slide to the fixed jaw"
            return
        yield from self._line(gt, gs, axis, GRIP_OPEN, 10, "slide")
        yield Target(qs, GRIP_CLOSED, rate=2.0, settle=int(p.get("close_hold", 12)), phase="close")
        lift = gs + [0, 0, LIFT]
        yield from self._line(gs, lift, axis, GRIP_CLOSED, 8, "lift")
        r = np.r_[dst[0] + p.get("release_dx", 0.0), dst[1] + p.get("release_dy", 0.0)]
        rz = gz + 0.006 + p.get("release_dz", 0.0)
        place = None
        for a in approaches(r, "side"):
            ec, qc, tc = self._site_for_object(r, gz + LIFT, a, self._measured())
            el, ql, tl = self._site_for_object(r, rz, a, qc)
            if qc is not None and ql is not None and ec < 0.01 and el < 0.006:
                place = (a, qc, tc, tl)
                break
        if place is None:
            self.failed = "no reachable side placement at the target"
            return
        ax2, qc, carry_site, low_site = place
        yield Target(qc, GRIP_CLOSED, rate=2.0, settle=6, phase="carry")
        yield from self._line(carry_site, low_site, ax2, GRIP_CLOSED, 8, "lower")
        yield Target(self._measured()[:5], GRIP_OPEN, settle=10, phase="release")
        back = low_site - ax2 * 0.06 + [0, 0, 0.03]
        yield from self._line(low_site, back, ax2, GRIP_OPEN, 8, "retreat")
        yield from self._go_home()

    # ------------------------------------------------------------------ act
    def _next(self) -> None:
        try:
            self.cur = next(self.moves)
            self.phase = self.cur.phase
        except StopIteration:
            self.cur = None
            self.finished = True
        self.steps_on, self.settled = 0, 0

    def act(self) -> np.ndarray:
        if self.cur is None and not self.finished:
            self._next()
        if self.cur is None:
            return self.cmd.astype(np.float32)
        tgt = np.r_[self.cur.q[:5], self.cur.grip]
        delta = tgt - self.cmd
        step = np.clip(delta, -self.cur.rate, self.cur.rate)
        step[5] = np.clip(delta[5], -8, 8)
        self.cmd = self.cmd + step
        self.steps_on += 1
        arrived = np.allclose(self.cmd, tgt) and np.abs(self._measured()[:5] - tgt[:5]).max() < 0.7
        if arrived and self.cur.until is not None:
            arrived = self.cur.until()
        if arrived:
            self.settled += 1
        if (arrived and self.settled >= self.cur.settle) or self.steps_on >= self.max_steps:
            self._next()
        return self.cmd.astype(np.float32)

    @property
    def done(self) -> bool:
        return self.finished or self.failed is not None


class PickPlacePolicy:
    """``sim_worker.run_episode``-compatible oracle (reset/act/ex) for intake scenes."""

    name = "pick-place skill (oracle, privileged state)"
    needs_pixels = False

    def __init__(self, params: dict[str, float] | None = None, height: float | None = None,
                 grasp_width: float | None = None, on_phase: Callable[[str], None] | None = None):
        self.params, self.height, self.width, self.on_phase = params, height, grasp_width, on_phase

    def reset(self, env, obs, seed):
        self.ex = PickPlaceSkill(env, self.params, self.height, self.width)
        self.ex.plan(obs["agent_pos"])

    def act(self, obs):
        a = self.ex.act()
        if self.on_phase:
            self.on_phase(self.ex.phase)
        return a
