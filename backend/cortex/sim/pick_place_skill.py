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

from .expert import GRIP_CLOSED, GRIP_OPEN, MAX_DEG_PER_STEP, approach_error_deg, solve_ik
from .grasp import approaches, grasp_height, grasp_mode, ik_seeds

LIFT = 0.06
GRIP_HOME = 30.0
# Site frame for side grasps (x = approach, z = jaw closing), measured on the Menagerie SO-101.
OPEN_GAP_OFFSET = np.array([-0.020, 0.0, 0.017])  # object centre in the middle of the open jaws
SLIDE_DEPTH = -0.020


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


class PickPlaceSkill:
    """Stateful per-episode skill. Call ``plan(state)`` after reset, then ``act()`` each step."""

    def __init__(self, env, params: dict[str, float] | None = None, height: float | None = None,
                 grasp_width: float | None = None, max_steps_per_target: int = 400):
        self.env = env
        self.p = params or {}
        self.h = height if height is not None else float(getattr(env, "obj_h", 0.03))
        self.r = (grasp_width if grasp_width is not None else 0.03) / 2
        self.mode = grasp_mode(self.h)
        self.max_steps = max_steps_per_target
        self.home = np.asarray(env.module.HOME, dtype=np.float64)

    # ------------------------------------------------------------------ IK helpers
    def _ik(self, xyz: np.ndarray, axis: np.ndarray | None, seed: np.ndarray | None = None,
            yaw: float | None = None, only_seed: bool = False) -> tuple[float, np.ndarray | None]:
        best_e, best_q = np.inf, None
        seeds = ([seed] if seed is not None else []) + ([] if only_seed else ik_seeds(xyz, self.home))
        for s in seeds:
            q, e = solve_ik(self.env, xyz, s, yaw=yaw, approach=axis, iters=150)
            if axis is not None and approach_error_deg(self.env, q, axis) > 10:
                continue
            if e < best_e:
                best_e, best_q = e, q
                if e < 0.002:
                    break
        return best_e, best_q

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

    def _side(self) -> Iterator[Target]:
        env, p = self.env, self.p
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
        ax2 = next((a for a in approaches(r, "side")
                    if self._ik(np.r_[r, gz + LIFT], a)[0] < 0.01 and self._ik(np.r_[r, rz], a)[0] < 0.006), None)
        if ax2 is None:
            self.failed = "no reachable side placement at the target"
            return
        _, qc = self._ik(np.r_[r, gz + LIFT], ax2, self._measured())
        yield Target(qc, GRIP_CLOSED, rate=2.0, settle=6, phase="carry")
        yield from self._line(np.r_[r, gz + LIFT], np.r_[r, rz], ax2, GRIP_CLOSED, 8, "lower")
        yield Target(self._measured()[:5], GRIP_OPEN, settle=10, phase="release")
        back = np.r_[r, rz] - ax2 * 0.06 + [0, 0, 0.03]
        yield from self._line(np.r_[r, rz], back, ax2, GRIP_OPEN, 8, "retreat")
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
