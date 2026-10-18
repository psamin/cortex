"""Scripted pick-and-place expert for PickPlaceEnv.

Damped-least-squares IK on a scratch copy of the MuJoCo state computes joint
targets for Cartesian waypoints (pre-grasp, grasp, lift, carry, release); the
expert then emits smooth, rate-limited joint-target actions in so101_follower
units. It reads privileged simulator state (cube pose), so it is an oracle
demonstrator: its episodes are genuine physics rollouts, but it is never
presented as a learned policy.
"""

from __future__ import annotations

import mujoco
import numpy as np

from .so101 import PickPlaceEnv

ARM = 5  # shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll
GRIP_OPEN, GRIP_CLOSED = 60.0, 0.0
DOWN = np.array([0.0, 0.0, -1.0])
MAX_DEG_PER_STEP = 4.0


def side_approach(xy: np.ndarray, pitch_deg: float = 0.0) -> np.ndarray:
    """Approach axis for side grasps of tall objects: away from the arm base, tilted `pitch_deg` below
    horizontal (a downward tilt brings the grasp closer to the base)."""
    v = np.array([xy[0], xy[1], 0.0])
    v /= np.linalg.norm(v) or 1.0
    p = np.deg2rad(pitch_deg)
    return np.array([v[0] * np.cos(p), v[1] * np.cos(p), -np.sin(p)])


def _jaw_tips(m: mujoco.MjModel) -> tuple[list[int], list[int]]:
    fixed = [i for i in range(m.ngeom) if m.geom(i).name.startswith("fixed_jaw_sph_tip")]
    moving = [i for i in range(m.ngeom) if m.geom(i).name.startswith("moving_jaw_sph_tip")]
    return fixed, moving


def closing_direction(m: mujoco.MjModel, d: mujoco.MjData) -> np.ndarray:
    """Unit vector from the fixed jaw's tips to the moving jaw's tips: the direction the jaws close along."""
    fixed, moving = _jaw_tips(m)
    v = d.geom_xpos[moving].mean(axis=0) - d.geom_xpos[fixed].mean(axis=0)
    return v / (np.linalg.norm(v) or 1.0)


def approach_error_deg(env: PickPlaceEnv, q_deg: np.ndarray, approach: np.ndarray | None = None) -> float:
    """Angle between the gripper's approach axis at q and the wanted axis (default: straight down)."""
    w = env.world
    d = mujoco.MjData(w.model)
    d.qpos[:] = w.data.qpos
    d.qpos[w.qadr[:ARM]] = np.deg2rad(q_deg[:ARM])
    mujoco.mj_kinematics(w.model, d)
    a = d.site_xmat[env.site].reshape(3, 3)[:, 0]
    want = DOWN if approach is None else approach / np.linalg.norm(approach)
    return float(np.rad2deg(np.arccos(np.clip(a @ want, -1, 1))))


def solve_ik(env: PickPlaceEnv, target: np.ndarray, q_init_deg: np.ndarray, yaw: float | None = None,
             iters: int = 200, ori_weight: float = 0.3, approach: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    """Joint degrees (5 arm joints) placing the gripper site at `target` with its
    approach axis along `approach` (default: pointing down, a top grasp).
    Returns (q_deg, position error in metres)."""
    axis = DOWN if approach is None else np.asarray(approach, dtype=np.float64)
    w = env.world
    m = w.model
    d = mujoco.MjData(m)
    d.qpos[:] = w.data.qpos
    qadr = w.qadr[:ARM]
    dofs = np.array([m.jnt_dofadr[m.joint(n).id] for n in
                     ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")])
    lo, hi = w.lo[:ARM], w.hi[:ARM]
    d.qpos[qadr] = np.clip(np.deg2rad(q_init_deg[:ARM]), lo, hi)
    jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    err_p = np.inf
    for _ in range(iters):
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        x = d.site_xpos[env.site]
        R = d.site_xmat[env.site].reshape(3, 3)
        e_p = target - x
        e_o = np.cross(R[:, 0], axis) * ori_weight
        err_p = float(np.linalg.norm(e_p))
        if err_p < 1e-3 and np.linalg.norm(e_o) < 1e-2:
            break
        mujoco.mj_jacSite(m, d, jp, jr, env.site)
        J = np.vstack([jp[:, dofs], jr[:, dofs] * ori_weight])
        e = np.concatenate([e_p, e_o])
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), e)
        d.qpos[qadr] = np.clip(d.qpos[qadr] + np.clip(dq, -0.2, 0.2), lo, hi)
    q = np.rad2deg(d.qpos[qadr]).copy()
    if approach is not None:
        q[4] = _horizontal_jaw_roll(m, d, qadr, lo[4], hi[4], q[4])
    if yaw is not None and approach is None:
        # Align the jaws with the cube faces: roll relative to the arm's pan angle.
        rel = (yaw - d.qpos[qadr[0]] + np.pi / 4) % (np.pi / 2) - np.pi / 4
        q[4] = np.clip(np.rad2deg(rel), np.rad2deg(lo[4]), np.rad2deg(hi[4]))
    return q, err_p


def _horizontal_jaw_roll(m: mujoco.MjModel, d: mujoco.MjData, qadr: np.ndarray, lo: float, hi: float,  # rad
                         current_deg: float) -> float:
    """Side grasps: the wrist roll (degrees) that makes the jaws close horizontally around a standing
    object, nearest the current roll. Roll does not move the gripper site, so reach is unchanged."""
    best, best_key = current_deg, (np.inf, np.inf)
    for r in np.arange(-180.0, 181.0, 15.0):
        if not lo <= np.deg2rad(r) <= hi:  # lo, hi: joint limits in radians
            continue
        d.qpos[qadr[4]] = np.deg2rad(r)
        mujoco.mj_kinematics(m, d)
        key = (round(abs(closing_direction(m, d)[2]), 2), abs(r - current_deg))
        if key < best_key:
            best, best_key = r, key
    d.qpos[qadr[4]] = np.deg2rad(best)
    mujoco.mj_kinematics(m, d)
    return float(best)


class ScriptedExpert:
    """Stateful per-episode expert. Call plan() after reset, then act() each step."""

    def __init__(self, env: PickPlaceEnv, rng: np.random.Generator | None = None):
        self.env = env
        self.rng = rng or np.random.default_rng(0)

    def plan(self, state_deg: np.ndarray) -> None:
        env = self.env
        cube, box = env.cube_pos(), env.box_pos()
        yaw = env.cube_yaw()
        z_grasp = 0.012 + self.rng.uniform(-0.002, 0.002)
        hover = 0.055 + self.rng.uniform(-0.005, 0.005)
        q = state_deg.copy()
        legs = []  # (target xyz, gripper, hold steps, use_yaw)
        legs.append((cube + [0, 0, hover], GRIP_OPEN, 8, True))
        legs.append((np.r_[cube[:2], z_grasp], GRIP_OPEN, 8, True))
        legs.append((np.r_[cube[:2], z_grasp], GRIP_CLOSED, 12, True))
        legs.append((np.r_[cube[:2], 0.075], GRIP_CLOSED, 4, True))
        legs.append((np.r_[box[:2], 0.075], GRIP_CLOSED, 6, False))
        legs.append((np.r_[box[:2], 0.055], GRIP_CLOSED, 4, False))
        legs.append((np.r_[box[:2], 0.055], GRIP_OPEN, 10, False))
        legs.append((np.r_[box[:2], 0.08], GRIP_OPEN, 6, False))
        self.waypoints: list[tuple[np.ndarray, int]] = []
        self.max_ik_err = 0.0
        for target, grip, hold, use_yaw in legs:
            arm, err = solve_ik(env, target, q, yaw if use_yaw else None)
            self.max_ik_err = max(self.max_ik_err, err)
            q = np.r_[arm, grip]
            self.waypoints.append((q, hold))
        self.cmd = state_deg.astype(np.float64).copy()
        self.i = 0
        self.hold = 0

    def act(self) -> np.ndarray:
        if self.i >= len(self.waypoints):
            return self.cmd.astype(np.float32)
        target, hold = self.waypoints[self.i]
        delta = target - self.cmd
        step = np.clip(delta, -MAX_DEG_PER_STEP, MAX_DEG_PER_STEP)
        step[5] = np.clip(delta[5], -8, 8)  # gripper moves faster
        self.cmd = self.cmd + step
        if np.abs(target - self.cmd).max() < 0.5:
            self.hold += 1
            if self.hold >= hold:
                self.i, self.hold = self.i + 1, 0
        return self.cmd.astype(np.float32)

    @property
    def done(self) -> bool:
        return self.i >= len(self.waypoints)
