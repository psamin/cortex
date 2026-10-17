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


def side_approach(xy: np.ndarray) -> np.ndarray:
    """Horizontal approach axis pointing away from the arm base, for side grasps of tall objects."""
    v = np.array([xy[0], xy[1], 0.0])
    return v / (np.linalg.norm(v) or 1.0)


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
    if yaw is not None and approach is None:
        # Align the jaws with the cube faces: roll relative to the arm's pan angle.
        rel = (yaw - d.qpos[qadr[0]] + np.pi / 4) % (np.pi / 2) - np.pi / 4
        q[4] = np.clip(np.rad2deg(rel), np.rad2deg(lo[4]), np.rad2deg(hi[4]))
    return q, err_p


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
