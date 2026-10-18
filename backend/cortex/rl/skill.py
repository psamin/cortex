"""The validated IK pick-and-place skill with RL-tunable corrections.

Same eight IK legs as ``ScriptedExpert`` (hover, grasp, close, lift, carry, lower,
release, retreat), with the sandbox's parameters applied as offsets. Every target still
goes through ``solve_ik`` (joint limits clipped) and the rate-limited joint stepping, so
RL only chooses *where* the validated skill goes, never raw motor commands. It reads
privileged simulator state (object pose), so it is labelled an oracle wherever it is used.
"""

from __future__ import annotations

import numpy as np

from ..sim.expert import GRIP_CLOSED, GRIP_OPEN, ScriptedExpert, solve_ik

LEG_NAMES = ("hover", "grasp", "close", "lift", "carry", "lower", "release", "retreat")
DEFAULTS = {"grasp_dx": 0.0, "grasp_dy": 0.0, "grasp_dz": 0.0, "yaw_offset": 0.0, "close_hold": 12.0,
            "carry_z": 0.075, "release_z": 0.055, "release_dx": 0.0, "release_dy": 0.0}


class TunedSkill(ScriptedExpert):
    def __init__(self, env, params: dict[str, float] | None = None):
        super().__init__(env, np.random.default_rng(0))
        self.p = {**DEFAULTS, **(params or {})}

    def plan(self, state_deg: np.ndarray) -> None:
        env, p = self.env, self.p
        cube, box = env.cube_pos(), env.box_pos()
        yaw = env.cube_yaw() + p["yaw_offset"]
        g = np.r_[cube[0] + p["grasp_dx"], cube[1] + p["grasp_dy"], 0.012 + p["grasp_dz"]]
        r = np.r_[box[0] + p["release_dx"], box[1] + p["release_dy"]]
        carry, release, hold = p["carry_z"], p["release_z"], int(round(p["close_hold"]))
        legs = [
            (g + [0, 0, 0.043], GRIP_OPEN, 8, True),
            (g, GRIP_OPEN, 8, True),
            (g, GRIP_CLOSED, hold, True),
            (np.r_[g[:2], carry], GRIP_CLOSED, 4, True),
            (np.r_[r, carry], GRIP_CLOSED, 6, False),
            (np.r_[r, release], GRIP_CLOSED, 4, False),
            (np.r_[r, release], GRIP_OPEN, 10, False),
            (np.r_[r, release + 0.025], GRIP_OPEN, 6, False),
        ]
        q = state_deg.copy()
        self.waypoints, self.max_ik_err = [], 0.0
        for target, grip, h, use_yaw in legs:
            arm, err = solve_ik(env, target, q, yaw if use_yaw else None)
            self.max_ik_err = max(self.max_ik_err, err)
            q = np.r_[arm, grip]
            self.waypoints.append((q, h))
        self.cmd = state_deg.astype(np.float64).copy()
        self.i, self.hold = 0, 0

    @property
    def phase(self) -> str:
        return LEG_NAMES[min(self.i, len(LEG_NAMES) - 1)]


class TunedPolicy:
    """``sim_worker.run_episode``-compatible wrapper (reset/act/ex). Reads sim state, never the cameras."""

    needs_pixels = False

    def __init__(self, params: dict[str, float] | None = None):
        self.params = params or {}
        self.name = "rl-tuned skill (oracle, privileged state)"

    def reset(self, env, obs, seed):
        self.ex = TunedSkill(env, self.params)
        self.ex.plan(obs["agent_pos"])

    def act(self, obs):
        return self.ex.act()
