"""A sandbox: domain randomization centred on one weak condition, and the shaped return.

Each seed draws its own object offset around the weak branch's offset, its own light scale,
and friction/mass jitter on the task object (restored afterwards), so RL can't overfit one
scene. The return is a weighted sum of measured terms:
  success        1 if the simulator's success test passes
  lifted         1 if the object's centre rose 3 cm above its resting height (h/2 on intake scenes)
  held_at_carry  1 if it rose 4.5 cm above it (it survived the lift into the carry)
Heights are relative to the resting height, so a tall object standing still earns nothing (#152).
  place_dist     1 - min(1, final object-target distance / 0.3 m)
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from ..sim.expert import MAX_DEG_PER_STEP
from ..sim.so101 import PickPlaceEnv, Variation
from ..workers.sim_worker import CUBE_REST_Z
from .types import SandboxEnv

LIFT_M, CARRY_M = 0.03, 0.045  # rise above the resting height: lifted, and survived into the carry
GRIP_RATE = 8.0  # deg/step: both skills move the gripper this fast


@dataclass
class Draw:
    var: Variation
    friction_scale: float
    mass_scale: float
    yaw: float = 0.0  # object yaw (rad), applied after reset


def draw(env_spec: SandboxEnv, seed: int) -> Draw:
    rng = np.random.default_rng(seed)
    j = env_spec.offset_jitter_m
    ox = env_spec.cube_offset[0] + rng.uniform(-j, j)
    oy = env_spec.cube_offset[1] + rng.uniform(-j, j)
    lo, hi = env_spec.light_scales
    var = Variation(cube_region=env_spec.condition.cube_region, lighting=env_spec.condition.lighting,
                    cube_offset=(float(ox), float(oy)), light_scale=float(rng.uniform(lo, hi)))
    f = 1 + rng.uniform(-env_spec.friction_jitter, env_spec.friction_jitter)
    m = 1 + rng.uniform(-env_spec.mass_jitter, env_spec.mass_jitter)
    yaw = rng.uniform(-env_spec.yaw_jitter_rad, env_spec.yaw_jitter_rad)
    return Draw(var, float(f), float(m), float(yaw))


def widen(env_spec: SandboxEnv, success_rate: float) -> bool:
    """ADR-style widening (docs/RL_DESIGN.md section 3): once the tuned skill succeeds more than ``widen_at``
    inside the current ranges, scale the offset and yaw jitter and the light range by ``widen_factor``, up to
    ``max_offset_jitter_m``. Returns True if the ranges changed. Mutates ``env_spec``."""
    if success_rate <= env_spec.widen_at or env_spec.offset_jitter_m >= env_spec.max_offset_jitter_m - 1e-9:
        return False
    k = env_spec.widen_factor
    env_spec.offset_jitter_m = round(min(env_spec.max_offset_jitter_m, env_spec.offset_jitter_m * k), 4)
    env_spec.yaw_jitter_rad = round(min(np.pi / 2, env_spec.yaw_jitter_rad * k), 4)
    lo, hi = env_spec.light_scales
    mid, half = (lo + hi) / 2, (hi - lo) / 2 * k
    env_spec.light_scales = (round(max(0.1, mid - half), 3), round(mid + half, 3))
    env_spec.widened += 1
    return True


class Physics:
    """Applies and restores the friction/mass draw on the task object's geoms and body."""

    def __init__(self, env: PickPlaceEnv):
        self.env = env
        self.saved: tuple[np.ndarray, float] | None = None

    def apply(self, d: Draw) -> None:
        w = self.env.world
        geoms = np.flatnonzero(w.model.geom_bodyid == self.env.cube)
        if self.saved is None:
            self.saved = (w.model.geom_friction[geoms].copy(), float(w.model.body_mass[self.env.cube]))
        fr, mass = self.saved
        w.model.geom_friction[geoms] = fr * d.friction_scale
        w.model.body_mass[self.env.cube] = mass * d.mass_scale
        a = self.env.cube_qadr
        w.data.qpos[a + 3:a + 7] = (np.cos(d.yaw / 2), 0, 0, np.sin(d.yaw / 2))
        mujoco.mj_forward(w.model, w.data)


# DART-style noise (Laskey et al. 2017): perturb the skill's free-space waypoints so demos visit off-nominal
# states and show the skill recovering from them. Contact legs (grasp, close, release) are left alone.
# The perturbed command is what runs; the demo's action label is the clean skill's command (act_labelled).
DART_LEGS = (0, 3, 4)  # TunedSkill legs: hover, lift, carry
DART_PHASES = ("hover", "transit", "carry")  # PickPlaceSkill single-target phases


def dart_perturb(skill, rng: np.random.Generator, sigma_deg: float = 3.0) -> None:
    """Add N(0, sigma_deg) to the arm joints of the skill's free-space targets, in place, after ``plan()``."""
    if hasattr(skill, "waypoints"):  # rl.skill.TunedSkill: a fixed list of (q, hold)
        skill.clean_waypoints = list(skill.waypoints)
        for i in DART_LEGS:
            q, hold = skill.waypoints[i]
            skill.waypoints[i] = (np.r_[q[:5] + rng.normal(0, sigma_deg, 5), q[5:]], hold)
        return
    moves = skill.moves  # sim.pick_place_skill.PickPlaceSkill: a generator of Targets

    def noisy():
        for t in moves:
            t.clean_q = np.asarray(t.q, dtype=np.float64).copy()
            if t.phase in DART_PHASES:
                t.q = np.asarray(t.q, dtype=np.float64) + rng.normal(0, sigma_deg, len(t.q))
            yield t
    skill.moves = noisy()


def _step_toward(cmd: np.ndarray, target: np.ndarray, rate: float) -> np.ndarray:
    d = target - cmd
    step = np.clip(d, -rate, rate)
    step[5] = np.clip(d[5], -GRIP_RATE, GRIP_RATE)
    return (cmd + step).astype(np.float32)


def act_labelled(skill) -> tuple[np.ndarray, np.ndarray]:
    """One step of a (possibly DART-perturbed) skill: ``(executed, label)``.

    ``executed`` is the perturbed command the skill issues. ``label`` is the command the unperturbed skill would
    issue from the same command state: the same rate-limited step, towards the leg's clean target. So the demo
    shows off-nominal states, labelled with the correction back towards the clean path, never the noise (#152).
    Without a perturbation the two are equal."""
    prev = np.asarray(skill.cmd, dtype=np.float64).copy()
    if hasattr(skill, "waypoints"):  # rl.skill.TunedSkill
        i = skill.i
        a = skill.act()
        clean = getattr(skill, "clean_waypoints", None)
        if clean is None or i >= len(clean):
            return a, a.copy()
        return a, _step_toward(prev, np.asarray(clean[i][0], dtype=np.float64), MAX_DEG_PER_STEP)
    if skill.cur is None and not skill.finished:  # what act() would do first; done here to see the target
        skill._next()
    cur = skill.cur
    a = skill.act()
    if cur is None:
        return a, a.copy()
    q = np.asarray(getattr(cur, "clean_q", cur.q), dtype=np.float64)
    return a, _step_toward(prev, np.r_[q[:5], cur.grip], cur.rate)


def episode_return(ep: dict, weights: dict[str, float]) -> tuple[float, dict[str, float]]:
    """``ep``: success, max_cube_z, final_cube_box_dist and rest_z (the object's resting centre height)."""
    lift = ep["max_cube_z"] - ep.get("rest_z", CUBE_REST_Z)
    terms = {
        "success": float(ep["success"]),
        "lifted": float(lift > LIFT_M),
        "held_at_carry": float(lift > CARRY_M),
        "place_dist": float(1 - min(1.0, ep["final_cube_box_dist"] / 0.3)),
    }
    total = sum(weights.get(k, 0.0) * v for k, v in terms.items())
    return total / max(sum(weights.values()), 1e-9), terms
