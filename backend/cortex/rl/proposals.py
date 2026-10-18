"""Turn measured weaknesses into RL sandbox proposals with labelled estimates.

Method ("cem_skill"): the base controller is the validated IK pick-and-place skill. RL
tunes a small vector of corrections to it (grasp offset, yaw, close timing, carry and
release heights) under domain randomization centred on the weak condition, maximising an
episodic return shaped for the dominant failure mode. Successful rollouts become targeted
demonstrations that retrain ACT (distillation). The RL result is never deployed directly.

The uplift estimate is arithmetic on measured numbers plus stated assumptions:
  sandbox_after = pilot + (1 - pilot) * k,   k in CLOSE_RANGE   (share of the skill's gap RL closes)
  policy_after  = policy + max(0, sandbox_after - policy) * t,  t in TRANSFER_RANGE
"""

from __future__ import annotations

from ..services.simulation_runner import wilson
from .types import Estimate, RLSandbox, SandboxBudget, SandboxEnv, SkillParam, Weakness

CLOSE_RANGE = (0.3, 0.7)  # assumption: RL closes 30-70% of the base skill's remaining gap in the sandbox
TRANSFER_RANGE = (0.3, 0.6)  # assumption: share of that skill-level gain that distillation transfers to ACT
STEPS_PER_EPISODE = 300  # PickPlaceEnv default episode length (control steps at 30 fps)

PARAMS: dict[str, SkillParam] = {p.name: p for p in [
    SkillParam(name="grasp_dx", lo=-0.012, hi=0.012, unit="m", why="grasp point along the table x axis"),
    SkillParam(name="grasp_dy", lo=-0.012, hi=0.012, unit="m", why="grasp point along the table y axis"),
    SkillParam(name="grasp_dz", lo=-0.006, hi=0.008, unit="m", why="grasp depth"),
    SkillParam(name="yaw_offset", lo=-0.3, hi=0.3, unit="rad", why="jaw alignment with the object"),
    SkillParam(name="close_hold", lo=6, hi=24, init=12, unit="steps", why="time the jaws squeeze before lifting"),
    SkillParam(name="carry_z", lo=0.06, hi=0.11, init=0.075, unit="m", why="carry height"),
    SkillParam(name="release_z", lo=0.04, hi=0.08, init=0.055, unit="m", why="release height above the target"),
    SkillParam(name="release_dx", lo=-0.02, hi=0.02, unit="m", why="release point along x"),
    SkillParam(name="release_dy", lo=-0.02, hi=0.02, unit="m", why="release point along y"),
]}

# Which corrections and which return terms matter for each measured failure mode.
FOCUS: dict[str, tuple[list[str], dict[str, float]]] = {
    "grasp_failed": (["grasp_dx", "grasp_dy", "grasp_dz", "yaw_offset", "close_hold"],
                     {"success": 1.0, "lifted": 0.6, "held_at_carry": 0.3, "place_dist": 0.1}),
    "dropped_before_target": (["close_hold", "carry_z", "grasp_dz", "yaw_offset"],
                              {"success": 1.0, "held_at_carry": 0.7, "lifted": 0.2, "place_dist": 0.2}),
    "released_outside_target": (["release_dx", "release_dy", "release_z", "carry_z"],
                                {"success": 1.0, "place_dist": 0.7, "held_at_carry": 0.2, "lifted": 0.1}),
}
DEFAULT_FOCUS = (list(PARAMS), {"success": 1.0, "lifted": 0.3, "held_at_carry": 0.3, "place_dist": 0.3})


def propose(project_id: str, weaknesses: list[Weakness], budget: SandboxBudget | None = None,
            scene_digest: str | None = None, max_sandboxes: int = 3) -> list[RLSandbox]:
    out = []
    for w in weaknesses[:max_sandboxes]:
        names, reward = FOCUS.get(w.dominant_failure or "", DEFAULT_FOCUS)
        env = SandboxEnv(condition=w.condition, cube_offset=w.cube_offset, light_scales=(
            round(w.light_scale * 0.8, 3), round(w.light_scale * 1.2, 3)), scene_digest=scene_digest)
        sb = RLSandbox(project_id=project_id, weakness=w, env=env, reward_weights=reward,
                       params=[PARAMS[n] for n in names], budget=budget or SandboxBudget())
        sb.estimate = Estimate(demos_expected=sb.budget.heldout_episodes // 2, assumptions=_assumptions())
        out.append(sb)
    return out


def estimate(sb: RLSandbox, pilot_successes: int, pilot_n: int, sim_steps_per_s: float,
             steps_per_episode: int = STEPS_PER_EPISODE) -> Estimate:
    """Fill a sandbox's estimate from a measured pilot (base skill in the sandbox) and measured throughput."""
    p = pilot_successes / pilot_n if pilot_n else 0.0
    lo, hi = (p + (1 - p) * k for k in CLOSE_RANGE)
    pol = sb.weakness.success_rate
    up_lo, up_hi = (max(0.0, s - pol) * t for s, t in ((lo, TRANSFER_RANGE[0]), (hi, TRANSFER_RANGE[1])))
    episodes = sb.budget.train_episodes + sb.budget.heldout_episodes + pilot_n
    cpu_min = episodes * steps_per_episode / max(sim_steps_per_s, 1e-6) / 60
    wall = cpu_min / sb.budget.max_parallel
    demos = round(sb.budget.heldout_episodes * hi)
    return Estimate(pilot_episodes=pilot_n, pilot_success=round(p, 4), pilot_ci95=wilson(pilot_successes, pilot_n),
                    sandbox_success_after=(round(lo, 3), round(hi, 3)),
                    policy_uplift=(round(up_lo, 3), round(up_hi, 3)),
                    policy_success_after=(round(min(1.0, pol + up_lo), 3), round(min(1.0, pol + up_hi), 3)),
                    demos_expected=demos, sim_steps_per_s=round(sim_steps_per_s, 1), cpu_minutes=round(cpu_min, 1),
                    wall_minutes=round(wall, 1), usd=0.0, gpu_hours=0.0, assumptions=_assumptions())


def _assumptions() -> list[str]:
    return [f"RL closes {CLOSE_RANGE[0]:.0%}-{CLOSE_RANGE[1]:.0%} of the base skill's remaining gap in the sandbox",
            f"distillation transfers {TRANSFER_RANGE[0]:.0%}-{TRANSFER_RANGE[1]:.0%} of that gain to the ACT policy",
            "cost is CPU time on this machine at the measured sim throughput; local runs cost $0",
            "only the re-evaluated ACT checkpoint counts as a measured improvement of the policy"]
