"""Turn measured weaknesses into RL sandbox proposals with labelled estimates.

Method ("cem_skill"): the base controller is the validated IK pick-and-place skill. RL
tunes a small vector of corrections to it (grasp offset, yaw, close timing, carry and
release heights) under domain randomization centred on the weak condition, maximising an
episodic return shaped for the dominant failure mode. Successful rollouts become targeted
demonstrations that retrain ACT (distillation). The RL result is never deployed directly.

Estimates (docs/RL_DESIGN.md section 4), by Monte Carlo over posteriors:
  * sandbox success: Beta posterior of the short CEM pilot's best candidate (measured). Without
    a CEM pilot, the base-skill pilot plus k ~ U(CLOSE_RANGE) of its gap (prior).
  * policy uplift = max(0, sandbox - policy) * t, with the policy rate a Beta posterior of its
    evaluation. t is learned from this project's past rounds (realised uplift / skill gain,
    bootstrapped) once there are >= MIN_ROUNDS of them; before that, t ~ U(TRANSFER_RANGE) (prior).
  * regression elsewhere: prior N(-0.02, 0.02), clipped to <= 0, until rounds measure it.
"""

from __future__ import annotations

import numpy as np

from .stats import beta_samples, interval
from .types import Cost, Estimate, Interval, Pilot, RLSandbox, SandboxBudget, SandboxEnv, SkillParam, Weakness

CLOSE_RANGE = (0.3, 0.7)  # assumption: RL closes 30-70% of the base skill's remaining gap in the sandbox
TRANSFER_RANGE = (0.3, 0.6)  # assumption: share of that skill-level gain that distillation transfers to ACT
STEPS_PER_EPISODE = 300  # PickPlaceEnv default episode length (control steps at 30 fps)
MIN_ROUNDS = 3  # past (skill gain -> realised uplift) rounds needed to use a learned transfer factor
N_MC = 20_000

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
                       params=[PARAMS[n] for n in names],
                       budget=budget.model_copy(deep=True) if budget else SandboxBudget())  # one per sandbox
        sb.estimate = Estimate(demos_expected=sb.budget.demos, assumptions=_assumptions())
        out.append(sb)
    return out


def estimate(sb: RLSandbox, pilot: Pilot, sim_steps_per_s: float, history: list[tuple[float, float]] | None = None,
             delta: float = 0.05, steps_per_episode: int = STEPS_PER_EPISODE, seed: int = 0,
             record_steps_per_s: float | None = None) -> Estimate:
    """Fill a sandbox's estimate from its measured pilot, measured throughput and past rounds.

    ``record_steps_per_s``: measured throughput while recording demos (camera renders on); defaults to
    ``sim_steps_per_s``. Demo attempts are scaled by the predicted sandbox success (failures are discarded).
    ``history``: this project's past rounds as (skill-level gain in the sandbox, realised ACT uplift on that
    condition) pairs, both measured.
    """
    rng = np.random.default_rng(seed)
    if pilot.cem_iterations and pilot.cem_best_n:
        sand = beta_samples(pilot.cem_best_successes or 0, pilot.cem_best_n, N_MC, seed)
        sand_src = "measured"
    else:
        base = beta_samples(pilot.successes, pilot.n, N_MC, seed)
        sand = base + (1 - base) * rng.uniform(*CLOSE_RANGE, N_MC)
        sand_src = "prior"
    w = sb.weakness
    pol = beta_samples(w.successes, w.n, N_MC, seed + 1)
    t, t_src, n_rounds = transfer_samples(history, rng)
    uplift = np.maximum(0.0, sand - pol) * t
    after = np.minimum(1.0, pol + uplift)
    regress = np.minimum(0.0, rng.normal(-0.02, 0.02, N_MC))

    b = sb.budget
    fast = (pilot.n + b.pilot_cem_iterations * b.population * b.episodes_per_candidate + b.train_episodes
            + 2 * b.heldout_episodes)
    demo_attempts = int(np.ceil(b.demos / max(float(sand.mean()), 1 / 3)))  # the worker caps attempts at 3x
    episodes = fast + demo_attempts
    cpu_h = (fast * steps_per_episode / max(sim_steps_per_s, 1e-6)
             + demo_attempts * steps_per_episode / max(record_steps_per_s or sim_steps_per_s, 1e-6)) / 3600
    wall_min = cpu_h * 60 / b.max_parallel
    mean_up = float(uplift.mean())
    return Estimate(
        pilot=pilot,
        predicted_sandbox_success=Interval(mean=round(float(sand.mean()), 4), ci90=interval(sand), source=sand_src),
        predicted_policy_uplift=Interval(mean=round(mean_up, 4), ci90=interval(uplift),
                                         source="learned" if t_src == "learned" else "prior"),
        predicted_policy_success=Interval(mean=round(float(after.mean()), 4), ci90=interval(after),
                                          source="learned" if t_src == "learned" else "prior"),
        predicted_regression_elsewhere=Interval(mean=round(float(regress.mean()), 4), ci90=interval(regress),
                                                source="prior"),
        p_improve_ge_delta=round(float((uplift >= delta).mean()), 4), delta=delta,
        uplift_per_hour=round(mean_up / max(wall_min / 60, 1 / 60), 4),
        demos_expected=b.demos,
        cost=Cost(episodes=episodes, cpu_hours=round(cpu_h, 3), wall_clock_est_min=round(wall_min, 1),
                  sim_steps_per_s=round(sim_steps_per_s, 1)),
        assumptions=_assumptions(sand_src, t_src, n_rounds))


def transfer_samples(history: list[tuple[float, float]] | None, rng: np.random.Generator,
                     n: int = N_MC) -> tuple[np.ndarray, str, int]:
    """Samples of the transfer factor t (realised ACT uplift / skill-level gain), its source and the rounds used.

    Learned (a bootstrap mean over this project's measured rounds, clipped to [0, 1]) once >= MIN_ROUNDS rounds
    with a skill gain above 1 pp exist; until then the labelled prior U(TRANSFER_RANGE)."""
    hist = [(g, u) for g, u in (history or []) if g > 0.01]
    if len(hist) >= MIN_ROUNDS:
        ratios = np.clip([u / g for g, u in hist], 0, 1)
        return rng.choice(ratios, size=(n, len(ratios))).mean(axis=1), "learned", len(hist)
    return rng.uniform(*TRANSFER_RANGE, n), "prior", len(hist)


def transfer_fit(history: list[tuple[float, float]] | None, seed: int = 0) -> dict:
    """The refitted transfer curve, as the estimates will use it next (``source`` says what it rests on)."""
    t, src, n = transfer_samples(history, np.random.default_rng(seed))
    return {"kind": "estimate", "source": src, "rounds_used": n, "min_rounds": MIN_ROUNDS,
            "t_mean": round(float(t.mean()), 4), "t_ci90": interval(t),
            "model": "policy uplift = max(0, sandbox skill success - policy success) * t, capped at success 1"}


def _assumptions(sand_src: str = "prior", t_src: str = "prior", rounds: int = 0) -> list[str]:
    out = []
    if sand_src == "measured":
        out.append("sandbox success is the Beta posterior of the short CEM pilot's best candidate (measured)")
    else:
        out.append(f"heuristic prior: RL closes {CLOSE_RANGE[0]:.0%}-{CLOSE_RANGE[1]:.0%} of the base skill's gap "
                   "in the sandbox (no literature basis; replaced once a CEM pilot runs)")
    if t_src == "learned":
        out.append(f"transfer to ACT learned from this project's {rounds} past rounds (bootstrapped)")
    else:
        out.append(f"heuristic prior: distillation transfers {TRANSFER_RANGE[0]:.0%}-{TRANSFER_RANGE[1]:.0%} of the "
                   f"skill-level gain to ACT (replaced after {MIN_ROUNDS} measured rounds)")
    out += ["regression on other conditions is a prior (-2 +/- 2 pp) until rounds measure it; targeted data is "
            "capped at 30-40% of the dataset to limit it",
            "cost is CPU time at the measured sim throughput (render-free for training and evaluation, with camera "
            "renders for demo recording); local runs cost $0",
            "only the re-evaluated ACT checkpoint counts as a measured improvement of the policy"]
    return out
