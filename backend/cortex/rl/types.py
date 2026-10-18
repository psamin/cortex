"""Contracts for RL sandboxes (#80): weaknesses, sandbox specs, estimates and progress.

Everything Cortex predicts carries ``kind="estimate"`` and lists its assumptions. Only
held-out simulator evaluations are "measured".
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from ..models import Condition, new_id, now

FailureMode = Literal["grasp_failed", "dropped_before_target", "released_outside_target", "sim_unstable", "other"]


class StartCluster(BaseModel):
    """Failures grouped by where the object started (1 cm bins) and how they failed."""

    start_bin: str
    failure_mode: str
    count: int
    seeds: list[int]


class Weakness(BaseModel):
    """One evaluation branch where the learned policy falls short of the target, measured in sim."""

    key: str  # e.g. "shifted_object"
    branch_id: str
    evaluation_id: str
    checkpoint_id: str | None
    policy_label: str = ""
    condition: Condition
    cube_offset: tuple[float, float] = (0.0, 0.0)
    light_scale: float = 1.0
    n: int
    successes: int
    success_rate: float
    ci95: tuple[float, float]
    target: float
    gap: float  # target - success_rate, floored at 0
    failure_modes: dict[str, int] = {}
    dominant_failure: str | None = None
    gap_at_upper: float = 0.0  # target - upper Wilson bound: the gap we're confident exists
    score: float  # ranking: gap at the upper bound, weighted by confidence (1 - CI width)
    start_clusters: list[StartCluster] = []
    vlm_hypotheses: list[str] = []  # unverified until a human confirms
    evidence: list[str] = []


class SkillParam(BaseModel):
    """A correction to the validated IK pick-and-place skill that RL tunes."""

    name: str
    lo: float
    hi: float
    init: float = 0.0
    unit: str = ""
    why: str = ""


class SandboxEnv(BaseModel):
    """Domain randomization centred on the weak condition. ADR widens the ranges (x widen_factor)
    once tuned-skill success inside them passes ``widen_at``."""

    condition: Condition
    cube_offset: tuple[float, float] = (0.0, 0.0)
    offset_jitter_m: float = 0.02
    yaw_jitter_rad: float = 0.6
    light_scales: tuple[float, float] = (0.8, 1.2)
    friction_jitter: float = 0.2
    mass_jitter: float = 0.2
    scene_digest: str | None = None  # the intake scene when one is built; None = the upstream tabletop
    widen_at: float = 0.8
    widen_factor: float = 1.5
    max_offset_jitter_m: float = 0.05
    widened: int = 0


class SandboxBudget(BaseModel):
    iterations: int = Field(12, ge=1, le=15)  # CEM stops earlier on the stopping rules
    population: int = Field(32, ge=4, le=128)
    pilot_episodes: int = Field(10, ge=4, le=100)
    pilot_cem_iterations: int = Field(2, ge=0, le=5)
    demos: int = Field(30, ge=0, le=200)  # targeted demos to generate (capped vs the dataset size)
    dart_noise_frac: float = Field(0.2, ge=0, le=0.5)  # share of demos generated with perturbed waypoints
    episodes_per_candidate: int = Field(2, ge=1, le=20)
    heldout_episodes: int = Field(20, ge=5, le=200)
    max_parallel: int = Field(2, ge=1, le=16)
    executor: Literal["local_cpu", "futurama_gpu"] = "local_cpu"

    @property
    def train_episodes(self) -> int:
        return self.iterations * self.population * self.episodes_per_candidate


class Interval(BaseModel):
    mean: float
    ci90: tuple[float, float]
    source: Literal["measured", "prior", "learned"]


class Pilot(BaseModel):
    """Measured in the sandbox before anything is accepted."""

    n: int
    successes: int
    wilson_95: tuple[float, float]
    cem_iterations: int = 0  # the short CEM pilot (0 = base skill only)
    cem_best_successes: int | None = None
    cem_best_n: int | None = None


class Cost(BaseModel):
    episodes: int
    cpu_hours: float
    gpu_hours: float = 0.0
    wall_clock_est_min: float
    usd: float = 0.0
    sim_steps_per_s: float


class Estimate(BaseModel):
    """Everything predicted before a sandbox runs. ``source`` says what each number rests on."""

    kind: Literal["estimate"] = "estimate"
    pilot: Pilot | None = None
    predicted_sandbox_success: Interval | None = None
    predicted_policy_uplift: Interval | None = None
    predicted_policy_success: Interval | None = None
    predicted_regression_elsewhere: Interval | None = None
    p_improve_ge_delta: float | None = None
    delta: float = 0.05
    uplift_per_hour: float | None = None
    demos_expected: int | None = None
    cost: Cost | None = None
    assumptions: list[str] = []


class IterationStats(BaseModel):
    iteration: int
    episodes: int
    mean_return: float
    success_rate: float
    ci95: tuple[float, float]
    elite_success_rate: float | None = None  # the top-return candidates' success (what ADR widens on)
    best_return: float
    best_params: dict[str, float]
    elapsed_s: float
    env: SandboxEnv | None = None  # the domain ranges this iteration ran in


class SandboxResult(BaseModel):
    kind: Literal["measured"] = "measured"
    heldout_seeds: list[int]
    base_success: float
    base_ci95: tuple[float, float]
    tuned_success: float
    tuned_ci95: tuple[float, float]
    p_value: float | None = None  # exact McNemar on matched held-out seeds
    discordant: tuple[int, int] = (0, 0)  # (base only, tuned only)
    params: dict[str, float]
    final_env: SandboxEnv | None = None
    stop_reason: str = ""
    demo_episode_ids: list[str] = []


class RLSandbox(BaseModel):
    id: str = Field(default_factory=lambda: new_id("rl"))
    project_id: str
    weakness: Weakness
    env: SandboxEnv
    reward_weights: dict[str, float]
    method: Literal["cem_skill"] = "cem_skill"
    params: list[SkillParam]
    budget: SandboxBudget = Field(default_factory=SandboxBudget)
    estimate: Estimate = Field(default_factory=Estimate)
    status: Literal["proposed", "piloting", "ready", "accepted", "running", "succeeded", "failed", "stopped",
                    "dismissed"] = "proposed"
    reason: str = ""
    progress: list[IterationStats] = []
    result: SandboxResult | None = None
    job_id: str | None = None  # the running job: the pilot's, then the run's
    accepted_by: str | None = None
    stop_requested_by: str | None = None
    phase: str | None = None  # pilot | train | heldout | demos (live)
    episodes_done: int = 0
    started_at: datetime | None = None
    finished_at: datetime | None = None
    created_at: datetime = Field(default_factory=now)
