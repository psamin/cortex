"""Contracts for RL sandboxes (#80): weaknesses, sandbox specs, estimates and progress.

Everything Cortex predicts carries ``kind="estimate"`` and lists its assumptions. Only
held-out simulator evaluations are "measured".
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from ..models import Condition, new_id, now

FailureMode = Literal["grasp_failed", "dropped_before_target", "released_outside_target", "other"]


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
    score: float  # ranking: gap x confidence (1 - CI width)
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
    """Domain randomization centred on the weak condition."""

    condition: Condition
    cube_offset: tuple[float, float] = (0.0, 0.0)
    offset_jitter_m: float = 0.02
    yaw_jitter_rad: float = 0.6
    light_scales: tuple[float, float] = (0.8, 1.2)
    friction_jitter: float = 0.2
    mass_jitter: float = 0.2
    scene_digest: str | None = None  # the intake scene when one is built; None = the upstream tabletop


class SandboxBudget(BaseModel):
    iterations: int = Field(8, ge=1, le=100)
    population: int = Field(12, ge=2, le=128)
    episodes_per_candidate: int = Field(2, ge=1, le=20)
    heldout_episodes: int = Field(20, ge=5, le=200)
    max_parallel: int = Field(2, ge=1, le=16)
    executor: Literal["local_cpu", "futurama_gpu"] = "local_cpu"

    @property
    def train_episodes(self) -> int:
        return self.iterations * self.population * self.episodes_per_candidate


class Estimate(BaseModel):
    kind: Literal["estimate"] = "estimate"
    pilot_episodes: int | None = None  # measured base-skill episodes in the sandbox
    pilot_success: float | None = None
    pilot_ci95: tuple[float, float] | None = None
    sandbox_success_after: tuple[float, float] | None = None  # range
    policy_uplift: tuple[float, float] | None = None  # absolute success-rate points on the weak branch
    policy_success_after: tuple[float, float] | None = None
    demos_expected: int | None = None
    sim_steps_per_s: float | None = None  # measured throughput
    cpu_minutes: float | None = None
    wall_minutes: float | None = None
    usd: float = 0.0
    gpu_hours: float = 0.0
    assumptions: list[str] = []


class IterationStats(BaseModel):
    iteration: int
    episodes: int
    mean_return: float
    success_rate: float
    ci95: tuple[float, float]
    best_return: float
    best_params: dict[str, float]
    elapsed_s: float


class SandboxResult(BaseModel):
    kind: Literal["measured"] = "measured"
    heldout_seeds: list[int]
    base_success: float
    base_ci95: tuple[float, float]
    tuned_success: float
    tuned_ci95: tuple[float, float]
    p_value: float | None = None
    params: dict[str, float]
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
    job_id: str | None = None
    accepted_by: str | None = None
    created_at: datetime = Field(default_factory=now)
