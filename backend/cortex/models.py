"""Typed contracts shared by the REST API, the MCP adapter and the web studio.

Every service takes and returns these models, so the GUI and MCP paths go
through identical domain logic. IDs are prefixed strings ("prj_...", "ep_...")
so an agent can tell what kind of object a reference points at.
"""

from __future__ import annotations

import secrets
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, Field


def new_id(prefix: str) -> str:
    return f"{prefix}_{secrets.token_hex(6)}"


def now() -> datetime:
    return datetime.now(UTC)


# ---------------------------------------------------------------- conditions

# "agent" = a vision-language planner (self-hosted Qwen3-VL or GPT-6 Astra) driving validated IK;
# "oracle" = the scripted IK expert. "astra" is accepted as a legacy alias for "agent".
Collector = Annotated[Literal["agent", "oracle"], BeforeValidator(lambda v: "agent" if v == "astra" else v)]

CubeRegion = Literal["left", "center", "right"]
Lighting = Literal["bright", "normal", "dim"]


class Condition(BaseModel):
    """A cell of the collection-condition grid. Sim and real share it."""

    cube_region: CubeRegion = "center"
    lighting: Lighting = "normal"

    @property
    def key(self) -> str:
        return f"{self.cube_region}/{self.lighting}"


# ------------------------------------------------------------------- project

class Mode(StrEnum):
    sim = "sim"
    real = "real"


class SuccessCriterion(BaseModel):
    id: str
    description: str
    measurable: str  # how the sim/human checks it


class TaskSpec(BaseModel):
    task: str
    robot: str = "so101"
    objects: list[str]
    goal: str
    success_criteria: list[SuccessCriterion]
    condition_axes: dict[str, list[str]]
    episode_seconds: float = 8.0
    fps: int = 10


class PlanCell(BaseModel):
    condition: Condition
    target: int
    collected: int = 0
    reason: str = ""


class CollectionPlan(BaseModel):
    cells: list[PlanCell]
    budget_total: int

    @property
    def collected(self) -> int:
        return sum(c.collected for c in self.cells)


class AutopilotPolicy(BaseModel):
    """Standing approvals the user grants up front so the loop can run unattended."""

    enabled: bool = True
    max_iterations: int = 3
    target_success_rate: float = 0.8  # stop when every eval branch reaches this
    demos_per_round: int = 9
    episodes_per_branch: int = 10
    auto_approve_free_training: bool = True  # Futurama/self-hosted GPUs cost the user nothing
    collector: Collector = "agent"  # who generates demonstrations
    model_budget_usd: float = 10.0  # standing approval for planner API spend on this project
    planning_mode: Literal["camera", "privileged"] = "camera"
    max_gpu_hours_per_job: float = 1.5
    allow_paid_jobs: bool = False  # paid cloud GPUs always need an explicit click


GPU_CHOICES = {"gpu:l40s:1": "1x L40S 48 GB", "gpu:l40:1": "1x L40 48 GB", "gpu:l4:1": "1x L4 24 GB",
               "gpu:h200:1": "1x H200 141 GB (one shared node: expect to queue)"}


class InfraChoice(BaseModel):
    """Where this project's stages run. GET /api/projects/{pid}/infra lists the options and their live
    availability. Futurama jobs always use the general partition (research use), one GPU."""

    planner: Literal["auto", "claude_code", "model"] = "auto"
    training_backend: Literal["auto", "futurama_slurm", "manual"] = "auto"
    training_gres: Literal["gpu:l40s:1", "gpu:l40:1", "gpu:l4:1", "gpu:h200:1"] = "gpu:l40s:1"
    training_walltime_min: int = Field(120, ge=15, le=24 * 60)  # the general partition's limit is 1 day


class Project(BaseModel):
    id: str = Field(default_factory=lambda: new_id("prj"))
    name: str
    task_prompt: str
    mode: Mode = Mode.sim
    spec: TaskSpec
    plan: CollectionPlan
    autopilot: AutopilotPolicy = Field(default_factory=AutopilotPolicy)
    infra: InfraChoice = Field(default_factory=InfraChoice)
    created_at: datetime = Field(default_factory=now)


class CreateProjectRequest(BaseModel):
    task_prompt: str = Field(examples=["Teach a robot to pick and place a cube."])
    mode: Mode = Mode.sim
    name: str | None = None
    demo_budget: int = 24
    autopilot: AutopilotPolicy | None = None
    start_autopilot: bool = False


class AutopilotStep(BaseModel):
    at: datetime = Field(default_factory=now)
    phase: str
    message: str
    ref_ids: list[str] = []
    status: Literal["running", "done", "waiting", "failed", "skipped"] = "done"


class AutopilotRun(BaseModel):
    id: str = Field(default_factory=lambda: new_id("ap"))
    project_id: str
    status: Literal["running", "waiting_approval", "succeeded", "stopped", "failed"] = "running"
    iteration: int = 0
    phase: str = "starting"
    # explicit state machine (see services/autopilot.py) and the ids it is waiting on, for resume
    state: str = "TASK_REQUESTED"
    context: dict[str, Any] = {}
    collector: Collector = "agent"
    steps: list[AutopilotStep] = []
    best: dict[str, Any] = {}
    error: str | None = None
    created_at: datetime = Field(default_factory=now)


# ------------------------------------------------------------------- episodes

class EpisodeSource(StrEnum):
    sim_scripted = "sim_scripted"  # scripted IK oracle in MuJoCo (privileged state)
    sim_agent = "sim_agent"  # an agent planner (Claude Code / VLM) drives validated IK, in MuJoCo
    imported = "imported"  # demonstrations imported from an existing LeRobot dataset (e.g. real teleop)
    sim_teleop = "sim_teleop"
    real_teleop = "real_teleop"


class EpisodeStatus(StrEnum):
    recorded = "recorded"
    accepted = "accepted"
    rejected = "rejected"


class QualityFlag(BaseModel):
    code: str
    severity: Literal["info", "warn", "error"]
    message: str
    evidence: dict[str, Any] = {}


class QualityReport(BaseModel):
    metrics: dict[str, float]
    flags: list[QualityFlag]
    score: float  # 0..1, 1 = no issues


class Episode(BaseModel):
    id: str = Field(default_factory=lambda: new_id("ep"))
    project_id: str
    index: int
    source: EpisodeSource
    condition: Condition
    seed: int
    num_frames: int
    fps: int
    success: bool | None  # sim: measured by the env; real: human label
    status: EpisodeStatus = EpisodeStatus.recorded
    quality: QualityReport | None = None
    artifacts: dict[str, str] = {}  # kind -> storage key (video, frames, thumb)
    # who produced the actions: {"planner": "scripted_ik_oracle"} or
    # {"planner": "gpt-6-astra", "provider", "model", "mode", "attempt_id", "collection_id"}
    provenance: dict[str, Any] = {}
    created_at: datetime = Field(default_factory=now)


# ---------------------------------------------------------- analysis / VLM

class Hypothesis(BaseModel):
    category: str
    confidence: float  # model-reported, 0..1, NOT a verified probability
    rationale: str
    frame_indices: list[int] = []


class EpisodeDiagnosis(BaseModel):
    episode_id: str
    model: str
    hypotheses: list[Hypothesis]
    status: Literal["hypothesis", "human_confirmed", "human_rejected"] = "hypothesis"
    human_label: str | None = None
    error: str | None = None


class FailureCluster(BaseModel):
    id: str = Field(default_factory=lambda: new_id("fc"))
    project_id: str
    analysis_id: str
    category: str
    source: Literal["quality", "vlm", "evaluation"]
    episode_ids: list[str]
    conditions: list[str]  # Condition.key values touched
    evidence: list[str]
    uncertainty: float  # 0..1
    verified_by_human: bool = False


class Analysis(BaseModel):
    id: str = Field(default_factory=lambda: new_id("an"))
    project_id: str
    status: Literal["running", "succeeded", "failed"] = "running"
    episode_count: int = 0
    summary: str = ""
    quality: dict[str, QualityReport] = {}
    diagnoses: list[EpisodeDiagnosis] = []
    vlm_provider: str | None = None
    created_at: datetime = Field(default_factory=now)


# ----------------------------------------------------------- recommendations

class Evidence(BaseModel):
    kind: Literal["coverage", "evaluation", "quality", "vlm", "budget"]
    description: str
    ref_ids: list[str] = []
    value: float | None = None


class Recommendation(BaseModel):
    id: str = Field(default_factory=lambda: new_id("rec"))
    project_id: str
    action: Literal["collect", "augment", "train", "evaluate", "rl"]
    title: str
    condition: Condition | None = None
    count: int = 0
    score: float
    components: dict[str, float]
    evidence: list[Evidence]
    support: Literal["evidence_backed", "hypothesis"]
    status: Literal["proposed", "accepted", "dismissed", "done"] = "proposed"
    created_at: datetime = Field(default_factory=now)


# ------------------------------------------------------------------ datasets

class Augmentation(BaseModel):
    type: Literal["brightness", "contrast", "color_temperature", "shadow"]
    params: dict[str, float]


class DatasetEpisodeLink(BaseModel):
    episode_id: str  # original robot episode; augmented copies point back here
    split: Literal["train", "eval"]
    augmentation: Augmentation | None = None
    lerobot_episode_index: int


class DatasetVersion(BaseModel):
    id: str = Field(default_factory=lambda: new_id("ds"))
    project_id: str
    name: str
    parent_id: str | None = None
    links: list[DatasetEpisodeLink]
    augmentations: list[Augmentation] = []
    eval_fraction: float = 0.2
    split_seed: int = 0
    lerobot_root: str | None = None  # local LeRobotDataset v3 directory
    storage_prefix: str | None = None
    imported_from: str | None = None  # e.g. "hf:lerobot/svla_so101_pickplace" (dataset used as-is for training)
    num_frames: int = 0
    created_at: datetime = Field(default_factory=now)

    @property
    def num_source_episodes(self) -> int:
        return len({ln.episode_id for ln in self.links})


class CreateDatasetVersionRequest(BaseModel):
    name: str | None = None
    parent_id: str | None = None
    episode_ids: list[str] | None = None  # default: all accepted episodes
    sources: list[EpisodeSource] | None = None  # e.g. only sim_agent demos (A/B datasets)
    augmentations: list[Augmentation] = []
    eval_fraction: float = 0.2
    split_seed: int = 0


# ------------------------------------------------------------------ training

class JobStatus(StrEnum):
    proposed = "proposed"
    approved = "approved"
    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"
    cancelled = "cancelled"


class ResourceEstimate(BaseModel):
    gpu: str
    gpu_hours: float
    est_cost_usd: float
    paid: bool
    notes: str


class TrainingJob(BaseModel):
    id: str = Field(default_factory=lambda: new_id("tj"))
    project_id: str
    dataset_version_id: str
    policy_type: Literal["act", "smolvla"]
    backend: Literal["futurama_slurm", "hf_jobs", "manual"]
    config: dict[str, Any]
    estimate: ResourceEstimate
    rationale: str
    status: JobStatus = JobStatus.proposed
    approved_by: str | None = None
    external_id: str | None = None
    checkpoint_ids: list[str] = []
    metrics: dict[str, Any] = {}  # e.g. loss_curve [[step, loss], ...] parsed from the training log
    error: str | None = None
    created_at: datetime = Field(default_factory=now)
    updated_at: datetime = Field(default_factory=now)


class Checkpoint(BaseModel):
    id: str = Field(default_factory=lambda: new_id("ck"))
    project_id: str
    training_job_id: str | None
    policy_type: str
    step: int
    path: str  # local dir or storage key holding pretrained_model/
    metrics: dict[str, float] = {}
    created_at: datetime = Field(default_factory=now)


# ---------------------------------------------------------------- evaluation

class PolicyRef(BaseModel):
    """What to evaluate. 'scripted_expert' is the oracle demo generator and is
    labelled as such everywhere; it is a reference ceiling, not a learned policy."""

    kind: Literal["checkpoint", "scripted_expert"]
    checkpoint_id: str | None = None


class BranchSpec(BaseModel):
    name: str
    condition: Condition
    cube_offset: tuple[float, float] = (0.0, 0.0)  # extra shift in metres
    light_scale: float = 1.0


class EpisodeResult(BaseModel):
    seed: int
    success: bool
    reward: float
    steps: int
    max_cube_z: float = 0.0
    final_cube_box_dist: float = 0.0
    failure_mode: str | None = None  # measured from sim telemetry
    video_key: str | None = None


class BranchResult(BaseModel):
    policy_name: str = ""
    wall_s: float = 0.0
    n: int
    successes: int
    success_rate: float
    ci95: tuple[float, float]
    mean_reward: float
    episodes: list[EpisodeResult]


class SimulationBranch(BaseModel):
    id: str = Field(default_factory=lambda: new_id("br"))
    evaluation_id: str
    spec: BranchSpec
    seeds: list[int]
    executor: Literal["local"] = "local"
    project_id: str = ""
    sandbox_id: str | None = None
    created_at: datetime = Field(default_factory=now)
    status: JobStatus = JobStatus.queued
    result: BranchResult | None = None
    log_key: str | None = None
    error: str | None = None


class EvaluationRun(BaseModel):
    id: str = Field(default_factory=lambda: new_id("ev"))
    project_id: str
    name: str = ""
    policy: PolicyRef
    policy_label: str = ""
    executor: str = "local"
    recommendation_id: str | None = None
    dataset_version_id: str | None = None
    branch_ids: list[str] = []
    job_id: str | None = None  # poll with GET /jobs/{id} or the get_job tool
    status: JobStatus = JobStatus.queued
    created_at: datetime = Field(default_factory=now)


class LaunchEvaluationRequest(BaseModel):
    policy: PolicyRef
    branches: list[BranchSpec] | None = None  # default: baseline/shifted/lighting
    episodes_per_branch: int = 10
    base_seed: int = 1000
    executor: Literal["auto", "local"] = "auto"
    name: str | None = None
    recommendation_id: str | None = None


class ExperimentComparison(BaseModel):
    evaluation_ids: list[str]
    rows: list[dict[str, Any]]  # one per (evaluation, branch)
    deltas: list[dict[str, Any]]
    summary: str


# ---------------------------------------------------------------- next steps

class NextAction(BaseModel):
    action: str  # MCP tool / UI action to invoke next
    reason: str
    requires_approval: bool = False
    args: dict[str, Any] = {}


class ProjectState(BaseModel):
    project: Project
    episode_counts: dict[str, int]
    latest_analysis_id: str | None
    dataset_versions: list[str]
    training_jobs: list[dict[str, Any]]
    evaluations: list[dict[str, Any]]
    open_recommendations: list[Recommendation]
    next_actions: list[NextAction]
    storage_backend: str
    db_backend: str


# --------------------------------------------------------- agent demonstrations

class Scenario(BaseModel):
    cube_region: CubeRegion = "center"
    lighting: Lighting = "normal"
    cube_offset: tuple[float, float] = (0.0, 0.0)
    box_offset: tuple[float, float] = (0.0, 0.0)

    @property
    def condition(self) -> Condition:
        return Condition(cube_region=self.cube_region, lighting=self.lighting)


class StartAgentCollectionRequest(BaseModel):
    num_demos: int = Field(6, ge=1, le=50, description="attempts to run (successes become demonstrations)")
    diversity: Literal["plan", "medium", "high"] = "medium"
    conditions: list[Condition] | None = Field(None, description="restrict scenes to these cells (targeted collection)")
    max_total_usd: float | None = Field(None, gt=0, description="planner spend cap for this collection")
    max_calls_per_episode: int = Field(16, ge=4, le=40)
    max_usd_per_episode: float = Field(1.5, gt=0, le=10)
    mode: Literal["camera", "privileged"] | None = None
    collector: Collector = "agent"
    recommendation_id: str | None = None
    reason: str = ""
    base_seed: int | None = Field(None, description="fix scenes and seeds (A/B runs use the same value)")
    task: str | None = Field(None, description="override the project's task text given to the planner")
    planner: Literal["auto", "claude_code", "model"] = Field(
        "auto", description="auto = the user's Claude Code when available, else the configured model route")


class AgentCollection(BaseModel):
    id: str = Field(default_factory=lambda: new_id("ac"))
    project_id: str
    collector: Collector
    provider: str | None = None
    model: str | None = None
    mode: str = "camera"
    request: StartAgentCollectionRequest
    status: Literal["queued", "running", "succeeded", "failed", "cancelled", "budget_exhausted"] = "queued"
    job_id: str | None = None
    scenarios: list[dict[str, Any]] = []
    totals: dict[str, Any] = {}
    reason: str = ""
    error: str | None = None
    created_at: datetime = Field(default_factory=now)


class AgentAttempt(BaseModel):
    """Every attempt, successful or not. Only successes become Episodes; failures are diagnostics."""

    id: str = Field(default_factory=lambda: new_id("at"))
    project_id: str
    collection_id: str
    index: int
    collector: Collector
    seed: int
    scenario: Scenario
    status: Literal["success", "failed"]
    episode_id: str | None = None
    provider: str | None = None
    model: str | None = None
    mode: str = "camera"
    usage: dict[str, Any] = {}
    recoveries: int = 0
    counters: dict[str, Any] = {}
    num_frames: int = 0
    wall_s: float = 0.0
    model_claimed_success: bool | None = None
    stop_reason: str | None = None
    error: str | None = None
    artifacts: dict[str, str] = {}
    trace_len: int = 0
    endpoint: str | None = None  # the planner URL actually called
    created_at: datetime = Field(default_factory=now)
