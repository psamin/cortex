// Hand-written mirror of backend/cortex/models.py and docs/API.md.
// Field names must match the Pydantic models exactly. Datetimes are ISO strings.

export type ISODate = string;

// ---------------------------------------------------------------- conditions
export type CubeRegion = "left" | "center" | "right";
export type Lighting = "bright" | "normal" | "dim";
export const CUBE_REGIONS: CubeRegion[] = ["left", "center", "right"];
export const LIGHTINGS: Lighting[] = ["dim", "normal", "bright"];

export interface Condition {
  cube_region: CubeRegion;
  lighting: Lighting;
}
/** Mirrors Condition.key on the backend. */
export const conditionKey = (c: Condition) => `${c.cube_region}/${c.lighting}`;

// ------------------------------------------------------------------- project
export type Mode = "sim" | "real";

export interface SuccessCriterion {
  id: string;
  description: string;
  measurable: string;
}

export interface TaskSpec {
  task: string;
  robot: string;
  objects: string[];
  goal: string;
  success_criteria: SuccessCriterion[];
  condition_axes: Record<string, string[]>;
  episode_seconds: number;
  fps: number;
}

export interface PlanCell {
  condition: Condition;
  target: number;
  collected: number;
  reason: string;
}

export interface CollectionPlan {
  cells: PlanCell[];
  budget_total: number;
}

export interface AutopilotPolicy {
  enabled: boolean;
  max_iterations: number;
  target_success_rate: number;
  demos_per_round: number;
  episodes_per_branch: number;
  auto_approve_free_training: boolean;
  collector: Collector;
  model_budget_usd: number;
  planning_mode: PlanningMode;
  max_gpu_hours_per_job: number;
  allow_paid_jobs: boolean;
}

export type Collector = "astra" | "oracle";
export type PlanningMode = "camera" | "privileged";

export const DEFAULT_AUTOPILOT_POLICY: AutopilotPolicy = {
  enabled: true,
  max_iterations: 3,
  target_success_rate: 0.8,
  demos_per_round: 9,
  episodes_per_branch: 10,
  auto_approve_free_training: true,
  collector: "astra",
  model_budget_usd: 10,
  planning_mode: "camera",
  max_gpu_hours_per_job: 1.5,
  allow_paid_jobs: false,
};

export interface Project {
  id: string;
  name: string;
  task_prompt: string;
  mode: Mode;
  spec: TaskSpec;
  plan: CollectionPlan;
  autopilot: AutopilotPolicy;
  created_at: ISODate;
}

export interface CreateProjectRequest {
  task_prompt: string;
  mode?: Mode;
  name?: string | null;
  demo_budget?: number;
  autopilot?: AutopilotPolicy | null;
  start_autopilot?: boolean;
}

export interface AutopilotStep {
  at: ISODate;
  phase: string;
  message: string;
  ref_ids: string[];
  status: "running" | "done" | "waiting" | "failed" | "skipped";
}

export interface AutopilotRun {
  id: string;
  project_id: string;
  status: "running" | "waiting_approval" | "succeeded" | "stopped" | "failed";
  iteration: number;
  phase: string;
  /** Explicit state machine (see AUTOPILOT_STATES). */
  state: string;
  context: Record<string, unknown>;
  collector: string;
  steps: AutopilotStep[];
  best: Record<string, unknown>;
  error: string | null;
  created_at: ISODate;
}

// ------------------------------------------------------------------- episodes
export type EpisodeSource = "sim_scripted" | "sim_agent" | "sim_teleop" | "real_teleop";

/** Who produced an episode's actions. planner: "gpt-6-astra" | "scripted_ik_oracle". */
export interface EpisodeProvenance {
  planner?: string;
  provider?: string | null;
  model?: string | null;
  mode?: string;
  attempt_id?: string;
  collection_id?: string;
  scenario?: Scenario;
  [k: string]: unknown;
}
export type EpisodeStatus = "recorded" | "accepted" | "rejected";

export interface QualityFlag {
  code: string;
  severity: "info" | "warn" | "error";
  message: string;
  evidence: Record<string, unknown>;
}

export interface QualityReport {
  metrics: Record<string, number>;
  flags: QualityFlag[];
  score: number;
}

export interface Episode {
  id: string;
  project_id: string;
  index: number;
  source: EpisodeSource;
  condition: Condition;
  seed: number;
  num_frames: number;
  fps: number;
  success: boolean | null;
  status: EpisodeStatus;
  quality: QualityReport | null;
  artifacts: Record<string, string>;
  provenance: EpisodeProvenance;
  created_at: ISODate;
}

export interface EpisodeMedia {
  video_url: string | null;
  thumb_url: string | null;
  frames: number;
}

// ---------------------------------------------------------- analysis / VLM
export interface Hypothesis {
  category: string;
  confidence: number;
  rationale: string;
  frame_indices: number[];
}

export type DiagnosisStatus = "hypothesis" | "human_confirmed" | "human_rejected";

export interface EpisodeDiagnosis {
  episode_id: string;
  model: string;
  hypotheses: Hypothesis[];
  status: DiagnosisStatus;
  human_label: string | null;
  error: string | null;
}

export interface FailureCluster {
  id: string;
  project_id: string;
  analysis_id: string;
  category: string;
  source: "quality" | "vlm" | "evaluation";
  episode_ids: string[];
  conditions: string[];
  evidence: string[];
  uncertainty: number;
  verified_by_human: boolean;
}

export interface Analysis {
  id: string;
  project_id: string;
  status: "running" | "succeeded" | "failed";
  episode_count: number;
  summary: string;
  quality: Record<string, QualityReport>;
  diagnoses: EpisodeDiagnosis[];
  vlm_provider: string | null;
  created_at: ISODate;
}

// ----------------------------------------------------------- recommendations
export interface Evidence {
  kind: "coverage" | "evaluation" | "quality" | "vlm" | "budget";
  description: string;
  ref_ids: string[];
  value: number | null;
}

export type RecommendationStatus = "proposed" | "accepted" | "dismissed" | "done";

export interface Recommendation {
  id: string;
  project_id: string;
  action: "collect" | "augment" | "train" | "evaluate" | "rl";
  title: string;
  condition: Condition | null;
  count: number;
  score: number;
  components: Record<string, number>;
  evidence: Evidence[];
  support: "evidence_backed" | "hypothesis";
  status: RecommendationStatus;
  created_at: ISODate;
}

// ------------------------------------------------------------------ datasets
export type AugmentationType = "brightness" | "contrast" | "color_temperature" | "shadow";

export interface Augmentation {
  type: AugmentationType;
  params: Record<string, number>;
}

export interface DatasetEpisodeLink {
  episode_id: string;
  split: "train" | "eval";
  augmentation: Augmentation | null;
  lerobot_episode_index: number;
}

export interface DatasetVersion {
  id: string;
  project_id: string;
  name: string;
  parent_id: string | null;
  links: DatasetEpisodeLink[];
  augmentations: Augmentation[];
  eval_fraction: number;
  split_seed: number;
  lerobot_root: string | null;
  storage_prefix: string | null;
  num_frames: number;
  created_at: ISODate;
}

export interface CreateDatasetVersionRequest {
  name?: string | null;
  parent_id?: string | null;
  episode_ids?: string[] | null;
  augmentations?: Augmentation[];
  eval_fraction?: number;
  split_seed?: number;
}

// ------------------------------------------------------------------ training
export type JobStatus =
  | "proposed"
  | "approved"
  | "queued"
  | "running"
  | "succeeded"
  | "failed"
  | "cancelled";

export interface ResourceEstimate {
  gpu: string;
  gpu_hours: number;
  est_cost_usd: number;
  paid: boolean;
  notes: string;
}

export type PolicyType = "act" | "smolvla";

export interface TrainingJob {
  id: string;
  project_id: string;
  dataset_version_id: string;
  policy_type: PolicyType;
  backend: "futurama_slurm" | "hf_jobs" | "manual";
  config: Record<string, unknown>;
  estimate: ResourceEstimate;
  rationale: string;
  status: JobStatus;
  approved_by: string | null;
  external_id: string | null;
  checkpoint_ids: string[];
  error: string | null;
  created_at: ISODate;
  updated_at: ISODate;
}

export interface Checkpoint {
  id: string;
  project_id: string;
  training_job_id: string | null;
  policy_type: string;
  step: number;
  path: string;
  metrics: Record<string, number>;
  created_at: ISODate;
}

// ---------------------------------------------------------------- evaluation
export interface PolicyRef {
  kind: "checkpoint" | "scripted_expert";
  checkpoint_id?: string | null;
}

export interface BranchSpec {
  name: string;
  condition: Condition;
  cube_offset: [number, number];
  light_scale: number;
}

export interface EpisodeResult {
  seed: number;
  success: boolean;
  reward: number;
  steps: number;
  max_cube_z: number;
  final_cube_box_dist: number;
  failure_mode: string | null;
  video_key: string | null;
}

export interface BranchResult {
  policy_name: string;
  wall_s: number;
  n: number;
  successes: number;
  success_rate: number;
  ci95: [number, number];
  mean_reward: number;
  episodes: EpisodeResult[];
}

export interface SimulationBranch {
  id: string;
  evaluation_id: string;
  spec: BranchSpec;
  seeds: number[];
  executor: string;
  project_id: string;
  sandbox_id: string | null;
  created_at: ISODate;
  status: JobStatus;
  result: BranchResult | null;
  log_key: string | null;
  error: string | null;
}

export interface EvaluationRun {
  id: string;
  project_id: string;
  name: string;
  policy: PolicyRef;
  policy_label: string;
  executor: string;
  recommendation_id: string | null;
  dataset_version_id: string | null;
  branch_ids: string[];
  status: JobStatus;
  created_at: ISODate;
}

export interface EvaluationWithBranches {
  run: EvaluationRun;
  branches: SimulationBranch[];
}

export interface LaunchEvaluationRequest {
  policy: PolicyRef;
  branches?: BranchSpec[] | null;
  episodes_per_branch?: number;
  base_seed?: number;
  name?: string | null;
  recommendation_id?: string | null;
}

export interface ExperimentComparison {
  evaluation_ids: string[];
  rows: Record<string, unknown>[];
  deltas: Record<string, unknown>[];
  summary: string;
}

// ---------------------------------------------------------------- next steps
export interface NextAction {
  action: string;
  reason: string;
  requires_approval: boolean;
  args: Record<string, unknown>;
}

export interface ProjectState {
  project: Project;
  episode_counts: Record<string, number>;
  latest_analysis_id: string | null;
  dataset_versions: string[];
  training_jobs: Record<string, unknown>[];
  evaluations: Record<string, unknown>[];
  open_recommendations: Recommendation[];
  next_actions: NextAction[];
  storage_backend: string;
  db_backend: string;
}

// ------------------------------------------------------------- jobs / events
/** From docs/API.md (no Pydantic model). "cancelled" is returned by /recordings/{id}/stop. */
export interface Job {
  id: string;
  kind: string;
  project_id: string;
  status: "queued" | "running" | "succeeded" | "failed" | "cancelled";
  progress: number;
  message: string;
  result_ref?: string | null;
  error?: string | null;
  idempotency_key?: string | null;
  created_at?: ISODate;
}

export interface EntityEvent {
  table: string;
  id: string;
}

export interface LogEvent {
  job_id: string;
  line: string;
}

export interface AuditRow {
  actor: "gui" | "mcp" | "agent" | "autopilot" | string;
  tool: string;
  args: Record<string, unknown>;
  status: string;
  created_at: ISODate;
}

// ---------------------------------------------------------------- health
export interface ProviderInfo {
  provider: string;
  model?: string | null;
  fallback: boolean;
}

export interface Health {
  db: ProviderInfo;
  storage: ProviderInfo;
  vlm: ProviderInfo;
  agent: ProviderInfo;
  executor: ProviderInfo;
  training: ProviderInfo;
  lerobot_version: string | null;
  sim_env: string | null;
  notes: string[];
}

// ---------------------------------------------------------------- agent
export type AgentEvent =
  | { type: "text"; delta: string }
  | { type: "tool_call"; tool: string; args: Record<string, unknown> }
  | { type: "tool_result"; tool: string; summary: string; ref_ids: string[] }
  | { type: "approval_required"; kind: string; ref_id: string; summary: string; args: Record<string, unknown> }
  | { type: "done" }
  | { type: "error"; message: string };

export interface AgentHistoryItem {
  role: string;
  content: string;
  tool?: string | null;
}

// ---------------------------------------------------------------- lineage graph
export type GraphNodeType =
  | "project"
  | "condition"
  | "episode"
  | "analysis"
  | "cluster"
  | "dataset"
  | "training"
  | "checkpoint"
  | "evaluation"
  | "branch"
  | "recommendation"
  | "autopilot";

export interface GraphNode {
  id: string;
  type: GraphNodeType | string;
  label: string;
  status: string | null;
  /** Where this row lives: e.g. backend "insforge postgres" or "sqlite (local fallback)". */
  source: { backend: string; table: string | null; row_id: string | null };
  meta: Record<string, unknown>;
}

export interface GraphEdge {
  source: string;
  target: string;
  kind: string;
}

export interface ProjectGraph {
  nodes: GraphNode[];
  edges: GraphEdge[];
  backend: { db: string; storage: string };
}

// ---------------------------------------------------------------- agent demonstrations
/** Order from backend/cortex/services/autopilot.py STATES. The loop returns to SCENARIO_PLANNED. */
export const AUTOPILOT_STATES = [
  "TASK_REQUESTED",
  "SCENARIO_PLANNED",
  "AGENT_COLLECTING",
  "EPISODES_VALIDATED",
  "DATASET_VERSIONED",
  "TRAINING_REQUESTED",
  "POLICY_TRAINED",
  "POLICY_EVALUATED",
  "FAILURES_ANALYZED",
  "NEXT_EXPERIMENT_SELECTED",
  "DONE",
] as const;

export interface Scenario {
  cube_region: CubeRegion;
  lighting: Lighting;
  cube_offset: [number, number];
  box_offset: [number, number];
}

export type PlannerStatus =
  | {
      available: true;
      model: string;
      provider: string;
      route: string;
      pricing_per_m_tokens: { input: number; output: number };
    }
  | { available: false; model?: string; provider?: null; reason: string };

export interface StartAgentCollectionRequest {
  num_demos: number;
  /** Planner task override; omitted = project spec goal. */
  task?: string | null;
  diversity: "plan" | "medium" | "high";
  conditions?: Condition[] | null;
  max_total_usd?: number | null;
  max_calls_per_episode?: number;
  max_usd_per_episode?: number;
  mode?: PlanningMode | null;
  collector: Collector;
  recommendation_id?: string | null;
  reason?: string;
}

export interface CollectionTotals {
  attempts: number;
  successes: number;
  failures: number;
  model_calls: number;
  cost_usd: number;
  recoveries: number;
  tool_errors: number;
  sim_seconds: number;
  wall_s: number;
  cells: Record<string, { attempts: number; successes: number }>;
  budget_cap_usd: number;
  stopped_reason?: string;
}

export interface AgentCollection {
  id: string;
  project_id: string;
  collector: Collector;
  provider: string | null;
  model: string | null;
  mode: string;
  request: StartAgentCollectionRequest;
  status: "queued" | "running" | "succeeded" | "failed" | "cancelled" | "budget_exhausted";
  job_id: string | null;
  scenarios: (Scenario & { seed: number })[];
  totals: Partial<CollectionTotals>;
  reason: string;
  error: string | null;
  created_at: ISODate;
}

export interface AgentAttempt {
  id: string;
  project_id: string;
  collection_id: string;
  index: number;
  collector: string;
  seed: number;
  scenario: Scenario;
  status: "success" | "failed";
  episode_id: string | null;
  provider: string | null;
  model: string | null;
  mode: string;
  usage: { calls?: number; cost_usd?: number; [k: string]: unknown };
  recoveries: number;
  counters: Record<string, unknown>;
  num_frames: number;
  wall_s: number;
  model_claimed_success: boolean | null;
  stop_reason: string | null;
  error: string | null;
  artifacts: Record<string, string>;
  trace_len: number;
  endpoint: string | null;
  created_at: ISODate;
}

/** One validated tool call. A rejected command has ok=false and result.error {code, message, details?}. */
export interface TraceStep {
  i: number;
  tool: string;
  args: Record<string, unknown>;
  phase: string;
  ok: boolean;
  result: Record<string, unknown> & {
    error?: { code?: string; message?: unknown; details?: { loc?: unknown[]; msg?: string }[]; [k: string]: unknown };
  };
  sim_step: number;
  model_call: number;
}

export interface AttemptDetail {
  attempt: AgentAttempt;
  trace: TraceStep[];
  model_summary: string | null;
  media: { front?: string; thumb?: string };
}

export interface CollectionDetail {
  collection: AgentCollection;
  attempts: AgentAttempt[];
  job: Job | null;
}

export interface CollectorStats {
  attempts: number;
  successes: number;
  success_rate: number;
  recoveries: number;
  mean_sim_seconds: number;
  mean_wall_s: number;
  model_calls: number;
  cost_usd: number;
  cells_covered: number;
}

export interface CollectorComparison {
  astra?: CollectorStats;
  oracle?: CollectorStats;
}

/** SSE `agent_step` payloads. */
export type AgentStepEvent =
  | { collection_id: string; type: "attempt_start"; index: number; total: number; seed: number; scenario: Scenario }
  | { collection_id: string; type: "episode_start"; index: number; total: number; seed: number; condition: Condition }
  | ({ collection_id: string; type: "agent_step"; index: number } & TraceStep)
  | { collection_id: string; type: "agent_thought"; index: number; text: string };

/** GET /agent-collections/{cid}/live: the in-progress attempt, for reconnecting viewers. */
export interface CollectionLive {
  index: number | null;
  start: { type: "attempt_start"; index: number; total: number; seed: number; scenario: Scenario } | null;
  steps: (({ type: "agent_step"; index: number } & TraceStep) | { type: "agent_thought"; index: number; text: string })[];
}

export interface PlannerBudget {
  budget_usd: number;
  spent_usd: number;
  remaining_usd: number;
}
