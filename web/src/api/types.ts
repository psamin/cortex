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
  /** autonomous: the chat agent keeps going and training inside this policy starts without a click; manual: one step per turn */
  autonomy: "autonomous" | "manual";
}

/** "agent" = a planning model drives the arm; "oracle" = scripted IK expert. "astra" is a legacy alias for "agent". */
export type Collector = "agent" | "oracle";
export const isAgent = (c: string | null | undefined) => c === "agent" || c === "astra";

/** People-facing name of the planning model behind the agent collector. */
export function agentLabel(model?: string | null): string {
  if (!model || model === "agent" || model === "astra") return "Agent";
  if (/claude[-_]?code/i.test(model)) return "Claude Code agent";
  if (/gpt-6-astra/i.test(model)) return "GPT-6 Astra";
  return model;
}
export type PlanningMode = "camera" | "privileged";

export const DEFAULT_AUTOPILOT_POLICY: AutopilotPolicy = {
  enabled: true,
  max_iterations: 3,
  target_success_rate: 0.8,
  demos_per_round: 9,
  episodes_per_branch: 10,
  auto_approve_free_training: true,
  collector: "agent",
  model_budget_usd: 10,
  planning_mode: "camera",
  max_gpu_hours_per_job: 1.5,
  allow_paid_jobs: false,
  autonomy: "autonomous",
};

export interface Project {
  id: string;
  name: string;
  task_prompt: string;
  mode: Mode;
  spec: TaskSpec;
  plan: CollectionPlan;
  autopilot: AutopilotPolicy;
  /** the sim scene built from the project's intake (build_simulation); absent for the built-in cube tabletop */
  scene?: { digest: string } | null;
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
export type EpisodeSource = "sim_scripted" | "sim_agent" | "sim_teleop" | "real_teleop" | "imported";

/** Who produced an episode's actions. planner: the agent's model (e.g. "claude-code") | "scripted_ik_oracle" | "human demonstration (imported)". */
export interface EpisodeProvenance {
  planner?: string;
  provider?: string | null;
  model?: string | null;
  mode?: string;
  attempt_id?: string;
  collection_id?: string;
  scenario?: Scenario;
  /** imported demos: "hf:<repo>" | "upload:<name>", the dataset's task text, camera keys */
  source?: string;
  task?: string;
  cameras?: string[];
  [k: string]: unknown;
}

/** Imported human demonstrations carry no simulator verdict and no known scene condition. */
export const isImported = (e: { source?: string; provenance?: { planner?: string } }) =>
  e.source === "imported" || /imported/i.test(e.provenance?.planner ?? "");
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
  /** why a failed job failed, classified by the backend (lib/trainingFailure.ts has the wording) */
  failure_kind?: string | null;
  created_at: ISODate;
  updated_at: ISODate;
  /** live training metrics; `loss_curve` is [[step, loss], ...] parsed from the trainer log */
  metrics?: { loss_curve?: [number, number][] } & Record<string, unknown>;
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
  /** technical text for internal errors; `error` is the readable sentence */
  error_detail?: string | null;
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
      /** numbers per million tokens, or a sentence (e.g. Claude Code subscription) */
      pricing_per_m_tokens: { input: number; output: number } | string;
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
  agent?: CollectorStats;
  /** legacy key */
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

// ---------------------------------------------------------------- viewer / pipeline (new endpoints; may 404)
export interface RemoteRollout {
  /** replay: a trained policy run open loop on a held-out recorded demo (joint error, never success),
   *  or (replay_kind live_sim) one live closed-loop sim run, whose success the simulator judged */
  kind: "demo" | "eval" | "attempt" | "replay";
  replay_kind?: "held_out_replay" | "live_inference" | "live_sim";
  /** replays: "0.12° joint error · 41 ms/frame"; live sim: "212 steps · 18 ms/frame" */
  metric?: string | null;
  partial?: boolean;
  /** replays: frames the policy acted on */
  frames?: number | null;
  /** failed attempts: why it failed (may be technical) */
  failure?: string | null;
  id: string;
  label: string;
  success: boolean | null;
  failure_mode?: string | null;
  /** cameras recorded for this rollout (id drives the video URL; label/detail are for people) */
  cameras?: { id: string; label: string; detail?: string }[] | string[];
  /** "left/normal" (Condition.key) on the live backend; an object is also accepted */
  condition?: Condition | string | null;
  seed?: number | null;
  video_url?: string | null;
  policy?: string | null;
  planner?: string | null;
  branch?: string | null;
  evaluation_id?: string | null;
  iteration?: number | null;
  created_at?: ISODate;
}

export interface IterationBranch {
  name: string;
  successes: number;
  n: number;
  success_rate: number;
  ci95: [number, number];
  failure_modes: Record<string, number>;
}

export interface Iteration {
  iteration: number;
  policy_version: string;
  checkpoint_id: string | null;
  demos_accepted: number;
  dataset_id: string | null;
  training: { id: string; status: string; backend: string; policy_type?: string; gpu_hours_est?: number; loss_curve: [number, number][]; error?: string | null } | null;
  evaluation: { id: string; branches: IterationBranch[] } | null;
  next: { reason: string; conditions: Condition[]; collection_id: string | null } | null;
  /** open-loop replays of this checkpoint on held-out recorded demos: joint error, not a success rate */
  replay?: { id: string | null; episodes: number; joint_mae_deg: number | null; live_sessions: number; inference_ms: number | null; note: string } | null;
}

// ---------------------------------------------------------------- real SO-101 (local studio only)
export interface HardwarePort {
  device: string;
  description: string;
  usb: boolean;
}

export interface CameraSpec {
  name: string;
  index_or_path: number | string;
  width: number;
  height: number;
  fps: number;
}

export interface HardwareConfig {
  follower_port: string | null;
  follower_id: string;
  leader_port: string | null;
  leader_id: string;
  cameras: CameraSpec[];
  max_step_deg: number;
}

export interface HardwareStatus {
  enabled: boolean;
  driver_error: string | null;
  ports: HardwarePort[];
  config: HardwareConfig;
  calibration: Record<"follower" | "leader", { id: string; path: string; exists: boolean }>;
  armed: { operator: string; estop_confirmed: boolean; armed_at: ISODate; expires_at: ISODate } | null;
  ready_for_motion: boolean;
  missing: string[];
  teleop_missing: string[];
  note: string;
}

/** lerobot-calibrate subprocess (torque off; the operator moves the joints by hand). */
export interface HardwareProcess {
  kind: string;
  state: "running" | "waiting_for_operator" | "finished" | "failed" | string;
  prompt: string | null;
  lines: string[];
  joint_ranges: Record<string, { min: number; pos: number; max: number }>;
  error: string | null;
  returncode: number | null;
}

export interface HardwareRun {
  run_id: string;
  project_id: string;
  kind: "teleop" | "policy";
  state: "running" | "holding" | "finished";
  warnings: string[];
  live: { joints?: Record<string, number>; target?: Record<string, number>; ticks?: number; episode?: number } | null;
  holding: { reason?: string; note?: string } | null;
  events: { type?: string; [k: string]: unknown }[];
  result: Record<string, unknown> | null;
  import_job_id: string | null;
}

// ---------------------------------------------------------------- infrastructure choices (per project)
export interface InfraOption {
  id: string;
  label: string;
  available: boolean;
  reason: string | null;
}

export interface InfraChoice {
  planner: string;
  training_backend: string;
  training_gres: string;
  training_walltime_min: number;
}

export interface Infra {
  choice: InfraChoice;
  planner: InfraOption[];
  training_backend: InfraOption[];
  gpus: InfraOption[];
  evaluation: InfraOption[];
  robot: InfraOption[];
  walltime_min: { min: number; max: number; note: string };
}

/** A past chat in a project (title = first user message). */
export interface AgentThread {
  id: string;
  title: string;
  created_at: ISODate;
  updated_at: ISODate;
}

/** GET /projects/{pid}/agent/state: whether a studio agent turn is really running in the backend (and its live
 *  line), one about to resume, a chain waiting on background work, or a pause that needs the Continue button. */
export interface AgentState {
  running: boolean;
  thread: string | null;
  status: string | null;
  resuming: boolean;
  waiting: boolean;
  paused: { thread: string; reason: string; at: number } | null;
}

/** GET /api/live-policy: the Futurama GPU server running trained policies (#157). */
export interface LivePolicyStatus {
  available: boolean;
  phase: "stopped" | "starting" | "ready";
  /** the current step while a GPU comes up, and every step's label in order */
  step?: string | null;
  steps?: Record<string, string>;
  reason?: string;
  error?: string | null;
  job?: string | null;
  gpu?: string;
  policies?: Record<string, { checkpoint: string; stage: string; held_out: number[] }>;
  /** sim-trained policies, run closed loop in MuJoCo on the GPU (best evaluated checkpoint per task) */
  sim?: Record<string, { variant: string; step: number; success_rate: number | null; eval_episodes: number; eval_seeds: string; instruction: string; layouts?: number | null }>;
  /** the project holding each served task (<task>, or sim_<task>), from infra/futurama/projects.json */
  projects?: Record<string, string>;
}

// ------------------------------------------------------------------ RL sandboxes (#80)
export interface Interval {
  mean: number;
  ci90?: [number, number];
  source?: "measured" | "prior" | "learned";
}

export interface RLProposal {
  id: string;
  condition: string;
  policy_success: number;
  dominant_failure?: string | null;
  predicted_policy_uplift?: Interval | null;
  predicted_sandbox_success?: Interval | null;
  p_improve_ge_delta?: number | null;
  cost?: { episodes: number; cpu_hours: number; gpu_hours: number; wall_clock_est_min: number; usd: number } | null;
  executor?: string;
  reason?: string;
}

export interface RLSandboxView {
  id: string;
  status: "proposed" | "piloting" | "ready" | "accepted" | "running" | "succeeded" | "failed" | "stopped" | "dismissed";
  phase?: string;
  reason?: string;
  weakness_id: string;
  method: string;
  iteration: number;
  max_iterations: number;
  episodes: number;
  success?: { rate: number; ci95: [number, number]; elite_rate?: number } | null;
  curve: { iteration: number; success_rate: number; ci95: [number, number]; elite_success_rate?: number }[];
  elapsed_s: number;
  cost_so_far?: { cpu_hours: number; usd: number };
  estimate: RLProposal;
  result?: { base_success: number; base_ci95: [number, number]; tuned_success: number; tuned_ci95: [number, number]; p_value?: number | null; demo_episode_ids: string[]; stop_reason?: string } | null;
  live_frame?: string;
  job_id?: string | null;
}

// ------------------------------------------------------------------ Physical Skills (#276)
/** What a skill's score measured. joint_error is imitation on held-out real demos, never a success rate. */
export interface SkillScore {
  kind: "sim_success" | "pending" | "joint_error" | "none";
  value: number | null;
  n: number | null;
  successes?: number;
  ci95?: [number, number] | null;
  layouts?: number | null;
  label: string;
  detail: string;
}

/** How to run a skill on the live GPU server: a held-out real demo (real) or a new sim layout (sim). */
export interface SkillRun {
  kind: "real" | "sim" | null;
  task: string | null;
  episodes: number[];
  live_now: boolean;
  note: string | null;
}

export interface SkillPastRun {
  id: string;
  kind: "live_inference" | "live_sim";
  label: string;
  when: string;
  partial: boolean;
  success: boolean | null;
  joint_mae_deg: number | null;
  steps: number | null;
  inference_ms: number | null;
  gpu: string | null;
  video_url: string | null;
  note: string;
}

export interface SkillEvidence {
  label: string;
  success?: boolean;
  steps?: number;
  joint_mae_deg?: number | null;
  video_url: string | null;
}

/** One trained policy, from GET /api/skills. */
export interface PhysicalSkill {
  id: string;
  name: string;
  task: string | null;
  data: "sim" | "real";
  instruction: string;
  robot: string;
  project_id: string | null;
  project_name: string | null;
  policy_type: string;
  policy: string;
  variant: string;
  steps: number | null;
  /** sim policies trained and scored on a few practice layouts (not unseen ones); null otherwise */
  layouts: number | null;
  trained_where: string | null;
  trained_at: string | null;
  weights: string;
  checkpoint_id: string | null;
  checkpoint: string;
  training_job_id: string | null;
  dataset_version_id: string | null;
  slurm_job: string | null;
  score: SkillScore;
  run: SkillRun;
  runs: SkillPastRun[];
  evidence: SkillEvidence[];
  thumbnail_url: string | null;
  /** recorded replay videos of a sim skill, newest first (#308); empty for real-demo policies. Missing from a
   *  backend older than #308. */
  replays?: string[];
  /** on the live GPU server now (missing from a backend older than #308) */
  live?: boolean;
}
