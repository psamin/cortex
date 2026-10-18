import type {
  AgentAttempt,
  AgentCollection,
  AttemptDetail,
  CollectionDetail,
  CollectionLive,
  PlannerBudget,
  CollectorComparison,
  PlannerStatus,
  StartAgentCollectionRequest,
  Analysis,
  AuditRow,
  AutopilotPolicy,
  AutopilotRun,
  Checkpoint,
  Condition,
  CreateDatasetVersionRequest,
  CreateProjectRequest,
  DiagnosisStatus,
  Episode,
  EpisodeDiagnosis,
  EpisodeMedia,
  EpisodeStatus,
  EvaluationWithBranches,
  ExperimentComparison,
  FailureCluster,
  Health,
  Job,
  LaunchEvaluationRequest,
  NextAction,
  PolicyType,
  HardwareConfig,
  Infra,
  InfraChoice,
  HardwareProcess,
  HardwareRun,
  HardwareStatus,
  RemoteRollout,
  Iteration,
  Project,
  ProjectState,
  Recommendation,
  RecommendationStatus,
  TrainingJob,
  DatasetVersion,
} from "./types";

export const API_BASE: string = (import.meta.env.VITE_API_URL as string | undefined)?.replace(/\/$/, "") ||
  "http://127.0.0.1:8000/api";

/** Every request identifies the studio so the audit log can attribute it. */
const ACTOR_HEADER = { "X-Cortex-Actor": "gui" } as const;

/** Origin of the API (for resolving root-relative media URLs). */
export const API_ORIGIN = new URL(API_BASE).origin;

export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
    public detail?: unknown,
    public body?: unknown,
  ) {
    super(message);
  }
  /** Machine-readable error code from the body, e.g. "planner_budget", "planner_unavailable". */
  get code(): string | undefined {
    const b = this.body as { code?: unknown } | undefined;
    return typeof b?.code === "string" ? b.code : undefined;
  }
  /** Technical text the backend keeps out of the headline (`error_detail`), for a "Details" toggle. */
  get technical(): string | undefined {
    const b = this.body as { error_detail?: unknown } | undefined;
    return typeof b?.error_detail === "string" && b.error_detail ? b.error_detail : undefined;
  }
  /** True when the backend could not be reached at all (network / CORS). */
  get unreachable() {
    return this.status === 0;
  }
}

async function request<T>(method: string, path: string, body?: unknown): Promise<T> {
  let res: Response;
  try {
    res = await fetch(API_BASE + path, {
      method,
      headers: body !== undefined ? { ...ACTOR_HEADER, "Content-Type": "application/json" } : ACTOR_HEADER,
      body: body !== undefined ? JSON.stringify(body) : undefined,
    });
  } catch {
    throw new ApiError(`Cannot reach the Cortex API at ${API_BASE}`, 0);
  }
  const text = await res.text();
  let data: unknown = undefined;
  if (text) {
    try {
      data = JSON.parse(text);
    } catch {
      data = text;
    }
  }
  if (!res.ok) {
    const detail = (data as { detail?: unknown } | undefined)?.detail ?? data;
    const msg = typeof detail === "string" ? detail : `${method} ${path} failed (${res.status})`;
    throw new ApiError(msg, res.status, detail, data);
  }
  return data as T;
}

const get = <T>(p: string) => request<T>("GET", p);
const post = <T>(p: string, b: unknown = {}) => request<T>("POST", p, b);
const patch = <T>(p: string, b: unknown) => request<T>("PATCH", p, b);
const put = <T>(p: string, b: unknown) => request<T>("PUT", p, b);

/** Media URLs from the backend may be absolute, root-relative, or relative to the API base. */
export function resolveMediaUrl(u: string | null | undefined): string | null {
  if (!u) return null;
  if (/^(https?:|data:|blob:)/.test(u)) return u;
  if (u.startsWith("/")) return API_ORIGIN + u;
  return `${API_BASE}/${u}`;
}

export const api = {
  health: () => get<Health>("/health"),

  // projects
  listProjects: () => get<Project[]>("/projects"),
  createProject: (req: CreateProjectRequest) => post<Project>("/projects", req),
  projectState: (pid: string) => get<ProjectState>(`/projects/${pid}/state`),
  audit: (pid: string) => get<AuditRow[]>(`/projects/${pid}/audit`),
  jobs: (pid: string) => get<Job[]>(`/projects/${pid}/jobs`),
  /** May 404 until the backend ships it; callers fall back to data assembled from episodes + evaluations. */
  rollouts: (pid: string) => get<RemoteRollout[]>(`/projects/${pid}/rollouts`),
  iterations: (pid: string) => get<Iteration[]>(`/projects/${pid}/iterations`),
  cameraUrl: (eid: string, cam: "front" | "wrist") => `${API_BASE}/episodes/${eid}/camera/${cam}.mp4`,
  job: (id: string) => get<Job>(`/jobs/${id}`),
  /** Resolve a storage key (e.g. EpisodeResult.video_key) to a browser URL. */
  media: (key: string) => get<{ url: string }>(`/media?key=${encodeURIComponent(key)}`),

  infra: (pid: string) => get<Infra>(`/projects/${pid}/infra`),
  saveInfra: (pid: string, choice: Partial<InfraChoice>) => put<Infra>(`/projects/${pid}/infra`, choice),

  // real SO-101 (local studio only; refusals are 403 {detail, code: "hardware_not_ready"})
  hardware: () => get<HardwareStatus>("/hardware"),
  saveHardwareConfig: (cfg: HardwareConfig) => put<HardwareStatus>("/hardware/config", cfg),
  arm: (operator: string) => post<HardwareStatus>("/hardware/arm", { operator, estop_confirmed: true }),
  disarm: () => post<HardwareStatus>("/hardware/disarm"),
  calibrate: (device: "follower" | "leader") => post<unknown>("/hardware/calibrate", { device }),
  hardwareProcess: () => get<HardwareProcess | null>("/hardware/process"),
  hardwareProcessInput: (text: "" | "c") => post<HardwareProcess | null>("/hardware/process/input", { text }),
  hardwareProcessStop: () => post<HardwareProcess | null>("/hardware/process/stop"),
  startTeleop: (pid: string, episodes: number, episode_s: number, task?: string) =>
    post<HardwareRun>(`/projects/${pid}/hardware/teleop`, { episodes, episode_s, task: task || undefined, operator_confirmed: true }),
  startPolicyRun: (pid: string, checkpoint_id: string, episodes: number, episode_s: number) =>
    post<HardwareRun>(`/projects/${pid}/hardware/policy`, { checkpoint_id, episodes, episode_s, operator_confirmed: true }),
  hardwareRun: () => get<HardwareRun | null>("/hardware/run"),
  hardwareRunOp: (op: "heartbeat" | "end_episode" | "stop" | "release") => post<unknown>(`/hardware/run/${op}`),

  // agent demonstrations
  planner: () => get<PlannerStatus>("/planner"),
  startAgentCollection: (pid: string, req: StartAgentCollectionRequest) =>
    post<AgentCollection>(`/projects/${pid}/agent-collections`, req),
  agentCollections: (pid: string) => get<AgentCollection[]>(`/projects/${pid}/agent-collections`),
  agentCollection: (cid: string) => get<CollectionDetail>(`/agent-collections/${cid}`),
  stopAgentCollection: (cid: string) => post<AgentCollection>(`/agent-collections/${cid}/stop`),
  agentCollectionLive: (cid: string) => get<CollectionLive>(`/agent-collections/${cid}/live`),
  plannerBudget: (pid: string) => get<PlannerBudget>(`/projects/${pid}/planner-budget`),
  agentAttempts: (pid: string) => get<AgentAttempt[]>(`/projects/${pid}/agent-attempts`),
  agentAttempt: (aid: string) => get<AttemptDetail>(`/agent-attempts/${aid}`),
  collectorComparison: (pid: string) => get<CollectorComparison>(`/projects/${pid}/collector-comparison`),

  // autopilot
  autopilot: (pid: string) => get<AutopilotRun | null>(`/projects/${pid}/autopilot`),
  startAutopilot: (pid: string) => post<AutopilotRun>(`/projects/${pid}/autopilot/start`),
  stopAutopilot: (pid: string) => post<AutopilotRun | null>(`/projects/${pid}/autopilot/stop`),
  setAutopilotPolicy: (pid: string, policy: Partial<AutopilotPolicy>) =>
    put<AutopilotPolicy>(`/projects/${pid}/autopilot/policy`, policy),
  eventsUrl: (pid: string) => `${API_BASE}/projects/${pid}/events`,

  // collect
  simFrameUrl: (pid: string, c: Condition, seed: number) =>
    `${API_BASE}/projects/${pid}/sim/frame?cube_region=${c.cube_region}&lighting=${c.lighting}&seed=${seed}`,
  simLiveUrl: (pid: string) => `${API_BASE}/projects/${pid}/sim/live`,
  record: (pid: string, condition: Condition, seed?: number) =>
    post<Job>(`/projects/${pid}/recordings`, seed === undefined ? { condition } : { condition, seed }),
  recordBatch: (pid: string, conditions: Condition[], per_condition: number, base_seed?: number) =>
    post<Job>(`/projects/${pid}/recordings/batch`, { conditions, per_condition, base_seed }),
  stopRecording: (jobId: string) => post<Job>(`/recordings/${jobId}/stop`),
  episodes: (pid: string) => get<Episode[]>(`/projects/${pid}/episodes`),
  setEpisodeStatus: (eid: string, status: Extract<EpisodeStatus, "accepted" | "rejected">) =>
    patch<Episode>(`/episodes/${eid}`, { status }),
  episodeMedia: (eid: string) => get<EpisodeMedia>(`/episodes/${eid}/media`),
  importHub: (pid: string, repo_id: string, max_episodes: number) => post<Job>(`/projects/${pid}/imports/hub`, { repo_id, max_episodes }),
  acceptPassing: (pid: string) => post<{ accepted: number }>(`/projects/${pid}/episodes/accept-passing`),

  // analyze
  runAnalysis: (pid: string, use_vlm: boolean, max_vlm_episodes?: number) =>
    post<Job>(`/projects/${pid}/analyses`, { use_vlm, max_vlm_episodes }),
  analysis: (aid: string) => get<Analysis>(`/analyses/${aid}`),
  latestAnalysis: (pid: string) => get<Analysis | null>(`/projects/${pid}/analyses/latest`),
  failureClusters: (pid: string) => get<FailureCluster[]>(`/projects/${pid}/failure-clusters`),
  correctDiagnosis: (
    aid: string,
    eid: string,
    status: Exclude<DiagnosisStatus, "hypothesis">,
    human_label?: string,
  ) => post<EpisodeDiagnosis>(`/analyses/${aid}/diagnoses/${eid}`, { status, human_label }),

  // improve
  recommend: (pid: string, budget?: number) =>
    post<Recommendation[]>(`/projects/${pid}/recommendations`, budget === undefined ? {} : { budget }),
  recommendations: (pid: string) => get<Recommendation[]>(`/projects/${pid}/recommendations`),
  setRecommendationStatus: (rid: string, status: RecommendationStatus) =>
    patch<Recommendation>(`/recommendations/${rid}`, { status }),
  createDataset: (pid: string, req: CreateDatasetVersionRequest) => post<Job>(`/projects/${pid}/datasets`, req),
  datasets: (pid: string) => get<DatasetVersion[]>(`/projects/${pid}/datasets`),
  proposeTraining: (pid: string, dataset_version_id: string, policy_type?: PolicyType, steps?: number) =>
    post<TrainingJob>(`/projects/${pid}/training/proposals`, { dataset_version_id, policy_type, steps }),
  approveTraining: (tid: string, approver: string, approval_token?: string) =>
    post<TrainingJob>(`/training/${tid}/approve`, { approver, approval_token }),
  launchTraining: (tid: string) => post<TrainingJob>(`/training/${tid}/launch`),
  training: (pid: string) => get<TrainingJob[]>(`/projects/${pid}/training`),
  trainingCommand: (tid: string) => get<{ command: string }>(`/training/${tid}/command`),
  registerCheckpoint: (pid: string, path: string, training_job_id?: string | null, step?: number) =>
    post<Checkpoint>(`/projects/${pid}/checkpoints`, { path, training_job_id, step }),
  checkpoints: (pid: string) => get<Checkpoint[]>(`/projects/${pid}/checkpoints`),
  launchEvaluation: (pid: string, req: LaunchEvaluationRequest) =>
    post<EvaluationWithBranches>(`/projects/${pid}/evaluations`, req),
  evaluation: (evid: string) => get<EvaluationWithBranches>(`/evaluations/${evid}`),
  evaluations: (pid: string) => get<EvaluationWithBranches[]>(`/projects/${pid}/evaluations`),
  compare: (pid: string, evaluation_ids: string[]) =>
    post<ExperimentComparison>(`/projects/${pid}/comparisons`, { evaluation_ids }),
  nextActions: (pid: string) => get<NextAction[]>(`/projects/${pid}/next-actions`),

  // agent
  agentHistory: (pid: string) => get<Record<string, unknown>[]>(`/projects/${pid}/agent/history`),
  resetAgent: (pid: string) => post<{ reset: boolean }>(`/projects/${pid}/agent/reset`),
  agentEngine: () => get<{ engine: string; label: string; model: string | null }>("/agent/engine"),
};

/**
 * POST /projects/{pid}/agent streams SSE over a POST body, so EventSource can't be used.
 * Parses `event:` / `data:` frames from a ReadableStream and yields {event, data}.
 */
export async function* streamAgent(
  pid: string,
  message: string,
  signal?: AbortSignal,
): AsyncGenerator<{ event: string; data: Record<string, unknown> }> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}/projects/${pid}/agent`, {
      method: "POST",
      headers: { ...ACTOR_HEADER, "Content-Type": "application/json", Accept: "text/event-stream" },
      body: JSON.stringify({ message }),
      signal,
    });
  } catch (e) {
    if ((e as Error).name === "AbortError") return;
    throw new ApiError(`Cannot reach the Cortex API at ${API_BASE}`, 0);
  }
  if (!res.ok || !res.body) {
    const text = await res.text().catch(() => "");
    throw new ApiError(text || `Agent request failed (${res.status})`, res.status);
  }
  const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += value;
    let idx: number;
    while ((idx = buf.search(/\r?\n\r?\n/)) !== -1) {
      const frame = buf.slice(0, idx);
      buf = buf.slice(idx).replace(/^\r?\n\r?\n/, "");
      const ev = parseFrame(frame);
      if (ev) {
        yield ev;
        if (ev.event === "done") return;
      }
    }
  }
  const tail = parseFrame(buf);
  if (tail) yield tail;
}

function parseFrame(frame: string): { event: string; data: Record<string, unknown> } | null {
  let event = "message";
  const data: string[] = [];
  for (const raw of frame.split(/\r?\n/)) {
    if (!raw || raw.startsWith(":")) continue;
    const colon = raw.indexOf(":");
    const field = colon === -1 ? raw : raw.slice(0, colon);
    const val = colon === -1 ? "" : raw.slice(colon + 1).replace(/^ /, "");
    if (field === "event") event = val;
    else if (field === "data") data.push(val);
  }
  if (event === "message" && data.length === 0) return null;
  const rawData = data.join("\n");
  try {
    const parsed = rawData ? JSON.parse(rawData) : {};
    return { event, data: parsed && typeof parsed === "object" ? parsed : { value: parsed } };
  } catch {
    return { event, data: { delta: rawData } };
  }
}

/**
 * Upload a LeRobot dataset archive (multipart field `file`). XHR rather than fetch so we get upload
 * progress. Resolves to the import Job; rejects with ApiError carrying the backend's `detail` on 4xx.
 */
export function uploadDemos(
  pid: string,
  file: File,
  maxEpisodes: number,
  onProgress: (loaded: number, total: number) => void,
): { promise: Promise<Job>; abort: () => void } {
  const xhr = new XMLHttpRequest();
  const promise = new Promise<Job>((resolve, reject) => {
    xhr.open("POST", `${API_BASE}/projects/${pid}/imports/upload?max_episodes=${maxEpisodes}`);
    xhr.setRequestHeader("X-Cortex-Actor", "gui");
    xhr.upload.onprogress = (e) => onProgress(e.loaded, e.lengthComputable ? e.total : file.size);
    xhr.onload = () => {
      let body: unknown = null;
      try {
        body = JSON.parse(xhr.responseText);
      } catch {
        body = xhr.responseText;
      }
      if (xhr.status >= 200 && xhr.status < 300) resolve(body as Job);
      else {
        const detail = (body as { detail?: unknown } | null)?.detail ?? body;
        reject(new ApiError(typeof detail === "string" && detail ? detail : `Upload failed (${xhr.status})`, xhr.status, detail, body));
      }
    };
    xhr.onerror = () => reject(new ApiError(`Cannot reach the Cortex API at ${API_BASE}`, 0));
    xhr.onabort = () => reject(new ApiError("Upload cancelled", 499));
    const form = new FormData();
    form.append("file", file);
    xhr.send(form);
  });
  return { promise, abort: () => xhr.abort() };
}
