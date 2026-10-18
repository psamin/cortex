// Pure chat model: SSE events and stored history both fold into the same item list.

export type ToolStatus = "running" | "ok" | "error";

export type ChatItem =
  | { kind: "user"; id: number; text: string }
  | { kind: "assistant"; id: number; text: string }
  | {
      kind: "tool";
      id: number;
      tool: string;
      status: ToolStatus;
      /** consecutive polls of the same tool collapse into one row */
      count: number;
      summary: string;
      refIds: string[];
      args: Record<string, unknown>;
    }
  | {
      kind: "approval";
      id: number;
      approvalKind: string;
      refId: string;
      summary: string;
      args: Record<string, unknown>;
    }
  | { kind: "meta"; id: number; turns?: number | null; durationMs?: number | null; costUsd?: number | null; isError?: boolean | null }
  | { kind: "error"; id: number; text: string }
  /** A demo import started from this studio (not in agent history): upload progress, then the live job card. */
  | { kind: "import"; id: number; title: string; detail: string; status: "uploading" | "started" | "error"; loaded: number; total: number; error?: string; jobId?: string }
  /** Written by Cortex (not the LLM) when a background job finishes. */
  | { kind: "job_done"; id: number; jobId: string; jobKind: string; status: string; text: string; refIds: string[]; detail?: string | null; superseded?: boolean };

export type ChatEvent =
  | { type: "user"; text: string }
  | { type: "text"; delta: string }
  | { type: "tool_call"; tool: string; args: Record<string, unknown> }
  | { type: "tool_result"; tool: string; summary: string; ref_ids: string[]; ok?: boolean }
  | { type: "approval_required"; kind: string; ref_id: string; summary: string; args: Record<string, unknown> }
  | { type: "done"; turns?: number | null; cost_usd?: number | null; duration_ms?: number | null; is_error?: boolean | null }
  | { type: "status"; state: string; tool?: string }
  | { type: "error"; message: string }
  | { type: "job_done"; job_id: string; job_kind: string; status: string; text: string; ref_ids: string[]; detail?: string | null; superseded?: boolean };

// ------------------------------------------------------------------ friendly labels
const LABELS: Record<string, string> = {
  start_agent_collection: "Collecting demonstrations",
  record_demonstrations: "Recording demonstrations",
  get_collection_progress: "Checking progress",
  get_job: "Checking progress",
  get_demonstration_results: "Reading the demonstrations",
  list_episodes: "Looking at the demos",
  analyze_dataset: "Checking demo quality",
  get_failure_clusters: "Grouping failures",
  recommend_demonstrations: "Ranking what to collect next",
  create_dataset_version: "Building the dataset",
  propose_training_job: "Preparing training",
  train_from_agent_demonstrations: "Preparing training",
  launch_training_job: "Starting training",
  launch_simulation_evaluation: "Testing in 3 conditions",
  evaluate_learned_policy: "Testing in 3 conditions",
  compare_experiments: "Comparing runs",
  improve_policy: "Targeting the weakest condition",
  get_project_state: "Reading the project",
  get_next_actions: "Reading the project",
  create_policy_project: "Setting up the project",
  create_project: "Setting up the project",
};

export const POLL_TOOLS = new Set(["get_collection_progress", "get_job"]);
/** Consecutive calls of these collapse into one updating row (the label follows the latest call). */
const COLLAPSE_TOOLS = new Set([...POLL_TOOLS, "robot_act"]);

/** Claude Code internals (ToolSearch, TodoWrite, …) are capitalised; they never appear in the chat. */
export const isHiddenTool = (tool: string) => /^[A-Z]/.test(tool);

export const normalizeTool = (tool: string) => tool.split("__").pop()!.replace(/^cortex_/, "").replace(/^cortex\./, "");

/** Phases of the arm-driving tools (args.tool on robot_act). */
const ROBOT_PHASES: Record<string, string> = {
  observe_scene: "observing the scene",
  get_robot_state: "reading robot state",
  propose_grasp: "selecting a grasp",
  move_end_effector: "moving the gripper",
  set_gripper: "operating the gripper",
  verify_grasp: "verifying the grasp",
  place_object: "placing the object",
  check_task_success: "checking task success",
};

/** robot_<verb> tools drive the simulated arm step by step. */
const ROBOT_VERBS: Record<string, string> = {
  start_episode: "starting an episode",
  finish_episode: "finishing the episode",
  observe: "looking at the scene",
  state: "reading robot state",
  propose_grasp: "choosing a grasp",
  move: "moving the gripper",
  verify_grasp: "checking the grasp",
  place: "placing the cube",
  check_success: "checking task success",
};

export function toolLabel(tool: string, args?: Record<string, unknown>): string {
  const t = normalizeTool(tool);
  if (t.startsWith("robot_") && t !== "robot_act") {
    const verb = t.slice(6);
    if (verb === "gripper") return `Driving the arm: ${args?.state === "closed" ? "closing" : "opening"} the gripper`;
    return `Driving the arm: ${ROBOT_VERBS[verb] ?? verb.replace(/_/g, " ")}`;
  }
  if (t === "robot_act") {
    const inner = typeof args?.tool === "string" ? args.tool : "";
    if (inner === "set_gripper" && typeof args?.args === "object" && args.args && "state" in (args.args as object))
      return `Driving the arm: ${(args.args as { state?: string }).state === "closed" ? "closing the gripper" : "opening the gripper"}`;
    return `Driving the arm${inner ? `: ${ROBOT_PHASES[inner] ?? inner.replace(/_/g, " ")}` : ""}`;
  }
  if (LABELS[t]) return LABELS[t];
  const words = t.replace(/_/g, " ");
  return words.charAt(0).toUpperCase() + words.slice(1);
}

/** Tool summaries are one-liners, but older history holds raw JSON payloads. Never show raw JSON. */
export function cleanSummary(summary: unknown, refIds: string[] = []): { summary: string; refIds: string[] } {
  const s = typeof summary === "string" ? summary.trim() : "";
  if (s.startsWith("{") || s.startsWith("[")) {
    try {
      const data = JSON.parse(s);
      if (data && typeof data === "object" && !Array.isArray(data)) {
        const refs = Array.isArray(data.ref_ids) ? data.ref_ids.map(String) : [];
        return { summary: typeof data.summary === "string" ? data.summary : "", refIds: refIds.length ? refIds : refs };
      }
    } catch {
      /* truncated JSON: fall through */
    }
    const m = s.match(/"summary"\s*:\s*"((?:[^"\\]|\\.)*)"/);
    return { summary: m ? m[1].replace(/\\"/g, '"') : "", refIds };
  }
  return { summary: s.length > 220 ? s.slice(0, 217) + "…" : s, refIds };
}

// ------------------------------------------------------------------ reducer
let nextId = 1;
const id = () => nextId++;
export const newItemId = id;

export function applyEvent(items: ChatItem[], ev: ChatEvent): ChatItem[] {
  switch (ev.type) {
    case "user":
      return [...items, { kind: "user", id: id(), text: ev.text }];
    case "text": {
      if (!ev.delta) return items;
      const last = items[items.length - 1];
      if (last?.kind === "assistant") return [...items.slice(0, -1), { ...last, text: `${last.text}\n\n${ev.delta}` }];
      return [...items, { kind: "assistant", id: id(), text: ev.delta }];
    }
    case "tool_call": {
      const tool = normalizeTool(ev.tool);
      if (isHiddenTool(tool)) return items;
      const last = items[items.length - 1];
      if (COLLAPSE_TOOLS.has(tool) && last?.kind === "tool" && (last.tool === tool || (POLL_TOOLS.has(tool) && POLL_TOOLS.has(last.tool)))) {
        return [...items.slice(0, -1), { ...last, tool, status: "running", count: last.count + 1, args: ev.args ?? {} }];
      }
      return [...items, { kind: "tool", id: id(), tool, status: "running", count: 1, summary: "", refIds: [], args: ev.args ?? {} }];
    }
    case "tool_result": {
      const tool = normalizeTool(ev.tool);
      if (isHiddenTool(tool)) return items;
      const clean = cleanSummary(ev.summary, ev.ref_ids ?? []);
      const status: ToolStatus = ev.ok === false ? "error" : "ok";
      for (let i = items.length - 1; i >= 0; i--) {
        const it = items[i];
        if (it.kind === "tool" && (it.status === "running" || (COLLAPSE_TOOLS.has(tool) && i === items.length - 1)) && (it.tool === tool || (POLL_TOOLS.has(it.tool) && POLL_TOOLS.has(tool)))) {
          const next = [...items];
          next[i] = { ...it, status, summary: clean.summary || it.summary, refIds: clean.refIds.length ? clean.refIds : it.refIds };
          return next;
        }
      }
      return [...items, { kind: "tool", id: id(), tool, status, count: 1, summary: clean.summary, refIds: clean.refIds, args: {} }];
    }
    case "approval_required": {
      const dup = items.some((it) => it.kind === "approval" && it.refId === ev.ref_id && it.approvalKind === ev.kind);
      if (dup) return items;
      return [...items, { kind: "approval", id: id(), approvalKind: ev.kind, refId: ev.ref_id, summary: cleanSummary(ev.summary).summary, args: ev.args ?? {} }];
    }
    case "done": {
      // a finished turn has no running tools left
      const settled = items.map((it) => (it.kind === "tool" && it.status === "running" ? { ...it, status: "ok" as const } : it));
      return [...settled, { kind: "meta", id: id(), turns: ev.turns, durationMs: ev.duration_ms, costUsd: ev.cost_usd, isError: ev.is_error }];
    }
    case "error":
      return [...items, { kind: "error", id: id(), text: ev.message }];
    case "job_done":
      // the same line can arrive live (SSE) and again from history: one row per job
      if (items.some((it) => it.kind === "job_done" && it.jobId === ev.job_id)) return items;
      {
        // a success makes earlier failures of the same kind history: they stop showing
        const prior = ev.status === "succeeded"
          ? items.map((it) => (it.kind === "job_done" && it.status === "failed" && it.jobKind === ev.job_kind ? { ...it, superseded: true } : it))
          : items;
        return [...prior, { kind: "job_done", id: id(), jobId: ev.job_id, jobKind: ev.job_kind, status: ev.status, text: ev.text, refIds: ev.ref_ids ?? [], detail: ev.detail ?? null, superseded: !!ev.superseded }];
      }
    case "status":
      return items;
  }
}

/** GET /agent/history rows → events (same reducer as the live stream). */
export function historyToEvents(rows: Record<string, unknown>[]): ChatEvent[] {
  const out: ChatEvent[] = [];
  for (const r of rows) {
    const k = r.kind;
    if (k === "user") out.push({ type: "user", text: String(r.text ?? "") });
    else if (k === "assistant") out.push({ type: "text", delta: String(r.text ?? "") });
    else if (k === "tool_call") out.push({ type: "tool_call", tool: String(r.tool ?? ""), args: (r.args as Record<string, unknown>) ?? {} });
    else if (k === "tool_result")
      out.push({ type: "tool_result", tool: String(r.tool ?? ""), summary: String(r.summary ?? ""), ref_ids: (r.ref_ids as string[]) ?? [], ok: r.ok as boolean | undefined });
    else if (k === "approval_required" || ((k === "training" || k === "evaluation") && "ref_id" in r))
      // card type is `card_kind`; older logs stored it in `kind` itself
      out.push({
        type: "approval_required",
        kind: String(r.card_kind ?? (k === "approval_required" ? "training" : k)),
        ref_id: String(r.training_job_id ?? r.ref_id ?? ""),
        summary: String(r.summary ?? ""),
        args: (r.args as Record<string, unknown>) ?? {},
      });
    else if (k === "job_done") out.push(jobDoneEvent(r));
    else if (k === "done") out.push({ type: "done", turns: r.turns as number, cost_usd: r.cost_usd as number, duration_ms: r.duration_ms as number, is_error: r.is_error as boolean });
  }
  return out;
}

// ------------------------------------------------------------------ result references
/** Id prefixes that get an inline result card. */
export const CARD_PREFIXES = ["ac", "ev", "ds", "tj", "an", "rec", "job", "ep"] as const;
export const prefixOf = (ref: string) => ref.split("_")[0];
export const isCardRef = (ref: string) => (CARD_PREFIXES as readonly string[]).includes(prefixOf(ref));

/** Priority for the preview pane's "latest result". */
export const PREVIEW_PREFIXES = ["ev", "ac", "ds", "tj", "an", "rec"];

/** A job_done payload (history row or live SSE `chat` event) as a chat event. */
export function jobDoneEvent(r: Record<string, unknown>): ChatEvent {
  return {
    type: "job_done",
    job_id: String(r.job_id ?? ""),
    job_kind: String(r.job_kind ?? "job"),
    status: String(r.status ?? "succeeded"),
    text: String(r.text ?? ""),
    ref_ids: Array.isArray(r.ref_ids) ? r.ref_ids.map(String) : [],
    detail: typeof r.detail === "string" ? r.detail : null,
    superseded: r.superseded === true,
  };
}

// ------------------------------------------------------------------ render plan
export type ToolItem = Extract<ChatItem, { kind: "tool" }>;

export type Block =
  | { type: "item"; item: Exclude<ChatItem, { kind: "tool" }>; key: string }
  /** consecutive tool calls between two messages; cards render after the group, outside the collapse */
  | { type: "tools"; tools: ToolItem[]; cards: string[]; budgetError: string | null; key: string };

/**
 * Deterministic layout: consecutive tool rows form one group; every result card renders once, at the
 * first mention of its id (so a live card stays put and updates in place); one training approval at most.
 */
export function buildBlocks(items: ChatItem[]): Block[] {
  const out: Block[] = [];
  const seen = new Set<string>();
  let trainingApproval = false;
  for (const it of items) {
    if (it.kind === "tool") {
      const last = out[out.length - 1];
      const group = last?.type === "tools" ? last : null;
      const refs = it.refIds.filter((r) => isCardRef(r) && !seen.has(r));
      refs.forEach((r) => seen.add(r));
      const budgetError = it.status === "error" && /budget/i.test(it.summary) ? it.summary : null;
      if (group) {
        group.tools.push(it);
        group.cards.push(...refs);
        group.budgetError = budgetError ?? group.budgetError;
      } else out.push({ type: "tools", tools: [it], cards: refs, budgetError, key: `t${it.id}` });
      if (refs.some((r) => prefixOf(r) === "tj")) trainingApproval = true;
      continue;
    }
    if (it.kind === "import" && it.jobId) seen.add(it.jobId); // its job card renders in the import row
    if (it.kind === "approval") {
      const training = it.approvalKind !== "evaluation";
      if (training && (trainingApproval || seen.has(it.refId))) continue;
      if (training) trainingApproval = true;
    }
    out.push({ type: "item", item: it, key: `i${it.id}` });
  }
  return out;
}
