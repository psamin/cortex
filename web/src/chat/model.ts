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
  | { kind: "error"; id: number; text: string };

export type ChatEvent =
  | { type: "user"; text: string }
  | { type: "text"; delta: string }
  | { type: "tool_call"; tool: string; args: Record<string, unknown> }
  | { type: "tool_result"; tool: string; summary: string; ref_ids: string[]; ok?: boolean }
  | { type: "approval_required"; kind: string; ref_id: string; summary: string; args: Record<string, unknown> }
  | { type: "done"; turns?: number | null; cost_usd?: number | null; duration_ms?: number | null; is_error?: boolean | null }
  | { type: "status"; state: string; tool?: string }
  | { type: "error"; message: string };

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
  robot_start_episode: "Driving the arm: starting an episode",
  robot_finish_episode: "Driving the arm: finishing the episode",
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

export function toolLabel(tool: string, args?: Record<string, unknown>): string {
  const t = normalizeTool(tool);
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
      // the backend log writes {kind: ev, **card}, so the card's own kind ("training") overwrites "approval_required"
      out.push({ type: "approval_required", kind: k === "approval_required" ? "training" : String(k), ref_id: String(r.ref_id ?? ""), summary: String(r.summary ?? ""), args: (r.args as Record<string, unknown>) ?? {} });
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
