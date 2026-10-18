// Pure: chat items → turns of milestone lines, one line per pipeline stage per turn. No React, no network.
// Cortex's own sentences (headlines) lead; raw tool text stays in the item for Details › Technical.
import { isCardRef, isWakeNotice, prefixOf, toolLabel, type ChatItem, type Evidence, type NextStep, type TodoItem, type ToolItem } from "./model";

export type { Evidence, NextStep } from "./model";

export type StageId = "scene" | "demos" | "train" | "test" | "improve";
export type MilestoneStatus = "done" | "working" | "needs_you" | "failed" | "stopped";

type JobDoneItem = Extract<ChatItem, { kind: "job_done" }>;
type NoticeItem = Extract<ChatItem, { kind: "notice" }>;
type ApprovalItem = Extract<ChatItem, { kind: "approval" }>;
type ChangesItem = Extract<ChatItem, { kind: "changes" }>;

/** Named after the Pipeline strip's Simulate / Demonstrate / Train / Evaluate / Improve. */
export const STAGES: { id: StageId; name: string }[] = [
  { id: "scene", name: "Set up the scene" },
  { id: "demos", name: "Collect demos" },
  { id: "train", name: "Train" },
  { id: "test", name: "Test" },
  { id: "improve", name: "Improve" },
];
export const STAGE_NAME = Object.fromEntries(STAGES.map((s) => [s.id, s.name])) as Record<StageId, string>;

export const STAGE_OF_TOOL: Record<string, StageId> = {
  start_intake: "scene",
  update_intake: "scene",
  build_simulation: "scene",
  capture_frame: "scene",
  create_project: "scene",
  create_policy_project: "scene",
  start_agent_collection: "demos",
  record_demonstrations: "demos",
  analyze_dataset: "demos",
  recommend_demonstrations: "demos",
  get_failure_clusters: "demos",
  import_demonstrations: "demos",
  create_dataset_version: "train",
  train_from_agent_demonstrations: "train",
  propose_training_job: "train",
  launch_training_job: "train",
  set_infrastructure: "train",
  launch_simulation_evaluation: "test",
  evaluate_learned_policy: "test",
  compare_experiments: "test",
  improve_policy: "improve",
  get_policy_weaknesses: "improve",
  propose_rl_sandboxes: "improve",
  accept_rl_sandboxes: "improve",
  stop_rl_sandbox: "improve",
  close_rl_round: "improve",
  start_autopilot: "improve",
};

export const STAGE_OF_JOB: Record<string, StageId> = {
  record: "demos",
  agent_collect: "demos",
  analyze: "demos",
  import: "demos",
  dataset: "train",
  train_from_agent: "train",
  train: "train",
  evaluate: "test",
  improve_policy: "improve",
};

/** Reads never get a line; they go into the Details of the nearest milestone. */
export const READ_TOOLS = new Set([
  "get_project_state",
  "get_next_actions",
  "get_job",
  "get_collection_progress",
  "list_episodes",
  "get_demonstration_results",
  "get_intake_status",
  "get_infrastructure",
  "get_session_link",
  "get_rl_sandbox_status",
]);

export const stageOfTool = (tool: string): StageId | null =>
  READ_TOOLS.has(tool) ? null : (STAGE_OF_TOOL[tool] ?? (tool.startsWith("robot_") ? "demos" : null));
export const stageOfJob = (kind: string): StageId | null => STAGE_OF_JOB[kind] ?? (kind.startsWith("rl_") ? "improve" : null);

export const STATUS_WORD: Record<MilestoneStatus, string> = {
  done: "Done",
  working: "Working",
  needs_you: "Needs you",
  failed: "Failed",
  stopped: "Stopped",
};

/** Headlines for rows without one (older history). Generic on purpose: never a quote of raw text. */
const SHARED = { working: "Working on it…", failed: "This step failed. Open Details for the reason.", stopped: "Stopped.", needs_you: "Waiting for your OK." };
export const FALLBACK: Record<StageId, Record<MilestoneStatus, string>> = {
  scene: { done: "Scene built.", ...SHARED },
  demos: { done: "Demos collected.", ...SHARED },
  train: { done: "Training step finished.", ...SHARED },
  test: { done: "Test finished.", ...SHARED },
  improve: { done: "Improvement step finished.", ...SHARED },
};

/** What an id chip says instead of the id (the id stays in its title). */
export const REF_NOUN: Record<string, string> = {
  job: "this step",
  tj: "the training run",
  ds: "the training set",
  ev: "the test",
  ck: "the trained policy",
  ac: "the demo collection",
  ep: "a demo",
  prj: "this project",
  an: "the quality check",
  rec: "the suggestion",
  rs: "this step",
  rr: "this step",
  run: "this step",
  th: "this step",
};

/** Cortex ids, and 12-hex scene digests (at least one letter and one digit, so plain numbers stay). */
const ID_RE = /\b(?:prj|job|tj|ds|ev|ac|ck|ep|an|rec|rs|rr|run|th)_[0-9a-f]{6,}\b|\b(?=[0-9a-f]*[a-f])(?=[0-9a-f]*\d)[0-9a-f]{12}\b/g;
const nounOf = (id: string) => (id.includes("_") ? (REF_NOUN[prefixOf(id)] ?? "this step") : "the scene");

/** Text split around ids, so a renderer can show each id as a chip with its noun. */
export function splitIds(text: string): (string | { id: string; noun: string })[] {
  const out: (string | { id: string; noun: string })[] = [];
  let last = 0;
  for (const m of text.matchAll(ID_RE)) {
    if (m.index! > last) out.push(text.slice(last, m.index));
    out.push({ id: m[0], noun: nounOf(m[0]) });
    last = m.index! + m[0].length;
  }
  if (last < text.length) out.push(text.slice(last));
  return out;
}

/** The text with every id replaced by its noun ("running as job_eee5…" → "running as this step"). */
export const stripIds = (text: string) => splitIds(text).map((p) => (typeof p === "string" ? p : p.noun)).join("");

/** Plain names for steps in Details, where the tool's own label would leak internal words. */
const STEP_LABEL: Record<string, string> = {
  start_intake: "Read the task",
  update_intake: "Filled in the scene details",
  get_intake_status: "Checked the scene details",
  build_simulation: "Built the scene",
  capture_frame: "Took a picture of the scene",
  get_project_state: "Checked the project",
  get_next_actions: "Checked what to do next",
  list_episodes: "Looked at the demos",
  get_demonstration_results: "Read the demo results",
  get_infrastructure: "Checked where training runs",
  get_session_link: "Got the studio link",
  get_rl_sandbox_status: "Checked the practice runs",
  create_dataset_version: "Building a training set",
  train_from_agent_demonstrations: "Building a training set, then setting up training",
  propose_training_job: "Setting up training",
  set_infrastructure: "Chose where training runs",
};
const JOB_LABEL: Record<string, string> = {
  record: "Recording demos",
  agent_collect: "Collecting demos",
  analyze: "Checking demo quality",
  import: "Importing demos",
  dataset: "Building the training set",
  train_from_agent: "Building the training set, then setting up training",
  train: "Training",
  evaluate: "Testing",
  improve_policy: "Improvement round",
};
export const stepLabel = (tool: string, args?: Record<string, unknown>) => STEP_LABEL[tool] ?? toolLabel(tool, args);
export const jobLabel = (kind: string) => JOB_LABEL[kind] ?? (kind.startsWith("rl_") ? "Practice run" : "Background step");

// ------------------------------------------------------------------ shapes
export interface Step {
  label: string;
  headline: string | null;
  status: MilestoneStatus;
}

export interface Milestone {
  key: string;
  stage: StageId;
  status: MilestoneStatus;
  headline: string;
  evidence: Evidence;
  assumed: string | null;
  next: NextStep;
  by: "cortex" | null;
  /** failures before the success that made this line Done */
  failedAttempts: number;
  /** Failed here, but the same stage succeeded in a later turn */
  laterSuccess: boolean;
  /** jobs this line started that have not reported back yet (live progress comes from these) */
  pendingJobs: string[];
  details: {
    steps: Step[];
    /** every tool row, reads included, for Technical */
    tools: ToolItem[];
    jobIds: string[];
    jobDone: JobDoneItem[];
    /** the agent's earlier messages in this turn, before this line */
    agentTexts: string[];
    notices: NoticeItem[];
    approvals: ApprovalItem[];
    /** result cards (collection, training, test…) first mentioned here; never job_ ids */
    cardRefs: string[];
  };
}

export type Trigger = { kind: "user"; text: string } | { kind: "wake"; after?: { stage: StageId; status: MilestoneStatus } };

export interface Turn {
  key: string;
  /** null only for rows before the first message */
  trigger: Trigger | null;
  milestones: Milestone[];
  /** the turn's last agent message (raw for rendering with id chips; `text` has no ids), and its plan */
  note: { text: string; raw: string; correction: string | null; plan: TodoItem[] | null } | null;
  footer: { durationMs: number | null; failed: number; isError: boolean; costUsd: number | null } | null;
  /** reads in a turn without milestones (shown under the note's Details) */
  reads: ToolItem[];
  /** earlier agent messages in a turn without milestones */
  earlierTexts: string[];
  /** which agent answers and why: shown above the lines */
  engine: NoticeItem[];
  /** rows that stay on the old path, in order: imports, errors, notices about nothing on a line */
  extras: ChatItem[];
  changes: ChangesItem[];
  budgetError: string | null;
}

// ------------------------------------------------------------------ build
type Entry =
  | { at: number; kind: "tool" | "read"; item: ToolItem }
  | { at: number; kind: "job"; item: JobDoneItem }
  | { at: number; kind: "approval"; item: ApprovalItem }
  | { at: number; kind: "notice"; item: NoticeItem };

interface Draft {
  key: string;
  stage: StageId;
  turn: number;
  entries: Entry[];
  texts: { at: number; text: string }[];
  out?: Milestone;
}

interface TurnDraft {
  turn: Turn;
  index: number;
  start: number;
  wakeRefs: string[] | null;
  byStage: Map<StageId, Draft>;
  drafts: Draft[];
  /** milestone items of this turn, in order (where a text or a read attaches) */
  touches: { at: number; d: Draft }[];
  texts: { at: number; text: string }[];
  reads: { at: number; item: ToolItem }[];
  plan: TodoItem[] | null;
}

/** Ids that tie a job, its approval and its notices to one line. */
const isLinkId = (id: string) => ["job", "tj", "ev"].includes(prefixOf(id));

type Outcome = "ok" | "fail" | "stop" | "run" | "attention" | "resolve" | null;
function outcomeOf(e: Entry): Outcome {
  switch (e.kind) {
    case "tool":
      return e.item.status === "ok" ? "ok" : e.item.status === "error" ? "fail" : e.item.status === "stopped" ? "stop" : "run";
    case "job":
      return e.item.status === "succeeded" ? "ok" : e.item.status === "cancelled" ? "stop" : e.item.status === "failed" ? "fail" : null;
    case "approval":
      return "attention";
    case "notice":
      return e.item.tone === "needs_you" ? "attention" : "resolve";
    default:
      return null;
  }
}
const headlineOf = (e: Entry | undefined) => (e && (e.kind === "tool" || e.kind === "job") ? (e.item.headline ?? null) : null);
const evidenceOfEntry = (e: Entry | undefined): Evidence => (e && (e.kind === "tool" || e.kind === "job") ? (e.item.evidence ?? null) : null);

const RUNNING_WORDS = /\b(running|started|in progress|underway|kicked off|is building|are building)\b/i;

export function buildTurns(items: ChatItem[]): Turn[] {
  const turns: TurnDraft[] = [];
  const all: Draft[] = [];
  const byId = new Map<string, Draft>();
  const importJobs = new Set<string>();

  const startTurn = (trigger: Trigger | null, key: string, at: number, wakeRefs: string[] | null = null) => {
    const t: TurnDraft = {
      turn: { key, trigger, milestones: [], note: null, footer: null, reads: [], earlierTexts: [], engine: [], extras: [], changes: [], budgetError: null },
      index: turns.length,
      start: at,
      wakeRefs,
      byStage: new Map(),
      drafts: [],
      touches: [],
      texts: [],
      reads: [],
      plan: null,
    };
    turns.push(t);
    return t;
  };
  const newDraft = (t: TurnDraft, stage: StageId, at: number) => {
    const d: Draft = { key: `${t.turn.key}-${stage}-${at}`, stage, turn: t.index, entries: [], texts: [] };
    t.byStage.set(stage, d);
    t.drafts.push(d);
    all.push(d);
    return d;
  };
  const lookup = (ids: string[]) => {
    for (const id of ids) if (isLinkId(id) && byId.has(id)) return byId.get(id)!;
    return null;
  };
  const register = (d: Draft, ids: string[]) => ids.filter(isLinkId).forEach((id) => byId.set(id, d));
  const latestOf = (stage: StageId | null) => (stage ? ([...all].reverse().find((d) => d.stage === stage) ?? null) : null);
  const attach = (t: TurnDraft, d: Draft, e: Entry) => {
    d.entries.push(e);
    if (d.turn === t.index) t.touches.push({ at: e.at, d });
  };

  let cur: TurnDraft | null = null;
  items.forEach((it, at) => {
    if (it.kind === "user") {
      cur = startTurn({ kind: "user", text: it.text }, `u${it.id}`, at);
      return;
    }
    if (isWakeNotice(it)) {
      cur = startTurn({ kind: "wake" }, `w${it.id}`, at, it.kind === "notice" ? it.refIds : []);
      return;
    }
    const t: TurnDraft = cur ?? (cur = startTurn(null, `s${it.id}`, at));
    switch (it.kind) {
      case "assistant":
        t.texts.push({ at, text: it.text });
        break;
      case "tool": {
        if (it.status === "error" && /budget/i.test(it.summary)) t.turn.budgetError = it.summary;
        const stage = stageOfTool(it.tool);
        if (!stage) {
          t.reads.push({ at, item: it }); // a read, or a tool with no stage: Details only, nothing lost
          break;
        }
        const d = t.byStage.get(stage) ?? newDraft(t, stage, at);
        attach(t, d, { at, kind: "tool", item: it });
        register(d, it.refIds);
        break;
      }
      case "job_done": {
        if (importJobs.has(it.jobId)) {
          t.turn.extras.push(it); // an upload from this tab: its own row shows the job
          break;
        }
        const stage = stageOfJob(it.jobKind);
        const d = lookup([it.jobId, ...it.refIds]) ?? latestOf(stage) ?? (stage ? newDraft(t, stage, at) : null);
        if (!d) {
          t.turn.extras.push(it);
          break;
        }
        attach(t, d, { at, kind: "job", item: it });
        register(d, [it.jobId, ...it.refIds]);
        break;
      }
      case "approval": {
        const stage: StageId = it.approvalKind === "evaluation" ? "test" : "train";
        const d = lookup([it.refId]) ?? latestOf(stage) ?? newDraft(t, stage, at);
        attach(t, d, { at, kind: "approval", item: it });
        register(d, [it.refId]);
        break;
      }
      case "notice": {
        if (it.tone === "engine") {
          t.turn.engine.push(it);
          break;
        }
        const d = lookup(it.refIds);
        if (d) attach(t, d, { at, kind: "notice", item: it });
        else t.turn.extras.push(it);
        break;
      }
      case "meta":
        t.turn.footer = { durationMs: it.durationMs ?? null, failed: 0, isError: !!it.isError, costUsd: it.costUsd ?? null };
        break;
      case "todos":
        t.plan = it.items;
        break;
      case "changes":
        t.turn.changes.push(it);
        break;
      case "import":
        if (it.jobId) importJobs.add(it.jobId);
        t.turn.extras.push(it);
        break;
      case "error":
        t.turn.extras.push(it);
        break;
    }
  });

  // earlier texts and reads attach to the next line in their turn, else the previous one
  for (const t of turns) {
    const note = t.texts[t.texts.length - 1];
    const near = (at: number) => t.touches.find((x) => x.at > at)?.d ?? [...t.touches].reverse().find((x) => x.at < at)?.d ?? null;
    for (const x of t.texts.slice(0, -1)) {
      const d = near(x.at);
      if (d) d.texts.push(x);
      else t.turn.earlierTexts.push(x.text);
    }
    for (const r of t.reads) {
      const d = near(r.at);
      if (d) d.entries.push({ at: r.at, kind: "read", item: r.item });
      else t.turn.reads.push(r.item);
    }
    if (note || t.plan) t.turn.note = { text: stripIds(note?.text ?? ""), raw: note?.text ?? "", correction: null, plan: t.plan };
  }

  // statuses first: "a later attempt worked" looks at later turns
  const doneJobs = new Set(all.flatMap((d) => d.entries.flatMap((e) => (e.kind === "job" ? [e.item.jobId] : []))));
  const seenCards = new Set<string>();
  for (const d of all) {
    d.entries.sort((a, b) => a.at - b.at);
    d.out = finish(d, all.some((x) => x.stage === d.stage && x.turn > d.turn), doneJobs, seenCards);
  }
  for (const d of all) {
    const m = d.out!;
    m.laterSuccess =
      m.status === "failed" &&
      all.some((x) => x.stage === d.stage && x.turn > d.turn && (x.out!.status === "done" || x.entries.some((e) => e.kind === "job" && e.item.status === "succeeded")));
  }

  return turns.map((t) => {
    const turn = t.turn;
    turn.milestones = t.drafts.map((d) => d.out!);
    if (turn.footer) turn.footer.failed = turn.milestones.filter((m) => m.status === "failed").length;
    if (t.wakeRefs && turn.trigger?.kind === "wake") {
      const d = lookup(t.wakeRefs);
      if (d) turn.trigger = { kind: "wake", after: { stage: d.stage, status: d.out!.status } };
    }
    // the agent said it is running, but a line in this turn had already failed before it wrote that
    const noteAt = t.texts[t.texts.length - 1]?.at;
    if (turn.note && noteAt !== undefined && RUNNING_WORDS.test(turn.note.raw)) {
      let hit: { at: number; stage: StageId } | null = null;
      for (const d of all) {
        if (d.out!.status !== "failed") continue;
        const fail = [...d.entries].reverse().find((e) => outcomeOf(e) === "fail");
        if (fail && fail.at >= t.start && fail.at < noteAt && (!hit || fail.at > hit.at)) hit = { at: fail.at, stage: d.stage };
      }
      if (hit) turn.note.correction = correctionText(hit.stage);
    }
    return turn;
  });
}

/** Status, headline and details of one line (R5, R15). */
function finish(d: Draft, laterInStage: boolean, doneJobs: Set<string>, seenCards: Set<string>): Milestone {
  let lastOk = -1;
  let lastFail = -1;
  let lastStop = -1;
  let lastAttention = -1;
  let lastResolve = -1;
  let running = false;
  const fails: number[] = [];
  d.entries.forEach((e, i) => {
    const o = outcomeOf(e);
    if (o === "ok") lastOk = i;
    else if (o === "fail") (lastFail = i), fails.push(i);
    else if (o === "stop") lastStop = i;
    else if (o === "attention") lastAttention = i;
    else if (o === "resolve") lastResolve = i;
    else if (o === "run") running = true;
  });
  const tools = d.entries.filter((e) => e.kind === "tool").map((e) => e.item as ToolItem);
  const startedJobs = tools.flatMap((t) => t.refIds.filter((r) => prefixOf(r) === "job"));
  const pendingJobs = [...new Set(startedJobs.filter((j) => !doneJobs.has(j)))];

  let status: MilestoneStatus;
  if (lastFail > Math.max(lastOk, lastAttention)) status = "failed";
  else if (lastAttention > Math.max(lastOk, lastStop, lastResolve) && !laterInStage) status = "needs_you";
  else if (running || pendingJobs.length) status = "working";
  else if (lastStop > lastOk) status = "stopped";
  else status = "done";

  const latest = (pred: (o: Outcome, e: Entry) => boolean, withHeadline = false) =>
    [...d.entries].reverse().find((e) => pred(outcomeOf(e), e) && (!withHeadline || !!headlineOf(e)));
  const source =
    status === "failed"
      ? d.entries[lastFail]
      : status === "stopped"
        ? d.entries[lastStop]
        : status === "working"
          ? latest((o, e) => e.kind === "tool" && (o === "run" || o === "ok"), true)
          : status === "done"
            ? latest((o) => o === "ok", true)
            : undefined;
  const headline = headlineOf(source);
  const failing = status === "failed" ? d.entries[lastFail] : undefined;

  const steps: Step[] = d.entries.flatMap((e): Step[] => {
    if (e.kind === "tool" || e.kind === "read") {
      const s = e.item.status;
      return [{ label: stepLabel(e.item.tool, e.item.args), headline: e.item.headline ?? null, status: s === "ok" ? "done" : s === "error" ? "failed" : s === "stopped" ? "stopped" : "working" }];
    }
    if (e.kind === "job") {
      const s = e.item.status;
      return [{ label: jobLabel(e.item.jobKind), headline: e.item.headline ?? null, status: s === "succeeded" ? "done" : s === "cancelled" ? "stopped" : "failed" }];
    }
    return [];
  });

  const cardRefs: string[] = [];
  for (const e of d.entries) {
    const refs = e.kind === "tool" || e.kind === "job" ? e.item.refIds : e.kind === "approval" ? [e.item.refId] : [];
    for (const r of refs)
      if (isCardRef(r) && prefixOf(r) !== "job" && !seenCards.has(r)) {
        seenCards.add(r);
        cardRefs.push(r);
      }
  }

  return {
    key: d.key,
    stage: d.stage,
    status,
    headline: headline ?? FALLBACK[d.stage][status],
    evidence: headline ? evidenceOfEntry(source) : null,
    assumed: [...tools].reverse().find((t) => t.assumed)?.assumed ?? null,
    next: failing?.kind === "job" ? (failing.item.next ?? null) : null,
    by: tools.length > 0 && tools.every((t) => t.by === "cortex") ? "cortex" : null,
    failedAttempts: fails.filter((i) => i < lastOk).length,
    laterSuccess: false,
    pendingJobs,
    details: {
      steps,
      tools: d.entries.filter((e) => e.kind === "tool" || e.kind === "read").map((e) => e.item as ToolItem),
      jobIds: [...new Set([...startedJobs, ...d.entries.flatMap((e) => (e.kind === "job" ? [e.item.jobId] : []))])],
      jobDone: d.entries.flatMap((e) => (e.kind === "job" ? [e.item] : [])),
      agentTexts: d.texts.sort((a, b) => a.at - b.at).map((x) => x.text),
      notices: d.entries.flatMap((e) => (e.kind === "notice" ? [e.item] : [])),
      approvals: d.entries.flatMap((e) => (e.kind === "approval" ? [e.item] : [])),
      cardRefs,
    },
  };
}

// ------------------------------------------------------------------ Cortex's words
/** "Done", or "Done, after 1 failed attempt". */
export function statusLabel(m: Pick<Milestone, "status" | "failedAttempts">): string {
  if (m.status === "done" && m.failedAttempts > 0) return `Done, after ${m.failedAttempts} failed attempt${m.failedAttempts === 1 ? "" : "s"}`;
  return STATUS_WORD[m.status];
}

export function footerText(f: NonNullable<Turn["footer"]>): string {
  const head = f.isError ? "Ended with an error" : `Done${f.durationMs != null ? ` in ${Math.round(f.durationMs / 1000)} s` : ""}`;
  return f.failed > 0 ? `${head} · ${f.failed} step${f.failed === 1 ? "" : "s"} failed` : head;
}

export function wakeText(t: Extract<Trigger, { kind: "wake" }>): string {
  return t.after ? `↻ Continuing on its own after ${STAGE_NAME[t.after.stage]} ${STATUS_WORD[t.after.status].toLowerCase()}` : "↻ Continuing on its own";
}

export const correctionText = (stage: StageId) => `Cortex: this step had already failed when the agent wrote this. See ${STAGE_NAME[stage]} above.`;

export const FIX_BUSY_TIP = "The agent is working on it. Stop it to take over.";

/** "1 of 4 done · stopped at: Build dataset…", or "all 4 done". */
export function planText(plan: TodoItem[], finished: boolean): string {
  const k = plan.filter((p) => p.status === "completed").length;
  if (k === plan.length) return `all ${plan.length} done`;
  const at = plan.find((p) => p.status === "in_progress");
  const where = at ? (finished ? ` · stopped at: ${at.text}` : ` · now: ${at.active || at.text}`) : "";
  return `${k} of ${plan.length} done${where}`;
}

/** The latest turn with lines ends on a failed line that offers a fix: its button replaces the generic next-step chips. */
export function endsOnFix(turns: Turn[]): boolean {
  const turn = [...turns].reverse().find((t) => t.milestones.length > 0);
  const m = turn?.milestones[turn.milestones.length - 1];
  return !!m && m.status === "failed" && !!m.next && !m.laterSuccess;
}
