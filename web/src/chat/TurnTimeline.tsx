import { useState, type ReactNode } from "react";
import { Ban, Check, ChevronRight, CircleDashed, Hand, ListChecks, Square, TriangleAlert, X } from "lucide-react";
import { api } from "../api/client";
import { isJobActive, useJob } from "../api/events";
import { useJobQuery, useProjectState, useTraining } from "../api/hooks";
import { cn, evidenceValue } from "../lib/format";
import { IsoIcon } from "../sketch";
import { GlowEdge } from "../components/ui/spotlight-card";
import { ApprovalItem, BudgetCard, ResultCards } from "./cards";
import { useChat } from "./ChatProvider";
import { Markdown } from "./Markdown";
import { prefixOf, toolLabel, type ChatItem, type Evidence, type ToolItem } from "./model";
import { toolIcon } from "./toolIcon";
import {
  FALLBACK,
  FIX_BUSY_TIP,
  footerText,
  planText,
  STAGE_NAME,
  stageOfTool,
  statusLabel,
  wakeText,
  type Milestone,
  type MilestoneStatus,
  type Trigger,
  type Turn,
} from "./timeline";

// ------------------------------------------------------------------ one turn
/** One turn: what started it, one line per stage that ran, the agent's note, the footer (R1). */
export function TurnTimeline({
  turn,
  pid,
  thumbKey,
  renderItem,
}: {
  turn: Turn;
  pid: string;
  /** the latest scene line in the chat: it gets the scene's picture */
  thumbKey: string | null;
  /** rows that stay on the old path (imports, errors, changes, notices about nothing on a line) */
  renderItem: (it: ChatItem) => ReactNode;
}) {
  return (
    <div className="space-y-3">
      {turn.trigger?.kind === "user" && (
        <div className="enter border-l border-ink pl-4">
          <div className="kn-label mb-1.5">You</div>
          <div className="whitespace-pre-wrap text-[16px] leading-relaxed text-ink">{turn.trigger.text}</div>
        </div>
      )}
      {turn.trigger?.kind === "wake" && <WakeHeader trigger={turn.trigger} />}
      {turn.engine.map((n) => (
        <div key={n.id}>{renderItem(n)}</div>
      ))}
      {turn.milestones.length > 0 && (
        <div className="space-y-2">
          {turn.milestones.map((m) => (
            <MilestoneRow key={m.key} m={m} pid={pid} thumb={m.key === thumbKey} />
          ))}
        </div>
      )}
      {turn.budgetError && <BudgetCard pid={pid} message={turn.budgetError} />}
      {turn.extras.map((it) => (
        <div key={it.id}>{renderItem(it)}</div>
      ))}
      {turn.note && <AgentNote turn={turn} pid={pid} />}
      {!turn.note && (turn.reads.length > 0 || turn.earlierTexts.length > 0) && <TurnDetails turn={turn} />}
      {turn.changes.map((c) => (
        <div key={c.id}>{renderItem(c)}</div>
      ))}
      {turn.footer && <Footer f={turn.footer} />}
    </div>
  );
}

/** R12: Cortex woke the agent; the line above already says why, so the reason is not repeated. */
export function WakeHeader({ trigger }: { trigger: Extract<Trigger, { kind: "wake" }> }) {
  return <div className="enter border-l border-line-strong pl-4 text-sm text-ink-2">{wakeText(trigger)}</div>;
}

function Footer({ f }: { f: NonNullable<Turn["footer"]> }) {
  const [head, ...failed] = footerText(f).split(" · ");
  return (
    <div className="flex items-center gap-3 font-mono text-[10px] uppercase tracking-[0.16em] text-ink-3" title={f.costUsd ? `agent cost $${f.costUsd.toFixed(2)}` : undefined}>
      <span className="h-px flex-1 bg-[var(--line)]" />
      <span>
        {head}
        {failed.length > 0 && <span className="text-fault"> · {failed.join(" · ")}</span>}
      </span>
      <span className="h-px flex-1 bg-[var(--line)]" />
    </div>
  );
}

// ------------------------------------------------------------------ one line
const STATUS_TONE: Record<MilestoneStatus, string> = {
  done: "text-pass",
  failed: "text-fault",
  needs_you: "text-caution",
  working: "text-servo",
  stopped: "text-ink-3",
};

const EVIDENCE_WORD: Record<Exclude<Evidence, null>, string> = {
  measured: "measured in sim",
  checked: "checked in sim",
  assumed: "assumed",
  hypothesis: "hypothesis, not checked",
};

function EvidenceTag({ kind }: { kind: Evidence }) {
  if (!kind) return null;
  return (
    <span className={cn("ml-1.5 inline-flex h-[18px] items-center whitespace-nowrap rounded-full border px-1.5 align-[1px] font-mono text-[9.5px] uppercase tracking-[0.1em]", kind === "hypothesis" ? "border-caution/45 text-caution" : "border-line-strong text-ink-3")}>
      {EVIDENCE_WORD[kind]}
    </span>
  );
}

/**
 * The line as it stands now: a job the chat never heard back from (its line would spin forever) takes its status
 * from the job record, and a training run that waits for approval makes a Done line need you.
 */
function useLiveLine(m: Milestone, pid: string): { status: MilestoneStatus; headline: string; jobId: string | null; progress: number | null } {
  const jobId = m.pendingJobs[m.pendingJobs.length - 1] ?? null;
  const live = useJob(jobId);
  const q = useJobQuery(jobId, !!jobId && (!live || isJobActive(live)));
  const job = live ?? q.data;
  const tj = m.details.cardRefs.find((r) => prefixOf(r) === "tj");
  const training = useTraining(pid); // shared with the pending-approval bar: no extra request
  const toolRunning = m.details.tools.some((t) => t.status === "running" && stageOfTool(t.tool));
  if (m.status === "working" && job && !isJobActive(job) && !toolRunning && m.pendingJobs.length === 1) {
    const status: MilestoneStatus = job.status === "succeeded" ? "done" : job.status === "cancelled" ? "stopped" : "failed";
    return { status, headline: FALLBACK[m.stage][status], jobId: null, progress: null };
  }
  if (m.status === "done" && tj && training.data?.find((t) => t.id === tj)?.status === "proposed")
    return { status: "needs_you", headline: FALLBACK[m.stage].needs_you, jobId: null, progress: null };
  return { status: m.status, headline: m.headline, jobId, progress: job && isJobActive(job) ? job.progress : null };
}

function MilestoneIcon({ status }: { status: MilestoneStatus }) {
  if (status === "needs_you") return <Hand className="h-3.5 w-3.5 text-caution" strokeWidth={1.75} />;
  return <StatusIcon status={status === "done" ? "ok" : status === "failed" ? "error" : status === "working" ? "running" : "stopped"} />;
}

/** One stage of one turn (R6): status icon and word, Cortex's headline, the fix, and inline what needs you. */
export function MilestoneRow({ m, pid, thumb }: { m: Milestone; pid: string; thumb?: boolean }) {
  const chat = useChat();
  const [open, setOpen] = useState(false);
  const line = useLiveLine(m, pid);
  const status = line.status;
  const failed = status === "failed";
  const approvalRefs = new Set(m.details.approvals.map((a) => a.refId));
  const approvals = m.details.approvals.filter((a, i, xs) => xs.findIndex((b) => b.refId === a.refId && b.approvalKind === a.approvalKind) === i);
  const cards = m.details.cardRefs;
  return (
    <div className={cn("enter sk-depth relative rounded border bg-plate", failed ? "border-fault/40" : status === "needs_you" ? "border-caution/50" : "border-line-strong")}>
      <GlowEdge />
      <div className="px-3.5 py-2.5">
        <div className="flex items-center gap-2.5">
          <span className="flex h-4 w-4 shrink-0 items-center justify-center">
            <MilestoneIcon status={status} />
          </span>
          <span className="text-sm font-medium text-ink">{STAGE_NAME[m.stage]}</span>
          <span className="ml-auto shrink-0 text-right font-mono text-[10.5px] uppercase tracking-[0.12em]">
            <span className={STATUS_TONE[status]}>{statusLabel({ status, failedAttempts: m.failedAttempts })}</span>
            {m.by === "cortex" && <span className="text-ink-3" title="Cortex did this itself, not the agent"> · by Cortex</span>}
          </span>
        </div>
        <div className="ml-[26px] mt-1 space-y-2">
          <p className={cn("text-sm leading-relaxed", failed ? "text-ink" : "text-ink-2")}>
            {line.headline}
            {line.headline === m.headline && <EvidenceTag kind={m.evidence} />}
          </p>
          {m.assumed && (
            <p className="text-[13px] leading-relaxed text-ink-3">
              {m.assumed}
              <EvidenceTag kind="assumed" />
            </p>
          )}
          {failed && m.laterSuccess && <p className="text-xs text-ink-3">A later attempt worked ↓</p>}
          {failed && m.next && !m.laterSuccess && (
            <div className="space-y-1.5">
              <p className="text-sm text-ink-2">
                <span className="font-medium text-ink">What to do: </span>
                {m.next.text}
              </p>
              {m.next.actions.length > 0 && (
                <div className="flex flex-wrap gap-1.5">
                  {m.next.actions.map((a) => (
                    // a disabled button shows no tooltip in most browsers: the wrapper carries it
                    <span key={a.label} title={chat.busy ? FIX_BUSY_TIP : undefined}>
                      <button
                        type="button"
                        disabled={chat.busy}
                        onClick={() => chat.send(a.message)}
                        className="sk-pill sk-pill-primary inline-flex h-8 items-center px-4 text-[13px] font-medium disabled:cursor-not-allowed disabled:opacity-45"
                      >
                        {a.label}
                      </button>
                    </span>
                  ))}
                </div>
              )}
            </div>
          )}
          {status === "needs_you" && (
            <div className="space-y-2">
              {approvals.map((a) => (
                <ApprovalItem key={a.id} pid={pid} kind={a.approvalKind} refId={a.refId} summary={a.summary} args={a.args} />
              ))}
              {cards.some((r) => !approvalRefs.has(r)) && <ResultCards pid={pid} refs={cards.filter((r) => !approvalRefs.has(r))} />}
            </div>
          )}
          {status === "working" && line.jobId && <ProgressBar value={line.progress} />}
          {status === "done" && cards.length > 0 && <ResultCards pid={pid} refs={cards} />}
          {thumb && <SceneThumb pid={pid} />}
          <div className="flex justify-end">
            <button type="button" onClick={() => setOpen((o) => !o)} aria-expanded={open} className="group inline-flex items-center gap-0.5 text-xs text-ink-3 hover:text-ink">
              Details
              <ChevronRight className={cn("h-3.5 w-3.5 transition-transform", open && "rotate-90")} />
            </button>
          </div>
        </div>
        {open && <MilestoneDetails m={m} />}
      </div>
    </div>
  );
}

/** Live progress of the job behind a working line; never the job's internal message. */
function ProgressBar({ value }: { value: number | null }) {
  return (
    <div className="flex items-center gap-2">
      <div className="h-1 flex-1 overflow-hidden bg-[var(--line)]">
        {value === null ? <div className="skeleton h-full w-full" /> : <div className="h-full bg-servo transition-[width] duration-500 ease-out" style={{ width: `${value * 100}%` }} />}
      </div>
      {value !== null && <span className="num w-9 text-right font-mono text-[10.5px] text-ink-3">{Math.round(value * 100)}%</span>}
    </div>
  );
}

/** The scene as built, from above (latest scene line only). */
function SceneThumb({ pid }: { pid: string }) {
  const digest = useProjectState(pid).data?.project.scene?.digest;
  const [broken, setBroken] = useState(false);
  if (!digest || broken) return null;
  return <img src={api.sceneFrameUrl(pid, digest)} alt="The scene from above" loading="lazy" onError={() => setBroken(true)} className="h-28 w-auto rounded-sm border border-line bg-well object-contain" />;
}

// ------------------------------------------------------------------ details
function DetailsHeading({ children }: { children: ReactNode }) {
  return <p className="mb-1.5 font-mono text-[10px] uppercase tracking-[0.14em] text-ink-3">{children}</p>;
}

/** R8: Steps in plain words, the agent's earlier messages, then exactly what the agent received. */
export function MilestoneDetails({ m }: { m: Milestone }) {
  const d = m.details;
  return (
    <div className="ml-[26px] mt-2.5 space-y-3.5 border-t border-line pt-2.5">
      <section>
        <DetailsHeading>Steps</DetailsHeading>
        <ul className="space-y-1">
          {d.steps.map((s, i) => (
            <li key={i} className="flex items-start gap-2 text-[13px] leading-snug text-ink-2">
              <span className="mt-0.5 flex h-4 w-4 shrink-0 items-center justify-center">
                <MilestoneIcon status={s.status} />
              </span>
              <span className="min-w-0 flex-1">{s.headline ?? s.label}</span>
              {s.status !== "done" && <span className={cn("shrink-0 font-mono text-[10px] uppercase tracking-[0.12em]", STATUS_TONE[s.status])}>{statusLabel({ status: s.status, failedAttempts: 0 })}</span>}
            </li>
          ))}
        </ul>
      </section>
      {d.agentTexts.length > 0 && <AgentTexts texts={d.agentTexts} />}
      <Technical tools={d.tools} jobDone={d.jobDone} notices={d.notices} approvals={d.approvals} />
    </div>
  );
}

function AgentTexts({ texts }: { texts: string[] }) {
  return (
    <section>
      <DetailsHeading>Agent's notes (its own words, not checked by Cortex)</DetailsHeading>
      <div className="space-y-2">
        {texts.map((t, i) => (
          <blockquote key={i} className="border-l-2 border-line pl-3">
            <Markdown text={t} muted />
          </blockquote>
        ))}
      </div>
    </section>
  );
}

function Technical({
  tools,
  jobDone = [],
  notices = [],
  approvals = [],
}: {
  tools: ToolItem[];
  jobDone?: Milestone["details"]["jobDone"];
  notices?: Milestone["details"]["notices"];
  approvals?: Milestone["details"]["approvals"];
}) {
  if (!tools.length && !jobDone.length && !notices.length && !approvals.length) return null;
  return (
    <section>
      <DetailsHeading>Technical (exactly what the agent received)</DetailsHeading>
      <div className="space-y-1.5">
        {tools.map((t) => (
          <ToolRow key={t.id} it={t} />
        ))}
        {jobDone.map((j) => (
          <div key={j.id} className="text-xs">
            <p className="font-mono text-[11px] text-ink-3">
              job {j.jobId} · {j.jobKind.replace(/_/g, " ")} · {j.status}
              {j.refIds.length > 0 && ` · ${j.refIds.join(", ")}`}
            </p>
            <p className="text-ink-2">{j.text}</p>
            {j.detail && <pre className="mt-1 max-h-40 overflow-auto whitespace-pre-wrap rounded-sm border border-line bg-well px-2 py-1.5 font-mono text-[11px] text-ink-3">{j.detail}</pre>}
          </div>
        ))}
        {approvals.map((a) => (
          <p key={a.id} className="text-xs text-ink-2">
            <span className="font-mono text-[11px] text-ink-3">approval {a.approvalKind} · {a.refId}</span> {a.summary}
          </p>
        ))}
        {notices.map((n) => (
          <p key={n.id} className="text-xs text-ink-2">
            <span className="font-mono text-[11px] text-ink-3">notice {n.tone}</span> {n.text}
          </p>
        ))}
      </div>
    </section>
  );
}

/** A turn with no lines (a question answered from reads): its reads and earlier messages. */
function TurnDetails({ turn }: { turn: Turn }) {
  const [open, setOpen] = useState(false);
  return (
    <div>
      <button type="button" onClick={() => setOpen((o) => !o)} aria-expanded={open} className="inline-flex items-center gap-0.5 text-xs text-ink-3 hover:text-ink">
        Details
        <ChevronRight className={cn("h-3.5 w-3.5 transition-transform", open && "rotate-90")} />
      </button>
      {open && (
        <div className="mt-2 space-y-3.5 border-t border-line pt-2.5">
          {turn.earlierTexts.length > 0 && <AgentTexts texts={turn.earlierTexts} />}
          <Technical tools={turn.reads} />
        </div>
      )}
    </div>
  );
}

// ------------------------------------------------------------------ the agent's note
/** R9/R10: the agent's last message, quoted as its own words; Cortex corrects it when a line had already failed. */
export function AgentNote({ turn, pid }: { turn: Turn; pid: string }) {
  const note = turn.note!;
  return (
    <div className="enter">
      {note.raw.trim() && (
        <>
          <div className="kn-label mb-1.5">Agent's note (its own words)</div>
          <blockquote className="border-l-2 border-line-strong pl-3.5">
            <Markdown text={note.raw} pid={pid} muted />
          </blockquote>
        </>
      )}
      {note.correction && (
        <p className="mt-2 flex items-start gap-1.5 text-[13px] leading-snug text-caution">
          <TriangleAlert className="mt-px h-3.5 w-3.5 shrink-0" strokeWidth={1.75} aria-label="Warning" />
          <span>{note.correction}</span>
        </p>
      )}
      {note.plan && note.plan.length > 0 && <Plan plan={note.plan} finished={!!turn.footer} />}
      {(turn.reads.length > 0 || turn.earlierTexts.length > 0) && (
        <div className="mt-1.5">
          <TurnDetails turn={turn} />
        </div>
      )}
    </div>
  );
}

/** The agent's checklist, verbatim, collapsed to one line. */
function Plan({ plan, finished }: { plan: NonNullable<NonNullable<Turn["note"]>["plan"]>; finished: boolean }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="mt-2">
      <button type="button" onClick={() => setOpen((o) => !o)} aria-expanded={open} className="flex w-full items-start gap-1.5 text-left text-xs text-ink-3 hover:text-ink">
        <ListChecks className="mt-px h-3.5 w-3.5 shrink-0" strokeWidth={1.5} />
        <span className="shrink-0 whitespace-nowrap">Agent's plan</span>
        <ChevronRight className={cn("mt-px h-3.5 w-3.5 shrink-0 transition-transform", open && "rotate-90")} />
        <span className="min-w-0 text-ink-4">{planText(plan, finished)}</span>
      </button>
      {open && (
        <ol className="ml-5 mt-1.5 space-y-1">
          {plan.map((t, i) => (
            <li key={i} className="flex items-start gap-2 text-[13px]">
              <span className="mt-0.5 flex h-4 w-4 shrink-0 items-center justify-center">
                {t.status === "completed" ? <Check className="h-3 w-3 text-ink-3" /> : <span className="h-2.5 w-2.5 rounded-full shadow-[inset_0_0_0_1.5px_rgb(var(--ink-4))]" />}
              </span>
              <span className={t.status === "completed" ? "text-ink-3" : t.status === "in_progress" ? "text-ink-2" : "text-ink-4"}>{t.text}</span>
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}

// ------------------------------------------------------------------ shared rows
export function StatusIcon({ status }: { status: "running" | "ok" | "error" | "cancelled" | "stopped" }) {
  if (status === "running") return <CircleDashed className="h-3.5 w-3.5 animate-spin text-servo" />;
  if (status === "stopped")
    return (
      <span className="flex h-3.5 w-3.5 items-center justify-center rounded-full border border-ink-3 text-ink-3">
        <Square className="h-1.5 w-1.5 fill-current" />
      </span>
    );
  if (status === "cancelled")
    return (
      <span className="flex h-3.5 w-3.5 items-center justify-center rounded-full border border-ink-3 text-ink-3">
        <Ban className="h-2 w-2" />
      </span>
    );
  if (status === "error")
    return (
      <span className="flex h-3.5 w-3.5 items-center justify-center rounded-full border border-fault text-fault">
        <X className="h-2 w-2" />
      </span>
    );
  return (
    <span className="flex h-3.5 w-3.5 items-center justify-center rounded-full border border-pass text-pass">
      <Check className="h-2 w-2" strokeWidth={3} />
    </span>
  );
}

/** A tool call as the agent saw it: label, status, raw summary, then its name, args and ids (Technical only). */
export function ToolRow({ it }: { it: ToolItem }) {
  const [open, setOpen] = useState(false);
  const args = Object.entries(it.args).filter(([k]) => k !== "project_id");
  const hasDetails = !!it.summary || args.length > 0 || it.refIds.length > 0;
  return (
    <div className="enter">
      <button type="button" onClick={() => hasDetails && setOpen((o) => !o)} className={cn("group flex w-full items-center gap-2.5 py-0.5 text-left", hasDetails && "cursor-pointer")}>
        <IsoIcon name={toolIcon(it.tool)} size={16} accent={it.status === "running"} />
        <span className="flex h-4 w-4 shrink-0 items-center justify-center">
          <StatusIcon status={it.status} />
        </span>
        <span className={cn("shrink-0 text-sm", it.status === "running" ? "text-ink" : "text-ink-2")}>
          {toolLabel(it.tool, it.args)}
          {it.status === "running" ? "…" : ""}
        </span>
        {it.count > 1 && <span className="num font-mono text-[11px] text-ink-3">×{it.count}</span>}
        {!open && it.summary && it.status !== "running" && <span className="min-w-0 flex-1 truncate text-xs text-ink-3">{it.summary}</span>}
        {hasDetails && <ChevronRight className={cn("ml-auto h-3.5 w-3.5 shrink-0 text-ink-4 transition-transform group-hover:text-ink-3", open && "rotate-90")} />}
      </button>
      {open && (
        <div className="ml-[50px] mt-1 space-y-1.5 border-l border-line pl-3 text-xs">
          <p className="font-mono text-[11px] text-ink-3">{it.tool}</p>
          {it.summary && <p className="text-ink-2">{it.summary}</p>}
          {args.length > 0 && (
            <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1">
              {args.map(([k, v]) => (
                <div key={k} className="contents">
                  <dt className="font-mono text-[10px] uppercase tracking-[0.12em] text-ink-3">{k.replace(/_/g, " ")}</dt>
                  <dd className="truncate text-ink-2">{friendly(v)}</dd>
                </div>
              ))}
            </dl>
          )}
          {it.refIds.length > 0 && <p className="font-mono text-[11px] text-ink-3">{it.refIds.join(", ")}</p>}
        </div>
      )}
    </div>
  );
}

/** Values for humans: lists and objects become short readable text, never JSON. */
export function friendly(v: unknown): string {
  if (v === null || v === undefined) return "—";
  if (Array.isArray(v)) return v.length > 4 ? `${v.length} items` : v.map(friendly).join(", ");
  if (typeof v === "object")
    return Object.entries(v as Record<string, unknown>)
      .map(([k, x]) => `${k.replace(/_/g, " ")} ${friendly(x)}`)
      .join(" · ");
  return evidenceValue(v);
}
