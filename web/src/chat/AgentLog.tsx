import { useEffect, useMemo, useRef, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { api, resolveMediaUrl } from "../api/client";
import { ErrorState, SkeletonRows } from "../components/ui";
import { GlowEdge } from "../components/ui/spotlight-card";
import { ago, cn, humanize, pct } from "../lib/format";
import { IsoIcon, type IsoIconName } from "../sketch";
import { agentsApi, type AgentCall } from "../agents/api";
import { WorkBoard } from "../agents/Work";
import { isActive, useSessionWork, type ProjectWork } from "../agents/sessionWork";

const label = "font-mono text-[10.5px] uppercase tracking-[0.14em]";

/** Who made a call, in words: the MCP client's name, the studio agent, a person or Autopilot. */
function who(c: AgentCall): string {
  if (c.agent_name) return c.agent_name;
  if (c.client) return c.client;
  if (c.actor === "agent") return "studio agent";
  if (c.actor === "autopilot") return "autopilot";
  if (c.actor === "gui") return "you";
  if (c.actor === "mcp") return "MCP agent";
  return c.actor ?? "agent";
}

// ------------------------------------------------------------------ training stages

type StageState = "waiting" | "running" | "done" | "attention" | "failed";
interface Stage {
  name: string;
  icon: IsoIconName;
  state: StageState;
  note?: string;
}

const okCall = (calls: AgentCall[], tool: string) => calls.some((c) => c.tool === tool && c.status === "ok");

/** The training process, stage by stage, from what the agent called and the work it started. */
function stagesOf(calls: AgentCall[], w: ProjectWork | undefined): Stage[] {
  const built = calls.filter((c) => c.tool === "build_simulation" && c.status === "ok");
  const lastBuild = built[built.length - 1];
  const infeasible = !!lastBuild && /not feasible/.test(lastBuild.summary ?? "");
  const cols = w?.collections ?? [];
  const recs = (w?.jobs ?? []).filter((j) => j.kind === "record" || j.kind === "import");
  const demos = cols.reduce((n, c) => n + (c.totals?.successes ?? 0), 0);
  const demoRunning = cols.some((c) => isActive(c.status)) || recs.some((j) => isActive(j.status));
  const demoDone = demos > 0 || recs.some((j) => j.status === "succeeded");
  const demoFailed = !demoDone && !demoRunning && (cols.length > 0 || recs.length > 0);
  const tj = w?.training ?? [];
  const evs = w?.evaluations ?? [];
  const sbs = (w?.sandboxes ?? []).filter((s) => s.status !== "dismissed");
  const improving = sbs.some((s) => isActive(s.status)) || (w?.jobs ?? []).some((j) => j.kind === "improve_policy" && isActive(j.status));
  const evDone = evs.filter((e) => e.run.status === "succeeded");
  const lastEv = evDone[evDone.length - 1];
  const evRate = lastEv
    ? (() => {
        const rs = lastEv.branches.map((b) => b.result).filter(Boolean);
        const n = rs.reduce((a, r) => a + (r?.n ?? 0), 0);
        return n ? rs.reduce((a, r) => a + (r?.successes ?? 0), 0) / n : null;
      })()
    : null;
  return [
    {
      name: "Describe",
      icon: "object",
      state: built.length ? "done" : calls.some((c) => c.tool === "start_intake" || c.tool === "update_intake") ? "running" : "waiting",
    },
    {
      name: "Simulate",
      icon: "robot",
      state: !lastBuild ? "waiting" : infeasible ? "attention" : "done",
      note: infeasible ? "not feasible" : lastBuild ? "scene built" : undefined,
    },
    {
      name: "Demonstrate",
      icon: "demonstrate",
      state: demoRunning ? "running" : demoDone ? "done" : demoFailed ? "failed" : "waiting",
      note: demos || demoDone ? `${demos} demos` : demoFailed ? "no demos yet" : undefined,
    },
    {
      name: "Train",
      icon: "train",
      state: tj.some((t) => t.status === "proposed")
        ? "attention"
        : tj.some((t) => isActive(t.status))
          ? "running"
          : tj.some((t) => t.status === "succeeded" || t.checkpoint_ids.length)
            ? "done"
            : tj.some((t) => t.status === "failed")
              ? "failed"
              : "waiting",
      note: tj.some((t) => t.status === "proposed") ? "needs your OK" : undefined,
    },
    {
      name: "Evaluate",
      icon: "evaluate",
      state: evs.some((e) => isActive(e.run.status)) ? "running" : evDone.length ? "done" : "waiting",
      note: evRate != null ? `${pct(evRate)} success` : undefined,
    },
    {
      name: "Improve",
      icon: "improve",
      state: improving ? "running" : sbs.some((s) => s.status === "succeeded") || okCall(calls, "improve_policy") ? "done" : "waiting",
      note: sbs.length ? `${sbs.length} RL sandbox${sbs.length > 1 ? "es" : ""}` : undefined,
    },
  ];
}

const STAGE_TONE: Record<StageState, string> = {
  waiting: "text-ink-4 border-line",
  running: "text-servo border-servo/60",
  done: "text-ink border-line-strong",
  attention: "text-caution border-caution/60",
  failed: "text-fault border-fault/50",
};
const STAGE_WORD: Record<StageState, string> = { waiting: "not yet", running: "now", done: "done", attention: "needs you", failed: "failed" };

function Stages({ stages }: { stages: Stage[] }) {
  return (
    <ol className="grid grid-cols-3 gap-2 sm:grid-cols-6">
      {stages.map((s, i) => (
        <li key={s.name} className={cn("relative rounded border bg-plate px-2.5 py-2", STAGE_TONE[s.state])}>
          <div className="flex items-center gap-1.5">
            <IsoIcon name={s.icon} size={16} accent={s.state === "running"} />
            <span className="font-mono text-[10px] text-ink-3">{String(i + 1).padStart(2, "0")}</span>
            {s.state === "running" && <span className="ml-auto h-1.5 w-1.5 rounded-full bg-servo live-dot" />}
          </div>
          <div className="mt-1 truncate text-[13px] text-ink">{s.name}</div>
          <div className={cn("truncate font-mono text-[10px] uppercase tracking-[0.1em]", STAGE_TONE[s.state].split(" ")[0])}>
            {s.note ?? STAGE_WORD[s.state]}
          </div>
        </li>
      ))}
    </ol>
  );
}

// ------------------------------------------------------------------ the feed

type FeedItem =
  | { kind: "decision"; at: string; call: AgentCall }
  | { kind: "call"; at: string; call: AgentCall }
  | { kind: "event"; at: string; icon: IsoIconName; text: string; tone?: "ok" | "bad" | "live" };

/** Stage events from the work the agent started: demos, training, evaluations, RL sandboxes. */
function eventsOf(w: ProjectWork | undefined): FeedItem[] {
  if (!w) return [];
  const out: FeedItem[] = [];
  for (const c of w.collections) {
    const t = c.totals ?? {};
    const tone = isActive(c.status) ? "live" : c.status === "succeeded" ? "ok" : c.status === "failed" ? "bad" : undefined;
    out.push({ kind: "event", at: c.created_at, icon: "demonstrate", tone, text: `Demonstrations (${c.collector}): ${humanize(c.status)}, ${t.successes ?? 0} of ${c.request?.num_demos ?? "?"} demos${c.error ? `. ${c.error}` : ""}` });
  }
  for (const j of w.jobs.filter((x) => x.kind === "record" || x.kind === "import" || x.kind === "dataset")) {
    const tone = isActive(j.status) ? "live" : j.status === "succeeded" ? "ok" : j.status === "failed" ? "bad" : undefined;
    out.push({ kind: "event", at: j.created_at ?? "", icon: j.kind === "dataset" ? "dataset" : "demonstrate", tone, text: `${humanize(j.kind)}: ${humanize(j.status)}${j.message ? `. ${j.message}` : ""}` });
  }
  for (const t of w.training) {
    const tone = isActive(t.status) && t.status !== "proposed" ? "live" : t.status === "succeeded" ? "ok" : t.status === "failed" ? "bad" : undefined;
    out.push({ kind: "event", at: t.created_at, icon: "train", tone, text: t.status === "proposed" ? `Training ${t.policy_type.toUpperCase()} proposed: waiting for a person to approve the GPU spend` : `Training ${t.policy_type.toUpperCase()}: ${humanize(t.status)}${t.checkpoint_ids.length ? `, ${t.checkpoint_ids.length} checkpoint(s)` : ""}` });
  }
  for (const e of w.evaluations) {
    const tone = isActive(e.run.status) ? "live" : e.run.status === "succeeded" ? "ok" : e.run.status === "failed" ? "bad" : undefined;
    const parts = e.branches.filter((b) => b.result).map((b) => `${humanize(b.spec.name)} ${b.result!.successes}/${b.result!.n}`);
    out.push({ kind: "event", at: e.run.created_at, icon: "evaluate", tone, text: `Evaluation of ${e.run.policy_label}: ${humanize(e.run.status)}${parts.length ? ` (${parts.join(", ")})` : ""}` });
  }
  return out;
}

const DECISION_WORD: Record<string, string> = { plan: "Plan", decision: "Decision", observation: "Noticed", question: "Question for you" };

function Decision({ c }: { c: AgentCall }) {
  const kind = String(c.args.kind ?? "decision");
  return (
    <li className={cn("relative rounded border-l-2 bg-plate py-2.5 pl-3.5 pr-3", kind === "question" ? "border-caution" : "border-ink")}>
      <div className="flex items-center gap-2 text-xs">
        <span className={cn(kind === "question" ? "text-caution" : "text-ink", label)}>{DECISION_WORD[kind] ?? kind}</span>
        <span className="text-ink-3">{who(c)} · {ago(c.started_at)}</span>
      </div>
      <p className="mt-1 text-[14px] leading-6 text-ink">{String(c.args.text ?? c.summary ?? "")}</p>
    </li>
  );
}

function CallRow({ c }: { c: AgentCall }) {
  const [open, setOpen] = useState(false);
  const bad = c.status === "error";
  return (
    <li className="rounded px-1">
      <button type="button" onClick={() => setOpen((o) => !o)} className="flex w-full items-start gap-2.5 py-1.5 text-left" aria-expanded={open}>
        <span
          className={cn(
            "mt-[7px] h-1.5 w-1.5 shrink-0 rounded-full",
            c.status === "running" ? "live-dot bg-servo" : bad ? "bg-fault" : c.status === "approval_required" ? "bg-caution" : "bg-ink-4",
          )}
        />
        <span className="min-w-0 flex-1 text-[13px] leading-5">
          <span className="font-mono text-ink">{c.tool}</span>
          <span className={cn("ml-2", bad ? "text-fault" : "text-ink-3")}>{c.status === "running" ? "running…" : (c.error ?? c.summary ?? c.status)}</span>
        </span>
        <span className="shrink-0 pt-px font-mono text-[11px] text-ink-4">
          {who(c)} · {c.duration_ms != null ? `${c.duration_ms} ms · ` : ""}
          {ago(c.started_at)}
        </span>
      </button>
      {(c.image_urls ?? []).length > 0 && (
        <div className="mb-1.5 ml-4 flex gap-1.5">
          {c.image_urls!.map((u) => (
            <img key={u} src={resolveMediaUrl(u) ?? u} alt={`${c.tool} image`} className="h-16 w-auto rounded-sm border border-line" />
          ))}
        </div>
      )}
      {open && (
        <pre className="mb-2 ml-4 max-h-56 overflow-auto whitespace-pre-wrap rounded-sm border border-line bg-well px-2.5 py-2 font-mono text-[11px] leading-5 text-ink-2">
          {JSON.stringify({ args: c.args, result: c.result_preview ? safeParse(c.result_preview) : undefined }, null, 2)}
        </pre>
      )}
    </li>
  );
}

const safeParse = (s: string) => {
  try {
    return JSON.parse(s);
  } catch {
    return s;
  }
};

function Event({ e }: { e: Extract<FeedItem, { kind: "event" }> }) {
  return (
    <li className="flex items-start gap-2.5 rounded-sm bg-rail/60 px-2.5 py-2">
      <IsoIcon name={e.icon} size={16} accent={e.tone === "live"} />
      <span className={cn("min-w-0 flex-1 text-[13px] leading-5", e.tone === "bad" ? "text-fault" : e.tone === "ok" ? "text-ink" : "text-ink-2")}>{e.text}</span>
      <span className="shrink-0 pt-px font-mono text-[11px] text-ink-4">{ago(e.at)}</span>
    </li>
  );
}

// ------------------------------------------------------------------ the view

/**
 * The chat's log mode: what the agent using the MCP is doing on this project, as it happens. The training
 * stages, then one feed of its plans and decisions (cortex_log_decision), its tool calls and the stage events
 * of the work it started, then the live work cards. Read-only; switch to Chat to step in.
 */
export function AgentLog({ pid, onIntervene }: { pid: string; onIntervene: () => void }) {
  const qc = useQueryClient();
  const calls = useQuery({ queryKey: ["agents", "project-calls", pid], queryFn: () => agentsApi.projectCalls(pid), refetchInterval: 15_000 });
  const { work, loading } = useSessionWork([pid]);
  const [onlyDecisions, setOnlyDecisions] = useState(false);

  // Every call row is an `entity` change on the project stream (start and end), whoever made it.
  const timer = useRef<ReturnType<typeof setTimeout>>();
  useEffect(() => {
    const es = new EventSource(api.eventsUrl(pid));
    es.addEventListener("entity", (m) => {
      try {
        if (JSON.parse((m as MessageEvent).data)?.table !== "tool_audit_logs") return;
      } catch {
        return;
      }
      clearTimeout(timer.current);
      timer.current = setTimeout(() => void qc.invalidateQueries({ queryKey: ["agents", "project-calls", pid] }), 250);
    });
    return () => {
      es.close();
      clearTimeout(timer.current);
    };
  }, [pid, qc]);

  const list = useMemo(() => calls.data ?? [], [calls.data]);
  const w = work[0];
  const stages = useMemo(() => stagesOf(list, w), [list, w]);
  const feed = useMemo<FeedItem[]>(() => {
    const items: FeedItem[] = list.map((c) => (c.tool === "log_decision" ? { kind: "decision", at: c.started_at, call: c } : { kind: "call", at: c.started_at, call: c }));
    const all = [...items, ...eventsOf(w)].filter((x) => !onlyDecisions || x.kind !== "call");
    return all.sort((a, b) => (a.at < b.at ? 1 : -1)); // newest first
  }, [list, w, onlyDecisions]);

  const running = [...list].reverse().find((c) => c.status === "running");
  const lastDecision = [...list].reverse().find((c) => c.tool === "log_decision");
  const approvals = (w?.training.filter((t) => t.status === "proposed").length ?? 0) + list.filter((c) => c.status === "approval_required").length;
  const narrates = list.some((c) => c.tool === "log_decision");

  return (
    <div className="min-h-0 flex-1 overflow-y-auto scroll-thin">
      <div className="mx-auto max-w-[920px] space-y-5 px-5 py-6">
        <Stages stages={stages} />

        <div className={cn("sk-depth relative rounded border bg-plate px-4 py-3", running ? "border-servo/60" : "border-line-strong")}>
          <GlowEdge />
          <div className="flex items-start gap-2.5">
            <span className={cn("mt-1.5 h-2.5 w-2.5 shrink-0 rounded-full", running ? "live-dot bg-servo" : "bg-ink-4")} />
            <div className="min-w-0 flex-1">
              <div className={cn("text-ink-3", label)}>now</div>
              <p className="mt-0.5 text-[14px] leading-6 text-ink [overflow-wrap:anywhere]">
                {running ? (
                  <>
                    {who(running)} is calling <span className="font-mono">{running.tool}</span>…
                  </>
                ) : list.length ? (
                  <>
                    {who(list[list.length - 1])} · <span className="font-mono">{list[list.length - 1].tool}</span>{" "}
                    <span className="text-ink-3">· {ago(list[list.length - 1].ended_at ?? list[list.length - 1].started_at)}</span>
                  </>
                ) : (
                  "No agent has worked on this project yet."
                )}
              </p>
              {lastDecision && (
                <p className="mt-1 text-[13px] leading-5 text-ink-2">
                  <span className="text-ink-3">Latest plan:</span> {String(lastDecision.args.text ?? "")}
                </p>
              )}
            </div>
          </div>
          {approvals > 0 && (
            <div className="mt-3 flex items-center justify-between gap-3 rounded-sm border border-caution/40 px-3 py-2 text-xs">
              <span className="text-caution">
                The agent is waiting on you: {approvals} approval{approvals > 1 ? "s" : ""}.
              </span>
              <button type="button" onClick={onIntervene} className="shrink-0 font-medium text-ink underline underline-offset-2">
                Switch to chat to respond
              </button>
            </div>
          )}
        </div>

        <section>
          <div className="mb-2.5 flex items-center gap-2.5">
            <h3 className={cn("text-ink", label)}>Agent log</h3>
            <span className={cn("text-ink-3", label)}>plans, tool calls and training stages · newest first</span>
            <span aria-hidden className="h-px flex-1 bg-line" />
            <button type="button" onClick={() => setOnlyDecisions((v) => !v)} className={cn("shrink-0 text-ink-3 underline-offset-2 hover:text-ink hover:underline", label)}>
              {onlyDecisions ? "show tool calls" : "decisions only"}
            </button>
          </div>
          {!narrates && list.length > 0 && (
            <p className="mb-2 text-xs text-ink-3">This agent hasn't shared its reasoning yet. Agents that follow Cortex's instructions log their plans with cortex_log_decision.</p>
          )}
          {calls.isLoading ? (
            <SkeletonRows rows={4} />
          ) : calls.error ? (
            <ErrorState error={calls.error} onRetry={() => calls.refetch()} />
          ) : feed.length === 0 ? (
            <p className="text-sm text-ink-3">Nothing yet. The agent's plans, tool calls and training stages appear here as they happen.</p>
          ) : (
            <ul className="space-y-1.5">
              {feed.map((it, i) =>
                it.kind === "decision" ? <Decision key={it.call.id} c={it.call} /> : it.kind === "call" ? <CallRow key={it.call.id} c={it.call} /> : <Event key={`e${i}`} e={it} />,
              )}
            </ul>
          )}
        </section>

        <WorkBoard work={work} loading={loading} compact />
      </div>
    </div>
  );
}
