import { useEffect, useMemo, useRef } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "../api/client";
import { ErrorState, SkeletonRows } from "../components/ui";
import { GlowEdge } from "../components/ui/spotlight-card";
import { ago, cn } from "../lib/format";
import { agentsApi, type AgentCall } from "../agents/api";
import { Timeline } from "../agents/Timeline";
import { WorkBoard } from "../agents/Work";
import { isActive, useSessionWork } from "../agents/sessionWork";

const label = "font-mono text-[10.5px] uppercase tracking-[0.14em]";

/** Who made a call, in words: the MCP client's name, the studio agent, a person or Autopilot. */
function who(c: AgentCall): string {
  if (c.client) return c.client;
  if (c.actor === "agent") return "studio agent";
  if (c.actor === "autopilot") return "autopilot";
  if (c.actor === "gui") return "you";
  if (c.actor === "mcp") return "MCP agent";
  return c.actor ?? "agent";
}

/**
 * The chat's log mode: watch what every agent is doing on this project without talking to it. Calls from MCP
 * sessions and the studio agent, the live work they started (demos, training, evaluations, RL) and anything
 * waiting on a person. Read-only. Switch back to Chat to step in.
 */
export function AgentLog({ pid, onIntervene }: { pid: string; onIntervene: () => void }) {
  const qc = useQueryClient();
  const calls = useQuery({ queryKey: ["agents", "project-calls", pid], queryFn: () => agentsApi.projectCalls(pid), refetchInterval: 15_000 });
  const { work, loading } = useSessionWork([pid]);

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
  const running = [...list].reverse().find((c) => c.status === "running");
  const last = list[list.length - 1];
  const recent = list.filter((c) => Date.now() - Date.parse(c.ended_at ?? c.started_at) < 10 * 60_000);
  const agents = [...new Set(recent.map(who))];
  const w = work[0];
  const busyJobs = w ? w.jobs.filter((j) => isActive(j.status)).length + w.sandboxes.filter((s) => s.status === "running" || s.status === "piloting").length : 0;
  const approvals = (w?.training.filter((t) => t.status === "proposed").length ?? 0) + list.filter((c) => c.status === "approval_required").length;
  const live = !!running || busyJobs > 0;

  return (
    <div className="min-h-0 flex-1 overflow-y-auto scroll-thin">
      <div className="mx-auto max-w-[920px] space-y-6 px-5 py-6">
        <div className={cn("sk-depth relative rounded border bg-plate px-4 py-3", live ? "border-servo/60" : "border-line-strong")}>
          <GlowEdge />
          <div className="flex flex-col gap-3">
            <div className="flex min-w-0 items-center gap-2.5">
              <span className={cn("h-2.5 w-2.5 shrink-0 rounded-full", running ? "live-dot bg-servo" : live ? "bg-pass" : "bg-ink-4")} />
              <span className={cn("shrink-0 text-ink-3", label)}>now</span>
              <span className="min-w-0 text-[14px] leading-5 text-ink [overflow-wrap:anywhere]">
                {running ? (
                  <>
                    {who(running)} is calling <span className="font-mono">{running.tool}</span>…
                  </>
                ) : last ? (
                  <>
                    {who(last)} · <span className="font-mono">{last.tool}</span>{" "}
                    <span className="text-ink-3">· {last.summary ?? last.status} · {ago(last.ended_at ?? last.started_at)}</span>
                  </>
                ) : (
                  "No agent has worked on this project yet."
                )}
              </span>
            </div>
            <div className="flex flex-wrap gap-x-8 gap-y-2">
              <div className="min-w-0">
                <div className={cn("text-ink-3", label)}>agents (10 min)</div>
                <div className="truncate text-[13px] text-ink">{agents.length ? agents.join(", ") : "none"}</div>
              </div>
              <div>
                <div className={cn("text-ink-3", label)}>background jobs</div>
                <div className="num text-[13px] text-ink">{busyJobs}</div>
              </div>
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

        <WorkBoard work={work} loading={loading} compact />

        <section>
          <div className="mb-2.5 flex items-center gap-2.5">
            <h3 className={cn("text-ink", label)}>Agent log</h3>
            <span className={cn("text-ink-3", label)}>{list.length} calls · every agent on this project</span>
            <span aria-hidden className="h-px flex-1 bg-line" />
          </div>
          {calls.isLoading ? (
            <SkeletonRows rows={4} />
          ) : calls.error ? (
            <ErrorState error={calls.error} onRetry={() => calls.refetch()} />
          ) : list.length === 0 ? (
            <p className="text-sm text-ink-3">Nothing yet. Calls from MCP agents and the studio agent appear here as they happen.</p>
          ) : (
            <Timeline calls={list} />
          )}
        </section>
      </div>
    </div>
  );
}
