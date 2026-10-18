import { useState, type ReactNode } from "react";
import { useAudit } from "../api/hooks";
import type { AuditRow } from "../api/types";
import { ChatThread } from "../chat/ChatThread";
import { EmptyState, ErrorState, Segmented, SkeletonRows } from "../components/ui";
import { AGENT_NAME } from "../lib/brand";
import { ago, cn } from "../lib/format";

export { RefChip, refRoute } from "./RefChip";
export { useApprover } from "../chat/approver";

/** The side panel on secondary pages: the same chat thread as the Chat page, plus the audit trail. */
export function AgentPanel({ pid, registerAsk }: { pid: string; registerAsk: (fn: (m: string) => void) => void }) {
  const [tab, setTab] = useState<"chat" | "activity">("chat");
  return (
    <div className="flex h-full flex-col">
      <div className="flex h-11 shrink-0 items-center justify-between gap-2 border-b border-line pl-10 pr-3">
        <span className="label !text-ink">{AGENT_NAME}</span>
        <Segmented
          size="sm"
          value={tab}
          onChange={setTab}
          ariaLabel="Agent panel view"
          options={[
            { value: "chat", label: "Chat" },
            { value: "activity", label: "Activity" },
          ]}
        />
      </div>
      <div className={cn("min-h-0 flex-1", tab !== "chat" && "hidden")}>
        <ChatThread pid={pid} variant="panel" registerAsk={registerAsk} />
      </div>
      {tab === "activity" && <Activity pid={pid} />}
    </div>
  );
}

// ------------------------------------------------------------------ activity
const actorStyle: Record<string, string> = {
  mcp: "text-servo bg-servo/10 shadow-[0_0_0_1px_rgb(var(--servo)/.3)]",
  autopilot: "text-servo-ink bg-servo",
  agent: "text-ink bg-rail shadow-[0_0_0_1px_var(--line-strong)]",
  gui: "text-ink-3 bg-transparent shadow-[0_0_0_1px_var(--line)]",
};

function Activity({ pid }: { pid: string }) {
  const q = useAudit(pid);
  const [filter, setFilter] = useState<"all" | "mcp" | "agent" | "autopilot" | "gui">("all");
  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="flex items-center justify-between gap-2 border-b border-line-soft px-3 py-2">
        <span className="text-2xs text-ink-3">Every API call, from any client</span>
        <Segmented
          size="sm"
          value={filter}
          onChange={setFilter}
          ariaLabel="Filter by actor"
          options={[
            { value: "all", label: "All" },
            { value: "mcp", label: "MCP" },
            { value: "agent", label: "Agent" },
            { value: "autopilot", label: "Auto" },
            { value: "gui", label: "GUI" },
          ]}
        />
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto scroll-thin">
        {q.isLoading ? (
          <div className="p-3">
            <SkeletonRows rows={6} />
          </div>
        ) : q.error ? (
          <div className="p-3">
            <ErrorState error={q.error} compact onRetry={() => q.refetch()} />
          </div>
        ) : (
          <AuditList rows={(q.data ?? []).filter((r) => filter === "all" || r.actor === filter)} />
        )}
      </div>
    </div>
  );
}

function AuditList({ rows }: { rows: AuditRow[] }) {
  if (rows.length === 0)
    return (
      <EmptyState title="No activity yet">
        Calls from this studio, the in-app agent, and Claude Code over MCP will appear here.
      </EmptyState>
    );
  const sorted = [...rows].sort((a, b) => (a.created_at < b.created_at ? 1 : -1));
  return (
    <ol className="divide-y divide-[var(--line-soft)]">
      {sorted.map((r, i) => (
        <AuditRowView key={`${r.created_at}-${i}`} r={r} />
      ))}
    </ol>
  );
}

function AuditRowView({ r }: { r: AuditRow }) {
  const [open, setOpen] = useState(false);
  const ok = /ok|succe|done|200/i.test(r.status);
  const bad = /fail|error|denied|refused/i.test(r.status);
  return (
    <li>
      <button type="button" onClick={() => setOpen((o) => !o)} className="flex w-full items-center gap-2 px-3 py-2 text-left hover:bg-rail/50">
        <span className={cn("w-12 shrink-0 rounded-sm py-px text-center font-mono text-[10px] font-medium uppercase", actorStyle[r.actor] ?? actorStyle.gui)}>
          {r.actor}
        </span>
        <span className="min-w-0 flex-1 truncate font-mono text-xs text-ink">{r.tool}</span>
        <span className={cn("text-2xs", ok ? "text-pass" : bad ? "text-fault" : "text-ink-3")}>{r.status}</span>
        <span className="w-14 shrink-0 text-right text-2xs text-ink-4">{ago(r.created_at)}</span>
      </button>
      {open && <Pre>{JSON.stringify(r.args, null, 2)}</Pre>}
    </li>
  );
}

function Pre({ children }: { children: ReactNode }) {
  return (
    <pre className="mx-3 mb-2 max-h-48 overflow-auto rounded bg-well p-2 font-mono text-[10.5px] leading-relaxed text-ink-2 scroll-thin">
      {children}
    </pre>
  );
}
