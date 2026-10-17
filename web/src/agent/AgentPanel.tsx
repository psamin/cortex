import { useEffect, useRef, useState, type ReactNode } from "react";
import { useNavigate } from "react-router-dom";
import { useQueryClient } from "@tanstack/react-query";
import { ArrowUp, Bot, Check, CornerDownRight, ShieldAlert, Square, Wrench, X } from "lucide-react";
import { api, ApiError, streamAgent } from "../api/client";
import { useAgentHistory, useAudit } from "../api/hooks";
import type { AuditRow } from "../api/types";
import { Badge, Button, EmptyState, ErrorState, Input, Segmented, SkeletonRows } from "../components/ui";
import { ago, cn, errorMessage } from "../lib/format";

type Item =
  | { id: number; kind: "user"; text: string }
  | { id: number; kind: "assistant"; text: string; streaming: boolean }
  | { id: number; kind: "tool_call"; tool: string; args: Record<string, unknown> }
  | { id: number; kind: "tool_result"; tool: string; summary: string; ref_ids: string[] }
  | {
      id: number;
      kind: "approval";
      approvalKind: string;
      ref_id: string;
      summary: string;
      args: Record<string, unknown>;
      state: "pending" | "launched" | "declined";
      error?: string;
    }
  | { id: number; kind: "error"; message: string };

const SUGGESTED = [
  "What should I collect next, and why?",
  "Review my latest episodes for quality problems.",
  "Why are demos failing? Separate evidence from hypotheses.",
  "Propose an ACT training run on the newest dataset.",
  "Compare my evaluations and recommend next steps.",
];

let nextId = 1;

export function AgentPanel({ pid, registerAsk }: { pid: string; registerAsk: (fn: (m: string) => void) => void }) {
  const [tab, setTab] = useState<"chat" | "activity">("chat");
  return (
    <div className="flex h-full flex-col">
      <div className="flex h-11 shrink-0 items-center justify-between gap-2 border-b border-line pl-10 pr-3">
        <span className="flex items-center gap-1.5 text-sm font-semibold">
          <Bot className="h-4 w-4 text-servo" />
          Agent
        </span>
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
        <Chat pid={pid} registerAsk={registerAsk} />
      </div>
      {tab === "activity" && <Activity pid={pid} />}
    </div>
  );
}

// ------------------------------------------------------------------ chat
function Chat({ pid, registerAsk }: { pid: string; registerAsk: (fn: (m: string) => void) => void }) {
  const qc = useQueryClient();
  const history = useAgentHistory(pid);
  const [items, setItems] = useState<Item[]>([]);
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const hydrated = useRef(false);

  // hydrate from server history once
  useEffect(() => {
    if (hydrated.current || !history.data) return;
    hydrated.current = true;
    setItems(
      history.data.map((h): Item => {
        if (h.role === "user") return { id: nextId++, kind: "user", text: h.content };
        if (h.role === "tool" || h.tool)
          return { id: nextId++, kind: "tool_result", tool: h.tool ?? "tool", summary: h.content, ref_ids: [] };
        return { id: nextId++, kind: "assistant", text: h.content, streaming: false };
      }),
    );
  }, [history.data]);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight });
  }, [items]);

  const update = (fn: (xs: Item[]) => Item[]) => setItems(fn);

  async function send(message: string) {
    const text = message.trim();
    if (!text || busy) return;
    setDraft("");
    setBusy(true);
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    update((xs) => [...xs, { id: nextId++, kind: "user", text }]);
    let assistantId: number | null = null;
    try {
      for await (const ev of streamAgent(pid, text, ctrl.signal)) {
        if (ev.type === "text") {
          if (assistantId === null) {
            const id = nextId++;
            assistantId = id;
            update((xs) => [...xs, { id, kind: "assistant", text: ev.delta, streaming: true }]);
          } else {
            const id = assistantId;
            update((xs) => xs.map((x) => (x.id === id && x.kind === "assistant" ? { ...x, text: x.text + ev.delta } : x)));
          }
          continue;
        }
        // any non-text event closes the current assistant bubble
        if (assistantId !== null) {
          const id = assistantId;
          update((xs) => xs.map((x) => (x.id === id && x.kind === "assistant" ? { ...x, streaming: false } : x)));
          assistantId = null;
        }
        if (ev.type === "tool_call") update((xs) => [...xs, { id: nextId++, kind: "tool_call", tool: ev.tool, args: ev.args }]);
        else if (ev.type === "tool_result") {
          update((xs) => [...xs, { id: nextId++, kind: "tool_result", tool: ev.tool, summary: ev.summary, ref_ids: ev.ref_ids }]);
          qc.invalidateQueries({ predicate: (q) => q.queryKey.includes(pid) });
        } else if (ev.type === "approval_required")
          update((xs) => [
            ...xs,
            { id: nextId++, kind: "approval", approvalKind: ev.kind, ref_id: ev.ref_id, summary: ev.summary, args: ev.args, state: "pending" },
          ]);
        else if (ev.type === "error") update((xs) => [...xs, { id: nextId++, kind: "error", message: ev.message }]);
      }
    } catch (e) {
      const msg =
        e instanceof ApiError && e.unreachable ? "Agent unreachable — the backend is not responding." : errorMessage(e);
      update((xs) => [...xs, { id: nextId++, kind: "error", message: msg }]);
    } finally {
      update((xs) => xs.map((x) => (x.kind === "assistant" && x.streaming ? { ...x, streaming: false } : x)));
      setBusy(false);
      abortRef.current = null;
    }
  }

  const sendRef = useRef(send);
  sendRef.current = send;
  useEffect(() => registerAsk((m) => void sendRef.current(m)), [registerAsk]);

  const setApproval = (id: number, patch: Partial<Extract<Item, { kind: "approval" }>>) =>
    update((xs) => xs.map((x) => (x.id === id && x.kind === "approval" ? { ...x, ...patch } : x)));

  return (
    <div className="flex h-full flex-col">
      <div ref={scrollRef} className="min-h-0 flex-1 space-y-3 overflow-y-auto px-3 py-3 scroll-thin">
        {history.isLoading && <SkeletonRows rows={3} />}
        {!history.isLoading && items.length === 0 && (
          <div className="space-y-3 pt-2">
            <p className="text-sm text-ink-2">
              Ask Cortex to plan collection, diagnose failures, or set up training. It acts through the same API as this
              studio. Paid jobs and hardware motion always come back to you for approval.
            </p>
            {history.error && !(history.error instanceof ApiError && history.error.status === 404) ? (
              <ErrorState error={history.error} compact onRetry={() => history.refetch()} />
            ) : null}
          </div>
        )}
        {items.map((it) => (
          <ChatItem key={it.id} it={it} pid={pid} setApproval={setApproval} send={send} />
        ))}
        {busy && items[items.length - 1]?.kind !== "assistant" && (
          <div className="flex items-center gap-2 text-xs text-ink-3">
            <span className="h-1.5 w-1.5 rounded-full bg-servo live-dot" />
            Working…
          </div>
        )}
      </div>

      <div className="shrink-0 border-t border-line p-2.5">
        {items.length === 0 && (
          <div className="mb-2 flex flex-wrap gap-1">
            {SUGGESTED.map((s) => (
              <button
                key={s}
                type="button"
                onClick={() => send(s)}
                className="rounded-sm bg-rail px-2 py-1 text-left text-2xs text-ink-2 shadow-[0_0_0_1px_var(--line)] hover:text-ink"
              >
                {s}
              </button>
            ))}
          </div>
        )}
        <form
          onSubmit={(e) => {
            e.preventDefault();
            send(draft);
          }}
          className="flex items-end gap-1.5 rounded bg-well p-1.5 shadow-[inset_0_0_0_1px_var(--line)] focus-within:shadow-[inset_0_0_0_1px_rgb(var(--servo)/.6)]"
        >
          <textarea
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                send(draft);
              }
            }}
            rows={2}
            placeholder="Ask the agent…"
            className="min-h-[40px] flex-1 resize-none bg-transparent px-1 text-sm text-ink placeholder:text-ink-4 focus:outline-none"
            aria-label="Message the agent"
          />
          {busy ? (
            <Button size="sm" variant="secondary" onClick={() => abortRef.current?.abort()} aria-label="Stop">
              <Square className="h-3 w-3" />
            </Button>
          ) : (
            <Button size="sm" variant="primary" type="submit" disabled={!draft.trim()} aria-label="Send">
              <ArrowUp className="h-3.5 w-3.5" />
            </Button>
          )}
        </form>
      </div>
    </div>
  );
}

function ChatItem({
  it,
  pid,
  setApproval,
  send,
}: {
  it: Item;
  pid: string;
  setApproval: (id: number, patch: Partial<Extract<Item, { kind: "approval" }>>) => void;
  send: (m: string) => void;
}) {
  switch (it.kind) {
    case "user":
      return (
        <div className="enter ml-6 rounded-md bg-rail px-3 py-2 text-sm text-ink shadow-[0_0_0_1px_var(--line)]">{it.text}</div>
      );
    case "assistant":
      return (
        <div className="enter whitespace-pre-wrap text-sm leading-relaxed text-ink">
          {it.text}
          {it.streaming && <span className="ml-0.5 inline-block h-3.5 w-1.5 translate-y-0.5 bg-servo live-dot" />}
        </div>
      );
    case "tool_call":
      return (
        <div className="enter flex items-start gap-2 text-xs text-ink-3">
          <Wrench className="mt-0.5 h-3.5 w-3.5 shrink-0" />
          <div className="min-w-0">
            <span className="font-mono text-ink-2">{it.tool}</span>
            {Object.keys(it.args).length > 0 && (
              <span className="ml-1.5 break-all font-mono text-2xs text-ink-4">{compactArgs(it.args)}</span>
            )}
          </div>
        </div>
      );
    case "tool_result":
      return (
        <div className="enter ml-5 flex items-start gap-2 text-xs">
          <CornerDownRight className="mt-0.5 h-3.5 w-3.5 shrink-0 text-ink-4" />
          <div className="min-w-0 space-y-1">
            <p className="text-ink-2">
              <span className="font-mono text-ink-3">{it.tool}</span> · {it.summary}
            </p>
            {it.ref_ids.length > 0 && (
              <div className="flex flex-wrap gap-1">
                {it.ref_ids.map((r) => (
                  <RefChip key={r} pid={pid} id={r} />
                ))}
              </div>
            )}
          </div>
        </div>
      );
    case "approval":
      return <ApprovalCard it={it} pid={pid} setApproval={setApproval} send={send} />;
    case "error":
      return (
        <div className="enter rounded bg-fault/5 px-2.5 py-2 text-xs text-fault shadow-[0_0_0_1px_rgb(var(--fault)/.25)]">
          {it.message}
        </div>
      );
  }
}

function compactArgs(a: Record<string, unknown>) {
  const s = JSON.stringify(a);
  return s.length > 140 ? s.slice(0, 137) + "…" : s;
}

/** Map an id prefix to where that object lives in the studio. */
export function refRoute(pid: string, id: string): string | null {
  const p = id.split("_")[0];
  const map: Record<string, string> = {
    ep: `/p/${pid}/analyze?ep=${id}`,
    an: `/p/${pid}/analyze`,
    fc: `/p/${pid}/analyze`,
    rec: `/p/${pid}/improve?tab=recommendations`,
    ds: `/p/${pid}/improve?tab=datasets`,
    tj: `/p/${pid}/improve?tab=training`,
    ck: `/p/${pid}/improve?tab=training`,
    ev: `/p/${pid}/improve?tab=evaluations`,
    br: `/p/${pid}/improve?tab=evaluations`,
    prj: `/p/${pid}`,
  };
  return map[p] ?? null;
}

export function RefChip({ pid, id }: { pid: string; id: string }) {
  const nav = useNavigate();
  const to = refRoute(pid, id);
  return (
    <button
      type="button"
      disabled={!to}
      onClick={() => to && nav(to)}
      className="rounded-sm bg-rail px-1.5 py-px font-mono text-[10.5px] text-ink-2 shadow-[0_0_0_1px_var(--line)] hover:text-servo disabled:cursor-default disabled:hover:text-ink-2"
      title={to ? `Open ${id}` : id}
    >
      {id}
    </button>
  );
}

function useApprover() {
  const [name, setName] = useState(() => {
    try {
      return localStorage.getItem("cortex.approver") ?? "";
    } catch {
      return "";
    }
  });
  const save = (n: string) => {
    setName(n);
    try {
      localStorage.setItem("cortex.approver", n);
    } catch {
      /* ignore */
    }
  };
  return [name, save] as const;
}
export { useApprover };

function ApprovalCard({
  it,
  pid,
  setApproval,
  send,
}: {
  it: Extract<Item, { kind: "approval" }>;
  pid: string;
  setApproval: (id: number, patch: Partial<Extract<Item, { kind: "approval" }>>) => void;
  send: (m: string) => void;
}) {
  const qc = useQueryClient();
  const [approver, setApprover] = useApprover();
  const [pending, setPending] = useState(false);
  const isTraining = it.approvalKind === "training";
  const isEvaluation = it.approvalKind === "evaluation";
  const actionable = isTraining || isEvaluation;

  async function approve() {
    setPending(true);
    try {
      if (isTraining) {
        await api.approveTraining(it.ref_id, approver.trim());
        await api.launchTraining(it.ref_id);
        qc.invalidateQueries({ queryKey: ["training", pid] });
      } else {
        const policy = String(it.args.policy ?? "scripted_expert");
        await api.launchEvaluation(pid, {
          policy: policy === "scripted_expert" ? { kind: "scripted_expert" } : { kind: "checkpoint", checkpoint_id: policy },
          episodes_per_branch: typeof it.args.episodes_per_branch === "number" ? it.args.episodes_per_branch : 10,
        });
        qc.invalidateQueries({ queryKey: ["evaluations", pid] });
      }
      setApproval(it.id, { state: "launched", error: undefined });
    } catch (e) {
      setApproval(it.id, { error: errorMessage(e) });
    } finally {
      setPending(false);
    }
  }
  return (
    <div className="enter rounded-md bg-servo/[.04] p-3 shadow-[0_0_0_1px_rgb(var(--servo)/.35)]">
      <div className="mb-1.5 flex items-center gap-1.5 text-xs font-semibold text-servo">
        <ShieldAlert className="h-3.5 w-3.5" />
        Approval required · <span className="font-mono font-normal">{it.approvalKind}</span>
      </div>
      <p className="text-sm text-ink">{it.summary}</p>
      <div className="mt-1.5">
        <RefChip pid={pid} id={it.ref_id} />
      </div>

      {it.state === "pending" && actionable && (
        <div className="mt-3 space-y-2">
          {isTraining && (
            <Input value={approver} onChange={(e) => setApprover(e.target.value)} placeholder="Your name (recorded as approver)" />
          )}
          <div className="flex gap-1.5">
            <Button size="sm" variant="primary" disabled={isTraining && !approver.trim()} loading={pending} onClick={approve}>
              <Check className="h-3.5 w-3.5" /> {isTraining ? "Approve & launch" : "Approve & run"}
            </Button>
            <Button
              size="sm"
              variant="ghost"
              onClick={() => {
                setApproval(it.id, { state: "declined" });
                send(`I decline ${it.approvalKind} ${it.ref_id}. Do not launch it.`);
              }}
            >
              <X className="h-3.5 w-3.5" /> Decline
            </Button>
          </div>
        </div>
      )}
      {it.state === "pending" && !actionable && (
        <div className="mt-3 space-y-2">
          <p className="text-2xs text-ink-3">
            The API has no approve endpoint for “{it.approvalKind}”. Reply to the agent to approve or decline.
          </p>
          <div className="flex gap-1.5">
            <Button size="sm" variant="primary" onClick={() => send(`Approved: ${it.approvalKind} ${it.ref_id}.`)}>
              Approve via agent
            </Button>
            <Button
              size="sm"
              variant="ghost"
              onClick={() => {
                setApproval(it.id, { state: "declined" });
                send(`I decline ${it.approvalKind} ${it.ref_id}.`);
              }}
            >
              Decline
            </Button>
          </div>
        </div>
      )}
      {it.state === "launched" && (
        <Badge tone="pass" className="mt-3">
          {isTraining ? "approved & launched — see Improve › Training" : "evaluation started — see Improve › Evaluations"}
        </Badge>
      )}
      {it.state === "declined" && <Badge className="mt-3">declined</Badge>}
      {it.error && <p className="mt-2 text-xs text-fault">{it.error}</p>}
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
