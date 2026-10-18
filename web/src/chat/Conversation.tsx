import { useEffect, useMemo, useRef, useState } from "react";
import { ArrowUp, Check, ChevronRight, CircleDashed, RotateCcw, Square, X } from "lucide-react";
import { useTraining } from "../api/hooks";
import { cn, evidenceValue } from "../lib/format";
import { ApprovalItem, BudgetCard, ResultCards, TrainingApproval } from "./cards";
import { useChat } from "./ChatProvider";
import { Markdown } from "./Markdown";
import { isCardRef, prefixOf, toolLabel, type ChatItem } from "./model";

const SUGGESTIONS = [
  "Where does this project stand?",
  "Collect 6 demonstrations and check their quality",
  "Train a policy on the demonstrations",
  "Test the policy in 3 conditions and tell me what failed",
];

export function Conversation({ pid }: { pid: string }) {
  const chat = useChat();
  const scroller = useRef<HTMLDivElement>(null);
  const [confirmReset, setConfirmReset] = useState(false);

  // stick to the bottom while new content streams in
  const atBottom = useRef(true);
  useEffect(() => {
    const el = scroller.current;
    if (el && atBottom.current) el.scrollTop = el.scrollHeight;
  }, [chat.items, chat.busy]);

  // Which result cards render where: each ref at its latest mention; a training approval only once (latest).
  const plan = useMemo(() => {
    const cards = new Map<number, string[]>();
    const approvalShown = new Map<number, boolean>();
    const seen = new Set<string>();
    let trainingApproval = false;
    for (let i = chat.items.length - 1; i >= 0; i--) {
      const it = chat.items[i];
      if (it.kind === "tool") {
        const refs = it.refIds.filter((r) => isCardRef(r) && !seen.has(r));
        refs.forEach((r) => seen.add(r));
        if (refs.length) cards.set(it.id, refs);
        if (refs.some((r) => prefixOf(r) === "tj")) {
          approvalShown.set(it.id, !trainingApproval);
          trainingApproval = true;
        }
      } else if (it.kind === "approval") {
        const isTraining = it.approvalKind !== "evaluation";
        approvalShown.set(it.id, isTraining ? !trainingApproval : true);
        if (isTraining) trainingApproval = true;
      }
    }
    return { cards, approvalShown, anyTrainingApproval: trainingApproval };
  }, [chat.items]);

  const empty = !chat.loading && chat.items.length === 0;
  const lastRunning = [...chat.items].reverse().find((i) => i.kind === "tool" && i.status === "running");

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex h-11 shrink-0 items-center gap-2 border-b border-line px-4">
        <span className="text-sm font-semibold">Agent</span>
        <div className="ml-auto">
          {confirmReset ? (
            <span className="flex items-center gap-1 text-xs text-ink-2">
              Start over?
              <button type="button" className="rounded px-1.5 py-0.5 text-ink hover:bg-rail" onClick={() => (setConfirmReset(false), void chat.reset())}>
                Yes
              </button>
              <button type="button" className="rounded px-1.5 py-0.5 hover:bg-rail" onClick={() => setConfirmReset(false)}>
                No
              </button>
            </span>
          ) : (
            <button type="button" onClick={() => setConfirmReset(true)} disabled={chat.items.length === 0} className="inline-flex items-center gap-1.5 rounded-md px-2 py-1 text-xs text-ink-3 hover:bg-rail hover:text-ink disabled:opacity-40">
              <RotateCcw className="h-3.5 w-3.5" /> New chat
            </button>
          )}
        </div>
      </div>

      <div
        ref={scroller}
        onScroll={(e) => {
          const el = e.currentTarget;
          atBottom.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
        }}
        className="min-h-0 flex-1 overflow-y-auto scroll-thin"
      >
        <div className="mx-auto max-w-[680px] space-y-4 px-5 py-6">
          {chat.loading && (
            <div className="space-y-3">
              <div className="skeleton ml-auto h-10 w-2/3 rounded-2xl" />
              <div className="skeleton h-16 w-full rounded-xl" />
            </div>
          )}
          {!!chat.loadError && <p className="text-sm text-fault">Couldn't load the conversation. Is the backend running?</p>}
          {empty && (
            <div className="pt-6">
              <p className="text-xl font-semibold tracking-[-0.015em]">What should we do next?</p>
              <p className="mt-1.5 text-sm text-ink-3">The agent collects demonstrations, trains, tests, and improves the policy. It asks before anything costs money or needs a GPU.</p>
            </div>
          )}
          {chat.items.map((it) => (
            <Item key={it.id} it={it} pid={pid} cards={plan.cards.get(it.id)} showApproval={plan.approvalShown.get(it.id) ?? true} />
          ))}
          {chat.busy && !lastRunning && <Thinking startedAt={chat.turnStartedAt} />}
        </div>
      </div>

      <div className="shrink-0 px-4 pb-4">
        <div className="mx-auto max-w-[680px]">
          {!plan.anyTrainingApproval && <PendingApproval pid={pid} />}
          {empty && (
            <div className="mb-2.5 flex flex-wrap gap-1.5">
              {SUGGESTIONS.map((s) => (
                <button key={s} type="button" onClick={() => chat.send(s)} className="rounded-full bg-plate px-3 py-1.5 text-xs text-ink-2 shadow-[0_0_0_1px_var(--line)] transition-colors hover:text-ink hover:shadow-[0_0_0_1px_var(--line-strong)]">
                  {s}
                </button>
              ))}
            </div>
          )}
          <Composer busy={chat.busy} onSend={chat.send} onStop={chat.stop} />
        </div>
      </div>
    </div>
  );
}

function Item({ it, pid, cards, showApproval }: { it: ChatItem; pid: string; cards?: string[]; showApproval: boolean }) {
  switch (it.kind) {
    case "user":
      return (
        <div className="enter flex justify-end">
          <div className="max-w-[85%] whitespace-pre-wrap rounded-2xl rounded-br-md bg-rail px-4 py-2.5 text-[15px] leading-relaxed text-ink shadow-[0_0_0_1px_var(--line)]">{it.text}</div>
        </div>
      );
    case "assistant":
      return (
        <div className="enter">
          <Markdown text={it.text} />
        </div>
      );
    case "tool":
      return (
        <div className="space-y-2.5">
          <ToolRow it={it} />
          {it.status === "error" && /budget/i.test(it.summary) && <BudgetCard pid={pid} message={it.summary} />}
          {cards && cards.length > 0 && (showApproval ? <ResultCards pid={pid} refs={cards} /> : <ResultCards pid={pid} refs={cards.filter((r) => prefixOf(r) !== "tj")} />)}
        </div>
      );
    case "approval":
      return showApproval ? <ApprovalItem pid={pid} kind={it.approvalKind} refId={it.refId} summary={it.summary} args={it.args} /> : null;
    case "meta":
      return (
        <div className="flex items-center gap-2 text-[11px] text-ink-4" title={it.costUsd ? `agent cost $${it.costUsd.toFixed(2)}` : undefined}>
          <span className="h-px flex-1 bg-[var(--line-soft)]" />
          {it.isError ? "Turn ended with an error" : `Done${it.durationMs ? ` in ${Math.round(it.durationMs / 1000)}s` : ""}`}
          <span className="h-px flex-1 bg-[var(--line-soft)]" />
        </div>
      );
    case "error":
      return <div className="rounded-xl bg-fault/5 px-4 py-2.5 text-sm text-fault shadow-[0_0_0_1px_rgb(var(--fault)/.25)]">{it.text}</div>;
  }
}

/** A tool call as a human status row: label, state, collapsed details (never raw JSON). */
function ToolRow({ it }: { it: Extract<ChatItem, { kind: "tool" }> }) {
  const [open, setOpen] = useState(false);
  const args = Object.entries(it.args).filter(([k]) => k !== "project_id");
  const hasDetails = !!it.summary || args.length > 0;
  return (
    <div className="enter">
      <button type="button" onClick={() => hasDetails && setOpen((o) => !o)} className={cn("group flex w-full items-center gap-2.5 rounded-lg py-1 text-left", hasDetails && "cursor-pointer")}>
        <span className="flex h-5 w-5 shrink-0 items-center justify-center">
          {it.status === "running" ? (
            <CircleDashed className="h-4 w-4 animate-spin text-servo" />
          ) : it.status === "error" ? (
            <span className="flex h-4 w-4 items-center justify-center rounded-full bg-fault/15 text-fault">
              <X className="h-2.5 w-2.5" />
            </span>
          ) : (
            <span className="flex h-4 w-4 items-center justify-center rounded-full bg-pass/15 text-pass">
              <Check className="h-2.5 w-2.5" />
            </span>
          )}
        </span>
        <span className={cn("text-sm", it.status === "running" ? "text-ink" : "text-ink-2")}>
          {toolLabel(it.tool, it.args)}
          {it.status === "running" ? "…" : ""}
        </span>
        {it.count > 1 && <span className="num font-mono text-[11px] text-ink-4">×{it.count}</span>}
        {!open && it.summary && it.status !== "running" && <span className="min-w-0 flex-1 truncate text-xs text-ink-3">{it.summary}</span>}
        {hasDetails && <ChevronRight className={cn("ml-auto h-3.5 w-3.5 shrink-0 text-ink-4 transition-transform group-hover:text-ink-3", open && "rotate-90")} />}
      </button>
      {open && (
        <div className="ml-[30px] mt-1 space-y-1 rounded-lg bg-plate px-3 py-2 text-xs shadow-[0_0_0_1px_var(--line)]">
          {it.summary && <p className="text-ink-2">{it.summary}</p>}
          {args.length > 0 && (
            <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5">
              {args.map(([k, v]) => (
                <div key={k} className="contents">
                  <dt className="text-ink-3">{k.replace(/_/g, " ")}</dt>
                  <dd className="truncate text-ink-2">{friendly(v)}</dd>
                </div>
              ))}
            </dl>
          )}
        </div>
      )}
    </div>
  );
}

/** Values for humans: lists and objects become short readable text, never JSON. */
function friendly(v: unknown): string {
  if (v === null || v === undefined) return "—";
  if (Array.isArray(v)) return v.length > 4 ? `${v.length} items` : v.map(friendly).join(", ");
  if (typeof v === "object")
    return Object.entries(v as Record<string, unknown>)
      .map(([k, x]) => `${k.replace(/_/g, " ")} ${friendly(x)}`)
      .join(" · ");
  return evidenceValue(v);
}

function Thinking({ startedAt }: { startedAt: number | null }) {
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(t);
  }, []);
  const s = startedAt ? Math.max(0, Math.round((now - startedAt) / 1000)) : 0;
  return (
    <div className="flex items-center gap-2.5 py-1 text-sm text-ink-3">
      <span className="flex gap-1">
        {[0, 1, 2].map((i) => (
          <span key={i} className="h-1.5 w-1.5 rounded-full bg-ink-3 live-dot" style={{ animationDelay: `${i * 0.18}s` }} />
        ))}
      </span>
      Thinking{s >= 3 ? ` · ${s}s` : ""}
    </div>
  );
}

/** Work that is waiting on the user but was not raised in this conversation. */
function PendingApproval({ pid }: { pid: string }) {
  const training = useTraining(pid);
  const [open, setOpen] = useState(false);
  const proposed = [...(training.data ?? [])].filter((t) => t.status === "proposed").sort((a, b) => (a.created_at < b.created_at ? 1 : -1));
  if (!proposed.length) return null;
  const tj = proposed[0];
  return (
    <div className="mb-2.5">
      {open ? (
        <div className="space-y-1.5">
          <div className="flex justify-end">
            <button type="button" onClick={() => setOpen(false)} className="rounded px-1.5 text-xs text-ink-3 hover:bg-rail hover:text-ink">
              Collapse
            </button>
          </div>
          <TrainingApproval pid={pid} tj={tj} />
        </div>
      ) : (
        <button
          type="button"
          onClick={() => setOpen(true)}
          className="flex w-full items-center gap-2.5 rounded-xl bg-servo/[.07] px-3.5 py-2.5 text-left text-sm shadow-[0_0_0_1px_rgb(var(--servo)/.4)] hover:bg-servo/[.1]"
        >
          <span className="h-2 w-2 shrink-0 rounded-full bg-servo live-dot" />
          <span className="min-w-0 flex-1 truncate text-ink">
            Waiting for you: start {tj.policy_type.toUpperCase()} training ({tj.estimate.paid ? "paid" : "no charge"})
            {proposed.length > 1 ? ` · ${proposed.length} jobs` : ""}
          </span>
          <span className="shrink-0 text-xs font-medium text-servo">Review</span>
        </button>
      )}
    </div>
  );
}

function Composer({ busy, onSend, onStop }: { busy: boolean; onSend: (t: string) => void; onStop: () => void }) {
  const [text, setText] = useState("");
  const ref = useRef<HTMLTextAreaElement>(null);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "0px";
    el.style.height = `${Math.min(200, el.scrollHeight)}px`;
  }, [text]);
  const submit = () => {
    if (!text.trim()) return;
    onSend(text);
    setText("");
  };
  return (
    <form
      onSubmit={(e) => {
        e.preventDefault();
        submit();
      }}
      className="flex items-end gap-2 rounded-2xl bg-plate p-2 pl-4 shadow-[0_0_0_1px_var(--line-strong),0_8px_24px_-12px_rgba(0,0,0,.35)] focus-within:shadow-[0_0_0_1px_rgb(var(--servo)/.6),0_8px_24px_-12px_rgba(0,0,0,.35)]"
    >
      <textarea
        ref={ref}
        rows={1}
        value={text}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
            e.preventDefault();
            submit();
          }
        }}
        placeholder={busy ? "The agent is working. You can queue a follow-up…" : "Ask the agent…"}
        className="max-h-[200px] min-h-[24px] flex-1 resize-none bg-transparent py-1.5 text-[15px] leading-6 text-ink placeholder:text-ink-4 focus:outline-none"
        aria-label="Message the agent"
      />
      {busy && !text.trim() ? (
        <button type="button" onClick={onStop} className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-rail text-ink hover:bg-rail/70" aria-label="Stop">
          <Square className="h-3.5 w-3.5 fill-current" />
        </button>
      ) : (
        <button type="submit" disabled={!text.trim()} className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-servo text-servo-ink transition-transform active:scale-95 disabled:opacity-35" aria-label="Send">
          <ArrowUp className="h-4 w-4" />
        </button>
      )}
    </form>
  );
}
