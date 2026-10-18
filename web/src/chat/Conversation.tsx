import { Fragment, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { ArrowDown, ArrowUp, Ban, Check, ChevronRight, CircleDashed, Download, History, Paperclip, Plus, Square, X } from "lucide-react";
import { api } from "../api/client";
import { useTraining } from "../api/hooks";
import { refRoute } from "../components/refs";
import { InlineError } from "../components/ui";
import { ago, cn, evidenceValue } from "../lib/format";
import { ApprovalItem, BudgetCard, ResultCards, TrainingApproval } from "./cards";
import { useChat } from "./ChatProvider";
import { ImportRow, ImportSheet, useImporter } from "./Imports";
import { Markdown } from "./Markdown";
import { IsoIcon } from "../sketch";
import { toolIcon } from "./toolIcon";
import { buildBlocks, prefixOf, toolLabel, type Block, type ChatItem, type ToolItem } from "./model";
import { GlowEdge } from "../components/ui/spotlight-card";

/** [chip label, message sent] */
const SUGGESTIONS: [string, string][] = [
  ["Status", "Where does this project stand?"],
  ["Collect 6 demos", "Collect 6 demonstrations and check their quality"],
  ["Train a policy", "Train a policy on the demonstrations"],
  ["Test in 3 conditions", "Test the policy in 3 conditions and tell me what failed"],
];

/** Within this many px of the end counts as "at the bottom". */
const STICK_PX = 48;

export function Conversation({ pid }: { pid: string }) {
  const chat = useChat();
  const scroller = useRef<HTMLDivElement>(null);
  const content = useRef<HTMLDivElement>(null);
  const blocks = useMemo(() => buildBlocks(chat.items), [chat.items]);
  const importer = useImporter(pid);
  const [sheet, setSheet] = useState<{ mode: "file" | "hub"; file: File | null } | null>(null);
  // drop a dataset archive anywhere on the chat column
  const [dragging, setDragging] = useState(false);
  const dragDepth = useRef(0);
  const hasFiles = (e: React.DragEvent) => Array.from(e.dataTransfer.types).includes("Files");

  // Scroll: follow new content only while the reader is at the bottom; otherwise offer a pill.
  const stuck = useRef(true);
  const [showJump, setShowJump] = useState(false);
  const [unseen, setUnseen] = useState(false);
  const toBottom = (smooth = false) => {
    const el = scroller.current;
    if (el) el.scrollTo({ top: el.scrollHeight, behavior: smooth ? "smooth" : "auto" });
  };
  useLayoutEffect(() => {
    if (stuck.current) toBottom();
    else setUnseen(true);
  }, [chat.items, chat.busy]);
  useEffect(() => {
    // cards load their data after mounting and grow: keep following while stuck
    const el = content.current;
    if (!el) return;
    const ro = new ResizeObserver(() => stuck.current && toBottom());
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  useEffect(() => {
    if (!chat.loading) {
      stuck.current = true;
      toBottom();
    }
  }, [chat.loading]);

  const trainingApprovalInChat = blocks.some(
    (b) => (b.type === "tools" && b.cards.some((r) => prefixOf(r) === "tj")) || (b.type === "item" && b.item.kind === "approval" && b.item.approvalKind !== "evaluation"),
  );
  const empty = !chat.loading && chat.items.length === 0;
  const lastRunning = [...chat.items].reverse().find((i) => i.kind === "tool" && i.status === "running");

  return (
    <div
      className="relative flex h-full min-h-0 flex-col"
      onDragEnter={(e) => {
        if (!hasFiles(e)) return;
        e.preventDefault();
        dragDepth.current += 1;
        setDragging(true);
      }}
      onDragOver={(e) => hasFiles(e) && e.preventDefault()}
      onDragLeave={() => {
        dragDepth.current = Math.max(0, dragDepth.current - 1);
        if (!dragDepth.current) setDragging(false);
      }}
      onDrop={(e) => {
        if (!hasFiles(e)) return;
        e.preventDefault();
        dragDepth.current = 0;
        setDragging(false);
        const f = e.dataTransfer.files[0];
        if (f) setSheet({ mode: "file", file: f });
      }}
    >
      {dragging && (
        <div className="pointer-events-none absolute inset-2 z-30 flex flex-col items-center justify-center gap-3 rounded border border-dashed border-servo bg-bench/95 text-center">
          <IsoIcon name="dataset" size={40} accent />
          <p className="kn-display text-[26px] text-ink">Drop a LeRobot dataset</p>
          <p className="text-xs text-ink-3">.zip or .tar.gz · up to 2 GB</p>
        </div>
      )}
      <div className="flex h-11 shrink-0 items-center gap-2 border-b border-line px-5">
        <span className="kn-label kn-label-rule text-ink">
          <span>Agent</span>
        </span>
        <div className="ml-auto flex items-center gap-3">
          <ChatHistory pid={pid} />
          <button
            type="button"
            onClick={() => void chat.newChat().catch(() => {})}
            disabled={chat.items.length === 0}
            title={chat.items.length === 0 ? "This chat is already empty" : "Start an empty chat; this one stays in History"}
            className="group inline-flex items-center gap-1.5 py-1 text-xs text-ink-3 hover:text-ink disabled:opacity-40"
          >
            <Plus className="h-3.5 w-3.5" strokeWidth={1.5} /> <span className="border-b border-line-strong group-hover:border-ink">New chat</span>
          </button>
        </div>
      </div>

      <div className="relative min-h-0 flex-1">
        <div
          ref={scroller}
          onScroll={(e) => {
            const el = e.currentTarget;
            const atEnd = el.scrollHeight - el.scrollTop - el.clientHeight < STICK_PX;
            stuck.current = atEnd;
            setShowJump(!atEnd);
            if (atEnd) setUnseen(false);
          }}
          className="h-full overflow-y-auto overscroll-contain scroll-thin"
        >
          <div ref={content} className="mx-auto max-w-[680px] space-y-5 px-5 py-7">
            {chat.loading && (
              <div className="space-y-3">
                <div className="skeleton h-6 w-2/3 rounded-sm" />
                <div className="skeleton h-16 w-full rounded-sm" />
              </div>
            )}
            {!!chat.loadError && <p className="text-sm text-fault">Couldn't load the chat. Is the backend running?</p>}
            {empty && (
              <div className="pt-6">
                <p className="kn-display text-[36px]">
                  What should we
                  <br />
                  <span className="kn-second">do next?</span>
                </p>
                <p className="mt-3 text-sm text-ink-3">Asks before anything paid or GPU.</p>
              </div>
            )}
            {blocks.map((b, i) => (
              <Fragment key={b.key}>
                {isAgentBlock(b) && !(i > 0 && isAgentBlock(blocks[i - 1])) && (
                  <div className="kn-label kn-label-rule -mb-2">
                    <span>
                      <span className="text-ink">Agent</span>
                    </span>
                  </div>
                )}
                <BlockView b={b} pid={pid} />
              </Fragment>
            ))}
            {chat.busy && !lastRunning && <Thinking startedAt={chat.turnStartedAt} />}
          </div>
        </div>
        {showJump && (
          <button
            type="button"
            onClick={() => {
              stuck.current = true;
              setUnseen(false);
              toBottom(true);
            }}
            className="enter absolute bottom-3 left-1/2 inline-flex -translate-x-1/2 items-center gap-1.5 rounded-full border border-line-strong bg-plate px-3 py-1.5 text-xs text-ink hover:border-ink"
          >
            <ArrowDown className="h-3.5 w-3.5" />
            {unseen ? "New activity" : "Jump to latest"}
            {unseen && <span className="h-1.5 w-1.5 rounded-full bg-servo" />}
          </button>
        )}
      </div>

      <div className="shrink-0 px-4 pb-4 pt-2">
        <div className="mx-auto max-w-[680px]">
          {!trainingApprovalInChat && <PendingApproval pid={pid} />}
          {empty && (
            <div className="mb-2.5 flex flex-wrap gap-1.5">
              {SUGGESTIONS.map(([label, s]) => (
                <button key={s} type="button" onClick={() => chat.send(s)} title={s} className="group inline-flex items-center gap-1.5 rounded-full border border-line-strong bg-plate px-3.5 py-1.5 text-xs text-ink-2 transition-colors hover:border-ink hover:text-ink">
                  {label}
                  <span aria-hidden className="text-ink-4 transition-transform group-hover:translate-x-0.5 group-hover:text-ink">→</span>
                </button>
              ))}
            </div>
          )}
          {sheet && (
            <ImportSheet
              mode={sheet.mode}
              file={sheet.file}
              onClose={() => setSheet(null)}
              onUpload={(f, n) => ((stuck.current = true), importer.upload(f, n))}
              onHub={(r, n) => ((stuck.current = true), importer.fromHub(r, n))}
            />
          )}
          <Composer
            busy={chat.busy}
            onSend={chat.send}
            onStop={chat.stop}
            onAttach={(f) => setSheet({ mode: "file", file: f })}
            onHub={() => setSheet({ mode: "hub", file: null })}
          />
        </div>
      </div>
    </div>
  );
}

/** True for blocks the agent produced (its text and tool rows), so one header covers a whole turn. */
function isAgentBlock(b: Block) {
  return b.type === "tools" || (b.item.kind !== "user" && b.item.kind !== "meta");
}

function BlockView({ b, pid }: { b: Block; pid: string }) {
  if (b.type === "tools")
    return (
      <div className="space-y-2.5">
        {b.tools.length === 1 ? <ToolRow it={b.tools[0]} /> : <ToolGroup tools={b.tools} />}
        {b.budgetError && <BudgetCard pid={pid} message={b.budgetError} />}
        {b.cards.length > 0 && <ResultCards pid={pid} refs={b.cards} />}
      </div>
    );
  const it = b.item;
  switch (it.kind) {
    case "user":
      return (
        <div className="enter border-l border-ink pl-4">
          <div className="kn-label mb-1.5">You</div>
          <div className="whitespace-pre-wrap text-[16px] leading-relaxed text-ink">{it.text}</div>
        </div>
      );
    case "assistant":
      return (
        <div className="enter">
          <Markdown text={it.text} />
        </div>
      );
    case "approval":
      return <ApprovalItem pid={pid} kind={it.approvalKind} refId={it.refId} summary={it.summary} args={it.args} />;
    case "job_done":
      return <JobDoneRow it={it} pid={pid} />;
    case "import":
      return <ImportRow it={it} pid={pid} />;
    case "meta":
      return (
        <div className="flex items-center gap-3 font-mono text-[10px] uppercase tracking-[0.16em] text-ink-3" title={it.costUsd ? `agent cost $${it.costUsd.toFixed(2)}` : undefined}>
          <span className="h-px flex-1 bg-[var(--line)]" />
          {it.isError ? "Ended with an error" : `Done${it.durationMs ? ` in ${Math.round(it.durationMs / 1000)}s` : ""}`}
          <span className="h-px flex-1 bg-[var(--line)]" />
        </div>
      );
    case "error":
      return (
        <div className="flex items-start gap-2.5 rounded border border-fault/35 bg-plate px-4 py-2.5 text-sm text-fault">
          <IsoIcon name="warning" size={18} className="mt-px" />
          <span className="min-w-0 flex-1">{it.text}</span>
        </div>
      );
  }
}

function StatusIcon({ status }: { status: "running" | "ok" | "error" | "cancelled" }) {
  if (status === "running") return <CircleDashed className="h-3.5 w-3.5 animate-spin text-servo" />;
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

/** Several tool calls in a row collapse into one expandable "n tool calls" row. */
function ToolGroup({ tools }: { tools: ToolItem[] }) {
  const [open, setOpen] = useState(false);
  const running = tools.find((t) => t.status === "running");
  const failed = tools.some((t) => t.status === "error");
  const calls = tools.reduce((n, t) => n + t.count, 0);
  const labels = [...new Set(tools.map((t) => toolLabel(t.tool, t.args)))];
  return (
    <div className="enter sk-depth relative rounded border border-line-strong bg-plate">
      <GlowEdge />
      <button type="button" onClick={() => setOpen((o) => !o)} className="group flex w-full items-center gap-2.5 px-3 py-2 text-left" aria-expanded={open}>
        <IsoIcon name={toolIcon((running ?? tools[tools.length - 1]).tool)} size={20} accent={!!running} />
        <span className="flex h-4 w-4 shrink-0 items-center justify-center">
          <StatusIcon status={running ? "running" : failed ? "error" : "ok"} />
        </span>
        <span className={cn("shrink-0 text-sm", running ? "text-ink" : "text-ink-2")}>
          {running ? `${toolLabel(running.tool, running.args)}…` : `${calls} tool calls`}
        </span>
        {!open && <span className="min-w-0 flex-1 truncate text-xs text-ink-3">{running ? `${calls} tool calls` : labels.join(" · ")}</span>}
        <ChevronRight className={cn("ml-auto h-3.5 w-3.5 shrink-0 text-ink-4 transition-transform group-hover:text-ink-3", open && "rotate-90")} />
      </button>
      {open && (
        <div className="space-y-1.5 border-t border-line px-3 py-2.5">
          {tools.map((t) => (
            <ToolRow key={t.id} it={t} nested />
          ))}
        </div>
      )}
    </div>
  );
}

const REF_NOUN: Record<string, string> = { ds: "dataset", ev: "evaluation", ac: "collection", tj: "training job", an: "analysis", ep: "demo", ck: "checkpoint", rec: "recommendation" };

/** A background job finished: one fixed line written by Cortex, with links to what it produced. */
function JobDoneRow({ it, pid }: { it: Extract<ChatItem, { kind: "job_done" }>; pid: string }) {
  const [showDetail, setShowDetail] = useState(false);
  const status = it.status === "succeeded" ? "ok" : it.status === "cancelled" ? "cancelled" : "error";
  if (it.superseded) return null; // a later run of the same kind succeeded: the old failure is history
  return (
    <div className="enter sk-depth relative flex items-start gap-2.5 rounded border border-line-strong bg-plate px-3 py-2.5">
      <GlowEdge />
      <IsoIcon name={toolIcon(it.jobKind)} size={20} />
      <span className="mt-0.5 flex h-4 w-4 shrink-0 items-center justify-center">
        <StatusIcon status={status} />
      </span>
      <div className="min-w-0 flex-1">
        <p className={cn("text-sm leading-snug", status === "error" ? "text-fault" : "text-ink-2")}>{it.text}</p>
        {it.detail && (
          <button type="button" onClick={() => setShowDetail((o) => !o)} className="mt-0.5 border-b border-line-strong text-xs text-ink-3 hover:border-ink hover:text-ink">
            {showDetail ? "Hide details" : "Details"}
          </button>
        )}
        {it.detail && showDetail && (
          <pre className="mt-1 max-h-40 overflow-auto whitespace-pre-wrap rounded-sm border border-line bg-well px-2 py-1.5 font-mono text-[11px] text-ink-3">{it.detail}</pre>
        )}
        {it.refIds.length > 0 && (
          <div className="mt-1 flex flex-wrap gap-x-3 gap-y-0.5">
            {it.refIds.map((r) => {
              const to = refRoute(pid, r);
              const label = `Open ${REF_NOUN[prefixOf(r)] ?? r}`;
              return to ? (
                <Link key={r} to={to} className="border-b border-line-strong text-xs text-ink hover:border-ink" title={r}>
                  {label}
                </Link>
              ) : (
                <span key={r} className="font-mono text-[11px] text-ink-3">
                  {r}
                </span>
              );
            })}
          </div>
        )}
      </div>
      <span className="shrink-0 pt-0.5 font-mono text-[10px] uppercase tracking-[0.14em] text-ink-3" title="Written by Cortex when the job finished">
        {it.jobKind.replace(/_/g, " ")}
      </span>
    </div>
  );
}

/** A tool call as a human status row: label, state, collapsed details (never raw JSON). */
function ToolRow({ it, nested }: { it: ToolItem; nested?: boolean }) {
  const [open, setOpen] = useState(false);
  const args = Object.entries(it.args).filter(([k]) => k !== "project_id");
  const hasDetails = !!it.summary || args.length > 0;
  return (
    <div className={cn("enter", !nested && "sk-depth relative rounded border border-line-strong bg-plate")}>
      {!nested && <GlowEdge />}
      <button type="button" onClick={() => hasDetails && setOpen((o) => !o)} className={cn("group flex w-full items-center gap-2.5 text-left", nested ? "py-0.5" : "px-3 py-2", hasDetails && "cursor-pointer")}>
        <IsoIcon name={toolIcon(it.tool)} size={nested ? 16 : 20} accent={it.status === "running"} />
        <span className="flex h-4 w-4 shrink-0 items-center justify-center">
          <StatusIcon status={it.status} />
        </span>
        <span className={cn("text-sm", it.status === "running" ? "text-ink" : "text-ink-2")}>
          {toolLabel(it.tool, it.args)}
          {it.status === "running" ? "…" : ""}
        </span>
        {it.count > 1 && <span className="num font-mono text-[11px] text-ink-3">×{it.count}</span>}
        {!open && it.summary && it.status !== "running" && <span className="min-w-0 flex-1 truncate text-xs text-ink-3">{it.summary}</span>}
        {hasDetails && <ChevronRight className={cn("ml-auto h-3.5 w-3.5 shrink-0 text-ink-4 transition-transform group-hover:text-ink-3", open && "rotate-90")} />}
      </button>
      {open && (
        <div className={cn("space-y-1.5 text-xs", nested ? "ml-[50px] mt-1 border-l border-line pl-3" : "border-t border-line px-3 py-2.5")}>
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
    <div className="flex items-center gap-3 py-1 font-mono text-[10.5px] uppercase tracking-[0.16em] text-ink-3">
      <span className="flex gap-1">
        {[0, 1, 2].map((i) => (
          <span key={i} className="h-1 w-1 bg-servo live-dot" style={{ animationDelay: `${i * 0.18}s` }} />
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
            <button type="button" onClick={() => setOpen(false)} className="border-b border-line-strong text-xs text-ink-3 hover:border-ink hover:text-ink">
              Collapse
            </button>
          </div>
          <TrainingApproval pid={pid} tj={tj} />
        </div>
      ) : (
        <button
          type="button"
          onClick={() => setOpen(true)}
          className="sk-depth sk-hover flex w-full items-center gap-2.5 rounded border border-servo/60 bg-plate px-3.5 py-2.5 text-left text-sm"
        >
          <IsoIcon name="approve" size={20} accent />
          <span className="h-1.5 w-1.5 shrink-0 rounded-full bg-servo live-dot" />
          <span className="min-w-0 flex-1 truncate text-ink">
            Approve {tj.policy_type.toUpperCase()} training · {tj.estimate.paid ? "paid" : "no charge"}
            {proposed.length > 1 ? ` · ${proposed.length} jobs` : ""}
          </span>
          <span className="shrink-0 border-b border-servo font-mono text-[10.5px] font-medium uppercase tracking-[0.14em] text-servo">Review →</span>
        </button>
      )}
    </div>
  );
}

function Composer({
  busy,
  onSend,
  onStop,
  onAttach,
  onHub,
}: {
  busy: boolean;
  onSend: (t: string) => void;
  onStop: () => void;
  onAttach: (f: File) => void;
  onHub: () => void;
}) {
  const [text, setText] = useState("");
  const ref = useRef<HTMLTextAreaElement>(null);
  const fileInput = useRef<HTMLInputElement>(null);
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
      className="sk-depth relative flex items-end gap-1.5 rounded border border-line-strong bg-plate p-2 transition-colors focus-within:border-ink"
    >
      <input
        ref={fileInput}
        type="file"
        accept=".zip,.tar.gz,.tgz,application/zip,application/gzip"
        className="hidden"
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) onAttach(f);
          e.target.value = "";
        }}
      />
      <button type="button" onClick={() => fileInput.current?.click()} className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full text-ink-3 hover:bg-rail hover:text-ink" aria-label="Attach demonstrations" title="Attach a LeRobot dataset (.zip or .tar.gz)">
        <Paperclip className="h-4 w-4" strokeWidth={1.5} />
      </button>
      <button type="button" onClick={onHub} className="flex h-9 shrink-0 items-center gap-1 rounded-full px-2 font-mono text-[10.5px] uppercase tracking-[0.12em] text-ink-3 hover:bg-rail hover:text-ink" aria-label="Import from Hugging Face" title="Import demonstrations from a Hugging Face dataset">
        <Download className="h-4 w-4" strokeWidth={1.5} /> Hub
      </button>
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
        placeholder={busy ? "Working… queue a follow-up" : "Ask the agent…"}
        className="max-h-[200px] min-h-[24px] flex-1 resize-none bg-transparent px-1 py-1.5 text-[15px] leading-6 text-ink placeholder:text-ink-4 focus:outline-none"
        aria-label="Message the agent"
      />
      {busy && !text.trim() ? (
        <button type="button" onClick={onStop} className="sk-pill sk-pill-secondary flex h-9 w-11 shrink-0 items-center justify-center" aria-label="Stop">
          <Square className="h-3 w-3 fill-current" />
        </button>
      ) : (
        <button type="submit" disabled={!text.trim()} className="sk-pill sk-pill-primary flex h-9 w-11 shrink-0 items-center justify-center disabled:opacity-35" aria-label="Send">
          <ArrowUp className="h-4 w-4" />
        </button>
      )}
    </form>
  );
}

/** Past chats of this project, newest first; picking one opens it and new messages continue it. */
function ChatHistory({ pid }: { pid: string }) {
  const chat = useChat();
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const threads = useQuery({ queryKey: ["agent-threads", pid], queryFn: () => api.agentThreads(pid), enabled: open, retry: 0 });
  useEffect(() => {
    if (!open) return;
    const off = (e: MouseEvent) => !ref.current?.contains(e.target as Node) && setOpen(false);
    const esc = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    document.addEventListener("mousedown", off);
    document.addEventListener("keydown", esc);
    return () => {
      document.removeEventListener("mousedown", off);
      document.removeEventListener("keydown", esc);
    };
  }, [open]);
  const list = threads.data ?? [];
  const current = chat.threadId ?? list[0]?.id ?? null;
  return (
    <div ref={ref} className="relative">
      <button type="button" onClick={() => setOpen((o) => !o)} aria-expanded={open} className="group inline-flex items-center gap-1.5 py-1 text-xs text-ink-3 hover:text-ink">
        <History className="h-3.5 w-3.5" strokeWidth={1.5} /> <span className="border-b border-line-strong group-hover:border-ink">History</span>
      </button>
      {open && (
        <div className="enter absolute right-0 top-full z-50 mt-1.5 max-h-[360px] w-72 overflow-y-auto rounded-xl bg-plate p-1.5 shadow-[0_0_0_1px_var(--line-strong),0_16px_32px_-12px_rgba(0,0,0,.5)] scroll-thin">
          {threads.isLoading ? (
            <div className="space-y-1.5 p-1.5">
              <div className="skeleton h-8 w-full rounded-md" />
              <div className="skeleton h-8 w-full rounded-md" />
            </div>
          ) : threads.error ? (
            <InlineError error={threads.error} fallback="Couldn't load past chats." className="p-2" />
          ) : list.length === 0 ? (
            <p className="p-2 text-xs text-ink-3">No past chats yet.</p>
          ) : (
            list.map((t) => (
              <button
                key={t.id}
                type="button"
                onClick={() => {
                  chat.openThread(t.id);
                  setOpen(false);
                }}
                className={cn("flex w-full items-baseline gap-2 rounded-lg px-2.5 py-2 text-left hover:bg-rail", t.id === current && "bg-rail")}
              >
                <span className="min-w-0 flex-1 truncate text-sm text-ink" title={t.title}>
                  {t.title || "Untitled chat"}
                </span>
                <span className="shrink-0 text-[11px] text-ink-4">{t.id === current ? "current" : ago(t.updated_at)}</span>
              </button>
            ))
          )}
        </div>
      )}
    </div>
  );
}
