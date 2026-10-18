import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { Link, useNavigate, useSearchParams } from "react-router-dom";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { ArrowDown, ArrowUp, Bot, Download, Hand, History, Info, Paperclip, Plus, Square } from "lucide-react";
import { api } from "../api/client";
import { useProjectState, useTraining } from "../api/hooks";
import { refRoute } from "../components/refs";
import { Segmented } from "../components/ui";
import { cn } from "../lib/format";
import { TrainingApproval } from "./cards";
import { useChat } from "./ChatProvider";
import { useChatHistoryPanel } from "./ChatHistoryPanel";
import { useViewerFocus } from "./focus";
import { ImportRow, ImportSheet, useImporter } from "./Imports";
import { IsoIcon } from "../sketch";
import { toolIcon } from "./toolIcon";
import { prefixOf, toolLabel, type ChatItem } from "./model";
import { buildTurns, endsOnFix } from "./timeline";
import { StatusIcon, TurnTimeline } from "./TurnTimeline";
import { GlowEdge } from "../components/ui/spotlight-card";
import { AgentLog } from "./AgentLog";

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
  const nav = useNavigate();
  // Chat: talk to the agent and step in. Agent log: just watch what every agent on this project is doing.
  const modeKey = `cortex.chatMode.${pid}`;
  const [params] = useSearchParams();
  const [mode, setModeState] = useState<"chat" | "log">(() => {
    const asked = params.get("view"); // ?view=log from the Agents page's "Open log"
    if (asked === "log" || asked === "chat") return asked;
    try {
      return localStorage.getItem(modeKey) === "log" ? "log" : "chat";
    } catch {
      return "chat";
    }
  });
  const setMode = (m: "chat" | "log") => {
    setModeState(m);
    try {
      localStorage.setItem(modeKey, m);
    } catch {
      /* storage unavailable */
    }
  };
  const scroller = useRef<HTMLDivElement>(null);
  const content = useRef<HTMLDivElement>(null);
  const turns = useMemo(() => buildTurns(chat.items), [chat.items]);
  // the scene's picture goes on the latest scene line only
  const thumbKey = useMemo(() => turns.flatMap((t) => t.milestones).filter((m) => m.stage === "scene").pop()?.key ?? null, [turns]);
  const agentMode = useAgentMode(pid);
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

  const trainingApprovalInChat = turns.some((t) =>
    t.milestones.some((m) => m.details.cardRefs.some((r) => prefixOf(r) === "tj") || m.details.approvals.some((a) => a.approvalKind !== "evaluation")),
  );
  const empty = !chat.loading && chat.items.length === 0;
  const projectState = useProjectState(pid);
  const hasNext = (projectState.data?.next_actions.length ?? 0) > 0;
  const autonomous = projectState.data?.project.autopilot.autonomy !== "manual";

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
        <Segmented
          size="sm"
          ariaLabel="View"
          value={mode}
          onChange={setMode}
          options={[
            { value: "chat", label: "Chat" },
            { value: "log", label: "Agent log" },
          ]}
        />
        {mode === "chat" && <ModeSwitch mode={agentMode.mode} onChange={agentMode.set} failed={agentMode.failed} />}
        <div className="ml-auto flex items-center gap-3">
          <HistoryToggle />
          <button
            type="button"
            onClick={() => nav("/studio")}
            title="Train something new: a new project with its own demos, datasets and policies (this chat stays in History)"
            className="group inline-flex items-center gap-1.5 py-1 text-xs text-ink-3 hover:text-ink"
          >
            <Plus className="h-3.5 w-3.5" strokeWidth={1.5} /> <span className="border-b border-line-strong group-hover:border-ink">New task</span>
          </button>
        </div>
      </div>

      {mode === "log" ? (
        <AgentLog pid={pid} onIntervene={() => setMode("chat")} />
      ) : (
        <>
      <p className="shrink-0 border-b border-line px-5 py-1.5 text-[12px] leading-snug text-ink-3">{modeLine(agentMode.mode, agentMode.cluster)}</p>
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
                <p className="mt-3 text-sm text-ink-3">
                  {autonomous ? "Runs the whole loop on its own. Needs you only for real robot motion and spend over your caps." : "One step at a time. Asks before anything paid or GPU."}
                </p>
              </div>
            )}
            {turns.map((t) => (
              <TurnTimeline key={t.key} turn={t} pid={pid} thumbKey={thumbKey} renderItem={(it) => <ItemView it={it} pid={pid} />} />
            ))}
            {chat.busy && <LiveStatus status={chat.status} startedAt={chat.turnStartedAt} />}
            {/* a turn that ends on a failed line with a fix: that button is the next step */}
            {!chat.busy && !chat.loading && chat.items.length > 0 && !endsOnFix(turns) && <NextSteps pid={pid} onPick={chat.send} />}
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
          {empty && hasNext && (
            <div className="mb-2.5">
              <NextSteps pid={pid} onPick={chat.send} />
            </div>
          )}
          {empty && !hasNext && (
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
        </>
      )}
    </div>
  );
}

/** Rows that stay on the old path inside a turn: imports, errors, changes, and notices about nothing on a line. */
function ItemView({ it, pid }: { it: ChatItem; pid: string }) {
  switch (it.kind) {
    case "job_done":
      return <JobDoneRow it={it} pid={pid} />;
    case "changes":
      return <ChangesRow it={it} pid={pid} />;
    case "notice":
      return <NoticeRow it={it} />;
    case "import":
      return <ImportRow it={it} pid={pid} />;
    case "error":
      return (
        <div className="flex items-start gap-2.5 rounded border border-fault/35 bg-plate px-4 py-2.5 text-sm text-fault">
          <IsoIcon name="warning" size={18} className="mt-px" />
          <span className="min-w-0 flex-1">{it.text}</span>
        </div>
      );
    default:
      return null;
  }
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

/** One live line while a turn runs: what the agent is doing right now, and for how long, so it never looks frozen. */
function LiveStatus({ status, startedAt }: { status: { state: string; tool?: string; text?: string } | null; startedAt: number | null }) {
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(t);
  }, []);
  const s = startedAt ? Math.max(0, Math.round((now - startedAt) / 1000)) : 0;
  const text = status?.text ?? (status?.state === "running" && status.tool ? `${toolLabel(status.tool)}…` : "Thinking…");
  return (
    <div className="flex items-center gap-2.5 py-1 text-xs text-ink-3" role="status" aria-live="polite">
      <span className="flex shrink-0 gap-1" aria-hidden>
        {[0, 1, 2].map((i) => (
          <span key={i} className="h-1 w-1 bg-servo live-dot" style={{ animationDelay: `${i * 0.18}s` }} />
        ))}
      </span>
      <span key={text} className="enter min-w-0 truncate" title={text}>
        {text}
      </span>
      {s >= 3 && <span className="num shrink-0 font-mono text-[10.5px] text-ink-4">{s}s</span>}
    </div>
  );
}

/** Work that is waiting on the user but was not raised in this conversation. */
function PendingApproval({ pid }: { pid: string }) {
  const training = useTraining(pid);
  const autonomous = useProjectState(pid).data?.project.autopilot.autonomy !== "manual";
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
            {autonomous ? "Needs you · " : ""}Approve {tj.policy_type.toUpperCase()} training · {tj.estimate.paid ? "paid" : "no charge"}
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

type AgentMode = "autonomous" | "manual";

/** The sentence under the header: what this mode does on its own and what it always asks you first (C1). */
function modeLine(mode: AgentMode, cluster: boolean): string {
  if (mode === "manual") return "Manual: does one step per message, then waits for you. Training always waits for your OK.";
  return cluster
    ? "Autonomous: works through the steps on its own in the simulator, including training within your limits. It asks you first before going over your limits or moving a real robot."
    : "Autonomous: works through the steps on its own in the simulator. It asks you first before training on the shared GPU cluster, going over your limits, or moving a real robot.";
}

/** Autonomous (default) or Manual, saved per project; `cluster`: autonomous mode may also start cluster training. */
function useAgentMode(pid: string) {
  const state = useProjectState(pid);
  const qc = useQueryClient();
  const saved = state.data?.project.autopilot.autonomy ?? "autonomous";
  const m = useMutation({
    mutationFn: (autonomy: AgentMode) => api.setAutopilotPolicy(pid, { autonomy }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["state", pid] });
      qc.invalidateQueries({ queryKey: ["projects"] });
    },
  });
  const mode: AgentMode = m.isPending && m.variables ? m.variables : saved;
  return { mode, set: (v: AgentMode) => v !== mode && m.mutate(v), failed: !!m.error, cluster: state.data?.autonomy_cluster === true };
}

function ModeSwitch({ mode, onChange, failed }: { mode: AgentMode; onChange: (v: AgentMode) => void; failed: boolean }) {
  return (
    <div className="flex items-center gap-1.5" title={failed ? "Couldn't save the mode" : undefined}>
      <Segmented
        size="sm"
        value={mode}
        onChange={onChange}
        ariaLabel="Agent mode"
        options={[
          { value: "autonomous", label: "Autonomous" },
          { value: "manual", label: "Manual" },
        ]}
      />
    </div>
  );
}

/** Cortex acted for you under your standing approval, a hard gate needs you, or which agent answers and why. */
function NoticeRow({ it }: { it: Extract<ChatItem, { kind: "notice" }> }) {
  const needs = it.tone === "needs_you";
  const Icon = needs ? Hand : it.tone === "engine" ? Info : Bot;
  return (
    <div className={cn("enter flex items-start gap-2 text-xs leading-relaxed", needs ? "rounded border border-caution/40 bg-caution/5 px-3 py-2 text-ink-2" : "text-ink-3")}>
      <Icon className={cn("mt-0.5 h-3.5 w-3.5 shrink-0", needs ? "text-caution" : "text-ink-3")} strokeWidth={1.5} />
      <p className="min-w-0">
        <span className={cn("mr-1.5 font-mono text-[10px] uppercase tracking-[0.14em]", needs ? "text-caution" : "text-ink-3")}>{needs ? "Needs you" : it.tone === "engine" ? "Agent" : "Autonomous"}</span>
        {it.text}
      </p>
    </div>
  );
}

/** Shows or hides the past-chats panel beside the rail (the same one the rail's Chat button opens). */
function HistoryToggle() {
  const panel = useChatHistoryPanel();
  return (
    <button type="button" onClick={() => panel.setOpen(!panel.open)} aria-expanded={panel.open} className="group inline-flex items-center gap-1.5 py-1 text-xs text-ink-3 hover:text-ink">
      <History className="h-3.5 w-3.5" strokeWidth={1.5} /> <span className="border-b border-line-strong group-hover:border-ink">{panel.open ? "Hide chats" : "All chats"}</span>
    </button>
  );
}

/** Where each kind of change lives in the studio. */
const CHANGE_TARGET: Record<string, string> = {
  checkpoints: "improve?tab=training",
  training_jobs: "improve?tab=training",
  dataset_versions: "improve?tab=datasets",
  evaluation_runs: "improve?tab=evaluations",
  simulation_branches: "improve?tab=evaluations",
  agent_collections: "collect?mode=agent",
  recommendations: "improve?tab=recommendations",
};

/** "What changed" at the end of a turn, measured by Cortex: one pill per change. */
function ChangesRow({ it, pid }: { it: Extract<ChatItem, { kind: "changes" }>; pid: string }) {
  const viewer = useViewerFocus();
  const pill = "inline-flex h-6 items-center rounded-full bg-rail px-2.5 text-xs text-ink shadow-[0_0_0_1px_var(--line)]";
  return (
    <div className="enter flex flex-wrap items-center gap-1.5">
      <span className="mr-1 text-xs text-ink-3">What changed</span>
      {it.items.map((c, i) =>
        c.what === "episodes" && viewer ? (
          <button key={i} type="button" onClick={() => viewer.filter("demo")} className={cn(pill, "hover:text-servo")} title="Show demos in the viewer">
            {c.text}
          </button>
        ) : CHANGE_TARGET[c.what] ? (
          <Link key={i} to={`/p/${pid}/${CHANGE_TARGET[c.what]}`} className={cn(pill, "hover:text-servo")}>
            {c.text}
          </Link>
        ) : (
          <span key={i} className={pill}>
            {c.text}
          </span>
        ),
      )}
    </div>
  );
}

/** Plain chip labels for the project's next actions; the label is also the message sent. */
const NEXT_LABEL: Record<string, string> = {
  record_demonstrations: "Collect demos",
  review_episodes: "Review the demos",
  analyze_dataset: "Check demo quality",
  recommend_demonstrations: "Suggest what to collect next",
  create_dataset_version: "Build a training set",
  propose_training_job: "Set up training",
  approve_training_job: "Approve the training",
  launch_training_job: "Start training",
  get_job: "Check progress",
  launch_simulation_evaluation: "Test the policy",
  compare_experiments: "Compare the runs",
};

/** 2–3 one-click next steps from the project's next actions. */
function NextSteps({ pid, onPick }: { pid: string; onPick: (text: string) => void }) {
  const state = useProjectState(pid);
  const actions = (state.data?.next_actions ?? []).slice(0, 3);
  if (!actions.length) return null;
  return (
    <div className="enter flex flex-wrap gap-1.5">
      {actions.map((a) => {
        const label = NEXT_LABEL[a.action] ?? a.action.replace(/_/g, " ").replace(/^\w/, (c) => c.toUpperCase());
        return (
          <button
            key={a.action}
            type="button"
            onClick={() => onPick(label)}
            title={a.reason}
            className="inline-flex items-center gap-1.5 rounded-full bg-plate px-3 py-1.5 text-xs text-ink-2 shadow-[0_0_0_1px_var(--line)] hover:text-ink hover:shadow-[0_0_0_1px_var(--line-strong)]"
          >
            {label}
            {a.requires_approval && <span className="text-[10px] text-caution">needs approval</span>}
          </button>
        );
      })}
    </div>
  );
}
