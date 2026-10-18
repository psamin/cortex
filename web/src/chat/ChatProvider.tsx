import { createContext, useCallback, useContext, useEffect, useRef, useState, type ReactNode } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { api, ApiError, streamAgent } from "../api/client";
import type { AgentState } from "../api/types";
import { useChatStream } from "../api/events";
import { splitError } from "../components/ui";
import { applyEvent, historyToEvents, jobDoneEvent, newItemId, noticeEvent, type ChangeItem, type ChatEvent, type ChatItem, type TodoItem } from "./model";

interface ChatState {
  items: ChatItem[];
  loading: boolean;
  loadError: unknown;
  /** a turn is running: the user's, or an automatic one (autonomous mode) */
  busy: boolean;
  /** an automatic turn (autonomous mode) is streaming over the project event stream */
  auto: boolean;
  /** latest `status` event while a turn runs; `text` is the live line ("Collecting 20 demos in sim…") */
  status: { state: string; tool?: string; text?: string } | null;
  turnStartedAt: number | null;
  /** bumps whenever a tool result lands, so the preview can follow the agent */
  resultTick: number;
  send: (text: string) => void;
  stop: () => void;
  /** the automatic steps stopped after repeated cut-offs (or a gate): the studio shows a Continue button */
  paused: AgentState["paused"];
  continueRun: () => void;
  reset: () => Promise<void>;
  /** the open chat thread (null = the backend's latest) */
  threadId: string | null;
  /** start an empty thread; past chats stay in History */
  newChat: () => Promise<void>;
  openThread: (id: string) => void;
  reload: () => void;
  /** local rows (e.g. a demo upload) that live only in this tab until the job's job_done line lands */
  addImport: (item: Omit<ImportItem, "id" | "kind">) => number;
  patchImport: (id: number, patch: Partial<ImportItem>) => void;
}

type ImportItem = Extract<ChatItem, { kind: "import" }>;

const Ctx = createContext<ChatState | null>(null);

export function useChat(): ChatState {
  const c = useContext(Ctx);
  if (!c) throw new Error("useChat must be used inside ChatProvider");
  return c;
}

function toEvent(event: string, d: Record<string, unknown>): ChatEvent | null {
  switch (event) {
    case "text":
      return { type: "text", delta: String(d.delta ?? "") };
    case "tool_call":
      return { type: "tool_call", tool: String(d.tool ?? ""), args: (d.args as Record<string, unknown>) ?? {} };
    case "tool_result":
      return {
        type: "tool_result",
        tool: String(d.tool ?? ""),
        summary: typeof d.summary === "string" ? d.summary : "",
        ref_ids: Array.isArray(d.ref_ids) ? (d.ref_ids as string[]) : [],
        ok: d.ok as boolean | undefined,
      };
    case "approval_required":
      return {
        type: "approval_required",
        kind: String(d.card_kind ?? d.kind ?? "training"),
        ref_id: String(d.training_job_id ?? d.ref_id ?? ""),
        summary: String(d.summary ?? ""),
        args: (d.args as Record<string, unknown>) ?? {},
      };
    case "done":
      return { type: "done", turns: d.turns as number, cost_usd: d.cost_usd as number, duration_ms: d.duration_ms as number, is_error: d.is_error as boolean };
    case "status":
      return { type: "status", state: String(d.state ?? "thinking"), tool: d.tool as string | undefined, text: typeof d.text === "string" ? d.text : undefined };
    case "error":
      console.warn("[cortex] agent error", d); // not shown in the chat (#328)
      return { type: "error", message: String(d.message ?? d.detail ?? "Agent error") };
    case "todos":
      return Array.isArray(d.items) ? { type: "todos", items: d.items as TodoItem[] } : null;
    case "changes":
      return Array.isArray(d.items) ? { type: "changes", items: d.items as ChangeItem[] } : null;
    case "notice":
      return noticeEvent(d);
    default:
      return null;
  }
}

export function ChatProvider({ pid, children }: { pid: string; children: ReactNode }) {
  const qc = useQueryClient();
  const nav = useNavigate();
  const [items, setItems] = useState<ChatItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<unknown>(null);
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState<ChatState["status"]>(null);
  const [turnStartedAt, setTurnStartedAt] = useState<number | null>(null);
  const [resultTick, setResultTick] = useState(0);
  const abort = useRef<AbortController | null>(null);
  const queue = useRef<string[]>([]);
  const busyRef = useRef(false);
  const [auto, setAuto] = useState(false);
  const autoRef = useRef(false);
  const [paused, setPaused] = useState<AgentState["paused"]>(null);

  // the open thread survives reloads, per project
  const storeKey = `cortex.thread.${pid}`;
  const [threadId, setThreadIdState] = useState<string | null>(() => {
    try {
      return localStorage.getItem(storeKey);
    } catch {
      return null;
    }
  });
  const setThreadId = useCallback(
    (id: string | null) => {
      setThreadIdState(id);
      try {
        if (id) localStorage.setItem(storeKey, id);
        else localStorage.removeItem(storeKey);
      } catch {
        /* storage unavailable */
      }
    },
    [storeKey],
  );
  const threadRef = useRef(threadId);
  threadRef.current = threadId;

  // only the latest history load may set the items: an older one landing late (StrictMode mounts twice, or a turn
  // started meanwhile) would wipe the message the user just sent
  const loadSeq = useRef(0);
  const load = useCallback(() => {
    const seq = ++loadSeq.current;
    const latest = () => seq === loadSeq.current;
    setLoading(true);
    setLoadError(null);
    api
      .agentHistory(pid, threadRef.current)
      .then((rows) => latest() && setItems(historyToEvents(rows).reduce(applyEvent, [] as ChatItem[])))
      .catch((e) => {
        if (!latest()) return;
        // a project with no chat yet may 404 on older backends; treat as empty
        if (e instanceof ApiError && e.status === 404) {
          setItems([]);
          if (threadRef.current) setThreadId(null); // a remembered thread that no longer exists
        }
        else {
          console.warn("[cortex] chat history", e);
          setLoadError(e);
        }
      })
      .finally(() => latest() && setLoading(false));
    // reads threadRef so a `thread` event mid-stream never triggers a reload
  }, [pid, setThreadId]);

  useEffect(() => {
    load();
    return () => abort.current?.abort();
  }, [load]);

  // a history load that failed tries again by itself; the chat only says it's retrying (#328)
  useEffect(() => {
    if (!loadError) return;
    const t = setTimeout(load, 5000);
    return () => clearTimeout(t);
  }, [loadError, load]);

  const runRef = useRef<(text: string) => Promise<void>>(async () => {});

  // The backend says whether a turn is really running. A live line for a turn that is gone (a restart, a crashed
  // agent) ends here instead of spinning forever, and a turn running without this tab (after a reload) shows live.
  const misses = useRef(0);
  const sync = useCallback(async () => {
    let s: AgentState;
    try {
      s = await api.agentState(pid);
    } catch {
      if (busyRef.current || autoRef.current) setStatus((prev) => ({ state: prev?.state ?? "thinking", text: "Reconnecting to Cortex…" }));
      return;
    }
    setPaused(s.paused);
    if (s.running || s.resuming) {
      misses.current = 0;
      const here = !s.thread || !threadRef.current || s.thread === threadRef.current;
      if (here && !busyRef.current && !autoRef.current) {
        autoRef.current = true;
        setAuto(true);
        setTurnStartedAt((t) => t ?? Date.now());
        setStatus({ state: "thinking", text: s.status ?? (s.resuming ? "Continuing on my own…" : "Working…") });
      }
      return;
    }
    if (!busyRef.current && !autoRef.current) return;
    if (++misses.current < 2) return; // one miss can be a turn starting this very moment
    misses.current = 0;
    abort.current?.abort();
    autoRef.current = false;
    setAuto(false);
    setStatus(null);
    setTurnStartedAt(null);
    load();
  }, [pid, load]);

  useEffect(() => {
    void sync();
  }, [sync]);

  useEffect(() => {
    if (!busy && !auto) return;
    const t = setInterval(() => void sync(), 8000);
    return () => clearInterval(t);
  }, [busy, auto, sync]);

  /** One event of an automatic turn (autonomous mode), relayed on the project event stream. */
  const onLive = useCallback(
    (event: string, data: Record<string, unknown>) => {
      if (event === "thread") return;
      if (!autoRef.current && event !== "done") {
        autoRef.current = true;
        setAuto(true);
        setTurnStartedAt((t) => t ?? Date.now());
        setStatus({ state: "starting", text: "Continuing on my own…" });
      }
      const ev = toEvent(event, data);
      if (event === "done") {
        if (ev && autoRef.current) setItems((xs) => applyEvent(xs, ev));
        autoRef.current = false;
        setAuto(false);
        if (!busyRef.current) {
          setStatus(null);
          setTurnStartedAt(null);
        }
        qc.invalidateQueries({ queryKey: ["agent-threads", pid] });
        const next = queue.current.shift();
        if (next && !busyRef.current) void runRef.current(next);
        return;
      }
      if (!ev) return;
      if (ev.type === "status") {
        setStatus((prev) => ({ state: ev.state, tool: ev.tool, text: ev.text ?? prev?.text }));
        return;
      }
      setItems((xs) => applyEvent(xs, ev));
      if (ev.type === "tool_result") {
        qc.invalidateQueries({ predicate: (q) => q.queryKey.includes(pid) });
        setResultTick((t) => t + 1);
      }
    },
    [pid, qc],
  );

  // Cortex-written lines (job finished, notices, automatic turns) arrive on the project event stream
  useChatStream((entry) => {
    if (entry.kind === "reconnected") {
      // back after a drop (e.g. the backend restarted): what happened meanwhile is in the history
      if (!busyRef.current) load();
      void sync();
      return;
    }
    const here = typeof entry.thread !== "string" || !threadRef.current || entry.thread === threadRef.current;
    if (entry.kind === "paused") {
      if (here) setPaused({ thread: String(entry.thread ?? ""), reason: String(entry.reason ?? ""), at: Date.now() / 1000 });
      return;
    }
    if (entry.kind === "live") {
      if (here) onLive(String(entry.event ?? ""), (entry.data as Record<string, unknown>) ?? {});
      return;
    }
    if (entry.kind === "notice") {
      if (here) setItems((xs) => applyEvent(xs, noticeEvent(entry)));
      return;
    }
    if (entry.kind !== "job_done") return;
    setItems((xs) => applyEvent(xs, jobDoneEvent(entry)));
    qc.invalidateQueries({ predicate: (q) => q.queryKey.includes(pid) });
  });

  const run = useCallback(
    async (text: string) => {
      loadSeq.current++; // this turn's rows win over a history load still in flight
      setLoading(false);
      setPaused(null); // a message picks the workflow up again
      busyRef.current = true;
      setBusy(true);
      setStatus({ state: "starting", text: "Starting…" });
      setTurnStartedAt(Date.now());
      setItems((xs) => applyEvent(xs, { type: "user", text }));
      const ctrl = new AbortController();
      abort.current = ctrl;
      let moveTo: { id: string; prompt: string } | null = null;
      try {
        for await (const { event, data } of streamAgent(pid, text, ctrl.signal, threadRef.current)) {
          if (event === "thread" && typeof data.id === "string") {
            // the backend says which thread this turn belongs to (a new one on the first message)
            threadRef.current = data.id;
            setThreadId(data.id);
            continue;
          }
          if (event === "project" && typeof data.id === "string") {
            // a different task than this project's: Cortex made it its own project; open it and ask there
            moveTo = { id: data.id, prompt: typeof data.prompt === "string" ? data.prompt : text };
            continue;
          }
          const ev = toEvent(event, data);
          if (!ev) continue;
          if (ev.type === "status") {
            // a status without a line keeps the last one, so the live line never goes blank
            setStatus((prev) => ({ state: ev.state, tool: ev.tool, text: ev.text ?? prev?.text }));
            continue;
          }
          setItems((xs) => applyEvent(xs, ev));
          if (ev.type === "tool_result") {
            // the agent changed something: refresh project data and let the preview follow
            qc.invalidateQueries({ predicate: (q) => q.queryKey.includes(pid) });
            setResultTick((t) => t + 1);
          }
        }
      } catch (e) {
        if ((e as Error).name !== "AbortError") {
          console.warn("[cortex] agent turn", e); // the chat leaves it out (#328)
          const msg = e instanceof ApiError && e.unreachable ? "The Cortex backend is not responding." : splitError(e, undefined, "The agent request failed.").text;
          // the error item stays out of the chat; the turn still ends with the plain "Stopped" line
          setItems((xs) => applyEvent(applyEvent(xs, { type: "error", message: msg }), { type: "done", is_error: true }));
        }
      } finally {
        setItems((xs) => xs.map((it) => (it.kind === "tool" && it.status === "running" ? { ...it, status: "ok" } : it)));
        abort.current = null;
        busyRef.current = false;
        setBusy(false);
        setStatus(null);
        setTurnStartedAt(null);
        qc.invalidateQueries({ queryKey: ["agent-threads", pid] });
        if (moveTo) queue.current = []; // messages typed meanwhile were for this project, not the new one
        const next = autoRef.current ? undefined : queue.current.shift(); // during an automatic turn it waits
        if (next) void run(next);
      }
      if (moveTo) {
        qc.invalidateQueries({ queryKey: ["projects"] });
        nav(`/p/${moveTo.id}`, { state: { autoSend: moveTo.prompt } });
      }
    },
    [pid, qc, nav, setThreadId],
  );
  runRef.current = run;

  const send = useCallback(
    (text: string) => {
      const t = text.trim();
      if (!t) return;
      if (busyRef.current || autoRef.current) queue.current.push(t);
      else void run(t);
    },
    [run],
  );

  /** End this tab's own turn (switching chats). */
  const abortOwn = useCallback(() => {
    queue.current = [];
    abort.current?.abort();
  }, []);

  /** The stop button: end this turn and the autonomous chain (no more automatic steps). */
  const stop = useCallback(() => {
    abortOwn();
    setPaused(null);
    void api.stopAgent(pid).catch(() => {});
  }, [abortOwn, pid]);

  /** The Continue button after a pause: the agent picks the workflow up where it stopped. */
  const continueRun = useCallback(() => {
    setPaused(null);
    autoRef.current = true;
    setAuto(true);
    setTurnStartedAt(Date.now());
    setStatus({ state: "starting", text: "Picking up where it stopped…" });
    void api
      .continueAgent(pid)
      .catch(() => {})
      .finally(() => void sync());
  }, [pid, sync]);

  const newChat = useCallback(async () => {
    abortOwn();
    const t = await api.newThread(pid);
    threadRef.current = t.id;
    setThreadId(t.id);
    setItems([]);
    qc.invalidateQueries({ queryKey: ["agent-threads", pid] });
  }, [pid, abortOwn, setThreadId, qc]);
  const reset = newChat;

  const openThread = useCallback(
    (id: string) => {
      if (id === threadRef.current) return;
      abortOwn();
      threadRef.current = id;
      setThreadId(id);
      load();
    },
    [abortOwn, setThreadId, load],
  );

  const addImport = useCallback((item: Omit<ImportItem, "id" | "kind">) => {
    const id = newItemId();
    setItems((xs) => [...xs, { ...item, kind: "import", id }]);
    return id;
  }, []);
  const patchImport = useCallback((id: number, patch: Partial<ImportItem>) => {
    setItems((xs) => xs.map((it) => (it.kind === "import" && it.id === id ? { ...it, ...patch, kind: "import", id } : it)));
  }, []);

  return (
    <Ctx.Provider value={{ items, loading, loadError, busy: busy || auto, auto, status, turnStartedAt, resultTick, send, stop, paused, continueRun, reset, reload: load, addImport, patchImport, threadId, newChat, openThread }}>
      {children}
    </Ctx.Provider>
  );
}
