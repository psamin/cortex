import { createContext, useCallback, useContext, useEffect, useRef, useState, type ReactNode } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { api, ApiError, streamAgent } from "../api/client";
import { useChatStream } from "../api/events";
import { splitError } from "../components/ui";
import { applyEvent, historyToEvents, jobDoneEvent, newItemId, type ChangeItem, type ChatEvent, type ChatItem, type TodoItem } from "./model";

interface ChatState {
  items: ChatItem[];
  loading: boolean;
  loadError: unknown;
  busy: boolean;
  /** latest `status` event while a turn runs */
  status: { state: string; tool?: string } | null;
  turnStartedAt: number | null;
  /** bumps whenever a tool result lands, so the preview can follow the agent */
  resultTick: number;
  send: (text: string) => void;
  stop: () => void;
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
      return { type: "status", state: String(d.state ?? "thinking"), tool: d.tool as string | undefined };
    case "error":
      return { type: "error", message: String(d.message ?? d.detail ?? "Agent error") };
    case "todos":
      return Array.isArray(d.items) ? { type: "todos", items: d.items as TodoItem[] } : null;
    case "changes":
      return Array.isArray(d.items) ? { type: "changes", items: d.items as ChangeItem[] } : null;
    default:
      return null;
  }
}

export function ChatProvider({ pid, children }: { pid: string; children: ReactNode }) {
  const qc = useQueryClient();
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

  const load = useCallback(() => {
    setLoading(true);
    setLoadError(null);
    api
      .agentHistory(pid, threadRef.current)
      .then((rows) => setItems(historyToEvents(rows).reduce(applyEvent, [] as ChatItem[])))
      .catch((e) => {
        // a project with no chat yet may 404 on older backends; treat as empty
        if (e instanceof ApiError && e.status === 404) {
          setItems([]);
          if (threadRef.current) setThreadId(null); // a remembered thread that no longer exists
        }
        else setLoadError(e);
      })
      .finally(() => setLoading(false));
    // reads threadRef so a `thread` event mid-stream never triggers a reload
  }, [pid, setThreadId]);

  useEffect(() => {
    load();
    return () => abort.current?.abort();
  }, [load]);

  // Cortex-written lines (job finished, …) arrive on the project event stream, outside agent turns
  useChatStream((entry) => {
    if (entry.kind !== "job_done") return;
    setItems((xs) => applyEvent(xs, jobDoneEvent(entry)));
    qc.invalidateQueries({ predicate: (q) => q.queryKey.includes(pid) });
  });

  const run = useCallback(
    async (text: string) => {
      busyRef.current = true;
      setBusy(true);
      setStatus({ state: "thinking" });
      setTurnStartedAt(Date.now());
      setItems((xs) => applyEvent(xs, { type: "user", text }));
      const ctrl = new AbortController();
      abort.current = ctrl;
      try {
        for await (const { event, data } of streamAgent(pid, text, ctrl.signal, threadRef.current)) {
          if (event === "thread" && typeof data.id === "string") {
            // the backend says which thread this turn belongs to (a new one on the first message)
            threadRef.current = data.id;
            setThreadId(data.id);
            continue;
          }
          const ev = toEvent(event, data);
          if (!ev) continue;
          if (ev.type === "status") {
            setStatus({ state: ev.state, tool: ev.tool });
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
          const msg = e instanceof ApiError && e.unreachable ? "The Cortex backend is not responding." : splitError(e, undefined, "The agent request failed.").text;
          setItems((xs) => applyEvent(xs, { type: "error", message: msg }));
        }
      } finally {
        setItems((xs) => xs.map((it) => (it.kind === "tool" && it.status === "running" ? { ...it, status: "ok" } : it)));
        abort.current = null;
        busyRef.current = false;
        setBusy(false);
        setStatus(null);
        setTurnStartedAt(null);
        qc.invalidateQueries({ queryKey: ["agent-threads", pid] });
        const next = queue.current.shift();
        if (next) void run(next);
      }
    },
    [pid, qc],
  );

  const send = useCallback(
    (text: string) => {
      const t = text.trim();
      if (!t) return;
      if (busyRef.current) queue.current.push(t);
      else void run(t);
    },
    [run],
  );

  const stop = useCallback(() => {
    queue.current = [];
    abort.current?.abort();
  }, []);

  const newChat = useCallback(async () => {
    stop();
    const t = await api.newThread(pid);
    threadRef.current = t.id;
    setThreadId(t.id);
    setItems([]);
    qc.invalidateQueries({ queryKey: ["agent-threads", pid] });
  }, [pid, stop, setThreadId, qc]);
  const reset = newChat;

  const openThread = useCallback(
    (id: string) => {
      if (id === threadRef.current) return;
      stop();
      threadRef.current = id;
      setThreadId(id);
      load();
    },
    [stop, setThreadId, load],
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
    <Ctx.Provider value={{ items, loading, loadError, busy, status, turnStartedAt, resultTick, send, stop, reset, reload: load, addImport, patchImport, threadId, newChat, openThread }}>
      {children}
    </Ctx.Provider>
  );
}
