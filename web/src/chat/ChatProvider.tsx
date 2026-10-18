import { createContext, useCallback, useContext, useEffect, useRef, useState, type ReactNode } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { api, ApiError, streamAgent } from "../api/client";
import { useChatStream } from "../api/events";
import { splitError } from "../components/ui";
import { applyEvent, historyToEvents, jobDoneEvent, newItemId, type ChatEvent, type ChatItem } from "./model";

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

  const load = useCallback(() => {
    setLoading(true);
    setLoadError(null);
    api
      .agentHistory(pid)
      .then((rows) => setItems(historyToEvents(rows).reduce(applyEvent, [] as ChatItem[])))
      .catch((e) => {
        // a project with no chat yet may 404 on older backends; treat as empty
        if (e instanceof ApiError && e.status === 404) setItems([]);
        else setLoadError(e);
      })
      .finally(() => setLoading(false));
  }, [pid]);

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
        for await (const { event, data } of streamAgent(pid, text, ctrl.signal)) {
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

  const reset = useCallback(async () => {
    stop();
    await api.resetAgent(pid);
    setItems([]);
  }, [pid, stop]);

  const addImport = useCallback((item: Omit<ImportItem, "id" | "kind">) => {
    const id = newItemId();
    setItems((xs) => [...xs, { ...item, kind: "import", id }]);
    return id;
  }, []);
  const patchImport = useCallback((id: number, patch: Partial<ImportItem>) => {
    setItems((xs) => xs.map((it) => (it.kind === "import" && it.id === id ? { ...it, ...patch, kind: "import", id } : it)));
  }, []);

  return (
    <Ctx.Provider value={{ items, loading, loadError, busy, status, turnStartedAt, resultTick, send, stop, reset, reload: load, addImport, patchImport }}>
      {children}
    </Ctx.Provider>
  );
}
