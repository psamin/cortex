import { useCallback, useEffect, useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { ApiError, streamAgent } from "../api/client";
import { useAgentHistory } from "../api/hooks";
import type { AgentHistoryItem, ChatCard, StepInfo } from "../api/types";
import { errorMessage } from "../lib/format";

export type ChatItem =
  | { id: number; kind: "user"; text: string }
  | { id: number; kind: "turn"; step?: StepInfo }
  | { id: number; kind: "assistant"; text: string; streaming: boolean }
  | { id: number; kind: "tool"; tool: string; summary?: string; ref_ids: string[]; done: boolean }
  | { id: number; kind: "card"; card: ChatCard }
  | { id: number; kind: "approval"; approvalKind: string; ref_id: string; summary: string; args: Record<string, unknown> }
  | { id: number; kind: "error"; message: string };

export type Send = (text: string, patch?: Record<string, unknown>) => void;

let nextId = 1;
const id = () => nextId++;

/** Rebuild the thread from GET /agent/history (cards and approvals are kept there as role "event"). */
function fromHistory(rows: AgentHistoryItem[]): ChatItem[] {
  const out: ChatItem[] = [];
  let open = false;
  const turn = (step?: StepInfo) => {
    if (!open) out.push({ id: id(), kind: "turn", step });
    else if (step) {
      // the turn opened with a tool line before its step arrived: label that turn instead of starting another
      for (let i = out.length - 1; i >= 0; i--) {
        const x = out[i];
        if (x.kind === "turn") {
          out[i] = { ...x, step };
          break;
        }
      }
    }
    open = true;
  };
  for (const h of rows) {
    if (h.role === "user") {
      out.push({ id: id(), kind: "user", text: h.content ?? "" });
      open = false;
    } else if (h.role === "assistant") {
      turn();
      out.push({ id: id(), kind: "assistant", text: h.content ?? "", streaming: false });
    } else if (h.role === "event" && h.data) {
      const d = h.data;
      if (h.event === "step") turn(d as unknown as StepInfo);
      else if (h.event === "card") (turn(), out.push({ id: id(), kind: "card", card: d as unknown as ChatCard }));
      else if (h.event === "approval_required") {
        turn();
        out.push({
          id: id(),
          kind: "approval",
          approvalKind: String(d.kind),
          ref_id: String(d.ref_id),
          summary: String(d.summary ?? ""),
          args: (d.args as Record<string, unknown>) ?? {},
        });
      } else if (h.event === "tool_result") {
        turn();
        out.push({ id: id(), kind: "tool", tool: String(d.tool), summary: String(d.summary ?? ""), ref_ids: (d.ref_ids as string[]) ?? [], done: true });
      }
    }
  }
  return out;
}

export function useChat(pid: string) {
  const qc = useQueryClient();
  const history = useAgentHistory(pid);
  const [items, setItems] = useState<ChatItem[]>([]);
  const [busy, setBusy] = useState(false);
  const [step, setStep] = useState<StepInfo | undefined>();
  const abortRef = useRef<AbortController | null>(null);
  const hydrated = useRef(false);
  const busyRef = useRef(false);

  useEffect(() => {
    if (hydrated.current || !history.data) return;
    hydrated.current = true;
    const xs = fromHistory(history.data);
    setItems(xs);
    const last = [...xs].reverse().find((x): x is Extract<ChatItem, { kind: "turn" }> => x.kind === "turn" && !!x.step);
    if (last?.step) setStep(last.step);
  }, [history.data]);

  const send: Send = useCallback(
    async (message: string, patch?: Record<string, unknown>) => {
      const text = message.trim();
      if (!text || busyRef.current) return;
      busyRef.current = true;
      setBusy(true);
      const ctrl = new AbortController();
      abortRef.current = ctrl;
      setItems((xs) => [...xs, { id: id(), kind: "user", text }]);
      let open = false;
      let assistantId: number | null = null;
      const push = (it: ChatItem) => setItems((xs) => [...xs, it]);
      const ensureTurn = () => {
        if (!open) push({ id: id(), kind: "turn" });
        open = true;
      };
      const closeText = () => {
        if (assistantId === null) return;
        const aid = assistantId;
        setItems((xs) => xs.map((x) => (x.id === aid && x.kind === "assistant" ? { ...x, streaming: false } : x)));
        assistantId = null;
      };
      try {
        for await (const ev of streamAgent(pid, text, ctrl.signal, patch)) {
          if (ev.type === "step") {
            setStep(ev.step);
            if (!open) push({ id: id(), kind: "turn", step: ev.step });
            else
              setItems((xs) => {
                const i = xs.map((x) => x.kind === "turn").lastIndexOf(true);
                return xs.map((x, j) => (j === i && x.kind === "turn" ? { ...x, step: ev.step } : x));
              });
            open = true;
            continue;
          }
          if (ev.type === "done") break;
          ensureTurn();
          if (ev.type === "text") {
            if (assistantId === null) {
              const aid = id();
              assistantId = aid;
              push({ id: aid, kind: "assistant", text: ev.delta, streaming: true });
            } else {
              const aid = assistantId;
              setItems((xs) => xs.map((x) => (x.id === aid && x.kind === "assistant" ? { ...x, text: `${x.text} ${ev.delta}` } : x)));
            }
            continue;
          }
          closeText();
          if (ev.type === "tool_call") push({ id: id(), kind: "tool", tool: ev.tool, ref_ids: [], done: false });
          else if (ev.type === "tool_result") {
            setItems((xs) => {
              const i = xs.map((x) => x.kind === "tool" && x.tool === ev.tool && !x.done).lastIndexOf(true);
              const done = { kind: "tool" as const, tool: ev.tool, summary: ev.summary, ref_ids: ev.ref_ids, done: true };
              if (i === -1) return [...xs, { id: id(), ...done }];
              return xs.map((x, j) => (j === i ? { ...x, ...done } : x));
            });
            qc.invalidateQueries({ predicate: (q) => q.queryKey.includes(pid) });
          } else if (ev.type === "card") push({ id: id(), kind: "card", card: ev.card });
          else if (ev.type === "approval_required")
            push({ id: id(), kind: "approval", approvalKind: ev.kind, ref_id: ev.ref_id, summary: ev.summary, args: ev.args });
          else if (ev.type === "error") push({ id: id(), kind: "error", message: ev.message });
        }
      } catch (e) {
        const msg = e instanceof ApiError && e.unreachable ? "The Cortex API is not responding." : errorMessage(e);
        push({ id: id(), kind: "error", message: msg });
      } finally {
        setItems((xs) => xs.map((x) => (x.kind === "assistant" && x.streaming ? { ...x, streaming: false } : x)));
        busyRef.current = false;
        setBusy(false);
        abortRef.current = null;
      }
    },
    [pid, qc],
  );

  const stop = () => abortRef.current?.abort();
  return { items, busy, step, send, stop, history };
}
