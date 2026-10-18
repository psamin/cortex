import { useEffect, useRef, useState } from "react";
import { ArrowUp, Square } from "lucide-react";
import type { StepInfo } from "../api/types";
import { RefChip } from "../agent/RefChip";
import { Button, SkeletonRows } from "../components/ui";
import { AGENT_NAME } from "../lib/brand";
import { cn } from "../lib/format";
import { ApprovalCard, ChatCardView, Label } from "./cards";
import { useChat, type ChatItem, type Send } from "./useChat";

export const PIPELINE = [
  { step: "intake", label: "Describe" },
  { step: "simulate", label: "Simulate" },
  { step: "collect", label: "Collect" },
  { step: "train", label: "Train" },
  { step: "evaluate", label: "Evaluate" },
];
const nn = (i: number) => String(i).padStart(2, "0");

/** "01 DESCRIBE   02 SIMULATE   03 COLLECT …" over a hairline rule. */
export function StepRail({ step, compact }: { step?: StepInfo; compact?: boolean }) {
  const cur = step ? PIPELINE.findIndex((p) => p.step === step.step) : -1;
  return (
    <ol className={cn("flex flex-wrap gap-x-6 gap-y-2 border-b border-line pb-3", compact && "gap-x-3.5")} aria-label="Pipeline progress">
      {PIPELINE.map((p, i) => (
        <li
          key={p.step}
          aria-current={i === cur ? "step" : undefined}
          className={cn(
            "label inline-flex items-center gap-2 transition-colors duration-300",
            i === cur ? "!text-ink" : i < cur ? "!text-ink-2" : "!text-ink-4",
            compact && "!text-[10px] !tracking-[0.12em]",
          )}
        >
          <span className={cn("h-1.5 w-1.5 rounded-full", i === cur ? "bg-servo live-dot" : i < cur ? "bg-ink-3" : "border border-ink-4")} />
          {nn(i + 1)} {p.label}
        </li>
      ))}
    </ol>
  );
}

export function ChatThread({
  pid,
  variant = "page",
  initial,
  registerAsk,
}: {
  pid: string;
  variant?: "page" | "panel";
  /** Sent once when the thread opens with no history (the prompt typed on the home page). */
  initial?: string;
  registerAsk?: (fn: (m: string) => void) => void;
}) {
  const chat = useChat(pid);
  const { items, busy, step, send, stop, history } = chat;
  const scrollRef = useRef<HTMLDivElement>(null);
  const sentInitial = useRef(false);
  const page = variant === "page";

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [items]);

  useEffect(() => {
    if (!initial || sentInitial.current || !history.data) return;
    sentInitial.current = true;
    if (history.data.length === 0) send(initial);
  }, [initial, history.data, send]);

  useEffect(() => registerAsk?.((m) => send(m)), [registerAsk, send]);

  const lastUser = items.map((x) => x.kind === "user").lastIndexOf(true);

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className={cn("shrink-0", page ? "px-5 pt-6 sm:px-10" : "px-3 pt-3")}>
        <div className={cn(page && "mx-auto max-w-[760px]")}>
          <StepRail step={step} compact={!page} />
        </div>
      </div>
      <div ref={scrollRef} className="min-h-0 flex-1 overflow-y-auto scroll-thin">
        <div className={cn("space-y-4", page ? "mx-auto max-w-[760px] px-5 pb-10 pt-8 sm:px-10" : "px-3 pb-4 pt-4")}>
          {history.isLoading && <SkeletonRows rows={3} />}
          {!history.isLoading && items.length === 0 && !initial && (
            <div className="space-y-4 pt-4">
              <h2 className={cn("display text-ink", page ? "text-[40px]" : "text-[26px]")}>
                Pick up where you left off.
                <br />
                <span className="text-muted">Cortex shows the next step.</span>
              </h2>
              <Button variant="secondary" onClick={() => send("Where are we?")}>
                Show the next step <span aria-hidden>→</span>
              </Button>
            </div>
          )}
          {items.map((it, i) => (
            <Item key={it.id} it={it} pid={pid} send={send} live={i > lastUser && !busy} page={page} />
          ))}
          {busy && (
            <Label className="flex items-center gap-2">
              <span className="h-1.5 w-1.5 rounded-full bg-servo live-dot" /> Working
            </Label>
          )}
        </div>
      </div>
      <Composer send={send} busy={busy} stop={stop} page={page} />
    </div>
  );
}

function Item({ it, pid, send, live, page }: { it: ChatItem; pid: string; send: Send; live: boolean; page: boolean }) {
  switch (it.kind) {
    case "user":
      return <p className={cn("enter whitespace-pre-wrap pt-6 text-ink", page ? "text-[19px] leading-snug" : "text-[15px]")}>{it.text}</p>;
    case "turn":
      return (
        <Label rule={page} className="enter pt-1">
          {AGENT_NAME}
          {it.step ? ` · Step ${nn(it.step.index)} ${it.step.label}` : ""}
        </Label>
      );
    case "assistant":
      return (
        <p className={cn("enter whitespace-pre-wrap leading-relaxed text-ink-2", page ? "text-[15px]" : "text-sm")}>
          {it.text}
          {it.streaming && <span className="ml-1 inline-block h-3.5 w-1 translate-y-0.5 bg-servo live-dot" />}
        </p>
      );
    case "tool":
      return (
        <div className="enter flex flex-wrap items-baseline gap-x-2 gap-y-1 font-mono text-[11.5px] text-ink-3">
          <span className={cn("uppercase tracking-[0.12em]", !it.done && "live-dot")}>→ {it.tool}</span>
          {it.summary && <span className="min-w-0 break-words normal-case text-ink-3">{it.summary}</span>}
          {it.ref_ids.slice(0, 3).map((r) => (
            <RefChip key={r} pid={pid} id={r} />
          ))}
        </div>
      );
    case "card":
      return <ChatCardView card={it.card} send={send} live={live} pid={pid} />;
    case "approval":
      return <ApprovalCard kind={it.approvalKind} refId={it.ref_id} summary={it.summary} args={it.args} pid={pid} send={send} live={live} />;
    case "error":
      return <p className="enter border-l border-fault pl-3 text-sm text-fault">{it.message}</p>;
  }
}

function Composer({ send, busy, stop, page }: { send: Send; busy: boolean; stop: () => void; page: boolean }) {
  const [draft, setDraft] = useState("");
  const submit = () => {
    if (!draft.trim() || busy) return;
    send(draft);
    setDraft("");
  };
  return (
    <div className={cn("shrink-0 border-t border-line bg-bench", page ? "px-5 py-4 sm:px-10" : "p-2.5")}>
      <form
        onSubmit={(e) => {
          e.preventDefault();
          submit();
        }}
        className={cn(
          "flex items-end gap-2 rounded-md border border-line bg-bench p-2 transition-colors focus-within:border-ink-3",
          page && "mx-auto max-w-[760px]",
        )}
      >
        <textarea
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              submit();
            }
          }}
          rows={page ? 2 : 2}
          placeholder="Answer, or tell Cortex what to do next…"
          className="min-h-[44px] flex-1 resize-none bg-transparent px-1.5 py-1 text-[15px] text-ink placeholder:text-ink-4 focus:outline-none"
          aria-label="Message Cortex"
        />
        {busy ? (
          <Button size="sm" variant="secondary" onClick={stop} aria-label="Stop">
            <Square className="h-3 w-3" />
          </Button>
        ) : (
          <Button size="sm" variant="primary" type="submit" disabled={!draft.trim()} aria-label="Send">
            <ArrowUp className="h-3.5 w-3.5" />
          </Button>
        )}
      </form>
    </div>
  );
}
