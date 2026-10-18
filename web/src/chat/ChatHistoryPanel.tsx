import { createContext, useContext } from "react";
import { useNavigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { PanelLeftClose, SquarePen } from "lucide-react";
import { api } from "../api/client";
import type { AgentThread } from "../api/types";
import { InlineError } from "../components/ui";
import { cn } from "../lib/format";
import { useChat } from "./ChatProvider";
import { groupThreads } from "./threadGroups";

/** Whether the past-chats panel next to the project rail is open (the rail's Chat button toggles it). */
export const ChatHistoryPanelCtx = createContext<{ open: boolean; setOpen: (open: boolean) => void }>({ open: false, setOpen: () => {} });
export const useChatHistoryPanel = () => useContext(ChatHistoryPanelCtx);

/**
 * Every chat of this project, like ChatGPT's sidebar, then past chats by recency.
 * "New task" opens the studio prompt, which creates a new project: a different thing to train gets its own
 * demos, datasets and policies instead of mixing into (and training on) this project's. "New chat on this
 * task" keeps the project and starts a fresh conversation about it.
 */
export function ChatHistoryPanel({ pid, onClose }: { pid: string; onClose: () => void }) {
  const chat = useChat();
  const nav = useNavigate();
  const threads = useQuery({ queryKey: ["agent-threads", pid], queryFn: () => api.agentThreads(pid), retry: 0 });
  const list: AgentThread[] = threads.data ?? [];
  const current = chat.threadId ?? list[0]?.id ?? null;
  const toChat = () => nav(`/p/${pid}`);

  return (
    <aside aria-label="Chats" className="enter flex w-[260px] shrink-0 flex-col border-r border-line bg-bench">
      <div className="flex h-11 shrink-0 items-center justify-between border-b border-line pl-4 pr-2">
        <span className="kn-label text-ink">Chats</span>
        <button type="button" onClick={onClose} aria-label="Close chats" title="Close" className="flex h-8 w-8 items-center justify-center rounded text-ink-3 hover:bg-rail hover:text-ink">
          <PanelLeftClose className="h-4 w-4" strokeWidth={1.5} />
        </button>
      </div>
      <div className="shrink-0 px-2 pt-2">
        <button
          type="button"
          onClick={() => nav("/studio")}
          title="Train something new: a new project with its own demos, datasets and policies"
          className="flex w-full items-center gap-2.5 rounded px-2.5 py-2 text-left text-sm text-ink hover:bg-rail"
        >
          <SquarePen className="h-4 w-4 shrink-0" strokeWidth={1.5} /> New task
        </button>
        <button
          type="button"
          onClick={() => {
            // an empty chat is already a new chat: just go to it
            if (chat.items.length > 0) void chat.newChat().catch(() => {});
            toChat();
          }}
          title="A fresh conversation about this project's task, with the same demos and policies"
          className="flex w-full items-center gap-2.5 rounded px-2.5 py-1.5 text-left text-xs text-ink-3 hover:bg-rail hover:text-ink"
        >
          <span className="w-4 shrink-0" /> New chat on this task
        </button>
      </div>
      <nav aria-label="Past chats" className="min-h-0 flex-1 overflow-y-auto px-2 pb-4 scroll-thin">
        {threads.isLoading ? (
          <div className="space-y-1.5 px-1 pt-3">
            {[0, 1, 2, 3].map((i) => (
              <div key={i} className="skeleton h-7 w-full rounded-sm" />
            ))}
          </div>
        ) : threads.error ? (
          <InlineError error={threads.error} fallback="Couldn't load past chats." className="px-2.5 pt-3" />
        ) : list.length === 0 ? (
          <p className="px-2.5 pt-3 text-xs text-ink-3">No chats yet.</p>
        ) : (
          groupThreads(list).map(([label, xs]) => (
            <section key={label} className="pt-4">
              <h3 className="px-2.5 pb-1 text-[11px] font-medium text-ink-3">{label}</h3>
              <ul>
                {xs.map((t) => {
                  const active = t.id === current;
                  return (
                    <li key={t.id}>
                      <button
                        type="button"
                        onClick={() => {
                          chat.openThread(t.id);
                          toChat();
                        }}
                        aria-current={active ? "page" : undefined}
                        title={t.title}
                        className={cn(
                          "flex w-full items-center gap-2 rounded px-2.5 py-1.5 text-left text-sm",
                          active ? "bg-rail text-ink" : "text-ink-2 hover:bg-rail/60 hover:text-ink",
                        )}
                      >
                        <span className="min-w-0 flex-1 truncate">{t.title || "New chat"}</span>
                        {active && chat.busy && <span className="h-1.5 w-1.5 shrink-0 rounded-full bg-servo live-dot" title="The agent is working" />}
                      </button>
                    </li>
                  );
                })}
              </ul>
            </section>
          ))
        )}
      </nav>
    </aside>
  );
}
