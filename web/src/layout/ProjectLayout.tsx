import { createContext, useCallback, useContext } from "react";
import { NavLink, Outlet, useMatch, useNavigate, useParams } from "react-router-dom";
import { ArrowLeft } from "lucide-react";
import { ProjectEventsProvider } from "../api/events";
import { ChatProvider, useChat } from "../chat/ChatProvider";
import { cn } from "../lib/format";

/** Lets any detail page hand a prompt to the agent (it opens the chat and sends). */
const AgentCtx = createContext<{ ask: (msg: string) => void }>({ ask: () => {} });
export const useAgent = () => useContext(AgentCtx);

export function ProjectLayout() {
  const { pid } = useParams<{ pid: string }>();
  if (!pid) return null;
  return (
    <ProjectEventsProvider pid={pid}>
      <ChatProvider pid={pid}>
        <Frame pid={pid} />
      </ChatProvider>
    </ProjectEventsProvider>
  );
}

function Frame({ pid }: { pid: string }) {
  const chat = useChat();
  const nav = useNavigate();
  const isChat = !!useMatch("/p/:pid");
  const ask = useCallback(
    (msg: string) => {
      chat.send(msg);
      nav(`/p/${pid}`);
    },
    [chat, nav, pid],
  );
  return (
    <AgentCtx.Provider value={{ ask }}>
      {isChat ? (
        <Outlet />
      ) : (
        <div className="flex h-full min-h-0 flex-col">
          <DetailsBar pid={pid} />
          <main className="min-h-0 flex-1 overflow-y-auto scroll-thin">
            <Outlet />
          </main>
        </div>
      )}
    </AgentCtx.Provider>
  );
}

const SECTIONS = [
  ["overview", "Overview"],
  ["collect", "Collect"],
  ["analyze", "Analyze"],
  ["improve", "Improve"],
] as const;

/** Secondary pages: a quiet bar with the way back to the chat. */
function DetailsBar({ pid }: { pid: string }) {
  const chat = useChat();
  return (
    <div className="flex h-10 shrink-0 items-center gap-1 border-b border-line px-3">
      <NavLink to={`/p/${pid}`} className="mr-2 inline-flex items-center gap-1.5 rounded-md px-2 py-1 text-sm text-ink-2 hover:bg-rail hover:text-ink">
        <ArrowLeft className="h-3.5 w-3.5" /> Back to chat
        {chat.busy && <span className="ml-1 h-1.5 w-1.5 rounded-full bg-servo live-dot" title="The agent is working" />}
      </NavLink>
      {SECTIONS.map(([path, label]) => (
        <NavLink
          key={path}
          to={`/p/${pid}/${path}`}
          className={({ isActive }) => cn("rounded-md px-2.5 py-1 text-sm transition-colors", isActive ? "bg-rail font-medium text-ink" : "text-ink-3 hover:text-ink")}
        >
          {label}
        </NavLink>
      ))}
    </div>
  );
}
