import { createContext, useCallback, useContext } from "react";
import { NavLink, Outlet, useLocation, useMatch, useNavigate, useParams } from "react-router-dom";
import { ArrowLeft, Camera, FlaskConical, LayoutDashboard, MessagesSquare, ScanSearch, type LucideIcon } from "lucide-react";
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
      <div className="flex h-full min-h-0">
        <ManualRail pid={pid} />
        <div className="min-w-0 flex-1">
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
        </div>
      </div>
    </AgentCtx.Provider>
  );
}

const SECTIONS = [
  { path: "overview", label: "Overview", hint: "Plan, autopilot and next actions", icon: LayoutDashboard },
  { path: "collect", label: "Collect", hint: "Record or import demonstrations by hand", icon: Camera },
  { path: "analyze", label: "Analyze", hint: "Quality checks and failure diagnoses", icon: ScanSearch },
  { path: "improve", label: "Improve", hint: "Datasets, training and evaluations", icon: FlaskConical },
] as const;

/**
 * Slim, always-visible rail: the chat (the default screen) on top, the manual dashboard below.
 * It never expands on its own; labels appear on hover/focus.
 */
function ManualRail({ pid }: { pid: string }) {
  const chat = useChat();
  return (
    <nav aria-label="Project" className="flex w-12 shrink-0 flex-col items-center gap-1 border-r border-line bg-bench py-2">
      <RailLink to={`/p/${pid}`} end label="Chat" hint="Work with the agent (default)" icon={MessagesSquare} busy={chat.busy} />
      <div className="my-1.5 h-px w-6 bg-[var(--line)]" />
      <span className="pb-0.5 text-[9px] font-medium uppercase tracking-[0.08em] text-ink-4">Manual</span>
      {SECTIONS.map((s) => (
        <RailLink key={s.path} to={`/p/${pid}/${s.path}`} label={s.label} hint={s.hint} icon={s.icon} />
      ))}
    </nav>
  );
}

function RailLink({ to, end, label, hint, icon: Icon, busy }: { to: string; end?: boolean; label: string; hint: string; icon: LucideIcon; busy?: boolean }) {
  return (
    <NavLink
      to={to}
      end={end}
      aria-label={label}
      className={({ isActive }) =>
        cn(
          "group relative flex h-9 w-9 items-center justify-center rounded-lg transition-colors",
          isActive ? "bg-rail text-ink shadow-[0_0_0_1px_var(--line)]" : "text-ink-3 hover:bg-rail/60 hover:text-ink",
        )
      }
    >
      {({ isActive }) => (
        <>
          {isActive && <span className="absolute -left-[7px] top-2 h-5 w-0.5 rounded-full bg-servo" aria-hidden />}
          <Icon className="h-[18px] w-[18px]" />
          {busy && <span className="absolute right-1.5 top-1.5 h-1.5 w-1.5 rounded-full bg-servo live-dot" title="The agent is working" />}
          <span
            role="tooltip"
            className="pointer-events-none absolute left-full top-1/2 z-50 ml-2.5 hidden -translate-y-1/2 whitespace-nowrap rounded-lg bg-plate px-2.5 py-1.5 text-left shadow-[0_0_0_1px_var(--line-strong),0_8px_20px_-8px_rgba(0,0,0,.5)] group-hover:block group-focus-visible:block"
          >
            <span className="block text-xs font-medium text-ink">{label}</span>
            <span className="block text-[11px] text-ink-3">{hint}</span>
          </span>
        </>
      )}
    </NavLink>
  );
}

/** Manual pages: a slim bar with the obvious way back to the chat. */
function DetailsBar({ pid }: { pid: string }) {
  const chat = useChat();
  const { pathname } = useLocation();
  const section = SECTIONS.find((s) => pathname.startsWith(`/p/${pid}/${s.path}`)) ?? null;
  return (
    <div className="flex h-10 shrink-0 items-center gap-2 border-b border-line px-3">
      <NavLink to={`/p/${pid}`} className="inline-flex items-center gap-1.5 rounded-md px-2 py-1 text-sm text-ink-2 hover:bg-rail hover:text-ink">
        <ArrowLeft className="h-3.5 w-3.5" /> Back to chat
        {chat.busy && <span className="ml-1 h-1.5 w-1.5 rounded-full bg-servo live-dot" title="The agent is working" />}
      </NavLink>
      {section && (
        <span className="text-sm text-ink-3">
          <span className="text-ink-4">Manual ·</span> {section.label}
        </span>
      )}
    </div>
  );
}
