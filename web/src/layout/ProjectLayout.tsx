import { createContext, useCallback, useContext, useState } from "react";
import { NavLink, Outlet, useLocation, useMatch, useNavigate, useParams } from "react-router-dom";
import { ArrowLeft, Camera, FlaskConical, Joystick, LayoutDashboard, Server, MessagesSquare, ScanSearch, Sparkles, type LucideIcon } from "lucide-react";
import { ProjectEventsProvider } from "../api/events";
import { ChatProvider, useChat } from "../chat/ChatProvider";
import { ChatHistoryPanel, ChatHistoryPanelCtx } from "../chat/ChatHistoryPanel";
import { cn } from "../lib/format";

/** Lets any detail page hand a prompt to the agent (it opens the chat and sends). */
const AgentCtx = createContext<{ ask: (msg: string) => void }>({ ask: () => {} });
export const useAgent = () => useContext(AgentCtx);

export function ProjectLayout() {
  const { pid } = useParams<{ pid: string }>();
  if (!pid) return null;
  // keyed by project: opening another project starts every piece of state fresh (its chat thread, the viewer's
  // live frame, rollout selection and filters), so nothing from the last simulation shows in the new one
  return (
    <ProjectEventsProvider key={pid} pid={pid}>
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
  const [historyOpen, setHistoryOpenState] = useState(() => {
    try {
      return localStorage.getItem("cortex.chatHistoryOpen") === "1";
    } catch {
      return false;
    }
  });
  const setHistoryOpen = useCallback((open: boolean) => {
    setHistoryOpenState(open);
    try {
      localStorage.setItem("cortex.chatHistoryOpen", open ? "1" : "0");
    } catch {
      /* storage unavailable */
    }
  }, []);
  const ask = useCallback(
    (msg: string) => {
      chat.send(msg);
      nav(`/p/${pid}`);
    },
    [chat, nav, pid],
  );
  return (
    <AgentCtx.Provider value={{ ask }}>
      <ChatHistoryPanelCtx.Provider value={{ open: historyOpen, setOpen: setHistoryOpen }}>
        <div className="flex h-full min-h-0">
          {/* Chat on a manual page opens the chat with its list; on the chat it toggles the list */}
          <ManualRail pid={pid} historyOpen={isChat && historyOpen} onChat={() => setHistoryOpen(isChat ? !historyOpen : true)} />
          {isChat && historyOpen && <ChatHistoryPanel pid={pid} onClose={() => setHistoryOpen(false)} />}
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
      </ChatHistoryPanelCtx.Provider>
    </AgentCtx.Provider>
  );
}

const SECTIONS = [
  { path: "overview", label: "Overview", hint: "Plan, autopilot and next actions", icon: LayoutDashboard },
  { path: "collect", label: "Collect", hint: "Record or import demonstrations by hand", icon: Camera },
  { path: "analyze", label: "Analyze", hint: "Quality checks and failure diagnoses", icon: ScanSearch },
  { path: "improve", label: "Improve", hint: "Datasets, training and evaluations", icon: FlaskConical },
  { path: "robot", label: "Robot", hint: "Real SO-101: setup, calibration, teleop, policy runs", icon: Joystick },
  { path: "setup", label: "Setup", hint: "Planner, training GPU and walltime", icon: Server },
] as const;

/**
 * Slim, always-visible rail: the chat (the default screen) on top, the manual dashboard below.
 * It never expands on its own; labels appear on hover/focus.
 */
function ManualRail({ pid, historyOpen, onChat }: { pid: string; historyOpen: boolean; onChat: () => void }) {
  const chat = useChat();
  return (
    <nav aria-label="Project" className="flex w-12 shrink-0 flex-col items-center gap-1 border-r border-line bg-bench py-3">
      <RailLink
        to={`/p/${pid}`}
        end
        label="Chat"
        hint={historyOpen ? "Hide past chats" : "Work with the agent · show past chats"}
        icon={MessagesSquare}
        busy={chat.busy}
        onClick={onChat}
        expanded={historyOpen}
      />
      <RailLink to="/studio/skills" label="Physical Skills" hint="Every policy you've trained, ready to run live" icon={Sparkles} />
      <div className="my-2 h-px w-6 bg-[var(--line)]" />
      <span className="pb-1 font-mono text-[8.5px] uppercase tracking-[0.06em] text-ink-3">Manual</span>
      {SECTIONS.map((s) => (
        <RailLink key={s.path} to={`/p/${pid}/${s.path}`} label={s.label} hint={s.hint} icon={s.icon} />
      ))}
    </nav>
  );
}

function RailLink({
  to,
  end,
  label,
  hint,
  icon: Icon,
  busy,
  onClick,
  expanded,
}: {
  to: string;
  end?: boolean;
  label: string;
  hint: string;
  icon: LucideIcon;
  busy?: boolean;
  onClick?: () => void;
  expanded?: boolean;
}) {
  return (
    <NavLink
      to={to}
      end={end}
      onClick={onClick}
      aria-expanded={expanded}
      aria-label={label}
      className={({ isActive }) =>
        cn(
          "group relative flex h-9 w-9 items-center justify-center rounded transition-colors",
          isActive ? "border border-line-strong bg-plate text-ink" : "text-ink-3 hover:bg-rail/60 hover:text-ink",
        )
      }
    >
      {({ isActive }) => (
        <>
          {isActive && <span className="absolute -left-[7px] top-2 h-5 w-0.5 bg-servo" aria-hidden />}
          <Icon className="h-[18px] w-[18px]" strokeWidth={1.5} />
          {busy && <span className="absolute right-1.5 top-1.5 h-1.5 w-1.5 rounded-full bg-servo live-dot" title="The agent is working" />}
          <span
            role="tooltip"
            className="pointer-events-none absolute left-full top-1/2 z-50 ml-2.5 hidden -translate-y-1/2 whitespace-nowrap rounded-sm border border-line-strong bg-plate px-2.5 py-1.5 text-left group-hover:block group-focus-visible:block"
          >
            <span className="block font-mono text-[10.5px] uppercase tracking-[0.14em] text-ink">{label}</span>
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
    <div className="flex h-11 shrink-0 items-center gap-5 border-b border-line px-4 sm:px-6">
      <NavLink to={`/p/${pid}`} className="group inline-flex items-center gap-1.5 text-[13px] text-ink-2 hover:text-ink">
        <ArrowLeft className="h-3.5 w-3.5" strokeWidth={1.5} />
        <span className="border-b border-line-strong group-hover:border-ink">Back to chat</span>
        {chat.busy && <span className="ml-1 h-1.5 w-1.5 rounded-full bg-servo live-dot" title="The agent is working" />}
      </NavLink>
      {section && (
        <span className="kn-label kn-label-rule">
          <span>
            Manual <span className="text-ink">· {section.label}</span>
          </span>
        </span>
      )}
    </div>
  );
}
