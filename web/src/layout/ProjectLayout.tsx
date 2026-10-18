import { createContext, useCallback, useContext, useRef, useState, type ReactNode } from "react";
import { NavLink, Outlet, useMatch, useParams } from "react-router-dom";
import { PanelRightClose, PanelRightOpen } from "lucide-react";
import { ProjectEventsProvider, useStreamStatus } from "../api/events";
import { useAutopilot, useEpisodes, useProjectState } from "../api/hooks";
import { AgentPanel } from "../agent/AgentPanel";
import { cn } from "../lib/format";

/** Lets any page hand a prompt to the persistent agent panel. */
const AgentCtx = createContext<{ ask: (msg: string) => void }>({ ask: () => {} });
export const useAgent = () => useContext(AgentCtx);

export function ProjectLayout() {
  const { pid } = useParams<{ pid: string }>();
  if (!pid) return null;
  return (
    <ProjectEventsProvider pid={pid}>
      <ProjectFrame pid={pid} />
    </ProjectEventsProvider>
  );
}

function ProjectFrame({ pid }: { pid: string }) {
  const [agentOpen, setAgentOpen] = useState(true);
  const onChat = !!useMatch("/p/:pid");
  const askRef = useRef<(m: string) => void>(() => {});
  const ask = useCallback((m: string) => {
    setAgentOpen(true);
    askRef.current(m);
  }, []);

  return (
    <AgentCtx.Provider value={{ ask }}>
      <div className="flex h-full min-h-0">
        <SideNav pid={pid} />
        <main className="min-w-0 flex-1 overflow-y-auto scroll-thin">
          <Outlet />
        </main>
        {/* The chat page is the chat; secondary pages keep it at hand in a side panel. */}
        {!onChat && (
          <aside
            className={cn(
              "relative hidden shrink-0 flex-col border-l border-line transition-[width] duration-200 ease-out md:flex",
              agentOpen ? "w-[360px] 2xl:w-[400px]" : "w-10",
            )}
          >
            <button
              type="button"
              onClick={() => setAgentOpen((o) => !o)}
              className="absolute left-2 top-2 z-10 rounded-full p-1 text-ink-3 hover:bg-rail hover:text-ink"
              aria-label={agentOpen ? "Collapse agent panel" : "Expand agent panel"}
            >
              {agentOpen ? <PanelRightClose className="h-4 w-4" /> : <PanelRightOpen className="h-4 w-4" />}
            </button>
            <div className={cn("min-h-0 flex-1", !agentOpen && "hidden")}>
              <AgentPanel pid={pid} registerAsk={(fn) => (askRef.current = fn)} />
            </div>
          </aside>
        )}
      </div>
    </AgentCtx.Provider>
  );
}

function SideNav({ pid }: { pid: string }) {
  const state = useProjectState(pid);
  const episodes = useEpisodes(pid);
  const stream = useStreamStatus();
  const ap = useAutopilot(pid).data;
  const autopilotMeta = ap ? (ap.status === "waiting_approval" ? "needs you" : ap.status === "running" ? "running" : undefined) : undefined;
  const plan = state.data?.project.plan;
  const collected = plan?.cells.reduce((a, c) => a + c.collected, 0);
  const evals = state.data?.evaluations.length;
  const accepted = episodes.data?.filter((e) => e.status === "accepted").length;

  return (
    <nav className="hidden w-[208px] shrink-0 flex-col border-r border-line px-3 py-6 sm:flex" aria-label="Project">
      <NavItem to={`/p/${pid}`} end label="Chat" />
      <div className="label mb-2 mt-7 px-2 !text-ink-4">Workbench</div>
      <NavItem to={`/p/${pid}/collect`} label="Collect" meta={plan ? `${collected}/${plan.budget_total}` : undefined} />
      <NavItem to={`/p/${pid}/analyze`} label="Analyze" meta={accepted !== undefined ? `${accepted} ok` : undefined} />
      <NavItem to={`/p/${pid}/improve`} label="Improve" meta={evals ? `${evals} eval${evals === 1 ? "" : "s"}` : undefined} />
      <NavItem to={`/p/${pid}/graph`} label="Lineage" />
      <NavItem to={`/p/${pid}/autopilot`} label="Autopilot" meta={autopilotMeta} />

      <div className="mt-auto space-y-1.5 px-2">
        <div className="label flex items-center gap-2 !tracking-[0.14em]" title="Project event stream (SSE)">
          <span
            className={cn(
              "h-1.5 w-1.5 rounded-full",
              stream === "open" ? "bg-servo live-dot" : stream === "connecting" ? "border border-caution" : "bg-fault",
            )}
          />
          {stream === "open" ? "Live" : stream === "connecting" ? "Connecting" : "Offline"}
        </div>
        <div className="truncate font-mono text-[10.5px] text-ink-4" title={pid}>
          {pid}
        </div>
      </div>
    </nav>
  );
}

function NavItem({ to, label, meta, end }: { to: string; label: string; meta?: ReactNode; end?: boolean }) {
  return (
    <NavLink
      to={to}
      end={end}
      className={({ isActive }) =>
        cn(
          "relative flex h-8 items-center gap-2.5 rounded px-2 text-sm transition-colors duration-100",
          isActive ? "text-ink" : "text-ink-3 hover:text-ink",
        )
      }
    >
      {({ isActive }) => (
        <>
          <span className={cn("absolute left-0 top-1.5 h-5 w-px", isActive ? "bg-servo" : "bg-transparent")} aria-hidden />
          <span className={cn("flex-1", isActive && "font-medium")}>{label}</span>
          {meta && <span className="num font-mono text-[10.5px] text-ink-4">{meta}</span>}
        </>
      )}
    </NavLink>
  );
}
