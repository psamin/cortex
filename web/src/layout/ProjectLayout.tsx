import { createContext, useCallback, useContext, useRef, useState, type ReactNode } from "react";
import { NavLink, Outlet, useParams } from "react-router-dom";
import { Camera, FlaskConical, Network, PanelRightClose, PanelRightOpen, ScanSearch, Zap } from "lucide-react";
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
        <aside
          className={cn(
            "relative flex shrink-0 flex-col border-l border-line transition-[width] duration-200 ease-out",
            agentOpen ? "w-[340px] 2xl:w-[380px]" : "w-10",
          )}
        >
          <button
            type="button"
            onClick={() => setAgentOpen((o) => !o)}
            className="absolute left-2 top-2 z-10 rounded p-1 text-ink-3 hover:bg-rail hover:text-ink"
            aria-label={agentOpen ? "Collapse agent panel" : "Expand agent panel"}
          >
            {agentOpen ? <PanelRightClose className="h-4 w-4" /> : <PanelRightOpen className="h-4 w-4" />}
          </button>
          <div className={cn("min-h-0 flex-1", !agentOpen && "hidden")}>
            <AgentPanel pid={pid} registerAsk={(fn) => (askRef.current = fn)} />
          </div>
        </aside>
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
    <nav className="flex w-[200px] shrink-0 flex-col border-r border-line px-2 py-3" aria-label="Project">
      <NavItem to={`/p/${pid}`} end icon={<Zap className="h-4 w-4" />} label="Autopilot" meta={autopilotMeta} />
      <NavItem to={`/p/${pid}/graph`} icon={<Network className="h-4 w-4" />} label="Lineage" />
      <NavItem
        to={`/p/${pid}/collect`}
        icon={<Camera className="h-4 w-4" />}
        label="Collect"
        meta={plan ? `${collected}/${plan.budget_total}` : undefined}
      />
      <NavItem
        to={`/p/${pid}/analyze`}
        icon={<ScanSearch className="h-4 w-4" />}
        label="Analyze"
        meta={accepted !== undefined ? `${accepted} ok` : undefined}
      />
      <NavItem
        to={`/p/${pid}/improve`}
        icon={<FlaskConical className="h-4 w-4" />}
        label="Improve"
        meta={evals ? `${evals} eval${evals === 1 ? "" : "s"}` : undefined}
      />

      <div className="mt-auto space-y-1 px-2 text-2xs text-ink-3">
        <div className="flex items-center gap-1.5" title="Project event stream (SSE)">
          <span
            className={cn(
              "h-1.5 w-1.5 rounded-full",
              stream === "open" ? "bg-pass" : stream === "connecting" ? "bg-caution live-dot" : "bg-fault",
            )}
          />
          {stream === "open" ? "Live updates" : stream === "connecting" ? "Connecting stream…" : "Stream offline"}
        </div>
        <div className="truncate font-mono text-ink-4" title={pid}>
          {pid}
        </div>
      </div>
    </nav>
  );
}

function NavItem({ to, icon, label, meta, end }: { to: string; icon: ReactNode; label: string; meta?: string; end?: boolean }) {
  return (
    <NavLink
      to={to}
      end={end}
      className={({ isActive }) =>
        cn(
          "group flex h-8 items-center gap-2.5 rounded px-2 text-sm transition-colors duration-100",
          isActive ? "bg-rail font-medium text-ink" : "text-ink-2 hover:bg-rail/60 hover:text-ink",
        )
      }
    >
      {({ isActive }) => (
        <>
          <span className={cn(isActive ? "text-servo" : "text-ink-3 group-hover:text-ink-2")}>{icon}</span>
          <span className="flex-1">{label}</span>
          {meta && <span className="num font-mono text-2xs text-ink-3">{meta}</span>}
        </>
      )}
    </NavLink>
  );
}
