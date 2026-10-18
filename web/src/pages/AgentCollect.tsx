import { useEffect, useMemo, useRef, useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { ArrowRight, Bot, ChevronDown, ChevronLeft, ChevronRight, Circle, Cog, Info, Square, VideoOff, Zap } from "lucide-react";
import { api, ApiError } from "../api/client";
import { isJobActive, useAgentLive, useHydrateLive, useJob, useStreamStatus, type LiveAttempt } from "../api/events";
import {
  useAgentAttempts,
  useAgentCollection,
  useAgentCollections,
  useCollectorComparison,
  useEpisodes,
  useNextActions,
  usePlanner,
  usePlannerBudget,
} from "../api/hooks";
import { agentLabel, conditionKey, isAgent, type AgentCollection, type Collector, type CollectorStats, type PlannerStatus, type PlanningMode, type Project } from "../api/types";
import { AttemptDrawer, PlanningModeBadge, ProvenanceBadge, ScenarioChip, ThoughtRow, TraceRow, Verdict } from "../components/AgentTrace";
import { Badge, Button, EmptyState, ErrorState, Field, InfoTip, Input, Meter, More, Panel, QueryState, SectionLabel, Segmented, Select, SkeletonRows, StatusBadge, Toggle, InlineError } from "../components/ui";
import { WorkcellGrid } from "../components/Workcell";
import { ago, cn, humanize, num, pct } from "../lib/format";

/** "Claude Code agent", "GPT-6 Astra", … or "Scripted IK oracle". */
const collectorName = (c: string, model?: string | null) => (isAgent(c) ? agentLabel(model) : "Scripted IK oracle");
const ACTIVE = new Set(["queued", "running"]);

function useElementWidth<T extends HTMLElement>() {
  const ref = useRef<T>(null);
  const [w, setW] = useState(1200);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => setW(e.contentRect.width));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return [ref, w] as const;
}

export function AgentCollect({ project, collector, onUseOracle }: { project: Project; collector: Collector; onUseOracle: () => void }) {
  const pid = project.id;
  const planner = usePlanner();
  const collections = useAgentCollections(pid);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [drawer, setDrawer] = useState<string | null>(null);
  const [wrapRef, width] = useElementWidth<HTMLDivElement>();

  // default to the newest collection of this collector (a running one first)
  const mine = useMemo(
    () => [...(collections.data ?? [])].filter((c) => c.collector === collector).sort((a, b) => (a.created_at < b.created_at ? 1 : -1)),
    [collections.data, collector],
  );
  const current = mine.find((c) => c.id === selectedId) ?? mine.find((c) => ACTIVE.has(c.status)) ?? mine[0] ?? null;
  const job = useJob(current?.job_id);
  const running = !!current && (ACTIVE.has(current.status) || isJobActive(job));
  const detail = useAgentCollection(current?.id, running);
  const col = detail.data?.collection ?? current;

  const plannerDown = isAgent(collector) && planner.data && !planner.data.available;

  // Rebuild the in-progress attempt on load and on every stream (re)connect, then SSE continues it.
  const hydrate = useHydrateLive();
  const stream = useStreamStatus();
  useEffect(() => {
    const cid = current?.id;
    if (!cid || !running || stream !== "open") return;
    let cancelled = false;
    api
      .agentCollectionLive(cid)
      .then((live) => !cancelled && hydrate(cid, live))
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [current?.id, running, stream]); // eslint-disable-line react-hooks/exhaustive-deps
  const cols = width >= 1120 ? "three" : width >= 760 ? "two" : "one";

  return (
    <div ref={wrapRef} className="space-y-4">
      {isAgent(collector) && planner.isLoading && <SkeletonRows rows={1} />}
      {plannerDown && planner.data && !planner.data.available && <PlannerUnavailable reason={planner.data.reason} onUseOracle={onUseOracle} />}
      {planner.error && isAgent(collector) && <ErrorState error={planner.error} compact onRetry={() => planner.refetch()} />}

      <StartForm
        project={project}
        collector={collector}
        planner={planner.data}
        disabled={!!plannerDown || running}
        running={running}
        current={col}
        onStarted={(c) => setSelectedId(c.id)}
      />

      {mine.length > 1 && (
        <div className="flex items-center gap-2 text-xs text-ink-3">
          <span>Collection</span>
          <Select value={current?.id ?? ""} onChange={(e) => setSelectedId(e.target.value)} className="h-7 w-auto min-w-[280px] text-xs" aria-label="Collection">
            {mine.map((c) => (
              <option key={c.id} value={c.id}>
                {c.id} · {c.status} · {c.totals.successes ?? 0}/{c.request.num_demos} · {ago(c.created_at)}
              </option>
            ))}
          </Select>
        </div>
      )}

      <div
        className={cn(
          "grid gap-4",
          cols === "three" && "grid-cols-[minmax(0,1.05fr)_minmax(0,1fr)_minmax(0,0.85fr)]",
          cols === "two" && "grid-cols-[minmax(0,1fr)_minmax(0,1fr)]",
          cols === "one" && "grid-cols-1",
        )}
      >
        <LiveViewport pid={pid} col={col} running={running} jobMessage={job?.message} progress={job?.progress} />
        <ActionTimeline col={col} running={running} collector={collector} onOpenAttempt={setDrawer} />
        <div className={cn(cols === "two" && "col-span-2")}>
          <DatasetProgress project={project} col={col} plannerRoute={planner.data?.available ? planner.data.route : undefined} />
        </div>
      </div>

      <AttemptsTable pid={pid} onOpen={setDrawer} collectionId={col?.id} />
      <CollectorComparisonCard pid={pid} />
      <AttemptDrawer attemptId={drawer} onClose={() => setDrawer(null)} />
    </div>
  );
}

// ------------------------------------------------------------------ planner status
function PlannerUnavailable({ reason, onUseOracle }: { reason: string; onUseOracle: () => void }) {
  return (
    <div className="flex flex-wrap items-start gap-3 rounded-md bg-caution/5 p-3.5 border border-caution/35">
      <Info className="mt-0.5 h-4 w-4 shrink-0 text-caution" />
      <div className="min-w-0 flex-1">
        <p className="text-sm font-medium text-ink">No planning model available.</p>
        <p className="mt-0.5 text-xs text-ink-3">{reason} No other model is substituted.</p>
      </div>
      <Button size="sm" onClick={onUseOracle}>
        <Cog className="h-3.5 w-3.5" /> Use oracle
      </Button>
    </div>
  );
}

// ------------------------------------------------------------------ start form
function StartForm({
  project,
  collector,
  planner,
  disabled,
  running,
  current,
  onStarted,
}: {
  project: Project;
  collector: Collector;
  planner?: PlannerStatus;
  disabled: boolean;
  running: boolean;
  current: AgentCollection | null;
  onStarted: (c: AgentCollection) => void;
}) {
  const pid = project.id;
  const qc = useQueryClient();
  const [numDemos, setNumDemos] = useState(6);
  const [diversity, setDiversity] = useState<"plan" | "medium" | "high">("medium");
  const [mode, setMode] = useState<PlanningMode>(project.autopilot.planning_mode ?? "camera");
  const [budget, setBudget] = useState(project.autopilot.model_budget_usd ?? 10);
  const [autopilot, setAutopilot] = useState(false);
  const [limitsOpen, setLimitsOpen] = useState(false);
  const [maxCalls, setMaxCalls] = useState(16);
  const [maxUsdEp, setMaxUsdEp] = useState(1.5);
  const [maxTotal, setMaxTotal] = useState("");
  const [task, setTask] = useState(project.spec.goal);
  const taskOverride = task.trim() && task.trim() !== project.spec.goal ? task.trim() : null;
  const budgetQ = usePlannerBudget(pid);
  useEffect(() => setBudget(project.autopilot.model_budget_usd ?? 10), [project.autopilot.model_budget_usd]);

  const invalidate = () => {
    for (const k of ["agent-collections", "state", "autopilot", "projects"]) qc.invalidateQueries({ queryKey: [k, pid] });
    qc.invalidateQueries({ queryKey: ["projects"] });
  };
  const start = useMutation({
    mutationFn: async () => {
      const policyPatch =
        isAgent(collector)
          ? { collector, planning_mode: mode, model_budget_usd: budget }
          : { collector };
      if (autopilot) {
        await api.setAutopilotPolicy(pid, policyPatch);
        await api.startAutopilot(pid);
        return null;
      }
      if (isAgent(collector) && budget !== project.autopilot.model_budget_usd) await api.setAutopilotPolicy(pid, { model_budget_usd: budget });
      return api.startAgentCollection(pid, {
        ...(taskOverride ? { task: taskOverride } : {}),
        num_demos: numDemos,
        diversity,
        collector,
        mode: isAgent(collector) ? mode : null,
        max_calls_per_episode: maxCalls,
        max_usd_per_episode: maxUsdEp,
        max_total_usd: maxTotal === "" ? null : Number(maxTotal),
      });
    },
    onSuccess: (c) => {
      invalidate();
      qc.invalidateQueries({ queryKey: ["planner-budget", pid] });
      if (c) onStarted(c);
    },
    onError: (e) => {
      // the planner went away since /planner was read: refresh so the notice shows
      if (e instanceof ApiError && e.code === "planner_unavailable") qc.invalidateQueries({ queryKey: ["planner"] });
    },
  });
  const stop = useMutation({
    mutationFn: () => api.stopAgentCollection(current!.id),
    onSuccess: invalidate,
  });
  const budgetBlocked = start.error instanceof ApiError && start.error.code === "planner_budget";

  return (
    <Panel
      title={isAgent(collector) ? "Agent demos" : "Oracle demos"}
      meta={isAgent(collector) ? `${agentLabel(planner?.model)} plans · IK executes` : "oracle · privileged"}
      actions={
        running && current ? (
          <Button size="sm" variant="danger" loading={stop.isPending} onClick={() => stop.mutate()}>
            <Square className="h-3 w-3 fill-current" /> Stop
          </Button>
        ) : (
          <Button size="sm" variant="primary" disabled={disabled} loading={start.isPending} onClick={() => start.mutate()}>
            {autopilot ? <Zap className="h-3.5 w-3.5" /> : <Circle className="h-3 w-3 fill-current" />}
            {autopilot ? "Autopilot" : `Collect ${numDemos}`}
          </Button>
        )
      }
    >
      <More className="mb-3" label="How it works">
        {isAgent(collector)
          ? `${agentLabel(planner?.model)} plans each step from the camera, a validator checks every command, IK executes it in MuJoCo, and only attempts the simulator verifies as successful become LeRobot demonstrations.`
          : "The scripted oracle follows fixed IK waypoints computed from simulator ground truth. It is a reference collector with privileged state, not a learned or camera-grounded agent."}
      </More>
      <div className="flex flex-wrap items-end gap-x-4 gap-y-3">
        <div className="min-w-[220px] flex-1">
          <Field
            label="Task"
            hint={
              autopilot
                ? "Autopilot uses the project goal"
                : taskOverride
                  ? "Overrides the goal for this run"
                  : "Project goal · edit to override"
            }
          >
            <Input value={autopilot ? project.spec.goal : task} disabled={autopilot} onChange={(e) => setTask(e.target.value)} aria-label="Task given to the planner" />
          </Field>
        </div>
        <div className="w-[88px]">
          <Field label="Attempts">
            <Input type="number" min={1} max={50} value={numDemos} disabled={autopilot} onChange={(e) => setNumDemos(Math.min(50, Math.max(1, Number(e.target.value) || 1)))} className="num font-mono" />
          </Field>
        </div>
        <Field label="Scene diversity">
          <Segmented size="md" value={diversity} onChange={setDiversity} ariaLabel="Scene diversity" options={[{ value: "plan", label: "plan" }, { value: "medium", label: "medium" }, { value: "high", label: "high" }]} />
        </Field>
        {isAgent(collector) && (
          <>
            <Field label="Planning mode">
              <Segmented size="md" value={mode} onChange={setMode} ariaLabel="Planning mode" options={[{ value: "camera", label: "camera" }, { value: "privileged", label: "privileged" }]} />
            </Field>
            <div className="w-[120px]">
              <Field label="Model budget">
                <div className="flex items-center gap-1">
                  <span className="text-xs text-ink-3">$</span>
                  <Input type="number" min={0} step={1} value={budget} onChange={(e) => setBudget(Math.max(0, Number(e.target.value) || 0))} className="num font-mono" />
                </div>
              </Field>
              {budgetQ.data && (
                <p className="num mt-1 font-mono text-[10.5px] text-ink-3">
                  ${budgetQ.data.spent_usd.toFixed(2)} spent · ${budgetQ.data.remaining_usd.toFixed(2)} left
                </p>
              )}
            </div>
          </>
        )}
        <div className="pb-0.5">
          <Toggle checked={autopilot} onChange={setAutopilot} label="Autopilot" hint="full loop" />
        </div>
      </div>

      {isAgent(collector) && (
        <div className="mt-2">
          <button type="button" onClick={() => setLimitsOpen((o) => !o)} className="inline-flex items-center gap-1 text-2xs text-ink-3 hover:text-ink-2" aria-expanded={limitsOpen}>
            Per-attempt limits <ChevronDown className={cn("h-3 w-3 transition-transform", limitsOpen && "rotate-180")} />
          </button>
          {limitsOpen && (
            <div className="mt-2 flex flex-wrap gap-3">
              <div className="w-[140px]">
                <Field label="Max calls / attempt">
                  <Input type="number" min={4} max={40} value={maxCalls} onChange={(e) => setMaxCalls(Math.min(40, Math.max(4, Number(e.target.value) || 4)))} className="num font-mono" />
                </Field>
              </div>
              <div className="w-[140px]">
                <Field label="Max $ / attempt">
                  <Input type="number" min={0.1} max={10} step={0.1} value={maxUsdEp} onChange={(e) => setMaxUsdEp(Number(e.target.value) || 0.1)} className="num font-mono" />
                </Field>
              </div>
              <div className="w-[160px]">
                <Field label="Collection cap $" hint="Blank = remaining budget">
                  <Input type="number" min={0} step={0.5} value={maxTotal} onChange={(e) => setMaxTotal(e.target.value)} placeholder="auto" className="num font-mono" />
                </Field>
              </div>
            </div>
          )}
        </div>
      )}
      {autopilot && (
        <p className="mt-2 text-2xs text-ink-3">
          {collectorName(collector, planner?.model)} · full loop under your standing approvals.
        </p>
      )}
      {planner && planner.available && isAgent(collector) && (
        <p className="mt-2 text-2xs text-ink-4">
          {typeof planner.pricing_per_m_tokens === "string"
            ? `Cost: ${planner.pricing_per_m_tokens}`
            : `$${planner.pricing_per_m_tokens.input}/M in · $${planner.pricing_per_m_tokens.output}/M out tokens`}
        </p>
      )}

      {start.error && !budgetBlocked && (
        <div className="mt-3">
          <ErrorState error={start.error} compact />
        </div>
      )}
      {budgetBlocked && (
        <RaiseBudget
          pid={pid}
          current={budgetQ.data?.budget_usd ?? project.autopilot.model_budget_usd}
          message={(start.error as Error).message}
          onRaised={(b) => {
            setBudget(b);
            start.reset();
          }}
        />
      )}
      {stop.error && <InlineError error={stop.error} className="mt-2" />}
      {start.isSuccess && autopilot && <p className="mt-2 text-xs text-pass">Autopilot started.</p>}
    </Panel>
  );
}

function RaiseBudget({ pid, current, message, onRaised }: { pid: string; current: number; message: string; onRaised: (b: number) => void }) {
  const qc = useQueryClient();
  const [value, setValue] = useState(Math.ceil((current ?? 0) + 5));
  const m = useMutation({
    mutationFn: () => api.setAutopilotPolicy(pid, { model_budget_usd: value }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["state", pid] });
      qc.invalidateQueries({ queryKey: ["projects"] });
      qc.invalidateQueries({ queryKey: ["planner-budget", pid] });
      onRaised(value);
    },
  });
  return (
    <div className="mt-3 flex flex-wrap items-center gap-2 rounded bg-caution/5 p-2.5 border border-caution/35">
      <p className="min-w-[240px] flex-1 text-xs text-ink-2">{message}</p>
      <span className="text-xs text-ink-3">$</span>
      <Input type="number" min={0} step={1} value={value} onChange={(e) => setValue(Number(e.target.value) || 0)} className="num h-7 w-20 font-mono" aria-label="New model budget" />
      <Button size="sm" variant="primary" loading={m.isPending} disabled={value <= (current ?? 0)} onClick={() => m.mutate()}>
        Raise budget
      </Button>
      {m.error && <InlineError error={m.error} className="w-full" />}
    </div>
  );
}

// ------------------------------------------------------------------ left: live viewport
function LiveViewport({
  pid,
  col,
  running,
  jobMessage,
  progress,
}: {
  pid: string;
  col: AgentCollection | null;
  running: boolean;
  jobMessage?: string;
  progress?: number;
}) {
  const live = useAgentLive(col?.id);
  const cur: LiveAttempt | undefined = live ? live.attempts[live.lastIndex] : undefined;
  const planned = col && cur ? col.scenarios[cur.index] : undefined;
  const src = running ? api.simLiveUrl(pid) : null;
  const [failed, setFailed] = useState(false);
  useEffect(() => setFailed(false), [src]);

  return (
    <Panel
      title="Live simulation"
      meta={running ? "MuJoCo" : "idle"}
      actions={col && <StatusBadge status={col.status} />}
      bodyClassName="p-0"
    >
      <div className="relative aspect-[4/3] w-full overflow-hidden bg-well">
        {src && !failed ? (
          <img src={src} alt="Live MuJoCo execution" onError={() => setFailed(true)} className="h-full w-full object-contain" />
        ) : (
          <div className="flex h-full flex-col items-center justify-center gap-2 px-6 text-center text-ink-4">
            <VideoOff className="h-6 w-6" />
            <p className="text-xs">{running ? "Live stream unavailable" : "Not running"}</p>
          </div>
        )}
        {running && (
          <div className="absolute left-3 top-3 flex items-center gap-1.5 rounded-sm bg-black/60 px-2 py-1 text-2xs font-semibold text-white">
            <span className="h-2 w-2 rounded-full bg-fault live-dot" /> LIVE
          </div>
        )}
        {running && progress !== undefined && (
          <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/70 to-transparent px-3 pb-2.5 pt-6">
            <div className="mb-1.5 flex justify-between text-2xs text-white/80">
              <span className="truncate">{jobMessage || "running…"}</span>
              <span className="num font-mono">{pct(progress)}</span>
            </div>
            <div className="h-1 overflow-hidden rounded-full bg-white/20">
              <div className="h-full bg-servo transition-[width] duration-300" style={{ width: `${progress * 100}%` }} />
            </div>
          </div>
        )}
      </div>
      <div className="flex flex-wrap items-center gap-2 border-t border-line-soft px-3 py-2">
        {cur || planned ? (
          <>
            <span className="text-2xs text-ink-3">
              attempt <span className="num font-mono text-ink">{(cur?.index ?? 0) + 1}</span>
              {live?.total ? <span className="text-ink-4"> / {live.total}</span> : null}
            </span>
            <ScenarioChip scenario={cur?.scenario ?? planned} condition={cur?.condition} seed={cur?.seed ?? planned?.seed} />
          </>
        ) : (
          <span className="text-2xs text-ink-4">{col ? `${col.scenarios.length} planned scenes` : "No scene yet"}</span>
        )}
      </div>
    </Panel>
  );
}

// ------------------------------------------------------------------ center: action timeline
function ActionTimeline({
  col,
  running,
  collector,
  onOpenAttempt,
}: {
  col: AgentCollection | null;
  running: boolean;
  collector: Collector;
  onOpenAttempt: (id: string) => void;
}) {
  const live = useAgentLive(col?.id);
  const indices = live ? Object.keys(live.attempts).map(Number).sort((a, b) => a - b) : [];
  const [pick, setPick] = useState<number | null>(null);
  useEffect(() => setPick(null), [col?.id]);
  const shown = pick ?? live?.lastIndex ?? null;
  const att = shown !== null && live ? live.attempts[shown] : undefined;
  const scroll = useRef<HTMLOListElement>(null);
  useEffect(() => {
    const el = scroll.current;
    if (el && pick === null) el.scrollTop = el.scrollHeight;
  }, [att?.items.length, att?.result, pick]);
  const pos = shown !== null ? indices.indexOf(shown) : -1;

  return (
    <Panel
      title={isAgent(collector) ? "Agent actions" : "Attempts"}
      meta={att ? (collector === "oracle" ? `${indices.length}${live?.total ? ` of ${live.total}` : ""} seen live` : `attempt ${att.index + 1}${live?.total ? ` of ${live.total}` : ""}`) : undefined}
      actions={
        isAgent(collector) &&
        indices.length > 1 && (
          <div className="flex items-center gap-0.5">
            <Button size="sm" variant="ghost" className="px-1.5" disabled={pos <= 0} onClick={() => setPick(indices[pos - 1])} aria-label="Previous attempt">
              <ChevronLeft className="h-3.5 w-3.5" />
            </Button>
            <Button size="sm" variant="ghost" className="px-1.5" disabled={pos >= indices.length - 1} onClick={() => setPick(pos + 1 >= indices.length - 1 ? null : indices[pos + 1])} aria-label="Next attempt">
              <ChevronRight className="h-3.5 w-3.5" />
            </Button>
          </div>
        )
      }
      bodyClassName="p-0"
    >
      {!col ? (
        <EmptyState title="No collection yet">
          {isAgent(collector)
            ? "Start one to see each tool call."
            : "Start one to see each attempt."}
        </EmptyState>
      ) : collector === "oracle" && live && indices.length > 0 ? (
        <ol className="max-h-[440px] min-h-[200px] divide-y divide-[var(--line-soft)] overflow-y-auto px-3.5 scroll-thin">
          <li className="py-2 text-2xs text-ink-3">Oracle · privileged · no tool trace</li>
          {indices.map((i) => {
            const a = live.attempts[i];
            return (
              <li key={i} className="flex flex-wrap items-center gap-x-3 gap-y-1 py-2">
                <span className="num w-6 font-mono text-xs text-ink-3">{i + 1}</span>
                <ScenarioChip scenario={a.scenario} condition={a.condition} seed={a.seed} className="min-w-0 flex-1" />
                {a.result ? (
                  <>
                    <Verdict a={a.result} />
                    <button type="button" onClick={() => onOpenAttempt(a.result!.id)} className="text-2xs text-ink-3 hover:text-servo">
                      video <ArrowRight className="inline h-3 w-3" />
                    </button>
                  </>
                ) : running && i === live.lastIndex ? (
                  <span className="flex items-center gap-1.5 text-2xs text-ink-3">
                    <span className="h-1.5 w-1.5 rounded-full bg-servo live-dot" /> executing
                  </span>
                ) : (
                  <span className="text-2xs text-ink-4">no verdict received</span>
                )}
              </li>
            );
          })}
        </ol>
      ) : !att ? (
        <EmptyState title={running ? "Waiting for a step" : "No live steps"}>
          {running ? "Earlier steps are in each attempt's trace." : "Open an attempt below for its trace."}
        </EmptyState>
      ) : (
        <>
          {att.scenario || att.condition ? (
            <div className="border-b border-line-soft px-3.5 py-2">
              <ScenarioChip scenario={att.scenario} condition={att.condition} seed={att.seed} />
            </div>
          ) : null}
          <ol ref={scroll} className="max-h-[440px] min-h-[200px] divide-y divide-[var(--line-soft)] overflow-y-auto px-3.5 scroll-thin">
            {att.items.map((it, i) => (it.kind === "step" ? <TraceRow key={i} step={it.step} dense /> : <ThoughtRow key={i} text={it.text} />))}
            {att.items.length === 0 && !att.result && (
              <li className="py-3 text-xs text-ink-3">
                {collector === "oracle" ? "Oracle · no tool trace" : "Waiting for the first model call…"}
              </li>
            )}
            {att.result && (
              <li className="space-y-1.5 py-2.5">
                <Verdict a={att.result} />
                <button type="button" onClick={() => onOpenAttempt(att.result!.id)} className="inline-flex items-center gap-1 text-2xs text-ink-3 hover:text-servo">
                  Video and trace <ArrowRight className="h-3 w-3" />
                </button>
              </li>
            )}
            {!att.result && running && att.index === live?.lastIndex && (
              <li className="flex items-center gap-2 py-2 text-2xs text-ink-3">
                <span className="h-1.5 w-1.5 rounded-full bg-servo live-dot" /> executing
              </li>
            )}
          </ol>
        </>
      )}
    </Panel>
  );
}

// ------------------------------------------------------------------ right: dataset progress
function DatasetProgress({ project, col, plannerRoute }: { project: Project; col: AgentCollection | null; plannerRoute?: string }) {
  const pid = project.id;
  const episodes = useEpisodes(pid);
  const budget = usePlannerBudget(pid);
  const next = useNextActions(pid);
  const t = col?.totals ?? {};
  const produced = (episodes.data ?? []).filter((e) => e.provenance?.collection_id && e.provenance.collection_id === col?.id);
  const scored = produced.filter((e) => e.quality);
  const meanQ = scored.length ? scored.reduce((s, e) => s + (e.quality?.score ?? 0), 0) / scored.length : null;
  const accepted = produced.filter((e) => e.status === "accepted").length;
  const plan = new Map(project.plan.cells.map((c) => [conditionKey(c.condition), c.target]));
  const cap = t.budget_cap_usd ?? 0;

  return (
    <Panel title="Dataset progress" meta={col ? collectorName(col.collector, col.model) : undefined}>
      {!col ? (
        <p className="text-xs text-ink-3">No collection yet.</p>
      ) : (
        <div className="space-y-4">
          <div className="grid grid-cols-2 gap-3">
            <div>
              <div className="num kn-display text-[40px] leading-none text-pass">{t.successes ?? 0}</div>
              <div className="mt-1 text-2xs text-ink-3">demos · sim-verified</div>
            </div>
            <div>
              <div className="num kn-display text-[40px] leading-none text-ink-2">{t.failures ?? 0}</div>
              <div className="mt-1 text-2xs text-ink-3">failed attempts</div>
            </div>
          </div>
          <p className="-mt-2 flex items-center gap-1.5 text-2xs text-ink-3">
            {`${t.attempts ?? 0}/${col.request.num_demos} attempts`}
            <InfoTip>Failed attempts are kept as diagnostics. They never become training data.</InfoTip>
          </p>

          <div>
            <SectionLabel className="mb-1.5">Coverage · ok / tries (target)</SectionLabel>
            <WorkcellGrid
              renderCell={(c) => {
                const k = conditionKey(c);
                const cell = t.cells?.[k];
                return (
                  <div className="px-1.5 py-1">
                    <span className={cn("num font-mono text-xs", cell?.successes ? "text-pass" : cell ? "text-ink-2" : "text-ink-4")}>
                      {cell ? `${cell.successes}/${cell.attempts}` : "—"}
                    </span>
                    {plan.get(k) ? <span className="num ml-1 font-mono text-[10px] text-ink-4">({plan.get(k)})</span> : null}
                  </div>
                );
              }}
            />
          </div>

          <div className="grid grid-cols-2 gap-3 text-xs">
            <div>
              <SectionLabel>Quality</SectionLabel>
              <div className="num mt-0.5 font-mono text-ink">{meanQ === null ? "—" : meanQ.toFixed(2)}</div>
              <div className="text-2xs text-ink-3">{accepted}/{produced.length} accepted</div>
            </div>
            <div>
              <SectionLabel>Recoveries</SectionLabel>
              <div className="num mt-0.5 font-mono text-ink">{t.recoveries ?? 0}</div>
              <div className="text-2xs text-ink-3">{t.tool_errors ?? 0} rejected cmds</div>
            </div>
          </div>

          {isAgent(col.collector) && (
            <div className="space-y-2">
              <SectionLabel>Model usage</SectionLabel>
              <div className="flex items-baseline justify-between text-xs">
                <span className="text-ink-3">
                  <span className="num font-mono text-ink">{t.model_calls ?? 0}</span> calls
                </span>
                <span className="num font-mono text-ink">
                  ${num(t.cost_usd ?? 0, 2)} <span className="text-ink-4">/ ${num(cap, 2)} cap</span>
                </span>
              </div>
              <Meter value={cap ? (t.cost_usd ?? 0) / cap : 0} tone={cap && (t.cost_usd ?? 0) / cap > 0.85 ? "caution" : "servo"} label="Spend vs collection cap" />
              <div className="flex justify-between text-2xs text-ink-3">
                <span>project total</span>
                <span className="num font-mono">
                  {budget.data
                    ? `$${num(budget.data.spent_usd, 2)} / $${num(budget.data.budget_usd, 2)} approved`
                    : budget.error
                      ? "unavailable"
                      : "…"}
                </span>
              </div>
              <div className="flex flex-wrap items-center gap-1.5 pt-1">
                <Badge title="provider">{col.provider ?? "provider ?"}</Badge>
                <Badge title="model">
                  <span className="font-mono">{col.model ?? "model ?"}</span>
                </Badge>
              </div>
              {plannerRoute && <p className="break-all font-mono text-[10.5px] text-ink-4">route {plannerRoute}</p>}
            </div>
          )}
          <div className="flex flex-wrap items-center gap-1.5">
            <PlanningModeBadge mode={col.mode} />
            {col.collector === "oracle" && <span className="text-2xs text-ink-3">privileged</span>}
          </div>
          {t.stopped_reason && <p className="text-xs text-caution">{t.stopped_reason}</p>}
          {col.error && <InlineError error={col.error} fallback="The collection stopped with an error." />}

          <div className="border-t border-line-soft pt-3">
            <SectionLabel className="mb-1">Next action</SectionLabel>
            {next.data && next.data.length > 0 ? (
              <div>
                <div className="text-sm font-medium capitalize text-ink">{humanize(next.data[0].action)}</div>
                <p className="text-xs text-ink-3">{next.data[0].reason}</p>
              </div>
            ) : (
              <p className="text-xs text-ink-4">{next.isLoading ? "…" : "None"}</p>
            )}
          </div>
        </div>
      )}
    </Panel>
  );
}

// ------------------------------------------------------------------ attempts table
function AttemptsTable({ pid, onOpen, collectionId }: { pid: string; onOpen: (id: string) => void; collectionId?: string }) {
  const q = useAgentAttempts(pid);
  const [onlyCurrent, setOnlyCurrent] = useState(false);
  const rows = useMemo(
    () =>
      [...(q.data ?? [])]
        .filter((a) => !onlyCurrent || a.collection_id === collectionId)
        .sort((a, b) => (a.created_at < b.created_at ? 1 : a.created_at > b.created_at ? -1 : b.index - a.index)),
    [q.data, onlyCurrent, collectionId],
  );
  return (
    <Panel
      title="Attempts"
      meta="click for video and trace"
      actions={
        collectionId && (
          <label className="flex items-center gap-1.5 text-2xs text-ink-3">
            <input type="checkbox" checked={onlyCurrent} onChange={(e) => setOnlyCurrent(e.target.checked)} className="accent-[rgb(var(--servo))]" />
            this run only
          </label>
        )
      }
      bodyClassName="p-0"
    >
      <QueryState q={q} skeleton={<div className="p-3.5"><SkeletonRows rows={3} /></div>}>
        {() =>
          rows.length === 0 ? (
            <EmptyState title="No attempts yet" />
          ) : (
            <div className="max-h-[420px] overflow-auto scroll-thin">
              <table className="w-full text-left text-sm">
                <thead className="sticky top-0 z-10 bg-plate text-2xs uppercase tracking-[0.05em] text-ink-3">
                  <tr className="border-b border-line-soft">
                    <th className="px-3.5 py-2 font-medium">#</th>
                    <th className="px-2 py-2 font-medium">Collector</th>
                    <th className="px-2 py-2 font-medium">Scene</th>
                    <th className="px-2 py-2 font-medium">Result</th>
                    <th className="px-2 py-2 text-right font-medium">Calls</th>
                    <th className="px-2 py-2 text-right font-medium">Cost</th>
                    <th className="px-2 py-2 text-right font-medium">Recov.</th>
                    <th className="px-3.5 py-2 text-right font-medium">Wall</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((a) => (
                    <tr key={a.id} onClick={() => onOpen(a.id)} className="cursor-pointer border-b border-line-soft last:border-0 hover:bg-rail/50">
                      <td className="px-3.5 py-2 font-mono text-xs text-ink-3">{a.index + 1}</td>
                      <td className="px-2 py-2">
                        <ProvenanceBadge planner={isAgent(a.collector) ? a.model ?? "agent" : "oracle"} mode={a.mode} />
                      </td>
                      <td className="px-2 py-2">
                        <ScenarioChip scenario={a.scenario} seed={a.seed} />
                      </td>
                      <td className="px-2 py-2">
                        {a.status === "success" ? <Badge tone="pass">demo</Badge> : <Badge tone="fault" title={a.stop_reason ?? (a.error ? "Ended with an error: open the attempt for details" : undefined)}>failed</Badge>}
                      </td>
                      <td className="num px-2 py-2 text-right font-mono text-xs">{a.usage.calls ?? 0}</td>
                      <td className="num px-2 py-2 text-right font-mono text-xs">{a.usage.cost_usd ? `$${Number(a.usage.cost_usd).toFixed(3)}` : "—"}</td>
                      <td className="num px-2 py-2 text-right font-mono text-xs">{a.recoveries}</td>
                      <td className="num px-3.5 py-2 text-right font-mono text-xs text-ink-3">{num(a.wall_s, 1)}s</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )
        }
      </QueryState>
    </Panel>
  );
}

// ------------------------------------------------------------------ comparison
const COMPARE_ROWS: { key: keyof CollectorStats; label: string; fmt: (v: number) => string; better?: "high" | "low" }[] = [
  { key: "attempts", label: "Attempts", fmt: (v) => String(v) },
  { key: "successes", label: "Successes", fmt: (v) => String(v) },
  { key: "success_rate", label: "Success rate", fmt: (v) => pct(v), better: "high" },
  { key: "recoveries", label: "Recoveries", fmt: (v) => String(v) },
  { key: "mean_sim_seconds", label: "Mean sim time", fmt: (v) => `${v.toFixed(1)}s`, better: "low" },
  { key: "mean_wall_s", label: "Mean wall time", fmt: (v) => `${v.toFixed(1)}s`, better: "low" },
  { key: "model_calls", label: "Model calls", fmt: (v) => String(v) },
  { key: "cost_usd", label: "Model cost", fmt: (v) => `$${v.toFixed(2)}` },
  { key: "cells_covered", label: "Cells covered", fmt: (v) => `${v} / 9`, better: "high" },
];

export function CollectorComparisonCard({ pid }: { pid: string }) {
  const q = useCollectorComparison(pid);
  return (
    <Panel title="Agent vs oracle" meta="measured · this project" bodyClassName="p-0">
      <QueryState q={q} skeleton={<div className="p-3.5"><SkeletonRows rows={3} /></div>}>
        {(c) => (
          <table className="w-full text-left text-sm">
            <thead className="text-2xs uppercase tracking-[0.05em] text-ink-3">
              <tr className="border-b border-line-soft">
                <th className="px-3.5 py-2 font-medium" />
                {(["agent", "oracle"] as const).map((side) => (
                  <th key={side} className="px-3.5 py-2 text-right font-medium">
                    <span className="inline-flex items-center gap-1">
                      {side === "agent" ? <Bot className="h-3 w-3" /> : <Cog className="h-3 w-3" />}
                      {side === "agent" ? "Agent" : "Oracle"}
                    </span>
                    {!(side === "agent" ? c.agent ?? c.astra : c[side]) && <div className="mt-0.5 font-sans text-[10px] normal-case tracking-normal text-ink-4">not measured</div>}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {COMPARE_ROWS.map((r) => (
                <tr key={r.key} className="border-b border-line-soft last:border-0">
                  <td className="px-3.5 py-1.5 text-xs text-ink-3">{r.label}</td>
                  {(["agent", "oracle"] as const).map((side) => {
                    const s = side === "agent" ? c.agent ?? c.astra : c[side];
                    return (
                      <td key={side} className="num px-3.5 py-1.5 text-right font-mono text-xs text-ink">
                        {s ? r.fmt(Number(s[r.key] ?? 0)) : <span className="text-ink-4">—</span>}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </QueryState>
    </Panel>
  );
}
