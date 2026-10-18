import { useMemo, type ReactNode } from "react";
import { Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { AlertTriangle, Check, ChevronDown, CircleDashed, Play, X } from "lucide-react";
import { isJobActive, useJobs } from "../api/events";
import { useAutopilot, useEpisodes, useEvaluations, useHealth, usePlannerBudget, useTraining } from "../api/hooks";
import type { AutopilotRun, EvaluationWithBranches, Iteration } from "../api/types";
import { cn, pct } from "../lib/format";
import { iterationSuccess, useIterations, wilson } from "./rollouts";
import type { ViewerSelection } from "./Viewer";

type StageState = "waiting" | "running" | "succeeded" | "failed" | "attention";
interface Stage {
  name: string;
  state: StageState;
  note?: string;
}

const STATE_TO_STAGE: Record<string, number> = {
  TASK_REQUESTED: 0,
  SCENARIO_PLANNED: 0,
  AGENT_COLLECTING: 1,
  EPISODES_VALIDATED: 1,
  DATASET_VERSIONED: 2,
  TRAINING_REQUESTED: 2,
  POLICY_TRAINED: 2,
  POLICY_EVALUATED: 3,
  FAILURES_ANALYZED: 4,
  NEXT_EXPERIMENT_SELECTED: 4,
};
const NAMES = ["Simulate", "Demonstrate", "Train", "Evaluate", "Improve", "Repeat"];

const learned = (e: EvaluationWithBranches) => e.run.policy.kind === "checkpoint";

function evalTotals(e: EvaluationWithBranches) {
  let k = 0;
  let n = 0;
  for (const b of e.branches) {
    k += b.result?.successes ?? 0;
    n += b.result?.n ?? 0;
  }
  return { k, n };
}

function useStages(pid: string, run: AutopilotRun | null | undefined, iterations: Iteration[]) {
  const jobs = useJobs();
  const episodes = useEpisodes(pid);
  const evals = useEvaluations(pid);
  const training = useTraining(pid);
  return useMemo<Stage[]>(() => {
    const active = Object.values(jobs).filter((j) => j.project_id === pid && isJobActive(j));
    const has = (re: RegExp) => active.some((j) => re.test(j.kind));
    const accepted = (episodes.data ?? []).filter((e) => e.status === "accepted").length;
    const evs = evals.data ?? [];
    const tjs = training.data ?? [];
    const proposed = tjs.filter((t) => t.status === "proposed").length;
    const trainRunning = tjs.some((t) => t.status === "running" || t.status === "queued");
    const trained = tjs.some((t) => t.status === "succeeded" || t.checkpoint_ids.length > 0);
    const evalRunning = has(/eval/) || evs.some((e) => e.branches.some((b) => b.status === "running" || b.status === "queued"));
    const learnedEval = evs.some(learned);

    if (run && (run.status === "running" || run.status === "waiting_approval" || run.state === "DONE" || run.status === "failed")) {
      const idx = run.state === "DONE" ? 5 : STATE_TO_STAGE[run.state] ?? 0;
      return NAMES.map((name, i) => {
        if (i === 5) return { name, state: run.iteration > 1 ? "succeeded" : "waiting", note: run.iteration ? `iteration ${run.iteration}` : undefined };
        if (i < idx) return { name, state: "succeeded" };
        if (i > idx) return { name, state: "waiting" };
        const st: StageState = run.status === "failed" ? "failed" : run.status === "waiting_approval" ? "attention" : run.state === "DONE" ? "succeeded" : "running";
        return { name, state: st, note: run.status === "waiting_approval" ? "needs your OK" : undefined };
      });
    }
    // no active autopilot: derive from what exists, plus live jobs
    return [
      { name: "Simulate", state: has(/record|collect|eval|sim/) ? "running" : (episodes.data?.length ?? 0) > 0 || evs.length > 0 ? "succeeded" : "waiting" },
      { name: "Demonstrate", state: has(/record|collect/) ? "running" : accepted > 0 ? "succeeded" : "waiting", note: accepted ? `${accepted} demos` : undefined },
      {
        name: "Train",
        state: trainRunning || has(/dataset/) ? "running" : trained ? "succeeded" : proposed ? "attention" : "waiting",
        note: proposed && !trained ? `${proposed} awaiting approval` : undefined,
      },
      {
        name: "Evaluate",
        state: evalRunning ? "running" : learnedEval ? "succeeded" : "waiting",
        note: !learnedEval && evs.length ? "oracle reference only" : undefined,
      },
      { name: "Improve", state: iterations.some((i) => i.next) ? "succeeded" : "waiting" },
      (() => {
        const evaluated = iterations.filter((i) => iterationSuccess(i)).length;
        const pending = iterations.length - evaluated;
        return {
          name: "Repeat",
          state: evaluated > 1 ? "succeeded" : "waiting",
          note: evaluated ? `${evaluated} evaluated${pending ? ` · ${pending} pending` : ""}` : iterations.length ? `${iterations.length} planned, none trained` : undefined,
        } as Stage;
      })(),
    ];
  }, [jobs, pid, run, episodes.data, evals.data, training.data, iterations]);
}

function StageIcon({ state }: { state: StageState }) {
  if (state === "running") return <CircleDashed className="h-4 w-4 animate-spin text-servo" />;
  if (state === "succeeded")
    return (
      <span className="flex h-4 w-4 items-center justify-center rounded-full bg-pass text-white">
        <Check className="h-2.5 w-2.5" />
      </span>
    );
  if (state === "failed")
    return (
      <span className="flex h-4 w-4 items-center justify-center rounded-full bg-fault text-white">
        <X className="h-2.5 w-2.5" />
      </span>
    );
  if (state === "attention") return <AlertTriangle className="h-4 w-4 text-caution" />;
  return <span className="h-4 w-4 rounded-full shadow-[inset_0_0_0_1.5px_rgb(var(--ink-4))]" />;
}

export function Pipeline({ pid, open, onToggle, sel }: { pid: string; open: boolean; onToggle: () => void; sel: ViewerSelection }) {
  const ap = useAutopilot(pid);
  const its = useIterations(pid);
  const iterations = its.data ?? [];
  const stages = useStages(pid, ap.data, iterations);

  return (
    <section className="border-t border-line bg-bench">
      <div className="flex items-center gap-3 px-4 py-2">
        <button type="button" onClick={onToggle} className="flex items-center gap-1.5 text-xs font-semibold text-ink-2 hover:text-ink" aria-expanded={open}>
          <ChevronDown className={cn("h-3.5 w-3.5 transition-transform", !open && "-rotate-90")} />
          Autonomous learning pipeline
        </button>
        <ol className="flex min-w-0 flex-1 items-center gap-1 overflow-x-auto scroll-thin">
          {stages.map((s, i) => (
            <li key={s.name} className="flex shrink-0 items-center">
              <span
                className={cn(
                  "inline-flex h-7 items-center gap-1.5 rounded-full px-2.5 text-xs",
                  s.state === "running" ? "bg-servo/10 text-ink" : s.state === "attention" ? "bg-caution/10 text-ink" : s.state === "failed" ? "bg-fault/10 text-ink" : "text-ink-2",
                )}
              >
                <StageIcon state={s.state} />
                <span className="font-medium">{s.name}</span>
                {s.note && <span className="text-ink-3">· {s.note}</span>}
              </span>
              {i < stages.length - 1 && <span className="mx-1 h-px w-4 bg-[var(--line-strong)]" />}
            </li>
          ))}
        </ol>
        {ap.data && <span className="hidden shrink-0 text-2xs text-ink-3 xl:inline">autopilot {ap.data.status.replace(/_/g, " ")}</span>}
      </div>
      {open && (
        <div className="grid grid-cols-[minmax(0,1.25fr)_minmax(0,1fr)_minmax(0,1.1fr)] gap-4 border-t border-line-soft px-4 py-3">
          <Metrics pid={pid} iterations={iterations} run={ap.data ?? null} />
          <Progress iterations={iterations} pid={pid} sel={sel} />
          <FailureAnalysis pid={pid} iterations={iterations} sel={sel} />
        </div>
      )}
    </section>
  );
}

// ------------------------------------------------------------------ metrics
function Tile({ label, value, sub, children }: { label: string; value: ReactNode; sub?: ReactNode; children?: ReactNode }) {
  return (
    <div className="min-w-0">
      <div className="truncate text-[11px] text-ink-3">{label}</div>
      <div className="num mt-0.5 truncate text-lg font-semibold leading-tight tracking-[-0.01em] text-ink">{value}</div>
      {sub && <div className="truncate text-[11px] text-ink-3">{sub}</div>}
      {children}
    </div>
  );
}

function Metrics({ pid, iterations, run }: { pid: string; iterations: Iteration[]; run: AutopilotRun | null }) {
  const episodes = useEpisodes(pid);
  const evals = useEvaluations(pid);
  const budget = usePlannerBudget(pid);
  const latestIt = [...iterations].reverse().find((i) => iterationSuccess(i));
  const latestLearned = [...(evals.data ?? [])].filter(learned).sort((a, b) => (a.run.created_at < b.run.created_at ? 1 : -1))[0];
  const policy = latestIt ? iterationSuccess(latestIt) : latestLearned ? (({ k, n }) => (n ? { k, n, rate: k / n } : null))(evalTotals(latestLearned)) : null;
  const oracle = (evals.data ?? []).filter((e) => !learned(e)).map(evalTotals).reduce((a, b) => ({ k: a.k + b.k, n: a.n + b.n }), { k: 0, n: 0 });
  const ci = policy ? wilson(policy.k, policy.n) : null;
  const accepted = (episodes.data ?? []).filter((e) => e.status === "accepted");
  const evalN = (evals.data ?? []).map(evalTotals).reduce((s, x) => s + x.n, 0);
  const avgLen = accepted.length ? accepted.reduce((s, e) => s + e.num_frames / (e.fps || 30), 0) / accepted.length : null;
  const loss = [...iterations].reverse().find((i) => i.training?.loss_curve?.length)?.training?.loss_curve ?? null;
  // a version counts only once it has a checkpoint; planned iterations are "(pending)"
  const trainedIt = [...iterations].reverse().find((i) => i.checkpoint_id);
  const version = trainedIt?.policy_version ?? null;
  const pendingIt = [...iterations].reverse().find((i) => !i.checkpoint_id);
  const iteration = run?.iteration || iterations.length || null;

  return (
    <div className="grid grid-cols-4 gap-x-4 gap-y-3">
      <Tile
        label="Policy success"
        value={policy ? pct(policy.rate) : "—"}
        sub={policy && ci ? `${policy.k}/${policy.n} · 95% CI ${pct(ci[0])}–${pct(ci[1])}` : oracle.n ? `oracle: ${pct(oracle.k / oracle.n)}` : "no learned policy yet"}
      />
      <Tile label="Demonstrations" value={episodes.data ? accepted.length : "—"} sub="accepted" />
      <Tile label="Iteration" value={iteration ?? "—"} sub={pendingIt ? pendingIt.policy_version : version ? `policy ${version}` : "no iterations yet"} />
      <Tile label="Eval episodes" value={evals.data ? evalN : "—"} sub={`${(evals.data ?? []).length} evaluations`} />
      <Tile label="Training loss" value={loss ? loss[loss.length - 1][1].toFixed(3) : "—"} sub={loss ? `step ${loss[loss.length - 1][0].toLocaleString()}` : "no training run yet"}>
        {loss && loss.length > 1 && (
          <div className="mt-1 h-6">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={loss.map(([s, l]) => ({ s, l }))}>
                <Line dataKey="l" stroke="var(--series-1)" strokeWidth={1.5} dot={false} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}
      </Tile>
      <Tile label="Avg demo length" value={avgLen ? `${avgLen.toFixed(1)}s` : "—"} sub="simulated time" />
      <Tile label="Policy version" value={version ?? "—"} sub={trainedIt?.checkpoint_id ?? "none trained yet"} />
      <Tile label="Model spend" value={budget.data ? `$${budget.data.spent_usd.toFixed(2)}` : "—"} sub={budget.data ? `of $${budget.data.budget_usd.toFixed(0)} approved` : undefined} />
    </div>
  );
}

// ------------------------------------------------------------------ progress across iterations
function Progress({ iterations, pid, sel }: { iterations: Iteration[]; pid: string; sel: ViewerSelection }) {
  const health = useHealth();
  const training = useTraining(pid);
  const evals = useEvaluations(pid);
  const points = iterations.map((it) => ({ it: it.policy_version || `it${it.iteration}`, rate: (iterationSuccess(it)?.rate ?? NaN) * 100, evalId: it.evaluation?.id ?? null }));
  const failures = useMemo(() => {
    const m = new Map<string, number>();
    for (const it of iterations) for (const b of it.evaluation?.branches ?? []) for (const [k, v] of Object.entries(b.failure_modes ?? {})) m.set(k, (m.get(k) ?? 0) + v);
    if (!iterations.length)
      for (const e of evals.data ?? []) for (const b of e.branches) for (const ep of b.result?.episodes ?? []) if (!ep.success) m.set(ep.failure_mode ?? "unlabelled", (m.get(ep.failure_mode ?? "unlabelled") ?? 0) + 1);
    return [...m.entries()].sort((a, b) => b[1] - a[1]).slice(0, 4);
  }, [iterations, evals.data]);
  const maxF = Math.max(1, ...failures.map(([, v]) => v));

  if (!points.some((p) => Number.isFinite(p.rate))) {
    const trainingProvider = health.data?.training?.provider ?? "";
    const proposed = (training.data ?? []).filter((t) => t.status === "proposed").length;
    return (
      <div className="min-w-0">
        <div className="text-xs font-medium text-ink-2">Success across iterations</div>
        <p className="mt-1.5 text-sm text-ink-2">
          {iterations.length ? `${iterations.length} iteration${iterations.length === 1 ? "" : "s"} planned, no policy trained yet` : "No policy trained yet"}
          {/futurama/i.test(trainingProvider) ? ". Training needs the Futurama GPU (GT VPN)." : health.data?.training?.fallback ? ". No GPU training backend is connected." : "."}
        </p>
        {proposed > 0 && <p className="mt-1 text-xs text-caution">{proposed} training job{proposed === 1 ? "" : "s"} awaiting your approval in the chat.</p>}
        <FailureBars failures={failures} maxF={maxF} empty="No failed rollouts recorded." />
      </div>
    );
  }
  return (
    <div className="min-w-0">
      <div className="flex items-baseline justify-between">
        <div className="text-xs font-medium text-ink-2">Success across iterations</div>
        <div className="flex gap-1">
          {iterations.map((it) => (
            <button
              key={it.iteration}
              type="button"
              disabled={!it.evaluation}
              onClick={() => {
                sel.setEvaluationId(it.evaluation?.id ?? null);
                sel.setFilter("eval");
              }}
              className={cn("rounded px-1.5 py-0.5 font-mono text-[10.5px]", sel.evaluationId && sel.evaluationId === it.evaluation?.id ? "bg-servo/15 text-servo" : "text-ink-3 hover:bg-rail hover:text-ink")}
              title="Load this iteration's rollouts into the viewer"
            >
              {it.policy_version || `it${it.iteration}`}
            </button>
          ))}
        </div>
      </div>
      <div className="mt-1 h-[88px]">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={points} margin={{ top: 6, right: 6, bottom: 0, left: -28 }}>
            <XAxis dataKey="it" tickLine={false} axisLine={false} tick={{ fill: "rgb(var(--ink-3))", fontSize: 10 }} />
            <YAxis domain={[0, 100]} ticks={[0, 50, 100]} tickLine={false} axisLine={false} tick={{ fill: "rgb(var(--ink-4))", fontSize: 10 }} />
            <Tooltip
              cursor={{ stroke: "var(--line-strong)" }}
              content={({ active, payload }) =>
                active && payload?.length ? (
                  <div className="rounded-md bg-rail px-2 py-1 text-xs text-ink shadow-[0_0_0_1px_var(--line-strong)]">
                    {String(payload[0].payload.it)}: {Number(payload[0].value).toFixed(0)}%
                  </div>
                ) : null
              }
            />
            <Line dataKey="rate" stroke="var(--series-1)" strokeWidth={2} dot={{ r: 3.5, fill: "var(--series-1)", strokeWidth: 0 }} isAnimationActive={false} connectNulls />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <FailureBars failures={failures} maxF={maxF} empty="No failures in evaluated iterations." />
    </div>
  );
}

function FailureBars({ failures, maxF, empty }: { failures: [string, number][]; maxF: number; empty: string }) {
  return (
    <div className="mt-2">
      <div className="mb-1 text-[11px] text-ink-3">Failure distribution</div>
      {failures.length === 0 ? (
        <p className="text-xs text-ink-4">{empty}</p>
      ) : (
        <ul className="space-y-1">
          {failures.map(([k, v]) => (
            <li key={k} className="grid grid-cols-[110px_1fr_24px] items-center gap-2 text-[11px]">
              <span className="truncate capitalize text-ink-2">{k.replace(/_/g, " ")}</span>
              <span className="h-1.5 overflow-hidden rounded-full bg-rail">
                <span className="block h-full rounded-full bg-fault/75" style={{ width: `${(v / maxF) * 100}%` }} />
              </span>
              <span className="num text-right font-mono text-ink-3">{v}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

// ------------------------------------------------------------------ failure analysis
function FailureAnalysis({ pid, iterations, sel }: { pid: string; iterations: Iteration[]; sel: ViewerSelection }) {
  const evals = useEvaluations(pid);
  // latest iteration whose learned policy failed somewhere
  const it = [...iterations].reverse().find((i) => (i.evaluation?.branches ?? []).some((b) => b.successes < b.n));
  const fallbackEval = !it ? [...(evals.data ?? [])].filter(learned).sort((a, b) => (a.run.created_at < b.run.created_at ? 1 : -1)).find((e) => e.branches.some((b) => (b.result?.successes ?? 0) < (b.result?.n ?? 0))) : undefined;

  const watch = (evaluationId: string | null) => {
    sel.setEvaluationId(evaluationId);
    sel.setFilter("failed");
    sel.setSelectedId(null);
  };

  if (it && it.evaluation) {
    const worst = [...it.evaluation.branches].sort((a, b) => a.success_rate - b.success_rate)[0];
    const failed = worst.n - worst.successes;
    const topMode = Object.entries(worst.failure_modes ?? {}).sort((a, b) => b[1] - a[1])[0]?.[0];
    const nextIt = iterations.find((x) => x.iteration === it.iteration + 1);
    return (
      <div className="min-w-0">
        <div className="text-xs font-medium text-ink-2">Failure analysis</div>
        <p className="mt-1.5 text-sm leading-relaxed text-ink">
          Policy <span className="font-semibold">{it.policy_version}</span> failed {failed}/{worst.n} trials in{" "}
          <span className="font-medium">{worst.name.replace(/_/g, " ")}</span>
          {topMode ? ` (mostly ${topMode.replace(/_/g, " ")})` : ""}
          {it.next ? (
            <>
              {" "}
              → <span className="text-servo">{it.next.reason}</span>
            </>
          ) : (
            "."
          )}
        </p>
        <div className="mt-2 flex flex-wrap items-center gap-2">
          <button type="button" onClick={() => watch(it.evaluation!.id)} className="inline-flex items-center gap-1.5 rounded-md bg-rail px-2.5 py-1 text-xs text-ink hover:bg-rail/70">
            <Play className="h-3 w-3" /> Watch the failed rollouts
          </button>
          {nextIt && (
            <span className="text-xs text-ink-3">
              Next: <span className="font-medium text-ink-2">{nextIt.policy_version}</span>
              {iterationSuccess(nextIt) ? ` · ${pct(iterationSuccess(nextIt)!.rate)} success` : " · not evaluated yet"}
            </span>
          )}
        </div>
      </div>
    );
  }
  if (fallbackEval) {
    const worst = [...fallbackEval.branches].filter((b) => b.result).sort((a, b) => a.result!.success_rate - b.result!.success_rate)[0];
    return (
      <div className="min-w-0">
        <div className="text-xs font-medium text-ink-2">Failure analysis</div>
        <p className="mt-1.5 text-sm text-ink">
          {fallbackEval.run.policy_label || "The learned policy"} failed {worst.result!.n - worst.result!.successes}/{worst.result!.n} trials in {worst.spec.name.replace(/_/g, " ")}.
        </p>
        <button type="button" onClick={() => watch(fallbackEval.run.id)} className="mt-2 inline-flex items-center gap-1.5 rounded-md bg-rail px-2.5 py-1 text-xs text-ink hover:bg-rail/70">
          <Play className="h-3 w-3" /> Watch the failed rollouts
        </button>
      </div>
    );
  }
  return (
    <div className="min-w-0">
      <div className="text-xs font-medium text-ink-2">Failure analysis</div>
      <p className="mt-1.5 text-sm text-ink-2">No learned-policy failures to analyze yet.</p>
      <p className="mt-1 text-xs text-ink-3">
        When a trained policy fails a test condition, Cortex explains it here, generates targeted demonstrations for that condition, and trains the next version.
      </p>
    </div>
  );
}
