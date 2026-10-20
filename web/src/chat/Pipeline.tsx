import { useMemo, type ReactNode } from "react";
import { Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { ChevronDown, Play } from "lucide-react";
import { isJobActive, useJobs } from "../api/events";
import { useAutopilot, useEpisodes, useEvaluations, useHealth, usePlannerBudget, useTraining } from "../api/hooks";
import { isImported, type AutopilotRun, type EvaluationWithBranches, type Iteration } from "../api/types";
import { cn, pct } from "../lib/format";
import { failureCause } from "../lib/trainingFailure";
import { IsoPlate } from "../sketch";
import { InfoTip } from "../components/ui";
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
    const acceptedEps = (episodes.data ?? []).filter((e) => e.status === "accepted");
    const accepted = acceptedEps.length;
    const importedN = acceptedEps.filter(isImported).length;
    // only simulator work counts as "Simulate": imported human demos were recorded on a real robot
    const simulated = (episodes.data ?? []).some((e) => !isImported(e)) || (evals.data ?? []).length > 0;
    const evs = evals.data ?? [];
    const tjs = training.data ?? [];
    const proposed = tjs.filter((t) => t.status === "proposed").length;
    const trainRunning = tjs.some((t) => t.status === "running" || t.status === "queued");
    const trained = tjs.some((t) => t.status === "succeeded" || t.checkpoint_ids.length > 0);
    const evalRunning = has(/eval/) || evs.some((e) => e.branches.some((b) => b.status === "running" || b.status === "queued"));
    const learnedEval = evs.some(learned);
    // the newest training job, when it failed and nothing newer replaced it
    const newest = tjs.length ? tjs.reduce((a, b) => (a.created_at > b.created_at ? a : b)) : undefined;
    const lastFailed = newest?.status === "failed" ? newest : undefined;

    if (run && (run.status === "running" || run.status === "waiting_approval" || run.state === "DONE" || run.status === "failed")) {
      const idx = run.state === "DONE" ? 5 : STATE_TO_STAGE[run.state] ?? 0;
      return NAMES.map((name, i) => {
        if (i === 5) return { name, state: run.iteration > 1 ? "succeeded" : "waiting", note: run.iteration ? `iteration ${run.iteration}` : undefined };
        if (i < idx) return { name, state: "succeeded" };
        if (i > idx) return { name, state: "waiting" };
        // a failed step says so (#348); the raw error stays in the backend logs (#328)
        const st: StageState = run.status === "failed" ? "failed" : run.status === "waiting_approval" ? "attention" : run.state === "DONE" ? "succeeded" : "running";
        const failedTj = run.status === "failed" && name === "Train" ? lastFailed : undefined;
        const note = run.status === "waiting_approval" ? "needs your OK" : run.status === "failed" ? (failedTj ? failureCause(failedTj)!.short : "didn't finish") : undefined;
        return { name, state: st, note };
      });
    }
    // no active autopilot: derive from what exists, plus live jobs
    return [
      {
        name: "Simulate",
        state: has(/record|collect|eval|sim/) ? "running" : simulated ? "succeeded" : "waiting",
        note: !simulated && importedN ? "no sim runs yet" : undefined,
      },
      {
        name: "Demonstrate",
        state: has(/record|collect|import/) ? "running" : accepted > 0 ? "succeeded" : "waiting",
        note: accepted ? `${accepted} demos${importedN ? ` (${importedN} imported)` : ""}` : undefined,
      },
      {
        name: "Train",
        state: trainRunning || has(/dataset/) ? "running" : trained ? "succeeded" : proposed ? "attention" : lastFailed ? "failed" : "waiting",
        note: proposed && !trained ? `${proposed} to approve` : lastFailed && !trained && !trainRunning ? failureCause(lastFailed)!.short : undefined,
      },
      {
        name: "Evaluate",
        state: evalRunning ? "running" : learnedEval ? "succeeded" : "waiting",
        note: !learnedEval && evs.length ? "oracle only" : undefined,
      },
      { name: "Improve", state: iterations.some((i) => i.next) ? "succeeded" : "waiting" },
      (() => {
        const evaluated = iterations.filter((i) => iterationSuccess(i)).length;
        const pending = iterations.length - evaluated;
        return {
          name: "Repeat",
          state: evaluated > 1 ? "succeeded" : "waiting",
          note: evaluated ? `${evaluated} evaluated${pending ? ` · ${pending} pending` : ""}` : iterations.length ? `${iterations.length} planned` : undefined,
        } as Stage;
      })(),
    ];
  }, [jobs, pid, run, episodes.data, evals.data, training.data, iterations]);
}

/** Each step is a small isometric plate: the live step lifts off its footprint, outlined in the accent. */
function StagePlate({ state }: { state: StageState }) {
  const lifted = state === "running" || state === "attention" || state === "failed";
  return (
    <span className={cn("flex h-[28px] w-[26px] shrink-0 items-end justify-center", state === "waiting" && "opacity-45")} aria-hidden>
      <IsoPlate w={1.25} d={1.25} h={state === "succeeded" ? 0.32 : 0.16} scale={9} lift={lifted ? 0.85 : 0} accent={state === "running"} />
    </span>
  );
}

const STATE_TEXT: Record<StageState, string> = {
  waiting: "text-ink-3",
  running: "text-servo",
  succeeded: "text-ink",
  failed: "text-fault",
  attention: "text-caution",
};

export function Pipeline({ pid, open, onToggle, sel }: { pid: string; open: boolean; onToggle: () => void; sel: ViewerSelection }) {
  const ap = useAutopilot(pid);
  const its = useIterations(pid);
  const iterations = its.data ?? [];
  const stages = useStages(pid, ap.data, iterations);

  return (
    <section className="border-t border-line bg-bench">
      <div className="flex items-center gap-5 px-4 py-2 sm:px-5">
        <button type="button" onClick={onToggle} className="flex shrink-0 items-center gap-2 font-mono text-[10.5px] font-medium uppercase tracking-[0.16em] text-ink hover:text-ink-2" aria-expanded={open}>
          <ChevronDown className={cn("h-3.5 w-3.5 transition-transform", !open && "-rotate-90")} />
          <span className="hidden sm:inline">Pipeline</span>
        </button>
        <ol className="relative flex min-w-0 flex-1 items-end gap-5 overflow-x-auto pb-1.5 scroll-thin">
          <span aria-hidden className="pointer-events-none absolute inset-x-0 bottom-0 h-px bg-[var(--line)]" />
          {stages.map((s, i) => (
            <li key={s.name} className="relative flex shrink-0 items-end gap-2" title={s.note}>
              <StagePlate state={s.state} />
              <span className={cn("pb-0.5 font-mono text-[10.5px] uppercase tracking-[0.16em]", STATE_TEXT[s.state])}>
                <span className={cn("mr-1.5", s.state === "waiting" ? "text-ink-4" : "")}>{String(i + 1).padStart(2, "0")}</span>
                {s.name}
                {s.note && <span className="ml-1.5 normal-case tracking-[0.04em] text-ink-3">· {s.note}</span>}
              </span>
              {(s.state === "running" || s.state === "attention") && (
                <span aria-hidden className={cn("absolute -bottom-1.5 left-0 right-0 h-px", s.state === "running" ? "bg-servo" : "bg-caution")} />
              )}
            </li>
          ))}
        </ol>
        {ap.data && <span className="hidden shrink-0 font-mono text-[10px] uppercase tracking-[0.14em] text-ink-3 xl:inline">autopilot · {ap.data.status.replace(/_/g, " ")}</span>}
      </div>
      {open && (
        <div className="grid grid-cols-[minmax(0,1.25fr)_minmax(0,1fr)_minmax(0,1.1fr)] gap-6 border-t border-line px-5 py-4">
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
      <div className="truncate font-mono text-[10px] uppercase tracking-[0.14em] text-ink-3">{label}</div>
      <div className="num kn-display mt-1 truncate text-[24px] text-ink">{value}</div>
      {sub && <div className="mt-0.5 truncate text-[11px] text-ink-3">{sub}</div>}
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
  // simulated and recorded (imported) time are different things: report them separately
  const meanLen = (xs: typeof accepted) => (xs.length ? xs.reduce((s, e) => s + e.num_frames / (e.fps || 30), 0) / xs.length : null);
  const simLen = meanLen(accepted.filter((e) => !isImported(e)));
  const recLen = meanLen(accepted.filter(isImported));
  const loss = [...iterations].reverse().find((i) => i.training?.loss_curve?.length)?.training?.loss_curve ?? null;
  // a version counts only once it has a checkpoint; planned iterations are "(pending)"
  const trainedIt = [...iterations].reverse().find((i) => i.checkpoint_id);
  const version = trainedIt?.policy_version ?? null;
  const pendingIt = [...iterations].reverse().find((i) => !i.checkpoint_id);
  const iteration = run?.iteration || iterations.length || null;
  const replay = [...iterations].reverse().find((i) => i.replay?.joint_mae_deg != null)?.replay ?? null;

  return (
    <div className="grid grid-cols-4 gap-x-5 gap-y-4">
      <Tile
        label="Policy success"
        value={policy ? pct(policy.rate) : "—"}
        sub={policy && ci ? `${policy.k}/${policy.n} · 95% CI ${pct(ci[0])}–${pct(ci[1])}` : oracle.n ? `oracle: ${pct(oracle.k / oracle.n)}` : replay ? `held-out replay: ${replay.joint_mae_deg!.toFixed(2)}° joint error` : version ? "not evaluated" : "no learned policy yet"}
      />
      <Tile label="Demos accepted" value={episodes.data ? accepted.length : "—"} />
      <Tile label="Iteration" value={iteration ?? "—"} sub={pendingIt ? pendingIt.policy_version : version ? `policy ${version}` : "none yet"} />
      <Tile label="Eval episodes" value={evals.data ? evalN : "—"} sub={`${(evals.data ?? []).length} evaluations`} />
      <Tile label="Training loss" value={loss ? loss[loss.length - 1][1].toFixed(3) : "—"} sub={loss ? `step ${loss[loss.length - 1][0].toLocaleString()}` : "not trained"}>
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
      <Tile
        label="Demo length"
        value={simLen !== null ? `${simLen.toFixed(1)}s` : recLen !== null ? `${recLen.toFixed(1)}s` : "—"}
        sub={simLen !== null ? (recLen !== null ? `simulated · imported ${recLen.toFixed(1)}s recorded` : "simulated time") : recLen !== null ? "recorded time (imported)" : "no demos yet"}
      />
      <Tile label="Policy version" value={version ?? "—"} sub={trainedIt?.checkpoint_id ?? "not trained"} />
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
        <div className="kn-label kn-label-rule"><span>Success across iterations</span></div>
        <p className="mt-2.5 text-sm text-ink-2">
          {iterations.some((i) => i.checkpoint_id)
            ? `${iterations.filter((i) => i.checkpoint_id).length} trained · no sim evaluation`
            : iterations.length ? `${iterations.length} planned · no policy trained` : "No policy trained"}
          {/futurama/i.test(trainingProvider) ? " · needs Futurama GPU (GT VPN)" : health.data?.training?.fallback ? " · no GPU connected" : ""}
        </p>
        {proposed > 0 && <p className="mt-1 text-xs text-caution">{proposed} training job{proposed === 1 ? "" : "s"} to approve in chat</p>}
        <Replays iterations={iterations} sel={sel} />
        {failures.length > 0 && <FailureBars failures={failures} maxF={maxF} empty="No failures." />}
      </div>
    );
  }
  return (
    <div className="min-w-0">
      <div className="flex items-baseline justify-between">
        <div className="kn-label kn-label-rule"><span>Success across iterations</span></div>
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
              className={cn("rounded-full border px-2 py-0.5 font-mono text-[10.5px]", sel.evaluationId && sel.evaluationId === it.evaluation?.id ? "border-servo text-servo" : "border-transparent text-ink-3 hover:border-line-strong hover:text-ink")}
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
            <XAxis dataKey="it" tickLine={false} axisLine={false} tick={{ fill: "rgb(var(--ink-3))", fontSize: 10, fontFamily: "IBM Plex Mono" }} />
            <YAxis domain={[0, 100]} ticks={[0, 50, 100]} tickLine={false} axisLine={false} tick={{ fill: "rgb(var(--ink-3))", fontSize: 10, fontFamily: "IBM Plex Mono" }} />
            <Tooltip
              cursor={{ stroke: "var(--line-strong)" }}
              content={({ active, payload }) =>
                active && payload?.length ? (
                  <div className="rounded-sm border border-line-strong bg-plate px-2 py-1 font-mono text-[11px] text-ink">
                    {String(payload[0].payload.it)}: {Number(payload[0].value).toFixed(0)}%
                  </div>
                ) : null
              }
            />
            <Line dataKey="rate" stroke="var(--series-1)" strokeWidth={1.25} dot={{ r: 3, fill: "rgb(var(--plate))", stroke: "var(--series-1)", strokeWidth: 1.25 }} isAnimationActive={false} connectNulls />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <FailureBars failures={failures} maxF={maxF} empty="No failures." />
    </div>
  );
}

/** Open-loop replays of each trained version on held-out recorded demos: joint error, not success. */
function Replays({ iterations, sel }: { iterations: Iteration[]; sel: ViewerSelection }) {
  const rows = iterations.filter((i) => i.replay);
  if (!rows.length) return null;
  return (
    <div className="mt-2.5 space-y-1">
      <div className="flex items-center gap-1.5 font-mono text-[10px] uppercase tracking-[0.14em] text-ink-3">
        Held-out replays
        <InfoTip>{`${rows[0].replay!.note}. The policy sees each recorded frame and its joint targets are compared with the human's; the arm does not move.`}</InfoTip>
      </div>
      {rows.map((it) => {
        const r = it.replay!;
        return (
          <button
            key={it.iteration}
            type="button"
            onClick={() => {
              sel.setEvaluationId(null);
              sel.setFilter("replay");
              sel.setSelectedId(null);
            }}
            className="block w-full truncate text-left text-xs text-ink-2 hover:text-ink"
            title="Show these replays in the viewer"
          >
            <span className="font-medium text-ink">{it.policy_version}</span>
            {r.episodes ? ` · ${r.episodes} episodes · ${r.joint_mae_deg?.toFixed(2) ?? "—"}° mean joint error` : ""}
            {r.live_sessions ? ` · ${r.live_sessions} live GPU run${r.live_sessions === 1 ? "" : "s"}${r.inference_ms != null ? `, ${r.inference_ms.toFixed(0)} ms/frame` : ""}` : ""}
          </button>
        );
      })}
    </div>
  );
}

function FailureBars({ failures, maxF, empty }: { failures: [string, number][]; maxF: number; empty: string }) {
  return (
    <div className="mt-2">
      <div className="mb-1.5 font-mono text-[10px] uppercase tracking-[0.14em] text-ink-3">Failure distribution</div>
      {failures.length === 0 ? (
        <p className="text-xs text-ink-4">{empty}</p>
      ) : (
        <ul className="space-y-1">
          {failures.map(([k, v]) => (
            <li key={k} className="grid grid-cols-[110px_1fr_24px] items-center gap-2 text-[11px]">
              <span className="truncate capitalize text-ink-2">{k.replace(/_/g, " ")}</span>
              <span className="h-1 overflow-hidden bg-[var(--line)]">
                <span className="block h-full bg-fault/75" style={{ width: `${(v / maxF) * 100}%` }} />
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
        <div className="kn-label kn-label-rule"><span>Failure analysis</span></div>
        <p className="mt-2.5 text-sm leading-relaxed text-ink">
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
          <button type="button" onClick={() => watch(it.evaluation!.id)} className="sk-pill sk-pill-secondary inline-flex h-7 items-center gap-1.5 px-3 text-xs font-medium">
            <Play className="h-3 w-3" /> Watch failures
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
        <div className="kn-label kn-label-rule"><span>Failure analysis</span></div>
        <p className="mt-2.5 text-sm text-ink">
          {fallbackEval.run.policy_label || "The learned policy"} failed {worst.result!.n - worst.result!.successes}/{worst.result!.n} trials in {worst.spec.name.replace(/_/g, " ")}.
        </p>
        <button type="button" onClick={() => watch(fallbackEval.run.id)} className="sk-pill sk-pill-secondary mt-2 inline-flex h-7 items-center gap-1.5 px-3 text-xs font-medium">
          <Play className="h-3 w-3" /> Watch failures
        </button>
      </div>
    );
  }
  return (
    <div className="min-w-0">
      <div className="kn-label kn-label-rule"><span>Failure analysis</span></div>
      <p className="mt-2.5 flex items-center gap-1.5 text-sm text-ink-2">
        No learned-policy failures yet
        <InfoTip>When a trained policy fails a test condition, Cortex explains it here, generates targeted demonstrations for that condition, and trains the next version.</InfoTip>
      </p>
    </div>
  );
}
