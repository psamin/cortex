import { useEffect, useMemo, useState } from "react";
import { useParams } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Bar, BarChart, CartesianGrid, ErrorBar, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { FlaskConical, GitCompare, Play } from "lucide-react";
import { api, resolveMediaUrl } from "../../api/client";
import { useCheckpoints, useEvaluations } from "../../api/hooks";
import type { EpisodeResult, EvaluationWithBranches, ExperimentComparison, LaunchEvaluationRequest, SimulationBranch } from "../../api/types";
import { Badge, Button, Dialog, EmptyState, ErrorState, Field, InfoTip, Input, Panel, QueryState, Select, SkeletonRows, StatusBadge, InlineError } from "../../components/ui";
import { ConditionLabel } from "../../components/Workcell";
import { NextActionsPanel } from "../Plan";
import { RealTrialsPanel } from "./RealTrials";
import { ago, cn, evidenceValue, num, pct } from "../../lib/format";

const SERIES = ["var(--series-1)", "var(--series-2)", "var(--series-3)", "var(--series-4)"];
const MAX_COMPARE = 4;

export function policyLabel(run: EvaluationWithBranches["run"]) {
  if (run.policy_label) return run.policy_label;
  return run.policy.kind === "scripted_expert" ? "Scripted expert" : run.policy.checkpoint_id ?? "checkpoint";
}

export function EvaluationsTab() {
  const { pid = "" } = useParams();
  const q = useEvaluations(pid);
  const [open, setOpen] = useState(false);
  const [sel, setSel] = useState<string[]>([]);
  const evals = useMemo(() => [...(q.data ?? [])].sort((a, b) => (a.run.created_at < b.run.created_at ? 1 : -1)), [q.data]);
  const anyActive = evals.some((e) => e.branches.some((b) => b.status === "queued" || b.status === "running"));

  // keep polling while branches run, in case the event stream is down
  useEvaluationsPolling(pid, anyActive);

  const toggle = (id: string) =>
    setSel((s) => (s.includes(id) ? s.filter((x) => x !== id) : s.length >= MAX_COMPARE ? s : [...s, id]));

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between gap-3">
        <p className="flex items-center gap-1.5 text-xs text-ink-3">
          3 sim branches · 95% CI
          <InfoTip>Each evaluation runs three sim branches in parallel local worker processes: baseline, shifted object, and changed lighting. Success rates carry 95% confidence intervals; with small n they are wide.</InfoTip>
        </p>
        <Button variant="primary" size="sm" onClick={() => setOpen(true)}>
          <Play className="h-3.5 w-3.5" /> Evaluate
        </Button>
      </div>

      <QueryState q={q} skeleton={<SkeletonRows rows={4} />}>
        {() =>
          evals.length === 0 ? (
            <Panel>
              <EmptyState
                icon={<FlaskConical className="h-5 w-5" />}
                title="No evaluations yet"
                action={
                  <Button size="sm" onClick={() => setOpen(true)}>
                    Evaluate
                  </Button>
                }
              >
                A checkpoint, or the scripted expert (oracle ceiling).
              </EmptyState>
            </Panel>
          ) : (
            <>
              <Comparison pid={pid} evals={evals.filter((e) => sel.includes(e.run.id))} order={sel} />
              <div className="space-y-3">
                {evals.map((e) => (
                  <EvaluationCard
                    key={e.run.id}
                    e={e}
                    selected={sel.includes(e.run.id)}
                    seriesColor={sel.includes(e.run.id) ? SERIES[sel.indexOf(e.run.id)] : undefined}
                    canSelect={sel.length < MAX_COMPARE}
                    onToggle={() => toggle(e.run.id)}
                  />
                ))}
              </div>
            </>
          )
        }
      </QueryState>

      <RealTrialsPanel pid={pid} />
      <NextActionsPanel pid={pid} />
      <LaunchDialog pid={pid} open={open} onClose={() => setOpen(false)} />
    </div>
  );
}

function useEvaluationsPolling(pid: string, active: boolean) {
  const qc = useQueryClient();
  useEffect(() => {
    if (!active) return;
    const t = setInterval(() => qc.invalidateQueries({ queryKey: ["evaluations", pid] }), 3000);
    return () => clearInterval(t);
  }, [active, pid, qc]);
}

// ------------------------------------------------------------------ cards
function EvaluationCard({
  e,
  selected,
  seriesColor,
  canSelect,
  onToggle,
}: {
  e: EvaluationWithBranches;
  selected: boolean;
  seriesColor?: string;
  canSelect: boolean;
  onToggle: () => void;
}) {
  const oracle = e.run.policy.kind === "scripted_expert";
  const done = e.branches.length > 0 && e.branches.every((b) => b.result);
  return (
    <article className={cn("rounded-md bg-plate border border-line", selected && "border border-line-strong")}>
      <header className="flex flex-wrap items-center gap-2 border-b border-line-soft px-3.5 py-2.5">
        <label className={cn("flex items-center gap-2", !done && "opacity-50")} title={done ? "Include in comparison" : "Wait for all branches to finish"}>
          <input
            type="checkbox"
            checked={selected}
            disabled={!done || (!selected && !canSelect)}
            aria-label="Include in comparison"
            onChange={onToggle}
            className="h-3.5 w-3.5 accent-[rgb(var(--servo))]"
          />
          {seriesColor && <span className="h-2.5 w-2.5 rounded-sm" style={{ background: seriesColor }} />}
        </label>
        <span className="text-sm font-medium">{policyLabel(e.run)}</span>
        {oracle && !/oracle/i.test(e.run.policy_label) && (
          <Badge tone="hypothesis" title="The scripted expert generates the demos. It is a reference ceiling, not a learned policy.">
            oracle · not learned
          </Badge>
        )}
        <StatusBadge status={e.run.status} />
        {e.run.name && <span className="text-xs text-ink-2">{e.run.name}</span>}
        <span className="font-mono text-2xs text-ink-3">{e.run.id}</span>
        {e.run.dataset_version_id && <span className="text-2xs text-ink-3">data {e.run.dataset_version_id}</span>}
        <span className="ml-auto text-2xs text-ink-4">{ago(e.run.created_at)}</span>
      </header>
      <div className="grid grid-cols-1 gap-2.5 p-3 lg:grid-cols-3">
        {e.branches.length === 0 ? (
          <p className="text-xs text-ink-3">No branches yet.</p>
        ) : (
          e.branches.map((b) => <BranchCard key={b.id} b={b} />)
        )}
      </div>
    </article>
  );
}

function BranchCard({ b }: { b: SimulationBranch }) {
  const r = b.result;
  const live = b.status === "queued" || b.status === "running";
  const shifted = b.spec.cube_offset[0] !== 0 || b.spec.cube_offset[1] !== 0;
  return (
    <div className="rounded bg-well p-3 border border-line">
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <div className="truncate text-sm font-medium capitalize">{b.spec.name.replace(/_/g, " ")}</div>
          <div className="mt-0.5 flex flex-wrap items-center gap-x-2 gap-y-0.5 text-2xs text-ink-3">
            <ConditionLabel c={b.spec.condition} className="text-2xs" />
            {shifted && (
              <span className="num font-mono">
                Δ({b.spec.cube_offset[0].toFixed(2)}, {b.spec.cube_offset[1].toFixed(2)}) m
              </span>
            )}
            {b.spec.light_scale !== 1 && <span className="num font-mono">light ×{b.spec.light_scale}</span>}
          </div>
        </div>
        <StatusBadge status={b.status} />
      </div>

      {r ? (
        <>
          <div className="mt-3 flex items-baseline gap-2">
            <span className="num kn-display text-[40px]">{pct(r.success_rate)}</span>
            <span className="num font-mono text-2xs text-ink-3" title="95% confidence interval">
              [{pct(r.ci95[0])}, {pct(r.ci95[1])}]
            </span>
          </div>
          {/* CI band on a 0..1 track */}
          <div className="relative mt-1.5 h-1.5 rounded-full bg-rail" aria-hidden>
            <div className="absolute inset-y-0 rounded-full bg-servo/30" style={{ left: `${r.ci95[0] * 100}%`, width: `${(r.ci95[1] - r.ci95[0]) * 100}%` }} />
            <div className="absolute -top-0.5 h-2.5 w-0.5 rounded-full bg-servo" style={{ left: `calc(${r.success_rate * 100}% - 1px)` }} />
          </div>
          <div className="mt-2 flex justify-between text-2xs text-ink-3">
            <span className="num">
              {r.successes}/{r.n} succeeded
            </span>
            <span className="num">mean reward {num(r.mean_reward)}</span>
          </div>
          <EpisodeStrip episodes={r.episodes} />
        </>
      ) : (
        <div className="mt-3 space-y-1.5">
          <div className={cn("h-7 w-20 rounded", live ? "skeleton" : "bg-rail")} />
          <p className="text-2xs text-ink-3">
            {live ? `${b.seeds.length} episodes · running` : b.error ? <InlineError error={b.error} fallback="This branch failed." /> : "No result"}
          </p>
        </div>
      )}

      <div className="mt-2.5 flex items-center justify-between border-t border-line-soft pt-2 text-2xs text-ink-4">
        <span className="truncate font-mono" title={b.seeds.join(", ")}>
          seeds {b.seeds.length ? `${b.seeds[0]}…${b.seeds[b.seeds.length - 1]}` : "—"}
        </span>
        {r && r.wall_s > 0 && <span className="num font-mono">{r.wall_s.toFixed(0)}s</span>}
        <span className="font-mono" title="Parallel local worker process">local worker</span>
      </div>
    </div>
  );
}

/** Per-episode outcome squares (measured). Click one with a video to play it. */
function EpisodeStrip({ episodes }: { episodes: EpisodeResult[] }) {
  const [sel, setSel] = useState<EpisodeResult | null>(null);
  const video = useQuery({
    queryKey: ["media-key", sel?.video_key],
    queryFn: () => api.media(sel!.video_key!),
    enabled: !!sel?.video_key,
    staleTime: 5 * 60_000,
  });
  return (
    <div className="mt-2.5">
      <div className="flex flex-wrap gap-[3px]" aria-label="Per-episode outcomes">
        {episodes.map((ep) => (
          <button
            type="button"
            key={ep.seed}
            onClick={() => setSel(sel?.seed === ep.seed ? null : ep)}
            title={[
              `seed ${ep.seed} · ${ep.success ? "success" : "fail"}`,
              ep.failure_mode ? `failure: ${ep.failure_mode}` : null,
              `reward ${num(ep.reward)} · ${ep.steps} steps`,
              `max cube z ${num(ep.max_cube_z, 3)} m · final cube→box ${num(ep.final_cube_box_dist, 3)} m`,
              ep.video_key ? "click to play" : "no video",
            ]
              .filter(Boolean)
              .join("\n")}
            className={cn(
              "h-3 w-3 rounded-[2px]",
              ep.success ? "bg-pass/80" : "bg-fault/70",
              sel?.seed === ep.seed && "outline outline-2 outline-offset-1 outline-servo",
            )}
          />
        ))}
      </div>
      {sel && (
        <div className="mt-2 rounded bg-plate p-2 text-2xs text-ink-3 border border-line">
          <div className="mb-1 flex flex-wrap gap-x-3">
            <span className="font-mono text-ink">seed {sel.seed}</span>
            <span className={sel.success ? "text-pass" : "text-fault"}>{sel.success ? "success" : sel.failure_mode ?? "fail"}</span>
            <span className="num font-mono">z {num(sel.max_cube_z, 3)}</span>
            <span className="num font-mono">d {num(sel.final_cube_box_dist, 3)}</span>
          </div>
          {!sel.video_key ? (
            <span>No video recorded for this episode.</span>
          ) : video.isLoading ? (
            <div className="skeleton aspect-[4/3] w-full rounded" />
          ) : video.data ? (
            <video src={resolveMediaUrl(video.data.url) ?? undefined} controls autoPlay loop playsInline className="w-full rounded" />
          ) : (
            <span className="text-fault">Video unavailable</span>
          )}
        </div>
      )}
    </div>
  );
}

// ------------------------------------------------------------------ comparison
function Comparison({ pid, evals, order }: { pid: string; evals: EvaluationWithBranches[]; order: string[] }) {
  const sorted = [...evals].sort((a, b) => order.indexOf(a.run.id) - order.indexOf(b.run.id));
  const [result, setResult] = useState<ExperimentComparison | null>(null);
  const m = useMutation({ mutationFn: () => api.compare(pid, sorted.map((e) => e.run.id)), onSuccess: setResult });

  if (sorted.length === 0)
    return (
      <div className="flex items-center gap-2 rounded-md px-3.5 py-2.5 text-xs text-ink-3 border border-line">
        <GitCompare className="h-3.5 w-3.5" />
        Tick finished evaluations (up to {MAX_COMPARE}) to compare them side by side.
      </div>
    );

  // rows = branch names (union, in first-seen order); one bar per evaluation, from measured branch results
  const branchNames: string[] = [];
  for (const e of sorted) for (const b of e.branches) if (!branchNames.includes(b.spec.name)) branchNames.push(b.spec.name);
  const data = branchNames.map((name) => {
    const row: Record<string, unknown> = { branch: name.replace(/_/g, " ") };
    sorted.forEach((e, i) => {
      const r = e.branches.find((b) => b.spec.name === name)?.result;
      if (r) {
        row[`e${i}`] = r.success_rate * 100;
        row[`e${i}_ci`] = [(r.success_rate - r.ci95[0]) * 100, (r.ci95[1] - r.success_rate) * 100];
        row[`e${i}_n`] = r.n;
      }
    });
    return row;
  });

  return (
    <Panel
      title="Comparison"
      meta="success rate by branch · whiskers = 95% CI"
      actions={
        <Button size="sm" loading={m.isPending} onClick={() => m.mutate()} disabled={sorted.length < 2}>
          <GitCompare className="h-3.5 w-3.5" /> {sorted.length < 2 ? "Select 2+ to diff" : "Compute deltas"}
        </Button>
      }
    >
      <div className="h-[240px]">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: -12 }} barGap={2} barCategoryGap="28%">
            <CartesianGrid vertical={false} stroke="var(--line)" />
            <XAxis dataKey="branch" tickLine={false} axisLine={{ stroke: "var(--line-strong)" }} tick={{ fill: "rgb(var(--ink-2))", fontSize: 12 }} />
            <YAxis domain={[0, 100]} ticks={[0, 25, 50, 75, 100]} tickFormatter={(v) => `${v}%`} tickLine={false} axisLine={false} tick={{ fill: "rgb(var(--ink-3))", fontSize: 11 }} />
            <Tooltip
              cursor={{ fill: "rgb(var(--rail) / .6)" }}
              content={({ active, payload, label }) =>
                active && payload?.length ? (
                  <div className="rounded bg-rail px-2.5 py-2 text-xs border border-line-strong">
                    <div className="mb-1 font-medium capitalize text-ink">{label}</div>
                    {payload.map((p) => {
                      const i = Number(String(p.dataKey).slice(1));
                      const row = p.payload as Record<string, unknown>;
                      const ci = row[`e${i}_ci`] as [number, number] | undefined;
                      const v = Number(p.value);
                      return (
                        <div key={String(p.dataKey)} className="flex items-center gap-2 text-ink-2">
                          <span className="h-2 w-2 rounded-sm" style={{ background: SERIES[i] }} />
                          <span className="min-w-[90px] truncate">{policyLabel(sorted[i].run)}</span>
                          <span className="num ml-auto font-mono text-ink">{v.toFixed(0)}%</span>
                          {ci && (
                            <span className="num font-mono text-ink-3">
                              [{(v - ci[0]).toFixed(0)}–{(v + ci[1]).toFixed(0)}] n={String(row[`e${i}_n`])}
                            </span>
                          )}
                        </div>
                      );
                    })}
                  </div>
                ) : null
              }
            />
            {sorted.length > 1 && (
              <Legend
                verticalAlign="top"
                align="right"
                height={24}
                iconType="square"
                iconSize={8}
                formatter={(value) => {
                  const i = Number(String(value).slice(1));
                  return <span className="text-xs text-ink-2">{`${policyLabel(sorted[i].run)} · ${sorted[i].run.id}`}</span>;
                }}
              />
            )}
            {sorted.map((e, i) => (
              <Bar key={e.run.id} dataKey={`e${i}`} fill={SERIES[i]} radius={[4, 4, 0, 0]} maxBarSize={36} isAnimationActive={false}>
                <ErrorBar dataKey={`e${i}_ci`} width={6} strokeWidth={1.5} stroke="rgb(var(--ink-2))" direction="y" />
              </Bar>
            ))}
          </BarChart>
        </ResponsiveContainer>
      </div>

      {m.error && (
        <div className="mt-3">
          <ErrorState error={m.error} compact />
        </div>
      )}
      {result && (
        <div className="mt-3 space-y-2 border-t border-line-soft pt-3">
          {result.summary && <p className="text-sm text-ink">{result.summary}</p>}
          <GenericTable title="Deltas" rows={result.deltas} />
          <details>
            <summary className="cursor-pointer text-2xs text-ink-3 hover:text-ink-2">Table view ({result.rows.length} rows)</summary>
            <div className="mt-2">
              <GenericTable rows={result.rows} />
            </div>
          </details>
        </div>
      )}
    </Panel>
  );
}

/** ExperimentComparison rows/deltas are free-form dicts in the contract; render whatever keys arrive. */
function GenericTable({ rows, title }: { rows: Record<string, unknown>[]; title?: string }) {
  if (rows.length === 0) return <p className="text-2xs text-ink-4">{title ? `${title}: none` : "No rows"}</p>;
  const cols = Array.from(new Set(rows.flatMap((r) => Object.keys(r))));
  const fmt = (k: string, v: unknown) => {
    if (typeof v === "number" && /rate|delta|success/i.test(k) && Math.abs(v) <= 1) {
      const s = `${(v * 100).toFixed(1)}%`;
      return /delta/i.test(k) && v > 0 ? `+${s}` : s;
    }
    return v === null || v === undefined ? "—" : evidenceValue(v);
  };
  return (
    <div className="overflow-x-auto scroll-thin">
      {title && <div className="mb-1 text-2xs font-medium uppercase tracking-[0.06em] text-ink-3">{title}</div>}
      <table className="w-full text-left font-mono text-2xs">
        <thead className="text-ink-3">
          <tr className="border-b border-line-soft">
            {cols.map((c) => (
              <th key={c} className="px-2 py-1.5 font-medium">
                {c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i} className="border-b border-line-soft last:border-0">
              {cols.map((c) => {
                const v = r[c];
                const neg = typeof v === "number" && /delta/i.test(c) && v < 0;
                const pos = typeof v === "number" && /delta/i.test(c) && v > 0;
                return (
                  <td key={c} className={cn("num px-2 py-1.5 text-ink", neg && "text-fault", pos && "text-pass")}>
                    {fmt(c, v)}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ------------------------------------------------------------------ launch
function LaunchDialog({ pid, open, onClose }: { pid: string; open: boolean; onClose: () => void }) {
  const qc = useQueryClient();
  const cks = useCheckpoints(pid);
  const [policy, setPolicy] = useState("expert");
  const [n, setN] = useState(10);
  const [seed, setSeed] = useState(1000);
  const req: LaunchEvaluationRequest = {
    policy: policy === "expert" ? { kind: "scripted_expert" } : { kind: "checkpoint", checkpoint_id: policy },
    episodes_per_branch: n,
    base_seed: seed,
  };
  const m = useMutation({
    mutationFn: () => api.launchEvaluation(pid, req),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["evaluations", pid] });
      onClose();
    },
  });
  return (
    <Dialog
      open={open}
      onClose={onClose}
      title="Launch evaluation"
      footer={
        <>
          <Button variant="ghost" onClick={onClose}>
            Cancel
          </Button>
          <Button variant="primary" loading={m.isPending} onClick={() => m.mutate()}>
            Launch 3 branches × {n}
          </Button>
        </>
      }
    >
      <div className="space-y-4">
        <Field
          label="Policy"
          hint={
            policy === "expert"
              ? "Oracle ceiling, not evidence a policy learned."
              : undefined
          }
        >
          <Select value={policy} onChange={(e) => setPolicy(e.target.value)}>
            <option value="expert">Scripted expert (oracle)</option>
            {(cks.data ?? []).map((c) => (
              <option key={c.id} value={c.id}>
                {c.policy_type.toUpperCase()} · step {c.step} · {c.id}
              </option>
            ))}
          </Select>
        </Field>
        {cks.data && cks.data.length === 0 && <p className="-mt-2 text-2xs text-ink-3">No checkpoints yet. Train a policy first.</p>}
        <div className="rounded bg-well p-3 text-xs text-ink-2 border border-line">
          <div className="mb-1 text-2xs font-medium uppercase tracking-[0.06em] text-ink-3">Branches (backend defaults)</div>
          <ul className="space-y-0.5">
            <li>
              <span className="font-medium text-ink">baseline</span> — the training distribution
            </li>
            <li>
              <span className="font-medium text-ink">shifted object</span> — cube moved off its usual spot
            </li>
            <li>
              <span className="font-medium text-ink">changed lighting</span> — scene light scaled
            </li>
          </ul>
        </div>
        <div className="grid grid-cols-2 gap-3">
          <Field label="Episodes per branch">
            <Input type="number" min={1} max={200} value={n} onChange={(e) => setN(Math.max(1, Number(e.target.value) || 1))} className="num font-mono" />
          </Field>
          <Field label="Base seed">
            <Input type="number" value={seed} onChange={(e) => setSeed(Number(e.target.value) || 0)} className="num font-mono" />
          </Field>
        </div>
        {m.error && <ErrorState error={m.error} compact />}
      </div>
    </Dialog>
  );
}
