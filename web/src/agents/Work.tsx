import { useEffect, useState, type ReactNode } from "react";
import { Link } from "react-router-dom";
import { resolveMediaUrl } from "../api/client";
import type { AgentCollection, EvaluationWithBranches, Job, RLSandboxView, TrainingJob } from "../api/types";
import { Meter, StatusBadge } from "../components/ui";
import { GlowEdge } from "../components/ui/spotlight-card";
import { ago, cn, humanize, pct, usd } from "../lib/format";
import { IsoIcon, type IsoIconName } from "../sketch";
import type { AgentCall, AgentSession } from "./api";
import { isActive, since, useSessionWork, type ProjectWork } from "./sessionWork";

const label = "font-mono text-[10.5px] uppercase tracking-[0.14em]";
/** Jobs whose work already has a richer card (collections, training, evaluations, RL sandboxes). */
const COVERED = new Set(["agent_collect", "train", "evaluate", "rl_pilot", "rl_run"]);

// ------------------------------------------------------------------ primitives

/** A minimal line chart with an optional band (e.g. a 95% interval), drawn in the current ink. */
function Spark({ ys, band, max, height = 44, className }: { ys: number[]; band?: [number, number][]; max?: number; height?: number; className?: string }) {
  if (ys.length < 2) return <div className={cn("flex items-center text-2xs text-ink-4", className)} style={{ height }}>waiting for data</div>;
  const w = 240;
  const top = max ?? Math.max(...ys, ...(band ?? []).map((b) => b[1]), 1e-6);
  const x = (i: number) => (i / (ys.length - 1)) * w;
  const y = (v: number) => height - 2 - (v / top) * (height - 4);
  const line = ys.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)} ${y(v).toFixed(1)}`).join("");
  const area = band
    ? band.map((b, i) => `${i ? "L" : "M"}${x(i).toFixed(1)} ${y(b[1]).toFixed(1)}`).join("") +
      [...band].reverse().map((b, j) => `L${x(band.length - 1 - j).toFixed(1)} ${y(b[0]).toFixed(1)}`).join("") + "Z"
    : null;
  return (
    <svg viewBox={`0 0 ${w} ${height}`} preserveAspectRatio="none" className={cn("w-full", className)} style={{ height }} aria-hidden>
      {area && <path d={area} className="fill-ink/10" />}
      <path d={line} fill="none" className="stroke-ink" strokeWidth={1.4} vectorEffect="non-scaling-stroke" />
    </svg>
  );
}

function Card({ children, live, className }: { children: ReactNode; live?: boolean; className?: string }) {
  return (
    <div className={cn("sk-depth relative rounded border bg-plate p-3.5", live ? "border-servo/60" : "border-line-strong", className)}>
      <GlowEdge />
      {children}
    </div>
  );
}

function Head({ icon, title, status, right }: { icon: IsoIconName; title: ReactNode; status?: string; right?: ReactNode }) {
  return (
    <div className="flex items-center gap-2">
      <IsoIcon name={icon} size={18} accent={isActive(status)} />
      <span className="min-w-0 flex-1 truncate text-[13.5px] text-ink">{title}</span>
      {right}
      {status && <StatusBadge status={status} />}
    </div>
  );
}

function Stat({ k, v }: { k: string; v: ReactNode }) {
  return (
    <div className="min-w-0">
      <div className={cn("text-ink-3", label)}>{k}</div>
      <div className="num truncate text-[13px] text-ink">{v}</div>
    </div>
  );
}

function Section({ icon, title, count, children, compact }: { icon: IsoIconName; title: string; count: number; children: ReactNode; compact?: boolean }) {
  return (
    <section>
      <div className="mb-2.5 flex items-center gap-2.5">
        <IsoIcon name={icon} size={18} />
        <h3 className={cn("text-ink", label)}>{title}</h3>
        <span className={cn("text-ink-3", label)}>{count}</span>
        <span aria-hidden className="h-px flex-1 bg-line" />
      </div>
      <div className={cn("grid gap-3", !compact && "md:grid-cols-2")}>{children}</div>
    </section>
  );
}

// ------------------------------------------------------------------ cards

function CollectionCard({ c }: { c: AgentCollection }) {
  const t = c.totals ?? {};
  const want = c.request?.num_demos ?? 0;
  const ok = t.successes ?? 0;
  return (
    <Card live={isActive(c.status)}>
      <Head icon="demonstrate" title={<>Demonstrations · <span className="font-mono text-xs text-ink-2">{c.collector}{c.model ? ` · ${c.model}` : ""}</span></>} status={c.status} />
      <Meter value={want ? ok / want : 0} tone={c.status === "failed" ? "fault" : "servo"} className="mt-3" label={`${ok} of ${want} demos`} />
      <div className="mt-3 grid grid-cols-3 gap-3">
        <Stat k="demos" v={`${ok} / ${want}`} />
        <Stat k="attempts" v={t.attempts ?? 0} />
        <Stat k="spend" v={`${usd(t.cost_usd ?? 0)}${t.budget_cap_usd ? ` of ${usd(t.budget_cap_usd)}` : ""}`} />
      </div>
      {(c.error || c.reason) && <p className="mt-2 line-clamp-2 text-xs text-ink-3">{c.error ?? c.reason}</p>}
    </Card>
  );
}

function JobCard({ j }: { j: Job }) {
  const icon: IsoIconName = j.kind === "record" || j.kind === "import" ? "demonstrate" : j.kind === "dataset" ? "dataset" : j.kind === "analyze" ? "evaluate" : "loop";
  return (
    <Card live={isActive(j.status)}>
      <Head icon={icon} title={humanize(j.kind)} status={j.status} right={<span className="font-mono text-[11px] text-ink-3">{ago(j.created_at)}</span>} />
      <Meter value={j.progress} tone={j.status === "failed" ? "fault" : "servo"} className="mt-3" label={`${pct(j.progress)} done`} />
      <p className={cn("mt-2 line-clamp-2 text-xs", j.status === "failed" ? "text-fault" : "text-ink-3")}>{j.error ?? j.message}</p>
    </Card>
  );
}

function TrainingCard({ t, pid }: { t: TrainingJob; pid: string }) {
  const curve = t.metrics?.loss_curve ?? [];
  const last = curve[curve.length - 1];
  const waiting = t.status === "proposed";
  return (
    <Card live={isActive(t.status)} className={waiting ? "border-caution/60" : undefined}>
      <Head icon="train" title={<>Training · <span className="font-mono text-xs uppercase text-ink-2">{t.policy_type}</span></>} status={t.status} />
      {waiting && (
        <div className="mt-3 flex items-center justify-between gap-2 rounded-sm border border-caution/40 px-2.5 py-2 text-xs">
          <span className="text-caution">Waiting for a person to approve the GPU spend.</span>
          <Link to={`/p/${pid}/improve`} className="shrink-0 font-medium text-ink underline underline-offset-2">Review</Link>
        </div>
      )}
      <div className="mt-3">
        <div className={cn("mb-1 flex justify-between text-ink-3", label)}>
          <span>loss</span>
          <span className="num">{last ? `${last[1].toFixed(3)} @ step ${last[0]}` : "—"}</span>
        </div>
        <Spark ys={curve.map((p) => p[1])} />
      </div>
      <div className="mt-3 grid grid-cols-3 gap-3">
        <Stat k="backend" v={humanize(t.backend)} />
        <Stat k="gpu est" v={`${t.estimate?.gpu_hours?.toFixed(2) ?? "—"} h`} />
        <Stat k="checkpoints" v={t.checkpoint_ids.length} />
      </div>
      {t.error && <p className="mt-2 line-clamp-2 text-xs text-fault">{t.error}</p>}
    </Card>
  );
}

function EvaluationCard({ e }: { e: EvaluationWithBranches }) {
  return (
    <Card live={isActive(e.run.status)}>
      <Head icon="evaluate" title={<>Evaluation · <span className="text-xs text-ink-2">{e.run.policy_label}</span></>} status={e.run.status} />
      <ul className="mt-3 space-y-2">
        {e.branches.map((b) => {
          const r = b.result;
          return (
            <li key={b.id}>
              <div className="flex items-baseline justify-between gap-2 text-xs">
                <span className="truncate text-ink-2">{humanize(b.spec.name)}</span>
                <span className="num shrink-0 text-ink">
                  {r ? `${r.successes}/${r.n} · ${pct(r.success_rate)}` : b.status === "running" || b.status === "queued" ? "running…" : b.error ? "failed" : "—"}
                </span>
              </div>
              <div className="relative mt-1 h-1.5 rounded-full bg-rail">
                {r && (
                  <>
                    <span className="absolute inset-y-0 rounded-full bg-ink/15" style={{ left: `${r.ci95[0] * 100}%`, width: `${(r.ci95[1] - r.ci95[0]) * 100}%` }} />
                    <span className="absolute inset-y-0 left-0 rounded-full bg-ink" style={{ width: `${r.success_rate * 100}%`, opacity: 0.75 }} />
                  </>
                )}
              </div>
            </li>
          );
        })}
      </ul>
    </Card>
  );
}

/** One RL sandbox in its own box: the weakness, the estimate it was accepted on, live progress, the measured result. */
function SandboxBox({ s }: { s: RLSandboxView }) {
  const live = s.status === "running" || s.status === "piloting";
  const [tick, setTick] = useState(0);
  useEffect(() => {
    if (!live) return;
    const id = setInterval(() => setTick((n) => n + 1), 1500);
    return () => clearInterval(id);
  }, [live]);
  const e = s.estimate;
  const up = e?.predicted_policy_uplift;
  const frame = s.live_frame && live ? `${resolveMediaUrl(s.live_frame)}?t=${tick}` : null;
  return (
    <Card live={live}>
      <Head icon="improve" title={<>RL sandbox · <span className="text-xs text-ink-2">{humanize(e?.condition ?? s.weakness_id)}</span></>} status={s.status} />
      <div className="mt-3 grid grid-cols-[minmax(0,1fr)_96px] gap-3">
        <div className="min-w-0">
          <div className={cn("mb-1 flex justify-between text-ink-3", label)}>
            <span>success · iteration {s.iteration}/{s.max_iterations}</span>
            <span className="num">{s.success ? pct(s.success.rate) : "—"}</span>
          </div>
          <Spark ys={s.curve.map((c) => c.success_rate)} band={s.curve.map((c) => c.ci95)} max={1} />
        </div>
        <div className="overflow-hidden rounded-sm border border-line bg-well">
          {frame ? <img src={frame} alt="live sandbox frame" className="h-full w-full object-cover" /> : <div className="flex h-full min-h-[60px] items-center justify-center text-2xs text-ink-4">{s.phase ?? "no frame"}</div>}
        </div>
      </div>
      <div className="mt-3 grid grid-cols-3 gap-3">
        <Stat k="policy now" v={pct(e?.policy_success)} />
        <Stat
          k="est. uplift"
          v={
            up ? (
              <span title={up.source === "prior" ? "a prior until 3 measured rounds exist" : up.source}>
                +{pct(up.mean)}
                {up.source && up.source !== "measured" && <span className="ml-1 text-2xs text-ink-3">{up.source}</span>}
              </span>
            ) : "—"
          }
        />
        <Stat k="cost" v={e?.cost ? `${usd(e.cost.usd)} · ${e.cost.cpu_hours.toFixed(2)} cpu-h` : "—"} />
      </div>
      {s.result && (
        <div className="mt-3 rounded-sm border border-line px-2.5 py-2 text-xs">
          <span className={cn("text-ink-3", label)}>measured on held-out seeds</span>
          <div className="num mt-1 text-ink">
            base {pct(s.result.base_success)} → tuned {pct(s.result.tuned_success)}
            {s.result.p_value != null && <span className="text-ink-3"> · p={s.result.p_value.toFixed(3)}</span>}
            <span className="text-ink-3"> · {s.result.demo_episode_ids.length} demos</span>
          </div>
        </div>
      )}
      {s.status === "ready" && <p className="mt-2 text-xs text-ink-3">Pilot done; waiting for the agent to accept it.</p>}
      {s.reason && !s.result && <p className="mt-2 line-clamp-2 text-xs text-ink-3">{s.reason}</p>}
    </Card>
  );
}

// ------------------------------------------------------------------ the board

/** What the agent is doing right now, in one line, plus the counts that matter. */
export function NowBar({ session, calls, work }: { session: AgentSession; calls: AgentCall[]; work: (ProjectWork | undefined)[] }) {
  const running = [...calls].reverse().find((c) => c.status === "running");
  const last = calls[calls.length - 1];
  const all = work.filter(Boolean) as ProjectWork[];
  const active =
    all.reduce((n, w) => n + w.jobs.filter((j) => isActive(j.status)).length, 0) +
    all.reduce((n, w) => n + w.sandboxes.filter((s) => s.status === "running" || s.status === "piloting").length, 0);
  const approvals = all.reduce((n, w) => n + w.training.filter((t) => t.status === "proposed").length, 0) + calls.filter((c) => c.status === "approval_required").length;
  return (
    <Card live={!!running || active > 0} className="mb-5">
      <div className="flex flex-wrap items-center gap-x-6 gap-y-2">
        <div className="flex min-w-0 flex-1 items-center gap-2.5">
          <span className={cn("h-2.5 w-2.5 shrink-0 rounded-full", running ? "live-dot bg-servo" : session.live ? "bg-pass" : "bg-ink-4")} />
          <span className={cn("shrink-0 text-ink-3", label)}>now</span>
          <span className="min-w-0 truncate text-[14px] text-ink">
            {running ? (
              <>calling <span className="font-mono">{running.tool}</span>…</>
            ) : last ? (
              <>
                <span className="font-mono">{last.tool}</span> <span className="text-ink-3">· {last.summary ?? last.status} · {ago(last.ended_at ?? last.started_at)}</span>
              </>
            ) : (
              "connected, no calls yet"
            )}
          </span>
        </div>
        <Stat k="background jobs" v={active} />
        <Stat k="needs you" v={<span className={approvals ? "text-caution" : undefined}>{approvals ? `${approvals} approval${approvals > 1 ? "s" : ""}` : "nothing"}</span>} />
        <Stat k="calls" v={session.call_count} />
      </div>
    </Card>
  );
}

/**
 * Everything these projects are doing: demos, training, evaluations, RL. With `from` (an agent session's
 * start), only work since then, with a toggle for earlier work; without it, all of the projects' work.
 */
export function WorkBoard({ from, work, loading, compact }: { from?: string; work: (ProjectWork | undefined)[]; loading: boolean; compact?: boolean }) {
  const [all, setAll] = useState(!from);
  const pick = <T extends { created_at?: string }>(xs: T[]) => (all ? xs : since(xs, from));
  const rows = work.filter(Boolean) as ProjectWork[];
  const collections = rows.flatMap((w) => pick(w.collections));
  const jobs = rows.flatMap((w) => pick(w.jobs).filter((j) => !COVERED.has(j.kind)));
  const demoJobs = jobs.filter((j) => j.kind === "record" || j.kind === "import");
  const otherJobs = jobs.filter((j) => j.kind !== "record" && j.kind !== "import");
  const training = rows.flatMap((w) => pick(w.training).map((t) => ({ t, pid: w.pid })));
  const evals = rows.flatMap((w) => (all ? w.evaluations : w.evaluations.filter((e) => since([e.run], from).length)));
  const sandboxes = rows.flatMap((w) => w.sandboxes.filter((s) => s.status !== "dismissed"));
  const empty = !collections.length && !demoJobs.length && !otherJobs.length && !training.length && !evals.length && !sandboxes.length;

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <h2 className={cn("text-ink", label)}>Work in progress</h2>
        <span className={cn("text-ink-3", label)}>{all ? (from ? "all work in these projects" : "everything in this project") : "since this session started"}</span>
        {from && (
          <button type="button" onClick={() => setAll((v) => !v)} className={cn("ml-auto text-ink-3 underline-offset-2 hover:text-ink hover:underline", label)}>
            {all ? "this session only" : "show earlier work"}
          </button>
        )}
      </div>
      {loading && !rows.length ? (
        <p className="text-sm text-ink-3">Loading the agent's work…</p>
      ) : empty ? (
        <Card>
          <p className="text-sm text-ink-2">Nothing running yet.</p>
          <p className="mt-1 text-xs text-ink-3">Demonstrations, training, evaluations and RL sandboxes the agent starts appear here live, with their progress, whether or not it polls for them.</p>
        </Card>
      ) : (
        <>
          {(collections.length > 0 || demoJobs.length > 0) && (
            <Section compact={compact} icon="demonstrate" title="Demonstrations" count={collections.length + demoJobs.length}>
              {collections.map((c) => <CollectionCard key={c.id} c={c} />)}
              {demoJobs.map((j) => <JobCard key={j.id} j={j} />)}
            </Section>
          )}
          {training.length > 0 && (
            <Section compact={compact} icon="train" title="Training" count={training.length}>
              {training.map(({ t, pid }) => <TrainingCard key={t.id} t={t} pid={pid} />)}
            </Section>
          )}
          {evals.length > 0 && (
            <Section compact={compact} icon="evaluate" title="Evaluations" count={evals.length}>
              {evals.map((e) => <EvaluationCard key={e.run.id} e={e} />)}
            </Section>
          )}
          {sandboxes.length > 0 && (
            <Section compact={compact} icon="improve" title="RL sandboxes" count={sandboxes.length}>
              {sandboxes.map((s) => <SandboxBox key={s.id} s={s} />)}
            </Section>
          )}
          {otherJobs.length > 0 && (
            <Section compact={compact} icon="loop" title="Other background work" count={otherJobs.length}>
              {otherJobs.map((j) => <JobCard key={j.id} j={j} />)}
            </Section>
          )}
        </>
      )}
    </div>
  );
}

export { useSessionWork };
