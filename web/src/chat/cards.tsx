import { useMemo, useState, type ReactNode } from "react";
import { Link } from "react-router-dom";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import {
  ArrowUpRight,
  BarChart3,
  Bot,
  Check,
  CircleDashed,
  Database,
  Dumbbell,
  Gauge,
  ListOrdered,
  Play,
  ScanSearch,
  ShieldCheck,
  Sparkles,
  Wallet,
  X,
} from "lucide-react";
import { api, resolveMediaUrl } from "../api/client";
import { isJobActive, useJob } from "../api/events";
import {
  useAgentCollection,
  useAnalysis,
  useDatasets,
  useEpisodeMedia,
  useEpisodes,
  useEvaluation,
  useJobQuery,
  usePlannerBudget,
  useRecommendations,
  useTraining,
} from "../api/hooks";
import { isImported, type AgentAttempt, type TrainingJob } from "../api/types";
import { AttemptDrawer, ScenarioChip } from "../components/AgentTrace";
import { Skeleton } from "../components/ui";
import { refRoute } from "../components/refs";
import { cn, errorMessage, num, pct, usd } from "../lib/format";
import { useChat } from "./ChatProvider";
import { prefixOf } from "./model";

// ------------------------------------------------------------------ shell
export function CardShell({
  icon,
  title,
  meta,
  status,
  pid,
  refId,
  children,
  large,
  tone = "default",
}: {
  icon: ReactNode;
  title: ReactNode;
  meta?: ReactNode;
  status?: ReactNode;
  pid: string;
  refId?: string;
  children?: ReactNode;
  large?: boolean;
  tone?: "default" | "approval";
}) {
  const to = refId ? refRoute(pid, refId) : null;
  return (
    <div
      className={cn(
        "enter overflow-hidden rounded-xl bg-plate",
        tone === "approval" ? "shadow-[0_0_0_1px_rgb(var(--servo)/.45)]" : "shadow-[0_0_0_1px_var(--line)]",
        large && "rounded-2xl",
      )}
    >
      <div className={cn("flex items-center gap-2.5", large ? "px-5 pt-4" : "px-4 pt-3")}>
        <span className={cn("flex shrink-0 items-center justify-center rounded-lg", large ? "h-8 w-8" : "h-7 w-7", tone === "approval" ? "bg-servo/15 text-servo" : "bg-rail text-ink-2")}>
          {icon}
        </span>
        <div className="min-w-0 flex-1">
          <div className={cn("truncate font-semibold text-ink", large ? "text-base" : "text-sm")}>{title}</div>
          {meta && <div className="truncate text-xs text-ink-3">{meta}</div>}
        </div>
        {status}
        {to && (
          <Link to={to} className="rounded-md p-1 text-ink-4 hover:bg-rail hover:text-ink-2" title="Open details" aria-label="Open details">
            <ArrowUpRight className="h-4 w-4" />
          </Link>
        )}
      </div>
      <div className={cn(large ? "px-5 pb-5 pt-3" : "px-4 pb-3.5 pt-2.5")}>{children}</div>
    </div>
  );
}

function Pill({ tone, children }: { tone: "live" | "ok" | "bad" | "muted" | "wait"; children: ReactNode }) {
  const cls = {
    live: "bg-servo/12 text-servo",
    ok: "bg-pass/12 text-pass",
    bad: "bg-fault/12 text-fault",
    muted: "bg-rail text-ink-3",
    wait: "bg-caution/12 text-caution",
  }[tone];
  return (
    <span className={cn("inline-flex h-6 shrink-0 items-center gap-1.5 rounded-full px-2.5 text-xs font-medium", cls)}>
      {tone === "live" && <span className="h-1.5 w-1.5 rounded-full bg-current live-dot" />}
      {children}
    </span>
  );
}

function Loading({ large }: { large?: boolean }) {
  return (
    <div className={cn("space-y-2 rounded-xl p-4 shadow-[0_0_0_1px_var(--line)]", large && "p-5")}>
      <Skeleton className="h-4 w-40" />
      <Skeleton className="h-10 w-full" />
    </div>
  );
}

function Missing({ what }: { what: string }) {
  return <div className="rounded-xl px-4 py-3 text-xs text-ink-3 shadow-[0_0_0_1px_var(--line)]">Couldn't load {what}.</div>;
}

// ------------------------------------------------------------------ dispatch
/** Renders the right card for a set of refs from one tool result. */
export function ResultCards({ pid, refs, large }: { pid: string; refs: string[]; large?: boolean }) {
  const eps = refs.filter((r) => prefixOf(r) === "ep");
  const recs = refs.filter((r) => prefixOf(r) === "rec");
  const singles = refs.filter((r) => !["ep", "rec"].includes(prefixOf(r))).slice(0, 3);
  return (
    <div className="space-y-2.5">
      {singles.map((r) => (
        <ResultCard key={r} pid={pid} refId={r} large={large} />
      ))}
      {recs.length > 0 && <RecommendationsCard pid={pid} ids={recs} large={large} />}
      {eps.length > 0 && <EpisodesCard pid={pid} ids={eps} large={large} />}
    </div>
  );
}

export function ResultCard({ pid, refId, large }: { pid: string; refId: string; large?: boolean }) {
  switch (prefixOf(refId)) {
    case "ac":
      return <CollectionCard pid={pid} id={refId} large={large} />;
    case "ev":
      return <EvaluationCard pid={pid} id={refId} large={large} />;
    case "ds":
      return <DatasetCard pid={pid} id={refId} large={large} />;
    case "tj":
      return <TrainingCard pid={pid} id={refId} large={large} />;
    case "an":
      return <AnalysisCard pid={pid} id={refId} large={large} />;
    case "job":
      return <JobCard pid={pid} id={refId} large={large} />;
    case "rec":
      return <RecommendationsCard pid={pid} ids={[refId]} large={large} />;
    case "ep":
      return <EpisodesCard pid={pid} ids={[refId]} large={large} />;
    default:
      return null;
  }
}

// ------------------------------------------------------------------ collection (ac_)
function CollectionCard({ pid, id, large }: { pid: string; id: string; large?: boolean }) {
  const first = useAgentCollection(id);
  const running = !!first.data && ["queued", "running"].includes(first.data.collection.status);
  const q = useAgentCollection(id, running);
  const [drawer, setDrawer] = useState<string | null>(null);
  if (q.isLoading) return <Loading large={large} />;
  if (!q.data) return <Missing what="this collection" />;
  const { collection: c, attempts } = q.data;
  const t = c.totals;
  const done = t.attempts ?? attempts.length;
  const total = c.request.num_demos;
  const recent = [...attempts].sort((a, b) => b.index - a.index).slice(0, large ? 8 : 4);
  const who = c.collector === "astra" ? "GPT-6 Astra" : "Scripted oracle";
  return (
    <CardShell
      pid={pid}
      refId={id}
      large={large}
      icon={c.collector === "astra" ? <Bot className="h-4 w-4" /> : <Sparkles className="h-4 w-4" />}
      title="Demonstrations"
      meta={`${who} · ${total} attempts`}
      status={
        running ? (
          <Pill tone="live">collecting</Pill>
        ) : c.status === "succeeded" ? (
          <Pill tone="ok">done</Pill>
        ) : (
          <Pill tone={c.status === "failed" ? "bad" : "muted"}>{c.status.replace(/_/g, " ")}</Pill>
        )
      }
    >
      <div className="flex items-end gap-6">
        <Stat value={t.successes ?? 0} label="demos" tone="pass" large={large} />
        <Stat value={t.failures ?? 0} label="failed" large={large} />
        {c.collector === "astra" && <Stat value={usd(t.cost_usd ?? 0)} label="model cost" large={large} />}
      </div>
      <div className="mt-3 h-1.5 overflow-hidden rounded-full bg-rail">
        <div className="h-full rounded-full bg-servo transition-[width] duration-500 ease-out" style={{ width: `${total ? (done / total) * 100 : 0}%` }} />
      </div>
      <div className="mt-1 text-xs text-ink-3">
        {done} of {total} attempts · failed attempts are diagnostics, never training data
      </div>
      {recent.length > 0 && (
        <ul className="mt-3 space-y-1">
          {recent.map((a) => (
            <AttemptRow key={a.id} a={a} onOpen={() => setDrawer(a.id)} />
          ))}
        </ul>
      )}
      <AttemptDrawer attemptId={drawer} onClose={() => setDrawer(null)} />
    </CardShell>
  );
}

function AttemptRow({ a, onOpen }: { a: AgentAttempt; onOpen: () => void }) {
  return (
    <li>
      <button type="button" onClick={onOpen} className="flex w-full items-center gap-2.5 rounded-md px-1.5 py-1 text-left hover:bg-rail/60">
        <span className={cn("flex h-4 w-4 shrink-0 items-center justify-center rounded-full", a.status === "success" ? "bg-pass/15 text-pass" : "bg-fault/15 text-fault")}>
          {a.status === "success" ? <Check className="h-2.5 w-2.5" /> : <X className="h-2.5 w-2.5" />}
        </span>
        <ScenarioChip scenario={a.scenario} className="min-w-0 flex-1" />
        <Play className="h-3 w-3 shrink-0 text-ink-4" />
      </button>
    </li>
  );
}

function Stat({ value, label, tone, large }: { value: ReactNode; label: string; tone?: "pass"; large?: boolean }) {
  return (
    <div>
      <div className={cn("num font-semibold leading-none tracking-[-0.02em]", large ? "text-4xl" : "text-2xl", tone === "pass" ? "text-pass" : "text-ink")}>{value}</div>
      <div className="mt-1 text-xs text-ink-3">{label}</div>
    </div>
  );
}

// ------------------------------------------------------------------ evaluation (ev_)
function EvaluationCard({ pid, id, large }: { pid: string; id: string; large?: boolean }) {
  const first = useEvaluation(id);
  const running = !!first.data && first.data.branches.some((b) => b.status === "queued" || b.status === "running");
  const q = useEvaluation(id, running);
  if (q.isLoading) return <Loading large={large} />;
  if (!q.data) return <Missing what="this evaluation" />;
  const { run, branches } = q.data;
  const label = run.policy_label || (run.policy.kind === "scripted_expert" ? "Scripted expert (oracle reference)" : run.policy.checkpoint_id ?? "policy");
  return (
    <CardShell
      pid={pid}
      refId={id}
      large={large}
      icon={<BarChart3 className="h-4 w-4" />}
      title="Test in 3 conditions"
      meta={label}
      status={running ? <Pill tone="live">testing</Pill> : run.status === "succeeded" ? <Pill tone="ok">done</Pill> : <Pill tone="muted">{run.status}</Pill>}
    >
      <div className="space-y-3">
        {branches.map((b) => {
          const r = b.result;
          return (
            <div key={b.id}>
              <div className="mb-1 flex items-baseline justify-between gap-2 text-sm">
                <span className="capitalize text-ink-2">{b.spec.name.replace(/_/g, " ")}</span>
                {r ? (
                  <span className="num font-mono text-ink">
                    <span className={cn("font-semibold", large && "text-base")}>{pct(r.success_rate)}</span>
                    <span className="ml-1.5 text-xs text-ink-3">
                      {r.successes}/{r.n}
                    </span>
                  </span>
                ) : (
                  <span className="text-xs text-ink-3">{b.status === "running" || b.status === "queued" ? "running…" : b.error ? "failed" : "no result"}</span>
                )}
              </div>
              <div className={cn("relative overflow-hidden rounded-full bg-rail", large ? "h-3" : "h-2")}>
                {r ? (
                  <>
                    <div className="absolute inset-y-0 bg-ink-4/40" style={{ left: `${r.ci95[0] * 100}%`, width: `${(r.ci95[1] - r.ci95[0]) * 100}%` }} title="95% confidence interval" />
                    <div className={cn("absolute inset-y-0 left-0 rounded-full", r.success_rate >= 0.8 ? "bg-pass" : r.success_rate >= 0.5 ? "bg-servo" : "bg-fault")} style={{ width: `${r.success_rate * 100}%`, opacity: 0.9 }} />
                  </>
                ) : (
                  <div className="skeleton absolute inset-0" />
                )}
              </div>
            </div>
          );
        })}
      </div>
      <div className="mt-2.5 text-xs text-ink-3">Grey band = 95% confidence interval. Small samples give wide bands.</div>
    </CardShell>
  );
}

// ------------------------------------------------------------------ dataset (ds_)
function DatasetCard({ pid, id, large }: { pid: string; id: string; large?: boolean }) {
  const q = useDatasets(pid);
  if (q.isLoading) return <Loading large={large} />;
  const d = q.data?.find((x) => x.id === id);
  if (!d) return <Missing what="this dataset" />;
  const sources = new Set(d.links.map((l) => l.episode_id)).size;
  const aug = d.links.filter((l) => l.augmentation).length;
  const train = d.links.filter((l) => l.split === "train").length;
  return (
    <CardShell pid={pid} refId={id} large={large} icon={<Database className="h-4 w-4" />} title={`Dataset ${d.name}`} meta="LeRobot format, ready to train" status={<Pill tone="ok">built</Pill>}>
      <div className="flex items-end gap-6">
        <Stat value={sources} label="source demos" large={large} />
        <Stat value={aug} label="augmented copies" large={large} />
        <Stat value={d.num_frames.toLocaleString()} label="frames" large={large} />
      </div>
      <div className="mt-2.5 text-xs text-ink-3">
        {train} train · {d.links.length - train} held out for eval
        {d.augmentations.length > 0 && ` · ${d.augmentations.map((a) => a.type.replace("_", " ")).join(", ")}`}
      </div>
    </CardShell>
  );
}

// ------------------------------------------------------------------ training (tj_)
function TrainingCard({ pid, id, large }: { pid: string; id: string; large?: boolean }) {
  const q = useTraining(pid);
  if (q.isLoading) return <Loading large={large} />;
  const t = q.data?.find((x) => x.id === id);
  if (!t) return <Missing what="this training job" />;
  if (t.status === "proposed") return <TrainingApproval pid={pid} tj={t} large={large} />;
  return (
    <CardShell
      pid={pid}
      refId={id}
      large={large}
      icon={<Dumbbell className="h-4 w-4" />}
      title={`${t.policy_type.toUpperCase()} training`}
      meta={trainingWhat(t)}
      status={
        t.status === "running" || t.status === "queued" ? (
          <Pill tone="live">{t.status === "queued" ? "queued" : "training"}</Pill>
        ) : t.status === "succeeded" ? (
          <Pill tone="ok">trained</Pill>
        ) : t.status === "approved" ? (
          <Pill tone="wait">approved</Pill>
        ) : (
          <Pill tone={t.status === "failed" ? "bad" : "muted"}>{t.status}</Pill>
        )
      }
    >
      <div className="text-sm text-ink-2">{estimateLine(t)}</div>
      {t.checkpoint_ids.length > 0 && <div className="mt-1 text-xs text-ink-3">{t.checkpoint_ids.length} checkpoint{t.checkpoint_ids.length === 1 ? "" : "s"} saved</div>}
      {t.backend === "manual" && t.status !== "succeeded" && (
        <div className="mt-1 text-xs text-ink-3">No GPU backend is connected; the training command is in Details › Improve › Training.</div>
      )}
      {t.error && <div className="mt-1 text-xs text-fault">{t.error}</div>}
    </CardShell>
  );
}

const trainingWhat = (t: TrainingJob) => {
  const steps = typeof t.config.steps === "number" ? `${(t.config.steps as number).toLocaleString()} steps` : null;
  return [`dataset ${t.dataset_version_id}`, steps].filter(Boolean).join(" · ");
};
const estimateLine = (t: TrainingJob) =>
  `${t.estimate.gpu} · about ${num(t.estimate.gpu_hours, 1)} GPU-hours · ${t.estimate.paid ? `${usd(t.estimate.est_cost_usd)} estimated` : "no charge"}`;

// ------------------------------------------------------------------ analysis (an_)
function AnalysisCard({ pid, id, large }: { pid: string; id: string; large?: boolean }) {
  const q = useAnalysis(id);
  const flags = useMemo(() => {
    const m = new Map<string, { n: number; severity: string; message: string }>();
    for (const rep of Object.values(q.data?.quality ?? {}))
      for (const f of rep.flags) {
        const cur = m.get(f.code) ?? { n: 0, severity: f.severity, message: f.message };
        cur.n++;
        m.set(f.code, cur);
      }
    return [...m.entries()].sort((a, b) => b[1].n - a[1].n).slice(0, 3);
  }, [q.data]);
  if (q.isLoading) return <Loading large={large} />;
  if (!q.data) return <Missing what="this analysis" />;
  const a = q.data;
  return (
    <CardShell pid={pid} refId={id} large={large} icon={<ScanSearch className="h-4 w-4" />} title="Demo quality check" meta={`${a.episode_count} demos checked`} status={a.status === "running" ? <Pill tone="live">checking</Pill> : <Pill tone="ok">done</Pill>}>
      {a.summary && <p className="text-sm leading-relaxed text-ink-2">{a.summary}</p>}
      {flags.length > 0 ? (
        <ul className="mt-2 space-y-1">
          {flags.map(([code, f]) => (
            <li key={code} className="flex items-center gap-2 text-xs">
              <span className={cn("h-1.5 w-1.5 shrink-0 rounded-full", f.severity === "error" ? "bg-fault" : f.severity === "warn" ? "bg-caution" : "bg-ink-4")} />
              <span className="text-ink-2">{f.message}</span>
              <span className="num ml-auto font-mono text-ink-3">×{f.n}</span>
            </li>
          ))}
        </ul>
      ) : (
        <p className="mt-1 text-xs text-ink-3">No quality flags raised.</p>
      )}
    </CardShell>
  );
}

// ------------------------------------------------------------------ recommendations (rec_)
function RecommendationsCard({ pid, ids, large }: { pid: string; ids: string[]; large?: boolean }) {
  const q = useRecommendations(pid);
  if (q.isLoading) return <Loading large={large} />;
  const all = q.data ?? [];
  const picked = all.filter((r) => ids.includes(r.id));
  const list = (picked.length ? picked : all).sort((a, b) => b.score - a.score).slice(0, 3);
  if (!list.length) return <Missing what="recommendations" />;
  return (
    <CardShell pid={pid} refId={list[0].id} large={large} icon={<ListOrdered className="h-4 w-4" />} title="What to do next" meta="ranked by expected gain">
      <ol className="space-y-2">
        {list.map((r, i) => (
          <li key={r.id} className="flex items-start gap-2.5">
            <span className="num mt-0.5 w-4 shrink-0 font-mono text-xs text-ink-4">{i + 1}</span>
            <div className="min-w-0 flex-1">
              <div className="text-sm text-ink">{r.title}</div>
              <div className="mt-0.5 flex items-center gap-2 text-xs">
                {r.support === "evidence_backed" ? (
                  <span className="inline-flex items-center gap-1 text-pass">
                    <ShieldCheck className="h-3 w-3" /> backed by measured data
                  </span>
                ) : (
                  <span className="inline-flex items-center gap-1 text-ink-3">
                    <Sparkles className="h-3 w-3" /> hypothesis, worth testing
                  </span>
                )}
              </div>
            </div>
          </li>
        ))}
      </ol>
    </CardShell>
  );
}

// ------------------------------------------------------------------ job (job_)
function JobCard({ pid, id, large }: { pid: string; id: string; large?: boolean }) {
  const live = useJob(id);
  const q = useJobQuery(id, !live || isJobActive(live));
  const j = live ?? q.data;
  if (!j) return q.isLoading ? <Loading large={large} /> : null;
  if (j.status === "succeeded" && j.result_ref && prefixOf(j.result_ref) !== "job") return <ResultCard pid={pid} refId={j.result_ref} large={large} />;
  const active = isJobActive(j);
  return (
    <CardShell
      pid={pid}
      large={large}
      icon={active ? <CircleDashed className="h-4 w-4 animate-spin" /> : <Gauge className="h-4 w-4" />}
      title={j.message || j.kind.replace(/_/g, " ")}
      meta={j.kind.replace(/_/g, " ")}
      status={active ? <Pill tone="live">{pct(j.progress)}</Pill> : j.status === "succeeded" ? <Pill tone="ok">done</Pill> : <Pill tone="bad">{j.status}</Pill>}
    >
      <div className="h-1.5 overflow-hidden rounded-full bg-rail">
        <div className="h-full rounded-full bg-servo transition-[width] duration-500 ease-out" style={{ width: `${j.progress * 100}%` }} />
      </div>
      {j.error && <div className="mt-1.5 text-xs text-fault">{j.error}</div>}
    </CardShell>
  );
}

// ------------------------------------------------------------------ episodes (ep_)
function EpisodesCard({ pid, ids, large }: { pid: string; ids: string[]; large?: boolean }) {
  const episodes = useEpisodes(pid);
  const [open, setOpen] = useState<string | null>(ids.length === 1 ? ids[0] : null);
  const shown = ids.slice(0, large ? 12 : 6);
  const first = episodes.data?.find((e) => e.id === ids[0]);
  const imported = first && isImported(first) ? `Human demo · imported${first.provenance?.source ? ` · ${first.provenance.source}` : ""}` : undefined;
  return (
    <CardShell pid={pid} refId={ids[0]} large={large} icon={<Play className="h-4 w-4" />} title={ids.length === 1 ? "Demo" : `${ids.length} demos`} meta={imported ?? (ids.length > shown.length ? `showing ${shown.length}` : undefined)}>
      {open && <EpisodeVideo id={open} />}
      {ids.length > 1 && (
        <div className={cn("grid gap-1.5", large ? "grid-cols-4" : "grid-cols-3", open && "mt-2")}>
          {shown.map((e) => (
            <EpisodeThumb key={e} id={e} active={e === open} ok={episodes.data?.find((x) => x.id === e)?.success} onClick={() => setOpen(e === open ? null : e)} />
          ))}
        </div>
      )}
    </CardShell>
  );
}

export function EpisodeVideo({ id, className }: { id: string; className?: string }) {
  const m = useEpisodeMedia(id);
  const video = resolveMediaUrl(m.data?.video_url);
  const thumb = resolveMediaUrl(m.data?.thumb_url);
  if (m.isLoading) return <Skeleton className={cn("aspect-[4/3] w-full rounded-lg", className)} />;
  if (video) return <video key={video} src={video} poster={thumb ?? undefined} controls autoPlay muted loop playsInline className={cn("aspect-[4/3] w-full rounded-lg bg-well object-contain", className)} />;
  if (thumb) return <img src={thumb} alt="Demo" className={cn("aspect-[4/3] w-full rounded-lg bg-well object-contain", className)} />;
  return <div className={cn("flex aspect-[4/3] w-full items-center justify-center rounded-lg bg-well text-xs text-ink-4", className)}>No video for this demo</div>;
}

export function EpisodeThumb({ id, active, ok, onClick }: { id: string; active?: boolean; ok?: boolean | null; onClick?: () => void }) {
  const m = useEpisodeMedia(id);
  const thumb = resolveMediaUrl(m.data?.thumb_url);
  return (
    <button type="button" onClick={onClick} className={cn("relative aspect-[4/3] overflow-hidden rounded-md bg-well", active ? "ring-2 ring-servo" : "hover:opacity-90")}>
      {thumb ? <img src={thumb} alt="" loading="lazy" className="h-full w-full object-cover" /> : <span className="skeleton absolute inset-0" />}
      {ok !== undefined && ok !== null && (
        <span className={cn("absolute right-1 top-1 flex h-4 w-4 items-center justify-center rounded-full", ok ? "bg-pass text-white" : "bg-fault text-white")}>
          {ok ? <Check className="h-2.5 w-2.5" /> : <X className="h-2.5 w-2.5" />}
        </span>
      )}
    </button>
  );
}

// ------------------------------------------------------------------ approvals
/** Training approval: what it does, what it costs, one decision. Approve = approve + launch + tell the agent. */
export function TrainingApproval({ pid, tj, large }: { pid: string; tj: TrainingJob; large?: boolean }) {
  const qc = useQueryClient();
  const chat = useChat();
  const [skipped, setSkipped] = useState(false);
  const m = useMutation({
    mutationFn: async () => {
      await api.approveTraining(tj.id, "you");
      await api.launchTraining(tj.id);
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["training", pid] });
      chat.send(`Approved training ${tj.id}`);
    },
  });
  if (skipped)
    return (
      <div className="flex items-center gap-2 rounded-xl px-4 py-2.5 text-sm text-ink-3 shadow-[0_0_0_1px_var(--line)]">
        Training not started.
        <button type="button" className="text-ink-2 underline underline-offset-2 hover:text-ink" onClick={() => setSkipped(false)}>
          Review again
        </button>
      </div>
    );
  return (
    <CardShell pid={pid} refId={tj.id} large={large} tone="approval" icon={<Dumbbell className="h-4 w-4" />} title="Start training?" meta="Needs your OK before anything runs">
      <p className="text-sm leading-relaxed text-ink-2">
        Train an <span className="font-medium text-ink">{tj.policy_type.toUpperCase()}</span> policy on dataset{" "}
        <span className="font-mono text-[13px] text-ink">{tj.dataset_version_id}</span>
        {typeof tj.config.steps === "number" ? ` for ${(tj.config.steps as number).toLocaleString()} steps` : ""}.
      </p>
      <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-sm">
        <span className="text-ink-2">{tj.estimate.gpu}</span>
        <span className="text-ink-2">~{num(tj.estimate.gpu_hours, 1)} GPU-hours</span>
        <span className={cn("font-medium", tj.estimate.paid ? "text-caution" : "text-pass")}>{tj.estimate.paid ? `${usd(tj.estimate.est_cost_usd)} estimated` : "No charge"}</span>
      </div>
      {tj.backend === "manual" && <p className="mt-1.5 text-xs text-ink-3">No GPU backend is connected, so starting gives you a command bundle to run on a GPU machine.</p>}
      <div className="mt-3.5 flex items-center gap-2">
        <button
          type="button"
          disabled={m.isPending}
          onClick={() => m.mutate()}
          className="inline-flex h-9 items-center gap-2 rounded-lg bg-servo px-4 text-sm font-medium text-servo-ink transition-transform active:scale-[0.97] disabled:opacity-60"
        >
          {m.isPending ? <CircleDashed className="h-4 w-4 animate-spin" /> : <Check className="h-4 w-4" />} Approve & start
        </button>
        <button type="button" onClick={() => setSkipped(true)} className="h-9 rounded-lg px-3 text-sm text-ink-2 hover:bg-rail">
          Not now
        </button>
      </div>
      {m.error && <p className="mt-2 text-xs text-fault">{errorMessage(m.error)}</p>}
    </CardShell>
  );
}

/** An approval_required item: training resolves to the proposed job; evaluation launches from args. */
export function ApprovalItem({ pid, kind, refId, summary, args }: { pid: string; kind: string; refId: string; summary: string; args: Record<string, unknown> }) {
  const training = useTraining(pid);
  const chat = useChat();
  const qc = useQueryClient();
  const [skipped, setSkipped] = useState(false);
  const evalM = useMutation({
    mutationFn: () => {
      const policy = String(args.policy ?? "scripted_expert");
      return api.launchEvaluation(pid, {
        policy: policy === "scripted_expert" ? { kind: "scripted_expert" } : { kind: "checkpoint", checkpoint_id: policy },
        episodes_per_branch: typeof args.episodes_per_branch === "number" ? args.episodes_per_branch : 10,
      });
    },
    onSuccess: (r) => {
      qc.invalidateQueries({ queryKey: ["evaluations", pid] });
      chat.send(`Approved the evaluation (${r.run.id}). Please continue.`);
    },
  });

  if (kind === "evaluation") {
    if (skipped) return null;
    return (
      <CardShell pid={pid} tone="approval" icon={<BarChart3 className="h-4 w-4" />} title="Run the 3-condition test?" meta="Needs your OK before anything runs">
        <p className="text-sm leading-relaxed text-ink-2">{summary || "Evaluate the policy in simulation: baseline, shifted object and changed lighting."}</p>
        <p className="mt-1 text-xs text-ink-3">Runs locally in simulation · no charge</p>
        <div className="mt-3.5 flex items-center gap-2">
          <button type="button" disabled={evalM.isPending || evalM.isSuccess} onClick={() => evalM.mutate()} className="inline-flex h-9 items-center gap-2 rounded-lg bg-servo px-4 text-sm font-medium text-servo-ink active:scale-[0.97] disabled:opacity-60">
            {evalM.isPending ? <CircleDashed className="h-4 w-4 animate-spin" /> : <Check className="h-4 w-4" />} {evalM.isSuccess ? "Started" : "Approve & start"}
          </button>
          {!evalM.isSuccess && (
            <button type="button" onClick={() => setSkipped(true)} className="h-9 rounded-lg px-3 text-sm text-ink-2 hover:bg-rail">
              Not now
            </button>
          )}
        </div>
        {evalM.error && <p className="mt-2 text-xs text-fault">{errorMessage(evalM.error)}</p>}
      </CardShell>
    );
  }

  // training: ref_id may be the job itself or (Claude Code engine) just the project
  if (training.isLoading) return <Loading />;
  const tj =
    training.data?.find((t) => t.id === refId) ??
    [...(training.data ?? [])].filter((t) => t.status === "proposed").sort((a, b) => (a.created_at < b.created_at ? 1 : -1))[0];
  if (!tj) return <div className="rounded-xl px-4 py-2.5 text-sm text-ink-3 shadow-[0_0_0_1px_var(--line)]">{summary || "An approval was requested"}. Nothing is waiting now.</div>;
  if (tj.status !== "proposed")
    return (
      <div className="flex items-center gap-2 rounded-xl px-4 py-2.5 text-sm text-ink-2 shadow-[0_0_0_1px_var(--line)]">
        <Check className="h-4 w-4 text-pass" /> Training {tj.status === "approved" ? "approved" : tj.status}
        {tj.approved_by ? ` by ${tj.approved_by}` : ""}
      </div>
    );
  return <TrainingApproval pid={pid} tj={tj} />;
}

/** The planner's standing budget ran out: offer to raise it, then tell the agent to continue. */
export function BudgetCard({ pid, message }: { pid: string; message: string }) {
  const b = usePlannerBudget(pid);
  const chat = useChat();
  const qc = useQueryClient();
  const target = Math.ceil((b.data?.budget_usd ?? 10) + 10);
  const m = useMutation({
    mutationFn: () => api.setAutopilotPolicy(pid, { model_budget_usd: target }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["planner-budget", pid] });
      chat.send(`I raised the planner budget to $${target}. Please continue.`);
    },
  });
  return (
    <CardShell pid={pid} tone="approval" icon={<Wallet className="h-4 w-4" />} title={`Raise planner budget to $${target}?`} meta="The agent's model budget for this project is used up">
      <p className="text-sm text-ink-2">{message}</p>
      {b.data && (
        <p className="mt-1 text-xs text-ink-3">
          ${b.data.spent_usd.toFixed(2)} of ${b.data.budget_usd.toFixed(2)} spent
        </p>
      )}
      <div className="mt-3.5 flex items-center gap-2">
        <button type="button" disabled={m.isPending || m.isSuccess} onClick={() => m.mutate()} className="inline-flex h-9 items-center gap-2 rounded-lg bg-servo px-4 text-sm font-medium text-servo-ink active:scale-[0.97] disabled:opacity-60">
          {m.isPending ? <CircleDashed className="h-4 w-4 animate-spin" /> : <Check className="h-4 w-4" />} {m.isSuccess ? "Raised" : `Raise to $${target}`}
        </button>
      </div>
      {m.error && <p className="mt-2 text-xs text-fault">{errorMessage(m.error)}</p>}
    </CardShell>
  );
}
