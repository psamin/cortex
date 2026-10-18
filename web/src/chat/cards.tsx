import { useMemo, useState, type ReactNode } from "react";
import { Link } from "react-router-dom";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import {
  ArrowUpRight,
  Check,
  CircleDashed,
  Play,
  ShieldCheck,
  Sparkles,
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
  useProjectState,
  useRecommendations,
  useTraining,
} from "../api/hooks";
import { agentLabel, isAgent, isImported, type TrainingJob } from "../api/types";
import { Skeleton, InlineError, InfoTip } from "../components/ui";
import { refRoute } from "../components/refs";
import { cn, num, pct, usd } from "../lib/format";
import { IsoIcon } from "../sketch";
import { useChat } from "./ChatProvider";
import { useViewerFocus } from "./focus";
import { AUG_NOTE, datasetBreakdown } from "../lib/augment";
import { prefixOf } from "./model";
import { GlowEdge } from "../components/ui/spotlight-card";

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
        "enter sk-depth relative rounded bg-plate border",
        tone === "approval" ? "border-servo/60" : "border-line-strong",
      )}
    >
      <div className={cn("flex items-center gap-2.5", large ? "px-5 pt-4" : "px-4 pt-3")}>
        <span className={cn("flex shrink-0 items-center justify-center", large ? "h-8 w-8" : "h-7 w-7", tone === "approval" ? "text-servo" : "text-ink-2")}>
          {icon}
        </span>
        <div className="min-w-0 flex-1">
          <div className={cn("truncate font-medium text-ink", large ? "text-[15px]" : "text-sm")}>{title}</div>
          {meta && <div className={cn("mt-0.5 truncate font-mono text-[10px] uppercase tracking-[0.14em]", tone === "approval" ? "text-servo" : "text-ink-3")}>{meta}</div>}
        </div>
        {status}
        {to && (
          <Link to={to} className="rounded-full p-1 text-ink-3 hover:bg-rail hover:text-ink" title="Open details" aria-label="Open details">
            <ArrowUpRight className="h-4 w-4" />
          </Link>
        )}
      </div>
      <div className={cn(large ? "px-5 pb-5 pt-3" : "px-4 pb-4 pt-3")}>{children}</div>
    </div>
  );
}

function Pill({ tone, children }: { tone: "live" | "ok" | "bad" | "muted" | "wait"; children: ReactNode }) {
  const cls = {
    live: "border-servo/50 text-servo",
    ok: "border-pass/40 text-pass",
    bad: "border-fault/40 text-fault",
    muted: "border-line-strong text-ink-3",
    wait: "border-caution/45 text-caution",
  }[tone];
  return (
    <span className={cn("inline-flex h-6 shrink-0 items-center gap-1.5 rounded-full border px-2.5 font-mono text-[10px] font-medium uppercase tracking-[0.12em]", cls)}>
      {tone === "live" && <span className="h-1.5 w-1.5 rounded-full bg-current live-dot" />}
      {children}
    </span>
  );
}

function Loading({ large }: { large?: boolean }) {
  return (
    <div className={cn("space-y-2 rounded border border-line p-4", large && "p-5")}>
      <Skeleton className="h-4 w-40" />
      <Skeleton className="h-10 w-full" />
    </div>
  );
}

function Missing({ what }: { what: string }) {
  return <div className="rounded border border-line px-4 py-3 text-xs text-ink-3">Couldn't load {what}.</div>;
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
function CollectionCard({ pid, id }: { pid: string; id: string; large?: boolean }) {
  const first = useAgentCollection(id);
  const running = !!first.data && ["queued", "running"].includes(first.data.collection.status);
  const q = useAgentCollection(id, running);
  if (q.isLoading) return <Loading />;
  if (!q.data) return <Missing what="this collection" />;
  const { collection: c, attempts } = q.data;
  const t = c.totals;
  const who = isAgent(c.collector) ? agentLabel(c.model) : "Scripted oracle";
  const demos = attempts.filter((a) => a.episode_id).map((a) => a.episode_id!);
  const done = t.attempts ?? attempts.length;
  return (
    <DemoRow
      pid={pid}
      refId={id}
      live={running}
      text={`${t.successes ?? 0} demo${t.successes === 1 ? "" : "s"} · ${who} · ${t.successes ?? 0}/${done} ok${running ? ` · ${done}/${c.request.num_demos} tries` : ""}`}
      ids={demos}
    />
  );
}

/** One compact line for demos in the chat; the videos live in the viewer. */
function DemoRow({ pid, refId, text, ids, live }: { pid: string; refId: string; text: string; ids: string[]; live?: boolean }) {
  const viewer = useViewerFocus();
  const to = refRoute(pid, refId);
  return (
    <div className="enter flex items-center gap-2.5 sk-depth relative rounded border border-line-strong bg-plate px-3 py-2">
      <GlowEdge />
      {live ? <CircleDashed className="h-4 w-4 shrink-0 animate-spin text-servo" /> : <Play className="h-4 w-4 shrink-0 text-ink-3" />}
      <span className="min-w-0 flex-1 truncate text-sm text-ink-2" title={text}>
        {text}
      </span>
      {viewer && ids.length > 0 && (
        <button type="button" onClick={() => viewer.focus(ids)} className="shrink-0 rounded-md px-2 py-1 text-xs font-medium text-servo hover:bg-rail">
          Watch
        </button>
      )}
      {to && (
        <Link to={to} className="shrink-0 rounded-md p-1 text-ink-4 hover:bg-rail hover:text-ink-2" title="Open details" aria-label="Open details">
          <ArrowUpRight className="h-4 w-4" />
        </Link>
      )}
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
  const label = run.policy_label || (run.policy.kind === "scripted_expert" ? "Scripted expert (oracle)" : run.policy.checkpoint_id ?? "policy");
  return (
    <CardShell
      pid={pid}
      refId={id}
      large={large}
      icon={<IsoIcon name="evaluate" size={22} />}
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
                  <span className="text-xs text-ink-3">{b.status === "running" || b.status === "queued" ? "running…" : "no result"}</span>
                )}
              </div>
              <div className={cn("relative overflow-hidden bg-[var(--line)]", large ? "h-2" : "h-1.5")}>
                {r ? (
                  <>
                    <div className="absolute inset-y-0 bg-ink-4/40" style={{ left: `${r.ci95[0] * 100}%`, width: `${(r.ci95[1] - r.ci95[0]) * 100}%` }} title="95% confidence interval" />
                    <div className={cn("absolute inset-y-0 left-0", r.success_rate >= 0.8 ? "bg-pass" : r.success_rate >= 0.5 ? "bg-servo" : "bg-fault")} style={{ width: `${r.success_rate * 100}%`, opacity: 0.9 }} />
                  </>
                ) : (
                  <div className="skeleton absolute inset-0" />
                )}
              </div>
            </div>
          );
        })}
      </div>
      <div className="mt-2.5 flex items-center gap-1.5 font-mono text-[10px] uppercase tracking-[0.12em] text-ink-3">
        grey = 95% CI <InfoTip>Grey band = 95% confidence interval. Small samples give wide bands.</InfoTip>
      </div>
    </CardShell>
  );
}

// ------------------------------------------------------------------ dataset (ds_)
function DatasetCard({ pid, id, large }: { pid: string; id: string; large?: boolean }) {
  const q = useDatasets(pid);
  const episodes = useEpisodes(pid);
  if (q.isLoading) return <Loading large={large} />;
  const d = q.data?.find((x) => x.id === id);
  if (!d) return <Missing what="this dataset" />;
  const b = datasetBreakdown(d, new Map((episodes.data ?? []).map((e) => [e.id, e])));
  return (
    <CardShell pid={pid} refId={id} large={large} icon={<IsoIcon name="dataset" size={22} />} title={`Dataset ${d.name}`} meta={`LeRobot · ${d.num_frames.toLocaleString()} frames`} status={<Pill tone="ok">built</Pill>}>
      <p className="text-sm text-ink-2">{b.text}</p>
      {b.hasAug && <p className="mt-1 text-xs text-ink-3">{AUG_NOTE}</p>}
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
      icon={<IsoIcon name="train" size={22} />}
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
          // a failed run reads "stopped": its error stays in the backend logs (#328)
          <Pill tone="muted">{t.status === "failed" ? "stopped" : t.status}</Pill>
        )
      }
    >
      <div className="text-sm text-ink-2">{estimateLine(t)}</div>
      {t.checkpoint_ids.length > 0 && <div className="mt-1 text-xs text-ink-3">{t.checkpoint_ids.length} checkpoint{t.checkpoint_ids.length === 1 ? "" : "s"}</div>}
      {t.backend === "manual" && t.status !== "succeeded" && (
        <div className="mt-1 text-xs text-ink-3">No GPU connected · command in Improve › Training</div>
      )}
    </CardShell>
  );
}

/** "Futurama general · gpu:l40s:1 · up to 120 min" from config.slurm (chosen in Setup); null for older jobs. */
const slurmLine = (t: TrainingJob) => {
  const s = t.config.slurm as { partition?: string; gres?: string; time_min?: number } | undefined;
  return s?.gres ? `Futurama ${s.partition ?? ""} · ${s.gres}${s.time_min ? ` · up to ${s.time_min} min` : ""}`.replace("  ", " ") : null;
};

const trainingWhat = (t: TrainingJob) => {
  const steps = typeof t.config.steps === "number" ? `${(t.config.steps as number).toLocaleString()} steps` : null;
  return [`dataset ${t.dataset_version_id}`, steps].filter(Boolean).join(" · ");
};
const estimateLine = (t: TrainingJob) =>
  `${t.estimate.gpu} · ~${num(t.estimate.gpu_hours, 1)} GPU-h · ${t.estimate.paid ? `${usd(t.estimate.est_cost_usd)} estimate` : "no charge"}`;

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
    <CardShell pid={pid} refId={id} large={large} icon={<IsoIcon name="check" size={22} />} title="Quality check" meta={`${a.episode_count} demos`} status={a.status === "running" ? <Pill tone="live">checking</Pill> : <Pill tone="ok">done</Pill>}>
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
        <p className="mt-1 text-xs text-ink-3">No flags.</p>
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
    <CardShell pid={pid} refId={list[0].id} large={large} icon={<IsoIcon name="improve" size={22} />} title="Next steps" meta="by expected gain">
      <ol className="space-y-2">
        {list.map((r, i) => (
          <li key={r.id} className="flex items-start gap-2.5">
            <span className="num mt-0.5 w-4 shrink-0 font-mono text-xs text-ink-4">{i + 1}</span>
            <div className="min-w-0 flex-1">
              <div className="text-sm text-ink">{r.title}</div>
              <div className="mt-0.5 flex items-center gap-2 text-xs">
                {r.support === "evidence_backed" ? (
                  <span className="inline-flex items-center gap-1 text-pass">
                    <ShieldCheck className="h-3 w-3" /> measured
                  </span>
                ) : (
                  <span className="inline-flex items-center gap-1 text-ink-3">
                    <Sparkles className="h-3 w-3" /> unverified hypothesis
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
  if (!j) {
    if (q.isLoading) return <Loading large={large} />;
    if (q.isError) return <p className="py-1 text-xs text-ink-3">Job record gone (ran before job history was kept).</p>;
    return null;
  }
  if (j.status === "succeeded" && j.result_ref && prefixOf(j.result_ref) !== "job") return <ResultCard pid={pid} refId={j.result_ref} large={large} />;
  const active = isJobActive(j);
  return (
    <CardShell
      pid={pid}
      large={large}
      icon={active ? <CircleDashed className="h-4 w-4 animate-spin" /> : <IsoIcon name="gpu" size={22} />}
      title={j.message || j.kind.replace(/_/g, " ")}
      meta={j.kind.replace(/_/g, " ")}
      status={active ? <Pill tone="live">{pct(j.progress)}</Pill> : j.status === "succeeded" ? <Pill tone="ok">done</Pill> : <Pill tone="muted">{j.status === "failed" ? "stopped" : j.status}</Pill>}
    >
      <div className="h-1 overflow-hidden bg-[var(--line)]">
        <div className="h-full bg-servo transition-[width] duration-500 ease-out" style={{ width: `${j.progress * 100}%` }} />
      </div>
    </CardShell>
  );
}

// ------------------------------------------------------------------ episodes (ep_)
function EpisodesCard({ pid, ids }: { pid: string; ids: string[]; large?: boolean }) {
  const episodes = useEpisodes(pid);
  const eps = (episodes.data ?? []).filter((e) => ids.includes(e.id));
  const first = eps[0];
  const who = !first ? "" : isImported(first) ? "Human (imported)" : first.source === "real_teleop" ? "Human (real teleop)" : first.source === "sim_scripted" ? "Scripted oracle" : agentLabel(first.provenance?.planner);
  const judged = eps.filter((e) => e.success !== null);
  const ok = judged.filter((e) => e.success).length;
  const n = ids.length;
  const text = [`${n} demo${n === 1 ? "" : "s"}`, who, judged.length ? `${ok}/${judged.length} succeeded` : first ? "not judged by the simulator" : ""].filter(Boolean).join(" · ");
  return <DemoRow pid={pid} refId={ids[0]} text={text} ids={ids} />;
}

export function EpisodeVideo({ id, className }: { id: string; className?: string }) {
  const m = useEpisodeMedia(id);
  const video = resolveMediaUrl(m.data?.video_url);
  const thumb = resolveMediaUrl(m.data?.thumb_url);
  if (m.isLoading) return <Skeleton className={cn("aspect-[4/3] w-full rounded-sm", className)} />;
  if (video) return <video key={video} src={video} poster={thumb ?? undefined} controls autoPlay muted loop playsInline className={cn("aspect-[4/3] w-full rounded-sm bg-well object-contain", className)} />;
  if (thumb) return <img src={thumb} alt="Demo" className={cn("aspect-[4/3] w-full rounded-sm bg-well object-contain", className)} />;
  return <div className={cn("flex aspect-[4/3] w-full items-center justify-center rounded-sm bg-well text-xs text-ink-4", className)}>No video for this demo</div>;
}

export function EpisodeThumb({ id, active, ok, onClick }: { id: string; active?: boolean; ok?: boolean | null; onClick?: () => void }) {
  const m = useEpisodeMedia(id);
  const thumb = resolveMediaUrl(m.data?.thumb_url);
  return (
    <button type="button" onClick={onClick} className={cn("relative aspect-[4/3] overflow-hidden rounded-sm border bg-well", active ? "border-servo ring-1 ring-servo" : "border-line hover:border-ink")}>
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
  // autonomous mode starts covered training by itself, so a card that still asks is over the standing approval
  const autonomous = useProjectState(pid).data?.project.autopilot.autonomy !== "manual";
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
      <div className="flex items-center gap-2 rounded border border-line px-4 py-2.5 text-sm text-ink-3">
        Not started.
        <button type="button" className="border-b border-line-strong text-ink-2 hover:border-ink hover:text-ink" onClick={() => setSkipped(false)}>
          Review
        </button>
      </div>
    );
  return (
    <CardShell pid={pid} refId={tj.id} large={large} tone="approval" icon={<IsoIcon name="approve" size={22} accent />} title="Start training?" meta={autonomous ? "Needs you · over your standing approval" : "Needs approval"}>
      <p className="text-sm leading-relaxed text-ink-2">
        <span className="font-medium text-ink">{tj.policy_type.toUpperCase()}</span> on{" "}
        <span className="font-mono text-[13px] text-ink">{tj.dataset_version_id}</span>
        {typeof tj.config.steps === "number" ? ` · ${(tj.config.steps as number).toLocaleString()} steps` : ""}
      </p>
      <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-sm">
        <span className="text-ink-2">{tj.estimate.gpu}</span>
        <span className="text-ink-2">~{num(tj.estimate.gpu_hours, 1)} GPU-hours</span>
        <span className={cn("font-medium", tj.estimate.paid ? "text-caution" : "text-pass")}>{tj.estimate.paid ? `${usd(tj.estimate.est_cost_usd)} estimate` : "No charge"}</span>
      </div>
      {slurmLine(tj) && <p className="mt-1.5 text-xs text-ink-3">{slurmLine(tj)}</p>}
      {tj.backend === "manual" && <p className="mt-1.5 text-xs text-ink-3">No GPU connected · you get a command to run</p>}
      <div className="mt-3.5 flex items-center gap-2">
        <button
          type="button"
          disabled={m.isPending}
          onClick={() => m.mutate()}
          className="sk-pill sk-pill-primary inline-flex h-9 items-center gap-2 px-5 text-[13px] font-medium disabled:opacity-60"
        >
          {m.isPending ? <CircleDashed className="h-4 w-4 animate-spin" /> : <Check className="h-4 w-4" />} Approve & start
        </button>
        <button type="button" onClick={() => setSkipped(true)} className="mx-2 h-9 text-[13px] text-ink-2 hover:text-ink">
          Not now
        </button>
      </div>
      {m.error && <InlineError error={m.error} className="mt-2" />}
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
      <CardShell pid={pid} tone="approval" icon={<IsoIcon name="approve" size={22} accent />} title="Run 3-condition test?" meta="Needs approval">
        <p className="text-sm leading-relaxed text-ink-2">{summary || "Baseline, shifted object, changed lighting."}</p>
        <p className="mt-1 text-xs text-ink-3">Local sim · no charge</p>
        <div className="mt-3.5 flex items-center gap-2">
          <button type="button" disabled={evalM.isPending || evalM.isSuccess} onClick={() => evalM.mutate()} className="sk-pill sk-pill-primary inline-flex h-9 items-center gap-2 px-5 text-[13px] font-medium disabled:opacity-60">
            {evalM.isPending ? <CircleDashed className="h-4 w-4 animate-spin" /> : <Check className="h-4 w-4" />} {evalM.isSuccess ? "Started" : "Approve & start"}
          </button>
          {!evalM.isSuccess && (
            <button type="button" onClick={() => setSkipped(true)} className="mx-2 h-9 text-[13px] text-ink-2 hover:text-ink">
              Not now
            </button>
          )}
        </div>
        {evalM.error && <InlineError error={evalM.error} className="mt-2" />}
      </CardShell>
    );
  }

  // training: ref_id may be the job itself or (Claude Code engine) just the project
  if (training.isLoading) return <Loading />;
  const tj =
    training.data?.find((t) => t.id === refId) ??
    [...(training.data ?? [])].filter((t) => t.status === "proposed").sort((a, b) => (a.created_at < b.created_at ? 1 : -1))[0];
  if (!tj) return <div className="rounded border border-line px-4 py-2.5 text-sm text-ink-3">{summary || "Approval requested"}. Nothing pending.</div>;
  if (tj.status !== "proposed")
    return (
      <div className="flex items-center gap-2 rounded border border-line px-4 py-2.5 text-sm text-ink-2">
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
    <CardShell pid={pid} tone="approval" icon={<IsoIcon name="approve" size={22} accent />} title={`Raise planner budget to $${target}?`} meta="Needs you · planner budget used up">
      <p className="text-sm text-ink-2">{message}</p>
      {b.data && (
        <p className="mt-1 text-xs text-ink-3">
          ${b.data.spent_usd.toFixed(2)} of ${b.data.budget_usd.toFixed(2)} spent
        </p>
      )}
      <div className="mt-3.5 flex items-center gap-2">
        <button type="button" disabled={m.isPending || m.isSuccess} onClick={() => m.mutate()} className="sk-pill sk-pill-primary inline-flex h-9 items-center gap-2 px-5 text-[13px] font-medium disabled:opacity-60">
          {m.isPending ? <CircleDashed className="h-4 w-4 animate-spin" /> : <Check className="h-4 w-4" />} {m.isSuccess ? "Raised" : `Raise to $${target}`}
        </button>
      </div>
      {m.error && <InlineError error={m.error} className="mt-2" />}
    </CardShell>
  );
}
