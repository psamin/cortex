import { useEffect, useMemo, useState } from "react";
import { useParams, useSearchParams } from "react-router-dom";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Bot, Check, Circle, Cog, Dices, Layers, RotateCcw, Square, VideoOff, X } from "lucide-react";
import { api } from "../api/client";
import { isJobActive, useJob, useJobs, useTrackJob } from "../api/events";
import { useEpisodeThumbs, useEpisodes, usePlanner, useProjectState, useRecommendations } from "../api/hooks";
import { CUBE_REGIONS, LIGHTINGS, conditionKey, type Condition, type Episode, type Job, type PlanCell } from "../api/types";
import {
  Badge,
  Button,
  Dialog,
  EmptyState,
  ErrorState,
  Field,
  Input,
  Panel,
  QueryState,
  Segmented,
  SkeletonRows,
  StatusBadge,
} from "../components/ui";
import { ConditionLabel, WorkcellGrid } from "../components/Workcell";
import { SupportBadge, EvidenceList } from "./improve/Recommendations";
import { AgentCollect } from "./AgentCollect";
import { AttemptDrawer, EpisodeProvenance } from "../components/AgentTrace";
import { resolveMediaUrl } from "../api/client";
import { cn, errorMessage, pct } from "../lib/format";
import { SceneView } from "../sim3d/SceneView";

const randSeed = () => Math.floor(Math.random() * 100_000);

type CollectMode = "agent" | "oracle" | "manual";

export function CollectPage() {
  const { pid = "" } = useParams();
  const [params, setParams] = useSearchParams();
  const planner = usePlanner();
  const state = useProjectState(pid);
  const raw = params.get("mode") as CollectMode | null;
  // Agent is the default only when the planner is actually available.
  const mode: CollectMode | null = raw ?? (planner.isLoading ? null : planner.data?.available ? "agent" : "oracle");
  const setMode = (m: CollectMode) => setParams({ mode: m }, { replace: true });
  const project = state.data?.project;

  return (
    <div className="space-y-4 px-5 py-5">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-xl font-semibold tracking-[-0.01em]">Collect demonstrations</h1>
          <p className="mt-0.5 text-sm text-ink-3">Only simulator-verified successes become LeRobot demonstrations.</p>
        </div>
        {mode && (
          <Segmented
            value={mode}
            onChange={setMode}
            ariaLabel="Collection mode"
            options={[
              { value: "agent", label: <span className="inline-flex items-center gap-1.5 normal-case"><Bot className="h-3.5 w-3.5" /> Agent demonstrations</span> },
              { value: "oracle", label: <span className="inline-flex items-center gap-1.5 normal-case"><Cog className="h-3.5 w-3.5" /> Scripted oracle</span> },
              { value: "manual", label: "Manual" },
            ]}
          />
        )}
      </div>
      {!mode || state.isLoading ? (
        <SkeletonRows rows={4} />
      ) : state.error ? (
        <ErrorState error={state.error} onRetry={() => state.refetch()} />
      ) : mode === "manual" || !project ? (
        <ManualCollect pid={pid} />
      ) : project.mode !== "sim" ? (
        <ErrorState error={new Error("Autonomous collection runs in simulation only. Use Manual for the real robot.")} compact />
      ) : (
        <AgentCollect key={mode} project={project} collector={mode === "agent" ? "astra" : "oracle"} onUseOracle={() => setMode("oracle")} />
      )}
    </div>
  );
}

function ManualCollect({ pid }: { pid: string }) {
  const qc = useQueryClient();
  const state = useProjectState(pid);
  const episodes = useEpisodes(pid);
  const track = useTrackJob();
  const jobs = useJobs();

  const [condition, setCondition] = useState<Condition>({ cube_region: "center", lighting: "normal" });
  const [seed, setSeed] = useState(randSeed);
  const [jobId, setJobId] = useState<string | null>(null);
  const [batchOpen, setBatchOpen] = useState(false);
  const [lastEpisodeId, setLastEpisodeId] = useState<string | null>(null);

  // adopt a recording started elsewhere (agent / MCP) so its live view shows here too
  const foreignActive = useMemo(
    () => Object.values(jobs).find((j) => j.project_id === pid && /record/.test(j.kind) && isJobActive(j)),
    [jobs, pid],
  );
  const job = useJob(jobId ?? foreignActive?.id);
  const recording = isJobActive(job);

  const isReal = state.data?.project.mode === "real";

  // when a recording finishes, surface the new take for review
  useEffect(() => {
    if (job?.status === "succeeded") {
      if (job.result_ref?.startsWith("ep_")) setLastEpisodeId(job.result_ref);
      else if (job.kind === "record" && episodes.data?.length) {
        const newest = [...episodes.data].sort((a, b) => (a.created_at < b.created_at ? 1 : -1))[0];
        setLastEpisodeId(newest.id);
      }
      qc.invalidateQueries({ queryKey: ["episodes", pid] });
      qc.invalidateQueries({ queryKey: ["state", pid] });
    }
  }, [job?.status]); // eslint-disable-line react-hooks/exhaustive-deps

  const record = useMutation({
    mutationFn: (c: Condition) => api.record(pid, c, seed),
    onSuccess: (j) => {
      track(j);
      setJobId(j.id);
      setLastEpisodeId(null);
      setSeed(randSeed());
    },
  });
  const stop = useMutation({
    mutationFn: (id: string) => api.stopRecording(id),
    onSuccess: (j) => track(j),
  });

  return (
    <div className="space-y-4">
      <div className="flex items-end justify-between gap-4">
        <div>
          <p className="text-sm text-ink-3">
            {isReal
              ? "Real robot: every motion needs your approval through the local LeRobot bridge."
              : "Record single demos by hand with the scripted expert, one condition at a time."}
          </p>
        </div>
        <Button onClick={() => setBatchOpen(true)} disabled={!state.data}>
          <Layers className="h-4 w-4" /> Record batch
        </Button>
      </div>

      <div className="grid grid-cols-1 gap-4 min-[1400px]:grid-cols-[minmax(0,1.2fr)_minmax(0,1fr)] 2xl:grid-cols-[minmax(0,1.35fr)_minmax(0,1fr)]">
        {/* viewport + transport */}
        <Panel
          title="Workcell"
          meta={recording ? undefined : `preview · seed ${seed}`}
          actions={
            <>
              {job && <StatusBadge status={job.status} />}
              {recording ? (
                <Button size="sm" variant="danger" loading={stop.isPending} onClick={() => job && stop.mutate(job.id)}>
                  <Square className="h-3 w-3 fill-current" /> Stop
                </Button>
              ) : (
                <Button size="sm" variant="primary" loading={record.isPending} onClick={() => record.mutate(condition)}>
                  <Circle className="h-3 w-3 fill-current" /> Record
                </Button>
              )}
            </>
          }
          bodyClassName="p-0"
        >
          <Viewport pid={pid} condition={condition} seed={seed} job={job} />
          <div className="flex flex-wrap items-center gap-2 border-t border-line-soft px-3 py-2.5">
            <Segmented
              size="sm"
              ariaLabel="Cube region"
              value={condition.cube_region}
              onChange={(v) => setCondition((c) => ({ ...c, cube_region: v }))}
              options={CUBE_REGIONS.map((r) => ({ value: r, label: r }))}
            />
            <Segmented
              size="sm"
              ariaLabel="Lighting"
              value={condition.lighting}
              onChange={(v) => setCondition((c) => ({ ...c, lighting: v }))}
              options={LIGHTINGS.map((l) => ({ value: l, label: l }))}
            />
            <Button size="sm" variant="ghost" onClick={() => setSeed(randSeed())} disabled={recording} title="New random seed">
              <Dices className="h-3.5 w-3.5" />
            </Button>
          </div>
          {(record.error || stop.error) && (
            <div className="px-3 pb-3">
              <ErrorState error={record.error ?? stop.error} compact />
            </div>
          )}
          {job?.status === "failed" && (
            <p className="px-3 pb-3 text-xs text-fault">Recording failed: {job.error ?? job.message}</p>
          )}
          {lastEpisodeId && !recording && (
            <LastTake
              pid={pid}
              episode={episodes.data?.find((e) => e.id === lastEpisodeId)}
              onRetry={() => record.mutate(condition)}
              onDone={() => setLastEpisodeId(null)}
            />
          )}
        </Panel>

        {/* coverage + picker */}
        <div className="space-y-4">
          <Panel title="Plan coverage" meta={coverageMeta(state.data?.project.plan.cells, state.data?.project.plan.budget_total)}>
            <QueryState q={state} skeleton={<SkeletonRows rows={3} />}>
              {(s) => (
                <Coverage cells={s.project.plan.cells} episodes={episodes.data ?? []} selected={condition} onSelect={setCondition} />
              )}
            </QueryState>
          </Panel>
          <SuggestedDemos pid={pid} onRecorded={(j) => (track(j), setJobId(j.id))} />
        </div>
      </div>

      <EpisodeStrip pid={pid} episodes={episodes} highlight={lastEpisodeId} />

      {state.data && (
        <BatchDialog
          open={batchOpen}
          onClose={() => setBatchOpen(false)}
          pid={pid}
          cells={state.data.project.plan.cells}
          onStarted={(j) => {
            track(j);
            setJobId(j.id);
            setBatchOpen(false);
          }}
        />
      )}
    </div>
  );
}

function coverageMeta(cells?: PlanCell[], budget?: number) {
  if (!cells) return undefined;
  const c = cells.reduce((a, x) => a + x.collected, 0);
  return `${c} of ${budget ?? cells.reduce((a, x) => a + x.target, 0)} demos`;
}

// ------------------------------------------------------------------ viewport
function Viewport({ pid, condition, seed, job }: { pid: string; condition: Condition; seed: number; job?: Job }) {
  const live = isJobActive(job);
  const src = live ? api.simLiveUrl(pid) : api.simFrameUrl(pid, condition, seed);
  const [failed, setFailed] = useState<string | null>(null);
  useEffect(() => setFailed(null), [src]);

  return (
    <div className="relative aspect-[4/3] max-h-[440px] w-full overflow-hidden bg-well">
      <SceneView
        pid={pid}
        seed={seed}
        condition={condition}
        live={live}
        camera={failed === src ? (
        <div className="flex h-full flex-col items-center justify-center gap-2 text-ink-4">
          <VideoOff className="h-6 w-6" />
          <p className="text-xs">{live ? "Live stream unavailable" : "Sim render unavailable — is the backend running?"}</p>
        </div>
      ) : (
        <img
          key={src}
          src={src}
          alt={live ? "Live recording" : `Preview: cube ${condition.cube_region}, ${condition.lighting} lighting`}
          onError={() => setFailed(src)}
          className="h-full w-full object-contain"
        />
      )}
      />
      {live && job && (
        <>
          <div className="absolute left-3 top-3 flex items-center gap-1.5 rounded-sm bg-black/60 px-2 py-1 text-2xs font-semibold text-white">
            <span className="h-2 w-2 rounded-full bg-fault live-dot" /> REC
          </div>
          <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/70 to-transparent px-3 pb-2.5 pt-6">
            <div className="mb-1.5 flex justify-between text-2xs text-white/80">
              <span className="truncate">{job.message || "recording…"}</span>
              <span className="num font-mono">{pct(job.progress)}</span>
            </div>
            <div className="h-1 overflow-hidden rounded-full bg-white/20">
              <div className="h-full bg-servo transition-[width] duration-300" style={{ width: `${job.progress * 100}%` }} />
            </div>
          </div>
        </>
      )}
      {!live && (
        <div className="absolute bottom-2.5 left-3 rounded-sm bg-black/55 px-1.5 py-0.5">
          <ConditionLabel c={condition} className="text-white/85" />
        </div>
      )}
    </div>
  );
}

function LastTake({
  pid,
  episode,
  onRetry,
  onDone,
}: {
  pid: string;
  episode?: Episode;
  onRetry: () => void;
  onDone: () => void;
}) {
  const qc = useQueryClient();
  const m = useMutation({
    mutationFn: (s: "accepted" | "rejected") => api.setEpisodeStatus(episode!.id, s),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["episodes", pid] });
      qc.invalidateQueries({ queryKey: ["state", pid] });
      onDone();
    },
  });
  return (
    <div className="enter flex flex-wrap items-center gap-3 border-t border-line-soft bg-servo/[.04] px-3 py-2.5">
      <div className="min-w-0 flex-1">
        <div className="text-xs font-medium text-ink">
          New take{episode ? <span className="ml-1 font-mono text-ink-3">#{episode.index}</span> : null}
        </div>
        {episode ? (
          <div className="mt-0.5 flex flex-wrap items-center gap-1.5">
            <ConditionLabel c={episode.condition} />
            <SuccessBadge s={episode.success} />
            {episode.quality && <QualityBadge score={episode.quality.score} />}
          </div>
        ) : (
          <div className="text-2xs text-ink-3">Loading episode…</div>
        )}
      </div>
      <Button size="sm" variant="ghost" onClick={onRetry}>
        <RotateCcw className="h-3.5 w-3.5" /> Retry
      </Button>
      <Button size="sm" variant="danger" disabled={!episode} loading={m.isPending && m.variables === "rejected"} onClick={() => m.mutate("rejected")}>
        <X className="h-3.5 w-3.5" /> Reject
      </Button>
      <Button size="sm" variant="primary" disabled={!episode} loading={m.isPending && m.variables === "accepted"} onClick={() => m.mutate("accepted")}>
        <Check className="h-3.5 w-3.5" /> Accept
      </Button>
      {m.error && <p className="w-full text-xs text-fault">{errorMessage(m.error)}</p>}
    </div>
  );
}

// ------------------------------------------------------------------ coverage
function Coverage({
  cells,
  episodes,
  selected,
  onSelect,
}: {
  cells: PlanCell[];
  episodes: Episode[];
  selected: Condition;
  onSelect: (c: Condition) => void;
}) {
  const byKey = new Map(cells.map((c) => [conditionKey(c.condition), c]));
  // quality-weighted: sum of measured quality scores of non-rejected episodes in the cell
  const qw = new Map<string, { sum: number; scored: number }>();
  for (const e of episodes) {
    if (e.status === "rejected") continue;
    const k = conditionKey(e.condition);
    const cur = qw.get(k) ?? { sum: 0, scored: 0 };
    if (e.quality) {
      cur.sum += e.quality.score;
      cur.scored += 1;
    }
    qw.set(k, cur);
  }
  return (
    <>
      <WorkcellGrid
        selected={selected}
        onSelect={onSelect}
        renderCell={(c) => {
          const k = conditionKey(c);
          const cell = byKey.get(k);
          const q = qw.get(k);
          const target = cell?.target ?? 0;
          const collected = cell?.collected ?? 0;
          const frac = target ? Math.min(1, collected / target) : 0;
          const qFrac = target && q ? Math.min(1, q.sum / target) : 0;
          return (
            <div className="px-2 py-1.5">
              <div className="flex items-baseline gap-1">
                <span className={cn("num text-base font-semibold", target && collected >= target ? "text-pass" : "text-ink")}>
                  {collected}
                </span>
                <span className="num text-2xs text-ink-3">/ {target}</span>
                {q && q.scored > 0 && (
                  <span className="num ml-auto font-mono text-[10px] text-ink-3" title="Σ quality score of non-rejected episodes">
                    q{q.sum.toFixed(1)}
                  </span>
                )}
              </div>
              {/* two-layer bar: raw count (faint) and quality-weighted (solid) */}
              <div className="relative mt-1 h-1 overflow-hidden rounded-full bg-rail">
                <div className="absolute inset-y-0 left-0 bg-servo/30" style={{ width: `${frac * 100}%` }} />
                <div className="absolute inset-y-0 left-0 bg-servo" style={{ width: `${qFrac * 100}%` }} />
              </div>
            </div>
          );
        }}
      />
      <div className="mt-2.5 flex items-center gap-3 text-2xs text-ink-3">
        <span className="flex items-center gap-1">
          <span className="h-1 w-3 rounded-full bg-servo/30" /> collected
        </span>
        <span className="flex items-center gap-1">
          <span className="h-1 w-3 rounded-full bg-servo" /> quality-weighted
        </span>
        <span className="ml-auto">Click a cell to set the recording condition</span>
      </div>
    </>
  );
}

// ------------------------------------------------------------------ suggestions
function SuggestedDemos({ pid, onRecorded }: { pid: string; onRecorded: (j: Job) => void }) {
  const state = useRecommendations(pid);
  const recs = (state.data ?? []).filter((r) => r.action === "collect" && r.condition && r.status === "proposed");
  const m = useMutation({
    mutationFn: ({ c, n }: { c: Condition; n: number }) => api.recordBatch(pid, [c], Math.max(1, n)),
    onSuccess: onRecorded,
  });
  return (
    <Panel title="Suggested next demos" meta="from recommendations">
      {state.isLoading ? (
        <SkeletonRows rows={2} />
      ) : recs.length === 0 ? (
        <p className="text-xs text-ink-3">
          No collection suggestions yet. Record a few demos and run analysis, or refresh recommendations in Improve.
        </p>
      ) : (
        <ul className="space-y-2.5">
          {recs.slice(0, 3).map((r) => (
            <li key={r.id} className={cn("rounded p-2.5", r.support === "hypothesis" ? "prov-hypothesis" : "prov-measured")}>
              <div className="flex items-start gap-2">
                <div className="min-w-0 flex-1">
                  <div className="text-sm font-medium text-ink">{r.title}</div>
                  <div className="mt-0.5 flex flex-wrap items-center gap-1.5">
                    {r.condition && <ConditionLabel c={r.condition} />}
                    <SupportBadge support={r.support} />
                  </div>
                </div>
                <Button
                  size="sm"
                  variant="primary"
                  loading={m.isPending && m.variables?.c === r.condition}
                  onClick={() => r.condition && m.mutate({ c: r.condition, n: r.count })}
                >
                  Record {r.count || 1}
                </Button>
              </div>
              <div className="mt-2">
                <div className="mb-1 text-2xs font-medium text-ink-3">Why</div>
                <EvidenceList evidence={r.evidence} pid={pid} compact />
              </div>
            </li>
          ))}
        </ul>
      )}
      {m.error && <p className="mt-2 text-xs text-fault">{errorMessage(m.error)}</p>}
    </Panel>
  );
}

// ------------------------------------------------------------------ episodes
export function QualityBadge({ score }: { score: number }) {
  const tone = score >= 0.8 ? "pass" : score >= 0.5 ? "caution" : "fault";
  return (
    <Badge tone={tone} title="Measured quality score (1 = no issues)">
      <span className="num font-mono">q {score.toFixed(2)}</span>
    </Badge>
  );
}

export function SuccessBadge({ s }: { s: boolean | null }) {
  if (s === null) return <Badge title="Not labelled">success ?</Badge>;
  return s ? <Badge tone="pass">success</Badge> : <Badge tone="fault">failed</Badge>;
}

function EpisodeStrip({
  pid,
  episodes,
  highlight,
}: {
  pid: string;
  episodes: ReturnType<typeof useEpisodes>;
  highlight: string | null;
}) {
  const sorted = useMemo(
    () => [...(episodes.data ?? [])].sort((a, b) => b.index - a.index),
    [episodes.data],
  );
  const thumbs = useEpisodeThumbs(sorted);
  const [trace, setTrace] = useState<string | null>(null);
  const qc = useQueryClient();
  const acceptPassing = useMutation({
    mutationFn: () => api.acceptPassing(pid),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["episodes", pid] });
      qc.invalidateQueries({ queryKey: ["state", pid] });
    },
  });
  const pending = sorted.filter((e) => e.status === "recorded").length;
  const counts = {
    accepted: sorted.filter((e) => e.status === "accepted").length,
    rejected: sorted.filter((e) => e.status === "rejected").length,
  };
  return (
    <Panel
      title="Episodes"
      meta={episodes.data ? `${sorted.length} recorded · ${counts.accepted} accepted · ${counts.rejected} rejected` : undefined}
      actions={
        <>
          {acceptPassing.data && <span className="text-2xs text-ink-3">accepted {acceptPassing.data.accepted}</span>}
          <Button
            size="sm"
            disabled={pending === 0}
            loading={acceptPassing.isPending}
            onClick={() => acceptPassing.mutate()}
            title="Accept every unreviewed episode that succeeded and passed the quality checks"
          >
            <Check className="h-3.5 w-3.5" /> Accept passing ({pending})
          </Button>
        </>
      }
    >
      <QueryState q={episodes} skeleton={<SkeletonRows rows={2} />}>
        {() =>
          sorted.length === 0 ? (
            <EmptyState title="No episodes yet">Pick a condition and press Record, or record a batch for the plan.</EmptyState>
          ) : (
            <div className="grid grid-cols-[repeat(auto-fill,minmax(150px,1fr))] gap-2.5">
              {sorted.map((e, i) => (
                <EpisodeCard key={e.id} pid={pid} e={e} thumb={thumbs[i]?.data?.thumb_url} highlight={e.id === highlight} onTrace={setTrace} />
              ))}
            </div>
          )
        }
      </QueryState>
      <AttemptDrawer attemptId={trace} onClose={() => setTrace(null)} />
    </Panel>
  );
}

function EpisodeCard({
  pid,
  e,
  thumb,
  highlight,
  onTrace,
}: {
  pid: string;
  e: Episode;
  thumb?: string | null;
  highlight: boolean;
  onTrace: (attemptId: string) => void;
}) {
  const qc = useQueryClient();
  const m = useMutation({
    mutationFn: (s: "accepted" | "rejected") => api.setEpisodeStatus(e.id, s),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["episodes", pid] });
      qc.invalidateQueries({ queryKey: ["state", pid] });
    },
  });
  const url = resolveMediaUrl(thumb);
  return (
    <div
      className={cn(
        "group overflow-hidden rounded bg-well shadow-[0_0_0_1px_var(--line)]",
        highlight && "shadow-[0_0_0_1.5px_rgb(var(--servo))]",
        e.status === "rejected" && "opacity-55",
      )}
    >
      <div className="relative aspect-[4/3] bg-rail">
        {url ? (
          <img src={url} alt={`Episode ${e.index}`} loading="lazy" className="h-full w-full object-cover" />
        ) : (
          <div className="flex h-full items-center justify-center text-ink-4">
            <VideoOff className="h-4 w-4" />
          </div>
        )}
        <span className="absolute left-1.5 top-1.5 rounded-sm bg-black/60 px-1 font-mono text-[10px] text-white">#{e.index}</span>
        {e.status !== "recorded" && (
          <span className="absolute right-1.5 top-1.5">
            <StatusBadge status={e.status} />
          </span>
        )}
      </div>
      <div className="space-y-1.5 p-2">
        <ConditionLabel c={e.condition} />
        <div className="flex flex-wrap items-center gap-1">
          <EpisodeProvenance e={e} />
          {e.provenance?.attempt_id && (
            <button type="button" onClick={() => onTrace(String(e.provenance.attempt_id))} className="text-2xs text-ink-3 underline decoration-dotted underline-offset-2 hover:text-servo">
              trace
            </button>
          )}
        </div>
        <div className="flex flex-wrap gap-1">
          <SuccessBadge s={e.success} />
          {e.quality ? <QualityBadge score={e.quality.score} /> : <Badge title="Run analysis to score">unscored</Badge>}
        </div>
        <div className="flex gap-1">
          <Button
            size="sm"
            variant={e.status === "accepted" ? "secondary" : "ghost"}
            className="h-6 flex-1 px-1 text-2xs"
            disabled={e.status === "accepted"}
            loading={m.isPending && m.variables === "accepted"}
            onClick={() => m.mutate("accepted")}
          >
            <Check className="h-3 w-3" /> Accept
          </Button>
          <Button
            size="sm"
            variant="ghost"
            className="h-6 flex-1 px-1 text-2xs"
            disabled={e.status === "rejected"}
            loading={m.isPending && m.variables === "rejected"}
            onClick={() => m.mutate("rejected")}
          >
            <X className="h-3 w-3" /> Reject
          </Button>
        </div>
      </div>
    </div>
  );
}

// ------------------------------------------------------------------ batch dialog
function BatchDialog({
  open,
  onClose,
  pid,
  cells,
  onStarted,
}: {
  open: boolean;
  onClose: () => void;
  pid: string;
  cells: PlanCell[];
  onStarted: (j: Job) => void;
}) {
  const remaining = cells.filter((c) => c.collected < c.target);
  const [sel, setSel] = useState<Set<string>>(new Set());
  const [per, setPer] = useState(2);
  const [baseSeed, setBaseSeed] = useState("");
  useEffect(() => {
    if (open) setSel(new Set((remaining.length ? remaining : cells).map((c) => conditionKey(c.condition))));
  }, [open]); // eslint-disable-line react-hooks/exhaustive-deps

  const conditions = cells.map((c) => c.condition).filter((c) => sel.has(conditionKey(c)));
  const m = useMutation({
    mutationFn: () => api.recordBatch(pid, conditions, per, baseSeed === "" ? undefined : Number(baseSeed)),
    onSuccess: onStarted,
  });
  const byKey = new Map(cells.map((c) => [conditionKey(c.condition), c]));

  return (
    <Dialog
      open={open}
      onClose={onClose}
      title="Record a batch"
      footer={
        <>
          <Button variant="ghost" onClick={onClose}>
            Cancel
          </Button>
          <Button variant="primary" disabled={conditions.length === 0 || per < 1} loading={m.isPending} onClick={() => m.mutate()}>
            Record {conditions.length * per} demos
          </Button>
        </>
      }
    >
      <p className="mb-3 text-xs text-ink-3">
        Pick conditions. Cells still short of their plan target are pre-selected.
      </p>
      <WorkcellGrid
        onSelect={(c) =>
          setSel((s) => {
            const n = new Set(s);
            const k = conditionKey(c);
            n.has(k) ? n.delete(k) : n.add(k);
            return n;
          })
        }
        renderCell={(c) => {
          const k = conditionKey(c);
          const cell = byKey.get(k);
          const on = sel.has(k);
          return (
            <div className={cn("flex items-center gap-2 px-2 py-2", on && "bg-servo/10")}>
              <span className={cn("flex h-3.5 w-3.5 items-center justify-center rounded-sm", on ? "bg-servo text-servo-ink" : "shadow-[inset_0_0_0_1px_var(--line-strong)]")}>
                {on && <Check className="h-2.5 w-2.5" />}
              </span>
              <span className="num text-2xs text-ink-3">
                {cell ? `${cell.collected}/${cell.target}` : "—"}
              </span>
            </div>
          );
        }}
      />
      <div className="mt-4 grid grid-cols-2 gap-3">
        <Field label="Demos per condition">
          <Input type="number" min={1} max={50} value={per} onChange={(e) => setPer(Math.max(1, Number(e.target.value) || 1))} className="num font-mono" />
        </Field>
        <Field label="Base seed" hint="Optional. Blank = backend default.">
          <Input type="number" value={baseSeed} onChange={(e) => setBaseSeed(e.target.value)} className="num font-mono" placeholder="auto" />
        </Field>
      </div>
      {m.error && (
        <div className="mt-3">
          <ErrorState error={m.error} compact />
        </div>
      )}
    </Dialog>
  );
}
