import { useEffect, useMemo, useState } from "react";
import { useParams, useSearchParams } from "react-router-dom";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Bot, Check, Cog, Play, ScanSearch, UserCheck, UserRound, VideoOff, X } from "lucide-react";
import { api, resolveMediaUrl } from "../api/client";
import { isJobActive, useJob, useTrackJob } from "../api/events";
import { useEpisodeMedia, useEpisodes, useFailureClusters, useLatestAnalysis } from "../api/hooks";
import type { Analysis, Episode, EpisodeDiagnosis, FailureCluster, QualityFlag, QualityReport } from "../api/types";
import { isImported } from "../api/types";
import {
  Badge,
  Button,
  EmptyState,
  ErrorState,
  HypothesisBadge,
  Input,
  Meter,
  Panel,
  QueryState,
  SectionLabel,
  Skeleton,
  SkeletonRows,
  StatusBadge,
  Toggle,
  InlineError,
} from "../components/ui";
import { ConditionLabel, Footprint } from "../components/Workcell";
import { describeProvider } from "../components/HealthPills";
import { AttemptDrawer, EpisodeProvenance, ImportedSource } from "../components/AgentTrace";
import { QualityBadge, SuccessBadge } from "./Collect";
import { ago, cn, evidenceValue, humanize, pct } from "../lib/format";

export function AnalyzePage() {
  const { pid = "" } = useParams();
  const [params, setParams] = useSearchParams();
  const episodes = useEpisodes(pid);
  const analysis = useLatestAnalysis(pid);
  const selected = params.get("ep");
  const select = (id: string) => setParams((p) => (p.set("ep", id), p), { replace: true });

  // default selection: first episode
  useEffect(() => {
    if (!selected && episodes.data?.length) select([...episodes.data].sort((a, b) => b.index - a.index)[0].id);
  }, [episodes.data, selected]); // eslint-disable-line react-hooks/exhaustive-deps

  const selectedEp = episodes.data?.find((e) => e.id === selected);

  return (
    <div className="space-y-4 px-5 py-5">
      <div className="flex items-end justify-between gap-4">
        <div>
          <h1 className="kn-display text-[36px] leading-tight">Analyze</h1>
          <p className="mt-0.5 text-sm text-ink-3">
            Measured quality checks plus VLM diagnoses. Diagnoses are hypotheses until you confirm them.
          </p>
        </div>
      </div>

      <RunAnalysis pid={pid} analysis={analysis.data ?? undefined} />

      <div className="grid grid-cols-1 gap-4 min-[1400px]:grid-cols-[minmax(0,1fr)_360px]">
        <Panel title="Episodes" meta={episodes.data ? `${episodes.data.length} total` : undefined} bodyClassName="p-0">
          <QueryState q={episodes} skeleton={<div className="p-3.5"><SkeletonRows rows={6} /></div>}>
            {(eps) =>
              eps.length === 0 ? (
                <EmptyState title="No episodes to analyze">Record demos in Collect first.</EmptyState>
              ) : (
                <EpisodeTable episodes={eps} analysis={analysis.data ?? undefined} selected={selected} onSelect={select} />
              )
            }
          </QueryState>
        </Panel>

        <div className="space-y-4">
          {selectedEp ? (
            <EpisodeDetail pid={pid} e={selectedEp} analysis={analysis.data ?? undefined} />
          ) : (
            <Panel title="Episode">
              {episodes.isLoading ? <Skeleton className="aspect-[4/3] w-full" /> : <EmptyState title="Select an episode" />}
            </Panel>
          )}
        </div>
      </div>

      <FailureClusters pid={pid} onSelectEpisode={select} />
    </div>
  );
}

// ------------------------------------------------------------------ run
function RunAnalysis({ pid, analysis }: { pid: string; analysis?: Analysis }) {
  const [useVlm, setUseVlm] = useState(true);
  const [maxVlm, setMaxVlm] = useState(8);
  const [jobId, setJobId] = useState<string | null>(null);
  const track = useTrackJob();
  const job = useJob(jobId);
  const running = isJobActive(job) || analysis?.status === "running";
  const m = useMutation({
    mutationFn: () => api.runAnalysis(pid, useVlm, useVlm ? maxVlm : undefined),
    onSuccess: (j) => {
      track(j);
      setJobId(j.id);
    },
  });
  const vlm = analysis?.vlm_provider ? describeProvider(analysis.vlm_provider) : null;

  return (
    <div className="flex flex-wrap items-center gap-x-6 gap-y-3 rounded-md bg-plate px-3.5 py-3 border border-line">
      <div className="min-w-[220px] flex-1">
        {analysis ? (
          <>
            <div className="flex flex-wrap items-center gap-2 text-sm">
              <span className="font-medium">Latest analysis</span>
              <StatusBadge status={analysis.status} />
              <span className="text-xs text-ink-3">
                {analysis.episode_count} episodes · {ago(analysis.created_at)}
              </span>
              {vlm && (
                <Badge tone={vlm.fallback ? "fallback" : "neutral"} title="VLM provider used for diagnoses">
                  vlm: <span className="font-mono">{vlm.name}</span>
                  {vlm.fallback && " (fallback)"}
                </Badge>
              )}
            </div>
            {analysis.summary && <p className="mt-1 text-xs text-ink-2">{analysis.summary}</p>}
          </>
        ) : (
          <p className="text-sm text-ink-3">No analysis yet. Run one to score episode quality and diagnose failures.</p>
        )}
        {job && isJobActive(job) && (
          <div className="mt-2 flex items-center gap-2">
            <Meter value={job.progress} className="max-w-[240px]" />
            <span className="truncate text-2xs text-ink-3">{job.message}</span>
          </div>
        )}
        {job?.status === "failed" && <InlineError prefix="Analysis failed: " error={job.error ?? job.message} detail={job.error_detail} className="mt-1" />}
      </div>
      <div className="flex items-center gap-4">
        <Toggle checked={useVlm} onChange={setUseVlm} label="VLM diagnosis" hint="Sends frames to the vision model" />
        {useVlm && (
          <label className="flex items-center gap-1.5 text-xs text-ink-3">
            max
            <Input
              type="number"
              min={1}
              max={100}
              value={maxVlm}
              onChange={(e) => setMaxVlm(Math.max(1, Number(e.target.value) || 1))}
              className="num h-7 w-14 font-mono"
              aria-label="Max episodes to send to the VLM"
            />
          </label>
        )}
        <Button variant="primary" loading={m.isPending || running} onClick={() => m.mutate()}>
          <ScanSearch className="h-4 w-4" /> {running ? "Analyzing" : "Run analysis"}
        </Button>
      </div>
      {m.error && (
        <div className="w-full">
          <ErrorState error={m.error} compact />
        </div>
      )}
    </div>
  );
}

// ------------------------------------------------------------------ table
const sevTone = { info: "neutral", warn: "caution", error: "fault" } as const;

function EpisodeTable({
  episodes,
  analysis,
  selected,
  onSelect,
}: {
  episodes: Episode[];
  analysis?: Analysis;
  selected: string | null;
  onSelect: (id: string) => void;
}) {
  const rows = useMemo(() => [...episodes].sort((a, b) => b.index - a.index), [episodes]);
  const diagByEp = new Map((analysis?.diagnoses ?? []).map((d) => [d.episode_id, d]));
  return (
    <div className="max-h-[560px] overflow-auto scroll-thin">
      <table className="w-full text-left text-sm">
        <thead className="sticky top-0 z-10 bg-plate text-2xs font-medium uppercase tracking-[0.05em] text-ink-3">
          <tr className="border-b border-line-soft">
            <th className="w-12 px-3.5 py-2 font-medium">#</th>
            <th className="px-2 py-2 font-medium">Condition</th>
            <th className="px-2 py-2 font-medium">Outcome</th>
            <th className="px-2 py-2 font-medium">Quality</th>
            <th className="px-2 py-2 font-medium">Flags</th>
            <th className="px-2 py-2 font-medium">Diagnosis</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((e) => {
            const q: QualityReport | null = analysis?.quality[e.id] ?? e.quality;
            const d = diagByEp.get(e.id);
            const on = e.id === selected;
            return (
              <tr
                key={e.id}
                onClick={() => onSelect(e.id)}
                className={cn(
                  "cursor-pointer border-b border-line-soft transition-colors duration-75",
                  on ? "bg-servo/[.06]" : "hover:bg-rail/50",
                  e.status === "rejected" && "opacity-55",
                )}
              >
                <td className={cn("px-3.5 py-2 font-mono text-xs", on ? "text-servo" : "text-ink-3")}>{e.index}</td>
                <td className="px-2 py-2">
                  <span className="inline-flex items-center gap-1.5">
                    {e.source === "imported" ? (
                      <UserRound className="h-3.5 w-3.5 text-ink-3" aria-label="Human demo, imported" />
                    ) : e.source === "sim_agent" ? (
                      <Bot className="h-3.5 w-3.5 text-servo" aria-label="Agent demo" />
                    ) : e.source === "sim_scripted" ? (
                      <Cog className="h-3.5 w-3.5 text-ink-4" aria-label="Scripted IK oracle" />
                    ) : null}
                    {isImported(e) ? <span className="text-xs text-ink-3" title="Imported demos have no known scene condition">condition unknown</span> : <ConditionLabel c={e.condition} />}
                  </span>
                </td>
                <td className="px-2 py-2">
                  <SuccessBadge s={e.success} />
                </td>
                <td className="px-2 py-2">{q ? <QualityBadge score={q.score} /> : <span className="text-2xs text-ink-4">—</span>}</td>
                <td className="px-2 py-2">
                  <div className="flex flex-wrap gap-1">
                    {q?.flags.length ? (
                      q.flags.slice(0, 3).map((f) => (
                        <Badge key={f.code} tone={sevTone[f.severity]} title={`${f.message}\n${evidenceLine(f)}`}>
                          <span className="font-mono">{f.code}</span>
                        </Badge>
                      ))
                    ) : q ? (
                      <span className="text-2xs text-ink-4">none</span>
                    ) : null}
                    {q && q.flags.length > 3 && <span className="text-2xs text-ink-3">+{q.flags.length - 3}</span>}
                  </div>
                </td>
                <td className="px-2 py-2">
                  {d ? (
                    d.status === "hypothesis" ? (
                      <span className="text-xs text-ink-2">{d.hypotheses[0]?.category ? humanize(d.hypotheses[0].category) : "—"}</span>
                    ) : (
                      <StatusBadge status={d.status} />
                    )
                  ) : (
                    <span className="text-2xs text-ink-4">—</span>
                  )}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

const evidenceLine = (f: QualityFlag) =>
  Object.entries(f.evidence)
    .map(([k, v]) => `${k}=${evidenceValue(v)}`)
    .join("  ");

// ------------------------------------------------------------------ detail
function EpisodeDetail({ pid, e, analysis }: { pid: string; e: Episode; analysis?: Analysis }) {
  const media = useEpisodeMedia(e.id);
  const q = analysis?.quality[e.id] ?? e.quality;
  const diag = analysis?.diagnoses.find((d) => d.episode_id === e.id);
  const video = resolveMediaUrl(media.data?.video_url);
  const thumb = resolveMediaUrl(media.data?.thumb_url);
  const [videoFailed, setVideoFailed] = useState(false);
  const [trace, setTrace] = useState<string | null>(null);
  useEffect(() => setVideoFailed(false), [e.id]);

  return (
    <>
      <Panel
        title={
          <span>
            Episode <span className="font-mono text-ink-3">#{e.index}</span>
          </span>
        }
        meta={`${e.num_frames} frames · ${e.fps} fps${isImported(e) ? "" : ` · seed ${e.seed}`}`}
        bodyClassName="p-0"
      >
        <div className="aspect-[4/3] bg-well">
          {media.isLoading ? (
            <Skeleton className="h-full w-full rounded-none" />
          ) : video && !videoFailed ? (
            <video key={video} src={video} controls loop playsInline poster={thumb ?? undefined} onError={() => setVideoFailed(true)} className="h-full w-full object-contain" />
          ) : thumb ? (
            <div className="relative h-full">
              <img src={thumb} alt={`Episode ${e.index} thumbnail`} className="h-full w-full object-contain" />
              <span className="absolute bottom-2 left-2 rounded-sm bg-black/60 px-1.5 py-0.5 text-2xs text-white/80">thumbnail · no video</span>
            </div>
          ) : (
            <div className="flex h-full flex-col items-center justify-center gap-1.5 text-ink-4">
              <VideoOff className="h-5 w-5" />
              <span className="text-xs">{media.error ? "Media unavailable" : "No media for this episode"}</span>
            </div>
          )}
        </div>
        <div className="space-y-3 p-3.5">
          <div className="flex flex-wrap items-center gap-1.5">
            {isImported(e) ? <span className="text-xs text-ink-3" title="Imported demos have no known scene condition">condition unknown</span> : <ConditionLabel c={e.condition} />}
            <SuccessBadge s={e.success} />
            <StatusBadge status={e.status} />
            <EpisodeProvenance e={e} />
            <ImportedSource e={e} className="w-full" />
            {e.provenance?.attempt_id && (
              <Button size="sm" variant="ghost" className="h-5 px-1.5 text-2xs" onClick={() => setTrace(String(e.provenance.attempt_id))}>
                Action trace
              </Button>
            )}
          </div>
          {q ? (
            <div>
              <SectionLabel className="mb-1.5">Quality checks · score {q.score.toFixed(2)}</SectionLabel>
              {q.flags.length === 0 ? (
                <p className="text-xs text-ink-3">No flags raised.</p>
              ) : (
                <ul className="space-y-1.5">
                  {q.flags.map((f) => (
                    <li key={f.code} className="rounded bg-well p-2 border border-line">
                      <div className="flex items-center gap-1.5">
                        <Badge tone={sevTone[f.severity]}>{f.severity}</Badge>
                        <span className="font-mono text-xs text-ink">{f.code}</span>
                      </div>
                      <p className="mt-1 text-xs text-ink-2">{f.message}</p>
                      {Object.keys(f.evidence).length > 0 && (
                        <dl className="mt-1.5 grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 font-mono text-2xs">
                          {Object.entries(f.evidence).map(([k, v]) => (
                            <div key={k} className="contents">
                              <dt className="text-ink-3">{k}</dt>
                              <dd className="num text-ink">{evidenceValue(v)}</dd>
                            </div>
                          ))}
                        </dl>
                      )}
                    </li>
                  ))}
                </ul>
              )}
              {Object.keys(q.metrics).length > 0 && (
                <details className="mt-2">
                  <summary className="cursor-pointer text-2xs text-ink-3 hover:text-ink-2">All metrics</summary>
                  <dl className="mt-1.5 grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 font-mono text-2xs">
                    {Object.entries(q.metrics).map(([k, v]) => (
                      <div key={k} className="contents">
                        <dt className="text-ink-3">{k}</dt>
                        <dd className="num text-ink">{evidenceValue(v)}</dd>
                      </div>
                    ))}
                  </dl>
                </details>
              )}
            </div>
          ) : (
            <p className="text-xs text-ink-3">Not scored yet — run analysis.</p>
          )}
        </div>
      </Panel>
      <DiagnosisPanel pid={pid} analysisId={analysis?.id} diag={diag} />
      <AttemptDrawer attemptId={trace} onClose={() => setTrace(null)} />
    </>
  );
}

function DiagnosisPanel({ pid, analysisId, diag }: { pid: string; analysisId?: string; diag?: EpisodeDiagnosis }) {
  const qc = useQueryClient();
  const [label, setLabel] = useState("");
  useEffect(() => setLabel(diag?.human_label ?? diag?.hypotheses[0]?.category ?? ""), [diag?.episode_id, diag?.human_label]); // eslint-disable-line react-hooks/exhaustive-deps
  const m = useMutation({
    mutationFn: (status: "human_confirmed" | "human_rejected") =>
      api.correctDiagnosis(analysisId!, diag!.episode_id, status, label.trim() || undefined),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["analysis", analysisId] });
      qc.invalidateQueries({ queryKey: ["latest-analysis", pid] });
      qc.invalidateQueries({ queryKey: ["failure-clusters", pid] });
    },
  });

  if (!diag)
    return (
      <Panel title="VLM diagnosis">
        <p className="text-xs text-ink-3">No diagnosis for this episode. Run analysis with VLM enabled.</p>
      </Panel>
    );

  return (
    <Panel
      title="VLM diagnosis"
      meta={<span className="font-mono">{diag.model}</span>}
      actions={diag.status === "hypothesis" ? <HypothesisBadge /> : <StatusBadge status={diag.status} />}
    >
      {diag.error && <InlineError prefix="Diagnosis failed: " error={diag.error} fallback="the vision model returned an error." className="mb-2" />}
      {diag.hypotheses.length === 0 && !diag.error && <p className="text-xs text-ink-3">The model raised no hypotheses.</p>}
      <ul className="space-y-2">
        {diag.hypotheses.map((h, i) => (
          <li key={i} className={cn("rounded p-2.5", diag.status === "human_confirmed" && i === 0 ? "prov-measured" : "prov-hypothesis")}>
            <div className="flex items-center justify-between gap-2">
              <span className="text-sm font-medium capitalize">{humanize(h.category)}</span>
              <span className="num font-mono text-2xs text-ink-3" title="Model-reported confidence. Not a calibrated probability.">
                conf {pct(h.confidence)}*
              </span>
            </div>
            <p className="mt-1 text-xs text-ink-2">{h.rationale}</p>
            {h.frame_indices.length > 0 && (
              <p className="mt-1 font-mono text-2xs text-ink-3">frames {h.frame_indices.join(", ")}</p>
            )}
          </li>
        ))}
      </ul>
      <p className="mt-2 text-2xs text-ink-4">* self-reported by the model, not a verified probability.</p>

      <div className="mt-3 space-y-2 border-t border-line-soft pt-3">
        <SectionLabel className="flex items-center gap-1.5">
          <UserCheck className="h-3 w-3" /> Human correction
        </SectionLabel>
        <Input value={label} onChange={(e) => setLabel(e.target.value)} placeholder="Failure label (e.g. grasp_slip)" aria-label="Failure label" />
        <div className="flex gap-1.5">
          <Button size="sm" variant="primary" disabled={!analysisId} loading={m.isPending && m.variables === "human_confirmed"} onClick={() => m.mutate("human_confirmed")}>
            <Check className="h-3.5 w-3.5" /> Confirm
          </Button>
          <Button size="sm" variant="danger" disabled={!analysisId} loading={m.isPending && m.variables === "human_rejected"} onClick={() => m.mutate("human_rejected")}>
            <X className="h-3.5 w-3.5" /> Reject
          </Button>
        </div>
        {diag.human_label && diag.status !== "hypothesis" && (
          <p className="text-2xs text-ink-3">
            Labelled <span className="font-mono text-ink-2">{diag.human_label}</span> by a human.
          </p>
        )}
        {m.error && <InlineError error={m.error} />}
      </div>
    </Panel>
  );
}

// ------------------------------------------------------------------ clusters
function FailureClusters({ pid, onSelectEpisode }: { pid: string; onSelectEpisode: (id: string) => void }) {
  const q = useFailureClusters(pid);
  return (
    <Panel title="Failure clusters" meta="grouped by category across episodes">
      <QueryState q={q} skeleton={<SkeletonRows rows={3} />}>
        {(clusters) =>
          clusters.length === 0 ? (
            <EmptyState title="No failure clusters">Clusters appear after an analysis finds repeated failure patterns.</EmptyState>
          ) : (
            <div className="grid grid-cols-1 gap-2.5 lg:grid-cols-2 2xl:grid-cols-3">
              {[...clusters]
                .sort((a, b) => b.episode_ids.length - a.episode_ids.length)
                .map((c) => (
                  <ClusterCard key={c.id} c={c} onSelectEpisode={onSelectEpisode} />
                ))}
            </div>
          )
        }
      </QueryState>
    </Panel>
  );
}

function ClusterCard({ c, onSelectEpisode }: { c: FailureCluster; onSelectEpisode: (id: string) => void }) {
  const hypothesis = c.source === "vlm" && !c.verified_by_human;
  return (
    <div className={cn("rounded p-3", hypothesis ? "prov-hypothesis" : "prov-measured")}>
      <div className="flex items-start justify-between gap-2">
        <div>
          <div className="text-sm font-medium capitalize">{humanize(c.category)}</div>
          <div className="mt-1 flex flex-wrap gap-1">
            <Badge tone={c.source === "vlm" ? "hypothesis" : "neutral"}>{c.source}</Badge>
            {hypothesis && <HypothesisBadge />}
            {c.verified_by_human && <Badge tone="pass">human verified</Badge>}
          </div>
        </div>
        <div className="flex flex-col items-end gap-1">
          <span className="num kn-display text-[26px] leading-none">{c.episode_ids.length}</span>
          <span className="text-2xs text-ink-3">episodes</span>
        </div>
      </div>
      <div className="mt-2.5 flex items-center gap-3">
        <Footprint keys={c.conditions} tone="fault" />
        <div className="min-w-0 flex-1">
          <div className="mb-0.5 flex justify-between text-2xs text-ink-3">
            <span>uncertainty</span>
            <span className="num font-mono">{pct(c.uncertainty)}</span>
          </div>
          <Meter value={c.uncertainty} tone="ink" label="Uncertainty" />
        </div>
      </div>
      {c.evidence.length > 0 && (
        <ul className="mt-2.5 space-y-0.5 text-xs text-ink-2">
          {c.evidence.slice(0, 4).map((ev, i) => (
            <li key={i} className="flex gap-1.5">
              <span className="text-ink-4">–</span>
              {ev}
            </li>
          ))}
        </ul>
      )}
      <div className="mt-2.5 flex flex-wrap gap-1">
        {c.episode_ids.slice(0, 10).map((id) => (
          <button
            key={id}
            type="button"
            onClick={() => onSelectEpisode(id)}
            className="rounded-sm bg-rail px-1.5 py-px font-mono text-[10.5px] text-ink-2 border border-line hover:text-servo"
          >
            <Play className="mr-0.5 inline h-2.5 w-2.5" />
            {id}
          </button>
        ))}
        {c.episode_ids.length > 10 && <span className="text-2xs text-ink-3">+{c.episode_ids.length - 10}</span>}
      </div>
    </div>
  );
}
