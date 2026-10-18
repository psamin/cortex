import { useState } from "react";
import { useParams } from "react-router-dom";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { RefreshCw, ShieldCheck, Sparkles } from "lucide-react";
import { api } from "../../api/client";
import { useRecommendations } from "../../api/hooks";
import type { Evidence, Recommendation, RecommendationStatus } from "../../api/types";
import { Badge, Button, EmptyState, ErrorState, Input, Panel, SkeletonRows, InlineError } from "../../components/ui";
import { ConditionLabel } from "../../components/Workcell";
import { RefChip } from "../../components/refs";
import { cn, evidenceValue, humanize } from "../../lib/format";

export function SupportBadge({ support }: { support: Recommendation["support"] }) {
  return support === "evidence_backed" ? (
    <Badge tone="pass" icon={<ShieldCheck className="h-3 w-3" />} title="Backed by measured data (coverage, quality checks, evaluations)">
      evidence-backed
    </Badge>
  ) : (
    <Badge tone="hypothesis" icon={<Sparkles className="h-3 w-3" />} title="Rests on model hypotheses or priors, not measurements">
      hypothesis
    </Badge>
  );
}

const MEASURED_KINDS = new Set<Evidence["kind"]>(["coverage", "evaluation", "quality", "budget"]);

export function EvidenceList({ evidence, pid, compact }: { evidence: Evidence[]; pid: string; compact?: boolean }) {
  if (evidence.length === 0) return <p className="text-2xs text-ink-4">No evidence attached.</p>;
  return (
    <ul className={cn("space-y-1", compact && "space-y-0.5")}>
      {evidence.map((ev, i) => {
        const measured = MEASURED_KINDS.has(ev.kind);
        return (
          <li key={i} className="flex items-start gap-2 text-xs">
            <span
              className={cn(
                "mt-px w-[68px] shrink-0 rounded-sm px-1 text-center font-mono text-[10px] leading-4",
                measured ? "bg-rail text-ink-2 border border-line" : "prov-hypothesis text-ink-3",
              )}
              title={measured ? "measured" : "model hypothesis"}
            >
              {ev.kind}
            </span>
            <span className="min-w-0 flex-1 text-ink-2">
              {ev.description}
              {ev.value !== null && <span className="num ml-1.5 font-mono text-ink">{evidenceValue(ev.value)}</span>}
              {!compact && ev.ref_ids.length > 0 && (
                <span className="ml-1.5 inline-flex flex-wrap gap-1 align-middle">
                  {ev.ref_ids.slice(0, 6).map((r) => (
                    <RefChip key={r} pid={pid} id={r} />
                  ))}
                  {ev.ref_ids.length > 6 && <span className="text-2xs text-ink-4">+{ev.ref_ids.length - 6}</span>}
                </span>
              )}
            </span>
          </li>
        );
      })}
    </ul>
  );
}

function ComponentBars({ components }: { components: Record<string, number> }) {
  const entries = Object.entries(components).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
  const max = Math.max(1e-9, ...entries.map(([, v]) => Math.abs(v)));
  if (entries.length === 0) return null;
  return (
    <div className="space-y-1">
      {entries.map(([k, v]) => (
        <div key={k} className="grid grid-cols-[110px_1fr_44px] items-center gap-2 text-2xs">
          <span className="truncate text-ink-3" title={k}>
            {humanize(k)}
          </span>
          <div className="h-1.5 overflow-hidden rounded-full bg-rail">
            <div className={cn("h-full rounded-full", v < 0 ? "bg-fault/70" : "bg-servo")} style={{ width: `${(Math.abs(v) / max) * 100}%` }} />
          </div>
          <span className={cn("num text-right font-mono", v < 0 ? "text-fault" : "text-ink-2")}>{v.toFixed(2)}</span>
        </div>
      ))}
    </div>
  );
}

export function RecommendationsTab() {
  const { pid = "" } = useParams();
  const qc = useQueryClient();
  const state = useRecommendations(pid);
  const [fresh, setFresh] = useState<Recommendation[] | null>(null);
  const [budget, setBudget] = useState("");
  const refresh = useMutation({
    mutationFn: () => api.recommend(pid, budget === "" ? undefined : Number(budget)),
    onSuccess: (r) => {
      setFresh(r);
      qc.invalidateQueries({ queryKey: ["recommendations", pid] });
    },
  });
  const recs = fresh ?? state.data ?? [];
  const ranked = [...recs].sort((a, b) => b.score - a.score);

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <p className="max-w-[560px] text-xs text-ink-3">
          Ranked by expected gain per demo. <span className="text-pass">Evidence-backed</span> items cite measured coverage,
          quality, or evaluation data; <span className="text-ink-2 underline decoration-dashed underline-offset-2">hypotheses</span>{" "}
          rest on model guesses and should be tested first.
        </p>
        <div className="flex items-center gap-2">
          <label className="flex items-center gap-1.5 whitespace-nowrap text-xs text-ink-3">
            demo budget
            <Input value={budget} onChange={(e) => setBudget(e.target.value)} type="number" min={1} placeholder="auto" className="num h-7 w-20 font-mono" />
          </label>
          <Button size="sm" loading={refresh.isPending} onClick={() => refresh.mutate()}>
            <RefreshCw className="h-3.5 w-3.5" /> Re-rank
          </Button>
        </div>
      </div>
      {refresh.error && <ErrorState error={refresh.error} compact />}
      {state.isLoading ? (
        <SkeletonRows rows={4} />
      ) : state.error && !fresh ? (
        <ErrorState error={state.error} onRetry={() => state.refetch()} />
      ) : ranked.length === 0 ? (
        <Panel>
          <EmptyState title="No open recommendations" action={<Button size="sm" onClick={() => refresh.mutate()}>Generate recommendations</Button>}>
            Recommendations are computed from plan coverage, episode quality, failure clusters, and evaluation results.
          </EmptyState>
        </Panel>
      ) : (
        <ol className="space-y-2.5">
          {ranked.map((r, i) => (
            <RecCard key={r.id} r={r} rank={i + 1} pid={pid} onChanged={(nr) => setFresh((xs) => xs && xs.map((x) => (x.id === nr.id ? nr : x)))} />
          ))}
        </ol>
      )}
    </div>
  );
}

function RecCard({ r, rank, pid, onChanged }: { r: Recommendation; rank: number; pid: string; onChanged: (r: Recommendation) => void }) {
  const qc = useQueryClient();
  const m = useMutation({
    mutationFn: (s: RecommendationStatus) => api.setRecommendationStatus(r.id, s),
    onSuccess: (nr) => {
      onChanged(nr);
      qc.invalidateQueries({ queryKey: ["recommendations", pid] });
      qc.invalidateQueries({ queryKey: ["state", pid] });
    },
  });
  const top = rank === 1;
  return (
    <li
      className={cn(
        "grid grid-cols-[36px_minmax(0,1fr)_minmax(0,300px)] gap-x-4 gap-y-2 rounded-md bg-plate p-3.5",
        r.support === "hypothesis" ? "prov-hypothesis" : "prov-measured",
        top && r.support === "evidence_backed" && "border border-servo/40",
      )}
    >
      <div className={cn("num pt-0.5 font-mono text-lg font-semibold", top ? "text-servo" : "text-ink-4")}>{rank}</div>
      <div className="min-w-0 space-y-2">
        <div>
          <div className="flex flex-wrap items-center gap-1.5">
            <Badge tone="servo" className="uppercase tracking-wide">
              {r.action}
            </Badge>
            <SupportBadge support={r.support} />
            {r.status !== "proposed" && <Badge>{r.status}</Badge>}
          </div>
          <h3 className="kn-display mt-1.5 text-[22px]">{r.title}</h3>
          <div className="mt-0.5 flex flex-wrap items-center gap-2 text-xs text-ink-3">
            {r.condition && <ConditionLabel c={r.condition} />}
            {r.count > 0 && <span className="num">× {r.count}</span>}
          </div>
        </div>
        <EvidenceList evidence={r.evidence} pid={pid} />
        <div className="flex gap-1.5 pt-1">
          {r.status === "proposed" && (
            <>
              <Button size="sm" variant="primary" loading={m.isPending && m.variables === "accepted"} onClick={() => m.mutate("accepted")}>
                Accept
              </Button>
              <Button size="sm" variant="ghost" loading={m.isPending && m.variables === "dismissed"} onClick={() => m.mutate("dismissed")}>
                Dismiss
              </Button>
            </>
          )}
          {r.status === "accepted" && (
            <Button size="sm" loading={m.isPending} onClick={() => m.mutate("done")}>
              Mark done
            </Button>
          )}
          {m.error && <InlineError error={m.error} className="self-center" />}
        </div>
      </div>
      <div className="space-y-2">
        <div className="flex items-baseline justify-between">
          <span className="text-2xs font-medium uppercase tracking-[0.06em] text-ink-3">Score</span>
          <span className="num font-mono text-2xl font-medium leading-none">{r.score.toFixed(2)}</span>
        </div>
        <ComponentBars components={r.components} />
      </div>
    </li>
  );
}
