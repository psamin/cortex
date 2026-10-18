import { useEffect, useRef, useState } from "react";
import { AlertOctagon, Bot, Check, Cog, Eye, UserRound, VideoOff, X } from "lucide-react";
import { resolveMediaUrl } from "../api/client";
import { useAgentAttempt } from "../api/hooks";
import { agentLabel, isAgent, type AgentAttempt, type Condition, type Episode, type Scenario, type TraceStep } from "../api/types";
import { Badge, ErrorState, Skeleton, InlineError } from "./ui";
import { LightGlyph } from "./Workcell";
import { cn, evidenceValue, num } from "../lib/format";

// ------------------------------------------------------------------ provenance
export function PlanningModeBadge({ mode }: { mode: string | null | undefined }) {
  if (mode === "privileged")
    return (
      <Badge tone="caution" title="The planner was given simulator ground-truth object poses. Not camera-grounded.">
        PRIVILEGED-STATE planning
      </Badge>
    );
  return (
    <Badge tone="neutral" title="The planner sees camera images only; object poses are not given to it.">
      <Eye className="h-3 w-3" /> camera-grounded
    </Badge>
  );
}

/** "Claude Code agent · camera" vs "Scripted IK oracle" vs "Human demo · imported". */
export function ProvenanceBadge({
  planner,
  mode,
  source,
  detail,
}: {
  planner?: string | null;
  mode?: string | null;
  source?: string;
  /** tooltip, e.g. the imported dataset's source and task */
  detail?: string;
}) {
  if (/imported/i.test(planner ?? "") || source === "imported")
    return (
      <Badge tone="neutral" icon={<UserRound className="h-3 w-3" />} title={detail ?? "Human demonstration imported from a LeRobot dataset; not judged by the simulator"}>
        Human demo · imported
      </Badge>
    );
  const oracle = planner === "scripted_ik_oracle" || planner === "oracle" || source === "sim_scripted";
  if (!oracle && (source === "sim_agent" || (planner && !/teleop/i.test(planner))))
    return (
      <Badge tone="servo" icon={<Bot className="h-3 w-3" />} title={`Planned by ${agentLabel(planner)}, executed by validated IK in MuJoCo`}>
        {agentLabel(planner)} · {mode === "privileged" ? "privileged" : "camera"}
      </Badge>
    );
  if (oracle)
    return (
      <Badge tone="neutral" icon={<Cog className="h-3 w-3" />} title="Fixed IK waypoints computed from simulator state (privileged)">
        Scripted IK oracle
      </Badge>
    );
  if (source === "sim_teleop" || source === "real_teleop") return <Badge>{source === "real_teleop" ? "Real teleop" : "Sim teleop"}</Badge>;
  return null;
}

export function EpisodeProvenance({ e }: { e: Episode }) {
  const p = e.provenance ?? {};
  return <ProvenanceBadge planner={p.planner} mode={p.mode} source={e.source} detail={p.source ? `${p.source}${p.task ? ` · ${p.task}` : ""}` : undefined} />;
}

/** Source repo/upload and task text of an imported demo (one muted line). */
export function ImportedSource({ e, className }: { e: Episode; className?: string }) {
  const p = e.provenance ?? {};
  if (e.source !== "imported" && !/imported/i.test(p.planner ?? "")) return null;
  return (
    <p className={cn("truncate text-[11px] text-ink-3", className)} title={[p.source, p.task].filter(Boolean).join(" · ")}>
      {p.source ?? "imported"}
      {p.task ? ` · ${p.task}` : ""}
    </p>
  );
}

export function ScenarioChip({ scenario, condition, seed, className }: { scenario?: Scenario; condition?: Condition; seed?: number; className?: string }) {
  const c = scenario ?? condition;
  if (!c) return null;
  const off = (v?: [number, number]) => (v && (v[0] !== 0 || v[1] !== 0) ? `${(v[0] * 100).toFixed(1)}, ${(v[1] * 100).toFixed(1)} cm` : null);
  const co = off(scenario?.cube_offset);
  const bo = off(scenario?.box_offset);
  return (
    <span className={cn("inline-flex flex-wrap items-center gap-x-2 gap-y-0.5 text-xs text-ink-2", className)}>
      <span className="inline-flex items-center gap-1.5 whitespace-nowrap">
        <LightGlyph lighting={c.lighting} />
        cube {c.cube_region} · {c.lighting} light
      </span>
      {co && <span className="whitespace-nowrap font-mono text-[10.5px] normal-case text-ink-3">cube Δ({co})</span>}
      {bo && <span className="whitespace-nowrap font-mono text-[10.5px] normal-case text-ink-3">bowl Δ({bo})</span>}
      {seed !== undefined && <span className="font-mono text-[10.5px] normal-case text-ink-3">seed {seed}</span>}
    </span>
  );
}

// ------------------------------------------------------------------ trace rows
function errorText(m: unknown): string {
  if (typeof m === "string") return m;
  if (Array.isArray(m))
    return m
      .map((x) => (x && typeof x === "object" && "msg" in x ? `${(x as { loc?: unknown[] }).loc?.join(".") ?? ""} ${String((x as { msg: unknown }).msg)}`.trim() : JSON.stringify(x)))
      .join("; ");
  return JSON.stringify(m);
}

/** One validated tool call. Rejected commands show the validator's code + message. */
export function TraceRow({ step, dense }: { step: TraceStep; dense?: boolean }) {
  const [open, setOpen] = useState(false);
  const err = !step.ok ? step.result?.error : undefined;
  return (
    <li className={cn("grid grid-cols-[18px_minmax(0,1fr)_auto] gap-x-2", dense ? "py-1" : "py-1.5")}>
      <span className={cn("mt-0.5 flex h-4 w-4 items-center justify-center rounded-full", step.ok ? "bg-pass/15 text-pass" : "bg-fault/15 text-fault")}>
        {step.ok ? <Check className="h-2.5 w-2.5" /> : <AlertOctagon className="h-2.5 w-2.5" />}
      </span>
      <div className="min-w-0">
        <button type="button" onClick={() => setOpen((o) => !o)} className="flex w-full min-w-0 items-baseline gap-2 text-left">
          <span className="truncate text-sm font-medium text-ink">{step.phase}</span>
          <span className="truncate font-mono text-[10.5px] text-ink-3">{step.tool}</span>
        </button>
        {err && (
          <div className="mt-0.5 text-xs text-fault">
            <span className="rounded-sm bg-fault/10 px-1 font-mono text-[10.5px]">{err.code ?? "rejected"}</span>{" "}
            <span className="text-ink-2">{errorText(err.message)}</span>
            {Array.isArray(err.details) && err.details.length > 0 && (
              <ul className="mt-0.5 space-y-0.5 pl-1">
                {err.details.map((d, k) => (
                  <li key={k} className="text-ink-3">
                    <span className="font-mono text-[10.5px]">{(d.loc ?? []).join(".")}</span> {d.msg}
                  </li>
                ))}
              </ul>
            )}
          </div>
        )}
        {open && (
          <pre className="mt-1 max-h-48 overflow-auto rounded bg-well p-2 font-mono text-[10.5px] leading-relaxed text-ink-2 scroll-thin">
            {JSON.stringify({ args: step.args, result: step.result }, null, 2)}
          </pre>
        )}
      </div>
      <span className="num mt-0.5 whitespace-nowrap text-right font-mono text-[10.5px] text-ink-4" title="model call # · simulator step">
        call {step.model_call} · step {step.sim_step}
      </span>
    </li>
  );
}

export function ThoughtRow({ text }: { text: string }) {
  return (
    <li className="py-1 pl-[26px] text-xs italic leading-relaxed text-ink-3">
      {text}
    </li>
  );
}

export function Verdict({ a }: { a: AgentAttempt }) {
  return (
    <div className="flex flex-wrap items-center gap-1.5 text-xs">
      {a.status === "success" ? (
        <Badge tone="pass" title="Measured by the simulator: the cube rests in the bowl">simulator: success</Badge>
      ) : (
        <Badge tone="fault" title="Measured by the simulator">simulator: failed</Badge>
      )}
      {a.model_claimed_success !== null && (
        <Badge tone={a.model_claimed_success === (a.status === "success") ? "neutral" : "caution"} title="What the planner said in finish_episode">
          model claimed {a.model_claimed_success ? "success" : "failure"}
        </Badge>
      )}
      {a.status === "failed" && <span className="text-2xs text-ink-3">kept as a diagnostic, not training data</span>}
    </div>
  );
}

// ------------------------------------------------------------------ drawer
export function AttemptDrawer({ attemptId, onClose }: { attemptId: string | null; onClose: () => void }) {
  const ref = useRef<HTMLDialogElement>(null);
  const q = useAgentAttempt(attemptId);
  useEffect(() => {
    const d = ref.current;
    if (!d) return;
    if (attemptId && !d.open) d.showModal();
    if (!attemptId && d.open) d.close();
  }, [attemptId]);
  const a = q.data?.attempt;
  const video = resolveMediaUrl(q.data?.media.front);
  const thumb = resolveMediaUrl(q.data?.media.thumb);

  return (
    <dialog
      ref={ref}
      onClose={onClose}
      onClick={(e) => e.target === ref.current && onClose()}
      className="m-0 ml-auto h-full max-h-none w-[600px] max-w-[100vw] bg-plate p-0 text-ink border border-line-strong"
    >
      {attemptId && (
        <div className="flex h-full flex-col">
          <header className="flex items-start gap-3 border-b border-line-soft px-4 py-3">
            <div className="min-w-0 flex-1">
              <div className="flex flex-wrap items-center gap-2">
                <h2 className="text-base font-semibold">Attempt {a ? `#${a.index + 1}` : ""}</h2>
                {a && <ProvenanceBadge planner={isAgent(a.collector) ? a.model ?? "agent" : "oracle"} mode={a.mode} />}
                {a && <PlanningModeBadge mode={a.mode} />}
              </div>
              <div className="mt-1 flex flex-wrap items-center gap-2">
                <span className="font-mono text-2xs text-ink-3">{attemptId}</span>
                {a && <ScenarioChip scenario={a.scenario} seed={a.seed} />}
              </div>
            </div>
            <button type="button" onClick={onClose} className="rounded p-1 text-ink-3 hover:bg-rail hover:text-ink" aria-label="Close">
              <X className="h-4 w-4" />
            </button>
          </header>

          <div className="min-h-0 flex-1 overflow-y-auto scroll-thin">
            {q.isLoading ? (
              <div className="space-y-2 p-4">
                <Skeleton className="aspect-[4/3] w-full" />
                <Skeleton className="h-6 w-1/2" />
              </div>
            ) : q.error ? (
              <div className="p-4">
                <ErrorState error={q.error} onRetry={() => q.refetch()} />
              </div>
            ) : a ? (
              <>
                <div className="aspect-[4/3] bg-well">
                  {video ? (
                    <video key={video} src={video} poster={thumb ?? undefined} controls autoPlay loop playsInline className="h-full w-full object-contain" />
                  ) : thumb ? (
                    <img src={thumb} alt="Attempt thumbnail" className="h-full w-full object-contain" />
                  ) : (
                    <div className="flex h-full flex-col items-center justify-center gap-1.5 text-ink-4">
                      <VideoOff className="h-5 w-5" />
                      <span className="text-xs">No video recorded for this attempt</span>
                    </div>
                  )}
                </div>
                <div className="space-y-3 p-4">
                  <Verdict a={a} />
                  <dl className="grid grid-cols-3 gap-x-4 gap-y-2 sm:grid-cols-6">
                    {[
                      ["model calls", a.usage.calls ?? 0],
                      ["cost", a.usage.cost_usd !== undefined ? `$${Number(a.usage.cost_usd).toFixed(3)}` : "—"],
                      ["recoveries", a.recoveries],
                      ["tool errors", a.counters.tool_errors ?? 0],
                      ["frames", a.num_frames],
                      ["wall", `${num(a.wall_s, 1)}s`],
                    ].map(([k, v]) => (
                      <div key={String(k)}>
                        <dt className="text-2xs text-ink-3">{String(k)}</dt>
                        <dd className="num font-mono text-sm text-ink">{evidenceValue(v)}</dd>
                      </div>
                    ))}
                  </dl>
                  {a.stop_reason && <p className="text-xs text-caution">Stopped: {a.stop_reason}</p>}
                  {a.error && <InlineError error={a.error} fallback="The attempt ended with an error." />}
                  {isAgent(a.collector) && (
                    <p className="text-2xs text-ink-3">
                      {a.provider ?? "provider unknown"} · <span className="font-mono">{a.model ?? "model unknown"}</span>
                      {a.endpoint && (
                        <>
                          {" "}
                          · <span className="break-all font-mono">{a.endpoint}</span>
                        </>
                      )}
                    </p>
                  )}
                  {q.data?.model_summary && (
                    <blockquote className="border-l-2 border-line-strong pl-3 text-xs italic text-ink-2">{q.data.model_summary}</blockquote>
                  )}
                  {a.episode_id && (
                    <p className="text-2xs text-ink-3">
                      Became episode <span className="font-mono text-ink-2">{a.episode_id}</span>
                    </p>
                  )}
                </div>
                <div className="border-t border-line-soft px-4 py-3">
                  <div className="mb-1 flex items-baseline justify-between">
                    <h3 className="text-sm font-semibold">Action trace</h3>
                    <span className="text-2xs text-ink-3">{q.data?.trace.length ?? 0} validated tool calls</span>
                  </div>
                  {q.data && q.data.trace.length > 0 ? (
                    <ol className="divide-y divide-[var(--line-soft)]">
                      {q.data.trace.map((s) => (
                        <TraceRow key={s.i} step={s} />
                      ))}
                    </ol>
                  ) : (
                    <p className="text-xs text-ink-3">
                      {a.collector === "oracle"
                        ? "The scripted oracle has no tool trace: it follows fixed IK waypoints computed from simulator state."
                        : "No trace was recorded for this attempt."}
                    </p>
                  )}
                </div>
              </>
            ) : null}
          </div>
        </div>
      )}
    </dialog>
  );
}
