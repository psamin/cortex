import { useEffect, useRef } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ArrowLeft, CircleDashed, Cpu, Sparkles, Square } from "lucide-react";
import { api, resolveMediaUrl, type ApiError } from "../api/client";
import type { LivePolicyStatus, PhysicalSkill } from "../api/types";
import { Badge, Button, EmptyState, ErrorState, Skeleton } from "../components/ui";
import { GlowEdge } from "../components/ui/spotlight-card";
import { cn } from "../lib/format";
import { runAction, scoreBadge, scoreWhere, trainedLine } from "./skillView";

const label = "font-mono text-[10.5px] uppercase tracking-[0.14em]";

/** The GPU server's status, shared with the project Inference tab (same query). Read-only: never requests a GPU. */
export function useGpu() {
  return useQuery({
    queryKey: ["live-policy"],
    queryFn: api.livePolicy,
    refetchInterval: (q) => (q.state.data?.available ? 30_000 : 3_000),
  });
}

/** A recorded video as a still (its first frames); plays while hovered, or loops muted with `autoPlay`. */
export function Thumb({ url, className, autoPlay = false }: { url: string | null; className?: string; autoPlay?: boolean }) {
  const ref = useRef<HTMLVideoElement>(null);
  const src = resolveMediaUrl(url);
  if (!src)
    return (
      <div className={cn("flex items-center justify-center bg-rail text-ink-4", className)}>
        <Sparkles className="h-6 w-6" strokeWidth={1.25} />
      </div>
    );
  return (
    <video
      ref={ref}
      src={`${src}#t=0.5`}
      muted
      loop
      playsInline
      autoPlay={autoPlay}
      preload={autoPlay ? "auto" : "metadata"}
      onMouseEnter={() => void ref.current?.play().catch(() => {})}
      onMouseLeave={() => !autoPlay && ref.current?.pause()}
      className={cn("bg-rail object-cover", className)}
    />
  );
}

function SkillCard({ s, gpu, onOpen }: { s: PhysicalSkill; gpu?: LivePolicyStatus; onOpen: () => void }) {
  const b = scoreBadge(s.score);
  const live = runAction(s, gpu).kind === "play";
  return (
    <li>
      <button
        type="button"
        onClick={onOpen}
        className="sk-depth group relative flex w-full flex-col overflow-hidden rounded border border-line-strong bg-plate text-left transition-colors hover:border-ink-3"
      >
        <GlowEdge />
        {/* a recorded replay plays on its own, so skills show what they do even with no GPU server up (#308) */}
        <Thumb url={s.replays[0] ?? s.thumbnail_url} autoPlay={!!s.replays.length} className="aspect-[16/10] w-full border-b border-line" />
        <div className="flex flex-1 flex-col p-3.5">
          <div className="flex items-start gap-2">
            <h3 className="min-w-0 flex-1 text-[15px] font-medium leading-snug text-ink">{s.name}</h3>
            {live && (
              <span className={cn("inline-flex shrink-0 items-center gap-1 text-pass", label, "text-[9.5px]")} title="On the GPU server now">
                <span className="live-dot h-1.5 w-1.5 rounded-full bg-pass" /> live
              </span>
            )}
          </div>
          <p className="mt-1 text-xs text-ink-3">
            {[s.robot, s.policy, "trained in simulation"].filter(Boolean).join(" · ")}
          </p>
          <div className="mt-3 flex flex-wrap items-center gap-2">
            <Badge tone={b.tone} title={s.score.label}>
              {b.text}
            </Badge>
            {scoreWhere(s.score) && <span className="text-[11px] text-ink-3">{scoreWhere(s.score)}</span>}
          </div>
          <p className="mt-auto pt-3 text-[11.5px] text-ink-3">{trainedLine(s) || "Trained elsewhere"}</p>
        </div>
      </button>
    </li>
  );
}

/** `compact` (the viewer's one-row header): just the dot below 2xl, the words in its tooltip. */
export function GpuPill({ st, compact = false }: { st?: LivePolicyStatus; compact?: boolean }) {
  if (!st) return null;
  const text = st.available ? `GPU server live${st.gpu ? ` · ${st.gpu}` : ""}` : st.phase === "starting" ? "GPU starting" : "GPU server off";
  return (
    <span className={cn("inline-flex items-center gap-1.5 text-ink-3", label)} title={st.available ? (compact ? text : undefined) : compact ? `${text}: ${st.reason ?? ""}` : st.reason}>
      <span className={cn("h-1.5 w-1.5 rounded-full", st.available ? "live-dot bg-pass" : st.phase === "starting" ? "bg-caution" : "bg-ink-4")} />
      <span className={compact ? "hidden 2xl:inline" : undefined}>{text}</span>
    </span>
  );
}

/** The GPU server, from anywhere on Physical Skills (#308): "Request a GPU" (one explicit click; the same request as
 *  the project Inference tab) with its step narration, or "Stop & release GPU" once it's live. */
export function GpuControl({ compact = false }: { compact?: boolean }) {
  const qc = useQueryClient();
  const gpu = useGpu();
  const refresh = () => {
    void qc.invalidateQueries({ queryKey: ["live-policy"] });
    void qc.invalidateQueries({ queryKey: ["skills"] });
  };
  const start = useMutation({ mutationFn: api.startLivePolicy, onSettled: refresh });
  const stop = useMutation({ mutationFn: api.stopLivePolicy, onSettled: refresh });
  const st = gpu.data;
  const err = (start.error ?? stop.error) as ApiError | null;
  const busy = st?.phase === "starting" || start.isPending;
  const step = start.isPending && !st?.step ? "Requesting a GPU on Georgia Tech's Futurama cluster…" : st?.step ? st.steps?.[st.step] : null;
  return (
    <div className="flex flex-col items-end gap-1">
      <div className="flex flex-wrap items-center justify-end gap-2">
        <GpuPill st={st} compact={compact} />
        {st?.available ? (
          <Button size="sm" onClick={() => stop.mutate()} loading={stop.isPending} title="Ends the live session and gives the GPU back to the cluster">
            <Square className="h-3 w-3" /> {stop.isPending ? "Releasing the GPU…" : "Stop & release GPU"}
          </Button>
        ) : busy ? (
          <>
            <span className="inline-flex items-center gap-1.5 text-xs text-ink-2">
              <CircleDashed className="h-3.5 w-3.5 animate-spin" /> {step ?? "Starting the GPU server…"}
            </span>
            {st?.job && (
              // a queued or starting job can be cancelled: otherwise it holds a GPU for its walltime
              <Button size="sm" onClick={() => stop.mutate()} loading={stop.isPending}>
                <Square className="h-3 w-3" /> Cancel GPU job
              </Button>
            )}
          </>
        ) : (
          st && (
            <Button variant="primary" size="sm" onClick={() => start.mutate()}>
              <Cpu className="h-3.5 w-3.5" /> {st.error ? "Try again" : "Request a GPU"}
            </Button>
          )
        )}
      </div>
      {err ? <p className="max-w-md text-right text-xs text-fault">{err.message}</p> : st?.error && !busy && <p className="max-w-md text-right text-xs text-fault">{st.error}</p>}
    </div>
  );
}

/** /studio/skills: every trained policy across projects, each runnable live on the GPU (#276). Opening one goes to
 *  the full-screen viewer, /studio/skills/:sid (#308). */
export function SkillsPage() {
  const nav = useNavigate();
  const qc = useQueryClient();
  const skills = useQuery({ queryKey: ["skills"], queryFn: api.skills, refetchInterval: 30_000 });
  const gpu = useGpu();
  const available = gpu.data?.available;
  useEffect(() => {
    // the list's run notes depend on the GPU server: read it again when the server comes up or goes away
    if (available !== undefined) void qc.invalidateQueries({ queryKey: ["skills"] });
  }, [available, qc]);
  const list = skills.data ?? [];

  return (
    <div className="h-full overflow-y-auto scroll-thin">
      <div className="mx-auto max-w-[1560px] px-4 pb-16 pt-8 sm:px-6">
        <Link to="/studio" className="mb-4 inline-flex h-8 items-center gap-1.5 rounded-full border border-line-strong px-3 text-[13px] text-ink-2 hover:text-ink">
          <ArrowLeft className="h-3.5 w-3.5" /> Back to studio
        </Link>
        <div className="kn-label kn-label-rule">
          <span className="text-ink">Physical Skills</span>
        </div>
        <div className="mt-4 flex flex-wrap items-end gap-x-6 gap-y-3">
          <div className="min-w-0">
            <h1 className="kn-display text-[clamp(1.9rem,3.6vw,2.6rem)]">What your robot can do</h1>
            <p className="mt-2 max-w-[44rem] text-sm leading-6 text-ink-2">
              Every policy you've trained, across all projects. Open one to watch its recorded runs, or run it live on the GPU.
            </p>
          </div>
          <div className="ml-auto">
            <GpuControl />
          </div>
        </div>

        <div className="mt-6">
          {skills.isLoading ? (
            <ul className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
              {[0, 1, 2, 3].map((i) => (
                <li key={i}>
                  <Skeleton className="h-[280px]" />
                </li>
              ))}
            </ul>
          ) : skills.error ? (
            <ErrorState error={skills.error} onRetry={() => skills.refetch()} />
          ) : !list.length ? (
            <EmptyState
              title="No trained skills yet"
              action={
                <Link to="/studio" className="sk-pill sk-pill-primary inline-flex h-8 items-center px-4 text-[12.5px] font-medium">
                  Start a project
                </Link>
              }
            >
              Train a policy in any project and it shows up here, ready to run.
            </EmptyState>
          ) : (
            <ul className="grid content-start gap-3 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
              {list.map((s) => (
                <SkillCard key={s.id} s={s} gpu={gpu.data} onOpen={() => nav(`/studio/skills/${encodeURIComponent(s.id)}`)} />
              ))}
            </ul>
          )}
        </div>
      </div>
    </div>
  );
}
