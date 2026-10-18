import { useEffect, useRef, useState, type ReactNode } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ArrowUpRight, Check, CircleDashed, Cpu, Play, Shuffle, Sparkles, Square, X } from "lucide-react";
import { api, resolveMediaUrl, type ApiError } from "../api/client";
import type { LivePolicyStatus, PhysicalSkill } from "../api/types";
import { Badge, Button, EmptyState, ErrorState, Skeleton } from "../components/ui";
import { GlowEdge } from "../components/ui/spotlight-card";
import { ago, cn } from "../lib/format";
import { LiveFrame } from "../chat/LivePolicy";
import { newSeed, runAction, scoreBadge, scoreWhere, simWhere, trainedLine } from "./skillView";

const label = "font-mono text-[10.5px] uppercase tracking-[0.14em]";

/** The GPU server's status, shared with the project Inference tab (same query). */
function useGpu() {
  return useQuery({
    queryKey: ["live-policy"],
    queryFn: api.livePolicy,
    refetchInterval: (q) => (q.state.data?.available ? 30_000 : 3_000),
  });
}

/** A recorded video as a still (its first frames); plays while hovered. */
function Thumb({ url, className }: { url: string | null; className?: string }) {
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
      preload="metadata"
      onMouseEnter={() => void ref.current?.play().catch(() => {})}
      onMouseLeave={() => ref.current?.pause()}
      className={cn("bg-rail object-cover", className)}
    />
  );
}

function SkillCard({ s, gpu, active, onOpen }: { s: PhysicalSkill; gpu?: LivePolicyStatus; active: boolean; onOpen: () => void }) {
  const b = scoreBadge(s.score);
  const live = runAction(s, gpu).kind === "play";
  return (
    <li>
      <button
        type="button"
        onClick={onOpen}
        aria-current={active || undefined}
        className={cn(
          "sk-depth group relative flex w-full flex-col overflow-hidden rounded border bg-plate text-left transition-colors",
          active ? "border-ink" : "border-line-strong hover:border-ink-3",
        )}
      >
        <GlowEdge />
        <Thumb url={s.thumbnail_url} className="aspect-[16/10] w-full border-b border-line" />
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
            {s.robot} · {s.policy} · trained in simulation
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

/** Run live: stream it on the GPU server if it's up; otherwise one explicit click requests a GPU (never automatic). */
function RunLive({ s }: { s: PhysicalSkill }) {
  const qc = useQueryClient();
  const gpu = useGpu();
  const refresh = () => {
    void qc.invalidateQueries({ queryKey: ["live-policy"] });
    void qc.invalidateQueries({ queryKey: ["skills"] });
  };
  const start = useMutation({ mutationFn: api.startLivePolicy, onSettled: refresh });
  const stop = useMutation({ mutationFn: api.stopLivePolicy, onSettled: refresh });
  const st = gpu.data;
  const a = runAction(s, st);
  // clicking a skill plays it straight away when the GPU server runs it (never requests a GPU by itself)
  const [seed, setSeed] = useState<number | null>(() => (a.kind === "play" ? newSeed() : null));
  const ready = a.kind === "play";
  useEffect(() => {
    if (ready) setSeed((cur) => cur ?? newSeed());
  }, [ready]);
  const err = (start.error ?? stop.error) as ApiError | null;
  const task = s.run.task ?? "";
  const end = () => {
    setSeed(null);
    // the run is saved when its stream ends; show it under past runs
    window.setTimeout(() => void qc.invalidateQueries({ queryKey: ["skills"] }), 4000);
  };

  if (a.kind === "checking")
    return (
      <p className="flex items-center gap-2 text-xs text-ink-3">
        <CircleDashed className="h-3.5 w-3.5 animate-spin" /> Checking the GPU server…
      </p>
    );
  if (a.kind === "unavailable") return <p className="text-[13px] leading-6 text-ink-2">{a.note}</p>;

  if (a.kind === "request" || a.kind === "starting") {
    const order = Object.keys(st?.steps ?? {});
    const at = st?.step ? order.indexOf(st.step) : start.isPending ? 0 : -1;
    const busy = a.kind === "starting" || start.isPending;
    return (
      <div className="space-y-3">
        <p className="text-[13px] leading-6 text-ink-2">
          {start.isPending && !st?.step ? "Requesting a GPU on Georgia Tech's Futurama cluster…" : (st?.reason ?? "The GPU server isn't running.")}
        </p>
        {at >= 0 && (
          <ol className="space-y-1 text-xs">
            {order.slice(0, -1).map((k, i) => (
              <li key={k} className={cn("flex items-center gap-2", i < at ? "text-ink-3" : i === at ? "text-ink" : "text-ink-4")}>
                {i < at ? <Check className="h-3 w-3" /> : i === at ? <CircleDashed className="h-3 w-3 animate-spin" /> : <span className="h-3 w-3" />}
                {st?.steps?.[k]}
              </li>
            ))}
          </ol>
        )}
        <div className="flex flex-wrap items-center gap-2">
          {!busy && (
            <Button variant="primary" size="sm" onClick={() => start.mutate()}>
              <Cpu className="h-3.5 w-3.5" /> {st?.error ? "Try again" : "Request a GPU"}
            </Button>
          )}
          {st?.job && (
            // a queued or starting job can be cancelled here: otherwise it holds a GPU for its walltime
            <Button size="sm" onClick={() => stop.mutate()} loading={stop.isPending}>
              <Square className="h-3 w-3" /> Cancel GPU job
            </Button>
          )}
        </div>
        {!busy && (
          <p className="text-[11.5px] leading-5 text-ink-3">
            Attaches to a GPU server already running on Futurama, or asks Slurm for one. Nothing starts until you click.
          </p>
        )}
        {err && err.message !== st?.reason && <p className="text-xs text-fault">{err.message}</p>}
      </div>
    );
  }

  // a.kind === "play": opening the skill plays it; "Run on a new layout" starts another run
  const layouts = s.layouts ?? st?.sim?.[task]?.layouts;
  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2">
        <Button variant="primary" size="sm" onClick={() => setSeed(newSeed())}>
          {seed == null ? <Play className="h-3.5 w-3.5" /> : <Shuffle className="h-3.5 w-3.5" />} {seed == null ? "Run live" : "Run on a new layout"}
        </Button>
        {seed != null && (
          <Button size="sm" variant="ghost" onClick={end}>
            <Square className="h-3 w-3" /> Stop
          </Button>
        )}
        <span className={cn("ml-auto text-ink-3", label)}>{st?.gpu}</span>
      </div>
      {seed != null && (
        <div className="aspect-[4/3] w-full bg-rail">
          <LiveFrame key={seed} src={api.livePolicySimUrl(task, seed)} label={`${s.name} running live on the GPU`} />
        </div>
      )}
      <p className="text-[11.5px] leading-5 text-ink-3">
        Closed loop in MuJoCo on the GPU; the simulator judges the end.{" "}
        Each run is {simWhere({ ...s, layouts: layouts ?? null })}. Saved to past runs when it ends.
      </p>
      <button type="button" onClick={() => stop.mutate()} disabled={stop.isPending} className="text-[11.5px] text-ink-3 underline underline-offset-2 hover:text-ink">
        {stop.isPending ? "Releasing the GPU…" : "Release the GPU when you're done"}
      </button>
    </div>
  );
}

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="border-t border-line px-5 py-4">
      <h3 className={cn("mb-2.5 text-ink-3", label)}>{title}</h3>
      {children}
    </section>
  );
}

function Detail({ s, onClose }: { s: PhysicalSkill; onClose: () => void }) {
  const b = scoreBadge(s.score);
  const [open, setOpen] = useState<string | null>(null);
  return (
    <div className="sk-depth relative rounded border border-line-strong bg-plate">
      <GlowEdge />
      <header className="flex items-start gap-3 px-5 pb-4 pt-5">
        <div className="min-w-0 flex-1">
          <h2 className="kn-display text-[26px] leading-tight">{s.name}</h2>
          {s.instruction && s.instruction.toLowerCase() !== s.name.toLowerCase() && <p className="mt-1.5 text-[13px] leading-5 text-ink-2">“{s.instruction}”</p>}
          <p className="mt-2 text-xs leading-5 text-ink-3">
            {[s.robot, s.policy, "trained in simulation", s.steps ? `${s.steps.toLocaleString()} steps` : null, `weights ${s.weights}`]
              .filter(Boolean)
              .join(" · ")}
          </p>
          <p className="mt-0.5 text-xs text-ink-3">{trainedLine(s)}</p>
        </div>
        <button type="button" onClick={onClose} aria-label="Close" className="flex h-8 w-8 shrink-0 items-center justify-center rounded text-ink-3 hover:bg-rail hover:text-ink">
          <X className="h-4 w-4" />
        </button>
      </header>

      <Section title="Score">
        <div className="flex flex-wrap items-center gap-2">
          <Badge tone={b.tone}>{b.text}</Badge>
          <span className="text-[13.5px] text-ink">{s.score.label}</span>
        </div>
        <p className="mt-2 text-[12.5px] leading-5 text-ink-2">{s.score.detail}</p>
        {s.evidence.some((e) => e.video_url) && (
          <ul className="mt-3 grid grid-cols-3 gap-2">
            {s.evidence
              .filter((e) => e.video_url)
              .slice(0, 6)
              .map((e) => (
                <li key={e.label} className="min-w-0">
                  <Thumb url={e.video_url} className="aspect-[4/3] w-full rounded-sm border border-line" />
                  <p className="mt-1 truncate text-[10.5px] text-ink-3">
                    {e.label}
                    {e.success != null && <span className={e.success ? "text-pass" : "text-fault"}> · {e.success ? "success" : "failed"}</span>}
                    {e.joint_mae_deg != null && ` · ${e.joint_mae_deg.toFixed(2)}°`}
                  </p>
                </li>
              ))}
          </ul>
        )}
      </Section>

      <Section title="Run live">
        <RunLive key={s.id} s={s} />
      </Section>

      <Section title={`Past live runs${s.runs.length ? ` · ${s.runs.length}` : ""}`}>
        {s.runs.length ? (
          <ul className="space-y-2">
            {s.runs.map((r) => (
              <li key={r.id} className="rounded-sm border border-line">
                <button type="button" onClick={() => setOpen(open === r.id ? null : r.id)} className="flex w-full items-center gap-3 px-3 py-2 text-left hover:bg-rail/50">
                  <span
                    className={cn("h-1.5 w-1.5 shrink-0 rounded-full", r.success === true ? "bg-pass" : r.success === false ? "bg-fault" : "bg-ink-4")}
                    aria-hidden
                  />
                  <span className="min-w-0 flex-1 truncate text-[13px] text-ink-2">
                    {r.label}
                    <span className="text-ink-3">
                      {" · "}
                      {r.partial ? "cut off, not judged" : r.success === true ? "succeeded" : r.success === false ? "failed" : r.joint_mae_deg != null ? `${r.joint_mae_deg.toFixed(2)}° joint error` : "recorded"}
                      {r.inference_ms != null && ` · ${Math.round(r.inference_ms)} ms/frame`}
                    </span>
                  </span>
                  <span className="shrink-0 text-[11px] text-ink-3">{ago(r.when)}</span>
                </button>
                {open === r.id && (
                  <div className="border-t border-line p-3">
                    {r.video_url ? (
                      <video src={resolveMediaUrl(r.video_url) ?? undefined} controls autoPlay muted playsInline className="block w-full bg-rail" />
                    ) : (
                      <p className="text-xs text-ink-3">No video was kept for this run.</p>
                    )}
                    <p className="mt-2 text-[11.5px] leading-5 text-ink-3">{r.note}</p>
                  </div>
                )}
              </li>
            ))}
          </ul>
        ) : (
          <p className="text-[12.5px] text-ink-3">No live runs yet. Run it live above and it's saved here.</p>
        )}
      </Section>

      {s.project_id && (
        <div className="border-t border-line px-5 py-3">
          <Link to={`/p/${s.project_id}`} className="inline-flex items-center gap-1 text-[12.5px] text-ink-2 hover:text-ink">
            Open its project{s.project_name ? `: ${s.project_name}` : ""} <ArrowUpRight className="h-3.5 w-3.5" />
          </Link>
        </div>
      )}
    </div>
  );
}

function GpuPill({ st }: { st?: LivePolicyStatus }) {
  if (!st) return null;
  return (
    <span className={cn("inline-flex items-center gap-1.5 text-ink-3", label)} title={st.available ? undefined : st.reason}>
      <span className={cn("h-1.5 w-1.5 rounded-full", st.available ? "live-dot bg-pass" : st.phase === "starting" ? "bg-caution" : "bg-ink-4")} />
      {st.available ? `GPU server live${st.gpu ? ` · ${st.gpu}` : ""}` : st.phase === "starting" ? "GPU starting" : "GPU server off"}
    </span>
  );
}

/** /studio/skills: every trained policy across projects, each runnable live on the GPU (#276). */
export function SkillsPage() {
  const { sid } = useParams<{ sid: string }>();
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
  const sel = sid ? list.find((s) => s.id === sid) : undefined;

  return (
    <div className="h-full overflow-y-auto scroll-thin">
      <div className="mx-auto max-w-[1560px] px-4 pb-16 pt-8 sm:px-6">
        <div className="kn-label kn-label-rule">
          <span className="text-ink">Physical Skills</span>
        </div>
        <div className="mt-4 flex flex-wrap items-end gap-x-6 gap-y-3">
          <div className="min-w-0">
            <h1 className="kn-display text-[clamp(1.9rem,3.6vw,2.6rem)]">What your robot can do</h1>
            <p className="mt-2 max-w-[44rem] text-sm leading-6 text-ink-2">
              Every policy you've trained, across all projects. Open one to see how it was measured and run it live on the GPU.
            </p>
          </div>
          <div className="ml-auto">
            <GpuPill st={gpu.data} />
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
            <div className={cn("grid gap-6", sel && "lg:grid-cols-[minmax(0,1fr)_minmax(400px,540px)]")}>
              {sel && (
                <aside className="lg:sticky lg:top-0 lg:order-2 lg:self-start">
                  <Detail s={sel} onClose={() => nav("/studio/skills")} />
                </aside>
              )}
              <ul className={cn("grid content-start gap-3 sm:grid-cols-2", sel ? "xl:grid-cols-2 2xl:grid-cols-3" : "lg:grid-cols-3 xl:grid-cols-4")}>
                {list.map((s) => (
                  <SkillCard key={s.id} s={s} gpu={gpu.data} active={s.id === sel?.id} onOpen={() => nav(`/studio/skills/${encodeURIComponent(s.id)}`)} />
                ))}
              </ul>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
