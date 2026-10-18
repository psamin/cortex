import { useEffect, useMemo, useRef, useState } from "react";
import { CircleDashed, Columns2, Pause, Play, Radio, VideoOff } from "lucide-react";
import { api } from "../api/client";
import { isJobActive, useJobs } from "../api/events";
import type { Job } from "../api/types";
import { LightGlyph } from "../components/Workcell";
import { cn } from "../lib/format";
import { useRolloutVideo, useRollouts, type Camera, type CameraInfo, type Rollout } from "./rollouts";

export type RolloutFilter = "all" | "demo" | "eval" | "failed";

export interface ViewerSelection {
  selectedId: string | null;
  setSelectedId: (id: string | null) => void;
  filter: RolloutFilter;
  setFilter: (f: RolloutFilter) => void;
  /** restrict to one evaluation (set from the iteration timeline / failure analysis) */
  evaluationId: string | null;
  setEvaluationId: (id: string | null) => void;
}

const LIVE_KINDS = /record|collect|eval|sim/i;

export function Viewer({
  pid,
  goal,
  asked,
  activity,
  sel,
}: {
  pid: string;
  goal: string;
  /** the user's request when the simulated task differs from it */
  asked?: string | null;
  activity: { label: string; busy: boolean } | null;
  sel: ViewerSelection;
}) {
  const rollouts = useRollouts(pid);
  const jobs = useJobs();
  const liveJob: Job | undefined = useMemo(
    () => Object.values(jobs).find((j) => j.project_id === pid && isJobActive(j) && LIVE_KINDS.test(j.kind)),
    [jobs, pid],
  );
  const [mode, setMode] = useState<"auto" | "live" | "replay">("auto");
  const showLive = mode === "live" || (mode === "auto" && !!liveJob);
  const [camera, setCamera] = useState<Camera>("presentation");
  const [compare, setCompare] = useState(false);

  const filtered = useMemo(() => {
    let xs = rollouts.data ?? [];
    if (sel.evaluationId) xs = xs.filter((r) => r.evaluationId === sel.evaluationId);
    if (sel.filter === "demo") xs = xs.filter((r) => r.kind === "demo");
    if (sel.filter === "eval") xs = xs.filter((r) => r.kind === "eval");
    if (sel.filter === "failed") xs = xs.filter((r) => r.success === false);
    return xs;
  }, [rollouts.data, sel.filter, sel.evaluationId]);

  const current = filtered.find((r) => r.id === sel.selectedId) ?? filtered[0] ?? null;
  const counterpart = useMemo(() => {
    if (!current || current.success === null) return null;
    const pool = (rollouts.data ?? []).filter((r) => r.success === !current.success && r.id !== current.id);
    const sameCond = pool.find((r) => r.condition && current.condition && r.condition.cube_region === current.condition.cube_region && r.condition.lighting === current.condition.lighting);
    return sameCond ?? pool.find((r) => r.kind === current.kind) ?? pool[0] ?? null;
  }, [current, rollouts.data]);
  // only the cameras this rollout actually has (backend wording); fall back to its first camera
  const cams = current?.cameras ?? [];
  const cam = cams.find((c) => c.id === camera) ?? cams[0] ?? null;
  useEffect(() => {
    if (cam && cam.id !== camera) setCamera(cam.id);
  }, [cam, camera]);

  const counts = useMemo(() => {
    const xs = rollouts.data ?? [];
    return { all: xs.length, demo: xs.filter((r) => r.kind === "demo").length, eval: xs.filter((r) => r.kind === "eval").length, failed: xs.filter((r) => r.success === false).length };
  }, [rollouts.data]);

  return (
    <section className="flex h-full min-h-0 flex-col bg-[#0b0c0e] text-[#eceDef]" aria-label="Robot simulation viewer">
      {/* toolbar */}
      <div className="flex items-center gap-2 border-b border-white/[.07] px-4 py-2.5">
        <div className="flex shrink-0 rounded-lg bg-white/[.05] p-0.5">
          <ToolbarTab
            active={showLive}
            onClick={() => setMode("live")}
            disabled={!liveJob && mode !== "live"}
            title={liveJob ? "Watch the simulation live" : "Nothing is running; the live view switches on by itself when a job starts"}
          >
            <Radio className={cn("h-3.5 w-3.5", liveJob && "text-[#ec6060]")} /> Live
          </ToolbarTab>
          <ToolbarTab active={!showLive} onClick={() => setMode("replay")}>
            Replays
          </ToolbarTab>
        </div>
        {!showLive && (
          <div className="flex min-w-0 flex-1 items-center gap-1 overflow-x-auto scroll-thin">
            {(
              [
                ["all", "All", counts.all],
                ["demo", "Demos", counts.demo],
                ["eval", "Eval rollouts", counts.eval],
                ["failed", "Failures", counts.failed],
              ] as const
            ).map(([k, label, n]) => (
              <button
                key={k}
                type="button"
                onClick={() => sel.setFilter(k)}
                className={cn(
                  "inline-flex h-7 shrink-0 items-center gap-1.5 whitespace-nowrap rounded-full px-2.5 text-xs transition-colors",
                  sel.filter === k ? "bg-white/[.12] text-white" : "text-white/55 hover:bg-white/[.06] hover:text-white/80",
                )}
              >
                {label}
                <span className="num font-mono text-[10.5px] text-white/40">{n}</span>
              </button>
            ))}
            {sel.evaluationId && (
              <button type="button" onClick={() => sel.setEvaluationId(null)} className="inline-flex h-7 shrink-0 items-center gap-1 whitespace-nowrap rounded-full bg-[#f0a43a]/15 px-3 text-xs text-[#f0a43a]">
                one evaluation · clear ×
              </button>
            )}
          </div>
        )}
        {!showLive && current && (
          <div className="ml-auto flex shrink-0 items-center gap-1.5">
            <select
              value={cam?.id ?? "presentation"}
              onChange={(e) => setCamera(e.target.value as Camera)}
              disabled={cams.length < 2}
              title={cam?.detail || cam?.label}
              className="h-7 max-w-[180px] rounded-md bg-white/[.06] px-2 text-xs text-white/80 outline-none disabled:opacity-70"
              aria-label="Camera"
            >
              {cams.map((c) => (
                <option key={c.id} value={c.id} title={c.detail}>
                  {c.label}
                </option>
              ))}
            </select>
            <button
              type="button"
              disabled={!counterpart}
              onClick={() => setCompare((c) => !c)}
              title={counterpart ? "Side by side: success vs failure" : "Needs both a successful and a failed rollout"}
              className={cn(
                "inline-flex h-7 items-center gap-1.5 rounded-md px-2.5 text-xs transition-colors disabled:opacity-35",
                compare ? "bg-white/[.14] text-white" : "bg-white/[.06] text-white/75 hover:text-white",
              )}
            >
              <Columns2 className="h-3.5 w-3.5" /> Compare
            </button>
          </div>
        )}
      </div>

      <Hud goal={goal} asked={asked} activity={activity} liveJob={showLive ? liveJob : undefined} rollout={showLive || compare ? null : current} cam={showLive ? null : cam} />

      {/* stage */}
      <div className="relative min-h-0 flex-1">
        {showLive ? (
          <LiveStage pid={pid} job={liveJob} />
        ) : rollouts.isLoading ? (
          <div className="flex h-full items-center justify-center">
            <CircleDashed className="h-6 w-6 animate-spin text-white/30" />
          </div>
        ) : !current ? (
          <div className="flex h-full flex-col items-center justify-center gap-2 px-8 text-center">
            <VideoOff className="h-7 w-7 text-white/25" />
            <p className="text-sm text-white/60">{(rollouts.data ?? []).length ? "Nothing matches this filter." : "No simulation videos yet."}</p>
            <p className="max-w-sm text-xs text-white/40">Ask the agent to collect demonstrations. The robot appears here live while it works.</p>
          </div>
        ) : compare && counterpart ? (
          <CompareStage a={current} b={counterpart} camera={camera} />
        ) : (
          <SingleStage r={current} camera={camera} />
        )}
      </div>

      {/* selector */}
      {!showLive && filtered.length > 0 && (
        <div className="border-t border-white/[.07]">
          <div className="flex gap-1.5 overflow-x-auto px-4 py-2.5 scroll-thin">
            {filtered.slice(0, 80).map((r) => (
              <button
                key={r.id}
                type="button"
                onClick={() => sel.setSelectedId(r.id)}
                className={cn(
                  "flex shrink-0 items-center gap-2 rounded-lg px-2.5 py-1.5 text-left text-xs transition-colors",
                  r.id === current?.id ? "bg-white/[.13] text-white" : "bg-white/[.04] text-white/65 hover:bg-white/[.08]",
                )}
              >
                <span className={cn("h-2 w-2 shrink-0 rounded-full", r.success === null ? "bg-white/30" : r.success ? "bg-[#4ac480]" : "bg-[#ec6060]")} />
                <span className="whitespace-nowrap">{r.kind === "demo" ? r.label : `Eval · ${r.label}`}</span>
                {r.imported && <span className="whitespace-nowrap rounded bg-white/[.08] px-1.5 text-[10.5px] text-white/60">human · imported</span>}
                {r.condition && !r.imported && (
                  <span className="inline-flex items-center gap-1 whitespace-nowrap capitalize text-white/40">
                    <LightGlyph lighting={r.condition.lighting} />
                    {r.condition.cube_region}
                  </span>
                )}
              </button>
            ))}
          </div>
        </div>
      )}
    </section>
  );
}

function ToolbarTab({
  active,
  onClick,
  disabled,
  title,
  children,
}: {
  active: boolean;
  onClick: () => void;
  disabled?: boolean;
  title?: string;
  children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      title={title}
      className={cn("inline-flex h-7 items-center gap-1.5 rounded-md px-3 text-xs font-medium transition-colors disabled:opacity-35", active ? "bg-white/[.12] text-white" : "text-white/55 hover:text-white/85")}
    >
      {children}
    </button>
  );
}

// ------------------------------------------------------------------ stages
function LiveStage({ pid, job }: { pid: string; job?: Job }) {
  const [failed, setFailed] = useState(false);
  const src = api.simLiveUrl(pid);
  return (
    <div className="flex h-full items-center justify-center p-4">
      {failed || !job ? (
        <div className="text-center text-sm text-white/50">{job ? "Live stream unavailable" : "Nothing is running right now."}</div>
      ) : (
        <img src={src} alt="Live MuJoCo simulation" onError={() => setFailed(true)} className="max-h-full max-w-full rounded-lg object-contain" />
      )}
      {job && (
        <div className="absolute inset-x-6 bottom-5">
          <div className="mb-1.5 flex justify-between text-xs text-white/70">
            <span className="truncate">{job.message || job.kind.replace(/_/g, " ")}</span>
            <span className="num font-mono">{Math.round(job.progress * 100)}%</span>
          </div>
          <div className="h-1 overflow-hidden rounded-full bg-white/15">
            <div className="h-full rounded-full bg-[#f0a43a] transition-[width] duration-500" style={{ width: `${job.progress * 100}%` }} />
          </div>
        </div>
      )}
    </div>
  );
}

function SingleStage({ r, camera }: { r: Rollout; camera: Camera }) {
  const v = useRolloutVideo(r, camera);
  const ref = useRef<HTMLVideoElement>(null);
  return (
    <div className="flex h-full flex-col">
      <div className="flex min-h-0 flex-1 items-center justify-center p-4 pb-2">
        <VideoSurface url={v.data} loading={v.isLoading} videoRef={ref} />
      </div>
      <Controls refs={[ref]} url={v.data} />
    </div>
  );
}

function CompareStage({ a, b, camera }: { a: Rollout; b: Rollout; camera: Camera }) {
  const has = (r: Rollout) => (r.cameras.some((c) => c.id === camera) ? camera : r.cameras[0]?.id ?? "presentation");
  const va = useRolloutVideo(a, has(a));
  const vb = useRolloutVideo(b, has(b));
  const ra = useRef<HTMLVideoElement>(null);
  const rb = useRef<HTMLVideoElement>(null);
  const [ok, bad] = a.success ? [a, b] : [b, a];
  const [vOk, vBad, rOk, rBad] = a.success ? [va, vb, ra, rb] : [vb, va, rb, ra];
  return (
    <div className="flex h-full flex-col">
      <div className="grid min-h-0 flex-1 grid-cols-2 gap-3 p-4 pb-2">
        {(
          [
            [ok, vOk, rOk, "Succeeded", "#4ac480"],
            [bad, vBad, rBad, "Failed", "#ec6060"],
          ] as const
        ).map(([r, v, ref, label, color]) => (
          <div key={r.id} className="flex min-h-0 flex-col gap-1.5">
            <div className="flex items-center gap-2 text-xs">
              <span className="h-2 w-2 rounded-full" style={{ background: color }} />
              <span className="font-medium text-white/85">{label}</span>
              <span className="truncate text-white/45">
                {r.kind === "demo" ? r.label : r.label}
                {r.failureMode ? ` · ${r.failureMode.replace(/_/g, " ")}` : ""}
              </span>
            </div>
            <div className="flex min-h-0 flex-1 items-center justify-center">
              <VideoSurface url={v.data} loading={v.isLoading} videoRef={ref} />
            </div>
          </div>
        ))}
      </div>
      <Controls refs={[ra, rb]} url={va.data && vb.data ? `${va.data}|${vb.data}` : null} />
    </div>
  );
}

function VideoSurface({ url, loading, videoRef }: { url: string | null | undefined; loading: boolean; videoRef: React.RefObject<HTMLVideoElement> }) {
  const [failed, setFailed] = useState<string | null>(null);
  if (loading) return <CircleDashed className="h-6 w-6 animate-spin text-white/30" />;
  if (!url || failed === url)
    return (
      <div className="flex flex-col items-center gap-2 text-white/40">
        <VideoOff className="h-6 w-6" />
        <span className="text-xs">{url ? "This camera has no video for this episode" : "No video recorded for this episode"}</span>
      </div>
    );
  return <video key={url} ref={videoRef} src={url} autoPlay muted loop playsInline onError={() => setFailed(url)} className="max-h-full max-w-full rounded-lg bg-black object-contain" />;
}

/** Play/pause + scrubbable timeline; drives one or two synced videos. */
function Controls({ refs, url }: { refs: React.RefObject<HTMLVideoElement>[]; url: string | null | undefined }) {
  const [t, setT] = useState(0);
  const [dur, setDur] = useState(0);
  const [playing, setPlaying] = useState(true);
  useEffect(() => {
    const v = refs[0].current;
    if (!v) return;
    const onTime = () => setT(v.currentTime);
    const onMeta = () => setDur(v.duration || 0);
    const onPlay = () => setPlaying(true);
    const onPause = () => setPlaying(false);
    v.addEventListener("timeupdate", onTime);
    v.addEventListener("loadedmetadata", onMeta);
    v.addEventListener("play", onPlay);
    v.addEventListener("pause", onPause);
    onMeta();
    return () => {
      v.removeEventListener("timeupdate", onTime);
      v.removeEventListener("loadedmetadata", onMeta);
      v.removeEventListener("play", onPlay);
      v.removeEventListener("pause", onPause);
    };
  }, [url]); // eslint-disable-line react-hooks/exhaustive-deps
  const all = () => refs.map((r) => r.current).filter((v): v is HTMLVideoElement => !!v);
  const toggle = () => {
    const vs = all();
    if (!vs.length) return;
    if (vs[0].paused) vs.forEach((v) => void v.play());
    else vs.forEach((v) => v.pause());
  };
  const seek = (x: number) => all().forEach((v) => (v.currentTime = x));
  const fmt = (s: number) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
  return (
    <div className="flex items-center gap-3 px-5 pb-3">
      <button type="button" onClick={toggle} disabled={!url} className="flex h-8 w-8 items-center justify-center rounded-full bg-white/[.1] text-white hover:bg-white/[.16] disabled:opacity-30" aria-label={playing ? "Pause" : "Play"}>
        {playing ? <Pause className="h-3.5 w-3.5" /> : <Play className="ml-0.5 h-3.5 w-3.5" />}
      </button>
      <input
        type="range"
        min={0}
        max={dur || 0}
        step={0.01}
        value={Math.min(t, dur || 0)}
        disabled={!url || !dur}
        onChange={(e) => seek(Number(e.target.value))}
        className="h-1 flex-1 cursor-pointer accent-[#f0a43a]"
        aria-label="Timeline"
      />
      <span className="num w-20 text-right font-mono text-[11px] text-white/55">
        {fmt(t)} / {fmt(dur)}
      </span>
    </div>
  );
}

/** Heads-up band above the stage: the task and what is happening now, plus what is on screen. */
function Hud({
  goal,
  asked,
  activity,
  liveJob,
  rollout,
  cam,
}: {
  goal: string;
  asked?: string | null;
  activity: { label: string; busy: boolean } | null;
  liveJob?: Job;
  rollout: Rollout | null;
  cam: CameraInfo | null;
}) {
  return (
    <div className="flex items-start justify-between gap-4 px-5 pt-3">
      <div className="min-w-0">
        <div className="text-[10px] font-medium uppercase tracking-[0.1em] text-white/40">{asked ? "Simulated task" : "Task"}</div>
        <div className="truncate text-[15px] text-white/90">{goal}</div>
        {asked && (
          <div className="truncate text-xs text-white/45" title={`The simulator only has the cube-and-bowl scene. You asked: ${asked}`}>
            You asked: “{asked}”. The simulator only has the cube-and-bowl scene.
          </div>
        )}
        <div className="mt-0.5 flex items-center gap-1.5 text-xs text-white/60">
          {liveJob ? (
            <>
              <span className="h-1.5 w-1.5 rounded-full bg-[#ec6060] live-dot" />
              <span title="Rendered for people; not what the policy sees">Live · studio view</span>
              <span className="text-white/40">· {liveJob.message || liveJob.kind.replace(/_/g, " ")}</span>
            </>
          ) : activity?.busy ? (
            <>
              <CircleDashed className="h-3 w-3 animate-spin text-[#f0a43a]" /> {activity.label}
            </>
          ) : (
            <>
              <span className="h-1.5 w-1.5 rounded-full bg-white/30" /> {activity?.label ?? "Idle"}
            </>
          )}
        </div>
      </div>
      {rollout && (
        <div className="flex shrink-0 flex-col items-end gap-1 pt-0.5 text-xs">
          <div className="flex items-center gap-1.5">
            <span className="text-white/60">
              {rollout.kind === "demo" ? "Demonstration" : "Evaluation rollout"}
              {rollout.policy ? ` · ${rollout.policy}` : ""}
            </span>
            {rollout.imported && <span className="rounded-full bg-white/[.1] px-2 py-0.5 font-medium text-white/80">Human demo · imported</span>}
            {rollout.success !== null && (
              <span className={cn("rounded-full px-2 py-0.5 font-medium", rollout.success ? "bg-[#4ac480]/15 text-[#7ddca5]" : "bg-[#ec6060]/15 text-[#f39a9a]")}>
                {rollout.success ? "Success · simulator-verified" : `Failed${rollout.failureMode ? ` · ${rollout.failureMode.replace(/_/g, " ")}` : ""}`}
              </span>
            )}
          </div>
          {rollout.imported ? (
            <span className="max-w-[360px] truncate text-white/50" title={[rollout.source, rollout.task].filter(Boolean).join(" · ")}>
              {rollout.source ?? "imported dataset"}
              {rollout.task ? ` · “${rollout.task}”` : ""} · no simulator verdict
            </span>
          ) : rollout.condition && (
            <span className="inline-flex items-center gap-1.5 text-white/50">
              <LightGlyph lighting={rollout.condition.lighting} />
              <span className="capitalize">
                cube {rollout.condition.cube_region} · {rollout.condition.lighting} light
              </span>
              {rollout.seed !== null && <span className="font-mono text-white/35">seed {rollout.seed}</span>}
            </span>
          )}
          {cam && (
            <span className="max-w-[360px] truncate text-white/40" title={cam.detail}>
              {cam.label}
              {cam.detail ? ` · ${cam.detail}` : ""}
            </span>
          )}
        </div>
      )}
    </div>
  );
}
