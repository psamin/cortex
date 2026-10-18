import { useEffect, useMemo, useRef, useState } from "react";
import { CircleDashed, Columns2, Cpu, Pause, Play, Radio, VideoOff } from "lucide-react";
import { api } from "../api/client";
import { isJobActive, useJobs } from "../api/events";
import { useDatasets } from "../api/hooks";
import { augmentedByEpisode } from "../lib/augment";
import type { Job } from "../api/types";
import { conditionWords } from "../components/Workcell";
import { cn } from "../lib/format";
import { DashedOrbit, SketchScene } from "../sketch";
import { InfoTip } from "../components/ui";
import { useRolloutVideo, useRollouts, type Camera, type Rollout, type RolloutOrigin } from "./rollouts";
import { GpuStage } from "./LivePolicy";

export type RolloutFilter = "all" | RolloutOrigin | "demo" | "failed";

const ORIGIN_LABEL: Record<RolloutOrigin, string> = {
  agent: "Agent",
  oracle: "Scripted oracle",
  imported: "Human imported",
  teleop: "Real teleop",
  eval: "Evaluations",
  attempt: "Failed attempts",
};
const ORIGIN_ORDER: RolloutOrigin[] = ["agent", "oracle", "imported", "teleop", "eval", "attempt"];

/** ✓ / ✗ failure / "not judged" */
const mark = (r: Rollout) => (r.success === null ? "not judged" : r.success ? "✓" : `✗${r.failureMode ? ` ${r.failureMode.replace(/_/g, " ")}` : ""}`);

/** "#27 · Claude Code · cube center, bright light · ✓" */
function chipText(r: Rollout): string {
  const num = r.kind === "demo" ? r.label.replace(/^Demo\s*/, "") : r.kind === "attempt" ? "Attempt" : r.who;
  const who =
    r.origin === "agent" ? r.who.replace(/ agent$/, "") : r.origin === "oracle" ? "Oracle" : r.origin === "imported" ? "Human (imported)" : r.origin === "teleop" ? "Human (teleop)" : r.origin === "attempt" ? r.who.replace(/ agent$/, "") : "";
  const where = r.imported ? shortSource(r.source) : r.condition ? conditionWords(r.condition) : "";
  return [num, who, where, mark(r)].filter(Boolean).join(" · ");
}
const shortSource = (s: string | null) => {
  const x = (s ?? "imported dataset").replace(/^(hf|upload):/, "");
  return x.length > 20 ? `${x.slice(0, 19)}…` : x;
};

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
  const datasets = useDatasets(pid);
  const augmented = useMemo(() => augmentedByEpisode(datasets.data ?? []), [datasets.data]);
  const jobs = useJobs();
  const liveJob: Job | undefined = useMemo(
    () => Object.values(jobs).find((j) => j.project_id === pid && isJobActive(j) && LIVE_KINDS.test(j.kind)),
    [jobs, pid],
  );
  const [mode, setMode] = useState<"auto" | "live" | "replay" | "gpu">("auto");
  const showLive = mode === "live" || (mode === "auto" && !!liveJob);
  const showGpu = mode === "gpu";
  const replay = !showLive && !showGpu;
  const [camera, setCamera] = useState<Camera>("presentation");
  const [compare, setCompare] = useState(false);

  const filtered = useMemo(() => {
    let xs = rollouts.data ?? [];
    if (sel.evaluationId) xs = xs.filter((r) => r.evaluationId === sel.evaluationId);
    if (sel.filter === "demo") xs = xs.filter((r) => r.kind === "demo");
    else if (sel.filter === "failed") xs = xs.filter((r) => r.success === false);
    else if (sel.filter !== "all") xs = xs.filter((r) => r.origin === sel.filter);
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
    const by = Object.fromEntries(ORIGIN_ORDER.map((o) => [o, xs.filter((r) => r.origin === o).length])) as Record<RolloutOrigin, number>;
    // name the agent chip after its planner when there is only one ("Claude Code agent")
    const agentNames = [...new Set(xs.filter((r) => r.origin === "agent").map((r) => r.who))];
    return { all: xs.length, by, agentLabel: agentNames.length === 1 ? agentNames[0] : "Agent" };
  }, [rollouts.data]);

  return (
    <section className="relative flex h-full min-h-0 flex-col bg-bench text-ink" aria-label="Robot simulation viewer">
      {/* toolbar */}
      <div className="relative flex items-center gap-3 border-b border-line bg-bench px-4 py-2.5">
        <div className="flex shrink-0 rounded-full border border-line-strong bg-plate p-0.5">
          <ToolbarTab
            active={showLive}
            onClick={() => setMode("live")}
            disabled={!liveJob && mode !== "live"}
            title={liveJob ? "Watch the simulation live" : "Turns on when a job starts"}
          >
            <Radio className={cn("h-3.5 w-3.5", liveJob && "text-fault")} /> Live
          </ToolbarTab>
          <ToolbarTab active={replay} onClick={() => setMode("replay")}>
            Replays
          </ToolbarTab>
          <ToolbarTab active={showGpu} onClick={() => setMode("gpu")} title="Trained policies running live on a cluster GPU">
            <Cpu className="h-3.5 w-3.5" /> GPU
          </ToolbarTab>
        </div>
        {replay && (
          <div className="flex min-w-0 flex-1 items-center gap-4 overflow-x-auto px-1 scroll-thin">
            {(
              [
                ["all", "All", counts.all] as const,
                ...ORIGIN_ORDER.filter((o) => counts.by[o] > 0 || sel.filter === o).map((o) => [o, o === "agent" ? counts.agentLabel : ORIGIN_LABEL[o], counts.by[o]] as const),
                ...(sel.filter === "failed" || sel.filter === "demo" ? [[sel.filter, sel.filter === "failed" ? "Failures" : "Demos", filtered.length] as const] : []),
              ] as const
            ).map(([k, label, n]) => (
              <button
                key={k}
                type="button"
                onClick={() => sel.setFilter(k)}
                className={cn(
                  "inline-flex h-7 shrink-0 items-center gap-1.5 whitespace-nowrap border-b font-mono text-[10.5px] uppercase tracking-[0.14em] transition-colors",
                  sel.filter === k ? "border-ink text-ink" : "border-transparent text-ink-3 hover:text-ink",
                )}
              >
                {label}
                <span className="num text-ink-4">{String(n).padStart(2, "0")}</span>
              </button>
            ))}
            {sel.evaluationId && (
              <button type="button" onClick={() => sel.setEvaluationId(null)} className="inline-flex h-6 shrink-0 items-center gap-1 whitespace-nowrap rounded-full border border-servo/60 px-3 font-mono text-[10px] uppercase tracking-[0.12em] text-servo">
                one evaluation · clear ×
              </button>
            )}
          </div>
        )}
        {replay && current && (
          <div className="ml-auto flex shrink-0 items-center gap-1.5">
            <select
              value={cam?.id ?? "presentation"}
              onChange={(e) => setCamera(e.target.value as Camera)}
              disabled={cams.length < 2}
              title={cam?.detail || cam?.label}
              className="h-7 max-w-[180px] rounded-full border border-line-strong bg-plate px-3 font-mono text-[10.5px] uppercase tracking-[0.1em] text-ink outline-none focus:border-servo disabled:opacity-70"
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
                "sk-pill inline-flex h-7 items-center gap-1.5 px-3 text-xs font-medium disabled:opacity-35",
                compare ? "sk-pill-primary" : "sk-pill-secondary",
              )}
            >
              <Columns2 className="h-3.5 w-3.5" /> Compare
            </button>
          </div>
        )}
      </div>

      <Hud aug={current ? augmented.get(current.id) : undefined} goal={goal} asked={asked} activity={activity} liveJob={showLive ? liveJob : undefined} rollout={!replay || compare ? null : current} />

      {/* stage */}
      <div className="relative min-h-0 flex-1 overflow-hidden">
        <DashedOrbit r={200} className="pointer-events-none absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 opacity-70" />
        {showGpu ? (
          <GpuStage />
        ) : showLive ? (
          <LiveStage pid={pid} job={liveJob} />
        ) : rollouts.isLoading ? (
          <div className="relative flex h-full items-center justify-center">
            <CircleDashed className="h-6 w-6 animate-spin text-ink-4" />
          </div>
        ) : !current ? (
          <div className="relative flex h-full flex-col items-center justify-center gap-2 px-8 text-center">
            <SketchScene width={300} label="SCENE" value="NO ROLLOUTS" className="mb-2" />
            <p className="kn-display text-[28px] text-ink">{(rollouts.data ?? []).length ? "No matches" : "No videos yet"}</p>
            <p className="max-w-sm text-xs leading-5 text-ink-3">Ask the agent to collect demos.</p>
          </div>
        ) : compare && counterpart ? (
          <CompareStage a={current} b={counterpart} camera={camera} />
        ) : (
          <SingleStage r={current} camera={camera} />
        )}
      </div>

      {/* selector */}
      {replay && filtered.length > 0 && (
        <div className="relative border-t border-line bg-bench">
          <p className="flex items-center gap-1.5 px-4 pt-2 font-mono text-[10px] uppercase tracking-[0.1em] text-ink-3">
            Newest first · ✓ sim-verified · ✗ failed
            <InfoTip>✓ succeeded (checked by the simulator) · ✗ failed · not judged: human demos have no simulator verdict</InfoTip>
          </p>
          <div className="flex gap-1.5 overflow-x-auto px-4 pb-2.5 pt-1.5 scroll-thin">
            {filtered.slice(0, 80).map((r) => (
              <button
                key={r.id}
                type="button"
                onClick={() => sel.setSelectedId(r.id)}
                className={cn(
                  "flex shrink-0 items-center gap-2 rounded-sm border px-2.5 py-1.5 text-left text-xs transition-colors",
                  r.id === current?.id ? "border-ink bg-plate text-ink" : "border-line bg-plate text-ink-2 hover:border-line-strong hover:text-ink",
                )}
              >
                <span className={cn("h-1.5 w-1.5 shrink-0 rounded-full", r.success === null ? "bg-ink-4" : r.success ? "bg-pass" : "bg-fault")} />
                <span className="whitespace-nowrap">{chipText(r)}</span>
                {augmented.get(r.id) && (
                  <span className="whitespace-nowrap rounded-sm border border-line px-1.5 font-mono text-[10px] uppercase tracking-[0.08em] text-ink-3" title={augmented.get(r.id)!.tip}>
                    +{augmented.get(r.id)!.n} augmented
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
      className={cn("inline-flex h-7 items-center gap-1.5 rounded-full px-3 text-xs font-medium transition-colors disabled:opacity-35", active ? "bg-ink text-bench" : "text-ink-3 hover:text-ink")}
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
    <div className="relative flex h-full items-center justify-center p-6 pb-14">
      {failed || !job ? (
        <div className="kn-label text-center">{job ? "Live stream unavailable" : "Nothing running"}</div>
      ) : (
        <Frame>
          <img src={src} alt="Live MuJoCo simulation" onError={() => setFailed(true)} className="block h-full w-auto max-w-full object-contain" />
        </Frame>
      )}
      {job && (
        <div className="absolute inset-x-6 bottom-5">
          <div className="mb-1.5 flex justify-between font-mono text-[10.5px] uppercase tracking-[0.12em] text-ink-2">
            <span className="truncate">{job.message || job.kind.replace(/_/g, " ")}</span>
            <span className="num text-ink">{Math.round(job.progress * 100)}%</span>
          </div>
          <div className="h-px overflow-hidden bg-[var(--line-strong)]">
            <div className="h-full bg-servo transition-[width] duration-500" style={{ width: `${job.progress * 100}%` }} />
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
      <div className="relative flex min-h-0 flex-1 items-center justify-center p-6 pb-3">
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
      <div className="relative grid min-h-0 flex-1 grid-cols-2 gap-6 p-6 pb-3">
        {(
          [
            [ok, vOk, rOk, "Succeeded", "bg-pass"],
            [bad, vBad, rBad, "Failed", "bg-fault"],
          ] as const
        ).map(([r, v, ref, label, color]) => (
          <div key={r.id} className="flex min-h-0 flex-col gap-2">
            <div className="flex items-center gap-2 font-mono text-[10.5px] uppercase tracking-[0.14em]">
              <span className="h-px w-6 bg-ink" />
              <span className={cn("h-1.5 w-1.5 rounded-full", color)} />
              <span className="text-ink">{label}</span>
              <span className="truncate normal-case tracking-normal text-ink-3">
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
  if (loading) return <CircleDashed className="h-6 w-6 animate-spin text-ink-4" />;
  if (!url || failed === url)
    return (
      <div className="flex flex-col items-center gap-2 text-ink-3">
        <VideoOff className="h-6 w-6" strokeWidth={1.25} />
        <span className="kn-label">{url ? "No video for this camera" : "No video recorded"}</span>
      </div>
    );
  return (
    <Frame>
      <video key={url} ref={videoRef} src={url} autoPlay muted loop playsInline onError={() => setFailed(url)} className="block h-full w-auto max-w-full bg-black object-contain" />
    </Frame>
  );
}

/** The drawing frame around a camera image: a hairline border with crop marks at the corners. */
function Frame({ children }: { children: React.ReactNode }) {
  return (
    <div className="relative h-full max-w-full border border-line-strong bg-plate p-1">
      {(["-left-2 -top-2 border-l border-t", "-right-2 -top-2 border-r border-t", "-bottom-2 -left-2 border-b border-l", "-bottom-2 -right-2 border-b border-r"] as const).map((c) => (
        <span key={c} aria-hidden className={cn("pointer-events-none absolute h-3 w-3 border-ink", c)} />
      ))}
      {children}
    </div>
  );
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
    <div className="relative flex items-center gap-4 px-6 pb-3">
      <button type="button" onClick={toggle} disabled={!url} className="sk-pill sk-pill-primary flex h-8 w-10 items-center justify-center disabled:opacity-30" aria-label={playing ? "Pause" : "Play"}>
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
        className="h-1 flex-1 cursor-pointer accent-[rgb(var(--servo))]"
        aria-label="Timeline"
      />
      <span className="num w-24 text-right font-mono text-[10.5px] uppercase tracking-[0.1em] text-ink-2">
        T {fmt(t)} <span className="text-ink-4">/ {fmt(dur)}</span>
      </span>
    </div>
  );
}

/** Heads-up band above the stage: the task and what is happening now, plus what is on screen. */
function Hud({
  aug,
  goal,
  asked,
  activity,
  liveJob,
  rollout,
}: {
  aug?: { n: number; tip: string };
  goal: string;
  asked?: string | null;
  activity: { label: string; busy: boolean } | null;
  liveJob?: Job;
  rollout: Rollout | null;
}) {
  return (
    <div className="relative flex items-start justify-between gap-6 px-6 pt-4">
      <div className="min-w-0">
        <div className="kn-label kn-label-rule">
          <span className="truncate">
            <span className="text-ink">{asked ? "Simulated task" : "Scene"}</span>
          </span>
        </div>
        <div className="kn-display mt-2 truncate text-[24px] text-ink">{goal}</div>
        {asked && (
          <div className="truncate text-xs text-ink-3" title={`The simulator only has the cube-and-bowl scene. You asked: ${asked}`}>
            Asked: “{asked}” · sim has cube and bowl only
          </div>
        )}
        <div className="mt-1.5 flex items-center gap-2 font-mono text-[10.5px] uppercase tracking-[0.12em] text-ink-2">
          {liveJob ? (
            <>
              <span className="h-1.5 w-1.5 rounded-full bg-fault live-dot" />
              <span title="Rendered for people; not what the policy sees">Live · studio view</span>
            </>
          ) : activity?.busy ? (
            <>
              <CircleDashed className="h-3 w-3 animate-spin text-servo" /> <span className="truncate">{activity.label}</span>
            </>
          ) : (
            <>
              <span className="h-1.5 w-1.5 rounded-full bg-ink-4" /> <span className="truncate">{activity?.label ?? "Idle"}</span>
            </>
          )}
        </div>
      </div>
      {rollout && (
        <p
          className="min-w-0 max-w-[50%] shrink truncate pt-0.5 font-mono text-[11px] uppercase tracking-[0.08em] text-ink-2"
          title={[
            rollout.origin === "oracle" ? "Scripted oracle (uses simulator state)" : rollout.who,
            rollout.imported ? [rollout.source, rollout.task].filter(Boolean).join(" · ") : null,
            rollout.seed !== null && rollout.seed >= 0 ? `seed ${rollout.seed}` : null,
            rollout.frames ? `${rollout.frames} frames` : null,
            aug?.tip ?? null,
          ].filter(Boolean).join("\n")}
        >
          {[
            rollout.origin === "oracle" ? "Oracle" : rollout.who,
            rollout.imported ? (rollout.source ?? "imported") : rollout.condition ? conditionWords(rollout.condition) : null,
            rollout.durationS ? `${rollout.durationS.toFixed(1)} s` : null,
            aug ? `+${aug.n} aug` : null,
          ].filter(Boolean).join(" · ")}
          {" · "}
          <span className={rollout.success === null ? "text-ink-3" : rollout.success ? "text-pass" : "text-fault"}>
            {rollout.success === null ? "not judged" : rollout.success ? "✓ success" : `✗ ${rollout.failureMode ? rollout.failureMode.replace(/_/g, " ") : "failed"}`}
          </span>
        </p>
      )}
    </div>
  );
}

