import { useEffect, useState, type ReactNode } from "react";
import { Link, useNavigate, useParams, useSearchParams } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ArrowLeft, ArrowUpRight, Check, CircleDashed, Cpu, Film, Play, Radio, Shuffle, Square, VideoOff } from "lucide-react";
import { api, resolveMediaUrl, type ApiError } from "../api/client";
import type { PhysicalSkill } from "../api/types";
import { Badge, Button, EmptyState, ErrorState, Skeleton } from "../components/ui";
import { cn } from "../lib/format";
import { LiveFrame } from "../chat/LivePolicy";
import { GpuControl, Thumb, useGpu } from "./SkillsPage";
import { newSeed, runAction, scoreBadge, simWhere, skillVideos } from "./skillView";

/** /studio/skills/:sid, full screen (#308): a sidebar of every physical skill, and the selected one's recorded runs.
 *  Nothing touches the GPU until "Live" is clicked: then it streams if a GPU server runs the policy, or offers one
 *  explicit "Request a GPU". */

const label = "font-mono text-[10.5px] uppercase tracking-[0.14em]";
type View = "both" | "front" | "wrist";
const VIEWS: [View, string][] = [
  ["both", "Front + wrist"],
  ["front", "Front"],
  ["wrist", "Wrist"],
];

function Segmented<T extends string>({ value, options, onChange, name }: { value: T; options: [T, string, ReactNode?][]; onChange: (v: T) => void; name: string }) {
  return (
    <div role="radiogroup" aria-label={name} className="inline-flex shrink-0 rounded-full border border-line-strong p-0.5">
      {options.map(([v, text, icon]) => (
        <button
          key={v}
          type="button"
          role="radio"
          aria-checked={value === v}
          onClick={() => onChange(v)}
          className={cn("inline-flex h-7 items-center gap-1.5 rounded-full px-3 text-xs", value === v ? "bg-ink text-bench" : "text-ink-2 hover:text-ink")}
        >
          {icon}
          {text}
        </button>
      ))}
    </div>
  );
}

/** The skill's recorded videos: the selected one plays large (looping, muted), the rest are thumbnails to pick, in a
 *  column beside it (not below), so the video keeps the full height. */
function Replays({ s }: { s: PhysicalSkill }) {
  const videos = skillVideos(s);
  const [i, setI] = useState(0);
  const cur = videos[i];
  if (!cur)
    return (
      <div className="flex h-full flex-col items-center justify-center gap-2 text-xs text-ink-2">
        <VideoOff className="h-6 w-6 text-ink-3" />
        <span>No recorded runs of this skill yet.</span>
        <span className="text-ink-3">Click Live to run it on the GPU.</span>
      </div>
    );
  return (
    <div className="flex h-full min-h-0 min-w-0 flex-col gap-2 md:flex-row">
      <div className="relative min-h-0 min-w-0 flex-1">
        <video
          key={cur.url}
          src={resolveMediaUrl(cur.url) ?? undefined}
          autoPlay
          loop
          muted
          playsInline
          controls
          className="absolute inset-0 block h-full w-full bg-rail object-contain"
        />
      </div>
      {videos.length > 1 && (
        <ul className="scroll-thin flex shrink-0 gap-2 overflow-x-auto pb-1 md:w-32 md:flex-col md:overflow-y-auto md:overflow-x-hidden md:pb-0">
          {videos.map((v, k) => (
            <li key={v.url} className="w-28 shrink-0 md:w-full">
              <button
                type="button"
                onClick={() => setI(k)}
                aria-current={k === i || undefined}
                className={cn("block w-full overflow-hidden rounded-sm border text-left", k === i ? "border-ink" : "border-line hover:border-ink-3")}
              >
                <Thumb url={v.url} className="aspect-[4/3] w-full" />
                <span className="block truncate px-1.5 py-1 text-[10.5px] text-ink-3">
                  {v.label}
                  {v.success != null && <span className={v.success ? "text-pass" : "text-fault"}> · {v.success ? "success" : "failed"}</span>}
                </span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

/** Mounted only after "Live" is clicked: streams when the GPU server runs this policy, else one explicit GPU request.
 *  The camera view and the layout seed live in the header (Stage), so the stream gets the whole area. */
function Live({ s, view, seed, setSeed }: { s: PhysicalSkill; view: View; seed: number | null; setSeed: (f: (cur: number | null) => number) => void }) {
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
  const ready = a.kind === "play";
  useEffect(() => {
    if (ready) setSeed((cur) => cur ?? newSeed());
  }, [ready, setSeed]);
  // a live run is saved when its stream ends: read the list again so it shows among the recorded runs
  useEffect(() => () => void window.setTimeout(() => void qc.invalidateQueries({ queryKey: ["skills"] }), 4000), [qc]);
  const err = (start.error ?? stop.error) as ApiError | null;
  const task = s.run.task ?? "";

  if (a.kind === "checking")
    return (
      <p className="flex h-full items-center justify-center gap-2 text-xs text-ink-3">
        <CircleDashed className="h-3.5 w-3.5 animate-spin" /> Checking the GPU server…
      </p>
    );
  if (a.kind === "unavailable") return <p className="flex h-full items-center justify-center text-center text-[13px] leading-6 text-ink-2">{a.note}</p>;

  if (a.kind === "request" || a.kind === "starting") {
    const order = Object.keys(st?.steps ?? {});
    const at = st?.step ? order.indexOf(st.step) : start.isPending ? 0 : -1;
    const busy = a.kind === "starting" || start.isPending;
    return (
      <div className="flex h-full flex-col items-center justify-center gap-4 px-6">
        {busy ? <CircleDashed className="h-6 w-6 animate-spin text-ink-3" /> : <Cpu className="h-6 w-6 text-ink-3" />}
        <p className="max-w-md text-center text-[13px] leading-6 text-ink-2">
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
        <div className="flex flex-wrap items-center justify-center gap-2">
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
        {!busy && <p className="max-w-md text-center text-[11.5px] leading-5 text-ink-3">Attaches to a GPU server already running on Futurama, or asks Slurm for one. Nothing starts until you click.</p>}
        {err && err.message !== st?.reason && <p className="text-xs text-fault">{err.message}</p>}
      </div>
    );
  }

  return (
    <div className="relative h-full min-h-0 min-w-0 bg-rail">
      {seed != null && (
        <div className="absolute inset-0">
          <LiveFrame fill key={`${seed}-${view}`} src={api.livePolicySimUrl(task, seed, view)} label={`${s.name} running live on the GPU`} />
        </div>
      )}
    </div>
  );
}

const BackToSkills = () => (
  <Link to="/studio/skills" className="inline-flex h-8 shrink-0 items-center gap-1.5 rounded-full border border-line-strong px-3 text-[13px] text-ink-2 hover:text-ink">
    <ArrowLeft className="h-3.5 w-3.5" /> Back
  </Link>
);

/** One skill: replays by default; "Live" is the only thing that connects to the GPU. Everything but the video sits in
 *  one header row, so the video takes the rest of the screen. */
function Stage({ s }: { s: PhysicalSkill }) {
  // ?seed= is an agent's View deployment link (cortex.deploy_physical_skill, #310): open live, on that run's layout
  const [params] = useSearchParams();
  const linked = Number(params.get("seed")) || null;
  const [mode, setMode] = useState<"replay" | "live">(linked ? "live" : "replay");
  const [view, setView] = useState<View>("both");
  const [seed, setSeed] = useState<number | null>(linked);
  const gpu = useGpu();
  const b = scoreBadge(s.score);
  const playing = mode === "live" && runAction(s, gpu.data).kind === "play";
  const layouts = s.layouts ?? gpu.data?.sim?.[s.run.task ?? ""]?.layouts;
  const meta = [s.robot, s.policy, "trained in simulation", s.steps ? `${s.steps.toLocaleString()} steps` : null].filter(Boolean).join(" · ");
  return (
    <>
      <header className="flex flex-wrap items-center gap-x-3 gap-y-2 border-b border-line px-3 py-2">
        <BackToSkills />
        <div className="flex min-w-[8rem] flex-1 items-center gap-2 overflow-hidden">
          <h1 className="kn-display min-w-0 truncate text-[18px] leading-tight" title={`${s.name} · ${meta}`}>
            {s.name}
          </h1>
          <Badge className={cn("shrink-0", playing && "hidden 2xl:inline-flex")} tone={b.tone} title={s.score.label !== b.text ? `${s.score.label}. ${s.score.detail}` : s.score.detail}>
            {b.text}
          </Badge>
          {s.project_id && (
            <Link to={`/p/${s.project_id}`} className="hidden shrink-0 items-center gap-0.5 text-[11.5px] text-ink-2 hover:text-ink xl:inline-flex">
              Open its project <ArrowUpRight className="h-3 w-3" />
            </Link>
          )}
        </div>
        <Segmented
          name="Watch"
          value={mode}
          onChange={(m) => {
            setMode(m);
            if (m === "replay") setSeed(null); // back to Live later starts a fresh run
          }}
          options={[
            ["replay", "Watch replay", <Film key="f" className="h-3.5 w-3.5" />],
            ["live", "Live", <Radio key="r" className="h-3.5 w-3.5" />],
          ]}
        />
        {playing && (
          <>
            <Segmented name="Camera view" value={view} options={VIEWS} onChange={setView} />
            <Button
              variant="primary"
              size="sm"
              onClick={() => setSeed(newSeed())}
              title={`Closed loop in MuJoCo on the GPU; the simulator judges the end. Each run is ${simWhere({ ...s, layouts: layouts ?? null })}. Saved to past runs when it ends.`}
            >
              {seed == null ? <Play className="h-3.5 w-3.5" /> : <Shuffle className="h-3.5 w-3.5" />} {seed == null ? "Run live" : "Run on a new layout"}
            </Button>
          </>
        )}
        {/* Request a GPU (with its steps) when none is live; Stop & release GPU when one is */}
        <GpuControl compact />
      </header>
      <div className="min-h-0 min-w-0 flex-1 p-2">{mode === "replay" ? <Replays s={s} /> : <Live s={s} view={view} seed={seed} setSeed={setSeed} />}</div>
    </>
  );
}

export function SkillViewer() {
  const { sid } = useParams<{ sid: string }>();
  const nav = useNavigate();
  const qc = useQueryClient();
  const skills = useQuery({ queryKey: ["skills"], queryFn: api.skills, refetchInterval: 30_000 });
  const gpu = useGpu();
  const available = gpu.data?.available;
  useEffect(() => {
    // which skills run live depends on the GPU server: read the list again when it comes up or goes away
    if (available !== undefined) void qc.invalidateQueries({ queryKey: ["skills"] });
  }, [available, qc]);
  const list = skills.data ?? [];
  const s = list.find((x) => x.id === sid);

  return (
    <div className="flex h-full flex-col bg-bench text-ink md:flex-row">
      <aside className="flex max-h-[30vh] shrink-0 flex-col border-b border-line md:max-h-none md:w-56 md:border-b-0 md:border-r">
        <h2 className={cn("px-4 pb-2 pt-4 text-ink-3", label)}>Physical skills</h2>
        <ul className="scroll-thin min-h-0 flex-1 overflow-y-auto px-2 pb-4">
          {list.map((x) => (
            <li key={x.id}>
              <button
                type="button"
                onClick={() => nav(`/studio/skills/${encodeURIComponent(x.id)}`, { replace: true })}
                aria-current={x.id === sid || undefined}
                className={cn("flex w-full items-center gap-3 rounded px-2 py-2 text-left", x.id === sid ? "bg-rail text-ink" : "text-ink-2 hover:bg-rail/50 hover:text-ink")}
              >
                <Thumb url={x.replays[0] ?? x.thumbnail_url} className="h-10 w-14 shrink-0 rounded-sm border border-line" />
                <span className="min-w-0 flex-1">
                  <span className="block truncate text-[13px]">{x.name}</span>
                  <span className="block truncate text-[11px] text-ink-3">{[x.policy, scoreBadge(x.score).text].filter(Boolean).join(" · ")}</span>
                </span>
                {x.live && <span className="live-dot h-1.5 w-1.5 shrink-0 rounded-full bg-pass" title="On the GPU server now" />}
              </button>
            </li>
          ))}
        </ul>
      </aside>
      <main className="flex min-h-0 min-w-0 flex-1 flex-col">
        {skills.isLoading ? (
          <Skeleton className="m-5 flex-1" />
        ) : skills.error ? (
          <div className="space-y-4 p-5">
            <BackToSkills />
            <ErrorState error={skills.error} onRetry={() => skills.refetch()} />
          </div>
        ) : s ? (
          <Stage key={s.id} s={s} />
        ) : (
          <div className="space-y-4 p-5">
            <BackToSkills />
            <EmptyState title="This skill isn't listed anymore">Pick another skill on the left, or go back to the list.</EmptyState>
          </div>
        )}
      </main>
    </div>
  );
}
