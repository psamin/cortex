import { useCallback, useEffect, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { AlertTriangle, Check, Hand, Plus, Power, Square, Trash2 } from "lucide-react";
import { api } from "../api/client";
import { useCheckpoints } from "../api/hooks";
import type { CameraSpec, HardwareConfig, HardwareProcess, HardwareRun, HardwareStatus } from "../api/types";
import { useApprover } from "../components/refs";
import { Badge, Button, Dialog, ErrorState, Field, InlineError, Input, Panel, Select, Skeleton } from "../components/ui";
import { ResultCard } from "../chat/cards";
import { cn } from "../lib/format";

/** Real SO-101 arm: setup, calibration, arming, teleop recording and policy runs (local studio only). */
export function RobotPage() {
  const { pid = "" } = useParams();
  const status = useQuery({ queryKey: ["hardware"], queryFn: api.hardware, refetchInterval: 5000 });
  const run = useQuery({
    queryKey: ["hardware-run"],
    queryFn: api.hardwareRun,
    refetchInterval: (q) => (q.state.data && q.state.data.state !== "finished" ? 300 : 3000),
  });
  const hw = status.data;
  return (
    <div className="space-y-4 px-5 py-5">
      <div>
        <h1 className="kn-display text-[36px] leading-tight">Real SO-101 arm</h1>
        <p className="mt-0.5 text-sm text-ink-3">Every run needs your approval and a held deadman.</p>
      </div>
      {status.isLoading ? (
        <Skeleton className="h-40 w-full" />
      ) : status.error || !hw ? (
        <ErrorState error={status.error} onRetry={() => status.refetch()} />
      ) : (
        <div className="grid grid-cols-1 gap-4 min-[1200px]:grid-cols-2">
          <div className="space-y-4">
            <Checklist hw={hw} />
            {hw.enabled && <SetupForm hw={hw} />}
            {hw.enabled && <Calibration hw={hw} />}
          </div>
          <div className="space-y-4">
            {run.data && <RunPanel run={run.data} pid={pid} />}
            {hw.enabled && <Arming hw={hw} />}
            {hw.enabled && <RunStarters pid={pid} hw={hw} busy={!!run.data && run.data.state !== "finished"} />}
          </div>
        </div>
      )}
    </div>
  );
}

// ------------------------------------------------------------------ readiness
/** A short setup guide: done steps are ticked, only the next step says what to do. Nothing in red. */
function Checklist({ hw }: { hw: HardwareStatus }) {
  const portOk = !!hw.config.follower_port && !hw.missing.some((m) => m.includes("port"));
  const steps = [
    { title: "Turn on real-arm mode", done: hw.enabled,
      hint: "On the computer the arm plugs into, add CORTEX_ENABLE_HARDWARE=1 to .env and restart Cortex." },
    { title: "Plug in the arm and choose its port", done: portOk, hint: "Pick the follower arm's USB port in Setup below." },
    { title: "Calibrate the arm", done: hw.calibration.follower.exists,
      hint: "Press Calibrate follower below; you move the joints by hand." },
    { title: "Arm it", done: !!hw.armed, hint: "Confirm the e-stop is in reach, below." },
  ];
  const current = steps.findIndex((st) => !st.done);
  const leaderReady = hw.teleop_missing.length === 0;
  return (
    <Panel title="Real arm setup" actions={hw.ready_for_motion ? <Badge tone="pass">ready</Badge> : <span className="text-xs text-ink-3">step {current + 1} of {steps.length}</span>}>
      {!hw.enabled && <p className="mb-3 text-sm text-ink-3">Simulator mode · no arm needed.</p>}
      {hw.driver_error && <InlineError error={hw.driver_error} fallback="The motor driver could not be loaded." className="mb-3" />}
      <ol className="space-y-2">
        {steps.map((st, i) => (
          <li key={st.title} className="flex items-start gap-2.5 text-sm">
            <span className={cn("mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full text-[11px]",
              st.done ? "bg-pass/15 text-pass" : i === current ? "border border-servo text-servo" : "border border-line text-ink-4")}>
              {st.done ? <Check className="h-3 w-3" /> : i + 1}
            </span>
            <div>
              <div className={st.done || i === current ? "text-ink" : "text-ink-4"}>{st.title}</div>
              {i === current && <div className="mt-0.5 text-xs text-ink-3">{st.hint}</div>}
            </div>
          </li>
        ))}
      </ol>
      {hw.ready_for_motion && (
        <p className="mt-3 text-xs text-ink-3">
          {leaderReady ? "Leader arm ready." : "Optional: calibrate a leader arm for teleop."} The arm moves only while you hold the deadman.
        </p>
      )}
    </Panel>
  );
}

// ------------------------------------------------------------------ setup
function SetupForm({ hw }: { hw: HardwareStatus }) {
  const qc = useQueryClient();
  const [cfg, setCfg] = useState<HardwareConfig>(hw.config);
  const save = useMutation({ mutationFn: () => api.saveHardwareConfig(cfg), onSuccess: (s) => qc.setQueryData(["hardware"], s) });
  const ports = [...hw.ports].sort((a, b) => Number(b.usb) - Number(a.usb));
  const set = <K extends keyof HardwareConfig>(k: K, v: HardwareConfig[K]) => setCfg((c) => ({ ...c, [k]: v }));
  const setCam = (i: number, patch: Partial<CameraSpec>) => set("cameras", cfg.cameras.map((c, j) => (j === i ? { ...c, ...patch } : c)));
  const portSelect = (k: "follower_port" | "leader_port", label: string) => (
    <Field label={label}>
      <Select value={cfg[k] ?? ""} onChange={(e) => set(k, e.target.value || null)}>
        <option value="">Not chosen</option>
        {ports.map((p) => (
          <option key={p.device} value={p.device}>
            {p.device}
            {p.usb ? " (USB)" : ""}
            {p.description && p.description !== "n/a" ? ` · ${p.description}` : ""}
          </option>
        ))}
      </Select>
    </Field>
  );
  return (
    <Panel title="Setup" meta="saving disarms the arm">
      <div className="grid grid-cols-2 gap-3">
        {portSelect("follower_port", "Follower port")}
        <Field label="Follower id">
          <Input value={cfg.follower_id} onChange={(e) => set("follower_id", e.target.value)} />
        </Field>
        {portSelect("leader_port", "Leader port (teleop)")}
        <Field label="Leader id">
          <Input value={cfg.leader_id} onChange={(e) => set("leader_id", e.target.value)} />
        </Field>
      </div>
      <div className="mt-3">
        <div className="mb-1 flex items-center justify-between text-xs font-medium text-ink-2">
          Cameras
          <Button size="sm" variant="ghost" onClick={() => set("cameras", [...cfg.cameras, { name: cfg.cameras.length ? "wrist" : "front", index_or_path: cfg.cameras.length, width: 640, height: 480, fps: 30 }])}>
            <Plus className="h-3.5 w-3.5" /> Add camera
          </Button>
        </div>
        {cfg.cameras.length === 0 && <p className="text-xs text-ink-3">No cameras. Image policies need matching camera names.</p>}
        {cfg.cameras.map((c, i) => (
          <div key={i} className="mb-1.5 grid grid-cols-[1fr_1fr_64px_64px_52px_28px] items-center gap-1.5">
            <Input value={c.name} onChange={(e) => setCam(i, { name: e.target.value })} aria-label="Camera name" placeholder="front" />
            <Input
              value={String(c.index_or_path)}
              onChange={(e) => setCam(i, { index_or_path: /^\d+$/.test(e.target.value) ? Number(e.target.value) : e.target.value })}
              aria-label="Camera index or path"
            />
            <Input type="number" value={c.width} onChange={(e) => setCam(i, { width: Number(e.target.value) })} aria-label="Width" />
            <Input type="number" value={c.height} onChange={(e) => setCam(i, { height: Number(e.target.value) })} aria-label="Height" />
            <Input type="number" value={c.fps} onChange={(e) => setCam(i, { fps: Number(e.target.value) })} aria-label="FPS" />
            <button type="button" onClick={() => set("cameras", cfg.cameras.filter((_, j) => j !== i))} className="rounded p-1 text-ink-3 hover:bg-rail hover:text-ink" aria-label="Remove camera">
              <Trash2 className="h-3.5 w-3.5" />
            </button>
          </div>
        ))}
      </div>
      <div className="mt-3 flex items-end gap-3">
        <div className="w-40">
          <Field label="Max step per tick (°)" hint="at most 3°">
            <Input type="number" step={0.5} min={0.5} max={3} value={cfg.max_step_deg} onChange={(e) => set("max_step_deg", Math.min(3, Math.max(0.1, Number(e.target.value) || 0.1)))} />
          </Field>
        </div>
        <Button variant="primary" loading={save.isPending} onClick={() => save.mutate()}>
          Save setup
        </Button>
        {save.isSuccess && <span className="pb-2 text-xs text-pass">Saved. The arm is disarmed.</span>}
      </div>
      {save.error && <InlineError error={save.error} className="mt-2" />}
    </Panel>
  );
}

// ------------------------------------------------------------------ calibration
function Calibration({ hw }: { hw: HardwareStatus }) {
  const qc = useQueryClient();
  const proc = useQuery({
    queryKey: ["hardware-process"],
    queryFn: api.hardwareProcess,
    refetchInterval: (q) => (q.state.data && ["running", "waiting_for_operator"].includes(q.state.data.state) ? 500 : false),
  });
  const refresh = () => qc.invalidateQueries({ queryKey: ["hardware-process"] });
  const start = useMutation({ mutationFn: (d: "follower" | "leader") => api.calibrate(d), onSuccess: refresh });
  const input = useMutation({ mutationFn: (t: "" | "c") => api.hardwareProcessInput(t), onSuccess: refresh });
  const stop = useMutation({ mutationFn: api.hardwareProcessStop, onSuccess: refresh });
  const p = proc.data;
  const live = !!p && ["running", "waiting_for_operator"].includes(p.state);
  return (
    <Panel title="Calibration" meta="lerobot-calibrate">
      <p className="text-sm text-ink-2">Torque off: move each joint through its full range when prompted.</p>
      <div className="mt-3 flex flex-wrap gap-2">
        {(["follower", "leader"] as const).map((d) => (
          <Button key={d} size="sm" disabled={live} loading={start.isPending && start.variables === d} onClick={() => start.mutate(d)} title={live ? "A calibration is already running" : undefined}>
            Calibrate {d} {hw.calibration[d].exists && <Badge tone="pass">calibrated</Badge>}
          </Button>
        ))}
        {live && (
          <Button size="sm" variant="danger" loading={stop.isPending} onClick={() => stop.mutate()}>
            <Square className="h-3 w-3 fill-current" /> Stop
          </Button>
        )}
      </div>
      {start.error && <InlineError error={start.error} className="mt-2" />}
      {p && <ProcessView p={p} onEnter={() => input.mutate("")} onRecalibrate={() => input.mutate("c")} busy={input.isPending} />}
      {(input.error || stop.error) && <InlineError error={input.error ?? stop.error} className="mt-2" />}
    </Panel>
  );
}

function ProcessView({ p, onEnter, onRecalibrate, busy }: { p: HardwareProcess; onEnter: () => void; onRecalibrate: () => void; busy: boolean }) {
  const joints = Object.entries(p.joint_ranges ?? {});
  const lo = Math.min(...joints.map(([, r]) => r.min), 0);
  const hi = Math.max(...joints.map(([, r]) => r.max), 4095);
  return (
    <div className="mt-3 space-y-3">
      <div className="flex items-center gap-2 text-xs text-ink-3">
        <Badge tone={p.state === "failed" ? "fault" : p.state === "finished" ? "pass" : "servo"}>{p.state.replace(/_/g, " ")}</Badge>
        {p.kind}
      </div>
      {p.prompt && p.state === "waiting_for_operator" && (
        <div className="rounded bg-servo/10 p-3 border border-servo/40">
          <p className="text-sm font-medium text-ink">{p.prompt}</p>
          <div className="mt-2 flex gap-2">
            <Button size="sm" variant="primary" loading={busy} onClick={onEnter}>
              Continue (Enter)
            </Button>
            {/'c'|\bc\b/i.test(p.prompt) && (
              <Button size="sm" onClick={onRecalibrate} disabled={busy}>
                Recalibrate
              </Button>
            )}
          </div>
        </div>
      )}
      {joints.length > 0 && (
        <ul className="space-y-1.5">
          {joints.map(([name, r]) => (
            <li key={name} className="grid grid-cols-[120px_1fr_120px] items-center gap-2 text-xs">
              <span className="truncate text-ink-2">{name}</span>
              <span className="relative h-2 rounded-full bg-rail">
                <span className="absolute inset-y-0 rounded-full bg-servo/35" style={{ left: `${((r.min - lo) / (hi - lo)) * 100}%`, width: `${((r.max - r.min) / (hi - lo)) * 100}%` }} />
                <span className="absolute -top-0.5 h-3 w-0.5 rounded bg-servo" style={{ left: `${((r.pos - lo) / (hi - lo)) * 100}%` }} />
              </span>
              <span className="num text-right font-mono text-ink-3">
                {r.min} · {r.pos} · {r.max}
              </span>
            </li>
          ))}
        </ul>
      )}
      {p.error && <InlineError error={p.error} fallback="Calibration failed." detail={p.lines.join("\n")} />}
    </div>
  );
}

// ------------------------------------------------------------------ arming
function Arming({ hw }: { hw: HardwareStatus }) {
  const qc = useQueryClient();
  const [name, setName] = useApprover();
  const [estop, setEstop] = useState(false);
  const done = (s: HardwareStatus) => qc.setQueryData(["hardware"], s);
  const arm = useMutation({ mutationFn: () => api.arm(name.trim()), onSuccess: done });
  const disarm = useMutation({ mutationFn: api.disarm, onSuccess: done });
  return (
    <Panel title="Arming" meta="expires after 30 minutes">
      {hw.armed ? (
        <div className="flex items-center gap-3">
          <Power className="h-4 w-4 text-pass" />
          <p className="flex-1 text-sm text-ink">
            Armed by <span className="font-medium">{hw.armed.operator}</span> until {new Date(hw.armed.expires_at).toLocaleTimeString()}
          </p>
          <Button size="sm" variant="danger" loading={disarm.isPending} onClick={() => disarm.mutate()}>
            Disarm
          </Button>
        </div>
      ) : (
        <div className="space-y-2">
          <Input value={name} onChange={(e) => setName(e.target.value)} placeholder="Operator name" aria-label="Operator name" />
          <label className="flex items-start gap-2 text-sm text-ink-2">
            <input type="checkbox" checked={estop} onChange={(e) => setEstop(e.target.checked)} className="mt-1 accent-[rgb(var(--servo))]" />
            The independent physical e-stop is in reach and was tested
          </label>
          <Button
            variant="primary"
            disabled={!estop || !name.trim()}
            title={!name.trim() ? "Type the operator's name" : !estop ? "Confirm the e-stop first" : "Arm for 30 minutes"}
            loading={arm.isPending}
            onClick={() => arm.mutate()}
          >
            Arm
          </Button>
        </div>
      )}
      {(arm.error || disarm.error) && <InlineError error={arm.error ?? disarm.error} className="mt-2" />}
    </Panel>
  );
}

// ------------------------------------------------------------------ runs
function RunStarters({ pid, hw, busy }: { pid: string; hw: HardwareStatus; busy: boolean }) {
  const qc = useQueryClient();
  const cks = useCheckpoints(pid);
  const [teleop, setTeleop] = useState({ episodes: 5, episode_s: 30, task: "" });
  const [policy, setPolicy] = useState({ checkpoint_id: "", episodes: 1, episode_s: 20 });
  const [confirm, setConfirm] = useState<"teleop" | "policy" | null>(null);
  const [atArm, setAtArm] = useState(false);
  const started = () => {
    setConfirm(null);
    qc.invalidateQueries({ queryKey: ["hardware-run"] });
  };
  const startTeleop = useMutation({ mutationFn: () => api.startTeleop(pid, teleop.episodes, teleop.episode_s, teleop.task.trim()), onSuccess: started });
  const startPolicy = useMutation({ mutationFn: () => api.startPolicyRun(pid, policy.checkpoint_id, policy.episodes, policy.episode_s), onSuccess: started });
  const ckId = policy.checkpoint_id || cks.data?.[0]?.id || "";
  const mutation = confirm === "teleop" ? startTeleop : startPolicy;
  const num = (v: string, lo: number, hi: number) => Math.min(hi, Math.max(lo, Number(v) || lo));
  return (
    <>
      <Panel title="Teleop recording" meta="saved as human demos">
        <div className="grid grid-cols-[90px_110px_1fr] gap-2">
          <Field label="Episodes">
            <Input type="number" value={teleop.episodes} onChange={(e) => setTeleop({ ...teleop, episodes: num(e.target.value, 1, 50) })} />
          </Field>
          <Field label="Seconds each">
            <Input type="number" value={teleop.episode_s} onChange={(e) => setTeleop({ ...teleop, episode_s: num(e.target.value, 3, 300) })} />
          </Field>
          <Field label="Task (optional)">
            <Input value={teleop.task} onChange={(e) => setTeleop({ ...teleop, task: e.target.value })} placeholder="the project's task" />
          </Field>
        </div>
        <Button className="mt-3" variant="primary" disabled={busy} title={busy ? "A run is already in progress" : undefined} onClick={() => (setAtArm(false), setConfirm("teleop"), setPolicy((p) => ({ ...p, checkpoint_id: ckId })))}>
          Start teleop
        </Button>
        {!confirm && startTeleop.error && <InlineError error={startTeleop.error} className="mt-2" />}
      </Panel>
      <Panel title="Run a policy">
        {cks.isLoading ? (
          <Skeleton className="h-8 w-full" />
        ) : !cks.data?.length ? (
          <p className="text-sm text-ink-2">
            No checkpoints yet. See{" "}
            <Link to={`/p/${pid}/improve?tab=training`} className="text-servo hover:underline">
              Improve › Training
            </Link>
            .
          </p>
        ) : (
          <>
            <div className="grid grid-cols-[1fr_90px_110px] gap-2">
              <Field label="Checkpoint">
                <Select value={ckId} onChange={(e) => setPolicy({ ...policy, checkpoint_id: e.target.value })}>
                  {cks.data.map((c) => (
                    <option key={c.id} value={c.id}>
                      {c.policy_type.toUpperCase()} · step {c.step} · {c.id}
                    </option>
                  ))}
                </Select>
              </Field>
              <Field label="Episodes">
                <Input type="number" value={policy.episodes} onChange={(e) => setPolicy({ ...policy, episodes: num(e.target.value, 1, 20) })} />
              </Field>
              <Field label="Seconds each">
                <Input type="number" value={policy.episode_s} onChange={(e) => setPolicy({ ...policy, episode_s: num(e.target.value, 3, 120) })} />
              </Field>
            </div>
            <p className="mt-2 flex items-start gap-1.5 text-xs text-caution">
              <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" /> Sim-trained policies may act differently on the real arm. Keep a hand on the deadman, e-stop in reach.
            </p>
            <Button className="mt-3" variant="primary" disabled={busy} title={busy ? "A run is already in progress" : undefined} onClick={() => (setAtArm(false), setPolicy((p) => ({ ...p, checkpoint_id: ckId })), setConfirm("policy"))}>
              Run on arm
            </Button>
          </>
        )}
        {!confirm && startPolicy.error && <InlineError error={startPolicy.error} className="mt-2" />}
      </Panel>
      <Dialog
        open={!!confirm}
        onClose={() => setConfirm(null)}
        title={confirm === "teleop" ? "Start teleop recording?" : "Run the policy on the real arm?"}
        footer={
          <>
            <Button variant="ghost" onClick={() => setConfirm(null)}>
              Cancel
            </Button>
            <Button variant="primary" disabled={!atArm} title={atArm ? undefined : "Confirm you are at the arm first"} loading={mutation.isPending} onClick={() => mutation.mutate()}>
              Start
            </Button>
          </>
        }
      >
        <p className="text-sm text-ink-2">
          {confirm === "teleop"
            ? `${teleop.episodes} episode${teleop.episodes === 1 ? "" : "s"} of ${teleop.episode_s} s. The arm only moves while you hold the deadman.`
            : `${policy.episodes} episode${policy.episodes === 1 ? "" : "s"} of ${policy.episode_s} s with ${ckId}. The arm only moves while you hold the deadman.`}
        </p>
        <label className="mt-3 flex items-start gap-2 text-sm text-ink">
          <input type="checkbox" checked={atArm} onChange={(e) => setAtArm(e.target.checked)} className="mt-1 accent-[rgb(var(--servo))]" />I am at the arm and the e-stop is in reach
        </label>
        {!hw.ready_for_motion && <p className="mt-2 text-xs text-ink-3">Setup incomplete; the backend will refuse and say why.</p>}
        {mutation.error && <InlineError error={mutation.error} className="mt-2" />}
      </Dialog>
    </>
  );
}

/** Hold-to-move: heartbeats every 100 ms while held; release, blur or tab switch stops them. */
function useDeadman(active: boolean) {
  const [held, setHeld] = useState(false);
  const timer = useRef<number | null>(null);
  const stop = useCallback(() => {
    if (timer.current !== null) window.clearInterval(timer.current);
    timer.current = null;
    setHeld(false);
  }, []);
  const start = useCallback(() => {
    if (!active || timer.current !== null) return;
    const beat = () => void api.hardwareRunOp("heartbeat").catch(() => {});
    beat();
    timer.current = window.setInterval(beat, 100);
    setHeld(true);
  }, [active]);
  useEffect(() => {
    const off = () => stop();
    window.addEventListener("blur", off);
    window.addEventListener("pointerup", off);
    document.addEventListener("visibilitychange", off);
    return () => {
      window.removeEventListener("blur", off);
      window.removeEventListener("pointerup", off);
      document.removeEventListener("visibilitychange", off);
      off();
    };
  }, [stop]);
  useEffect(() => {
    if (!active) stop();
  }, [active, stop]);
  return { held, start, stop };
}

function RunPanel({ run, pid }: { run: HardwareRun; pid: string }) {
  const qc = useQueryClient();
  const live = run.state !== "finished";
  const dead = useDeadman(run.state === "running");
  const op = useMutation({ mutationFn: (o: "end_episode" | "stop" | "release") => api.hardwareRunOp(o), onSuccess: () => qc.invalidateQueries({ queryKey: ["hardware-run"] }) });
  const joints = Object.entries(run.live?.joints ?? {});
  return (
    <Panel title={run.kind === "teleop" ? "Teleop run" : "Policy run"} actions={<Badge tone={run.state === "holding" ? "caution" : live ? "servo" : "neutral"}>{run.state}</Badge>}>
      {run.warnings.length > 0 && (
        <ul className="mb-3 space-y-1 rounded bg-caution/10 p-3 border border-caution/35">
          {run.warnings.map((w) => (
            <li key={w} className="flex items-start gap-1.5 text-sm text-ink">
              <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-caution" /> {w}
            </li>
          ))}
        </ul>
      )}
      {live && (
        <>
          <button
            type="button"
            onPointerDown={(e) => (e.currentTarget.setPointerCapture(e.pointerId), dead.start())}
            onPointerUp={dead.stop}
            onPointerCancel={dead.stop}
            onKeyDown={(e) => e.code === "Space" && !e.repeat && (e.preventDefault(), dead.start())}
            onKeyUp={(e) => e.code === "Space" && dead.stop()}
            onContextMenu={(e) => e.preventDefault()}
            disabled={run.state !== "running"}
            className={cn(
              "flex h-24 w-full select-none items-center justify-center gap-2 rounded text-lg font-semibold transition-colors disabled:opacity-50",
              dead.held ? "bg-pass text-white" : "bg-rail text-ink border border-line-strong",
            )}
          >
            <Hand className="h-6 w-6" /> {dead.held ? "Moving: release to stop" : "Hold to move (mouse or space bar)"}
          </button>
          <p className="mt-1.5 text-xs text-ink-3">Release stops it within one tick; it holds position.</p>
          <div className="mt-3 flex flex-wrap gap-2">
            <Button size="sm" onClick={() => op.mutate("end_episode")}>End episode</Button>
            <Button size="sm" variant="danger" onClick={() => op.mutate("stop")}>
              <Square className="h-3 w-3 fill-current" /> Stop
            </Button>
            {run.state === "holding" && (
              <Button size="sm" variant="danger" title="Support the arm first: it will drop when torque is off" onClick={() => op.mutate("release")}>
                Release torque
              </Button>
            )}
          </div>
          {run.state === "holding" && (
            <p className="mt-2 text-xs text-caution">
              Holding{run.holding?.reason ? `: ${run.holding.reason}` : ""}. Support the arm before releasing torque.
            </p>
          )}
          {op.error && <InlineError error={op.error} className="mt-2" />}
        </>
      )}
      {joints.length > 0 && (
        <dl className="mt-3 grid grid-cols-3 gap-x-4 gap-y-1 text-xs">
          {joints.map(([k, v]) => (
            <div key={k} className="flex justify-between gap-2">
              <dt className="truncate text-ink-3">{k}</dt>
              <dd className="num font-mono text-ink">{typeof v === "number" ? v.toFixed(1) : String(v)}</dd>
            </div>
          ))}
        </dl>
      )}
      {run.events.length > 0 && (
        <ul className="mt-3 space-y-0.5 text-xs text-ink-2">
          {run.events.slice(-8).map((e, i) => (
            <li key={i}>{String(e.type ?? "event").replace(/_/g, " ")}{e.episode !== undefined ? ` · episode ${String(e.episode)}` : ""}{typeof e.reason === "string" ? ` · ${e.reason}` : ""}</li>
          ))}
        </ul>
      )}
      {run.import_job_id && (
        <div className="mt-3">
          <p className="mb-1.5 text-xs text-ink-3">Importing as demos (human teleop, real SO-101):</p>
          <ResultCard pid={pid} refId={run.import_job_id} />
        </div>
      )}
    </Panel>
  );
}
