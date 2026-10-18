import { useEffect, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Check, ChevronDown, Copy, Pause, Play, ShieldAlert, Upload } from "lucide-react";
import { api } from "../api/client";
import { useAutopilot, useTraining } from "../api/hooks";
import { AUTOPILOT_STATES, isAgent, type AutopilotPolicy, type AutopilotRun, type AutopilotStep, type Project } from "../api/types";
import { ProvenanceBadge } from "../components/AgentTrace";
import { RefChip, useApprover } from "../components/refs";
import { Badge, Button, ErrorState, Input, Segmented, Skeleton, Toggle, type Tone, InlineError } from "../components/ui";
import { cn, evidenceValue, humanize, pct } from "../lib/format";

// ------------------------------------------------------------------ policy fields
/** Standing approvals: what autopilot may do without asking. Shared by onboarding and the hero. */
export function PolicyFields({ value, onChange }: { value: AutopilotPolicy; onChange: (p: AutopilotPolicy) => void }) {
  const set = <K extends keyof AutopilotPolicy>(k: K, v: AutopilotPolicy[K]) => onChange({ ...value, [k]: v });
  const numField = (k: keyof AutopilotPolicy, label: string, step = 1, scale = 1, suffix?: string) => (
    <label className="block space-y-1">
      <span className="block text-2xs font-medium text-ink-3">{label}</span>
      <div className="flex items-center gap-1.5">
        <Input
          type="number"
          step={step}
          value={Number(value[k]) * scale}
          onChange={(e) => set(k, (Number(e.target.value) / scale) as never)}
          className="num h-7 w-full font-mono"
        />
        {suffix && <span className="text-2xs text-ink-3">{suffix}</span>}
      </div>
    </label>
  );
  return (
    <div className="space-y-3">
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        {numField("max_iterations", "Max rounds")}
        {numField("target_success_rate", "Stop at success", 5, 100, "%")}
        {numField("demos_per_round", "Demos / round")}
        {numField("episodes_per_branch", "Eval eps / branch")}
      </div>
      <div className="grid grid-cols-1 gap-x-6 sm:grid-cols-2">
        <Toggle
          checked={value.auto_approve_free_training}
          onChange={(v) => set("auto_approve_free_training", v)}
          label="Auto-approve free training"
          hint="University / self-hosted GPUs that cost you nothing"
        />
        <Toggle
          checked={value.allow_paid_jobs}
          onChange={(v) => set("allow_paid_jobs", v)}
          label="Allow paid GPU jobs"
          hint="Off: paid cloud jobs always wait for your click"
        />
      </div>
      <div className="w-[180px]">{numField("max_gpu_hours_per_job", "Max GPU-hours per job", 0.5)}</div>
      <div className="flex flex-wrap items-end gap-4 border-t border-line-soft pt-3">
        <label className="block space-y-1">
          <span className="block text-2xs font-medium text-ink-3">Demonstrations from</span>
          <Segmented
            size="sm"
            value={value.collector}
            onChange={(v) => set("collector", v)}
            ariaLabel="Collector"
            options={[
              { value: "agent", label: "Agent" },
              { value: "oracle", label: "Scripted oracle" },
            ]}
          />
        </label>
        {isAgent(value.collector) && (
          <>
            <label className="block space-y-1">
              <span className="block text-2xs font-medium text-ink-3">Planning mode</span>
              <Segmented
                size="sm"
                value={value.planning_mode}
                onChange={(v) => set("planning_mode", v)}
                ariaLabel="Planning mode"
                options={[
                  { value: "camera", label: "camera" },
                  { value: "privileged", label: "privileged" },
                ]}
              />
            </label>
            <div className="w-[150px]">{numField("model_budget_usd", "Planner budget (USD)", 1)}</div>
          </>
        )}
      </div>
    </div>
  );
}

/** The run's explicit state machine; the loop returns to SCENARIO_PLANNED after NEXT_EXPERIMENT_SELECTED. */
function StateMachine({ run }: { run: AutopilotRun }) {
  const idx = (AUTOPILOT_STATES as readonly string[]).indexOf(run.state);
  const live = run.status === "running" || run.status === "waiting_approval";
  return (
    <div className="border-b border-line-soft px-4 py-2.5">
      <ol className="flex flex-wrap items-center gap-y-1.5 text-[10.5px]" aria-label="Autopilot state machine">
        {AUTOPILOT_STATES.map((s, i) => {
          const cur = i === idx;
          const past = idx >= 0 && i < idx;
          return (
            <li key={s} className="flex items-center">
              <span
                aria-current={cur ? "step" : undefined}
                className={cn(
                  "inline-flex h-5 items-center gap-1 rounded-sm px-1.5 font-mono tracking-tight",
                  cur
                    ? run.status === "waiting_approval"
                      ? "bg-caution/15 text-caution shadow-[0_0_0_1px_rgb(var(--caution)/.45)]"
                      : run.status === "failed"
                        ? "bg-fault/10 text-fault shadow-[0_0_0_1px_rgb(var(--fault)/.4)]"
                        : "bg-servo/15 text-servo shadow-[0_0_0_1px_rgb(var(--servo)/.45)]"
                    : past
                      ? "text-ink-2"
                      : "text-ink-4",
                )}
              >
                {cur && live && <span className="h-1.5 w-1.5 rounded-full bg-current live-dot" />}
                {past && <Check className="h-2.5 w-2.5" />}
                {s}
              </span>
              {i < AUTOPILOT_STATES.length - 1 && <span className="mx-0.5 text-ink-4">→</span>}
            </li>
          );
        })}
      </ol>
      {idx < 0 && <p className="mt-1 font-mono text-[10.5px] text-ink-3">state: {run.state}</p>}
      <p className="mt-1 text-2xs text-ink-4">After NEXT_EXPERIMENT_SELECTED the loop returns to SCENARIO_PLANNED with targeted conditions.</p>
    </div>
  );
}

// ------------------------------------------------------------------ hero
const runTone: Record<AutopilotRun["status"], Tone> = {
  running: "servo",
  waiting_approval: "caution",
  succeeded: "pass",
  stopped: "neutral",
  failed: "fault",
};

export function AutopilotHero({ project }: { project: Project }) {
  const pid = project.id;
  const qc = useQueryClient();
  const q = useAutopilot(pid);
  const run = q.data ?? null;
  const active = run?.status === "running" || run?.status === "waiting_approval";
  const setRun = (r: AutopilotRun | null) => qc.setQueryData(["autopilot", pid], r);
  const start = useMutation({ mutationFn: () => api.startAutopilot(pid), onSuccess: setRun });
  const stop = useMutation({
    mutationFn: () => api.stopAutopilot(pid),
    onSuccess: (r) => (r ? setRun(r) : qc.invalidateQueries({ queryKey: ["autopilot", pid] })),
  });
  const [showPolicy, setShowPolicy] = useState(false);

  return (
    <section className="overflow-hidden rounded-md bg-plate shadow-[0_0_0_1px_var(--line)]">
      <header className="flex flex-wrap items-center gap-x-4 gap-y-2 border-b border-line-soft px-4 py-3">
        <div className="flex items-center gap-2">
          <span
            className={cn(
              "h-2 w-2 rounded-full",
              run?.status === "running" ? "bg-servo live-dot" : run?.status === "waiting_approval" ? "bg-caution live-dot" : "bg-ink-4",
            )}
          />
          <h2 className="text-base font-semibold tracking-[-0.01em]">Autopilot</h2>
          {run && <Badge tone={runTone[run.status]}>{humanize(run.status)}</Badge>}
        </div>
        {run && (
          <div className="flex items-center gap-3 text-xs text-ink-3">
            <span>
              round <span className="num font-mono text-ink">{run.iteration}</span>
              <span className="text-ink-4"> / {project.autopilot.max_iterations}</span>
            </span>
            <span>
              phase <span className="font-mono text-ink">{run.phase}</span>
            </span>
            <ProvenanceBadge planner={run.collector} mode={project.autopilot.planning_mode} />
          </div>
        )}
        <div className="ml-auto flex items-center gap-1.5">
          <Button size="sm" variant="ghost" onClick={() => setShowPolicy((s) => !s)} aria-expanded={showPolicy}>
            Standing approvals <ChevronDown className={cn("h-3.5 w-3.5 transition-transform", showPolicy && "rotate-180")} />
          </Button>
          {active ? (
            <Button size="sm" variant="danger" loading={stop.isPending} onClick={() => stop.mutate()}>
              <Pause className="h-3.5 w-3.5" /> Stop
            </Button>
          ) : (
            <Button size="sm" variant="primary" loading={start.isPending} onClick={() => start.mutate()}>
              <Play className="h-3.5 w-3.5" /> {run ? "Run again" : "Start autopilot"}
            </Button>
          )}
        </div>
      </header>

      {showPolicy && <PolicyEditor project={project} />}
      {(start.error || stop.error) && (
        <div className="px-4 pt-3">
          <ErrorState error={start.error ?? stop.error} compact />
        </div>
      )}

      {q.isLoading ? (
        <div className="space-y-2 p-4">
          <Skeleton className="h-6 w-2/3" />
          <Skeleton className="h-6 w-1/2" />
        </div>
      ) : q.error ? (
        <div className="p-4">
          <ErrorState error={q.error} onRetry={() => q.refetch()} compact />
        </div>
      ) : !run ? (
        <div className="px-4 py-6">
          <p className="max-w-[560px] text-sm text-ink-2">
            Autopilot runs the whole loop for you: collect demos, check quality, build a dataset, train, evaluate in three
            sim branches, and decide what to collect next. It stops at your target success rate or round limit, and only
            pauses for things outside your standing approvals.
          </p>
        </div>
      ) : (
        <>
          <StateMachine run={run} />
          {run.error && <InlineError error={run.error} fallback="Autopilot stopped with an error." className="px-4 pt-3" />}
          <Best best={run.best} />
          <Timeline run={run} pid={pid} />
        </>
      )}
    </section>
  );
}

function PolicyEditor({ project }: { project: Project }) {
  const qc = useQueryClient();
  const [draft, setDraft] = useState<AutopilotPolicy>(project.autopilot);
  useEffect(() => setDraft(project.autopilot), [project.autopilot]);
  const m = useMutation({
    mutationFn: () => api.setAutopilotPolicy(project.id, draft),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["state", project.id] });
      qc.invalidateQueries({ queryKey: ["projects"] });
    },
  });
  const dirty = JSON.stringify(draft) !== JSON.stringify(project.autopilot);
  return (
    <div className="border-b border-line-soft bg-well/60 px-4 py-3">
      <PolicyFields value={draft} onChange={setDraft} />
      <div className="mt-3 flex items-center gap-2">
        <Button size="sm" variant="primary" disabled={!dirty} loading={m.isPending} onClick={() => m.mutate()}>
          Save policy
        </Button>
        {m.isSuccess && !dirty && <span className="text-2xs text-pass">Saved</span>}
        {m.error && <InlineError error={m.error} />}
      </div>
    </div>
  );
}

/** `best` is a free-form dict from the backend; show numbers as numbers, rates as %. */
function Best({ best }: { best: Record<string, unknown> }) {
  const entries = Object.entries(best ?? {}).filter(([, v]) => v !== null && typeof v !== "object");
  if (!entries.length) return null;
  return (
    <div className="flex flex-wrap gap-x-8 gap-y-2 border-b border-line-soft px-4 py-3">
      {entries.map(([k, v]) => (
        <div key={k}>
          <div className="text-2xs font-medium uppercase tracking-[0.06em] text-ink-3">best {humanize(k)}</div>
          <div className="num mt-0.5 font-mono text-lg font-semibold">
            {typeof v === "number" && /rate|success/i.test(k) && v <= 1 ? pct(v) : evidenceValue(v)}
          </div>
        </div>
      ))}
    </div>
  );
}

const dot: Record<AutopilotStep["status"], string> = {
  running: "bg-servo live-dot",
  done: "bg-pass",
  waiting: "bg-caution live-dot",
  failed: "bg-fault",
  skipped: "bg-ink-4",
};

function Timeline({ run, pid }: { run: AutopilotRun; pid: string }) {
  const ref = useRef<HTMLOListElement>(null);
  useEffect(() => {
    const el = ref.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [run.steps.length]);
  if (run.steps.length === 0) return <p className="px-4 py-4 text-xs text-ink-3">Starting…</p>;
  const last = run.steps.length - 1;
  return (
    <ol ref={ref} className="max-h-[420px] overflow-y-auto px-4 py-3 scroll-thin">
      {run.steps.map((s, i) => {
        // only the latest waiting step is actionable; earlier ones were resolved
        const actionable = s.status === "waiting" && i === last && run.status === "waiting_approval";
        return (
          <li key={i} className="relative grid grid-cols-[16px_64px_minmax(0,1fr)_auto] gap-x-2 pb-3 last:pb-0">
            {i < last && <span className="absolute left-[7px] top-3 h-full w-px bg-[var(--line)]" aria-hidden />}
            <span className={cn("relative mt-1.5 h-2 w-2 justify-self-center rounded-full", dot[s.status])} />
            <span className="mt-px truncate font-mono text-2xs uppercase tracking-wide text-ink-3">{s.phase}</span>
            <div className="min-w-0">
              <p className={cn("text-sm", s.status === "failed" ? "text-fault" : s.status === "skipped" ? "text-ink-3" : "text-ink")}>
                {s.message}
              </p>
              {s.ref_ids.length > 0 && (
                <div className="mt-1 flex flex-wrap gap-1">
                  {s.ref_ids.map((r) => (
                    <RefChip key={r} pid={pid} id={r} />
                  ))}
                </div>
              )}
              {actionable && <WaitingAction pid={pid} step={s} />}
            </div>
            <time className="mt-px text-2xs text-ink-4" dateTime={s.at}>
              {new Date(s.at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })}
            </time>
          </li>
        );
      })}
    </ol>
  );
}

/** Inline resolution for a waiting step: approve a training job, or finish the manual training path. */
function WaitingAction({ pid, step }: { pid: string; step: AutopilotStep }) {
  const training = useTraining(pid);
  const tid = step.ref_ids.find((r) => r.startsWith("tj_"));
  const tj = training.data?.find((t) => t.id === tid);
  if (!tid) return null;
  if (training.isLoading) return <Skeleton className="mt-2 h-8 w-64" />;
  if (!tj) return null;
  if (tj.backend === "manual") return <ManualTraining pid={pid} tid={tj.id} />;
  if (tj.status !== "proposed") return null;
  return <ApproveTraining tid={tj.id} pid={pid} cost={tj.estimate.est_cost_usd} paid={tj.estimate.paid} />;
}

function ApproveTraining({ tid, pid, cost, paid }: { tid: string; pid: string; cost: number; paid: boolean }) {
  const qc = useQueryClient();
  const [approver, setApprover] = useApprover();
  const m = useMutation({
    mutationFn: () => api.approveTraining(tid, approver.trim()),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["training", pid] }),
  });
  return (
    <div className="mt-2 flex flex-wrap items-center gap-2 rounded bg-caution/5 p-2 shadow-[0_0_0_1px_rgb(var(--caution)/.3)]">
      <ShieldAlert className="h-3.5 w-3.5 text-caution" />
      <Input value={approver} onChange={(e) => setApprover(e.target.value)} placeholder="Your name" className="h-7 w-40" aria-label="Approver name" />
      <Button size="sm" variant="primary" disabled={!approver.trim()} loading={m.isPending} onClick={() => m.mutate()}>
        <Check className="h-3.5 w-3.5" /> Approve{paid ? ` $${cost.toFixed(2)}` : ""}
      </Button>
      <span className="text-2xs text-ink-3">Autopilot launches it once approved.</span>
      {m.error && <InlineError error={m.error} className="w-full" />}
    </div>
  );
}

/** No GPU backend: show the command bundle and let the user register the resulting checkpoint. */
export function ManualTraining({ pid, tid }: { pid: string; tid: string }) {
  const qc = useQueryClient();
  const cmd = useQuery({ queryKey: ["training-command", tid], queryFn: () => api.trainingCommand(tid) });
  const [path, setPath] = useState("");
  const [copied, setCopied] = useState(false);
  const reg = useMutation({
    mutationFn: () => api.registerCheckpoint(pid, path.trim(), tid),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["checkpoints", pid] });
      qc.invalidateQueries({ queryKey: ["training", pid] });
      setPath("");
    },
  });
  return (
    <div className="mt-2 space-y-2 rounded bg-well p-2.5 shadow-[inset_0_0_0_1px_var(--line)]">
      <div className="flex items-center justify-between text-2xs text-ink-3">
        <span>Run this on a GPU machine</span>
        {cmd.data && (
          <button
            type="button"
            className="inline-flex items-center gap-1 hover:text-ink"
            onClick={() => {
              navigator.clipboard?.writeText(cmd.data.command).then(() => setCopied(true));
            }}
          >
            <Copy className="h-3 w-3" /> {copied ? "copied" : "copy"}
          </button>
        )}
      </div>
      {cmd.isLoading ? (
        <Skeleton className="h-10 w-full" />
      ) : cmd.error ? (
        <ErrorState error={cmd.error} compact />
      ) : (
        <pre className="max-h-40 overflow-auto whitespace-pre-wrap break-all font-mono text-[11px] leading-relaxed text-ink-2 scroll-thin">
          {cmd.data?.command}
        </pre>
      )}
      <div className="flex gap-1.5">
        <Input value={path} onChange={(e) => setPath(e.target.value)} placeholder="Checkpoint path (…/pretrained_model)" className="h-7 font-mono text-xs" />
        <Button size="sm" disabled={!path.trim()} loading={reg.isPending} onClick={() => reg.mutate()}>
          <Upload className="h-3.5 w-3.5" /> Register
        </Button>
      </div>
      {reg.isSuccess && <p className="text-2xs text-pass">Checkpoint registered. Autopilot will pick it up.</p>}
      {reg.error && <InlineError error={reg.error} />}
    </div>
  );
}
