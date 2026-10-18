import { useEffect, useState, type ReactNode } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { Check, Copy } from "lucide-react";
import { api, resolveMediaUrl } from "../api/client";
import { useJob, useLogs, useTrackJob } from "../api/events";
import type { ChatCard, FeasCheck, Job, PatchOption } from "../api/types";
import { Button, Input } from "../components/ui";
import { cn, errorMessage } from "../lib/format";
import { useApprover } from "./approver";
import type { Send } from "./useChat";

// ------------------------------------------------------------------ primitives
export function Label({ children, rule, className }: { children: ReactNode; rule?: boolean; className?: string }) {
  return <div className={cn("label", rule && "label-rule", className)}>{children}</div>;
}

export function Frame({ label, meta, children, tone }: { label: ReactNode; meta?: ReactNode; children: ReactNode; tone?: "accent" }) {
  return (
    <section className={cn("enter rounded-md border bg-plate", tone === "accent" ? "border-servo/50" : "border-line")}>
      <header className="flex items-center justify-between gap-3 border-b border-line px-4 py-2.5">
        <Label>{label}</Label>
        {meta && <span className="label !tracking-[0.12em]">{meta}</span>}
      </header>
      <div className="px-4 py-3.5">{children}</div>
    </section>
  );
}

export function Pill({ children, onClick, disabled, primary }: { children: ReactNode; onClick: () => void; disabled?: boolean; primary?: boolean }) {
  return (
    <Button size="sm" variant={primary ? "primary" : "secondary"} onClick={onClick} disabled={disabled}>
      {children}
      {!primary && <span aria-hidden>→</span>}
    </Button>
  );
}

export function TextLink({ children, onClick, disabled }: { children: ReactNode; onClick: () => void; disabled?: boolean }) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className="border-b border-[var(--line-strong)] pb-px text-sm text-ink hover:border-ink disabled:pointer-events-none disabled:opacity-40"
    >
      {children} <span aria-hidden>→</span>
    </button>
  );
}

const pct = (x: number) => `${Math.round(x * 100)}%`;

// ------------------------------------------------------------------ dispatcher
export function ChatCardView({ card, send, live, pid }: { card: ChatCard; send: Send; live: boolean; pid: string }) {
  switch (card.type) {
    case "question":
      return <QuestionCard card={card} send={send} live={live} />;
    case "image":
      return <SceneFigure card={card} />;
    case "feasibility":
      return <FeasibilityCard card={card} send={send} live={live} />;
    case "job":
      return <JobCard card={card} send={send} live={live} />;
    case "training_manual":
      return <ManualTrainingCard card={card} send={send} live={live} pid={pid} />;
    case "results":
      return <ResultsCard card={card} />;
    case "actions":
      return (
        <div className="enter flex flex-wrap gap-2">
          {card.actions.map((a) => (
            <Pill key={a.label} onClick={() => send(a.message)} disabled={!live}>
              {a.label}
            </Pill>
          ))}
        </div>
      );
  }
}

// ------------------------------------------------------------------ question
function QuestionCard({ card, send, live }: { card: Extract<ChatCard, { type: "question" }>; send: Send; live: boolean }) {
  return (
    <Frame label={`Question · ${card.path}`} meta={card.question_type}>
      <p className="text-[15px] leading-relaxed text-ink">{card.prompt}</p>
      <p className="mt-1 text-sm text-ink-3">{card.why}</p>
      {card.options.length > 0 && (
        <div className="mt-3.5 flex flex-wrap gap-2">
          {card.options.map((o: PatchOption) => (
            <Pill key={o.label} onClick={() => send(o.label, o.patch)} disabled={!live}>
              {o.label}
            </Pill>
          ))}
        </div>
      )}
      {live && card.allow_text && <p className="mt-3 text-xs text-ink-3">Or answer in your own words below.</p>}
    </Frame>
  );
}

// ------------------------------------------------------------------ scene figure
function SceneFigure({ card }: { card: Extract<ChatCard, { type: "image" }> }) {
  const src = card.image_jpeg_base64 ? `data:image/jpeg;base64,${card.image_jpeg_base64}` : resolveMediaUrl(card.url);
  return (
    <figure className="enter overflow-hidden rounded-md border border-line">
      <div className="drafting relative flex justify-center px-6 py-8 sm:px-12">
        <svg className="pointer-events-none absolute inset-0 h-full w-full text-ink-4" aria-hidden>
          <circle cx="50%" cy="50%" r="46%" fill="none" stroke="currentColor" strokeDasharray="2 5" strokeWidth="1" opacity=".55" />
          <line x1="50%" y1="2%" x2="50%" y2="98%" stroke="currentColor" strokeDasharray="1 6" opacity=".4" />
          <line x1="2%" y1="50%" x2="98%" y2="50%" stroke="currentColor" strokeDasharray="1 6" opacity=".4" />
        </svg>
        {src ? (
          <img src={src} alt={`Simulated scene from the ${card.camera} camera`} className="relative z-[1] aspect-square w-full max-w-[360px] border border-line object-cover" />
        ) : (
          <div className="relative z-[1] grid aspect-square w-full max-w-[360px] place-items-center border border-dashed border-line-strong">
            <Label>No image</Label>
          </div>
        )}
        <div className="absolute left-4 top-4 z-[2] flex items-center gap-2">
          <Label className="!text-ink-2">{card.label}</Label>
        </div>
        <div className="absolute bottom-4 right-4 z-[2] flex items-center gap-2">
          <span className="h-px w-10 bg-[var(--line-strong)]" aria-hidden />
          <Label>{card.scene ? `Scene ${card.scene.slice(0, 8)}` : "Sim render"}</Label>
        </div>
      </div>
    </figure>
  );
}

// ------------------------------------------------------------------ feasibility
function Mark({ c }: { c: FeasCheck }) {
  return <span className={cn("font-mono text-xs", c.ok ? "text-servo" : "text-fault")}>{c.ok ? "OK" : "FAIL"}</span>;
}

function FeasibilityCard({ card, send, live }: { card: Extract<ChatCard, { type: "feasibility" }>; send: Send; live: boolean }) {
  return (
    <Frame
      label="Feasibility · measured on the arm model"
      meta={card.jaw_opening_m ? `Jaw ${(card.jaw_opening_m * 1000).toFixed(0)} mm` : undefined}
      tone={card.ok ? undefined : "accent"}
    >
      <div className="divide-y divide-[var(--line-soft)]">
        {card.objects.map((o) => (
          <div key={o.name} className="grid gap-x-6 gap-y-1 py-2.5 first:pt-0 sm:grid-cols-[140px_1fr]">
            <div>
              <div className="text-sm font-medium text-ink">{o.name.replace(/_/g, " ")}</div>
              <Label className="mt-0.5">
                {o.grasp_mode} grasp{o.tall ? " · tall" : ""}
              </Label>
            </div>
            <dl className="space-y-1 text-sm">
              <div className="flex gap-3">
                <dt className="label w-16 shrink-0 pt-0.5">Grasp</dt>
                <dd className="flex-1 text-ink-2">
                  <Mark c={o.grasp} /> · {o.grasp.detail}
                </dd>
              </div>
              <div className="flex gap-3">
                <dt className="label w-16 shrink-0 pt-0.5">Payload</dt>
                <dd className="flex-1 text-ink-2">
                  <Mark c={o.payload} /> · {o.payload.detail}
                </dd>
              </div>
            </dl>
          </div>
        ))}
        <div className="grid gap-x-6 py-2.5 last:pb-0 sm:grid-cols-[140px_1fr]">
          <div className="text-sm font-medium text-ink">Reach</div>
          <ul className="space-y-0.5">
            {card.places.map((p) => (
              <li key={`${p.kind}-${p.name}`} className="font-mono text-xs uppercase tracking-[0.08em] text-ink-2">
                <span className={p.ok ? "text-servo" : "text-fault"}>{p.ok ? "●" : "○"}</span> {p.name} {p.kind} · IK residual{" "}
                {Math.max(p.grasp_err_mm, p.carry_err_mm).toFixed(1)} mm
              </li>
            ))}
          </ul>
        </div>
      </div>
      {card.suggestions.length > 0 && (
        <div className="mt-3 border-t border-line pt-3">
          <Label className="mb-2">Suggested fixes</Label>
          <div className="flex flex-wrap gap-2">
            {card.suggestions.map((s) => (
              <Pill key={s.label} onClick={() => send(s.label, s.patch)} disabled={!live}>
                {s.label}
              </Pill>
            ))}
          </div>
        </div>
      )}
    </Frame>
  );
}

// ------------------------------------------------------------------ job progress (live over SSE)
function JobCard({ card, send, live }: { card: Extract<ChatCard, { type: "job" }>; send: Send; live: boolean }) {
  const track = useTrackJob();
  useEffect(() => {
    track(card.job);
    api.job(card.job_id).then(track).catch(() => {});
  }, [card.job_id]); // eslint-disable-line react-hooks/exhaustive-deps
  const j: Job = useJob(card.job_id) ?? card.job;
  const logs = useLogs([card.job_id]);
  const active = j.status === "queued" || j.status === "running";
  const state = j.status === "succeeded" ? "Done" : j.status === "failed" ? "Failed" : j.status === "cancelled" ? "Stopped" : j.status;
  return (
    <Frame
      label={card.title}
      meta={
        <span className="inline-flex items-center gap-1.5">
          <span className={cn("h-1.5 w-1.5 rounded-full", active ? "bg-servo live-dot" : j.status === "succeeded" ? "bg-servo" : "bg-fault")} />
          {state} · {pct(j.progress)}
        </span>
      }
    >
      <div className="h-px w-full bg-[var(--line)]">
        <div className="h-px bg-servo transition-[width] duration-500 ease-out" style={{ width: pct(Math.min(1, j.progress)) }} />
      </div>
      <p className="mt-2.5 min-h-[20px] text-sm text-ink-2">{j.error ?? j.message ?? ""}</p>
      {logs.length > 0 && (
        <pre className="mt-2 max-h-28 overflow-auto whitespace-pre-wrap font-mono text-[11px] leading-relaxed text-ink-3 scroll-thin">
          {logs.slice(-6).join("\n")}
        </pre>
      )}
      <div className="mt-2 flex items-center justify-between">
        <Label>{card.job_id}</Label>
        {!active && live && <TextLink onClick={() => send("Continue")}>{j.status === "succeeded" ? "Continue" : "Check again"}</TextLink>}
      </div>
    </Frame>
  );
}

// ------------------------------------------------------------------ approvals (human click only)
export function ApprovalCard({
  kind,
  refId,
  summary,
  args,
  pid,
  send,
  live,
}: {
  kind: string;
  refId: string;
  summary: string;
  args: Record<string, unknown>;
  pid: string;
  send: Send;
  live: boolean;
}) {
  const qc = useQueryClient();
  const [approver, setApprover] = useApprover();
  const [state, setState] = useState<"idle" | "pending" | "done" | "declined">("idle");
  const [error, setError] = useState<string | null>(null);
  const training = kind === "training";
  const evaluation = kind === "evaluation";

  async function approve() {
    setState("pending");
    setError(null);
    try {
      if (training) {
        await api.approveTraining(refId, approver.trim());
        await api.launchTraining(refId);
      } else if (evaluation) {
        await api.launchEvaluation(pid, {
          policy: { kind: "checkpoint", checkpoint_id: String(args.policy ?? refId) },
          episodes_per_branch: typeof args.episodes_per_branch === "number" ? args.episodes_per_branch : 10,
        });
      }
      qc.invalidateQueries({ predicate: (q) => q.queryKey.includes(pid) });
      setState("done");
      send("Continue");
    } catch (e) {
      setError(errorMessage(e));
      setState("idle");
    }
  }

  const facts: [string, unknown][] = training
    ? [
        ["Backend", args.backend],
        ["Steps", args.steps],
        ["GPU-hours", args.gpu_hours],
        ["Cost", typeof args.est_cost_usd === "number" ? `$${args.est_cost_usd.toFixed(2)}` : undefined],
      ]
    : [["Policy", args.policy], ["Seeds / branch", args.episodes_per_branch]];

  return (
    <Frame label={`Approval required · ${kind}`} meta={refId} tone="accent">
      <p className="text-sm leading-relaxed text-ink">{summary}</p>
      <dl className="mt-3 grid grid-cols-2 gap-x-6 gap-y-1.5 sm:grid-cols-4">
        {facts
          .filter(([, v]) => v !== undefined && v !== null)
          .map(([k, v]) => (
            <div key={k}>
              <dt className="label">{k}</dt>
              <dd className="num font-mono text-sm text-ink">{String(v)}</dd>
            </div>
          ))}
      </dl>
      {live && (state === "idle" || state === "pending") && (
        <div className="mt-4 flex flex-wrap items-center gap-2">
          {training && (
            <Input
              value={approver}
              onChange={(e) => setApprover(e.target.value)}
              placeholder="Your name (recorded as approver)"
              className="!h-8 max-w-[240px] !rounded-full"
              aria-label="Approver name"
            />
          )}
          <Button size="sm" variant="primary" loading={state === "pending"} disabled={training && !approver.trim()} onClick={approve}>
            <Check className="h-3.5 w-3.5" /> {training ? "Approve and launch" : "Approve and evaluate"}
          </Button>
          <Button
            size="sm"
            variant="ghost"
            onClick={() => {
              setState("declined");
              send(`I decline ${kind} ${refId}. Do not launch it.`);
            }}
          >
            Decline
          </Button>
        </div>
      )}
      {state === "done" && <Label className="mt-3 !text-servo">● Approved by {training ? approver : "you"} · launched</Label>}
      {state === "declined" && <Label className="mt-3">Declined</Label>}
      {!live && state === "idle" && <Label className="mt-3">Superseded by a later step</Label>}
      {error && <p className="mt-2 text-sm text-fault">{error}</p>}
    </Frame>
  );
}

// ------------------------------------------------------------------ manual training (the GPU gap, stated plainly)
function ManualTrainingCard({
  card,
  send,
  live,
}: {
  card: Extract<ChatCard, { type: "training_manual" }>;
  send: Send;
  live: boolean;
  pid: string;
}) {
  const [path, setPath] = useState("");
  const [copied, setCopied] = useState(false);
  return (
    <Frame label="Training · needs a GPU" meta={`${card.backend} backend · ${card.training_job_id}`} tone="accent">
      <p className="text-sm leading-relaxed text-ink">
        No GPU cluster is configured, so Cortex cannot run this job itself. Run the command on any machine with a GPU, then
        register the checkpoint here. Nothing is marked as trained until a real checkpoint is on disk.
      </p>
      <div className="relative mt-3">
        <pre className="max-h-52 overflow-auto whitespace-pre-wrap break-all rounded border border-line bg-well p-3 pr-10 font-mono text-[11.5px] leading-relaxed text-ink-2 scroll-thin">
          {card.command}
        </pre>
        <button
          type="button"
          className="absolute right-2 top-2 rounded-full p-1.5 text-ink-3 hover:bg-rail hover:text-ink"
          aria-label="Copy command"
          onClick={() => {
            void navigator.clipboard?.writeText(card.command).then(() => setCopied(true));
            setTimeout(() => setCopied(false), 1500);
          }}
        >
          {copied ? <Check className="h-3.5 w-3.5" /> : <Copy className="h-3.5 w-3.5" />}
        </button>
      </div>
      {live && (
        <form
          className="mt-3 flex flex-wrap items-center gap-2"
          onSubmit={(e) => {
            e.preventDefault();
            if (path.trim()) send(path.trim());
          }}
        >
          <Input
            value={path}
            onChange={(e) => setPath(e.target.value)}
            placeholder={card.checkpoint_hint}
            className="!h-8 min-w-[260px] flex-1 font-mono !text-xs"
            aria-label="Checkpoint path"
          />
          <Button size="sm" variant="primary" type="submit" disabled={!path.trim()}>
            Register checkpoint
          </Button>
        </form>
      )}
      <div className="mt-4 border-t border-line pt-3">
        <Label className="mb-1.5">No GPU? Smoke test on this machine</Label>
        {card.smoke_available ? (
          <div className="flex flex-wrap items-center justify-between gap-3">
            <p className="max-w-[460px] text-sm text-ink-3">
              Runs lerobot-train on the CPU for {card.smoke_steps} steps. It proves the dataset trains end to end and gives a real
              checkpoint, but the policy will not succeed at the task. It is labelled as a smoke checkpoint.
            </p>
            <TextLink onClick={() => send("Run the CPU smoke test")} disabled={!live}>
              Run {card.smoke_steps}-step smoke test
            </TextLink>
          </div>
        ) : (
          <p className="text-sm text-ink-3">Not available here: {card.smoke_reason}</p>
        )}
      </div>
    </Frame>
  );
}

// ------------------------------------------------------------------ results (measured, with Wilson CIs)
function ResultsCard({ card }: { card: Extract<ChatCard, { type: "results" }> }) {
  return (
    <Frame label="Results · measured in simulation" meta={`${card.checkpoint_id} · step ${card.checkpoint_step}`}>
      {card.smoke && <p className="mb-3 text-sm text-caution">CPU smoke checkpoint ({card.checkpoint_step} steps): a pipeline check, not a trained policy.</p>}
      <div className="space-y-3">
        {card.branches.map((b) => (
          <div key={b.name} className="grid grid-cols-[120px_1fr_88px] items-center gap-4">
            <div className="truncate text-sm text-ink">{b.name.replace(/_/g, " ")}</div>
            <div className="relative h-4" aria-label={`95% CI ${pct(b.ci95[0])} to ${pct(b.ci95[1])}`}>
              <div className="absolute inset-x-0 top-1/2 h-px bg-[var(--line)]" />
              <div className="absolute top-1/2 h-[5px] -translate-y-1/2 bg-servo/25" style={{ left: pct(b.ci95[0]), width: pct(b.ci95[1] - b.ci95[0]) }} />
              <div className="absolute top-1/2 h-3 w-px -translate-y-1/2 bg-servo" style={{ left: pct(b.success_rate) }} />
            </div>
            <div className="num text-right font-mono text-sm text-ink">
              {pct(b.success_rate)} <span className="text-ink-3">{b.successes}/{b.n}</span>
            </div>
          </div>
        ))}
      </div>
      <Label className="mt-3">Bars show the 95% Wilson interval · {card.evaluation_id}</Label>
    </Frame>
  );
}
