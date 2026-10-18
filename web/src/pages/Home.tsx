import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useQueryClient } from "@tanstack/react-query";
import { api } from "../api/client";
import { useProjects } from "../api/hooks";
import type { Project } from "../api/types";
import { StepRail } from "../chat/ChatThread";
import { Label, TextLink } from "../chat/cards";
import { Button, ErrorState, Skeleton } from "../components/ui";
import { ago, errorMessage } from "../lib/format";

const EXAMPLES = [
  "Pick up the cube and put it on the left spot",
  "Move the small water bottle onto the coaster",
  "Put the tennis ball in a zone on the right",
];

export function HomePage() {
  return (
    <div className="h-full overflow-y-auto scroll-thin">
      <div className="mx-auto grid max-w-[1180px] gap-16 px-5 pb-20 pt-14 sm:px-10 lg:grid-cols-[minmax(0,1fr)_380px] lg:pt-20">
        <Ask />
        <ReachDrawing />
      </div>
      <div className="mx-auto max-w-[1180px] px-5 pb-24 sm:px-10">
        <Recent />
      </div>
    </div>
  );
}

function Ask() {
  const nav = useNavigate();
  const qc = useQueryClient();
  const [prompt, setPrompt] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function start(text: string) {
    const t = text.trim();
    if (!t || busy) return;
    setBusy(true);
    setError(null);
    try {
      const st = await api.startIntake(t);
      qc.invalidateQueries({ queryKey: ["projects"] });
      nav(`/p/${st.project_id}`, { state: { initial: t } });
    } catch (e) {
      setError(errorMessage(e));
      setBusy(false);
    }
  }

  return (
    <section>
      <Label rule>Fig.00 Teach a task · physics sim</Label>
      <h1 className="display mt-7 text-[46px] text-ink sm:text-[64px] lg:text-[76px]">
        What should your
        <br />
        <span className="text-muted">robot learn?</span>
      </h1>
      <p className="mt-6 max-w-[520px] text-[15px] leading-relaxed text-ink-2">
        Describe the task in plain words. Cortex builds the scene in MuJoCo, checks the arm can do it, collects demonstrations and
        trains a policy for your SO-101, step by step in one conversation.
      </p>
      <form
        className="mt-9 max-w-[640px]"
        onSubmit={(e) => {
          e.preventDefault();
          void start(prompt);
        }}
      >
        <div className="rounded-md border border-line-strong bg-bench p-2 transition-colors focus-within:border-ink">
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                void start(prompt);
              }
            }}
            rows={3}
            autoFocus
            placeholder="Pick up the red cube and put it on the left spot"
            className="w-full resize-none bg-transparent px-2 py-1.5 text-[17px] text-ink placeholder:text-ink-4 focus:outline-none"
            aria-label="What should your robot learn?"
          />
          <div className="flex items-center justify-between gap-3 px-1 pb-0.5">
            <Label className="!text-ink-4">SO-101 · MuJoCo</Label>
            <Button variant="primary" type="submit" loading={busy} disabled={!prompt.trim()}>
              Start teaching
            </Button>
          </div>
        </div>
        {error && <p className="mt-2 text-sm text-fault">{error}</p>}
      </form>
      <ul className="mt-7 space-y-2.5">
        {EXAMPLES.map((x) => (
          <li key={x}>
            <TextLink onClick={() => void start(x)} disabled={busy}>
              {x}
            </TextLink>
          </li>
        ))}
      </ul>
      <div className="mt-14 max-w-[640px]">
        <StepRail />
        <p className="mt-3 text-sm text-ink-3">
          Training and evaluation always wait for your click. Prefer forms?{" "}
          <Link to="/new" className="border-b border-[var(--line-strong)] text-ink-2 hover:border-ink hover:text-ink">
            Advanced setup →
          </Link>
        </p>
      </div>
    </section>
  );
}

/** Top view of the SO-101's reach, drawn from the intake defaults (workspace box, home pose). */
function ReachDrawing() {
  const home = [
    ["J01 / Shoulder pan", "0.0°"],
    ["J02 / Shoulder lift", "−60.0°"],
    ["J03 / Elbow", "60.0°"],
    ["J04 / Wrist flex", "35.0°"],
    ["J05 / Wrist roll", "0.0°"],
    ["J06 / Gripper", "30 / 100"],
  ];
  // table frame: x away from the base (up on the drawing), y to the arm's left; 1 m = 600 px
  const s = 600;
  const cx = 190;
  const cy = 300;
  const rect = { x0: 0.08, x1: 0.36, y0: -0.22, y1: 0.24 };
  return (
    <aside className="hidden lg:block" aria-label="SO-101 reach, top view">
      <Label rule>Fig.01 Reach · top view</Label>
      <div className="drafting mt-7 rounded-md border border-line">
        <svg viewBox="0 0 380 330" className="h-auto w-full text-ink-3" fill="none" stroke="currentColor">
          {[0.42, 0.3, 0.18].map((r) => (
            <path key={r} d={`M ${cx - r * s} ${cy} A ${r * s} ${r * s} 0 0 1 ${cx + r * s} ${cy}`} strokeDasharray="2 5" opacity={r === 0.42 ? 0.9 : 0.5} />
          ))}
          <rect
            x={cx - rect.y1 * s}
            y={cy - rect.x1 * s}
            width={(rect.y1 - rect.y0) * s}
            height={(rect.x1 - rect.x0) * s}
            stroke="rgb(var(--servo))"
            strokeWidth="1"
          />
          <line x1={cx} y1={cy} x2={cx} y2={cy - 0.42 * s} strokeDasharray="1 4" />
          <circle cx={cx} cy={cy} r="7" fill="rgb(var(--bench))" />
          <circle cx={cx} cy={cy} r="2" fill="currentColor" stroke="none" />
          <line x1={cx + 0.24 * s} y1={cy - 0.36 * s} x2={330} y2={34} opacity=".7" />
          <text x="236" y="28" className="fill-current font-mono" fontSize="9" letterSpacing="1.4" stroke="none">
            WORKSPACE
          </text>
          <text x={cx + 6} y={cy - 0.42 * s + 12} className="fill-current font-mono" fontSize="9" letterSpacing="1.4" stroke="none">
            REACH 0.42 M
          </text>
          <text x={cx + 12} y={cy - 4} className="fill-current font-mono" fontSize="9" letterSpacing="1.4" stroke="none">
            BASE
          </text>
        </svg>
      </div>
      <dl className="mt-5 divide-y divide-[var(--line-soft)] border-y border-line">
        {home.map(([k, v]) => (
          <div key={k} className="flex items-center justify-between py-1.5">
            <dt className="label">{k}</dt>
            <dd className="num font-mono text-xs text-ink">{v}</dd>
          </div>
        ))}
      </dl>
      <div className="mt-3 flex justify-between">
        <Label>Home pose · sim default</Label>
        <Label className="!text-ink-2">● Idle · hold</Label>
      </div>
    </aside>
  );
}

function Recent() {
  const q = useProjects();
  if (q.isLoading) return <Skeleton className="h-24 w-full" />;
  if (q.error) return <ErrorState error={q.error} onRetry={() => q.refetch()} />;
  const rows = [...(q.data ?? [])].sort((a, b) => (a.created_at < b.created_at ? 1 : -1)).slice(0, 12);
  if (rows.length === 0) return null;
  return (
    <section>
      <Label rule>Recent projects</Label>
      <ul className="mt-5 border-t border-line">
        {rows.map((p: Project) => (
          <li key={p.id} className="border-b border-line">
            <Link to={`/p/${p.id}`} className="group grid grid-cols-[1fr_auto] items-baseline gap-4 py-3.5 sm:grid-cols-[1fr_120px_90px_60px]">
              <div className="min-w-0">
                <div className="truncate text-[15px] text-ink">{p.task_prompt}</div>
                <div className="label mt-0.5 truncate !tracking-[0.12em]">{p.name}</div>
              </div>
              <span className="label hidden sm:block">{p.intake ? "Guided" : "Classic"}</span>
              <span className="label hidden sm:block">{p.mode === "sim" ? "Sim" : "Real"}</span>
              <span className="text-right text-xs text-ink-3 group-hover:text-ink">
                {ago(p.created_at)} <span aria-hidden>→</span>
              </span>
            </Link>
          </li>
        ))}
      </ul>
    </section>
  );
}
