import { useEffect, useState, type ReactNode } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { ArrowRight, Box, ChevronDown, Cpu, Info, Zap } from "lucide-react";
import { api } from "../api/client";
import { usePlanner } from "../api/hooks";
import { DEFAULT_AUTOPILOT_POLICY, type AutopilotPolicy, type Mode, type Project } from "../api/types";
import { Button, ErrorState, Input } from "../components/ui";
import { ProjectSpecView } from "./Plan";
import { PolicyFields } from "./Autopilot";
import { cn } from "../lib/format";

const PLACEHOLDER = "Teach a robot to pick and place a cube.";

export function NewProjectForm({ onCreated }: { onCreated: (p: Project, autopilot: boolean) => void }) {
  const qc = useQueryClient();
  const [prompt, setPrompt] = useState("");
  const [autopilot, setAutopilot] = useState(true);
  const [policy, setPolicy] = useState<AutopilotPolicy>(DEFAULT_AUTOPILOT_POLICY);
  const planner = usePlanner();
  const [policyTouched, setPolicyTouched] = useState(false);
  // demonstrations come from GPT-6 Astra only when it is configured; never pretend otherwise
  useEffect(() => {
    if (!policyTouched && planner.data) setPolicy((p) => ({ ...p, collector: planner.data.available ? "astra" : "oracle" }));
  }, [planner.data, policyTouched]);
  const [open, setOpen] = useState<"none" | "approvals" | "options">("none");
  const [mode, setMode] = useState<Mode>("sim");
  const [budget, setBudget] = useState(24);
  const [name, setName] = useState("");
  const create = useMutation({
    mutationFn: () =>
      api.createProject({
        task_prompt: prompt.trim() || PLACEHOLDER,
        mode,
        demo_budget: budget,
        name: name.trim() || null,
        autopilot: { ...policy, enabled: autopilot },
        start_autopilot: autopilot,
      }),
    onSuccess: (p) => {
      qc.invalidateQueries({ queryKey: ["projects"] });
      onCreated(p, autopilot);
    },
  });
  const toggle = (k: "approvals" | "options") => setOpen((o) => (o === k ? "none" : k));

  return (
    <form
      onSubmit={(e) => {
        e.preventDefault();
        create.mutate();
      }}
    >
      <label htmlFor="task-prompt" className="mb-2 block text-3xl font-semibold tracking-[-0.02em]">
        What should your robot learn?
      </label>
      <p className="mb-5 text-sm text-ink-3">
        Describe the skill in plain words. Cortex writes the task spec and collection plan, then (on autopilot) collects
        demos, trains, and evaluates until it hits your target.
      </p>

      <div className="rounded-lg bg-well shadow-[inset_0_0_0_1px_var(--line-strong)] focus-within:shadow-[inset_0_0_0_1px_rgb(var(--servo)/.7)]">
        <textarea
          id="task-prompt"
          autoFocus
          rows={3}
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) create.mutate();
          }}
          placeholder={PLACEHOLDER}
          className="block w-full resize-none bg-transparent px-4 pb-2 pt-3.5 text-lg text-ink placeholder:text-ink-4 focus:outline-none"
        />
        <div className="flex flex-wrap items-center gap-2 px-3 pb-3">
          <button
            type="button"
            role="switch"
            aria-checked={autopilot}
            onClick={() => setAutopilot((a) => !a)}
            className={cn(
              "inline-flex h-8 items-center gap-2 rounded-full pl-1.5 pr-3 text-sm font-medium transition-colors duration-150",
              autopilot ? "bg-servo/15 text-servo shadow-[0_0_0_1px_rgb(var(--servo)/.45)]" : "bg-rail text-ink-3 shadow-[0_0_0_1px_var(--line)]",
            )}
          >
            <span className={cn("flex h-5 w-5 items-center justify-center rounded-full", autopilot ? "bg-servo text-servo-ink" : "bg-ink-4/40 text-ink-3")}>
              <Zap className="h-3 w-3" />
            </span>
            Autopilot {autopilot ? "on" : "off"}
          </button>
          {autopilot && (
            <Disclosure label="Standing approvals" open={open === "approvals"} onClick={() => toggle("approvals")} />
          )}
          <Disclosure
            label={`${mode === "sim" ? "Simulation" : "Real robot"} · ${budget} demos`}
            open={open === "options"}
            onClick={() => toggle("options")}
          />
          <div className="ml-auto flex items-center gap-2">
            <span className="hidden text-2xs text-ink-4 sm:inline">⌘ Enter</span>
            <Button variant="primary" type="submit" loading={create.isPending} className="h-8 px-3.5">
              {autopilot ? "Start" : "Create project"} <ArrowRight className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </div>

      {open === "approvals" && autopilot && (
        <div className="enter mt-3 rounded-md bg-plate p-3.5 shadow-[0_0_0_1px_var(--line)]">
          <p className="mb-3 text-xs text-ink-3">
            What autopilot may do without asking. Anything outside these limits pauses the run and waits for your click.
          </p>
          {planner.data && !planner.data.available && (
            <p className="mb-3 text-2xs text-caution">GPT-6 Astra not configured ({planner.data.reason}) Autopilot will use the scripted oracle.</p>
          )}
          <PolicyFields
            value={policy}
            onChange={(p) => {
              setPolicyTouched(true);
              setPolicy(p);
            }}
          />
        </div>
      )}

      {open === "options" && (
        <div className="enter mt-3 space-y-3 rounded-md bg-plate p-3.5 shadow-[0_0_0_1px_var(--line)]">
          <div className="grid grid-cols-2 gap-2.5">
            <ModeCard
              active={mode === "sim"}
              onClick={() => setMode("sim")}
              icon={<Box className="h-4 w-4" />}
              title="Simulation"
              sub="MuJoCo scene. Scripted-expert demos, no hardware needed."
              tag="default"
            />
            <ModeCard active={mode === "real"} onClick={() => setMode("real")} icon={<Cpu className="h-4 w-4" />} title="Real robot" sub="SO-101 via LeRobot teleoperation." />
          </div>
          {mode === "real" && (
            <div className="flex gap-2.5 rounded bg-caution/5 p-3 text-xs text-ink-2 shadow-[0_0_0_1px_rgb(var(--caution)/.3)]">
              <Info className="mt-px h-4 w-4 shrink-0 text-caution" />
              <p>
                Real-robot mode needs the local LeRobot bridge running next to the arm. Every motion command requires your
                explicit approval. The agent can plan, but it cannot move hardware on its own.
              </p>
            </div>
          )}
          <div className="grid grid-cols-[160px_1fr] gap-2.5">
            <label className="block space-y-1.5">
              <span className="block text-xs font-medium text-ink-2">Demo budget</span>
              <Input type="number" min={3} max={500} value={budget} onChange={(e) => setBudget(Math.max(1, Number(e.target.value) || 0))} className="num font-mono" />
            </label>
            <label className="block space-y-1.5">
              <span className="block text-xs font-medium text-ink-2">Project name (optional)</span>
              <Input value={name} onChange={(e) => setName(e.target.value)} placeholder="Derived from the prompt" />
            </label>
          </div>
        </div>
      )}

      {create.error && (
        <div className="mt-3">
          <ErrorState error={create.error} compact />
        </div>
      )}
    </form>
  );
}

function Disclosure({ label, open, onClick }: { label: string; open: boolean; onClick: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-expanded={open}
      className={cn("inline-flex h-8 items-center gap-1 rounded-full px-3 text-xs text-ink-2 hover:text-ink", open ? "bg-rail" : "hover:bg-rail/60")}
    >
      {label}
      <ChevronDown className={cn("h-3.5 w-3.5 transition-transform duration-150", open && "rotate-180")} />
    </button>
  );
}

function ModeCard({
  active,
  onClick,
  icon,
  title,
  sub,
  tag,
}: {
  active: boolean;
  onClick: () => void;
  icon: ReactNode;
  title: string;
  sub: string;
  tag?: string;
}) {
  return (
    <button
      type="button"
      role="radio"
      aria-checked={active}
      onClick={onClick}
      className={cn(
        "rounded-md p-3 text-left transition-[box-shadow,background-color] duration-100",
        active ? "bg-servo/[.06] shadow-[0_0_0_1.5px_rgb(var(--servo))]" : "bg-well shadow-[0_0_0_1px_var(--line)] hover:shadow-[0_0_0_1px_var(--line-strong)]",
      )}
    >
      <div className="flex items-center gap-2 text-sm font-medium">
        <span className={active ? "text-servo" : "text-ink-3"}>{icon}</span>
        {title}
        {tag && <span className="ml-auto text-2xs font-normal text-ink-4">{tag}</span>}
      </div>
      <p className="mt-1 text-xs text-ink-3">{sub}</p>
    </button>
  );
}

export function NewProjectPage() {
  const nav = useNavigate();
  const [created, setCreated] = useState<Project | null>(null);

  return (
    <div className="h-full overflow-y-auto scroll-thin">
      <div className="mx-auto max-w-[880px] px-6 py-10">
        {!created ? (
          <div className="mx-auto max-w-[680px] pt-[6vh]">
            <NewProjectForm onCreated={(p, auto) => (auto ? nav(`/p/${p.id}`) : setCreated(p))} />
          </div>
        ) : (
          <div className="enter space-y-5">
            <div className="flex items-end justify-between gap-4">
              <div>
                <p className="text-xs font-medium text-pass">Project created</p>
                <h1 className="mt-1 text-2xl font-semibold tracking-[-0.015em]">{created.name}</h1>
                <p className="mt-1 text-sm text-ink-3">“{created.task_prompt}”</p>
              </div>
              <div className="flex shrink-0 gap-2">
                <Link to={`/p/${created.id}`}>
                  <Button>Open project</Button>
                </Link>
                <Button variant="primary" onClick={() => nav(`/p/${created.id}/collect`)}>
                  Start collecting <ArrowRight className="h-4 w-4" />
                </Button>
              </div>
            </div>
            <ProjectSpecView project={created} />
          </div>
        )}
      </div>
    </div>
  );
}
