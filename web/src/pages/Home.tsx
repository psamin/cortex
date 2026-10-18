import { useEffect, useRef, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { ArrowUp, CircleDashed } from "lucide-react";
import { api } from "../api/client";
import { useProjects } from "../api/hooks";
import { ErrorState } from "../components/ui";
import { DashedOrbit, IsoIcon } from "../sketch";
import { ago } from "../lib/format";

const PLACEHOLDER = "Teach a robot to pick and place a cube.";
const EXAMPLES = ["Pick up the cube and place it in the bowl", "Push the cube to the left edge of the table", "Stack the cube on top of the box"];

/** Landing: one prompt. Submitting creates the project and hands the prompt to the agent as the first message. */
export function HomePage() {
  const nav = useNavigate();
  const qc = useQueryClient();
  const projects = useProjects();
  const [prompt, setPrompt] = useState("");
  const ref = useRef<HTMLTextAreaElement>(null);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "0px";
    el.style.height = `${Math.min(220, Math.max(56, el.scrollHeight))}px`;
  }, [prompt]);

  const create = useMutation({
    mutationFn: (text: string) => api.createProject({ task_prompt: text }),
    onSuccess: (p, text) => {
      qc.invalidateQueries({ queryKey: ["projects"] });
      nav(`/p/${p.id}`, { state: { autoSend: text } });
    },
  });
  const submit = (t = prompt) => {
    const text = t.trim() || PLACEHOLDER;
    if (!create.isPending) create.mutate(text);
  };
  const recent = [...(projects.data ?? [])].sort((a, b) => (a.created_at < b.created_at ? 1 : -1)).slice(0, 6);

  return (
    <div className="relative h-full overflow-y-auto overflow-x-hidden scroll-thin">
      <DashedOrbit r={260} className="pointer-events-none absolute -right-64 -top-24 hidden opacity-60 xl:block" />
      <div className="relative mx-auto flex min-h-full max-w-[760px] flex-col px-6 pb-12 pt-[12vh]">
        <div className="kn-label kn-label-rule">
          <span>
            <span className="text-ink">FIG.00</span>&nbsp;&nbsp;New skill · Simulation first
          </span>
        </div>
        <h1 className="kn-display mt-6 text-[clamp(2.4rem,5vw,3.6rem)]">
          What should your
          <br />
          <span className="kn-second">robot learn?</span>
        </h1>
        <p className="mt-5 max-w-[34rem] text-[15px] leading-[1.65] text-ink-2">
          Describe a skill. Cortex collects demonstrations in simulation, trains a policy, tests it, and keeps improving it.
        </p>
        <form
          onSubmit={(e) => {
            e.preventDefault();
            submit();
          }}
          className="sk-depth sk-depth-lg relative mt-9 flex items-end gap-2 rounded border border-line-strong bg-plate p-2.5 pl-5 transition-colors focus-within:border-ink"
        >
          <textarea
            ref={ref}
            autoFocus
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
                e.preventDefault();
                submit();
              }
            }}
            placeholder={PLACEHOLDER}
            className="flex-1 resize-none bg-transparent py-2.5 text-lg leading-7 text-ink placeholder:text-ink-4 focus:outline-none"
            aria-label="What should your robot learn?"
          />
          <button type="submit" disabled={create.isPending} className="sk-pill sk-pill-primary flex h-11 w-14 shrink-0 items-center justify-center disabled:opacity-60" aria-label="Start">
            {create.isPending ? <CircleDashed className="h-5 w-5 animate-spin" /> : <ArrowUp className="h-5 w-5" />}
          </button>
        </form>
        <div className="mt-5 flex flex-wrap gap-2">
          {EXAMPLES.map((x) => (
            <button key={x} type="button" onClick={() => setPrompt(x)} className="group inline-flex items-center gap-1.5 rounded-full border border-line-strong bg-plate px-3.5 py-1.5 text-xs text-ink-2 transition-colors hover:border-ink hover:text-ink">
              {x}
              <span aria-hidden className="text-ink-4 transition-transform group-hover:translate-x-0.5 group-hover:text-ink">→</span>
            </button>
          ))}
        </div>
        {create.error && (
          <div className="mt-4">
            <ErrorState error={create.error} compact />
          </div>
        )}

        <div className="mt-auto pt-16">
          {projects.error ? (
            <ErrorState error={projects.error} compact onRetry={() => projects.refetch()} />
          ) : recent.length > 0 ? (
            <>
              <div className="kn-label kn-label-rule mb-3">
                <span>Recent projects</span>
              </div>
              <ul className="divide-y divide-[var(--line)] border-y border-line">
                {recent.map((p) => (
                  <li key={p.id}>
                    <Link to={`/p/${p.id}`} className="group flex items-center gap-3 px-1 py-3 hover:bg-rail/50">
                      <IsoIcon name="cube" size={16} />
                      <span className="truncate text-sm text-ink">{p.name}</span>
                      <span className="hidden min-w-0 flex-1 truncate text-xs text-ink-3 sm:block">{p.task_prompt}</span>
                      <span className="ml-auto shrink-0 font-mono text-[10px] uppercase tracking-[0.14em] text-ink-3">{ago(p.created_at)}</span>
                      <span aria-hidden className="text-ink-4 transition-transform group-hover:translate-x-0.5 group-hover:text-ink">→</span>
                    </Link>
                  </li>
                ))}
              </ul>
            </>
          ) : null}
        </div>
      </div>
    </div>
  );
}
