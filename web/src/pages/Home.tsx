import { useEffect, useRef, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { ArrowUp, CircleDashed } from "lucide-react";
import { api } from "../api/client";
import { useProjects } from "../api/hooks";
import { ErrorState } from "../components/ui";
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
    <div className="h-full overflow-y-auto scroll-thin">
      <div className="mx-auto flex min-h-full max-w-[720px] flex-col px-6 pb-12 pt-[16vh]">
        <h1 className="text-center text-[40px] font-semibold leading-tight tracking-[-0.03em]">What should your robot learn?</h1>
        <p className="mt-3 text-center text-[15px] text-ink-3">
          Describe a skill. Cortex collects demonstrations in simulation, trains a policy, tests it, and keeps improving it.
        </p>
        <form
          onSubmit={(e) => {
            e.preventDefault();
            submit();
          }}
          className="mt-9 flex items-end gap-2 rounded-2xl bg-plate p-2.5 pl-5 shadow-[0_0_0_1px_var(--line-strong),0_24px_48px_-24px_rgba(0,0,0,.5)] focus-within:shadow-[0_0_0_1px_rgb(var(--servo)/.6),0_24px_48px_-24px_rgba(0,0,0,.5)]"
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
          <button type="submit" disabled={create.isPending} className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl bg-servo text-servo-ink transition-transform active:scale-95 disabled:opacity-60" aria-label="Start">
            {create.isPending ? <CircleDashed className="h-5 w-5 animate-spin" /> : <ArrowUp className="h-5 w-5" />}
          </button>
        </form>
        <div className="mt-3 flex flex-wrap justify-center gap-1.5">
          {EXAMPLES.map((x) => (
            <button key={x} type="button" onClick={() => setPrompt(x)} className="rounded-full px-3 py-1.5 text-xs text-ink-3 shadow-[0_0_0_1px_var(--line)] hover:text-ink">
              {x}
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
              <div className="mb-2 text-xs text-ink-4">Recent</div>
              <ul className="divide-y divide-[var(--line-soft)]">
                {recent.map((p) => (
                  <li key={p.id}>
                    <Link to={`/p/${p.id}`} className="flex items-baseline gap-3 rounded-md px-2 py-2.5 hover:bg-rail/60">
                      <span className="truncate text-sm text-ink-2">{p.name}</span>
                      <span className="hidden min-w-0 flex-1 truncate text-xs text-ink-4 sm:block">{p.task_prompt}</span>
                      <span className="ml-auto shrink-0 text-xs text-ink-4">{ago(p.created_at)}</span>
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
