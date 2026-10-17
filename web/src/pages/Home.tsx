import { Link, useNavigate } from "react-router-dom";
import { Plus } from "lucide-react";
import { useProjects } from "../api/hooks";
import type { Project } from "../api/types";
import { Button, ErrorState, Skeleton } from "../components/ui";
import { NewProjectForm } from "./NewProject";
import { ago } from "../lib/format";

export function HomePage() {
  const q = useProjects();
  const nav = useNavigate();

  return (
    <div className="h-full overflow-y-auto scroll-thin">
      <div className="mx-auto max-w-[880px] px-6 py-10">
        {q.isLoading && (
          <div className="space-y-2">
            <Skeleton className="h-8 w-48" />
            <Skeleton className="h-16 w-full" />
            <Skeleton className="h-16 w-full" />
          </div>
        )}
        {q.error && (
          <div className="space-y-8">
            <ErrorState error={q.error} onRetry={() => q.refetch()} />
            <div className="pointer-events-none opacity-50" aria-hidden>
              <NewProjectForm onCreated={() => {}} />
            </div>
          </div>
        )}
        {q.data && q.data.length === 0 && (
          <div className="mx-auto max-w-[640px]">
            <NewProjectForm onCreated={(p) => nav(`/p/${p.id}`)} />
          </div>
        )}
        {q.data && q.data.length > 0 && <ProjectList projects={q.data} />}
      </div>
    </div>
  );
}

function ProjectList({ projects }: { projects: Project[] }) {
  const sorted = [...projects].sort((a, b) => (a.created_at < b.created_at ? 1 : -1));
  return (
    <div className="space-y-4">
      <div className="flex items-end justify-between">
        <div>
          <h1 className="text-2xl font-semibold tracking-[-0.015em]">Projects</h1>
          <p className="mt-1 text-sm text-ink-3">Each project is one skill you are teaching.</p>
        </div>
        <Link to="/new">
          <Button variant="primary">
            <Plus className="h-4 w-4" /> New project
          </Button>
        </Link>
      </div>
      <ul className="overflow-hidden rounded-md bg-plate shadow-[0_0_0_1px_var(--line)]">
        {sorted.map((p, i) => {
          const collected = p.plan.cells.reduce((a, c) => a + c.collected, 0);
          const frac = p.plan.budget_total ? collected / p.plan.budget_total : 0;
          return (
            <li key={p.id} className={i > 0 ? "border-t border-line-soft" : undefined}>
              <Link to={`/p/${p.id}`} className="grid grid-cols-[1fr_140px_90px_80px] items-center gap-4 px-4 py-3 hover:bg-rail/50">
                <div className="min-w-0">
                  <div className="truncate text-sm font-medium">{p.name}</div>
                  <div className="truncate text-xs text-ink-3">{p.task_prompt}</div>
                </div>
                <div className="space-y-1">
                  <div className="num font-mono text-2xs text-ink-3">
                    {collected}/{p.plan.budget_total} demos
                  </div>
                  <div className="h-1 overflow-hidden rounded-full bg-rail">
                    <div className="h-full bg-servo" style={{ width: `${Math.min(1, frac) * 100}%` }} />
                  </div>
                </div>
                <span className="text-2xs font-medium uppercase tracking-wide text-ink-3">{p.mode === "sim" ? "sim" : "real"}</span>
                <span className="text-right text-2xs text-ink-4">{ago(p.created_at)}</span>
              </Link>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
