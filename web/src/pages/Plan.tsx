import { useNavigate, useParams } from "react-router-dom";
import { ArrowRight, Bot, ShieldAlert } from "lucide-react";
import { useNextActions, useProjectState } from "../api/hooks";
import { conditionKey, type NextAction, type Project } from "../api/types";
import { Badge, Button, EmptyState, Panel, QueryState, SectionLabel, Skeleton } from "../components/ui";
import { WorkcellGrid } from "../components/Workcell";
import { useAgent } from "../layout/ProjectLayout";
import { AutopilotHero } from "./Autopilot";
import { LineagePreview } from "./Lineage";
import { cn, humanize } from "../lib/format";
import { ScenePanel } from "../sim3d/ScenePanel";

export function PlanPage() {
  const { pid = "" } = useParams();
  const state = useProjectState(pid);
  return (
    <div className="mx-auto max-w-[1100px] space-y-4 px-5 py-5">
      <QueryState
        q={state}
        skeleton={
          <div className="space-y-3">
            <Skeleton className="h-8 w-72" />
            <Skeleton className="h-40 w-full" />
          </div>
        }
      >
        {(s) => (
          <>
            <div>
              <h1 className="text-2xl font-semibold tracking-[-0.015em]">{s.project.name}</h1>
              <p className="mt-1 text-sm text-ink-3">“{s.project.task_prompt}”</p>
            </div>
            <AutopilotHero project={s.project} />
            <ScenePanel project={s.project} />
            <LineagePreview pid={pid} />
            <NextActionsPanel pid={pid} />
            <ProjectSpecView project={s.project} />
          </>
        )}
      </QueryState>
    </div>
  );
}

export function ProjectSpecView({ project }: { project: Project }) {
  const { spec, plan } = project;
  const byKey = new Map(plan.cells.map((c) => [conditionKey(c.condition), c]));
  const targetTotal = plan.cells.reduce((a, c) => a + c.target, 0);
  return (
    <div className="grid grid-cols-1 gap-4 min-[1400px]:grid-cols-[minmax(0,1fr)_minmax(0,1.1fr)]">
      <Panel title="Task spec" meta={`${spec.robot} · ${spec.episode_seconds}s @ ${spec.fps} fps`}>
        <dl className="space-y-3 text-sm">
          <div>
            <SectionLabel>Goal</SectionLabel>
            <dd className="mt-1 text-ink">{spec.goal}</dd>
          </div>
          <div>
            <SectionLabel>Objects</SectionLabel>
            <dd className="mt-1 flex flex-wrap gap-1">
              {spec.objects.map((o) => (
                <Badge key={o}>{o}</Badge>
              ))}
            </dd>
          </div>
          <div>
            <SectionLabel>Success criteria</SectionLabel>
            <ol className="mt-1.5 space-y-2">
              {spec.success_criteria.map((c, i) => (
                <li key={c.id} className="grid grid-cols-[20px_1fr] gap-1">
                  <span className="num font-mono text-xs text-ink-4">{i + 1}.</span>
                  <div>
                    <div className="text-ink">{c.description}</div>
                    <div className="mt-0.5 font-mono text-2xs text-ink-3">check: {c.measurable}</div>
                  </div>
                </li>
              ))}
            </ol>
          </div>
          <div>
            <SectionLabel>Condition axes</SectionLabel>
            <dd className="mt-1 space-y-1">
              {Object.entries(spec.condition_axes).map(([axis, vals]) => (
                <div key={axis} className="flex gap-2 text-xs">
                  <span className="w-24 shrink-0 font-mono text-ink-3">{axis}</span>
                  <span className="text-ink-2">{vals.join(" · ")}</span>
                </div>
              ))}
            </dd>
          </div>
        </dl>
      </Panel>

      <Panel title="Collection plan" meta={`${targetTotal} targeted · budget ${plan.budget_total}`}>
        <WorkcellGrid
          renderCell={(c) => {
            const cell = byKey.get(conditionKey(c));
            if (!cell) return <div className="px-2.5 py-3 text-2xs text-ink-4">not planned</div>;
            const done = cell.collected >= cell.target && cell.target > 0;
            return (
              <div className="px-2.5 py-2" title={cell.reason || undefined}>
                <div className="flex items-baseline gap-1">
                  <span className={cn("num text-xl font-semibold", done ? "text-pass" : "text-ink")}>{cell.collected}</span>
                  <span className="num text-xs text-ink-3">/ {cell.target}</span>
                </div>
                {cell.reason && <div className="mt-0.5 line-clamp-2 text-2xs leading-snug text-ink-3">{cell.reason}</div>}
              </div>
            );
          }}
        />
        <p className="mt-3 text-2xs text-ink-3">Collected / target demos per condition. Columns are where the cube starts, rows are the lighting.</p>
      </Panel>
    </div>
  );
}

/** Where a next-action name most likely lives in the studio (by name heuristics). */
function routeForAction(pid: string, action: string): string | null {
  const a = action.toLowerCase();
  if (/record|collect|demo/.test(a)) return `/p/${pid}/collect`;
  if (/analy|diagnos|quality/.test(a)) return `/p/${pid}/analyze`;
  if (/dataset|augment/.test(a)) return `/p/${pid}/improve?tab=datasets`;
  if (/train|checkpoint/.test(a)) return `/p/${pid}/improve?tab=training`;
  if (/eval|compar/.test(a)) return `/p/${pid}/improve?tab=evaluations`;
  if (/recommend/.test(a)) return `/p/${pid}/improve?tab=recommendations`;
  return null;
}

export function NextActionsPanel({ pid }: { pid: string }) {
  const q = useNextActions(pid);
  return (
    <Panel title="Next actions" meta="from the orchestrator" bodyClassName="p-0">
      <QueryState q={q} skeleton={<div className="p-3.5"><Skeleton className="h-10 w-full" /></div>}>
        {(actions) =>
          actions.length === 0 ? (
            <EmptyState title="No next actions right now" />
          ) : (
            <ul>
              {actions.map((a, i) => (
                <NextActionRow key={`${a.action}-${i}`} a={a} pid={pid} first={i === 0} />
              ))}
            </ul>
          )
        }
      </QueryState>
    </Panel>
  );
}

function NextActionRow({ a, pid, first }: { a: NextAction; pid: string; first: boolean }) {
  const nav = useNavigate();
  const { ask } = useAgent();
  const to = routeForAction(pid, a.action);
  const args = Object.keys(a.args).length ? JSON.stringify(a.args) : "";
  return (
    <li className={cn("flex items-start gap-3 px-3.5 py-2.5", !first && "border-t border-line-soft")}>
      <span className={cn("mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full", first ? "bg-servo" : "bg-ink-4")} />
      <div className="min-w-0 flex-1">
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-sm font-medium capitalize text-ink">{humanize(a.action)}</span>
          {a.requires_approval && (
            <Badge tone="servo" icon={<ShieldAlert className="h-3 w-3" />}>
              needs approval
            </Badge>
          )}
        </div>
        <p className="mt-0.5 text-xs text-ink-3">{a.reason}</p>
        {args && <p className="mt-0.5 truncate font-mono text-2xs text-ink-4">{args}</p>}
      </div>
      <div className="flex shrink-0 gap-1">
        <Button size="sm" variant="ghost" onClick={() => ask(`Do the next action "${a.action}"${args ? ` with ${args}` : ""}. Reason given: ${a.reason}`)}>
          <Bot className="h-3.5 w-3.5" /> Ask agent
        </Button>
        {to && (
          <Button size="sm" onClick={() => nav(to)}>
            Open <ArrowRight className="h-3.5 w-3.5" />
          </Button>
        )}
      </div>
    </li>
  );
}
