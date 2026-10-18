import { Link } from "react-router-dom";
import { Play } from "lucide-react";
import type { AgentCall } from "../agents/api";
import { GlowEdge } from "../components/ui/spotlight-card";
import { ago, cn } from "../lib/format";
import type { IsoIconName } from "../sketch";

/** The end of the agent's golden path (#310): it saves the trained checkpoint as a Physical Skill
 *  (cortex.save_physical_skill), then deploys it in the deployment simulator (cortex.deploy_physical_skill). */

type StageState = "waiting" | "running" | "done" | "attention" | "failed";
export interface SkillStage {
  name: string;
  icon: IsoIconName;
  state: StageState;
  note?: string;
}

export interface SavedSkill {
  at: string;
  skillId: string;
  name: string;
  score?: string;
}

export interface Deployment {
  at: string;
  skillId: string;
  name: string;
  /** false while the GPU comes up: the page still opens, and plays the run once the GPU is live */
  deployed: boolean;
  url: string;
  runId?: string;
  seed?: number;
}

const parse = (s?: string): Record<string, unknown> => {
  try {
    const v = JSON.parse(s ?? "");
    return v && typeof v === "object" ? (v as Record<string, unknown>) : {};
  } catch {
    return {};
  }
};
const str = (v: unknown) => (typeof v === "string" && v ? v : undefined);
const last = <T,>(xs: T[]) => xs[xs.length - 1];

export function savedSkills(calls: AgentCall[]): SavedSkill[] {
  return calls
    .filter((c) => c.tool === "save_physical_skill" && c.status === "ok")
    .map((c) => {
      const r = parse(c.result_preview);
      const score = r.score as { label?: string } | undefined;
      return { at: c.ended_at ?? c.started_at, skillId: str(r.skill_id) ?? "", name: str(r.name) ?? String(c.args.name ?? "skill"), score: str(score?.label) };
    });
}

/** Deployments with a page to open (a call cut off before its result has none). */
export function deployments(calls: AgentCall[]): Deployment[] {
  return calls
    .filter((c) => c.tool === "deploy_physical_skill" && c.status === "ok")
    .flatMap((c) => {
      const r = parse(c.result_preview);
      const url = str(r.deployment_url);
      if (!url) return [];
      return [{
        at: c.ended_at ?? c.started_at,
        skillId: str(r.skill_id) ?? String(c.args.skill_id ?? ""),
        name: str(r.skill_name) ?? "skill",
        deployed: r.deployed === true,
        url,
        runId: str(r.run_id),
        seed: typeof r.seed === "number" ? r.seed : undefined,
      }];
    });
}

/** The two stages after Improve: Save skill, then Deploy. */
export function skillStages(calls: AgentCall[]): SkillStage[] {
  const saved = last(savedSkills(calls));
  const dep = last(deployments(calls));
  const running = (tool: string) => calls.some((c) => c.tool === tool && c.status === "running");
  return [
    {
      name: "Save skill",
      icon: "check",
      state: saved ? "done" : running("save_physical_skill") ? "running" : "waiting",
      note: saved ? saved.name : undefined,
    },
    {
      name: "Deploy",
      icon: "robot",
      state: running("deploy_physical_skill") || (dep && !dep.deployed) ? "running" : dep ? "done" : "waiting",
      note: dep ? (dep.deployed ? "live run ready" : "GPU starting") : undefined,
    },
  ];
}

/** Feed milestones: "Saved physical skill 'Jenga v1' (62% success …)", "Deploying physical skill 'Jenga v1'…". */
export function skillEvents(calls: AgentCall[]): { at: string; icon: IsoIconName; text: string; tone?: "ok" | "live" }[] {
  return [
    ...savedSkills(calls).map((s) => ({ at: s.at, icon: "check" as const, tone: "ok" as const, text: `Saved physical skill '${s.name}'${s.score ? ` (${s.score})` : ""}` })),
    ...deployments(calls).map((d) => ({
      at: d.at,
      icon: "robot" as const,
      tone: "live" as const,
      text: d.deployed ? `Deploying physical skill '${d.name}'${d.seed != null ? ` on unseen layout ${d.seed}` : ""}…` : `Deploying physical skill '${d.name}': bringing up a GPU…`,
    })),
  ];
}

/** The studio route a deployment URL points at (the URL is absolute, built from the backend's studio address). */
export function deploymentPath(url: string): string {
  try {
    const u = new URL(url, "http://studio.local");
    return `${u.pathname}${u.search}`;
  } catch {
    return url;
  }
}

/** A rectangle with a play button: opens the deployment simulator on this skill's live run. */
export function DeployCard({ d }: { d: Deployment }) {
  return (
    <Link
      to={deploymentPath(d.url)}
      className="sk-depth group relative flex items-center gap-4 rounded border border-line-strong bg-plate px-4 py-3.5 transition-colors hover:border-ink"
    >
      <GlowEdge />
      <span className="flex h-11 w-11 shrink-0 items-center justify-center rounded-full bg-ink text-bench transition-transform group-hover:scale-105">
        <Play className="ml-0.5 h-4 w-4" fill="currentColor" />
      </span>
      <span className="min-w-0 flex-1">
        <span className="block text-[15px] font-medium text-ink">View deployment</span>
        <span className="block truncate text-xs text-ink-3">
          {d.name} · {d.deployed ? `live run in the deployment simulator${d.seed != null ? ` on unseen layout ${d.seed}` : ""}` : "the GPU is coming up; the run plays once it's live"}
        </span>
      </span>
      <span className={cn("shrink-0 font-mono text-[10.5px] uppercase tracking-[0.12em]", d.deployed ? "text-servo" : "text-ink-3")}>{ago(d.at)}</span>
    </Link>
  );
}
