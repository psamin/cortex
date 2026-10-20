import type { MCPServer } from "mcp-use";
import { getJson } from "./backend.js";

// Pipeline-stage metadata for every cortex.* tool, applied in a `tools/list` middleware:
//  - tools are listed in golden-path order, each with _meta {"cortex/stage", "cortex/tier", "cortex/step"};
//  - the MCP URL can narrow the list: ?toolset=core hides the custom-path and step-by-step robot tools,
//    ?project=prj_... advertises only the tools for that project's current stage (plus the always-on reads,
//    and without the step-by-step robot tools).
// The server is stateless, so a plain /mcp URL lists everything (in order). Hidden tools stay callable.

export type Stage = "setup" | "collect" | "train" | "evaluate" | "improve" | "anytime" | "manual";
type Tier = "golden" | "custom" | "advanced";
type Meta = { stage: Stage; tier: Tier; step?: number };

const META: Record<string, Meta> = {
  create_policy_project: { stage: "setup", tier: "golden", step: 1 },
  // anytime: improve_policy starts a collection after a checkpoint exists and says to poll this next
  get_collection_progress: { stage: "anytime", tier: "golden", step: 2 },
  train_from_agent_demonstrations: { stage: "train", tier: "golden", step: 3 },
  launch_training_job: { stage: "train", tier: "golden", step: 5 },
  evaluate_learned_policy: { stage: "evaluate", tier: "golden", step: 7 },
  improve_policy: { stage: "improve", tier: "golden", step: 8 },
  save_physical_skill: { stage: "improve", tier: "golden", step: 9 },
  deploy_physical_skill: { stage: "improve", tier: "golden", step: 10 },
  get_project_state: { stage: "anytime", tier: "golden", step: 6 },
  get_next_actions: { stage: "anytime", tier: "golden" },
  get_job: { stage: "anytime", tier: "golden" },
  // custom scene (intake) path
  start_intake: { stage: "setup", tier: "custom" },
  update_intake: { stage: "setup", tier: "custom" },
  get_intake_status: { stage: "setup", tier: "custom" },
  build_simulation: { stage: "setup", tier: "custom" },
  capture_frame: { stage: "setup", tier: "custom" },
  start_agent_collection: { stage: "collect", tier: "custom" },
  get_demonstration_results: { stage: "collect", tier: "custom" },
  // manual data / training path
  create_project: { stage: "setup", tier: "advanced" },
  record_demonstrations: { stage: "collect", tier: "advanced" },
  import_demonstrations: { stage: "collect", tier: "advanced" },
  list_episodes: { stage: "collect", tier: "advanced" },
  analyze_dataset: { stage: "improve", tier: "advanced" },
  get_failure_clusters: { stage: "improve", tier: "advanced" },
  recommend_demonstrations: { stage: "improve", tier: "advanced" },
  create_dataset_version: { stage: "train", tier: "advanced" },
  propose_training_job: { stage: "train", tier: "custom" }, // AGENTS.md step 4 names it
  register_training_checkpoint: { stage: "train", tier: "advanced" },
  launch_simulation_evaluation: { stage: "evaluate", tier: "advanced" },
  compare_experiments: { stage: "evaluate", tier: "advanced" },
  get_infrastructure: { stage: "anytime", tier: "advanced" },
  set_infrastructure: { stage: "anytime", tier: "advanced" },
  // Tools AGENTS.md's path uses, and their companions (#358). Without an entry a tool counts as
  // {stage: "manual", tier: "advanced"}: ?toolset=core and ?project= would hide it.
  get_session_link: { stage: "anytime", tier: "golden" }, // AGENTS.md step 5
  register_agent: { stage: "anytime", tier: "advanced" },
  log_decision: { stage: "anytime", tier: "advanced" },
  list: { stage: "anytime", tier: "advanced" },
  cancel_job: { stage: "anytime", tier: "advanced" },
  show_training_review: { stage: "anytime", tier: "custom" },
  show_collection: { stage: "anytime", tier: "custom" },
  show_evaluation: { stage: "anytime", tier: "custom" },
  get_policy_weaknesses: { stage: "improve", tier: "custom" }, // AGENTS.md step 8
  propose_rl_sandboxes: { stage: "improve", tier: "custom" },
  accept_rl_sandboxes: { stage: "improve", tier: "custom" },
  get_rl_sandbox_status: { stage: "improve", tier: "custom" },
  stop_rl_sandbox: { stage: "improve", tier: "advanced" },
  close_rl_round: { stage: "improve", tier: "advanced" },
  get_rl_rounds: { stage: "improve", tier: "advanced" },
};
const metaOf = (tool: string): Meta => META[tool.replace(/^cortex\./, "")] ?? { stage: "manual", tier: "advanced" };

const STAGE_ORDER: Stage[] = ["setup", "collect", "train", "evaluate", "improve", "anytime", "manual"];
const TIER_ORDER: Tier[] = ["golden", "custom", "advanced"];
function rank(name: string): number {
  const m = metaOf(name);
  if (m.step !== undefined) return m.step;
  return 100 + TIER_ORDER.indexOf(m.tier) * 100 + STAGE_ORDER.indexOf(m.stage) * 10;
}

/** Which stages a project is in, from its state (episodes -> datasets -> checkpoints -> evaluations). */
async function stagesFor(projectId: string): Promise<Set<Stage>> {
  type State = { episode_counts?: { total?: number; accepted?: number }; dataset_versions?: unknown[]; training_jobs?: { status: string; checkpoint_ids?: string[] }[]; evaluations?: unknown[]; project?: { intake?: unknown; scene?: unknown } };
  const st = await getJson<State>(`/projects/${encodeURIComponent(projectId)}/state`);
  const out = new Set<Stage>(["anytime"]);
  const episodes = st.episode_counts?.total ?? 0;
  const checkpoints = (st.training_jobs ?? []).some((t) => (t.checkpoint_ids ?? []).length > 0);
  if (st.project?.intake && !st.project.scene) out.add("setup");
  if (!checkpoints) out.add("collect");
  if (episodes > 0 && !checkpoints) out.add("train");
  if (checkpoints) out.add("evaluate").add("train");
  if ((st.evaluations ?? []).length > 0) out.add("improve").add("evaluate");
  return out;
}

/** Registers the `tools/list` middleware (ordering, stage _meta, optional ?toolset= / ?project= filters). */
export function registerStages(server: MCPServer) {
  server.use("mcp:tools/list", async (ctx, next) => {
    const tools = await next();
    const toolset = ctx.request?.query("toolset") ?? process.env.CORTEX_MCP_TOOLSET ?? "all";
    const project = ctx.request?.query("project");
    let stages: Set<Stage> | undefined;
    if (project) {
      try {
        stages = await stagesFor(project);
      } catch {
        stages = undefined; // unknown project or backend down: list everything rather than nothing
      }
    }
    return tools
      .filter((t) => {
        if (!t.name.startsWith("cortex.")) return true;
        const m = metaOf(t.name);
        if (toolset === "core" && m.tier !== "golden" && m.tier !== "custom") return false;
        return !stages || stages.has(m.stage);
      })
      .sort((a, b) => rank(a.name) - rank(b.name))
      .map((t) => {
        if (!t.name.startsWith("cortex.")) return t;
        const m = metaOf(t.name);
        return { ...t, _meta: { ...t._meta, "cortex/stage": m.stage, "cortex/tier": m.tier, ...(m.step !== undefined ? { "cortex/golden_path_step": m.step } : {}) } };
      });
  });
}
