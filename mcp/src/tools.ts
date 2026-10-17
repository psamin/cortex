import type { MCPServer, ToolAnnotations } from "mcp-use";
import { z } from "zod";
import { callTool } from "./backend.js";

// Input schemas mirror the backend registry (GET /api/tools). Where the
// registry says plain `string` but the domain model only accepts a fixed set
// (cube_region, lighting, policy_type, augmentation_preset), the
// enum is spelled out here so agents cannot pick a value the backend rejects.
// Optional fields are left unset so the backend applies its own defaults.

const projectId = z.string().describe("Cortex project id (prj_...), from cortex.create_project");

const waitS = (max: number, typical: string) =>
  z.number().int().min(0).max(max).optional().describe(
    `Seconds to block waiting for the background job to finish (0-${max}, default 0 = return immediately with the job id). ` +
      `${typical} Many MCP clients time out a tool call after about 60 s, so prefer wait_s <= 45 and then poll with cortex.get_job.`,
  );

const READ: ToolAnnotations = { readOnlyHint: true, destructiveHint: false, idempotentHint: true, openWorldHint: false };
const WRITE: ToolAnnotations = { readOnlyHint: false, destructiveHint: false, idempotentHint: false, openWorldHint: false };

type Def = {
  name: string;
  title: string;
  description: string;
  inputSchema: z.ZodObject;
  annotations: ToolAnnotations;
};

const TOOLS: Def[] = [
  {
    name: "create_project",
    title: "Create project",
    description:
      "Start a robot-learning project from a plain-language task (SO-101 arm, MuJoCo tabletop sim). " +
      "Returns the project id, a task spec with measurable success criteria, and an initial demonstration plan " +
      "(condition cells with target counts). Use the returned project_id for every other cortex.* tool.",
    inputSchema: z.object({
      task_prompt: z.string().describe("What the robot should learn, in plain language, e.g. 'pick up the cube and place it in the bowl'"),
      mode: z.enum(["sim", "real"]).optional().describe(
        "'sim' (default) or 'real'. Agents can only record demonstrations in sim; real-robot recording needs a human operator.",
      ),
      demo_budget: z.number().int().optional().describe("Total demonstrations to plan for (default 24)"),
    }),
    annotations: WRITE,
  },
  {
    name: "get_project_state",
    title: "Get project state",
    description:
      "Snapshot of a project: task spec and plan, episode counts (recorded/accepted/rejected), latest analysis id, " +
      "dataset version ids, training jobs (with status and checkpoint ids), evaluations (per-branch successes), " +
      "open recommendations and suggested next actions. Use it to poll evaluations, which have no job id.",
    inputSchema: z.object({ project_id: projectId }),
    annotations: READ,
  },
  {
    name: "list_episodes",
    title: "List episodes",
    description:
      "List recorded demonstrations with id, condition (cube_region/lighting), seed, success, status, frame count, " +
      "quality score and quality flags. Returns ids and storage keys only, never video data.",
    inputSchema: z.object({
      project_id: projectId,
      status: z.enum(["recorded", "accepted", "rejected"]).nullish().describe("Only episodes with this status (default: all)"),
    }),
    annotations: READ,
  },
  {
    name: "record_demonstrations",
    title: "Record demonstrations (sim)",
    description:
      "Record scripted-expert demonstrations in the MuJoCo simulator for the given conditions. Simulation projects only: " +
      "on a 'real' project the backend refuses and a human must record with the physical robot. " +
      "Starts a background job; pass wait_s to block, otherwise poll cortex.get_job with the returned job id.",
    inputSchema: z.object({
      project_id: projectId,
      conditions: z.array(z.object({
        cube_region: z.enum(["left", "center", "right"]).optional().describe("Where the cube starts (default 'center')"),
        lighting: z.enum(["bright", "normal", "dim"]).optional().describe("Scene lighting (default 'normal')"),
        count: z.number().int().min(1).max(20).optional().describe("Demonstrations to record for this condition (1-20, default 1)"),
      })).describe(
        "Condition cells to record, e.g. [{cube_region:'left', lighting:'normal', count:2}]. " +
          "cortex.create_project's plan and cortex.recommend_demonstrations suggest which cells to fill.",
      ),
      auto_accept: z.boolean().optional().describe("Accept demos that succeed and pass quality checks (default true)"),
      recommendation_id: z.string().nullish().describe("If this recording fulfils a recommendation (rec_...), its id, so it is marked accepted"),
      wait_s: waitS(600, "Recording takes roughly 5 s per demonstration."),
    }),
    annotations: WRITE,
  },
  {
    name: "analyze_dataset",
    title: "Analyze dataset",
    description:
      "Run quality checks, telemetry failure analysis and (optionally) VLM failure hypotheses over the project's episodes, " +
      "and rebuild the failure clusters. Background job: pass wait_s or poll cortex.get_job. " +
      "When it finishes, the summary is the analysis summary; read clusters with cortex.get_failure_clusters.",
    inputSchema: z.object({
      project_id: projectId,
      use_vlm: z.boolean().optional().describe("Also ask a vision-language model for failure hypotheses (default true). VLM output is a hypothesis, not evidence."),
      wait_s: waitS(600, "Without the VLM it finishes in about a second; VLM review takes longer."),
    }),
    annotations: WRITE,
  },
  {
    name: "get_failure_clusters",
    title: "Get failure clusters",
    description:
      "Current failure clusters: category, source ('quality' | 'evaluation' | 'vlm'), affected episode ids and conditions, " +
      "evidence and uncertainty (0-1). Clusters with source='vlm' are UNVERIFIED hypotheses unless verified_by_human is true: " +
      "report them as hypotheses, never as established causes.",
    inputSchema: z.object({ project_id: projectId }),
    annotations: READ,
  },
  {
    name: "recommend_demonstrations",
    title: "Recommend demonstrations",
    description:
      "Rank what to do next for data collection: which demonstrations to collect (condition + count) and any augment/train/evaluate " +
      "suggestions, each with evidence and a support level ('evidence_backed' or 'hypothesis'). " +
      "Replaces any earlier proposed recommendations. Act on a 'collect' recommendation with cortex.record_demonstrations, passing its recommendation_id.",
    inputSchema: z.object({
      project_id: projectId,
      budget: z.number().int().nullish().describe("Demonstrations you can still afford to collect (default: the plan budget + 12 minus accepted episodes)"),
    }),
    annotations: WRITE,
  },
  {
    name: "create_dataset_version",
    title: "Create dataset version",
    description:
      "Snapshot all accepted episodes as a versioned LeRobotDataset v3 with lineage, a train/eval split and optional image-only " +
      "augmentation. Background job: the dataset version id appears in the job's result_ref (and in ref_ids once done). " +
      "Needs at least 10 accepted episodes before training makes sense.",
    inputSchema: z.object({
      project_id: projectId,
      augmentation_preset: z.enum(["lighting_robustness"]).nullish().describe(
        "Named augmentation set. 'lighting_robustness' adds darker, brighter, warmer and shadowed copies of the train episodes.",
      ),
      augmentations: z.array(z.object({
        type: z.enum(["brightness", "contrast", "color_temperature", "shadow"]),
        params: z.record(z.string(), z.number()).describe("e.g. {factor: 0.55} for brightness, {shift: 0.6} for color_temperature, {strength: 0.45} for shadow"),
      })).optional().describe("Explicit image augmentations, added to the preset's (default none)"),
      eval_fraction: z.number().optional().describe("Fraction of source episodes held out for eval (default 0.2)"),
      name: z.string().nullish().describe("Optional human-readable name for the version"),
      wait_s: waitS(600, "11 episodes with the lighting preset took about 20 s."),
    }),
    annotations: WRITE,
  },
  {
    name: "propose_training_job",
    title: "Propose training job",
    description:
      "Propose an ACT or SmolVLA training job on a dataset version, with its config, rationale and GPU-hours/cost estimate. " +
      "This does NOT start training: the proposal must be approved by a human in the Cortex studio before cortex.launch_training_job " +
      "will run it. Show the human the estimate and ask them to approve.",
    inputSchema: z.object({
      project_id: projectId,
      dataset_version_id: z.string().describe("Dataset version id (ds_...), from cortex.create_dataset_version or get_project_state"),
      policy_type: z.enum(["act", "smolvla"]).optional().describe("'act' (default; fast, single-task) or 'smolvla' (language-conditioned, ~5x slower per step)"),
      steps: z.number().int().nullish().describe("Training steps (default 20000)"),
    }),
    annotations: WRITE,
  },
  {
    name: "launch_training_job",
    title: "Launch training job (needs human approval)",
    description:
      "Launch a proposed training job. It only runs if a human has already approved the job in the Cortex studio, or if you were " +
      "given the operator's approval_token. Otherwise nothing is launched and the result says APPROVAL REQUIRED: stop, ask the human " +
      "to approve it in the studio, and call this again after they confirm. Never guess or invent an approval_token.",
    inputSchema: z.object({
      training_job_id: z.string().describe("Training job id (tj_...), from cortex.propose_training_job"),
      approval_token: z.string().nullish().describe("Operator approval token, only if the human gave you one; otherwise omit and have them approve in the studio"),
    }),
    annotations: { readOnlyHint: false, destructiveHint: false, idempotentHint: true, openWorldHint: true },
  },
  {
    name: "launch_simulation_evaluation",
    title: "Launch simulation evaluation",
    description:
      "Evaluate a policy in three parallel MuJoCo branches (baseline, shifted object, dim lighting), recording seeds, successes, " +
      "rewards and video storage keys. Use policy='scripted_expert' for a reference ceiling, or a trained checkpoint id (ck_...) " +
      "from get_project_state's training_jobs[].checkpoint_ids. Returns the evaluation id (ev_...) and its job (job.id): if it is still running " +
      "when wait_s runs out, poll cortex.get_job with job.id. " +
      "get_next_actions marks evaluations requires_approval: confirm with the human first unless they already asked you to run it.",
    inputSchema: z.object({
      project_id: projectId,
      policy: z.string().optional().describe("'scripted_expert' (default) or a checkpoint id (ck_...)"),
      episodes_per_branch: z.number().int().min(1).max(50).optional().describe("Episodes per branch (1-50, default 10)"),
      base_seed: z.number().int().optional().describe("First seed; branch episodes use base_seed, base_seed+1, ... (default 1000). Reuse it to compare policies on identical seeds."),
      wait_s: waitS(900, "A scripted-expert run took about 30 s at 2 episodes per branch and about 2 min at 10."),
    }),
    annotations: WRITE,
  },
  {
    name: "compare_experiments",
    title: "Compare experiments",
    description:
      "Branch-by-branch comparison of evaluation runs with success-rate deltas and p-values. " +
      "Defaults to the project's evaluations when evaluation_ids is omitted.",
    inputSchema: z.object({
      project_id: projectId,
      evaluation_ids: z.array(z.string()).nullish().describe("Evaluation ids (ev_...) to compare (default: the project's runs)"),
    }),
    annotations: READ,
  },
  {
    name: "get_next_actions",
    title: "Get next actions",
    description:
      "What to do next for this project, and why. Each item has an action, reason, requires_approval and args. " +
      "Most actions map to a cortex.<action> tool called with those args. 'review_episodes' and 'approve_training_job' are human " +
      "steps in the Cortex studio: ask the human. When requires_approval is true, confirm with the human before calling the tool.",
    inputSchema: z.object({ project_id: projectId }),
    annotations: READ,
  },
  {
    name: "get_job",
    title: "Get job",
    description:
      "Status (queued/running/succeeded/failed/cancelled), progress (0-1), message and result_ref of a background job " +
      "(recording, analysis, dataset build, evaluation, training). Use it to poll after a call returned before the job finished.",
    inputSchema: z.object({ job_id: z.string().describe("Job id (job_...)") }),
    annotations: READ,
  },

  // ---- autonomous demonstrations (an agent VLM plans, validated IK executes in MuJoCo)
  {
    name: "create_policy_project",
    title: "Create policy project",
    description:
      "One call to start training a manipulation policy: creates the project and (by default) starts autonomous " +
      "demonstration collection, where Cortex's agent planner (our self-hosted Qwen3-VL, or GPT-6 Astra) plans from rendered camera images and a validated IK controller executes " +
      "in MuJoCo. Only simulator-verified successes become LeRobot demonstrations. Returns project_id, collection_id and job_id. " +
      "Paid planner spend is capped by the server's standing budget; you can lower it but never raise it. If the planner is not " +
      "configured, collection_not_started explains why (Cortex never substitutes another model).",
    inputSchema: z.object({
      task_prompt: z.string().describe("e.g. 'Train an SO-101 policy to pick up a cube and place it in a bowl'"),
      num_demos: z.number().int().min(1).max(50).optional().describe("Autonomous attempts to run (default 20)"),
      start_collection: z.boolean().optional().describe("Start agent collection right away (default true)"),
      diversity: z.enum(["plan", "medium", "high"]).optional().describe(
        "Scene variety: plan = the project's condition grid; medium = + jittered cube positions and lighting; high = + moved goal box",
      ),
      model_budget_usd: z.number().positive().optional().describe("Planner spend cap in USD (may only lower the default)"),
    }),
    annotations: WRITE,
  },
  {
    name: "start_agent_collection",
    title: "Start agent collection",
    description:
      "Generate demonstrations autonomously in fresh simulator seeds. collector='agent' (default): the agent planner (self-hosted VLM or GPT-6 Astra) chooses actions " +
      "through validated tools (observe, point at the cube, move, grip, verify, place); collector='oracle': the scripted IK " +
      "expert on the same kind of scenes (for A/B). Pass conditions to target weak cells, e.g. [{cube_region:'left', " +
      "lighting:'normal'}]. Returns collection_id and job_id; poll with cortex.get_collection_progress. Each attempt takes " +
      "~5 s of sim plus model latency. A 403 means the project's planner budget is exhausted: ask the human to raise it.",
    inputSchema: z.object({
      project_id: projectId,
      num_demos: z.number().int().min(1).max(50).optional().describe("Attempts to run (default 6)"),
      diversity: z.enum(["plan", "medium", "high"]).optional(),
      conditions: z.array(z.object({
        cube_region: z.enum(["left", "center", "right"]),
        lighting: z.enum(["dim", "normal", "bright"]).optional(),
      })).nullish().describe("Restrict scenes to these cells (targeted collection)"),
      collector: z.enum(["agent", "oracle"]).optional().describe("agent (VLM planner) or oracle (scripted IK expert)"),
      max_total_usd: z.number().positive().nullish().describe("Spend cap for this collection (within the project budget)"),
      mode: z.enum(["camera", "privileged"]).nullish().describe(
        "camera (default): the planner only sees images + proprioception. privileged: also simulator object poses (labelled as such).",
      ),
    }),
    annotations: WRITE,
  },
  {
    name: "get_collection_progress",
    title: "Get collection progress",
    description:
      "Status of an agent collection: successes, failed attempts, model calls, USD spent vs cap, recoveries, per-cell " +
      "coverage, and the underlying job's progress.",
    inputSchema: z.object({ collection_id: z.string().describe("Collection id (ac_...)") }),
    annotations: READ,
  },
  {
    name: "get_demonstration_results",
    title: "Get demonstration results",
    description:
      "Every attempt with provenance (agent planner model vs scripted oracle, provider, planning mode), outcome, episode_id for " +
      "successes, recoveries, model calls and cost, plus per-collector metrics. Failed attempts are diagnostics only and are " +
      "never used as training data. Report differences between collectors only as measured here.",
    inputSchema: z.object({
      project_id: projectId,
      collection_id: z.string().nullish().describe("Limit to one collection (ac_...)"),
    }),
    annotations: READ,
  },
  {
    name: "train_from_agent_demonstrations",
    title: "Train from agent demonstrations",
    description:
      "Build a LeRobotDataset v3 version from accepted demos of the chosen sources (default only agent-planner demos) and " +
      "propose ACT training. Returns job_id; its result_ref is the proposed training job id. Training does NOT start until a " +
      "human approves it in the studio (or the project's standing approval covers it). Never claim a policy was retrained " +
      "until a checkpoint exists.",
    inputSchema: z.object({
      project_id: projectId,
      sources: z.array(z.enum(["sim_agent", "sim_scripted"])).optional().describe("Demo sources (default ['sim_agent'])"),
      augmentation_preset: z.enum(["lighting_robustness"]).nullish(),
      policy_type: z.enum(["act", "smolvla"]).optional(),
    }),
    annotations: WRITE,
  },
  {
    name: "evaluate_learned_policy",
    title: "Evaluate learned policy",
    description:
      "Evaluate a trained checkpoint (ck_..., never the oracle) in three parallel MuJoCo conditions (baseline, shifted object, " +
      "dim lighting) on held-out seeds. Returns the evaluation id and job (poll cortex.get_job). Success rates come with 95% CIs.",
    inputSchema: z.object({
      project_id: projectId,
      checkpoint_id: z.string().regex(/^ck_/).describe("Checkpoint id from get_project_state training_jobs[].checkpoint_ids"),
      episodes_per_branch: z.number().int().min(1).max(50).optional(),
    }),
    annotations: WRITE,
  },
  {
    name: "improve_policy",
    title: "Improve policy",
    description:
      "Close the loop once: find the weakest measured condition of the latest learned-policy evaluation (or the top coverage " +
      "gap), collect targeted agent demonstrations there in fresh seeds within the budget, build a new dataset version and " +
      "propose retraining (needs approval). Returns the target cell, the evidence, collection_id and job_id.",
    inputSchema: z.object({
      project_id: projectId,
      max_new_demos: z.number().int().min(1).max(20).optional(),
      max_usd: z.number().positive().nullish(),
    }),
    annotations: WRITE,
  },
  // ---------------------------------------------------------------- intake (spec v2)
  {
    name: "start_intake",
    title: "Start intake",
    description:
      "Start teaching a task by describing your setup. Creates a draft project and returns typed questions about the arm, " +
      "cameras, objects, place zones, surroundings and task. Every question carries a ready-to-send `patch` for " +
      "cortex.update_intake, so you can accept defaults or edit them. Objects named in the prompt (e.g. 'water bottle', " +
      "'cube', 'cup') are pre-filled from a catalogue and marked as assumed until you confirm them.",
    inputSchema: z.object({
      prompt: z.string().optional().describe("The task in plain language, e.g. 'pick up the small water bottle and put it on the coaster'"),
      mode: z.enum(["sim", "real"]).optional().describe("'sim' (default; the MVP runs everything in physics sim) or 'real'"),
    }),
    annotations: WRITE,
  },
  {
    name: "update_intake",
    title: "Update intake",
    description:
      "Answer intake questions by merging a partial spec into the project. Lists of objects, zones, cameras and " +
      "environment.obstacles merge by `name` ({name, _delete: true} removes one; {name, confirm: true} confirms an object's " +
      "catalogue values). Units: metres and kilograms; table frame with x away from the arm base, y to its left. Unknown " +
      "keys are rejected with the reason. Returns the merged spec, `changed`, `still_assumed` and the remaining questions.",
    inputSchema: z.object({
      project_id: projectId,
      patch: z.record(z.string(), z.unknown()).describe(
        "Partial intake, e.g. {objects:[{name:'cup', mass_kg:0.2}], zones:[{name:'coaster', center:[0.3,0.1]}], task:{pick:'cup', place:['coaster']}, environment:{lighting:'dim', obstacles:[...]}}",
      ),
    }),
    annotations: WRITE,
  },
  {
    name: "get_intake_status",
    title: "Get intake status",
    description: "The project's intake spec, open questions, whether it is complete and whether the sim scene is built.",
    inputSchema: z.object({ project_id: projectId }),
    annotations: READ,
  },
  {
    name: "build_simulation",
    title: "Build simulation",
    description:
      "Build the MuJoCo scene from the completed intake, then check feasibility against the arm model: jaw opening " +
      "vs object width, holding torque vs actuator limits, and IK reach to every start and zone. Infeasible items " +
      "come back with alternatives (a smaller object, the nearest reachable spot). Returns a top-camera preview image. " +
      "Takes 10-60 s.",
    inputSchema: z.object({ project_id: projectId }),
    annotations: WRITE,
  },
  {
    name: "capture_frame",
    title: "Capture frame",
    description:
      "Render a camera of the built scene (top, front or wrist_cam) as an image, so you can check that the simulation " +
      "matches the setup you described. Fix differences with cortex.update_intake, then cortex.build_simulation again.",
    inputSchema: z.object({
      project_id: projectId,
      camera: z.enum(["top", "front", "wrist_cam"]).optional(),
      seed: z.number().int().optional().describe("Scene randomization seed (default 0)"),
    }),
    annotations: READ,
  },
];

/** Registers every backend tool as `cortex.<name>`. */
export function registerTools(server: MCPServer) {
  for (const t of TOOLS) {
    server.tool(
      {
        name: `cortex.${t.name}`,
        title: t.title,
        description: t.description,
        inputSchema: t.inputSchema,
        annotations: { title: t.title, ...t.annotations },
      },
      async (input, ctx) => callTool(t.name, input as Record<string, unknown>, ctx.signal),
    );
  }
}
