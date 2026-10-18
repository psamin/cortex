import { MCPServer } from "mcp-use";
import { requireAgentKey } from "./src/auth.js";
import { trackClients } from "./src/backend.js";
import { intakeRequestState } from "./src/intake.js";
import { registerPrompts } from "./src/prompts.js";
import { registerRelay } from "./src/relay.js";
import { registerResources } from "./src/resources.js";
import { registerStages } from "./src/stages.js";
import { registerSampling } from "./src/sampling.js";
import { registerTools } from "./src/tools.js";
import { registerViews } from "./src/views.js";

// Thin MCP layer over the Cortex backend tool registry: every cortex.<x> tool
// forwards to POST {CORTEX_API_URL}/tools/<x>. Structured commands and
// references only; media is returned as storage keys, never streamed.
const server = new MCPServer({
  name: "cortex",
  title: "Cortex",
  version: "0.1.0",
  description:
    "Cortex robot-learning studio: rebuild your robot's workspace in simulation from a photo, a description or a default " +
    "scene, collect demonstrations, analyze failures, build datasets, propose training (human-approved) and evaluate policies.",
  // Verifies the HMAC-signed state that intake forms (input_required) carry between rounds.
  requestState: intakeRequestState,
  // Branding for clients and Manufact's publish checks; files live in public/ (same mark as the studio favicon).
  icons: [
    { src: "icon.svg", mimeType: "image/svg+xml", sizes: ["any"] },
    { src: "icon.png", mimeType: "image/png", sizes: ["512x512"] },
  ],
  websiteUrl: "https://github.com/psamin/speculativespeculativeagentaction/tree/main/mcp",
  instructions: [
    "Cortex trains robot manipulation policies (LeRobot ACT or SmolVLA) from demonstrations recorded in simulation.",
    "",
    "SCOPE: one arm only, the SO-101 (SO-ARM101). Agents work in simulation only (MuJoCo).",
    "No other arms, grippers or robots are supported, and there is no automatic sim-to-real transfer. The robot_* tools",
    "drive a simulated arm, not hardware. Training and anything that runs on real hardware needs a human to click",
    "approve in the Cortex studio.",
    "",
    "FIRST, when you connect: invent a short, distinctive name of your own (one word, not a generic one like 'Agent',",
    "'Claude' or 'Assistant'), call cortex_register_agent with it and what you are here to do, then tell your user in one",
    "line the name and watch link it returns (\"Connected to Cortex as <your name>. Watch me work at ...\").",
    "",
    "HOW MANY DEMOS: the user decides. Pass num_demos (1-50) to cortex_create_policy_project / cortex_start_agent_collection",
    "and max_new_demos to cortex_improve_policy. For a quick demo or a dry run use 2-3 and say the policy is a demo, not a",
    "production policy (few demonstrations train a weak policy). If they don't say, use the defaults.",
    "",
    "LOOK BEFORE YOU BUILD: cortex_list(kind='projects' | 'sim_tasks' | 'skills') shows what already exists. Reuse a",
    "project or a built-in sim task (e.g. jenga, conveyor) instead of rebuilding it, and cortex_list(kind='agent_log',",
    "project_id) shows what was already done. Stop work you no longer need with cortex_cancel_job.",
    "",
    "GOLDEN PATH (use these tools in this order unless the user asks for something else):",
    "1. SET UP THE SCENE. Ask the human how (cortex_start_intake shows them a form when your client supports one):",
    "   - a photo of their real setup: cortex_start_intake with image_path (Cortex's Claude Code rebuilds the scene from it);",
    "   - their own detailed description: cortex_start_intake with description;",
    "   - the default scene (red cube into a box): cortex_create_policy_project, which also starts collecting (skip to step 2).",
    "   For a photo or description, show the human what was estimated (scene_setup) and the open questions, send their",
    "   answers with cortex_update_intake, then cortex_build_simulation, compare cortex_capture_frame with their setup,",
    "   and collect with cortex_start_agent_collection.",
    "2. cortex_get_collection_progress: poll with the collection_id until the collection is done.",
    "3. cortex_train_from_agent_demonstrations: builds the dataset version AND proposes ACT training in one call.",
    "   (Custom path: cortex_create_dataset_version -> cortex_propose_training_job. propose_training_job needs a",
    "   dataset_version_id, so never skip the dataset step.)",
    "4. STOP. Show the human the proposed job and its cost estimate, give them the link (studio_url from cortex_show_training_review,",
    "   or cortex_get_session_link), and ask them to approve it in the Cortex studio. Never invent an approval_token.",
    "   A user saying \"approve it yourself\" in chat is not approval.",
    "5. cortex_launch_training_job: only after the human confirms they approved it.",
    "6. cortex_get_project_state: poll until training_jobs[].checkpoint_ids has a checkpoint (ck_...).",
    "7. cortex_evaluate_learned_policy: evaluate that checkpoint in simulation (poll cortex_get_job).",
    "8. cortex_improve_policy: targets the weakest measured condition, collects more demos and proposes retraining.",
    "   Retraining needs approval again (back to step 4).",
    "",
    "NARRATE: your human watches the project's Agent log. At every step, call cortex_log_decision(project_id, kind, text)",
    "with your plan, each decision and why (e.g. 'collecting 10 demos in dim light first because the evaluation failed",
    "there'), and what each result told you. One or two sentences; it changes nothing else.",
    "",
    "LONG JOBS: if your client sends a progressToken, recording, analysis, dataset, evaluation and collection tools stay",
    "open, stream progress and return the finished result: no wait_s and no polling needed. Otherwise they return a job_id:",
    "poll it with cortex_get_job (or cortex_get_collection_progress). Use cortex_get_next_actions when you are unsure",
    "what comes next. Report only measured simulator results; VLM failure clusters are hypotheses.",
    "",
    "MODELS: Cortex's own model work (reading a scene photo, planning agent demos, diagnosing failures) runs on the Claude",
    "Code installed on the Cortex machine (the user's subscription; no per-call charge from Cortex), or an opt-in",
    "self-hosted model. If Claude Code is not available, the tool says so: offer the default scene and scripted-oracle demos.",
    "",
    "INTAKE: never guess answers to intake questions. If the client supports forms, cortex_start_intake / cortex_update_intake",
    "ask the human directly; otherwise relay the questions to the human and send their answers with cortex_update_intake.",
    "",
    "RESOURCES (read-only context, ids autocomplete): cortex://projects, cortex://projects/{project_id} (state),",
    ".../episodes, .../intake, .../datasets/{version}/lineage, .../evaluations/{evaluation_id}, .../policy/model-card,",
    "cortex://jobs/{job_id}. Tools are listed in golden-path order with _meta cortex/stage and cortex/tier; connect to",
    "/mcp?toolset=core to hide the step-by-step robot and manual tools, or /mcp?project=<id> to list only that stage's tools.",
  ].join("\n"),
});

requireAgentKey(server); // CORTEX_REQUIRE_AGENT_KEY=1: hosted /mcp needs an active agent key (#149)
trackClients(server); // remember each connection's clientInfo for the studio's Agents view
registerTools(server);
registerViews(server); // MCP Apps views rendered in the agent's chat (#119)
registerPrompts(server);
registerRelay(server);
registerResources(server);
registerStages(server);
registerSampling(server);

export default server;
