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
    "Cortex robot-learning studio: create projects, record simulated demonstrations, analyze failures, build datasets, " +
    "propose training (human-approved) and evaluate policies in simulation.",
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
    "GOLDEN PATH (use these tools in this order unless the user asks for something else):",
    "1. cortex_create_policy_project: creates the project and starts autonomous demonstration collection. This is the fast",
    "   path for the built-in tabletop pick-and-place scene (red cube into a box). For a custom scene (the user's own objects, zones or",
    "   obstacles), use cortex_start_intake -> cortex_update_intake (answer the questions) -> cortex_build_simulation",
    "   instead, then collect with cortex_start_agent_collection.",
    "2. cortex_get_collection_progress: poll with the collection_id until the collection is done.",
    "3. cortex_train_from_agent_demonstrations: builds the dataset version AND proposes ACT training in one call.",
    "   (Custom path: cortex_create_dataset_version -> cortex_propose_training_job. propose_training_job needs a",
    "   dataset_version_id, so never skip the dataset step.)",
    "4. STOP. Show the human the proposed job and its cost estimate, and ask them to approve it in the Cortex studio.",
    "   Never invent an approval_token.",
    "5. cortex_launch_training_job: only after the human confirms they approved it.",
    "6. cortex_get_project_state: poll until training_jobs[].checkpoint_ids has a checkpoint (ck_...).",
    "7. cortex_evaluate_learned_policy: evaluate that checkpoint in simulation (poll cortex_get_job).",
    "8. cortex_improve_policy: targets the weakest measured condition, collects more demos and proposes retraining.",
    "   Retraining needs approval again (back to step 4).",
    "",
    "LONG JOBS: if your client sends a progressToken, recording, analysis, dataset, evaluation and collection tools stay",
    "open, stream progress and return the finished result: no wait_s and no polling needed. Otherwise they return a job_id:",
    "poll it with cortex_get_job (or cortex_get_collection_progress). Use cortex_get_next_actions when you are unsure",
    "what comes next. Report only measured simulator results; VLM failure clusters are hypotheses.",
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
