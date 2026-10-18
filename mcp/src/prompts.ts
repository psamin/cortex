import type { MCPServer } from "mcp-use";
import { z } from "zod";
import { STUDIO_URL } from "./backend.js";

const steps = (task: string, projectId?: string) => `You are driving Cortex, a robot-learning studio, through its cortex.* MCP tools.
Goal: teach the robot this task in simulation: "${task}".

Work through the loop below. After every tool call, read its summary and ref_ids; use ids from results, never invent them.
${projectId ? `\n0. A project already exists: ${projectId}. Call cortex.get_project_state and cortex.get_next_actions, then resume at the first step that is not done.\n` : ""}
1. Set up the scene. Ask the human: a photo of their setup, a detailed description, or the default scene?
   - photo / description: cortex.start_intake {prompt, image_path | description} (Cortex's Claude Code rebuilds the
     scene). Show them scene_setup and the open questions, apply their answers with cortex.update_intake, then
     cortex.build_simulation, compare cortex.capture_frame with their setup, and cortex.start_agent_collection {project_id}.
   - default: cortex.create_policy_project {task_prompt, num_demos: 12} -> project_id, collection_id, job_id.
   Cortex's agent planner generates demonstrations autonomously (it plans from camera images; a validated IK controller
   executes in MuJoCo). If collection_not_started says the planner is unavailable, tell the human and use
   cortex.start_agent_collection {project_id, collector: "oracle"} instead (the scripted IK oracle), clearly labelled as such.
2. If the call returned a job_id/collection_id instead of the finished result, poll cortex.get_collection_progress {collection_id} every ~20 s until it finishes. Report successes vs failed attempts,
   model calls and USD spent. Failed attempts are diagnostics, never training data.
3. cortex.get_demonstration_results {project_id} for provenance and per-collector metrics (only claim what is measured).
4. cortex.train_from_agent_demonstrations {project_id} -> job_id; poll cortex.get_job; result_ref = training job tj_....
5. STOP AND ASK THE HUMAN to approve tj_... in the Cortex studio (${STUDIO_URL}, project -> Improve tab). Do not invent an
   approval token. Once they confirm, call cortex.launch_training_job {training_job_id}. Training takes tens of minutes on the
   GPU cluster; never say the policy was retrained before a checkpoint (ck_...) appears in get_project_state.
6. cortex.evaluate_learned_policy {project_id, checkpoint_id} -> three parallel MuJoCo conditions on held-out seeds;
   poll cortex.get_job. This measures the learned policy, not the oracle.
7. cortex.improve_policy {project_id, max_new_demos: 5} -> it targets the weakest measured condition with new agent demos and
   proposes retraining (approval again). Then cortex.compare_experiments after the next evaluation.
8. cortex.get_next_actions {project_id} and report: demos collected (agent planner vs oracle), evaluation results with CIs, the
   weakest condition and what Cortex is doing about it.

Rules: never stream or paste raw video; refer to episodes, evaluations and media by id or storage key.
Physical-robot recording and paid or cluster jobs always need a human.`;

/** Registers the `cortex.teach_task` walkthrough prompt. */
export function registerPrompts(server: MCPServer) {
  server.prompt(
    {
      name: "cortex.teach_task",
      title: "Teach a robot a task with Cortex",
      description:
        "Step-by-step plan for the full Cortex loop: set up the scene (photo, description or default), collect demos, build a " +
        "dataset version, propose training, get human approval, evaluate in simulation, compare, and pick next actions.",
      schema: z.object({
        task: z.string().describe("What the robot should learn, e.g. 'pick up the cube and place it in the bowl'"),
        project_id: z.string().optional().describe("Resume an existing project (prj_...) instead of creating one"),
      }),
    },
    async ({ task, project_id }) => ({
      messages: [{ role: "user", content: { type: "text", text: steps(task, project_id) } }],
    }),
  );
}
