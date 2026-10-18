import type { CallToolResult, MCPServer, ToolAnnotations } from "mcp-use";
import { z } from "zod";
import { callTool, STUDIO_URL } from "./backend.js";

// MCP Apps views (#119): Cortex results rendered inside the agent's chat (Claude, ChatGPT, any MCP
// Apps host). Each model-visible `cortex.show_*` tool is bound to a view under views/<name>/view.tsx;
// the `cortex.view_*` tools have visibility "app": the view calls them (live refresh, Approve) and
// hosts hide them from the model. Hosts without MCP Apps still get the text summary.

const READ: ToolAnnotations = { readOnlyHint: true, destructiveHint: false, idempotentHint: true, openWorldHint: false };
const projectId = z.string().describe("Cortex project id (prj_...)");
type Json = Record<string, unknown>;

type Failure = { isError: true; content: CallToolResult["content"] };
const text = (s: string) => [{ type: "text" as const, text: s }];

/** Backend JSON for a registry tool, or an MCP error result to return as-is. */
async function backend(tool: string, args: Json, signal?: AbortSignal): Promise<Json | Failure> {
  const res = await callTool(tool, args, signal);
  if (res.isError || !res.structuredContent) return { isError: true, content: res.content ?? text(`cortex.${tool} failed`) };
  const data = res.structuredContent as Json;
  // A refused approval comes back as a non-error "ask the human" result; for a view it is an error.
  if (data.approval_required === true) return { isError: true, content: text(String(data.detail ?? data.summary)) };
  return data;
}
const isResult = (x: Json | Failure): x is Failure => x.isError === true && Array.isArray(x.content);

// ------------------------------------------------------------------ collection

const attempt = z.object({
  index: z.number(),
  status: z.string(),
  seed: z.number(),
  condition: z.string(),
  model_calls: z.number().nullable(),
  cost_usd: z.number().nullable(),
  stop_reason: z.string().nullable(),
  recoveries: z.number().nullable(),
});
const collectionOut = z.object({
  project_id: z.string(),
  collection_id: z.string(),
  status: z.string(),
  running: z.boolean(),
  collector: z.string(),
  model: z.string().nullable(),
  provider: z.string().nullable(),
  planned: z.number(),
  attempts: z.number(),
  successes: z.number(),
  model_calls: z.number(),
  cost_usd: z.number(),
  budget_cap_usd: z.number(),
  reason: z.string(),
  error: z.string().nullable(),
  recent: z.array(attempt),
  studio_url: z.string(),
});
export type CollectionOut = z.infer<typeof collectionOut>;

function collection(d: Json): { out: CollectionOut; frame?: string; age?: number } {
  const out = collectionOut.parse({ ...d, studio_url: `${STUDIO_URL}/p/${d.project_id}/collect` });
  return { out, frame: (d.live_frame_b64 as string | null) ?? undefined, age: (d.live_frame_age_s as number | null) ?? undefined };
}

// ------------------------------------------------------------------ evaluation

const branch = z.object({
  name: z.string(),
  condition: z.record(z.string(), z.unknown()),
  status: z.string(),
  n: z.number(),
  successes: z.number(),
  success_rate: z.number().nullable(),
  ci95: z.array(z.number()).nullable(),
  failure_modes: z.record(z.string(), z.number()),
});
const evaluationOut = z.object({
  project_id: z.string(),
  evaluation_id: z.string(),
  policy_label: z.string(),
  policy_kind: z.string(),
  status: z.string(),
  running: z.boolean(),
  branches: z.array(branch),
  weakest: z.string().nullable(),
  studio_url: z.string(),
});

export type EvaluationOut = z.infer<typeof evaluationOut>;

// ------------------------------------------------------------------ training review

const reviewOut = z.object({
  project_id: z.string(),
  training_job_id: z.string(),
  status: z.string(),
  awaiting_approval: z.boolean(),
  policy_type: z.string(),
  backend: z.string(),
  rationale: z.string(),
  steps: z.number().nullable(),
  approved_by: z.string().nullable(),
  estimate: z.object({ gpu: z.string(), gpu_hours: z.number(), est_cost_usd: z.number(), paid: z.boolean(), notes: z.string() }),
  dataset: z
    .object({
      id: z.string(),
      name: z.string(),
      source_episodes: z.number(),
      frames: z.number(),
      train_episodes: z.number(),
      eval_episodes: z.number(),
      augmentations: z.array(z.string()),
    })
    .nullable(),
  studio_url: z.string(),
});

export type ReviewOut = z.infer<typeof reviewOut>;

export function registerViews(server: MCPServer) {
  server.tool(
    {
      name: "cortex.show_collection",
      title: "Show demonstration collection",
      description:
        "Show a live view of an agent demonstration collection in the chat: the simulator frame, successes vs attempts, " +
        "model calls, spend and recent attempts. Read-only. Use after cortex_create_policy_project or " +
        "cortex_start_agent_collection so the human can watch it run.",
      inputSchema: z.object({ project_id: projectId, collection_id: z.string().optional().describe("default: latest") }),
      outputSchema: collectionOut,
      annotations: { title: "Show demonstration collection", ...READ },
      view: { name: "collection", description: "Live MuJoCo collection viewport", prefersBorder: true },
    },
    async (input, ctx) => {
      const d = await backend("collection_snapshot", input, ctx.signal);
      if (isResult(d)) return d;
      const { out, frame, age } = collection(d);
      return {
        content: text(String(d.summary)),
        structuredContent: out,
        _meta: { "cortex/live_frame_b64": frame, "cortex/live_frame_age_s": age },
      };
    },
  );

  server.tool(
    {
      name: "cortex.view_collection_snapshot",
      title: "Collection snapshot (view refresh)",
      description: "Used by the collection view to refresh itself. Not for the model.",
      inputSchema: z.object({ project_id: projectId, collection_id: z.string().optional().describe("ac_...; default: latest") }),
      outputSchema: collectionOut.extend({ live_frame_b64: z.string().nullable(), live_frame_age_s: z.number().nullable() }),
      annotations: { title: "Collection snapshot", ...READ },
      visibility: "app",
    },
    async (input, ctx) => {
      const d = await backend("collection_snapshot", input, ctx.signal);
      if (isResult(d)) return d;
      const { out, frame, age } = collection(d);
      return {
        content: text(String(d.summary)),
        structuredContent: { ...out, live_frame_b64: frame ?? null, live_frame_age_s: age ?? null },
      };
    },
  );

  server.tool(
    {
      name: "cortex.show_evaluation",
      title: "Show policy evaluation",
      description:
        "Show a policy evaluation in the chat: success per condition with Wilson 95% intervals, measured failure modes " +
        "and the weakest condition. Read-only. Use after cortex_evaluate_learned_policy or cortex_launch_simulation_evaluation.",
      inputSchema: z.object({ project_id: projectId, evaluation_id: z.string().optional().describe("ev_...; default: latest") }),
      outputSchema: evaluationOut,
      annotations: { title: "Show policy evaluation", ...READ },
      view: { name: "evaluation", description: "Evaluation results per condition", prefersBorder: true },
    },
    async (input, ctx) => {
      const d = await backend("evaluation_report", input, ctx.signal);
      if (isResult(d)) return d;
      const out = evaluationOut.parse({ ...d, studio_url: `${STUDIO_URL}/p/${d.project_id}/improve` });
      return { content: text(String(d.summary)), structuredContent: out };
    },
  );

  server.tool(
    {
      name: "cortex.view_evaluation_report",
      title: "Evaluation report (view refresh)",
      description: "Used by the evaluation view to refresh while branches run. Not for the model.",
      inputSchema: z.object({ project_id: projectId, evaluation_id: z.string().optional().describe("ev_...; default: latest") }),
      outputSchema: evaluationOut,
      annotations: { title: "Evaluation report", ...READ },
      visibility: "app",
    },
    async (input, ctx) => {
      const d = await backend("evaluation_report", input, ctx.signal);
      if (isResult(d)) return d;
      return {
        content: text(String(d.summary)),
        structuredContent: evaluationOut.parse({ ...d, studio_url: `${STUDIO_URL}/p/${d.project_id}/improve` }),
      };
    },
  );

  server.tool(
    {
      name: "cortex.show_training_review",
      title: "Show training job for human approval",
      description:
        "Show a proposed training job (cost, GPU hours, dataset, rationale) to the human in the chat, with an Approve " +
        "button only they can press. This does NOT approve anything, and you cannot approve on their behalf. Call it " +
        "after cortex_propose_training_job or cortex_train_from_agent_demonstrations, then wait for the human. Once they " +
        "say they approved, call cortex_launch_training_job.",
      inputSchema: z.object({ project_id: projectId, training_job_id: z.string().optional().describe("tj_...; default: latest proposed") }),
      outputSchema: reviewOut,
      annotations: { title: "Show training job for approval", ...READ, idempotentHint: false },
      view: { name: "approval", description: "Training approval card (human click only)", prefersBorder: true },
    },
    async (input, ctx) => {
      const d = await backend("review_training_job", input, ctx.signal);
      if (isResult(d)) return d;
      const nonce = d.view_approval_nonce as string | undefined;
      const out = reviewOut.parse({ ...d, studio_url: `${STUDIO_URL}/p/${d.project_id}/improve` });
      const ask = out.awaiting_approval
        ? " The human can approve it with the button in this view or in the studio. Wait for them; do not retry."
        : "";
      // The nonce goes only to the view (result _meta). It is never in content or structuredContent.
      return { content: text(String(d.summary) + ask), structuredContent: out, _meta: nonce ? { "cortex/approval_nonce": nonce } : {} };
    },
  );

  server.tool(
    {
      name: "cortex.view_approve_training",
      title: "Approve training (human click in the view)",
      description: "Called only by the approval view's Approve button with a single-use nonce the model never sees.",
      inputSchema: z.object({
        project_id: projectId,
        training_job_id: z.string().describe("tj_... being approved"),
        nonce: z.string().describe("single-use approval nonce from the review view (never shown to the model)"),
      }),
      outputSchema: z.object({ training_job_id: z.string(), status: z.string() }),
      annotations: { title: "Approve training", readOnlyHint: false, destructiveHint: false, idempotentHint: false, openWorldHint: false },
      visibility: "app",
    },
    async (input, ctx) => {
      const d = await backend("approve_training_from_view", input, ctx.signal);
      if (isResult(d)) return d;
      return {
        content: text(String(d.summary)),
        structuredContent: { training_job_id: String(d.training_job_id), status: String(d.status) },
      };
    },
  );
}
