import { inputRequired, inputResponse, type CallToolResult, type MCPServer, type RequestContext, type ToolAnnotations } from "mcp-use";
import { z } from "zod";
import { callTool, identityOf, type Identity } from "./backend.js";

// Bring your own model: VLM failure diagnosis answered by the MCP client's model instead of a
// model route Cortex holds a key for. Two ways in:
//
//  cortex.diagnose_failures_with_my_model  one call. Fetches the diagnosis inputs, asks the client's
//      model through MCP sampling (sampling/createMessage, one request per contact sheet, returned
//      as an `input_required` result; the SDK sends real server->client requests to 2025-era
//      clients), then submits the raw replies. Route "mcp_sampling", model = the sampling result's
//      `model`. Needs a client that supports sampling.
//  cortex.get_diagnosis_inputs + cortex.submit_failure_hypotheses  the same thing for clients
//      without sampling: the agent looks at the contact sheets itself. Route "mcp_agent".
//
// The backend validates every answer against the fixed taxonomy (one invalid answer rejects the
// submission) and stores accepted ones as unverified VLM hypotheses with that provenance.

type Json = Record<string, unknown>;
type Target = { ref_id: string; prompt: string; image_b64: string; mime_type: string; telemetry: Json };
type Inputs = { analysis_id: string; targets: Target[]; summary: string; taxonomy: Record<string, string> };
type State = { analysis_id: string; refs: string[] };

const projectId = z.string().describe("Cortex project id (prj_...)");
const maxTargets = z.number().int().min(1).max(6).optional().describe(
  "Episodes to review, failed evaluation rollouts first (default 4). Each one is a separate model request.",
);
const WRITE: ToolAnnotations = { readOnlyHint: false, destructiveHint: false, idempotentHint: false, openWorldHint: false };

const SYSTEM =
  "You review robot manipulation episodes from a simulator. Answer with only the JSON object the user asks for. " +
  "Your answer is stored as an unverified hypothesis; the simulator's measurements are ground truth.";

async function fetchInputs(project_id: string, max_targets: number | undefined, signal: AbortSignal, who: Identity) {
  const res = await callTool("get_diagnosis_inputs", { project_id, ...(max_targets ? { max_targets } : {}) }, signal, who);
  return { res, inputs: res.isError ? undefined : (res.structuredContent as Inputs | undefined) };
}

function clientName(ctx: RequestContext): string | undefined {
  const { name, version } = ctx.client.info();
  return name ? `${name}${version ? ` ${version}` : ""}` : undefined;
}

function replyText(content: unknown): string | undefined {
  const blocks = (Array.isArray(content) ? content : [content]) as { type?: string; text?: string }[];
  const text = blocks.filter((b) => b?.type === "text" && typeof b.text === "string").map((b) => b.text).join("\n");
  return text || undefined;
}

function prefixed(res: CallToolResult, ...lines: string[]): CallToolResult {
  return { ...res, content: [{ type: "text", text: lines.join("\n") }, ...(res.content ?? [])] };
}

export function registerSampling(server: MCPServer) {
  server.tool(
    {
      name: "cortex.diagnose_failures_with_my_model",
      title: "Diagnose failures with my model (MCP sampling)",
      description:
        "Bring your own model: VLM failure diagnosis without a Cortex model key. Cortex runs quality checks and the " +
        "measured failure analysis, then asks YOUR client's model (MCP sampling) to look at each failed episode's " +
        "keyframe contact sheet and name failure categories from a fixed taxonomy. Answers are validated and stored " +
        "as unverified VLM hypotheses recording route 'mcp_sampling' and the model your client reports; failure " +
        "clusters and recommendations then use them. Your client may ask you to approve each request.\n" +
        "If your client does not support sampling, use cortex_get_diagnosis_inputs and cortex_submit_failure_hypotheses " +
        "instead (you look at the images yourself).",
      inputSchema: z.object({ project_id: projectId, max_targets: maxTargets }),
      annotations: { title: "Diagnose failures with my model", ...WRITE },
    },
    async ({ project_id, max_targets }, ctx) => {
      const state = ctx.requestState<string>();
      if (!state && !ctx.client.can("sampling")) {
        // Checked before any backend work. 2025-era requests have no per-request capabilities, and
        // mcp-use serves them statelessly, so it could not send them a sampling request anyway.
        return {
          isError: true,
          content: [{
            type: "text",
            text:
              "Your MCP client did not declare the sampling capability on this request (or uses the 2025 protocol, " +
              "which this server answers statelessly), so Cortex cannot ask its model. Nothing was started. " +
              "Use cortex.get_diagnosis_inputs, look at the contact sheets yourself, then cortex.submit_failure_hypotheses.",
          }],
        };
      }
      if (!state) {
        // Round 1: prepare the analysis and ask the client's model, one request per contact sheet.
        const { res, inputs } = await fetchInputs(project_id, max_targets, ctx.signal, identityOf(ctx));
        if (!inputs) return res;
        if (inputs.targets.length === 0) return res;
        const inputRequests = Object.fromEntries(
          inputs.targets.map((t, i) => [
            `target${i}`,
            inputRequired.createMessage({
              systemPrompt: SYSTEM,
              messages: [
                { role: "user", content: { type: "image", data: t.image_b64, mimeType: t.mime_type } },
                { role: "user", content: { type: "text", text: t.prompt } },
              ],
              maxTokens: 600,
              temperature: 0,
              includeContext: "none",
              modelPreferences: { intelligencePriority: 0.8, speedPriority: 0.2, costPriority: 0.2 },
            }),
          ]),
        );
        const next: State = { analysis_id: inputs.analysis_id, refs: inputs.targets.map((t) => t.ref_id) };
        return inputRequired({ inputRequests, requestState: JSON.stringify(next) });
      }

      // Round 2: the client answered (or not). The state and answers come back through the client,
      // so the backend re-checks the analysis id, the refs and every answer.
      let st: State;
      try {
        st = JSON.parse(state) as State;
      } catch {
        return { isError: true, content: [{ type: "text", text: "Malformed request state; call cortex.diagnose_failures_with_my_model again." }] };
      }
      const models = new Set<string>();
      const answers = st.refs.map((ref_id, i) => {
        const r = inputResponse(ctx.inputResponses, `target${i}`);
        if (r.kind !== "sampling") return { ref_id, error: "the client returned no sampling result for this episode" };
        if (r.result.model) models.add(r.result.model);
        const text = replyText(r.result.content);
        return text ? { ref_id, reply_text: text } : { ref_id, error: "the sampling result had no text content" };
      });
      const model = [...models].join(", ") || "unknown (the client did not report a model)";
      const client = clientName(ctx);
      const res = await callTool(
        "submit_failure_hypotheses",
        { project_id, analysis_id: st.analysis_id, route: "mcp_sampling", model, ...(client ? { client } : {}), answers },
        ctx.signal,
        identityOf(ctx),
      );
      if (res.isError) {
        return prefixed(
          res,
          `Cortex rejected the replies from ${model} for analysis ${st.analysis_id}; nothing was stored.`,
          "Either run cortex.diagnose_failures_with_my_model again, or look at the images yourself with " +
            "cortex.get_diagnosis_inputs and submit with cortex.submit_failure_hypotheses.",
        );
      }
      return res;
    },
  );

  server.tool(
    {
      name: "cortex.get_diagnosis_inputs",
      title: "Get failure diagnosis inputs",
      description:
        "Bring your own model, for clients without MCP sampling: runs quality checks and the measured failure analysis, " +
        "then returns, for each episode a VLM should review, its keyframe contact sheet (image), the measured " +
        "telemetry and the fixed-taxonomy prompt. Look at each image yourself, then call " +
        "cortex_submit_failure_hypotheses with the analysis_id. Takes a few seconds per recorded episode.",
      inputSchema: z.object({ project_id: projectId, max_targets: maxTargets }),
      annotations: { title: "Get failure diagnosis inputs", ...WRITE },
    },
    async ({ project_id, max_targets }, ctx) => {
      const { res, inputs } = await fetchInputs(project_id, max_targets, ctx.signal, identityOf(ctx));
      if (!inputs) return res;
      if (inputs.targets.length === 0) return res;
      // Images go in content; no structuredContent, since some clients (Claude Code) then hide content.
      return {
        content: [
          {
            type: "text",
            text: [
              inputs.summary,
              `analysis_id: ${inputs.analysis_id}`,
              "For each episode below, answer its prompt from the image. Then call cortex_submit_failure_hypotheses once " +
                "with every answer (ref_id + hypotheses, or ref_id + error if you cannot tell) and your model name. " +
                "Use only the listed categories; invalid answers reject the whole submission.",
            ].join("\n"),
          },
          ...inputs.targets.flatMap((t, i) => [
            { type: "text" as const, text: `--- episode ${i + 1}/${inputs.targets.length}: ref_id ${t.ref_id}\n${t.prompt}` },
            { type: "image" as const, data: t.image_b64, mimeType: t.mime_type },
          ]),
        ],
      };
    },
  );

  const hypothesis = z.object({
    category: z.string().describe("a category from the taxonomy in the prompt, e.g. grasp_miss"),
    confidence: z.number().min(0).max(1).describe("your own confidence, 0-1"),
    rationale: z.string().min(1).max(300).describe("<= 30 words"),
    frame_indices: z.array(z.number().int().min(0).max(5)).max(6).optional().describe("keyframes 0-5 that show it"),
  });
  server.tool(
    {
      name: "cortex.submit_failure_hypotheses",
      title: "Submit failure hypotheses",
      description:
        "Store your failure hypotheses for an analysis from cortex_get_diagnosis_inputs. Cortex validates them against " +
        "the fixed taxonomy (any invalid answer rejects the submission, nothing stored) and stores them as unverified " +
        "VLM hypotheses with route 'mcp_agent' and the model name you give, then rebuilds failure clusters. " +
        "A human confirms or rejects them in the studio.",
      inputSchema: z.object({
        project_id: projectId,
        analysis_id: z.string().describe("analysis_id from cortex_get_diagnosis_inputs (an_...)"),
        model: z.string().min(1).max(120).describe("the model that looked at the images (yours), e.g. claude-sonnet-4-5"),
        answers: z.array(
          z.object({
            ref_id: z.string().describe("ref_id of the episode"),
            hypotheses: z.array(hypothesis).min(1).max(3).optional().describe("up to 3 hypotheses"),
            error: z.string().max(300).optional().describe("instead of hypotheses: why you could not assess this episode"),
          }),
        ).min(1).max(6).describe("one entry per episode from cortex_get_diagnosis_inputs (1-6)"),
      }),
      annotations: { title: "Submit failure hypotheses", ...WRITE },
    },
    async ({ project_id, analysis_id, model, answers }, ctx) => {
      const client = clientName(ctx);
      return callTool(
        "submit_failure_hypotheses",
        { project_id, analysis_id, route: "mcp_agent", model, ...(client ? { client } : {}), answers },
        ctx.signal,
        identityOf(ctx),
      );
    },
  );
}
