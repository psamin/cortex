import type { CallToolResult } from "mcp-use";

// The Cortex backend's tool registry is the single source of truth:
//   GET  {CORTEX_API_URL}/tools        -> tool names + JSON Schemas
//   POST {CORTEX_API_URL}/tools/{name} -> {summary, ref_ids, ...data}
// A hosted MCP server cannot reach localhost, so on Manufact CORTEX_API_URL
// must be a public URL (e.g. a cloudflared tunnel) and CORTEX_MCP_KEY must
// match the backend's key.
export const API_URL = (process.env.CORTEX_API_URL ?? "http://127.0.0.1:8000/api").replace(/\/+$/, "");
export const STUDIO_URL = (process.env.CORTEX_STUDIO_URL ?? "http://localhost:5173").replace(/\/+$/, "");
const MCP_KEY = process.env.CORTEX_MCP_KEY;

type Json = Record<string, unknown>;

function ids(data: Json): string[] {
  return Array.isArray(data.ref_ids) ? data.ref_ids.map(String) : [];
}

function text(...lines: (string | undefined | false)[]): CallToolResult["content"] {
  return [{ type: "text", text: lines.filter(Boolean).join("\n") }];
}

function error(message: string): CallToolResult {
  return { isError: true, content: text(message) };
}

/** Pydantic 422 detail -> "conditions.0.count: Input should be ...; ..." */
function validationDetail(detail: unknown): string {
  if (!Array.isArray(detail)) return typeof detail === "string" ? detail : JSON.stringify(detail);
  return detail
    .map((e: { loc?: unknown[]; msg?: string }) => `${(e.loc ?? []).join(".") || "(body)"}: ${e.msg ?? "invalid"}`)
    .join("; ");
}

/** A human must act in the studio. Not an error: the agent should stop and ask. */
function approvalRequired(tool: string, args: Json, detail: string): CallToolResult {
  const where = typeof args.project_id === "string" ? `${STUDIO_URL}/p/${args.project_id}` : STUDIO_URL;
  const tj = typeof args.training_job_id === "string" ? args.training_job_id : undefined;
  const ask = tj
    ? `Ask the human to open the Cortex studio at ${where}, go to the project's Improve tab and approve training job ${tj}. After they confirm, call cortex.${tool} again with the same training_job_id.`
    : `Ask the human to do this in the Cortex studio at ${where}.`;
  const headline = `HUMAN APPROVAL REQUIRED - cortex.${tool} was not executed. Do not retry until the human confirms.`;
  // Some clients (Claude Code) show the model structuredContent instead of the
  // text, so the instruction goes in both.
  return {
    content: text(headline, `Backend: ${detail}`, ask),
    structuredContent: {
      summary: headline,
      ref_ids: tj ? [tj] : [],
      approval_required: true,
      executed: false,
      detail,
      next_step: ask,
      studio_url: where,
    },
  };
}

/** POST /tools/{tool} and turn the response into an MCP tool result. */
export async function callTool(tool: string, args: Json, signal?: AbortSignal): Promise<CallToolResult> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    Accept: "application/json",
    "X-Cortex-Actor": "mcp",
  };
  if (MCP_KEY) headers.Authorization = `Bearer ${MCP_KEY}`;

  let res: Response;
  try {
    res = await fetch(`${API_URL}/tools/${encodeURIComponent(tool)}`, {
      method: "POST",
      headers,
      body: JSON.stringify(args),
      signal,
    });
  } catch (e) {
    return error(`Cannot reach the Cortex backend at ${API_URL} (${(e as Error).message}). Is it running, and is CORTEX_API_URL correct?`);
  }

  const raw = await res.text();
  let body: unknown;
  try {
    body = raw ? JSON.parse(raw) : {};
  } catch {
    body = undefined;
  }
  const data = (body && typeof body === "object" && !Array.isArray(body) ? body : {}) as Json;
  const detail = data.detail !== undefined ? validationDetail(data.detail) : raw.slice(0, 500);

  if (res.ok && body !== undefined) {
    const refs = ids(data);
    // Scene renders come back as base64 JPEG; hand them to the agent as an MCP image, not as text.
    const { image_jpeg_base64: image, ...rest } = data;
    const content = text(
      String(data.summary ?? `cortex.${tool} ok`),
      refs.length > 0 && `ref_ids: ${refs.join(", ")}`,
    );
    if (typeof image === "string" && image.length > 0) {
      content.push({ type: "image", data: image, mimeType: "image/jpeg" });
    }
    return { content, structuredContent: rest };
  }
  if (res.status === 403 && data.approval_required === true) return approvalRequired(tool, args, detail);
  if (res.status === 403) {
    return error(`Cortex backend refused access (403): ${detail}. When the backend is reached through a tunnel, set CORTEX_MCP_KEY on this MCP server to the backend's CORTEX_MCP_KEY.`);
  }
  if (res.status === 404) return error(`Not found (404): ${detail}. Check the id.`);
  if (res.status === 422) return error(`Invalid arguments for cortex.${tool} (422): ${detail}`);
  return error(`Cortex backend error ${res.status} on cortex.${tool}: ${detail}`);
}
