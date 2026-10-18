import type { CallToolResult } from "mcp-use";
import { relayCall, relayEnabled, runnerConnected } from "./relay.js";

// The Cortex backend's tool registry is the single source of truth:
//   GET  {CORTEX_API_URL}/tools        -> tool names + JSON Schemas
//   POST {CORTEX_API_URL}/tools/{name} -> {summary, ref_ids, ...data}
// Two transports, picked per call:
//   runner  a local runner (backend/cortex/mcp_runner.py) is long-polling this
//           server (src/relay.ts); used whenever one is connected. This is how
//           a Manufact-hosted server reaches a laptop with no tunnel.
//   direct  POST to CORTEX_API_URL (local dev, or a public/tunneled backend).
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

const RUNNER_HINT =
  "Start the local runner next to your backend: `scripts/dev.sh --runner <this server's URL>` or " +
  "`cd backend && uv run python -m cortex.mcp_runner --server <this server's URL>` (same CORTEX_MCP_KEY on both).";

type Transport = { status: number; raw: string } | { error: CallToolResult };

async function send(tool: string, args: Json, signal?: AbortSignal): Promise<Transport> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    Accept: "application/json",
    "X-Cortex-Actor": "mcp",
  };
  if (relayEnabled() && runnerConnected()) {
    try {
      const r = await relayCall(tool, args, headers, signal);
      return r;
    } catch (e) {
      return { error: error(`The Cortex runner did not complete cortex.${tool} (${(e as Error).message}). Is it still running?`) };
    }
  }
  if (MCP_KEY) headers.Authorization = `Bearer ${MCP_KEY}`;
  try {
    const res = await fetch(`${API_URL}/tools/${encodeURIComponent(tool)}`, {
      method: "POST",
      headers,
      body: JSON.stringify(args),
      signal,
    });
    return { status: res.status, raw: await res.text() };
  } catch (e) {
    const why = `Cannot reach the Cortex backend: no local runner is connected and ${API_URL} is unreachable (${(e as Error).message}).`;
    return {
      error: error(
        relayEnabled()
          ? `${why} ${RUNNER_HINT}`
          : `${why} Is the backend running and CORTEX_API_URL correct? (To reach a laptop from a hosted server, set CORTEX_MCP_KEY here and start the runner.)`,
      ),
    };
  }
}

/** POST /tools/{tool} (directly or through the runner) and turn the response into an MCP tool result. */
export async function callTool(tool: string, args: Json, signal?: AbortSignal): Promise<CallToolResult> {
  const sent = await send(tool, args, signal);
  if ("error" in sent) return sent.error;
  const res = { status: sent.status, ok: sent.status >= 200 && sent.status < 300 };
  const raw = sent.raw;
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
    if (typeof data.image_b64 === "string" && data.image_b64) {
      // Camera image for the agent to look at. Claude Code shows the model structuredContent
      // instead of content when both exist, so image results carry no structuredContent.
      const { image_b64, ...rest } = data;
      const result = (rest.result ?? {}) as Json;
      return {
        content: [
          ...text(
            String(rest.summary ?? `cortex.${tool} ok`),
            typeof rest.image_kind === "string" && `image: ${rest.image_kind}`,
            `result: ${JSON.stringify(result).slice(0, 1500)}`,
          ),
          { type: "image", data: image_b64 as string, mimeType: "image/jpeg" },
        ],
      };
    }
    return {
      content: text(
        String(data.summary ?? `cortex.${tool} ok`),
        refs.length > 0 && `ref_ids: ${refs.join(", ")}`,
      ),
      structuredContent: data,
    };
  }
  if (res.status === 403 && data.approval_required === true) return approvalRequired(tool, args, detail);
  if (res.status === 403) {
    return error(`Cortex backend refused access (403): ${detail}. When the backend is reached through a tunnel, set CORTEX_MCP_KEY on this MCP server to the backend's CORTEX_MCP_KEY.`);
  }
  if (res.status === 404) return error(`Not found (404): ${detail}. Check the id.`);
  if (res.status === 422) return error(`Invalid arguments for cortex.${tool} (422): ${detail}`);
  return error(`Cortex backend error ${res.status} on cortex.${tool}: ${detail}`);
}
