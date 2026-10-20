import type { CallToolResult, MCPServer } from "mcp-use";
import { relayCall, relayEnabled, relayGet, runnerConnected } from "./relay.js";

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
// Agent key (ck_...) from the studio's Settings -> Agents: ties these calls to a person (#105).
// A client can also send its own with `--header "X-Cortex-Agent-Key: ck_..."` (or `Authorization: Bearer ck_...`).
const AGENT_KEY = process.env.CORTEX_AGENT_KEY;
// Same check as src/auth.ts. When the hosted gate is on, a request that sent no key must not be
// attributed to the operator's CORTEX_AGENT_KEY (#152 finding 6). auth.ts imports this module, so
// the env check is repeated here instead of imported.
const REQUIRE_AGENT_KEY = /^(1|true|yes)$/i.test(process.env.CORTEX_REQUIRE_AGENT_KEY ?? "");

/** Who is calling, forwarded on every backend call so the studio can show the agent's session live. */
export type Identity = { sessionId?: string; client?: string; agentKey?: string; project?: string };

type ClientInfo = { name?: unknown; version?: unknown } | undefined;
const fmt = (i: ClientInfo) =>
  i && typeof i.name === "string" && i.name ? (typeof i.version === "string" && i.version ? `${i.name}/${i.version}` : i.name) : undefined;

/** MCP session id -> "name/version" from that connection's `initialize` (legacy clients send it only there). */
const clients = new Map<string, string>();

/**
 * Give each connection a session id and remember its clientInfo.
 *
 * mcp-use v2 serves every request statelessly and assigns no `Mcp-Session-Id`, so on a 2025-era
 * `initialize` this middleware assigns one (`cx-<uuid>`) in the response header. Streamable-HTTP
 * clients (Claude Code, Cursor, VS Code, the Inspector) then send it on every later request, which
 * is what lets the studio group one connection's calls into a session.
 */
export function trackClients(server: MCPServer) {
  server.use("/mcp", async (c, next) => {
    let info: string | undefined;
    let isInit = false;
    if (c.req.method === "POST") {
      try {
        const body: unknown = await c.req.raw.clone().json();
        const init = (Array.isArray(body) ? body : [body]).find((m) => (m as { method?: string })?.method === "initialize") as
          | { params?: { clientInfo?: ClientInfo } }
          | undefined;
        isInit = init !== undefined;
        info = fmt(init?.params?.clientInfo);
      } catch {
        /* not JSON: nothing to learn */
      }
    }
    await next();
    let sid = c.res.headers.get("mcp-session-id");
    if (isInit && !sid && c.res.ok) {
      sid = `cx-${crypto.randomUUID()}`;
      const headers = new Headers(c.res.headers);
      headers.set("mcp-session-id", sid);
      c.res = new Response(c.res.body, { status: c.res.status, statusText: c.res.statusText, headers });
    }
    if (info && sid) {
      clients.set(sid, info);
      if (clients.size > 5000) clients.delete(clients.keys().next().value as string);
    }
  });
}

type ToolCtx = {
  request?: { header(name: string): string | undefined };
  client: { info(): ClientInfo };
};

/** Identity for one tool call: MCP session id, clientInfo (per-request envelope, else from initialize), agent key. */
export function identityOf(ctx: ToolCtx): Identity {
  const header = (n: string) => ctx.request?.header(n)?.trim() || undefined;
  const bearer = header("authorization")?.match(/^Bearer\s+(ck_\S+)$/i)?.[1];
  const presented = header("x-cortex-agent-key") ?? bearer;
  const agentKey = presented ?? (REQUIRE_AGENT_KEY ? undefined : AGENT_KEY);
  let sessionId = header("mcp-session-id");
  const client = fmt(ctx.client.info()) ?? (sessionId ? clients.get(sessionId) : undefined);
  // 2026-era clients have no session at all: group their calls per client (and key) instead.
  if (!sessionId && client) sessionId = `client-${client}-${agentKey ? agentKey.slice(0, 7) : "local"}`;
  return { sessionId, client, agentKey, project: header("x-cortex-project") };
}

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

export function backendHeaders(accept = "application/json"): Record<string, string> {
  const headers: Record<string, string> = { "Content-Type": "application/json", Accept: accept, "X-Cortex-Actor": "mcp" };
  if (MCP_KEY) headers.Authorization = `Bearer ${MCP_KEY}`;
  return headers;
}

/**
 * GET {API_URL}{path} as JSON (resources and stage hints). Read-only. Goes through the runner when one is
 * connected (a hosted server cannot reach the backend's REST routes otherwise), else directly.
 * Throws on a non-2xx status.
 */
export async function getJson<T = unknown>(path: string, signal?: AbortSignal): Promise<T> {
  let status: number;
  let raw: string;
  if (relayEnabled() && runnerConnected()) {
    const { Authorization: _auth, ...headers } = backendHeaders();
    ({ status, raw } = await relayGet(path, headers, signal));
  } else {
    const res = await fetch(`${API_URL}${path}`, { headers: backendHeaders(), signal });
    ({ status, raw } = { status: res.status, raw: await res.text() });
  }
  if (status < 200 || status >= 300) throw new Error(`GET ${path} -> ${status}: ${raw.slice(0, 300)}`);
  return JSON.parse(raw) as T;
}

const RUNNER_HINT =
  "Start the local runner next to your backend: `scripts/dev.sh --runner <this server's URL>` or " +
  "`cd backend && uv run python -m cortex.mcp_runner --server <this server's URL>` (same CORTEX_MCP_KEY on both).";

type Transport = { status: number; raw: string } | { error: CallToolResult };

async function send(tool: string, args: Json, signal?: AbortSignal, who: Identity = {}): Promise<Transport> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    Accept: "application/json",
    "X-Cortex-Actor": "mcp",
    // This server's channel: the backend hands MCP App view approval nonces only to it (it keeps them out of the
    // model's view). With CORTEX_MCP_KEY set the backend checks the key instead (#354).
    "X-Cortex-View-Channel": "1",
  };
  // Agent identity (#105) travels on both transports, so the studio can show this agent's session live.
  if (who.agentKey) headers["X-Cortex-Agent-Key"] = who.agentKey;
  if (who.sessionId) headers["X-Cortex-Agent-Session"] = who.sessionId;
  if (who.client) headers["X-Cortex-Client"] = who.client;
  if (who.project) headers["X-Cortex-Project"] = who.project; // the studio chat's project, when it launched this agent
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

/** Raw status and JSON body of a registry tool call, over the same transport as callTool (runner or direct). */
export async function toolStatus(tool: string, args: Json, who: Identity = {}): Promise<{ status: number; data: Json } | { error: string }> {
  const sent = await send(tool, args, undefined, who);
  if ("error" in sent) {
    const first = sent.error.content?.[0];
    return { error: first && first.type === "text" ? first.text : `cortex.${tool} could not reach the backend` };
  }
  let data: unknown;
  try {
    data = sent.raw ? JSON.parse(sent.raw) : {};
  } catch {
    data = {};
  }
  return { status: sent.status, data: (data && typeof data === "object" && !Array.isArray(data) ? data : {}) as Json };
}

/** POST /tools/{tool} (directly or through the runner) and turn the response into an MCP tool result. */
export async function callTool(tool: string, args: Json, signal?: AbortSignal, who: Identity = {}): Promise<CallToolResult> {
  const sent = await send(tool, args, signal, who);
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
  if (res.status === 403 && data.approval_required === true) {
    // launch_training_job's args have no project_id: the backend names the job's project, so the link opens it (#359)
    const project = args.project_id ?? (typeof data.project_id === "string" ? data.project_id : undefined);
    return approvalRequired(tool, project === undefined ? args : { ...args, project_id: project }, detail);
  }
  if (res.status === 403) {
    return error(`Cortex backend refused access (403): ${detail}. When the backend is reached through a tunnel, set CORTEX_MCP_KEY on this MCP server to the backend's CORTEX_MCP_KEY.`);
  }
  if (res.status === 404) return error(`Not found (404): ${detail}. Check the id.`);
  if (res.status === 422) return error(`Invalid arguments for cortex.${tool} (422): ${detail}`);
  return error(`Cortex backend error ${res.status} on cortex.${tool}: ${detail}`);
}
