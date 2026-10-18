import type { CallToolResult, MCPServer } from "mcp-use";

// The Cortex backend's tool registry is the single source of truth:
//   GET  {CORTEX_API_URL}/tools        -> tool names + JSON Schemas
//   POST {CORTEX_API_URL}/tools/{name} -> {summary, ref_ids, ...data}
// A hosted MCP server cannot reach localhost, so on Manufact CORTEX_API_URL
// must be a public URL (e.g. a cloudflared tunnel) and CORTEX_MCP_KEY must
// match the backend's key.
export const API_URL = (process.env.CORTEX_API_URL ?? "http://127.0.0.1:8000/api").replace(/\/+$/, "");
export const STUDIO_URL = (process.env.CORTEX_STUDIO_URL ?? "http://localhost:5173").replace(/\/+$/, "");
const MCP_KEY = process.env.CORTEX_MCP_KEY;
// Agent key (ck_...) from the studio's Settings -> Agents: ties these calls to a person (#105).
// A client can also send its own with `--header "X-Cortex-Agent-Key: ck_..."` (or `Authorization: Bearer ck_...`).
const AGENT_KEY = process.env.CORTEX_AGENT_KEY;

/** Who is calling, forwarded on every backend call so the studio can show the agent's session live. */
export type Identity = { sessionId?: string; client?: string; agentKey?: string };

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
  const agentKey = header("x-cortex-agent-key") ?? bearer ?? AGENT_KEY;
  let sessionId = header("mcp-session-id");
  const client = fmt(ctx.client.info()) ?? (sessionId ? clients.get(sessionId) : undefined);
  // 2026-era clients have no session at all: group their calls per client (and key) instead.
  if (!sessionId && client) sessionId = `client-${client}-${agentKey ? agentKey.slice(0, 7) : "local"}`;
  return { sessionId, client, agentKey };
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

/** POST /tools/{tool} and turn the response into an MCP tool result. */
export async function callTool(tool: string, args: Json, signal?: AbortSignal, who: Identity = {}): Promise<CallToolResult> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    Accept: "application/json",
    "X-Cortex-Actor": "mcp",
  };
  if (MCP_KEY) headers.Authorization = `Bearer ${MCP_KEY}`; // tunnel guard, unrelated to the agent key
  if (who.agentKey) headers["X-Cortex-Agent-Key"] = who.agentKey;
  if (who.sessionId) headers["X-Cortex-Agent-Session"] = who.sessionId;
  if (who.client) headers["X-Cortex-Client"] = who.client;

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
  if (res.status === 403 && data.code === "agent_key_invalid") {
    return error(`Cortex refused this agent key: ${detail}`);
  }
  if (res.status === 403) {
    return error(`Cortex backend refused access (403): ${detail}. When the backend is reached through a tunnel, set CORTEX_MCP_KEY on this MCP server to the backend's CORTEX_MCP_KEY.`);
  }
  if (res.status === 404) return error(`Not found (404): ${detail}. Check the id.`);
  if (res.status === 422) return error(`Invalid arguments for cortex.${tool} (422): ${detail}`);
  return error(`Cortex backend error ${res.status} on cortex.${tool}: ${detail}`);
}
