import type { MCPServer } from "mcp-use";
import { toolStatus } from "./backend.js";

// Hosted-endpoint gate (#105). With CORTEX_REQUIRE_AGENT_KEY=1 every request to /mcp must carry an
// active Cortex agent key, as `Authorization: Bearer ck_...` or `X-Cortex-Agent-Key: ck_...`. The key is
// checked with the backend's verify_agent_key tool (it stores only the key's SHA-256). That goes through
// /api/tools like every other call, so it works directly, over a tunnel and through the runner relay. Unset (local dev),
// /mcp stays keyless and calls without a key belong to the backend's local owner.
// A key only says who the agent acts for: training approval and the real arm stay human clicks.
export const REQUIRE_AGENT_KEY = /^(1|true|yes)$/i.test(process.env.CORTEX_REQUIRE_AGENT_KEY ?? "");
const OK_TTL_MS = 60_000;
const BAD_TTL_MS = 10_000;

/** key -> {ok, until}: one backend round trip per key per minute, not per request. */
const verdicts = new Map<string, { ok: boolean; until: number; detail?: string }>();

/** The agent key a request carries, if any (same rules as identityOf in backend.ts). */
export function agentKeyOf(headers: Headers): string | undefined {
  const own = headers.get("x-cortex-agent-key")?.trim();
  const bearer = headers.get("authorization")?.match(/^Bearer\s+(ck_\S+)$/i)?.[1];
  return own || bearer || undefined;
}

async function verify(key: string): Promise<{ ok: boolean; detail?: string }> {
  const hit = verdicts.get(key);
  if (hit && hit.until > Date.now()) return hit;
  const r = await toolStatus("verify_agent_key", {}, { agentKey: key });
  if ("error" in r) throw new Error(r.error);
  const data = r.data as { code?: string; detail?: string };
  const res = { ok: r.status >= 200 && r.status < 300, status: r.status };
  if (res.ok) {
    verdicts.set(key, { ok: true, until: Date.now() + OK_TTL_MS });
    return { ok: true };
  }
  if (res.status === 403 && data.code === "agent_key_invalid") {
    const v = { ok: false, until: Date.now() + BAD_TTL_MS, detail: data.detail };
    verdicts.set(key, v);
    if (verdicts.size > 5000) verdicts.delete(verdicts.keys().next().value as string);
    return v;
  }
  throw new Error(`backend answered ${res.status} to the key check`);
}

function refuse(status: number, message: string, code: number, extra: Record<string, string> = {}): Response {
  return new Response(JSON.stringify({ jsonrpc: "2.0", id: null, error: { code, message } }), {
    status,
    headers: { "Content-Type": "application/json", ...extra },
  });
}

const CHALLENGE = {
  "WWW-Authenticate":
    'Bearer realm="cortex", error="invalid_token", error_description="Send a Cortex agent key (ck_...) from the studio\'s Agents page"',
};

/** Install the /mcp gate when CORTEX_REQUIRE_AGENT_KEY=1. Call before trackClients so refused requests get no session. */
export function requireAgentKey(server: MCPServer) {
  if (!REQUIRE_AGENT_KEY) return;
  server.use("/mcp", async (c, next) => {
    if (c.req.method === "OPTIONS") return next(); // CORS preflight carries no credentials
    const key = agentKeyOf(c.req.raw.headers);
    if (!key) {
      return refuse(401, "This Cortex MCP server requires an agent key. Create one in the Cortex studio (Agents → Settings · Agent keys) and send it as `Authorization: Bearer ck_...` or `X-Cortex-Agent-Key: ck_...`.", -32001, CHALLENGE);
    }
    let v: { ok: boolean; detail?: string };
    try {
      v = await verify(key);
    } catch (e) {
      return refuse(503, `Could not check the agent key with the Cortex backend: ${(e as Error).message}`, -32002);
    }
    if (!v.ok) return refuse(401, `Cortex refused this agent key: ${v.detail ?? "unknown or revoked"}`, -32001, CHALLENGE);
    return next();
  });
}
