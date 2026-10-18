import { randomUUID, timingSafeEqual } from "node:crypto";
import type { MCPServer } from "mcp-use";

// Runner relay: lets a hosted MCP server (Manufact) reach a backend on a
// developer's laptop with no tunnel and no inbound port. The local runner
// (backend/cortex/mcp_runner.py) dials OUT to this server and long-polls:
//
//   POST /relay/poll    Bearer CORTEX_MCP_KEY  -> 200 {calls:[{id, tool, args, headers}]} | 204 after ~25 s
//   POST /relay/result  Bearer CORTEX_MCP_KEY  <- {id, status, body}   (the backend's HTTP status + raw body)
//
// Plain HTTPS long-polling instead of a WebSocket: mcp-use's Hono app has no
// upgrade hook, and it passes through any proxy in front of the hosted server.
// The relay is off unless CORTEX_MCP_KEY is set on this server.

const KEY = process.env.CORTEX_MCP_KEY;
const HOLD_MS = 25_000; // how long one poll waits for work
// A live runner re-polls within milliseconds, so it always has a poll open; a short grace covers the gap.
const SEEN_MS = 5_000;
const PICKUP_MS = HOLD_MS + 5_000; // a queued call nobody takes means the runner is gone
const CALL_TIMEOUT_MS = 15 * 60_000; // backend wait_s tops out at 900 s

type Json = Record<string, unknown>;
export interface RelayCall {
  id: string;
  tool: string;
  args: Json;
  headers: Record<string, string>;
}
export interface RelayResponse {
  status: number;
  raw: string;
}

const queue: RelayCall[] = [];
const pending = new Map<string, { resolve: (r: RelayResponse) => void; reject: (e: Error) => void }>();
const waiters = new Set<() => void>();
let lastPoll = 0;
let activePolls = 0;

/** True while a runner has a poll open, or polled within the last SEEN_MS. */
export function runnerConnected(): boolean {
  return activePolls > 0 || Date.now() - lastPoll < SEEN_MS;
}

export function relayEnabled(): boolean {
  return Boolean(KEY);
}

/** Queue a tool call for the runner and wait for the backend's response. */
export function relayCall(tool: string, args: Json, headers: Record<string, string>, signal?: AbortSignal): Promise<RelayResponse> {
  const call: RelayCall = { id: randomUUID(), tool, args, headers };
  return new Promise<RelayResponse>((resolve, reject) => {
    const done = () => {
      clearTimeout(timer);
      clearTimeout(pickup);
      signal?.removeEventListener("abort", onAbort);
      pending.delete(call.id);
      const i = queue.indexOf(call);
      if (i >= 0) queue.splice(i, 1);
    };
    const onAbort = () => {
      done();
      reject(new Error("tool call cancelled"));
    };
    const timer = setTimeout(() => {
      done();
      reject(new Error(`the runner did not answer within ${CALL_TIMEOUT_MS / 60_000} min`));
    }, CALL_TIMEOUT_MS);
    const pickup = setTimeout(() => {
      if (!queue.includes(call)) return;
      done();
      reject(new Error(`no runner picked the call up within ${PICKUP_MS / 1000} s; it seems to have disconnected`));
    }, PICKUP_MS);
    if (signal?.aborted) return onAbort();
    signal?.addEventListener("abort", onAbort, { once: true });
    pending.set(call.id, {
      resolve: (r) => (done(), resolve(r)),
      reject: (e) => (done(), reject(e)),
    });
    queue.push(call);
    for (const wake of waiters) wake();
  });
}

function authorized(header: string | undefined): boolean {
  if (!KEY || !header?.startsWith("Bearer ")) return false;
  const got = Buffer.from(header.slice(7));
  const want = Buffer.from(KEY);
  return got.length === want.length && timingSafeEqual(got, want);
}

/** Wait until there is work, the hold expires, or the runner hangs up. */
function nextCalls(abort: AbortSignal): Promise<RelayCall[]> {
  if (queue.length > 0) return Promise.resolve(queue.splice(0));
  return new Promise((resolve) => {
    const finish = () => {
      clearTimeout(timer);
      waiters.delete(finish);
      abort.removeEventListener("abort", finish);
      resolve(abort.aborted ? [] : queue.splice(0));
    };
    const timer = setTimeout(finish, HOLD_MS);
    waiters.add(finish);
    abort.addEventListener("abort", finish, { once: true });
  });
}

export function registerRelay(server: MCPServer): void {
  server.post("/relay/poll", async (c) => {
    if (!KEY) return c.json({ detail: "Runner relay is disabled: set CORTEX_MCP_KEY on the MCP server." }, 503);
    if (!authorized(c.req.header("authorization"))) return c.json({ detail: "Invalid runner key" }, 401);
    activePolls++;
    lastPoll = Date.now();
    try {
      const calls = await nextCalls(c.req.raw.signal);
      if (c.req.raw.signal.aborted) {
        queue.unshift(...calls); // hand them to the next poll instead of losing them
        return c.body(null, 204);
      }
      return calls.length > 0 ? c.json({ calls }) : c.body(null, 204);
    } finally {
      activePolls--;
      lastPoll = Date.now();
    }
  });

  server.post("/relay/result", async (c) => {
    if (!KEY) return c.json({ detail: "Runner relay is disabled: set CORTEX_MCP_KEY on the MCP server." }, 503);
    if (!authorized(c.req.header("authorization"))) return c.json({ detail: "Invalid runner key" }, 401);
    lastPoll = Date.now();
    const msg = (await c.req.json().catch(() => ({}))) as Json;
    const call = typeof msg.id === "string" ? pending.get(msg.id) : undefined;
    if (!call) return c.json({ detail: "unknown or expired call id" }, 404);
    if (typeof msg.error === "string") call.reject(new Error(msg.error));
    else call.resolve({ status: Number(msg.status) || 502, raw: typeof msg.body === "string" ? msg.body : "" });
    return c.json({ ok: true });
  });
}
