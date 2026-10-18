import { useEffect, useRef, useState } from "react";
import { ApiError, API_BASE } from "../api/client";

/** Agent observability API (#105): /api/agents/*. Scoped to the signed-in user (or the local owner). */

export interface AgentSession {
  id: string;
  mcp_session_id?: string;
  key_id?: string | null;
  client?: string | null;
  client_version?: string | null;
  first_seen: string;
  last_seen: string;
  call_count: number;
  projects: string[];
  last_tool?: string;
  status: "live" | "idle";
  live: boolean;
}

export type CallStatus = "running" | "ok" | "error" | "approval_required";

export interface AgentCall {
  id: string;
  tool: string;
  args: Record<string, unknown>;
  project_id?: string | null;
  session_id?: string | null;
  client?: string | null;
  status: CallStatus;
  started_at: string;
  ended_at?: string;
  duration_ms?: number;
  summary?: string;
  error?: string;
  result_preview?: string;
  images?: string[];
  image_urls?: string[];
  needs_approval?: boolean;
}

export interface AgentKey {
  id: string;
  label: string;
  prefix: string;
  status: "active" | "revoked";
  created_at: string;
  revoked_at?: string | null;
  key?: string; // only in the create response
}

export interface Me {
  mode: "local" | "insforge";
  signed_in: boolean;
  user: { id: string; email?: string | null; name?: string | null } | null;
  note?: string;
}

const TOKEN = "cortex.userToken";
export const token = {
  get: (): string | null => {
    try {
      return localStorage.getItem(TOKEN);
    } catch {
      return null;
    }
  },
  set: (t: string | null) => {
    try {
      if (t) localStorage.setItem(TOKEN, t);
      else localStorage.removeItem(TOKEN);
    } catch {
      /* storage unavailable */
    }
  },
};

async function req<T>(method: string, path: string, body?: unknown): Promise<T> {
  const headers: Record<string, string> = { "X-Cortex-Actor": "gui" };
  const t = token.get();
  if (t) headers["X-Cortex-User-Token"] = t;
  if (body !== undefined) headers["Content-Type"] = "application/json";
  let res: Response;
  try {
    res = await fetch(`${API_BASE}/agents${path}`, { method, headers, body: body === undefined ? undefined : JSON.stringify(body) });
  } catch {
    throw new ApiError(`Cannot reach the Cortex API at ${API_BASE}`, 0);
  }
  const data = await res.json().catch(() => undefined);
  if (!res.ok) {
    const detail = (data as { detail?: unknown } | undefined)?.detail;
    throw new ApiError(typeof detail === "string" ? detail : `${method} ${path} failed (${res.status})`, res.status, detail, data);
  }
  return data as T;
}

export const agentsApi = {
  me: () => req<Me>("GET", "/me"),
  login: (email: string, password: string) => req<{ access_token: string | null }>("POST", "/login", { email, password }),
  sessions: () => req<AgentSession[]>("GET", "/sessions"),
  calls: (sid: string) => req<AgentCall[]>("GET", `/sessions/${sid}/calls`),
  keys: () => req<AgentKey[]>("GET", "/keys"),
  createKey: (label: string) => req<AgentKey>("POST", "/keys", { label }),
  revokeKey: (id: string) => req<AgentKey>("DELETE", `/keys/${id}`),
};

export type AgentEvent = { event: "session_start" | "tool_start" | "tool_end" | "approval_required" | "session_idle"; data: AgentCall };

/** The user-scoped SSE stream. Returns whether it is connected; `onEvent` gets every event. */
export function useAgentEvents(onEvent: (e: AgentEvent) => void, enabled = true): boolean {
  const [open, setOpen] = useState(false);
  const cb = useRef(onEvent);
  cb.current = onEvent;
  useEffect(() => {
    if (!enabled) return;
    const t = token.get();
    const es = new EventSource(`${API_BASE}/agents/events${t ? `?token=${encodeURIComponent(t)}` : ""}`);
    es.addEventListener("hello", () => setOpen(true));
    es.onerror = () => setOpen(false);
    for (const ev of ["session_start", "tool_start", "tool_end", "approval_required", "session_idle"] as const) {
      es.addEventListener(ev, (m) => {
        try {
          cb.current({ event: ev, data: JSON.parse((m as MessageEvent).data) });
        } catch {
          /* malformed event */
        }
      });
    }
    return () => es.close();
  }, [enabled]);
  return open;
}
