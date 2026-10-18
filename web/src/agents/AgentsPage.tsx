import { useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Button, ErrorState, Field, Input, Panel, Segmented, SkeletonRows } from "../components/ui";
import { ago, cn } from "../lib/format";
import { agentsApi, token, useAgentEvents, type AgentEvent, type AgentSession } from "./api";
import { KeysPanel } from "./Keys";
import { GlowEdge } from "../components/ui/spotlight-card";
import { useProjects } from "../api/hooks";

const label = "font-mono text-[10.5px] uppercase tracking-[0.14em]";

function SignIn({ onDone }: { onDone: () => void }) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const login = useMutation({
    mutationFn: () => agentsApi.login(email, password),
    onSuccess: (r) => {
      token.set(r.access_token);
      onDone();
    },
  });
  return (
    <Panel title="Sign in" meta="InsForge account" className="mx-auto max-w-sm">
      <form
        className="space-y-3"
        onSubmit={(e) => {
          e.preventDefault();
          login.mutate();
        }}
      >
        <Field label="Email">
          <Input type="email" autoComplete="email" value={email} onChange={(e) => setEmail(e.target.value)} required />
        </Field>
        <Field label="Password">
          <Input type="password" autoComplete="current-password" value={password} onChange={(e) => setPassword(e.target.value)} required />
        </Field>
        {login.error && <ErrorState error={login.error} compact />}
        <Button type="submit" variant="primary" loading={login.isPending}>
          Sign in
        </Button>
      </form>
    </Panel>
  );
}

function SessionCard({ s, names }: { s: AgentSession; names: Record<string, string> }) {
  return (
    <li className={cn("sk-depth relative rounded border bg-plate p-4", s.live ? "border-servo/60" : "border-line-strong")}>
      <GlowEdge />
      <div className="flex items-center gap-2">
        <span className={cn("h-2 w-2 shrink-0 rounded-full", s.live ? "live-dot bg-pass" : "bg-ink-4")} title={s.live ? "live" : "idle"} />
        <span className="truncate text-[14px] text-ink">{s.client ?? "unknown client"}</span>
        {s.client_version && <span className="font-mono text-[11px] text-ink-3">{s.client_version}</span>}
        <span className={cn("ml-auto shrink-0 text-ink-3", label)}>{s.live ? "live" : "idle"}</span>
      </div>
      <p className="mt-1.5 text-xs text-ink-3">
        {s.last_tool ? <span className="font-mono text-ink-2">{s.last_tool}</span> : "no calls yet"} · {ago(s.last_seen)} · {s.call_count} calls
      </p>
      {s.projects.length > 0 ? (
        <ul className="mt-3 space-y-1.5">
          {s.projects.map((p) => (
            <li key={p} className="flex items-center justify-between gap-3 rounded-sm border border-line px-2.5 py-1.5">
              <span className="min-w-0 truncate text-[13px] text-ink-2">{names[p] ?? <span className="font-mono text-xs">{p}</span>}</span>
              <Link to={`/p/${p}?view=log`} className="shrink-0 text-xs font-medium text-ink underline underline-offset-2 hover:text-servo">
                Open log
              </Link>
            </li>
          ))}
        </ul>
      ) : (
        <p className="mt-3 text-xs text-ink-3">Not working on a project yet.</p>
      )}
    </li>
  );
}

/**
 * Which agents are connected and what they are working on. The detail (every call, demos, training,
 * evaluations, RL) lives in each project's Agent log; this page only routes you there.
 */
function Sessions() {
  const qc = useQueryClient();
  const sessions = useQuery({ queryKey: ["agents", "sessions"], queryFn: agentsApi.sessions, refetchInterval: 30_000 });
  const projects = useProjects();
  const names = Object.fromEntries((projects.data ?? []).map((p) => [p.id, p.name]));
  const live = useAgentEvents((e: AgentEvent) => {
    if (e.event === "session_start" || e.event === "session_idle" || e.event === "tool_end") void qc.invalidateQueries({ queryKey: ["agents", "sessions"] });
  });
  const list = sessions.data ?? [];
  const liveCount = list.filter((s) => s.live).length;
  return (
    <div className="space-y-4">
      <p className="max-w-[46rem] text-sm leading-6 text-ink-2">
        Every agent connected to Cortex over MCP. To watch one work (its calls, demos, training, evaluations and RL), open the project's{" "}
        <span className="text-ink">Agent log</span>.
        <span className={cn("ml-2 inline-flex items-center gap-1.5 align-middle text-ink-3", label)}>
          <span className={cn("h-1.5 w-1.5 rounded-full", live ? "bg-pass" : "bg-caution")} />
          {live ? `${liveCount} live` : "connecting"}
        </span>
      </p>
      {sessions.isLoading ? (
        <SkeletonRows rows={3} />
      ) : sessions.error ? (
        <ErrorState error={sessions.error} onRetry={() => sessions.refetch()} />
      ) : !list.length ? (
        <Panel title="No agents yet">
          <p className="text-sm leading-6 text-ink-2">
            Connect Claude Code (or any MCP client): <span className="font-mono text-xs">claude mcp add --transport http cortex http://localhost:3000/mcp</span>. It
            shows up here the moment it calls a tool.
          </p>
        </Panel>
      ) : (
        <ul className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
          {list.map((s) => (
            <SessionCard key={s.id} s={s} names={names} />
          ))}
        </ul>
      )}
    </div>
  );
}

/** /studio/agents: what your MCP agents are doing, live; and Settings → Agents (keys). */
export function AgentsPage() {
  const qc = useQueryClient();
  const [params, setParams] = useSearchParams();
  const tab = params.get("tab") === "keys" ? "keys" : "sessions";
  const me = useQuery({ queryKey: ["agents", "me"], queryFn: agentsApi.me });

  return (
    <div className="h-full overflow-y-auto scroll-thin">
      <div className="mx-auto max-w-[1560px] px-4 pb-16 pt-8 sm:px-6">
        <div className="kn-label kn-label-rule">
          <span className="text-ink">Agents</span>
        </div>
        <div className="mt-4 flex flex-wrap items-end gap-4">
          <h1 className="kn-display text-[clamp(1.9rem,3.6vw,2.6rem)]">Connected agents</h1>
          <div className="ml-auto flex items-center gap-3">
            {me.data && (
              <span className={cn("text-ink-3", label)} title={me.data.note}>
                {me.data.mode === "local" ? "local mode, single user" : me.data.user ? `signed in · ${me.data.user.email ?? me.data.user.id}` : "InsForge · signed out"}
              </span>
            )}
            {me.data?.mode === "insforge" && me.data.signed_in && (
              <Button size="sm" variant="ghost" onClick={() => { token.set(null); void qc.invalidateQueries({ queryKey: ["agents"] }); }}>
                Sign out
              </Button>
            )}
            <Segmented
              ariaLabel="View"
              value={tab}
              onChange={(v) => setParams(v === "keys" ? { tab: "keys" } : {})}
              options={[
                { value: "sessions", label: "Agents" },
                { value: "keys", label: "Settings · Agent keys" },
              ]}
            />
          </div>
        </div>
        <div className="mt-6">
          {me.isLoading ? (
            <SkeletonRows rows={4} />
          ) : me.error ? (
            <ErrorState error={me.error} onRetry={() => me.refetch()} />
          ) : me.data && !me.data.signed_in ? (
            <SignIn onDone={() => void qc.invalidateQueries({ queryKey: ["agents"] })} />
          ) : tab === "keys" ? (
            <KeysPanel />
          ) : (
            <Sessions />
          )}
        </div>
      </div>
    </div>
  );
}
