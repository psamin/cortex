import { useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Button, ErrorState, Field, Input, Panel, Segmented, SkeletonRows } from "../components/ui";
import { ago, cn } from "../lib/format";
import { agentsApi, token, useAgentEvents, type AgentCall, type AgentEvent, type AgentSession } from "./api";
import { KeysPanel } from "./Keys";
import { Timeline } from "./Timeline";
import { NowBar, WorkBoard, useSessionWork } from "./Work";

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

function SessionRow({ s, active }: { s: AgentSession; active: boolean }) {
  return (
    <li>
      <Link
        to={`/studio/agents?session=${s.id}`}
        className={cn("block rounded-sm border px-3 py-2.5 transition-colors", active ? "border-ink bg-plate" : "border-transparent hover:bg-rail")}
      >
        <div className="flex items-center gap-2">
          <span className={cn("h-2 w-2 shrink-0 rounded-full", s.live ? "animate-pulse bg-pass" : "bg-ink-4")} title={s.live ? "live" : "idle"} />
          <span className="truncate text-[13.5px] text-ink">{s.client ?? "unknown client"}</span>
          {s.client_version && <span className="font-mono text-[11px] text-ink-3">{s.client_version}</span>}
          <span className={cn("ml-auto shrink-0 text-ink-3", label)}>{s.live ? "live" : "idle"}</span>
        </div>
        <div className="mt-1 flex flex-wrap gap-x-3 text-xs text-ink-3">
          <span>
            {s.last_tool ? <span className="font-mono">{s.last_tool}</span> : "no calls"} · {ago(s.last_seen)}
          </span>
          <span>{s.call_count} calls</span>
          {s.projects.length > 0 && <span className="font-mono">{s.projects.slice(-2).join(", ")}{s.projects.length > 2 ? ` +${s.projects.length - 2}` : ""}</span>}
        </div>
      </Link>
    </li>
  );
}

function Sessions() {
  const qc = useQueryClient();
  const [params] = useSearchParams();
  const sessions = useQuery({ queryKey: ["agents", "sessions"], queryFn: agentsApi.sessions, refetchInterval: 60_000 });
  const selected = params.get("session") ?? sessions.data?.[0]?.id;
  const calls = useQuery({ queryKey: ["agents", "calls", selected], queryFn: () => agentsApi.calls(selected!), enabled: !!selected });

  const live = useAgentEvents((e: AgentEvent) => {
    if (e.event === "session_start" || e.event === "session_idle" || e.event === "tool_end") void qc.invalidateQueries({ queryKey: ["agents", "sessions"] });
    if ((e.event === "tool_start" || e.event === "tool_end") && e.data.session_id) {
      // Merge the call into its session's timeline: start rows show live, the end row replaces it.
      qc.setQueryData<AgentCall[]>(["agents", "calls", e.data.session_id], (old) => {
        if (!old) return old;
        const i = old.findIndex((c) => c.id === e.data.id);
        if (i < 0) return [...old, e.data];
        const next = old.slice();
        next[i] = { ...old[i], ...e.data };
        return next;
      });
      if (e.data.session_id !== selected) void qc.invalidateQueries({ queryKey: ["agents", "calls", e.data.session_id] });
    }
  });

  const current = sessions.data?.find((s) => s.id === selected);
  const { work, loading } = useSessionWork(current?.projects ?? []);
  return (
    <div className="grid gap-6 lg:grid-cols-[320px_minmax(0,1fr)]">
      <Panel title="Sessions" meta={<span className="inline-flex items-center gap-1.5"><span className={cn("h-1.5 w-1.5 rounded-full", live ? "bg-pass" : "bg-caution")} />{live ? "live tail on" : "connecting"}</span>} bodyClassName="p-2">
        {sessions.isLoading ? (
          <SkeletonRows rows={3} />
        ) : sessions.error ? (
          <ErrorState error={sessions.error} onRetry={() => sessions.refetch()} compact />
        ) : !sessions.data?.length ? (
          <div className="space-y-2 p-2 text-xs leading-5 text-ink-3">
            <p>No agent sessions yet.</p>
            <p>
              Connect Claude Code (or any MCP client) to the Cortex MCP server; every call shows up here live. Ask it for{" "}
              <span className="font-mono">cortex.get_session_link</span> to jump straight to its session.
            </p>
          </div>
        ) : (
          <ul className="space-y-1">
            {sessions.data.map((s) => (
              <SessionRow key={s.id} s={s} active={s.id === selected} />
            ))}
          </ul>
        )}
      </Panel>
      <section className="min-w-0">
        {current && (
          <div className="mb-4 flex flex-wrap items-baseline gap-x-4 gap-y-1">
            <h2 className="kn-display text-[24px]">{current.client ?? "Agent"} session</h2>
            <span className={cn("text-ink-3", label)}>
              started {ago(current.first_seen)} · {current.call_count} calls
            </span>
            {current.projects.map((p) => (
              <Link key={p} to={`/p/${p}`} className="font-mono text-xs text-ink-2 underline-offset-2 hover:text-ink hover:underline">
                {p}
              </Link>
            ))}
          </div>
        )}
        {current && <NowBar session={current} calls={calls.data ?? []} work={work} />}
        {!selected ? null : (
          <div className="grid gap-6 2xl:grid-cols-[minmax(0,1fr)_420px]">
            {current && <WorkBoard session={current} work={work} loading={loading} />}
            <div className="min-w-0">
              <h2 className={cn("mb-2.5 text-ink", label)}>Tool calls</h2>
              {calls.isLoading ? (
                <SkeletonRows rows={4} />
              ) : calls.error ? (
                <ErrorState error={calls.error} onRetry={() => calls.refetch()} />
              ) : (
                <Timeline calls={calls.data ?? []} />
              )}
            </div>
          </div>
        )}
      </section>
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
          <h1 className="kn-display text-[clamp(1.9rem,3.6vw,2.6rem)]">What your agents are doing</h1>
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
                { value: "sessions", label: "Sessions" },
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
