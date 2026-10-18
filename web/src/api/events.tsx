import { createContext, useContext, useEffect, useMemo, useRef, useSyncExternalStore, type ReactNode } from "react";
import { useQueryClient, type QueryClient } from "@tanstack/react-query";
import { api } from "./client";
import type { AgentAttempt, AgentStepEvent, AutopilotRun, CollectionLive, Condition, EntityEvent, Job, LogEvent, Scenario, TraceStep } from "./types";

type StreamStatus = "connecting" | "open" | "error";

export type LiveItem = { kind: "step"; step: TraceStep } | { kind: "thought"; text: string };

export interface LiveAttempt {
  index: number;
  seed?: number;
  scenario?: Scenario;
  condition?: Condition;
  items: LiveItem[];
  /** Set when the `agent_attempt` event for this index arrives (simulator verdict). */
  result?: AgentAttempt;
}

/** What this browser has seen of a collection over SSE (events are not replayed on reload). */
export interface LiveCollection {
  total: number;
  lastIndex: number;
  attempts: Record<number, LiveAttempt>;
}

interface Snapshot {
  status: StreamStatus;
  jobs: Record<string, Job>;
  logs: Record<string, string[]>;
  lastEntityAt: number;
  agent: Record<string, LiveCollection>;
}

const MAX_ITEMS_PER_ATTEMPT = 400;

const MAX_LOG_LINES = 2000;

/** Tiny external store for one project's SSE stream (jobs + logs). */
class ProjectEventStore {
  private snap: Snapshot = { status: "connecting", jobs: {}, logs: {}, lastEntityAt: 0, agent: {} };
  private listeners = new Set<() => void>();
  /** SSE `chat` lines (e.g. job_done) are pushed to subscribers, not kept in the snapshot. */
  chatListeners = new Set<(entry: Record<string, unknown>) => void>();
  subscribe = (fn: () => void) => {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  };
  get = () => this.snap;
  private set(next: Partial<Snapshot>) {
    this.snap = { ...this.snap, ...next };
    this.listeners.forEach((l) => l());
  }
  setStatus(status: StreamStatus) {
    if (this.snap.status !== status) this.set({ status });
  }
  seedJobs(jobs: Job[]) {
    const next = { ...this.snap.jobs };
    for (const j of jobs) if (!next[j.id]) next[j.id] = j;
    this.set({ jobs: next });
  }
  upsertJob(job: Job) {
    this.set({ jobs: { ...this.snap.jobs, [job.id]: { ...this.snap.jobs[job.id], ...job } } });
  }
  appendLog(ev: LogEvent) {
    const prev = this.snap.logs[ev.job_id] ?? [];
    const next = prev.length >= MAX_LOG_LINES ? [...prev.slice(-MAX_LOG_LINES + 1), ev.line] : [...prev, ev.line];
    this.set({ logs: { ...this.snap.logs, [ev.job_id]: next } });
  }
  touchEntity() {
    this.set({ lastEntityAt: Date.now() });
  }
  private patchAttempt(cid: string, index: number, fn: (a: LiveAttempt, c: LiveCollection) => LiveAttempt, total?: number) {
    const c: LiveCollection = this.snap.agent[cid] ?? { total: 0, lastIndex: -1, attempts: {} };
    const prev = c.attempts[index] ?? { index, items: [] };
    const nextC: LiveCollection = {
      total: total ?? c.total,
      lastIndex: Math.max(c.lastIndex, index),
      attempts: { ...c.attempts, [index]: fn(prev, c) },
    };
    this.set({ agent: { ...this.snap.agent, [cid]: nextC } });
  }
  agentStep(ev: AgentStepEvent) {
    const push = (a: LiveAttempt, item: LiveItem): LiveAttempt => ({
      ...a,
      items: a.items.length >= MAX_ITEMS_PER_ATTEMPT ? [...a.items.slice(1), item] : [...a.items, item],
    });
    switch (ev.type) {
      case "attempt_start":
        return this.patchAttempt(ev.collection_id, ev.index, (a) => ({ ...a, seed: ev.seed, scenario: ev.scenario }), ev.total);
      case "episode_start":
        return this.patchAttempt(ev.collection_id, ev.index, (a) => ({ ...a, seed: ev.seed, condition: ev.condition }), ev.total);
      case "agent_thought":
        return this.patchAttempt(ev.collection_id, ev.index, (a) => push(a, { kind: "thought", text: ev.text }));
      case "agent_step": {
        const { collection_id, type: _t, index, ...step } = ev;
        void _t;
        return this.patchAttempt(collection_id, index, (a) => push(a, { kind: "step", step: step as TraceStep }));
      }
    }
  }
  /** Rebuild the in-progress attempt from GET /agent-collections/{cid}/live, unless SSE already has more. */
  hydrateLive(cid: string, live: CollectionLive) {
    if (live.index === null || !live.start) return;
    const items: LiveItem[] = live.steps.map((ev) => {
      if (ev.type === "agent_thought") return { kind: "thought", text: ev.text };
      const { type: _t, index: _i, ...step } = ev;
      void _t;
      void _i;
      return { kind: "step", step: step as TraceStep };
    });
    const start = live.start;
    this.patchAttempt(
      cid,
      live.index,
      (a) => ({ ...a, seed: start.seed, scenario: start.scenario, items: a.items.length >= items.length ? a.items : items }),
      start.total,
    );
  }
  agentAttempt(a: AgentAttempt) {
    this.patchAttempt(a.collection_id, a.index, (prev) => ({ ...prev, seed: a.seed, scenario: a.scenario, result: a }));
  }
}

/** Which react-query keys to refresh when a table changes. */
const TABLE_KEYS: Record<string, string[]> = {
  projects: ["projects", "state"],
  episodes: ["episodes", "state"],
  episode_artifacts: ["episodes", "media"],
  analyses: ["analysis", "latest-analysis", "state"],
  failure_clusters: ["failure-clusters"],
  recommendations: ["state", "recommendations"],
  dataset_versions: ["datasets", "state"],
  dataset_episode_links: ["datasets"],
  training_jobs: ["training", "state"],
  checkpoints: ["checkpoints", "training"],
  evaluation_runs: ["evaluations", "evaluation", "state"],
  simulation_branches: ["evaluations", "evaluation"],
  tool_audit_logs: ["audit"],
  agent_collections: ["agent-collections", "agent-collection", "collector-comparison", "planner-budget"],
  agent_attempts: ["agent-attempts", "agent-collection", "collector-comparison"],
};

function invalidateFor(qc: QueryClient, pid: string, table: string) {
  const keys = TABLE_KEYS[table] ?? ["state"];
  for (const k of keys) {
    if (k === "projects") qc.invalidateQueries({ queryKey: ["projects"] });
    else if (k === "analysis" || k === "media" || k === "agent-collection" || k === "evaluation") qc.invalidateQueries({ queryKey: [k] });
    else qc.invalidateQueries({ queryKey: [k, pid] });
  }
  qc.invalidateQueries({ queryKey: ["next-actions", pid] });
}

const Ctx = createContext<ProjectEventStore | null>(null);

export function ProjectEventsProvider({ pid, children }: { pid: string; children: ReactNode }) {
  const qc = useQueryClient();
  const store = useMemo(() => new ProjectEventStore(), [pid]); // eslint-disable-line react-hooks/exhaustive-deps
  const qcRef = useRef(qc);
  qcRef.current = qc;

  // seed the job store so work started before this page loaded (agent, MCP, autopilot) shows up
  useEffect(() => {
    let cancelled = false;
    api
      .jobs(pid)
      .then((js) => !cancelled && store.seedJobs(js))
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [pid, store]);

  useEffect(() => {
    const es = new EventSource(api.eventsUrl(pid));
    store.setStatus("connecting");
    let dropped = false;
    es.onopen = () => {
      store.setStatus("open");
      if (!dropped) return;
      // back after a drop (e.g. the backend restarted): events in between are lost, so take the jobs as they are
      // now (a run the restart ended no longer shows as live) and refresh this project's data
      dropped = false;
      api.jobs(pid).then((js) => js.forEach((j) => store.upsertJob(j))).catch(() => {});
      qcRef.current.invalidateQueries({ predicate: (q) => q.queryKey.includes(pid) });
      store.chatListeners.forEach((fn) => fn({ kind: "reconnected" })); // the chat re-reads what is really running
    };
    es.onerror = () => {
      dropped = true;
      store.setStatus(es.readyState === EventSource.CLOSED ? "error" : "connecting");
    };
    const parse = <T,>(e: MessageEvent): T | null => {
      try {
        return JSON.parse(e.data) as T;
      } catch {
        return null;
      }
    };
    es.addEventListener("job", (e) => {
      const job = parse<Job>(e as MessageEvent);
      if (!job) return;
      store.upsertJob(job);
      if (job.status === "succeeded" || job.status === "failed") {
        // the job's output landed somewhere; refresh broadly
        qcRef.current.invalidateQueries({ predicate: (q) => q.queryKey.includes(pid) });
        qcRef.current.invalidateQueries({ queryKey: ["analysis"] });
      }
    });
    es.addEventListener("entity", (e) => {
      const ev = parse<EntityEvent>(e as MessageEvent);
      if (!ev) return;
      store.touchEntity();
      invalidateFor(qcRef.current, pid, ev.table);
    });
    es.addEventListener("autopilot", (e) => {
      const run = parse<AutopilotRun>(e as MessageEvent);
      if (run) qcRef.current.setQueryData(["autopilot", pid], run);
    });
    es.addEventListener("chat", (e) => {
      const entry = parse<Record<string, unknown>>(e as MessageEvent);
      if (entry) store.chatListeners.forEach((fn) => fn(entry));
    });
    es.addEventListener("agent_step", (e) => {
      const ev = parse<AgentStepEvent>(e as MessageEvent);
      if (ev) store.agentStep(ev);
    });
    es.addEventListener("agent_attempt", (e) => {
      const a = parse<AgentAttempt & { collection_id: string }>(e as MessageEvent);
      if (!a) return;
      store.agentAttempt(a);
      const qc = qcRef.current;
      for (const k of ["agent-attempts", "agent-collections", "collector-comparison", "planner-budget", "episodes", "state"])
        qc.invalidateQueries({ queryKey: [k, pid] });
      qc.invalidateQueries({ queryKey: ["agent-collection", a.collection_id] });
    });
    es.addEventListener("log", (e) => {
      const ev = parse<LogEvent>(e as MessageEvent);
      if (ev) store.appendLog(ev);
    });
    return () => es.close();
  }, [pid, store]);

  return <Ctx.Provider value={store}>{children}</Ctx.Provider>;
}

function useStore() {
  const s = useContext(Ctx);
  if (!s) throw new Error("useProjectEvents must be used inside ProjectEventsProvider");
  return s;
}

export function useStreamStatus(): StreamStatus {
  const s = useStore();
  return useSyncExternalStore(s.subscribe, () => s.get().status);
}

/** Live Job state from SSE. Seed it with the Job returned by a POST so it shows immediately. */
export function useJob(jobId: string | null | undefined): Job | undefined {
  const s = useStore();
  return useSyncExternalStore(s.subscribe, () => (jobId ? s.get().jobs[jobId] : undefined));
}

export function useJobs(): Record<string, Job> {
  const s = useStore();
  return useSyncExternalStore(s.subscribe, () => s.get().jobs);
}

export function useTrackJob() {
  const s = useStore();
  return (job: Job) => s.upsertJob(job);
}

export function useLogs(ids: (string | null | undefined)[]): string[] {
  const s = useStore();
  const logs = useSyncExternalStore(s.subscribe, () => s.get().logs);
  return useMemo(() => ids.flatMap((id) => (id ? logs[id] ?? [] : [])), [logs, ids.join("|")]); // eslint-disable-line react-hooks/exhaustive-deps
}

/** Timestamp of the last `entity` event (any table), for consumers that refresh on any change. */
export function useLastEntityAt(): number {
  const s = useStore();
  return useSyncExternalStore(s.subscribe, () => s.get().lastEntityAt);
}

export function useHydrateLive() {
  const s = useStore();
  return (cid: string, live: CollectionLive) => s.hydrateLive(cid, live);
}

/** Everything this tab has seen over SSE for one agent collection. */
export function useAgentLive(collectionId: string | null | undefined): LiveCollection | undefined {
  const s = useStore();
  return useSyncExternalStore(s.subscribe, () => (collectionId ? s.get().agent[collectionId] : undefined));
}

/** Subscribe to Cortex-written chat lines arriving on the project stream (SSE `chat`). */
export function useChatStream(fn: (entry: Record<string, unknown>) => void) {
  const s = useStore();
  const ref = useRef(fn);
  ref.current = fn;
  useEffect(() => {
    const h = (e: Record<string, unknown>) => ref.current(e);
    s.chatListeners.add(h);
    return () => {
      s.chatListeners.delete(h);
    };
  }, [s]);
}

export const isJobActive = (j?: Job) => !!j && (j.status === "queued" || j.status === "running");
