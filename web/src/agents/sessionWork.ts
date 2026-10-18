import { useEffect, useRef } from "react";
import { useQueries, useQueryClient } from "@tanstack/react-query";
import { api } from "../api/client";
import type { AgentCollection, EvaluationWithBranches, Job, RLSandboxView, TrainingJob } from "../api/types";

/**
 * Everything the agent's projects are doing, for the session view: demos, training, evaluations and RL
 * sandboxes. One query per project; each project's SSE stream (job / entity / rl / agent_attempt) invalidates
 * it, so cards move live without the agent having to poll. A slow poll is the fallback while work is active.
 */
export interface ProjectWork {
  pid: string;
  jobs: Job[];
  collections: AgentCollection[];
  training: TrainingJob[];
  evaluations: EvaluationWithBranches[];
  sandboxes: RLSandboxView[];
}

const ACTIVE = new Set(["queued", "running", "proposed", "approved", "piloting", "accepted", "ready"]);
export const isActive = (s: string | undefined) => !!s && ACTIVE.has(s);

async function load(pid: string): Promise<ProjectWork> {
  const soft = <T>(p: Promise<T>, empty: T) => p.catch(() => empty); // a missing feature must not blank the board
  const [jobs, collections, training, evaluations, sandboxes] = await Promise.all([
    soft(api.jobs(pid), [] as Job[]),
    soft(api.agentCollections(pid), [] as AgentCollection[]),
    soft(api.training(pid), [] as TrainingJob[]),
    soft(api.evaluations(pid), [] as EvaluationWithBranches[]),
    soft(api.rlSandboxes(pid), [] as RLSandboxView[]),
  ]);
  return { pid, jobs, collections, training, evaluations, sandboxes };
}

function anyActive(w: ProjectWork | undefined): boolean {
  if (!w) return false;
  return (
    w.jobs.some((j) => isActive(j.status)) ||
    w.collections.some((c) => isActive(c.status)) ||
    w.training.some((t) => isActive(t.status)) ||
    w.evaluations.some((e) => isActive(e.run.status)) ||
    w.sandboxes.some((s) => isActive(s.status))
  );
}

/** Live work for these projects. Returns one entry per project (undefined while loading). */
export function useSessionWork(pids: string[]): { work: (ProjectWork | undefined)[]; loading: boolean } {
  const qc = useQueryClient();
  const results = useQueries({
    queries: pids.map((pid) => ({
      queryKey: ["work", pid],
      queryFn: () => load(pid),
      refetchInterval: (q: { state: { data?: ProjectWork } }) => (anyActive(q.state.data) ? 5_000 : 30_000),
    })),
  });

  // One EventSource per project; any progress event schedules a (debounced) refresh of that project's work.
  const timers = useRef<Record<string, ReturnType<typeof setTimeout>>>({});
  const key = pids.join(",");
  useEffect(() => {
    const sources = pids.map((pid) => {
      const es = new EventSource(api.eventsUrl(pid));
      const bump = () => {
        clearTimeout(timers.current[pid]);
        timers.current[pid] = setTimeout(() => void qc.invalidateQueries({ queryKey: ["work", pid] }), 400);
      };
      for (const ev of ["job", "entity", "rl", "autopilot", "agent_attempt"]) es.addEventListener(ev, bump);
      return es;
    });
    const t = timers.current;
    return () => {
      sources.forEach((es) => es.close());
      Object.values(t).forEach(clearTimeout);
    };
  }, [key]); // eslint-disable-line react-hooks/exhaustive-deps

  return { work: results.map((r) => r.data), loading: results.some((r) => r.isLoading) };
}

/** Only work that started during this session (with a minute of slack), newest first. */
export function since<T extends { created_at?: string }>(items: T[], from: string | undefined): T[] {
  const t0 = from ? Date.parse(from) - 60_000 : 0;
  return items
    .filter((x) => !x.created_at || Date.parse(x.created_at) >= t0)
    .sort((a, b) => (a.created_at && b.created_at ? (a.created_at < b.created_at ? 1 : -1) : 0));
}
