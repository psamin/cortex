import { useQuery, useQueries } from "@tanstack/react-query";
import { api } from "./client";
import type { Episode } from "./types";

// Query keys are [resource, pid?] so SSE `entity` events can invalidate precisely (see events.tsx).

export const useHealth = () =>
  useQuery({ queryKey: ["health"], queryFn: api.health, refetchInterval: 15_000, retry: 0 });

export const useProjects = () => useQuery({ queryKey: ["projects"], queryFn: api.listProjects });

/** Home: projects with history only (["projects", …] so project events refresh it too). */
export const useProjectsWithHistory = () =>
  useQuery({ queryKey: ["projects", "history"], queryFn: api.listProjectsWithHistory });

export const useProjectState = (pid: string) =>
  useQuery({ queryKey: ["state", pid], queryFn: () => api.projectState(pid) });

export const useEpisodes = (pid: string) =>
  useQuery({ queryKey: ["episodes", pid], queryFn: () => api.episodes(pid) });

export const useEpisodeMedia = (eid: string | null | undefined) =>
  useQuery({
    queryKey: ["media", eid],
    queryFn: () => api.episodeMedia(eid!),
    enabled: !!eid,
    staleTime: 5 * 60_000,
    retry: 0,
  });

/** Thumbnails for a list of episodes (one media call each; see README contract notes). */
export const useEpisodeThumbs = (episodes: Episode[] | undefined) =>
  useQueries({
    queries: (episodes ?? []).map((e) => ({
      queryKey: ["media", e.id],
      queryFn: () => api.episodeMedia(e.id),
      staleTime: 5 * 60_000,
      retry: 0,
    })),
  });

export const usePlanner = () =>
  useQuery({ queryKey: ["planner"], queryFn: api.planner, staleTime: 60_000, retry: 0 });

export const useAgentCollections = (pid: string) =>
  useQuery({ queryKey: ["agent-collections", pid], queryFn: () => api.agentCollections(pid) });

export const useAgentCollection = (cid: string | null | undefined, poll = false) =>
  useQuery({
    queryKey: ["agent-collection", cid],
    queryFn: () => api.agentCollection(cid!),
    enabled: !!cid,
    // fallback when the event stream is down: keep totals moving while a collection runs
    refetchInterval: poll ? 4000 : false,
  });

export const useAgentAttempts = (pid: string) =>
  useQuery({ queryKey: ["agent-attempts", pid], queryFn: () => api.agentAttempts(pid) });

export const useAgentAttempt = (aid: string | null | undefined) =>
  useQuery({ queryKey: ["agent-attempt", aid], queryFn: () => api.agentAttempt(aid!), enabled: !!aid });

export const usePlannerBudget = (pid: string) =>
  useQuery({ queryKey: ["planner-budget", pid], queryFn: () => api.plannerBudget(pid) });

export const useCollectorComparison = (pid: string) =>
  useQuery({ queryKey: ["collector-comparison", pid], queryFn: () => api.collectorComparison(pid) });

export const useAutopilot = (pid: string) =>
  useQuery({ queryKey: ["autopilot", pid], queryFn: () => api.autopilot(pid) });

export const useLatestAnalysis = (pid: string) =>
  useQuery({ queryKey: ["latest-analysis", pid], queryFn: () => api.latestAnalysis(pid) });

export const useRecommendations = (pid: string) =>
  useQuery({ queryKey: ["recommendations", pid], queryFn: () => api.recommendations(pid) });

export const useAnalysis = (aid: string | null | undefined) =>
  useQuery({ queryKey: ["analysis", aid], queryFn: () => api.analysis(aid!), enabled: !!aid });

export const useFailureClusters = (pid: string) =>
  useQuery({ queryKey: ["failure-clusters", pid], queryFn: () => api.failureClusters(pid) });

export const useDatasets = (pid: string) =>
  useQuery({ queryKey: ["datasets", pid], queryFn: () => api.datasets(pid) });

export const useTraining = (pid: string) =>
  useQuery({ queryKey: ["training", pid], queryFn: () => api.training(pid) });

export const useCheckpoints = (pid: string) =>
  useQuery({ queryKey: ["checkpoints", pid], queryFn: () => api.checkpoints(pid) });

export const useEvaluations = (pid: string) =>
  useQuery({ queryKey: ["evaluations", pid], queryFn: () => api.evaluations(pid) });

export const useNextActions = (pid: string) =>
  useQuery({ queryKey: ["next-actions", pid], queryFn: () => api.nextActions(pid) });

export const useAudit = (pid: string) =>
  useQuery({ queryKey: ["audit", pid], queryFn: () => api.audit(pid), refetchInterval: 10_000 });

export const useEngine = () => useQuery({ queryKey: ["agent-engine"], queryFn: api.agentEngine, staleTime: 5 * 60_000, retry: 0 });

export const useEvaluation = (evid: string | null | undefined, poll = false) =>
  useQuery({
    queryKey: ["evaluation", evid],
    queryFn: () => api.evaluation(evid!),
    enabled: !!evid,
    refetchInterval: poll ? 4000 : false,
  });

export const useJobQuery = (id: string | null | undefined, poll = false) =>
  useQuery({
    queryKey: ["job", id],
    queryFn: () => api.job(id!),
    enabled: !!id,
    // a job that no longer exists (e.g. from before job history survived restarts) is not polled forever
    refetchInterval: (q) => (poll && q.state.status !== "error" ? 3000 : false),
    retry: 0,
  });
