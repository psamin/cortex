import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { api, ApiError, resolveMediaUrl } from "../api/client";
import { useEpisodes, useEvaluations } from "../api/hooks";
import { agentLabel, isImported, type Condition, type Episode, type EvaluationWithBranches, type Iteration } from "../api/types";

/** One playable MuJoCo episode: a demonstration or an evaluation rollout. */
export interface Rollout {
  kind: "demo" | "eval";
  id: string;
  label: string;
  success: boolean | null;
  failureMode: string | null;
  condition: Condition | null;
  seed: number | null;
  /** direct URL (from /rollouts); otherwise resolved lazily from episodeId / videoKey */
  videoUrl: string | null;
  episodeId: string | null;
  videoKey: string | null;
  evaluationId: string | null;
  policy: string | null;
  iteration: number | null;
  createdAt: string;
  /** imported human demo: no simulator verdict, no known scene condition */
  imported: boolean;
  source: string | null;
  task: string | null;
  /** cameras this rollout actually has, in the backend's order and wording */
  cameras: CameraInfo[];
}

export type Camera = "presentation" | "front" | "wrist";
export interface CameraInfo {
  id: Camera;
  label: string;
  detail: string;
}

const CAMERA_IDS: Camera[] = ["presentation", "front", "wrist"];
/** Used only when the backend sends no camera list (older backends). */
const FALLBACK_DEMO_CAMERAS: CameraInfo[] = [
  { id: "presentation", label: "Studio view", detail: "Rendered for people; the policy never sees it." },
  { id: "front", label: "Front camera", detail: "" },
  { id: "wrist", label: "Wrist camera", detail: "" },
];
const FALLBACK_STUDIO_ONLY = FALLBACK_DEMO_CAMERAS.slice(0, 1);

function parseCameras(raw: unknown, hasEpisode: boolean): CameraInfo[] {
  if (Array.isArray(raw) && raw.length && typeof raw[0] === "object") {
    const cams = (raw as { id?: string; label?: string; detail?: string }[])
      .filter((c): c is { id: Camera; label?: string; detail?: string } => CAMERA_IDS.includes(c.id as Camera))
      .map((c) => ({ id: c.id, label: c.label || c.id, detail: c.detail ?? "" }));
    if (cams.length) return cams;
  }
  return hasEpisode ? FALLBACK_DEMO_CAMERAS : FALLBACK_STUDIO_ONLY;
}

function parseCondition(c: Condition | string | null | undefined): Condition | null {
  if (!c) return null;
  if (typeof c === "object") return c;
  const [cube_region, lighting] = c.split("/");
  return cube_region && lighting ? ({ cube_region, lighting } as Condition) : null;
}

const plannerLabel = (p: string | null | undefined) =>
  !p || /imported/i.test(p) ? null : p === "scripted_ik_oracle" ? "Scripted oracle" : agentLabel(p);

/** 404 → null (endpoint not shipped yet); other errors propagate. */
async function optional<T>(fn: () => Promise<T>): Promise<T | null> {
  try {
    return await fn();
  } catch (e) {
    if (e instanceof ApiError && e.status === 404) return null;
    throw e;
  }
}

export function useIterations(pid: string) {
  return useQuery({ queryKey: ["iterations", pid], queryFn: () => optional(() => api.iterations(pid)), retry: 0 });
}

function importedFields(e: Episode) {
  const imported = isImported(e);
  return { imported, source: imported ? e.provenance?.source ?? null : null, task: imported ? e.provenance?.task ?? null : null };
}

function fromEpisodes(eps: Episode[]): Rollout[] {
  return eps
    .filter((e) => e.status !== "rejected")
    .map((e) => ({
      kind: "demo" as const,
      id: e.id,
      label: `Demo #${e.index}`,
      success: e.success,
      failureMode: null,
      condition: e.condition,
      seed: e.seed,
      videoUrl: null,
      episodeId: e.id,
      videoKey: null,
      evaluationId: null,
      policy: plannerLabel(e.provenance?.planner) ?? (e.source === "sim_scripted" ? "Scripted oracle" : null),
      iteration: null,
      createdAt: e.created_at,
      cameras: FALLBACK_DEMO_CAMERAS,
      ...importedFields(e),
    }));
}

function fromEvaluations(evs: EvaluationWithBranches[]): Rollout[] {
  const out: Rollout[] = [];
  for (const { run, branches } of evs) {
    const policy = run.policy.kind === "scripted_expert" ? "Scripted oracle" : run.policy_label || run.policy.checkpoint_id || "policy";
    for (const b of branches)
      for (const ep of b.result?.episodes ?? [])
        out.push({
          kind: "eval",
          id: `${b.id}:${ep.seed}`,
          label: `${b.spec.name.replace(/_/g, " ")} · seed ${ep.seed}`,
          success: ep.success,
          failureMode: ep.failure_mode,
          condition: b.spec.condition,
          seed: ep.seed,
          videoUrl: null,
          episodeId: null,
          videoKey: ep.video_key,
          evaluationId: run.id,
          policy,
          iteration: null,
          createdAt: run.created_at,
          imported: false,
          cameras: FALLBACK_STUDIO_ONLY,
          source: null,
          task: null,
        });
  }
  return out;
}

/** GET /rollouts when available; otherwise assembled from real episodes + evaluation results. */
export function useRollouts(pid: string) {
  const remote = useQuery({ queryKey: ["rollouts", pid], queryFn: () => optional(() => api.rollouts(pid)), retry: 0 });
  const episodes = useEpisodes(pid);
  const evals = useEvaluations(pid);
  const data = useMemo<Rollout[] | undefined>(() => {
    const epById = new Map((episodes.data ?? []).map((e) => [e.id, e]));
    if (remote.data)
      return remote.data.map((r) => ({
        kind: r.kind,
        id: r.id,
        label: r.label,
        success: r.success,
        failureMode: r.failure_mode ?? null,
        condition: parseCondition(r.condition),
        seed: r.seed ?? null,
        videoUrl: r.video_url ?? null,
        cameras: parseCameras(r.cameras, r.kind === "demo"),
        episodeId: r.kind === "demo" ? r.id : null,
        videoKey: null,
        evaluationId: r.evaluation_id ?? null,
        policy: r.policy ?? plannerLabel(r.planner),
        iteration: r.iteration ?? null,
        createdAt: r.created_at ?? "",
        ...(epById.get(r.id) ? importedFields(epById.get(r.id)!) : { imported: /imported/i.test(r.planner ?? ""), source: null, task: null }),
      }));
    if (remote.isLoading || !episodes.data || !evals.data) return undefined;
    return [...fromEvaluations(evals.data), ...fromEpisodes(episodes.data)].sort((a, b) => (a.createdAt < b.createdAt ? 1 : -1));
  }, [remote.data, remote.isLoading, episodes.data, evals.data]);
  return { data, isLoading: data === undefined && !remote.error, error: remote.error ?? episodes.error ?? evals.error, fromApi: !!remote.data };
}

/** Resolve a playable URL for a rollout and camera. Robot cameras exist only for recorded episodes. */
export function useRolloutVideo(r: Rollout | null, camera: Camera) {
  return useQuery({
    queryKey: ["rollout-video", r?.id, camera, r?.videoUrl, r?.videoKey],
    enabled: !!r,
    staleTime: 10 * 60_000,
    retry: 0,
    queryFn: async (): Promise<string | null> => {
      if (!r) return null;
      if (camera !== "presentation") return r.episodeId ? api.cameraUrl(r.episodeId, camera) : null;
      if (r.videoUrl) return resolveMediaUrl(r.videoUrl);
      if (r.episodeId) return resolveMediaUrl((await api.episodeMedia(r.episodeId)).video_url);
      if (r.videoKey) return resolveMediaUrl((await api.media(r.videoKey)).url);
      return null;
    },
  });
}

export const iterationSuccess = (it: Iteration) => {
  const b = it.evaluation?.branches ?? [];
  const n = b.reduce((s, x) => s + x.n, 0);
  const k = b.reduce((s, x) => s + x.successes, 0);
  return n ? { k, n, rate: k / n } : null;
};

/** Wilson 95% interval from measured counts. */
export function wilson(k: number, n: number): [number, number] {
  if (!n) return [0, 0];
  const z = 1.96;
  const p = k / n;
  const d = 1 + (z * z) / n;
  const c = (p + (z * z) / (2 * n)) / d;
  const h = (z * Math.sqrt((p * (1 - p)) / n + (z * z) / (4 * n * n))) / d;
  return [Math.max(0, c - h), Math.min(1, c + h)];
}
