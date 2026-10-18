import { useQuery } from "@tanstack/react-query";
import { API_BASE } from "../api/client";
import type { Condition } from "../api/types";

/** MuJoCo conventions: metres, z up, quaternions as [w, x, y, z]. */
export type Vec3 = [number, number, number];
export type QuatWXYZ = [number, number, number, number];

export interface SimMeshGeom {
  body: string;
  mesh: string;
  /** Places the raw STL in the body frame (MuJoCo's geom * inverse(mesh frame)). */
  pos: Vec3;
  quat: QuatWXYZ;
  scale: Vec3;
  role: "arm" | "motor";
}

export type PrimitiveRole =
  | "table"
  | "leg"
  | "floor"
  | "wall"
  | "object"
  | "zone"
  | "obstacle"
  | "box"
  | "parked"
  | "motor"
  | "prop";

export interface SimPrimitive {
  name: string | null;
  body: string;
  type: "box" | "cylinder" | "sphere" | "capsule" | "plane" | "ellipsoid" | string;
  /** MuJoCo size convention (half extents, radius + half length, ...). */
  size: Vec3;
  pos: Vec3;
  quat: QuatWXYZ;
  rgba: [number, number, number, number];
  role: PrimitiveRole;
}

export interface SimJoint {
  name: string;
  label: string;
  body: string;
  range_deg: [number, number];
}

export interface SimModel {
  scene: string;
  bodies: string[];
  meshes: SimMeshGeom[];
  primitives: SimPrimitive[];
  joints: SimJoint[];
  tcp: { site: string; body: string; pos: Vec3 };
  project_id?: string;
  pick?: string | null;
}

export interface SimState {
  scene: string;
  seed: number;
  source: "reset" | string;
  time: number;
  bodies: Record<string, { pos: Vec3; quat: QuatWXYZ }>;
  joints_deg: Record<string, number>;
  tcp: Vec3;
  pick: string | null;
  pick_body: string | null;
  project_id: string;
}

/** Same upstream-tabletop scene id the backend uses before a project's scene is built. */
export const UPSTREAM_SCENE = "so101_tabletop";

async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(API_BASE + path, { headers: { "X-Cortex-Actor": "gui" } });
  if (!res.ok) {
    let detail = `${res.status}`;
    try {
      detail = ((await res.json()) as { detail?: string }).detail ?? detail;
    } catch {
      /* not json */
    }
    throw new Error(detail);
  }
  return (await res.json()) as T;
}

export const meshUrl = (name: string) => `${API_BASE}/sim/meshes/${encodeURIComponent(name)}.stl`;

/**
 * Body poses from MuJoCo. Workers run in their own processes and publish only camera frames, so
 * during a live recording this is still the scene's reset state; it is polled at 2 Hz then so a
 * future pose feed shows up without UI changes.
 */
export function useSimState(pid: string, seed: number, condition?: Condition, live = false, sceneKey?: string) {
  const q = condition ? `&cube_region=${condition.cube_region}&lighting=${condition.lighting}` : "";
  return useQuery({
    queryKey: ["sim-state", pid, seed, condition?.cube_region, condition?.lighting, sceneKey],
    queryFn: () => getJson<SimState>(`/projects/${pid}/simulation/state?seed=${seed}${q}`),
    refetchInterval: live ? 500 : false,
    staleTime: 30_000,
    placeholderData: (prev) => prev,
    retry: 0,
  });
}

/** The scene description; immutable per scene id, so it is cached for the session. */
export function useSimModel(pid: string, scene: string | undefined) {
  return useQuery({
    queryKey: ["sim-model", scene],
    queryFn: () => getJson<SimModel>(scene === UPSTREAM_SCENE ? "/sim/model" : `/sim/model?pid=${pid}`),
    enabled: !!scene,
    staleTime: Infinity,
    retry: 0,
  });
}
