import { maxReach, ROBOT, shoulderHeight } from "./robotConfig";
import type { RobotConfig } from "./types";

export interface Workspace {
  /** Shoulder center, world. */
  center: [number, number, number];
  rMax: number;
  rMin: number;
  /** Lowest allowed TCP height. */
  minY: number;
  /** Highest allowed TCP height. */
  maxY: number;
  /** Minimum horizontal distance from the base yaw axis (keeps targets off the base). */
  minHorizontal: number;
}

export function workspaceFor(cfg: RobotConfig = ROBOT): Workspace {
  const reach = maxReach(cfg);
  return {
    center: [cfg.baseOrigin[0], shoulderHeight(cfg), cfg.baseOrigin[2]],
    rMax: reach * 0.8,
    rMin: 0.42,
    minY: 0.07,
    maxY: shoulderHeight(cfg) + 0.85,
    minHorizontal: 0.38,
  };
}

/**
 * Project a desired TCP target into a conservative reachable region.
 * Keeps the solver fed with reachable goals so the arm never thrashes at the boundary.
 * Writes into `out` (may alias `p`). Returns true if the point was modified.
 */
export function clampToWorkspace(
  p: ArrayLike<number>,
  ws: Workspace,
  out: Float64Array | number[],
): boolean {
  let x = p[0];
  let y = p[1];
  let z = p[2];
  const changedIn = [x, y, z];

  // Height band.
  y = Math.min(Math.max(y, ws.minY), ws.maxY);

  // Keep away from the base axis.
  let hx = x - ws.center[0];
  let hz = z - ws.center[2];
  let h = Math.hypot(hx, hz);
  if (h < ws.minHorizontal) {
    if (h < 1e-6) {
      hx = 1;
      hz = 0;
      h = 1;
    }
    x = ws.center[0] + (hx / h) * ws.minHorizontal;
    z = ws.center[2] + (hz / h) * ws.minHorizontal;
  }

  // Spherical shell around the shoulder.
  const dx = x - ws.center[0];
  const dy = y - ws.center[1];
  const dz = z - ws.center[2];
  const r = Math.hypot(dx, dy, dz);
  if (r > ws.rMax) {
    const s = ws.rMax / r;
    x = ws.center[0] + dx * s;
    y = ws.center[1] + dy * s;
    z = ws.center[2] + dz * s;
  } else if (r < ws.rMin && r > 1e-9) {
    const s = ws.rMin / r;
    x = ws.center[0] + dx * s;
    y = ws.center[1] + dy * s;
    z = ws.center[2] + dz * s;
  }
  y = Math.max(y, ws.minY);

  out[0] = x;
  out[1] = y;
  out[2] = z;
  return x !== changedIn[0] || y !== changedIn[1] || z !== changedIn[2];
}
