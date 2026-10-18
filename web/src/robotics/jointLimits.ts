import type { RobotConfig } from "./types";

/** Clamp joint angles to the configured limits in place. Returns true if any joint was clamped. */
export function clampToLimits(q: Float64Array | number[], cfg: RobotConfig): boolean {
  let clamped = false;
  for (let i = 0; i < cfg.joints.length; i++) {
    const { min, max } = cfg.joints[i];
    if (q[i] < min) {
      q[i] = min;
      clamped = true;
    } else if (q[i] > max) {
      q[i] = max;
      clamped = true;
    }
  }
  return clamped;
}

export function withinLimits(q: ArrayLike<number>, cfg: RobotConfig, eps = 1e-9): boolean {
  for (let i = 0; i < cfg.joints.length; i++) {
    const { min, max } = cfg.joints[i];
    if (!(q[i] >= min - eps && q[i] <= max + eps)) return false;
  }
  return true;
}

/** True if joint i sits on (or within eps of) a limit. */
export function atLimit(q: ArrayLike<number>, cfg: RobotConfig, i: number, eps = 1e-6): boolean {
  const { min, max } = cfg.joints[i];
  return q[i] <= min + eps || q[i] >= max - eps;
}
