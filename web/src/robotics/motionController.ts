import type { RobotConfig } from "./types";

/** Largest frame delta we integrate; longer gaps (tab switch) are treated as this. */
export const MAX_FRAME_DT = 0.1;
/** Integration substep so behaviour is identical at 30, 60 or 144 Hz. */
export const SUBSTEP = 1 / 240;

export interface JointMotionState {
  q: Float64Array;
  v: Float64Array;
}

export function createMotionState(q0: ArrayLike<number>): JointMotionState {
  return { q: Float64Array.from(q0), v: new Float64Array(q0.length) };
}

/**
 * Advance joints toward `qDesired` respecting per-joint velocity and acceleration limits.
 * Velocity command is a P-term capped by vmax and by the speed from which the joint
 * can still brake to zero before the goal (sqrt(2 a |err|)), so it never overshoots
 * by more than one substep. Frame-rate independent via fixed substeps.
 */
export function stepJoints(
  state: JointMotionState,
  qDesired: ArrayLike<number>,
  cfg: RobotConfig,
  dt: number,
  options: { gain?: number; speedScale?: number } = {},
): void {
  const gain = options.gain ?? 6;
  const speedScale = options.speedScale ?? 1;
  let remaining = Math.min(Math.max(dt, 0), MAX_FRAME_DT);
  const { q, v } = state;
  while (remaining > 1e-9) {
    const h = Math.min(SUBSTEP, remaining);
    remaining -= h;
    for (let i = 0; i < cfg.joints.length; i++) {
      const j = cfg.joints[i];
      const vmax = j.maxVelocity * speedScale;
      const amax = j.maxAcceleration * speedScale;
      const err = qDesired[i] - q[i];
      let vt = gain * err;
      const vBrake = Math.sqrt(2 * amax * Math.abs(err));
      const lim = Math.min(vmax, vBrake);
      if (vt > lim) vt = lim;
      else if (vt < -lim) vt = -lim;
      let dv = vt - v[i];
      const dvMax = amax * h;
      if (dv > dvMax) dv = dvMax;
      else if (dv < -dvMax) dv = -dvMax;
      v[i] += dv;
      q[i] += v[i] * h;
      if (q[i] < j.min) {
        q[i] = j.min;
        v[i] = 0;
      } else if (q[i] > j.max) {
        q[i] = j.max;
        v[i] = 0;
      }
    }
  }
}

export function maxJointError(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
}

/** Frame-rate independent exponential smoothing of a vector toward a target. */
export function smoothToward(
  current: Float64Array,
  target: ArrayLike<number>,
  dt: number,
  timeConstant: number,
): void {
  const a = 1 - Math.exp(-Math.min(dt, MAX_FRAME_DT) / Math.max(timeConstant, 1e-4));
  for (let i = 0; i < current.length; i++) current[i] += (target[i] - current[i]) * a;
}
