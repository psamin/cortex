import type { ForwardKinematics } from "./forwardKinematics";

/**
 * Geometric Jacobian of a revolute chain, written into `out` (row-major, rows x n).
 * Rows 0-2: linear velocity of the TCP    J_i = a_i x (p_tcp - p_i)
 * Rows 3-5: angular velocity (optional)   J_i = a_i
 * Angular rows are scaled by `angularWeight`.
 * Assumes `fk.compute(q)` has already been called.
 */
export function computeJacobian(
  fk: ForwardKinematics,
  out: Float64Array,
  includeAngular: boolean,
  angularWeight = 1,
): number {
  const n = fk.n;
  const t = fk.tcpPosition;
  for (let i = 0; i < n; i++) {
    const a = fk.axes[i];
    const p = fk.positions[i];
    const dx = t[0] - p[0];
    const dy = t[1] - p[1];
    const dz = t[2] - p[2];
    out[0 * n + i] = a[1] * dz - a[2] * dy;
    out[1 * n + i] = a[2] * dx - a[0] * dz;
    out[2 * n + i] = a[0] * dy - a[1] * dx;
    if (includeAngular) {
      out[3 * n + i] = a[0] * angularWeight;
      out[4 * n + i] = a[1] * angularWeight;
      out[5 * n + i] = a[2] * angularWeight;
    }
  }
  return includeAngular ? 6 : 3;
}
