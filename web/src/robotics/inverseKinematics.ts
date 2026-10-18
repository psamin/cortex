import { solveInPlace } from "./math";
import { ForwardKinematics } from "./forwardKinematics";
import { computeJacobian } from "./jacobian";
import { atLimit, clampToLimits } from "./jointLimits";
import type { IKOptions, IKResult, RobotConfig, Vec3 } from "./types";

export const DEFAULT_IK: IKOptions = {
  maxIterations: 24,
  tolerance: 1e-3,
  damping: 0.06,
  maxStep: 0.1,
  maxJointStep: 0.18,
  postureGain: 0.08,
  projectorDamping: 0.005,
  direction: null,
  directionWeight: 0.5,
  directionTolerance: 0.03,
};

/**
 * Damped-least-squares IK with task priorities:
 *   1. position     dq  = J^T (J J^T + lambda^2 I)^-1 e
 *   2. direction    dq += N_pos  * J_dir^+ e_dir         (optional tool approach axis)
 *   3. posture      dq += N_all  * k (q_posture - q)
 *
 * - Direction rows are projected perpendicular to the tool axis so tool roll stays
 *   free for the posture term.
 * - Warm-started from the q passed in; q is updated in place.
 * - Joint-limit projection after every step; per-iteration step limits.
 * - Never reports convergence unless the residuals are within tolerance.
 *
 * All buffers are preallocated: no allocation inside solve().
 */
export class IKSolver {
  readonly fk: ForwardKinematics;
  private readonly n: number;
  private readonly J = new Float64Array(36);
  private readonly A = new Float64Array(36);
  private readonly e = new Float64Array(6);
  private readonly y = new Float64Array(6);
  private readonly z: Float64Array;
  private readonly dq: Float64Array;
  private readonly Jz = new Float64Array(6);
  private readonly tmp: Float64Array;
  private lambda2 = DEFAULT_IK.damping * DEFAULT_IK.damping;

  constructor(readonly config: RobotConfig) {
    this.fk = new ForwardKinematics(config);
    this.n = config.joints.length;
    this.z = new Float64Array(this.n);
    this.dq = new Float64Array(this.n);
    this.tmp = new Float64Array(this.n);
  }

  solve(q: Float64Array, target: Vec3 | Float64Array, options: Partial<IKOptions> = {}): IKResult {
    const o = { ...DEFAULT_IK, ...options };
    const { fk, n, J, e, z, dq } = this;
    const dir = o.direction ?? null;
    this.lambda2 = o.damping * o.damping;
    let posErr = Infinity;
    let dirErr = 0;
    let iterations = 0;

    for (let it = 0; it <= o.maxIterations; it++) {
      fk.compute(q);
      const t = fk.tcpPosition;
      let ex = target[0] - t[0];
      let ey = target[1] - t[1];
      let ez = target[2] - t[2];
      posErr = Math.hypot(ex, ey, ez);

      // Tool direction error: rotation axis * sin(angle) that aligns tool -> desired.
      const d = fk.toolDirection;
      let rx = 0;
      let ry = 0;
      let rz = 0;
      if (dir) {
        rx = d[1] * dir[2] - d[2] * dir[1];
        ry = d[2] * dir[0] - d[0] * dir[2];
        rz = d[0] * dir[1] - d[1] * dir[0];
        const dot = d[0] * dir[0] + d[1] * dir[1] + d[2] * dir[2];
        dirErr = Math.atan2(Math.hypot(rx, ry, rz), dot);
        if (dot < 0) {
          // > 90 deg away: rotate at full rate about the cross axis (or any perpendicular).
          const s = Math.hypot(rx, ry, rz) || 1;
          rx /= s;
          ry /= s;
          rz /= s;
        }
      }

      if (posErr < o.tolerance && (!dir || dirErr < o.directionTolerance)) break;
      if (it === o.maxIterations) break;
      iterations = it + 1;

      // Clamp the Cartesian step so far targets are approached gradually.
      if (posErr > o.maxStep) {
        const s = o.maxStep / posErr;
        ex *= s;
        ey *= s;
        ez *= s;
      }

      // Rows 0-2: position Jacobian. Rows 3-5: angular rows (tool roll removed).
      computeJacobian(fk, J, dir !== null, 1);
      if (dir) {
        for (let i = 0; i < n; i++) {
          const ax = J[3 * n + i];
          const ay = J[4 * n + i];
          const az = J[5 * n + i];
          const k = ax * d[0] + ay * d[1] + az * d[2];
          J[3 * n + i] = ax - k * d[0];
          J[4 * n + i] = ay - k * d[1];
          J[5 * n + i] = az - k * d[2];
        }
      }

      // Task priority 1 (position): dq = J^T (J J^T + lambda^2 I)^-1 e
      e[0] = ex;
      e[1] = ey;
      e[2] = ez;
      if (!this.dampedSolve(0, 3, e, dq, this.lambda2)) break;

      // Priority 2 (tool direction), projected into the null space of the position task,
      // so styling the approach angle can never cost position accuracy.
      if (dir) {
        e[0] = rx * o.directionWeight;
        e[1] = ry * o.directionWeight;
        e[2] = rz * o.directionWeight;
        if (this.dampedSolve(3, 3, e, z, this.lambda2)) {
          this.projectNull(z, 0, 3, o.projectorDamping);
          for (let i = 0; i < n; i++) dq[i] += z[i];
        }
      }

      // Priority 3 (posture), projected into the null space of every active task.
      if (o.postureGain > 0 && o.posture) {
        for (let i = 0; i < n; i++) z[i] = o.postureGain * (o.posture[i] - q[i]);
        this.projectNull(z, 0, dir ? 6 : 3, o.projectorDamping);
        for (let i = 0; i < n; i++) dq[i] += z[i];
      }

      // Step limiting (uniform scale keeps the direction of motion).
      let maxAbs = 0;
      for (let i = 0; i < n; i++) maxAbs = Math.max(maxAbs, Math.abs(dq[i]));
      const scale = maxAbs > o.maxJointStep ? o.maxJointStep / maxAbs : 1;
      for (let i = 0; i < n; i++) q[i] += dq[i] * scale;
      clampToLimits(q, this.config);
    }

    let limited = false;
    for (let i = 0; i < n; i++) if (atLimit(q, this.config, i)) limited = true;
    const converged = posErr < o.tolerance && (!dir || dirErr < o.directionTolerance);
    return { converged, positionError: posErr, directionError: dir ? dirErr : 0, iterations, limited };
  }

  /** A = Jr Jr^T + lambda^2 I for the rows [r0, r0 + m). */
  private buildA(r0: number, m: number, lambda2: number) {
    const { J, A, n } = this;
    for (let r = 0; r < m; r++) {
      for (let c = r; c < m; c++) {
        let s = 0;
        for (let i = 0; i < n; i++) s += J[(r0 + r) * n + i] * J[(r0 + c) * n + i];
        A[r * m + c] = s;
        A[c * m + r] = s;
      }
      A[r * m + r] += lambda2;
    }
  }

  /** out = Jr^T (Jr Jr^T + lambda^2 I)^-1 rhs  (damped pseudo-inverse of a row block). */
  private dampedSolve(r0: number, m: number, rhs: Float64Array, out: Float64Array, lambda2: number): boolean {
    const { J, A, y, n } = this;
    this.buildA(r0, m, lambda2);
    for (let r = 0; r < m; r++) y[r] = rhs[r];
    if (!solveInPlace(A, y, m)) return false;
    for (let i = 0; i < n; i++) {
      let s = 0;
      for (let r = 0; r < m; r++) s += J[(r0 + r) * n + i] * y[r];
      out[i] = s;
    }
    return true;
  }

  /** v <- (I - Jr^+ Jr) v  using a near-exact (tiny damping) pseudo-inverse. */
  private projectNull(v: Float64Array, r0: number, m: number, damping: number) {
    const { J, Jz, n, tmp } = this;
    for (let r = 0; r < m; r++) {
      let s = 0;
      for (let i = 0; i < n; i++) s += J[(r0 + r) * n + i] * v[i];
      Jz[r] = s;
    }
    if (!this.dampedSolve(r0, m, Jz, tmp, damping * damping)) return;
    for (let i = 0; i < n; i++) v[i] -= tmp[i];
  }
}
